#include "../kv-runtime-demotion.hpp"
#include "../tlsf-allocator.hpp"
#include "kv-region-test-model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz). Use an explicit check that
// always runs, per the test-zone-sizing.cpp precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Numeric variant printing got/want, per the test-kv-slice-sizing.cpp
// check_eq() precedent. Both operands are cast to long long so size_t and int
// fields share one macro without format-specifier mismatches.
#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using ggml_sycl::context_demand_reconcile;
using ggml_sycl::context_side_demand;
using ggml_sycl::demand_scope;
using ggml_sycl::held_slot;
using ggml_sycl::kv_admission_mismatch;
using ggml_sycl::kv_alloc_slack_per_layer;
using ggml_sycl::kv_buffer_layer_owner;
using ggml_sycl::kv_demotion_input;
using ggml_sycl::kv_demotion_result;
using ggml_sycl::kv_device_fit_input;
using ggml_sycl::kv_device_residency_changed;
using ggml_sycl::kv_hot_layers_override_active;
using ggml_sycl::kv_layer_cells;
using ggml_sycl::kv_layer_desc;
using ggml_sycl::kv_layer_tensor_bytes;
using ggml_sycl::kv_optional_layout_yield;
using ggml_sycl::kv_optional_layouts;
using ggml_sycl::kv_reads_device_arena;
using ggml_sycl::kv_residency_input;
using ggml_sycl::kv_residency_needs_refit;
using ggml_sycl::kv_residency_result;
using ggml_sycl::kv_shape;
using ggml_sycl::kv_shape_changed;
using ggml_sycl::kv_vram_available;
using ggml_sycl::kv_weight_capacity;
using ggml_sycl::layer_block_kv_device;
using ggml_sycl::plan_device_kv_fit;
using ggml_sycl::plan_optional_layout_yield;
using ggml_sycl::plan_runtime_kv_demotion;
using ggml_sycl::plan_runtime_kv_residency;
using ggml_sycl::rs_buffer_bytes;
using ggml_sycl::rs_layer_desc;
using ggml_sycl::tlsf_allocator;

// Helper: n_layers alternating geometry like GPT-OSS (even = full-attn, odd = SWA).
//
// llama.cpp-3aos: layer_kv_bytes is now a REAL per-layer
// vector (kv_per_layer/kv_per_swa_layer scalars removed) -- this helper
// still builds a uniform-width vector (kv_full on every even layer, kv_swa
// on every odd one) so every existing case below keeps its original
// numbers; that uniformity is the TEST's choice, not something the
// production type assumes.
static kv_demotion_input make_input(size_t budget, size_t vram, size_t kv_full, size_t kv_swa, int n_layers) {
    kv_demotion_input in;
    in.vram_budget = budget;
    in.vram_bytes  = vram;
    in.kv_device.assign(n_layers, 0);  // all on device 0
    in.swa_layer_mask.assign(n_layers, 0);
    in.layer_kv_bytes.assign(n_layers, kv_full);
    for (int l = 1; l < n_layers; l += 2) {
        in.swa_layer_mask[l] = 1;
        in.layer_kv_bytes[l] = kv_swa;
    }
    return in;
}

// A dense residency (index = layer id) as a plan's kv_device map.
// One optional layout copy of copy_bytes with free_bytes free beside it, in a
// zone of its own: releasing it frees one extent of both.
static kv_optional_layouts one_copy_beside_free(size_t free_bytes, size_t copy_bytes) {
    kv_optional_layouts copies;
    copies.zone.emplace_back(free_bytes + copy_bytes);
    copies.copies.push_back({ 0, copies.zone[0].allocate(copy_bytes) });
    copies.bytes.push_back(copy_bytes);
    return copies;
}

// Case 29's zone: four primaries (40 MB) each followed by its copy (48 MB),
// sealed by one more primary, with 8 MB free on top.
static kv_optional_layouts interleaved_copies() {
    const size_t        mb = 1024 * 1024;
    kv_optional_layouts copies;
    copies.zone.emplace_back(400 * mb);
    for (int k = 0; k < 4; ++k) {
        (void) copies.zone[0].allocate(40 * mb);
        copies.copies.push_back({ 0, copies.zone[0].allocate(48 * mb) });
        copies.bytes.push_back(48 * mb);
    }
    (void) copies.zone[0].allocate(40 * mb);
    return copies;
}

static std::unordered_map<int, int> kv_device_map(const std::vector<int> & kv_device) {
    std::unordered_map<int, int> map;
    for (size_t l = 0; l < kv_device.size(); ++l) {
        map[(int) l] = kv_device[l];
    }
    return map;
}

// Row sizes for the types the cases use, from ggml's own table (type ids and block
// sizes of ggml.h); the TU links no ggml-base, so the rule is restated here.
enum { TEST_TYPE_F32 = 0, TEST_TYPE_F16 = 1, TEST_TYPE_Q8_0 = 8 };

static size_t test_row_size(int32_t type, int64_t n_elements) {
    switch (type) {
        case TEST_TYPE_F32:
            return static_cast<size_t>(n_elements) * 4;
        case TEST_TYPE_F16:
            return static_cast<size_t>(n_elements) * 2;
        case TEST_TYPE_Q8_0:
            return static_cast<size_t>(n_elements / 32) * 34;
        default:
            // A type this table lacks must not read as "0 bytes" and slip through
            // an equality of zeros.
            std::fprintf(stderr, "test_row_size: unknown type id %d\n", (int) type);
            std::abort();
    }
}

// ---------------------------------------------------------------------------
// kv_region_fit cases (llama.cpp-moua H2, H3, H6).  Every case runs the fit on a
// geometry copied out of the real allocator and replays its carve on that
// allocator, so fit == carve is checked against tlsf_allocator, not a model of it.
// ---------------------------------------------------------------------------
namespace krt = kv_region_test;
using ggml_sycl::kv_head_slot_request;
using ggml_sycl::kv_layer_alloc_bytes;
using ggml_sycl::kv_layer_slot_request;
using ggml_sycl::kv_region_fit;
using ggml_sycl::kv_region_fit_result;
using ggml_sycl::kv_region_request;
using krt::KiB;
using krt::MiB;

static kv_region_request layers_request(const std::vector<uint32_t> & full,
                                        size_t                        full_bytes,
                                        const std::vector<uint32_t> & swa       = {},
                                        size_t                        swa_bytes = 0) {
    kv_region_request r;
    for (uint32_t l : full) {
        kv_layer_slot_request s;
        s.layer    = l;
        s.group    = ggml_sycl::KV_SLOT_FULL;
        s.kv_bytes = full_bytes;
        r.layers.push_back(s);
    }
    for (uint32_t l : swa) {
        kv_layer_slot_request s;
        s.layer    = l;
        s.group    = ggml_sycl::KV_SLOT_SWA;
        s.kv_bytes = swa_bytes;
        r.layers.push_back(s);
    }
    return r;
}

static kv_head_slot_request head_slot(const char * cohort,
                                      uint32_t     index,
                                      size_t       size,
                                      bool         names_weight = false,
                                      uint64_t     owner        = 1) {
    kv_head_slot_request h;
    h.cohort       = cohort;
    h.index        = index;
    h.size         = size;
    h.owner        = owner;
    h.names_weight = names_weight;
    return h;
}

static std::vector<uint32_t> iota_layers(uint32_t from, uint32_t to) {
    std::vector<uint32_t> v;
    for (uint32_t l = from; l < to; ++l) {
        v.push_back(l);
    }
    return v;
}

static size_t device_layers(const kv_region_fit_result & f) {
    size_t n = 0;
    for (const auto & l : f.layers) {
        n += l.device ? 1 : 0;
    }
    return n;
}

// The smallest k with gap + the top k ladder entries >= need.
static size_t min_prefix(const std::vector<size_t> & ladder, size_t gap, size_t need) {
    size_t acc = gap;
    for (size_t k = 0; k <= ladder.size(); ++k) {
        if (acc >= need) {
            return k;
        }
        if (k < ladder.size()) {
            acc += ladder[k];
        }
    }
    return SIZE_MAX;
}

// H2's A2 zone: a 64 MiB context block at the top, a gap of 84.1 MiB under it,
// seven yieldable optional tenants (2870 MiB) below the gap, weights below
// those.  After the full yield the zone has 2954.1 MiB for KV and head slots.
struct a2_zone {
    krt::device_model   dev;
    std::vector<size_t> ladder;    // highest first
    size_t              gap  = 84 * MiB + 100 * KiB;
    size_t              room = 0;  // gap + every tenant
};

static a2_zone make_a2_zone() {
    a2_zone      a;
    const size_t tenants[7] = { 370 * MiB, 500 * MiB, 400 * MiB, 600 * MiB, 300 * MiB, 200 * MiB, 500 * MiB };
    size_t       total      = 0;
    for (size_t t : tenants) {
        total += t;
    }
    const size_t size = 1024 * MiB + total + a.gap + 64 * MiB;
    a.dev.tlsfs.emplace_back(size);
    krt::zone_model & z = a.dev.tlsfs[0];
    z.context(64 * MiB);
    z.weight(1024 * MiB);
    for (int i = 6; i >= 0; --i) {
        z.optional_tenant(tenants[i]);
    }
    for (size_t t : tenants) {
        a.ladder.push_back(t);
    }
    a.room = a.gap + total;
    return a;
}

// H2 A2: device = 23 - ceil(max(0, H - 10.1 MiB) / 128 MiB), host = 32 - device,
// and a yield prefix equal to the minimal strict prefix covering
// 128 MiB * device + H - gap.
static int case_h2_a2() {
    const size_t slack   = 10 * MiB + 100 * KiB;  // 10.1 MiB
    const size_t hs[3]   = { 0, 10 * MiB, 11 * MiB };
    const size_t want[3] = { 23, 23, 22 };
    for (int c = 0; c < 3; ++c) {
        a2_zone a = make_a2_zone();
        CHECK_EQ(a.room, 23 * 128 * MiB + slack, "h2: A2's room is 23 slots and 10.1 MiB");
        kv_region_request r = layers_request(iota_layers(0, 32), 128 * MiB);
        if (hs[c] != 0) {
            r.head_slots.push_back(head_slot("ring", 0, hs[c]));
        }
        const kv_region_fit_result f = kv_region_fit(a.dev.snapshot(), r);
        CHECK(f.fits, "h2: A2 fits");
        CHECK_EQ(device_layers(f), want[c], "h2: A2 device layers");
        CHECK_EQ(f.layers.size() - device_layers(f), 32 - want[c], "h2: A2 host layers");
        for (size_t l = 0; l < f.layers.size(); ++l) {
            CHECK(f.layers[l].device == (l < want[c]), "h2: the highest-indexed layers are the ones demoted");
        }
        const size_t need = 128 * MiB * want[c] + hs[c];
        CHECK_EQ(f.yield_prefix[0], min_prefix(a.ladder, a.gap, need),
                 "h2: the yield prefix is the minimal strict prefix");
        CHECK(a.dev.replay(f), "h2: A2's carve lands at the fit's offsets");
        for (size_t l = want[c]; l < 32; ++l) {
            // With the head slot removed one more layer would have stayed (H = 11 MiB),
            // and none would at H = 0 and 10 MiB: layer 22 is the head slot's, the rest are capacity.
            const uint8_t cause = (c == 2 && l == 22) ? ggml_sycl::KV_DEMOTE_HEAD_SLOT : ggml_sycl::KV_DEMOTE_CAPACITY;
            CHECK_EQ(f.layers[l].cause, cause, "h2: the demotion cause");
        }
    }
    return 0;
}

// The RED: the same zone with its top optional copy leased.  The ladder ends at a
// block that cannot be released, so nothing yields and every layer demotes.
static int case_h2_a2_red() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(1024 * MiB + 2870 * MiB + 84 * MiB + 64 * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(64 * MiB);
    z.weight(1024 * MiB);
    const size_t tenants[7] = { 370 * MiB, 500 * MiB, 400 * MiB, 600 * MiB, 300 * MiB, 200 * MiB, 500 * MiB };
    for (int i = 6; i >= 0; --i) {
        z.optional_tenant(tenants[i], /*leased=*/i == 0);
    }
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), layers_request(iota_layers(0, 32), 128 * MiB));
    CHECK(f.fits, "h2 red: the fit itself succeeds");
    CHECK_EQ(device_layers(f), 0, "h2 red: expected 0 device layers behind a leased top copy (23 when it yields)");
    CHECK_EQ(f.yield_prefix[0], 0, "h2 red: nothing yields");
    return 0;
}

// P4: seeded random zones and requests.  For every request the fit's plan must
// carve at the fit's offsets on the real allocator, and the demotion must be
// minimal: asking again with one more layer cannot place everything.
struct lcg {
    uint64_t s;

    explicit lcg(uint64_t x) : s(x * 2862933555777941757ull + 3037000493ull) {}

    uint32_t next(uint32_t n) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t) ((s >> 33) % n);
    }
};

// Oracles NOT in this property, deferred (the L3 reviews' m-2): the minimality check
// below asks the fit again, so it is self-referential, and jehw's byte-total oracle
// over a kv_zone_snapshot is absent until the production census exists (L4).  KV
// refusal has no independent oracle of its own either; the first-head refusal below
// and i6's single-gap arithmetic are the independent ones.
static int case_h2_property() {
    size_t placed  = 0;
    size_t refused = 0;
    size_t yielded = 0;
    size_t demoted = 0;
    size_t refits  = 0;
    for (uint64_t seed = 1; seed <= 600; ++seed) {
        lcg               rng(seed);
        krt::device_model dev;
        const uint32_t    n_tlsf = 1 + rng.next(3);
        for (uint32_t t = 0; t < n_tlsf; ++t) {
            struct item {
                bool   optional;
                bool   leased;
                size_t size;
            };

            std::vector<size_t> ctx;
            for (uint32_t i = rng.next(3); i > 0; --i) {
                ctx.push_back((4 + rng.next(60)) * MiB);
            }
            std::vector<item> items;
            for (uint32_t i = 1 + rng.next(4); i > 0; --i) {
                items.push_back({ false, false, (20 + rng.next(180)) * MiB });
            }
            for (uint32_t i = rng.next(5); i > 0; --i) {
                items.push_back({ true, rng.next(4) == 0, (20 + rng.next(280)) * MiB });
            }
            for (size_t i = items.size(); i > 1; --i) {
                std::swap(items[i - 1], items[rng.next((uint32_t) i)]);
            }
            const size_t gap   = rng.next(400) * MiB + 256 * rng.next(64);
            size_t       total = gap;
            for (size_t c : ctx) {
                total += c;
            }
            for (const item & it : items) {
                total += it.size;
            }
            dev.tlsfs.emplace_back(total + 4 * MiB);
            krt::zone_model & z  = dev.tlsfs.back();
            z.takes_kv           = t == 0 || rng.next(2);
            z.takes_weight_named = t + 1 == n_tlsf || rng.next(2);
            for (size_t c : ctx) {
                const size_t off = z.context(c);
                if (off != SIZE_MAX && rng.next(3) == 0) {
                    z.retain(off, c);  // a context-side block handed back as a retained run
                }
            }
            std::vector<std::pair<size_t, bool>> carved;  // offset, freeable
            for (const item & it : items) {
                const size_t off = it.optional ? z.optional_tenant(it.size, it.leased) : z.weight(it.size);
                carved.push_back({ off, !it.leased });
            }
            for (auto & c : carved) {
                if (c.second && rng.next(10) < 3) {
                    z.free(c.first);
                }
            }
            // Another transaction's pending ranges, anywhere in the TLSF: the fit must
            // plan around them and the replay must find nothing placed on them.
            for (uint32_t i = rng.next(3); i > 0; --i) {
                const size_t lo = 256 * (size_t) rng.next((uint32_t) (z.size() / 256 - 1));
                z.pend(lo, std::min((1 + (size_t) rng.next(48)) * MiB, z.size() - lo));
            }
        }
        kv_region_request r;
        const uint32_t    n_layers = 2 + rng.next(13);
        for (uint32_t l = 0; l < n_layers; ++l) {
            kv_layer_slot_request s;
            s.layer         = l;
            s.group         = (l % 2 == 0) ? ggml_sycl::KV_SLOT_FULL : ggml_sycl::KV_SLOT_SWA;
            s.kv_bytes      = (s.group == ggml_sycl::KV_SLOT_FULL ? 8 + rng.next(120) : 1 + rng.next(8)) * MiB;
            s.sidecar_bytes = rng.next(10) < 3 ? MiB : 0;
            r.layers.push_back(s);
        }
        for (uint32_t h = rng.next(4); h > 0; --h) {
            r.head_slots.push_back(
                head_slot("head", (uint32_t) r.head_slots.size(), (1 + rng.next(40)) * MiB, rng.next(3) == 0));
        }
        const ggml_sycl::shared_zone_geometry geo = dev.snapshot();
        const kv_region_fit_result            f   = kv_region_fit(geo, r);
        // The refusal oracle, from the census and the pending ranges alone: the first
        // head slot (nothing placed before it) is refused exactly when no chunk the
        // commit can carve, on a TLSF that may take it, holds it.
        if (!r.head_slots.empty()) {
            const size_t need  = kv_layer_alloc_bytes(r.head_slots[0].size);
            bool         chunk = false;
            for (const krt::zone_model & z : dev.tlsfs) {
                if (r.head_slots[0].names_weight ? z.takes_weight_named : z.takes_kv) {
                    for (size_t c : z.carvable_chunks()) {
                        chunk = chunk || c >= need;
                    }
                }
            }
            const bool refused0 = std::find(f.refused_heads.begin(), f.refused_heads.end(), 0) != f.refused_heads.end();
            CHECK(refused0 != chunk, "property: the first head slot is refused iff no carvable chunk holds it");
        }
        if (!f.fits) {
            CHECK(!f.refused_heads.empty(), "property: a refusal names its head slots");
            ++refused;
            continue;
        }
        ++placed;
        krt::device_model before_yield = dev;
        if (!dev.replay(f)) {
            std::fprintf(stderr, "property: seed %llu\n", (unsigned long long) seed);
            CHECK(false, "property: the carve lands at the fit's offsets");
        }
        // Slots lie inside their extents and no two placements overlap.
        std::vector<std::pair<size_t, size_t>> ranges;  // tlsf-tagged absolute [lo, hi)
        for (const auto & e : f.extents) {
            ranges.push_back({ e.tlsf * (1ull << 40) + e.offset, e.tlsf * (1ull << 40) + e.offset + e.size });
        }
        for (const auto & h : f.heads) {
            if (!h.reused) {
                ranges.push_back({ h.tlsf * (1ull << 40) + h.offset, h.tlsf * (1ull << 40) + h.offset + h.size });
            }
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            CHECK(ranges[i - 1].second <= ranges[i].first, "property: placements are disjoint");
        }
        for (const auto & l : f.layers) {
            if (l.device) {
                CHECK(l.slot_offset + l.size <= f.extents[l.extent].size,
                      "property: a slot is never split across extents");
            }
        }
        // The demotion order: per group a suffix by layer, and SWA only after every full layer.
        bool any_full_device = false;
        bool swa_host        = false;
        for (const auto & l : f.layers) {
            const bool full = (l.layer % 2) == 0;
            if (full && l.device) {
                any_full_device = true;
            }
            if (!full && !l.device) {
                swa_host = true;
            }
        }
        CHECK(!(any_full_device && swa_host), "property: an SWA layer demotes only after every full layer");
        for (int parity = 0; parity < 2; ++parity) {
            bool seen_host = false;
            for (const auto & l : f.layers) {
                if ((int) (l.layer % 2) != parity) {
                    continue;
                }
                CHECK(!(seen_host && l.device), "property: a group's host layers are its highest indices");
                seen_host = seen_host || !l.device;
            }
        }
        // Minimality: restore the last layer the loop demoted and the fit cannot
        // place everything.  Demotion order is full attention highest first, then SWA.
        int64_t last = -1;
        for (int group = 0; group < 2; ++group) {
            for (size_t i = f.layers.size(); i-- > 0;) {
                if (((f.layers[i].layer % 2) == (group == 0 ? 0u : 1u)) && !f.layers[i].device) {
                    last = (int64_t) f.layers[i].layer;
                }
            }
        }
        if (last >= 0) {
            ++demoted;
            kv_region_request r2 = r;
            r2.layers.clear();
            for (const auto & l : r.layers) {
                bool keep = (int64_t) l.layer == last;
                for (const auto & p : f.layers) {
                    keep = keep || (p.layer == l.layer && p.device);
                }
                if (keep) {
                    r2.layers.push_back(l);
                }
            }
            const kv_region_fit_result f2 = kv_region_fit(geo, r2);
            CHECK(f2.fits && device_layers(f2) < r2.layers.size(),
                  "property: with the last demoted layer restored the fit cannot place everything");
        }
        yielded += f.yield_prefix[0] + f.buried_released.size();

        // The commit re-fit, after the plan's own yield: it may place only inside the
        // ranges the plan carved, and what it places must carve as it says.
        {
            for (size_t t = 0; t < before_yield.tlsfs.size(); ++t) {
                std::vector<size_t> buried;
                for (const auto & b : f.buried_released) {
                    if (b.tlsf == t) {
                        buried.push_back(b.offset);
                    }
                }
                before_yield.tlsfs[t].yield(f.yield_prefix[t], buried);
            }
            // The plan's ranges: every placement records one, including a retained run it
            // claimed whole and an after-KV charge, which carve nothing.
            for (const auto & op : f.carve_order) {
                before_yield.tlsfs[op.tlsf].own(op.offset, op.size);
            }
            kv_region_request re         = r;
            re.commit_refit              = true;
            const kv_region_fit_result g = kv_region_fit(before_yield.snapshot(), re);
            if (!before_yield.replay(g)) {
                std::fprintf(stderr, "property: re-fit seed %llu\n", (unsigned long long) seed);
                CHECK(false, "property: the commit re-fit carves at its offsets, clear of pending ranges");
            }
            // Nothing changed between the plan and the re-fit but the plan's own carve, so
            // the re-fit must realize exactly the plan: the same layers on the device, and
            // no head the plan placed refused (plan == reality, in the refusing direction).
            if (!g.fits || device_layers(g) != device_layers(f)) {
                std::fprintf(stderr, "property: re-fit seed %llu fits=%d layers %zu plan %zu\n",
                             (unsigned long long) seed, (int) g.fits, device_layers(g), device_layers(f));
            }
            CHECK(g.fits && g.refused_heads.empty(), "property: the re-fit refuses no head the plan placed");
            CHECK_EQ(device_layers(g), device_layers(f), "property: the re-fit places exactly the plan's layers");
            for (size_t i = 0; i < f.layers.size(); ++i) {
                CHECK(g.layers[i].device == f.layers[i].device, "property: and the same layers, not just as many");
            }
            ++refits;
        }
    }
    CHECK(placed > 100 && refused > 0 && yielded > 50 && demoted > 50 && refits > 20,
          "property: the generator exercises every outcome");
    return 0;
}

// A zone of one gap: a context block at the top, weights at the bottom, `gap`
// between them.
static krt::device_model gap_zone(size_t gap) {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + gap);
    dev.tlsfs[0].context(MiB);
    dev.tlsfs[0].weight(16 * MiB);
    return dev;
}

// Commit under churn (r2 N-I7; r4 I8).  The plan's yield is done and its ranges
// are recorded; the re-fit may place only inside them.
static int case_h2_churn() {
    kv_region_request r = layers_request(iota_layers(0, 32), 128 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 10 * MiB));
    // Churn outside the ranges: a weight at the bottom and another transaction's range.
    {
        a2_zone                    a    = make_a2_zone();
        const kv_region_fit_result plan = kv_region_fit(a.dev.snapshot(), r);
        CHECK(plan.fits && device_layers(plan) == 23, "churn: the plan");
        krt::zone_model & z = a.dev.tlsfs[0];
        z.yield(plan.yield_prefix[0]);
        size_t lowest = SIZE_MAX;
        for (const auto & op : plan.carve_order) {
            if (op.carve) {
                z.own(op.offset, op.size);
                lowest = std::min(lowest, op.offset);
            }
        }
        // The minimal prefix leaves 100 KiB of slack under the ranges: a weight and
        // another transaction's pending range live there.
        const size_t slack_lo = z.anchor() - z.allocator().gap_below(z.anchor());
        CHECK_EQ(lowest - slack_lo, 100 * KiB, "churn: 100 KiB of slack under the planned ranges");
        CHECK(z.weight(64 * KiB) == slack_lo, "churn: the weight lands in the slack");
        z.pend(slack_lo + 64 * KiB, 32 * KiB);
        kv_region_request re         = r;
        re.commit_refit              = true;
        const kv_region_fit_result f = kv_region_fit(a.dev.snapshot(), re);
        CHECK(f.fits && device_layers(f) == 23, "churn: outside the ranges nothing changes");
        CHECK_EQ(f.yield_prefix[0], 0, "churn: a re-fit yields nothing");
        for (const auto & op : plan.carve_order) {
            bool found = false;
            for (const auto & o2 : f.carve_order) {
                found = found || (o2.offset == op.offset && o2.size == op.size && o2.kind == op.kind);
            }
            CHECK(found, "churn: the commit carves exactly the planned offsets");
        }
        CHECK(a.dev.replay(f), "churn: the carve succeeds beside another transaction's range");
    }
    // A weight that landed in the planned room: the intersection shrinks, the
    // re-fit demotes inside the ranges and ends.
    {
        a2_zone                    a    = make_a2_zone();
        const kv_region_fit_result plan = kv_region_fit(a.dev.snapshot(), r);
        krt::zone_model &          z    = a.dev.tlsfs[0];
        z.yield(plan.yield_prefix[0]);
        size_t lowest = SIZE_MAX;
        for (const auto & op : plan.carve_order) {
            if (op.carve) {
                z.own(op.offset, op.size);
                lowest = std::min(lowest, op.offset);
            }
        }
        // A weight that took everything below the lowest range and 130 MiB of it.
        const size_t gap_lo = z.anchor() - z.allocator().gap_below(z.anchor());
        CHECK(z.weight(lowest - gap_lo + 130 * MiB) != SIZE_MAX, "churn: the weight lands in the planned room");
        kv_region_request re         = r;
        re.commit_refit              = true;
        const kv_region_fit_result f = kv_region_fit(a.dev.snapshot(), re);
        CHECK(f.fits, "churn: the re-fit fits");
        CHECK_EQ(device_layers(f), 21, "churn: 130 MiB lost from the bottom of the planned room costs two slots");
        CHECK(a.dev.replay(f), "churn: and what is left carves at the fit's offsets");
    }
    return 0;
}

// The ring rows are a head slot (r3 I5): KV plus the rows does not fit, KV minus
// one layer plus the rows does, so the fit demotes one layer and places the rows;
// and rows that do not fit with every KV layer on the host refuse, naming them.
static int case_h2_ring_head_slot() {
    krt::device_model dev = gap_zone(900 * MiB);
    kv_region_request r   = layers_request(iota_layers(0, 8), 100 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 150 * MiB));
    kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits, "ring: fits after one demotion");
    CHECK_EQ(device_layers(f), 7, "ring: one layer demoted for the rows");
    CHECK(!f.layers[7].device, "ring: the highest layer is the one");
    CHECK_EQ(f.layers[7].cause, ggml_sycl::KV_DEMOTE_HEAD_SLOT, "ring: the rows are the cause");
    CHECK(f.heads[0].size == 150 * MiB, "ring: the rows are placed");
    CHECK(dev.replay(f), "ring: the carve lands at the fit's offsets");

    krt::device_model small = gap_zone(100 * MiB);
    f                       = kv_region_fit(small.snapshot(), r);
    CHECK(!f.fits, "ring: rows that do not fit with every KV layer on the host refuse");
    CHECK(f.refused_heads == std::vector<size_t>({ 0 }), "ring: the refusal names the rows");
    CHECK_EQ(f.tlsf_free[0], 100 * MiB, "ring: and the TLSF's free space");
    return 0;
}

// Head slots are mandatory and only tiers 1-3 serve them; the byte budget is not
// an input at all.
static int case_h2_head_slot_rules() {
    // A head slot that would fit only by yielding an optional copy is refused,
    // while KV of the same size yields.
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 100 * MiB + 10 * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(MiB);
    z.weight(16 * MiB);
    z.optional_tenant(100 * MiB);
    kv_region_request r;
    r.head_slots.push_back(head_slot("moe_control", 0, 50 * MiB));
    kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(!f.fits && f.refused_heads == std::vector<size_t>({ 0 }), "head: no yield is made for a head slot");
    r = layers_request({ 0 }, 50 * MiB);
    f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 1 && f.yield_prefix[0] == 1, "head: KV of the same size yields the copy");
    CHECK(dev.replay(f), "head: and carves where the fit said");

    // A head slot that fits the geometry is placed whatever any budget says: KV
    // demotes only by what the geometry needs.
    krt::device_model two = gap_zone(250 * MiB);
    r                     = layers_request(iota_layers(0, 4), 50 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 100 * MiB));
    f = kv_region_fit(two.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 3, "head: the geometry alone decides");
    return 0;
}

// The ring-held yield (§2.7): the rows the context already holds are reused in
// place and counted once.  The jehw-merge order held the rows while the yield
// modelled the KV zone and then admitted them again: it places fewer layers.
static int case_h2_ring_held_yield() {
    auto zone = [](uint64_t owner) {
        krt::device_model dev;
        dev.tlsfs.emplace_back(500 * MiB + 16 * MiB + 600 * MiB + 800 * MiB);
        krt::zone_model & z    = dev.tlsfs[0];
        const size_t      rows = z.context(500 * MiB);
        z.weight(16 * MiB);
        z.optional_tenant(400 * MiB);
        z.optional_tenant(400 * MiB);
        z.reserve(ggml_sycl::demand_scope::CONTEXT, owner, "ring", 0, rows, 500 * MiB);
        return dev;
    };
    kv_region_request r = layers_request(iota_layers(0, 8), 200 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 500 * MiB, false, /*owner=*/1));
    krt::device_model          held = zone(1);
    const kv_region_fit_result f    = kv_region_fit(held.snapshot(), r);
    CHECK(f.fits && f.heads[0].reused && f.heads[0].reservation == 0, "ring held: the rows are reused in place");
    CHECK_EQ(device_layers(f), 7, "ring held: 1400 MiB of room holds seven 200 MiB layers with the rows counted once");
    for (const auto & op : f.carve_order) {
        CHECK(op.kind != ggml_sycl::KV_CARVE_HEAD, "ring held: nothing is carved for the rows");
    }
    CHECK(held.replay(f), "ring held: the carve lands at the fit's offsets");
    // Another owner's rows are not ours: the same zone with the rows re-admitted.
    krt::device_model          other = zone(2);
    const kv_region_fit_result g     = kv_region_fit(other.snapshot(), r);
    CHECK(g.fits && !g.heads[0].reused, "ring held: another context's rows are not reused");
    CHECK_EQ(device_layers(g), 4, "ring held: re-admitting the rows (900 MiB left) places fewer layers");
    return 0;
}

// The rows are per context, in REGION (rulings §M32 I-1, I-2).
static int case_h2_rows_per_context() {
    krt::device_model dev = gap_zone(2000 * MiB);
    kv_region_request a   = layers_request(iota_layers(0, 4), 100 * MiB);
    a.head_slots.push_back(head_slot("ring", 0, 100 * MiB, false, 1));  // -ub 512
    kv_region_request b = layers_request(iota_layers(0, 4), 100 * MiB);
    b.head_slots.push_back(head_slot("ring", 0, 200 * MiB, false, 2));  // -ub 1024
    const kv_region_fit_result fa = kv_region_fit(dev.snapshot(), a);
    CHECK(fa.fits && dev.replay(fa), "rows: the first context places its rows");
    dev.tlsfs[0].reserve(ggml_sycl::demand_scope::CONTEXT, 1, "ring", 0, fa.heads[0].offset, fa.heads[0].size);
    const kv_region_fit_result fb = kv_region_fit(dev.snapshot(), b);
    CHECK(fb.fits && !fb.heads[0].reused, "rows: the second context's rows are its own");
    CHECK(dev.replay(fb), "rows: and carve at the fit's offsets");
    CHECK(fb.heads[0].offset + fb.heads[0].size <= fa.heads[0].offset ||
              fa.heads[0].offset + fa.heads[0].size <= fb.heads[0].offset,
          "rows: the two contexts' rows are disjoint");
    CHECK_EQ(fa.heads[0].size, 100 * MiB, "rows: at its own n_ubatch");
    CHECK_EQ(fb.heads[0].size, 200 * MiB, "rows: at its own n_ubatch");
    // Closing the -ub 1024 context releases its rows and leaves the other's.
    dev.tlsfs[0].free(fb.heads[0].offset);
    CHECK(dev.tlsfs[0].allocator().tag_at(fa.heads[0].offset) == ggml_sycl::SHARED_ZONE_TAG_CONTEXT,
          "rows: closing one context leaves the other's rows allocated");
    CHECK(dev.tlsfs[0].invariants(), "rows: the allocator stays consistent");
    // The context's rows grow: the smaller slot is superseded and the publish releases it.
    kv_region_request grown       = a;
    grown.head_slots[0].size      = 300 * MiB;
    const kv_region_fit_result fg = kv_region_fit(dev.snapshot(), grown);
    CHECK(fg.fits && !fg.heads[0].reused, "rows: a larger claim is carved new");
    CHECK(fg.superseded.size() == 1 && fg.superseded[0].reservation == 0, "rows: the smaller slot is superseded");
    return 0;
}

// Demotion order and free_after_full_kv (§2.4.1; rulings §M32 I-2, §M60).  A
// GPT-OSS-shaped fixture with the fixture's own room and heads: 12 full layers of
// 128 MiB and 12 SWA layers of 2.5 MiB, both ring rows (540 MiB), a compute head
// slot and the small ones, room set so free_after_full_kv is -700.0 MiB.
static int case_h2_demotion_order() {
    const size_t          full_bytes = 128 * MiB;
    const size_t          swa_bytes  = 5 * MiB / 2;
    std::vector<uint32_t> full, swa;
    for (uint32_t l = 0; l < 24; ++l) {
        (l % 2 == 0 ? full : swa).push_back(l);
    }
    kv_region_request r = layers_request(full, full_bytes, swa, swa_bytes);
    r.head_slots.push_back(head_slot("ring_act", 0, 180 * MiB));
    r.head_slots.push_back(head_slot("ring_out", 0, 360 * MiB));
    r.head_slots.push_back(head_slot("compute", 0, 400 * MiB));
    r.head_slots.push_back(head_slot("moe_control", 0, 33024));
    r.head_slots.push_back(head_slot("onednn_pp_a", 0, 8388608));
    r.head_slots.push_back(head_slot("jzvq", 0, 223560));
    size_t heads = 0;
    for (const auto & h : r.head_slots) {
        heads += kv_layer_alloc_bytes(h.size);
    }
    CHECK_EQ(12 * full_bytes + 12 * swa_bytes, 1566 * MiB, "demotion: 1536 MiB of full KV and 30 MiB of SWA");

    krt::device_model    dev = gap_zone(heads + 866 * MiB);
    kv_region_fit_result f   = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits, "demotion: fits");
    CHECK_EQ(f.free_after_full_kv[0], -700ll * (long long) MiB, "demotion: free_after_full_kv reads -700.0 MiB");
    size_t host_full = 0;
    for (const auto & l : f.layers) {
        if (l.layer % 2 == 1) {
            CHECK(l.device, "demotion: no SWA layer is demoted");
        } else if (!l.device) {
            ++host_full;
            CHECK(l.layer >= 12, "demotion: the highest-indexed full layers go");
            CHECK_EQ(l.cause, ggml_sycl::KV_DEMOTE_HEAD_SLOT, "demotion: every cause is the head slots");
        }
    }
    CHECK_EQ(host_full, 6, "demotion: 5 x 128 = 640 is short by 60, 6 x 128 = 768 covers it");
    CHECK(f.sub_slot_holes.empty(), "demotion: no sub-slot hole, no WARN");
    CHECK(dev.replay(f), "demotion: the carve lands at the fit's offsets");

    // The same deficit with 166 MiB of the room in sub-slot weight holes demotes 7.
    krt::device_model holes;
    holes.tlsfs.emplace_back(MiB + (6 * 10 + 4 * 33 + 34) * MiB + heads + 700 * MiB);
    krt::zone_model & z = holes.tlsfs[0];
    z.context(MiB);
    std::vector<size_t> hole_offsets;
    z.weight(10 * MiB);
    for (int i = 0; i < 5; ++i) {
        hole_offsets.push_back(z.weight((i == 4 ? 34 : 33) * MiB));
        z.weight(10 * MiB);
    }
    for (size_t o : hole_offsets) {
        z.free(o);
    }
    f = kv_region_fit(holes.snapshot(), r);
    CHECK(f.fits, "demotion holes: fits");
    CHECK_EQ(f.free_after_full_kv[0], -700ll * (long long) MiB, "demotion holes: the room is the same -700.0 MiB");
    size_t demoted = 0;
    for (const auto & l : f.layers) {
        demoted += l.device ? 0 : 1;
    }
    CHECK_EQ(demoted, 7, "demotion holes: the holes take no 128 MiB slot, so 7 demote");
    CHECK_EQ(f.sub_slot_holes.size(), 5, "demotion holes: the sub-slot WARN names the five holes");
    CHECK(holes.replay(f), "demotion holes: the carve lands at the fit's offsets");
    return 0;
}

// Recurrent state (r4 I9): the RS slot is placed first, attention KV demotes
// around it, and an RS slot that cannot fit with every KV layer on the host
// refuses the transaction, naming it.
static int case_h2_recurrent_state() {
    kv_region_request r = layers_request(iota_layers(0, 8), 100 * MiB);
    r.head_slots.push_back(head_slot("rs", 0, 300 * MiB));
    krt::device_model          dev = gap_zone(1000 * MiB);
    const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 7, "rs: KV demotes around the recurrent slot");
    CHECK_EQ(f.layers[7].cause, ggml_sycl::KV_DEMOTE_HEAD_SLOT, "rs: the slot is the cause");
    CHECK(dev.replay(f), "rs: the carve lands at the fit's offsets");
    krt::device_model          tiny = gap_zone(200 * MiB);
    const kv_region_fit_result g    = kv_region_fit(tiny.snapshot(), r);
    CHECK(!g.fits && g.refused_heads == std::vector<size_t>({ 0 }), "rs: a slot that cannot fit refuses, naming it");
    return 0;
}

static std::vector<uint32_t> host_layers(const kv_region_fit_result & f) {
    std::vector<uint32_t> v;
    for (const auto & l : f.layers) {
        if (!l.device) {
            v.push_back(l.layer);
        }
    }
    return v;
}

// H3: the slot table, the demotion order, and packing that never splits a slot.
static int case_h3_slot_table_and_order() {
    // Interleaved full and SWA layers, SWA listed first: the table is full first, then SWA.
    kv_region_request r = layers_request({ 0, 2, 4 }, 100 * MiB, { 1, 3, 5 }, 10 * MiB);
    std::reverse(r.layers.begin(), r.layers.end());
    krt::device_model     dev = gap_zone(1000 * MiB);
    kv_region_fit_result  f   = kv_region_fit(dev.snapshot(), r);
    std::vector<uint32_t> order;
    for (const auto & l : f.layers) {
        order.push_back(l.layer);
    }
    CHECK(order == std::vector<uint32_t>({ 0, 2, 4, 1, 3, 5 }),
          "h3: the slot table groups full attention first, then SWA");
    CHECK(dev.replay(f), "h3: the carve lands at the fit's offsets");
    // 135 MiB: full layers 4 then 2 demote (latest first), no SWA layer.
    krt::device_model a = gap_zone(135 * MiB);
    f                   = kv_region_fit(a.snapshot(), r);
    CHECK(host_layers(f) == std::vector<uint32_t>({ 2, 4 }), "h3: latest full layers first");
    CHECK(a.replay(f), "h3: and the carve matches");
    // 90 MiB: every full layer, SWA stays.
    krt::device_model b = gap_zone(90 * MiB);
    f                   = kv_region_fit(b.snapshot(), r);
    CHECK(host_layers(f) == std::vector<uint32_t>({ 0, 2, 4 }), "h3: every full layer before any SWA");
    // 25 MiB: the SWA layers demote, latest first, after every full layer.
    krt::device_model c = gap_zone(25 * MiB);
    f                   = kv_region_fit(c.snapshot(), r);
    CHECK(host_layers(f) == std::vector<uint32_t>({ 0, 2, 4, 5 }), "h3: then the latest SWA layer");
    CHECK(c.replay(f), "h3: and the carve matches");
    return 0;
}

// Multi-extent packing: retained run first, the gap second, a weight hole last;
// a slot is never split, and each tier is a separate extent.
static int case_h3_multi_extent() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(300 * MiB + 16 * MiB + 150 * MiB + 16 * MiB + 250 * MiB);
    krt::zone_model & z        = dev.tlsfs[0];
    const size_t      retained = z.context(300 * MiB);
    z.weight(16 * MiB);
    const size_t hole = z.weight(150 * MiB);
    z.weight(16 * MiB);
    z.free(hole);
    z.retain(retained, 300 * MiB);
    const kv_region_request    r = layers_request(iota_layers(0, 6), 100 * MiB);
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 6, "multi: all six slots are placed");
    CHECK_EQ(f.extents.size(), 3, "multi: three extents");
    size_t kinds[4] = { 0, 0, 0, 0 };
    for (const auto & e : f.extents) {
        ++kinds[e.kind];
    }
    CHECK(kinds[ggml_sycl::KV_EXTENT_RETAINED] == 1 && kinds[ggml_sycl::KV_EXTENT_FRONTIER] == 1 &&
              kinds[ggml_sycl::KV_EXTENT_HOLE] == 1,
          "multi: a retained run, the frontier and a hole");
    for (size_t l = 0; l < 6; ++l) {
        const auto & lp = f.layers[l];
        const auto & e  = f.extents[lp.extent];
        CHECK(lp.slot_offset + lp.size <= e.size, "multi: a slot is never split across extents");
        const uint8_t want =
            l < 3 ? ggml_sycl::KV_EXTENT_RETAINED : (l < 5 ? ggml_sycl::KV_EXTENT_FRONTIER : ggml_sycl::KV_EXTENT_HOLE);
        CHECK_EQ(e.kind, want, "multi: retained, then the gap, then a hole");
    }
    CHECK(dev.replay(f), "multi: the carve lands at the fit's offsets");
    return 0;
}

// The cost-ordered pack: nothing yields while the gap holds the slots; when one
// must, the cheaper of the frontier prefix and a buried run goes, the frontier
// first at a tie.
static int case_h3_cost_order() {
    // [buried B][weight][T1][gap 200 MiB]
    auto zone = [](size_t b, size_t t1) {
        krt::device_model dev;
        dev.tlsfs.emplace_back(MiB + b + 16 * MiB + t1 + 200 * MiB);
        krt::zone_model & z = dev.tlsfs[0];
        z.context(MiB);
        z.optional_tenant(b);
        z.weight(16 * MiB);
        z.optional_tenant(t1);
        return dev;
    };
    krt::device_model    dev = zone(120 * MiB, 100 * MiB);
    kv_region_fit_result f   = kv_region_fit(dev.snapshot(), layers_request(iota_layers(0, 2), 100 * MiB));
    CHECK(f.fits && device_layers(f) == 2 && f.yield_prefix[0] == 0 && f.buried_released.empty(),
          "cost: the gap alone holds two");
    // Three slots: the third yields.  T1 (100) is cheaper than B (120).
    f = kv_region_fit(dev.snapshot(), layers_request(iota_layers(0, 3), 100 * MiB));
    CHECK(f.fits && device_layers(f) == 3 && f.yield_prefix[0] == 1 && f.buried_released.empty(),
          "cost: the frontier prefix is cheaper");
    CHECK(dev.replay(f), "cost: and carves where the fit said");
    // B and T1 equal: the frontier first.
    krt::device_model tie = zone(100 * MiB, 100 * MiB);
    f                     = kv_region_fit(tie.snapshot(), layers_request(iota_layers(0, 3), 100 * MiB));
    CHECK(f.yield_prefix[0] == 1 && f.buried_released.empty(), "cost: the frontier before a buried run at a tie");
    // T1 dearer than B: the buried run goes.
    krt::device_model buried = zone(100 * MiB, 150 * MiB);
    f                        = kv_region_fit(buried.snapshot(), layers_request(iota_layers(0, 3), 100 * MiB));
    CHECK(f.fits && device_layers(f) == 3 && f.yield_prefix[0] == 0 && f.buried_released.size() == 1,
          "cost: the buried run is cheaper");
    CHECK(buried.replay(f), "cost: the buried hole carves where the fit said");
    bool hole = false;
    for (const auto & e : f.extents) {
        hole = hole || e.kind == ggml_sycl::KV_EXTENT_HOLE;
    }
    CHECK(hole, "cost: the buried slot sits in a hole extent");
    return 0;
}

// The strict prefix and the two-way classification (r4 addendum D).
static int case_h3_strict_prefix() {
    // ladder: T1 yieldable, T2 leased, T3 yieldable; gap 50 MiB.
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 300 * MiB + 50 * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(MiB);
    z.weight(16 * MiB);
    z.optional_tenant(100 * MiB);                   // T3
    z.optional_tenant(100 * MiB, /*leased=*/true);  // T2
    z.optional_tenant(100 * MiB);                   // T1
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), layers_request(iota_layers(0, 3), 100 * MiB));
    CHECK(f.fits && device_layers(f) == 1, "strict: the ladder ends at the leased copy; T3 is not skipped to");
    CHECK_EQ(f.yield_prefix[0], 1, "strict: only T1 is released");
    CHECK(dev.replay(f), "strict: and carves where the fit said");
    return 0;
}

// Pending ranges are allocated; the model's first context counts its FIRST_CONTEXT ranges.
static int case_h3_pending_ranges() {
    auto zone = [](bool first_context) {
        krt::device_model dev = gap_zone(300 * MiB);
        krt::zone_model & z   = dev.tlsfs[0];
        const size_t      top = z.anchor() - 75 * MiB;
        z.pend(top - 150 * MiB, 150 * MiB, first_context);
        return dev;
    };
    const kv_region_request r     = layers_request(iota_layers(0, 3), 100 * MiB);
    krt::device_model       other = zone(false);
    kv_region_fit_result    f     = kv_region_fit(other.snapshot(), r);
    CHECK_EQ(device_layers(f), 0,
             "pending: a pending range splits the gap into two 75 MiB pieces that take no 100 MiB slot");
    CHECK(other.replay(f), "pending: the host-only plan carves nothing and replays clean");
    kv_region_request first = r;
    first.first_context     = true;
    krt::device_model mine  = zone(true);
    f                       = kv_region_fit(mine.snapshot(), first);
    CHECK_EQ(device_layers(f), 3, "pending: the first context counts its FIRST_CONTEXT ranges as free");
    CHECK(mine.replay(f), "pending: and its carve lands at the fit's offsets");
    krt::device_model foreign = zone(true);
    f                         = kv_region_fit(foreign.snapshot(), r);
    CHECK_EQ(device_layers(f), 0, "pending: another context does not");
    CHECK(foreign.replay(f), "pending: and its plan replays clean");
    return 0;
}

// forced_host, self extents, sidecars, and the after-KV charge.
static int case_h3_request_forms() {
    // forced_host layers are never promoted.
    krt::device_model dev  = gap_zone(1000 * MiB);
    kv_region_request r    = layers_request(iota_layers(0, 4), 100 * MiB);
    r.forced_host          = { 3 };
    kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && host_layers(f) == std::vector<uint32_t>({ 3 }), "forms: a forced-host layer stays on the host");
    CHECK_EQ(f.layers[3].cause, ggml_sycl::KV_DEMOTE_FORCED_HOST, "forms: its cause is the inherited demotion");

    // The packed-K sidecar is a slice of the layer's slot.
    kv_region_request s       = layers_request(iota_layers(0, 2), 100 * MiB);
    s.layers[0].sidecar_bytes = 10 * MiB + 1;
    s.layers[1].sidecar_bytes = 10 * MiB + 1;
    krt::device_model sd      = gap_zone(1000 * MiB);
    f                         = kv_region_fit(sd.snapshot(), s);
    const size_t slot         = 100 * MiB + kv_layer_alloc_bytes(10 * MiB + 1);
    CHECK(f.fits && f.layers[0].size == slot && f.layers[1].slot_offset == slot,
          "forms: the sidecar is part of the slot");
    CHECK_EQ(f.layers[0].sidecar_offset, 100 * MiB, "forms: at the aligned end of the KV bytes");
    CHECK(sd.replay(f), "forms: the carve is one extent of both slots");

    // Self extents: layers 0 and 1 are already carved by this call.
    krt::device_model         self = gap_zone(450 * MiB);  // 250 MiB remain under the self extent
    const size_t              off  = self.tlsfs[0].context(200 * MiB);
    kv_region_request         q    = layers_request(iota_layers(0, 4), 100 * MiB);
    ggml_sycl::kv_self_extent ext;
    ext.tlsf   = 0;
    ext.offset = off;
    ext.size   = 200 * MiB;
    ext.slots  = {
        { 0, 0,         100 * MiB },
        { 1, 100 * MiB, 100 * MiB }
    };
    q.self_extents.push_back(ext);
    f = kv_region_fit(self.snapshot(), q);
    CHECK(f.fits && device_layers(f) == 4, "forms: the self layers are device, the others fit the remaining 250 MiB");
    CHECK(f.layers[0].extent != SIZE_MAX && f.extents[f.layers[0].extent].kind == ggml_sycl::KV_EXTENT_SELF,
          "forms: a self extent is copied in");
    CHECK_EQ(f.layers[1].slot_offset, 100 * MiB, "forms: with its slot offsets");

    // After-KV: ascending admission, declined when it does not fit, never a demotion cause.
    krt::device_model ak = gap_zone(300 * MiB + 65 * MiB);  // three layers, 65 MiB left
    kv_region_request a  = layers_request(iota_layers(0, 4), 100 * MiB);
    a.forced_host        = { 3 };
    a.after_kv           = {
        { 0, 50 * MiB },
        { 1, 10 * MiB },
        { 2, 30 * MiB },
        { 3, 20 * MiB }
    };
    f = kv_region_fit(ak.snapshot(), a);
    CHECK(f.fits && device_layers(f) == 3, "after-KV: three layers stay; the charge demotes nothing");
    CHECK(f.after_kv[1].admitted && f.after_kv[2].admitted, "after-KV: the 10 and 30 MiB terms are admitted");
    CHECK(!f.after_kv[0].admitted, "after-KV: the 50 MiB term no longer fits and is declined");
    CHECK(!f.after_kv[3].admitted, "after-KV: a host layer's term is not charged");
    CHECK(ak.replay(f), "after-KV: the carve of what is carved lands at the fit's offsets");
    return 0;
}

// Carve mirroring (r1 M1): the extent is top-carved, and the same rule decides
// the offset when the gap is exactly the size or one MIN_BLOCK_SIZE over it.
static int case_h3_carve_mirroring() {
    for (size_t extra : { size_t(0), size_t(256), size_t(512), 4 * MiB }) {
        krt::device_model dev = gap_zone(150 * MiB + extra);
        kv_region_request r   = layers_request(iota_layers(0, 1), 100 * MiB);
        r.head_slots.push_back(head_slot("ring", 0, 50 * MiB));
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
        CHECK(f.fits && device_layers(f) == 1, "mirror: both fit");
        CHECK(dev.replay(f), "mirror: the offsets are the allocator's");
        CHECK_EQ(f.extents[0].size, 100 * MiB, "mirror: a top carve takes exactly the slots' bytes");
    }
    return 0;
}

// H6: two weight TLSFs plus the tail KV TLSF.
static int case_h6_n_chunk() {
    krt::device_model dev;
    for (int t = 0; t < 3; ++t) {
        dev.tlsfs.emplace_back(MiB + 16 * MiB + 250 * MiB);
        dev.tlsfs[t].context(MiB);
        dev.tlsfs[t].weight(16 * MiB);
    }
    dev.tlsfs[0].takes_kv           = false;
    dev.tlsfs[0].takes_weight_named = false;
    dev.tlsfs[1].takes_kv           = false;  // the last weight chunk
    dev.tlsfs[1].takes_weight_named = true;
    dev.tlsfs[2].takes_kv           = true;   // the tail chunk's KV TLSF
    dev.tlsfs[2].takes_weight_named = false;
    kv_region_request r             = layers_request(iota_layers(0, 2), 100 * MiB);
    r.head_slots.push_back(head_slot("moe_control", 0, 20 * MiB, /*names_weight=*/true));
    r.head_slots.push_back(head_slot("ring", 0, 30 * MiB, /*names_weight=*/false));
    kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 2, "h6: fits");
    CHECK_EQ(f.heads[0].tlsf, 1, "h6: a WEIGHT-naming head slot goes to the last weight chunk");
    CHECK_EQ(f.heads[1].tlsf, 2, "h6: a KV-naming head slot goes to the KV TLSF");
    for (const auto & e : f.extents) {
        CHECK_EQ(e.tlsf, 2, "h6: KV goes to the KV TLSF");
    }
    CHECK(dev.replay(f), "h6: each carve lands at the fit's offsets on its own TLSF");

    // Greedy across TLSFs: both weight chunks take KV too; slots fill the first TLSF with room.
    krt::device_model many;
    for (int t = 0; t < 3; ++t) {
        many.tlsfs.emplace_back(MiB + 16 * MiB + 250 * MiB);
        many.tlsfs[t].context(MiB);
        many.tlsfs[t].weight(16 * MiB);
    }
    kv_region_request k = layers_request(iota_layers(0, 6), 100 * MiB);
    f                   = kv_region_fit(many.snapshot(), k);
    CHECK(f.fits && device_layers(f) == 6, "h6: six slots over three TLSFs");
    size_t per_tlsf[3] = { 0, 0, 0 };
    for (const auto & l : f.layers) {
        ++per_tlsf[f.extents[l.extent].tlsf];
    }
    CHECK(per_tlsf[0] == 2 && per_tlsf[1] == 2 && per_tlsf[2] == 2,
          "h6: each TLSF takes the two slots it has room for, in order");
    CHECK_EQ(f.layers[0].extent, f.layers[1].extent, "h6: slots of one TLSF share its extent");
    CHECK(many.replay(f), "h6: fit == carve across TLSFs and extents");
    CHECK_EQ(f.free_after_full_kv[0] + f.free_after_full_kv[1] + f.free_after_full_kv[2],
             3 * 250ll * (long long) MiB - 600ll * (long long) MiB,
             "h6: free_after_full_kv is signed per TLSF and sums to room minus KV");
    return 0;
}

// ---------------------------------------------------------------------------
// Fit == carve under pending ranges, retained runs and ring growth (the L3 review).
// A placement the commit cannot realize is a placement the fit must not claim:
// tlsf_allocator::allocate_below carves at the top of the free block directly
// under an ALLOCATED block (or the region end), and a pending range is not an
// allocated block.  So a free piece whose top edge is another transaction's
// pending range is not carvable, and everything the fit places must also lie
// clear of every such range.
// ---------------------------------------------------------------------------
static int case_i1_pending_ranges_carve() {
    const kv_region_request two = layers_request(iota_layers(0, 2), 100 * MiB);
    {
        // A 300 MiB gap with a 10 MiB pending range in the middle.  The piece above
        // the range ends at the allocated context block and takes one 100 MiB slot;
        // the piece below ends at the pending range, which the commit cannot carve under.
        krt::device_model dev    = gap_zone(300 * MiB);
        krt::zone_model & z      = dev.tlsfs[0];
        const size_t      gap_lo = z.anchor() - 300 * MiB;
        z.pend(gap_lo + 145 * MiB, 10 * MiB);
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), two);
        CHECK(f.fits, "pending carve: fits");
        CHECK_EQ(device_layers(f), 1, "pending carve: the 145 MiB piece above the range takes one slot");
        CHECK(f.layers[0].device && !f.layers[1].device, "pending carve: the highest layer is the one demoted");
        CHECK_EQ(f.layers[1].cause, ggml_sycl::KV_DEMOTE_CAPACITY, "pending carve: for capacity");
        CHECK(dev.replay(f), "pending carve: the plan carves at the fit's offsets, clear of the range");
        // The room counted is the room the commit can carve: 145 MiB, not 290.
        CHECK_EQ(f.free_after_full_kv[0], -55ll * (long long) MiB,
                 "pending carve: free_after_full_kv counts only the carvable 145 MiB against 200 MiB of KV");
        kv_region_request refused = two;
        refused.layers.clear();
        refused.head_slots.push_back(head_slot("ring", 0, 200 * MiB));
        // replay() above carved into `dev`; the refusal reads a fresh zone of the same shape.
        krt::device_model fresh = gap_zone(300 * MiB);
        fresh.tlsfs[0].pend(gap_lo + 145 * MiB, 10 * MiB);
        const kv_region_fit_result h = kv_region_fit(fresh.snapshot(), refused);
        CHECK(!h.fits && h.refused_heads == std::vector<size_t>({ 0 }),
              "pending carve: a 200 MiB head slot is refused although 290 MiB are free in all");
        CHECK_EQ(h.tlsf_free[0], 145 * MiB, "pending carve: and the TLSF's free room is the carvable 145 MiB");
    }
    {
        // The range abuts the context block: nothing above it, and the piece below
        // ends at the range.
        krt::device_model dev = gap_zone(300 * MiB);
        krt::zone_model & z   = dev.tlsfs[0];
        z.pend(z.anchor() - 10 * MiB, 10 * MiB);
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), two);
        CHECK_EQ(device_layers(f), 0, "pending carve: a range under the anchor leaves nothing the commit can carve");
        CHECK(dev.replay(f), "pending carve: an empty plan replays");
    }
    {
        // The range at the bottom of the gap: the whole piece above it is carvable.
        krt::device_model dev = gap_zone(300 * MiB);
        krt::zone_model & z   = dev.tlsfs[0];
        z.pend(z.anchor() - 300 * MiB, 10 * MiB);
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), two);
        CHECK_EQ(device_layers(f), 2, "pending carve: 290 MiB above a range at the bottom hold both slots");
        CHECK(dev.replay(f), "pending carve: and they carve at the fit's offsets");
    }
    {
        // A region that ends off the 256 grid cannot be top-carved (carve_gap refuses),
        // so its gap is not claimed.
        krt::device_model dev;
        dev.tlsfs.emplace_back(16 * MiB + 300 * MiB + 100);
        dev.tlsfs[0].weight(16 * MiB);
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), two);
        CHECK_EQ(device_layers(f), 0, "off-grid: a gap whose top is off the grid is not carvable");
        CHECK(dev.replay(f), "off-grid: an empty plan replays");
    }
    return 0;
}

// Retained runs obey the same ranges (the L3 review I-2): another transaction's pending
// range is allocated, and a commit re-fit may place only inside its own ranges.
static int case_i2_retained_runs_respect_ranges() {
    auto zone = [](size_t pend_lo, size_t pend_size, size_t own_lo, size_t own_size) {
        krt::device_model dev;
        dev.tlsfs.emplace_back(16 * MiB + 300 * MiB);
        krt::zone_model & z = dev.tlsfs[0];
        z.weight(16 * MiB);
        const size_t off = z.context(300 * MiB);
        z.retain(off, 300 * MiB);
        if (pend_size != 0) {
            z.pend(off + pend_lo, pend_size);
        }
        if (own_size != 0) {
            z.own(off + own_lo, own_size);
        }
        return dev;
    };
    const kv_region_request three = layers_request(iota_layers(0, 3), 100 * MiB);
    {
        krt::device_model          dev = zone(0, 300 * MiB, 0, 0);
        const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), three);
        CHECK_EQ(device_layers(f), 0, "retained: a run fully under another transaction's pending range is not claimed");
        CHECK(dev.replay(f), "retained: and nothing overlaps");
    }
    {
        // Pending over the middle 100 MiB: a fragment of 100 MiB on each side.
        krt::device_model          dev = zone(100 * MiB, 100 * MiB, 0, 0);
        const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), three);
        CHECK_EQ(device_layers(f), 2, "retained: the two fragments around a pending range take one slot each");
        CHECK(dev.replay(f), "retained: neither lies on the range");
    }
    {
        // A commit re-fit whose own ranges lie elsewhere may not use the run.
        krt::device_model dev = zone(0, 0, 0, 0);
        dev.tlsfs[0].own(0, 4 * MiB);
        kv_region_request re         = three;
        re.commit_refit              = true;
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), re);
        CHECK_EQ(device_layers(f), 0, "retained: a re-fit does not place in a run outside its own ranges");
    }
    {
        // Inside its own range the run is usable: a re-fit owning the middle 100 MiB takes one slot.
        krt::device_model dev        = zone(0, 0, 100 * MiB, 100 * MiB);
        kv_region_request re         = three;
        re.commit_refit              = true;
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), re);
        CHECK_EQ(device_layers(f), 1, "retained: a re-fit places inside the part of the run it owns");
        CHECK(dev.replay(f), "retained: within the owned range");
    }
    return 0;
}

// Ring growth is the first cause, and the superseded slot is free room together with
// what is next to it (the L3 review I-3).
static int case_i3_ring_growth_cause() {
    // Joined: weights | 130 MiB gap | the context's old 60 MiB ring row | context block.
    // Apart: weights | 100 MiB hole | weight wall | the old row | context block, so the
    // row has no free neighbour and the room is a hole far from it.
    auto zone = [](bool apart) {
        krt::device_model dev;
        dev.tlsfs.emplace_back(16 * MiB + (apart ? 100 + 10 : 130) * MiB + 60 * MiB + MiB);
        krt::zone_model & z = dev.tlsfs[0];
        z.context(MiB);
        const size_t row = z.context(60 * MiB);
        z.weight(16 * MiB);
        if (apart) {
            const size_t hole = z.weight(100 * MiB);
            z.weight(10 * MiB);
            z.free(hole);
        }
        z.reserve(ggml_sycl::demand_scope::CONTEXT, 1, "ring", 0, row, 60 * MiB);
        return dev;
    };
    kv_region_request r = layers_request(iota_layers(0, 2), 100 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 80 * MiB));
    {
        // Counting the old row free the room is 130 + 60 - 80 = 110 MiB: layer 0 stays on
        // the device, so its demotion is the ring's growth.  Layer 1 is short with every
        // cause removed.
        krt::device_model          dev = zone(false);
        const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), r);
        CHECK(f.fits && f.superseded.size() == 1, "ring growth: the grown slot supersedes the old row");
        CHECK_EQ(device_layers(f), 0, "ring growth: 130 - 80 = 50 MiB holds no 100 MiB slot");
        CHECK_EQ(f.layers[0].cause, ggml_sycl::KV_DEMOTE_RING_GROWTH,
                 "ring growth: layer 0 would stay if the old row were free");
        CHECK_EQ(f.layers[1].cause, ggml_sycl::KV_DEMOTE_CAPACITY, "ring growth: layer 1 is capacity");
        CHECK(dev.replay(f), "ring growth: the head slot carves at the fit's offset");
    }
    {
        // With a wall under the row there is nothing to merge with: the old row is a 60 MiB
        // piece on its own, which no 100 MiB slot fits, so the demotion is the head slot's
        // (without the 80 MiB, the 100 MiB hole holds layer 0).
        krt::device_model          dev = zone(true);
        const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), r);
        CHECK(f.fits && device_layers(f) == 0, "ring growth, apart: the rows leave 20 MiB of the hole");
        CHECK_EQ(f.layers[0].cause, ggml_sycl::KV_DEMOTE_HEAD_SLOT,
                 "ring growth, apart: an isolated old row cannot hold the slot, so the head slot is the cause");
        CHECK_EQ(f.layers[1].cause, ggml_sycl::KV_DEMOTE_CAPACITY, "ring growth, apart: layer 1 is capacity");
        CHECK(dev.replay(f), "ring growth, apart: the head slot carves at the fit's offset");
    }
    return 0;
}

// Within a tier the best fit decides, across TLSFs too (the L3 review m-3).
static int case_i4_best_fit_across_tlsfs() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 500 * MiB);
    dev.tlsfs[0].context(MiB);
    dev.tlsfs[0].weight(16 * MiB);
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 130 * MiB);
    dev.tlsfs[1].context(MiB);
    dev.tlsfs[1].weight(16 * MiB);
    const kv_region_request    r = layers_request(iota_layers(0, 1), 100 * MiB);
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 1, "best fit: the slot is placed");
    CHECK_EQ(f.extents[0].tlsf, 1, "best fit: the 130 MiB gap is the tightest room that holds the slot");
    CHECK(dev.replay(f), "best fit: and carves there");

    // The same in tier 1: of two retained runs on different TLSFs, the tighter one takes it.
    krt::device_model retained;
    for (int t = 0; t < 2; ++t) {
        retained.tlsfs.emplace_back(16 * MiB + (t == 0 ? 500 : 130) * MiB);
        krt::zone_model & z = retained.tlsfs[t];
        z.weight(16 * MiB);
        const size_t off = z.context((t == 0 ? 500 : 130) * MiB);
        z.retain(off, (t == 0 ? 500 : 130) * MiB);
    }
    const kv_region_fit_result g = kv_region_fit(retained.snapshot(), r);
    CHECK(g.fits && device_layers(g) == 1, "best fit, retained: the slot is placed");
    CHECK_EQ(g.extents[0].tlsf, 1, "best fit, retained: the 130 MiB run is the tightest");
    return 0;
}

// Best fit must not strand a constrained head: the reviewer's repro.  T0 takes KV
// only and has a 20 MiB gap; T1 takes KV and weight-named heads and has 10 MiB.  Head
// a (KV, 10 MiB) then head b (names weight, 10 MiB): a global best fit puts a in T1's
// tighter room and b has nowhere to go.  The constrained head is placed first.
static int case_i8_constrained_head_first() {
    auto zones = [] {
        krt::device_model dev;
        dev.tlsfs.emplace_back(MiB + 16 * MiB + 20 * MiB);
        dev.tlsfs[0].takes_weight_named = false;
        dev.tlsfs[0].context(16 * MiB);
        dev.tlsfs.emplace_back(MiB + 16 * MiB + 10 * MiB);
        dev.tlsfs[1].takes_weight_named = true;
        dev.tlsfs[1].context(16 * MiB);
        return dev;
    };
    kv_region_request r;
    r.head_slots.push_back(head_slot("a", 0, 10 * MiB, false));
    r.head_slots.push_back(head_slot("b", 1, 10 * MiB, true));
    krt::device_model          dev = zones();
    const kv_region_fit_result f   = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && f.refused_heads.empty(), "constrained: both heads are placed");
    CHECK_EQ(f.heads[0].tlsf, 0, "constrained: the KV head takes the TLSF only it may use");
    CHECK_EQ(f.heads[1].tlsf, 1, "constrained: the weight-naming head takes the one TLSF that may take it");
    CHECK(dev.replay(f), "constrained: and both carve at the fit's offsets");

    // The order of the request does not matter, and best fit still decides within a set.
    kv_region_request swapped;
    swapped.head_slots.push_back(head_slot("b", 0, 10 * MiB, true));
    swapped.head_slots.push_back(head_slot("a", 1, 10 * MiB, false));
    krt::device_model          dev2 = zones();
    const kv_region_fit_result g    = kv_region_fit(dev2.snapshot(), swapped);
    CHECK(g.fits && g.heads[0].tlsf == 1 && g.heads[1].tlsf == 0, "constrained: request order is irrelevant");
    CHECK(dev2.replay(g), "constrained: swapped, the carve still lands");

    // Two unconstrained heads on the same pair still use best fit: the 10 MiB gap first.
    kv_region_request both;
    both.head_slots.push_back(head_slot("a", 0, 10 * MiB, false));
    krt::device_model          dev3 = zones();
    const kv_region_fit_result h    = kv_region_fit(dev3.snapshot(), both);
    CHECK(h.fits && h.heads[0].tlsf == 1, "constrained: an unconstrained head alone takes the tighter room");

    // The commit re-fit of the first plan places both heads again.
    krt::device_model before = zones();
    for (size_t t = 0; t < before.tlsfs.size(); ++t) {
        before.tlsfs[t].yield(f.yield_prefix[t], {});
    }
    for (const auto & op : f.carve_order) {
        before.tlsfs[op.tlsf].own(op.offset, op.size);
    }
    kv_region_request re         = r;
    re.commit_refit              = true;
    const kv_region_fit_result k = kv_region_fit(before.snapshot(), re);
    CHECK(k.fits && k.refused_heads.empty(), "constrained: the re-fit refuses neither head");
    CHECK(k.heads[0].tlsf == 0 && k.heads[1].tlsf == 1, "constrained: and puts them where the plan did");
    return 0;
}

// A re-fit's rooms are shrunk to what the plan put in them, which changes which room
// a greedy best fit finds tightest.  Retained runs of 10 and 12 MiB, slots of 4, 6 and
// 5 MiB: the plan puts 4 and 6 in the 10 and 5 in the 12 (the 12 keeps 7 spare).  A
// re-fit whose rooms are 10 and 5 puts the 4 in the 5, and the 5 then has no room.
static int case_i9_refit_exact_in_shrunk_rooms() {
    auto zones = [] {
        krt::device_model dev;
        dev.tlsfs.emplace_back(16 * MiB + 22 * MiB);
        krt::zone_model & z = dev.tlsfs[0];
        z.weight(16 * MiB);
        const size_t x = z.context(10 * MiB);
        const size_t y = z.context(12 * MiB);
        z.retain(x, 10 * MiB);
        z.retain(y, 12 * MiB);
        return dev;
    };
    kv_region_request r             = layers_request(iota_layers(0, 3), 4 * MiB);
    r.layers[1].kv_bytes            = 6 * MiB;
    r.layers[2].kv_bytes            = 5 * MiB;
    krt::device_model          dev  = zones();
    const kv_region_fit_result plan = kv_region_fit(dev.snapshot(), r);
    CHECK(plan.fits && device_layers(plan) == 3, "shrunk: the plan places every slot");
    CHECK(dev.replay(plan), "shrunk: and carves");
    krt::device_model before = zones();
    for (const auto & op : plan.carve_order) {
        before.tlsfs[op.tlsf].own(op.offset, op.size);
    }
    kv_region_request re         = r;
    re.commit_refit              = true;
    const kv_region_fit_result g = kv_region_fit(before.snapshot(), re);
    CHECK(g.fits, "shrunk: the re-fit fits");
    CHECK_EQ(device_layers(g), 3, "shrunk: and places every slot the plan placed");
    CHECK(before.replay(g), "shrunk: and carves at its offsets");
    return 0;
}

// A greedy best fit of the heads can refuse one the plan placed (the L3 r2 fuzz,
// seed 8989): own rooms of 33 and 48 MiB, heads of 27 (names weight), 33 and 16 MiB.
// Best fit puts the 27 in the 33 and the 33 in the 48, and the 16 has nowhere to go;
// the placement the plan had is 27 and 16 in the 48, the 33 in the 33.
static int case_i12_refit_heads_search() {
    krt::device_model dev;
    for (size_t gap : { 33 * MiB, 48 * MiB }) {
        dev.tlsfs.emplace_back(MiB + 16 * MiB + gap);
        krt::zone_model & z = dev.tlsfs.back();
        z.context(MiB);
        z.weight(16 * MiB);
        z.own(z.anchor() - gap, gap);
    }
    kv_region_request r;
    r.head_slots.push_back(head_slot("h", 0, 27 * MiB, true));
    r.head_slots.push_back(head_slot("h", 1, 33 * MiB, false));
    r.head_slots.push_back(head_slot("h", 2, 16 * MiB, false));
    r.commit_refit               = true;
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && f.refused_heads.empty(), "heads search: the re-fit refuses no head the rooms can hold");
    CHECK(f.heads[0].tlsf == 1 && f.heads[2].tlsf == 1 && f.heads[1].tlsf == 0,
          "heads search: 27 and 16 share the 48, the 33 takes the 33");
    CHECK(dev.replay(f), "heads search: and the carve lands at the fit's offsets");

    // Control: with the rooms one MiB too small for the three there is no placement,
    // and the search says so instead of inventing one.
    krt::device_model tight;
    for (size_t gap : { 33 * MiB, 42 * MiB }) {
        tight.tlsfs.emplace_back(MiB + 16 * MiB + gap);
        krt::zone_model & z = tight.tlsfs.back();
        z.context(MiB);
        z.weight(16 * MiB);
        z.own(z.anchor() - gap, gap);
    }
    const kv_region_fit_result g = kv_region_fit(tight.snapshot(), r);
    CHECK(!g.fits && !g.refused_heads.empty(), "heads search: control: rooms that cannot hold all three refuse");
    return 0;
}

// An optional tenant counts as room only where a yield could use it (the L3 r2 review
// I-2): here nothing carvable holds 100 MiB, so the layer demotes and the room the
// fit reports is the 21 MiB gap plus the 50 MiB of the hole above its pending range.
static int case_i10_opt_room_needs_a_carvable_span() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 40 * MiB + 100 * MiB + 60 * MiB + 30 * MiB + 20 * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(16 * MiB);
    z.weight(40 * MiB);
    const size_t opt  = z.optional_tenant(100 * MiB);
    const size_t hole = z.weight(60 * MiB);
    z.weight(30 * MiB);
    z.free(hole);
    z.pend(opt + 100 * MiB, 10 * MiB);  // the bottom of the hole: the tenant's span now tops under it
    const kv_region_request    r = layers_request(iota_layers(0, 1), 100 * MiB);
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 0, "opt room: no carvable room holds 100 MiB, so the layer demotes");
    CHECK_EQ(f.free_after_full_kv[0], (71ll - 100ll) * (long long) MiB,
             "opt room: the tenant under the pending range is not counted (21 + 50 - 100 MiB)");
    CHECK(f.buried_released.empty(), "opt room: and no yield is claimed for it");

    // The tenant's span tops at the FREE piece above it, not at the tenant: a pending
    // range at the top of that hole leaves the tenant and the hole both unreachable.
    krt::device_model above;
    above.tlsfs.emplace_back(MiB + 16 * MiB + 40 * MiB + 100 * MiB + 60 * MiB + 30 * MiB + 20 * MiB);
    krt::zone_model & u = above.tlsfs[0];
    u.context(16 * MiB);
    u.weight(40 * MiB);
    u.optional_tenant(100 * MiB);
    const size_t hole3 = u.weight(60 * MiB);
    u.weight(30 * MiB);
    u.free(hole3);
    u.pend(hole3 + 50 * MiB, 10 * MiB);
    const kv_region_fit_result t = kv_region_fit(above.snapshot(), r);
    CHECK(t.fits && device_layers(t) == 0, "opt room: a span topped by a non-carvable hole holds no layer");
    CHECK_EQ(t.free_after_full_kv[0], (21ll - 100ll) * (long long) MiB,
             "opt room: neither the tenant nor the hole under the range counts (only the 21 MiB gap)");

    // Control: without the range the tenant is reachable and counts, and a yield places the layer.
    krt::device_model open;
    open.tlsfs.emplace_back(MiB + 16 * MiB + 40 * MiB + 100 * MiB + 60 * MiB + 30 * MiB + 20 * MiB);
    krt::zone_model & w = open.tlsfs[0];
    w.context(16 * MiB);
    w.weight(40 * MiB);
    w.optional_tenant(100 * MiB);
    const size_t hole2 = w.weight(60 * MiB);
    w.weight(30 * MiB);
    w.free(hole2);
    const kv_region_fit_result g = kv_region_fit(open.snapshot(), r);
    CHECK(g.free_after_full_kv[0] > f.free_after_full_kv[0], "opt room: control: with no range the room is larger");
    return 0;
}

// An optional tenant a pending range covers is a hard barrier (still an allocated
// block): the free piece under it stays carvable.
static int case_i11_free_piece_under_a_cut_tenant() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + 150 * MiB + 50 * MiB + 10 * MiB + 5 * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(MiB);
    z.weight(16 * MiB);
    const size_t hole = z.weight(150 * MiB);
    const size_t opt  = z.optional_tenant(50 * MiB, false);
    z.weight(10 * MiB);
    z.free(hole);
    z.pend(opt + 10 * MiB, 5 * MiB);  // inside the tenant: it is a barrier whole, not a soft one
    const kv_region_request    r = layers_request(iota_layers(0, 1), 100 * MiB);
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 1, "cut tenant: the hole under the tenant takes the layer");
    CHECK(dev.replay(f), "cut tenant: and the carve lands under the allocated tenant");
    return 0;
}

// A sub-slot hole a head slot already filled is not reported (the L3 review m-7).
static int case_i5_sub_slot_hole_after_heads() {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + (10 + 33 + 10 + 33 + 10 + 20) * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(MiB);
    z.weight(10 * MiB);
    const size_t h0 = z.weight(33 * MiB);
    z.weight(10 * MiB);
    const size_t h1 = z.weight(33 * MiB);
    z.weight(10 * MiB);
    z.free(h0);
    z.free(h1);
    kv_region_request r = layers_request(iota_layers(0, 4), 128 * MiB);
    r.head_slots.push_back(head_slot("ring", 0, 33 * MiB));
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK(f.fits && device_layers(f) == 0, "sub-slot: a 20 MiB gap and two 33 MiB holes take no 128 MiB slot");
    CHECK_EQ(f.heads[0].offset, h0, "sub-slot: the head slot fills the lower hole");
    CHECK_EQ(f.sub_slot_holes.size(), 1, "sub-slot: only the hole the head slot left is reported");
    CHECK_EQ(f.sub_slot_holes[0].offset, h1, "sub-slot: it is the upper one");
    CHECK(dev.replay(f), "sub-slot: the head slot carves at the fit's offset");
    return 0;
}

// A yield whose freed block would be topped by a pending range gains no room, so it is not
// the cheapest yield, whatever it costs.
static int case_i7_yield_span_under_a_range() {
    // weight | optional A (60) | 80 MiB hole | weight | optional B (100) | weight | context
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + (10 + 60 + 80 + 10 + 100 + 10) * MiB);
    krt::zone_model & z = dev.tlsfs[0];
    z.context(MiB);
    z.weight(10 * MiB);
    z.optional_tenant(60 * MiB);
    const size_t hole = z.weight(80 * MiB);
    z.weight(10 * MiB);
    const size_t b = z.optional_tenant(100 * MiB);
    z.weight(10 * MiB);
    z.free(hole);
    // A pending range in the middle of the hole: the 40 MiB under it plus tenant A would
    // make 100 MiB, but the commit cannot carve under the range.
    z.pend(hole + 40 * MiB, 10 * MiB);
    const kv_region_request    r = layers_request(iota_layers(0, 1), 100 * MiB);
    const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
    CHECK_EQ(device_layers(f), 1, "yield span: the slot is placed");
    CHECK_EQ(f.buried_released.size(), 1, "yield span: one buried tenant is released");
    CHECK_EQ(f.buried_released[0].offset, b, "yield span: it is B, not the cheaper A under the range");
    CHECK(dev.replay(f), "yield span: the carve lands at the fit's offset");
    return 0;
}

// An independent arithmetic oracle (the spec's second oracle, in the form that needs no
// fit machinery): on a zone of one gap the device layers are what is left after
// demoting, in the stated order, until the slots' sum fits the gap.
static int case_i6_single_gap_oracle() {
    size_t demoting = 0;
    for (uint64_t seed = 1; seed <= 300; ++seed) {
        lcg               rng(seed + 9000);
        const size_t      gap = (size_t) rng.next(1500) * MiB + 256 * rng.next(64);
        krt::device_model dev = gap_zone(gap);
        kv_region_request r;
        const uint32_t    n = 1 + rng.next(16);
        for (uint32_t l = 0; l < n; ++l) {
            kv_layer_slot_request s;
            s.layer         = l;
            s.group         = (l % 3 == 2) ? ggml_sycl::KV_SLOT_SWA : ggml_sycl::KV_SLOT_FULL;
            s.kv_bytes      = (s.group == ggml_sycl::KV_SLOT_FULL ? 16 + rng.next(240) : 2 + rng.next(20)) * MiB;
            s.sidecar_bytes = rng.next(10) < 3 ? 3 * MiB + rng.next(1000) : 0;
            r.layers.push_back(s);
        }
        std::vector<size_t> total(n);
        std::vector<char>   on(n, 1);
        size_t              sum = 0;
        for (uint32_t l = 0; l < n; ++l) {
            total[l] = kv_layer_alloc_bytes(r.layers[l].kv_bytes) +
                       (r.layers[l].sidecar_bytes ? kv_layer_alloc_bytes(r.layers[l].sidecar_bytes) : 0);
            sum += total[l];
        }
        for (int group = 0; group < 2 && sum > gap; ++group) {
            for (size_t i = n; i-- > 0 && sum > gap;) {
                if ((int) r.layers[i].group == group) {
                    on[i] = 0;
                    sum -= total[i];
                }
            }
        }
        const kv_region_fit_result f = kv_region_fit(dev.snapshot(), r);
        CHECK(f.fits, "oracle: fits");
        demoting += std::count(on.begin(), on.end(), 0) != 0 ? 1 : 0;
        // The fit lists layers in slot-table order (full attention, then SWA), so look each up by layer.
        std::vector<char> got(n, 0);
        for (const auto & pl : f.layers) {
            got[pl.layer] = pl.device ? 1 : 0;
        }
        for (uint32_t l = 0; l < n; ++l) {
            if (got[l] != on[l]) {
                std::fprintf(stderr, "oracle: seed %llu layer %u, gap %zu, slots:", (unsigned long long) seed, l, gap);
                for (uint32_t k = 0; k < n; ++k) {
                    std::fprintf(stderr, " %zu%s/%d/%d", total[k],
                                 r.layers[k].group == ggml_sycl::KV_SLOT_SWA ? "s" : "f", (int) on[k], (int) got[k]);
                }
                std::fprintf(stderr, "\n");
                CHECK(false, "oracle: the device layers are what the slots' sum leaves after the stated demotion");
            }
        }
        CHECK(dev.replay(f), "oracle: and the carve is the allocator's");
    }
    CHECK(demoting > 100, "oracle: the seeds demote often enough to mean something");
    return 0;
}

// Runs a case unless KRT_ONLY names another one (a RED capture runs one case alone).
static int run_case(const char * name, int (*fn)()) {
    const char * only = std::getenv("KRT_ONLY");
    if (only != nullptr && std::string(only) != name) {
        return 0;
    }
    const int rc = fn();
    if (rc != 0) {
        std::fprintf(stderr, "case %s failed\n", name);
    }
    return rc;
}

int main() {
    // 1. identity: already fits -> zero demotions, bytes unchanged
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 900, 100, 10, 4));
        CHECK(r.fits, "case 1: fits");
        CHECK(r.demoted_layers.empty(), "case 1: no demotions");
        CHECK_EQ(r.vram_bytes_after, 900, "case 1: vram_bytes_after unchanged");
    }
    // 2. exact boundary: vram_bytes == budget is ADMITTED with zero demotions
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1000, 100, 10, 4));
        CHECK(r.fits, "case 2: exact boundary fits");
        CHECK(r.demoted_layers.empty(), "case 2: no demotions at exact boundary");
    }
    // 3. one-layer overshoot demotes exactly the LAST full-attn layer
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1050, 100, 10, 6));
        CHECK(r.fits, "case 3: fits after one demotion");
        CHECK_EQ(r.demoted_layers.size(), 1, "case 3: exactly one demotion");
        CHECK_EQ(r.demoted_layers[0], 4, "case 3: demotes last full-attn layer (4)");
        CHECK_EQ(r.vram_bytes_after, 950, "case 3: vram_bytes_after reduced by one layer");
        CHECK_EQ(r.host_kv_bytes_added, 100, "case 3: host_kv_bytes_added == one layer");
    }
    // 4. multi-layer overshoot demotes latest-first until fit
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1250, 100, 10, 6));
        CHECK(r.fits, "case 4: fits after multiple demotions");
        CHECK(r.demoted_layers == (std::vector<int>{ 4, 2, 0 }), "case 4: demotes latest-first 4,2,0");
        CHECK_EQ(r.vram_bytes_after, 950, "case 4: vram_bytes_after settles at 950");
    }
    // 5. SWA layers are never demoted, even when insufficient
    {
        auto r = plan_runtime_kv_demotion(make_input(100, 500, 100, 10, 4));
        // full-attn layers 0,2 demoted (200 recovered) -> 300 > 100 budget: no fit
        CHECK(!r.fits, "case 5: cannot fit without demoting SWA layers");
        // Non-vacuous: an empty demoted_layers would satisfy the loop below by
        // never iterating, so pin the count before checking parity.
        CHECK_EQ(r.demoted_layers.size(), 2, "case 5: both full-attn layers demoted");
        for (int l : r.demoted_layers) {
            CHECK(l % 2 == 0, "case 5: only full-attn (even) layers demoted");
        }
    }
    // 6. layers already on host are not re-demoted or double-counted
    {
        auto in         = make_input(1000, 1050, 100, 10, 6);
        in.kv_device[4] = -1;  // pre-demoted
        auto r          = plan_runtime_kv_demotion(in);
        CHECK(r.fits, "case 6: fits");
        CHECK(r.demoted_layers == (std::vector<int>{ 2 }), "case 6: only layer 2 newly demoted");
    }
    // 7. a single layer's KV exceeds the remaining total VRAM: refuse without
    // wrapping vram_bytes_after (size_t underflow would report a huge total
    // instead of "did not fit").
    {
        kv_demotion_input in;
        in.vram_budget = 0;
        in.vram_bytes  = 50;
        in.layer_kv_bytes.assign(3, 100);
        in.kv_device.assign(3, 0);
        in.swa_layer_mask.assign(3, 0);
        auto r = plan_runtime_kv_demotion(in);
        CHECK(!r.fits, "case 7: cannot fit without wrapping");
        CHECK_EQ(r.vram_bytes_after, 50, "case 7: vram_bytes_after unchanged, no underflow");
        CHECK_EQ(r.host_kv_bytes_added, 0, "case 7: no bytes added");
    }
    // 8. llama.cpp-3aos: a device-resident, non-SWA layer with
    // layer_kv_bytes[l] == 0 (e.g. a SHARED layer that holds no
    // independent KV of its own) must never be demoted -- there is nothing
    // for it to give back, and counting it would silently manufacture
    // phantom VRAM savings.
    {
        auto in              = make_input(1000, 1050, 100, 10, 6);
        in.layer_kv_bytes[4] = 0;  // layer 4 (full-attn slot) holds no KV of its own
        auto r               = plan_runtime_kv_demotion(in);
        // Layer 4 cannot be demoted (0 bytes); the next full-attn layer (2)
        // is demoted instead to reach the same 950 target.
        CHECK(r.fits, "case 8: fits by demoting the next real full-attn layer");
        CHECK(r.demoted_layers == (std::vector<int>{ 2 }), "case 8: layer 4 (0 bytes) skipped, layer 2 demoted");
        CHECK_EQ(r.vram_bytes_after, 950, "case 8: vram_bytes_after reduced by layer 2's real bytes");
        CHECK_EQ(r.host_kv_bytes_added, 100, "case 8: host_kv_bytes_added == layer 2's bytes, not layer 4's 0");
    }
    // 9. The KV+weight capacity of a device is the shared arena zone when one
    // exists, never the larger budget the zone was carved from: the B70 at
    // GGML_SYCL_VRAM_BUDGET_PCT=22 has a 5136 MB budget but a 3856 MB zone.
    {
        const size_t mb = 1024 * 1024;
        CHECK_EQ(kv_weight_capacity(5136 * mb, 3856 * mb), 3856 * mb, "case 9: zone below budget wins");
        CHECK_EQ(kv_weight_capacity(3000 * mb, 3856 * mb), 3000 * mb, "case 9: budget below zone wins");
        CHECK_EQ(kv_weight_capacity(5136 * mb, 0), 5136 * mb, "case 9: no arena zone -> budget");
    }
    // 10. The measured split (Mistral 7B, n_ctx=2048): device 0 holds layers
    // 0-29 with 3683.8 MB of weights in a 3856 MB zone, device 1 holds 30-31.
    // As plan_runtime_kv_residency() calls it, the capacity is the live KV
    // headroom the weights left (172.2 MB) less the slack of the 30 resident
    // layers, which fits 21 of device 0's 8 MB layers: exactly 21..29 demote,
    // and device 1's layers are never touched by device 0's pass.
    {
        const size_t        mb = 1024 * 1024;
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = (3856 * mb - (3683 * mb + 819 * mb / 1000)) - 30 * kv_alloc_slack_per_layer;
        in.layer_kv_bytes.assign(32, 8 * mb);
        in.kv_device.assign(32, 0);
        in.kv_device[30] = 1;
        in.kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 10: device 0 fits after demotion");
        CHECK_EQ(r.demoted_layers.size(), 9, "case 10: nine layers demoted");
        CHECK_EQ(r.demoted_layers.front(), 29, "case 10: latest device-0 layer first");
        CHECK_EQ(r.demoted_layers.back(), 21, "case 10: stops at layer 21");
        CHECK_EQ(r.host_kv_bytes_added, 72 * mb, "case 10: 72 MB of KV moves to host");

        in.device   = 1;
        in.capacity = (676 * mb - 234 * mb) - 2 * kv_alloc_slack_per_layer;
        auto r1     = plan_device_kv_fit(in);
        CHECK(r1.fits, "case 10: device 1 fits");
        CHECK(r1.demoted_layers.empty(), "case 10: device 1 demotes nothing");
    }
    // 11. The same split at n_ctx=1024 (4 MB layers) fits with no demotion.
    {
        const size_t        mb = 1024 * 1024;
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = (3856 * mb - (3683 * mb + 819 * mb / 1000)) - 30 * kv_alloc_slack_per_layer;
        in.layer_kv_bytes.assign(32, 4 * mb);
        in.kv_device.assign(32, 0);
        in.kv_device[30] = 1;
        in.kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 11: fits");
        CHECK(r.demoted_layers.empty(), "case 11: nothing demoted");
    }
    // 12. No KV headroom left at all (the weights filled the zone): every layer
    // moves to the host tier and the context still fits, with no KV on the
    // device -- never shrink context.
    {
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = 0;
        in.layer_kv_bytes.assign(2, 10);
        in.kv_device.assign(2, 0);
        in.swa_layer_mask.assign(2, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 12: fits with every layer on the host tier");
        CHECK_EQ(r.demoted_layers.size(), 2, "case 12: both layers demoted");
        CHECK_EQ(r.vram_bytes_after, 0, "case 12: no KV left on the device");
    }
    // 13. A block's KV owner: the one device every KV-holding layer uses, -2
    // when they disagree (a demoted layer inside a device block), -1 with no
    // KV at all. Layers with no KV of their own do not vote.
    {
        std::vector<int>    kv_dev   = { 0, 0, -1, 0, 1, 1 };
        std::vector<size_t> kv_bytes = { 8, 8, 8, 0, 8, 8 };
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 0, 1), 0, "case 13: uniform block");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 0, 2), -2, "case 13: demoted layer makes it mixed");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 3, 3), -1, "case 13: no KV -> -1");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 3, 5), 1, "case 13: KV-less layer does not vote");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 4, 9), 1, "case 13: range clamped to the vectors");
    }
    // 14. Only a new KV shape re-decides residency. The load-time plan always
    // does; a republish that changes nothing the KV size depends on (the auto
    // micro-batch trial) keeps the published residency.
    {
        kv_shape published;
        published.runtime   = true;
        published.n_ctx     = 2048;
        published.n_seq_max = 1;
        kv_shape next       = published;
        CHECK(!kv_shape_changed(published, next), "case 14: same shape keeps residency");
        kv_shape load = published;
        load.runtime  = false;
        CHECK(kv_shape_changed(load, next), "case 14: the load-time plan is always re-decided");
        next.n_ctx = 512;
        CHECK(kv_shape_changed(published, next), "case 14: n_ctx");
        next           = published;
        next.n_seq_max = 4;
        CHECK(kv_shape_changed(published, next), "case 14: n_seq_max");
        next            = published;
        next.kv_unified = true;
        CHECK(kv_shape_changed(published, next), "case 14: kv_unified");
        next          = published;
        next.swa_full = true;
        CHECK(kv_shape_changed(published, next), "case 14: swa_full");
    }
    // 15. Demote at a large context, then a small one gets the load-time
    // residency back: the split with 172.2 MB of live headroom on device 0.
    // The slack (30 resident layers x 64 KiB) does not change the count.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.load_kv_device[30] = 1;
        in.load_kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        in.devices   = { 0, 1 };
        in.available = { 172 * mb + 205 * mb / 1000, 442 * mb };
        in.layer_kv_bytes.assign(32, 8 * mb);
        auto big = plan_runtime_kv_residency(in);
        CHECK(big.fits, "case 15: n_ctx=2048 fits after demotion");
        CHECK_EQ(big.per_device[0].demoted_layers.size(), 9, "case 15: nine layers demoted on device 0");
        CHECK(big.per_device[1].demoted_layers.empty(), "case 15: device 1 keeps its layers");
        for (int l = 0; l < 32; ++l) {
            const int want = l >= 21 && l <= 29 ? -1 : in.load_kv_device[l];
            CHECK_EQ(big.kv_device[l], want, "case 15: layers 21..29 on host, the rest where the planner put them");
        }

        in.layer_kv_bytes.assign(32, 2 * mb);  // n_ctx=512
        auto small = plan_runtime_kv_residency(in);
        CHECK(small.fits, "case 15: n_ctx=512 fits");
        CHECK(small.kv_device == in.load_kv_device, "case 15: the load-time residency comes back");
    }
    // 16. KV that does not fit with every full-attention layer demoted places
    // its SWA layers on the host tier too, full-attention first, instead of
    // refusing the context. When full-attention demotion is enough, SWA stays.
    {
        kv_residency_input in;
        in.load_kv_device = { 0, 0 };
        in.layer_kv_bytes = { 100, 100 };
        in.swa_layer_mask = { 1, 0 };
        in.devices        = { 0 };
        in.available      = { 50 + 2 * kv_alloc_slack_per_layer };
        auto r            = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 16: SWA KV over headroom is placed on the host tier, not refused");
        CHECK(r.per_device[0].demoted_layers == std::vector<int>({ 1, 0 }),
              "case 16: full-attention layer first, then the SWA layer");
        CHECK(r.kv_device == std::vector<int>({ -1, -1 }), "case 16: both layers on the host tier");

        in.available = { 100 + 2 * kv_alloc_slack_per_layer };
        auto enough  = plan_runtime_kv_residency(in);
        CHECK(enough.fits && enough.kv_device == std::vector<int>({ 0, -1 }),
              "case 16: the SWA layer stays when full-attention demotion is enough");
    }
    // 17. The slack is enough for the arena allocator: layers admitted against
    // exactly sum(bytes) + n * slack of headroom all place, one allocation at a
    // time and in two KV buffers, at the tiered allocator's 512-byte alignment.
    {
        const size_t        mb     = 1024 * 1024;
        std::vector<size_t> layers = { 8 * mb + 1, 8 * mb + 511, 128 * mb + 300, 3 * mb / 2 + 7, 1, 256 };
        size_t              sum    = 0;
        for (size_t b : layers) {
            sum += b;
        }
        tlsf_allocator arena(sum + layers.size() * kv_alloc_slack_per_layer);
        for (size_t pass = 0; pass < 2; ++pass) {  // full-attention buffer, then SWA buffer
            for (size_t i = pass; i < layers.size(); i += 2) {
                const size_t request = (layers[i] + 511) & ~size_t(511);
                const size_t before  = arena.used();
                CHECK(arena.allocate(request) != SIZE_MAX, "case 17: an admitted layer places");
                CHECK(arena.used() - before <= layers[i] + kv_alloc_slack_per_layer,
                      "case 17: one allocation costs at most its bytes plus the slack");
            }
        }
    }
    // 18. Two same-shape contexts against one device. The first is admitted
    // and allocates its KV; the second, not yet admitted, re-fits against the
    // headroom A left and places its overflow on the host tier through the
    // plan -- and the allocator's backstop never fires.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.layer_kv_bytes.assign(32, 8 * mb);
        in.swa_layer_mask.assign(32, 0);
        in.devices   = { 0 };
        in.available = { 300 * mb };
        kv_shape load;
        kv_shape shape;
        shape.runtime   = true;
        shape.n_ctx     = 2048;
        shape.n_seq_max = 1;

        // Admission's demand on device 0: every resident layer plus its slack.
        auto demand = [&](const std::vector<int> & kv_device) {
            size_t bytes = 0;
            for (size_t l = 0; l < kv_device.size(); ++l) {
                bytes += kv_device[l] == 0 ? in.layer_kv_bytes[l] + kv_alloc_slack_per_layer : 0;
            }
            return bytes;
        };

        // Context A: the load-time plan is re-decided and all 32 layers fit.
        CHECK(kv_residency_needs_refit(load, shape, false), "case 18: A is decided");
        auto a = plan_runtime_kv_residency(in);
        CHECK(a.fits && a.per_device[0].demoted_layers.empty(), "case 18: A keeps every layer on the device");
        size_t a_bytes = 0;
        for (int l = 0; l < 32; ++l) {
            a_bytes += a.kv_device[l] == 0 ? in.layer_kv_bytes[l] : 0;
        }
        const size_t shared_available = in.available[0] - a_bytes;  // A's KV is allocated

        // Context B: same shape, published residency is A's.
        const bool       refit = kv_residency_needs_refit(shape, shape, false);
        std::vector<int> b_kv  = a.kv_device;
        if (refit) {
            in.available = { shared_available };
            auto b       = plan_runtime_kv_residency(in);
            CHECK(b.fits, "case 18: B fits after demotion");
            b_kv = b.kv_device;
        }
        size_t b_bytes = 0;
        for (int l = 0; l < 32; ++l) {
            b_bytes += b_kv[l] == 0 ? in.layer_kv_bytes[l] : 0;
        }
        CHECK(!kv_admission_mismatch(b_bytes, shared_available), "case 18: the allocator backstop never fires");
        CHECK(refit, "case 18: the second same-shape context re-fits");
        CHECK(demand(b_kv) <= shared_available, "case 18: B's admitted demand fits");
        CHECK(b_bytes < a_bytes, "case 18: B's overflow is on the host tier");
    }
    // 19. An admitted context's same-shape republish (its micro-batch trial;
    // its KV is allocated, so the live headroom no longer covers it) keeps its
    // residency. A new context always re-fits, from the load-time residency,
    // so it gets full device residency when that fits even if the published
    // residency is more demoted.
    {
        kv_shape shape;
        shape.runtime   = true;
        shape.n_ctx     = 2048;
        shape.n_seq_max = 1;
        CHECK(!kv_residency_needs_refit(shape, shape, true),
              "case 19: an admitted context's republish keeps its residency");
        CHECK(kv_residency_needs_refit(shape, shape, false), "case 19: a new context re-fits");

        kv_residency_input in;
        in.load_kv_device = { 0, 0, 0, 0 };
        in.layer_kv_bytes = { 64, 64, 64, 64 };
        in.swa_layer_mask = { 0, 0, 0, 0 };
        in.devices        = { 0 };
        in.available      = { 256 + 4 * kv_alloc_slack_per_layer };

        const std::vector<int> published = { 0, 0, -1, -1 };  // an earlier, larger context's residency
        auto                   r         = plan_runtime_kv_residency(in);
        CHECK(r.fits && r.kv_device == in.load_kv_device,
              "case 19: a new context that fits gets full device residency");
        CHECK(r.kv_device != published, "case 19: it does not inherit the published residency");
    }
    // 20. A layer of device 0's KV buffer that the plan gives to device 1 is in
    // device 1's VRAM, although device 0's tier layout marks it off-device.
    {
        CHECK(kv_buffer_layer_owner(true, 1, false, 0, false) == 1,
              "case 20: another device's layer is device VRAM, not host memory");
        CHECK(kv_buffer_layer_owner(true, 0, true, 0, false) == 0, "case 20: this device's layer");
        CHECK(kv_buffer_layer_owner(true, -1, true, 0, false) == -1, "case 20: a demoted layer is host memory");
        CHECK(kv_buffer_layer_owner(false, -1, true, 0, false) == 0, "case 20: without a plan the layout decides");
        CHECK(kv_buffer_layer_owner(true, 0, true, 0, true) == -1, "case 20: GGML_SYCL_KV_HOST=1 is host memory");
    }
    // 21. A full arena KV zone leaves no KV headroom; only a device without an
    // arena uses the budget-based headroom.
    {
        CHECK_EQ(kv_vram_available(true, 0, 4096), 0, "case 21: a full KV zone is not a missing arena");
        CHECK_EQ(kv_vram_available(true, 512, 4096), 512, "case 21: the arena's KV zone");
        CHECK_EQ(kv_vram_available(false, 0, 4096), 4096, "case 21: no arena, the budget headroom");
    }
    // 22. On a split, each device's backend publishes the same context and
    // re-fits it. All of them publish before llama_kv_cache allocates that KV,
    // so each re-fit restarts from the load residency against headroom that
    // still includes it (plan_runtime_kv_residency is a pure function, so the
    // same input gives the same answer). Were one to run after the KV is
    // allocated, the headroom would no longer include it and it would demote
    // more; the publish ordering is pinned on the source
    // (test-sycl-kv-layer-sizing-source.py).
    {
        kv_residency_input in;
        in.load_kv_device = { 0, 0, 0, 1, 1, 1 };
        in.layer_kv_bytes = { 64, 64, 64, 64, 64, 64 };
        in.swa_layer_mask = { 0, 0, 0, 0, 0, 0 };
        in.devices        = { 0, 1 };
        in.available      = { 128 + 3 * kv_alloc_slack_per_layer, 192 + 3 * kv_alloc_slack_per_layer };
        auto before       = plan_runtime_kv_residency(in);
        CHECK(before.fits, "case 22: the re-fit before allocation fits");
        CHECK_EQ(before.per_device[0].demoted_layers.size(), 1, "case 22: device 0 demotes one layer");

        in.available[0] -= 2 * 64;  // after allocation: the headroom no longer includes the two resident layers
        auto late = plan_runtime_kv_residency(in);
        CHECK(late.per_device[0].demoted_layers.size() > before.per_device[0].demoted_layers.size(),
              "case 22: a re-fit after allocation would demote more");
    }
    // 23. GGML_SYCL_KV_HOT_LAYERS is read by value, as the tier manager reads
    // it: a count >= 0 overrides, -1 (or unset) does not.
    {
        CHECK(!kv_hot_layers_override_active(nullptr), "case 23: unset");
        CHECK(!kv_hot_layers_override_active("-1"), "case 23: -1 means off");
        CHECK(kv_hot_layers_override_active("0"), "case 23: 0 hot layers is an override");
        CHECK(kv_hot_layers_override_active("16"), "case 23: 16 hot layers");
    }
    // 24. Only a multi-device plan in GLOBAL cache mode reads no per-device
    // arena; every other combination reads the device's own zones.
    {
        CHECK(!kv_reads_device_arena(true, true), "case 24: multi-device GLOBAL has one zone for all devices");
        CHECK(kv_reads_device_arena(true, false), "case 24: multi-device PER_DEVICE");
        CHECK(kv_reads_device_arena(false, true), "case 24: single-device GLOBAL");
        CHECK(kv_reads_device_arena(false, false), "case 24: single-device PER_DEVICE");
    }
    // 25. A device's KV residency changed only when one of the layers it holds
    // at load moved between the published and the next plan. An absent layer
    // is on the host tier, like -1.
    {
        const auto load      = kv_device_map({ 0, 0, 1, 1 });
        const auto published = kv_device_map({ 0, -1, 1, 1 });
        auto       absent    = published;
        absent.erase(1);
        CHECK(!kv_device_residency_changed(load, published, published, 0), "case 25: same residency");
        CHECK(!kv_device_residency_changed(load, published, absent, 0),
              "case 25: an absent layer is the host tier, like -1");
        const auto other = kv_device_map({ 0, -1, 1, -1 });
        CHECK(!kv_device_residency_changed(load, published, other, 0), "case 25: only device 1 changed");
        CHECK(kv_device_residency_changed(load, published, other, 1), "case 25: device 1 demoted one more layer");
        const auto more = kv_device_map({ -1, -1, 1, 1 });
        CHECK(kv_device_residency_changed(load, published, more, 0), "case 25: device 0 demoted one more layer");
        CHECK(kv_device_residency_changed(load, more, published, 0), "case 25: device 0 got a layer back");
    }
    // 26. Optional layout copies yield before any KV layer demotes. The B50
    // Mistral Q4_0 run at GGML_SYCL_VRAM_BUDGET_PCT=60, -c 2048: 32 layers of
    // 8 MB, 84.4 MB of live headroom left beside 2869.8 MB of oneDNN WOQ
    // copies. Without the copies counted it demotes 22 layers, as that run did.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.swa_layer_mask.assign(32, 0);
        in.layer_kv_bytes.assign(32, 8 * mb);
        in.devices   = { 0 };
        in.available = { 84 * mb + 4 * mb / 10 };

        auto without = plan_runtime_kv_residency(in);
        CHECK(without.fits, "case 26: fits without the copies, by demotion");
        CHECK_EQ(without.per_device[0].demoted_layers.size(), 22, "case 26: 22 layers demoted without the copies");

        in.optional_layouts = { one_copy_beside_free(in.available[0], 2869 * mb + 8 * mb / 10) };
        auto r              = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 26: fits");
        CHECK(r.per_device[0].demoted_layers.empty(), "case 26: no KV layer demotes while copies can yield");
        CHECK(r.kv_device == in.load_kv_device, "case 26: all 32 layers stay on the device");
        CHECK_EQ(r.yield_bytes.size(), 1, "case 26: one yield per device");
        CHECK_EQ(r.yield_bytes[0], 32 * 8 * mb + 32 * kv_alloc_slack_per_layer - in.available[0],
                 "case 26: yields exactly the shortfall");
        CHECK(r.yields[0].groups == std::vector<std::vector<size_t>>({ { 0 } }), "case 26: the copy is the pick");
        CHECK_EQ(r.yields[0].kv_layers, 32, "case 26: every layer lands once it goes");
    }
    // 27. The same device at -c 32768 (128 MB a layer): by bytes every copy
    // yields and only what is still over demotes -- 9 layers instead of all 32.
    // The copy and the free space beside it are one extent, so the 23 layers
    // the bytes admit also land.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.swa_layer_mask.assign(32, 0);
        in.layer_kv_bytes.assign(32, 128 * mb);
        in.devices   = { 0 };
        in.available = { 84 * mb + 4 * mb / 10 };

        auto without = plan_runtime_kv_residency(in);
        CHECK_EQ(without.per_device[0].demoted_layers.size(), 32, "case 27: every layer demoted without the copies");

        in.optional_layouts = { one_copy_beside_free(in.available[0], 2869 * mb + 8 * mb / 10) };
        auto r              = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 27: fits");
        CHECK_EQ(r.yield_bytes[0], in.optional_layouts[0].bytes[0], "case 27: every copy yields");
        CHECK_EQ(r.yields[0].kv_layers, 23, "case 27: 23 layers land once it goes");
        CHECK_EQ(r.per_device[0].demoted_layers.size(), 9, "case 27: only the rest demotes");
    }
    // 28. Nothing yields while the KV fits, and a device without a shortfall
    // keeps its copies while another device yields.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device = { 0, 0, 1, 1 };
        in.swa_layer_mask.assign(4, 0);
        in.layer_kv_bytes.assign(4, 8 * mb);
        in.devices          = { 0, 1 };
        in.available        = { 100 * mb, 10 * mb };
        in.optional_layouts = { one_copy_beside_free(100 * mb, 500 * mb), one_copy_beside_free(10 * mb, 500 * mb) };
        auto r              = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 28: fits");
        CHECK_EQ(r.yield_bytes[0], 0, "case 28: device 0 fits, nothing yields");
        CHECK(r.yields[0].groups.empty(), "case 28: and device 0 picks nothing");
        CHECK(r.yields[1].groups == std::vector<std::vector<size_t>>({ { 0 } }), "case 28: device 1 picks its copy");
        CHECK_EQ(r.yield_bytes[1], 16 * mb + 2 * kv_alloc_slack_per_layer - 10 * mb,
                 "case 28: device 1 yields its shortfall");
        CHECK(r.kv_device == in.load_kv_device, "case 28: no layer demotes");
    }
    // 29. What lands is decided by the allocator, not by bytes. The zone below
    // is S1-PRELOAD's order -- each dense weight's primary (40 MB), then its
    // WOQ copy (48 MB) -- sealed by one more primary, with 8 MB free on top.
    // Freeing every copy frees 192 MB, but into four 48 MB holes between live
    // primaries: not one 128 MB layer lands, so no copy is worth releasing.
    // 8 MB layers do land in them, and only as many copies go as the layers
    // need, from the top down -- each its own group, since each hole holds
    // layers without the other.
    {
        using ggml_sycl::kv_zone_block;
        const size_t              mb     = 1024 * 1024;
        const kv_optional_layouts copies = interleaved_copies();
        CHECK_EQ(copies.zone[0].largest_free_block(), 8 * mb, "case 29: 8 MB free on top");

        const std::vector<size_t> big(2, 128 * mb);
        CHECK_EQ(ggml_sycl::kv_layers_allocatable(copies.zone, big), 0, "case 29: no 128 MB layer lands");
        kv_optional_layout_yield y = plan_optional_layout_yield(copies, big);
        CHECK(y.groups.empty(), "case 29: holes between primaries give a 128 MB layer nothing, so every copy stays");
        CHECK_EQ(y.kv_layers, 0, "case 29: and no 128 MB layer lands");

        const std::vector<size_t> small(13, 8 * mb);
        CHECK_EQ(ggml_sycl::kv_layers_allocatable(copies.zone, small), 1,
                 "case 29: one 8 MB layer lands before a yield");
        y = plan_optional_layout_yield(copies, small);
        CHECK(y.groups == std::vector<std::vector<size_t>>({ { 3 }, { 2 } }),
              "case 29: two holes of six layers each, the top two copies, one group each");
        CHECK_EQ(y.kv_layers, 13, "case 29: all thirteen land once both go");
        y = plan_optional_layout_yield(copies, {});
        CHECK(y.groups.empty() && y.kv_layers == 0, "case 29: no layers asked, nothing yields");

        // A copy the allocator does not model (allocator == SIZE_MAX) is never picked.
        kv_optional_layouts unmodelled = copies;
        unmodelled.copies              = { kv_zone_block{} };
        unmodelled.bytes               = { 48 * mb };
        CHECK(plan_optional_layout_yield(unmodelled, small).groups.empty(), "case 29: an unmodelled copy stays");

        // No zone (no arena): KV is not carved from a zone, so none limits it.
        kv_optional_layouts no_zone;
        no_zone.copies = { kv_zone_block{} };
        no_zone.bytes  = { 48 * mb };
        y              = plan_optional_layout_yield(no_zone, small);
        CHECK(y.groups.empty(), "case 29: without a zone nothing is picked");
        CHECK_EQ(y.kv_layers, small.size(), "case 29: and no zone holds the layers back");
    }
    // 30. The same copies staged together, after every primary: they free into
    // one extent with each other and the free space on top, so a 128 MB layer
    // lands once the top three go -- and only those three, even with a second
    // layer asked for that the fourth still would not make room for. They are
    // one group: any two of them free an extent the layer does not fit.
    {
        const size_t        mb = 1024 * 1024;
        kv_optional_layouts copies;
        copies.zone.emplace_back(360 * mb);
        for (int k = 0; k < 4; ++k) {
            (void) copies.zone[0].allocate(40 * mb);
        }
        for (int k = 0; k < 4; ++k) {
            copies.copies.push_back({ 0, copies.zone[0].allocate(48 * mb) });
            copies.bytes.push_back(48 * mb);
        }
        CHECK_EQ(copies.zone[0].largest_free_block(), 8 * mb, "case 30: 8 MB free on top");
        const std::vector<size_t>      big       = std::vector<size_t>(2, 128 * mb);
        const kv_optional_layout_yield y         = plan_optional_layout_yield(copies, big);
        const std::vector<size_t>      top_three = { 3, 2, 1 };
        CHECK(y.groups.size() == 1 && y.groups[0] == top_three,
              "case 30: the top three copies, walked down from the top, as one group");
        CHECK_EQ(y.kv_layers, 1, "case 30: and one 128 MB layer lands");
        ggml_sycl::kv_zone_model freed = copies.zone;
        for (size_t i : y.groups[0]) {
            freed[0].free(copies.copies[i].offset);
        }
        CHECK_EQ(ggml_sycl::kv_layers_allocatable(freed, big), (long long) y.kv_layers,
                 "case 30: kv_layers is what the zone places once the groups go");
    }
    // 31. A copy the gain does not need is dropped again: the 4 MB copy on top
    // joins the free space first, but only the isolated 48 MB hole below holds
    // a 40 MB layer.
    {
        const size_t        mb = 1024 * 1024;
        kv_optional_layouts copies;
        copies.zone.emplace_back(140 * mb);
        (void) copies.zone[0].allocate(40 * mb);
        copies.copies.push_back({ 0, copies.zone[0].allocate(48 * mb) });
        (void) copies.zone[0].allocate(40 * mb);
        copies.copies.push_back({ 0, copies.zone[0].allocate(4 * mb) });
        copies.bytes = { 48 * mb, 4 * mb };
        CHECK_EQ(copies.zone[0].largest_free_block(), 8 * mb, "case 31: 8 MB free on top");
        CHECK(plan_optional_layout_yield(copies, { 40 * mb }).groups == std::vector<std::vector<size_t>>({ { 0 } }),
              "case 31: only the hole that holds the layer");
    }
    // 32. Layers land in allocation order, each in the first allocator with
    // room (zone_alloc(KV)'s order), and counting stops at the first that
    // does not land even if a later, smaller one would.
    {
        using ggml_sycl::kv_zone_model;
        const size_t  mb = 1024 * 1024;
        kv_zone_model zone{ tlsf_allocator(16 * mb), tlsf_allocator(64 * mb) };
        CHECK_EQ(ggml_sycl::kv_layers_allocatable(zone, { 16 * mb, 32 * mb, 32 * mb }), 3,
                 "case 32: the first allocator, then the second");
        CHECK_EQ(ggml_sycl::kv_layers_allocatable(zone, { 64 * mb, 128 * mb, 8 * mb }), 1,
                 "case 32: stops at the first layer that does not land");
        CHECK_EQ(zone[1].largest_free_block(), 64 * mb, "case 32: the model passed in is not changed");
    }
    // 33. fit_capacity holds a device's KV to what its zone can place,
    // whatever its headroom: 2954 MB free, but only ten 128 MB layers land.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.swa_layer_mask.assign(32, 0);
        in.layer_kv_bytes.assign(32, 128 * mb);
        in.devices   = { 0 };
        in.available = { 2954 * mb };
        CHECK_EQ(plan_runtime_kv_residency(in).per_device[0].demoted_layers.size(), 9,
                 "case 33: by bytes, 23 layers stay");
        in.fit_capacity = { 10 * 128 * mb };
        auto r          = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 33: fits");
        CHECK_EQ(r.per_device[0].demoted_layers.size(), 22, "case 33: held to the ten that land");
        for (int l = 0; l < 10; ++l) {
            CHECK_EQ(r.kv_device[l], 0, "case 33: the leading layers are the ones kept");
        }
        in.fit_capacity = { SIZE_MAX };
        CHECK_EQ(plan_runtime_kv_residency(in).per_device[0].demoted_layers.size(), 9, "case 33: SIZE_MAX is no cap");
    }
    // 34. kv_layer_alloc_bytes is the tiered allocator's 512-byte layer size.
    {
        CHECK_EQ(ggml_sycl::kv_layer_alloc_bytes(1), 512, "case 34: rounds up");
        CHECK_EQ(ggml_sycl::kv_layer_alloc_bytes(512), 512, "case 34: exact stays");
        CHECK_EQ(ggml_sycl::kv_layer_alloc_bytes(513), 1024, "case 34: next block");
    }
    // 35. The fit counts a device's copies as headroom only where its layers
    // land in them: it hands back the groups worth releasing as the pick list,
    // and holds the device's KV to the leading layers its zone places once they
    // go. Case 29's zone, with 8 MB of live headroom beside it.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.devices          = { 0 };
        in.available        = { 8 * mb };
        in.optional_layouts = { interleaved_copies() };

        // Two 128 MB layers: 200 MB of headroom admits one by bytes, but no
        // layer lands, so nothing is picked and both demote.
        in.load_kv_device.assign(2, 0);
        in.swa_layer_mask.assign(2, 0);
        in.layer_kv_bytes.assign(2, 128 * mb);
        kv_residency_result r = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 35: fits");
        CHECK(r.yield_bytes[0] > 0, "case 35: by bytes the copies are headroom");
        CHECK(r.yields[0].groups.empty(), "case 35: no copy lets a 128 MB layer land");
        CHECK_EQ(r.per_device[0].demoted_layers.size(), 2, "case 35: held to the zone, both layers demote");

        // Thirteen 8 MB layers: the top two copies, one group each, and none demote.
        in.load_kv_device.assign(13, 0);
        in.swa_layer_mask.assign(13, 0);
        in.layer_kv_bytes.assign(13, 8 * mb);
        r = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 35: fits");
        CHECK(r.yields[0].groups == std::vector<std::vector<size_t>>({ { 3 }, { 2 } }),
              "case 35: the fit's pick list is the zone model's groups");
        CHECK_EQ(r.yields[0].kv_layers, 13, "case 35: all land");
        CHECK(r.per_device[0].demoted_layers.empty(), "case 35: nothing demotes");

        // Twenty-six, with headroom enough by bytes: all four copies make
        // room for only 25, so the zone, not the bytes, demotes one.
        in.available = { 40 * mb };
        in.load_kv_device.assign(26, 0);
        in.swa_layer_mask.assign(26, 0);
        in.layer_kv_bytes.assign(26, 8 * mb);
        r = plan_runtime_kv_residency(in);
        CHECK_EQ(r.yields[0].groups.size(), 4, "case 35: every copy is its own group");
        CHECK_EQ(r.yields[0].kv_layers, 25, "case 35: 25 of 26 land");
        CHECK(r.per_device[0].demoted_layers == std::vector<int>({ 25 }), "case 35: the last layer demotes");
    }
    // 36. A device's layers in allocation order, and the bytes of its leading
    // ones: another device's layers and KV-less layers are not its.
    {
        kv_residency_input in;
        in.load_kv_device = { 0, 1, 0, 0, -1, 0 };
        in.layer_kv_bytes = { 1000, 2000, 0, 3000, 4000, 513 };
        CHECK(ggml_sycl::kv_device_layer_alloc_bytes(in, 0) == std::vector<size_t>({ 1024, 3072, 1024 }),
              "case 36: device 0's layers, at their allocation size");
        CHECK(ggml_sycl::kv_device_layer_alloc_bytes(in, 1) == std::vector<size_t>({ 2048 }), "case 36: device 1");
        CHECK_EQ(ggml_sycl::kv_device_leading_layer_bytes(in, 0, 0), 0, "case 36: no layers");
        CHECK_EQ(ggml_sycl::kv_device_leading_layer_bytes(in, 0, 2), 4000, "case 36: the first two, at KV bytes");
        CHECK_EQ(ggml_sycl::kv_device_leading_layer_bytes(in, 0, 99), 4513, "case 36: clamped to the device's");
    }
    // 37. The cell arithmetic, on the shapes llama.cpp-3aos and llama.cpp-uajm
    // verified against live runs, and on the GPT-OSS 20B shape the fit's worked
    // prediction uses.
    {
        using namespace ggml_sycl;
        // GPT-OSS 20B (n_swa 128) at -c 4096 -np 4 -ub 512, per-stream KV: n_ctx_seq 1024,
        // PAD(min(1024, 128 + 512), 256) = 768 cells a stream, four streams.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 4, false, false, 128), 3072,
                 "case 37: GPT-OSS SWA, 4 streams");
        // Gemma 4 E4B (n_swa 512): PAD(min(1024, 512 + 512), 256) = 1024 a stream.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 4, false, false, 512), 4096, "case 37: Gemma SWA, 4 streams");
        // kv_unified: one stream whose window scales with n_seq_max, 512 * 4 + 512 = 2560.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 4, true, false, 512), 2560, "case 37: unified SWA window");
        // swa_full: an SWA layer holds the whole window, like a FULL one.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 4, false, true, 128), 4096, "case 37: swa_full is n_ctx");
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 4, true, false, 0), 0, "case 37: no window, no cells");
        CHECK_EQ(kv_layer_cells(KV_CELLS_FULL, 65536, 1024, 1, false, false, 128), 65536, "case 37: FULL is n_ctx");
        CHECK_EQ(kv_layer_cells(KV_CELLS_SHARED, 65536, 1024, 1, false, false, 128), 0, "case 37: SHARED holds none");
        // The fit's worked prediction: PAD(n_swa + n_ubatch, 256) = PAD(128 + 1024, 256) = 1280.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 65536, 1024, 1, false, false, 128), 1280, "case 37: GA's SWA cells");
        // n_seq_max 0 reads as 1, as it always did.
        CHECK_EQ(kv_layer_cells(KV_CELLS_SWA, 4096, 512, 0, false, false, 128), 768, "case 37: n_seq_max 0 is 1");
    }
    // 38. One layer's bytes: the row-size rule per type, K and V separately, each
    // tensor padded the way ggml_backend_alloc_ctx_tensors_from_buft pads it.
    {
        using namespace ggml_sycl;
        kv_layer_desc gptoss;  // 8 KV heads x head dim 64
        gptoss.n_embd_k_gqa = 512;
        gptoss.n_embd_v_gqa = 512;
        gptoss.has_kv       = 1;
        // f16: 2 KiB a cell, K and V together; 65536 cells is 128 MiB.
        CHECK_EQ(kv_layer_tensor_bytes(gptoss, TEST_TYPE_F16, TEST_TYPE_F16, 65536, 1, test_row_size),
                 128ll * 1024 * 1024, "case 38: f16 layer");
        // K and V at q8_0: 512 elements are 16 blocks of 34 bytes.
        CHECK_EQ(kv_layer_tensor_bytes(gptoss, TEST_TYPE_Q8_0, TEST_TYPE_Q8_0, 65536, 1, test_row_size),
                 2ll * 65536 * 544, "case 38: q8_0 K and V");
        CHECK_EQ(kv_layer_tensor_bytes(gptoss, TEST_TYPE_Q8_0, TEST_TYPE_F16, 100, 1, test_row_size),
                 100ll * (544 + 1024), "case 38: q8_0 K, f16 V");
        // Padding is per tensor: 37 cells of a 96-wide f16 row are 7104 B a tensor, 7168 padded to 128.
        kv_layer_desc narrow;
        narrow.n_embd_k_gqa = 96;
        narrow.n_embd_v_gqa = 96;
        narrow.has_kv       = 1;
        CHECK_EQ(kv_layer_tensor_bytes(narrow, TEST_TYPE_F16, TEST_TYPE_F16, 37, 1, test_row_size), 2 * 7104,
                 "case 38: unpadded");
        CHECK_EQ(kv_layer_tensor_bytes(narrow, TEST_TYPE_F16, TEST_TYPE_F16, 37, 128, test_row_size), 2 * 7168,
                 "case 38: each tensor padded to the alignment");
        // v_trans with a variable V width: llama pads V to n_embd_v_gqa_max, so V is wider than K.
        kv_layer_desc vtrans = narrow;
        vtrans.n_embd_v_gqa  = 160;
        CHECK_EQ(kv_layer_tensor_bytes(vtrans, TEST_TYPE_F16, TEST_TYPE_F16, 64, 1, test_row_size), 64ll * (192 + 320),
                 "case 38: V wider than K");
        // MLA: no V at all.
        kv_layer_desc mla = narrow;
        mla.n_embd_v_gqa  = 0;
        CHECK_EQ(kv_layer_tensor_bytes(mla, TEST_TYPE_F16, TEST_TYPE_F16, 64, 128, test_row_size), 64 * 192,
                 "case 38: MLA has K only");
        // A filtered, shared or reused layer holds no slot.
        kv_layer_desc none = narrow;
        none.has_kv        = 0;
        CHECK_EQ(kv_layer_tensor_bytes(none, TEST_TYPE_F16, TEST_TYPE_F16, 64, 128, test_row_size), 0,
                 "case 38: has_kv 0 is 0");
    }
    // 39. The recurrent-state buffer: r_l then s_l of every layer, row size times
    // rows, each tensor padded.  Qwen3.5-35B-A3B at n_seq_max 1 is the doc's 65863680 B.
    {
        using namespace ggml_sycl;
        std::vector<rs_layer_desc> layers;
        for (uint32_t il = 0; il < 30; ++il) {
            rs_layer_desc d;
            d.il       = il;
            d.type_r   = TEST_TYPE_F32;
            d.type_s   = TEST_TYPE_F32;
            d.n_embd_r = 3 * (4096 + 2 * 16 * 128);
            d.n_embd_s = 128 * 4096;
            d.n_rows   = 1;
            layers.push_back(d);
        }
        CHECK_EQ(rs_buffer_bytes(layers, 1, test_row_size), 65863680, "case 39: the Qwen recurrent state");
        for (rs_layer_desc & d : layers) {
            d.n_rows = 4;
        }
        CHECK_EQ(rs_buffer_bytes(layers, 128, test_row_size), 4ll * 65863680, "case 39: scales with n_seq_max");
        // A padded tensor: 3 rows of a 33-wide f32 r and a 1-wide s.
        rs_layer_desc odd;
        odd.type_r   = TEST_TYPE_F32;
        odd.type_s   = TEST_TYPE_F32;
        odd.n_embd_r = 33;
        odd.n_embd_s = 1;
        odd.n_rows   = 3;
        CHECK_EQ(rs_buffer_bytes({ odd }, 128, test_row_size), 512 + 128, "case 39: 396 B and 12 B pad to 512 and 128");
        CHECK_EQ(rs_buffer_bytes({}, 128, test_row_size), 0, "case 39: no layers");
    }
    // 40. Reconciliation: a slot is reused in place when its owner holds the same
    // (cohort, index) with cap >= the need; every other slot is carved, the old
    // one it supersedes is released at the publish, and a held slot no record
    // names stays as planned room.
    {
        using namespace ggml_sycl;
        context_side_demand ring;
        ring.device = 0;
        ring.owner  = 7;
        ring.cohort = "ring";
        ring.slots  = { 400, 500 };
        context_side_demand compute;
        compute.device                           = 0;
        compute.owner                            = 7;
        compute.cohort                           = "compute";
        compute.slots                            = { 300 };
        std::vector<context_side_demand> demands = { ring, compute };

        auto held_at = [](const char * cohort, uint32_t index, size_t cap, uint64_t owner = 7, int device = 0) {
            held_slot h;
            h.device = device;
            h.owner  = owner;
            h.cohort = cohort;
            h.index  = index;
            h.cap    = cap;
            return h;
        };
        // Nothing held: everything is carved.
        demand_reconciliation r = context_demand_reconcile(demands, {});
        CHECK_EQ(r.carved.size(), 3, "case 40: three slots carved");
        CHECK(r.reused.empty() && r.superseded.empty() && r.unused.empty(), "case 40: nothing else");
        // A larger slot serves a smaller claim; an equal one too; a smaller one is superseded.
        std::vector<held_slot> held = { held_at("ring", 0, 1000),    held_at("ring", 1, 500),
                                        held_at("compute", 0, 299),  held_at("ring", 2, 64),
                                        held_at("ring", 0, 9999, 8), held_at("ring", 0, 9999, 7, 1) };
        r                           = context_demand_reconcile(demands, held);
        CHECK_EQ(r.reused.size(), 2, "case 40: ring 0 and ring 1 reused");
        CHECK_EQ(r.reused[0].held, 0, "case 40: ring 0 is served by the 1000 B slot");
        CHECK_EQ(r.superseded.size(), 1, "case 40: compute grew");
        CHECK_EQ(r.superseded[0].held, 2, "case 40: the 299 B slot is the one superseded");
        CHECK(r.carved.empty(), "case 40: nothing carved with no key held");
        CHECK(r.unused == std::vector<size_t>({ 3 }),
              "case 40: ring index 2 is named by no record; another owner's and another device's slots are not ours");
        // A zero-byte index needs no carve and is served by anything held.
        context_side_demand zero = ring;
        zero.slots               = { 0, 0 };
        r                        = context_demand_reconcile({ zero }, { held_at("ring", 1, 8) });
        CHECK_EQ(r.reused.size(), 1, "case 40: a held slot serves a zero need");
        CHECK(r.carved.empty() && r.superseded.empty(), "case 40: a zero need carves nothing");

        // The scope is part of the key: a MODEL slot of the same owner id, device,
        // cohort and index is a different reservation from the CONTEXT one.
        {
            held_slot other_scope   = held_at("ring", 0, 1000);
            other_scope.scope       = demand_scope::MODEL;
            context_side_demand one = ring;
            one.slots               = { 400 };
            r                       = context_demand_reconcile({ one }, { other_scope });
            CHECK(r.reused.empty(), "case 40: a held slot of the other scope does not serve the claim");
            CHECK_EQ(r.carved.size(), 1, "case 40: so the claim is carved");
            CHECK(r.unused.empty(), "case 40: and the other-scope slot is not ours to report");
            CHECK(!demand_slot_key_matches(other_scope, one, 0), "case 40: the key differs by scope alone");
            other_scope.scope = demand_scope::CONTEXT;
            CHECK(demand_slot_key_matches(other_scope, one, 0),
                  "case 40: control: the same key with the scope equal matches");
        }
        // Cohorts compare by content, not by pointer: two distinct buffers holding
        // "ring" match; a different string does not; a null cohort matches only null.
        {
            const std::string   spelled = std::string("ri") + "ng";
            const std::string   other   = std::string("ri") + "ngs";
            context_side_demand one     = ring;
            one.slots                   = { 400 };
            held_slot copy              = held_at(spelled.c_str(), 0, 1000);
            CHECK(copy.cohort != one.cohort, "case 40: control: the two cohort pointers differ");
            CHECK(demand_slot_key_matches(copy, one, 0), "case 40: equal cohort text under distinct pointers matches");
            r = context_demand_reconcile({ one }, { copy });
            CHECK_EQ(r.reused.size(), 1, "case 40: and the slot is reused");
            held_slot different = held_at(other.c_str(), 0, 1000);
            CHECK(!demand_slot_key_matches(different, one, 0), "case 40: different cohort text does not match");
            context_side_demand null_demand = one;
            null_demand.cohort              = nullptr;
            held_slot null_held             = held_at(nullptr, 0, 1000);
            CHECK(demand_slot_key_matches(null_held, null_demand, 0), "case 40: null matches null");
            CHECK(!demand_slot_key_matches(null_held, one, 0) && !demand_slot_key_matches(copy, null_demand, 0),
                  "case 40: null does not match a named cohort");
        }
    }
    if (int rc = case_h2_a2()) {
        return rc;
    }
    if (int rc = case_h2_a2_red()) {
        return rc;
    }
    if (int rc = run_case("h2_property", case_h2_property)) {
        return rc;
    }
    if (int rc = case_h2_churn()) {
        return rc;
    }
    if (int rc = case_h2_ring_head_slot()) {
        return rc;
    }
    if (int rc = case_h2_head_slot_rules()) {
        return rc;
    }
    if (int rc = case_h2_ring_held_yield()) {
        return rc;
    }
    if (int rc = case_h2_rows_per_context()) {
        return rc;
    }
    if (int rc = case_h2_demotion_order()) {
        return rc;
    }
    if (int rc = case_h2_recurrent_state()) {
        return rc;
    }
    if (int rc = case_h3_slot_table_and_order()) {
        return rc;
    }
    if (int rc = case_h3_multi_extent()) {
        return rc;
    }
    if (int rc = case_h3_cost_order()) {
        return rc;
    }
    if (int rc = case_h3_strict_prefix()) {
        return rc;
    }
    if (int rc = case_h3_pending_ranges()) {
        return rc;
    }
    if (int rc = case_h3_carve_mirroring()) {
        return rc;
    }
    if (int rc = case_h3_request_forms()) {
        return rc;
    }
    if (int rc = case_h6_n_chunk()) {
        return rc;
    }
    if (int rc = run_case("i1_pending_ranges_carve", case_i1_pending_ranges_carve)) {
        return rc;
    }
    if (int rc = run_case("i2_retained_runs_respect_ranges", case_i2_retained_runs_respect_ranges)) {
        return rc;
    }
    if (int rc = run_case("i3_ring_growth_cause", case_i3_ring_growth_cause)) {
        return rc;
    }
    if (int rc = run_case("i4_best_fit_across_tlsfs", case_i4_best_fit_across_tlsfs)) {
        return rc;
    }
    if (int rc = run_case("i8_constrained_head_first", case_i8_constrained_head_first)) {
        return rc;
    }
    if (int rc = run_case("i9_refit_exact_in_shrunk_rooms", case_i9_refit_exact_in_shrunk_rooms)) {
        return rc;
    }
    if (int rc = run_case("i10_opt_room_needs_a_carvable_span", case_i10_opt_room_needs_a_carvable_span)) {
        return rc;
    }
    if (int rc = run_case("i11_free_piece_under_a_cut_tenant", case_i11_free_piece_under_a_cut_tenant)) {
        return rc;
    }
    if (int rc = run_case("i12_refit_heads_search", case_i12_refit_heads_search)) {
        return rc;
    }
    if (int rc = run_case("i5_sub_slot_hole_after_heads", case_i5_sub_slot_hole_after_heads)) {
        return rc;
    }
    if (int rc = run_case("i6_single_gap_oracle", case_i6_single_gap_oracle)) {
        return rc;
    }
    if (int rc = run_case("i7_yield_span_under_a_range", case_i7_yield_span_under_a_range)) {
        return rc;
    }
    std::printf("test-kv-runtime-demotion: all ok\n");
    return 0;
}
