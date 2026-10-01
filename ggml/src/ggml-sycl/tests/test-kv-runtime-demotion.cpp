#include "../kv-runtime-demotion.hpp"
#include "../tlsf-allocator.hpp"

#include <cstdio>
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
            return 0;
    }
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
    }
    std::printf("test-kv-runtime-demotion: all ok\n");
    return 0;
}
