// The host WEIGHT zone charges the loader's SYCL_Host buffer once, and only the pool that holds it commits it; a host
// expert the loader serves in place takes no block of its own.
//
// On a load that does not map files, with evictable weights, the model loader binds the SYCL_Host buffer type for the
// weights the plan places and keeps that buffer for the model's life. When the backend scheduler sees one device, every
// chunk of that buffer comes from that device's pinned pool (the load's loader_device); when it sees more, the buffer
// is split across their pools by layer and no one pool holds it (loader_device -1). The plan charges the buffer as the
// loader_residency term once, host-wide, when one pool holds it: the bytes of every entry the buffer holds, each
// rounded up to the loader's tensor alignment (exact for a dense weight, a bound for an expert slice). Every card's
// host zones are sized from the plan, so only loader_device's commit terms carry the term; every other card's WEIGHT
// zone leaves it out. The buffer holds every entry of a bound inventory, the late plan's. An unbound inventory, the
// early plan's, also names the host-planned dense weights the cpu and plain layouts put in a CPU-owned buffer instead,
// and neither the term nor the replay of the buffers counts those.
// A plan made for loader_device also serves each host-planned PINNED expert from the buffer in place, under the
// buffer's owner, so the expert is counted once, inside loader_residency, as loader_served, and holds no block. A
// dense host weight still holds a block, and so does every host weight of a multi-device plan or of a plan made for
// another device: S1-PRELOAD copies them.
//
// The loader's buffers are charged as the pool holds them. ggml-alloc splits the loader's tensors into buffers no
// larger than the buffer type's chunk cap, next-fit in their order, and the pinned pool serves each buffer from one
// chunk segment of the WEIGHT zone, so a chunk holds one buffer of the loader's size and the rest of the chunk is lost
// to it. The commit terms charge the zone bytes those buffers take (loader_zone) and the chunks the pool grows for a
// buffer no segment holds (loader_grown): once the plan admits, the loader's WEIGHT zone allocations need no growth the
// plan did not count, where the pool's own budget is at least the plan's host budget. The oracle replays the pool's own
// rules (configure_host_zones' segments, zone_alloc's first fit, grow_zone's one chunk sized to the request), and it is
// checked against the B70 Qwen load that admitted a plan charging raw bytes: the zone held 34 of the 37 buffers, and
// the pool grew two chunks and was refused the third.
//
// Host-only: the fact rule is a constexpr function of the load's inputs, and the terms are a pure function of a
// hand-built plan. No queue, no device, no allocation.

#include "ggml-impl.h"
#include "ggml-sycl/unified-cache.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::host_weight_zone_terms;
using ggml_sycl::host_zone_commit_terms;
using ggml_sycl::placement_host_backing;
using ggml_sycl::placement_loader_fact;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr size_t kDense       = 1000;  // not a multiple of the alignment: its loader bytes round up
constexpr size_t kDenseHost   = 2048;
constexpr size_t kExpert      = 4096;
constexpr int    kHostExperts = 4;
constexpr size_t kChunk       = 64 * 1024;  // small, so the loader term moves the chunk count
constexpr int    kMultiDevice = -1;         // the plan device of the multi-device builder

size_t align_up(size_t n, size_t a) {
    return (n + a - 1) / a * a;
}

// One dense weight on the device, one on the host, four host experts and one device expert, every backing PINNED as a
// load that maps no file plans them. The weights are evictable exactly when the loader holds them, the buffer is in
// `loader_device`'s pool, and the plan is made for `plan_device` (kMultiDevice for the multi-device builder); the
// loader facts are the ones the builder copies for that load. The plan keeps loader_holds_host_dense's default, a bound
// inventory's, so the loader's buffer holds the host dense weight too.
placement_plan make_plan(bool holds, int loader_device, int plan_device) {
    placement_plan plan{};
    plan.entries.push_back({ "blk.0.attn_q.weight", kDense, kDense, 0, placement_priority::ATTENTION, 0, -1,
                             expert_tensor_role::UNKNOWN, true, 0 });
    plan.entries.push_back({ "blk.0.attn_k.weight", kDenseHost, kDenseHost, 0, placement_priority::ATTENTION, 0, -1,
                             expert_tensor_role::UNKNOWN, false, -1 });
    for (int e = 0; e < kHostExperts + 1; ++e) {
        const bool host = e < kHostExperts;
        plan.entries.push_back({ "blk.0.ffn_up_exps.weight", kExpert, kExpert, 0, placement_priority::MOE_UP, 0, e,
                                 expert_tensor_role::UP, !host, host ? -1 : 0 });
    }
    const placement_loader_fact fact = ggml_sycl::placement_loader_fact_for(false, holds, loader_device, plan_device);
    plan.device_id                   = plan_device;
    plan.multi_device                = plan_device == kMultiDevice;
    plan.loader_holds_weights        = fact.holds_weights;
    plan.loader_device               = fact.device;
    plan.loader_serves_host_experts  = fact.serves_host_experts;
    plan.build_index();
    return plan;
}

// The inventory make_plan's plan was made from, one row per tensor in the plan's order: an expert tensor is the sum of
// its slices.
std::vector<ggml_sycl::placement_tensor_info> inventory_of(const placement_plan & plan) {
    std::vector<ggml_sycl::placement_tensor_info> rows;
    for (const auto & e : plan.entries) {
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const auto & r) { return r.name == e.name; });
        if (it == rows.end()) {
            rows.emplace_back(e.name, e.src_size);
        } else {
            it->size += e.src_size;
        }
    }
    return rows;
}

// The zone record populate_host_zone_sizing() makes from the terms: the WEIGHT zone's total, its loader term and, with
// the term, the loader's tensors, through the replay's own function.
void record_weight_zone(placement_plan & plan) {
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    plan.host_zone_weight_bytes        = terms.total();
    plan.host_zone_loader_bytes        = terms.loader_residency;
    plan.loader_tensor_bytes           = ggml_sycl::placement_plan_loader_tensor_bytes(plan, inventory_of(plan));
}

bool same_terms(const host_zone_commit_terms & a, const host_zone_commit_terms & b) {
    return a.weight == b.weight && a.kv == b.kv && a.staging == b.staging && a.scratch == b.scratch &&
           a.runtime == b.runtime && a.rounding == b.rounding && a.loader_zone == b.loader_zone &&
           a.loader_grown == b.loader_grown && a.onednn_live == b.onednn_live && a.chunk == b.chunk;
}

// The loader's buffer when it holds every entry of make_plan's plan: a bound inventory's, or under the off layout.
size_t loader_bytes() {
    return align_up(kDense, TENSOR_ALIGNMENT) + kDenseHost + (kHostExperts + 1) * kExpert;
}

// The same buffer without the host dense weight, which an unbound inventory under the cpu or plain layout names but the
// loader puts in a CPU-owned buffer.
size_t loader_bytes_without_host_dense() {
    return align_up(kDense, TENSOR_ALIGNMENT) + (kHostExperts + 1) * kExpert;
}

void check_fact_rule() {
    // The loader binds SYCL_Host only for evictable weights on a load that maps no file. The buffer is in the load's
    // loader_device's pool, and a plan serves host experts from it in place only when made for that device: a
    // multi-device plan stages its experts for every device, and a chunk is ring-streamable only on the device whose
    // cache queue allocated it.
    for (const bool maps : { false, true }) {
        for (const bool evictable : { false, true }) {
            for (const int loader_device : { -1, 0, 1 }) {
                for (const int plan_device : { kMultiDevice, 0, 1 }) {
                    const placement_loader_fact f =
                        ggml_sycl::placement_loader_fact_for(maps, evictable, loader_device, plan_device);
                    const bool holds = !maps && evictable;
                    CHECK(f.holds_weights == holds);
                    CHECK(f.device == (holds ? loader_device : -1));
                    CHECK(f.serves_host_experts == (holds && loader_device >= 0 && plan_device == loader_device));
                }
            }
        }
    }
}

void check_no_loader_buffer() {
    // The control: a plan whose loader holds no weight keeps today's terms, every host weight a block.
    const placement_plan         plan  = make_plan(false, 0, 0);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    const size_t                 host  = kDenseHost + kHostExperts * kExpert;
    CHECK(plan.loader_device == -1);
    CHECK(terms.host_weights == host);
    CHECK(terms.blocks == 1 + kHostExperts);
    CHECK(terms.loader_residency == 0);
    CHECK(terms.loader_served == 0);
    CHECK(terms.tlsf_bound == terms.blocks * ggml_sycl::tlsf_allocator::block_grain);
    CHECK(terms.total() == host + terms.host_compute_reserve + terms.tlsf_bound);
    CHECK(terms.matches_charge(host));
}

void check_loader_serves_host_experts() {
    const placement_plan plan = make_plan(true, 0, 0);
    CHECK(plan.loader_device == 0);
    CHECK(plan.loader_serves_host_experts);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    // The dense host weight is copied, so it alone holds a block; the experts are served from the loader's buffer.
    CHECK(terms.host_weights == kDenseHost);
    CHECK(terms.blocks == 1);
    CHECK(terms.loader_served == kHostExperts * kExpert);
    // The loader's buffer of a bound inventory holds every weight the plan places, the device-planned ones and the
    // host dense one included.
    CHECK(terms.loader_residency == loader_bytes());
    CHECK(terms.tlsf_bound == ggml_sycl::tlsf_allocator::block_grain);
    CHECK(terms.total() == kDenseHost + loader_bytes() + terms.host_compute_reserve + terms.tlsf_bound);
    // The plan charged every host weight to weight_host_bytes; the served experts are part of that charge.
    CHECK(terms.matches_charge(kDenseHost + kHostExperts * kExpert));
    // Positive control: a charge that left the served experts out is a disagreement.
    CHECK(!terms.matches_charge(kDenseHost));
}

// The terms of a plan that charges the loader's buffer but copies every host weight: a multi-device plan, or a plan
// made for a device whose pool does not hold the buffer.
void check_charged_and_copied(const placement_plan & plan) {
    CHECK(plan.loader_holds_weights);
    CHECK(plan.loader_device == 0);
    CHECK(!plan.loader_serves_host_experts);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    const size_t                 host  = kDenseHost + kHostExperts * kExpert;
    CHECK(terms.loader_residency == loader_bytes());
    CHECK(terms.loader_served == 0);
    CHECK(terms.host_weights == host);
    CHECK(terms.blocks == 1 + kHostExperts);
    CHECK(terms.total() == host + loader_bytes() + terms.host_compute_reserve + terms.tlsf_bound);
    CHECK(terms.matches_charge(host));
}

void check_multi_device_charges_loader_once() {
    // (i) A multi-device plan whose loader buffer is in device 0's pool: the plan charges the buffer once, host-wide,
    // and copies every host weight. Device 0's commit terms are the plan's own; device 1's WEIGHT zone leaves the
    // buffer out, and its chunk rounding follows.
    placement_plan plan = make_plan(true, 0, kMultiDevice);
    check_charged_and_copied(plan);
    record_weight_zone(plan);
    CHECK(plan.host_zone_loader_bytes == loader_bytes());
    const host_zone_commit_terms record = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, kChunk);
    const host_zone_commit_terms card0  = ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, record, 0);
    const host_zone_commit_terms card1  = ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, record, 1);
    CHECK(same_terms(card0, record));
    // One buffer smaller than a chunk: the zone bytes it takes are its own.
    CHECK(record.loader_zone == loader_bytes());
    CHECK(card1.weight == record.weight - loader_bytes());
    CHECK(card1.loader_zone == 0 && card1.loader_grown == 0);
    CHECK(card1.kv == record.kv && card1.staging == record.staging && card1.scratch == record.scratch);
    CHECK(card1.runtime == record.runtime && card1.onednn_live == record.onednn_live && card1.chunk == record.chunk);
    CHECK(card1.zones() < record.zones());
    CHECK(card1.total() % (kChunk + ggml_sycl::pinned_chunk_pool::DEFAULT_ALIGNMENT) == 0);
    // The oracle: the same plan with the loader term taken out of its WEIGHT zone before the terms are computed.
    placement_plan without = plan;
    without.host_zone_weight_bytes -= loader_bytes();
    without.host_zone_loader_bytes = 0;
    without.loader_tensor_bytes.clear();
    CHECK(same_terms(card1, ggml_sycl::placement_plan_host_zone_commit_terms(without, 0, kChunk)));
}

void check_other_device_plan_copies_host_experts() {
    // A single-device plan made for device 1 while the loader's buffer is in device 0's pool: the buffer is charged
    // once, but the plan's own device cannot stream from it, so every host weight is copied, and device 1's commit
    // terms leave the buffer out.
    placement_plan plan = make_plan(true, 0, 1);
    check_charged_and_copied(plan);
    record_weight_zone(plan);
    const host_zone_commit_terms record = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, kChunk);
    CHECK(ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, record, 1).weight ==
          record.weight - loader_bytes());
}

void check_split_loader_charges_nothing() {
    // (ii) loader_device -1: the scheduler sees more than one device, so the buffer is split across their pools by
    // layer. No pool holds it, so no plan charges it or serves from it, every host weight is copied, and no device's
    // commit terms carry it.
    placement_plan plan = make_plan(true, -1, 0);
    CHECK(plan.loader_holds_weights);
    CHECK(plan.loader_device == -1);
    CHECK(!plan.loader_serves_host_experts);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    const size_t                 host  = kDenseHost + kHostExperts * kExpert;
    CHECK(terms.loader_residency == 0);
    CHECK(terms.loader_served == 0);
    CHECK(terms.host_weights == host);
    CHECK(terms.blocks == 1 + kHostExperts);
    CHECK(terms.matches_charge(host));
    record_weight_zone(plan);
    CHECK(plan.host_zone_loader_bytes == 0);
    const host_zone_commit_terms record = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, kChunk);
    for (const int device : { 0, 1 }) {
        CHECK(same_terms(ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, record, device), record));
    }
    // Positive control: the same entries with the buffer in device 0's pool charge it.
    CHECK(ggml_sycl::placement_plan_host_weight_zone_terms(make_plan(true, 0, 0)).loader_residency == loader_bytes());
}

void check_mapped_expert_is_neither() {
    // A FILE_MAPPED host expert is served from the file's mapping, not the loader's buffer: neither a block nor
    // loader_served, under either fact.
    for (const bool serves : { false, true }) {
        placement_plan plan = make_plan(serves, 0, 0);
        CHECK(plan.loader_serves_host_experts == serves);
        for (auto & e : plan.entries) {
            if (e.expert_id == 0) {
                e.host_backing = placement_host_backing::FILE_MAPPED;
            }
        }
        const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
        const size_t                 left  = (kHostExperts - 1) * kExpert;
        CHECK(terms.host_weights + terms.loader_served == kDenseHost + left);
    }
}

void check_loader_holds_what_it_places() {
    // The term and the replay count exactly the tensors the loader puts in its SYCL_Host buffer. The early pass plans
    // an unbound inventory, which names every weight of the file; under the cpu and plain layouts the loader puts the
    // host dense weight in a CPU-owned buffer, so neither counts it. Under off it stays in SYCL_Host. The late pass
    // plans a bound inventory, which keeps only the tensors on SYCL buffers, so both count every entry whatever the
    // layout.
    struct arm {
        uint8_t layout;
        bool    unbound;
        bool    holds_host_dense;
    };

    const arm arms[] = {
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_CPU,   true,  false },
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_PLAIN, true,  false },
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_OFF,   true,  true  },
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_CPU,   false, true  },
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_PLAIN, false, true  },
        { GGML_SYCL_HOST_WEIGHT_LAYOUT_OFF,   false, true  },
    };
    for (const arm & a : arms) {
        const bool holds = ggml_sycl::placement_loader_holds_host_dense(a.layout, a.unbound);
        if (holds != a.holds_host_dense) {
            std::fprintf(stderr, "  layout=%u unbound=%d: the loader holds the host dense weight=%d, want %d\n",
                         (unsigned) a.layout, (int) a.unbound, (int) holds, (int) a.holds_host_dense);
        }
        CHECK(holds == a.holds_host_dense);
        placement_plan plan          = make_plan(true, 0, 0);
        plan.loader_holds_host_dense = holds;
        record_weight_zone(plan);
        const size_t want = a.holds_host_dense ? loader_bytes() : loader_bytes_without_host_dense();
        CHECK(plan.host_zone_loader_bytes == want);
        // The replay's tensors, from the plan's inventory: the device dense weight, the expert tensor and, when the
        // buffer holds it, the host dense weight, each at the loader's alignment.
        const std::vector<size_t> replay = ggml_sycl::placement_plan_loader_tensor_bytes(plan, inventory_of(plan));
        CHECK(replay.size() == (a.holds_host_dense ? 3u : 2u));
        size_t replayed = 0;
        for (const size_t b : replay) {
            replayed += b;
        }
        CHECK(replayed == want);
        // The plan still charges the host dense weight a pinned block, whichever buffer the loader then gives it.
        CHECK(ggml_sycl::placement_plan_host_weight_zone_terms(plan).host_weights == kDenseHost);
    }
    // A device-planned dense weight and every expert stay in the buffer under every layout and pass.
    placement_plan plan          = make_plan(true, 0, 0);
    plan.loader_holds_host_dense = false;
    for (const auto & e : plan.entries) {
        const bool host_dense = !e.on_device && e.expert_id < 0;
        CHECK(ggml_sycl::placement_plan_loader_holds_entry(plan, e) == !host_dense);
    }
}

void check_layout_byte_reads_one_way() {
    // Each layout as the readers take it: under off the loader keeps a host dense weight in its SYCL_Host buffer, under
    // cpu ggml-cpu repacks it into a buffer of its own on a mapping load, and under plain the CPU reads the mapping.
    using ggml_sycl::host_weight_layout_facts;
    const host_weight_layout_facts off   = ggml_sycl::host_weight_layout_facts_of(GGML_SYCL_HOST_WEIGHT_LAYOUT_OFF);
    const host_weight_layout_facts cpu   = ggml_sycl::host_weight_layout_facts_of(GGML_SYCL_HOST_WEIGHT_LAYOUT_CPU);
    const host_weight_layout_facts plain = ggml_sycl::host_weight_layout_facts_of(GGML_SYCL_HOST_WEIGHT_LAYOUT_PLAIN);
    CHECK(off.known && off.loader_holds_dense && !off.dense_cpu_owned);
    CHECK(cpu.known && !cpu.loader_holds_dense && cpu.dense_cpu_owned);
    CHECK(plain.known && !plain.loader_holds_dense && !plain.dense_cpu_owned);
    // A byte no layout names is a malformed inventory, which every entry point refuses. A plan built from one anyway
    // reads it on the side that charges more: the loader's term holds the dense weight, and a mapping load charges it
    // as a CPU-owned buffer rather than a mapping no zone holds.
    // The enum has no fixed underlying type, so its values are 0-3 and casting 255 to it is undefined. A constant
    // evaluation that reaches such a cast does not compile, so this pins that the byte is refused before the cast.
    static_assert(!ggml_sycl::host_weight_layout_facts_of(255).known, "an unknown byte must not reach the enum cast");
    for (const unsigned byte : { 3u, 255u }) {
        const host_weight_layout_facts unknown = ggml_sycl::host_weight_layout_facts_of(static_cast<uint8_t>(byte));
        CHECK(!unknown.known);
        CHECK(unknown.loader_holds_dense && unknown.dense_cpu_owned);
        CHECK(ggml_sycl::placement_loader_holds_host_dense(static_cast<uint8_t>(byte), true));
    }
}

bool role_says(const placement_plan & plan, const char * text) {
    const std::string role =
        ggml_sycl::placement_plan_loader_role(plan, ggml_sycl::placement_plan_host_weight_zone_terms(plan));
    return role.find(text) != std::string::npos;
}

void check_loader_role_follows_what_it_serves() {
    // The WEIGHT-zone line names the loader's buffer as serving host experts in place only when the terms serve some
    // (loader_served > 0). A plan made for loader_device whose host weights are all dense, a dense model's, serves
    // none, so its line says the buffer is charged and serves nothing in place.
    constexpr const char * kServing = "serving the host experts in place";
    constexpr const char * kCharged = "charged once, in loader_device's pool";
    CHECK(role_says(make_plan(true, 0, 0), kServing));

    placement_plan dense = make_plan(true, 0, 0);
    dense.entries.erase(std::remove_if(dense.entries.begin(), dense.entries.end(),
                                       [](const ggml_sycl::placement_entry & e) { return e.expert_id >= 0; }),
                        dense.entries.end());
    dense.build_index();
    CHECK(dense.loader_serves_host_experts);
    CHECK(ggml_sycl::placement_plan_host_weight_zone_terms(dense).loader_served == 0);
    CHECK(!role_says(dense, kServing));
    CHECK(role_says(dense, kCharged));

    // The other roles follow their facts as before.
    CHECK(role_says(make_plan(true, 0, 1), kCharged));
    CHECK(role_says(make_plan(true, -1, 0), "not charged"));
    CHECK(role_says(make_plan(false, 0, 0), "binds no SYCL_Host buffer"));
}

// ---- The loader's buffers in the pinned pool ----

constexpr size_t kMiB       = 1024 * 1024;
constexpr size_t kPoolChunk = ggml_sycl::pinned_chunk_pool::CHUNK_SIZE;         // the default pool chunk, 2 GiB
constexpr size_t kHeader    = ggml_sycl::pinned_chunk_pool::DEFAULT_ALIGNMENT;  // a chunk's header, and its alignment
constexpr size_t kGrain     = ggml_sycl::tlsf_allocator::block_grain;           // a zone segment's allocation grain

// ggml-alloc's split of a buffer type's tensors (ggml_backend_buft_alloc_buffer_n_plan): next-fit in their order, a
// buffer closed when the next tensor would take it past max_size.
std::vector<size_t> split_buffers(const std::vector<size_t> & tensors, size_t max_size) {
    std::vector<size_t> buffers;
    size_t              cur = 0;
    for (const size_t t : tensors) {
        if (cur > 0 && cur + t > max_size) {
            buffers.push_back(cur);
            cur = t;
        } else {
            cur += t;
        }
    }
    if (cur > 0) {
        buffers.push_back(cur);
    }
    return buffers;
}

// The largest buffer the SYCL_Host buffer type reports during a load, as ggml_sycl_host_buft_max_size answers it on a
// fresh WEIGHT zone whose largest free block is a whole chunk: the chunk cap (0: none), never below the default pool
// chunk, which zone growth can always add.
size_t load_max_size(size_t cap, size_t chunk) {
    const size_t optimistic = std::max(chunk, kPoolChunk);
    return cap == 0 ? optimistic : std::min(cap, optimistic);
}

struct pool_run {
    size_t placed       = 0;  // buffers served from the zone's own segments
    size_t grown_chunks = 0;  // chunks grown for the others
    size_t grown_bytes  = 0;  // those chunks' backing, headers included
    size_t short_zone   = 0;  // chunks grown for a buffer one chunk holds: the zone was too small for it
};

// The loader's buffers allocated into a fresh pool. configure_host_zones carves the WEIGHT zone first, from logical 0,
// over chunks of `chunk`: whole segments, then what is left. zone_alloc serves each buffer, rounded to the grain, from
// the first segment with room, and grow_zone adds one chunk for a buffer no segment holds, sized from the request
// unified_alloc makes for it (zone_grow_request, at the alignment the SYCL_Host buffer type ends up with).
pool_run run_pool(const std::vector<size_t> & buffers, size_t weight_zone, size_t chunk) {
    std::vector<size_t> room(weight_zone / chunk, chunk);
    if (weight_zone % chunk != 0) {
        room.push_back(weight_zone % chunk);
    }
    pool_run run;
    for (const size_t buffer : buffers) {
        const size_t bytes  = align_up(buffer, kGrain);
        const auto   served = std::find_if(room.begin(), room.end(), [bytes](size_t r) { return r >= bytes; });
        if (served != room.end()) {
            *served -= bytes;
            run.placed++;
            continue;
        }
        const ggml_sycl::pinned_chunk_pool::chunk_footprint grown = ggml_sycl::pinned_chunk_pool::chunk_footprint_for(
            chunk, ggml_sycl::pinned_chunk_pool::zone_grow_request(buffer, kHeader));
        run.grown_chunks++;
        run.grown_bytes += grown.backing;
        run.short_zone += bytes <= chunk ? 1 : 0;
        CHECK(grown.usable >= bytes);  // the chunk grown for the buffer holds it
        room.push_back(grown.usable >= bytes ? grown.usable - bytes : 0);
    }
    return run;
}

// The 37 buffers the B70 Qwen3.8 IQ4_NL load requested (-lm none, ub 2048): the next-fit split of its 1223 tensors in
// creation order at a 2 GiB cap, which reproduces every logged request size.
std::vector<size_t> qwen_b70_buffers() {
    std::vector<size_t> b = { 1910764416, 2008730496 };
    for (int i = 0; i < 11; ++i) {
        b.insert(b.end(), { 2054480768, 1973709696, 1973709696 });
    }
    b.insert(b.end(), { 2054480768, 1420810240 });
    return b;
}

constexpr size_t kQwenLoader = 73415687680;  // that load's loader_residency
constexpr size_t kQwenWeight = 73482796544;  // and the WEIGHT zone the plan charged it, 70078.7 MB

// A routed-expert model shaped like that load: two 644 MiB embedding tensors, then per layer a few small tensors and
// three 450 MiB expert tensors.
std::vector<size_t> moe_tensors() {
    std::vector<size_t> t = { 675430400, 675430400 };
    for (int layer = 0; layer < 48; ++layer) {
        for (const size_t s : { 13107200, 5242880, 471859200, 2621440, 471859200, 471859200, 10240 }) {
            t.push_back(align_up(s, TENSOR_ALIGNMENT));
        }
    }
    return t;
}

// The same tensors in other orders: the plan replays the order the loader allocates in, whatever it is.
std::vector<std::vector<size_t>> orders_of(const std::vector<size_t> & tensors) {
    std::vector<size_t> reversed(tensors.rbegin(), tensors.rend());
    std::vector<size_t> ascending = tensors;
    std::sort(ascending.begin(), ascending.end());
    std::vector<size_t> descending(ascending.rbegin(), ascending.rend());
    return { tensors, reversed, ascending, descending };
}

size_t sum_of(const std::vector<size_t> & v) {
    size_t s = 0;
    for (const size_t x : v) {
        s += x;
    }
    return s;
}

// A plan made for device 0 whose loader buffer holds `tensors` in device 0's pool, with the zone record the planner
// makes: the WEIGHT zone is the loader term plus `other_weight` bytes.
placement_plan make_loader_plan(const std::vector<size_t> & tensors, size_t cap, size_t other_weight) {
    placement_plan plan         = make_plan(true, 0, 0);
    plan.loader_buffer_cap      = cap;
    plan.loader_tensor_bytes    = tensors;
    plan.host_zone_loader_bytes = sum_of(tensors);
    plan.host_zone_weight_bytes = plan.host_zone_loader_bytes + other_weight;
    return plan;
}

void check_pool_oracle_reproduces_the_device_log() {
    // The positive control: the oracle must reproduce what the pool did. The plan charged raw bytes, a 70078.7 MB
    // WEIGHT zone (34 whole chunks and a 446.7 MB rest); the zone held 34 buffers, the pool grew two chunks for the
    // next two and was refused the third by its budget.
    const std::vector<size_t> qwen = qwen_b70_buffers();
    CHECK(qwen.size() == 37);
    CHECK(sum_of(qwen) == kQwenLoader);
    // Each buffer is more than half a chunk, so each closes the one before it: the split leaves them as they are.
    CHECK(split_buffers(qwen, kPoolChunk) == qwen);
    const pool_run run = run_pool(qwen, kQwenWeight, kPoolChunk);
    CHECK(run.placed == 34);
    CHECK(run.grown_chunks == 3 && run.short_zone == 3);
}

// The footprint is the pool's: with loader_zone bytes of WEIGHT zone every buffer a chunk holds is served without
// growth, one grain less leaves one short, and the chunks grown are the ones the footprint counts.
void check_footprint_against_pool(const std::vector<size_t> & tensors, size_t cap, size_t chunk) {
    const size_t                                     max_size = load_max_size(cap, chunk);
    const std::vector<size_t>                        buffers  = split_buffers(tensors, max_size);
    const ggml_sycl::placement_loader_pool_footprint fp =
        ggml_sycl::placement_loader_pool_footprint_for(tensors, max_size, chunk);
    CHECK(fp.buffers == buffers.size());
    const pool_run fits = run_pool(buffers, fp.zone, chunk);
    CHECK(fits.short_zone == 0);
    CHECK(fits.grown_bytes == fp.grown);
    CHECK(fits.placed == fp.buffers - fits.grown_chunks);
    CHECK(fp.chunks * chunk >= fp.zone && (fp.chunks == 0 || (fp.chunks - 1) * chunk < fp.zone));
    if (fp.zone >= kGrain) {
        CHECK(run_pool(buffers, fp.zone - kGrain, chunk).short_zone > 0);
    }
}

void check_loader_footprint_is_the_pools() {
    const std::vector<size_t> qwen = qwen_b70_buffers();
    check_footprint_against_pool(qwen, kPoolChunk, kPoolChunk);
    const ggml_sycl::placement_loader_pool_footprint fp =
        ggml_sycl::placement_loader_pool_footprint_for(qwen, kPoolChunk, kPoolChunk);
    // 37 chunks, the last filled to its buffer: 36 x 2 GiB + 1355.0 MB.
    CHECK(fp.buffers == 37 && fp.chunks == 37 && fp.grown == 0);
    CHECK(fp.zone == 36 * kPoolChunk + align_up(1420810240, kGrain));
    for (const std::vector<size_t> & order : orders_of(moe_tensors())) {
        check_footprint_against_pool(order, 0, kPoolChunk);
        check_footprint_against_pool(order, kPoolChunk, kPoolChunk);
        check_footprint_against_pool(order, 1024 * kMiB, kPoolChunk);  // a smaller cap: more, smaller buffers
    }
    // One buffer smaller than a chunk takes its own bytes, no whole chunk.
    const std::vector<size_t> small = { 300 * kMiB, 200 * kMiB };
    CHECK(ggml_sycl::placement_loader_pool_footprint_for(small, kPoolChunk, kPoolChunk).zone == 500 * kMiB);
    CHECK(ggml_sycl::placement_loader_pool_footprint_for(small, kPoolChunk, kPoolChunk).chunks == 1);
}

void check_oversize_buffer_grows_a_planned_chunk() {
    // A tensor larger than a chunk is a buffer of its own that no segment holds: the pool grows one chunk sized to it.
    const size_t        big     = align_up(2856 * kMiB + 12345, TENSOR_ALIGNMENT);
    std::vector<size_t> tensors = moe_tensors();
    tensors.insert(tensors.begin() + 5, big);
    check_footprint_against_pool(tensors, kPoolChunk, kPoolChunk);
    const ggml_sycl::placement_loader_pool_footprint fp =
        ggml_sycl::placement_loader_pool_footprint_for(tensors, kPoolChunk, kPoolChunk);
    CHECK(fp.grown == align_up(align_up(big, kGrain), kHeader) + kHeader);
    // A pool chunk smaller than the buffer type's cap: every buffer past a chunk grows its own.
    for (const std::vector<size_t> & order : orders_of(moe_tensors())) {
        check_footprint_against_pool(order, kPoolChunk, 1024 * kMiB);
    }
    CHECK(ggml_sycl::placement_loader_pool_footprint_for(moe_tensors(), kPoolChunk, 1024 * kMiB).grown > 0);
}

// A chunk the pool grows for one contiguous request no chunk holds serves that request. zone_alloc aligns the size to
// the request's alignment and asks the chunk's fresh TLSF for it at the larger of that alignment and the grain, and the
// chunk's arena is chunk_footprint_for(chunk, zone_grow_request(size, alignment)).usable bytes. Every residue of the
// size modulo the grain that ggml-alloc's 32 B padding gives, at each alignment the TLSF takes (it refuses one above
// the grain), and a request of exactly one chunk.
void check_grown_chunk_serves_its_request() {
    using ggml_sycl::pinned_chunk_pool;
    const size_t chunk  = 64 * kMiB;
    const auto   served = [&](size_t size, size_t alignment) {
        const size_t usable =
            pinned_chunk_pool::chunk_footprint_for(chunk, pinned_chunk_pool::zone_grow_request(size, alignment)).usable;
        ggml_sycl::tlsf_allocator fresh(usable);
        const bool ok = fresh.allocate(align_up(size, alignment), std::max(alignment, kGrain)) != SIZE_MAX;
        if (!ok) {
            std::printf("not served: size=%zu (residue %zu) alignment=%zu arena=%zu\n", size, size % kGrain, alignment,
                          usable);
        }
        return ok;
    };
    for (const size_t alignment : { kHeader, 2 * kHeader, kGrain }) {
        for (size_t residue = 0; residue < kGrain; residue += TENSOR_ALIGNMENT) {
            CHECK(served(3 * chunk + residue, alignment));
        }
    }
    CHECK(served(chunk, kHeader));
}

// What the oracle says of an admitted plan: every chunk the loader grows is one the plan counted.
void check_admitted_plan(const placement_plan & plan, size_t chunk) {
    const host_zone_commit_terms terms = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, chunk);
    const std::vector<size_t>    buffers =
        split_buffers(plan.loader_tensor_bytes, load_max_size(plan.loader_buffer_cap, chunk));
    const pool_run run = run_pool(buffers, terms.weight, chunk);
    CHECK(run.short_zone == 0);
    CHECK(run.grown_bytes == terms.loader_grown);
    CHECK(terms.total() == terms.zones() + terms.runtime + terms.rounding + terms.loader_grown);
    // The WEIGHT zone is the rest of the record plus the loader's zone bytes.
    CHECK(terms.weight == plan.host_zone_weight_bytes - plan.host_zone_loader_bytes + terms.loader_zone);
    // Device 1's pool holds none of the buffer: its zone leaves the loader out, and it grows nothing for it.
    const host_zone_commit_terms card1 = ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, terms, 1);
    CHECK(card1.weight == plan.host_zone_weight_bytes - plan.host_zone_loader_bytes);
    CHECK(card1.loader_zone == 0 && card1.loader_grown == 0);
}

void check_admitted_plan_grows_nothing_unplanned() {
    constexpr size_t kReserve = 64 * kMiB;  // the WEIGHT zone's host_compute_reserve, its only other term here

    // The B70 Qwen load: the plan admits only what its 37 chunks need.
    placement_plan qwen = make_loader_plan(qwen_b70_buffers(), kPoolChunk, kQwenWeight - kQwenLoader);
    check_admitted_plan(qwen, kPoolChunk);
    // With the other terms its [HOST-LEDGER] line logged, the plan that charged raw bytes reproduces the logged commit
    // total (zones=79456897344 on its [HOST-PLAN] lazy line), and the footprint's total is what the lazy-table decision
    // then weighs against the logged budget, less the logged reserve.
    qwen.host_zone_kv_bytes      = 67108864;
    qwen.host_zone_staging_bytes = 1370882048;
    qwen.host_zone_scratch_bytes = 1180844032;
    qwen.onednn_scratchpad_bytes = 1048576000;
    placement_plan raw           = qwen;
    raw.loader_tensor_bytes.clear();
    const host_zone_commit_terms was   = ggml_sycl::placement_plan_host_zone_commit_terms(raw, 0, kPoolChunk);
    const host_zone_commit_terms terms = ggml_sycl::placement_plan_host_zone_commit_terms(qwen, 0, kPoolChunk);
    CHECK(was.weight == kQwenWeight && was.total() == 79456897344);
    constexpr long long kQwenBudget  = 85755396096;  // host_budget
    constexpr long long kQwenReserve = 2147483648;   // the room rule's reserve
    // What the lazy-table decision must fit: the STORAGE table's cache at its floor, its logged resident bytes less
    // the yieldable share handed out above the floor (2147483648 - 1721761792), and the block ring's charge (2097152).
    constexpr long long kQwenFloor   = 427819008;
    std::printf(
        "Qwen B70 WEIGHT zone: %zu bytes (%.1f MB) for the loader's 37 buffers, loader_zone=%zu loader_grown=%zu; "
        "raw bytes charged %zu (%.1f MB)\n",
        terms.weight, terms.weight / double(kMiB), terms.loader_zone, terms.loader_grown, kQwenWeight,
        kQwenWeight / double(kMiB));
    std::printf(
        "Qwen B70 commit total: %zu bytes (raw %zu); lazy fit room %lld bytes (raw %lld) for a %lld-byte floor\n",
        terms.total(), was.total(), kQwenBudget - (long long) terms.total() - kQwenReserve,
        kQwenBudget - (long long) was.total() - kQwenReserve, kQwenFloor);
    for (const std::vector<size_t> & order : orders_of(moe_tensors())) {
        check_admitted_plan(make_loader_plan(order, kPoolChunk, kReserve), kPoolChunk);
        check_admitted_plan(make_loader_plan(order, 0, kReserve), 1024 * kMiB);
    }
    std::vector<size_t> oversize = moe_tensors();
    oversize.insert(oversize.begin() + 5, align_up(2856 * kMiB + 12345, TENSOR_ALIGNMENT));
    check_admitted_plan(make_loader_plan(oversize, kPoolChunk, kReserve), kPoolChunk);
}

// A hand-built record whose WEIGHT zone is smaller than its own loader term (the planner's never is: the zone holds the
// term and the HOST_COMPUTE reserve). The WEIGHT term is the rest of the record, none here, plus the loader's zone
// bytes; it never wraps below zero.
void check_record_below_its_loader_term() {
    const std::vector<size_t> tensors  = { 100 * kMiB, align_up(2856 * kMiB + 12345, TENSOR_ALIGNMENT) };
    placement_plan            plan     = make_loader_plan(tensors, kPoolChunk, 0);
    plan.host_zone_weight_bytes        = 64 * kMiB;
    const host_zone_commit_terms terms = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, kPoolChunk);
    // The oversize tensor is a chunk of its own, so the zone holds only the first buffer.
    CHECK(terms.loader_zone == 100 * kMiB && terms.loader_grown > 0);
    CHECK(terms.weight == terms.loader_zone);
    CHECK(terms.total() >= terms.weight + terms.loader_grown);
}

// `plan` with its commit terms recorded on a fresh pool of `chunk` chunks and a host budget of `budget` bytes, as the
// planner's ledger record leaves it.
placement_plan with_budget(placement_plan plan, size_t chunk, size_t budget) {
    plan.host_zone_terms = ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, chunk);
    plan.host_budget     = budget;
    return plan;
}

size_t commit_total(const placement_plan & plan, size_t chunk) {
    return ggml_sycl::placement_plan_host_zone_commit_terms(plan, 0, chunk).total();
}

bool refused_as_loader_pool(const std::string & refusal) {
    return refusal.rfind("loader-pool-over-budget", 0) == 0;
}

bool carries(const std::string & refusal, const std::string & field, size_t value) {
    return refusal.find(" " + field + "=" + std::to_string(value)) != std::string::npos;
}

// A load whose loader buffers the pinned pool cannot hold within the host budget fails in grow_zone partway through, so
// the stage refuses a plan charged the loader term whose commit total, loader_grown included, exceeds its budget, by
// name and with its figures; it admits one within the budget. A plan without the term keeps the ledger's WARN-only
// check, however far over its budget it is.
void check_loader_pool_over_budget_is_refused() {
    const placement_plan         qwen  = make_loader_plan(qwen_b70_buffers(), kPoolChunk, kQwenWeight - kQwenLoader);
    const host_zone_commit_terms terms = ggml_sycl::placement_plan_host_zone_commit_terms(qwen, 0, kPoolChunk);
    const size_t                 total = terms.total();
    const std::string over = ggml_sycl::placement_plan_loader_pool_refusal(with_budget(qwen, kPoolChunk, total - 1));
    CHECK(refused_as_loader_pool(over));
    CHECK(carries(over, "total", total) && carries(over, "host_budget", total - 1));
    CHECK(carries(over, "loader_zone", terms.loader_zone) && carries(over, "loader_grown", 0));
    CHECK(carries(over, "loader_device", 0));
    std::printf("over budget: %s\n", over.c_str());
    CHECK(ggml_sycl::placement_plan_loader_pool_refusal(with_budget(qwen, kPoolChunk, total)).empty());

    // loader_grown is in the total the budget is weighed against: a budget that covers everything but the chunk grown
    // for an oversize buffer refuses.
    std::vector<size_t> oversize = moe_tensors();
    oversize.insert(oversize.begin() + 5, align_up(2856 * kMiB + 12345, TENSOR_ALIGNMENT));
    const placement_plan         big       = make_loader_plan(oversize, kPoolChunk, 64 * kMiB);
    const host_zone_commit_terms big_terms = ggml_sycl::placement_plan_host_zone_commit_terms(big, 0, kPoolChunk);
    CHECK(big_terms.loader_grown > 0);
    const std::string grown = ggml_sycl::placement_plan_loader_pool_refusal(
        with_budget(big, kPoolChunk, big_terms.total() - big_terms.loader_grown));
    CHECK(refused_as_loader_pool(grown) && carries(grown, "loader_grown", big_terms.loader_grown));
    CHECK(ggml_sycl::placement_plan_loader_pool_refusal(with_budget(big, kPoolChunk, big_terms.total())).empty());

    // A plan made for another device is charged the term too, host-wide: it is refused alike.
    placement_plan other = make_plan(true, 0, 1);
    record_weight_zone(other);
    CHECK(refused_as_loader_pool(
        ggml_sycl::placement_plan_loader_pool_refusal(with_budget(other, kChunk, commit_total(other, kChunk) - 1))));

    // No loader term: no buffer bound (weights not evictable), or a buffer split across the scheduler's devices. Both
    // keep the WARN-only ledger at any budget.
    for (placement_plan plan : { make_plan(false, 0, 0), make_plan(true, -1, kMultiDevice) }) {
        record_weight_zone(plan);
        CHECK(plan.host_zone_loader_bytes == 0 && commit_total(plan, kChunk) > 0);
        CHECK(ggml_sycl::placement_plan_loader_pool_refusal(with_budget(plan, kChunk, 0)).empty());
    }
}

// The loader pool's check at the preload sees a packing drift: it measures how far loader_device's WEIGHT zone grew
// from the capacity the zone had when the load decided the host zones, against the loader_grown the plan recorded.
// The oracle's pool gives the growth an admitted plan's buffers take.
void check_weight_zone_growth_is_checked() {
    std::vector<size_t> oversize = moe_tensors();
    oversize.insert(oversize.begin() + 5, align_up(2856 * kMiB + 12345, TENSOR_ALIGNMENT));
    const placement_plan plan = with_budget(make_loader_plan(oversize, kPoolChunk, 64 * kMiB), kPoolChunk, SIZE_MAX);
    const host_zone_commit_terms terms =
        ggml_sycl::placement_plan_host_zone_commit_terms_on_device(plan, plan.host_zone_terms, 0);
    const pool_run run =
        run_pool(split_buffers(plan.loader_tensor_bytes, load_max_size(plan.loader_buffer_cap, kPoolChunk)),
                 terms.weight, kPoolChunk);
    CHECK(terms.loader_grown > 0 && run.grown_bytes == terms.loader_grown);
    // configure_zones aligns the zone, and each chunk grow_zone adds puts its usable bytes in the zone; loader_grown
    // counts each with its header, so a growth the plan counted reads below it.
    const size_t configured = align_up(terms.weight, kHeader);
    const size_t grown      = run.grown_bytes - run.grown_chunks * kHeader;
    const auto   check      = [&](size_t baseline, size_t capacity) {
        return ggml_sycl::host_zone_weight_growth_check(0, terms, baseline, capacity);
    };
    CHECK(check(configured, configured + grown).empty());
    CHECK(check(configured, configured).empty());
    // The bound is inclusive: a zone grown by exactly loader_grown passes and one byte more warns. A real load grows
    // the usable bytes, one header per chunk below loader_grown, so this boundary is a synthetic capacity.
    CHECK(grown < terms.loader_grown);
    CHECK(check(configured, configured + terms.loader_grown).empty());
    CHECK(!check(configured, configured + terms.loader_grown + 1).empty());

    // One chunk more than the plan counted: the line names the device and the planned, actual and excess bytes.
    const size_t      actual = configured + grown + kPoolChunk;
    const std::string bug    = check(configured, actual);
    std::printf("drifted: %s\n", bug.c_str());
    CHECK(bug.rfind("[HOST-PLAN-BUG] device=0 ", 0) == 0);
    CHECK(carries(bug, "planned", configured + terms.loader_grown));
    CHECK(carries(bug, "actual", actual));
    CHECK(carries(bug, "excess", actual - configured - terms.loader_grown));
    CHECK(carries(bug, "baseline", configured) && carries(bug, "loader_grown", terms.loader_grown));

    // A zone an earlier load configured and grew, kept by this load: the growth it brought is in the baseline, so a
    // load whose buffers fit it, or that grows what its plan counted, is no alarm.
    const size_t kept = configured + 3 * kPoolChunk;
    CHECK(check(kept, kept).empty());
    CHECK(check(kept, kept + grown).empty());
    // But the plan replays the loader's buffers into an empty zone: a kept zone they do not fit, grown past this load's
    // own loader_grown, is a true alarm.
    CHECK(!check(kept, kept + grown + kPoolChunk).empty());

    // A plan that records no growth, its WEIGHT zone not a multiple of the zone alignment: no growth is no alarm, and
    // one chunk is.
    host_zone_commit_terms flat;
    flat.weight          = 1000;
    const size_t aligned = align_up(flat.weight, kHeader);
    CHECK(ggml_sycl::host_zone_weight_growth_check(1, flat, aligned, aligned).empty());
    const std::string one = ggml_sycl::host_zone_weight_growth_check(1, flat, aligned, aligned + kPoolChunk);
    CHECK(one.rfind("[HOST-PLAN-BUG] device=1 ", 0) == 0 && carries(one, "excess", kPoolChunk));
    // A capacity below its baseline grew nothing.
    CHECK(ggml_sycl::host_zone_weight_growth_check(1, flat, aligned, aligned - kHeader).empty());
}

}  // namespace

int main() {
    check_fact_rule();
    check_no_loader_buffer();
    check_loader_serves_host_experts();
    check_multi_device_charges_loader_once();
    check_other_device_plan_copies_host_experts();
    check_split_loader_charges_nothing();
    check_mapped_expert_is_neither();
    check_loader_role_follows_what_it_serves();
    check_loader_holds_what_it_places();
    check_layout_byte_reads_one_way();
    check_pool_oracle_reproduces_the_device_log();
    check_loader_footprint_is_the_pools();
    check_oversize_buffer_grows_a_planned_chunk();
    check_grown_chunk_serves_its_request();
    check_admitted_plan_grows_nothing_unplanned();
    check_record_below_its_loader_term();
    check_loader_pool_over_budget_is_refused();
    check_weight_zone_growth_is_checked();
    if (failures != 0) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
