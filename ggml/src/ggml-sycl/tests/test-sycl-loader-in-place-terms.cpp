// The host WEIGHT zone charges the loader's SYCL_Host buffer once, and only the pool that holds it commits it; a host
// expert the loader serves in place takes no block of its own.
//
// On a load that does not map files, with evictable weights, the model loader binds the SYCL_Host buffer type for
// every weight the plan places and keeps that buffer for the model's life. When the backend scheduler sees one device,
// every chunk of that buffer comes from that device's pinned pool (the load's loader_device); when it sees more, the
// buffer is split across their pools by layer and no one pool holds it (loader_device -1). The plan charges the buffer
// as the loader_residency term once, host-wide, when one pool holds it: every entry's bytes, each rounded up to the
// loader's tensor alignment (exact for a dense weight, a bound for an expert slice). Every card's host zones are sized
// from the plan, so only loader_device's commit terms carry the term; every other card's WEIGHT zone leaves it out.
// A plan made for loader_device also serves each host-planned PINNED expert from the buffer in place, under the
// buffer's owner, so the expert is counted once, inside loader_residency, as loader_served, and holds no block. A
// dense host weight still holds a block, and so does every host weight of a multi-device plan or of a plan made for
// another device: S1-PRELOAD copies them.
//
// Host-only: the fact rule is a constexpr function of the load's inputs, and the terms are a pure function of a
// hand-built plan. No queue, no device, no allocation.

#include "ggml-impl.h"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdio>
#include <string>

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
// loader facts are the ones the builder copies for that load.
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

// The zone record populate_host_zone_sizing() makes from the terms: the WEIGHT zone's total and its loader term.
void record_weight_zone(placement_plan & plan) {
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    plan.host_zone_weight_bytes        = terms.total();
    plan.host_zone_loader_bytes        = terms.loader_residency;
}

bool same_terms(const host_zone_commit_terms & a, const host_zone_commit_terms & b) {
    return a.weight == b.weight && a.kv == b.kv && a.staging == b.staging && a.scratch == b.scratch &&
           a.runtime == b.runtime && a.rounding == b.rounding && a.onednn_live == b.onednn_live && a.chunk == b.chunk;
}

size_t loader_bytes() {
    return align_up(kDense, TENSOR_ALIGNMENT) + kDenseHost + (kHostExperts + 1) * kExpert;
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
    // The loader's buffer holds every weight the plan places, the device-planned ones included.
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
    CHECK(card1.weight == record.weight - loader_bytes());
    CHECK(card1.kv == record.kv && card1.staging == record.staging && card1.scratch == record.scratch);
    CHECK(card1.runtime == record.runtime && card1.onednn_live == record.onednn_live && card1.chunk == record.chunk);
    CHECK(card1.zones() < record.zones());
    CHECK(card1.total() % (kChunk + ggml_sycl::pinned_chunk_pool::DEFAULT_ALIGNMENT) == 0);
    // The oracle: the same plan with the loader term taken out of its WEIGHT zone before the terms are computed.
    placement_plan without = plan;
    without.host_zone_weight_bytes -= loader_bytes();
    without.host_zone_loader_bytes = 0;
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

}  // namespace

int main() {
    check_fact_rule();
    check_no_loader_buffer();
    check_loader_serves_host_experts();
    check_multi_device_charges_loader_once();
    check_other_device_plan_copies_host_experts();
    check_split_loader_charges_nothing();
    check_mapped_expert_is_neither();
    if (failures != 0) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
