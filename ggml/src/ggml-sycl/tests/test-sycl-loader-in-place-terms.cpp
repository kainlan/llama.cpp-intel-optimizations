// The host WEIGHT zone charges the loader's SYCL_Host buffer, and a host expert the loader serves in place takes no
// block of its own.
//
// On a load that does not map files, with evictable weights, the model loader binds the SYCL_Host buffer type for
// every weight the plan places and keeps that buffer for the model's life. The zone charges it as the
// loader_residency term: every entry's bytes, each rounded up to the loader's tensor alignment (exact for a dense
// weight, a bound for an expert slice). On a single-device plan S1-PRELOAD serves each host-planned PINNED expert from
// that buffer in place, under the buffer's owner, so the expert is counted once, inside loader_residency, as
// loader_served, and holds no block. A dense host weight still holds a block: S1-PRELOAD copies it.
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

size_t align_up(size_t n, size_t a) {
    return (n + a - 1) / a * a;
}

// One dense weight on the device, one on the host, four host experts and one device expert, every backing PINNED as a
// load that maps no file plans them.
placement_plan make_plan(bool holds, bool serves) {
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
    plan.loader_holds_weights       = holds;
    plan.loader_serves_host_experts = serves;
    plan.build_index();
    return plan;
}

void check_fact_rule() {
    // The loader binds SYCL_Host only for evictable weights on a load that maps no file; it serves experts in place
    // only on a single-device plan.
    for (const bool maps : { false, true }) {
        for (const bool evictable : { false, true }) {
            for (const bool multi : { false, true }) {
                const placement_loader_fact f = ggml_sycl::placement_loader_fact_for(maps, evictable, multi);
                CHECK(f.holds_weights == (!maps && evictable));
                CHECK(f.serves_host_experts == (!maps && evictable && !multi));
            }
        }
    }
}

void check_no_loader_buffer() {
    // The control: a plan whose loader holds no weight keeps today's terms, every host weight a block.
    const placement_plan         plan  = make_plan(false, false);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    const size_t                 host  = kDenseHost + kHostExperts * kExpert;
    CHECK(terms.host_weights == host);
    CHECK(terms.blocks == 1 + kHostExperts);
    CHECK(terms.loader_residency == 0);
    CHECK(terms.loader_served == 0);
    CHECK(terms.tlsf_bound == terms.blocks * ggml_sycl::tlsf_allocator::block_grain);
    CHECK(terms.total() == host + terms.host_compute_reserve + terms.tlsf_bound);
    CHECK(terms.matches_charge(host));
}

size_t loader_bytes() {
    return align_up(kDense, TENSOR_ALIGNMENT) + kDenseHost + (kHostExperts + 1) * kExpert;
}

void check_loader_serves_host_experts() {
    const placement_plan         plan  = make_plan(true, true);
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

void check_loader_holds_but_copies() {
    // A multi-device plan: the loader's buffer is charged, and every host weight is still copied into a block.
    const placement_plan         plan  = make_plan(true, false);
    const host_weight_zone_terms terms = ggml_sycl::placement_plan_host_weight_zone_terms(plan);
    const size_t                 host  = kDenseHost + kHostExperts * kExpert;
    CHECK(terms.host_weights == host);
    CHECK(terms.blocks == 1 + kHostExperts);
    CHECK(terms.loader_served == 0);
    CHECK(terms.loader_residency == loader_bytes());
    CHECK(terms.total() == host + loader_bytes() + terms.host_compute_reserve + terms.tlsf_bound);
    CHECK(terms.matches_charge(host));
}

void check_mapped_expert_is_neither() {
    // A FILE_MAPPED host expert is served from the file's mapping, not the loader's buffer: neither a block nor
    // loader_served, under either fact.
    for (const bool serves : { false, true }) {
        placement_plan plan = make_plan(serves, serves);
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
    check_loader_holds_but_copies();
    check_mapped_expert_is_neither();
    if (failures != 0) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
