// Host-only contract tests for the retained MoE route-batch foundation.
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/moe-resolved-batch.hpp"

#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

#define CHECK(c)                                                 \
    do {                                                         \
        if (!(c)) {                                              \
            std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #c); \
            return false;                                        \
        }                                                        \
    } while (0)

namespace ggml_sycl {

// Test-target-only stable lease. This friend-backed definition deliberately
// does not exist in libggml-sycl and never probes device/cache state.
mem_handle test_make_stable_weight_lease(const ggml_sycl_cache_id & key_id,
                                         int                        device,
                                         void *                     ptr,
                                         ggml_layout_mode           layout,
                                         bool                       on_device,
                                         std::shared_ptr<void>      storage_owner) {
    mem_handle h;
    h.kind_                 = mem_handle_kind::WEIGHT;
    h.device_               = device;
    h.key_                  = { cache_entry_type::DENSE_WEIGHT, key_id, -1, -1 };
    h.gen_                  = cache_generation();
    h.cached_               = { ptr, layout, on_device, false, sycl::event{} };
    h.leased_storage_owner_ = std::move(storage_owner);
    return h;
}

}  // namespace ggml_sycl

static ggml_sycl_cache_id key_for(int id) {
    // Synthetic logical identity: deliberately independent of the backing
    // pointer so the test cannot accidentally bless pointer-derived identity.
    ggml_sycl_cache_id key{};
    key.valid     = true;
    key.model_id  = 0x4d4f4500;
    key.name_hash = static_cast<uint64_t>(id);
    key.aux_id    = static_cast<uint64_t>(id);
    return key;
}

static ggml_sycl::mem_handle weight_handle(void *           ptr,
                                           int              owner,
                                           ggml_layout_mode layout,
                                           int              identity,
                                           bool             on_device) {
    return ggml_sycl::test_make_stable_weight_lease(key_for(identity), owner, ptr, layout, on_device,
                                                    std::make_shared<int>(identity));
}

static ggml_sycl::moe_batch_route route_for(void *                         ptr,
                                            int                            owner,
                                            ggml_sycl::moe_batch_residency residency,
                                            ggml_layout_mode               requested,
                                            ggml_layout_mode               actual,
                                            int                            identity) {
    ggml_sycl::moe_batch_route route;
    route.residency         = residency;
    route.transient_ptr     = ptr;
    route.owning_device     = owner;
    route.requested_layout  = requested;
    route.actual_layout     = actual;
    route.plan_found        = true;
    route.planned_on_device = residency != ggml_sycl::moe_batch_residency::HOST;
    route.planned_device    = route.planned_on_device ? owner : ggml_sycl::mem_handle::HOST_DEVICE;
    route.lease =
        weight_handle(ptr, owner < 0 ? 0 : owner, actual, identity, residency != ggml_sycl::moe_batch_residency::HOST);
    return route;
}

static bool test_host_primary_secondary_mixed_and_occurrences() {
    int           primary = 1, secondary = 2, host = 3;
    const int32_t ids[]  = { 1, 2, 1, 3 };
    int           calls  = 0;
    auto          result = ggml_sycl::build_moe_resolved_batch(ids, 4, 2, 0, [&](int32_t id) {
        ++calls;
        if (id == 1) {
            return route_for(&primary, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA,
                                      GGML_LAYOUT_XMX_TILED, 1);
        }
        if (id == 2) {
            return route_for(&secondary, 1, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_SOA,
                                      GGML_LAYOUT_SOA, 2);
        }
        return route_for(&host, -1, ggml_sycl::moe_batch_residency::HOST, GGML_LAYOUT_SOA, GGML_LAYOUT_AOS, 3);
    });
    CHECK(result);
    CHECK(calls == 4);  // repeated occurrences resolve independently and remain semantic entries
    CHECK(result.batch.expert_ids == std::vector<int32_t>({ 1, 2, 1, 3 }));
    CHECK(result.batch.operands.size() == 4);
    CHECK(result.batch.operands[0].occurrence == 0 && result.batch.operands[2].occurrence == 2);
    CHECK(result.batch.operands[1].token_index == 0 && result.batch.operands[1].slot_index == 1);
    CHECK(result.batch.operands[2].token_index == 1 && result.batch.operands[2].slot_index == 0);
    CHECK(result.batch.operands[3].token_index == 1 && result.batch.operands[3].slot_index == 1);
    CHECK(result.batch.operands[0].requested_layout == GGML_LAYOUT_SOA);
    CHECK(result.batch.operands[0].actual_layout == GGML_LAYOUT_XMX_TILED);
    CHECK(result.batch.operands[1].residency == ggml_sycl::moe_batch_residency::SECONDARY_DEVICE);
    CHECK(result.batch.operands[3].residency == ggml_sycl::moe_batch_residency::HOST);
    CHECK(result.batch.operands[0].lease.stable_identity_equal(result.batch.operands[2].lease));
    return true;
}

static bool test_explicit_planned_alternate_on_submit_device() {
    // The positive path crosses the canonical production normalizer and an
    // actual placement_plan entry; this generic route cannot self-authorize.
    int alternate = 6;
    CHECK(ggml_sycl::test_moe_resolved_batch_accepts_actual_planned_alternate(
        weight_handle(&alternate, 0, GGML_LAYOUT_SOA, 6, true)));

    const int32_t ids[] = { 6 };
    auto          unproved =
        route_for(&alternate, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA, GGML_LAYOUT_SOA, 6);
    unproved.planned_device = 1;
    CHECK(!ggml_sycl::build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return unproved; }));
    return true;
}

static bool test_identity_sharing_and_ready_event() {
    int           shared = 4;
    const int32_t ids[]  = { 4, 5 };
    auto          result = ggml_sycl::build_moe_resolved_batch(ids, 2, 1, 0, [&](int32_t) {
        auto route =
            route_for(&shared, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 9);
        route.has_ready_event = true;
        route.ready_event     = sycl::event{};
        return route;
    });
    CHECK(result);
    CHECK(result.batch.operands[0].expert_id == 4 && result.batch.operands[1].expert_id == 5);
    CHECK(result.batch.operands[0].lease.stable_identity_equal(result.batch.operands[1].lease));
    CHECK(result.batch.operands[0].has_ready_event && result.batch.operands[1].has_ready_event);
    CHECK(result.batch.operands[0].lease.valid() && result.batch.operands[1].lease.valid());
    CHECK(result.batch.operands[0].lease.resolve().ptr == &shared);

    int  same_pointer = 5;
    int  identity     = 11;
    auto distinct     = ggml_sycl::build_moe_resolved_batch(ids, 2, 1, 0, [&](int32_t) {
        return route_for(&same_pointer, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS,
                             GGML_LAYOUT_AOS, identity++);
    });
    CHECK(distinct);
    CHECK(!distinct.batch.operands[0].lease.stable_identity_equal(distinct.batch.operands[1].lease));
    CHECK(distinct.batch.operands[0].lease.resolve().ptr == distinct.batch.operands[1].lease.resolve().ptr);
    return true;
}

static bool test_executor_choice_is_residency_and_capability_driven() {
    int value = 12;
    auto host = route_for(&value, -1, ggml_sycl::moe_batch_residency::HOST, GGML_LAYOUT_AOS,
                          GGML_LAYOUT_AOS, 12);
    auto primary = route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA,
                             GGML_LAYOUT_SOA, 13);
    auto secondary = route_for(&value, 1, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_SOA,
                               GGML_LAYOUT_SOA, 14);
    const int32_t ids[] = { 12 };
    auto build_one = [&](ggml_sycl::moe_batch_route route) {
        return ggml_sycl::build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return route; });
    };

    auto host_batch = build_one(host);
    CHECK(host_batch);
    auto choice = ggml_sycl::choose_moe_batch_executor(host_batch.batch.operands[0], 0,
                                                       /*device_capable=*/false,
                                                       /*owning_queue_available=*/false);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::HOST_CPU);

    auto primary_batch = build_one(primary);
    CHECK(primary_batch);
    choice = ggml_sycl::choose_moe_batch_executor(primary_batch.batch.operands[0], 0, false, true);
    CHECK(!choice && choice.reject == ggml_sycl::moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);
    choice = ggml_sycl::choose_moe_batch_executor(primary_batch.batch.operands[0], 0, true, true);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::PRIMARY_DEVICE);

    auto secondary_batch = build_one(secondary);
    CHECK(secondary_batch);
    choice = ggml_sycl::choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, true, false);
    CHECK(!choice && choice.reject == ggml_sycl::moe_batch_reject_reason::WRONG_QUEUE);
    choice = ggml_sycl::choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, false, true);
    CHECK(!choice && choice.reject == ggml_sycl::moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);
    choice = ggml_sycl::choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, true, true);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::SECONDARY_DEVICE);
    return true;
}

static bool test_fail_closed_contract() {
    int           value = 7;
    const int32_t ids[] = { 7 };
    auto          run   = [&](ggml_sycl::moe_batch_route route) {
        return ggml_sycl::build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return route; });
    };

    auto missing =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    missing.lease = {};
    CHECK(run(missing).reject == ggml_sycl::moe_batch_reject_reason::MISSING_HANDLE);

    auto raw =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    raw.lease = ggml_sycl::mem_handle::from_direct(&value, GGML_LAYOUT_AOS, true, 0);
    CHECK(run(raw).reject == ggml_sycl::moe_batch_reject_reason::RAW_COMPAT_HANDLE);

    auto wrong_device =
        route_for(&value, 1, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    CHECK(run(wrong_device).reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);

    auto wrong_secondary =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    CHECK(run(wrong_secondary).reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);

    auto primary_lease_mismatch =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    primary_lease_mismatch.lease = weight_handle(&value, 1, GGML_LAYOUT_AOS, 7, true);
    CHECK(run(primary_lease_mismatch).reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);

    auto secondary_lease_mismatch =
        route_for(&value, 1, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    secondary_lease_mismatch.lease = weight_handle(&value, 2, GGML_LAYOUT_AOS, 7, true);
    CHECK(run(secondary_lease_mismatch).reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);

    auto wrong_host = route_for(&value, -1, ggml_sycl::moe_batch_residency::HOST, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    wrong_host.owning_device = 0;
    CHECK(run(wrong_host).reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);

    auto wrong_layout =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    wrong_layout.actual_layout = GGML_LAYOUT_SOA;
    CHECK(run(wrong_layout).reject == ggml_sycl::moe_batch_reject_reason::LAYOUT_MISMATCH);

    auto unavailable =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    unavailable.residency = ggml_sycl::moe_batch_residency::UNAVAILABLE;
    CHECK(run(unavailable).reject == ggml_sycl::moe_batch_reject_reason::ROUTE_UNAVAILABLE);

    auto wrong_plan =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    wrong_plan.planned_device = 1;
    CHECK(run(wrong_plan).reject == ggml_sycl::moe_batch_reject_reason::PLAN_MISMATCH);

    auto pointer_mismatch =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 7);
    int other                      = 8;
    pointer_mismatch.transient_ptr = &other;
    CHECK(run(pointer_mismatch).reject == ggml_sycl::moe_batch_reject_reason::POINTER_MISMATCH);

    CHECK(ggml_sycl::build_moe_resolved_batch(nullptr, 1, 1, 0, [&](int32_t) { return missing; }).reject ==
          ggml_sycl::moe_batch_reject_reason::INVALID_REQUEST);
    CHECK(ggml_sycl::build_moe_resolved_batch(ids, 1, 0, 0, [&](int32_t) { return missing; }).reject ==
          ggml_sycl::moe_batch_reject_reason::INVALID_REQUEST);
    return true;
}

static bool test_prompt_local_view_uses_exact_retained_handles() {
    int           first  = 21;
    int           second = 22;
    const int32_t ids[]  = { 3, 3, 4, 3 };
    auto          result = ggml_sycl::build_moe_resolved_batch(ids, 4, 2, 0, [&](int32_t id) {
        return route_for(id == 3 ? &first : &second, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA,
                         GGML_LAYOUT_SOA, id);
    });
    CHECK(result);
    CHECK(result.batch.occurrence(0, 0) == &result.batch.operands[0]);
    CHECK(result.batch.occurrence(1, 1) == &result.batch.operands[3]);
    CHECK(result.batch.occurrence(2, 0) == nullptr);

    auto view = ggml_sycl::make_moe_batch_local_view(result.batch, GGML_LAYOUT_SOA);
    CHECK(view);
    CHECK(view.expert_ids == std::vector<int32_t>({ 3, 4 }));
    CHECK(view.expert_ptrs == std::vector<void *>({ &first, &second }));
    CHECK(view.leases.size() == 2);
    CHECK(view.leases[0].stable_identity_equal(result.batch.operands[0].lease));
    CHECK(view.leases[1].stable_identity_equal(result.batch.operands[2].lease));

    auto wrong_layout = ggml_sycl::make_moe_batch_local_view(result.batch, GGML_LAYOUT_AOS);
    CHECK(!wrong_layout && wrong_layout.reject == ggml_sycl::moe_batch_reject_reason::LAYOUT_MISMATCH);

    int  identity    = 30;
    auto conflicting = ggml_sycl::build_moe_resolved_batch(ids, 4, 2, 0, [&](int32_t id) {
        return route_for(id == 3 ? &first : &second, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA,
                         GGML_LAYOUT_SOA, identity++);
    });
    CHECK(conflicting);
    auto conflict_view = ggml_sycl::make_moe_batch_local_view(conflicting.batch, GGML_LAYOUT_SOA);
    CHECK(!conflict_view && conflict_view.reject == ggml_sycl::moe_batch_reject_reason::POINTER_MISMATCH);
    return true;
}

static bool test_decode_admission_is_route_mode_independent() {
    int           local = 13;
    const int32_t ids[] = { 2, 2 };

    // An all-local placement plan still admits every occurrence into one batch.
    int  planned_calls  = 0;
    auto all_local_plan = ggml_sycl::build_moe_resolved_batch(ids, 2, 2, 0, [&](int32_t) {
        ++planned_calls;
        return route_for(&local, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS,
                         13);
    });
    CHECK(all_local_plan);
    CHECK(planned_calls == 2);
    CHECK(all_local_plan.batch.operands.size() == 2);
    for (const auto & operand : all_local_plan.batch.operands) {
        auto choice = ggml_sycl::choose_moe_batch_executor(operand, 0, true, true);
        CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::PRIMARY_DEVICE);
    }

    // No placement plan/nonhybrid residency follows the same retained contract.
    int  unplanned_calls = 0;
    auto no_plan         = ggml_sycl::build_moe_resolved_batch(ids, 2, 2, 0, [&](int32_t) {
        ++unplanned_calls;
        auto route =
            route_for(&local, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 13);
        route.plan_found = false;
        return route;
    });
    CHECK(no_plan);
    CHECK(unplanned_calls == 2);
    CHECK(no_plan.batch.expert_ids == std::vector<int32_t>({ 2, 2 }));
    return true;
}

int main() {
    if (!test_host_primary_secondary_mixed_and_occurrences() || !test_explicit_planned_alternate_on_submit_device() ||
        !test_identity_sharing_and_ready_event() || !test_executor_choice_is_residency_and_capability_driven() ||
        !test_fail_closed_contract() || !test_prompt_local_view_uses_exact_retained_handles() ||
        !test_decode_admission_is_route_mode_independent()) {
        return 1;
    }
    std::puts("PASS: retained MoE route-batch host contract");
    return 0;
}
