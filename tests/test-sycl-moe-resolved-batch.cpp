// Host-only contract tests for the retained MoE route-batch foundation.
#include "ggml-sycl/cpu-traits-support.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/moe-resolved-batch.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

struct cpu_moe_host_aos_task {
    ggml_sycl::moe_execution_recipe recipe;
    size_t                          admitted_recipe_signature = 0;
    ggml_sycl::mem_handle           weight_lease;
    const float *                   activations = nullptr;
    float *                         output = nullptr;
    void *                          workspace = nullptr;
    size_t                          workspace_bytes = 0;
    ggml_sycl::mem_handle           workspace_lease;
    size_t                          execution_rows = 0;
};

bool ggml_sycl_cpu_moe_host_aos_execute(const cpu_moe_host_aos_task & task,
                                        ggml_sycl::moe_batch_reject_reason * reject = nullptr);

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
    h.cached_                  = { ptr, layout, on_device, false, sycl::event{} };
#ifdef GGML_SYCL_RETENTION_IDENTITY_TESTING
    h.set_canonical_identity_for_test(key_id.aux_id, h.gen_, 4096, 64, 256);
#endif
    h.leased_storage_owner_    = std::move(storage_owner);
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
    route.recipe.valid        = true;
    route.recipe.request      = { ggml_sycl::moe_route_phase::DECODE, 32, 32, 1, GGML_TYPE_Q4_0, 0 };
    route.recipe.layout       = actual;
    route.recipe.owner_device = owner;
    route.recipe.kind =
        residency == ggml_sycl::moe_batch_residency::HOST           ? ggml_sycl::moe_batch_executor::HOST_CPU :
        residency == ggml_sycl::moe_batch_residency::PRIMARY_DEVICE ? ggml_sycl::moe_batch_executor::PRIMARY_DEVICE :
                                                                      ggml_sycl::moe_batch_executor::SECONDARY_DEVICE;
    route.recipe.kernel   = residency == ggml_sycl::moe_batch_residency::HOST ? ggml_sycl::moe_route_kernel::HOST_CPU :
                                                                                ggml_sycl::moe_route_kernel::MMVQ_COMPAT;
    route.recipe.queue    = residency == ggml_sycl::moe_batch_residency::SECONDARY_DEVICE ?
                                ggml_sycl::moe_recipe_queue::OWNER :
                                ggml_sycl::moe_recipe_queue::SUBMIT;
    route.recipe.transfer = residency == ggml_sycl::moe_batch_residency::HOST ?
                                ggml_sycl::moe_recipe_transfer::HOST_ACTIVATION :
                                ggml_sycl::moe_recipe_transfer::NONE;
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

static bool test_canonical_bind_is_zero_based_and_exact() {
    int storage = 0;
    auto handle = weight_handle(&storage, 0, GGML_LAYOUT_AOS, 321, true);
    auto first = ggml_sycl::moe::canonical_allocation_integration::bind(handle, GGML_LAYOUT_AOS + 1, 0);
    CHECK(first && first->identity.occurrence == 0);
    CHECK(first->identity.allocation_id == key_for(321).aux_id && first->identity.generation != 0);
    CHECK(first->identity.byte_offset == 64 && first->identity.byte_size == 256);
    const auto expected = handle.debug_info();
    auto check_copy = [&](const ggml_sycl::mem_handle & copy, uint32_t occurrence) {
        const auto info = copy.debug_info();
        CHECK(info.canonical_allocation_id == expected.canonical_allocation_id);
        CHECK(info.canonical_generation == expected.canonical_generation);
        CHECK(info.canonical_extent == expected.canonical_extent);
        CHECK(info.offset == expected.offset && info.size == expected.size);
        auto binding = ggml_sycl::moe::canonical_allocation_integration::bind(
            copy, GGML_LAYOUT_AOS + 1, occurrence);
        CHECK(binding && binding->identity.allocation_id == expected.canonical_allocation_id);
        CHECK(binding->identity.generation == expected.canonical_generation);
        CHECK(binding->owner.extent() == expected.canonical_extent);
        CHECK(binding->identity.byte_offset == expected.offset && binding->identity.byte_size == expected.size);
        CHECK(binding->identity.occurrence == occurrence);
        return true;
    };

    ggml_sycl::mem_handle copy_constructed(handle);
    CHECK(check_copy(copy_constructed, 1));
    ggml_sycl::mem_handle copy_assigned;
    copy_assigned = handle;
    CHECK(check_copy(copy_assigned, 2));
    ggml_sycl::mem_handle nested_copy(copy_assigned);
    CHECK(check_copy(nested_copy, 3));
    ggml_sycl::mem_handle move_constructed(std::move(copy_constructed));
    CHECK(check_copy(move_constructed, 4));
    CHECK(!copy_constructed.valid() && !copy_constructed.has_stable_owner_identity());
    ggml_sycl::mem_handle move_assigned;
    move_assigned = std::move(copy_assigned);
    CHECK(check_copy(move_assigned, 5));
    CHECK(!copy_assigned.valid() && !copy_assigned.has_stable_owner_identity());
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
    int  value = 12;
    auto host  = route_for(&value, -1, ggml_sycl::moe_batch_residency::HOST, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 12);
    auto primary =
        route_for(&value, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA, GGML_LAYOUT_SOA, 13);
    auto secondary =
        route_for(&value, 1, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_SOA, GGML_LAYOUT_SOA, 14);
    const int32_t ids[]     = { 12 };
    auto          build_one = [&](ggml_sycl::moe_batch_route route) {
        return ggml_sycl::build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return route; });
    };

    auto host_batch = build_one(host);
    CHECK(host_batch);
    auto choice = ggml_sycl::choose_moe_batch_executor(host_batch.batch.operands[0], 0,
                                                       /*owning_queue_available=*/false, 0);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::HOST_CPU);

    auto primary_batch = build_one(primary);
    CHECK(primary_batch);
    choice = ggml_sycl::choose_moe_batch_executor(primary_batch.batch.operands[0], 0, true, 0);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::PRIMARY_DEVICE);

    auto secondary_batch = build_one(secondary);
    CHECK(secondary_batch);
    choice = ggml_sycl::choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, false, 0);
    CHECK(!choice && choice.reject == ggml_sycl::moe_batch_reject_reason::WRONG_QUEUE);
    choice = ggml_sycl::choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, true, 0);
    CHECK(choice && choice.executor == ggml_sycl::moe_batch_executor::SECONDARY_DEVICE);
    return true;
}

static bool test_owned_direct_slice_route_acceptance() {
    std::vector<sycl::device> devices;
    try { devices = sycl::device::get_devices(); } catch (...) { return true; }
    if (devices.empty()) return true;
    sycl::queue q(devices.front());
    ggml_sycl::alloc_request req{};
    req.queue = &q;
    req.device = 0;
    req.size = 512;
    req.intent.role = ggml_sycl::alloc_role::WEIGHT;
    req.intent.constraints.must_device = true;
    ggml_sycl::alloc_handle allocation{};
    CHECK(ggml_sycl::unified_alloc(req, &allocation));
    auto owner = ggml_sycl::mem_handle::from_owned_alloc(std::move(allocation), GGML_LAYOUT_AOS);

    // Every special member must preserve the allocator-minted capability as one
    // tuple with the owner and range.  This is deliberately a real unified_alloc
    // handle: synthetic pointer/key identity would not exercise direct binding.
    const auto expected = owner.debug_info();
    CHECK(expected.canonical_allocation_id != 0 && expected.canonical_generation != 0 &&
          expected.canonical_extent == req.size && expected.offset == 0 && expected.size == req.size);
    auto check_identity = [&](const ggml_sycl::mem_handle & handle, size_t offset, size_t size) {
        const auto info = handle.debug_info();
        CHECK(info.canonical_allocation_id == expected.canonical_allocation_id);
        CHECK(info.canonical_generation == expected.canonical_generation);
        CHECK(info.canonical_extent == expected.canonical_extent);
        CHECK(info.offset == offset && info.size == size && handle.has_stable_owner_identity());
        auto binding = ggml_sycl::moe::canonical_allocation_integration::bind(handle, 0xabc, 7);
        CHECK(binding && binding->identity.allocation_id == expected.canonical_allocation_id);
        CHECK(binding->identity.generation == expected.canonical_generation);
        CHECK(binding->owner.extent() == expected.canonical_extent);
        CHECK(binding->identity.byte_offset == offset && binding->identity.byte_size == size);
        CHECK(binding->identity.occurrence == 7);
        return true;
    };

    ggml_sycl::mem_handle copy_constructed(owner);
    CHECK(check_identity(copy_constructed, 0, 512));
    ggml_sycl::mem_handle copy_assigned;
    copy_assigned = owner;
    CHECK(check_identity(copy_assigned, 0, 512));
    ggml_sycl::mem_handle nested_copy(copy_assigned);
    CHECK(check_identity(nested_copy, 0, 512));

    ggml_sycl::mem_handle move_source(owner);
    ggml_sycl::mem_handle move_constructed(std::move(move_source));
    CHECK(check_identity(move_constructed, 0, 512));
    CHECK(!move_source.valid() && !move_source.has_stable_owner_identity());
    ggml_sycl::mem_handle move_assign_source(owner);
    ggml_sycl::mem_handle move_assigned = ggml_sycl::mem_handle::from_direct(&req, GGML_LAYOUT_AOS, false);
    move_assigned = std::move(move_assign_source);
    CHECK(check_identity(move_assigned, 0, 512));
    CHECK(!move_assign_source.valid() && !move_assign_source.has_stable_owner_identity());

    auto slice = nested_copy.slice(128, 256);
    auto nested_slice = slice.slice(32, 64);
    CHECK(check_identity(slice, 128, 256));
    CHECK(check_identity(nested_slice, 160, 64));
    ggml_sycl::mem_handle slice_copy = nested_slice;
    CHECK(check_identity(slice_copy, 160, 64));

    const auto resolved = slice.resolve(0);
    CHECK(slice.kind() == ggml_sycl::mem_handle_kind::DIRECT);
    CHECK(slice.has_stable_owner_identity() && resolved.ptr && resolved.on_device);
    auto route = route_for(resolved.ptr, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE,
                           GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 1);
    route.lease = slice;
    const int32_t ids[] = { 1 };
    auto accepted = ggml_sycl::build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return route; });
    CHECK(accepted);
    CHECK(accepted.batch.operands[0].lease.stable_identity_equal(slice));

    int raw = 0;
    auto ownerless = ggml_sycl::mem_handle::from_direct(&raw, GGML_LAYOUT_AOS, false);
    CHECK(!ownerless.has_stable_owner_identity());
    CHECK(!ggml_sycl::moe::canonical_allocation_integration::bind(ownerless, 0xabc, 0));
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

    // A stale same-size external/cache array cannot overwrite the admitted ID snapshot.
    int32_t mutable_ids[] = { 6, 7 };
    auto    snapshot      = ggml_sycl::build_moe_resolved_batch(mutable_ids, 2, 2, 0, [&](int32_t id) {
        return route_for(id == 6 ? &first : &second, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_SOA,
                         GGML_LAYOUT_SOA, id);
    });
    CHECK(snapshot);
    mutable_ids[0] = 7;
    mutable_ids[1] = 6;
    CHECK(snapshot.batch.expert_ids == std::vector<int32_t>({ 6, 7 }));
    CHECK(snapshot.batch.occurrence(0, 0)->expert_id == 6);
    CHECK(snapshot.batch.occurrence(0, 1)->expert_id == 7);
    return true;
}

static bool test_planned_prompt_hybrid_identity_readiness_and_layout_miss() {
    int           primary = 31, secondary = 32, host = 33;
    const int32_t ids[] = { 1, 2, 3 };
    auto          batch = ggml_sycl::build_moe_resolved_batch(ids, 3, 3, 0, [&](int32_t id) {
        ggml_sycl::moe_batch_route route;
        if (id == 1) {
            route = route_for(&primary, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS,
                                       GGML_LAYOUT_AOS, 31);
        } else if (id == 2) {
            route = route_for(&secondary, 1, ggml_sycl::moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_AOS,
                                       GGML_LAYOUT_AOS, 32);
        } else {
            route = route_for(&host, -1, ggml_sycl::moe_batch_residency::HOST, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 33);
        }
        route.has_ready_event = true;
        route.ready_event     = sycl::event{};
        return route;
    });
    CHECK(batch);
    CHECK(batch.batch.operands.size() == 3);
    for (const auto & operand : batch.batch.operands) {
        CHECK(operand.has_ready_event);
    }
    // The all-local fast path rejects this mixed batch. The hybrid fallback must
    // still partition every exact occurrence instead of publishing an empty dst.
    auto fast_reject = ggml_sycl::make_moe_batch_local_view(batch.batch, GGML_LAYOUT_AOS);
    CHECK(!fast_reject && fast_reject.reject == ggml_sycl::moe_batch_reject_reason::WRONG_DEVICE);
    size_t primary_count = 0, secondary_count = 0, host_count = 0;
    for (size_t slot = 0; slot < 3; ++slot) {
        const auto * operand = batch.batch.occurrence(0, slot);
        CHECK(operand);
        const auto choice = ggml_sycl::choose_moe_batch_executor(*operand, 0, true, 0);
        CHECK(choice);
        primary_count += choice.executor == ggml_sycl::moe_batch_executor::PRIMARY_DEVICE;
        secondary_count += choice.executor == ggml_sycl::moe_batch_executor::SECONDARY_DEVICE;
        host_count += choice.executor == ggml_sycl::moe_batch_executor::HOST_CPU;
    }
    CHECK(primary_count == 1 && secondary_count == 1 && host_count == 1);

    // A selected local fast path must fail preflight when the admitted layout differs.
    auto layout_miss = ggml_sycl::make_moe_batch_local_view(batch.batch, GGML_LAYOUT_SOA);
    CHECK(!layout_miss && layout_miss.reject == ggml_sycl::moe_batch_reject_reason::LAYOUT_MISMATCH);

    // Same expert ID with a changed stable identity is not groupable.
    int32_t repeated_ids[] = { 5, 5 };
    int     identity       = 50;
    auto    drift          = ggml_sycl::build_moe_resolved_batch(repeated_ids, 2, 2, 0, [&](int32_t) {
        return route_for(&primary, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS,
                                     identity++);
    });
    CHECK(drift);
    auto drift_view = ggml_sycl::make_moe_batch_local_view(drift.batch, GGML_LAYOUT_AOS);
    CHECK(!drift_view && drift_view.reject == ggml_sycl::moe_batch_reject_reason::POINTER_MISMATCH);
    return true;
}

static bool test_retained_role_alignment_and_terminal_transaction() {
    int           gate_a = 61, gate_b = 62, up_a = 71, up_b = 72, down_a = 81, down_b = 82;
    const int32_t ids[] = { 4, 4, 9, 4 };
    auto          build = [&](ggml_sycl::moe_batch_role role) {
        const ggml_layout_mode layout = role == ggml_sycl::moe_batch_role::GATE ? GGML_LAYOUT_SOA :
                                                 role == ggml_sycl::moe_batch_role::UP   ? GGML_LAYOUT_XMX_TILED :
                                                                                           GGML_LAYOUT_AOS;
        return ggml_sycl::build_moe_resolved_batch(ids, 4, 2, 0, [&](int32_t id) {
            int * ptr      = nullptr;
            int   identity = 0;
            if (role == ggml_sycl::moe_batch_role::GATE) {
                ptr      = id == 4 ? &gate_a : &gate_b;
                identity = id == 4 ? 61 : 62;
            } else if (role == ggml_sycl::moe_batch_role::UP) {
                ptr      = id == 4 ? &up_a : &up_b;
                identity = id == 4 ? 71 : 72;
            } else {
                ptr      = id == 4 ? &down_a : &down_b;
                identity = id == 4 ? 81 : 82;
            }
            return route_for(ptr, 0, ggml_sycl::moe_batch_residency::PRIMARY_DEVICE, layout, layout, identity);
        });
    };
    auto gate = build(ggml_sycl::moe_batch_role::GATE);
    auto up   = build(ggml_sycl::moe_batch_role::UP);
    auto down = build(ggml_sycl::moe_batch_role::DOWN);
    CHECK(gate && up && down);

    ggml_sycl::moe_retained_role_bundle_result absent_pair;
    CHECK(!absent_pair && absent_pair.reject == ggml_sycl::moe_batch_reject_reason::MISSING_ROLE);
    auto missing_gate = ggml_sycl::align_moe_retained_role_batches(
        { ggml_sycl::moe_batch_role::GATE, nullptr, {} }, { ggml_sycl::moe_batch_role::UP, nullptr, up.batch },
        { ggml_sycl::moe_batch_role::DOWN, nullptr, down.batch });
    auto missing_up = ggml_sycl::align_moe_retained_role_batches(
        { ggml_sycl::moe_batch_role::GATE, nullptr, gate.batch }, { ggml_sycl::moe_batch_role::UP, nullptr, {} },
        { ggml_sycl::moe_batch_role::DOWN, nullptr, down.batch });
    auto missing_down = ggml_sycl::align_moe_retained_role_batches(
        { ggml_sycl::moe_batch_role::GATE, nullptr, gate.batch }, { ggml_sycl::moe_batch_role::UP, nullptr, up.batch },
        { ggml_sycl::moe_batch_role::DOWN, nullptr, {} });
    CHECK(!missing_gate && missing_gate.role == ggml_sycl::moe_batch_role::GATE);
    CHECK(!missing_up && missing_up.role == ggml_sycl::moe_batch_role::UP);
    CHECK(!missing_down && missing_down.role == ggml_sycl::moe_batch_role::DOWN);

    auto aligned = ggml_sycl::align_moe_retained_role_batches(
        { ggml_sycl::moe_batch_role::GATE, reinterpret_cast<const ggml_tensor *>(1), gate.batch },
        { ggml_sycl::moe_batch_role::UP, reinterpret_cast<const ggml_tensor *>(2), up.batch },
        { ggml_sycl::moe_batch_role::DOWN, reinterpret_cast<const ggml_tensor *>(3), down.batch });
    CHECK(aligned);
    CHECK(aligned.bundle.retained_lease_count() == 12);
    CHECK(aligned.bundle.gate.weight_identity != aligned.bundle.up.weight_identity);
    CHECK(aligned.bundle.gate.batch.operands[0].actual_layout == GGML_LAYOUT_SOA);
    CHECK(aligned.bundle.up.batch.operands[0].actual_layout == GGML_LAYOUT_XMX_TILED);
    CHECK(aligned.bundle.down.batch.operands[0].actual_layout == GGML_LAYOUT_AOS);

    // Duplicate occurrences are valid, but any role-local ID/token/slot drift is not.
    auto drift_up                   = up.batch;
    drift_up.operands[2].slot_index = 1;
    auto drift =
        ggml_sycl::align_moe_retained_role_batches({ ggml_sycl::moe_batch_role::GATE, nullptr, gate.batch },
                                                   { ggml_sycl::moe_batch_role::UP, nullptr, std::move(drift_up) },
                                                   { ggml_sycl::moe_batch_role::DOWN, nullptr, down.batch });
    CHECK(!drift && drift.reject == ggml_sycl::moe_batch_reject_reason::ROLE_ALIGNMENT_MISMATCH);
    CHECK(drift.role == ggml_sycl::moe_batch_role::UP && drift.occurrence == 2);

    ggml_sycl::moe_retained_pointer_table down_table;
    down_table.table_handle = weight_handle(&down_a, 0, GGML_LAYOUT_AOS, 180, true);
    for (const auto & operand : down.batch.operands) {
        down_table.role_leases.push_back(operand.lease);
    }
    down_table.has_ready_event = true;
    CHECK(down_table.valid() && down_table.role_leases.size() == 4);

    ggml_sycl::moe_retained_terminal_bundle terminal;
    terminal.roles = aligned.bundle;
    terminal.tables.push_back(std::move(down_table));
    terminal.intermediates.push_back(gate.batch.operands.front().lease);
    terminal.terminal_submitted = true;
    CHECK(terminal.retained_handle_count() == 18);
    return true;
}

static bool test_recipe_matrix_workspace_and_immutability() {
    using namespace ggml_sycl;
    for (ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
        for (moe_route_phase phase : { moe_route_phase::DECODE, moe_route_phase::PROMPT }) {
            const size_t         rows = phase == moe_route_phase::DECODE ? 1 : 7;
            moe_route_request    request{ phase, 256, 96, rows, type, 0 };
            moe_workspace_recipe ws;
            CHECK(plan_moe_host_workspace(request, 272, 4 * sizeof(size_t), &ws));
            CHECK(ws.activation_f32_bytes == rows * 256 * sizeof(float));
            CHECK(ws.activation_q8_bytes == rows * 272);
            CHECK(ws.output_f32_bytes == rows * 96 * sizeof(float));
            CHECK(ws.descriptor_bytes == rows * 4 * sizeof(size_t));
            CHECK(ws.total_bytes >=
                  ws.activation_f32_bytes + ws.activation_q8_bytes + ws.output_f32_bytes + ws.descriptor_bytes);

            int  value = 90;
            auto host  = route_for(&value, -1, moe_batch_residency::HOST, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 90);
            host.recipe.request    = request;
            host.recipe.workspace  = ws;
            const int32_t ids[]    = { 9 };
            auto          admitted = build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return host; });
            CHECK(admitted);
            CHECK(choose_moe_batch_executor(admitted.batch.operands[0], 0, false, ws.total_bytes));
            auto mutated = admitted.batch.operands[0];
            mutated.recipe.request.rows++;
            auto choice = choose_moe_batch_executor(mutated, 0, false, ws.total_bytes);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::RECIPE_MISMATCH);
            choice = choose_moe_batch_executor(admitted.batch.operands[0], 0, false, ws.total_bytes - 1);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::WORKSPACE_UNDERSIZED);

            auto device =
                route_for(&value, 0, moe_batch_residency::PRIMARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 91);
            device.recipe.request = request;
            device.recipe.kernel = moe_route_kernel::DEVICE_MMVQ_Q1_NVFP4_AOS;
            CHECK(plan_moe_q1_nvfp4_device_workspace(request, 272, 4, &device.recipe.workspace));
            auto device_batch = build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return device; });
            CHECK(device_batch);
            choice = choose_moe_batch_executor(device_batch.batch.operands[0], 0, true,
                                               device.recipe.workspace.total_bytes);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);
            moe_admitted_workspace_bundle ordinary_bundle;
            choice = choose_moe_batch_executor(device_batch.batch.operands[0], 0, true,
                                               device.recipe.workspace.total_bytes, &ordinary_bundle);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);

            auto secondary =
                route_for(&value, 1, moe_batch_residency::SECONDARY_DEVICE, GGML_LAYOUT_AOS, GGML_LAYOUT_AOS, 92);
            secondary.recipe.request = request;
            secondary.recipe.kind = moe_batch_executor::SECONDARY_DEVICE;
            secondary.recipe.kernel = moe_route_kernel::DEVICE_MMVQ_Q1_NVFP4_AOS;
            secondary.recipe.queue = moe_recipe_queue::OWNER;
            secondary.recipe.transfer = moe_recipe_transfer::HOST_BOUNCE;
            secondary.recipe.owner_device = 1;
            secondary.recipe.workspace = device.recipe.workspace;
            auto secondary_batch = build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return secondary; });
            CHECK(secondary_batch);
            choice = choose_moe_batch_executor(secondary_batch.batch.operands[0], 0, true,
                                               secondary.recipe.workspace.total_bytes);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);

            auto soa = device_batch.batch.operands[0];
            soa.recipe.layout = GGML_LAYOUT_SOA;
            soa.admitted_recipe_signature = moe_execution_recipe_signature(soa.recipe);
            choice = choose_moe_batch_executor(soa, 0, true, soa.recipe.workspace.total_bytes);
            CHECK(!choice && choice.reject == moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);
        }
    }
    moe_workspace_recipe overflow;
    CHECK(!plan_moe_host_workspace({ moe_route_phase::PROMPT, INT64_MAX, INT64_MAX, SIZE_MAX, GGML_TYPE_Q1_0, 0 },
                                   SIZE_MAX, SIZE_MAX, &overflow));
    return true;
}

static bool test_numerical_q1_nvfp4_host_executor() {
    using namespace ggml_sycl;
    int identity = 200;
    for (ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
        const int64_t K = type == GGML_TYPE_Q1_0 ? 128 : 64;
        const int64_t N = 2;
        const auto * traits = ggml_sycl_get_type_traits_cpu(type);
        CHECK(traits && traits->vec_dot);
        for (moe_route_phase phase : { moe_route_phase::DECODE, moe_route_phase::PROMPT }) {
            const size_t rows = phase == moe_route_phase::DECODE ? 1 : 3;
            std::vector<uint8_t> weights(ggml_row_size(type, K) * N, 0);
            std::vector<float> activations(rows * K);
            for (size_t i = 0; i < activations.size(); ++i) activations[i] = float(int(i % 13) - 6) / 7.0f;
            std::vector<float> output(rows * N, 99.0f);
            auto route = route_for(weights.data(), -1, moe_batch_residency::HOST, GGML_LAYOUT_AOS,
                                   GGML_LAYOUT_AOS, identity++);
            route.recipe.request = { phase, K, N, rows, type, 0 };
            CHECK(plan_moe_host_workspace(route.recipe.request, ggml_row_size(traits->vec_dot_type, K),
                                          sizeof(size_t) * 4, &route.recipe.workspace));
            const int32_t ids[] = { 1 };
            auto admitted = build_moe_resolved_batch(ids, 1, 1, 0, [&](int32_t) { return route; });
            CHECK(admitted);
            const size_t bytes = route.recipe.workspace.total_bytes;
            std::vector<uint8_t> storage(bytes + 128);
            void * raw = storage.data();
            size_t available = storage.size();
            void * aligned = std::align(route.recipe.workspace.alignment, bytes, raw, available);
            CHECK(aligned);
            cpu_moe_host_aos_task task;
            task.recipe = admitted.batch.operands[0].recipe;
            task.admitted_recipe_signature = admitted.batch.operands[0].admitted_recipe_signature;
            task.weight_lease = admitted.batch.operands[0].lease;
            task.activations = activations.data();
            task.output = output.data();
            task.workspace = aligned;
            task.workspace_bytes = bytes;
            task.workspace_lease = weight_handle(aligned, 0, GGML_LAYOUT_AOS, identity++, false);
            moe_batch_reject_reason reject;
            CHECK(ggml_sycl_cpu_moe_host_aos_execute(task, &reject));
            for (float value : output) CHECK(std::isfinite(value) && value == 0.0f);
            task.workspace_bytes = bytes - 1;
            CHECK(!ggml_sycl_cpu_moe_host_aos_execute(task, &reject) &&
                  reject == moe_batch_reject_reason::WORKSPACE_UNDERSIZED);
            task.workspace_bytes = bytes;
            task.workspace = static_cast<uint8_t *>(aligned) + 1;
            task.workspace_lease = weight_handle(task.workspace, 0, GGML_LAYOUT_AOS, identity++, false);
            CHECK(!ggml_sycl_cpu_moe_host_aos_execute(task, &reject));

#if !defined(_WIN32)
            if (phase == moe_route_phase::PROMPT) {
                const long page_size = sysconf(_SC_PAGESIZE);
                CHECK(page_size > 0 && static_cast<size_t>(page_size) >= static_cast<size_t>(K) * sizeof(float));
                void * mapping = mmap(nullptr, static_cast<size_t>(page_size) * 2, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                CHECK(mapping != MAP_FAILED);
                CHECK(mprotect(static_cast<uint8_t *>(mapping) + page_size, static_cast<size_t>(page_size),
                               PROT_NONE) == 0);
                float * one_row = reinterpret_cast<float *>(static_cast<uint8_t *>(mapping) + page_size -
                                                            static_cast<size_t>(K) * sizeof(float));
                for (int64_t k = 0; k < K; ++k) {
                    one_row[k] = float(k % 11) / 5.0f;
                }
                std::vector<float> one_output(N, 7.0f);
                task.activations     = one_row;
                task.output          = one_output.data();
                task.execution_rows  = 1;
                task.workspace       = aligned;
                task.workspace_bytes = bytes;
                task.workspace_lease = weight_handle(aligned, 0, GGML_LAYOUT_AOS, identity++, false);
                CHECK(ggml_sycl_cpu_moe_host_aos_execute(task, &reject));
                for (float value : one_output) {
                    CHECK(std::isfinite(value) && value == 0.0f);
                }
                CHECK(munmap(mapping, static_cast<size_t>(page_size) * 2) == 0);
            }
#endif
        }
    }
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
        auto choice = ggml_sycl::choose_moe_batch_executor(operand, 0, true, 0);
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
        !test_canonical_bind_is_zero_based_and_exact() || !test_identity_sharing_and_ready_event() || !test_executor_choice_is_residency_and_capability_driven() ||
        !test_owned_direct_slice_route_acceptance() || !test_fail_closed_contract() ||
        !test_prompt_local_view_uses_exact_retained_handles() ||
        !test_planned_prompt_hybrid_identity_readiness_and_layout_miss() ||
        !test_retained_role_alignment_and_terminal_transaction() || !test_recipe_matrix_workspace_and_immutability() ||
        !test_numerical_q1_nvfp4_host_executor() || !test_decode_admission_is_route_mode_independent()) {
        return 1;
    }
    std::puts("PASS: retained MoE route-batch host contract");
    return 0;
}
