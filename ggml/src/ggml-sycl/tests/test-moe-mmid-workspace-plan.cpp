#include "moe-mmid-workspace.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl;

#if defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define MMID_TSAN_BUILD 1
#    endif
#endif
#if defined(__SANITIZE_THREAD__)
#    define MMID_TSAN_BUILD 1
#endif

#ifndef MMID_TSAN_BUILD
static std::atomic<size_t> g_heap_allocations{ 0 };
static std::atomic<bool>   g_fail_heap_allocations{ false };

void * operator new(std::size_t bytes) {
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
    if (g_fail_heap_allocations.load(std::memory_order_relaxed)) {
        throw std::bad_alloc();
    }
    if (void * ptr = std::malloc(bytes)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void * ptr) noexcept {
    std::free(ptr);
}

void operator delete(void * ptr, std::size_t) noexcept {
    std::free(ptr);
}
#endif

static void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static void geometry_exact_and_boundary() {
    moe_mmid_workspace_geometry g;
    check(moe_mmid_plan_workspace({ 64, 96, 1, 2, 3 }, true, &g), "valid shape rejected");
    check(g.activation_rows == 3 && g.occurrences == 6, "A/O formula mismatch");
    check(g.q8_ne10_row_bytes == 72 && g.q8_ne01_row_bytes == 108, "exact Q8_1 row mismatch");
    check(g.activation_f32_bytes == 768 && g.activation_q8_bytes == 216, "activation bytes mismatch");
    check(g.output_f32_bytes == 2304 && g.output_q8_bytes == 648, "output bytes mismatch");
    check(g.device_slot_bytes == 4096, "256-byte device slice accounting mismatch");
    check(g.activation_f32_offset % 256 == 0 && g.activation_q8_offset % 256 == 0 && g.output_f32_offset % 256 == 0 &&
              g.output_q8_offset % 256 == 0,
          "device slice start lost 256-byte alignment");
    check(g.activation_q8_offset == 768 && g.output_f32_offset == 1024 && g.output_q8_offset == 3328,
          "device slice-start mutant survived");
    check(g.descriptor_host_bytes == 96, "16-byte descriptor accounting mismatch");
    check(g.secondary_activation_d2h_bytes == 768 && g.secondary_activation_h2d_bytes == 768 &&
              g.secondary_output_d2h_bytes == 2304 && g.secondary_output_h2d_bytes == 2304,
          "four secondary bounce slices mismatch");
    check(g.secondary_bounce_bytes == 6144 && g.host_slot_bytes == 6240, "host total mismatch");

    size_t device = 0, host = 0;
    check(moe_mmid_checked_pool_bytes(g, MOE_MMID_WORKSPACE_DEPTH, &device, &host), "exact pool rejected");
    check(device == 8192 && host == 12480, "pool multiplication mismatch");
    size_t remaining = device;
    check(moe_mmid_debit_device_budget(device, &remaining) && remaining == 0,
          "exact per-device workspace debit failed");
    remaining = device - 1;
    check(!moe_mmid_debit_device_budget(device, &remaining) && remaining == device - 1,
          "T-1 per-device budget was mutated or admitted");
    check(!moe_mmid_checked_pool_bytes(g, MOE_MMID_WORKSPACE_DEPTH - 1, &device, &host), "T-1 depth silently clamped");
}

static void exact_256_slice_start_after_128_bytes() {
    moe_mmid_workspace_geometry geometry;
    check(moe_mmid_plan_workspace({ 32, 32, 1, 1, 1 }, false, &geometry), "128-byte geometry rejected");
    check(geometry.activation_f32_bytes == 128, "geometry did not produce 128-byte first slice");
    check(geometry.activation_q8_offset == 256, "128-byte end was not advanced to exact 256-byte slice start");
    check(geometry.activation_q8_offset != 128, "128-byte alignment mutant survived");
}

static void broadcast_and_occurrences() {
    moe_mmid_workspace_geometry broadcast, expanded;
    check(moe_mmid_plan_workspace({ 64, 96, 1, 2, 3 }, false, &broadcast), "ne11=1 rejected");
    check(moe_mmid_plan_workspace({ 64, 96, 2, 2, 3 }, false, &expanded), "ne11=top_k rejected");
    check(expanded.activation_rows == 6 && expanded.occurrences == 6, "expanded row formulas mismatch");
    check(expanded.device_slot_bytes == 5120, "expanded device geometry mismatch");
    check(expanded.descriptor_host_bytes == broadcast.descriptor_host_bytes,
          "occurrence descriptors must not depend on activation broadcast");
    check(expanded.secondary_bounce_bytes == 0, "primary owner got secondary bounces");
    check(!moe_mmid_plan_workspace({ 64, 96, 3, 2, 3 }, false, &expanded), "invalid ne11 admitted");
}

static void maxima_and_overflow() {
    size_t capacity = 0;
    check(moe_mmid_capacity(4096, 512, 8, &capacity) && capacity == 512,
          "n_seq_max incorrectly multiplied workspace C");
    check(moe_mmid_capacity(128, 512, 8, &capacity) && capacity == 128, "nonzero n_ctx cap ignored");

    moe_mmid_workspace_geometry a, b, maximum;
    check(moe_mmid_plan_workspace({ 64, 32, 2, 2, 5 }, true, &a), "shape a rejected");
    check(moe_mmid_plan_workspace({ 32, 96, 1, 2, 5 }, true, &b), "shape b rejected");
    check(moe_mmid_component_max(&maximum, a) && moe_mmid_component_max(&maximum, b), "max failed");
    check(maximum.activation_f32_bytes == a.activation_f32_bytes, "activation component not maxed");
    check(maximum.output_f32_bytes == b.output_f32_bytes, "output component not maxed");
    check(maximum.device_slot_bytes >= a.device_slot_bytes && maximum.device_slot_bytes >= b.device_slot_bytes,
          "mixed K/N component maxima total under-sized");
    const size_t expected_bounce = 2 * std::max(a.activation_f32_bytes, b.activation_f32_bytes) +
                                   2 * std::max(a.output_f32_bytes, b.output_f32_bytes);
    check(maximum.secondary_bounce_bytes == expected_bounce &&
              maximum.host_slot_bytes == maximum.descriptor_host_bytes + expected_bounce,
          "four independently merged bounce maxima were not recomputed");

    moe_mmid_workspace_geometry unchanged          = maximum;
    moe_mmid_workspace_geometry overflow_candidate = b;
    overflow_candidate.secondary_output_h2d_bytes  = std::numeric_limits<size_t>::max();
    check(!moe_mmid_component_max(&maximum, overflow_candidate), "merged bounce overflow admitted");
    check(maximum.secondary_bounce_bytes == unchanged.secondary_bounce_bytes &&
              maximum.host_slot_bytes == unchanged.host_slot_bytes,
          "failed merge partially mutated aggregate");

    size_t row = 7;
    check(!moe_mmid_q8_1_row_bytes(31, &row) && row == 7, "partial Q8_1 row accepted or output changed");
    check(!moe_mmid_plan_workspace({ 64, 96, 1, 2, std::numeric_limits<size_t>::max() }, true, &a),
          "shape multiplication overflow admitted");
    moe_mmid_workspace_geometry huge;
    huge.valid             = true;
    huge.device_slot_bytes = std::numeric_limits<size_t>::max();
    huge.host_slot_bytes   = 1;
    size_t d = 11, h = 12;
    check(!moe_mmid_checked_pool_bytes(huge, MOE_MMID_WORKSPACE_DEPTH, &d, &h) && d == 11 && h == 12,
          "pool overflow did not fail atomically");
    size_t zone = 19;
    check(!moe_mmid_checked_zone_total(std::numeric_limits<size_t>::max(), 1, &zone) && zone == 19,
          "host-zone addition overflow did not fail atomically");
    check(!moe_mmid_checked_product(std::numeric_limits<size_t>::max(), 2, &zone) && zone == 19,
          "host-zone product overflow did not fail atomically");
}

static void pool_identity_generation_and_terminal_release() {
    moe_mmid_workspace_pool pool;
    auto                    first  = pool.acquire(10, 100);
    auto                    second = pool.acquire(20, 200);
    check(first.status == moe_mmid_lease_status::ACQUIRED && second.status == moe_mmid_lease_status::ACQUIRED,
          "fixed slots unavailable");
    check(pool.acquire(30, 300).status == moe_mmid_lease_status::BUSY, "depth exhaustion not BUSY");
    check(pool.terminal_release(first.lease, 11, 100) == moe_mmid_release_status::WRONG_QUEUE,
          "wrong queue released lease");
    check(pool.terminal_release(first.lease, 10, 101) == moe_mmid_release_status::WRONG_EPOCH,
          "wrong epoch released lease");
    check(pool.terminal_release(first.lease, 10, 100) == moe_mmid_release_status::RELEASED, "terminal release failed");
    auto reused = pool.acquire(10, 101);
    check(reused.status == moe_mmid_lease_status::ACQUIRED && reused.lease.slot() == first.lease.slot() &&
              reused.lease.generation() == first.lease.generation() + 1,
          "slot generation/reuse mismatch");
    check(pool.terminal_release(first.lease, 10, 100) == moe_mmid_release_status::STALE,
          "stale generation released reused slot");
    check(pool.terminal_release(reused.lease, 10, 101) == moe_mmid_release_status::RELEASED, "reuse release failed");
    check(pool.terminal_release(second.lease, 20, 200) == moe_mmid_release_status::RELEASED, "second release failed");

    moe_mmid_workspace_pool other;
    check(other.terminal_release(reused.lease, 10, 101) == moe_mmid_release_status::INVALID,
          "cross-pool lease identity accepted");
    moe_mmid_workspace_pool bad_depth(1);
    check(bad_depth.depth() == 0 && bad_depth.acquire(1, 1).status == moe_mmid_lease_status::INVALID,
          "non-fixed depth did not fail closed");

    moe_mmid_workspace_pool generation_edge;
    check(generation_edge.set_generation_for_test(0, std::numeric_limits<uint64_t>::max()),
          "generation edge setup failed");
    auto later_slot = generation_edge.acquire(77, 88);
    check(later_slot.status == moe_mmid_lease_status::ACQUIRED && later_slot.lease.slot() == 1,
          "generation-max slot prevented scanning a later reusable slot");
    check(generation_edge.terminal_release(later_slot.lease, 77, 88) == moe_mmid_release_status::RELEASED,
          "later-slot terminal release failed");
}

struct fake_allocation {
    explicit fake_allocation(size_t bytes, std::atomic<int> * destroyed) : storage(bytes + 255), destroyed(destroyed) {}

    ~fake_allocation() {
        if (destroyed) {
            destroyed->fetch_add(1);
        }
    }

    std::vector<unsigned char> storage;
    std::atomic<int> *         destroyed;
};

static moe_mmid_blob fake_blob(bool host, int device, size_t bytes, std::atomic<int> * destroyed) {
    auto          allocation = std::make_shared<fake_allocation>(bytes, destroyed);
    moe_mmid_blob blob;
    blob.owner          = allocation;
    const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation->storage.data());
    blob.ptr            = reinterpret_cast<void *>((raw + 255) & ~uintptr_t{ 255 });
    blob.bytes          = bytes;
    blob.device         = device;
    blob.host_pinned    = host;
    return blob;
}

static moe_mmid_materialized_owner_plan owner_plan(int device, uint64_t queue, bool secondary = false) {
    moe_mmid_materialized_owner_plan owner;
    owner.owner_device = device;
    owner.queue_cookie = queue;
    check(moe_mmid_plan_workspace({ 64, 96, 2, 2, 3 }, secondary, &owner.geometry), "owner geometry failed");
    check(moe_mmid_checked_pool_bytes(owner.geometry, MOE_MMID_WORKSPACE_DEPTH, &owner.device_pool_bytes,
                                      &owner.host_pool_bytes),
          "owner pool failed");
    return owner;
}

static void finalized_owner_accounting() {
    std::vector<moe_mmid_owner_accounting> owners = {
        { 0, 1000, 600, 0 },
        { 1, 8,    7,   0 },
    };
    size_t total = 607;
    // Device 1 is eligible but unused: absence from finalized charges means its
    // deliberately tiny budget cannot poison primary acceptance.
    check(moe_mmid_account_actual_owners(
              {
                  { 0, 100 }
    },
              &owners, &total),
          "unused low-budget secondary was charged speculatively");
    check(owners[0].used_bytes == 700 && owners[0].workspace_bytes == 100 && owners[1].used_bytes == 7 &&
              owners[1].workspace_bytes == 0 && total == 707,
          "finalized primary totals mismatch");

    auto   before       = owners;
    size_t before_total = total;
    check(!moe_mmid_account_actual_owners(
              {
                  { 1, 2 }
    },
              &owners, &total),
          "actual alternate over budget was accepted");
    check(owners[1].used_bytes == before[1].used_bytes && total == before_total,
          "failed actual-alternate charge was not atomic");

#ifndef MMID_TSAN_BUILD
    const auto                                oom_before = owners;
    const size_t                              oom_total  = total;
    const std::vector<std::pair<int, size_t>> oom_charges{
        { 0, 1 }
    };
    g_fail_heap_allocations.store(true);
    const bool oom_result = moe_mmid_account_actual_owners(oom_charges, &owners, &total);
    g_fail_heap_allocations.store(false);
    check(!oom_result && owners.size() == oom_before.size() && owners[0].used_bytes == oom_before[0].used_bytes &&
              owners[1].used_bytes == oom_before[1].used_bytes && total == oom_total,
          "accounting owner-copy OOM escaped noexcept or mutated outputs");
#endif

    owners[1].budget_bytes = 20;
    check(moe_mmid_account_actual_owners(
              {
                  { 1, 2 }
    },
              &owners, &total),
          "actual alternate workspace was not charged");
    check(owners[1].used_bytes == 9 && owners[1].workspace_bytes == 2 && total == 709,
          "actual alternate/per-device/global totals mismatch");
}

static void stable_geometry_single_device_budget_boundary() {
    size_t old_capacity = 0, grown_capacity = 0;
    check(moe_mmid_capacity(4, 8, 1, &old_capacity) && moe_mmid_capacity(8, 8, 1, &grown_capacity) &&
              old_capacity == 4 && grown_capacity == 8,
          "runtime n_ctx growth setup failed");

    // The immutable pool was conservatively materialized for the grown demand;
    // runtime n_ctx growth therefore remains a stable-geometry fit.
    moe_mmid_workspace_geometry immutable, grown;
    check(moe_mmid_plan_workspace({ 64, 96, 2, 2, grown_capacity }, false, &immutable) &&
              moe_mmid_plan_workspace({ 64, 96, 2, 2, grown_capacity }, false, &grown) &&
              immutable.device_slot_bytes == grown.device_slot_bytes,
          "stable runtime geometry setup failed");

    const size_t exact_budget = 1000 + immutable.device_slot_bytes;
    size_t       admitted     = 77;
    check(moe_mmid_admit_single_device_total(1000, immutable.device_slot_bytes, exact_budget, &admitted) &&
              admitted == exact_budget,
          "stable geometry exact-budget runtime refresh was rejected");
    const size_t unchanged = admitted;
    check(!moe_mmid_admit_single_device_total(1001, immutable.device_slot_bytes, exact_budget, &admitted) &&
              admitted == unchanged,
          "stable geometry budget+1 refresh was accepted or mutated output");
    check(!moe_mmid_admit_single_device_total(std::numeric_limits<size_t>::max(), 1, std::numeric_limits<size_t>::max(),
                                              &admitted) &&
              admitted == unchanged,
          "stable geometry overflow mutated admission output");
}

static void runtime_per_device_rebuild_is_atomic() {
    const std::vector<int>    devices{ 0, 1 };
    const std::vector<size_t> budgets{ 1000, 400 };
    std::vector<size_t>       used{ 1, 2 };
    check(moe_mmid_rebuild_per_device_usage(
              {
                  500, 200
    },
              { { 0, 100 }, { 1, 50 } }, { { 0, 80 }, { 1, 40 } }, devices, budgets, &used) &&
              used == std::vector<size_t>({ 680, 290 }),
          "KV growth stable-fit rebuild preserved stale per-device totals");
    const auto before = used;
    check(!moe_mmid_rebuild_per_device_usage(
              {
                  500, 200
    },
              { { 0, 100 }, { 1, 50 } }, { { 0, 80 }, { 1, 151 } }, devices, budgets, &used) &&
              used == before,
          "unequal-budget replacement rebuild was not atomic");
}

static void replacement_accounting_is_atomic() {
    const std::vector<int>    devices{ 0, 1 };
    const std::vector<size_t> budgets{ 1000, 300 };
    std::vector<size_t>       used{ 700, 250 };
    size_t                    total = 950;
    check(moe_mmid_reaccount_replacement(
              {
                  { 0, 100 },
                  { 1, 50  }
    },
              { { 0, 100 }, { 1, 50 } }, devices, budgets, &used, &total) &&
              used == std::vector<size_t>({ 700, 250 }) && total == 950,
          "stable-fit replacement changed original totals");
    check(moe_mmid_reaccount_replacement(
              {
                  { 0, 100 },
                  { 1, 50  }
    },
              { { 0, 150 }, { 1, 70 } }, devices, budgets, &used, &total) &&
              used == std::vector<size_t>({ 750, 270 }) && total == 1020,
          "growth replacement double-counted its base");
    const auto   before       = used;
    const size_t before_total = total;
    check(!moe_mmid_reaccount_replacement(
              {
                  { 0, 150 },
                  { 1, 70  }
    },
              { { 0, 150 }, { 1, 101 } }, devices, budgets, &used, &total) &&
              used == before && total == before_total,
          "unequal per-device budget rejection was not atomic");
    used                       = { std::numeric_limits<size_t>::max(), 1 };
    total                      = std::numeric_limits<size_t>::max();
    const auto overflow_before = used;
    check(!moe_mmid_reaccount_replacement(
              {
    },
              { { 0, 1 } }, devices, { std::numeric_limits<size_t>::max(), 2 }, &used, &total) &&
              used == overflow_before && total == std::numeric_limits<size_t>::max(),
          "replacement overflow mutated accounting");
}

static void cookie_saturates_without_reuse() {
    std::atomic<uint64_t> cookie{ std::numeric_limits<uint64_t>::max() - 2 };
    check(moe_mmid_mint_monotonic_cookie(cookie) == std::numeric_limits<uint64_t>::max() - 1,
          "last valid queue cookie was not minted");
    check(moe_mmid_mint_monotonic_cookie(cookie) == 0 && moe_mmid_mint_monotonic_cookie(cookie) == 0 &&
              cookie.load() == std::numeric_limits<uint64_t>::max() - 1,
          "exhausted queue cookie wrapped or reused zero/MAX");
}

static void registry_materialization_and_rollback() {
    const moe_mmid_model_token  token{ 1, 2, 3 };
    const auto                  owner = owner_plan(0, 100);
    std::atomic<int>            destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto                        allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 9, 0, { owner }, allocator) == moe_mmid_materialize_status::PUBLISHED,
          "registry publication failed");
    check(registry.published_contexts() == 1, "published context missing");
    const auto listed = registry.list();
    check(listed.size() == 1 && listed[0].token.model_id == token.model_id && listed[0].plan_identity == 9 &&
              listed[0].submit_device == 0 && listed[0].owner_count == 1,
          "published lifecycle context listing mismatch");
    check(registry.materialize(token, 9, 0, { owner }, allocator) == moe_mmid_materialize_status::ALREADY_PUBLISHED,
          "duplicate publication replaced context");

    auto too_small = owner;
    --too_small.device_pool_bytes;
    check(registry.materialize({ 4, 5, 6 }, 10, 0, { too_small }, allocator) == moe_mmid_materialize_status::INVALID,
          "T-1 immutable plan admitted");

    moe_mmid_workspace_registry rollback_registry;
    std::atomic<int>            rollback_destroyed{ 0 };
    int                         calls   = 0;
    auto                        failing = [&](bool host, int device, size_t bytes, size_t) {
        ++calls;
        return calls == 2 ? moe_mmid_blob{} : fake_blob(host, device, bytes, &rollback_destroyed);
    };
    check(rollback_registry.materialize({ 7, 8, 9 }, 11, 0, { owner, owner_plan(1, 101, true) }, failing) ==
              moe_mmid_materialize_status::ALLOCATION_FAILED,
          "allocation failure did not roll back");
    check(rollback_registry.published_contexts() == 0 && rollback_destroyed.load() == 1,
          "failed transaction retained or published its first blob");

    auto misaligned = [&](bool host, int device, size_t bytes, size_t) {
        moe_mmid_blob blob = fake_blob(host, device, bytes + 1, &rollback_destroyed);
        blob.ptr           = static_cast<unsigned char *>(blob.ptr) + 1;
        blob.bytes         = bytes;
        return blob;
    };
    check(rollback_registry.materialize({ 12, 13, 14 }, 15, 0, { owner }, misaligned) ==
                  moe_mmid_materialize_status::ALLOCATION_FAILED &&
              rollback_registry.published_contexts() == 0,
          "misaligned device base published instead of rolling back");

    auto throwing_slice = [&](bool host, int device, size_t bytes, size_t) {
        auto blob        = fake_blob(host, device, bytes, &rollback_destroyed);
        blob.slice_owner = [](size_t, size_t) -> std::shared_ptr<void> {
            throw std::bad_alloc();
        };
        return blob;
    };
    check(rollback_registry.materialize({ 15, 16, 17 }, 18, 0, { owner }, throwing_slice) ==
                  moe_mmid_materialize_status::ALLOCATION_FAILED &&
              rollback_registry.published_contexts() == 0,
          "throwing slice owner terminated or published a partial context");
}

static void registry_multi_owner_context_and_identity() {
    moe_mmid_workspace_registry registry;
    std::atomic<int>            destroyed{ 0 };
    std::atomic<int>            allocation_calls{ 0 };
    auto                        allocator = [&](bool host, int device, size_t bytes, size_t) {
        allocation_calls.fetch_add(1);
        return fake_blob(host, device, bytes, &destroyed);
    };
    const moe_mmid_model_token a{ 10, 11, 12 }, b{ 10, 13, 14 };
    check(registry.materialize(a, 1000, 0, { owner_plan(0, 20), owner_plan(1, 21, true) }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "multi-owner context failed");
    check(registry.materialize(b, 1001, 1, { owner_plan(1, 31) }, allocator) == moe_mmid_materialize_status::PUBLISHED,
          "second context failed");
    auto wrong_queue = registry.acquire(a, 1000, 0, 1, 20);
    check(wrong_queue.status == moe_mmid_lease_status::INVALID, "wrong owner queue admitted");
    const int allocation_calls_before_acquire = allocation_calls.load();
#ifndef MMID_TSAN_BUILD
    const size_t heap_before_acquire = g_heap_allocations.load();
    g_fail_heap_allocations.store(true);
#endif
    auto lease = registry.acquire(a, 1000, 0, 1, 21);
#ifndef MMID_TSAN_BUILD
    g_fail_heap_allocations.store(false);
#endif
    check(lease.status == moe_mmid_lease_status::ACQUIRED && lease.lease.owner_device() == 1 &&
              lease.lease.submit_device() == 0 && lease.lease.plan_identity() == 1000,
          "lease identities mismatch");
    check(allocation_calls.load() == allocation_calls_before_acquire, "acquire allocated backing storage");
#ifndef MMID_TSAN_BUILD
    check(g_heap_allocations.load() == heap_before_acquire, "acquire performed a heap allocation");
#endif
    check(lease.lease.slices().activation_f32.valid() && lease.lease.slices().host.valid(),
          "exact retained slices missing");
    check(lease.lease.terminal_release(21, lease.lease.generation() + 1) == moe_mmid_release_status::STALE,
          "wrong lease generation released slot");
    check(lease.lease.terminal_release(20, lease.lease.generation()) == moe_mmid_release_status::WRONG_QUEUE,
          "wrong terminal queue released slot");
    check(lease.lease.terminal_release(21, lease.lease.generation()) == moe_mmid_release_status::RELEASED,
          "terminal release failed");
}

static void registry_replacement_and_queue_reset() {
    moe_mmid_workspace_registry registry;
    std::atomic<int>            destroyed{ 0 };
    auto                        allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    const moe_mmid_model_token token{ 20, 21, 22 };
    check(
        registry.materialize(token, 200, 0, { owner_plan(0, 70) }, allocator) == moe_mmid_materialize_status::PUBLISHED,
        "replacement baseline failed");
    check(
        registry.materialize(token, 201, 0, { owner_plan(0, 71) }, allocator) == moe_mmid_materialize_status::PUBLISHED,
        "replacement transaction failed");
    check(registry.acquire(token, 201, 0, 0, 70).status == moe_mmid_lease_status::INVALID,
          "queue reset accepted stale generation cookie");
    check(registry.retire(token, 200), "exact old-plan retirement failed");
    check(registry.acquire(token, 200, 0, 0, 70).status == moe_mmid_lease_status::INVALID,
          "retired replacement baseline remained visible");
    auto current = registry.acquire(token, 201, 0, 0, 71);
    check(current.status == moe_mmid_lease_status::ACQUIRED, "new replacement context disappeared with old retirement");
    check(current.lease.terminal_release(71, current.lease.generation()) == moe_mmid_release_status::RELEASED,
          "replacement terminal release failed");
}

static void registry_concurrent_depth_busy() {
    moe_mmid_workspace_registry registry;
    std::atomic<int>            destroyed{ 0 };
    auto                        allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    const moe_mmid_model_token token{ 30, 31, 32 };
    check(
        registry.materialize(token, 40, 0, { owner_plan(0, 50) }, allocator) == moe_mmid_materialize_status::PUBLISHED,
        "concurrent context publish failed");
    std::atomic<int>         ready{ 0 }, attempted{ 0 }, acquired{ 0 }, busy{ 0 };
    std::atomic<bool>        go{ false };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            auto result = registry.acquire(token, 40, 0, 0, 50);
            if (result.status == moe_mmid_lease_status::ACQUIRED) {
                acquired.fetch_add(1);
            }
            if (result.status == moe_mmid_lease_status::BUSY) {
                busy.fetch_add(1);
            }
            attempted.fetch_add(1);
            while (attempted.load() != 8) {
                std::this_thread::yield();
            }
            if (result.status == moe_mmid_lease_status::ACQUIRED) {
                check(result.lease.terminal_release(50, result.lease.generation()) == moe_mmid_release_status::RELEASED,
                      "concurrent registry release failed");
            }
            (void) i;
        });
    }
    while (ready.load() != 8) {
        std::this_thread::yield();
    }
    go.store(true);
    for (auto & thread : threads) {
        thread.join();
    }
    check(acquired.load() == 2 && busy.load() == 6, "registry depth grew under BUSY pressure");
}

static void registry_retirement_retains_outstanding_lease() {
    moe_mmid_registry_lease delayed;
    std::atomic<int>        destroyed{ 0 };
    {
        moe_mmid_workspace_registry registry;
        auto                        allocator = [&](bool host, int device, size_t bytes, size_t) {
            return fake_blob(host, device, bytes, &destroyed);
        };
        const moe_mmid_model_token token{ 40, 41, 42 };
        check(registry.materialize(token, 50, 0, { owner_plan(0, 60) }, allocator) ==
                  moe_mmid_materialize_status::PUBLISHED,
              "delayed context publish failed");
        auto acquired = registry.acquire(token, 50, 0, 0, 60);
        check(acquired.status == moe_mmid_lease_status::ACQUIRED, "delayed lease acquire failed");
        delayed = acquired.lease;
        check(registry.retire(token), "model retirement failed");
        check(registry.acquire(token, 50, 0, 0, 60).status == moe_mmid_lease_status::INVALID,
              "retired model remained admissible");
        check(destroyed.load() == 0, "retirement destroyed outstanding lease blobs");
    }
    check(destroyed.load() == 0, "registry destruction destroyed outstanding lease blobs");
    check(delayed.terminal_release(60, delayed.generation()) == moe_mmid_release_status::RELEASED,
          "delayed terminal release failed");
    delayed = {};
    check(destroyed.load() == 2, "device/host blobs were not destroyed after final lease");
}

static void concurrent_depth_busy_and_reuse() {
    moe_mmid_workspace_pool  pool;
    std::atomic<int>         ready{ 0 };
    std::atomic<bool>        go{ false };
    std::atomic<int>         attempted{ 0 };
    std::atomic<int>         acquired{ 0 };
    std::atomic<int>         busy{ 0 };
    std::vector<std::thread> threads;
    for (uint64_t i = 1; i <= 8; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            auto result = pool.acquire(i, i);
            if (result.status == moe_mmid_lease_status::ACQUIRED) {
                acquired.fetch_add(1);
            } else if (result.status == moe_mmid_lease_status::BUSY) {
                busy.fetch_add(1);
            }
            attempted.fetch_add(1);
            while (attempted.load() < 8) {
                std::this_thread::yield();
            }
            if (result.status == moe_mmid_lease_status::ACQUIRED) {
                check(pool.terminal_release(result.lease, i, i) == moe_mmid_release_status::RELEASED,
                      "concurrent release failed");
            }
        });
    }
    while (ready.load() != 8) {
        std::this_thread::yield();
    }
    go.store(true);
    for (auto & thread : threads) {
        thread.join();
    }
    check(acquired.load() == 2 && busy.load() == 6, "concurrent fixed-depth result mismatch");
    check(pool.acquire(99, 99).status == moe_mmid_lease_status::ACQUIRED, "released pool was not reusable");
}

int main() {
    try {
        geometry_exact_and_boundary();
        exact_256_slice_start_after_128_bytes();
        broadcast_and_occurrences();
        maxima_and_overflow();
        pool_identity_generation_and_terminal_release();
        concurrent_depth_busy_and_reuse();
        finalized_owner_accounting();
        stable_geometry_single_device_budget_boundary();
        runtime_per_device_rebuild_is_atomic();
        replacement_accounting_is_atomic();
        cookie_saturates_without_reuse();
        registry_materialization_and_rollback();
        registry_multi_owner_context_and_identity();
        registry_replacement_and_queue_reset();
        registry_concurrent_depth_busy();
        registry_retirement_retains_outstanding_lease();
        std::cout << "moe-mmid-workspace-plan: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
