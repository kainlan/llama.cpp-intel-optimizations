#define GGML_SYCL_RETENTION_TESTING 1
#include "moe-mmid-workspace.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace ggml_sycl;

static_assert(!std::is_constructible<queue_submission_authority,
                                     moe_mmid_queue_capability, uint64_t,
                                     std::shared_ptr<moe::device_terminal>>::value,
              "foreign terminal must not be publicly bindable to a queue label");

namespace ggml_sycl { struct lifecycle_plan_snapshot {}; }

// This standalone target intentionally does not link moe-graph-retention.cpp;
// provide the capability constructors used only by its friend-backed fixture.
namespace ggml_sycl::moe {
struct test_graph_publication {
    graph_owner_key key;
    published_graph_token token;
    std::shared_ptr<const graph_retention_record> snapshot;
};
static std::vector<test_graph_publication> test_graph_publications;
static graph_retention_registry test_graph_registry;

std::shared_ptr<const graph_retention_record> graph_retention_registry::snapshot(graph_owner_key key) const noexcept {
    for (const auto & publication : test_graph_publications)
        if (publication.key.context.value == key.context.value && publication.key.epoch.value == key.epoch.value)
            return publication.snapshot;
    return {};
}
retention_error graph_retention_registry::acquire_published_token(graph_owner_key key,
                                                                  published_graph_token * out) const noexcept {
    if (!out) return retention_error::MISMATCH;
    for (const auto & publication : test_graph_publications) {
        if (publication.key.context.value == key.context.value && publication.key.epoch.value == key.epoch.value) {
            *out = publication.token;
            return retention_error::OK;
        }
    }
    return retention_error::STALE;
}

retained_allocation_owner::retained_allocation_owner(uint64_t allocation_id, uint64_t generation, int device,
                                                     size_t extent, std::shared_ptr<const void> handle) :
    allocation_id_(allocation_id), generation_(generation), device_(device), extent_(extent), handle_(std::move(handle)) {}

graph_private_table_owner::graph_private_table_owner(graph_owner_key owner, uint64_t table_id, uint64_t layout_id,
                                                     int device, std::vector<entry_type> entries) :
    owner_(owner), table_id_(table_id), layout_id_(layout_id), device_(device), entries_(std::move(entries)) {}

bool queue_quiescence_proof::ready() const noexcept { return ready_ && ready_(state_.get()); }
bool queue_quiescence_proof::wait_and_confirm() noexcept { return wait_ && wait_(state_.get()) && ready(); }

std::shared_ptr<const graph_private_table_owner> graph_private_table_owner::create(
    graph_owner_key owner, uint64_t table_id, uint64_t layout_id, int device, std::vector<entry_type> entries) {
    return std::shared_ptr<const graph_private_table_owner>(
        new graph_private_table_owner(owner, table_id, layout_id, device, std::move(entries)));
}
} // namespace ggml_sycl::moe

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
    owner.secondary_owner = secondary;
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

    // Planner component maxima need not correspond to one source shape: K and
    // N may come from different tensors. Materialization consumes this exact
    // canonical aggregate rather than reconstructing it from a zero/fake shape.
    moe_mmid_workspace_geometry mixed_k, mixed_n, aggregate;
    check(moe_mmid_plan_workspace({ 128, 32, 2, 2, 3 }, false, &mixed_k) &&
              moe_mmid_plan_workspace({ 32, 128, 2, 2, 3 }, false, &mixed_n) &&
              moe_mmid_component_max(&aggregate, mixed_k) && moe_mmid_component_max(&aggregate, mixed_n),
          "mixed K/N aggregate setup failed");
    moe_mmid_materialized_owner_plan aggregate_owner;
    aggregate_owner.owner_device = 0;
    aggregate_owner.queue_cookie = 901;
    aggregate_owner.geometry = aggregate;
    check(moe_mmid_checked_pool_bytes(aggregate, MOE_MMID_WORKSPACE_DEPTH, &aggregate_owner.device_pool_bytes,
                                      &aggregate_owner.host_pool_bytes), "mixed aggregate pool sizing failed");
    const moe_mmid_model_token aggregate_token{ 91, 92, 93 };
    check(registry.materialize(aggregate_token, 902, 0, { aggregate_owner }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "mixed K/N canonical aggregate was not materialized");
    check(registry.published_contexts() == 2, "published contexts missing");
    const auto listed = registry.list();
    check(listed.size() == 2 && listed[0].token.model_id == token.model_id && listed[0].plan_identity == 9 &&
              listed[0].submit_device == 0 && listed[0].owner_count == 1,
          "published lifecycle context listing mismatch");
    check(registry.materialize(token, 9, 0, { owner }, allocator) == moe_mmid_materialize_status::ALREADY_PUBLISHED,
          "duplicate publication replaced context");

    auto overlap = owner;
    overlap.geometry.output_f32_offset = overlap.geometry.activation_f32_offset;
    check(registry.materialize({ 2, 3, 4 }, 91, 0, { overlap }, allocator) == moe_mmid_materialize_status::INVALID,
          "overlapping caller geometry bypassed canonical registry geometry");

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
    std::atomic<int>            destroyed{ 0 };
    std::atomic<int>            allocation_calls{ 0 };
    moe_mmid_workspace_registry registry;
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
    std::atomic<int>            destroyed{ 0 };
    moe_mmid_workspace_registry registry;
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
    std::atomic<int>            destroyed{ 0 };
    moe_mmid_workspace_registry registry;
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

struct ready_terminal final : moe::device_terminal {
    explicit ready_terminal(std::shared_ptr<std::atomic<bool>> ready) : ready_(std::move(ready)) {}
    bool ready() const noexcept override { return ready_->load(); }
    void wait() noexcept override { ready_->store(true); }
    std::shared_ptr<std::atomic<bool>> ready_;
};

static moe::published_graph_token graph_token_for(moe::graph_owner_key key, uint64_t serial) {
    struct token_wire { moe::graph_owner_key key; uint64_t serial; } wire{ key, serial };
    moe::published_graph_token token;
    static_assert(sizeof(token) == sizeof(wire), "graph token test fixture ABI");
    std::memcpy(&token, &wire, sizeof(token));
    return token;
}

static moe_mmid_admission_request admission_request(const moe_mmid_model_token & token, uint64_t plan,
                                                     std::vector<moe_mmid_admission_owner> owners,
                                                     uint64_t epoch = 900,
                                                     std::shared_ptr<std::atomic<bool>> ready =
                                                         std::make_shared<std::atomic<bool>>(true)) {
    moe_mmid_admission_request request;
    request.token = token;
    request.plan_identity = plan;
    request.submit_device = 0;
    request.top_k = 2;
    request.ne11 = 2;
    request.K = 64;
    request.N = 96;
    request.type = 1;
    request.owners = std::move(owners);

    const moe::graph_owner_key key{ { token.model_id }, { epoch } };
    auto snapshot = std::make_shared<moe::graph_retention_record>();
    snapshot->key = key;
    snapshot->root = { { token.model_id }, { token.load_txn_id }, { 1, token.generation } };
    snapshot->phase = moe::retention_phase::INSTALLED;
    snapshot->publication_serial = 12345;
    std::vector<moe::retained_allocation_owner> table_entries;
    auto bindings = std::make_shared<std::vector<moe::mmid_batch_binding>>();
    const uint64_t ids[] = { 101, 102, 101, 103, 104, 101 };
    for (uint32_t i = 0; i < 6; ++i) {
        const int device = static_cast<int>(ids[i] % 2);
        auto owner = moe::retained_allocation_test_factory::mint(ids[i], 7, device, 4096,
                                                                 std::make_shared<int>(static_cast<int>(ids[i])));
        moe::mmid_batch_binding binding{ { ids[i], 7, 55, device, 64, 128, i }, owner };
        bindings->push_back(binding);
        snapshot->batches.push_back(binding);
        bool present = false;
        for (const auto & entry : table_entries) present |= entry.allocation_id() == ids[i];
        if (!present) table_entries.push_back(owner);
    }
    auto table = moe::graph_private_table_owner::create(key, 888, 55, 0, std::move(table_entries));
    snapshot->tables.push_back({ table->table_id(), table->layout_id(), table->device(), table });
    for (const auto & owner : request.owners)
        snapshot->terminals.emplace(owner.owner_device, std::make_shared<ready_terminal>(ready));
    request.graph_token = graph_token_for(key, snapshot->publication_serial);
    moe::test_graph_publications.push_back({ key, request.graph_token, snapshot });
    request.graph_registry = &moe::test_graph_registry;
    request.graph_snapshot = snapshot;
    request.retained_occurrences = bindings;
    request.table_owner = table;
    return request;
}

static void registry_component_max_constituent_coverage() {
    const moe_mmid_model_token token{ 49, 50, 51 };
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    moe_mmid_workspace_geometry k_max, n_max, aggregate;
    check(moe_mmid_plan_workspace({ 128, 32, 2, 2, 3 }, false, &k_max) &&
              moe_mmid_plan_workspace({ 32, 128, 2, 2, 3 }, false, &n_max) &&
              moe_mmid_component_max(&aggregate, k_max) && moe_mmid_component_max(&aggregate, n_max),
          "constituent aggregate setup failed");
    moe_mmid_materialized_owner_plan owner;
    owner.owner_device = 0; owner.queue_cookie = 690; owner.geometry = aggregate;
    check(moe_mmid_checked_pool_bytes(aggregate, MOE_MMID_WORKSPACE_DEPTH, &owner.device_pool_bytes,
                                      &owner.host_pool_bytes) &&
              registry.materialize(token, 691, 0, { owner }, allocator) == moe_mmid_materialize_status::PUBLISHED,
          "constituent aggregate publication failed");
    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>>{ { 128, 32 }, { 32, 128 } }) {
        auto request = admission_request(token, 691, { { 0, 690 } }, 692 + shape.first);
        request.K = shape.first; request.N = shape.second;
        auto admitted = registry.admit(request);
        check(admitted.status == moe_mmid_lease_status::ACQUIRED, "constituent coverage rejected");
        const auto & geometry = admitted.bundle.owner_leases()[0].geometry();
        const auto & slices = admitted.bundle.owner_leases()[0].slices();
        check(slices.activation_f32.bytes == geometry.activation_f32_bytes &&
                  slices.output_f32.bytes == geometry.output_f32_bytes &&
                  slices.activation_f32.bytes <= aggregate.activation_f32_bytes &&
                  slices.output_f32.bytes <= aggregate.output_f32_bytes,
              "admitted slices were not exact requested subranges");
        check(admitted.bundle.terminal_release(), "constituent release failed");
    }
    auto too_large = admission_request(token, 691, { { 0, 690 } }, 999);
    too_large.K = 160;
    check(registry.admit(too_large).status == moe_mmid_lease_status::INVALID,
          "T-1 aggregate capacity admitted oversized constituent");
}

static void registry_atomic_bundle_authority() {
    const moe_mmid_model_token token{ 51, 52, 53 };
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 700, 0, { owner_plan(0, 701), owner_plan(1, 702, true) }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "atomic bundle context publication failed");
    auto request = admission_request(token, 700, { { 1, 702 }, { 0, 701 } });
    auto admitted = registry.admit(request);
    check(admitted.status == moe_mmid_lease_status::ACQUIRED && admitted.bundle.valid(),
          "all-owner atomic admission failed");
    check(admitted.bundle.owner_count() == 2 && admitted.bundle.graph_owners()[0].owner_device == 1 &&
              admitted.bundle.owner_leases()[0].owner_device() == 1 && admitted.bundle.owner_leases()[1].owner_device() == 0,
          "graph owner enumeration lost request order");
    check(admitted.bundle.retained_occurrences().size() == 6 &&
              admitted.bundle.retained_occurrences()[0].identity.allocation_id ==
                  admitted.bundle.retained_occurrences()[2].identity.allocation_id &&
              admitted.bundle.identity_digest() != 0,
          "repeated occurrence identity was deduplicated or digest missing");
    std::array<moe::mmid_operand_identity, 6> exact_identities{};
    for (size_t i = 0; i < exact_identities.size(); ++i)
        exact_identities[i] = admitted.bundle.retained_occurrences()[i].identity;
    check(admitted.bundle.matches(0, 1, 64, 96, 1, exact_identities.data(), exact_identities.size()),
          "complete exact identity match rejected");
    exact_identities[2].byte_offset++;
    check(!admitted.bundle.matches(0, 1, 64, 96, 1, exact_identities.data(), exact_identities.size()),
          "offset identity substitution matched authority");
    for (size_t i = 0; i < admitted.bundle.owner_count(); ++i) {
        const auto & lease = admitted.bundle.owner_leases()[i];
        check(lease.epoch() == request.graph_snapshot->key.epoch.value && lease.slices().activation_f32.valid() &&
                  lease.slices().output_q8.valid(), "bundle lease omitted epoch or exact slices");
    }
    auto second = registry.admit(request);
    check(second.status == moe_mmid_lease_status::ACQUIRED, "second depth slot unavailable");
    check(registry.admit(request).status == moe_mmid_lease_status::BUSY, "all-owner depth exhaustion not BUSY");
    check(admitted.bundle.terminal_release(), "exact all-owner terminal release failed");
    check(second.bundle.terminal_release(), "second all-owner terminal release failed");

    auto wrong = request;
    wrong.owners[0].queue_cookie++;
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "wrong queue admitted");
    wrong = request; wrong.plan_identity++;
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "wrong plan admitted");
    wrong = request; wrong.submit_device = 1;
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "wrong submit device admitted");
    wrong = request; wrong.token.generation++;
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "stale model generation admitted");
    wrong = request;
    wrong.graph_token = graph_token_for({ request.graph_snapshot->key.context,
                                          { request.graph_snapshot->key.epoch.value + 1 } },
                                        request.graph_snapshot->publication_serial);
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "wrong graph capability admitted");
    wrong = request;
    auto substituted = std::make_shared<std::vector<moe::mmid_batch_binding>>(*request.retained_occurrences);
    substituted->at(2).identity.generation++;
    wrong.retained_occurrences = substituted;
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "identity substitution admitted");
    wrong = request;
    wrong.table_owner = moe::graph_private_table_owner::create(
        request.graph_snapshot->key, request.table_owner->table_id(), request.table_owner->layout_id(),
        request.table_owner->device(), request.table_owner->entries());
    check(registry.admit(wrong).status == moe_mmid_lease_status::INVALID, "table-owner substitution admitted");
}

static void common_direct_authority_plan_queue_and_lifetime() {
    const moe_mmid_model_token token{ 151, 152, 153 };
    auto plan = std::make_shared<const lifecycle_plan_snapshot>();
    int exact_queue = 0;
    auto queue_lifetime = std::make_shared<int>(7);
    auto capability = moe_mmid_queue_capability_test_factory::mint(0, &exact_queue, 1701, queue_lifetime);
    auto owner = owner_plan(0, 1701); owner.queue_capability = capability;
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 1700, plan, 0, { owner }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED, "authoritative context publication failed");
    check(registry.exact_queue(token, plan, 0, 0, &exact_queue).valid(),
          "materialized exact shared plan/queue was not discoverable");
    int substituted_queue = 0;
    check(!registry.exact_queue(token, plan, 0, 0, &substituted_queue).valid(),
          "different queue object acquired exact capability");
    auto equal_but_foreign_plan = std::make_shared<const lifecycle_plan_snapshot>();
    check(!registry.exact_queue(token, equal_but_foreign_plan, 0, 0, &exact_queue).valid(),
          "equal-metadata foreign plan acquired exact capability");

    auto graph = admission_request(token, 1700, { { 0, 1701 } }, 1702);
    auto pin = std::make_shared<int>(9); std::weak_ptr<int> weak_pin = pin;
    auto terminal = graph.graph_snapshot->terminals.at(0);
    auto make_authority = [&](moe_mmid_queue_capability queue, std::shared_ptr<void> invocation_pin) {
        auto submission = workspace_admission_authority_test_factory::terminal(queue, 1702, terminal);
        return workspace_admission_authority_test_factory::direct(
            token, 1700, 1702, 0, plan, graph.retained_occurrences, graph.table_owner,
            { { 0, 1701 } }, { std::move(queue) }, std::move(invocation_pin), { std::move(submission) });
    };
    moe_mmid_authoritative_admission_request direct{ make_authority(capability, pin), { 2, 2, 64, 96, 1 } };
    pin.reset();
    auto committed = registry.admit(std::move(direct));
    check(committed.status == moe_mmid_lease_status::ACQUIRED && committed.bundle.valid() && !weak_pin.expired(),
          "direct authority did not retain invocation pin");

    auto forged = moe_mmid_queue_capability_test_factory::mint(0, &exact_queue, 1701, queue_lifetime);
    moe_mmid_authoritative_admission_request wrong_queue{ make_authority(forged, {}), { 2, 2, 64, 96, 1 } };
    check(registry.admit(std::move(wrong_queue)).status == moe_mmid_lease_status::INVALID,
          "same-device same-cookie different queue capability admitted");
    auto wrong_submission = workspace_admission_authority_test_factory::terminal(forged, 1702, terminal);
    auto wrong_terminal_authority = workspace_admission_authority_test_factory::direct(
        token, 1700, 1702, 0, plan, graph.retained_occurrences, graph.table_owner,
        { { 0, 1701 } }, { capability }, {}, { wrong_submission });
    check(registry.admit({ std::move(wrong_terminal_authority), { 2, 2, 64, 96, 1 } }).status ==
              moe_mmid_lease_status::INVALID, "substituted terminal queue authority admitted");

    auto forged_graph_authority = workspace_admission_authority_test_factory::graph(
        token, 1700, 0, plan, graph.graph_registry, graph.graph_token, graph.graph_snapshot,
        graph.retained_occurrences, graph.table_owner, { { 0, 1701 } }, { forged });
    check(registry.admit({ std::move(forged_graph_authority), { 2, 2, 64, 96, 1 } }).status ==
              moe_mmid_lease_status::INVALID, "GRAPH substituted exact queue capability admitted");

    auto graph_authority = workspace_admission_authority_test_factory::graph(
        token, 1700, 0, plan, graph.graph_registry, graph.graph_token, graph.graph_snapshot,
        graph.retained_occurrences, graph.table_owner, { { 0, 1701 } }, { capability });
#ifndef MMID_TSAN_BUILD
    g_fail_heap_allocations.store(true);
#endif
    auto graph_admitted = registry.admit({ std::move(graph_authority), { 2, 2, 64, 96, 1 } });
#ifndef MMID_TSAN_BUILD
    g_fail_heap_allocations.store(false);
#endif
    check(graph_admitted.status == moe_mmid_lease_status::ACQUIRED,
          "GRAPH common helper allocated or rejected exact authority");
    check(graph_admitted.bundle.terminal_release(), "GRAPH common helper release failed");

    auto replacement = std::make_shared<const lifecycle_plan_snapshot>();
    std::atomic<bool> late_entered{ false }, late_release{ false };
    moe_mmid_materialize_status late_status = moe_mmid_materialize_status::PUBLISHED;
    auto late_allocator = [&](bool host, int device, size_t bytes, size_t alignment) {
        late_entered.store(true, std::memory_order_release);
        while (!late_release.load(std::memory_order_acquire)) std::this_thread::yield();
        return allocator(host, device, bytes, alignment);
    };
    std::thread late_materialize([&] {
        late_status = registry.materialize(token, 1700, plan, 0, { owner }, late_allocator);
    });
    while (!late_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    check(registry.replace_plan(plan, replacement, 1703, true), "stable exact plan replacement failed");
    late_release.store(true, std::memory_order_release); late_materialize.join();
    check(late_status == moe_mmid_materialize_status::INVALID,
          "paused late stale-plan materialization crossed replacement tombstone");
    moe_mmid_authoritative_admission_request stale{ make_authority(capability, {}), { 2, 2, 64, 96, 1 } };
    check(registry.admit(std::move(stale)).status == moe_mmid_lease_status::INVALID,
          "stale exact plan admitted after replacement rebind");
    auto replacement_submission = workspace_admission_authority_test_factory::terminal(capability, 1702, terminal);
    auto replacement_authority = workspace_admission_authority_test_factory::direct(
        token, 1703, 1702, 0, replacement, graph.retained_occurrences, graph.table_owner,
        { { 0, 1701 } }, { capability }, {}, { replacement_submission });
    auto replacement_admitted = registry.admit({ std::move(replacement_authority), { 2, 2, 64, 96, 1 } });
    check(replacement_admitted.status == moe_mmid_lease_status::ACQUIRED,
          "stable replacement did not rebind unchanged materialized context");
    check(replacement_admitted.bundle.terminal_release(), "replacement bundle release failed");
    check(committed.bundle.terminal_release(), "committed old bundle did not finish after plan replacement");
    committed = {};
    check(weak_pin.expired(), "terminal direct bundle retained invocation pin");

    // Generation tickets are reclaimable: repeated stable replacements do not
    // consume tombstone capacity.
    auto current = replacement;
    uint64_t current_id = 1703;
    for (size_t i = 0; i < 1000; ++i) {
        auto next = std::make_shared<const lifecycle_plan_snapshot>();
        check(registry.replace_plan(current, next, ++current_id, true), "1000 stable replacements exhausted protocol");
        current = std::move(next);
    }

    // Changed geometry is prepared off to the side, then atomically selected;
    // the old context is retired from registry ownership.
    auto changed = owner; changed.geometry = {};
    check(moe_mmid_plan_workspace({ 128, 96, 2, 2, 3 }, false, &changed.geometry) &&
              moe_mmid_checked_pool_bytes(changed.geometry, MOE_MMID_WORKSPACE_DEPTH,
                                          &changed.device_pool_bytes, &changed.host_pool_bytes),
          "changed geometry setup failed");
    auto changed_plan = std::make_shared<const lifecycle_plan_snapshot>();
    check(registry.materialize(token, current_id + 1, changed_plan, 0, { changed }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "changed geometry replacement context was not prepared");
    check(registry.replace_plan(current, changed_plan, current_id + 1, false),
          "changed geometry prepared context did not publish atomically");

    auto loser = std::make_shared<const lifecycle_plan_snapshot>();
    auto winner = std::make_shared<const lifecycle_plan_snapshot>();
    std::atomic<int> replacement_wins{ 0 };
    std::atomic<bool> replacement_go{ false };
    std::thread replace_a([&] {
        while (!replacement_go.load(std::memory_order_acquire)) std::this_thread::yield();
        replacement_wins += registry.replace_plan(changed_plan, winner, current_id + 2, true);
    });
    std::thread replace_b([&] {
        while (!replacement_go.load(std::memory_order_acquire)) std::this_thread::yield();
        replacement_wins += registry.replace_plan(changed_plan, loser, current_id + 3, true);
    });
    replacement_go.store(true, std::memory_order_release);
    replace_a.join(); replace_b.join();
    check(replacement_wins.load() == 1, "concurrent replacement did not provide one CAS winner");
}

static void model_scoped_replacement_and_retired_recovery() {
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    int qa = 0, qb = 0;
    auto plan_a = std::make_shared<const lifecycle_plan_snapshot>();
    auto plan_b = std::make_shared<const lifecycle_plan_snapshot>();
    auto cap_a = moe_mmid_queue_capability_test_factory::mint(0, &qa, 2101);
    auto cap_b = moe_mmid_queue_capability_test_factory::mint(0, &qb, 2201);
    auto owner_a = owner_plan(0, 2101); owner_a.queue_capability = cap_a;
    auto owner_b = owner_plan(0, 2201); owner_b.queue_capability = cap_b;
    const moe_mmid_model_token token_a{ 211, 212, 213 }, token_b{ 221, 222, 223 };
    check(registry.materialize(token_b, 2200, plan_b, 0, { owner_b }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED, "cross-model active setup failed");
    auto no_mmid_expected = std::make_shared<const lifecycle_plan_snapshot>();
    auto no_mmid_replacement = std::make_shared<const lifecycle_plan_snapshot>();
    check(registry.replace_plan(no_mmid_expected, no_mmid_replacement, 2300, true),
          "no-MMID model replacement was poisoned by unrelated active model");

    check(registry.materialize(token_a, 2100, plan_a, 0, { owner_a }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED, "retired recovery old setup failed");
    auto graph = admission_request(token_a, 2100, { { 0, 2101 } }, 2102,
                                   std::make_shared<std::atomic<bool>>(false));
    auto terminal = graph.graph_snapshot->terminals.at(0);
    auto submission = workspace_admission_authority_test_factory::terminal(cap_a, 2102, terminal);
    auto authority = workspace_admission_authority_test_factory::direct(
        token_a, 2100, 2102, 0, plan_a, graph.retained_occurrences, graph.table_owner,
        { { 0, 2101 } }, { cap_a }, {}, { submission });
    auto admitted = registry.admit({ std::move(authority), { 2, 2, 64, 96, 1 } });
    check(admitted.status == moe_mmid_lease_status::ACQUIRED && admitted.bundle.mark_possible_submit(),
          "retired recovery submitted-old setup failed");
    auto plan_a2 = std::make_shared<const lifecycle_plan_snapshot>();
    auto changed = owner_a;
    check(moe_mmid_plan_workspace({ 128, 96, 2, 2, 3 }, false, &changed.geometry) &&
              moe_mmid_checked_pool_bytes(changed.geometry, MOE_MMID_WORKSPACE_DEPTH,
                                          &changed.device_pool_bytes, &changed.host_pool_bytes) &&
              registry.materialize(token_a, 2103, plan_a2, 0, { changed }, allocator) ==
                  moe_mmid_materialize_status::PUBLISHED &&
              registry.replace_plan(plan_a, plan_a2, 2103, false),
          "changed replacement did not retain submitted old context");
    terminal->wait();
    admitted = {}; // destroyed submitted bundle; retired registry must retain recovery authority
    check(registry.recover_quarantined(token_a, 2100, false) == 1,
          "retired exact context was not searchable for recovery");

    std::atomic<bool> entered{ false }, release{ false };
    auto paused = [&](bool host, int device, size_t bytes, size_t alignment) {
        entered.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
        return allocator(host, device, bytes, alignment);
    };
    auto plan_b_late = std::make_shared<const lifecycle_plan_snapshot>();
    moe_mmid_materialize_status late = moe_mmid_materialize_status::INVALID;
    std::thread late_thread([&] { late = registry.materialize(token_b, 2202, plan_b_late, 0, { owner_b }, paused); });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    auto plan_a3 = std::make_shared<const lifecycle_plan_snapshot>();
    check(registry.replace_plan(plan_a2, plan_a3, 2104, true),
          "model A replacement failed during model B materialization");
    release.store(true, std::memory_order_release); late_thread.join();
    check(late == moe_mmid_materialize_status::PUBLISHED,
          "model A replacement invalidated model B generation ticket");
}

static void changed_replacement_terminal_release_gc() {
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    int queue_object = 0;
    auto queue = moe_mmid_queue_capability_test_factory::mint(0, &queue_object, 3101);
    auto owner = owner_plan(0, 3101); owner.queue_capability = queue;
    const moe_mmid_model_token token{ 311, 312, 313 };
    auto current = std::make_shared<const lifecycle_plan_snapshot>();
    uint64_t plan_id = 3100;
    check(registry.materialize(token, plan_id, current, 0, { owner }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED, "repeated replacement baseline failed");
    for (size_t iteration = 0; iteration < 24; ++iteration) {
        const uint64_t epoch = 3200 + iteration;
        auto graph = admission_request(token, plan_id, { { 0, 3101 } }, epoch);
        auto terminal = graph.graph_snapshot->terminals.at(0);
        auto submission = workspace_admission_authority_test_factory::terminal(queue, epoch, terminal);
        auto authority = workspace_admission_authority_test_factory::direct(
            token, plan_id, epoch, 0, current, graph.retained_occurrences, graph.table_owner,
            { { 0, 3101 } }, { queue }, {}, { submission });
        auto old = registry.admit({ std::move(authority), { 2, 2, 64, 96, 1 } });
        check(old.status == moe_mmid_lease_status::ACQUIRED, "repeated old admission failed");
        auto next = std::make_shared<const lifecycle_plan_snapshot>();
        auto changed = owner;
        check(moe_mmid_plan_workspace({ 64 + 32 * (iteration + 1), 96, 2, 2, 3 }, false, &changed.geometry) &&
                  moe_mmid_checked_pool_bytes(changed.geometry, MOE_MMID_WORKSPACE_DEPTH,
                                              &changed.device_pool_bytes, &changed.host_pool_bytes) &&
                  registry.materialize(token, plan_id + 1, next, 0, { changed }, allocator) ==
                      moe_mmid_materialize_status::PUBLISHED,
              "repeated changed context preparation failed");
        std::atomic<bool> go{ false };
        bool replaced = false, released = false;
        std::thread replacer([&] { while (!go.load()) std::this_thread::yield();
            replaced = registry.replace_plan(current, next, plan_id + 1, false); });
        std::thread releaser([&] { while (!go.load()) std::this_thread::yield(); released = old.bundle.terminal_release(); });
        go.store(true); replacer.join(); releaser.join();
        check(replaced && released && registry.retired_contexts_for_test() == 0 &&
                  registry.published_contexts() == 1,
              "normal release did not GC retired changed context");
        current = std::move(next); ++plan_id;
    }
    check(registry.retired_contexts_for_test() == 0 && registry.published_contexts() == 1,
          "repeated changed replacement grew registry owners");
}

static void registry_bundle_rollback_move_oom_and_quarantine() {
    const moe_mmid_model_token token{ 61, 62, 63 };
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 800, 0, { owner_plan(0, 801), owner_plan(1, 802, true) }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "rollback context publication failed");
    auto both = admission_request(token, 800, { { 0, 801 }, { 1, 802 } });
    auto owner1_only = both;
    owner1_only.owners.erase(owner1_only.owners.begin());
    // Exact owner-set mismatch cannot partially acquire owner 0.
    check(registry.admit(owner1_only).status == moe_mmid_lease_status::INVALID, "missing owner admitted");

    // Exhaust only the second deterministic owner through the legacy exact-pool
    // API. Atomic admission must inspect owner 0 but leave it completely free.
    auto owner1_a = registry.acquire(token, 800, 0, 1, 802);
    auto owner1_b = registry.acquire(token, 800, 0, 1, 802);
    check(owner1_a.status == moe_mmid_lease_status::ACQUIRED && owner1_b.status == moe_mmid_lease_status::ACQUIRED,
          "second-owner BUSY setup failed");
    check(registry.admit(both).status == moe_mmid_lease_status::BUSY, "second-owner BUSY did not reject bundle");
    auto owner0_a = registry.acquire(token, 800, 0, 0, 801);
    auto owner0_b = registry.acquire(token, 800, 0, 0, 801);
    check(owner0_a.status == moe_mmid_lease_status::ACQUIRED && owner0_b.status == moe_mmid_lease_status::ACQUIRED,
          "second-owner BUSY partially consumed first-owner slot");
    check(owner0_a.lease.terminal_release(801, owner0_a.lease.generation()) == moe_mmid_release_status::RELEASED &&
              owner0_b.lease.terminal_release(801, owner0_b.lease.generation()) == moe_mmid_release_status::RELEASED &&
              owner1_a.lease.terminal_release(802, owner1_a.lease.generation()) == moe_mmid_release_status::RELEASED &&
              owner1_b.lease.terminal_release(802, owner1_b.lease.generation()) == moe_mmid_release_status::RELEASED,
          "BUSY rollback setup cleanup failed");
    {
        auto first = registry.admit(both);
        check(first.status == moe_mmid_lease_status::ACQUIRED, "pre-BUSY admission failed");
        moe_admitted_workspace_bundle moved(std::move(first.bundle));
        check(moved.valid() && !first.bundle.valid(), "bundle move duplicated authority");
        // Scope destruction before possible submit releases every owner.
    }
    auto reusable = registry.admit(both);
    check(reusable.status == moe_mmid_lease_status::ACQUIRED, "pre-submit destruction did not release owners");
    check(reusable.bundle.mark_possible_submit() && reusable.bundle.quarantined(), "possible-submit quarantine absent");
    check(reusable.bundle.terminal_release(), "drained quarantine did not terminally release");

#ifndef MMID_TSAN_BUILD
    g_fail_heap_allocations.store(true);
    auto oom = registry.admit(both);
    g_fail_heap_allocations.store(false);
    check(oom.status == moe_mmid_lease_status::ACQUIRED && oom.bundle.valid(),
          "fail-new admission allocated or rejected preassembled authority");
    check(oom.bundle.terminal_release(), "fail-new bundle release failed");
    auto after_oom = registry.admit(both);
    check(after_oom.status == moe_mmid_lease_status::ACQUIRED, "fail-new admission mutated owner slots");
    check(after_oom.bundle.terminal_release(), "post-fail-new bundle release failed");
#endif
}

static void registry_destroyed_quarantine_recovery() {
    const moe_mmid_model_token token{ 66, 67, 68 };
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 805, 0, { owner_plan(0, 806), owner_plan(1, 807, true) }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED, "quarantine context publication failed");
    auto ready = std::make_shared<std::atomic<bool>>(false);
    auto request = admission_request(token, 805, { { 0, 806 }, { 1, 807 } }, 901, ready);
    auto admitted = registry.admit(request);
    check(admitted.status == moe_mmid_lease_status::ACQUIRED && admitted.bundle.mark_possible_submit(),
          "quarantine admission failed");
    auto copied_lease = admitted.bundle.owner_leases()[0];
    check(copied_lease.terminal_release(copied_lease.queue_cookie(), copied_lease.generation()) ==
              moe_mmid_release_status::WRONG_EPOCH,
          "copied lease bypassed registry-owned quarantine");
    check(!admitted.bundle.terminal_release(), "premature terminal release recycled submitted slots");
    check(!registry.retire(token, 805), "retirement discarded quarantined bundle");
    check(registry.recover_quarantined(token, 805, false) == 0, "unready quarantine recovered");
    ready->store(true);
    std::atomic<size_t> recovered{ 0 };
    std::atomic<bool> terminal_released{ false };
    std::thread recover_thread([&] { recovered.store(registry.recover_quarantined(token, 805, false)); });
    std::thread release_thread([&] { terminal_released.store(admitted.bundle.terminal_release()); });
    recover_thread.join(); release_thread.join();
    check((recovered.load() == 2 && !terminal_released.load()) ||
              (recovered.load() == 0 && terminal_released.load()),
          "release/recover race partially transitioned multi-owner bundle");
    check(registry.retire(token, 805), "context did not retire after atomic quarantine recovery race");
}

static void registry_atomic_concurrency_multi_owner() {
    const moe_mmid_model_token token{ 71, 72, 73 };
    std::atomic<int> destroyed{ 0 };
    moe_mmid_workspace_registry registry;
    auto allocator = [&](bool host, int device, size_t bytes, size_t) {
        return fake_blob(host, device, bytes, &destroyed);
    };
    check(registry.materialize(token, 810, 0, { owner_plan(0, 811), owner_plan(1, 812, true) }, allocator) ==
              moe_mmid_materialize_status::PUBLISHED,
          "concurrent atomic context publication failed");
    auto request = admission_request(token, 810, { { 0, 811 }, { 1, 812 } });
    std::atomic<int> ready{ 0 }, acquired{ 0 }, busy{ 0 };
    std::atomic<bool> go{ false };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            auto local_request = request;
            if (i & 1) std::reverse(local_request.owners.begin(), local_request.owners.end());
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            auto result = registry.admit(local_request);
            if (result.status == moe_mmid_lease_status::ACQUIRED) acquired.fetch_add(1);
            if (result.status == moe_mmid_lease_status::BUSY) busy.fetch_add(1);
            while (acquired.load() + busy.load() < 8) std::this_thread::yield();
        });
    }
    while (ready.load() != 8) std::this_thread::yield();
    go.store(true);
    for (auto & thread : threads) thread.join();
    check(acquired.load() == 2 && busy.load() == 6, "multi-owner admission was not atomic at fixed depth");
    auto reusable = registry.admit(request);
    check(reusable.status == moe_mmid_lease_status::ACQUIRED, "concurrent bundle destruction did not release");
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
        registry_component_max_constituent_coverage();
        registry_atomic_bundle_authority();
        common_direct_authority_plan_queue_and_lifetime();
        model_scoped_replacement_and_retired_recovery();
        changed_replacement_terminal_release_gc();
        registry_bundle_rollback_move_oom_and_quarantine();
        registry_destroyed_quarantine_recovery();
        registry_atomic_concurrency_multi_owner();
        std::cout << "moe-mmid-workspace-plan: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
