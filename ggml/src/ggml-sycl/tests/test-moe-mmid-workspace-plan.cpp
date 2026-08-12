#include "moe-mmid-workspace.hpp"

#include <atomic>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl;

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
        broadcast_and_occurrences();
        maxima_and_overflow();
        pool_identity_generation_and_terminal_release();
        concurrent_depth_busy_and_reuse();
        std::cout << "moe-mmid-workspace-plan: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
