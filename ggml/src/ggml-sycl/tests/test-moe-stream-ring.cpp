// Device test of moe_stream_ring (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// 1. Reserve 8 slots of 1 MiB, use the first region's 4 slots, flip, use the
//    second region's 4 slots, flip, and check that the next slot handed out is
//    the first region's first slot again, gated on the first boundary's event,
//    resolving on the device with the bytes the first fill wrote. Release and
//    check the allocation's registry row and the raw device bytes are gone.
// 2. Geometries that cannot form two regions are refused before allocating.
// 3. The region gate is a device dependency: a copy chained on it does not run
//    while the event that ended the region's previous use is incomplete, and
//    acquire_slot() hands the slot out regardless (it never consults an event).
//
// A test may wait on events; the ring itself never does, and nothing here
// status-queries an event.

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../moe-stream-ring.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

using namespace ggml_sycl;

static int g_failures = 0;

#define CHECK(cond, ...)                       \
    do {                                       \
        if (cond) {                            \
            std::printf("PASS: " __VA_ARGS__); \
        } else {                               \
            std::printf("FAIL: " __VA_ARGS__); \
            ++g_failures;                      \
        }                                      \
        std::printf("\n");                     \
    } while (0)

static moe_stream_ring_geometry make_geometry(size_t slot_bytes, uint32_t n_slots, size_t bytes) {
    moe_stream_ring_geometry geo;
    geo.slot_bytes   = slot_bytes;
    geo.n_slots      = n_slots;
    geo.bytes        = bytes;
    geo.region_slots = n_slots / 2;
    return geo;
}

// Case 1: two regions, reuse of the first region's first slot, release.
static void test_regions_and_release(sycl::queue & q, int device, size_t held_before) {
    constexpr size_t   kSlot   = 1024 * 1024;
    constexpr uint32_t kSlots  = 8;
    constexpr uint32_t kRegion = kSlots / 2;
    const auto         geo     = make_geometry(kSlot, kSlots, kSlots * kSlot);

    void * base_ptr = nullptr;
    {
        moe_stream_ring ring;
        const bool      reserved = ring.reserve(device, geo);
        CHECK(reserved, "reserve(%u slots x %zu bytes) on device %d", kSlots, kSlot, device);
        if (!reserved) {
            return;
        }
        CHECK(ring.geometry().n_slots == kSlots && ring.geometry().region_slots == kRegion &&
                  ring.geometry().bytes == geo.bytes,
              "the reserved geometry is %u slots, %u per region, %zu bytes", kSlots, kRegion, geo.bytes);
        CHECK(ring.slots_free() == kRegion, "a fresh ring's current region has %u free slots (got %u)", kRegion,
              ring.slots_free());

        // Region 0: four slots, each filled with its own byte.
        std::vector<mem_handle> first_region;
        std::vector<void *>     first_ptrs;
        sycl::event             first_fill;
        for (uint32_t i = 0; i < kRegion; ++i) {
            mem_handle slot     = ring.acquire_slot();
            const auto resolved = slot.resolve(device);
            CHECK(resolved.ptr != nullptr && resolved.on_device, "region 0 slot %u resolves on device", i);
            first_fill = mem_fill_async(slot, 0x10 + static_cast<int>(i), kSlot, q, { ring.current_region_gate() });
            first_ptrs.push_back(resolved.ptr);
            first_region.push_back(slot);
        }
        base_ptr = first_ptrs[0];

        // Positive control for the release checks below, on either path: the
        // reservation's registry row exists while the ring holds it.
        CHECK(allocation_registry_test_contains(base_ptr), "the reservation's registry row exists while held");
        alloc_metadata md{};
        const bool     zone_managed = first_region.front().owner_metadata(&md) && md.zone_managed;
        std::printf("INFO: reservation is %s\n", zone_managed ? "a WEIGHT-zone sub-allocation" : "a raw device row");
        if (!zone_managed) {
            // A raw row must show up in the held-bytes ledger while held.
            const size_t held_during = unified_cache_raw_device_held_bytes(device, nullptr);
            CHECK(held_during >= held_before + geo.bytes,
                  "the raw reservation is in the held-bytes ledger (before %zu, during %zu)", held_before, held_during);
        }
        CHECK(!ring.acquire_slot().valid(), "an exhausted region hands out an invalid handle");
        CHECK(ring.slots_free() == 0, "an exhausted region has no free slots (got %u)", ring.slots_free());
        ring.epoch_boundary(first_fill);

        // Region 1: the other four slots.
        std::vector<mem_handle> second_region;
        sycl::event             second_fill;
        for (uint32_t i = 0; i < kRegion; ++i) {
            mem_handle slot     = ring.acquire_slot();
            const auto resolved = slot.resolve(device);
            const auto offset   = static_cast<const char *>(resolved.ptr) - static_cast<const char *>(first_ptrs[0]);
            CHECK(resolved.ptr != nullptr && resolved.on_device &&
                      offset == static_cast<std::ptrdiff_t>((kRegion + i) * kSlot),
                  "region 1 slot %u resolves on device at slot offset %u (got byte offset %td)", i, kRegion + i,
                  offset);
            second_fill = mem_fill_async(slot, 0x20 + static_cast<int>(i), kSlot, q, { ring.current_region_gate() });
            second_region.push_back(slot);
        }
        ring.epoch_boundary(second_fill);

        // Region 0 again: its gate is the first boundary's event, and the slot
        // comes back without the ring asking whether that event completed.
        const sycl::event gate = ring.current_region_gate();
        CHECK(gate == first_fill, "after two boundaries the current region's gate is the first boundary's event");
        mem_handle again    = ring.acquire_slot();
        const auto resolved = again.resolve(device);
        CHECK(resolved.ptr != nullptr && resolved.on_device && resolved.ptr == first_ptrs[0],
              "after two boundaries the next slot is region 0's first slot, on device (got %p, want %p)", resolved.ptr,
              first_ptrs[0]);
        std::vector<uint8_t> host(kSlot, 0);
        if (resolved.ptr != nullptr) {
            mem_copy_ptr_async(host.data(), resolved.ptr, kSlot, q, { gate }).wait();
        }
        size_t mismatched = 0;
        for (uint8_t b : host) {
            mismatched += b != 0x10 ? 1 : 0;
        }
        CHECK(mismatched == 0, "region 0's first slot still holds the first fill's byte 0x10 (%zu bytes differ)",
              mismatched);

        q.wait();
        again = mem_handle();
        first_region.clear();
        second_region.clear();
        ring.release();
        CHECK(!ring.acquire_slot().valid(), "a released ring hands out an invalid handle");
    }

    // release() handed the backing to the retained-handle drain; let it run.
    CHECK(drain_retained_handles(true), "the retained-handle drain completes");
    CHECK(!allocation_registry_test_contains(base_ptr), "the reservation's registry row is gone after release");
    const size_t held_after = unified_cache_raw_device_held_bytes(device, nullptr);
    CHECK(held_after == held_before, "raw device held bytes return to their pre-reserve value (before %zu, after %zu)",
          held_before, held_after);
}

// Case 2: geometries that cannot form two regions are refused before allocating.
static void test_refused_geometries(int device) {
    constexpr size_t kSlot = 1024 * 1024;
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 1, kSlot)), "reserve refuses n_slots = 1");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot().valid(),
              "a refused n_slots = 1 leaves the ring unreserved");
    }
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 8, 8 * kSlot - 1)),
              "reserve refuses bytes < n_slots x slot_bytes");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot().valid(),
              "a refused short geometry leaves the ring unreserved");
    }
}

// Case 3: the gate orders reuse on the device.
static void test_gate_orders_reuse_on_device(sycl::queue & q, int device) {
    constexpr size_t kSlot = 64 * 1024;
    moe_stream_ring  ring;
    const bool       reserved = ring.reserve(device, make_geometry(kSlot, 4, 4 * kSlot));
    CHECK(reserved, "reserve(4 slots x %zu bytes) for the gate case", kSlot);
    if (!reserved) {
        return;
    }

    // The previous use of region 0 "ends" at a host_task on a separate,
    // non-profiling in-order queue that holds until the test lets it go.
    sycl::queue       blocker_queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    std::atomic<bool> release_blocker{ false };
    std::atomic<bool> copy_done{ false };

    mem_handle  first_use = ring.acquire_slot();
    sycl::event blocker   = blocker_queue.submit([&](sycl::handler & h) {
        h.host_task([&]() {
            while (!release_blocker.load()) {
                std::this_thread::yield();
            }
        });
    });
    ring.epoch_boundary(blocker);        // region 0 gated on the blocker
    ring.epoch_boundary(sycl::event());  // region 1: nothing acquired, no gate stored
    CHECK(ring.current_region_gate() == blocker, "region 0's gate is the blocker after a round trip");
    ring.epoch_boundary(sycl::event());  // region 0: nothing acquired since it became current
    ring.epoch_boundary(sycl::event());
    CHECK(ring.current_region_gate() == blocker,
          "a boundary taken while a region was never acquired from keeps its older gate");

    mem_handle slot = ring.acquire_slot();
    CHECK(slot.valid(), "acquire_slot() hands the slot out while the gate is still pending");
    const sycl::event gate = ring.current_region_gate();
    sycl::event       copy = mem_fill_async(slot, 0x5A, kSlot, q, { gate });
    sycl::event       done = q.submit([&](sycl::handler & h) {
        h.depends_on(copy);
        h.host_task([&]() { copy_done.store(true); });
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool ran_early = copy_done.load();
    release_blocker.store(true);
    done.wait();
    CHECK(!ran_early, "a copy chained on the gate does not run while the gate is pending");
    CHECK(copy_done.load(), "the chained copy runs once the gate completes");

    blocker_queue.wait();
    q.wait();
    first_use = mem_handle();
    slot      = mem_handle();
    ring.release();
}

int main() {
    std::optional<sycl::device> dev_opt = sycl_test_require_gpu("the MoE stream ring");
    if (!dev_opt) {
        return SYCL_TEST_SKIP;
    }
    // The backend owner's device and context, as test-mem-handle-eviction does:
    // a raw gpu_selector queue is not authoritative for logical device 0.
    const auto & sycl_info = ggml_sycl_info();
    if (sycl_info.total_gpu_count <= 0) {
        sycl_test_print_skip("the MoE stream ring backend owner selection");
        return SYCL_TEST_SKIP;
    }
    auto &        owner       = ggml_sycl_get_gpu_device(0);
    sycl::queue & owner_queue = owner.default_queue();
    sycl::queue   q(owner_queue.get_context(), owner_queue.get_device(), sycl::property::queue::in_order{});
    const int     device = ggml_sycl_get_device_id_from_queue(q);
    std::printf("Device: %s (logical %d)\n", q.get_device().get_info<sycl::info::device::name>().c_str(), device);
    if (device != 0) {
        std::printf("FAIL: backend owner queue did not round-trip to logical GPU 0\n");
        return 1;
    }

    // Create the device's cache (and its arena, if any) before the baseline, so
    // the baseline does not move for reasons other than the ring.
    (void) get_unified_cache_for_device(device);
    const size_t held_before = unified_cache_raw_device_held_bytes(device, nullptr);

    test_regions_and_release(q, device, held_before);
    test_refused_geometries(device);
    test_gate_orders_reuse_on_device(q, device);

    CHECK(drain_retained_handles(true), "the final retained-handle drain completes");
    const size_t held_end = unified_cache_raw_device_held_bytes(device, nullptr);
    CHECK(held_end == held_before, "raw device held bytes end at their starting value (before %zu, end %zu)",
          held_before, held_end);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
