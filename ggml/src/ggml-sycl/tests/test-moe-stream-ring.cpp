// Device test of moe_stream_ring (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// Reserve 8 slots of 1 MiB, use the first region's 4 slots, flip, use the
// second region's 4 slots, flip, and check that the next slot handed out is the
// first region's first slot again, resolving on the device with the bytes the
// first fill wrote. Then release and check the raw device bytes the cache holds
// return to their value before the reservation.
//
// A test may wait on events; the ring itself never does.

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../moe-stream-ring.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <sycl/sycl.hpp>
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

    constexpr size_t         kSlot   = 1024 * 1024;
    constexpr uint32_t       kSlots  = 8;
    constexpr uint32_t       kRegion = kSlots / 2;
    moe_stream_ring_geometry geo;
    geo.slot_bytes = kSlot;
    geo.n_slots    = kSlots;
    geo.bytes      = kSlots * kSlot;

    bool zone_managed = false;
    {
        moe_stream_ring ring;
        const bool      reserved = ring.reserve(device, geo);
        CHECK(reserved, "reserve(%u slots x %zu bytes) on device %d", kSlots, kSlot, device);
        if (!reserved) {
            std::printf("FAILED: %d case(s)\n", g_failures);
            return 1;
        }
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
            first_fill = mem_fill_async(slot, 0x10 + static_cast<int>(i), kSlot, q);
            first_ptrs.push_back(resolved.ptr);
            first_region.push_back(slot);
        }
        alloc_metadata md{};
        if (first_region.front().owner_metadata(&md)) {
            zone_managed = md.zone_managed;
        }
        std::printf("INFO: reservation is %s\n", zone_managed ? "a WEIGHT-zone sub-allocation" : "a raw device row");
        if (!zone_managed) {
            // Positive control for the release check below: a raw row must show
            // up in the ledger while the ring holds it.
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
            second_fill = mem_fill_async(slot, 0x20 + static_cast<int>(i), kSlot, q);
            second_region.push_back(slot);
        }
        ring.epoch_boundary(second_fill);

        // The ring only queries; the test makes region 0's event complete.
        first_fill.wait();
        mem_handle again    = ring.acquire_slot();
        const auto resolved = again.resolve(device);
        CHECK(resolved.ptr != nullptr && resolved.on_device && resolved.ptr == first_ptrs[0],
              "after two boundaries the next slot is region 0's first slot, on device (got %p, want %p)", resolved.ptr,
              first_ptrs[0]);
        std::vector<uint8_t> host(kSlot, 0);
        if (resolved.ptr != nullptr) {
            mem_copy_ptr_async(host.data(), resolved.ptr, kSlot, q).wait();
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
    const size_t held_after = unified_cache_raw_device_held_bytes(device, nullptr);
    CHECK(held_after == held_before, "raw device held bytes return to their pre-reserve value (before %zu, after %zu)",
          held_before, held_after);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
