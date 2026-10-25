// Device test of moe_stream_ring (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// 1. Reserve 8 slots of 1 MiB, use the first region's 4 slots, flip, use the
//    second region's 4 slots, flip, and check that the next slot handed out is
//    the first region's first slot again, gated on the first boundary's event,
//    resolving on the device with the bytes the first fill wrote. Each slot is
//    its own allocation (llama.cpp-am87): no two overlap, and each has its own
//    registry row while held. Release and check every slot's registry row and
//    the raw device bytes are gone.
//    The first region's slots are acquired as AOS and SOA in turn, and each
//    resolves with the layout it was acquired for (lead ruling c-o9uw-1).
// 2. Geometries that are not two equal aligned regions (one slot, an odd slot
//    count, short or surplus bytes, a region_slots other than n_slots / 2, an
//    unaligned slot_bytes) are refused before allocating.
// 3. The region gate is a device dependency: a copy chained on it does not run
//    while the event that ended the region's previous use is incomplete, and
//    acquire_slot(layout) hands the slot out regardless (it never consults an
//    event).
// 4. A WEIGHT zone whose free room is scattered holes gives the ring
//    (llama.cpp-am87). The zone is filled, then every other one of its first
//    fillers is freed: eight holes, each one slot wide, none holding two, and
//    one allocation of the whole ring refused. One reserve() call takes the
//    ring, every slot inside a hole. With room for only three slots reserve()
//    refuses, short by the five slots still to allocate, and keeps the three
//    (no slot of room is left); once the other holes are free the next call
//    completes the ring. A partial reservation of one geometry is given back
//    before a reserve() of another. Each release() gives the holes back. This
//    case reserves a small arena when the cache has none, so it runs last.
//
// A test may wait on events; the ring itself never does, and nothing here
// status-queries an event.

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../moe-stream-ring.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <sycl/sycl.hpp>
#include <thread>
#include <utility>
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

// Whether the reservation came from the WEIGHT zone (sub-allocation). The
// held-bytes ledger does not see zone sub-allocations, so its checks hold only
// on the raw path; on the zone path the registry checks carry the release proof.
static bool g_zone_managed = false;

static bool is_slot_aligned(const void * ptr) {
    return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % MOE_STREAM_RING_SLOT_ALIGN == 0;
}

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

    std::vector<void *> slot_ptrs;
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
            const ggml_layout_mode want     = i % 2 == 0 ? GGML_LAYOUT_AOS : GGML_LAYOUT_SOA;
            mem_handle             slot     = ring.acquire_slot(want);
            const auto             resolved = slot.resolve(device);
            CHECK(resolved.ptr != nullptr && resolved.on_device, "region 0 slot %u resolves on device", i);
            CHECK(resolved.layout == want,
                  "region 0 slot %u resolves with the layout it was acquired for (%d, want %d)", i,
                  static_cast<int>(resolved.layout), static_cast<int>(want));
            CHECK(is_slot_aligned(resolved.ptr), "region 0 slot %u starts at a multiple of %zu (got %p)", i,
                  MOE_STREAM_RING_SLOT_ALIGN, resolved.ptr);
            first_fill = mem_fill_async(slot, 0x10 + static_cast<int>(i), kSlot, q, { ring.current_region_gate() });
            first_ptrs.push_back(resolved.ptr);
            first_region.push_back(slot);
        }
        slot_ptrs = first_ptrs;

        alloc_metadata md{};
        const bool     zone_managed = first_region.front().owner_metadata(&md) && md.zone_managed;
        g_zone_managed              = zone_managed;
        std::printf("INFO: reservation is %s\n", zone_managed ? "a WEIGHT-zone sub-allocation" : "a raw device row");
        if (!zone_managed) {
            // A raw row must show up in the held-bytes ledger while held.
            const size_t held_during = unified_cache_raw_device_held_bytes(device, nullptr);
            CHECK(held_during >= held_before + geo.bytes,
                  "the raw reservation is in the held-bytes ledger (before %zu, during %zu)", held_before, held_during);
        }
        CHECK(!ring.acquire_slot(GGML_LAYOUT_AOS).valid(), "an exhausted region hands out an invalid handle");
        CHECK(ring.slots_free() == 0, "an exhausted region has no free slots (got %u)", ring.slots_free());
        ring.epoch_boundary(first_fill);

        // Region 1: the other four slots, each its own allocation, so each
        // overlaps no slot handed out before it.
        std::vector<mem_handle> second_region;
        sycl::event             second_fill;
        for (uint32_t i = 0; i < kRegion; ++i) {
            mem_handle      slot     = ring.acquire_slot(GGML_LAYOUT_AOS);
            const auto      resolved = slot.resolve(device);
            const uintptr_t begin    = reinterpret_cast<uintptr_t>(resolved.ptr);
            bool            apart    = resolved.ptr != nullptr;
            for (void * other : slot_ptrs) {
                const uintptr_t o = reinterpret_cast<uintptr_t>(other);
                apart             = apart && (begin + kSlot <= o || o + kSlot <= begin);
            }
            CHECK(is_slot_aligned(resolved.ptr), "region 1 slot %u starts at a multiple of %zu (got %p)", i,
                  MOE_STREAM_RING_SLOT_ALIGN, resolved.ptr);
            CHECK(resolved.ptr != nullptr && resolved.on_device && apart,
                  "region 1 slot %u resolves on device and overlaps none of the %zu slots before it (got %p)", i,
                  slot_ptrs.size(), resolved.ptr);
            second_fill = mem_fill_async(slot, 0x20 + static_cast<int>(i), kSlot, q, { ring.current_region_gate() });
            slot_ptrs.push_back(resolved.ptr);
            second_region.push_back(slot);
        }
        ring.epoch_boundary(second_fill);

        // Positive control for the release checks below, on either path: each
        // slot's registry row exists while the ring holds it.
        size_t rows_held = 0;
        for (void * ptr : slot_ptrs) {
            rows_held += allocation_registry_test_contains(ptr) ? 1 : 0;
        }
        CHECK(rows_held == kSlots, "each of the %u slots has its own registry row while held (%zu found)", kSlots,
              rows_held);

        // Region 0 again: its gate is the first boundary's event, and the slot
        // comes back without the ring asking whether that event completed.
        const sycl::event gate = ring.current_region_gate();
        CHECK(gate == first_fill, "after two boundaries the current region's gate is the first boundary's event");
        mem_handle again    = ring.acquire_slot(GGML_LAYOUT_AOS);
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
        CHECK(!ring.acquire_slot(GGML_LAYOUT_AOS).valid(), "a released ring hands out an invalid handle");
    }

    // release() handed the slots to the retained-handle drain; let it run.
    CHECK(drain_retained_handles(true), "the retained-handle drain completes");
    size_t rows_left = 0;
    for (void * ptr : slot_ptrs) {
        rows_left += allocation_registry_test_contains(ptr) ? 1 : 0;
    }
    CHECK(rows_left == 0, "every slot's registry row is gone after release (%zu left)", rows_left);
    if (g_zone_managed) {
        std::printf("INFO: held-bytes check skipped: a zone sub-allocation is outside that ledger\n");
    } else {
        const size_t held_after = unified_cache_raw_device_held_bytes(device, nullptr);
        CHECK(held_after == held_before,
              "raw device held bytes return to their pre-reserve value (before %zu, after %zu)", held_before,
              held_after);
    }
}

// Case 2: geometries that cannot form two regions are refused before allocating.
static void test_refused_geometries(int device) {
    constexpr size_t kSlot = 1024 * 1024;
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 1, kSlot)), "reserve refuses n_slots = 1");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused n_slots = 1 leaves the ring unreserved");
    }
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 8, 8 * kSlot - 1)),
              "reserve refuses bytes < n_slots x slot_bytes");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused short geometry leaves the ring unreserved");
    }
    {
        moe_stream_ring          ring;
        moe_stream_ring_geometry geo = make_geometry(kSlot, 8, 8 * kSlot);
        geo.region_slots             = 3;
        CHECK(!ring.reserve(device, geo), "reserve refuses region_slots != n_slots / 2");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused region_slots mismatch leaves the ring unreserved");
    }
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 7, 7 * kSlot)), "reserve refuses an odd n_slots");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused odd n_slots leaves the ring unreserved");
    }
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot, 8, 8 * kSlot + 1)),
              "reserve refuses bytes > n_slots x slot_bytes");
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused surplus-bytes geometry leaves the ring unreserved");
    }
    {
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, make_geometry(kSlot + 1, 8, 8 * (kSlot + 1))),
              "reserve refuses a slot_bytes that is not a multiple of %zu", MOE_STREAM_RING_SLOT_ALIGN);
        CHECK(ring.geometry().n_slots == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a refused unaligned slot_bytes leaves the ring unreserved");
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

    mem_handle first_use = ring.acquire_slot(GGML_LAYOUT_AOS);
    void *     slot_ptr  = first_use.resolve(device).ptr;
    CHECK(slot_ptr != nullptr && allocation_registry_test_contains(slot_ptr),
          "the gate case's first slot has a registry row while held");
    sycl::event blocker = blocker_queue.submit([&](sycl::handler & h) {
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

    mem_handle slot = ring.acquire_slot(GGML_LAYOUT_AOS);
    CHECK(slot.valid(), "acquire_slot(layout) hands the slot out while the gate is still pending");
    const sycl::event gate = ring.current_region_gate();

    // The release is NOT sequenced behind the main thread: this thread releases
    // on a timer. A host_task submission can block on an in-flight dependency
    // (presumably the gated fill on the same in-order queue), so a release
    // placed on the submitting thread after that submit would wait on itself.
    // Measured on the B50 on 2026-10-21: in 1 of 3 runs the submit returned
    // 1.8 ms after the release fired. That is consistent with the 30 s stall
    // (then the SYCL watchdog) seen before this thread existed, which was not
    // reproduced with stamps. done.wait() returned after every release in all
    // 3 runs, which argues against a runtime deadlock.
    //
    // The INFO lines below are stamped by this process, in microseconds since
    // the case began, and flushed as they print. A submit stamp later than the
    // releaser's is that blocking path, not a ring defect.
    const auto t0    = std::chrono::steady_clock::now();
    auto       stamp = [&](const char * what) {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0);
        std::printf("INFO: gate case +%lld us: %s\n", static_cast<long long>(us.count()), what);
        std::fflush(stdout);
    };
    std::atomic<bool> submitted{ false };
    std::atomic<bool> ran_early{ false };
    std::atomic<bool> submitted_before_release{ false };
    std::thread       releaser([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        ran_early.store(copy_done.load());
        submitted_before_release.store(submitted.load());
        stamp("releaser fires");
        release_blocker.store(true);
    });

    sycl::event copy = mem_fill_async(slot, 0x5A, kSlot, q, { gate });
    stamp("mem_fill_async returned");
    sycl::event done = q.submit([&](sycl::handler & h) {
        h.depends_on(copy);
        h.host_task([&]() { copy_done.store(true); });
    });
    stamp("host_task submit returned");

    submitted.store(true);
    done.wait();
    stamp("done.wait() returned");
    releaser.join();
    std::printf("INFO: gate case: submission %s the release\n",
                submitted_before_release.load() ? "finished before" : "had NOT finished at");
    CHECK(!ran_early.load(), "a copy chained on the gate does not run while the gate is pending");
    CHECK(copy_done.load(), "the chained copy runs once the gate completes");

    blocker_queue.wait();
    q.wait();
    first_use = mem_handle();
    slot      = mem_handle();
    ring.release();
    // Region 1 was never gated: its release is the has_gate == false branch.
    CHECK(drain_retained_handles(true), "the gate case's retained-handle drain completes");
    stamp("retained-handle drain completed");
    CHECK(!allocation_registry_test_contains(slot_ptr), "the gate case's first slot has no registry row after release");
}

// Holds `bytes` of the WEIGHT zone as an owned allocation, or an invalid handle when the zone refuses.
static mem_handle hold_weight_bytes(int device, size_t bytes) {
    allocation_result allocation =
        unified_cache_zone_allocate_owner(device, vram_zone_id::WEIGHT, bytes, MOE_STREAM_RING_SLOT_ALIGN);
    return allocation ? mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS) : mem_handle();
}

// The WEIGHT zone's free bytes, or SIZE_MAX when its figures are not available (no arena).
static size_t weight_zone_available(unified_cache * cache) {
    size_t capacity  = 0;
    size_t used      = 0;
    size_t available = 0;
    size_t largest   = 0;
    return cache->zone_free_figures(vram_zone_id::WEIGHT, capacity, used, available, largest) ? available : SIZE_MAX;
}

// `bytes` rounded down to the slot alignment, and at least one alignment unit.
static size_t align_down_one_unit(size_t bytes) {
    return std::max(MOE_STREAM_RING_SLOT_ALIGN, bytes / MOE_STREAM_RING_SLOT_ALIGN * MOE_STREAM_RING_SLOT_ALIGN);
}

// Takes the rest of the WEIGHT zone: each filler is the largest free block the zone reports, rounded down to the
// alignment and halved while the zone refuses it, until not even one alignment unit is given.
static void fill_weight_zone(unified_cache * cache, int device, std::vector<mem_handle> & fillers) {
    for (int guard = 0; guard < 4096; ++guard) {
        size_t capacity = 0;
        size_t used     = 0;
        size_t avail    = 0;
        size_t largest  = 0;
        if (!cache->zone_free_figures(vram_zone_id::WEIGHT, capacity, used, avail, largest)) {
            return;
        }
        size_t     want   = align_down_one_unit(largest);
        mem_handle filler = hold_weight_bytes(device, want);
        while (!filler.valid() && want > MOE_STREAM_RING_SLOT_ALIGN) {
            want   = align_down_one_unit(want / 2);
            filler = hold_weight_bytes(device, want);
        }
        if (!filler.valid()) {
            return;
        }
        fillers.push_back(std::move(filler));
    }
}

struct byte_range {
    uintptr_t begin = 0;
    uintptr_t end   = 0;
};

// Sorted, with touching ranges merged: two freed fillers that sat side by side are one free block.
static std::vector<byte_range> merge_ranges(std::vector<byte_range> ranges) {
    std::sort(ranges.begin(), ranges.end(),
              [](const byte_range & a, const byte_range & b) { return a.begin < b.begin; });
    std::vector<byte_range> merged;
    for (const byte_range & r : ranges) {
        if (!merged.empty() && r.begin <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, r.end);
        } else {
            merged.push_back(r);
        }
    }
    return merged;
}

static bool inside_one(const std::vector<byte_range> & holes, const void * ptr, size_t bytes) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    for (const byte_range & h : holes) {
        if (begin >= h.begin && begin + bytes <= h.end) {
            return true;
        }
    }
    return false;
}

static bool disjoint(const std::vector<byte_range> & ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                return false;
            }
        }
    }
    return true;
}

// Acquires both regions' slots of a reserved ring and checks each is on the device inside one of `holes`, and that no
// two overlap; then releases the ring.
static void use_and_release(moe_stream_ring &               ring,
                            int                             device,
                            const std::vector<byte_range> & holes,
                            const char *                    what) {
    const moe_stream_ring_geometry geo = ring.geometry();
    std::vector<mem_handle>        held;
    std::vector<byte_range>        slots;
    bool                           fits = true;
    for (int region = 0; region < 2; ++region) {
        for (uint32_t i = 0; i < geo.region_slots; ++i) {
            mem_handle      slot     = ring.acquire_slot(GGML_LAYOUT_AOS);
            const auto      resolved = slot.resolve(device);
            const uintptr_t begin    = reinterpret_cast<uintptr_t>(resolved.ptr);
            fits                     = fits && resolved.ptr != nullptr && resolved.on_device &&
                   inside_one(holes, resolved.ptr, geo.slot_bytes);
            slots.push_back({ begin, begin + geo.slot_bytes });
            held.push_back(std::move(slot));
        }
        ring.epoch_boundary(sycl::event());
    }
    CHECK(fits, "%s: each of the %u slots is on the device inside one hole", what, geo.n_slots);
    CHECK(disjoint(slots), "%s: no two slots overlap", what);
    held.clear();
    ring.release();
}

// Case 4: one reserve() takes the ring from scattered holes; a partial reservation holds its slots, reports the slots
// still to allocate and completes at the next call; another geometry gives the partial slots back first.
static void test_reserve_completes_across_two_calls_from_scattered_room(sycl::queue & q, int device) {
    constexpr size_t   kSlot  = 1024 * 1024;
    constexpr uint32_t kSlots = 8;
    constexpr uint32_t kLeft  = 3;  // the holes the partial sub-cases leave free
    // A hole is a slot plus one alignment unit: it holds one slot, never two.
    constexpr size_t   kHole  = kSlot + MOE_STREAM_RING_SLOT_ALIGN;
    const auto         geo    = make_geometry(kSlot, kSlots, kSlots * kSlot);

    unified_cache * cache = get_unified_cache_for_device(device);
    if (cache != nullptr && !cache->arena_active()) {
        constexpr size_t kArena = 64ull * 1024 * 1024;
        (void) cache->arena_reserve(q, kArena, kArena, kArena, 0, 0, 0, 0);
    }
    const bool zoned = cache != nullptr && weight_zone_available(cache) != SIZE_MAX;
    CHECK(zoned, "device %d has a WEIGHT zone for the scattered-room case", device);
    if (!zoned) {
        return;
    }
    const size_t available_empty = weight_zone_available(cache);

    // The hole fillers first, so they sit side by side in the zone's free room, then the rest of the zone. The last
    // hole filler is never freed: it keeps the last hole apart from whatever follows.
    std::vector<mem_handle> hole_fillers;
    bool                    all_given = true;
    for (uint32_t i = 0; i < 2 * kSlots + 1; ++i) {
        hole_fillers.push_back(hold_weight_bytes(device, kHole));
        all_given = all_given && hole_fillers.back().valid();
    }
    CHECK(all_given, "the WEIGHT zone gives %u fillers of %zu bytes", 2 * kSlots + 1, kHole);
    if (!all_given) {
        return;
    }
    std::vector<mem_handle> rest;
    fill_weight_zone(cache, device, rest);
    CHECK(!hold_weight_bytes(device, MOE_STREAM_RING_SLOT_ALIGN).valid(),
          "the WEIGHT zone is full after the fill (%zu more fillers)", rest.size());

    std::vector<byte_range> holes;
    for (uint32_t i = 0; i < kSlots; ++i) {
        mem_handle &    filler = hole_fillers[2 * i];
        const uintptr_t begin  = reinterpret_cast<uintptr_t>(filler.resolve(device).ptr);
        holes.push_back({ begin, begin + kHole });
        filler = mem_handle();
    }
    CHECK(drain_retained_handles(true), "the scattered-room case's hole drain completes");
    holes                        = merge_ranges(holes);
    const size_t available_holes = weight_zone_available(cache);
    std::printf("INFO: %u holes of %zu bytes in %zu free blocks; WEIGHT zone available %zu (empty %zu)\n", kSlots,
                kHole, holes.size(), available_holes, available_empty);
    CHECK(available_holes >= geo.bytes, "the holes together hold the ring (%zu >= %zu)", available_holes, geo.bytes);
    CHECK(!hold_weight_bytes(device, geo.bytes).valid(), "one allocation of the ring's %zu bytes is refused",
          geo.bytes);

    // Sub-case 1: one call, every slot in a hole.
    {
        moe_stream_ring ring;
        size_t          short_by = SIZE_MAX;
        const bool      reserved = ring.reserve(device, geo, &short_by);
        CHECK(reserved && short_by == 0, "one reserve() takes the %u-slot ring from %zu scattered holes (short %zu)",
              kSlots, holes.size(), short_by);
        if (reserved) {
            use_and_release(ring, device, holes, "one call");
        }
    }
    CHECK(drain_retained_handles(true), "the one-call ring's drain completes");
    CHECK(weight_zone_available(cache) == available_holes, "the one-call ring gives the holes back (%zu, want %zu)",
          weight_zone_available(cache), available_holes);

    // Sub-case 2: room for kLeft slots, then for the rest.
    {
        std::vector<mem_handle> refill;
        bool                    refilled = true;
        for (uint32_t i = 0; i < kSlots - kLeft; ++i) {
            refill.push_back(hold_weight_bytes(device, kHole));
            refilled = refilled && refill.back().valid();
        }
        CHECK(refilled, "%u fillers take all but %u holes back", kSlots - kLeft, kLeft);
        moe_stream_ring ring;
        size_t          short_by = 0;
        const bool      first    = ring.reserve(device, geo, &short_by);
        const size_t    want     = static_cast<size_t>(kSlots - kLeft) * kSlot;
        CHECK(!first && short_by == want,
              "with room for %u slots reserve() refuses, short by the %u slots left to allocate (%zu, want %zu)", kLeft,
              kSlots - kLeft, short_by, want);
        CHECK(ring.slots_free() == 0 && !ring.acquire_slot(GGML_LAYOUT_AOS).valid(),
              "a partial reservation hands out no slot (%u free)", ring.slots_free());
        CHECK(!hold_weight_bytes(device, kSlot).valid(),
              "the refused reserve() keeps its %u slots: the zone has no slot of room left", kLeft);
        refill.clear();
        CHECK(drain_retained_handles(true), "the refill's drain completes");
        const bool second = ring.reserve(device, geo, &short_by);
        CHECK(second && short_by == 0, "with the other holes free the next reserve() completes the ring (short %zu)",
              short_by);
        if (second) {
            use_and_release(ring, device, holes, "two calls");
        }
    }
    CHECK(drain_retained_handles(true), "the two-call ring's drain completes");
    CHECK(weight_zone_available(cache) == available_holes, "the two-call ring gives the holes back (%zu, want %zu)",
          weight_zone_available(cache), available_holes);

    // Sub-case 3: the partial slots of one geometry go back before a reserve() of another takes its own.
    {
        std::vector<mem_handle> refill;
        bool                    refilled = true;
        for (uint32_t i = 0; i < kSlots - kLeft; ++i) {
            refill.push_back(hold_weight_bytes(device, kHole));
            refilled = refilled && refill.back().valid();
        }
        CHECK(refilled, "%u fillers take all but %u holes back again, for the other geometry", kSlots - kLeft, kLeft);
        moe_stream_ring ring;
        CHECK(!ring.reserve(device, geo), "with room for %u slots the %u-slot ring is refused again", kLeft, kSlots);
        const auto small = make_geometry(kSlot, 2, 2 * kSlot);
        const bool other = ring.reserve(device, small);
        CHECK(other && ring.geometry().n_slots == 2,
              "a reserve() of 2 slots gives the %u partial slots back first and takes two of their holes", kLeft);
        if (other) {
            use_and_release(ring, device, holes, "another geometry");
        }
        refill.clear();
    }
    CHECK(drain_retained_handles(true), "the other geometry's drain completes");
    CHECK(weight_zone_available(cache) == available_holes, "the other geometry gives the holes back (%zu, want %zu)",
          weight_zone_available(cache), available_holes);

    hole_fillers.clear();
    rest.clear();
    CHECK(drain_retained_handles(true), "the scattered-room case's final drain completes");
    CHECK(weight_zone_available(cache) == available_empty,
          "the WEIGHT zone's room is what it was before the case (%zu, want %zu)", weight_zone_available(cache),
          available_empty);
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
    if (g_zone_managed) {
        std::printf("INFO: final held-bytes check skipped: a zone sub-allocation is outside that ledger\n");
    } else {
        const size_t held_end = unified_cache_raw_device_held_bytes(device, nullptr);
        CHECK(held_end == held_before, "raw device held bytes end at their starting value (before %zu, end %zu)",
              held_before, held_end);
    }

    // Last: it may reserve an arena, which the held-bytes checks above do not expect.
    test_reserve_completes_across_two_calls_from_scattered_room(q, device);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
