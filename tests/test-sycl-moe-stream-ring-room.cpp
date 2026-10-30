// Characterization of the window ring's planned room on a real device cache: n_slots x slot_bytes free in the WEIGHT
// zone funds every slot of the ring, and one grain less funds one slot fewer.
//
// The planner holds the ring's room as exactly n_slots x slot_bytes (hold_moe_stream_ring_room in unified-cache.cpp),
// with no padding. moe_stream_ring::reserve allocates each slot from the WEIGHT zone at MOE_STREAM_RING_SLOT_ALIGN,
// which is the zone allocator's grain, so each slot should take exactly slot_bytes. This test checks that on the
// device, with no model loaded:
//   - with exactly the room free in the WEIGHT zone, as one block, reserve() holds every slot, reports nothing short,
//     and the slots take exactly the room from the zone;
//   - releasing the ring returns every slot's bytes to the zone;
//   - with one grain less free, reserve() holds one slot fewer and reports exactly one slot short.
// The slot is the Qwen3.8 window ring's (921600 bytes, one IQ4_NL 640x2560 matrix). It is not a size class of the
// zone allocator, so the ring's last slot takes the exact-fit path of tlsf_allocator::allocate.
//
// The zone is whatever small arena the test reserves, filled with plain zone allocations until only the room is
// free, so the run costs the arena's 64 MB of device memory.
//
// Usage (GPU; ctest supplies the selector):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-moe-stream-ring-room

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/mem-handle.hpp"
#include "ggml-sycl/moe-stream-ring.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the window ring's room.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache;
using ggml_sycl::vram_zone_id;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t   SLOT_MATRIX_BYTES = 921600;  // one IQ4_NL 640x2560 matrix: the Qwen3.8 window ring's slot
constexpr uint32_t REGION_SLOTS      = 2;
constexpr size_t   GRAIN             = ggml_sycl::tlsf_allocator::block_grain;

// A freed zone block can be held by a retained reference until the drain worker sees it; settle that before a
// figure is read, so a check reads the zone and not the worker's timing.
void settle() {
    (void) ggml_sycl::drain_retained_handles(true);
}

// Fills the WEIGHT zone so that exactly `room` bytes are free, as one block: the room is taken first, every other
// free byte of the zone is filled, and the room is given back. `fill` keeps the fillers. False when the zone cannot be
// brought to that state.
bool leave_free(int device, unified_cache * cache, size_t room, std::vector<ggml_sycl::alloc_owner> * fill) {
    ggml_sycl::allocation_result hole =
        ggml_sycl::unified_cache_zone_allocate_owner(device, vram_zone_id::WEIGHT, room, GRAIN);
    if (!hole) {
        return false;
    }
    while (cache->zone_available(vram_zone_id::WEIGHT) != 0) {
        const size_t want = cache->zone_largest_free(vram_zone_id::WEIGHT) / GRAIN * GRAIN;
        if (want == 0) {
            break;
        }
        ggml_sycl::allocation_result filler =
            ggml_sycl::unified_cache_zone_allocate_owner(device, vram_zone_id::WEIGHT, want, GRAIN);
        if (!filler) {
            break;
        }
        fill->push_back(std::move(filler.owner));
    }
    const bool full = cache->zone_available(vram_zone_id::WEIGHT) == 0;
    (void) hole.owner.reset();
    settle();
    return full;
}

void release_fill(std::vector<ggml_sycl::alloc_owner> * fill) {
    for (auto & owner : *fill) {
        (void) owner.reset();
    }
    fill->clear();
    settle();
}

void test_room_funds_every_slot(int device, unified_cache * cache) {
    printf("the room the planner holds funds every slot of the ring:\n");
    const ggml_sycl::moe_stream_ring_geometry geo =
        ggml_sycl::moe_stream_ring_size(REGION_SLOTS, SLOT_MATRIX_BYTES, SIZE_MAX);
    check(geo.n_slots == 2 * REGION_SLOTS && geo.region_slots == REGION_SLOTS && geo.slot_bytes == SLOT_MATRIX_BYTES &&
              geo.bytes == geo.n_slots * geo.slot_bytes,
          "the ring is two regions of two slots of the matrix's bytes, which are a whole number of grains");
    const size_t room = geo.bytes;  // what hold_moe_stream_ring_room holds for this ring

    std::vector<ggml_sycl::alloc_owner> fill;
    const bool                          ready   = leave_free(device, cache, room, &fill);
    const size_t                        avail   = cache->zone_available(vram_zone_id::WEIGHT);
    const size_t                        largest = cache->zone_largest_free(vram_zone_id::WEIGHT);
    printf("  room=%zu: zone_available=%zu largest_free=%zu fillers=%zu\n", room, avail, largest, fill.size());
    check(ready && avail == room && largest == room, "exactly the room is free in the WEIGHT zone, as one block");
    if (ready && avail == room && largest == room) {
        ggml_sycl::moe_stream_ring ring;
        size_t                     short_bytes = SIZE_MAX;
        const bool                 ok          = ring.reserve(device, geo, &short_bytes);
        const size_t               left        = cache->zone_available(vram_zone_id::WEIGHT);
        printf("  reserve: ok=%d slots_held=%u of %u bytes_short=%zu zone_available=%zu\n", ok ? 1 : 0,
               ring.slots_held(), geo.n_slots, short_bytes, left);
        check(ok && ring.valid() && short_bytes == 0, "reserve() holds every slot with nothing short");
        check(left == 0, "and the slots took exactly the room from the WEIGHT zone");
        ring.release();
        settle();
        check(cache->zone_available(vram_zone_id::WEIGHT) == room,
              "releasing the ring returns every slot's bytes to the zone");
    }
    release_fill(&fill);
}

void test_one_grain_less_funds_one_slot_fewer(int device, unified_cache * cache) {
    printf("one grain less than the room funds one slot fewer:\n");
    const ggml_sycl::moe_stream_ring_geometry geo =
        ggml_sycl::moe_stream_ring_size(REGION_SLOTS, SLOT_MATRIX_BYTES, SIZE_MAX);
    const size_t free_bytes = geo.bytes - GRAIN;

    std::vector<ggml_sycl::alloc_owner> fill;
    const bool                          ready   = leave_free(device, cache, free_bytes, &fill);
    const size_t                        avail   = cache->zone_available(vram_zone_id::WEIGHT);
    const size_t                        largest = cache->zone_largest_free(vram_zone_id::WEIGHT);
    printf("  free=%zu: zone_available=%zu largest_free=%zu fillers=%zu\n", free_bytes, avail, largest, fill.size());
    check(ready && avail == free_bytes && largest == free_bytes,
          "one grain less than the room is free in the WEIGHT zone, as one block");
    if (ready && avail == free_bytes && largest == free_bytes) {
        ggml_sycl::moe_stream_ring ring;
        size_t                     short_bytes = 0;
        const bool                 ok          = ring.reserve(device, geo, &short_bytes);
        printf("  reserve: ok=%d slots_held=%u of %u bytes_short=%zu zone_available=%zu\n", ok ? 1 : 0,
               ring.slots_held(), geo.n_slots, short_bytes, cache->zone_available(vram_zone_id::WEIGHT));
        check(!ok && !ring.valid() && ring.slots_held() == geo.n_slots - 1,
              "reserve() holds one slot fewer and refuses the ring");
        check(short_bytes == geo.slot_bytes, "and reports exactly one slot short");
        ring.release();
        settle();
        check(cache->zone_available(vram_zone_id::WEIGHT) == free_bytes,
              "releasing the partial ring returns its slots' bytes to the zone");
    }
    release_fill(&fill);
}

}  // namespace

int main(int, char ** argv) {
    sycl_test_selector_fallback(argv, "level_zero:1");

    const int      device  = 0;  // in-process index after selector filtering
    ggml_backend_t backend = ggml_backend_sycl_init(device);
    if (backend == nullptr) {
        fprintf(stderr, "ggml_backend_sycl_init(%d) failed; cannot run.\n", device);
        return 1;
    }
    unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device);
    if (cache == nullptr) {
        fprintf(stderr, "no unified cache for device %d; cannot run.\n", device);
        ggml_backend_free(backend);
        return 1;
    }
    sycl::queue * queue = &cache->get_queue();
    // The arena is normally reserved by a model load, which this test does not do. Reserve a small one the way the
    // backend's own staging tests do (test_ensure_weight_zone in ggml-sycl.cpp).
    if (!cache->arena_active()) {
        constexpr size_t arena_bytes = 64u << 20;
        (void) cache->arena_reserve(*queue, arena_bytes, arena_bytes, arena_bytes, 0, 0, 0, 0);
    }
    const size_t ring_bytes = 2 * REGION_SLOTS * SLOT_MATRIX_BYTES;
    printf("setup: arena_active=%d zone_capacity(WEIGHT)=%zu zone_available(WEIGHT)=%zu\n",
           cache->arena_active() ? 1 : 0, cache->zone_capacity(vram_zone_id::WEIGHT),
           cache->zone_available(vram_zone_id::WEIGHT));
    // Without an arena there is no zone to fill, and every check below would pass or fail for reasons unrelated to
    // the room.
    check(cache->arena_active() && cache->zone_available(vram_zone_id::WEIGHT) >= 2 * ring_bytes,
          "the device cache has an arena with room for the ring in the WEIGHT zone");

    if (g_failures == 0) {
        test_room_funds_every_slot(device, cache);
        test_one_grain_less_funds_one_slot_fewer(device, cache);
    }

    ggml_backend_free(backend);

    printf("%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
