// Host test for the zone-exact owner request, unified_cache_zone_allocate_owner(): the caller asks for a block of
// one VRAM zone and accepts nothing else, so the request is served by that zone or refused there.
//
//   full zone   the request is refused without a raw-allocation gate reading its figure (a walk of the whole runtime
//               registry), without a raw device allocation, and with the error a caller reads as "no room":
//               PHYSICAL_ALLOCATION_FAILED, never the control failure direct staging treats as fatal
//   room        the request is served from the zone as an exact, zone-managed block, and no raw gate is read either
//
// The cache is a real unified_cache on the OpenCL CPU device, registered through unified_cache_register_for_queue(),
// so no GPU is touched (the registration pins the selector to the CPU device). Its arena is one chunk, where the
// WEIGHT zone allocates from the allocator it shares with KV. The zone is filled with plain zone-exact allocations
// around one hole of the request's size; the bytes are never written, so the run costs address space, not memory.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build/bin/test-sycl-zone-exact-alloc

#include "ggml.h"
#include "mem-handle.hpp"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"
#include "unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

using namespace ggml_sycl;

namespace {

constexpr size_t MOCK_BUDGET = 320ULL * 1024 * 1024;
constexpr int    MOCK_DEVICE = 0;
constexpr size_t GRAIN       = tlsf_allocator::block_grain;
// One IQ4_NL 640x2560 matrix, a member of the expert triplets an admit asks the WEIGHT zone for.
constexpr size_t REQUEST     = 921600;

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

// A freed zone block can be held by a retained reference until the drain worker sees it; settle that before a
// figure is read, so a check reads the zone and not the worker's timing.
void settle() {
    (void) drain_retained_handles(true);
}

// The mock's environment, set before anything reads it. A 1 MiB compute and runtime arena keep the zone tail small
// enough for the budget.
void set_mock_environment() {
    setenv("GGML_SYCL_UNIFIED_CACHE_MODE", "per_device", 1);
    setenv("GGML_SYCL_COMPUTE_ARENA_MB", "1", 1);
    setenv("GGML_SYCL_RUNTIME_ARENA_MB", "1", 1);
    unsetenv("GGML_SYCL_NONFA_ATTN_SCRATCH_MB");
    unsetenv("GGML_SYCL_STRICT_LEASES");
    set_unified_cache_budget(MOCK_BUDGET);
}

// A mock whose arena is one chunk, or an exit that says the precondition did not hold (VOID, never FAIL).
unified_cache * make_mock(sycl::queue & q) {
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    if (max_alloc / 100 * 95 < MOCK_BUDGET) {
        std::fprintf(stderr, "VOID: max_mem_alloc_size %zu is below the single-chunk precondition\n", max_alloc);
        std::exit(LLAMA_TEST_EXIT_SKIP);
    }
    unified_cache * cache = unified_cache_register_for_queue(MOCK_DEVICE, q);
    if (cache == nullptr || !cache->arena_active() || cache->zone_capacity(vram_zone_id::WEIGHT) == 0) {
        std::fprintf(stderr, "VOID: the mock cache has no WEIGHT arena, so no zone-exact request can be measured\n");
        std::exit(LLAMA_TEST_EXIT_SKIP);
    }
    return cache;
}

// Fills the WEIGHT zone around one hole of REQUEST bytes: the hole is taken first, then every other free byte, largest
// block first. `fill` keeps the fillers and `hole` the hole. False when the zone cannot be brought to that state.
bool fill_around_hole(unified_cache * cache, alloc_owner * hole, std::vector<alloc_owner> * fill) {
    allocation_result first = unified_cache_zone_allocate_owner(MOCK_DEVICE, vram_zone_id::WEIGHT, REQUEST);
    if (!first) {
        return false;
    }
    *hole = std::move(first.owner);
    while (cache->zone_available(vram_zone_id::WEIGHT) != 0) {
        const size_t want = cache->zone_largest_free(vram_zone_id::WEIGHT) / GRAIN * GRAIN;
        if (want == 0) {
            break;
        }
        allocation_result filler = unified_cache_zone_allocate_owner(MOCK_DEVICE, vram_zone_id::WEIGHT, want, GRAIN);
        if (!filler) {
            break;
        }
        fill->push_back(std::move(filler.owner));
    }
    return cache->zone_available(vram_zone_id::WEIGHT) == 0;
}

void test_full_zone_refuses_at_the_zone(unified_cache * cache) {
    std::printf("a zone-exact request on a full zone is refused at the zone:\n");
    const uint64_t reads_before = unified_cache_test_raw_gate_reads(MOCK_DEVICE);
    const uint64_t raw_before   = unified_cache_ext_alloc_count_for_testing(MOCK_DEVICE);

    allocation_result miss = unified_cache_zone_allocate_owner(MOCK_DEVICE, vram_zone_id::WEIGHT, REQUEST);

    const uint64_t reads = unified_cache_test_raw_gate_reads(MOCK_DEVICE) - reads_before;
    const uint64_t raw   = unified_cache_ext_alloc_count_for_testing(MOCK_DEVICE) - raw_before;
    std::printf("  request=%zu zone_available=%zu: served=%d error=%d raw_gate_reads=%llu raw_allocations=%llu\n",
                REQUEST, cache->zone_available(vram_zone_id::WEIGHT), miss ? 1 : 0, static_cast<int>(miss.error),
                static_cast<unsigned long long>(reads), static_cast<unsigned long long>(raw));
    CHECK(!miss, "full zone: the request was served");
    CHECK_EQ(reads, 0, "full zone: a raw gate read its figure for a request that accepts no raw block");
    CHECK_EQ(raw, 0, "full zone: a raw device allocation was made for a request that accepts none");
    CHECK_EQ(static_cast<int>(miss.error), static_cast<int>(allocation_error::PHYSICAL_ALLOCATION_FAILED),
             "full zone: the miss does not read as a failed physical allocation");
    CHECK_EQ(cache->zone_available(vram_zone_id::WEIGHT), 0, "full zone: the refusal changed the zone's free bytes");
}

void test_room_is_served_from_the_zone(unified_cache * cache, alloc_owner * hole) {
    std::printf("a zone-exact request with room is served from the zone:\n");
    (void) hole->reset();
    settle();
    CHECK_EQ(cache->zone_available(vram_zone_id::WEIGHT), REQUEST, "room: the hole's bytes did not come back");

    const uint64_t reads_before = unified_cache_test_raw_gate_reads(MOCK_DEVICE);
    const uint64_t raw_before   = unified_cache_ext_alloc_count_for_testing(MOCK_DEVICE);

    allocation_result hit = unified_cache_zone_allocate_owner(MOCK_DEVICE, vram_zone_id::WEIGHT, REQUEST);

    const uint64_t reads = unified_cache_test_raw_gate_reads(MOCK_DEVICE) - reads_before;
    const uint64_t raw   = unified_cache_ext_alloc_count_for_testing(MOCK_DEVICE) - raw_before;
    std::printf("  request=%zu: served=%d error=%d raw_gate_reads=%llu raw_allocations=%llu zone_available=%zu\n",
                REQUEST, hit ? 1 : 0, static_cast<int>(hit.error), static_cast<unsigned long long>(reads),
                static_cast<unsigned long long>(raw), cache->zone_available(vram_zone_id::WEIGHT));
    CHECK(static_cast<bool>(hit), "room: the request was refused");
    if (hit) {
        const alloc_metadata exact = hit.owner.metadata();
        CHECK(exact.zone_managed && exact.zone == vram_zone_id::WEIGHT, "room: the block is not the WEIGHT zone's");
        CHECK_EQ(exact.extent, REQUEST, "room: the block is not exactly the bytes asked");
        CHECK(exact.ptr != nullptr && cache->vram_owns(exact.ptr), "room: the block is not inside the arena");
    }
    CHECK_EQ(reads, 0, "room: a raw gate read its figure for a request the zone served");
    CHECK_EQ(raw, 0, "room: a raw device allocation was made for a request the zone served");
    CHECK_EQ(cache->zone_available(vram_zone_id::WEIGHT), 0, "room: the block did not take the hole");
    (void) hit.owner.reset();
    settle();
}

}  // namespace

int main(int, char ** argv) {
    // A bare run must not reach a GPU: no code here enumerates one, and the selector keeps the backend from doing it
    // on this process's behalf. A plain setenv() here is too late (oneCCL makes libsycl memoize the selector before
    // main()), so this re-execs; under ctest the registration already sets it.
    sycl_test_selector_fallback(argv, "opencl:cpu");
    sycl::queue q{ sycl::cpu_selector_v };

    set_mock_environment();
    unified_cache * cache = make_mock(q);
    std::printf("mock: arena %zu bytes, WEIGHT capacity %zu, available %zu\n", cache->arena_total_size(),
                cache->zone_capacity(vram_zone_id::WEIGHT), cache->zone_available(vram_zone_id::WEIGHT));

    alloc_owner              hole;
    std::vector<alloc_owner> fill;
    const bool               full = fill_around_hole(cache, &hole, &fill);
    std::printf("fill: %zu fillers, zone_available=%zu\n", fill.size(), cache->zone_available(vram_zone_id::WEIGHT));
    CHECK(full, "the WEIGHT zone could not be filled");
    if (full) {
        test_full_zone_refuses_at_the_zone(cache);
        test_room_is_served_from_the_zone(cache, &hole);
    }

    (void) hole.reset();
    for (alloc_owner & owner : fill) {
        (void) owner.reset();
    }
    fill.clear();
    settle();
    CHECK(shutdown_unified_cache(), "the cache did not shut down");

    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-zone-exact-alloc: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-zone-exact-alloc: all ok\n");
    return 0;
}
