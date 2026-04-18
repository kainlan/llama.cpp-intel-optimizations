//
// Test: mem_handle validity under eviction (RESEARCH-A4 / llama.cpp-2jmzn)
//
// Targeted unit test for the contracts that mem_handle relies on across the
// unified cache's eviction lifecycle:
//
//   (1) DIRECT handles are stable — resolve() returns the original pointer
//       regardless of cache_generation() bumps.
//   (2) Explicit eviction via cache.evict() bumps the global cache generation
//       AND removes the entry from lookup.  get_weight_ptr must not return a
//       stale device pointer for an evicted key.
//   (3) Re-insertion after eviction produces a fresh lookup result.
//
// Companion to llama.cpp-goegc.1 ("stale pointer after eviction"): this test
// covers the PRIMARY goegc.1 failure mode — key-based lookup returning null
// for an evicted key rather than a stale device pointer.  The kernel-args-
// in-flight variant (a kernel already submitted with an evicted VRAM ptr
// baked into its arg buffer) is the residual concern and needs its own
// test when goegc.1's fix lands.
//
// Notes for future modifiers:
//
//   * Budget is 16 MB per test, not 1 MB.  The unified_cache's VRAM arena
//     reserves ~1 GB of minimum zones (scratch+runtime+oneDNN) up-front,
//     so very small budgets push entries to host-pinned where the evict
//     path is quieter and the test loses coverage.  16 MB is large enough
//     that `malloc_device_raw` at unified-cache.cpp:1852 is exercised and
//     the entries land on device.
//
//   * After `cache.ensure_cached()`, entries start in state IN_PROGRESS
//     (unified-cache.cpp:1908) while the H2D copy event drains.
//     `get_weight_ptr()` / `try_get_cached_fast()` reject non-READY
//     entries, but `cache.get()` (unified-cache.cpp:2353) transitions the
//     state on observing a complete event.  Tests that need stable
//     lookup must call `q.wait()` + `cache.get(key, layout)` after
//     ensure_cached before asserting on other lookups.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>

#include "../unified-cache.hpp"
#include "../mem-handle.hpp"

// =============================================================================
// Test harness (mirrors test-unified-cache-fast-path.cpp)
// =============================================================================

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

// =============================================================================
// Helpers
// =============================================================================

static ggml_sycl_cache_id make_test_cache_id(uint64_t model_id, uint64_t aux_id, size_t nbytes) {
    ggml_sycl_cache_id id = {};
    id.valid              = true;
    id.model_id           = model_id;
    id.aux_id             = aux_id;
    id.has_gguf           = false;
    id.file_idx           = 0;
    id.file_offs          = 0;
    id.nbytes             = nbytes;
    id.name_hash          = model_id ^ aux_id;
    id.type               = GGML_TYPE_F32;
    id.tp_sharded         = false;
    id.tp_rank            = 0;
    id.tp_world_size      = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = (i == 0) ? static_cast<int64_t>(nbytes / sizeof(float)) : 1;
        id.tp_local_ne[i]  = id.ne[i];
        id.tp_offset_ne[i] = 0;
    }
    return id;
}

// =============================================================================
// Test 1: DIRECT mem_handle is immune to generation bumps.
// =============================================================================
static bool test_direct_handle_stable_across_bumps() {
    TEST_BEGIN("direct_handle_stable_across_bumps");

    int                   marker  = 0;
    void *                raw_ptr = &marker;
    ggml_sycl::mem_handle h       = ggml_sycl::mem_handle::from_direct(raw_ptr, GGML_LAYOUT_AOS, true);

    const uint64_t gen_before = ggml_sycl::cache_generation();

    auto r1 = h.resolve();
    TEST_ASSERT(r1.ptr == raw_ptr, "first resolve should return the direct ptr");
    TEST_ASSERT(r1.on_device, "direct on_device flag should be preserved");

    // Simulate evictions happening elsewhere.
    ggml_sycl::cache_generation_bump();
    ggml_sycl::cache_generation_bump();
    TEST_ASSERT(ggml_sycl::cache_generation() == gen_before + 2, "two bumps should produce gen+2");

    auto r2 = h.resolve();
    TEST_ASSERT(r2.ptr == raw_ptr, "DIRECT handle must still return same ptr after bumps");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Explicit eviction removes the entry and bumps generation.
//
// Uses a budget large enough to avoid arena-minimum shenanigans, inserts two
// entries, then calls cache.evict(size) directly to force an eviction.
// =============================================================================
static bool test_explicit_evict_bumps_gen_and_removes_entry(sycl::queue & q) {
    TEST_BEGIN("explicit_evict_bumps_gen_and_removes_entry");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;  // 16 MB — well above arena min
    ggml_sycl::unified_cache cache(q, budget);

    // Use sycl::malloc_host for src so ensure_cached's H2D copy uses USM-visible
    // memory; mirrors the fast-path test's pattern.
    void * src_a_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_a_host != nullptr, "malloc_host for src_a should succeed");
    std::memset(src_a_host, 0xAB, entry_bytes);
    ggml_sycl_cache_id key_a = make_test_cache_id(100, 1, entry_bytes);

    void * ptr_a = cache.ensure_cached(key_a, src_a_host, entry_bytes,
                                       ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                       -1, -1, GGML_LAYOUT_AOS, false);
    TEST_ASSERT(ptr_a != nullptr, "ensure_cached(A) should succeed");
    // Drain any in-flight H2D, then call cache.get() to drive the IN_PROGRESS
    // → READY state transition (get() checks event_complete and flips state).
    q.wait();
    void * drove_a = cache.get(key_a, GGML_LAYOUT_AOS);
    TEST_ASSERT(drove_a == ptr_a, "cache.get() should return the same ptr after event completes");

    // Sanity: A is resolvable before eviction.
    auto result_a_before = cache.get_weight_ptr(key_a);
    TEST_ASSERT(static_cast<bool>(result_a_before), "A should be resolvable before eviction");
    TEST_ASSERT(result_a_before.ptr == ptr_a, "pre-eviction ptr matches ensure_cached return");

    const uint64_t gen_before_evict = ggml_sycl::cache_generation();

    // Explicitly evict.  Ask for more bytes than A itself to motivate eviction of
    // everything evictable.  evict() calls evict_one in a loop.
    size_t freed = cache.evict(entry_bytes * 2);

    // The amount freed may be 0 (host-resident) or entry_bytes (device-resident);
    // either way, A should be removed from entries_ and gen should have bumped.
    (void) freed;
    const uint64_t gen_after_evict = ggml_sycl::cache_generation();
    TEST_ASSERT(gen_after_evict > gen_before_evict,
                "explicit evict() must bump cache_generation when it removed at least one entry");

    // Stale-pointer check: lookup of evicted key must not return the cached device ptr.
    auto result_a_after = cache.get_weight_ptr(key_a);
    TEST_ASSERT(!result_a_after, "get_weight_ptr(A) must return null result after explicit eviction");

    sycl::free(src_a_host, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Re-insertion after eviction produces a fresh lookup result and
// bumps the generation again.
// =============================================================================
static bool test_reinsert_after_evict_recovers_lookup(sycl::queue & q) {
    TEST_BEGIN("reinsert_after_evict_recovers_lookup");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;
    ggml_sycl::unified_cache cache(q, budget);

    void * src_a_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_a_host != nullptr, "malloc_host for src_a should succeed");
    std::memset(src_a_host, 0x11, entry_bytes);
    ggml_sycl_cache_id key_a = make_test_cache_id(200, 1, entry_bytes);

    (void) cache.ensure_cached(key_a, src_a_host, entry_bytes,
                               ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                               -1, -1, GGML_LAYOUT_AOS, false);
    q.wait();
    (void) cache.get(key_a, GGML_LAYOUT_AOS);  // drive state → READY

    (void) cache.evict(entry_bytes * 2);
    const uint64_t gen_after_evict = ggml_sycl::cache_generation();

    // Verify A is really gone.
    TEST_ASSERT(!cache.get_weight_ptr(key_a), "A should be evicted before re-insert step");

    // Re-insert A.
    void * ptr_a2 = cache.ensure_cached(key_a, src_a_host, entry_bytes,
                                        ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                        -1, -1, GGML_LAYOUT_AOS, false);
    TEST_ASSERT(ptr_a2 != nullptr, "re-ensure_cached(A) should succeed");
    q.wait();
    (void) cache.get(key_a, GGML_LAYOUT_AOS);  // drive state → READY

    // Lookup sees the fresh pointer.
    auto result_a = cache.get_weight_ptr(key_a);
    TEST_ASSERT(static_cast<bool>(result_a), "get_weight_ptr(A) must resolve after re-insert");
    TEST_ASSERT(result_a.ptr == ptr_a2, "re-insert lookup must return the new ptr");

    // Generation monotonicity: gen after re-insert is >= gen immediately after evict.
    // (ensure_cached does not currently bump generation on its own, only via eviction,
    // so we only assert non-decrease here — not strict monotonic increase.)
    const uint64_t gen_after_readd = ggml_sycl::cache_generation();
    TEST_ASSERT(gen_after_readd >= gen_after_evict,
                "generation must be monotonically non-decreasing across re-insert");

    sycl::free(src_a_host, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Async eviction (SOA layout) reaches EVICTING state.  VRAM stays
// mapped until finalize_evictions() runs; generation bumps somewhere along
// the way (at evict_one for sync, at finalize for async).
// =============================================================================
static bool test_async_eviction_finalize_bumps_gen(sycl::queue & q) {
    TEST_BEGIN("async_eviction_finalize_bumps_gen");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;
    ggml_sycl::unified_cache cache(q, budget);

    void * src_a_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_a_host != nullptr, "malloc_host for src_a should succeed");
    std::memset(src_a_host, 0x55, entry_bytes);
    ggml_sycl_cache_id key_a = make_test_cache_id(300, 1, entry_bytes);

    void * ptr_a = cache.ensure_cached(key_a, src_a_host, entry_bytes,
                                       ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                       -1, -1, GGML_LAYOUT_SOA, false);
    TEST_ASSERT(ptr_a != nullptr, "ensure_cached(A, SOA) should succeed");
    q.wait();
    (void) cache.get(key_a, GGML_LAYOUT_SOA);  // drive state → READY

    const uint64_t gen_before = ggml_sycl::cache_generation();

    // Evict.  SOA layout is the qualifying condition for async_evict in evict_one
    // (unified-cache.cpp:3829-3830: has_transformed_layout && async_evict_enabled_).
    // The default constructor enables async_evict unless GGML_SYCL_ASYNC_EVICT=0.
    // If the env var disables it, this test falls through to the sync path and
    // still passes (gen bumps at unified-cache.cpp:3915 instead of :3988) —
    // acceptable but reduces coverage of the async path specifically.  We do
    // not assert on the mode chosen because neither unified_cache nor async_evict
    // exposes a public getter for the runtime mode.
    (void) cache.evict(entry_bytes * 2);

    // Allow any in-flight DMA to complete before finalizing. The cache submits
    // async D2H on its internal dma_queue_, which is distinct from the test's
    // own queue, so we must drain it explicitly via get_dma_queue().wait().
    cache.get_dma_queue().wait();
    (void) cache.finalize_evictions();

    const uint64_t gen_after = ggml_sycl::cache_generation();
    TEST_ASSERT(gen_after > gen_before, "generation must bump across evict + finalize_evictions");

    auto result_a = cache.get_weight_ptr(key_a);
    TEST_ASSERT(!result_a, "A must be unreachable after evict + finalize");

    sycl::free(src_a_host, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "mem_handle / eviction lifecycle tests\n");
    fprintf(stderr, "===========================================\n");

    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU device found: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "-------------------------------------------\n");

    sycl::queue q(dev, sycl::property::queue::in_order{});

    bool all_passed = true;
    all_passed &= test_direct_handle_stable_across_bumps();
    all_passed &= test_explicit_evict_bumps_gen_and_removes_entry(q);
    all_passed &= test_reinsert_after_evict_recovers_lookup(q);
    all_passed &= test_async_eviction_finalize_bumps_gen(q);

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
