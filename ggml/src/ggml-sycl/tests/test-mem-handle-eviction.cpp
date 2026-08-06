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

#include "sycl-test-skip.hpp"

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
// Test 1b: CONTROL for llama.cpp-gzea.  get_weight_ptr() must resolve an entry
// that direct_stage_weight() published, at the SAME 16 MB budget test 2 uses.
//
// This exists to discriminate between two competing explanations of the
// get_weight_ptr() miss in test 2:
//
//   (H1) key space — get_weight_ptr() probes only make_direct_stage_key()
//        keys, while ensure_cached() files the entry under the plain
//        {type, id, layer_id, expert_id} key reachable via id_to_key_.
//   (H2) allocation class — get_weight_ptr() resolves only arena-backed
//        entries, and the VRAM arena declines to reserve at a 16 MB budget
//        (it logs "falling back to per-entry allocation").
//
// The budget here is identical to test 2's, so the arena declines identically;
// the only thing that differs is WHICH key the writer files the entry under.
// H1 predicts this test passes.  H2 predicts it fails.  It is deliberately
// written so that a passing run refutes H2 rather than confirming H1.
// =============================================================================
static bool test_get_weight_ptr_resolves_direct_staged_entry(sycl::queue & q) {
    TEST_BEGIN("get_weight_ptr_resolves_direct_staged_entry");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;  // same as test 2
    ggml_sycl::unified_cache cache(q, budget);

    void * src_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_host != nullptr, "malloc_host for src should succeed");
    std::memset(src_host, 0x7E, entry_bytes);
    ggml_sycl_cache_id key = make_test_cache_id(700, 1, entry_bytes);

    auto staged =
        cache.direct_stage_weight(key, src_host, entry_bytes, entry_bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q);
    TEST_ASSERT(staged.ok && staged.ptr, "direct_stage_weight should succeed");
    staged.event.wait();
    // Same READY-driving step test 2 performs: get_weight_ptr() holds only a
    // shared_lock and so cannot flip IN_PROGRESS -> READY itself.
    void * drove = cache.get(key, GGML_LAYOUT_AOS);
    TEST_ASSERT(drove == staged.ptr, "cache.get() should return the direct-staged ptr");

    auto resolved = cache.get_weight_ptr(key);
    TEST_ASSERT(static_cast<bool>(resolved), "get_weight_ptr must resolve a direct_stage_weight entry at 16 MB budget");
    TEST_ASSERT(resolved.ptr == staged.ptr, "resolved ptr must match direct_stage_weight's");

    sycl::free(src_host, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 1c: get_weight_ptr() and acquire_weight_lease() must resolve the same
// key space.  A narrower key space in the lease variant means a caller that
// takes ownership fails to resolve a weight the non-owning accessor returns —
// llama.cpp-gzea one level down.
//
// NOTE: an earlier version of this comment said the lease variant is "the
// refcount-safe entry point used by mem_handle::resolve_slow()".  That is
// false.  resolve_slow() calls acquire_entry_lease() directly
// (mem-handle.cpp:476); acquire_weight_lease()'s only callers are
// cpu-dispatch.cpp:2620 and :2657, on the DNNL host-pointer path.  See the
// comment above acquire_weight_lease() in unified-cache.cpp for the latent gap
// that leaves in resolve_slow().
// =============================================================================
static bool test_lease_and_plain_lookup_agree(sycl::queue & q) {
    TEST_BEGIN("lease_and_plain_lookup_agree");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;
    ggml_sycl::unified_cache cache(q, budget);

    void * src_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_host != nullptr, "malloc_host for src should succeed");
    std::memset(src_host, 0x2D, entry_bytes);
    ggml_sycl_cache_id key = make_test_cache_id(800, 1, entry_bytes);

    // These ids must NOT be (-1, -1), and that is the whole point of the test.
    //
    // ensure_cached() files under {DENSE_WEIGHT, id, layer_id, expert_id}.  With
    // (-1, -1) the entry lands on exactly the key acquire_weight_lease() tries
    // second — `unified_cache_key{DENSE_WEIGHT, key, -1, -1}` — so the lookup
    // succeeds there and returns before the id_to_key_ fallback is reached.  The
    // fallback is also guarded by `!(mapped == ckey)`, which that same key fails.
    // A (-1, -1) version of this test therefore passes identically whether or not
    // the fallback exists: it would assert nothing about the code it was written
    // for.  Non-(-1, -1) ids make the direct-stage sweep and the {-1, -1} key both
    // miss, leaving id_to_key_ as the only route.
    //
    // Verified to discriminate: with the id_to_key_ fallback removed from
    // acquire_weight_lease(), the assertion below fails.
    constexpr int layer_id  = 3;
    constexpr int expert_id = 7;

    void * ptr = cache.ensure_cached(key, src_host, entry_bytes, ggml_sycl::cache_entry_type::DENSE_WEIGHT, layer_id,
                                     expert_id, GGML_LAYOUT_AOS, false);
    TEST_ASSERT(ptr != nullptr, "ensure_cached should succeed");
    q.wait();
    (void) cache.get(key, GGML_LAYOUT_AOS);  // drive state → READY

    auto plain = cache.get_weight_ptr(key);
    TEST_ASSERT(static_cast<bool>(plain), "get_weight_ptr must resolve an ensure_cached entry");

    auto lease = cache.acquire_weight_lease(key);
    TEST_ASSERT(lease.ptr != nullptr, "acquire_weight_lease must resolve whatever get_weight_ptr resolves");
    TEST_ASSERT(lease.entry != nullptr, "a successful lease must carry the entry to release against");

    // Release exactly once, per the contract documented on the declaration in
    // unified-cache.hpp, and do it BEFORE the remaining assertion: TEST_ASSERT
    // returns early on failure, which would otherwise skip the release and leave
    // in_use_count pinned on an entry the later tests expect to be evictable.
    void * leased_ptr = lease.ptr;
    lease.entry->in_use_count.fetch_sub(1);

    TEST_ASSERT(leased_ptr == plain.ptr, "lease and plain lookup must agree on the pointer");

    sycl::free(src_host, q);
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

    // Sibling accessors that DO consult id_to_key_ must agree with get().  If
    // these hold while get_weight_ptr() below does not, the disagreement is a
    // key-space gap in get_weight_ptr, not a property of the entry itself.
    TEST_ASSERT(cache.is_cached(key_a, GGML_LAYOUT_AOS), "is_cached(A, AOS) must agree with get()");
    TEST_ASSERT(cache.is_cached_any(key_a), "is_cached_any(A) must agree with get()");

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

    // Positive precondition.  Without it the "A should be evicted" assertion
    // below passes whenever get_weight_ptr() resolves NOTHING — it would then
    // be certifying llama.cpp-gzea as the expected behaviour rather than
    // catching it.  Asserting resolvable-then-unresolvable makes the pair fail
    // both when the accessor is blind and when it is over-permissive.
    TEST_ASSERT(cache.get_weight_ptr(key_a), "A must be resolvable before it is evicted");

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

    // Positive precondition — see the identical note in test 3.  The
    // "A must be unreachable after evict + finalize" assertion at the end of
    // this test passed for the wrong reason while llama.cpp-gzea was open:
    // get_weight_ptr() could not resolve this entry at any point, so the
    // post-eviction check was vacuously true and the test stayed green
    // throughout the defect it was meant to cover.
    TEST_ASSERT(cache.get_weight_ptr(key_a), "A must be resolvable before evict + finalize");

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

    // Enumeration comes FIRST: the bare `sycl::device dev;` this replaced
    // default-constructs through the default selector and THROWS on a device-less
    // host, from outside the try -- so the process aborted (exit 134) without ever
    // reaching the check below. See sycl-test-skip.hpp.
    std::optional<sycl::device> dev_opt = sycl_test_require_gpu("mem_handle / eviction lifecycle");
    if (!dev_opt) {
        return SYCL_TEST_SKIP;
    }
    sycl::device & dev = *dev_opt;
    fprintf(stderr, "Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "-------------------------------------------\n");

    sycl::queue q(dev, sycl::property::queue::in_order{});

    bool all_passed = true;
    all_passed &= test_direct_handle_stable_across_bumps();
    all_passed &= test_get_weight_ptr_resolves_direct_staged_entry(q);
    all_passed &= test_lease_and_plain_lookup_agree(q);
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
