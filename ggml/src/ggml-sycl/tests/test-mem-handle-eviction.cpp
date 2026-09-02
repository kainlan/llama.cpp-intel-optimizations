//
// Test: mem_handle validity under eviction (RESEARCH-A4 / llama.cpp-2jmzn)
//
// Targeted unit test for the contracts that mem_handle relies on across the
// unified cache's eviction lifecycle:
//
//   (1) DIRECT handles are stable — resolve() returns the original pointer
//       regardless of cache_generation() bumps.
//   (2) Explicit eviction via cache.evict() bumps the global cache generation
//       when it actually removes an entry, and removes that entry from lookup.
//       get_weight_ptr must not return a stale device pointer for an evicted key.
//   (3) Re-insertion after eviction produces a fresh lookup result.
//
// Companion to llama.cpp-goegc.1 ("stale pointer after eviction"): this test
// covers the PRIMARY goegc.1 failure mode — key-based lookup returning null
// for an evicted key rather than a stale device pointer.  The kernel-args-
// in-flight variant (a kernel already submitted with A's VRAM ptr baked into
// its arg buffer while an eviction runs) is covered by
// test_in_flight_kernel_lease_blocks_eviction (llama.cpp-goegc.1 residual,
// absorbing llama.cpp-32dg8.15.3) for the WEIGHT entry lease, and by
// test_in_flight_kernel_arena_chunk_lease_blocks_replan for the device VRAM
// arena chunk lease, which gates a different reclaim path (see that test's
// comment).  The host-pinned chunk lease equivalent is a documented
// structural gap — see the "Scope note" above
// test_in_flight_kernel_lease_blocks_eviction.
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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sycl/sycl.hpp>

#include "../unified-cache.hpp"
#include "../mem-handle.hpp"
#include "../common.hpp"
#include "../mmvq-rmsnorm.hpp"

#include "sycl-test-skip.hpp"

// =============================================================================
// Test harness (mirrors test-unified-cache-fast-path.cpp)
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

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

// A skip is not a pass and not a failure: it means the environment could not
// exercise the case (e.g. the device lacks a capability the case needs), so
// nothing about the case's correctness was proven either way.  Unlike
// TEST_BEGIN's g_tests_run, this decrements g_tests_run back out -- a skip
// was not a completed run of the case's logic -- so "N run, M passed,
// K skipped" always satisfies N == M + (failures) with skips counted
// separately, matching test-mem-handle-wrong-device.cpp's TEST_SKIP.
#define TEST_SKIP(reason)                        \
    do {                                         \
        g_tests_skipped++;                       \
        g_tests_run--;                            \
        fprintf(stderr, "SKIPPED: %s\n", reason); \
        return true;                              \
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

    // Sanity: A is resolvable before eviction.  get_weight_ptr() returns a
    // retained handle by contract, so confine the precondition result to this
    // scope.  Keeping it alive across evict() correctly makes A ineligible;
    // an eviction request alone neither removes A nor bumps the generation.
    {
        auto result_a_before = cache.get_weight_ptr(key_a);
        TEST_ASSERT(static_cast<bool>(result_a_before), "A should be resolvable before eviction");
        TEST_ASSERT(result_a_before.ptr == ptr_a, "pre-eviction ptr matches ensure_cached return");
    }

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
// Test 5: exact retirement is an immediate discovery transition but a deferred
// lifetime transition. A live lease remains usable until release.
// =============================================================================
static bool test_expert_retirement_with_live_lease(sycl::queue & q) {
    TEST_BEGIN("expert_retirement_with_live_lease");

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;
    ggml_sycl::unified_cache cache(q, budget);
    void *                   src_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_host != nullptr, "malloc_host for retirement source should succeed");
    std::memset(src_host, 0x5A, entry_bytes);

    ggml_sycl_cache_id   key = make_test_cache_id(705, 6, entry_bytes);
    ggml_sycl::mem_handle old_lease;
    auto staged = cache.direct_stage_expert(key, src_host, entry_bytes, entry_bytes, GGML_LAYOUT_AOS,
                                            nullptr, nullptr, &q, &old_lease);
    TEST_ASSERT(staged.ok && staged.ptr && old_lease.valid(), "expert stage should publish a lease");
    staged.event.wait_and_throw();
    const auto before = old_lease.resolve();
    TEST_ASSERT(before.ptr == staged.ptr, "old lease did not resolve staged storage");

    const auto retired = cache.retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-live-lease");
    TEST_ASSERT(retired == ggml_sycl::expert_retire_status::DEFERRED,
                "live lease retirement should defer reclamation");
    TEST_ASSERT(cache.retired_pending_count_for_test() == 1,
                "false->true retirement must increment pending exactly once");
    TEST_ASSERT(cache.retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-idempotent-retire") ==
                    ggml_sycl::expert_retire_status::DEFERRED,
                "repeat retirement should remain deferred");
    TEST_ASSERT(cache.retired_pending_count_for_test() == 1,
                "repeat retirement must not double-count pending entries");

    ggml_sycl::expert_resolve_request request{};
    request.key              = key;
    request.requested_layout = GGML_LAYOUT_AOS;
    TEST_ASSERT(!cache.resolve_expert(request), "new resolve discovered retired expert");
    TEST_ASSERT(old_lease.resolve().ptr == before.ptr, "retirement invalidated the already acquired lease");

    old_lease = {};
    cache.process_deferred_frees_public();
    TEST_ASSERT(cache.lookup_expert(key) == nullptr, "retired expert survived final lease release");
    TEST_ASSERT(cache.retired_pending_count_for_test() == 0,
                "retirement finalization did not balance the pending counter");
    sycl::free(src_host, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: publication and exact retirement linearize under the same joint lock.
// The hook pauses after canonical insertion but before direct-mirror insertion;
// retirement must block, then withdraw both views without a mirror ABA.
// =============================================================================
static bool test_expert_publication_retirement_linearization(sycl::queue & q) {
    TEST_BEGIN("expert_publication_retirement_linearization");

    struct hook_state {
        std::mutex              mutex;
        std::condition_variable cv;
        bool                    entered = false;
        bool                    release = false;
    } state;
    auto hook = [](void * opaque) {
        auto & s = *static_cast<hook_state *>(opaque);
        std::unique_lock<std::mutex> lock(s.mutex);
        s.entered = true;
        s.cv.notify_all();
        s.cv.wait(lock, [&] { return s.release; });
    };

    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    std::vector<uint8_t>     first(4096, 0x31);
    std::vector<uint8_t>     second(4096, 0x72);
    const auto key = make_test_cache_id(706, 7, first.size());
    std::atomic<bool> publish_ok{ false };
    std::atomic<bool> retire_done{ false };
    ggml_sycl::expert_retire_status retire_status = ggml_sycl::expert_retire_status::INVALID;

    ggml_sycl::unified_cache_set_expert_publication_test_hook(hook, &state);
    std::thread publisher([&] {
        publish_ok.store(cache.register_host_expert(key, first.data(), first.size(), GGML_LAYOUT_AOS),
                         std::memory_order_release);
    });
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cv.wait(lock, [&] { return state.entered; });
    }
    std::thread retiree([&] {
        retire_status = cache.retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-publication-race");
        retire_done.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const bool retirement_blocked_at_joint_lock = !retire_done.load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.release = true;
    }
    state.cv.notify_all();
    publisher.join();
    retiree.join();
    ggml_sycl::unified_cache_set_expert_publication_test_hook(nullptr, nullptr);

    ggml_sycl::expert_resolve_request request{};
    request.key              = key;
    request.requested_layout = GGML_LAYOUT_AOS;
    TEST_ASSERT(publish_ok.load(std::memory_order_acquire), "host expert publisher failed");
    TEST_ASSERT(retirement_blocked_at_joint_lock, "retirement crossed a partial publication");
    TEST_ASSERT(ggml_sycl::expert_retire_succeeded(retire_status), "exact retirement failed");
    TEST_ASSERT(!cache.resolve_expert(request), "canonical expert remained discoverable after retirement");
    TEST_ASSERT(cache.lookup_expert(key) == nullptr, "direct mirror ABA survived retirement");

    cache.process_deferred_frees_public();
    TEST_ASSERT(cache.register_host_expert(key, second.data(), second.size(), GGML_LAYOUT_AOS),
                "same-key restage did not progress after synchronize/GC");
    TEST_ASSERT(cache.resolve_expert(request), "restaged expert was not discoverable");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: host publication commit failure is all-or-nothing.
// =============================================================================
static bool test_host_publication_fault_is_transactional(sycl::queue & q) {
    TEST_BEGIN("host_publication_fault_is_transactional");

    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    std::vector<uint8_t>     bytes(4096, 0x44);
    const auto key = make_test_cache_id(707, 8, bytes.size());

    ggml_sycl::unified_cache_fail_next_expert_phase_for_test(
        ggml_sycl::expert_fault_phase::HOST_BEFORE_COMMIT);
    TEST_ASSERT(!cache.register_host_expert(key, bytes.data(), bytes.size(), GGML_LAYOUT_AOS),
                "faulted host publisher unexpectedly committed");
    ggml_sycl::expert_resolve_request request{};
    request.key              = key;
    request.requested_layout = GGML_LAYOUT_AOS;
    TEST_ASSERT(!cache.resolve_expert(request), "faulted host publisher leaked canonical state");
    TEST_ASSERT(cache.lookup_expert(key) == nullptr, "faulted host publisher leaked direct mirror");
    TEST_ASSERT(cache.validate(), "faulted host publisher left inconsistent maps");

    TEST_ASSERT(cache.register_host_expert(key, bytes.data(), bytes.size(), GGML_LAYOUT_AOS),
                "host publisher did not recover after one-shot fault");
    TEST_ASSERT(cache.resolve_expert(request), "successful retry was not discoverable");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 8: every device publication fault before the commit point leaves both
// indices empty, releases/defer-releases backing, and permits a clean retry.
// =============================================================================
static bool test_device_publication_fault_phases(sycl::queue & q) {
    TEST_BEGIN("device_publication_fault_phases");

    constexpr size_t bytes = 4096;
    // Configure the fixture geometry before cache construction instead of
    // assuming the production 1280 MiB tail (which made a 512 MiB assertion
    // impossible on constrained B50/B70 test runs). oneDNN retains its 256 MiB
    // minimum; 1 MiB runtime + scratch leaves a real WEIGHT region.
    const char * old_compute = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB");
    const char * old_runtime = std::getenv("GGML_SYCL_RUNTIME_ARENA_MB");
    const bool had_compute = old_compute != nullptr;
    const bool had_runtime = old_runtime != nullptr;
    const std::string saved_compute = old_compute ? old_compute : "";
    const std::string saved_runtime = old_runtime ? old_runtime : "";
#if defined(_WIN32)
    _putenv_s("GGML_SYCL_COMPUTE_ARENA_MB", "1");
    _putenv_s("GGML_SYCL_RUNTIME_ARENA_MB", "1");
#else
    setenv("GGML_SYCL_COMPUTE_ARENA_MB", "1", 1);
    setenv("GGML_SYCL_RUNTIME_ARENA_MB", "1", 1);
#endif
    ggml_sycl::unified_cache cache(q, 512ull * 1024 * 1024);
#if defined(_WIN32)
    _putenv_s("GGML_SYCL_COMPUTE_ARENA_MB", had_compute ? saved_compute.c_str() : "");
    _putenv_s("GGML_SYCL_RUNTIME_ARENA_MB", had_runtime ? saved_runtime.c_str() : "");
#else
    had_compute ? (void) setenv("GGML_SYCL_COMPUTE_ARENA_MB", saved_compute.c_str(), 1) :
                  (void) unsetenv("GGML_SYCL_COMPUTE_ARENA_MB");
    had_runtime ? (void) setenv("GGML_SYCL_RUNTIME_ARENA_MB", saved_runtime.c_str(), 1) :
                  (void) unsetenv("GGML_SYCL_RUNTIME_ARENA_MB");
#endif

    ggml_sycl::alloc_handle geometry{};
    const int device = ggml_sycl_get_device_id_from_queue(q);
    TEST_ASSERT(ggml_sycl::unified_cache_zone_allocate(
                    device, ggml_sycl::vram_zone_id::WEIGHT, bytes, &geometry, 256),
                "exact WEIGHT geometry allocation failed");
    TEST_ASSERT(geometry.alloc_id != 0 && geometry.arena_generation != 0,
                "exact allocation omitted id/generation");
    TEST_ASSERT(geometry.vram_zone == ggml_sycl::vram_zone_id::WEIGHT && geometry.arena_extent == bytes,
                "exact allocation omitted zone/extent");
    ggml_sycl::mem_handle geometry_owner =
        ggml_sycl::detail::from_legacy_owned_alloc(std::move(geometry), GGML_LAYOUT_AOS);
    const auto geometry_info = geometry_owner.debug_info();
    TEST_ASSERT(geometry_info.valid && geometry_info.canonical_allocation_id != 0 &&
                    geometry_info.canonical_generation != 0 && geometry_info.canonical_extent == bytes,
                "allocation-time tuple was not preserved by mem_handle");
    geometry_owner = {};
    // direct_stage_expert() is given the exact source capacity.  Use ordinary
    // owned host storage so its raw-pointer bridge mints a bounded DIRECT view;
    // an external sycl::malloc_host pointer is classified as HOST_PINNED and
    // the compatibility chunk bridge cannot infer that allocation's extent.
    std::vector<uint8_t> src(bytes * 2, 0x61);

    const ggml_sycl::expert_fault_phase single_phases[] = {
        ggml_sycl::expert_fault_phase::SINGLE_AFTER_ALLOC,
        ggml_sycl::expert_fault_phase::SINGLE_BEFORE_COMMIT,
    };
    for (size_t i = 0; i < 2; ++i) {
        const auto key = make_test_cache_id(710 + i, 20 + i, bytes);
        ggml_sycl::unified_cache_fail_next_expert_phase_for_test(single_phases[i]);
        const auto failed = cache.direct_stage_expert(
            key, src.data(), bytes, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr);
        TEST_ASSERT(!failed.ok, "faulted single publication committed");
        TEST_ASSERT(cache.lookup_expert(key) == nullptr, "faulted single publication leaked a mirror");
        TEST_ASSERT(cache.register_host_expert(key, src.data(), bytes, GGML_LAYOUT_AOS),
                    "single publication did not recover after fault");
    }

    const ggml_sycl::expert_fault_phase bulk_phases[] = {
        ggml_sycl::expert_fault_phase::BULK_AFTER_ALLOC,
        ggml_sycl::expert_fault_phase::BULK_BEFORE_COMMIT,
    };
    for (size_t i = 0; i < 2; ++i) {
        std::vector<ggml_sycl_cache_id> keys{
            make_test_cache_id(720 + i * 2, 30 + i * 2, bytes),
            make_test_cache_id(721 + i * 2, 31 + i * 2, bytes),
        };
        ggml_sycl::unified_cache_fail_next_expert_phase_for_test(bulk_phases[i]);
        const auto failed = cache.direct_stage_expert_tensor(
            keys, src.data(), bytes * 2, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr);
        TEST_ASSERT(!failed.ok, "faulted bulk publication committed");
        TEST_ASSERT(cache.lookup_expert(keys[0]) == nullptr && cache.lookup_expert(keys[1]) == nullptr,
                    "faulted bulk publication leaked a mirror");
    }

    const auto duplicate = make_test_cache_id(730, 40, bytes);
    std::vector<ggml_sycl_cache_id> duplicates{ duplicate, duplicate };
    TEST_ASSERT(!cache.direct_stage_expert_tensor(
                     duplicates, src.data(), bytes * 2, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr).ok,
                "bulk publication accepted duplicate keys");
    TEST_ASSERT(cache.lookup_expert(duplicate) == nullptr, "duplicate bulk request partially published");

    q.wait_and_throw();
    cache.process_deferred_frees_public();
    TEST_ASSERT(cache.validate(), "device fault phases left inconsistent bookkeeping");
    TEST_PASS();
    return true;
}

static sycl::event count_unexpected_stage_fill(sycl::queue & queue,
                                               void *, size_t, const void *, size_t,
                                               const void * ctx,
                                               const std::vector<sycl::event> &) {
    static_cast<std::atomic<int> *>(const_cast<void *>(ctx))->fetch_add(1, std::memory_order_relaxed);
    return queue.ext_oneapi_submit_barrier();
}

// Owner-control failure must happen before any allocation pointer can be copied,
// submitted to a fill callback, or published through the output handle.
static bool test_owner_failure_prevents_raw_stage_use(sycl::queue & q) {
    TEST_BEGIN("owner_failure_prevents_raw_stage_use");

    constexpr size_t bytes = 4096;
    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    std::vector<uint8_t> src(bytes * 2, 0x73);
    std::atomic<int> fill_calls{ 0 };
    int sentinel = 0x274;
    ggml_sycl::mem_handle sentinel_handle =
        ggml_sycl::mem_handle::from_direct(&sentinel, GGML_LAYOUT_AOS, false,
                                           ggml_sycl::mem_handle::HOST_DEVICE, sizeof(sentinel));

    const auto dense_key = make_test_cache_id(2740, 2740, bytes);
    ggml_sycl::allocation_owner_test_fail_next_control_allocations(2);
    const auto dense = cache.direct_stage_weight(
        dense_key, src.data(), bytes, bytes, GGML_LAYOUT_AOS, count_unexpected_stage_fill,
        &fill_calls, &q, &sentinel_handle);
    TEST_ASSERT(!dense.ok && dense.ptr == nullptr, "owner-failed dense stage exposed a raw pointer");
    TEST_ASSERT(fill_calls.load(std::memory_order_relaxed) == 0,
                "owner-failed dense stage submitted a fill");
    TEST_ASSERT(sentinel_handle.resolve().ptr == &sentinel,
                "owner-failed dense stage changed the output sentinel");

    const auto expert_key = make_test_cache_id(2741, 2741, bytes);
    ggml_sycl::allocation_owner_test_fail_next_control_allocations(2);
    const auto single = cache.direct_stage_expert(
        expert_key, src.data(), bytes, bytes, GGML_LAYOUT_AOS, count_unexpected_stage_fill,
        &fill_calls, &q, &sentinel_handle);
    TEST_ASSERT(!single.ok && single.ptr == nullptr, "owner-failed expert stage exposed a raw pointer");
    TEST_ASSERT(fill_calls.load(std::memory_order_relaxed) == 0,
                "owner-failed expert stage submitted a fill");
    TEST_ASSERT(sentinel_handle.resolve().ptr == &sentinel,
                "owner-failed expert stage changed the output sentinel");

    std::vector<ggml_sycl_cache_id> bulk_keys{
        make_test_cache_id(2742, 2742, bytes), make_test_cache_id(2743, 2743, bytes) };
    std::vector<ggml_sycl::mem_handle> bulk_sentinel{ sentinel_handle };
    ggml_sycl::allocation_owner_test_fail_next_control_allocations(1);
    const auto bulk = cache.direct_stage_expert_tensor(
        bulk_keys, src.data(), bytes * 2, bytes, GGML_LAYOUT_AOS, count_unexpected_stage_fill,
        &fill_calls, &q, &bulk_sentinel);
    TEST_ASSERT(!bulk.ok && bulk.ptr == nullptr, "owner-failed bulk stage exposed a raw pointer");
    TEST_ASSERT(fill_calls.load(std::memory_order_relaxed) == 0,
                "owner-failed bulk stage submitted a fill");
    TEST_ASSERT(bulk_sentinel.size() == 1 && bulk_sentinel.front().resolve().ptr == &sentinel,
                "owner-failed bulk stage changed the output sentinel");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 9: abort cleanup failure is observable and a caller can fail closed;
// retry performs the allocation-free withdrawal.
// =============================================================================
static bool test_abort_cleanup_status_is_observable(sycl::queue & q) {
    TEST_BEGIN("abort_cleanup_status_is_observable");

    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    std::vector<uint8_t> bytes(4096, 0x52);
    const auto key = make_test_cache_id(735, 45, bytes.size());
    constexpr uint64_t load_txn = 0xB70;
    TEST_ASSERT(cache.register_host_expert(key, bytes.data(), bytes.size(), GGML_LAYOUT_AOS),
                "abort fixture publication failed");
    TEST_ASSERT(cache.test_mark_entry_touched_by_load(key, GGML_LAYOUT_AOS, load_txn),
                "abort fixture was not tagged");

    ggml_sycl::unified_cache_fail_next_expert_phase_for_test(
        ggml_sycl::expert_fault_phase::ABORT_BEFORE_RETIRE);
    TEST_ASSERT(!cache.note_model_load_abort(load_txn), "faulted abort reported false success");
    TEST_ASSERT(cache.lookup_expert(key) != nullptr, "faulted abort partially withdrew publication");

    TEST_ASSERT(cache.note_model_load_abort(load_txn), "allocation-free abort retry failed");
    TEST_ASSERT(cache.lookup_expert(key) == nullptr, "successful abort left direct publication visible");
    TEST_ASSERT(cache.retired_pending_count_for_test() == 0, "successful abort left reclaimable retirement pending");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 10: a ready-event status query exception is unknown and therefore keeps
// retired backing deferred until a later successful query proves completion.
// =============================================================================
static bool test_retired_status_query_failure_is_deferred(sycl::queue & q) {
    TEST_BEGIN("retired_status_query_failure_is_deferred");

    constexpr size_t bytes = 4096;
    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    void * src = sycl::malloc_host(bytes, q);
    TEST_ASSERT(src != nullptr, "malloc_host for GC status test failed");
    const auto key = make_test_cache_id(740, 50, bytes);
    auto staged = cache.direct_stage_expert(
        key, src, bytes, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr);
    TEST_ASSERT(staged.ok, "GC status test stage failed");

    ggml_sycl::unified_cache_fail_next_expert_phase_for_test(
        ggml_sycl::expert_fault_phase::GC_READY_EVENT);
    TEST_ASSERT(cache.retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-status-query") ==
                    ggml_sycl::expert_retire_status::DEFERRED,
                "status-query failure was incorrectly treated as terminal");
    TEST_ASSERT(cache.retired_pending_count_for_test() == 1,
                "status-query failure released retired backing");

    staged.event.wait_and_throw();
    cache.process_deferred_frees_public();
    TEST_ASSERT(cache.retired_pending_count_for_test() == 0,
                "retired backing was not reclaimed after terminal proof");
    sycl::free(src, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 10: terminal ticket state transitions commit exact event authority.
// =============================================================================
static bool test_terminal_retention_ticket_state_machine(sycl::queue & q) {
    TEST_BEGIN("terminal_retention_ticket_state_machine");

    int marker = 0;
    auto owner = ggml_sycl::mem_handle::from_direct(&marker, GGML_LAYOUT_AOS, false);
    auto ticket = ggml_sycl::terminal_retention_ticket::prepare({}, { owner });
    TEST_ASSERT(ticket.current_state() == ggml_sycl::terminal_retention_ticket::state::PREPARED,
                "ticket did not enter PREPARED");
    TEST_ASSERT(ticket.owner_count() == 1, "ticket did not escrow owner before submit");
    sycl::event terminal = q.submit([&](sycl::handler & cgh) { cgh.host_task([]() {}); });
    ticket.mark_submitted(q);
    TEST_ASSERT(ticket.current_state() == ggml_sycl::terminal_retention_ticket::state::SUBMITTED,
                "ticket did not enter SUBMITTED");
    ticket.commit(terminal);
    TEST_ASSERT(ticket.current_state() == ggml_sycl::terminal_retention_ticket::state::COMMITTED,
                "ticket did not enter COMMITTED");
    TEST_ASSERT(ggml_sycl::drain_retained_handles(true, 1000), "committed terminal did not drain");

    TEST_PASS();
    return true;
}

static bool test_mmvq_rmsnorm_second_batch_failure_drains_first(sycl::queue & q) {
    TEST_BEGIN("mmvq_rmsnorm_second_batch_failure_drains_first");

    std::atomic<bool> release_first{ false };
    std::atomic<bool> cleanup_entered{ false };
    std::atomic<int> submitted_callbacks{ 0 };
    int marker = 0;

    auto result = std::async(std::launch::async, [&] {
        try {
            auto ticket = ggml_sycl::terminal_retention_ticket::prepare(
                {}, { ggml_sycl::mem_handle::from_direct(&marker, GGML_LAYOUT_AOS, false) });
            ggml_sycl_mmvq_rmsnorm_submit_batches(
                2,
                [&](int batch) {
                    ggml_sycl_mmvq_rmsnorm_fail_submit_n_for_test(batch, 1, cleanup_entered);
                    q.submit([&](sycl::handler & cgh) {
                        cgh.host_task([&] {
                            while (!release_first.load(std::memory_order_acquire)) std::this_thread::yield();
                        });
                    });
                },
                [&] {
                    ticket.mark_submitted(q);
                    submitted_callbacks.fetch_add(1, std::memory_order_release);
                });
        } catch (const std::runtime_error &) {
            return true;
        }
        return false;
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!cleanup_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!cleanup_entered.load(std::memory_order_acquire)) {
        release_first.store(true, std::memory_order_release);
        (void) result.get();
        fprintf(stderr, "FAIL: RMSNorm second-submit cleanup barrier was not reached\n");
        return false;
    }

    // The private failpoint signals immediately before exception unwinding
    // enters the submitted ticket's queue wait. The future cannot complete
    // until the independently controlled first-batch barrier is released.
    const bool blocked_at_cleanup = result.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
    const int callbacks_before_release = submitted_callbacks.load(std::memory_order_acquire);
    release_first.store(true, std::memory_order_release);
    const bool caught_second_failure = result.get();

    TEST_ASSERT(blocked_at_cleanup, "second-batch failure released ticket owners before first batch drained");
    TEST_ASSERT(callbacks_before_release == 1, "production batch loop callback did not mark exactly the successful submit");
    TEST_ASSERT(caught_second_failure, "second-batch submit failure was not observed");
    TEST_PASS();
    return true;
}

static bool test_graph_recording_epoch_reuse_rejected(sycl::queue & q) {
    TEST_BEGIN("graph_recording_epoch_reuse_rejected");

    int marker = 0;
    std::vector<ggml_sycl::mem_handle> sink;
    ggml_sycl::set_graph_retained_handle_sink(&sink);
    auto ticket = ggml_sycl::terminal_retention_ticket::prepare(
        {}, { ggml_sycl::mem_handle::from_direct(&marker, GGML_LAYOUT_AOS, false) });
    sycl::event terminal = q.submit([&](sycl::handler & cgh) { cgh.host_task([]() {}); });
    ticket.mark_submitted(q);

    // Reuse the exact vector address for a later recording attempt. Pointer-only
    // identity would incorrectly publish this old ticket into the new graph.
    ggml_sycl::set_graph_retained_handle_sink(nullptr);
    ggml_sycl::set_graph_retained_handle_sink(&sink);
    bool rejected = false;
    try { ticket.commit(terminal); } catch (const std::runtime_error &) { rejected = true; }
    ggml_sycl::set_graph_retained_handle_sink(nullptr);
    terminal.wait_and_throw();
    TEST_ASSERT(rejected, "stale ticket committed into a reused graph sink");
    TEST_ASSERT(sink.empty(), "stale epoch published retained handles");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 11: failed retained publication leaves the caller's ticket active until
// the exact submission queue has drained. A concurrent graph-boundary drain
// must observe that ticket throughout the failure cleanup window.
// =============================================================================
static bool test_retained_publication_failure_is_transactional(sycl::queue & q) {
    TEST_BEGIN("retained_publication_failure_is_transactional");

    TEST_ASSERT(ggml_sycl::drain_retained_handles(true, 1000), "retainer must start empty");

    std::mutex              gate_mutex;
    std::condition_variable gate_cv;
    bool                    gate_open = false;
    sycl::event submitted = q.submit([&](sycl::handler & cgh) {
        cgh.host_task([&]() {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_cv.wait(lock, [&]() { return gate_open; });
        });
    });

    int marker = 0;
    std::vector<ggml_sycl::mem_handle> owners{
        ggml_sycl::mem_handle::from_direct(&marker, GGML_LAYOUT_AOS, false),
    };
    auto ticket = ggml_sycl::begin_retained_handle_publish();

    ggml_sycl::fail_next_retained_handle_publication_for_test();
    bool publication_threw = false;
    try {
        ggml_sycl::retain_handles_until_event_transactional(owners, submitted, ticket);
    } catch (const std::bad_alloc &) {
        publication_threw = true;
    }

    std::atomic<bool> drain_started{ false };
    std::atomic<bool> drain_done{ false };
    bool              drain_result = false;
    std::thread drainer([&]() {
        drain_started.store(true, std::memory_order_release);
        drain_result = ggml_sycl::drain_retained_handles(true, 1000);
        drain_done.store(true, std::memory_order_release);
    });
    while (!drain_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const bool blocked_before_queue_drain = !drain_done.load(std::memory_order_acquire);

    // This is the failure catch's required ordering: drain the same queue while
    // both owners and the publication ticket remain alive.
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        gate_open = true;
    }
    gate_cv.notify_all();
    q.wait_and_throw();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const bool blocked_after_queue_drain = !drain_done.load(std::memory_order_acquire);
    const bool ticket_survived_failure   = static_cast<bool>(ticket);

    ticket = {};
    drainer.join();

    TEST_ASSERT(publication_threw, "fault injection did not fail publication");
    TEST_ASSERT(ticket_survived_failure, "failed publication consumed caller-owned ticket");
    TEST_ASSERT(blocked_before_queue_drain, "concurrent drain cleared while failed-stage queue was blocked");
    TEST_ASSERT(blocked_after_queue_drain, "concurrent drain cleared before caller released its ticket");
    TEST_ASSERT(drain_result, "concurrent drain did not clear after queue drain and ticket release");

    TEST_PASS();
    return true;
}

// =============================================================================
// Shared helpers for the in-flight-kernel lease cases below (llama.cpp-goegc.1
// spec-review round).
// =============================================================================

// RAII guard for a USM allocation obtained via sycl::malloc_host/malloc_device.
// Frees on every path out of scope, including an early TEST_ASSERT return,
// instead of relying on one unconditional free at the very end of a function
// that an early return would skip.  Callers must have already drained/waited
// any in-flight work that references the pointer before letting a guard for
// it go out of scope -- this guard does not itself synchronize.
struct sycl_free_guard {
    sycl::queue * q_   = nullptr;
    void *        ptr_ = nullptr;

    sycl_free_guard() = default;

    sycl_free_guard(sycl::queue & q, void * ptr) : q_(&q), ptr_(ptr) {}

    sycl_free_guard(const sycl_free_guard &)             = delete;
    sycl_free_guard & operator=(const sycl_free_guard &) = delete;

    ~sycl_free_guard() {
        if (ptr_ && q_) {
            sycl::free(ptr_, *q_);
        }
    }
};

// RAII guard for a device VRAM arena chunk lease (arena_acquire_chunk_lease /
// arena_release_chunk_lease).  Unlike a WEIGHT-entry lease -- which the
// mem_handle wrapping it releases in its own destructor no matter how the
// function exits -- arena_acquire_chunk_lease() returns a bare int with no
// RAII of its own.  Left unguarded, an early TEST_ASSERT return between
// acquiring the lease and this file's explicit release point would leave the
// lease outstanding when the owning unified_cache's destructor runs at
// function exit; unlike the WEIGHT-entry path, that destructor's chunk-lease
// drain is UNCONDITIONAL (arena_destroy(), unified-cache.cpp ~20585-20601):
// it is not gated by GGML_SYCL_STRICT_LEASES, and it does not merely warn --
// it waits up to 5s and then GGML_ASSERTs, aborting the whole test binary
// (every later TEST_* in main(), not just this one case).  release_now() is
// idempotent so the deliberate release point in the case below and the
// destructor's own call cannot double-release.
struct arena_chunk_lease_guard {
    ggml_sycl::unified_cache * cache_     = nullptr;
    int                        chunk_idx_ = -1;

    arena_chunk_lease_guard() = default;

    arena_chunk_lease_guard(ggml_sycl::unified_cache & cache, int chunk_idx) : cache_(&cache), chunk_idx_(chunk_idx) {}

    arena_chunk_lease_guard(const arena_chunk_lease_guard &)             = delete;
    arena_chunk_lease_guard & operator=(const arena_chunk_lease_guard &) = delete;

    void release_now() {
        if (cache_ && chunk_idx_ >= 0) {
            cache_->arena_release_chunk_lease(chunk_idx_);
        }
        cache_     = nullptr;
        chunk_idx_ = -1;
    }

    ~arena_chunk_lease_guard() { release_now(); }
};

// If drain_retained_handles() does not report success on its first bounded
// wait, the retained copy it was waiting on may still be queued (its wait is
// itself bounded -- mem-handle.cpp's drain loop -- so a timeout there is not
// proof the copy was dropped).  A caller may NOT simply return in that case:
// the queued copy holds a raw pointer into that caller's stack-local cache
// (mem-handle.cpp ~1116-1131), and returning while it is still queued lets a
// later drain dereference that pointer after the cache is destroyed --
// llama.cpp-goegc.1 finding #3, the same use-after-free class as finding #4,
// one level further out: finding #4 was about an early TEST_ASSERT skipping
// the drain entirely; this is about the drain call itself not being trusted
// on a single non-success result.
//
// The event this file's cases wait via ev.wait_and_throw() before calling
// this proves the kernel itself has completed, so a genuine indefinite hang
// here is not expected -- this is a belt-and-braces guard against a spurious
// wakeup or a future change to drain_retained_handles()'s contract, not a
// path measured to occur.  Retrying is cheap (each retry is itself a bounded
// wait); giving up and returning is not an option once the first wait has
// been tried, so the only two outcomes are "eventually drained" and a loud,
// deterministic abort -- never a silent return with a dangling reference.
static void require_fully_drained_or_abort(const char * case_name, bool already_drained) {
    bool drained = already_drained;
    for (int attempt = 0; !drained && attempt < 5; ++attempt) {
        drained = ggml_sycl::drain_retained_handles(true, 10000);
    }
    if (!drained) {
        fprintf(stderr,
                "[ABORT] %s: drain_retained_handles did not clear after repeated retries although the kernel "
                "event it was waiting on already completed; a retained handle would otherwise be returned to "
                "the caller still queued against a cache object about to be destroyed (use-after-free) -- "
                "aborting instead of returning\n",
                case_name);
        std::fflush(stderr);
        std::abort();
    }
}

// Both in-flight-kernel cases below mint a device-side sycl::atomic_ref over
// memory_scope::system so a GPU kernel can observe a host-USM store without a
// queue round-trip.  This file is the ONLY place in the SYCL backend that
// does so (verified: `grep -rn 'memory_scope::system' ggml/src/ggml-sycl/`
// matches only the two cases below).  Not every device/backend combination is
// guaranteed to support a GPU-side atomic observing a host-USM write -- query
// before relying on it instead of assuming.
static bool device_supports_system_scope_atomics(const sycl::device & dev) {
    if (!dev.has(sycl::aspect::usm_atomic_host_allocations)) {
        return false;
    }
    const auto caps = dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
    return std::find(caps.begin(), caps.end(), sycl::memory_scope::system) != caps.end();
}

// Iteration bound for the device-side gate spin in both cases below.
//
// The ORIGINAL bound here was 400,000,000 -- an ITERATION count, not a time
// bound.  A system-scope atomic load against host USM is not a cheap
// device-local load; a stuck gate could cost up to ~1 microsecond per
// iteration on comparable hardware.  At that per-iteration cost 400,000,000
// iterations is up to ~400 SECONDS worst case -- minutes, not the "ends
// quickly" the old comment implied -- against a ctest TIMEOUT of 120s for
// this whole binary (CMakeLists.txt, the `add_test(NAME mem-handle-eviction
// ...)` / `set_tests_properties(... TIMEOUT 120 ...)` pair).  A run that hit
// that worst case would be SIGTERMed by ctest with a kernel still resident on
// the card: the documented reboot-only wedge, not a clean failure.
//
// 2,000,000 iterations bounds the same worst case at ~2 seconds
// (2,000,000 x 1us), comfortably under both a 10s per-case design target and
// the 120s binary-wide TIMEOUT, while still large enough that a healthy run
// (host opens the gate within milliseconds -- see the wall-clock check in
// each case below) never gets close to it.
static constexpr uint32_t kInFlightKernelSpinBound = 2000000u;

// =============================================================================
// Test: in-flight kernel + event-bound lease blocks eviction (llama.cpp-goegc.1,
// residual case; absorbs llama.cpp-32dg8.15.3).
//
// The kernel-args-in-flight hazard: a kernel has been submitted with entry A's
// raw VRAM pointer baked into its argument buffer, and an eviction runs before
// the kernel's event completes.  The contract (Ruling 2, canonical §12.6) is
// that the submitter retains a WEIGHT-lease handle copy until the event
// completes (retain_handles_until_event), and evict_one() skips any entry whose
// in_use_count > 0 -- so eviction is DEFERRED, never a wait() and never a free.
//
// Shape:
//   1. ensure_cached(A) -> READY; acquire_weight_lease(A) and package it into a
//      WEIGHT-kind mem_handle (from_weight_lease_locked transfers the bump).
//   2. Submit a kernel that spins on a host-USM gate, then sums A's bytes into
//      an output buffer.
//   3. retain_handles_until_event({copy of handle}, event); drop the local
//      handle.  The retained copy is now the ONLY lease.
//   4. cache.evict() must not remove A: get_weight_ptr(A) still resolves to the
//      same pointer and cache_generation() is unchanged.
//   5. Open the gate; drain_retained_handles(true) waits for the event and
//      releases the copy.  The kernel must have read A's ORIGINAL bytes.
//   6. Now cache.evict() removes A and bumps the generation -- proving step 4
//      was the lease, not some unrelated ineligibility.
//
// Steps 4 and 6 together are the discriminating pair: 6 is the positive control
// that shows the eviction path WOULD have freed A absent the retained lease.
//
// What step 4 does NOT prove by itself: that the kernel was still executing
// at the instant evict() ran.  out_host[1] < spin_bound is satisfied equally
// by "the kernel spun and was interrupted by the gate" and by "the kernel had
// not even started yet"; neither disproves "the kernel had already finished".
// The real discriminator this test relies on is structural, not timing: step
// 4 finding A still resolvable AND step 5 reading back A's correct original
// bytes can only both hold if the retained lease was live in in_use_count at
// the moment evict() ran -- that is exactly what evict_one()'s in_use_count
// check gates, independent of the kernel's precise progress.  As an
// ADDITIONAL, non-asserted, informational discriminator, this case also
// samples the kernel event's execution status immediately before the
// eviction attempt and reports whether it observed "not yet complete" -- a
// genuine positive when it holds, but its absence does not indict the test
// (a fast enough run can legitimately finish before that sample is taken),
// so the case does not gate on it.
//
// Scope note: this exercises the unified_cache entry lease (in_use_count) and
// (see test_in_flight_kernel_arena_chunk_lease_blocks_replan below) the
// device-side VRAM arena chunk lease.  The host-pinned chunk lease
// (pinned_chunk_pool) is NOT exercised by either case, and it is a
// structural gap, not an oversight: pinned_chunk_pool::chunk_has_leases()
// (pinned-pool.cpp:1023) has zero production call sites anywhere in the
// backend (verified: `grep -rn chunk_has_leases ggml/src/ggml-sycl/` matches
// only its own declaration and definition) -- the only place a host-pinned
// chunk lease is actually consulted is ~pinned_chunk_pool()'s destructor
// (pinned-pool.cpp:141-163), which holds the pool's own mutex_ for the ENTIRE
// destructor body, including the up-to-5s wait_for_chunk_drain_or_assert()
// spin (pinned-pool.cpp:1038-1057, "Caller (destructor) already holds
// mutex_").  release_chunk_lease() -- the only sanctioned way to drop a
// lease (pinned-pool.cpp:1001-1010) -- itself needs that same mutex_.  A test
// that tried to release a host-pinned chunk lease concurrently with pool
// destruction could therefore never observe the "refuse-then-succeed" shape
// this file uses everywhere else: the release call cannot make progress
// while the destructor holds mutex_, so the only reachable outcomes are
// "released before destruction starts" (which proves nothing about a
// live-lease refusal) or "destructor times out and GGML_ASSERTs after 5s" (a
// deliberate process abort a unit test may not trigger).  There is no third,
// safely-testable reclaim path for the host-pinned chunk lease to substitute
// -- unlike the VRAM arena side, where arena_chunk_has_leases() gates
// ensure_planned_arena_zones()'s rebuild-refusal (unified-cache.cpp
// ~3597-3612) via a plain atomic load, with no lock a release call needs.
// =============================================================================
static bool test_in_flight_kernel_lease_blocks_eviction(sycl::queue & q) {
    TEST_BEGIN("in_flight_kernel_lease_blocks_eviction");
    if (!device_supports_system_scope_atomics(q.get_device())) {
        TEST_SKIP(
            "device lacks system-scope USM atomic support (aspect::usm_atomic_host_allocations and/or "
            "memory_scope::system not in atomic_memory_scope_capabilities)");
    }

    constexpr size_t         entry_bytes = 4 * 1024;
    constexpr size_t         budget      = 16 * 1024 * 1024;
    const uint32_t           spin_bound  = kInFlightKernelSpinBound;
    ggml_sycl::unified_cache cache(q, budget);
    const int                dev = ggml_sycl_get_device_id_from_queue(q);

    void * src_host = sycl::malloc_host(entry_bytes, q);
    TEST_ASSERT(src_host != nullptr, "malloc_host for src should succeed");
    sycl_free_guard src_host_guard(q, src_host);
    std::memset(src_host, 0x5A, entry_bytes);
    ggml_sycl_cache_id key = make_test_cache_id(900, 1, entry_bytes);

    void * ptr = cache.ensure_cached(key, src_host, entry_bytes, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1,
                                     GGML_LAYOUT_AOS, false);
    TEST_ASSERT(ptr != nullptr, "ensure_cached(A) should succeed");
    q.wait();
    TEST_ASSERT(cache.get(key, GGML_LAYOUT_AOS) == ptr, "cache.get() should drive A to READY");

    // Package the lease into a handle whose dtor releases it (Ruling 2: the
    // handle, not the caller, is the release point).
    auto lease = cache.acquire_weight_lease(key);
    TEST_ASSERT(lease.ptr == ptr && lease.entry != nullptr, "acquire_weight_lease(A) must resolve with an entry");
    ggml_sycl::mem_handle handle = ggml_sycl::mem_handle::from_weight_lease_locked(key, dev, lease.ptr, lease.layout,
                                                                                   lease.on_device, lease.entry);
    TEST_ASSERT(handle.valid(), "lease-backed handle must be valid");
    TEST_ASSERT(lease.entry->in_use_count.load() == 1, "exactly one lease outstanding after packaging");

    // Gate in host USM so the host can open it while the kernel is resident.
    int *      gate = sycl::malloc_host<int>(1, q);
    uint32_t * out  = sycl::malloc_device<uint32_t>(2, q);
    TEST_ASSERT(gate != nullptr && out != nullptr, "gate/out allocation should succeed");
    sycl_free_guard gate_guard(q, gate);
    sycl_free_guard out_guard(q, out);
    *gate = 0;
    q.memset(out, 0, 2 * sizeof(uint32_t)).wait();

    const unsigned char * data        = static_cast<const unsigned char *>(ptr);
    const auto            submit_time = std::chrono::steady_clock::now();
    sycl::event           ev          = q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                sycl::access::address_space::global_space>
                     g(*gate);
            // Bounded spin: a stuck gate ends the kernel instead of wedging the
            // card (GT reset cascade); the host-side assertion then fails loudly.
            uint32_t spins = 0;
            while (g.load() == 0 && spins < spin_bound) {
                ++spins;
            }
            uint32_t sum = 0;
            for (size_t i = 0; i < entry_bytes; ++i) {
                sum += data[i];
            }
            out[0] = sum;
            out[1] = spins;
        });
    });

    // Hand the lease to the event: this is the production shape for every
    // async submit (canonical §12.6).  The local handle is dropped so the
    // retained copy is the only thing keeping in_use_count > 0.
    {
        std::vector<ggml_sycl::mem_handle> retained;
        retained.push_back(handle);
        ggml_sycl::retain_handles_until_event(std::move(retained), ev);
    }
    handle                         = ggml_sycl::mem_handle{};
    const uint32_t leases_retained = lease.entry->in_use_count.load();

    // Step 4: eviction while the kernel is in flight must be deferred.
    // Informational-only discriminator (see the block comment above): sample
    // whether the kernel event still reports "not complete" right before the
    // eviction attempt.  Not asserted -- see that comment for why.
    const bool kernel_apparently_still_running =
        ev.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete;
    const uint64_t gen_before = ggml_sycl::cache_generation();
    const size_t   freed_live = cache.evict(entry_bytes * 2);
    const uint64_t gen_live   = ggml_sycl::cache_generation();
    bool           still_there;
    void *         ptr_live;
    {
        auto r      = cache.get_weight_ptr(key);
        still_there = static_cast<bool>(r);
        ptr_live    = r.ptr;
    }
    // Open the gate BEFORE any assertion returns, so a failed assertion cannot
    // leave a spinning kernel behind.  Everything from kernel submission to
    // here is synchronous, cheap, host-only bookkeeping (no waits, no
    // sleeps), so this happens within milliseconds on every path through this
    // function -- the wall-clock check below turns that guarantee into an
    // assertion instead of leaving it implicit.
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        host_gate(*gate);
    host_gate.store(1);
    const auto gate_open_time    = std::chrono::steady_clock::now();
    const auto host_side_latency = std::chrono::duration_cast<std::chrono::milliseconds>(gate_open_time - submit_time);

    // Step 5: the event completes, the drain worker drops the retained copy.
    // This -- along with ev.wait_and_throw() and require_fully_drained_or_abort()
    // -- MUST happen before any TEST_ASSERT below can return, not after:
    // retain_handles_until_event has already queued a copy of `handle` that
    // holds a raw lease.entry pointer into this function's stack-local
    // `cache`.  An early return here would leave that copy queued against a
    // cache object about to be destroyed on return, and the drain worker
    // would later dereference the dangling entry while unrelated later
    // tests are running (llama.cpp-goegc.1 finding #4: mem-handle.cpp
    // ~1120-1129 is where that dereference happens).  A single
    // drain_retained_handles() call reporting failure is not itself proof
    // the copy is safe to leave queued (finding #3) -- see
    // require_fully_drained_or_abort()'s comment for why that path aborts
    // rather than returns.
    const bool drained_first_attempt = ggml_sycl::drain_retained_handles(true, 10000);
    ev.wait_and_throw();
    require_fully_drained_or_abort("in_flight_kernel_lease_blocks_eviction", drained_first_attempt);

    uint32_t out_host[2] = { 0, 0 };
    q.memcpy(out_host, out, sizeof(out_host)).wait();
    const uint32_t leases_after_drain = lease.entry->in_use_count.load();

    // From here on it is safe to assert and return early: no retained handle
    // is outstanding regardless of which assertion below fails --
    // require_fully_drained_or_abort() above guarantees that, or the process
    // has already aborted rather than reaching here.
    TEST_ASSERT(host_side_latency < std::chrono::seconds(5),
                "host must open the gate within a few seconds of kernel submission on every path");
    TEST_ASSERT(leases_retained == 1, "retained copy must hold exactly one lease");
    TEST_ASSERT(freed_live == 0, "evict() must free nothing while the entry is leased by in-flight work");
    TEST_ASSERT(still_there && ptr_live == ptr, "A must still resolve to the same pointer during in-flight work");
    TEST_ASSERT(gen_live == gen_before, "cache_generation must not bump while eviction is deferred");
    TEST_ASSERT(out_host[1] < spin_bound, "kernel must have observed the gate open (not the spin bound)");
    TEST_ASSERT(out_host[0] == 0x5Au * entry_bytes, "kernel must have read A's original bytes, not freed memory");
    TEST_ASSERT(leases_after_drain == 0, "drain must have released the retained lease");
    if (!kernel_apparently_still_running) {
        fprintf(stderr,
                "  [NOTE] in_flight_kernel_lease_blocks_eviction: kernel event already reported complete at "
                "the eviction-attempt sample; the informational discriminator did not fire this run\n");
    }

    // Step 6: positive control -- with the lease gone the same eviction frees A.
    (void) cache.evict(entry_bytes * 2);
    (void) cache.finalize_evictions();
    const uint64_t gen_after = ggml_sycl::cache_generation();
    TEST_ASSERT(gen_after > gen_before, "positive control: eviction must proceed once the lease is released");
    TEST_ASSERT(!cache.get_weight_ptr(key), "positive control: A must be gone after the lease is released");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test: in-flight kernel + device VRAM arena chunk lease blocks a zone re-plan
// (llama.cpp-goegc.1 spec-review round, finding #1 -- the device half of "if
// host-pinned chunk leases and device arena chunk leases are separate code
// paths, repeat the case for both").
//
// Unlike the WEIGHT entry lease above, the arena CHUNK_LEASE counter
// (arena_chunks_[i].lease_count) does not gate evict_one() at all -- it gates
// a DIFFERENT reclaim path: ensure_planned_arena_zones()'s in-place rebuild,
// which refuses when has_chunk_leases is true (unified-cache.cpp ~3597-3612)
// and otherwise tears the arena down and re-reserves it at the newly planned
// zone sizes.  This case exercises that path directly.
//
// mem_handle::from_chunk_ptr() (mem-handle.cpp:640) is the production entry
// point that mints a CHUNK_LEASE handle, but it resolves its owning cache via
// get_existing_unified_cache_for_device(device) -- the per-device GLOBAL
// registry, not whichever unified_cache instance a caller happens to hold.
// Every case in this file (including the WEIGHT-lease case above) constructs
// its own LOCAL, unregistered unified_cache, exactly so one case's state
// cannot leak into another -- so from_chunk_ptr() cannot be used here.
// Instead this case calls arena_acquire_chunk_lease() / arena_release_
// chunk_lease() directly: this is the exact same primitive from_chunk_ptr()
// itself calls once it has a cache (mem-handle.cpp:673), so holding it
// directly -- without the mem_handle wrapper a globally-registered cache
// would allow -- still exercises production's chunk-lease bookkeeping, not a
// substitute for it.
//
// Shape (mirrors the WEIGHT-lease case's steps 2-6):
//   1. cache.arena_base() is a pointer into arena_chunks_[0] once the arena
//      is actually reserved.  That reservation is NOT automatic at any
//      budget: arena_reserve()'s single-chunk admission check (unified-
//      cache.cpp ~19222-19236) refuses unless
//        alloc_size >= tail_bytes + k_min_shared_bytes
//      where tail_bytes = oneDNN + RUNTIME + SCRATCH zone bytes (defaults
//      256 + 512 + 512 MB unless GGML_SYCL_ONEDNN/RUNTIME/COMPUTE_ARENA_MB
//      override them) and k_min_shared_bytes = 16 MB -- i.e. roughly
//      1296 MB, not this file's usual 16 MB per-entry-cache budget (the
//      N-chunk fallback at ~19348/~19367-19372 refuses on the same shape
//      of arithmetic against per_chunk_cap).  A prior version of this case
//      used the 16 MB budget and silently [SKIP]ped on every run as a
//      result -- caught in spec review.  budget below is sized with ~750 MB
//      of headroom over that ~1296 MB floor (live zone sizes are read back
//      via zone_capacity() rather than assumed, so this also tolerates an
//      externally-set GGML_SYCL_*_ARENA_MB or a planner-raised oneDNN
//      zone).  arena_active() is asserted, not skipped, below: with this
//      budget a false reading is a test bug, not an environment condition.
//   2. arena_acquire_chunk_lease(base) leases that chunk; submit a kernel
//      that spins on a host-USM gate, then reads a few bytes at the base
//      pointer into an output buffer -- proof the chunk is in use by
//      in-flight device work, mirroring the WEIGHT-lease kernel.
//   3. Raise GGML_SYCL_RUNTIME_ARENA_MB by 1 MB over THIS cache's own
//      current zone_capacity(RUNTIME) -- independent of whatever any
//      earlier case in this binary already left in the environment, and
//      deliberately a 1 MB step rather than a large one: the positive
//      control in step 5 re-runs arena_reserve() at this raised target
//      after arena_destroy(), so it is bound by the SAME admission
//      arithmetic as step 1 (tail_bytes + k_min_shared_bytes <= budget) --
//      a large step could blow that budget and turn the positive control
//      red for a reason unrelated to the lease.  Then call
//      ensure_planned_arena_zones() again while the chunk lease (and the
//      kernel) are still live: must return false (refused).
//   4. Open the gate; wait the event; release the chunk lease.
//   5. Call ensure_planned_arena_zones() again with the same raised target:
//      must now return true -- the positive control showing step 3's refusal
//      was the lease, not some unrelated ineligibility.
// =============================================================================
static bool test_in_flight_kernel_arena_chunk_lease_blocks_replan(sycl::queue & q) {
    TEST_BEGIN("in_flight_kernel_arena_chunk_lease_blocks_replan");
    if (!device_supports_system_scope_atomics(q.get_device())) {
        TEST_SKIP(
            "device lacks system-scope USM atomic support (aspect::usm_atomic_host_allocations and/or "
            "memory_scope::system not in atomic_memory_scope_capabilities)");
    }

    // See the block comment above for the arena_reserve() admission
    // arithmetic this budget must clear (~1296 MB floor with default zone
    // sizes); 2048 MB leaves ~750 MB of margin.
    constexpr size_t         budget     = 2048ull * 1024 * 1024;
    const uint32_t           spin_bound = kInFlightKernelSpinBound;
    ggml_sycl::unified_cache cache(q, budget);
    TEST_ASSERT(cache.arena_active(),
                "arena must be active at this budget (see the admission arithmetic in the block comment above); "
                "a false reading here is a test bug, not an environment condition");

    void * arena_ptr = cache.arena_base();
    TEST_ASSERT(arena_ptr != nullptr, "an active arena must have a base pointer");

    const int chunk_idx = cache.arena_acquire_chunk_lease(arena_ptr);
    TEST_ASSERT(chunk_idx >= 0, "arena_acquire_chunk_lease must find arena_base() inside a chunk");
    // Guard from here on: see the class comment above for why an unguarded
    // early return in this window is an abort hazard, not just a leak.
    arena_chunk_lease_guard chunk_lease_guard(cache, chunk_idx);
    TEST_ASSERT(cache.arena_chunk_has_leases(chunk_idx), "chunk lease must be visible immediately after acquire");

    int *      gate = sycl::malloc_host<int>(1, q);
    uint32_t * out  = sycl::malloc_device<uint32_t>(2, q);
    TEST_ASSERT(gate != nullptr && out != nullptr, "gate/out allocation should succeed");
    sycl_free_guard gate_guard(q, gate);
    sycl_free_guard out_guard(q, out);
    *gate = 0;
    q.memset(out, 0, 2 * sizeof(uint32_t)).wait();

    constexpr size_t      read_bytes  = 4096;  // arbitrary "in use" touch, well inside the ~1GB+ default arena
    const unsigned char * data        = static_cast<const unsigned char *>(arena_ptr);
    const auto            submit_time = std::chrono::steady_clock::now();
    sycl::event           ev          = q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                sycl::access::address_space::global_space>
                     g(*gate);
            uint32_t spins = 0;
            while (g.load() == 0 && spins < spin_bound) {
                ++spins;
            }
            uint32_t sum = 0;
            for (size_t i = 0; i < read_bytes; ++i) {
                sum += data[i];
            }
            out[0] = sum;
            out[1] = spins;
        });
    });

    // Force the next ensure_planned_arena_zones() call to see insufficient
    // zones regardless of what any earlier case in this binary left in the
    // environment: target strictly above THIS cache's own current RUNTIME
    // zone capacity, not a hardcoded absolute value.  +1 MB, not a larger
    // step: see the block comment above this function -- the positive
    // control at step 5 must clear the SAME tail_bytes + k_min_shared_bytes
    // <= budget admission check this cache's own construction did, and a
    // large step risks failing that arithmetic against `budget` for a
    // reason unrelated to the lease.
    const size_t      current_runtime_mb = cache.zone_capacity(ggml_sycl::vram_zone_id::RUNTIME) / (1024 * 1024);
    const size_t      target_runtime_mb  = current_runtime_mb + 1;
    const char *      old_runtime_env    = std::getenv("GGML_SYCL_RUNTIME_ARENA_MB");
    const bool        had_runtime_env    = old_runtime_env != nullptr;
    const std::string saved_runtime_env  = old_runtime_env ? old_runtime_env : "";
    char              target_buf[32];
    std::snprintf(target_buf, sizeof(target_buf), "%zu", target_runtime_mb);
#if defined(_WIN32)
    _putenv_s("GGML_SYCL_RUNTIME_ARENA_MB", target_buf);
#else
    setenv("GGML_SYCL_RUNTIME_ARENA_MB", target_buf, 1);
#endif

    // Attempt the reclaim path while the chunk lease (and the kernel) are
    // live: must be refused.
    const bool replanned_while_leased = cache.ensure_planned_arena_zones();

    // Open the gate BEFORE any assertion returns -- same reasoning as the
    // WEIGHT-lease case above: a stuck kernel here would hold arena chunk 0's
    // pointer live, and this cache's destructor (shutdown_resources())
    // GGML_ABORTs the whole process if it cannot drain live allocator owners
    // at teardown.
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        host_gate(*gate);
    host_gate.store(1);
    const auto gate_open_time    = std::chrono::steady_clock::now();
    const auto host_side_latency = std::chrono::duration_cast<std::chrono::milliseconds>(gate_open_time - submit_time);

    // Wait the event and release the lease BEFORE any assertion can return
    // early -- chunk_idx keys into this function's stack-local cache's
    // arena_chunks_, so nothing here may leak past this function's return
    // with an outstanding acquire (this cache's destructor aborts the
    // process on a live allocator owner, per the comment above).  Unlike a
    // mem_handle-retained release, the chunk lease is not routed through the
    // async retained-handle queue (see the class comment above this test),
    // so releasing it is a direct, synchronous call once the event completes
    // -- arena_release_chunk_lease() is a plain atomic decrement with no
    // bounded internal wait of its own, unlike drain_retained_handles().
    // release_now() therefore has no "reported failure but may still be
    // queued" outcome to retry or abort on (llama.cpp-goegc.1 finding #3 is
    // specific to drain_retained_handles()'s bounded-wait contract, which
    // this path does not go through), so this case has no equivalent window
    // to guard beyond the RAII guard already in place above.
    ev.wait_and_throw();
    chunk_lease_guard.release_now();
    const bool chunk_lease_cleared = !cache.arena_chunk_has_leases(chunk_idx);

    uint32_t out_host[2] = { 0, 0 };
    q.memcpy(out_host, out, sizeof(out_host)).wait();

    const bool replanned_after_release = cache.ensure_planned_arena_zones();

    // Restore the environment for later cases in this binary.
#if defined(_WIN32)
    _putenv_s("GGML_SYCL_RUNTIME_ARENA_MB", had_runtime_env ? saved_runtime_env.c_str() : "");
#else
    had_runtime_env ? (void) setenv("GGML_SYCL_RUNTIME_ARENA_MB", saved_runtime_env.c_str(), 1) :
                      (void) unsetenv("GGML_SYCL_RUNTIME_ARENA_MB");
#endif

    TEST_ASSERT(host_side_latency < std::chrono::seconds(5),
                "host must open the gate within a few seconds of kernel submission on every path");
    TEST_ASSERT(out_host[1] < spin_bound, "kernel must have observed the gate open (not the spin bound)");
    TEST_ASSERT(!replanned_while_leased,
                "ensure_planned_arena_zones() must refuse an in-place rebuild while the chunk is leased by "
                "in-flight work");
    TEST_ASSERT(chunk_lease_cleared, "arena_release_chunk_lease must clear the lease once the kernel event completes");
    TEST_ASSERT(replanned_after_release,
                "positive control: ensure_planned_arena_zones() must succeed once the chunk lease is released");

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
    // Initialize the backend's logical-to-physical device map before choosing
    // the fixture queue.  A raw gpu_selector queue is not authoritative when
    // GGML_SYCL_DEVICE remaps logical device 0 (for example, raw B50 device 1
    // becomes backend owner 0).  Publishing handles from a cache built on that
    // raw queue then labels them with a different owner than the active queue.
    // Keep production's exact wrong-device checks intact: make the fixture use
    // the backend owner's device and context instead.
    const auto & sycl_info = ggml_sycl_info();
    if (sycl_info.total_gpu_count <= 0) {
        sycl_test_print_skip("mem_handle / eviction lifecycle backend owner selection");
        return SYCL_TEST_SKIP;
    }
    auto &        owner       = ggml_sycl_get_gpu_device(0);
    sycl::queue & owner_queue = owner.default_queue();
    sycl::queue   q(owner_queue.get_context(), owner_queue.get_device(), sycl::property::queue::in_order{});
    const int     queue_device = ggml_sycl_get_device_id_from_queue(q);

    fprintf(stderr, "Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "Backend owner: logical GPU 0, queue device %d\n", queue_device);
    fprintf(stderr, "-------------------------------------------\n");
    if (queue_device != 0) {
        fprintf(stderr, "FAILED: backend owner queue did not round-trip to logical GPU 0\n");
        return 1;
    }

    bool all_passed = true;
    all_passed &= test_direct_handle_stable_across_bumps();
    all_passed &= test_get_weight_ptr_resolves_direct_staged_entry(q);
    all_passed &= test_lease_and_plain_lookup_agree(q);
    all_passed &= test_explicit_evict_bumps_gen_and_removes_entry(q);
    all_passed &= test_reinsert_after_evict_recovers_lookup(q);
    all_passed &= test_async_eviction_finalize_bumps_gen(q);
    all_passed &= test_in_flight_kernel_lease_blocks_eviction(q);
    all_passed &= test_in_flight_kernel_arena_chunk_lease_blocks_replan(q);
    all_passed &= test_expert_retirement_with_live_lease(q);
    all_passed &= test_expert_publication_retirement_linearization(q);
    all_passed &= test_host_publication_fault_is_transactional(q);
    all_passed &= test_device_publication_fault_phases(q);
    all_passed &= test_owner_failure_prevents_raw_stage_use(q);
    all_passed &= test_abort_cleanup_status_is_observable(q);
    all_passed &= test_retired_status_query_failure_is_deferred(q);
    all_passed &= test_terminal_retention_ticket_state_machine(q);
    all_passed &= test_mmvq_rmsnorm_second_batch_failure_drains_first(q);
    all_passed &= test_graph_recording_epoch_reuse_rejected(q);
    all_passed &= test_retained_publication_failure_is_transactional(q);

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n", g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    if (g_tests_skipped > 0) {
        // A skip means an environment condition (e.g. no system-scope USM
        // atomic support) kept a case from running its logic at all -- that
        // case proved nothing, in either direction.  Reporting this as
        // "ALL TESTS PASSED" would read as an unqualified green when some
        // fraction of the suite is actually unverified this run.
        fprintf(stderr, "PASSED WITH SKIPS\n");
        return 0;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
