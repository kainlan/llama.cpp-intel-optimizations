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

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
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
    ggml_sycl::unified_cache cache(q, 16 * 1024 * 1024);
    void * src = sycl::malloc_host(bytes * 2, q);
    TEST_ASSERT(src != nullptr, "malloc_host for fault phases failed");
    std::memset(src, 0x61, bytes * 2);

    const ggml_sycl::expert_fault_phase single_phases[] = {
        ggml_sycl::expert_fault_phase::SINGLE_AFTER_ALLOC,
        ggml_sycl::expert_fault_phase::SINGLE_BEFORE_COMMIT,
    };
    for (size_t i = 0; i < 2; ++i) {
        const auto key = make_test_cache_id(710 + i, 20 + i, bytes);
        ggml_sycl::unified_cache_fail_next_expert_phase_for_test(single_phases[i]);
        const auto failed = cache.direct_stage_expert(
            key, src, bytes, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr);
        TEST_ASSERT(!failed.ok, "faulted single publication committed");
        TEST_ASSERT(cache.lookup_expert(key) == nullptr, "faulted single publication leaked a mirror");
        TEST_ASSERT(cache.register_host_expert(key, src, bytes, GGML_LAYOUT_AOS),
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
            keys, src, bytes * 2, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr);
        TEST_ASSERT(!failed.ok, "faulted bulk publication committed");
        TEST_ASSERT(cache.lookup_expert(keys[0]) == nullptr && cache.lookup_expert(keys[1]) == nullptr,
                    "faulted bulk publication leaked a mirror");
    }

    const auto duplicate = make_test_cache_id(730, 40, bytes);
    std::vector<ggml_sycl_cache_id> duplicates{ duplicate, duplicate };
    TEST_ASSERT(!cache.direct_stage_expert_tensor(
                     duplicates, src, bytes * 2, bytes, GGML_LAYOUT_AOS, nullptr, nullptr, &q, nullptr).ok,
                "bulk publication accepted duplicate keys");
    TEST_ASSERT(cache.lookup_expert(duplicate) == nullptr, "duplicate bulk request partially published");

    q.wait_and_throw();
    cache.process_deferred_frees_public();
    TEST_ASSERT(cache.validate(), "device fault phases left inconsistent bookkeeping");
    sycl::free(src, q);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 9: a ready-event status query exception is unknown and therefore keeps
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
// Test 10: failed retained publication leaves the caller's ticket active until
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
    all_passed &= test_expert_retirement_with_live_lease(q);
    all_passed &= test_expert_publication_retirement_linearization(q);
    all_passed &= test_host_publication_fault_is_transactional(q);
    all_passed &= test_device_publication_fault_phases(q);
    all_passed &= test_retired_status_query_failure_is_deferred(q);
    all_passed &= test_retained_publication_failure_is_transactional(q);

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
