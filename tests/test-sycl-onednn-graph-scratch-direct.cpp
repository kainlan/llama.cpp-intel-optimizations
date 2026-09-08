// Gate for llama.cpp-0oxf: the oneDNN Graph-scratch allocator's DIRECT
// (non-arena) fallback must BOUND its outstanding bytes, REUSE freed buffers
// instead of repeatedly round-tripping through unified_alloc(), and FAIL
// LOUDLY rather than silently handing oneDNN a null scratch pointer.
//
// BACKGROUND. unified_cache::onednn_graph_scratch_alloc() serves oneDNN
// Graph's SYCL allocator callback. When a request does not fit the planned
// ONEDNN zone (measured: a 144 MB Graph-scratch request at n_kv=8192 on
// Mistral 7B Q4_0 -- see the ticket), it falls back to a real, individually
// mem_handle-owned unified_alloc() (the "DIRECT" path).
//
// TWO DESIGNS, AND WHY THIS FILE TESTS THE SECOND ONE. The first version of
// this fix deferred a freed DIRECT buffer's physical release to the shared
// background drain worker (mem-handle.cpp retained_handle_drain_loop) via
// retain_handles_until_event(), and bounded outstanding bytes by waiting for
// that worker to catch up. That was superseded (ticket verification 2, "item
// 3 as built is a no-op" plus the mechanics finding that follows it) once it
// became clear the ONEDNN zone cannot grow after weights load at all
// (arena_reserve()'s existing-arena branch ignores the zone-size arguments
// it is passed) -- so a model whose real n_ctx makes the shape-derived floor
// too small to plan for at load time takes the DIRECT path on EVERY request
// of that shape, not just an occasional one, and a background-drain-and-wait
// design just means repeatedly paying a zeMemAllocDevice round trip for
// buffers of the identical size. The current design instead PARKS a freed
// DIRECT buffer in a size-bucketed reuse pool (event-gated, exactly like the
// superseded design's release gating) and hands it back to the next request
// of the SAME size instead of a fresh allocation -- the common case, since
// ubatches at a fixed n_ubatch/n_ctx-so-far shape repeat across `-r N`
// benchmark reps and across decode steps. The bound from the superseded
// design (GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB) still applies: a
// pooled-but-idle buffer's bytes stay charged against it (it is still real
// resident VRAM), so a request that cannot be served from the pool AND would
// exceed the cap evicts (real release) completed pool entries of OTHER sizes
// until it fits, waiting (bounded) for an in-flight entry to complete first
// if none are immediately evictable. Two further bounds, not directly
// exercised by this file (both are exercised via unified-cache.cpp's own
// source-contract gate, tests/test-sycl-onednn-graph-allocator-source.py,
// and via reading the counters this file DOES check): each size bucket also
// caps at onednn_graph_scratch_pool_depth_per_size() entries (default 8,
// env-overridable) so a workload walking many distinct sizes cannot grow the
// pool without limit even while every individual size stays under the byte
// cap; and the pool is cleared (real release) at cache teardown and at the
// same point arena_reserve() reclaims the KV/RUNTIME zones for a new
// context, so a pooled buffer cannot outlive the context it belongs to.
//
// This test asserts four properties:
//
//   (a) POOL REUSE: freeing a DIRECT buffer and immediately requesting the
//       SAME size again must be served from the pool -- no fresh
//       unified_alloc() round trip, observable as
//       onednn_graph_scratch_pool_hit_count() incrementing.
//
//   (b) BOUNDED EVICTION/WAIT: driving outstanding DIRECT bytes over an
//       artificially small cap (via GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB)
//       with a DIFFERENT size than what is already pooled (so pool reuse in
//       (a) cannot short-circuit this path) must make the allocator wait for
//       the pooled entry's release event to complete and then evict it,
//       rather than allocating past the cap -- observable as the second
//       allocation taking a substantial fraction of the release delay and
//       onednn_graph_scratch_direct_wait_count() incrementing.
//
//   (c) LOUD FAILURE: an allocation that is genuinely exhausted (both the
//       initial attempt and the drain-and-retry fail) must never return a
//       null scratch pointer to oneDNN -- it must log the failure and abort.
//       Actually triggering GGML_ABORT() in-process would crash this test
//       binary and, per this host's own hard-won rules (CLAUDE.md: forking a
//       process that has touched the SYCL/Level-Zero runtime is a known hang
//       hazard here), a fork()-based death-check is not a safe way to work
//       around that on this machine. So the fix ships two ALWAYS-compiled
//       (never gated behind a separate _TESTING object-library build --
//       see the comment on their declarations in unified-cache.hpp for why)
//       test-only hooks that let this test observe "this would have aborted"
//       without crashing: ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail()
//       forces the DIRECT unified_alloc() to fail without touching the GPU,
//       and ggml_sycl_test_onednn_graph_scratch_suppress_abort() turns the
//       GGML_ABORT() into a latched flag plus a nullptr return. This test
//       asserts the ERROR log line, the latch, and the nullptr return.
//
// RED-FIRST NOTE. All three properties are RED against the pre-0oxf code:
// (a) and (b) have no pool, cap, or wait at all (every DIRECT allocation is
// a fresh unified_alloc() with no bound), and (c) had no abort hook to
// suppress -- the pre-fix function simply returned nullptr with
// `req.suppress_failure_log = true`, so the log capture in this test would
// find nothing and the "abort triggered" latch would not exist.
//
//   (d) RECLAIM SAFETY (spec review finding #1, BLOCKING): reclaiming the
//       pool (onednn_graph_scratch_reclaim_pool(), reached from cache
//       teardown, arena_reserve()'s context-reclaim branch, and
//       ggml_backend_sycl_set_runtime_context()) must not destruct a pooled
//       entry whose release event has not yet completed -- doing so would
//       return that VRAM to the general unified_alloc() pool while a queued
//       SDPA kernel might still be reading it, the exact fault class this
//       whole ticket exists to close. Two of the three reclaim call sites do
//       NOT drain the queue first (only cache teardown does), so this
//       property must hold on its own. Exercised by parking an entry with a
//       real, unwaited slow-release event, reclaiming while it is still
//       incomplete, and observing (a) the process stays healthy, (b) the
//       reclaim counted it as an eviction, and (c) a fresh request of the
//       same size misses the pool -- see the test's own comment for why
//       this behavioral proxy is the strongest property observable without
//       adding test-only introspection into mem-handle.cpp's drain worker.
//
// SKIPS (77, ctest SKIP_RETURN_CODE): no SYCL device, or GGML_SYCL_DNNL not
// compiled in (the whole onednn_graph_scratch_* subsystem is `#if
// GGML_SYCL_DNNL`-only).

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL/GGML_SYCL_DNNL not both enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

#    include <chrono>
#    include <thread>

using ggml_sycl::get_unified_cache_for_device;
using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_abort_triggered;
using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail;
using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_suppress_abort;
using ggml_sycl::unified_cache;
using ggml_sycl::unified_cache_reclaim_onednn_graph_scratch_pool;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

std::string g_captured_log;

void capture_log(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    if (text != nullptr) {
        g_captured_log += text;
    }
}

bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

// Two sizes that never fit any realistic ONEDNN zone (the zone defaults to
// 256 MB and this repository's own comments record requests up to 144 MB as
// the largest measured), so both deterministically take the DIRECT path
// regardless of what a live model's planning left the zone sized to. They
// are deliberately DIFFERENT from each other so a request for kSizeB cannot
// be silently served by a pool entry parked for kSizeA -- that would let
// pool reuse mask the eviction/wait path this file also needs to exercise.
constexpr size_t kSizeA = 300ull * 1024 * 1024;
constexpr size_t kSizeB = 320ull * 1024 * 1024;

// A release event that only completes after this many milliseconds -- long
// enough to be unambiguously distinguishable from "the wait didn't really
// wait", short enough to keep the gate fast.
constexpr int kSlowReleaseMs = 1500;

sycl::event submit_slow_release(sycl::queue & q) {
    return q.submit([&](sycl::handler & h) {
        h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(kSlowReleaseMs)); });
    });
}

// --- (a) freed buffer is reused for a same-size request ---------------------
void test_pool_reuse(unified_cache * cache) {
    printf("DIRECT path pool reuse:\n");

    sycl::queue & q = cache->get_queue();

    void * ptr1 = cache->onednn_graph_scratch_alloc(kSizeA, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the reuse setup succeeds");
    if (!ptr1) {
        return;
    }

    // Free with a REAL (not nullptr) event -- "served from the pool once its
    // event completes", not merely "served because no event was given". Wait
    // for it explicitly here (unlike test_bounded_eviction below, this test
    // is not measuring how long the allocator itself waits) so the pool
    // lookup that follows finds a genuinely event-complete entry rather than
    // racing the host_task.
    sycl::event release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr1, &release);
    release.wait();

    const size_t hits_before   = cache->onednn_graph_scratch_pool_hit_count();
    const size_t misses_before = cache->onednn_graph_scratch_pool_miss_count();
    void *       ptr2          = cache->onednn_graph_scratch_alloc(kSizeA, 256, &q);

    check(ptr2 != nullptr, "the reused allocation succeeds");
    check(ptr2 == ptr1,
          "the SAME pointer is handed back -- this is reuse, not a fresh allocation that happens to "
          "land at the same address");
    check(cache->onednn_graph_scratch_pool_hit_count() > hits_before,
          "onednn_graph_scratch_pool_hit_count() increased -- the allocator served this from the reuse pool");
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before,
          "onednn_graph_scratch_pool_miss_count() did NOT increase -- a pool hit is not also counted as a miss");

    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }
}

// --- (b) bounded eviction/wait under the DIRECT-path cap --------------------
void test_bounded_eviction(unified_cache * cache) {
    printf("DIRECT path bounded eviction/wait:\n");

    sycl::queue & q = cache->get_queue();

    void * ptr1 = cache->onednn_graph_scratch_alloc(kSizeA, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the eviction setup succeeds");
    if (!ptr1) {
        return;
    }

    // A host_task event that only completes after kSlowReleaseMs -- this is
    // the "slow-to-complete event" the fix spec calls for: ptr1's bytes stay
    // charged against the outstanding-DIRECT counter (it is parked in the
    // reuse pool, not physically released, until this event completes and
    // the pool eviction sweep observes that) for long enough that the
    // request below is guaranteed to observe the cap and enter the wait
    // loop, rather than racing a fast release.
    sycl::event slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr1, &slow_release);

    const size_t wait_count_before     = cache->onednn_graph_scratch_direct_wait_count();
    const size_t eviction_count_before = cache->onednn_graph_scratch_pool_eviction_count();
    check(cache->onednn_graph_scratch_pool_peak_bytes() >= kSizeA,
          "onednn_graph_scratch_pool_peak_bytes() reflects ptr1 sitting in the pool");

    // kSizeB, not kSizeA: an exact-size request would be served by the pool
    // reuse path tested above WITHOUT ever reaching the cap check at all,
    // which would prove nothing about eviction/waiting. kSizeA (pooled,
    // pending) + kSizeB (requested) together exceed the 350 MB cap set in
    // main(), so this can only succeed by evicting ptr1's pooled entry once
    // its event completes.
    const auto start = std::chrono::steady_clock::now();
    void *     ptr2  = cache->onednn_graph_scratch_alloc(kSizeB, 256, &q);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    check(ptr2 != nullptr, "the kSizeB allocation eventually succeeds");
    check(cache->onednn_graph_scratch_direct_wait_count() > wait_count_before,
          "onednn_graph_scratch_direct_wait_count() increased -- the allocator actually waited for eviction");
    // Half of kSlowReleaseMs, not the whole thing: the poll interval inside
    // the wait loop (kOnednnGraphDirectWaitPollTimeoutMs, 200 ms in
    // unified-cache.cpp) means the observed wait can undershoot the event's
    // own completion time by up to one poll, and this only needs to
    // distinguish "actually waited" from "returned immediately".
    check(elapsed.count() >= kSlowReleaseMs / 2,
          "the kSizeB allocation took a substantial fraction of the release delay, not ~0 ms");
    check(cache->onednn_graph_scratch_pool_eviction_count() > eviction_count_before,
          "onednn_graph_scratch_pool_eviction_count() increased -- ptr1's pooled entry was actually released, "
          "not just waited on");
    printf("    (elapsed=%lld ms, cap wait count %zu -> %zu, eviction count %zu -> %zu)\n",
           static_cast<long long>(elapsed.count()), wait_count_before, cache->onednn_graph_scratch_direct_wait_count(),
           eviction_count_before, cache->onednn_graph_scratch_pool_eviction_count());

    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }
}

// --- (c) exhausted DIRECT allocation fails loudly, never silently ----------
void test_loud_failure(unified_cache * cache) {
    printf("DIRECT path exhaustion fails loudly:\n");

    sycl::queue & q = cache->get_queue();

    g_captured_log.clear();
    ggml_log_set(capture_log, nullptr);

    ggml_sycl_test_onednn_graph_scratch_suppress_abort(true);
    // 2: force BOTH the initial attempt and the drain-and-retry to fail,
    // without touching the GPU -- this is what drives the allocator all the
    // way to its "give up" decision deterministically. Uses a size not
    // exercised above so no pool entry can accidentally satisfy the request
    // before the forced-fail path is even reached.
    ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail(2);

    constexpr size_t kSizeC = 340ull * 1024 * 1024;
    void *           ptr    = cache->onednn_graph_scratch_alloc(kSizeC, 256, &q);

    ggml_sycl_test_onednn_graph_scratch_suppress_abort(false);
    ggml_log_set(nullptr, nullptr);

    check(ptr == nullptr,
          "the exhausted allocation returns nullptr (a real prior version returned nullptr too --"
          " the property under test is the log+abort below, not this alone)");
    check(ggml_sycl_test_onednn_graph_scratch_abort_triggered(),
          "the abort path was reached (would have called GGML_ABORT in production)");
    check(contains(g_captured_log, "Refusing to hand oneDNN a null scratch pointer"),
          "the loud ERROR log explains the refusal");
    check(contains(g_captured_log, "llama.cpp-0oxf"), "the loud ERROR log cites this ticket");
}

// --- (d) reclaim never destructs a pooled entry whose release event is
//         still pending (spec review finding #1, BLOCKING) ----------------
void test_pending_event_reclaim_does_not_destruct_in_flight(unified_cache * cache, int device) {
    printf("Reclaim defers a pooled entry with an incomplete release event:\n");

    // Start from a clean pool so this test's assertions are not sensitive to
    // whatever the earlier tests in this process left pooled.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup");

    sycl::queue & q = cache->get_queue();

    constexpr size_t kSizeD = 360ull * 1024 * 1024;  // distinct from every other size in this file

    void * ptr = cache->onednn_graph_scratch_alloc(kSizeD, 256, &q);
    check(ptr != nullptr, "DIRECT allocation for the pending-event reclaim setup succeeds");
    if (!ptr) {
        return;
    }

    // Free with a slow event and do NOT wait on it -- the entry parks in the
    // pool with an INCOMPLETE release event, exactly the state
    // onednn_graph_scratch_clear_pool_locked() must handle without
    // destructing the owning mem_handle out from under a still-in-flight
    // host_task.
    sycl::event slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr, &slow_release);
    check(cache->onednn_graph_scratch_pool_peak_bytes() >= kSizeD,
          "the entry is sitting in the pool (peak bytes reflects it) before the event completes");

    const size_t evictions_before = cache->onednn_graph_scratch_pool_eviction_count();

    // Reclaim now, while slow_release is (overwhelmingly likely, given
    // kSlowReleaseMs) still incomplete -- this is the exact call
    // ggml_backend_sycl_set_runtime_context() and arena_reserve()'s
    // context-reclaim branch make, and unlike cache teardown, NEITHER of
    // those two call sites drains the queue first. The pre-fix
    // onednn_graph_scratch_clear_pool_locked() destructed every pooled
    // entry unconditionally here, regardless of its release event's
    // completion -- this is the property that would regress.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test reclaim");

    check(cache->onednn_graph_scratch_pool_eviction_count() > evictions_before,
          "the reclaim counted the pending entry as an eviction (it left the pool)");

    // There is no public accessor for "is a handle still sitting in the
    // shared background drain worker's retained queue rather than already
    // freed" (retain_handles_until_event() hands ownership into
    // mem-handle.cpp's own internal worker, which this file deliberately
    // does not reach into) -- so the strongest property observable from
    // here without adding test-only introspection into that worker is
    // behavioral: the process stays healthy after the reclaim, and the pool
    // is genuinely empty for this size afterward, not merely that a counter
    // moved. A fresh request of the identical size must miss the pool
    // (nothing left to reuse) and still succeed as a real allocation.
    const size_t misses_before = cache->onednn_graph_scratch_pool_miss_count();
    void *       fresh         = cache->onednn_graph_scratch_alloc(kSizeD, 256, &q);
    check(fresh != nullptr, "a fresh allocation of the same size succeeds after the reclaim");
    check(cache->onednn_graph_scratch_pool_miss_count() > misses_before,
          "the fresh allocation missed the pool -- the reclaimed entry was not left behind for reuse");

    slow_release.wait();  // let the host_task finish before the process exits
    if (fresh) {
        cache->onednn_graph_scratch_free(fresh, nullptr);
    }
}

}  // namespace

int main(int, char ** argv) {
    // Match the sibling SYCL gates: pin the validation card so a bare ctest
    // run cannot perturb a measurement on the other GPU. ctest also sets this
    // via ENVIRONMENT; this is only a fallback for bare invocation.
    sycl_test_selector_fallback(argv, "level_zero:1");

    // Set BEFORE any onednn_graph_scratch_* call in this process:
    // onednn_graph_scratch_direct_cap_bytes() memoizes this env var on its
    // first read (matching the sibling zone-floor override's own once-only
    // lazy-static pattern), so setting it later in the process would be a
    // silent no-op. 350 MB sits strictly between one and two of
    // {kSizeA, kSizeB} (300/320 MB), so kSizeA (pooled) + kSizeB (requested)
    // together exceed it while either alone fits comfortably.
    setenv("GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB", "350", 1);

    const int device = 0;  // in-process index after selector filtering

    ggml_backend_t backend = ggml_backend_sycl_init(device);
    if (backend == nullptr) {
        fprintf(stderr, "SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    unified_cache * cache = get_unified_cache_for_device(device);
    if (cache == nullptr) {
        fprintf(stderr, "no unified cache for device %d; cannot run.\n", device);
        ggml_backend_free(backend);
        return 1;
    }

    test_pool_reuse(cache);
    test_bounded_eviction(cache);
    test_loud_failure(cache);
    test_pending_event_reclaim_does_not_destruct_in_flight(cache, device);

    ggml_backend_free(backend);

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
