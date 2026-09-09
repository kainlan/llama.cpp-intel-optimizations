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
// exceed the cap evicts (real release) completed pool entries of any size,
// including the requested size's own bucket, until it fits, waiting
// (bounded) for an in-flight entry to complete first if none are immediately
// evictable. Two further bounds, not directly exercised by this file (both
// are exercised via unified-cache.cpp's own source-contract gate,
// tests/test-sycl-onednn-graph-allocator-source.py, and via reading the
// counters this file DOES check): each size bucket also caps at
// onednn_graph_scratch_pool_depth_per_size() entries (default 8,
// env-overridable) so a workload walking many distinct sizes cannot grow the
// pool without limit even while every individual size stays under the byte
// cap; and the pool is cleared (real release) at cache teardown and at the
// same point arena_reserve() reclaims the KV/RUNTIME zones for a new
// context, so a pooled buffer cannot outlive the context it belongs to.
//
// This test asserts five properties:
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
//   (d) RECLAIM SAFETY (BLOCKING): reclaiming the
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
//   (e) OVERSIZED REQUEST EARLY-OUT: a single request larger than the whole
//       DIRECT cap must skip the bounded poll-loop wait entirely --
//       observable as onednn_graph_scratch_direct_wait_count() staying
//       unchanged, the primary proof the early-out fired rather than the
//       wait loop -- while the eviction sweep that always runs first still
//       does, observable as onednn_graph_scratch_pool_eviction_count()
//       increasing for a genuinely parked entry, and the allocation still
//       succeeds as a real (uncapped-by-this-check) allocation. Added by
//       llama.cpp-pqgl's review, not part of the original 0oxf RED-FIRST
//       set below.
//
//   (f) IN-FLIGHT ENTRIES ARE SKIPPED, NOT WAITED ON: the reuse pool's
//       completion checks must not BLOCK a calling thread on a pooled
//       entry's release_event -- event_complete()'s bare
//       command_execution_status query does exactly that on any
//       profiling-enabled queue, which every backend stream is. Parking an
//       entry with a slow, unwaited release event and immediately
//       re-requesting the identical size must return a fresh allocation
//       (miss, not hit) in well under the release delay, leaving the
//       in-flight entry pooled and un-evicted; the same must hold for the
//       cap-eviction sweep when a DIFFERENT, already-complete entry alone
//       supplies the needed headroom. A test-only hook forces the pre-fix
//       blocking query as a positive control within this same (fixed)
//       binary. Added by llama.cpp-c6ah, not part of the original 0oxf
//       RED-FIRST set below.
//
// RED-FIRST NOTE. Three of these six properties, (a)-(c), are RED against
// the pre-0oxf code: (a) and (b) have no pool, cap, or wait at all (every
// DIRECT allocation is a fresh unified_alloc() with no bound), and (c) had
// no abort hook to suppress -- the pre-fix function simply returned nullptr
// with `req.suppress_failure_log = true`, so the log capture in this test
// would find nothing and the "abort triggered" latch would not exist. (d),
// (e) and (f) above were added by later tickets (llama.cpp-0oxf's own
// reclaim-safety finding, llama.cpp-pqgl's review, and llama.cpp-c6ah's
// blocking-query finding, respectively) and are not part of this original
// RED-FIRST set -- (f) is RED against the pre-c6ah code specifically (its
// own force_blocking_pool_check hook reproduces that RED behaviour on
// demand within this GREEN binary, since the pre-fix source is no longer
// buildable standalone once this fix has landed).
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
using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check;
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

// A level TAG ("[ERROR] " / "[WARN] ") is prefixed onto each new
// (non-continuation) ERROR or WARN line so a check can assert not just that
// some text was logged, but that it was logged AT that specific level --
// e.g. distinguishing the oversized-request WARN from what would otherwise
// be an indistinguishable-by-text ERROR (llama.cpp-pqgl: capturing text
// alone made a WARN-vs-ERROR downgrade unverifiable). INFO and DEBUG lines
// are captured with no tag at all -- no check in this file needs to tell
// them apart from each other. GGML_LOG_LEVEL_CONT (a continuation of the
// previous log call) also gets no fresh tag, so a message split across
// multiple callback invocations still reads as one tagged line rather than
// being interrupted mid-sentence.
void capture_log(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(user_data);
    if (text != nullptr) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            g_captured_log += "[ERROR] ";
        } else if (level == GGML_LOG_LEVEL_WARN) {
            g_captured_log += "[WARN] ";
        }
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
//         still pending (BLOCKING) ----------------------------------------
void test_pending_event_reclaim_does_not_destruct_in_flight(unified_cache * cache, int device) {
    printf("Reclaim defers a pooled entry with an incomplete release event:\n");

    // Start from a clean pool so this test's assertions are not sensitive to
    // whatever the earlier tests in this process left pooled.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup");

    sycl::queue & q = cache->get_queue();

    // 310 MiB, not 360: this must stay UNDER the 350 MB cap main() sets via
    // GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB, so both allocations below take
    // the NORMAL pool path this test actually exercises (reclaim/pending-
    // event handling) rather than the size>cap early-out
    // (onednn_graph_scratch_wait_for_direct_headroom_locked(), llama.cpp-pqgl)
    // -- that early-out has its own dedicated test below,
    // test_oversized_request_skips_wait_loop(). Still distinct from
    // kSizeA/kSizeB/kSizeC (300/320/340 MiB) above.
    constexpr size_t kSizeD = 310ull * 1024 * 1024;

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
    // Not asserted here: onednn_graph_scratch_pool_peak_bytes() is a
    // cumulative process-lifetime high-water mark that never resets, so a
    // "peak >= kSizeD" check right after this free() would be vacuous once
    // an earlier test (test_bounded_eviction, kSizeA/kSizeB = 300/320 MiB)
    // has already pushed the peak above kSizeD (310 MiB) -- it would hold
    // whether or not THIS entry ever made it into the pool. The eviction-
    // count delta asserted below is the real proof, and it is attributable
    // to exactly THIS entry (not, say, test_bounded_eviction's kSizeB
    // buffer, which its own free() left parked with a complete event)
    // because this function's very first line above already reclaimed the
    // pool under "test setup" -- the pool was empty of every leftover entry
    // before this test ever allocated kSizeD, so the only thing that can be
    // sitting in it when evictions_before is captured just below is this
    // entry: eviction_count_ can only increase for an entry that
    // clear_pool_locked() actually found IN the pool, and there is nothing
    // else in it to find.
    sycl::event slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr, &slow_release);

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

// --- (e) a request larger than the cap takes the size>cap early-out, not
//         the full poll-loop wait, AND the eviction sweep still runs first
//         (llama.cpp-pqgl) --------------------------------------------------
//
// This must remain the ONLY oversized (size>cap) request issued anywhere in
// this process, and the FIRST one if any other test is ever added after it:
// onednn_graph_scratch_oversized_request_logged_ is a per-instance "log
// once" latch, so a second oversized request anywhere else in this binary
// -- before or after this test -- would silently not log the WARN this test
// asserts on.
void test_oversized_request_skips_wait_loop(unified_cache * cache, int device) {
    printf("DIRECT path request larger than the cap skips the poll-loop wait:\n");

    // Start from a clean pool.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup");

    sycl::queue & q = cache->get_queue();

    // Park ONE genuinely event-complete pooled entry, well under the cap,
    // BEFORE issuing the oversized request below -- so the eviction sweep
    // inside onednn_graph_scratch_wait_for_direct_headroom_locked() (which a
    // prior fix moved to run BEFORE its size>cap early-out) has something
    // REAL to evict. Without this setup the pool would already be empty and
    // this test could not tell "the sweep ran before the early-out" apart
    // from "the sweep never ran at all" -- both pass identically on an
    // empty pool.
    //
    // 280 MiB, not 200: this process reserves a 256 MB ONEDNN zone even with
    // no model loaded -- measured on hardware, `[VRAM-ARENA] Reserved single
    // chunk: ... oneDNN=256.0 ...` -- and a request at or under that zone
    // size is served FROM THE ZONE, never takes the DIRECT path, and so
    // never enters the pool this setup needs it to. 200 MiB silently did
    // exactly that: measured FAIL on hardware, the eviction-count assertion
    // below never saw an increase, because the "parked" allocation was
    // never pooled to begin with. 280 MiB is above that zone -- matching
    // every other size in this file (kSizeA/kSizeB/kSizeC/kSizeD at
    // 300/320/340/310 MiB are all above it for the identical reason, and a
    // 300 MiB request logs "did not fit the ONEDNN zone" confirming it),
    // distinct from all four of them, and comfortably under the 350 MB cap
    // main() sets -- so it can be parked and later evicted without itself
    // ever engaging the cap machinery this test isn't exercising. The
    // miss-count assertion right after the allocation below is this setup's
    // own self-check against silently regressing back to a zone-served size.
    constexpr size_t kSizeParked             = 280ull * 1024 * 1024;
    const size_t     misses_before_park      = cache->onednn_graph_scratch_pool_miss_count();
    const size_t     outstanding_before_park = cache->onednn_graph_scratch_direct_outstanding_bytes();
    void *           parked                  = cache->onednn_graph_scratch_alloc(kSizeParked, 256, &q);
    check(parked != nullptr, "the parked-entry setup allocation succeeds");
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before_park + 1,
          "the parked allocation was a DIRECT-path pool miss, not served from the ONEDNN zone -- proves this "
          "setup actually parks a poolable entry rather than silently zone-serving it");
    check(cache->onednn_graph_scratch_direct_outstanding_bytes() == outstanding_before_park + kSizeParked,
          "onednn_graph_scratch_direct_outstanding_bytes() increased by exactly the parked allocation's size -- "
          "proves the accessor tracks a real DIRECT-path charge rather than staying inert");

    // onednn_graph_scratch_high_water_bytes() had no caller anywhere in
    // this binary before this test -- every sibling accessor above has one,
    // so give it one here too. Its only write site,
    // note_onednn_graph_scratch_alloc_locked()
    // (unified-cache.cpp:10595-10600), is called from the zone-fit path
    // (unified-cache.cpp:10208) AND both DIRECT paths (pool-hit at
    // unified-cache.cpp:10143, fresh alloc at unified-cache.cpp:10419) --
    // so, unlike onednn_graph_scratch_direct_outstanding_bytes() above, it
    // tracks zone-served and DIRECT allocations combined, as a running max
    // per unified_cache INSTANCE (unified-cache.hpp:~3790), not scoped to
    // this test -- and this binary drives a single instance for device 0.
    // By the time this test runs, main() has already run the three earlier
    // tests that actually complete an allocation (pool_reuse,
    // bounded_eviction, pending_event_reclaim); test_loud_failure's kSizeC
    // (340 MiB) request is deliberately forced to fail
    // (ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail(2),
    // asserted via ptr == nullptr) and returns before ever reaching this
    // counter's write site, so it contributes nothing. kSizeA/kSizeB/kSizeD
    // (300/320/310 MiB) already push the high-water past kSizeParked (280
    // MiB) on their own -- so the check below cannot isolate the parked
    // allocation's own contribution to that floor; it only proves the
    // accessor is not inert (it would read 0 if the write site above never
    // ran). The non-decrease-after-release check further below
    // distinguishes a high-water mark from a live outstanding count; it
    // cannot fully discriminate that from a counter merely frozen at or
    // above kSizeParked, since the combined
    // onednn_graph_scratch_outstanding_bytes_ it tracks has no accessor of
    // its own to check directly, and no cheaper, fully discriminating check
    // is available here.
    check(cache->onednn_graph_scratch_high_water_bytes() >= kSizeParked,
          "onednn_graph_scratch_high_water_bytes() is not inert -- it reports at least the parked allocation's "
          "size, a floor the earlier completed tests in this binary already exceeded on their own");
    const size_t high_water_after_park = cache->onednn_graph_scratch_high_water_bytes();

    if (parked) {
        sycl::event release = submit_slow_release(q);
        cache->onednn_graph_scratch_free(parked, &release);
        release.wait();  // ensure event-complete before the oversized request below
    }

    // High-water is a max, never lowered by a free() -- note_onednn_graph_-
    // scratch_free_locked() (unified-cache.cpp:10602-10604) only decrements
    // onednn_graph_scratch_outstanding_bytes_, and nothing anywhere writes
    // onednn_graph_scratch_high_water_bytes_ except the max-update inside
    // note_onednn_graph_scratch_alloc_locked() cited above, so releasing
    // the parked allocation must not move it back down.
    check(cache->onednn_graph_scratch_high_water_bytes() >= high_water_after_park,
          "onednn_graph_scratch_high_water_bytes() does not decrease after the parked allocation is released -- "
          "it is a high-water mark, not a live outstanding count");

    const size_t evictions_before  = cache->onednn_graph_scratch_pool_eviction_count();
    const size_t wait_count_before = cache->onednn_graph_scratch_direct_wait_count();

    // main() sets the cap to 350 MB via GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB.
    // 366 MiB is comfortably over that ON ITS OWN -- no amount of eviction
    // or waiting could ever bring outstanding+size under the cap for this
    // request.
    constexpr size_t kSizeOversized = 366ull * 1024 * 1024;

    g_captured_log.clear();
    ggml_log_set(capture_log, nullptr);

    const auto start = std::chrono::steady_clock::now();
    void *     ptr   = cache->onednn_graph_scratch_alloc(kSizeOversized, 256, &q);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    ggml_log_set(nullptr, nullptr);

    check(ptr != nullptr,
          "the oversized allocation still succeeds -- the cap bounds pooling/waiting, it is not a hard refusal");
    check(cache->onednn_graph_scratch_pool_eviction_count() > evictions_before,
          "the eviction sweep still ran and released the parked entry before the size>cap early-out fired");
    // Primary proof the early-out fired rather than the poll loop:
    // onednn_graph_scratch_direct_wait_count() is only incremented once the
    // poll loop past the early-out actually starts, which the early-out
    // returns before reaching -- this is a stronger, non-timing-based
    // signal than the elapsed-time check below, and not vulnerable to a
    // slow but otherwise-correct allocation false-failing it.
    check(cache->onednn_graph_scratch_direct_wait_count() == wait_count_before,
          "onednn_graph_scratch_direct_wait_count() did NOT increase -- the early-out returned before any wait "
          "was attempted, not after a completed-but-unsuccessful one");
    // Secondary timing check only -- 4000 ms, not 2000: 2000 ms equals
    // kOnednnGraphDirectFailureDrainTimeoutMs, so a single drain-and-retry
    // on this call's underlying unified_alloc() would false-fail a 2000 ms
    // bound even though the early-out itself worked correctly. What this
    // bounds is "did not run the full 5000 ms poll loop", not
    // "was instantaneous".
    check(elapsed.count() < 4000, "the allocation returned well under the 5 s poll-loop timeout");
    printf("    (elapsed=%lld ms)\n", static_cast<long long>(elapsed.count()));
    // Adjacency, not three independent substring searches: the "[WARN] " tag
    // is prepended immediately before the message text (capture_log above),
    // and GGML_LOG_WARN emits the whole line in one callback, so the tag and
    // the message start are always contiguous when this really was logged
    // at WARN -- a check that only searched for "[WARN] " and the message
    // text independently could pass on an unrelated WARN elsewhere plus an
    // ERROR carrying this text, which is exactly the confusion this check
    // exists to rule out.
    check(contains(g_captured_log, "[WARN] [UNIFIED-CACHE] oneDNN Graph scratch DIRECT path: requested") &&
              contains(g_captured_log, "cap by itself"),
          "the latched oversized-request WARN was logged, and at WARN level (not ERROR)");
    // llama.cpp-pqgl: the caller (onednn_graph_scratch_alloc_direct_locked)
    // must NOT also log its own "gave up waiting for headroom" ERROR for
    // this call -- that message describes a genuine timed-out wait, and
    // this call never waited at all, having returned via the early-out
    // instead. This binds only while
    // onednn_graph_scratch_gave_up_waiting_logged_ has not already latched
    // from an earlier genuine timeout elsewhere in this process (no test in
    // this file triggers one) -- otherwise the message would stay silent
    // regardless of whether this call's own early-out guard still exists,
    // and this check would pass vacuously.
    check(!contains(g_captured_log, "gave up waiting for headroom"),
          "the caller did not log its timed-out-wait ERROR for a call that never waited");

    if (ptr) {
        cache->onednn_graph_scratch_free(ptr, nullptr);
    }

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test teardown");
}

// --- (f) an in-flight pooled entry is SKIPPED, not WAITED ON -- both by a
//         fresh same-size request and by the eviction sweep (llama.cpp-c6ah)
//
// BACKGROUND. unified_cache::event_complete()'s bare
// command_execution_status query BLOCKS rather than polls on any
// profiling-enabled queue, and every backend stream -- including the one
// oneDNN's free callback supplies release_event on -- is profiling-enabled.
// The DIRECT reuse pool's completion checks
// (onednn_graph_scratch_entry_usable_locked(),
// onednn_graph_scratch_evict_pool_until_fits_locked(),
// onednn_graph_scratch_clear_pool_locked()) used to query that event
// directly, so instead of SKIPPING an in-flight pooled entry -- the design
// documented at their own declarations -- the allocator WAITED for its
// SDPA kernel to finish, serializing the allocation path behind running
// device work. The fix routes every one of those checks through
// onednn_graph_scratch_pool_entry_release_complete(), which prefers a
// host-visible std::atomic<bool> flag (armed by a host_task on a new,
// dedicated, non-profiling watcher queue) over querying the event.
//
// WHY THIS TEST USES A SHRUNK ONEDNN ZONE. Proving "skipped, not waited"
// with a wall-clock bound requires a request that does NOT need the
// DIRECT-path cap machinery to engage at all -- otherwise a genuine,
// unavoidable physical wait for the SAME in-flight entry's completion is
// indistinguishable from the bug this fix closes (both take about
// kSlowReleaseMs either way; only WHICH mechanism produces that wait
// differs, and this file has no way to observe that internal difference
// through the public API alone). That requires parking the SAME size twice
// simultaneously while staying under the DIRECT_CAP_MB=350 cap main() sets
// -- but the ONEDNN zone's natural (no-model) floor is measured elsewhere
// in this file at ~256 MB (see kSizeParked's own comment above), and no
// size can simultaneously exceed that floor (to reach the DIRECT path at
// all) AND have its DOUBLED cost fit under a 350 MB cap (2 x 257 MB always
// exceeds 350 MB). GGML_SYCL_ONEDNN_GRAPH_ZONE_MB is this codebase's own
// sanctioned escape hatch for exactly this shape of problem ("always wins
// over the FORMULA when set" -- see onednn_graph_scratch_zone_floor_bytes()'s
// comment) -- main() sets it to a few MB, safely below EVERY size used
// anywhere in this file (this test's own two-digit-MiB sizes included, and
// every existing three-digit-MiB size above by a wide margin), so nothing
// in this file becomes zone-served that was not already. This does not
// change what any earlier test in this file measures -- every one of them
// already relies on its own size exceeding whatever the zone floor is, not
// on the floor's specific value.
void test_in_flight_entry_is_skipped_not_waited(unified_cache * cache, int device) {
    printf("DIRECT path in-flight pooled entry is skipped, not waited on:\n");

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup");

    sycl::queue & q = cache->get_queue();

    // Small (relative to the shrunk zone) and DISTINCT from every size used
    // elsewhere in this file, so this test's pool state cannot alias theirs
    // even without today's leading reclaim call.
    constexpr size_t kSizeSkip = 32ull * 1024 * 1024;

    const size_t misses_before_setup = cache->onednn_graph_scratch_pool_miss_count();
    void *       ptr1                = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the skip-not-wait setup succeeds");
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before_setup + 1,
          "the setup allocation was a DIRECT-path pool miss, not served from the (shrunk) ONEDNN zone -- proves "
          "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB actually took effect for this size");
    if (!ptr1) {
        return;
    }

    // Park with a slow, UNWAITED release event -- the entry sits in the pool
    // with an INCOMPLETE release_event for the rest of this sub-scenario.
    sycl::event slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr1, &slow_release);

    const size_t misses_before    = cache->onednn_graph_scratch_pool_miss_count();
    const size_t hits_before      = cache->onednn_graph_scratch_pool_hit_count();
    const size_t evictions_before = cache->onednn_graph_scratch_pool_eviction_count();

    const auto start = std::chrono::steady_clock::now();
    void *     ptr2  = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    check(ptr2 != nullptr, "the immediate same-size re-request succeeds");
    check(ptr2 != ptr1,
          "...as a genuinely FRESH allocation, not the still in-flight entry handed back (that would be a "
          "USE-AFTER-FREE-shaped bug: ptr1's SDPA kernel may still be running)");
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before + 1,
          "onednn_graph_scratch_pool_miss_count() increased -- served by a fresh allocation, not the pool");
    check(cache->onednn_graph_scratch_pool_hit_count() == hits_before,
          "onednn_graph_scratch_pool_hit_count() did NOT increase -- the in-flight entry was never handed back");
    check(cache->onednn_graph_scratch_pool_eviction_count() == evictions_before,
          "onednn_graph_scratch_pool_eviction_count() did NOT increase -- the in-flight entry is still pooled, "
          "not released");
    // kSlowReleaseMs/3, not /2: this call does strictly less work than
    // test_bounded_eviction's cap-constrained wait (no cap is engaged here
    // at all -- see this test's header comment), so it can afford a
    // tighter bound while remaining well clear of scheduler/CI noise.
    check(elapsed.count() < kSlowReleaseMs / 3,
          "the re-request returned in well under a third of the release delay -- it did not wait for ptr1's "
          "release event to complete");
    printf("    (elapsed=%lld ms, immediate re-request)\n", static_cast<long long>(elapsed.count()));

    // Now let ptr1's release actually complete, and confirm a THIRD
    // same-size request is a genuine pool HIT -- the in-flight entry was
    // skipped above, not lost.
    slow_release.wait();
    const size_t hits_before_reuse = cache->onednn_graph_scratch_pool_hit_count();
    void *       ptr3              = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    check(ptr3 == ptr1, "once ptr1's release event completes, it is reused for a same-size request");
    check(cache->onednn_graph_scratch_pool_hit_count() == hits_before_reuse + 1,
          "onednn_graph_scratch_pool_hit_count() increased -- this time it really was a pool hit");

    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }
    if (ptr3) {
        cache->onednn_graph_scratch_free(ptr3, nullptr);
    }

    // --- Positive control: the RED arm this GREEN binary can still produce.
    // With the hook forced, onednn_graph_scratch_pool_entry_release_complete()
    // falls back to a direct event_complete() query -- reproducing the
    // pre-fix blocking wait -- so the SAME scenario above should now take
    // roughly the full release delay AND come back as a pool HIT (blocking
    // until complete makes the entry look immediately USABLE once it
    // returns), the opposite of both outcomes just asserted.
    ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check(true);

    void * ptr4 = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    check(ptr4 != nullptr, "RED-arm setup allocation succeeds");
    if (ptr4) {
        sycl::event slow_release2 = submit_slow_release(q);
        cache->onednn_graph_scratch_free(ptr4, &slow_release2);

        const auto start2 = std::chrono::steady_clock::now();
        void *     ptr5   = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
        const auto elapsed2 =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start2);

        // 80%, not 100%: the same poll-granularity slack test_bounded_eviction
        // already accepts for the analogous genuine-wait case, applied here
        // to a real blocking call instead of a poll loop -- comfortably
        // above the /3 bound the GREEN arm above must stay under, so the two
        // checks cannot both pass on the same (broken) code.
        check(elapsed2.count() >= (kSlowReleaseMs * 8) / 10,
              "with the RED hook forced, the re-request took most of the release delay -- reproducing the "
              "pre-fix blocking behaviour this test would otherwise never exercise");
        check(ptr5 == ptr4,
              "...and the blocking query left the entry USABLE by the time it returned, so this was a pool HIT, "
              "not a fresh allocation -- the opposite of the GREEN-arm outcome above");
        printf("    (elapsed=%lld ms, RED arm / force_blocking_pool_check)\n",
               static_cast<long long>(elapsed2.count()));

        if (ptr5) {
            cache->onednn_graph_scratch_free(ptr5, nullptr);
        }
    }

    ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check(false);

    // --- Eviction sweep must apply the identical skip, not just the
    // try-reuse lookup above. Park one in-flight (incomplete) entry and one
    // already-complete entry of a DIFFERENT size, then request a THIRD,
    // different size that needs headroom the completed entry alone can
    // supply -- the in-flight entry must be left alone (not evicted, not
    // waited on) either way.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (eviction sweep)");

    constexpr size_t kSizeEvictInFlight = 200ull * 1024 * 1024;
    constexpr size_t kSizeEvictComplete = 100ull * 1024 * 1024;
    constexpr size_t kSizeEvictRequest  = 90ull * 1024 * 1024;
    // kSizeEvictInFlight + kSizeEvictComplete (300 MiB) fits under the
    // 350 MB cap on its own -- parking both must not itself force anything;
    // + kSizeEvictRequest (390 MiB) does not, so satisfying this request
    // needs SOME eviction, and kSizeEvictInFlight + kSizeEvictRequest alone
    // (290 MiB) fits -- so evicting kSizeEvictComplete is both necessary and
    // SUFFICIENT; the in-flight entry never needs to be touched to succeed.

    void * evict_in_flight = cache->onednn_graph_scratch_alloc(kSizeEvictInFlight, 256, &q);
    check(evict_in_flight != nullptr, "eviction-sweep in-flight setup allocation succeeds");
    sycl::event evict_slow_release = submit_slow_release(q);
    if (evict_in_flight) {
        cache->onednn_graph_scratch_free(evict_in_flight, &evict_slow_release);
    }

    void * evict_complete = cache->onednn_graph_scratch_alloc(kSizeEvictComplete, 256, &q);
    check(evict_complete != nullptr, "eviction-sweep completed-entry setup allocation succeeds");
    if (evict_complete) {
        sycl::event complete_release = submit_slow_release(q);
        complete_release.wait();  // genuinely complete before parking
        cache->onednn_graph_scratch_free(evict_complete, &complete_release);
    }

    const size_t sweep_evictions_before = cache->onednn_graph_scratch_pool_eviction_count();

    const auto sweep_start = std::chrono::steady_clock::now();
    void *     evict_ptr   = cache->onednn_graph_scratch_alloc(kSizeEvictRequest, 256, &q);
    const auto sweep_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sweep_start);

    check(evict_ptr != nullptr, "the headroom-needing request succeeds");
    check(cache->onednn_graph_scratch_pool_eviction_count() == sweep_evictions_before + 1,
          "exactly one entry was evicted -- the completed one, never the in-flight one");
    check(sweep_elapsed.count() < kSlowReleaseMs / 3,
          "the eviction sweep did not wait for the in-flight entry's release event -- evicting the completed "
          "entry alone was enough headroom");
    printf("    (elapsed=%lld ms, eviction sweep)\n", static_cast<long long>(sweep_elapsed.count()));

    if (evict_ptr) {
        cache->onednn_graph_scratch_free(evict_ptr, nullptr);
    }
    evict_slow_release.wait();  // let the still-pooled in-flight entry's host_task finish before the process exits
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test teardown");
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
    // Also memoized once per process (onednn_graph_scratch_test_hooks_enabled(),
    // unified-cache.cpp), so it must be set here too, not only via ctest's
    // ENVIRONMENT: a bare (non-ctest) invocation of this binary would
    // otherwise silently no-op force_direct_alloc_fail()/suppress_abort()
    // (see their gated setters), making test_loud_failure() actually
    // exhaust real VRAM instead of exercising the forced-fail path, and
    // false-FAIL on the "abort triggered" assertion. A test binary
    // self-enabling its OWN test-only hooks is the sanctioned case the gate
    // in unified-cache.cpp exists to allow -- only ctest's ENVIRONMENT and a
    // direct run of this specific binary should ever set this, never a
    // production process.
    setenv("GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS", "1", 1);
    // llama.cpp-c6ah: also memoized once per process
    // (onednn_graph_scratch_zone_floor_bytes(), unified-cache.cpp), so set
    // BEFORE ggml_backend_sycl_init() below plans the arena, same
    // once-only-lazy-static reasoning as the cap above. Shrinks the ONEDNN
    // zone from its natural (no-model) ~256 MB floor down to a few MB --
    // this codebase's own sanctioned override for exactly this purpose (see
    // onednn_graph_scratch_zone_floor_bytes()'s comment: "always wins over
    // the FORMULA when set"). test_in_flight_entry_is_skipped_not_waited()
    // needs to park the SAME size twice simultaneously while staying under
    // the 350 MB cap set above, which is impossible at the natural ~256 MB
    // floor (any size big enough to miss that zone, doubled, exceeds
    // 350 MB) -- see that test's own header comment for the full argument.
    // Harmless to every earlier test in this file: none of them depends on
    // the zone's specific size, only on their own sizes exceeding whatever
    // it is (all >= 280 MiB, comfortably above this override either way).
    setenv("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB", "8", 1);

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
    test_oversized_request_skips_wait_loop(cache, device);
    test_in_flight_entry_is_skipped_not_waited(cache, device);

    ggml_backend_free(backend);

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
