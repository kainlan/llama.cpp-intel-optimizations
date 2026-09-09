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

#    include <algorithm>
#    include <chrono>
#    include <mutex>
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

// llama.cpp-c6ah: the process-wide DIRECT-path cap every size constant in
// this file is chosen relative to, and the natural (no-model) ONEDNN zone
// floor every DIRECT-triggering size in this file must exceed. Defined
// once here so each size's own static_assert (immediately next to its
// definition, throughout this file) can check its relation to both by
// name instead of by a hand-copied number -- a future edit that breaks the
// arithmetic then fails the BUILD, not just a hardware run.
//
// 700, not the historical 350: this file previously set the cap to 350 MB
// and shrank the zone via GGML_SYCL_ONEDNN_GRAPH_ZONE_MB so sizes could
// stay small. That zone override does NOT take effect for the no-model
// arena this test binary plans -- measured on hardware, the process still
// reports `[VRAM-ARENA] Reserved single chunk: ... oneDNN=256.0` regardless
// of the override -- so every DIRECT-triggering size here, old or new,
// must clear that ~256 MB floor the same way the pre-existing tests
// already did, and the override cannot shrink sizes below it instead.
// test_in_flight_entry_is_skipped_not_waited() needs TWO such sizes
// outstanding simultaneously without engaging the cap machinery, which is
// impossible at a 350 MB cap (2 x 257 MB already exceeds it); raising the
// cap to 700 MB is what actually makes that possible, not shrinking the
// zone. main() sets GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB from kCapMiB
// directly so the two can never drift apart.
constexpr size_t kCapMiB         = 700;
constexpr size_t kCapBytes       = kCapMiB * 1024ull * 1024ull;
// Measured on hardware (see kCapMiB's own comment above); NOT
// configurable via GGML_SYCL_ONEDNN_GRAPH_ZONE_MB for the no-model case
// this binary plans against, so it is a fact this file's sizes must
// respect, not a value this file can tune.
constexpr size_t kZoneFloorBytes = 256ull * 1024 * 1024;

// Two sizes that exceed the natural ONEDNN zone floor (this repository's
// own comments separately record requests up to 144 MB as the largest
// measured from a real model), so both deterministically take the DIRECT
// path regardless of what a live model's planning left the zone sized to.
// They are deliberately DIFFERENT from each other so a request for kSizeB
// cannot be silently served by a pool entry parked for kSizeA -- that would
// let pool reuse mask the eviction/wait path this file also needs to
// exercise.
constexpr size_t kSizeA = 360ull * 1024 * 1024;
constexpr size_t kSizeB = 380ull * 1024 * 1024;
static_assert(kSizeA > kZoneFloorBytes && kSizeB > kZoneFloorBytes,
              "kSizeA/kSizeB must exceed the no-model ONEDNN zone floor to take the DIRECT path");
static_assert(kSizeA < kCapBytes && kSizeB < kCapBytes,
              "each size alone must still fit under the cap, so parking either one alone applies no cap pressure");
static_assert(kSizeA + kSizeB > kCapBytes,
              "test_bounded_eviction needs kSizeA (pooled) + kSizeB (requested) together to exceed the cap");

// The CALIBRATION TARGET for every "slow release" event in this file: long
// enough to be unambiguously distinguishable from "the wait didn't really
// wait", short enough to keep the gate fast. submit_slow_release() below
// scales a device kernel's iteration count to hit this target on whatever
// hardware this binary actually runs on -- it is a target, not a literal
// duration; see that function's own comment for why a hardcoded assumption
// is not good enough here.
constexpr int kSlowReleaseMs = 1500;

// llama.cpp-c6ah: this release event is now a DEVICE KERNEL, not a
// host_task -- measured on hardware (tests/test-sycl-event-status-blocking-probe.cpp,
// both cards: B50 kernel/profiling query=118 ms vs kernel duration=122 ms;
// B70 110 ms vs 114 ms; host_task on either queue kind, and a kernel on a
// NON-profiling queue, all measured ~0 ms), the bare
// command_execution_status query BLOCKS until completion ONLY for a
// device-kernel-produced event on a profiling-enabled queue -- exactly the
// shape a real oneDNN Graph-scratch release event actually is in
// production (oneDNN's free callback supplies a device-kernel event, not a
// host_task one). A host_task-produced "slow release" here would exercise
// a DIFFERENT case than the one this ticket's fix and premise are about --
// this file used one until this ticket's own RED arm measured ~6 ms
// instead of the predicted ~1500 ms and a dedicated probe traced the
// discrepancy to exactly this event-source difference.
//
// One work-item spin kernel, reading and writing a device-global cell
// every iteration so the loop cannot be folded away at compile time (each
// iteration's value depends on the PREVIOUS iteration's write to device
// memory, which the compiler cannot know ahead of time) -- deliberately
// serial, not parallel: the point is wall-clock duration on one device
// compute unit, matching test-sycl-event-status-blocking-probe.cpp's own
// kernel shape exactly, so this file's timings are comparable to that
// probe's.
sycl::event submit_spin_kernel(sycl::queue & q, int * cell, long long iterations) {
    return q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            int acc = 0;
            for (long long i = 0; i < iterations; ++i) {
                acc   = acc + *cell + 1;
                *cell = acc;
            }
        });
    });
}

// llama.cpp-c6ah: the ACTUAL device-measured duration of an event ALREADY
// known to be complete (command_end - command_start, both in nanoseconds
// per SYCL's profiling info), as opposed to submit_slow_release()'s own
// pre-submission CALIBRATED ESTIMATE (kSlowReleaseMs) -- printed alongside
// that estimate at every call site where the event's completion is
// confirmed before printing, so a failing bound is diagnosable as "the
// query did not block" rather than confusable with "the kernel ran
// shorter than the estimate" (calibration folds submission overhead into
// its rate, and the FIRST spin kernel submitted in a process also pays a
// one-time JIT/compilation cost the calibration run itself absorbs but a
// caller's own estimate cannot see). Profiling info is only valid once the
// command has finished -- NEVER call this on a still-pending event (most
// of this file's own release events are deliberately still pending at
// their own "elapsed=" print site; this helper is for the OTHER prints,
// after an explicit wait()). Requires the queue the event was submitted on
// to be profiling-enabled (every real backend stream in this codebase is,
// via default_queue_properties()) -- returns -1 if the query itself
// throws, so a caller can print "n/a" rather than propagate the exception
// into an unrelated check. Defined here, ahead of
// ensure_slow_release_calibrated() below, because that function also uses
// it to print the calibration kernel's own measured duration.
long long actual_kernel_duration_ms(sycl::event & evt) {
    try {
        const auto start_ns = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end_ns   = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
        return static_cast<long long>((end_ns - start_ns) / 1'000'000ull);
    } catch (const sycl::exception &) {
        return -1;
    }
}

// A submit_slow_release() result: the still-pending release event (never
// waited on here -- that is the whole point, callers park it in-flight)
// plus this run's CALIBRATED ESTIMATE of that event's own duration, so
// callers can bound their own timing checks against reality on whatever
// hardware ctest actually runs this on, rather than against the
// kSlowReleaseMs constant directly. The estimate is mathematically
// kSlowReleaseMs by construction (the iteration count is scaled
// specifically to hit that target using this run's OWN measured
// iterations-per-ms rate, not an assumption baked in at compile time) --
// callers reference `duration_ms` anyway, not the constant, so a future
// change to how the estimate is derived does not require touching every
// call site.
struct slow_release_result {
    sycl::event release_event;
    long long   duration_ms;
};

// llama.cpp-c6ah: the scaled iteration count every
// submit_slow_release() call in this process uses, computed exactly ONCE
// by ensure_slow_release_calibrated() below and cached here. 0 means
// "not yet calibrated" -- submit_slow_release() treats that as a hard
// failure, not a signal to scale from zero (see its own comment). The
// device-global cell submit_spin_kernel()'s loop reads/writes every
// iteration (purely to prevent the compiler folding the loop away; its
// VALUE is never read by any caller) is allocated once, by
// ensure_slow_release_calibrated(), before any spin kernel -- calibration
// or real -- is submitted.
long long g_slow_release_scaled_iterations = 0;
int *     g_slow_release_cell              = nullptr;

// llama.cpp-c6ah: calibrates ONCE per process, against `q`
// (the cache's own backend queue -- the SAME queue a real oneDNN
// Graph-scratch release event actually completes on), and caches the
// resulting scaled iteration count in g_slow_release_scaled_iterations for
// every later submit_slow_release() call to reuse. Must be called
// explicitly from main(), after the cache and its queue exist but BEFORE
// any test runs -- NOT lazily from the first submit_slow_release() call,
// which would let whichever test happens to run first (rather than
// main()) decide the timing context, and a future test-ordering change
// could make that first call calibrate from behind an in-flight kernel a
// DIFFERENT test deliberately left unwaited on this same in-order queue.
//
// This split (calibrate once vs. reuse per call) exists because the
// original per-call design silently timed the WRONG thing whenever an
// earlier test left a slow kernel running on `q`: calibration's own host
// timer captures everything already queued ahead of it on an in-order
// queue, not just the calibration kernel, so a calibration run queued
// behind an earlier sub-test's own unwaited ~1500 ms kernel read ~900 ms
// for an ~84 ms kernel, and the resulting scale collapsed by roughly that
// same factor -- reproduced with a standalone probe (both cards): a
// calibration kernel queued behind a running one measured host=922 ms /
// profiling=83.9 ms on the B50, host=856 ms / profiling=77.9 ms on the
// B70, against ~110 ms host / ~90 ms profiling when run on a drained
// queue. Calibrating once, at the very start of main() before any test has
// had a chance to leave anything in flight, removes the possibility
// entirely rather than trying to detect it after the fact.
//
// A drain (wait_and_throw()) is issued before timing starts anyway, even
// though main() calling this first should already guarantee an idle queue
// -- defensive, not load-bearing, and cheap on an already-idle queue. A
// full kCalibrationIterations run is submitted and discarded first as a
// warm-up: the FIRST spin kernel ever submitted in this process pays a
// one-time JIT/compilation cost that a later real "slow release" kernel
// does not, so timing the very first submission would systematically
// undershoot the target (measured: ~25% low on both cards without a
// warm-up -- still within submit_slow_release()'s own 0.8x floor, but with
// little margin).
void ensure_slow_release_calibrated(sycl::queue & q) {
    static std::once_flag once;
    std::call_once(once, [&]() {
        q.wait_and_throw();

        if (!g_slow_release_cell) {
            g_slow_release_cell = sycl::malloc_device<int>(1, q);
            q.memset(g_slow_release_cell, 0, sizeof(int)).wait_and_throw();
        }

        constexpr long long kCalibrationIterations = 2'000'000;

        sycl::event warmup_evt = submit_spin_kernel(q, g_slow_release_cell, kCalibrationIterations);
        warmup_evt.wait_and_throw();

        const auto  cal_start = std::chrono::steady_clock::now();
        sycl::event cal_evt   = submit_spin_kernel(q, g_slow_release_cell, kCalibrationIterations);
        cal_evt.wait_and_throw();
        long long cal_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cal_start).count();
        const long long cal_profiling_ms = actual_kernel_duration_ms(cal_evt);

        // llama.cpp-c6ah: an implausible calibration reading means something is
        // wrong with the hardware/driver/queue state (e.g. GPU thermal
        // throttling to a crawl, or a stuck queue returning near-instantly
        // without the kernel actually having run), not with the scaling math
        // below -- letting it drive that math unchecked has two failure modes,
        // not one: an implausibly SMALL cal_ms (near zero) scales toward
        // billions of iterations (~3e9 at cal_ms=1), a multi-MINUTE spin that
        // blows this test's own 300 s ctest TIMEOUT rather than failing fast;
        // an implausibly LARGE cal_ms scales toward a near-zero iteration
        // count, silently producing a "slow release" that is not actually
        // slow. Fail loudly (via check(), so the rest of this test still runs)
        // and clamp into the plausible band rather than let either play out
        // silently. 10-2000 ms: comfortably wider than the 110-122 ms this
        // fork's own two discrete cards measured for kCalibrationIterations
        // (test-sycl-event-status-blocking-probe.cpp), without being so wide
        // that a genuinely broken reading could sneak through as "plausible".
        constexpr long long kMinPlausibleCalMs = 10;
        constexpr long long kMaxPlausibleCalMs = 2000;
        check(cal_ms >= kMinPlausibleCalMs && cal_ms <= kMaxPlausibleCalMs,
              "calibration kernel duration is within a plausible band (10-2000 ms) -- an implausible value means "
              "something is wrong with the hardware/driver/queue state, not with the scaling math that follows");
        if (cal_ms < kMinPlausibleCalMs || cal_ms > kMaxPlausibleCalMs) {
            cal_ms = std::max(kMinPlausibleCalMs, std::min(cal_ms, kMaxPlausibleCalMs));
        }

        long long scaled_iterations =
            static_cast<long long>((static_cast<double>(kCalibrationIterations) / static_cast<double>(cal_ms)) *
                                   static_cast<double>(kSlowReleaseMs));
        if (scaled_iterations < 1) {
            scaled_iterations = 1;
        }
        // Upper clamp independent of the plausibility band above: even a
        // plausible calibration rate could scale to an unreasonable iteration
        // count if kSlowReleaseMs were ever raised far beyond its current
        // 1500 ms without revisiting this clamp. 64x kCalibrationIterations
        // -- at the clamped worst case (cal_ms == kMaxPlausibleCalMs, i.e. the
        // slowest rate this function will still scale from) that bounds the
        // resulting kernel to at most 64 x kMaxPlausibleCalMs = 128 s, still
        // comfortably under this test's 300 s ctest TIMEOUT with margin for
        // everything else this binary does.
        constexpr long long kMaxScaledIterationsMultiple = 64;
        const long long     max_scaled_iterations        = kCalibrationIterations * kMaxScaledIterationsMultiple;
        if (scaled_iterations > max_scaled_iterations) {
            scaled_iterations = max_scaled_iterations;
        }

        g_slow_release_scaled_iterations = scaled_iterations;

        printf(
            "  (ensure_slow_release_calibrated: cal_ms=%lld (profiling=%lld ms) for %lld iterations -> "
            "scaled_iterations=%lld targeting %d ms)\n",
            cal_ms, cal_profiling_ms, kCalibrationIterations, g_slow_release_scaled_iterations, kSlowReleaseMs);
    });
}

// Submits (never waits on -- callers park it in-flight, that is the whole
// point) a device-kernel event scaled to hit (a calibrated estimate of)
// kSlowReleaseMs, using the iteration count ensure_slow_release_calibrated()
// computed once for this whole process. Requires that function to have
// already run -- main() calls it explicitly before any test starts (see its
// own comment for why explicitly, and why not lazily from here) -- so a
// zero g_slow_release_scaled_iterations here means a caller (or a future
// test added ahead of main()'s calibration call) skipped that setup, not a
// legitimately-scaled-to-nothing kernel; check() names the missing call
// rather than silently submitting a degenerate (but still memory-safe: 0
// iterations never dereferences g_slow_release_cell) kernel.
slow_release_result submit_slow_release(sycl::queue & q) {
    check(g_slow_release_scaled_iterations > 0,
          "the slow-release kernel was calibrated in main() before this submit -- if this fails, main() is "
          "missing the explicit calibration call");

    sycl::event evt = submit_spin_kernel(q, g_slow_release_cell, g_slow_release_scaled_iterations);
    return { evt, kSlowReleaseMs };
}

// llama.cpp-c6ah: onednn_graph_scratch_pool_entry_release_complete()'s
// completion flag is armed by a SEPARATE, asynchronous watcher marker
// kernel (a device kernel, not a host_task -- see
// onednn_graph_scratch_pool_entry::flag_slot's own comment in
// unified-cache.hpp for why that distinction matters) dispatched once a
// pooled entry's own release event completes (see
// get_event_watch_queue()'s comment in unified-cache.hpp) -- waiting on
// that event itself (sycl::event::wait()) only proves the underlying SYCL
// command is done, not that the watcher has already run and stored the
// flag. Polls for `target` (a specific, already-parked pointer of size
// `size`) to become the pool's own hit for a same-size request, retrying
// up to a bound rather than asserting on the very first call. Returns
// whether `target` itself was ever returned -- a caller must check the
// return value, not just that SOME allocation eventually succeeded (see
// the very first paragraph below for why).
//
// A retry that lands before the watcher fires is itself a genuine MISS --
// a fresh, already-complete decoy allocation -- immediately freed
// (re-parked with a nullptr event, which reads as complete
// unconditionally: no flag, no watcher, no race of its own -- see
// event_complete()'s handling of a default-constructed event) before the
// next attempt, so it cannot leak and cannot be mistaken for `target`.
// `target` always sits FIRST in its size bucket (onednn_graph_scratch_free()
// appends new entries with push_back(), and `target` is never popped until
// it is truly USABLE), so a decoy -- always appended AFTER it -- can never
// be handed back INSTEAD of `target` once `target` itself becomes ready;
// a decoy can only delay how many retries this loop needs, never mask a
// genuine failure to arm `target`'s own flag. IMPORTANT: a decoy CAN
// itself be returned as a hit on some EARLIER iteration than `target`'s
// own (once re-parked, it is immediately reusable) -- callers must not
// infer "target became a hit" from onednn_graph_scratch_pool_hit_count()
// increasing by exactly one across a call to this function; the pointer
// identity this function itself checks is the only reliable signal.
//
// 2000 ms / 5 ms: generous relative to the dispatch gap this bridges (a
// marker kernel dispatched after an already-satisfied dependency is normally
// sub-millisecond even under load), tight enough to fail fast (returning
// false) on a genuine regression rather than hanging the test.
//
// RESIDUAL RACE (documented, not fully closed): at a call site where
// `target`'s own size plus current outstanding DIRECT bytes exceeds the
// cap main() sets (kCapBytes), a MISS retry's fresh allocation attempt engages
// the cap's eviction sweep (onednn_graph_scratch_evict_pool_until_fits_locked()),
// which walks every bucket including `target`'s own. If the watcher arms
// `target`'s flag in the narrow window between this function's own
// try-pool check (EVENT_PENDING) and the sweep's later check on that same
// entry, the sweep can legitimately EVICT `target` for real (it is now
// complete, and headroom is needed) before this function ever sees it as
// a hit -- `target` is then genuinely gone, not merely still pending, and
// this function will correctly time out. This is a real, rare interaction
// between two correct mechanisms (the completion flag and the cap sweep),
// not a bug in either; call sites where outstanding+size exceeds the cap
// document this residual explicitly. On timeout, distinguish the two
// failure shapes via the eviction counter so a flake is diagnosable rather
// than a bare false return.
bool poll_for_pool_hit(unified_cache * cache, sycl::queue & q, size_t size, void * target) {
    const size_t evictions_at_start = cache->onednn_graph_scratch_pool_eviction_count();
    const auto   deadline           = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    for (;;) {
        void * ptr = cache->onednn_graph_scratch_alloc(size, 256, &q);
        if (ptr == target) {
            return true;
        }
        if (ptr) {
            cache->onednn_graph_scratch_free(ptr, nullptr);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            if (cache->onednn_graph_scratch_pool_eviction_count() > evictions_at_start) {
                printf(
                    "    (poll_for_pool_hit: target=%p size=%zu timed out, AND the eviction count rose "
                    "during the poll -- target may have been evicted by the cap-headroom sweep instead of "
                    "becoming a hit; see this call site's own residual-race comment)\n",
                    target, size);
            } else {
                printf(
                    "    (poll_for_pool_hit: target=%p size=%zu timed out with no eviction observed -- the "
                    "completion flag genuinely never armed within the timeout)\n",
                    target, size);
            }
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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
    // lookup that follows finds a genuinely complete entry rather than
    // racing the kernel.
    slow_release_result release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr1, &release.release_event);
    release.release_event.wait();

    // llama.cpp-c6ah: release.wait() above proves the underlying
    // SYCL command is done, but onednn_graph_scratch_pool_entry_release_complete()
    // reads a completion flag armed by a SEPARATE, asynchronous watcher
    // marker kernel (see poll_for_pool_hit()'s own comment) -- the wait
    // does not guarantee that watcher has already run. Poll rather than
    // asserting on the very first call. The dropped "miss count did NOT
    // increase" assertion this replaced is no longer reliably true here: a
    // retry that lands before the watcher fires is itself a genuine miss
    // (see poll_for_pool_hit()) -- the hit-count check below is the
    // property that actually matters (a pool hit is never ALSO counted as
    // a miss for that SAME successful call, which the production code's
    // own hit/miss invariant, documented at
    // onednn_graph_scratch_pool_hit_count_'s declaration, already
    // guarantees independent of this test).
    //
    // RESIDUAL RACE, documented rather than closed (see
    // poll_for_pool_hit()'s own "RESIDUAL RACE" paragraph for the
    // mechanism): kSizeA is 360 MiB, so a MISS retry's fresh allocation
    // (360 MiB new + 360 MiB already parked = 720 MiB) exceeds the 700 MB
    // cap main() sets (via kCapBytes) and engages the cap's eviction sweep,
    // which could in principle evict ptr1 for real in the same narrow
    // window this file's other affected sites describe. Unlike kSizeParked
    // in test_oversized_request_skips_wait_loop(), kSizeA is shared with
    // test_bounded_eviction()'s own cap-triggering logic below (see
    // kSizeA's own static_assert) and is not free to shrink to make this
    // site cap-safe.
    const size_t hits_before = cache->onednn_graph_scratch_pool_hit_count();
    const bool   became_hit  = poll_for_pool_hit(cache, q, kSizeA, ptr1);
    void *       ptr2        = became_hit ? ptr1 : nullptr;

    // became_hit alone (not also a separate ptr2 != nullptr check --
    // ptr2 is became_hit ? ptr1 : nullptr, and ptr1 is already known
    // non-null, so the two predicates are identical) proves both "the
    // reused allocation succeeds" and "the SAME pointer is handed back".
    check(became_hit,
          "the reused allocation succeeds and hands back the SAME pointer -- this is reuse, not a fresh "
          "allocation that happens to land at the same address (polled up to 2 s to tolerate the completion "
          "flag's async watcher dispatch)");
    check(cache->onednn_graph_scratch_pool_hit_count() > hits_before,
          "onednn_graph_scratch_pool_hit_count() increased -- the allocator served this from the reuse pool");

    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }
}

// --- (b) bounded eviction/wait under the DIRECT-path cap --------------------
void test_bounded_eviction(unified_cache * cache, int device) {
    printf("DIRECT path bounded eviction/wait:\n");

    sycl::queue & q = cache->get_queue();

    // llama.cpp-c6ah: leading reclaim -- test_pool_reuse()
    // just above shares kSizeA with this test (deliberately, per its own
    // header comment: the exact-size pool-hit path this test's own comment
    // says to avoid re-testing), and its own poll_for_pool_hit() loop
    // (called on ptr1 there) submits a FRESH kSizeA allocation on every
    // failed poll attempt and immediately re-parks it with a nullptr event
    // (an unconditionally-complete decoy) whenever that attempt does not
    // land the real target -- this is by design (see poll_for_pool_hit()'s
    // own comment), but it means the kSizeA bucket can hold a stray
    // already-complete decoy left over from that churn, on top of the
    // genuine hit-then-reparked entry test_pool_reuse() itself leaves
    // behind at its very end. Without reclaiming here first, THIS test's
    // own setup allocation below can silently reuse one of those decoys
    // (a pool HIT, not the fresh miss this test's cap-pressure design
    // assumes), and -- worse -- a leftover decoy can still be sitting in
    // the kSizeA bucket ALONGSIDE this test's own slow-release entry when
    // the eviction sweep runs a few lines down, letting it evict the
    // decoy (genuinely, correctly complete) instead of ever touching the
    // slow entry this test means to exercise, which is indistinguishable
    // from the outside (eviction count still increases by exactly one) but
    // proves nothing about waiting for an in-flight release. This is what
    // actually explained a both-cards failure once mistaken for a possible
    // flag-correctness bug: the flag itself was never wrong (see event_complete()'s and
    // onednn_graph_scratch_pool_entry_release_complete()'s own comments for
    // the runtime fact that WAS real and required its own fix), but this
    // test's setup could hand the eviction sweep an unrelated, genuinely-
    // complete decoy to evict instead of the entry the test's own elapsed-
    // time bound assumes it is measuring.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (bounded eviction)");

    const size_t misses_before_setup = cache->onednn_graph_scratch_pool_miss_count();
    void *       ptr1                = cache->onednn_graph_scratch_alloc(kSizeA, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the eviction setup succeeds");
    // llama.cpp-c6ah: proves the leading reclaim above actually
    // worked -- without it, this setup allocation could silently be served
    // by a leftover decoy from test_pool_reuse() (a pool HIT), which is
    // exactly the failure mode this reclaim exists to prevent. A regression
    // here fails on this line by name, not as a confusing miss on the
    // eviction-count or elapsed-time bounds further down.
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before_setup + 1,
          "the setup allocation was a DIRECT-path pool miss, not served from a leftover pool entry -- proves "
          "the leading reclaim above actually left the kSizeA bucket empty");
    if (!ptr1) {
        return;
    }

    // A device-kernel event that only completes after (a calibrated
    // estimate of) kSlowReleaseMs -- this is the "slow-to-complete event"
    // the fix spec calls for: ptr1's bytes stay charged against the
    // outstanding-DIRECT counter (it is parked in the reuse pool, not
    // physically released, until this event completes and the pool
    // eviction sweep observes that) for long enough that the request below
    // is guaranteed to observe the cap and enter the wait loop, rather than
    // racing a fast release.
    slow_release_result slow_release    = submit_slow_release(q);
    // llama.cpp-c6ah: the free() call itself must return fast -- this is
    // the exact regression this bound catches: the earlier host_task-based
    // watcher design made SUBMITTING the watcher block the calling thread
    // until the kernel it depended on
    // completed, so onednn_graph_scratch_free() itself stalled for the
    // parked kernel's whole duration on every single park (measured, both
    // cards, 2026-09-09). A device marker kernel submitted the same way
    // does not block its submitting thread, so this call must return in a
    // small fraction of the release delay, not most of it.
    const auto free_call_start = std::chrono::steady_clock::now();
    cache->onednn_graph_scratch_free(ptr1, &slow_release.release_event);
    const long long free_call_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - free_call_start)
            .count();
    printf("    (onednn_graph_scratch_free() itself returned in %lld ms, release duration (calibrated)=%lld ms)\n",
           free_call_ms, slow_release.duration_ms);
    check(free_call_ms < slow_release.duration_ms / 3,
          "onednn_graph_scratch_free() itself returned in well under a third of the release delay -- arming the "
          "completion flag did not block the park call the way the earlier host_task-based watcher design "
          "did");

    const size_t wait_count_before     = cache->onednn_graph_scratch_direct_wait_count();
    const size_t eviction_count_before = cache->onednn_graph_scratch_pool_eviction_count();
    check(cache->onednn_graph_scratch_pool_peak_bytes() >= kSizeA,
          "onednn_graph_scratch_pool_peak_bytes() reflects ptr1 sitting in the pool");

    // llama.cpp-c6ah: a t_flag/t_wait timing diagnostic that used to live
    // here was moved to its own standalone sub-test,
    // test_flag_timing_diagnostic() -- see that function's own comment.
    // Running it here (between the park just above and the timed kSizeB
    // request just below) consumed the ~1500 ms delay this test's own
    // cap-wait bound depends on measuring: the pre-wait eviction sweep
    // observed ptr1 already complete (the diagnostic
    // had already polled the flag AND waited on the event before the timed
    // request even started) and evicted it in 3-4 ms with zero cap waits,
    // making the bound below vacuous.

    // kSizeB, not kSizeA: an exact-size request would be served by the pool
    // reuse path tested above WITHOUT ever reaching the cap check at all,
    // which would prove nothing about eviction/waiting. kSizeA (pooled,
    // pending) + kSizeB (requested) together exceed the 700 MB cap set in
    // main() (see their shared static_assert), so this can only succeed by
    // evicting ptr1's pooled entry once its event completes.
    const auto start = std::chrono::steady_clock::now();
    void *     ptr2  = cache->onednn_graph_scratch_alloc(kSizeB, 256, &q);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    check(ptr2 != nullptr, "the kSizeB allocation eventually succeeds");
    check(cache->onednn_graph_scratch_direct_wait_count() > wait_count_before,
          "onednn_graph_scratch_direct_wait_count() increased -- the allocator actually waited for eviction");
    // Half of the release's own (calibrated) duration, not the whole thing:
    // the poll interval inside the wait loop (kOnednnGraphDirectWaitPollTimeoutMs,
    // 200 ms in unified-cache.cpp) means the observed wait can undershoot
    // the event's own completion time by up to one poll, and this only
    // needs to distinguish "actually waited" from "returned immediately".
    check(elapsed.count() >= slow_release.duration_ms / 2,
          "the kSizeB allocation took a substantial fraction of the release delay, not ~0 ms");
    check(cache->onednn_graph_scratch_pool_eviction_count() > eviction_count_before,
          "onednn_graph_scratch_pool_eviction_count() increased -- ptr1's pooled entry was actually released, "
          "not just waited on");
    // slow_release.release_event is confirmed complete at this point -- the
    // eviction just asserted above only happens once
    // onednn_graph_scratch_pool_entry_release_complete() reports it so --
    // so reading its ACTUAL device-measured duration here (as opposed to
    // submit_slow_release()'s own pre-submission calibrated ESTIMATE) is
    // valid; see actual_kernel_duration_ms()'s own comment for why both
    // numbers are printed rather than just the estimate.
    const long long actual_ms = actual_kernel_duration_ms(slow_release.release_event);
    printf(
        "    (elapsed=%lld ms, release duration (calibrated)=%lld ms, actual=%lld ms, cap wait count %zu -> "
        "%zu, eviction count %zu -> %zu)\n",
        static_cast<long long>(elapsed.count()), slow_release.duration_ms, actual_ms, wait_count_before,
        cache->onednn_graph_scratch_direct_wait_count(), eviction_count_before,
        cache->onednn_graph_scratch_pool_eviction_count());
    // llama.cpp-c6ah: the calibrated ESTIMATE printed above is
    // only useful as a timing-bound denominator if the kernel actually ran
    // close to it -- check the ACTUAL measured duration directly, rather
    // than trusting the estimate, so a calibration undershoot (e.g. a
    // future test reordering that calibrates behind an in-flight kernel
    // again) fails on this line by name instead of surfacing as a
    // confusing miss on the elapsed-time bound above.
    check(actual_ms >= (slow_release.duration_ms * 8) / 10,
          "the release kernel's ACTUAL measured duration is at least 80% of its calibrated target -- otherwise "
          "this test's timing bounds are being checked against a kernel that undershot calibration");

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

    constexpr size_t kSizeC = 400ull * 1024 * 1024;
    static_assert(kSizeC > kZoneFloorBytes && kSizeC < kCapBytes,
                  "kSizeC must exceed the zone floor and fit under the cap alone -- distinctness from the "
                  "other size constants in this file (all integer literals, so distinctness is verifiable by "
                  "inspection rather than a static_assert) is what actually avoids aliasing a pooled entry "
                  "from another test, not any relation to the cap or zone");
    void * ptr = cache->onednn_graph_scratch_alloc(kSizeC, 256, &q);

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

    // 370 MiB: comfortably under the 700 MB cap main() sets via
    // GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB, so both allocations below take
    // the NORMAL pool path this test actually exercises (reclaim/pending-
    // event handling) rather than the size>cap early-out
    // (onednn_graph_scratch_wait_for_direct_headroom_locked(), llama.cpp-pqgl)
    // -- that early-out has its own dedicated test below,
    // test_oversized_request_skips_wait_loop(). Distinct from
    // kSizeA/kSizeB/kSizeC (360/380/400 MiB) above.
    constexpr size_t kSizeD = 370ull * 1024 * 1024;
    static_assert(kSizeD > kZoneFloorBytes && kSizeD < kCapBytes, "see the prose above");

    void * ptr = cache->onednn_graph_scratch_alloc(kSizeD, 256, &q);
    check(ptr != nullptr, "DIRECT allocation for the pending-event reclaim setup succeeds");
    if (!ptr) {
        return;
    }

    // Free with a slow event and do NOT wait on it -- the entry parks in the
    // pool with an INCOMPLETE release event, exactly the state
    // onednn_graph_scratch_clear_pool_locked() must handle without
    // destructing the owning mem_handle out from under a still-in-flight
    // device kernel.
    // Not asserted here: onednn_graph_scratch_pool_peak_bytes() is a
    // cumulative process-lifetime high-water mark that never resets, so a
    // "peak >= kSizeD" check right after this free() would be vacuous once
    // an earlier test (test_bounded_eviction, kSizeA/kSizeB = 360/380 MiB)
    // has already pushed the peak above kSizeD (370 MiB) -- it would hold
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
    slow_release_result slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr, &slow_release.release_event);

    const size_t evictions_before = cache->onednn_graph_scratch_pool_eviction_count();

    // Reclaim now, while slow_release is (overwhelmingly likely, given its
    // calibrated duration) still incomplete -- this is the exact call
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

    slow_release.release_event.wait();  // let the kernel finish before the process exits
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
    // 280 MiB: exceeds the ~256 MB no-model ONEDNN zone floor this process
    // reserves (measured on hardware: `[VRAM-ARENA] Reserved single chunk:
    // ... oneDNN=256.0` -- see kCapMiB's own comment for why that floor is
    // not configurable away for this no-model case) -- 200 MiB measured
    // FAIL there, zone-served rather than pooled, which is why this test
    // still checks a miss below rather than trusting the size by
    // inspection. Distinct from kSizeA/kSizeB/kSizeC/kSizeD
    // (360/380/400/370 MiB) elsewhere in this file, and comfortably under
    // the 700 MB cap main() sets so it can be parked and later evicted
    // without itself engaging the cap machinery this test isn't
    // exercising. Also small enough that DOUBLING it (this entry parked
    // plus a same-size probe requested while polling for it to become
    // ready, below) still fits under the cap on its own (280 x 2 = 560 <=
    // 700), so that poll cannot engage the eviction sweep and risk evicting
    // THIS entry for real before the poll ever sees it as a hit (see
    // poll_for_pool_hit()'s own "RESIDUAL RACE" comment) -- unlike kSizeA
    // (shared with test_bounded_eviction's own cap-triggering logic and not
    // free to shrink), this constant is scoped to this function alone, so
    // it costs nothing to keep it cap-safe here, unlike the other two
    // affected call sites in this file, which must document the residual
    // race instead (see their own comments). The miss-count assertion right
    // after the allocation below is this setup's own self-check against
    // silently regressing back to a zone-served size.
    constexpr size_t kSizeParked = 280ull * 1024 * 1024;
    static_assert(kSizeParked > kZoneFloorBytes, "see the prose above");
    static_assert(kSizeParked * 2 <= kCapBytes,
                  "kSizeParked parked plus a same-size retry probe must fit under the cap without engaging the "
                  "eviction sweep -- closes the residual race documented in poll_for_pool_hit()");
    const size_t misses_before_park      = cache->onednn_graph_scratch_pool_miss_count();
    const size_t outstanding_before_park = cache->onednn_graph_scratch_direct_outstanding_bytes();
    void *       parked                  = cache->onednn_graph_scratch_alloc(kSizeParked, 256, &q);
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
    // (400 MiB) request is deliberately forced to fail
    // (ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail(2),
    // asserted via ptr == nullptr) and returns before ever reaching this
    // counter's write site, so it contributes nothing. kSizeA/kSizeB/kSizeD
    // (360/380/370 MiB) already push the high-water past kSizeParked (280
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
        slow_release_result release = submit_slow_release(q);
        cache->onednn_graph_scratch_free(parked, &release.release_event);
        release.release_event.wait();  // ensure the underlying event is complete

        // llama.cpp-c6ah: release.wait() above only proves the
        // underlying SYCL command is done, not that
        // onednn_graph_scratch_pool_entry_release_complete()'s asynchronous
        // watcher marker kernel has already run and armed the flag (see
        // poll_for_pool_hit()'s own comment). The oversized request below
        // is a ONE-SHOT for this whole binary (see this test's own
        // top-of-function comment: it must remain the only oversized
        // request issued anywhere in this process), so that race cannot be
        // resolved by retrying the real request itself the way the other
        // tests retry a same-size allocation. Resolve it here instead: round-
        // trip `parked` through a same-size probe (poll_for_pool_hit only
        // succeeds once the flag is genuinely armed) and re-park it with a
        // nullptr event, which reads as complete unconditionally -- no
        // flag, no watcher, no race (see event_complete()'s handling of a
        // default-constructed event) -- so the real request below cannot
        // race this entry's own watcher.
        const bool became_ready = poll_for_pool_hit(cache, q, kSizeParked, parked);
        check(became_ready,
              "the parked entry becomes reusable once its release event completes (polled up to "
              "2 s to tolerate the completion flag's async watcher dispatch)");
        if (became_ready) {
            cache->onednn_graph_scratch_free(parked, nullptr);
        }
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

    // main() sets the cap to kCapMiB (700) MB via
    // GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB. 720 MiB is comfortably over
    // that ON ITS OWN -- no amount of eviction or waiting could ever bring
    // outstanding+size under the cap for this request.
    constexpr size_t kSizeOversized = 720ull * 1024 * 1024;
    static_assert(kSizeOversized > kCapBytes, "must exceed the cap on its own to take the size>cap early-out");

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
// host-visible flag (armed by a device marker kernel -- not a host_task,
// see onednn_graph_scratch_pool_entry::flag_slot's own comment in
// unified-cache.hpp for why -- on a new, dedicated, non-profiling watcher
// queue) over querying the event.
//
// CONFIRMED ON HARDWARE, both discrete cards this fork validates against
// (tests/test-sycl-event-status-blocking-probe.cpp, a standalone
// decisive-experiment isolating this exact query from the pool allocator
// entirely): the query blocks specifically for a DEVICE-KERNEL-produced
// event on a profiling-enabled queue (B50: query=118 ms vs. the kernel's
// own 122 ms duration; B70: 110 ms vs. 114 ms), and polls (a few ms or
// less) for a host_task-produced event on either queue kind, and for a
// device-kernel event on a non-profiling queue. Every real pool release
// event in production is device-kernel-produced (oneDNN's own free
// callback), so this test's own "slow release" events are too -- see
// submit_slow_release()'s own comment for the history (this file used a
// host_task-produced release event until the probe measured otherwise,
// which measured a ~6 ms RED arm below instead of reproducing the block
// this ticket's fix actually closes).
//
// WHY THIS TEST NEEDS A RAISED CAP, NOT A SHRUNK ZONE. Proving "skipped,
// not waited" with a wall-clock bound requires a request that does NOT
// need the DIRECT-path cap machinery to engage at all -- otherwise a
// genuine, unavoidable physical wait for the SAME in-flight entry's
// completion is indistinguishable from the bug this fix closes (both take
// about kSlowReleaseMs either way; only WHICH mechanism produces that wait
// differs, and this file has no way to observe that internal difference
// through the public API alone). That requires parking the SAME size twice
// simultaneously while staying under the cap main() sets, and every
// DIRECT-triggering size in this file (this test's own included) must
// exceed the ~256 MB no-model ONEDNN zone floor this process reserves --
// see kCapMiB's own comment for the measurement, and for why
// GGML_SYCL_ONEDNN_GRAPH_ZONE_MB, despite being this codebase's documented
// escape hatch for exactly this shape of problem, does NOT actually shrink
// that floor for this no-model case (an earlier version of this test tried
// exactly that override and it measured inert on hardware -- the arena
// still reported `oneDNN=256.0` -- so kSizeSkip below WAS being served from
// the zone, not the DIRECT path, and every downstream assertion in this
// function was vacuous). kCapMiB is 700, not the historical 350, precisely
// so that two zone-exceeding sizes CAN be outstanding together (kSizeSkip
// below is chosen with its own static_assert proving exactly that). This
// does not change what any earlier test in this file measures -- every one
// of them already relies on its own size exceeding the zone floor, which
// is unaffected by the cap's value.
void test_in_flight_entry_is_skipped_not_waited(unified_cache * cache, int device) {
    printf("DIRECT path in-flight pooled entry is skipped, not waited on:\n");

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup");

    sycl::queue & q = cache->get_queue();

    // 300 MiB: exceeds the no-model ONEDNN zone floor (see kCapMiB's own
    // comment), and genuinely distinct from every other size constant in
    // this file (including this SAME function's own later
    // kSizeEvictInFlight/kSizeEvictRequest below, which happen to share
    // this value too -- but sharing a VALUE across two DIFFERENT constants
    // is not an aliasing risk by itself; onednn_graph_scratch_reuse_pool_
    // is keyed on the byte value, not on which C++ name produced it, so two
    // same-sized entries from different sub-tests genuinely are the same
    // pool bucket). What actually makes this test's own leading reclaim
    // call above load-bearing is NOT name/value collision at all: every
    // earlier test in this file frees its buffers with a nullptr event,
    // which PARKS them (still charged against
    // onednn_graph_scratch_direct_outstanding_bytes_) rather than
    // releasing them for real. In the CURRENT call order in main(), the
    // immediately preceding test (test_oversized_request_skips_wait_loop)
    // already reclaims at its own teardown, so this test's leading reclaim
    // is not strictly needed to reach a clean pool THIS run -- it is
    // ORDER-INDEPENDENCE INSURANCE against a future reordering of the
    // main() call sequence, a future test inserted between that one and
    // this one that does not clean up after itself, or that teardown call
    // being removed later. Without it, whatever bytes an earlier test left
    // parked would still be charged against the shared 700 MB cap when
    // this test starts, and could leave less headroom under the cap than
    // kSizeSkip's own doubled cost needs -- exactly the headroom its
    // `kSizeSkip * 2 <= kCapBytes` static_assert below proves is available
    // in isolation, a proof this reclaim is what actually makes true at
    // runtime. Two of these must be outstanding at once (this one parked,
    // plus a fresh same-size request) without engaging the cap.
    constexpr size_t kSizeSkip = 300ull * 1024 * 1024;
    static_assert(kSizeSkip > kZoneFloorBytes, "must exceed the zone floor to take the DIRECT path");
    static_assert(kSizeSkip * 2 <= kCapBytes,
                  "two outstanding at once must not themselves engage the cap machinery this test isn't "
                  "exercising in its GREEN-arm scenario");

    const size_t misses_before_setup = cache->onednn_graph_scratch_pool_miss_count();
    void *       ptr1                = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the skip-not-wait setup succeeds");
    check(cache->onednn_graph_scratch_pool_miss_count() == misses_before_setup + 1,
          "the setup allocation was a DIRECT-path pool miss, not served from the ONEDNN zone -- proves this "
          "setup actually parks a poolable entry rather than silently zone-serving it");
    if (!ptr1) {
        return;
    }

    // Park with a slow, UNWAITED release event -- the entry sits in the pool
    // with an INCOMPLETE release_event for the rest of this sub-scenario.
    slow_release_result slow_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr1, &slow_release.release_event);

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
    // Third of the release's own (calibrated) duration, not half: this call
    // does strictly less work than test_bounded_eviction's cap-constrained
    // wait (no cap is engaged here at all -- see this test's header
    // comment), so it can afford a tighter bound while remaining well clear
    // of scheduler/CI noise.
    // ASSUMPTION (llama.cpp-c6ah, unverified until this runs on
    // hardware): the DIRECT allocation path's own unified_alloc() does not
    // itself submit anything onto `q` -- `q` currently has ptr1's slow
    // kernel still running on it (submit_slow_release() submitted it to
    // this same in-order queue). If that assumption is wrong, this bound
    // would fail even with the fix correctly applied, because the fresh
    // allocation would have to wait its turn behind the slow kernel
    // regardless of how onednn_graph_scratch_pool_entry_release_complete()
    // itself behaves -- a queue-ordering wait, not the blocking-query wait
    // this check targets. Elapsed is printed either way so a failure here
    // is diagnosable rather than a bare FAIL.
    check(elapsed.count() < slow_release.duration_ms / 3,
          "the re-request returned in well under a third of the release delay -- it did not wait for ptr1's "
          "release event to complete");
    printf("    (elapsed=%lld ms, release duration (calibrated)=%lld ms, immediate re-request)\n",
           static_cast<long long>(elapsed.count()), slow_release.duration_ms);

    // llama.cpp-c6ah: free ptr2 HERE, before polling for ptr1 below, not
    // after -- see the poll's own comment for why the ORDER matters, not
    // just that it eventually happens.
    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }

    // Now let ptr1's release actually complete, and confirm a THIRD
    // same-size request is a genuine pool HIT -- the in-flight entry was
    // skipped above, not lost.
    //
    // llama.cpp-c6ah: slow_release.wait() above only proves the
    // underlying SYCL command is done, not that the async watcher marker
    // kernel that arms the completion flag has already run (see
    // poll_for_pool_hit()'s own comment) -- poll rather than asserting on
    // the very first call.
    //
    // llama.cpp-c6ah: freeing ptr2 BEFORE this poll (immediately above,
    // not after it) is load-bearing, not cosmetic. ptr2's free re-parks it
    // as an unconditionally-complete (nullptr-event) decoy in the SAME
    // kSizeSkip bucket ptr1 sits in, so onednn_graph_scratch_try_pool_locked()'s
    // exact-size lookup -- which every onednn_graph_scratch_alloc() call
    // checks FIRST, unconditionally -- always finds a usable entry in this
    // bucket (ptr2's decoy, whenever ptr1 itself is not yet ready) and
    // NEVER falls through to onednn_graph_scratch_alloc_direct_locked()'s
    // cap-headroom machinery at all, for any poll iteration. This is about
    // POOLABILITY, not outstanding bytes: a parked entry stays charged
    // against onednn_graph_scratch_direct_outstanding_bytes_ regardless of
    // whether it is complete or pending (see that field's own comment in
    // unified-cache.hpp, and onednn_graph_scratch_try_reuse_pool_locked()'s
    // own note that a pool hit does not touch it) -- so outstanding here is
    // ptr1 + ptr2 = kSizeSkip x 2 = 600 MiB either way, WITH or without this
    // fix. What the fix changes is that the bucket ALWAYS holds a usable
    // entry (ptr1 once ready, or ptr2's decoy otherwise), so every retry is
    // served from the pool and never reaches the cap check at all -- not
    // that a fresh miss would somehow stay under the cap on its own (it
    // would not: 600 + 300 = 900 MiB exceeds the 700 MB cap, same as
    // before). Freeing ptr2 AFTER this poll instead (the ordering this
    // replaced) left it checked out rather than parked -- still charged
    // identically, but NOT poolable -- so the bucket could be empty
    // whenever ptr1 itself was not yet ready, a miss retry's fresh 300 MiB
    // request DID fall through to the cap machinery, and every such retry
    // engaged the exact residual race poll_for_pool_hit() documents
    // (undocumented at this specific site), risking each retry taking up
    // to that machinery's own internal 5 s bounded wait, not the "up to 2 s"
    // this function's own message claims.
    slow_release.release_event.wait();
    // ptr1's release event is confirmed complete by the wait() just above
    // (unlike at the "immediate re-request" print earlier in this
    // function, where it was deliberately still pending) -- safe to read
    // its ACTUAL device-measured duration here, alongside the calibrated
    // ESTIMATE printed earlier, so a reader can see how close the estimate
    // actually landed on this run's hardware.
    const long long ptr1_actual_ms = actual_kernel_duration_ms(slow_release.release_event);
    printf("    (ptr1's release: calibrated estimate=%lld ms, actual=%lld ms)\n", slow_release.duration_ms,
           ptr1_actual_ms);
    // llama.cpp-c6ah: same reasoning as test_bounded_eviction's
    // own actual-vs-calibrated check -- fail on this line, by name, rather
    // than let a calibration undershoot surface as a confusing miss on the
    // GREEN/RED arm bounds below that both depend on slow_release's
    // duration being close to its target.
    check(ptr1_actual_ms >= (slow_release.duration_ms * 8) / 10,
          "ptr1's release kernel ACTUAL measured duration is at least 80% of its calibrated target -- otherwise "
          "this test's timing bounds are being checked against a kernel that undershot calibration");
    const size_t hits_before_reuse = cache->onednn_graph_scratch_pool_hit_count();

    // llama.cpp-c6ah: time the poll itself (not just assert its
    // eventual outcome below) and print when it turned into a hit -- direct
    // evidence for how long the watcher's async dispatch actually took after
    // ptr1's kernel completed, alongside test_bounded_eviction's own flag
    // poll instrumentation.
    const auto      poll_reuse_start = std::chrono::steady_clock::now();
    const bool      became_hit_reuse = poll_for_pool_hit(cache, q, kSizeSkip, ptr1);
    // static_cast<long long>, not a bare .count(): std::chrono::milliseconds::rep
    // is `long` on this platform, not `long long`, so printing it directly
    // against %lld is a -Wformat mismatch -- same cast this file already
    // applies at every other %lld duration print (e.g. this function's own
    // elapsed2/actual_ms prints above).
    const long long poll_reuse_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - poll_reuse_start)
            .count();
    printf("    (poll_for_pool_hit turned into a hit at %lld ms: %s)\n", poll_reuse_ms,
           became_hit_reuse ? "hit" : "timed out, never a hit");
    void * ptr3 = became_hit_reuse ? ptr1 : nullptr;
    check(became_hit_reuse,
          "once ptr1's release event completes (and its watcher marker kernel has run), a "
          "same-size request eventually becomes a genuine pool hit (polled up to 2 s to tolerate the async "
          "dispatch)");
    // `>`, not `== + 1`: poll_for_pool_hit() itself documents why -- a
    // retry that lands before ptr1's own watcher fires can be served by a
    // DECOY hit first (a fresh miss from an earlier retry, re-parked and
    // then reused), which also increments this counter. ptr3 == ptr1
    // (checked via became_hit_reuse above) is the real proof this was
    // ptr1 specifically; this check only needs "at least one real hit
    // happened", not an exact count.
    check(cache->onednn_graph_scratch_pool_hit_count() > hits_before_reuse,
          "onednn_graph_scratch_pool_hit_count() increased -- this time it really was a pool hit");

    if (ptr3) {
        cache->onednn_graph_scratch_free(ptr3, nullptr);
    }

    // llama.cpp-c6ah: reclaim the pool before the RED arm below.
    // Without this, ptr2's free earlier above (before the poll) and ptr3's
    // free immediately above each PARK a fresh, already-complete
    // (nullptr-event) decoy of kSizeSkip into the SAME bucket rather than
    // releasing it for real (onednn_graph_scratch_free()'s DIRECT branch
    // always parks under the per-size depth limit) -- so the RED arm below
    // would find a ready-made complete entry sitting in the bucket
    // regardless of what ptr4's OWN (still in-flight) entry is
    // doing, making both RED-arm assertions vacuous (ptr5 would reuse a
    // leftover decoy immediately, not ptr4).
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (RED arm)");

    // --- Positive control: the RED arm this GREEN binary can still produce.
    // llama.cpp-c6ah: forcing this hook makes onednn_graph_scratch_free() SKIP ARMING the
    // completion flag for ptr4's park below (flag_slot stays -1 for that
    // one entry), rather than -- an earlier version of this hook --
    // making onednn_graph_scratch_pool_entry_release_complete() branch on
    // the hook and query release_event directly regardless of flag_slot.
    // Skipping the arm means the SAME flag_slot == -1 fallback
    // release_complete() already uses when arming fails now runs against a
    // genuinely UNWATCHED event for ptr4's entry, which BLOCKS for real (a
    // bare command_execution_status query on an unwatched device-kernel
    // event blocks on a profiling-enabled queue -- measured on hardware,
    // both cards; see event_complete()'s own comment) -- so the SAME
    // scenario above should now take roughly the full release delay AND
    // come back as a pool HIT (blocking until complete makes the entry
    // look immediately USABLE once it returns), the opposite of both
    // outcomes just asserted. Must be set BEFORE the free() below that
    // parks ptr4 -- the hook is read at PARK time, not at check time.
    //
    // llama.cpp-c6ah: confirmed real on hardware, not the misdiagnosis an
    // earlier version of this bound's own comment used to describe --
    // measured the forced-fallback bare query blocking for the kernel's
    // full ~1515/1516 ms on both cards, matching this bound. The mechanism
    // this arm relies on (flag_slot == -1 -> a bare query on a genuinely
    // unwatched event) was never wrong; what was wrong was a SEPARATE,
    // since-removed design -- arming the flag via a host_task,
    // which was found to block onednn_graph_scratch_free() itself, not the
    // query this RED arm exercises.
    ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check(true);

    void * ptr4 = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
    check(ptr4 != nullptr, "RED-arm setup allocation succeeds");
    if (ptr4) {
        slow_release_result slow_release2 = submit_slow_release(q);
        cache->onednn_graph_scratch_free(ptr4, &slow_release2.release_event);

        const auto start2 = std::chrono::steady_clock::now();
        void *     ptr5   = cache->onednn_graph_scratch_alloc(kSizeSkip, 256, &q);
        const auto elapsed2 =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start2);

        // 80%, not 100%: the same poll-granularity slack test_bounded_eviction
        // already accepts for the analogous genuine-wait case, applied here
        // to a real blocking call instead of a poll loop -- comfortably
        // above the /3 bound the GREEN arm above must stay under, so the two
        // checks cannot both pass on the same (broken) code.
        //
        // This bound is now expected to actually hold: measured on hardware
        // (tests/test-sycl-event-status-blocking-probe.cpp, both cards),
        // the bare command_execution_status query blocks until completion
        // specifically for a device-kernel-produced event on a
        // profiling-enabled queue -- exactly what slow_release2 now is (see
        // submit_slow_release()'s own comment for why this test switched
        // away from a host_task release event, which measured ~6 ms here
        // instead of reproducing the block at all). If this still shows
        // ~0 ms on hardware despite that, something has regressed in either
        // this test's own event source or the driver's behaviour changed --
        // do not loosen this bound to make a ~0 ms result pass silently.
        check(elapsed2.count() >= (slow_release2.duration_ms * 8) / 10,
              "with the RED hook forced, the re-request took most of the release delay -- reproducing the "
              "pre-fix blocking behaviour this test would otherwise never exercise");
        check(ptr5 == ptr4,
              "...and the blocking query left the entry USABLE by the time it returned, so this was a pool HIT, "
              "not a fresh allocation -- the opposite of the GREEN-arm outcome above");
        // slow_release2.release_event is confirmed complete at this point:
        // ptr5 == ptr4 just above only holds if the blocking query itself
        // already observed it complete before returning. Reading its
        // ACTUAL device-measured duration here, alongside the calibrated
        // ESTIMATE, is what actually distinguishes "the query did not
        // block" from "the kernel happened to run shorter than the
        // estimate" if this bound ever fails.
        const long long red_actual_ms = actual_kernel_duration_ms(slow_release2.release_event);
        printf(
            "    (elapsed=%lld ms, release duration (calibrated)=%lld ms, actual=%lld ms, RED arm / "
            "force_blocking_pool_check)\n",
            static_cast<long long>(elapsed2.count()), slow_release2.duration_ms, red_actual_ms);
        // llama.cpp-c6ah: same reasoning as test_bounded_eviction's
        // own actual-vs-calibrated check -- this is the RED arm's own
        // positive-control bound (elapsed2 >= 80% of duration_ms just
        // above), so a calibration undershoot here would otherwise read as
        // "the blocking premise did not reproduce" rather than what it
        // actually is: the kernel used for the control finished early.
        check(red_actual_ms >= (slow_release2.duration_ms * 8) / 10,
              "the RED-arm release kernel's ACTUAL measured duration is at least 80% of its calibrated target -- "
              "otherwise this test's positive control is being checked against a kernel that undershot "
              "calibration");

        if (ptr5) {
            cache->onednn_graph_scratch_free(ptr5, nullptr);
        }
    }

    ggml_sycl_test_onednn_graph_scratch_force_blocking_pool_check(false);

    // --- Eviction sweep must apply the identical skip, not just the
    // try-reuse lookup above. Park one already-complete entry and one
    // in-flight (incomplete) entry of a DIFFERENT size, then request a
    // THIRD, different size that needs headroom the completed entry alone
    // can supply -- the in-flight entry must be left alone (not evicted,
    // not waited on) either way.
    //
    // llama.cpp-c6ah: park the COMPLETED entry FIRST, and
    // confirm via poll_for_pool_hit's probe-and-reparent round trip (same
    // async-watcher reasoning as elsewhere in this test) that it is
    // genuinely, unambiguously complete BEFORE the in-flight entry is even
    // allocated. The version this replaced submitted the in-flight entry's
    // slow release FIRST and the "completed" entry's release SECOND, both
    // on this same IN-ORDER queue `q`, then waited only on the second --
    // on an in-order queue that wait is satisfied only once BOTH
    // host_tasks have run, so the "in-flight" entry was ALSO already
    // complete by the time the eviction sweep ran below, making the whole
    // sub-test vacuous (it could not have told a fixed allocator apart
    // from a broken one -- both would evict either entry with nothing to
    // stop them).
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (eviction sweep)");

    constexpr size_t kSizeEvictInFlight = 300ull * 1024 * 1024;
    constexpr size_t kSizeEvictComplete = 260ull * 1024 * 1024;
    constexpr size_t kSizeEvictRequest  = 300ull * 1024 * 1024;
    static_assert(kSizeEvictInFlight > kZoneFloorBytes && kSizeEvictComplete > kZoneFloorBytes &&
                      kSizeEvictRequest > kZoneFloorBytes,
                  "every size here must exceed the zone floor to take the DIRECT path");
    static_assert(kSizeEvictInFlight + kSizeEvictComplete <= kCapBytes, "parking both must not itself force anything");
    static_assert(kSizeEvictInFlight + kSizeEvictComplete + kSizeEvictRequest > kCapBytes,
                  "satisfying the request needs SOME eviction");
    static_assert(kSizeEvictInFlight + kSizeEvictRequest <= kCapBytes,
                  "evicting kSizeEvictComplete alone must be sufficient -- the in-flight entry never needs to "
                  "be touched to succeed");
    // kSizeEvictInFlight + kSizeEvictComplete (560 MiB) fits under the
    // 700 MB cap on its own; + kSizeEvictRequest (860 MiB) does not, so
    // satisfying this request needs SOME eviction, and kSizeEvictInFlight +
    // kSizeEvictRequest alone (600 MiB) fits -- so evicting
    // kSizeEvictComplete is both necessary and SUFFICIENT (see the three
    // static_asserts above, which are the actual proof; these numbers are
    // restated here only for a reader scanning prose, not code). Deliberately
    // DIFFERENT sizes for in-flight vs. completed (not just distinct from
    // each other for bucket separation): it lets
    // onednn_graph_scratch_direct_outstanding_bytes() alone distinguish
    // WHICH entry the sweep evicted below, without a third probe allocation
    // that would itself need headroom. kSizeEvictRequest deliberately equal
    // to kSizeEvictInFlight is fine and not a bug: a same-size request still
    // correctly skips a NOT-YET-USABLE entry in its own exact-size bucket
    // (EVENT_PENDING is checked regardless of which caller asked), so it
    // falls through to the general cap-eviction path exactly like a
    // different-sized request would.

    // llama.cpp-c6ah: kSizeEvictComplete (260 MiB) has the thinnest margin
    // over kZoneFloorBytes (256 MiB) of any size in this file, so this
    // allocation gets its own explicit zone-miss self-check, matching the
    // pattern kSizeSkip/kSizeParked already use elsewhere -- without it, a
    // regression that shrank the effective zone floor enough to zone-serve
    // this specific size would surface as a confusing poll_for_pool_hit()
    // failure below rather than naming the actual cause: a zone-served
    // allocation for a fixed size returns the SAME address on every
    // request (unlike a DIRECT allocation), so it could even satisfy
    // `ptr == target` on the very first poll call and mask the regression
    // entirely rather than timing out.
    const size_t evict_complete_misses_before = cache->onednn_graph_scratch_pool_miss_count();
    void *       evict_complete               = cache->onednn_graph_scratch_alloc(kSizeEvictComplete, 256, &q);
    check(evict_complete != nullptr, "eviction-sweep completed-entry setup allocation succeeds");
    check(cache->onednn_graph_scratch_pool_miss_count() == evict_complete_misses_before + 1,
          "the completed-entry setup allocation was a DIRECT-path pool miss, not served from the ONEDNN "
          "zone -- proves this setup actually parks a poolable entry rather than silently zone-serving it");
    if (evict_complete) {
        slow_release_result complete_release = submit_slow_release(q);
        cache->onednn_graph_scratch_free(evict_complete, &complete_release.release_event);
        complete_release.release_event.wait();  // ensure the underlying event is complete

        const bool complete_ready = poll_for_pool_hit(cache, q, kSizeEvictComplete, evict_complete);
        check(complete_ready,
              "the completed entry becomes reusable once its release event completes (polled "
              "up to 2 s to tolerate the completion flag's async watcher dispatch)");
        if (complete_ready) {
            // Re-park with a nullptr event -- unconditionally complete (no
            // flag, no watcher, no race of its own) for the real request
            // below.
            cache->onednn_graph_scratch_free(evict_complete, nullptr);
        }
    }

    void * evict_in_flight = cache->onednn_graph_scratch_alloc(kSizeEvictInFlight, 256, &q);
    check(evict_in_flight != nullptr, "eviction-sweep in-flight setup allocation succeeds");
    slow_release_result evict_slow_release = submit_slow_release(q);
    if (evict_in_flight) {
        cache->onednn_graph_scratch_free(evict_in_flight, &evict_slow_release.release_event);
    }

    const size_t sweep_evictions_before   = cache->onednn_graph_scratch_pool_eviction_count();
    const size_t sweep_outstanding_before = cache->onednn_graph_scratch_direct_outstanding_bytes();

    const auto sweep_start = std::chrono::steady_clock::now();
    void *     evict_ptr   = cache->onednn_graph_scratch_alloc(kSizeEvictRequest, 256, &q);
    const auto sweep_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sweep_start);

    check(evict_ptr != nullptr, "the headroom-needing request succeeds");
    check(cache->onednn_graph_scratch_pool_eviction_count() == sweep_evictions_before + 1,
          "exactly one entry was evicted");
    // llama.cpp-c6ah: check WHICH entry the count above refers
    // to, not just that some count went up by one -- kSizeEvictInFlight and
    // kSizeEvictComplete are deliberately different sizes (see their own
    // comment above), so outstanding bytes after this request can only
    // equal kSizeEvictInFlight + kSizeEvictRequest if the COMPLETED entry
    // (kSizeEvictComplete) was the one removed; it would instead equal
    // kSizeEvictComplete + kSizeEvictRequest if the sweep had wrongly
    // evicted the in-flight entry.
    check(cache->onednn_graph_scratch_direct_outstanding_bytes() ==
              sweep_outstanding_before - kSizeEvictComplete + kSizeEvictRequest,
          "outstanding bytes reflect exactly the in-flight entry's size plus the fresh request -- the "
          "completed entry's bytes were removed (evicted), the in-flight entry's were not (still pooled, "
          "untouched)");
    // Same queue-ordering assumption as the GREEN arm's own bound above
    // (see the "ASSUMPTION" comment there): this bound also presumes
    // onednn_graph_scratch_alloc()'s own unified_alloc() call does not
    // itself submit anything onto `q`, which currently has evict_in_flight's
    // slow kernel still running on it.
    check(sweep_elapsed.count() < evict_slow_release.duration_ms / 3,
          "the eviction sweep did not wait for the in-flight entry's release event -- evicting the completed "
          "entry alone was enough headroom");
    printf("    (elapsed=%lld ms, release duration (calibrated)=%lld ms, eviction sweep)\n",
           static_cast<long long>(sweep_elapsed.count()), evict_slow_release.duration_ms);

    if (evict_ptr) {
        cache->onednn_graph_scratch_free(evict_ptr, nullptr);
    }

    // llama.cpp-c6ah: the in-flight entry must have SURVIVED,
    // not merely gone unevicted by coincidence -- confirm it is still
    // pooled and becomes a genuine hit once its own release completes.
    //
    // RESIDUAL RACE, documented rather than closed (see
    // poll_for_pool_hit()'s own "RESIDUAL RACE" paragraph): outstanding
    // bytes here are kSizeEvictInFlight (300 MiB, still parked) plus
    // evict_ptr's own re-parked 300 MiB (kSizeEvictRequest) = 600 MiB, so a
    // MISS retry's fresh 300 MiB allocation (600 + 300 = 900 MiB) exceeds
    // the 700 MB cap and engages the eviction sweep. Adjusting the sizes
    // here would not close this the way kSizeParked was chosen to be
    // cap-safe elsewhere in this file: evict_ptr's 300 MiB is itself
    // already charged and outstanding with no cheap way to release it for
    // real before this check (a nullptr free only re-parks it; only a
    // genuine reclaim or driving its bucket to the per-size depth limit
    // would truly release it, and both are more invasive than this
    // residual is worth closing here).
    evict_slow_release.release_event.wait();
    // Confirmed complete by the wait() just above (unlike at the eviction
    // sweep's own "elapsed=" print earlier, where it was deliberately
    // still pending) -- safe to read its ACTUAL device-measured duration
    // here, alongside the calibrated ESTIMATE printed there.
    const long long evict_actual_ms = actual_kernel_duration_ms(evict_slow_release.release_event);
    printf("    (evict_in_flight's release: calibrated estimate=%lld ms, actual=%lld ms)\n",
           evict_slow_release.duration_ms, evict_actual_ms);
    // llama.cpp-c6ah: same reasoning as test_bounded_eviction's
    // own actual-vs-calibrated check -- the sweep's own timing bound above
    // (sweep_elapsed < duration_ms / 3) depends on evict_slow_release
    // having actually run close to its calibrated target.
    check(evict_actual_ms >= (evict_slow_release.duration_ms * 8) / 10,
          "evict_in_flight's release kernel ACTUAL measured duration is at least 80% of its calibrated target -- "
          "otherwise this test's timing bound is being checked against a kernel that undershot calibration");
    const bool in_flight_survived = poll_for_pool_hit(cache, q, kSizeEvictInFlight, evict_in_flight);
    check(in_flight_survived,
          "the in-flight entry is still pooled and becomes a hit once its release event "
          "completes -- it was skipped by the sweep, not silently dropped");
    if (in_flight_survived) {
        cache->onednn_graph_scratch_free(evict_in_flight, nullptr);
    }

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test teardown");
}

// llama.cpp-c6ah: a pool entry whose marker kernel is still in flight when
// the pool is reclaimed must have its flag_slot RETIRED (never returned to
// the free list), not handed to a later entry. Before this was fixed,
// onednn_graph_scratch_clear_pool_locked() returned every entry's slot
// unconditionally, including one whose marker was still pending; since the
// watch queue is out-of-order, that stale marker could fire AFTER a new
// occupant's own marker already wrote its generation, overwriting the
// slot back to the OLD generation and permanently flipping an
// already-complete entry back to "not complete" -- its bytes then never
// reclaimed, cap waits on every future request of that size running to
// their 5 s timeout. This test parks an entry with a slow release,
// reclaims the pool while it is still in flight (exactly the hazard),
// parks a fresh same-size entry, confirms the fresh entry becomes complete
// at its OWN kernel's end, and then -- after finally letting the old,
// now-retired kernel finish too -- confirms the fresh entry's completion
// is unaffected by that stale write.
void test_reclaim_while_in_flight_retires_the_slot(unified_cache * cache, int device) {
    printf("Reclaim while in flight retires the slot, not the entry it later hands out:\n");

    sycl::queue & q = cache->get_queue();

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (reclaim retirement)");

    constexpr size_t kSizeReclaimRetire = 320ull * 1024 * 1024;
    static_assert(kSizeReclaimRetire > kZoneFloorBytes,
                  "must exceed the no-model ONEDNN zone floor to take the DIRECT path");
    static_assert(kSizeReclaimRetire < kCapBytes, "must fit under the cap alone -- no cap pressure needed here");

    void * ptr_old = cache->onednn_graph_scratch_alloc(kSizeReclaimRetire, 256, &q);
    check(ptr_old != nullptr, "DIRECT allocation for the old (in-flight-at-reclaim) entry succeeds");
    if (!ptr_old) {
        return;
    }
    slow_release_result old_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr_old, &old_release.release_event);

    // RED witness (llama.cpp-c6ah): capture ptr_old's own slot
    // and the free list's size BEFORE the reclaim, both via hooks-gated
    // accessors. The free list is LIFO, and on the pre-fix code ptr_old's
    // marker fires within milliseconds of the wait below -- well before
    // ptr_new's park below could reset that same slot and arm a new
    // generation -- so a check that only waits and then re-polls cannot
    // distinguish "retired" from "returned and already overwritten by the
    // time anything looked". Comparing the free list's size and ptr_new's
    // slot against these captured values does not have that blind spot:
    // both fail on the pre-fix code (which returns ptr_old's slot to the
    // free list at reclaim time) and both pass on the fix, independent of
    // timing.
    const int32_t old_entry_flag_slot = cache->onednn_graph_scratch_pool_entry_flag_slot_for_test(kSizeReclaimRetire);
    check(old_entry_flag_slot >= 0,
          "the old entry actually armed a slot -- otherwise the two witnesses below prove nothing");
    const size_t free_list_size_before_reclaim = cache->onednn_graph_scratch_flag_slot_free_list_size_for_test();

    // Reclaim the pool WHILE ptr_old's marker kernel is still in flight --
    // the exact hazard this test exists to catch. Before the fix,
    // ptr_old's slot returned to the free list here even though its
    // marker had not fired yet.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test reclaim (entry still in flight)");

    const size_t free_list_size_after_reclaim = cache->onednn_graph_scratch_flag_slot_free_list_size_for_test();
    printf("    (old entry's slot=%d; free list size: before reclaim=%zu, after reclaim=%zu)\n", old_entry_flag_slot,
           free_list_size_before_reclaim, free_list_size_after_reclaim);
    // RED witness 1/2: ptr_old's slot must be PERMANENTLY retired, never
    // returned to the free list -- the free list's size must be unchanged
    // by this reclaim (this test's pool holds no other entry, so nothing
    // else could legitimately change it either).
    check(free_list_size_after_reclaim == free_list_size_before_reclaim,
          "the reclaim does not grow the free list -- ptr_old's slot was retired, not returned");

    // q is a single IN-ORDER queue and old_release's kernel was never waited
    // on above, so without this wait, new_release's kernel (submitted next,
    // on the same queue) would queue BEHIND old_release's still-running one
    // and could not itself complete until old_release's kernel also did --
    // long enough to blow the poll deadline below for a reason that has
    // nothing to do with the retirement logic under test. The reclaim call
    // above already exercised the hazard this test checks (the slot
    // decision is made at reclaim time, while old_release's marker is still
    // in flight); waiting here only serializes the two kernels so the poll
    // below measures new_release's own completion, not queue backpressure
    // from old_release.
    old_release.release_event.wait_and_throw();

    void * ptr_new = cache->onednn_graph_scratch_alloc(kSizeReclaimRetire, 256, &q);
    check(ptr_new != nullptr, "DIRECT allocation for the new entry succeeds");
    if (!ptr_new) {
        return;
    }
    slow_release_result new_release = submit_slow_release(q);
    cache->onednn_graph_scratch_free(ptr_new, &new_release.release_event);

    const int32_t  new_entry_flag_slot = cache->onednn_graph_scratch_pool_entry_flag_slot_for_test(kSizeReclaimRetire);
    const uint32_t new_entry_flag_generation =
        cache->onednn_graph_scratch_pool_entry_flag_generation_for_test(kSizeReclaimRetire);
    printf("    (old entry's slot=%d; new entry parked: flag_slot=%d (%s), flag_generation=%u)\n", old_entry_flag_slot,
           new_entry_flag_slot,
           new_entry_flag_slot >= 0 ? "armed" : "unwatched -- falls back to the bare blocking query",
           new_entry_flag_generation);
    // RED witness 2/2: the new entry must never be handed ptr_old's own
    // (retired) slot. On the pre-fix code, with the free list LIFO and
    // this test's pool otherwise empty, ptr_old's slot -- if wrongly
    // returned -- would be exactly the next slot popped, so this comparison
    // fails on the pre-fix code and passes on the fix regardless of timing.
    check(new_entry_flag_slot != old_entry_flag_slot, "the new entry's slot is never ptr_old's own (retired) slot");

    // Poll the new entry's own flag directly (no alloc()/free() round trip
    // that would itself consume/re-park it) until it reads true -- must
    // happen at ITS OWN kernel's end.
    const auto flag_poll_start    = std::chrono::steady_clock::now();
    const auto flag_poll_deadline = flag_poll_start + std::chrono::milliseconds(2500);
    bool       new_entry_complete = false;
    while (std::chrono::steady_clock::now() < flag_poll_deadline) {
        if (cache->onednn_graph_scratch_pool_entry_flag_true_for_test(kSizeReclaimRetire)) {
            new_entry_complete = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check(new_entry_complete, "the new entry's own completion flag reads true once its own release event completes");

    // old_release's kernel already finished (waited above, to keep it off
    // new_release's in-order queue); this re-confirms that. If the fix is
    // wrong -- if ptr_old's slot had been handed to ptr_new -- old_release's
    // marker write, landing in that same (wrongly shared) slot, would have
    // already flipped ptr_new's completion status back to false by now.
    old_release.release_event.wait_and_throw();
    check(cache->onednn_graph_scratch_pool_entry_flag_true_for_test(kSizeReclaimRetire),
          "the new entry STAYS complete after the old (retired-slot) kernel finally finishes -- proves the "
          "retired slot was never handed to the new entry");

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test teardown (reclaim retirement)");
}

// llama.cpp-c6ah: standalone t_flag/t_wait diagnostic, moved out of
// test_bounded_eviction() (its original home) after that placement was
// found to consume the ~1500 ms delay that test's own cap-wait bound
// depends on measuring -- the pre-wait eviction sweep observed ptr1
// already complete (this diagnostic's own poll+wait had already let the
// kernel finish before the timed request even started) and evicted it in
// 3-4 ms with zero cap waits, making that bound vacuous. The mechanism
// itself is confirmed understood, both cards: t_flag lands just after the
// kernel's own actual duration (B50: t_flag=1533 ms vs actual=1515 ms;
// B70: 1527 ms vs 1516 ms) and t_wait taken right after t_flag reads
// ~0 ms, so this function keeps the measurement available as a standing,
// assertion-free diagnostic (beyond the one bound every slow-release test
// in this file already applies to its own free() call) rather than
// dropping it entirely. Deliberately called LAST in main(), after every
// other test's own teardown reclaim, so nothing it does here can perturb
// another test's own timing the way its old placement did.
void test_flag_timing_diagnostic(unified_cache * cache, int device) {
    printf("Flag timing diagnostic:\n");

    sycl::queue & q = cache->get_queue();

    // Leading reclaim -- order-independence insurance, same rationale as
    // every other test in this file that reuses a size (here, kSizeA)
    // another test also uses.
    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test setup (flag timing diagnostic)");

    void * ptr = cache->onednn_graph_scratch_alloc(kSizeA, 256, &q);
    check(ptr != nullptr, "DIRECT allocation for the flag timing diagnostic succeeds");
    if (!ptr) {
        return;
    }

    slow_release_result slow_release    = submit_slow_release(q);
    // Same bound every other slow-release park in this file applies (see
    // test_bounded_eviction()'s own comment for the full history) -- kept
    // here too since this is the last place in the binary that parks a
    // slow-release event, so it is also the last chance to catch a
    // regression back to a blocking arm.
    const auto          free_call_start = std::chrono::steady_clock::now();
    cache->onednn_graph_scratch_free(ptr, &slow_release.release_event);
    const long long free_call_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - free_call_start)
            .count();
    printf("    (onednn_graph_scratch_free() itself returned in %lld ms, release duration (calibrated)=%lld ms)\n",
           free_call_ms, slow_release.duration_ms);
    check(free_call_ms < slow_release.duration_ms / 3,
          "onednn_graph_scratch_free() itself returned in well under a third of the release delay -- arming the "
          "completion flag did not block the park call");

    const auto flag_poll_start    = std::chrono::steady_clock::now();
    const auto flag_poll_deadline = flag_poll_start + std::chrono::milliseconds(2500);
    long long  t_flag_ms          = -1;
    while (std::chrono::steady_clock::now() < flag_poll_deadline) {
        if (cache->onednn_graph_scratch_pool_entry_flag_true_for_test(kSizeA)) {
            t_flag_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                              flag_poll_start)
                            .count();
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (t_flag_ms >= 0) {
        const auto wait_start = std::chrono::steady_clock::now();
        slow_release.release_event.wait_and_throw();
        const long long t_wait_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wait_start)
                .count();
        printf(
            "    (flag poll: t_flag=%lld ms, t_wait=%lld ms, actual kernel duration=%lld ms, kernel "
            "target=%lld ms)\n",
            t_flag_ms, t_wait_ms, actual_kernel_duration_ms(slow_release.release_event), slow_release.duration_ms);
    } else {
        printf("    (flag poll: never read true within 2.5 s, kernel target=%lld ms)\n", slow_release.duration_ms);
    }

    unified_cache_reclaim_onednn_graph_scratch_pool(device, "test teardown (flag timing diagnostic)");
}

}  // namespace

int main(int, char ** argv) {
    // Match the sibling SYCL gates: pin the validation card so a bare ctest
    // run cannot perturb a measurement on the other GPU. ctest also sets this
    // via ENVIRONMENT; this is only a fallback for bare invocation.
    sycl_test_selector_fallback(argv, "level_zero:1");

    // Set BEFORE any onednn_graph_scratch_* call in this process:
    // onednn_graph_scratch_direct_cap_bytes() memoizes this env var in a
    // function-local static on its first read, so setting it later in the
    // process would be a silent no-op; nothing before this point in main()
    // allocates, so that first read cannot have happened yet. kCapMiB (700)
    // is the single source of truth for this value -- see its own comment
    // at the top of this file for why 700, not the historical 350: an
    // earlier version of this test tried to reach the same goal (two
    // same-size DIRECT allocations outstanding at once) by shrinking the
    // ONEDNN zone via GGML_SYCL_ONEDNN_GRAPH_ZONE_MB instead of raising this
    // cap, and that override measured INERT on hardware for this no-model
    // arena (the process still reserved the natural ~256 MB zone
    // regardless), silently zone-serving what was meant to be a DIRECT-path
    // test and voiding every assertion downstream of it. Built as a string
    // (not a string literal) so it can never drift from kCapMiB.
    const std::string cap_mb_str = std::to_string(kCapMiB);
    setenv("GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB", cap_mb_str.c_str(), 1);
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

    // llama.cpp-c6ah: calibrate the slow-release kernel's
    // scaled iteration count exactly once, HERE, before any test has a
    // chance to leave an unwaited kernel in flight on the cache's queue --
    // see ensure_slow_release_calibrated()'s own comment for why calling it
    // explicitly from main() (rather than lazily from the first
    // submit_slow_release() call) is load-bearing, not stylistic.
    ensure_slow_release_calibrated(cache->get_queue());

    test_pool_reuse(cache);
    test_bounded_eviction(cache, device);
    test_loud_failure(cache);
    test_pending_event_reclaim_does_not_destruct_in_flight(cache, device);
    test_oversized_request_skips_wait_loop(cache, device);
    test_in_flight_entry_is_skipped_not_waited(cache, device);
    test_reclaim_while_in_flight_retires_the_slot(cache, device);

    // llama.cpp-c6ah: LAST, deliberately -- see this
    // function's own comment for why.
    test_flag_timing_diagnostic(cache, device);

    ggml_backend_free(backend);

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
