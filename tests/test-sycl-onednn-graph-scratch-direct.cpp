// Gate for llama.cpp-0oxf: the oneDNN Graph-scratch allocator's DIRECT
// (non-arena) fallback must BOUND its outstanding bytes and FAIL LOUDLY
// rather than silently handing oneDNN a null scratch pointer.
//
// BACKGROUND. unified_cache::onednn_graph_scratch_alloc() serves oneDNN
// Graph's SYCL allocator callback. When a request does not fit the planned
// ONEDNN zone (measured: a 144 MB Graph-scratch request at n_kv=8192 on
// Mistral 7B Q4_0 -- see the ticket), it falls back to a real, individually
// mem_handle-owned unified_alloc() (the "DIRECT" path). That allocation's
// physical release is deferred to the shared background drain worker
// (mem-handle.cpp retained_handle_drain_loop). Under host CPU contention that
// worker lags, DIRECT buffers pile up, unified_alloc() eventually fails, and
// the pre-fix code returned nullptr SILENTLY -- oneDNN then executed the SDPA
// kernel against an invalid scratch pointer: a GPU page fault ("Engine memory
// CAT error"), an engine reset, then a segfault.
//
// The fix has two parts, and this test asserts both:
//
//   (a) BOUNDED WAIT: before taking the DIRECT path, the allocator now caps
//       outstanding DIRECT bytes (onednn_graph_scratch_direct_cap_bytes()) and
//       waits (bounded, polling the shared drain worker) for headroom rather
//       than allocating past the cap. This test drives outstanding DIRECT
//       bytes over an artificially small cap (via
//       GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB) with a release event that
//       completes only after a real, measurable delay, and asserts the next
//       allocation (1) actually blocks for roughly that delay rather than
//       returning immediately, and (2) counts the wait
//       (onednn_graph_scratch_direct_wait_count()).
//
//   (b) LOUD FAILURE: an allocation that is genuinely exhausted (both the
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
// RED-FIRST NOTE. Both properties are RED against the pre-0oxf code: (a) has
// no cap or wait at all (every DIRECT allocation proceeds immediately, so
// nothing here would ever measure a delay or increment a wait counter), and
// (b) had no abort hook to suppress -- the pre-fix function simply returns
// nullptr with `req.suppress_failure_log = true`, so the log capture in this
// test would find nothing and the "abort triggered" latch would not exist.
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

// One Graph-scratch request this size never fits any realistic ONEDNN zone
// (the zone defaults to 256 MB and this repository's own comments record
// requests up to 144 MB as the largest measured), so it deterministically
// takes the DIRECT path regardless of what a live model's planning left the
// zone sized to.
constexpr size_t kDirectRequestBytes = 300ull * 1024 * 1024;

// A release event that only completes after this many milliseconds -- long
// enough to be unambiguously distinguishable from "the wait didn't really
// wait", short enough to keep the gate fast.
constexpr int kSlowReleaseMs = 1500;

// --- (a) bounded wait under the DIRECT-path cap -----------------------------
void test_bounded_wait(unified_cache * cache) {
    printf("DIRECT path bounded wait:\n");

    sycl::queue & q = cache->get_queue();

    void * ptr1 = cache->onednn_graph_scratch_alloc(kDirectRequestBytes, 256, &q);
    check(ptr1 != nullptr, "DIRECT allocation for the bounded-wait setup succeeds");
    if (!ptr1) {
        return;
    }

    // A host_task event that only completes after kSlowReleaseMs -- this is
    // the "slow-to-complete event" the fix spec calls for: it keeps ptr1's
    // bytes charged against the outstanding-DIRECT counter (see
    // onednn_graph_scratch_free()'s DIRECT branch: the physical release is
    // deferred to the shared drain worker, gated on exactly this event) long
    // enough that the SECOND allocation below is guaranteed to observe the
    // cap and enter the wait loop, rather than racing a fast release.
    sycl::event slow_release = q.submit([&](sycl::handler & h) {
        h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(kSlowReleaseMs)); });
    });
    cache->onednn_graph_scratch_free(ptr1, &slow_release);

    const size_t wait_count_before = cache->onednn_graph_scratch_direct_wait_count();

    const auto start = std::chrono::steady_clock::now();
    void *     ptr2  = cache->onednn_graph_scratch_alloc(kDirectRequestBytes, 256, &q);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

    check(ptr2 != nullptr, "second DIRECT allocation eventually succeeds");
    check(cache->onednn_graph_scratch_direct_wait_count() > wait_count_before,
          "onednn_graph_scratch_direct_wait_count() increased -- the allocator actually waited");
    // Half of kSlowReleaseMs, not the whole thing: the poll interval inside
    // the wait loop (kOnednnGraphDirectWaitPollTimeoutMs, 200 ms in
    // unified-cache.cpp) means the observed wait can undershoot the event's
    // own completion time by up to one poll, and this only needs to
    // distinguish "actually waited" from "returned immediately".
    check(elapsed.count() >= kSlowReleaseMs / 2,
          "the second allocation took a substantial fraction of the release delay, not ~0 ms");
    printf("    (elapsed=%lld ms, cap wait count %zu -> %zu)\n", static_cast<long long>(elapsed.count()),
           wait_count_before, cache->onednn_graph_scratch_direct_wait_count());

    if (ptr2) {
        cache->onednn_graph_scratch_free(ptr2, nullptr);
    }
}

// --- (b) exhausted DIRECT allocation fails loudly, never silently ----------
void test_loud_failure(unified_cache * cache) {
    printf("DIRECT path exhaustion fails loudly:\n");

    sycl::queue & q = cache->get_queue();

    g_captured_log.clear();
    ggml_log_set(capture_log, nullptr);

    ggml_sycl_test_onednn_graph_scratch_suppress_abort(true);
    // 2: force BOTH the initial attempt and the drain-and-retry to fail,
    // without touching the GPU -- this is what drives the allocator all the
    // way to its "give up" decision deterministically.
    ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail(2);

    void * ptr = cache->onednn_graph_scratch_alloc(kDirectRequestBytes, 256, &q);

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
    // silent no-op. 350 MB sits strictly between one and two
    // kDirectRequestBytes (300 MB) allocations, so the second concurrent one
    // is exactly what should trip the cap.
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

    test_bounded_wait(cache);
    test_loud_failure(cache);

    ggml_backend_free(backend);

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
