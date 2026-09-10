// Decisive-experiment probe for llama.cpp-c6ah's own premise: does the bare
// `evt.get_info<sycl::info::event::command_execution_status>()` query
// actually BLOCK (rather than poll) on a profiling-enabled queue, the way
// unified-cache.hpp's own get_dma_queue() comment claims?
//
// WHY THIS EXISTS. llama.cpp-c6ah's fix replaced a direct call to that query
// with a host-visible completion flag, on the premise that the direct call
// blocks. The fix's own GPU test (test-sycl-onednn-graph-scratch-direct.cpp)
// measured its RED-arm positive control (force_blocking_pool_check(true),
// which re-enables the direct query) returning in ~6 ms with a fresh
// allocation, not the ~1500 ms blocking wait the premise predicts -- i.e.
// the RED arm did not reproduce blocking on this driver/queue combination,
// for a release event produced by a host_task. This file isolates the
// question to the smallest possible reproduction, independent of the pool
// allocator, the completion flag, or any of this ticket's own code: submit
// something onto a queue, then issue exactly ONE bare status query
// immediately afterward (never wait() first), and time that ONE call.
//
// TWO ORTHOGONAL AXES, both varied, because either alone would leave the
// other as an unexamined confound:
//
//   - QUEUE PROPERTIES: in_order + enable_profiling (what every real
//     backend stream in this codebase uses, via default_queue_properties()
//     in common.hpp) vs. in_order alone (what dma_queue_ uses, and what
//     unified-cache.hpp's own comment claims is the one queue where the
//     query is actually non-blocking).
//
//   - EVENT SOURCE: a host_task (what test-sycl-onednn-graph-scratch-direct.cpp's
//     own "slow release" events were produced by, for convenience, when its
//     RED arm measured that ~6 ms result) vs. a real DEVICE KERNEL (a
//     single-work-item spin loop -- the shape oneDNN's free callback
//     actually supplies release events as in production). The hypothesis
//     this axis tested: the hpp comment this ticket's fix is built on
//     documents driver behaviour that might be specific to device-kernel
//     events, and a host_task might not carry the same profiling counters
//     at all, which would make it query-cheap regardless of queue
//     properties -- a distinct explanation for the RED arm's ~6 ms result
//     that would have nothing to do with whether the ticket's premise
//     holds for the case that actually matters in production. See the
//     MEASURED section below for the answer this probe found.
//
// Both axes together give four combinations, each measured once. This test
// asserts only that no query hangs and every event completes -- it prints
// the four timings (plus each kernel's own total duration, measured
// separately via wait(), so a human can see the actual kernel duration
// GGML_TEST_SPIN_ITERATIONS produced on the hardware it ran on, and compare
// the query time against THAT measured duration rather than against a
// fixed target) and leaves the INTERPRETATION to whoever reads the numbers.
// Two outcomes were possible in principle:
//
//   - If all four queries return in a few tens of ms, event_complete()
//     already polls on this driver regardless of queue properties or event
//     source, and this ticket's premise (that it blocks) does not hold on
//     this hardware/driver combination -- the fix may still be correct
//     defensively, but it is not closing an observed bug here.
//   - If the DEVICE-KERNEL query on the PROFILING queue takes close to the
//     kernel's own full duration (i.e. the query effectively absorbed the
//     wait, and the subsequent wait() returns almost immediately), the
//     premise holds, but specifically for device-kernel events -- and the
//     ticket's own GPU test should be using a kernel-produced release
//     event, not a host_task one, to actually exercise it.
//
// MEASURED (GGML_TEST_SPIN_ITERATIONS=2000000, both discrete cards this
// fork validates against): the SECOND outcome. host_task query times were
// ~0 ms on both queue kinds on both cards; the device-kernel query on the
// NON-profiling queue was also ~0 ms on both cards. The device-kernel query
// on the PROFILING queue took nearly the kernel's own full duration on
// both cards -- B50: query=118 ms vs. kernel duration=122 ms; B70:
// query=110 ms vs. kernel duration=114 ms. So the bare
// command_execution_status query BLOCKS until completion specifically for
// a device-kernel-produced event on a profiling-enabled queue, exactly as
// unified-cache.hpp's own comment states, and exactly the shape a real
// oneDNN Graph-scratch release event actually is in production. This is
// why test-sycl-onednn-graph-scratch-direct.cpp's own "slow release" events
// are device-kernel-produced (via a helper matching this file's own kernel
// shape), not host_task-produced -- see that file's submit_slow_release()
// for the full history of why a host_task-produced release event there
// measured a ~6 ms RED arm instead of reproducing this block at all.
//
// RELATED, DISTINCT FACT (llama.cpp-c6ah), not measured by this file: the
// pool's original fix armed a host_task, on a SEPARATE
// queue, depending on a device-kernel release event via depends_on() --
// not the bare-query pattern this file measures at all. A follow-up
// measurement (both cards, 2026-09-09) found that SUBMITTING such a
// host_task blocks the SUBMITTING thread until the dependency completes,
// whenever the dependency comes from another queue. That fact does not
// change anything measured here (this file never submits a host_task with
// a cross-queue dependency), but it explains why an earlier version of
// this ticket's own source comments (unified-cache.cpp's event_complete()
// and onednn_graph_scratch_pool_entry_release_complete()) briefly claimed
// the bare query above "lies" once a watcher exists -- that was a
// misdiagnosis of the host_task-submission-blocks effect, not a property
// of the query this file actually isolates. See
// onednn_graph_scratch_pool_entry::flag_slot's comment in unified-cache.hpp
// for the full incident history and the device-marker-kernel design that
// replaced the host_task watcher.
//
// Exits 77 (ctest SKIP_RETURN_CODE) when no SYCL GPU device is present.

#include "sycl-selector-fallback.hpp"
#include "sycl-spin-kernel.hpp"
#include "test-skip.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <thread>

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

using clock_type = std::chrono::steady_clock;

long long ms_since(clock_type::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now() - start).count();
}

// The ~1000 ms operation whose release event's completion query is what
// this whole file measures.
constexpr int kHostTaskMs = 1000;

sycl::event submit_host_task(sycl::queue & q) {
    return q.submit([&](sycl::handler & h) {
        h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(kHostTaskMs)); });
    });
}

// submit_spin_kernel() (matching the property this file's own header
// comment describes, "a single work-item loop") now lives in the shared
// tests/sycl-spin-kernel.hpp (llama.cpp-me60 F7) -- included above.

// The one measurement this whole file exists to take: how long does a
// SINGLE bare status query take, issued immediately after submission with
// no wait() first. Never call this more than once per event -- a caller
// that queried twice could not tell "the first call blocked" apart from
// "polling happened to need two tries", and this file's whole point is to
// measure ONE call's cost.
long long measure_query_ms(sycl::event & evt) {
    const auto start  = clock_type::now();
    const auto status = evt.get_info<sycl::info::event::command_execution_status>();
    (void) status;  // the STATUS is not what this file is testing -- how LONG the call itself took is.
    return ms_since(start);
}

}  // namespace

int main(int, char ** argv) {
    // Same pinning convention as every sibling SYCL gate in this directory
    // -- see sycl-selector-fallback.hpp's own comment for why a plain
    // setenv() in main() would be too late. MUST be the first statement of
    // main(): the fallback can re-exec the whole process, and anything run
    // before it would run twice (once before the re-exec, once after).
    sycl_test_selector_fallback(argv, "level_zero:1");

    // Line-buffer stdout (regardless of whether it lands on a terminal or,
    // as under ctest, a redirected file) so a run killed mid-measurement
    // (e.g. an unexpectedly long default iteration count, or a hung device)
    // still shows every line printed before the kill, not just whatever
    // happened to be in a full stdio buffer. Measured need: the very first
    // default iteration count this file shipped with (2,000,000,000) did
    // not finish within a 120 s observation window on real hardware, and
    // its output was fully buffered (the default for a non-terminal
    // stdout), so nothing printed at all before that run was killed --
    // this fixes both the count (below) and the buffering.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // GGML_TEST_SPIN_ITERATIONS: the spin kernel's iteration count is a
    // hardware-speed-dependent guess (this file was authored without GPU
    // access to calibrate it directly -- see CLAUDE.md's division of labour
    // between subagents and the lead session for GPU work). Overridable
    // without a rebuild so the actual kernel duration this run prints (see
    // below) can be used to correct it on the next run rather than needing
    // a source edit and a full SYCL rebuild for a constant tune. 2,000,000
    // (not the original 2,000,000,000, three orders of magnitude larger):
    // measured on real hardware (both discrete cards this fork validates
    // against) at ~110-122 ms for 2,000,000 iterations -- comfortably close
    // to kHostTaskMs (1000 ms) once scaled by roughly 8-9x, and the ORIGINAL
    // 2,000,000,000 default measured as "did not finish in 120 s", three
    // orders of magnitude too slow rather than too fast.
    long long iterations = 2'000'000LL;
    if (const char * env = std::getenv("GGML_TEST_SPIN_ITERATIONS")) {
        if (env[0] != '\0') {
            iterations = std::atoll(env);
        }
    }

    // Device/context/queue construction and every measurement below share
    // ONE try block, matching this directory's own established pattern for
    // a device probe (test-sycl-onednn-int8-grouped-probe.cpp) -- a
    // sycl::exception anywhere in this setup, not only "no device found",
    // is reported as a skip rather than a hard failure, since this probe's
    // job is to report timings when it CAN run, not to be a strict
    // regression gate.
    try {
        sycl::device  dev{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        // Matches default_queue_properties() in common.hpp -- every real
        // backend stream in this codebase is in_order + enable_profiling.
        sycl::queue   q_profiling(
            ctx, dev,
            sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} });
        // Matches dma_queue_'s own construction in unified-cache.cpp --
        // in_order, deliberately WITHOUT enable_profiling.
        sycl::queue q_non_profiling(ctx, dev, sycl::property_list{ sycl::property::queue::in_order{} });

        printf("device=%s\n", dev.get_info<sycl::info::device::name>().c_str());

        // One device-global cell, shared by both queues (they share `ctx`,
        // so a pointer allocated against it is valid on either queue).
        int * cell = sycl::malloc_device<int>(1, q_profiling);
        check(cell != nullptr, "device cell allocation succeeds");
        if (!cell) {
            return 1;
        }
        q_profiling.memset(cell, 0, sizeof(int)).wait_and_throw();

        printf(
            "Bare command_execution_status query timing (single call, issued immediately\n"
            "after submission, no wait() first):\n");

        // (1) host_task on the PROFILING queue -- the shape
        // test-sycl-onednn-graph-scratch-direct.cpp's own "slow release"
        // events used to be produced by, before this probe's own MEASURED
        // result (above) showed that was not the shape that actually
        // matters and that file switched to a device-kernel release event.
        {
            sycl::event     evt  = submit_host_task(q_profiling);
            const long long q_ms = measure_query_ms(evt);
            evt.wait_and_throw();
            // llama.cpp-c6ah: assert the relation this file's own
            // MEASURED result found, not merely "did not hang" -- a driver
            // change that made this combination block would leave a bare
            // check(true, ...) green. kHostTaskMs (not kernel_ms -- no
            // kernel runs in this combination) is the relevant duration to
            // compare against.
            check(q_ms < kHostTaskMs / 2,
                  "host_task / profiling queue: query returns well under the host_task's "
                  "own sleep duration, not close to it");
            printf("  host_task,     profiling queue: query=%lld ms\n", q_ms);
        }

        // (2) host_task on the NON-PROFILING queue -- the one queue
        // unified-cache.hpp's own comment says the query is actually
        // non-blocking on.
        {
            sycl::event     evt  = submit_host_task(q_non_profiling);
            const long long q_ms = measure_query_ms(evt);
            evt.wait_and_throw();
            // llama.cpp-c6ah: same reasoning as the profiling-queue
            // host_task case above.
            check(q_ms < kHostTaskMs / 2,
                  "host_task / non-profiling queue: query returns well under the "
                  "host_task's own sleep duration, not close to it");
            printf("  host_task, non-profiling queue: query=%lld ms\n", q_ms);
        }

        // (3) device kernel on the PROFILING queue -- the shape a REAL
        // oneDNN Graph-scratch release event actually is in production
        // (oneDNN's free callback supplies a device-kernel event, not a
        // host_task one). This is the combination the ticket's premise
        // most needs to hold for.
        {
            const auto      k_start = clock_type::now();
            sycl::event     evt     = submit_spin_kernel(q_profiling, cell, iterations);
            const long long q_ms    = measure_query_ms(evt);
            evt.wait_and_throw();
            const long long kernel_ms = ms_since(k_start);
            // llama.cpp-c6ah: this is the combination the
            // ticket's premise most needs to hold for (see the comment
            // above) -- assert it against THIS run's own measured
            // kernel_ms, matching the printf below's own stated
            // interpretation, rather than leaving a bare check(true, ...)
            // that a driver change making this non-blocking would leave
            // green.
            check(q_ms >= kernel_ms / 2,
                  "device kernel / profiling queue: query takes at least half the "
                  "kernel's own measured duration, i.e. it blocks rather than polls");
            // The default GGML_TEST_SPIN_ITERATIONS (2,000,000) targets
            // roughly 120 ms on the hardware this fork validates against
            // (measured range: 110-122 ms across both discrete cards) --
            // NOT kHostTaskMs (1000 ms), which only bounds the unrelated
            // host_task combinations above. Compare query= against THIS
            // line's own measured "kernel total duration", not a fixed
            // target: what this file's interpretation actually needs is
            // "did the query take close to the kernel's own duration",
            // which holds regardless of what that duration numerically is.
            printf(
                "  kernel,        profiling queue: query=%lld ms  (kernel total duration=%lld ms, "
                "iterations=%lld -- compare query= against this run's OWN measured kernel duration, not a "
                "fixed target)\n",
                q_ms, kernel_ms, iterations);
        }

        // (4) device kernel on the NON-PROFILING queue.
        {
            const auto      k_start = clock_type::now();
            sycl::event     evt     = submit_spin_kernel(q_non_profiling, cell, iterations);
            const long long q_ms    = measure_query_ms(evt);
            evt.wait_and_throw();
            const long long kernel_ms = ms_since(k_start);
            // llama.cpp-c6ah: the queue property the ticket's
            // fix does NOT depend on, but this file measures anyway (see
            // the file header) -- assert against this run's own kernel_ms.
            check(q_ms < kernel_ms / 2,
                  "device kernel / non-profiling queue: query returns well under the "
                  "kernel's own measured duration, not close to it");
            printf(
                "  kernel,    non-profiling queue: query=%lld ms  (kernel total duration=%lld ms, "
                "iterations=%lld)\n",
                q_ms, kernel_ms, iterations);
        }

        sycl::free(cell, ctx);
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: device/queue setup or a probe operation failed: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
