// Regression test for llama.cpp-oze0: retain_handles_until_event() must decide
// handle-lifetime routing from THIS thread's graph-recording state.
//
// The defect, in one sentence: the guard was process-global while the sink it
// selects is thread_local, so when the two disagree every handle fell into a
// global limbo nobody drains.
//
//   * ggml_sycl_graph_recording_active() is
//       g_ggml_sycl_graph_recording                       (thread_local bool)
//     || g_ggml_sycl_graph_recording_depth > 0            (process-global atomic)
//   * g_graph_retained_handle_sink, the thing that guard picks, is thread_local
//     and is installed only by the context that is actually recording.
//
// So a thread that is NOT recording, while any other thread IS, took the
// graph-lifetime branch and found no sink -- and fell through to the
// process-global graph_unwaitable list.  drain_retained_handles() deliberately
// does not wait on that list, and only release_graph_retained_handles(), called
// when some *other* context invalidates its graph, ever clears it.
//
// Two consequences, both real:
//
//   1. LOUD.  Transient scratch stays live in the shared zones for as long as
//      any other context keeps recording.  The next graph boundary's zone reset
//      is then correctly refused ("refusing zone_reset of VRAM zone SCRATCH
//      with N live registered allocations") and graph launch aborts.  Observed
//      in test-thread-safety on both of get_rows' staging cohorts:
//      get_rows:seq_device (VRAM SCRATCH, getrows.cpp) and
//      get_rows_indices_small_host (host STAGING, getrows.cpp) -- two escape
//      axes, one root cause, because both release through this one function.
//   2. SILENT, and worse.  release_graph_retained_handles() frees those handles
//      from whichever context next invalidates a graph, which is not the one
//      whose submitted work still depends on them.
//
// Property 2 below is the one that is easy to lose.  Narrowing the guard all the
// way to "never retain for graph lifetime" also makes property 1 pass, and would
// reintroduce the bug graph retention exists for: events produced by recorded
// commands are not waitable, so a recording thread's handles genuinely must be
// held for the executable graph's lifetime rather than waited on.
//
// CPU-only by construction: the handles are DIRECT views over stack buffers
// (non-owning -- their destructor frees nothing) and the events are default-
// constructed, i.e. already complete.  Nothing here touches a device, a queue or
// the unified cache, so it runs anywhere, including a machine with no GPU.
//
// Usage:
//   ./build/bin/test-sycl-graph-retention-scope                # all three
//   ./build/bin/test-sycl-graph-retention-scope other-thread
//   ./build/bin/test-sycl-graph-retention-scope own-thread
//   ./build/bin/test-sycl-graph-retention-scope own-thread-sink

#include "common.hpp"
#include "mem-handle.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// A non-owning view over a stack buffer: valid() is true, so
// retain_handles_until_event() treats it like any other retained lease, but
// releasing it frees nothing and needs no allocator.
ggml_sycl::mem_handle make_direct_handle(void * buf) {
    return ggml_sycl::mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false);
}

void retain_one(void * buf) {
    std::vector<ggml_sycl::mem_handle> handles;
    handles.push_back(make_direct_handle(buf));
    ggml_sycl::retain_handles_until_event(std::move(handles), sycl::event{});
}

// Hold the process-global recording depth up for as long as the scope lives,
// exactly as another context's in-flight graph capture would.
struct other_thread_recording {
    std::thread             worker;
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    recording = false;
    bool                    stop      = false;

    other_thread_recording() {
        worker = std::thread([this]() {
            // The depth counter is what a recording context bumps; the
            // thread_local flag stays false on every OTHER thread, which is the
            // whole point of the test.
            g_ggml_sycl_graph_recording_depth.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lock(mutex);
                recording = true;
            }
            cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return stop; });
            }
            g_ggml_sycl_graph_recording_depth.fetch_sub(1, std::memory_order_acq_rel);
        });

        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return recording; });
    }

    ~other_thread_recording() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        worker.join();
    }
};

// Property 1: another context's graph capture must not capture THIS thread's
// handle release.  A non-recording thread's event is a real, waitable event, so
// the event-bound path releases it naturally at the next drain.
int test_other_thread_recording_does_not_capture_our_handles() {
    ggml_sycl::drain_retained_handles(true);
    ggml_sycl::release_graph_retained_handles();
    const size_t before = ggml_sycl::graph_retained_handle_count();

    alignas(64) std::uint8_t buf[64] = {};
    {
        other_thread_recording other;
        if (!ggml_sycl_graph_recording_active()) {
            std::fprintf(stderr,
                         "FAIL: test setup is inert -- the global recording depth did not make\n"
                         "      ggml_sycl_graph_recording_active() true, so nothing was exercised\n");
            return 1;
        }
        retain_one(buf);

        const size_t after = ggml_sycl::graph_retained_handle_count();
        if (after != before) {
            std::fprintf(stderr,
                         "FAIL: a non-recording thread's handle was parked for command-graph\n"
                         "      lifetime because ANOTHER thread was recording: count %zu -> %zu.\n"
                         "      drain_retained_handles() does not wait on that list, so the\n"
                         "      allocation stays live in its zone and the next graph boundary's\n"
                         "      zone reset is refused.\n",
                         before, after);
            return 1;
        }
    }

    // And it must actually be released, not merely routed elsewhere.
    ggml_sycl::drain_retained_handles(true);
    std::puts("PASS: another thread's recording does not capture this thread's handle release");
    return 0;
}

// Property 2 (positive control): a thread that IS recording, with no sink
// installed, must still retain for graph lifetime.  Its event is not waitable.
int test_own_recording_without_sink_retains_for_graph_lifetime() {
    ggml_sycl::drain_retained_handles(true);
    ggml_sycl::release_graph_retained_handles();
    const size_t before = ggml_sycl::graph_retained_handle_count();

    alignas(64) std::uint8_t buf[64] = {};
    g_ggml_sycl_graph_recording      = true;
    retain_one(buf);
    g_ggml_sycl_graph_recording = false;

    const size_t after = ggml_sycl::graph_retained_handle_count();
    if (after != before + 1) {
        std::fprintf(stderr,
                     "FAIL: a recording thread's handle was NOT retained for graph lifetime:\n"
                     "      count %zu -> %zu (expected %zu). Events produced by recorded\n"
                     "      commands are not waitable, so waiting on one either hangs or throws\n"
                     "      and the lease is dropped while the graph still references it.\n",
                     before, after, before + 1);
        return 1;
    }

    ggml_sycl::release_graph_retained_handles();
    if (ggml_sycl::graph_retained_handle_count() != 0) {
        std::fprintf(stderr, "FAIL: release_graph_retained_handles() left %zu handles parked\n",
                     ggml_sycl::graph_retained_handle_count());
        return 1;
    }
    std::puts("PASS: a recording thread's handle is retained for command-graph lifetime");
    return 0;
}

// Property 3: with a sink installed the handle belongs to that context's graph,
// not to the process-global fallback list.
int test_own_recording_with_sink_routes_to_the_sink() {
    ggml_sycl::drain_retained_handles(true);
    ggml_sycl::release_graph_retained_handles();
    const size_t before = ggml_sycl::graph_retained_handle_count();

    alignas(64) std::uint8_t           buf[64] = {};
    std::vector<ggml_sycl::mem_handle> sink;
    ggml_sycl::set_graph_retained_handle_sink(&sink);
    retain_one(buf);
    ggml_sycl::set_graph_retained_handle_sink(nullptr);

    if (sink.size() != 1) {
        std::fprintf(stderr,
                     "FAIL: the installed graph sink received %zu handles (expected 1);\n"
                     "      a context's own graph must own its retained leases\n",
                     sink.size());
        return 1;
    }
    if (ggml_sycl::graph_retained_handle_count() != before) {
        std::fprintf(stderr,
                     "FAIL: a handle leaked into the process-global fallback list while a\n"
                     "      per-context sink was installed: count %zu -> %zu\n",
                     before, ggml_sycl::graph_retained_handle_count());
        return 1;
    }
    std::puts("PASS: an installed graph sink owns its context's retained leases");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * only     = argc > 1 ? argv[1] : nullptr;
    auto         selected = [only](const char * name) {
        return !only || std::strcmp(only, name) == 0;
    };

    if (only && !selected("other-thread") && !selected("own-thread") && !selected("own-thread-sink")) {
        std::fprintf(stderr, "unknown test '%s' (other-thread | own-thread | own-thread-sink)\n", only);
        return 2;
    }

    if (selected("other-thread")) {
        if (int rc = test_other_thread_recording_does_not_capture_our_handles()) {
            return rc;
        }
    }
    if (selected("own-thread")) {
        if (int rc = test_own_recording_without_sink_retains_for_graph_lifetime()) {
            return rc;
        }
    }
    if (selected("own-thread-sink")) {
        if (int rc = test_own_recording_with_sink_routes_to_the_sink()) {
            return rc;
        }
    }
    std::puts("PASS: graph-retention routing is scoped to the recording thread");
    return 0;
}
