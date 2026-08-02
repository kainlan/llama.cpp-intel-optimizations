// Regression test for llama.cpp-f9tg: the scoped transient allocators must decide their
// allocation SHAPE from THIS thread's graph-recording state.
//
// The defect, in one sentence: a process-global recording predicate drove a per-thread
// allocation decision, so one context's transient was labelled and routed according to an
// unrelated context's phase.
//
//   * ggml_sycl_graph_recording_active() is
//       g_ggml_sycl_graph_recording                       (thread_local bool)
//     || g_ggml_sycl_graph_recording_depth > 0            (process-global atomic)
//   * three scoped_unified_*_temp templates in ggml-sycl.cpp took `graph_lifetime` from it and
//     used it to pick alloc_role, runtime_category, whether prefer_vram_zone = SCRATCH is set
//     at all, and (host form) use_pinned_pool / require_host_usm_base.
//
// Unlike its sibling llama.cpp-oze0, this is NOT a correctness bug. The wide predicate is a
// superset of the narrow one, so it only ever OVER-applies graph treatment -- it fails safe,
// and therefore announces itself nowhere. What it corrupts is attribution: llama.cpp-iiff
// Phase 0's inventory reads role / category / cohort / zone directly, and it is the
// authoritative work-list every Phase 1 cohort ticket is keyed on. A capture taken while any
// context was recording mislabelled COMPUTE/CONTROL allocations as GRAPH. The host form also
// paid for a different allocator path -- a standalone USM base instead of a pinned-pool slice
// -- to satisfy a graph the calling thread had no part in.
//
// WHAT THIS TEST COVERS, AND WHAT IT DOES NOT.
//
// It exercises ggml_sycl_transient_device_intent() / ggml_sycl_transient_host_pinned_intent()
// (common.hpp), which now hold the entire decision -- predicate included -- and which are the
// only copy of that mapping. It does NOT drive the three templates themselves: those are
// internal to ggml-sycl.cpp, and their allocation is not reachable without a device
// (unified_alloc -> get_unified_cache_for_device -> the SYCL device manager, which aborts with
// "No SYCL devices available"; verified under ONEAPI_DEVICE_SELECTOR=level_zero:99). That each
// template calls the matching builder and computes nothing itself is a one-line property of
// each site, visible in the diff and greppable; that the builders decide correctly is what is
// asserted here.
//
// Properties 2 and 3 exist because property 1 -- the one that names the bug -- is satisfied by a
// WRONG fix. Both wrong fixes were built against this file and run; exit code per subtest:
//
//   predicate variant                 other-thread   own-thread   predicates
//   per-thread (shipped)                   0             0            0
//   merged / wide (pre-fix)                1             0            1
//   always false (over-correction)         0             1            1
//
// Narrowing to always-false passes property 1 while stripping the graph handling a recording
// thread genuinely needs.  Merging the two predicates back together "for consistency" restores
// the bug but leaves property 2 green.  Neither is hypothetical: the first is the shape of an
// over-eager fix, the second is what a tidy-up does to an asymmetry that looks accidental.
//
// CPU-only by construction: nothing here allocates, touches a device or a queue, or reaches the
// unified cache. It reads and writes the recording flags and calls two pure functions.
//
// Usage:
//   ./build/bin/test-sycl-transient-alloc-intent-scope             # all three
//   ./build/bin/test-sycl-transient-alloc-intent-scope other-thread
//   ./build/bin/test-sycl-transient-alloc-intent-scope own-thread
//   ./build/bin/test-sycl-transient-alloc-intent-scope predicates

#include "common.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

const char * role_name(ggml_sycl::alloc_role role) {
    switch (role) {
        case ggml_sycl::alloc_role::WEIGHT:
            return "WEIGHT";
        case ggml_sycl::alloc_role::COMPUTE:
            return "COMPUTE";
        case ggml_sycl::alloc_role::KV:
            return "KV";
        case ggml_sycl::alloc_role::STAGING:
            return "STAGING";
        case ggml_sycl::alloc_role::GRAPH_TMP:
            return "GRAPH_TMP";
        case ggml_sycl::alloc_role::TP_TMP:
            return "TP_TMP";
        case ggml_sycl::alloc_role::EXPERT_STAGING:
            return "EXPERT_STAGING";
        case ggml_sycl::alloc_role::CONTROL:
            return "CONTROL";
        default:
            return "OTHER";
    }
}

const char * category_name(ggml_sycl::runtime_category category) {
    switch (category) {
        case ggml_sycl::runtime_category::KV_CACHE:
            return "KV_CACHE";
        case ggml_sycl::runtime_category::COMPUTE:
            return "COMPUTE";
        case ggml_sycl::runtime_category::STAGING:
            return "STAGING";
        case ggml_sycl::runtime_category::GRAPH:
            return "GRAPH";
        case ggml_sycl::runtime_category::HOST_COMPUTE:
            return "HOST_COMPUTE";
        case ggml_sycl::runtime_category::EXPERT_CACHE:
            return "EXPERT_CACHE";
        case ggml_sycl::runtime_category::CONTROL:
            return "CONTROL";
        default:
            return "OTHER";
    }
}

const char * zone_name(ggml_sycl::vram_zone_id zone) {
    switch (zone) {
        case ggml_sycl::vram_zone_id::KV:
            return "KV";
        case ggml_sycl::vram_zone_id::WEIGHT:
            return "WEIGHT";
        case ggml_sycl::vram_zone_id::ONEDNN:
            return "ONEDNN";
        case ggml_sycl::vram_zone_id::RUNTIME:
            return "RUNTIME";
        case ggml_sycl::vram_zone_id::SCRATCH:
            return "SCRATCH";
        default:
            return "COUNT (no zone preference)";
    }
}

int check_device_intent(const char *                what,
                        ggml_sycl::alloc_role       want_role,
                        ggml_sycl::runtime_category want_category,
                        ggml_sycl::vram_zone_id     want_zone) {
    const ggml_sycl::alloc_intent got = ggml_sycl_transient_device_intent("f9tg_device_probe");

    int failures = 0;
    if (got.role != want_role) {
        std::fprintf(stderr, "FAIL [%s]: device transient role is %s, expected %s\n", what, role_name(got.role),
                     role_name(want_role));
        failures++;
    }
    if (got.category != want_category) {
        std::fprintf(stderr, "FAIL [%s]: device transient category is %s, expected %s\n", what,
                     category_name(got.category), category_name(want_category));
        failures++;
    }
    if (got.constraints.prefer_vram_zone != want_zone) {
        std::fprintf(stderr, "FAIL [%s]: device transient prefer_vram_zone is %s, expected %s\n", what,
                     zone_name(got.constraints.prefer_vram_zone), zone_name(want_zone));
        failures++;
    }
    if (!got.constraints.must_device) {
        std::fprintf(stderr, "FAIL [%s]: device transient lost must_device\n", what);
        failures++;
    }
    if (!got.cohort_id || std::strcmp(got.cohort_id, "f9tg_device_probe") != 0) {
        std::fprintf(stderr, "FAIL [%s]: device transient cohort_id is '%s', expected 'f9tg_device_probe'\n", what,
                     got.cohort_id ? got.cohort_id : "(null)");
        failures++;
    }
    return failures;
}

int check_host_intent(const char *                what,
                      ggml_sycl::alloc_role       want_role,
                      ggml_sycl::runtime_category want_category,
                      bool                        want_pinned_pool,
                      bool                        want_usm_base) {
    const ggml_sycl::alloc_intent got = ggml_sycl_transient_host_pinned_intent("f9tg_host_probe");

    int failures = 0;
    if (got.role != want_role) {
        std::fprintf(stderr, "FAIL [%s]: host transient role is %s, expected %s\n", what, role_name(got.role),
                     role_name(want_role));
        failures++;
    }
    if (got.category != want_category) {
        std::fprintf(stderr, "FAIL [%s]: host transient category is %s, expected %s\n", what,
                     category_name(got.category), category_name(want_category));
        failures++;
    }
    if (got.constraints.use_pinned_pool != want_pinned_pool) {
        std::fprintf(stderr, "FAIL [%s]: host transient use_pinned_pool is %d, expected %d\n", what,
                     got.constraints.use_pinned_pool ? 1 : 0, want_pinned_pool ? 1 : 0);
        failures++;
    }
    if (got.constraints.require_host_usm_base != want_usm_base) {
        std::fprintf(stderr, "FAIL [%s]: host transient require_host_usm_base is %d, expected %d\n", what,
                     got.constraints.require_host_usm_base ? 1 : 0, want_usm_base ? 1 : 0);
        failures++;
    }
    if (!got.constraints.must_host_pinned) {
        std::fprintf(stderr, "FAIL [%s]: host transient lost must_host_pinned\n", what);
        failures++;
    }
    if (!got.cohort_id || std::strcmp(got.cohort_id, "f9tg_host_probe") != 0) {
        std::fprintf(stderr, "FAIL [%s]: host transient cohort_id is '%s', expected 'f9tg_host_probe'\n", what,
                     got.cohort_id ? got.cohort_id : "(null)");
        failures++;
    }
    return failures;
}

// Hold the process-global recording depth up for as long as the scope lives, exactly as another
// context's in-flight graph capture would.  The thread_local flag stays false on this thread,
// which is the whole point.
struct other_thread_recording {
    std::thread             worker;
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    recording = false;
    bool                    stop      = false;

    other_thread_recording() {
        worker = std::thread([this]() {
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

// Property 1: another context's graph capture must not change the shape of THIS thread's
// transients.  This is the bug.
int test_other_thread_recording_does_not_reshape_our_allocations() {
    other_thread_recording other;

    // Inertness guard only -- if the worker thread failed to raise the process-global depth,
    // nothing below is exercised and a green result would mean nothing.
    if (!ggml_sycl_graph_recording_active()) {
        std::fprintf(stderr,
                     "FAIL: test setup is inert -- the global recording depth did not make\n"
                     "      ggml_sycl_graph_recording_active() true, so nothing was exercised\n");
        return 1;
    }

    int failures = 0;
    failures += check_device_intent("another thread recording", ggml_sycl::alloc_role::COMPUTE,
                                    ggml_sycl::runtime_category::COMPUTE, ggml_sycl::vram_zone_id::SCRATCH);
    failures += check_host_intent("another thread recording", ggml_sycl::alloc_role::CONTROL,
                                  ggml_sycl::runtime_category::CONTROL, /*want_pinned_pool=*/true,
                                  /*want_usm_base=*/false);

    // Checked AFTER the fields, not before: this is the mechanism, and reporting it first
    // would return early and hide which fields the mechanism actually corrupted.
    if (ggml_sycl_graph_recording_this_thread()) {
        std::fprintf(stderr,
                     "FAIL: this thread reports itself as recording while only ANOTHER thread is.\n"
                     "      ggml_sycl_graph_recording_this_thread() must read the thread_local flag\n"
                     "      alone, not the process-global depth counter.\n");
        failures++;
    }

    if (failures) {
        std::fprintf(stderr,
                     "  ^ %d field(s) of a NON-recording thread's transient were decided by an\n"
                     "    unrelated context's recording phase.  Correctness is unaffected (graph\n"
                     "    treatment is a superset), but role/category/cohort/zone is exactly what\n"
                     "    llama.cpp-iiff Phase 0's inventory reads, and the host form also pays for\n"
                     "    a standalone USM base instead of a pinned-pool slice.\n",
                     failures);
        return 1;
    }
    std::puts("PASS: another thread's recording does not reshape this thread's transients");
    return 0;
}

// Property 2 (positive control): a thread that IS recording must still get graph-lifetime
// treatment.  Narrowing the predicate all the way to false would make property 1 pass while
// putting a graph-captured transient into the SCRATCH zone that the next graph boundary resets.
int test_own_recording_still_gets_graph_lifetime() {
    g_ggml_sycl_graph_recording = true;

    int failures = 0;
    if (!ggml_sycl_graph_recording_this_thread()) {
        std::fprintf(stderr, "FAIL: this thread set g_ggml_sycl_graph_recording but does not report recording\n");
        failures++;
    }
    failures += check_device_intent("this thread recording", ggml_sycl::alloc_role::GRAPH_TMP,
                                    ggml_sycl::runtime_category::GRAPH, ggml_sycl::vram_zone_id::COUNT);
    failures += check_host_intent("this thread recording", ggml_sycl::alloc_role::GRAPH_TMP,
                                  ggml_sycl::runtime_category::GRAPH, /*want_pinned_pool=*/false,
                                  /*want_usm_base=*/true);

    g_ggml_sycl_graph_recording = false;

    if (failures) {
        std::fprintf(stderr,
                     "  ^ %d field(s) denied graph-lifetime treatment to a thread that IS recording.\n"
                     "    A SCRATCH-zone slice does not survive the graph boundary that resets it,\n"
                     "    and a pinned-pool slice is an interior TLSF suballocation, not the USM base\n"
                     "    a recorded host copy captures.\n",
                     failures);
        return 1;
    }
    std::puts("PASS: a recording thread's transients still get graph-lifetime treatment");
    return 0;
}

// Property 3: the two predicates must remain DIFFERENT.  ggml_sycl_graph_recording_active() is
// process-wide on purpose -- it answers "may I emit a memcpy node?" -- and unifying the two for
// consistency is what reintroduces both this bug and llama.cpp-oze0's cross-context free.
int test_the_two_predicates_stay_distinct() {
    if (ggml_sycl_graph_recording_this_thread()) {
        std::fprintf(stderr, "FAIL: this thread is already recording at test entry\n");
        return 1;
    }
    if (ggml_sycl_graph_recording_active()) {
        std::fprintf(stderr, "FAIL: some other thread is still recording at test entry\n");
        return 1;
    }

    {
        other_thread_recording other;
        if (!ggml_sycl_graph_recording_active()) {
            std::fprintf(stderr,
                         "FAIL: ggml_sycl_graph_recording_active() went NARROW -- it no longer sees\n"
                         "      another thread's recording.  It is process-wide by design; the\n"
                         "      submission-constraint callers depend on that.\n");
            return 1;
        }
        if (ggml_sycl_graph_recording_this_thread()) {
            std::fprintf(stderr,
                         "FAIL: ggml_sycl_graph_recording_this_thread() went WIDE -- it reports this\n"
                         "      thread as recording because ANOTHER thread is.  That is the whole\n"
                         "      defect (llama.cpp-f9tg / llama.cpp-oze0); the asymmetry is the fix.\n");
            return 1;
        }
    }

    g_ggml_sycl_graph_recording = true;
    const bool wide_when_own    = ggml_sycl_graph_recording_active();
    const bool narrow_when_own  = ggml_sycl_graph_recording_this_thread();
    g_ggml_sycl_graph_recording = false;

    if (!wide_when_own || !narrow_when_own) {
        std::fprintf(stderr,
                     "FAIL: with THIS thread recording, both predicates must be true "
                     "(wide=%d narrow=%d)\n",
                     wide_when_own ? 1 : 0, narrow_when_own ? 1 : 0);
        return 1;
    }

    std::puts("PASS: the wide and per-thread recording predicates remain distinct");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * only     = argc > 1 ? argv[1] : nullptr;
    auto         selected = [only](const char * name) {
        return !only || std::strcmp(only, name) == 0;
    };

    if (only && !selected("other-thread") && !selected("own-thread") && !selected("predicates")) {
        std::fprintf(stderr, "unknown test '%s' (other-thread | own-thread | predicates)\n", only);
        return 2;
    }

    if (selected("other-thread")) {
        if (int rc = test_other_thread_recording_does_not_reshape_our_allocations()) {
            return rc;
        }
    }
    if (selected("own-thread")) {
        if (int rc = test_own_recording_still_gets_graph_lifetime()) {
            return rc;
        }
    }
    if (selected("predicates")) {
        if (int rc = test_the_two_predicates_stay_distinct()) {
            return rc;
        }
    }
    std::puts("PASS: transient allocation shape is scoped to the allocating thread");
    return 0;
}
