// Canary D0.3 — post-scheduler-split CPY node visibility + stability.
//
// Question (from docs/plans/2026-04-22-planner-canaries-implementation.md):
// when the SYCL scheduler inserts synthetic CPY nodes for cross-backend
// edges, do those nodes have stable deterministic names across runs?
// If yes, the planner's plan.ops table can key on op name; if no, it
// must key on op_id or another stable identifier.
//
// Status today: this canary is intentionally SKELETAL. The full test
// requires a multi-device scheduler split, which requires loading a
// model with `n_gpu_layers=999` plus `LLAMA_SPLIT_MODE_ROW` across two
// visible SYCL devices. That model-load path is currently wedged by
// `llama.cpp-m09zb` (staging_buffer_pool multi-context re-entry); the
// fix is scoped as epic prerequisite E1, not this session.
//
// Until E1 lands, this canary writes INCONCLUSIVE with a documented
// reason so the A3a design-doc update in Task 5 can cite it.

#include "test-planner-canary-common.hpp"
#include "llama.h"
#include "ggml-sycl.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/d0.3-cpy-visibility.md";
static const char * JSON_PATH = "tests/data/planner-canaries/d0.3.json";

int main(int /*argc*/, char ** /*argv*/) {
    // Unbuffer stdout so nothing is lost if a later canary iteration
    // aborts mid-run — per feedback from the Task 2 stdio race.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    findings f;
    f.canary_id = "D0.3";
    f.result    = status::INCONCLUSIVE;

    // Initialize the llama backend so the SYCL device registry is
    // populated before we query device count.
    llama_backend_init();

    const int n_sycl = ggml_backend_sycl_get_device_count();
    add(f, "sycl_device_count", std::to_string(n_sycl));

    if (n_sycl < 2) {
        // Single-device path: the scenario the canary was designed to
        // probe (scheduler-inserted CPY nodes on cross-device edges)
        // cannot occur without at least two SYCL devices. Per the
        // assignment this is a PASS-equivalent INCONCLUSIVE:
        // "scenario-unavailable" not "canary-failure".
        f.summary        = "Single SYCL device visible; cross-device CPY scenario unavailable";
        f.recommendation = "Re-run once ≥2 SYCL devices are safely usable on this host "
                           "(B50 currently disabled per host policy; see CLAUDE.md memory "
                           "feedback_disable_b50.md)";
        add(f, "reason", "single_device");
    } else {
        // Multi-device path: the full test would call
        // llama_model_load_from_file(.., n_gpu_layers=999,
        // split_mode=LLAMA_SPLIT_MODE_ROW, tensor_split={0.5,0.5}),
        // run N decodes, capture CPY node names, and compare sets
        // across runs. That load path is currently wedged by
        // `llama.cpp-m09zb` (staging_buffer_pool multi-context re-entry
        // observed during implementer-1's D0.1 + implementer-2's D0.2
        // multi-context attempts). Per the team-lead's "feasibility
        // test, not a fix" direction, we record this as blocked
        // on E1 rather than attempting the load.
        f.summary        = "Multi-device CPY stability untestable: "
                           "blocked on llama.cpp-m09zb staging_buffer_pool wedge (epic prerequisite E1)";
        f.recommendation = "switch-to-plan-B: re-run this canary after E1 "
                           "(staging_buffer_pool restructure) lands and validates "
                           "multi-context model loads on SYCL";
        add(f, "reason",         "blocked_on_m09zb");
        add(f, "blocker_bead",   "llama.cpp-m09zb");
        add(f, "e1_requirement", "staging_buffer_pool restructure to support "
                                 "multi-context model load");
    }

    add(f, "plan_section",
        "docs/plans/2026-04-22-planner-canaries-implementation.md §Task 3 (D0.3)");
    add(f, "what_this_would_test_when_unblocked",
        "5 consecutive model-load + decode cycles with "
        "split_mode=ROW, tensor_split={0.5,0.5}; capture scheduler-inserted "
        "CPY node names across runs; verify set equality to decide whether "
        "C2 can key plan.ops on op_name or must use op_id");

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);

    llama_backend_free();
    // INCONCLUSIVE is a PASS-equivalent per the plan's acceptance rubric
    // for "scenario unavailable", so return 0.
    return 0;
}
