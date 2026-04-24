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
// `llama.cpp-m09zb` (single-context slot acquire-after-release wedge in
// the SYCL staging_buffer_pool); the fix is scoped as epic prerequisite
// E1, not this session.
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
    findings f;
    f.canary_id = "D0.3";
    f.result    = status::INCONCLUSIVE;

    // Initialize the llama backend so the SYCL device registry is
    // populated before we query device count.
    llama_backend_init();

    const int n_sycl = ggml_backend_sycl_get_device_count();
    add(f, "sycl_device_count", std::to_string(n_sycl));

    // Single-device path is the only path this canary implements today.
    // The scenario the canary was designed to probe (scheduler-inserted
    // CPY nodes on cross-device edges) cannot occur without ≥2 SYCL
    // devices, so with 1 device we report INCONCLUSIVE /
    // reason=single_device — the "scenario-unavailable" PASS-equivalent
    // per the plan's acceptance rubric.
    //
    // TODO: when E1 lands (staging_buffer_pool restructure, llama.cpp-m09zb)
    // AND ≥2 SYCL devices are safely usable on this host, replace this
    // canary body with the real 5-run multi-device CPY-name stability
    // check: load model with split_mode=ROW + tensor_split={0.5,0.5},
    // decode, capture scheduler-inserted CPY node names, repeat 5x,
    // verify set equality.
    f.summary        = "Single SYCL device visible; cross-device CPY scenario unavailable";
    f.recommendation = "Re-run once ≥2 SYCL devices are safely usable on this host "
                       "(B50 currently disabled per host policy; see CLAUDE.md memory "
                       "feedback_disable_b50.md) AND the E1 staging_buffer_pool fix "
                       "(llama.cpp-m09zb) has landed";
    add(f, "reason", "single_device");

    add(f, "plan_section",
        "docs/plans/2026-04-22-planner-canaries-implementation.md §Task 3 (D0.3)");

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);

    llama_backend_free();
    // INCONCLUSIVE is a PASS-equivalent per the plan's acceptance rubric
    // for "scenario unavailable", so return 0.
    return 0;
}
