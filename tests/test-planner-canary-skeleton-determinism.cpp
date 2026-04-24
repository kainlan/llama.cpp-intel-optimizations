// Canary D0.1 — skeleton graph size determinism.
//
// Question (from docs/plans/2026-04-22-planner-canaries-implementation.md):
// does the SYCL scheduler produce the same per-backend reserve sizes
// across repeated identical context constructions, and does the size
// change after a forward pass? If yes, A3a's mini-context approach can
// produce authoritative compute-plan sizes from a throwaway graph
// without inference state.
//
// Status today: this canary is intentionally SKELETAL. The full test
// requires `llama_init_from_model` to return cleanly on a GPU-offloaded
// Mistral 7B, which is currently wedged by bead `llama.cpp-m09zb`: the
// L0 DirectSubmission non-flush surfaces during `ggml_sycl_preload_model_weights`
// at `ggml/src/ggml-sycl/common.hpp:1863` (gdb-confirmed backtrace;
// strace shows only the outer `timeout` SIGTERM, no abort). Canary
// D0.4 reproduces the same L0 hang signature via a different code
// path (`ggml_backend_sycl_buffer_set_tensor` at
// `ggml-sycl.cpp:12685`, no staging pool involved) — corroborating
// witness that the bug is broader than any single amplifier.
//
// The fix is scoped as the main epic's prerequisite E1 (see plan doc
// §Track E). Until E1 lands, this canary writes INCONCLUSIVE with a
// documented reason and exits 0 so Task 5 aggregation can cite it.

#include "test-planner-canary-common.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace planner_canary;

static const char * MD_PATH   = "docs/plans/data/planner-canaries/d0.1-skeleton-determinism.md";
static const char * JSON_PATH = "tests/data/planner-canaries/d0.1.json";

int main(int /*argc*/, char ** /*argv*/) {
    findings f;
    f.canary_id = "D0.1";
    f.result    = status::INCONCLUSIVE;

    // TODO: when E1 lands (staging_buffer_pool restructure + broader
    // L0 flush path, bead llama.cpp-m09zb), replace this canary body
    // with the real determinism check. The planned three-sample
    // protocol on Mistral 7B Q4_0:
    //   (1) ctx1 = fresh create-destroy-recreate pair → capture
    //       per-backend reserve size via llama_state_get_size
    //   (2) ctx2 = second fresh construction with identical cparams
    //       (n_ctx=4096, n_batch=512, n_ubatch=512, flash_attn_type=AUTO)
    //       → capture size; assert s2 == s1
    //   (3) ctx_fwd = construct, run one-token decode, capture pre-
    //       and post-decode sizes; assert s_post == s_pre
    // Sequential lifecycle (one ctx alive at a time) avoids the
    // previously-hypothesized two-contexts-on-same-model crash; a
    // guarded probe via `D0_1_PROBE_MULTICONTEXT=1` re-tests that
    // hypothesis separately once the backend is fixed.
    f.summary        = "Canary cannot complete today: llama_model_load_from_file "
                       "wedges in the SYCL S1-PRELOAD weight stage before any "
                       "determinism measurement can run (bead llama.cpp-m09zb)";
    f.recommendation = "Re-run once the E1 staging_buffer_pool fix + broader L0 "
                       "flush work (llama.cpp-m09zb) has landed; D0.4 is the "
                       "corroborating witness that the bug is not "
                       "staging-pool-specific";

    add(f, "reason",                "blocked_on_m09zb");
    add(f, "blocker_bead",           "llama.cpp-m09zb");
    add(f, "hang_location",          "ggml/src/ggml-sycl/common.hpp:1863");
    add(f, "hang_function",          "staging_buffer_pool::acquire");
    add(f, "caller_api",             "llama_model_load_from_file via ggml_sycl_preload_model_weights");
    add(f, "corroborating_witness",  "D0.4 - ggml_backend_sycl_buffer_set_tensor at ggml-sycl.cpp:12685");
    add(f, "plan_section",           "docs/plans/2026-04-22-planner-canaries-implementation.md §Task 1 (D0.1)");

    write_markdown(f, MD_PATH);
    write_json    (f, JSON_PATH);

    // INCONCLUSIVE is a PASS-equivalent per the plan's acceptance
    // rubric for "scenario unavailable", so return 0.
    return 0;
}
