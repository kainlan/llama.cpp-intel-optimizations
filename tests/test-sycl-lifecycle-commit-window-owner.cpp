// Owner resolution must survive the load_end COMMITTING window (llama.cpp-mik6).
//
// ---------------------------------------------------------------------------
// What this pins
// ---------------------------------------------------------------------------
//
// S1-PRELOAD does not run while a load transaction is ACTIVE. It runs from
// ggml_backend_sycl_model_load_end(), AFTER prepare_end() has moved the txn to
// finish_phase::COMMITTING and BEFORE the ModelState is published LIVE:
//
//   load_end: prepare_end()                 -> phase = COMMITTING
//             acquire_finisher_effect()     -> the effects window opens
//             ggml_sycl_model_loading_effects(false, outer)
//                -> ggml_sycl_preload_model_weights()   <-- S1-PRELOAD
//             ... publish ModelState LIVE ...
//
// Inside that window ggml_sycl_exact_wrapper_owner() must still resolve the
// owning model. It is the sole authority behind
// ggml_sycl_get_moe_expert_cache_key(), which fails CLOSED on a partial owner,
// and behind ggml_sycl_get_tensor_usage()'s registered-usage lookup. When it
// returns a zero token during preload, every MoE expert cache key is invalid
// and all 72 GPT-OSS expert tensors are skipped in silence -- no log line, no
// exception, just `moe cached=0 failed=72` and a rolled-back load.
//
// Regression history: acquire_load_effect() admits only finish_phase::ACTIVE
// transactions, so once bfd28f84b routed wrapper-owner resolution through it,
// the entire preload window resolved to {}. The window is only reachable from
// inside load_end, which is why this fixture drives the registry directly
// rather than calling the public load_end wrapper.
//
// ---------------------------------------------------------------------------
// Why it is host-only
// ---------------------------------------------------------------------------
//
// Nothing here creates a queue, allocates device memory, or loads a model. It
// drives the lifecycle Registry and reads one test seam, so it is safe at any
// parallelism and is NOT a member of the GPU-allocating OOM-hazard family.

#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/model-lifecycle.hpp"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (condition) {
        std::printf("  ok  : %s\n", what);
        return;
    }
    std::printf("  FAIL: %s\n", what);
    ++failures;
}

}  // namespace

int main() {
    auto & registry = ggml_sycl::lifecycle::global_registry();

    const auto begin = registry.begin_outer();
    if (begin.code != ggml_sycl::lifecycle::error::OK || begin.token.model.value == 0) {
        std::fprintf(stderr, "failed to open a load transaction fixture\n");
        return 1;
    }
    const auto     txn      = begin.txn;
    const uint64_t model_id = begin.token.model.value;

    // load_end binds the candidate for the duration of the end sequence
    // (ggml_sycl_load_candidate_end_scope). Mirror that here so the fixture
    // observes the same binding state the real window has.
    registry.bind_candidate(txn);

    std::printf("=== phase ACTIVE (positive control) ===\n");
    // The control matters: if the seam could not report `true` under ANY phase,
    // its `false` inside the COMMITTING window below would be uninformative.
    check(ggml_sycl::test_exact_wrapper_owner_matches(model_id, model_id), "ACTIVE phase resolves the owning model");
    check(!ggml_sycl::test_exact_wrapper_owner_matches(model_id + 1, model_id),
          "ACTIVE phase refuses a mismatched wrapper model");

    auto ticket = registry.prepare_end(txn, /*explicit_success=*/true, /*output_available=*/true);
    if (!ticket.finisher || !ticket.commit) {
        std::fprintf(stderr, "prepare_end did not hand this caller the committing finisher\n");
        registry.unbind_candidate(txn);
        return 1;
    }

    auto finisher_effect = registry.acquire_finisher_effect(ticket);
    if (!finisher_effect) {
        std::fprintf(stderr, "failed to open the finisher effect window\n");
        (void) registry.finalize_end(ticket, false);
        registry.unbind_candidate(txn);
        return 1;
    }

    std::printf("=== phase COMMITTING (the S1-PRELOAD window) ===\n");
    // THE REGRESSION. Everything owner-scoped that runs during preload depends
    // on this: the MoE expert cache key, which has no fallback and fails
    // closed, and the registered tensor-usage lookup, whose fallback silently
    // reclassifies ffn_*_exps.bias out of the MoE branch.
    check(ggml_sycl::test_exact_wrapper_owner_matches(model_id, model_id),
          "COMMITTING window resolves the owning model");

    // The guard the fix must not weaken: a wrapper naming a different model is
    // still refused, so no model can borrow another's identity in this window.
    check(!ggml_sycl::test_exact_wrapper_owner_matches(model_id + 1, model_id),
          "COMMITTING window refuses a mismatched wrapper model");
    check(!ggml_sycl::test_exact_wrapper_owner_matches(0x5a17, model_id),
          "COMMITTING window refuses an unknown wrapper model");

    finisher_effect = {};
    (void) registry.finalize_end(ticket, false);
    registry.unbind_candidate(txn);

    std::printf("\nSYCL commit-window owner resolution: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
