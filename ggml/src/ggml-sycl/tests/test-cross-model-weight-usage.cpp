// Exact-owner weight-usage regression. A and B remain LIVE concurrently and
// classify the same tensor name differently. Activating A, then B, then A
// again must select immutable metadata by exact ModelToken rather than a
// process-global name row or "most recently loaded" state.
//
// All tensor names below are synthetic ("zzz_test_*") and deliberately
// unlike any real GGUF tensor name, so ggml_sycl_get_tensor_usage()'s
// pattern-based fallback (infer_tensor_usage(), used only when the name is
// absent from the registry) cannot accidentally produce the "expected"
// answer on its own. Every check below is therefore a check of the
// registry/reset behavior, not of the fallback classifier -- see
// tests/test-sycl-tensor-usage.cpp for that.
//
// The exception is checks 7 to 11, which need expert-family names: the loader
// registers an expert family's usage from the family alone, so an expert's
// per-expert ".scale" or ".input_scale" arrives here as MOE_EXPERT_WEIGHT, and
// the registry must store MOE_INTERMEDIATE for it or the S1 preload, which
// reads the registry, skips the tensor while the planner gave it a dense entry.
// Checks 8 and 9 use the ffn_norm_exps family (arctic), which the loader
// registers as MOE_EXPERT_WEIGHT but which is outside the name rule's expert
// families: the name rule calls those names NORM (check 7 pins that), so only
// a stored, corrected registry row can read MOE_INTERMEDIATE for them.
//
// Mutation control: key g_sycl_weight_usages by bare name, or clear it at B's
// load boundary. The B or reactivated-A exact-owner checks then fail.

#include "../../../../tests/sycl-selector-fallback.hpp"
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

static int g_checks   = 0;
static int g_failures = 0;

static void check(bool cond, const char * label) {
    g_checks++;
    if (!cond) {
        g_failures++;
        fprintf(stderr, "FAIL: %s\n", label);
    }
}

static tensor_usage usage_of(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 8, 8);
    ggml_set_name(t, name);
    return ggml_sycl_get_tensor_usage(t);
}

int main(int, char ** argv) {
    sycl_test_selector_fallback(argv, "level_zero:0");

    ggml_init_params params{};
    params.mem_size    = 16 * 1024 * 1024;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        return 1;
    }

    // ---- Model A --------------------------------------------------------
    ggml_sycl_tensor_inventory inventory{};
    inventory.n_ctx = 32;
    inventory.n_ubatch = 8;
    ggml_sycl_placement_envelope envelope{};
    envelope.n_ctx = 32;
    envelope.n_ubatch = 8;
    envelope.n_seq_max = 1;

    ggml_sycl_load_txn load_a{};
    ggml_sycl_model_token model_a{};
    check(ggml_backend_sycl_model_load_begin(&load_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle begin");
    check(ggml_backend_sycl_stage_load_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
          "model A plan stage");

    // Check 1 (positive control): first-sight registration of a name unique
    // to this run classifies correctly. Failing this means the harness
    // itself -- register/lookup through the map -- is broken, independent
    // of any cross-model leak.
    ggml_backend_sycl_register_weight_usage("zzz_test_first_sight", GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT);
    check(usage_of(ctx, "zzz_test_first_sight") == tensor_usage::ATTENTION_WEIGHT,
          "check 1: first-sight registration classifies correctly");

    // Set up model A's own tied-weight case: the SAME name registered twice
    // with DIFFERENT usages forces UNKNOWN. This is the correct outcome FOR
    // MODEL A -- it is what the tied-weight detection is for.
    ggml_backend_sycl_register_weight_usage("zzz_test_tied", GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT);
    ggml_backend_sycl_register_weight_usage("zzz_test_tied", GGML_SYCL_TENSOR_USAGE_FFN_WEIGHT);

    // Check 2: confirm the precondition -- model A really did force UNKNOWN
    // for its own tied-weight name. If this fails, check 4 below would prove
    // nothing (there would be no stale UNKNOWN to leak in the first place).
    check(usage_of(ctx, "zzz_test_tied") == tensor_usage::UNKNOWN,
          "check 2: model A's own tied-weight case forces UNKNOWN (precondition for check 4)");

    // A name every architecture reuses verbatim with a CONSISTENT usage.
    ggml_backend_sycl_register_weight_usage("zzz_test_consistent", GGML_SYCL_TENSOR_USAGE_NORM);

    // Check 7 (precondition for checks 8 and 9): the name rule does not call these names MOE_INTERMEDIATE, so the
    // fallback cannot produce the answer checks 8 and 9 expect. Registered nowhere before this line.
    check(infer_tensor_usage("blk.0.ffn_norm_exps.scale") == tensor_usage::NORM &&
              infer_tensor_usage("blk.0.ffn_norm_exps.input_scale") == tensor_usage::NORM,
          "check 7: the name rule classifies the ffn_norm_exps scale names NORM (precondition for checks 8 and 9)");
    // Checks 8 and 9: the loader registers an expert family's per-expert scale with the family's usage. The registry
    // must store MOE_INTERMEDIATE for it, the class the planner placed it by.
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_norm_exps.scale", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);
    check(usage_of(ctx, "blk.0.ffn_norm_exps.scale") == tensor_usage::MOE_INTERMEDIATE,
          "check 8: an expert .scale registered as MOE_EXPERT_WEIGHT reads MOE_INTERMEDIATE from its registry row");
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_norm_exps.input_scale",
                                            GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);
    check(usage_of(ctx, "blk.0.ffn_norm_exps.input_scale") == tensor_usage::MOE_INTERMEDIATE,
          "check 9: an expert .input_scale registered as MOE_EXPERT_WEIGHT reads MOE_INTERMEDIATE");
    // Check 10 (control): the expert matrix of a family keeps the family's usage.
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_gate_up_exps.weight", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);
    check(usage_of(ctx, "blk.0.ffn_gate_up_exps.weight") == tensor_usage::MOE_EXPERT_WEIGHT,
          "check 10 (control): the expert .weight registered as MOE_EXPERT_WEIGHT reads MOE_EXPERT_WEIGHT");
    // Check 11 pins today's behaviour, not the goal: a family's .bias is stored as the family's MOE_EXPERT_WEIGHT,
    // while the name rule calls it MOE_INTERMEDIATE (owed: "Tracker tasks owed" in
    // docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md). A fix of .bias must change this check
    // on purpose.
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_down_exps.bias", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);
    check(usage_of(ctx, "blk.0.ffn_down_exps.bias") == tensor_usage::MOE_EXPERT_WEIGHT,
          "check 11 (today's behaviour): an expert .bias registered as MOE_EXPERT_WEIGHT reads MOE_EXPERT_WEIGHT");

    check(ggml_backend_sycl_model_load_end(load_a, true, &model_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle commit");

    // ---- Model B: a DIFFERENT, unrelated model in the SAME process ------
    ggml_sycl_load_txn load_b{};
    ggml_sycl_model_token model_b{};
    check(ggml_backend_sycl_model_load_begin(&load_b) == GGML_SYCL_LIFECYCLE_OK, "model B lifecycle begin");
    check(ggml_backend_sycl_stage_load_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
          "model B plan stage");

    // Check 3 (negative control): the reused name, registered fresh with
    // the SAME usage as model A, must still read correctly after the reset.
    // This must stay green whether or not the fix (or its revert) is
    // present -- it demonstrates the mutation named above is scoped to the
    // tied-weight leak and does not just break the reset wholesale.
    ggml_backend_sycl_register_weight_usage("zzz_test_consistent", GGML_SYCL_TENSOR_USAGE_NORM);
    check(usage_of(ctx, "zzz_test_consistent") == tensor_usage::NORM,
          "check 3 (negative control): reused consistent name survives the reset");

    // Check 4 (the bug): model B has its own, single, unrelated tensor that
    // happens to share model A's tied-weight name. Model A's UNKNOWN must
    // not survive into model B's first, and only, registration of that name.
    ggml_backend_sycl_register_weight_usage("zzz_test_tied", GGML_SYCL_TENSOR_USAGE_EMBEDDING);
    check(usage_of(ctx, "zzz_test_tied") == tensor_usage::EMBEDDING,
          "check 4: model B's shared name gets its OWN usage, not model A's stale UNKNOWN");

    check(ggml_backend_sycl_model_load_end(load_b, true, &model_b) == GGML_SYCL_LIFECYCLE_OK, "model B lifecycle commit");

    // Reactivating A must select A's exact immutable metadata even though B's
    // LIVE same-name row has a different classification.
    check(ggml_backend_sycl_activate_model_plan(model_a) == GGML_SYCL_LIFECYCLE_OK, "reactivate exact model A");
    check(usage_of(ctx, "zzz_test_tied") == tensor_usage::UNKNOWN,
          "check 5: A->B->reactivate-A lookup restores A exact same-name usage");
    check(ggml_backend_sycl_activate_model_plan(model_b) == GGML_SYCL_LIFECYCLE_OK, "reactivate exact model B");
    check(usage_of(ctx, "zzz_test_tied") == tensor_usage::EMBEDDING,
          "check 6: exact B lookup remains independent of A");
    ggml_free(ctx);
    (void) ggml_backend_sycl_model_unloaded_token(model_b);
    (void) ggml_backend_sycl_model_unloaded_token(model_a);

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif
