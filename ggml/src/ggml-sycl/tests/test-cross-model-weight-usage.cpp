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
// Mutation control: key g_sycl_weight_usages by bare name, or clear it at B's
// load boundary. The B or reactivated-A exact-owner checks then fail.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
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

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

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
    check(ggml_backend_sycl_stage_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
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

    check(ggml_backend_sycl_model_load_end(load_a, true, &model_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle commit");

    // ---- Model B: a DIFFERENT, unrelated model in the SAME process ------
    ggml_sycl_load_txn load_b{};
    ggml_sycl_model_token model_b{};
    check(ggml_backend_sycl_model_load_begin(&load_b) == GGML_SYCL_LIFECYCLE_OK, "model B lifecycle begin");
    check(ggml_backend_sycl_stage_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
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
