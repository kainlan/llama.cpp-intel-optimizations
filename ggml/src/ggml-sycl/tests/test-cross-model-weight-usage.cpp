// Cross-model state-leak regression test (llama.cpp-k7b0).
//
// test-llama-archs loads ~131 architectures sequentially in ONE process.
// Any process-global that describes "the model currently being loaded" and
// is never reset between loads leaks into the next model -- the same shape
// of bug as the kv_tier_manager resize()-vs-assign() defect that motivated
// this audit, one level up (a process-global instead of a per-device
// singleton). This test proves one confirmed instance of that class
// end-to-end through the REAL production API:
// ggml_backend_sycl_register_weight_usage() / ggml_sycl_get_tensor_usage()
// (g_sycl_weight_usages, ggml-sycl.cpp), bracketed by the REAL load
// boundary, ggml_backend_sycl_model_load_begin(), whose outer branch
// calls ggml_sycl_reset_model_load_scratch_state() (the fix under test).
//
// The bug: ggml_backend_sycl_register_weight_usage() only *emplaces* a
// name's usage on first sight, and forces UNKNOWN on a *mismatch* against
// whatever is already mapped -- correct within one model's own tied-weight
// detection (e.g. a legacy checkpoint tying output.weight to
// token_embd.weight). Never clearing the map between models means a name a
// PREVIOUS, unrelated model forced to UNKNOWN for its own reasons poisons a
// DIFFERENT model's first, and only, registration of that same name:
// `it->second != mapped` reads true against the stale UNKNOWN, so the new
// model's real classification is silently discarded before it is ever used.
//
// All tensor names below are synthetic ("zzz_test_*") and deliberately
// unlike any real GGUF tensor name, so ggml_sycl_get_tensor_usage()'s
// pattern-based fallback (infer_tensor_usage(), used only when the name is
// absent from the registry) cannot accidentally produce the "expected"
// answer on its own. Every check below is therefore a check of the
// registry/reset behavior, not of the fallback classifier -- see
// tests/test-sycl-tensor-usage.cpp for that.
//
// Mutation that proves this test is specific to the fix and not a
// tautology: comment out the `g_sycl_weight_usages.clear();` line inside
// ggml_sycl_reset_model_load_scratch_state() (ggml-sycl.cpp). Only check 4
// fails ("model B: shared name gets its OWN usage, not model A's stale
// UNKNOWN"); checks 1-3 stay green. That is the specificity bar this test
// is held to: it detects THIS leak, not "something changed".

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

    const auto & info = ggml_sycl_info();
    if (info.device_count <= 0) {
        // 77 (ctest SKIP_RETURN_CODE), not 0: nothing was verified, so this
        // must not report success.
        fprintf(stderr, "SKIP: no SYCL devices available -- NO DEVICE WORK WAS PERFORMED.\n");
        return 77;
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
    ggml_sycl_load_txn load_a{};
    ggml_sycl_model_token model_a{};
    check(ggml_backend_sycl_model_load_begin(&load_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle begin");

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
    ggml_free(ctx);
    (void) ggml_backend_sycl_model_unloaded_token(model_b);
    (void) ggml_backend_sycl_model_unloaded_token(model_a);

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif
