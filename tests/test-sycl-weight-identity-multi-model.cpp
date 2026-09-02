// Regression test pinning the multi-model weight-identity lookup order
// (llama.cpp-s83n).
//
// Bug (found and fixed in passing by llama.cpp-n3pw, commit 6f38363f1;
// verified present at HEAD fed0b58e2, ggml-sycl.cpp around line 13968):
// ggml_backend_sycl_get_weight_cache_key() used to resolve a tensor's
// weight-identity OWNER from the PUBLISHED PLACEMENT PLAN, which names
// exactly one model. With two models loaded, whichever model was NOT
// currently published found nothing under its own owner-scoped
// g_sycl_weight_identities_by_name key and silently fell through to the
// UUID/no-identity path (has_gguf=false), losing its file identity
// entirely.
//
// The fix resolves the owner from the tensor's own extra->model_id FIRST --
// the same order ggml_sycl_get_tensor_usage() already used -- and only
// falls back to the published-plan snapshot when extra->model_id is 0 (no
// owner known). See the comment beginning "Resolve the owner from the
// tensor's own extra where it has one" directly above the fix.
//
// This test drives the real lifecycle load-transaction API
// (ggml_backend_sycl_model_load_begin/model_load_end,
// ggml_backend_sycl_register_weight_identity) so each synthetic model gets
// a genuine, LIVE ggml_sycl_model_token from the registry -- not an
// arbitrary caller-chosen id -- then builds standalone tensors whose
// extra->model_id names their own model, independent of which model's plan
// is currently PUBLISHED (ggml_backend_sycl_activate_model_plan). Both
// tensors must resolve their own registered file identity in every
// published-plan configuration; neither may resolve the other's identity or
// silently drop to a UUID.
//
// Mutation control (see the task report for the exact diffs and captured
// output): reverting the owner resolution to consult the published plan
// FIRST -- either by swapping ggml_sycl_exact_wrapper_owner(extra_model_id)
// for ggml_sycl_exact_wrapper_owner(0) unconditionally, or by deleting the
// `if (!name.empty() ...)` block's extra->model_id read so owner resolution
// always starts from 0 -- makes round 1's model-A checks fail (B is
// published after B's commit) while round 2's model-B checks fail (A is
// published after the explicit reactivation). The round that currently
// holds the published plan keeps passing either way, which is exactly the
// asymmetry the bug produced: whichever model is NOT published loses its
// identity.

#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_SYCL)
int main() {
    // 77 is ctest's SKIP_RETURN_CODE. Exiting 0 here would report a run that
    // tested nothing as a pass -- see CLAUDE.md, "a SKIP line with status 0
    // is not a pass".
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the lookup order.\n");
    return 77;
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

static void print_cache_id(const char * label, const ggml_sycl_cache_id & id) {
    fprintf(stderr, "  %s: valid=%d has_gguf=%d file_id=0x%llx file_offs=%zu model_id=%llu\n", label, id.valid,
            id.has_gguf, (unsigned long long) id.file_id, id.file_offs, (unsigned long long) id.model_id);
}

int main() {
    // Never enumerate the iGPU by accident (CLAUDE.md, llama.cpp-403s) -- this
    // test never touches a device at all, but stay consistent with every
    // other test in this family that links ggml-sycl.
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1", 1);
    }

    const char * shared_name = "zzz_s83n_shared.weight";

    ggml_init_params params{};
    params.mem_size    = 4 * 1024 * 1024;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "ggml_init failed");

    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    check(tensor_a != nullptr && tensor_b != nullptr, "tensor allocation failed");
    ggml_set_name(tensor_a, shared_name);
    ggml_set_name(tensor_b, shared_name);

    // ---- Model A: register its own identity for the shared tensor name ---
    ggml_sycl_load_txn    load_a{};
    ggml_sycl_model_token model_a{};
    check(ggml_backend_sycl_model_load_begin(&load_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle begin");
    // model_id=0: adopt the load transaction's own token (see the API
    // comment on ggml_backend_sycl_register_weight_identity).
    ggml_backend_sycl_register_weight_identity(tensor_a, /*file_idx=*/0, /*file_offs=*/0x1000, ggml_nbytes(tensor_a),
                                               /*model_id=*/0);
    check(ggml_backend_sycl_model_load_end(load_a, true, &model_a) == GGML_SYCL_LIFECYCLE_OK,
          "model A lifecycle commit");
    check(model_a.model_id != 0, "model A got a real, nonzero model id");

    // ---- Model B: a second, unrelated model over the SAME tensor name ----
    ggml_sycl_load_txn    load_b{};
    ggml_sycl_model_token model_b{};
    check(ggml_backend_sycl_model_load_begin(&load_b) == GGML_SYCL_LIFECYCLE_OK, "model B lifecycle begin");
    ggml_backend_sycl_register_weight_identity(tensor_b, /*file_idx=*/0, /*file_offs=*/0x2000, ggml_nbytes(tensor_b),
                                               /*model_id=*/0);
    check(ggml_backend_sycl_model_load_end(load_b, true, &model_b) == GGML_SYCL_LIFECYCLE_OK,
          "model B lifecycle commit");
    check(model_b.model_id != 0 && model_b.model_id != model_a.model_id,
          "model B got a real, nonzero, distinct model id");

    // Point each tensor's extra at its own model -- this is the field
    // ggml_backend_sycl_get_weight_cache_key() must consult first.
    ggml_tensor_extra_gpu extra_a{};
    extra_a.model_id = model_a.model_id;
    tensor_a->extra  = &extra_a;

    ggml_tensor_extra_gpu extra_b{};
    extra_b.model_id = model_b.model_id;
    tensor_b->extra  = &extra_b;

    // ---- Round 1: model B's load just committed, so B -- not A -- is the
    // currently published/active plan. This is the exact configuration the
    // bug requires: querying A's tensor while B is published.
    {
        ggml_sycl_cache_id key_a = ggml_backend_sycl_get_weight_cache_key(tensor_a, 0);
        ggml_sycl_cache_id key_b = ggml_backend_sycl_get_weight_cache_key(tensor_b, 0);
        print_cache_id("round 1, model A (not published)", key_a);
        print_cache_id("round 1, model B (published)", key_b);

        check(key_a.valid, "round 1: model A key is valid");
        check(key_a.has_gguf, "round 1: model A resolves its own GGUF identity, not the UUID fallback");
        check(key_a.file_offs == 0x1000, "round 1: model A resolves its OWN file_offs, not B's or a UUID");

        check(key_b.valid, "round 1: model B key is valid");
        check(key_b.has_gguf, "round 1: model B resolves its own GGUF identity");
        check(key_b.file_offs == 0x2000, "round 1: model B resolves its OWN file_offs");

        check(key_a.file_offs != key_b.file_offs, "round 1: the two models' identities are not conflated");
    }

    // ---- Round 2: explicitly (re)activate A, then re-query both. If the
    // lookup depended on which plan is currently published, this would flip
    // the answers; it must not, because extra->model_id already names the
    // exact owner. B is NOT published in this round, so B's checks here are
    // the mirror image of A's checks in round 1.
    check(ggml_backend_sycl_activate_model_plan(model_a) == GGML_SYCL_LIFECYCLE_OK, "reactivate exact model A");
    {
        ggml_sycl_cache_id key_a = ggml_backend_sycl_get_weight_cache_key(tensor_a, 0);
        ggml_sycl_cache_id key_b = ggml_backend_sycl_get_weight_cache_key(tensor_b, 0);
        print_cache_id("round 2, model A (published)", key_a);
        print_cache_id("round 2, model B (not published)", key_b);

        check(key_a.has_gguf && key_a.file_offs == 0x1000,
              "round 2: model A still resolves its own identity with A published");
        check(key_b.has_gguf, "round 2: model B (not published) still resolves its own GGUF identity");
        check(key_b.file_offs == 0x2000, "round 2: model B (not published) resolves its OWN file_offs, not a UUID");
    }

    (void) ggml_backend_sycl_model_unloaded_token(model_b);
    (void) ggml_backend_sycl_model_unloaded_token(model_a);
    ggml_free(ctx);

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    if (g_failures > 0) {
        printf(
            "FAILED: the multi-model weight-identity lookup order regressed to published-plan-first "
            "(llama.cpp-s83n).\n");
        return 1;
    }
    printf("PASS: both models resolve their own weight identity regardless of the published plan.\n");
    return 0;
}

#endif
