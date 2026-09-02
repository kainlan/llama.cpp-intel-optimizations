// Regression test pinning the multi-model weight-identity lookup order
// (llama.cpp-s83n).
//
// Bug (found and fixed in passing by llama.cpp-n3pw, commit 6f38363f1;
// verified present at HEAD fed0b58e2, inside
// ggml_backend_sycl_get_weight_cache_key() in ggml/src/ggml-sycl/ggml-sycl.cpp --
// locate it with
//     cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n 'ggml_backend_sycl_get_weight_cache_key'
// rather than an absolute line number, which llama.cpp-qq19 is about to move):
// that function used to resolve a tensor's weight-identity OWNER from the
// PUBLISHED PLACEMENT PLAN, which names exactly one model. With two models
// loaded, whichever model was NOT currently published found nothing under
// its own owner-scoped g_sycl_weight_identities_by_name key and silently
// fell through to the UUID/no-identity path (has_gguf=false), losing its
// file identity entirely.
//
// The fix resolves the owner from the tensor's own extra->model_id FIRST --
// the same order ggml_sycl_get_tensor_usage() already used -- and only
// falls back to the published-plan snapshot when extra->model_id is 0 (no
// owner known). This lives inside ggml_backend_sycl_get_weight_cache_key()
// (ggml/src/ggml-sycl/ggml-sycl.cpp; the grep above locates it -- a line
// number is not cited here because llama.cpp-qq19 is about to move this
// resolution behind a new ggml_sycl_resolve_tensor_owner() helper). Today
// that reads:
//         const uint64_t extra_model_id = extra ? extra->model_id : 0;
//         const auto     owner          = ggml_sycl_exact_wrapper_owner(extra_model_id);
//
// -----------------------------------------------------------------------
// Why this test is genuinely device-free (CLAUDE.md "GPU/model-loading work
// is SERIALISED THROUGH THE LEAD SESSION" -- a hostonly-labelled test must
// never enumerate a SYCL device).
//
// The obvious way to drive this code is the public model-load lifecycle API
// (ggml_backend_sycl_model_load_begin/model_load_end,
// ggml_backend_sycl_activate_model_plan, ggml_backend_sycl_model_unloaded_token).
// All four call ggml_sycl_info() -- device enumeration, which GGML_ABORTs
// with no devices present (dpct/helper.hpp ~:1186) -- somewhere in their
// call graph:
//   - ggml_backend_sycl_model_load_begin -> ggml_sycl_model_loading_effects(true, outer=true)
//     -> ggml_sycl_info().total_gpu_count and get_unified_cache_for_device() per device.
//   - ggml_backend_sycl_model_load_end -> ggml_sycl_model_loading_effects(false, outer=true)
//     -> ggml_sycl_info().total_gpu_count to reserve a 512 MB compute arena per device
//     -> ggml_sycl_prepare_plan_publication_locked()/publish, which touches per-device caches.
//   - ggml_backend_sycl_activate_model_plan -> the same publish path.
//   - ggml_backend_sycl_model_unloaded_token -> ggml_sycl_teardown_owner_effects()
//     -> `for (int device = 0; device < ggml_sycl_info().device_count; ++device)`.
// None of those four functions is called anywhere below.
//
// Instead this test drives two lower layers directly, both confirmed
// device-free by inspection:
//   1. ggml_sycl::lifecycle::Registry (ggml-sycl/model-lifecycle.hpp,
//      implemented in model-lifecycle.cpp) is the SAME process-wide
//      singleton ggml-sycl.cpp itself uses via
//      ggml_sycl::lifecycle::global_registry() -- model-lifecycle.cpp is one
//      of the .cpp files globbed into the ggml-sycl target this binary
//      links, and it contains zero `sycl::`/`dpct::` references anywhere in
//      the file (grep confirms). begin_outer()/bind_candidate()/end()/
//      unbind_candidate()/teardown() are plain mutex+map bookkeeping. This
//      is the exact same Registry class test-sycl-lifecycle-load-txn.cpp
//      drives standalone, linked with nothing but ggml-base.
//   2. ggml_backend_sycl_register_weight_identity() and
//      ggml_backend_sycl_get_weight_cache_key() (ggml-sycl.cpp) -- read in
//      full for this test: the only calls either function makes are mutex-
//      guarded std::unordered_map lookups (g_sycl_weight_identities_by_name,
//      g_sycl_weight_identities_unowned, g_sycl_gguf_file_ids),
//      registry.acquire_load_effect()/bound_candidate() (layer 1, above),
//      ggml_sycl::dispatch_tuning::ensure_model_loaded() (env-var gated
//      local-file read, no device access -- dispatch-tuning.cpp:356), and
//      sycl_module_mutation_guard (a plain mutex/counter in ggml-sycl.cpp;
//      `grep -n 'class sycl_module_mutation_guard'` locates it).
//      get_weight_cache_key()'s one branch that WOULD touch a device
//      (ggml_backend_sycl_reg()/ggml_backend_reg_dev_get(), guarded by
//      `!extra && tensor->buffer && ...`) is never reached here because
//      every tensor queried below always has tensor->extra set before the
//      query.
//
// Bypassing the four lifecycle-API entry points means no placement plan is
// ever staged or published (g_placement_publication stays null for the
// whole process) and no compute arena is reserved -- this test proves only
// the owner-resolution order inside ggml_backend_sycl_get_weight_cache_key(),
// which is exactly what llama.cpp-s83n is about. Registry::end() alone is
// sufficient to make a model LIVE in the Registry's own bookkeeping
// (registry.find(model_id)), which is everything ggml_sycl_exact_wrapper_owner()
// consults for a nonzero model id.
//
// Mutation control: inside ggml_backend_sycl_get_weight_cache_key()
// (ggml/src/ggml-sycl/ggml-sycl.cpp; `cat ggml/src/ggml-sycl/ggml-sycl.cpp |
// grep -n 'ggml_backend_sycl_get_weight_cache_key'` locates it), make the
// owner resolution ignore extra->model_id, i.e. resolve as if
// extra_model_id were 0. Today that means changing the `owner` assignment
// line (whitespace-aligned in the source; quoted here without the
// alignment):
//     const auto owner = ggml_sycl_exact_wrapper_owner(extra_model_id);
// to
//     const auto owner = ggml_sycl_exact_wrapper_owner(0);
// After llama.cpp-qq19 lands, the same resolution is behind a call to
// ggml_sycl_resolve_tensor_owner() instead, and the mutation is the
// analogous change to that call. Either form always takes the model_id==0
// "published plan" branch (ggml_sycl_identity_owner(ggml_sycl_identity_plan_snapshot())).
// Since this test never publishes any plan, that branch resolves to a
// zeroed ModelToken for every query, which falls through to the (empty)
// g_sycl_weight_identities_unowned map:
//   - Round 1 (shared tensor name registered separately by A and B): BOTH
//     key_a.has_gguf and key_b.has_gguf go false, and both file_offs checks
//     (0x1000, 0x2000) fail -- neither model resolves anything under the
//     mutation, which is the "whichever is not the fallback owner loses its
//     identity" failure this test exists to catch, just simultaneously
//     instead of alternating (there is no "published" round to alternate
//     without going through the device-touching publish path -- see above).
//   - Round 2's first query (model C's tensor queried under B, which never
//     registered it) stays has_gguf=false either way -- it is a coverage
//     check for the false branch, not a regression discriminator, and is
//     not expected to flip.
//   - Round 2's second query (model C's tensor queried under its real owner,
//     model A) DOES flip: has_gguf goes false and file_offs goes to 0 under
//     the mutation, since the mutated owner is zero rather than A's real
//     token.
// This test's file_offs/has_gguf checks in round 1 and round 2's second
// query are therefore genuinely RED under the mutation.

#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/model-lifecycle.hpp"
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
    // test is device-free by construction (see the header), but stay
    // consistent with every other test in this family that links ggml-sycl.
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1", 1);
    }

    using ggml_sycl::lifecycle::error;
    using ggml_sycl::lifecycle::ModelToken;

    const char * shared_name = "zzz_s83n_shared.weight";
    const char * only_a_name = "zzz_s83n_only_a.weight";

    ggml_init_params params{};
    params.mem_size    = 4 * 1024 * 1024;
    params.mem_buffer  = nullptr;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "ggml_init failed");

    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * tensor_c = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    check(tensor_a != nullptr && tensor_b != nullptr && tensor_c != nullptr, "tensor allocation failed");
    ggml_set_name(tensor_a, shared_name);
    ggml_set_name(tensor_b, shared_name);
    ggml_set_name(tensor_c, only_a_name);

    ggml_sycl::lifecycle::Registry & registry = ggml_sycl::lifecycle::global_registry();

    // ---- Model A: a real, committed Registry model, minted and driven
    // entirely through the Registry -- no lifecycle-API device work. Model A
    // registers its own identity for the shared tensor name plus a name
    // model B will never touch.
    const auto begin_a = registry.begin_outer();
    check(begin_a.code == error::OK, "model A begin_outer");
    registry.bind_candidate(begin_a.txn);
    // model_id=0: adopt the load transaction's own token (see the API
    // comment on ggml_backend_sycl_register_weight_identity).
    ggml_backend_sycl_register_weight_identity(tensor_a, /*file_idx=*/0, /*file_offs=*/0x1000, ggml_nbytes(tensor_a),
                                               /*model_id=*/0);
    ggml_backend_sycl_register_weight_identity(tensor_c, /*file_idx=*/0, /*file_offs=*/0x3000, ggml_nbytes(tensor_c),
                                               /*model_id=*/0);
    const auto end_a = registry.end(begin_a.txn, /*success=*/true);
    check(end_a.code == error::OK && end_a.committed, "model A commit");
    registry.unbind_candidate(begin_a.txn);
    const ModelToken token_a = end_a.token;
    check(token_a.model.value != 0, "model A got a real, nonzero model id");

    // ---- Model B: a second, unrelated model over the SAME shared tensor
    // name. It never registers only_a_name.
    const auto begin_b = registry.begin_outer();
    check(begin_b.code == error::OK, "model B begin_outer");
    registry.bind_candidate(begin_b.txn);
    ggml_backend_sycl_register_weight_identity(tensor_b, /*file_idx=*/0, /*file_offs=*/0x2000, ggml_nbytes(tensor_b),
                                               /*model_id=*/0);
    const auto end_b = registry.end(begin_b.txn, /*success=*/true);
    check(end_b.code == error::OK && end_b.committed, "model B commit");
    registry.unbind_candidate(begin_b.txn);
    const ModelToken token_b = end_b.token;
    check(token_b.model.value != 0 && token_b.model.value != token_a.model.value,
          "model B got a real, nonzero, distinct model id");

    // Point each tensor's extra at its own model -- this is the field
    // ggml_backend_sycl_get_weight_cache_key() must consult first.
    ggml_tensor_extra_gpu extra_a{};
    extra_a.model_id = token_a.model.value;
    tensor_a->extra  = &extra_a;

    ggml_tensor_extra_gpu extra_b{};
    extra_b.model_id = token_b.model.value;
    tensor_b->extra  = &extra_b;

    // ---- Round 1: shared tensor name, registered separately by A and B.
    // Neither model was ever "published" (no plan is ever staged or
    // published in this test -- see the header), so this round exercises
    // the exact-owner path for both queries and is the discriminating
    // file_offs case: under the fix each tensor resolves its own identity
    // regardless of the other model's existence.
    {
        ggml_sycl_cache_id key_a = ggml_backend_sycl_get_weight_cache_key(tensor_a, 0);
        ggml_sycl_cache_id key_b = ggml_backend_sycl_get_weight_cache_key(tensor_b, 0);
        print_cache_id("round 1, model A", key_a);
        print_cache_id("round 1, model B", key_b);
        // id.valid is set unconditionally for any non-null tensor in
        // ggml_backend_sycl_get_weight_cache_key() -- a plain-string grep for
        // 'id.valid = true' matches nothing against the vertically-aligned
        // source, so use `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -nE
        // 'id\.valid +='` (14011 as of this writing, of three matches --
        // disambiguate by enclosing function) -- it cannot go false here, so
        // it is printed
        // above for context but not asserted as a counted check.

        check(key_a.has_gguf, "round 1: model A resolves its own GGUF identity, not the UUID fallback");
        check(key_a.file_offs == 0x1000, "round 1: model A resolves its OWN file_offs, not B's or a UUID");

        check(key_b.has_gguf, "round 1: model B resolves its own GGUF identity");
        check(key_b.file_offs == 0x2000, "round 1: model B resolves its OWN file_offs");

        check(key_a.file_offs != key_b.file_offs, "round 1: the two models' identities are not conflated");
    }

    // ---- Round 2: only_a_name was registered ONLY by model A. Querying it
    // under model B's owner must reach the genuine has_gguf==false branch
    // (not just "always true because both sides registered the same name",
    // which round 1 alone cannot rule out). Querying it under model A's own
    // owner afterwards must resolve correctly.
    {
        ggml_tensor_extra_gpu extra_c_as_b{};
        extra_c_as_b.model_id                = token_b.model.value;
        tensor_c->extra                      = &extra_c_as_b;
        ggml_sycl_cache_id key_c_wrong_owner = ggml_backend_sycl_get_weight_cache_key(tensor_c, 0);
        print_cache_id("round 2, only_a_name queried under model B (never registered it)", key_c_wrong_owner);
        check(!key_c_wrong_owner.has_gguf,
              "round 2: model B never registered only_a_name -- has_gguf is reachable-false");

        ggml_tensor_extra_gpu extra_c_as_a{};
        extra_c_as_a.model_id                = token_a.model.value;
        tensor_c->extra                      = &extra_c_as_a;
        ggml_sycl_cache_id key_c_right_owner = ggml_backend_sycl_get_weight_cache_key(tensor_c, 0);
        print_cache_id("round 2, only_a_name queried under model A (its real owner)", key_c_right_owner);
        check(key_c_right_owner.has_gguf, "round 2: model A resolves the identity it actually registered");
        check(key_c_right_owner.file_offs == 0x3000, "round 2: model A resolves its OWN file_offs for only_a_name");
    }

    // Registry hygiene: both synthetic models must tear down cleanly through
    // the Registry itself (no lifecycle-API/device work -- see the header).
    const error teardown_a = registry.teardown(token_a);
    check(teardown_a == error::OK, "model A registry teardown succeeded");
    const error teardown_b = registry.teardown(token_b);
    check(teardown_b == error::OK, "model B registry teardown succeeded");

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
