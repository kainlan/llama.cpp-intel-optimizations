// Owner-scoped canonical-checksum diagnostic (llama.cpp-x3ou).
//
// g_sycl_canonical_checksums is keyed by BARE TENSOR NAME and is never erased,
// so in a process that loads more than one model the second model's capture
// overwrites the first's row under every tensor name the two share -- which,
// for two GGUF models, is most of them. The only reader is the
// GGML_SYCL_UNIFIED_COMPARE diagnostic, which then prints `match=0` for a
// tensor that is byte-for-byte correct: a false corruption verdict from the
// check that exists to detect corruption. Nothing numerical reads the map, so
// the blast radius is diagnostic only -- see the consumer trace in the tracker.
//
// The fix keys the map by ggml_sycl_owner_name_key(owner, name), the same
// scoping g_sycl_weight_identities_by_name already uses, and erases an owner's
// rows in ggml_sycl_erase_weight_identities_for_owner().
//
// Mutation control. The two reverts are NOT equivalent, and an earlier version
// of this comment claimed they were:
//
//   * SYMMETRIC -- make ggml_sycl_canonical_checksum_key() return the bare
//     name, so capture and lookup still agree. The same-owner round trip keeps
//     working and only the cross-model isolation breaks: check 5 fails, alone.
//     (Derived, not yet measured on hardware.)
//   * SINGLE-SITE -- revert only the capture subscript to
//     `std::string(tensor->name)`, leaving the lookup owner-keyed. Now every
//     lookup misses, and checks 1, 2, 3 and 4 fail. Measured 2026-08-09.
//
// The single-site result is why `row::present` exists. Before it, that revert
// left checks 5 and 6 -- the two assertions this test is for -- passing on
// `0 == 0`, so the run reported the map's total failure as four unrelated
// failures and green on the ownership checks. Both now fail too, which is the
// honest reading of a build where nothing is recorded.
//
// The erase-on-teardown half is not observable from here -- a reused slot
// carries a new generation, so a missing erase leaks memory rather than
// answers. That half is the source-contract gate's job, not this test's.
//
// All names are synthetic ("zzz_ckchk_*") and cannot collide with a real GGUF
// tensor name, so nothing below can be satisfied by a real model's capture.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 77;
}
#else

using capture_fn = void (*)(const ggml_tensor *, const void *, size_t);
using lookup_fn  = bool (*)(const ggml_tensor *, size_t *, uint64_t *);

static capture_fn g_capture = nullptr;
static lookup_fn  g_lookup  = nullptr;

static int g_checks   = 0;
static int g_failures = 0;

static void check(bool cond, const char * label) {
    g_checks++;
    if (!cond) {
        g_failures++;
        fprintf(stderr, "FAIL: %s\n", label);
    }
}

// Bare tensors: extra is null, so the capture and the lookup both resolve the
// owner from the published plan -- exactly the path a tensor takes when its
// extra has no model_id yet.
static ggml_tensor * named(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 64);
    ggml_set_name(t, name);
    return t;
}

// A recorded row, carrying presence separately from the value.
//
// `present` is what keeps the equality checks below non-vacuous, and it is here
// because the first version of this test did NOT have it. Returning a bare 0
// for "absent" meant checks 5 and 6 compared 0 against 0 and PASSED on a build
// where every lookup missed -- measured, not hypothesised: reverting only the
// capture-site key made checks 1-4 fail while the two checks the test exists
// for reported green. An absent sentinel that ties with itself is invisible to
// a value-only comparison, so presence is asserted explicitly at every site.
struct row {
    bool     present  = false;
    uint64_t checksum = 0;
};

static row row_of(const ggml_tensor * t) {
    row    r{};
    size_t bytes = 0;
    r.present    = g_lookup(t, &bytes, &r.checksum) && bytes != 0;
    return r;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }
    // Drive the real env gate rather than bypassing it: the substring filter
    // must admit every zzz_ckchk_* name below, and the capture caches its
    // enabled/-filter decision in a function-local static on first call.
    setenv("GGML_SYCL_CANONICAL_CHECKSUM_TENSOR", "zzz_ckchk", 1);

    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
    if (reg == nullptr) {
        fprintf(stderr, "FAIL: no SYCL backend registry\n");
        return 1;
    }

    g_capture = reinterpret_cast<capture_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_capture_canonical_checksum"));
    g_lookup = reinterpret_cast<lookup_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_lookup_canonical_checksum"));
    // A missing hook is a build without the fix, not a reason to skip: a skip
    // here would read as "owner scoping verified" on a binary that has none.
    if (g_capture == nullptr || g_lookup == nullptr) {
        fprintf(stderr, "FAIL: canonical-checksum test hooks absent from this build\n");
        return 1;
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

    // Two payloads of equal length whose contents differ, so a checksum
    // difference can only come from the data and never from the byte count.
    uint8_t payload_a[64];
    uint8_t payload_b[64];
    for (size_t i = 0; i < sizeof(payload_a); ++i) {
        payload_a[i] = static_cast<uint8_t>(0x11 + i);
        payload_b[i] = static_cast<uint8_t>(0xE0 - i);
    }

    ggml_sycl_tensor_inventory inventory{};
    inventory.n_ctx    = 32;
    inventory.n_ubatch = 8;
    ggml_sycl_placement_envelope envelope{};
    envelope.n_ctx     = 32;
    envelope.n_ubatch  = 8;
    envelope.n_seq_max = 1;

    ggml_tensor * solo   = named(ctx, "zzz_ckchk_solo");
    ggml_tensor * shared = named(ctx, "zzz_ckchk_shared");
    ggml_tensor * b_only = named(ctx, "zzz_ckchk_b_only");

    // ---- Model A --------------------------------------------------------
    ggml_sycl_load_txn    load_a{};
    ggml_sycl_model_token model_a{};
    check(ggml_backend_sycl_model_load_begin(&load_a) == GGML_SYCL_LIFECYCLE_OK, "model A lifecycle begin");
    check(ggml_backend_sycl_stage_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
          "model A plan stage");

    // Check 1 (positive control): the harness itself -- capture, then read
    // back through the map -- works. Failing this makes every check below
    // vacuous, so it is asserted before anything about ownership.
    g_capture(solo, payload_a, sizeof(payload_a));
    check(row_of(solo).present, "check 1 (positive control): a capture is readable under its own owner");

    g_capture(shared, payload_a, sizeof(payload_a));
    const row shared_a = row_of(shared);

    // Check 2 (precondition): model A really did record the shared name. With
    // no row here there would be nothing for model B to overwrite and check 5
    // would prove nothing.
    check(shared_a.present, "check 2: model A recorded the shared name (precondition for check 5)");

    check(ggml_backend_sycl_model_load_end(load_a, true, &model_a) == GGML_SYCL_LIFECYCLE_OK,
          "model A lifecycle commit");

    // ---- Model B: a different, unrelated model in the SAME process ------
    ggml_sycl_load_txn    load_b{};
    ggml_sycl_model_token model_b{};
    check(ggml_backend_sycl_model_load_begin(&load_b) == GGML_SYCL_LIFECYCLE_OK, "model B lifecycle begin");
    check(ggml_backend_sycl_stage_inventory_plan(&inventory, &envelope, false) == GGML_SYCL_LIFECYCLE_OK,
          "model B plan stage");

    // Check 3 (negative control): a name only model B ever captures reads back
    // correctly. This stays green under the SYMMETRIC revert -- it shows that
    // mutation is scoped to the cross-model case and does not break capture
    // wholesale. It does NOT survive the single-site revert, which breaks
    // every lookup; that is the point of distinguishing the two.
    g_capture(b_only, payload_b, sizeof(payload_b));
    check(row_of(b_only).present, "check 3 (negative control): model B's own unique name reads back");

    g_capture(shared, payload_b, sizeof(payload_b));
    const row shared_b = row_of(shared);

    // Check 4: model B's capture of the shared name is B's own data. Under
    // symmetric bare-name keying this also passes -- last write wins -- which
    // is exactly why check 5 and not this one is the discriminating assertion.
    check(shared_b.present && shared_b.checksum != shared_a.checksum,
          "check 4: model B's shared-name capture differs from model A's");

    check(ggml_backend_sycl_model_load_end(load_b, true, &model_b) == GGML_SYCL_LIFECYCLE_OK,
          "model B lifecycle commit");

    // ---- A -> B -> A ----------------------------------------------------
    // Check 5 (the bug): reactivating A must read A's OWN row for the shared
    // name. Under bare-name keying model B's capture destroyed it and this
    // returns B's checksum, which is the false `match=0` the diagnostic then
    // reports against model A's correct weights.
    // `present &&` is not decoration: without it an absent row compares equal
    // to another absent row and this reports green on a build recording
    // nothing at all.
    check(ggml_backend_sycl_activate_model_plan(model_a) == GGML_SYCL_LIFECYCLE_OK, "reactivate exact model A");
    const row shared_a_again = row_of(shared);
    check(shared_a_again.present && shared_a_again.checksum == shared_a.checksum,
          "check 5: A->B->reactivate-A reads A's own shared-name checksum, not B's");

    // Check 6: B's row is independent of A's and survived A's reactivation.
    check(ggml_backend_sycl_activate_model_plan(model_b) == GGML_SYCL_LIFECYCLE_OK, "reactivate exact model B");
    const row shared_b_again = row_of(shared);
    check(shared_b_again.present && shared_b_again.checksum == shared_b.checksum,
          "check 6: exact B lookup remains independent of A");

    ggml_free(ctx);
    (void) ggml_backend_sycl_model_unloaded_token(model_b);
    (void) ggml_backend_sycl_model_unloaded_token(model_a);

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif
