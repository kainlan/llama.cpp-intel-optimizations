//
// Discriminator for llama.cpp-81gx (gemma3n wrong answers on B70).
//
// GGML_SYCL_DISABLE_FUSION=1 fixes gemma3n; GGML_SYCL_FUSION_MASK bisection (lead,
// a1edc1599) narrowed it to bit1 -- the RMS_NORM + MUL + ADD 3-way fusion
// (ggml_sycl_op_rms_norm_fused_add, norm.cpp). Four static reads of that kernel found
// no defect. The one axis not yet tested: bit1 IS exercised by other (passing) archs
// via build_norm()'s optional bias parameter (arwkv7.cpp:143, rwkv6qwen2.cpp:106/156),
// but ALWAYS as a broadcasting {ncols} bias vector -- add_nrows == 1 in kernel terms.
// gemma3n's two bit1 call sites (laurel_post_norm, per_layer_proj_norm) both add a
// FULL matching-shape residual instead -- add_nrows == nrows, a non-trivial per-row
// addend. This may be the only place in the passing suite that exercises that branch
// of add_row_ptr's addressing.
//
// This test drives the exact production entry point (a real ggml_cgraph, allocated on
// a real SYCL backend buffer, executed via ggml_backend_graph_compute -- NOT a hand
// call into ggml_sycl_op_rms_norm_fused_add) at gemma3n's two actual ncols values
// (n_embd=64, n_embd_altup=256; both < 1024, so both hit the plain, non-SLM-cached
// rms_norm_mul_add_f32 kernel) for two ADD-operand shapes:
//   Case A (broadcast):     add_src shape {ncols},         add_nrows == 1
//   Case B (full residual): add_src shape {ncols, nrows},  add_nrows == nrows
// and diffs the fused ADD-node output against a host-computed RMS_NORM->MUL->ADD
// reference.
//
// Case A is the positive control: if it fails too, the broadcast-vs-residual framing
// is wrong and the search should go elsewhere (bit1 kernel body after all, or
// cgraph->use_counts accuracy under gemma3n's graph size). Case B is the suspected
// break. Both must be true for the axis to be confirmed.
//
// UPDATE (this ticket, continued): Cases A and B above both PASS (max_err ~2.4e-07 for
// all four), refuting the broadcast-vs-residual axis. That result held: a
// GGML_SYCL_FUSION_MASK use_count instrumentation pass then found bit1 firing on FIVE
// chains in gemma3n, not the two this file's original comment names, and a per-chain
// GGML_SYCL_FUSION_SKIP_MUL bisection isolated the actual culprit to
// `first_prediction_out` (gemma3n.cpp:227-253). Its ADD operand, `slice_rest`, is a
// ggml_view_3d into a graph-internal tensor (`corrected`) at a NON-ZERO byte offset
// (skipping altup index 0) -- an axis Case A/B never varied, since both used plain,
// zero-offset, freshly-allocated operands. Case C below reproduces that exact shape:
// add_src is a view into a larger tensor, offset to its SECOND slice, mirroring
// slice_rest precisely rather than approximating it.
//
// UPDATE 2 (this ticket, continued further): Case C's FIRST version read op_seq=0
// (GGML_SYCL_FUSION_BIT1_REACH_DEBUG) -- it never reached bit1 at all, on either ncols.
// Root cause: `ggml_add(ctx, mul, add_src)` put the VIEW operand at src[1]. ggml's
// post-order graph builder visits src[0] before src[1], and a VIEW is its own
// cgraph->nodes[] entry (not a leaf), so that ordering placed the view node BETWEEN mul
// and the ADD (next_ops=[MUL,VIEW]) -- structurally ineligible before bit1's op-sequence
// check could even look at it. Fixed by matching gemma3n's actual argument order --
// `ggml_add(ctx0, slice_rest, first_prediction)` puts the view FIRST -- which flushes the
// view's (trivial) dependency chain to cgraph->nodes[] before the rms->mul chain is even
// visited, landing RMS_NORM/MUL/ADD consecutively regardless of what precedes them. See
// the comment at Case C's own ggml_add call for the full DFS trace. This was a real,
// separate defect in the test's construction -- not the same bug as gemma3n's, but a
// concrete demonstration of the "numerically identical shape, structurally ineligible"
// trap: building the right operand shape is not sufficient to reach the code that
// operand shape is supposed to exercise.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

static int g_checks   = 0;
static int g_failures = 0;

// Host-side reference: dst[row][col] = rms_scale(row) * x[row][col] * gamma[col] + add[row][col]
// where add[row][col] is add_src[col] (broadcast) if add_nrows==1, else add_src[row*ncols+col].
static void ref_rms_norm_mul_add(const std::vector<float> & x,
                                 const std::vector<float> & gamma,
                                 const std::vector<float> & add_src,
                                 int                        ncols,
                                 int                        nrows,
                                 bool                       broadcast_add,
                                 float                      eps,
                                 std::vector<float> &       out) {
    out.resize((size_t) ncols * nrows);
    for (int row = 0; row < nrows; row++) {
        double sumsq = 0.0;
        for (int col = 0; col < ncols; col++) {
            const double v = x[(size_t) row * ncols + col];
            sumsq += v * v;
        }
        const float scale = 1.0f / std::sqrt((float) (sumsq / ncols) + eps);
        for (int col = 0; col < ncols; col++) {
            const float xi                  = x[(size_t) row * ncols + col];
            const float gi                  = gamma[col];
            const float addend              = broadcast_add ? add_src[col] : add_src[(size_t) row * ncols + col];
            out[(size_t) row * ncols + col] = scale * xi * gi + addend;
        }
    }
}

// Builds RMS_NORM(x) -> MUL(*, gamma) -> ADD(*, add_src) as three consecutive graph
// nodes (ggml's post-order builder emits a single-consumer linear chain consecutively,
// which is exactly what ggml_can_fuse_subgraph's {i,i+1,i+2} adjacency check requires),
// runs it on the real SYCL backend (letting the production fusion-detection loop in
// ggml_backend_sycl_graph_compute_impl pick it up -- GGML_SYCL_FUSION_MASK defaults to
// all-bits-on, so bit1 fires if eligible), and compares the ADD node's device output
// against the host reference.
static bool run_case(ggml_backend_t backend, int ncols, int nrows, bool broadcast_add, const char * label) {
    g_checks++;

    std::mt19937                          rng(1234u + (unsigned) ncols * 7u + (broadcast_add ? 1u : 0u));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_x((size_t) ncols * nrows);
    std::vector<float> h_gamma((size_t) ncols);
    std::vector<float> h_add(broadcast_add ? (size_t) ncols : (size_t) ncols * nrows);

    for (auto & v : h_x) {
        v = dist(rng);
    }
    for (auto & v : h_gamma) {
        v = 0.8f + 0.4f * dist(rng);  // keep gamma away from 0
    }
    for (auto & v : h_add) {
        v = dist(rng);
    }

    const float eps = 1e-6f;

    std::vector<float> h_ref;
    ref_rms_norm_mul_add(h_x, h_gamma, h_add, ncols, nrows, broadcast_add, eps, h_ref);

    struct ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_init failed\n", label);
        g_failures++;
        return false;
    }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_input(x);
    struct ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols);
    ggml_set_input(gamma);
    struct ggml_tensor * add_src = broadcast_add ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols) :
                                                   ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_input(add_src);

    struct ggml_tensor * rms   = ggml_rms_norm(ctx, x, eps);
    struct ggml_tensor * mul   = ggml_mul(ctx, rms, gamma);
    struct ggml_tensor * added = ggml_add(ctx, mul, add_src);
    ggml_set_output(added);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, added);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_backend_alloc_ctx_tensors failed\n", label);
        ggml_free(ctx);
        g_failures++;
        return false;
    }

    ggml_backend_tensor_set(x, h_x.data(), 0, h_x.size() * sizeof(float));
    ggml_backend_tensor_set(gamma, h_gamma.data(), 0, h_gamma.size() * sizeof(float));
    ggml_backend_tensor_set(add_src, h_add.data(), 0, h_add.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    bool                   ok     = (status == GGML_STATUS_SUCCESS);
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_backend_graph_compute status=%d\n", label, (int) status);
    }

    std::vector<float> h_out((size_t) ncols * nrows);
    if (ok) {
        ggml_backend_tensor_get(added, h_out.data(), 0, h_out.size() * sizeof(float));

        float max_err = 0.0f;
        for (size_t i = 0; i < h_out.size(); i++) {
            max_err = std::max(max_err, std::fabs(h_out[i] - h_ref[i]));
        }
        // F32 kernel arithmetic vs a double-accumulated host reference; 1e-3 is loose
        // enough to absorb that and tight enough to catch a wrong add_row_ptr index
        // (which produces an error on the order of the addend's own magnitude, ~O(1)
        // here, not a rounding-sized error).
        const float tol = 1e-3f;
        ok              = max_err < tol;
        std::fprintf(stderr, "  [%s] %s (ncols=%d nrows=%d %s, max_err=%.3e, tol=%.1e)\n", ok ? "PASS" : "FAIL", label,
                     ncols, nrows, broadcast_add ? "broadcast" : "full-residual", max_err, tol);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (!ok) {
        g_failures++;
    }
    return ok;
}

// Case C: reproduces gemma3n's actual first_prediction_out shape (gemma3n.cpp:227-253)
// rather than approximating it. `big` stands in for `corrected` -- a graph-internal 3D
// tensor with 2 "altup-like" slices along its outermost dimension. `add_src` is a VIEW
// into big's SECOND slice only (byte offset = big->nb[2], i.e. skip slice 0), exactly
// mirroring `slice_rest = ggml_view_3d(ctx0, corrected, ..., n_embd*n_tokens*elemsize)`:
// a view at a non-zero offset into a tensor that is itself a fresh compute-node output,
// not a leaf/weight. Cases A and B (above) never varied this axis -- their add_src was
// always either a leaf 1D vector or a leaf 2D tensor at offset 0.
static bool run_case_view_offset(ggml_backend_t backend, int ncols, int nrows, const char * label) {
    g_checks++;

    std::mt19937                          rng(5678u + (unsigned) ncols * 13u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_x((size_t) ncols * nrows);
    std::vector<float> h_gamma((size_t) ncols);
    // Both slices of `big` get random data; only the second slice (index 1) is the
    // addend `slice_rest` actually points at, mirroring corrected's altup index 1.
    std::vector<float> h_big((size_t) ncols * nrows * 2);

    for (auto & v : h_x) {
        v = dist(rng);
    }
    for (auto & v : h_gamma) {
        v = 0.8f + 0.4f * dist(rng);
    }
    for (auto & v : h_big) {
        v = dist(rng);
    }

    const float eps = 1e-6f;

    // Reference addend is big[slice 1] = h_big[ncols*nrows .. 2*ncols*nrows).
    std::vector<float> h_add_slice1(h_big.begin() + (size_t) ncols * nrows, h_big.end());
    std::vector<float> h_ref;
    ref_rms_norm_mul_add(h_x, h_gamma, h_add_slice1, ncols, nrows, /*broadcast_add=*/false, eps, h_ref);

    struct ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_init failed\n", label);
        g_failures++;
        return false;
    }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_input(x);
    struct ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols);
    ggml_set_input(gamma);
    struct ggml_tensor * big = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncols, nrows, 2);
    ggml_set_input(big);
    // View into big's second slice only -- offset = big->nb[2] (the byte stride between
    // slices), the same "skip slice 0" pattern as slice_rest's offset into `corrected`.
    struct ggml_tensor * add_src = ggml_view_2d(ctx, big, ncols, nrows, big->nb[1], big->nb[2]);

    struct ggml_tensor * rms   = ggml_rms_norm(ctx, x, eps);
    struct ggml_tensor * mul   = ggml_mul(ctx, rms, gamma);
    // Operand order matters here in a way it doesn't for Cases A/B, and getting it backwards
    // is exactly why this case originally read op_seq=0 (GGML_SYCL_FUSION_BIT1_REACH_DEBUG,
    // this ticket): ggml_build_forward_expand does a post-order DFS that visits src[0] before
    // src[1], appending each child to cgraph->nodes[] only after ALL of it is visited. add_src
    // is a VIEW node (op=GGML_OP_VIEW, not a leaf), so wherever it sits in the src[] order, it
    // becomes its own entry in cgraph->nodes[] -- the only question is where.
    //   ggml_add(ctx, mul, add_src)  -- add_src is src[1], visited AFTER the whole
    //     rms->mul chain is already appended, landing the VIEW node BETWEEN mul and this ADD:
    //     nodes[] = [..., rms, mul, add_src(VIEW), added]. next_ops=[MUL,VIEW], not
    //     [MUL,ADD] -- bit1's adjacency check fails before it can even look at the operands.
    //   ggml_add(ctx, add_src, mul)  -- add_src is src[0], visited FIRST; its own (trivial,
    //     one-level) view chain resolves and flushes to nodes[] before DFS ever starts on
    //     src[1]. The rms->mul chain, visited second, lands immediately adjacent to `added`
    //     when DFS returns: nodes[] = [..., add_src(VIEW), rms, mul, added]. RMS_NORM, MUL,
    //     ADD are consecutive regardless of what precedes them.
    // gemma3n's real chain (gemma3n.cpp:253) already uses this order --
    // `ggml_add(ctx0, slice_rest, first_prediction)`, view first -- which is why it reaches
    // bit1 in the real graph while this test, built with the arguments the "natural" way
    // round, silently didn't.
    struct ggml_tensor * added = ggml_add(ctx, add_src, mul);
    ggml_set_output(added);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, added);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_backend_alloc_ctx_tensors failed\n", label);
        ggml_free(ctx);
        g_failures++;
        return false;
    }

    ggml_backend_tensor_set(x, h_x.data(), 0, h_x.size() * sizeof(float));
    ggml_backend_tensor_set(gamma, h_gamma.data(), 0, h_gamma.size() * sizeof(float));
    ggml_backend_tensor_set(big, h_big.data(), 0, h_big.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    bool                   ok     = (status == GGML_STATUS_SUCCESS);
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] %s: ggml_backend_graph_compute status=%d\n", label, (int) status);
    }

    std::vector<float> h_out((size_t) ncols * nrows);
    if (ok) {
        ggml_backend_tensor_get(added, h_out.data(), 0, h_out.size() * sizeof(float));

        float max_err = 0.0f;
        for (size_t i = 0; i < h_out.size(); i++) {
            max_err = std::max(max_err, std::fabs(h_out[i] - h_ref[i]));
        }
        const float tol = 1e-3f;
        ok              = max_err < tol;
        std::fprintf(stderr, "  [%s] %s (ncols=%d nrows=%d view-nonzero-offset, max_err=%.3e, tol=%.1e)\n",
                     ok ? "PASS" : "FAIL", label, ncols, nrows, max_err, tol);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (!ok) {
        g_failures++;
    }
    return ok;
}

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        // See test-mem-ops.cpp for why this is 77, not 0 -- a device test that never
        // reached a device has verified nothing, and 0 would make that indistinguishable
        // from a pass. ctest's SKIP_RETURN_CODE reads this as skipped; a bare shell run
        // (setvars.sh never sourced) gets a non-zero status instead of a silent green.
        std::fprintf(stderr,
                     "SKIP: no SYCL GPU devices available -- NO DEVICE WORK WAS PERFORMED.\n"
                     "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return 77;
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP: ggml_backend_sycl_init(0) failed -- NO DEVICE WORK WAS PERFORMED.\n");
        return 77;
    }

    std::fprintf(stderr, "=== bit1 (RMS_NORM+MUL+ADD) broadcast-vs-full-residual discriminator ===\n");
    std::fprintf(stderr, "llama.cpp-81gx: does add_src's broadcast shape change fused-ADD correctness?\n\n");

    // ncols=64 -> gemma3n's laurel_post_norm (n_embd in the test fixture).
    // ncols=256 -> gemma3n's per_layer_proj_norm (n_embd_altup struct default,
    // llama-hparams.h:211 -- the fixture never overrides it). nrows=8 is an arbitrary
    // >1 row count so Case B's add_nrows==nrows is a genuine non-broadcast shape; it
    // does not need to match gemma3n's actual n_tokens/n_layer to exercise the same
    // kernel branch.
    run_case(backend, 64, 8, /*broadcast_add=*/true, "ncols=64 broadcast (positive control)");
    run_case(backend, 64, 8, /*broadcast_add=*/false, "ncols=64 full-residual (suspected)");
    run_case(backend, 256, 8, /*broadcast_add=*/true, "ncols=256 broadcast (positive control)");
    run_case(backend, 256, 8, /*broadcast_add=*/false, "ncols=256 full-residual (suspected)");

    // Case C: gemma3n's actual root cause (first_prediction_out / slice_rest) -- a view
    // at a non-zero offset into a graph-internal tensor. GGML_SYCL_FUSION_BIT1_REACH_DEBUG
    // should now show op_seq=1 and a [FUSION-BIT1-GUARD] line for both ncols (it read
    // op_seq=0 -- never reached bit1 at all -- before the ggml_add argument-order fix; see
    // the file header and the comment at run_case_view_offset's ggml_add call). With the
    // guard (ggml_sycl_fusion_operand_view_offset_safe, ggml-sycl.cpp) intact, expect PASS;
    // with it mutated to unconditionally return true, expect FAIL -- that A/B is what
    // proves this case actually exercises the bug rather than merely reaching the code.
    run_case_view_offset(backend, 64, 8, "ncols=64 view-at-nonzero-offset (root cause)");
    run_case_view_offset(backend, 256, 8, "ncols=256 view-at-nonzero-offset (root cause)");

    ggml_backend_free(backend);

    std::fprintf(stderr, "\n=== %d checks, %d failures ===\n", g_checks, g_failures);

    if (g_failures > 0) {
        std::fprintf(stderr, "SOME CHECKS FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "ALL CHECKS PASSED\n");
    return 0;
}
