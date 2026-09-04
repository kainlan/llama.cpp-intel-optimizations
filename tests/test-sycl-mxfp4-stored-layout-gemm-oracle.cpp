// Host CPU reference oracle for MXFP4 GEMMs on the two device-materialized
// expert weight layouts (llama.cpp-vtfs task 1, "option C" -- owner ruling
// 2026-08-23). Every stored-layout MXFP4 kernel written in this track (G4-G8)
// is scored against the `reference_gemm` + `max_rel_violations` this file
// defines; this test only proves the oracle itself is trustworthy.
//
// What is checked:
//
//   1. Cross-decoder equality. `decode_soa_mxfp4` and `decode_xmx_tiled_mxfp4`
//      are independent re-derivations (from the documented byte formulas
//      below, NOT the production headers -- those are SYCL device kernels
//      and cannot run without a queue) of the two stored layouts. A weight
//      tensor is quantized once via `ggml_quantize_chunk` (AOS block_mxfp4,
//      row-major), materialized both ways with this file's own byte-layout
//      writers, and decoded three independent ways: through decode_soa_mxfp4,
//      through decode_xmx_tiled_mxfp4, and through ggml's own
//      dequantize_row_mxfp4 called directly on the AOS source. All three must
//      be bit-identical -- same table lookup, same E8M0 scale, same order of
//      operations, just different byte addressing. Positive control: a single
//      corrupted byte in each stored buffer must make its decode diverge from
//      the AOS reference, proving the equality check actually reads the bytes
//      it claims to and is not vacuously true.
//
//   2. `reference_gemm(X, W_decoded, M, N, K, act)` computes
//      Y[M,N] = X[M,K] . W[N,K]^T in double, for two activation
//      pre-processing modes: ACT_F16 (round-trip X through fp16, matching a
//      pure-f16 SDPA-style consumer) and ACT_Q8_1 (quantize X row-wise with
//      ggml's own quantize_row_q8_1_ref and reconstruct qs*d in double,
//      matching what an int8 DPAS kernel effectively consumes after its
//      int32-accumulate-then-scale epilogue). Sanity-checked against a plain
//      double dot product (no activation quantization at all) and timed at
//      the GPT-OSS expert shape (K=N=2880) for every M this track's kernels
//      will be scored at.
//
//   3. `max_rel_violations(out, ref, rel_tol, abs_floor)` is the shared
//      scorer every later G4-G8 numerics test calls, for comparing a real
//      device kernel against this oracle at the SAME activation
//      representation (NOT for comparing two different activation
//      representations against each other -- see the Frobenius-norm note on
//      case_reference_gemm_gptoss_shape's own sanity check below, which
//      deliberately does NOT use this scorer). The tolerance contract it is
//      built around: the proven 2-D WOQ GEMM arm's hardware verdict
//      (llama.cpp-sr83 fix cycle 2, team-lead hardware verdict 2026-08-22,
//      documented at ggml/src/ggml-sycl/gemm.hpp near the pp_woq_3d_enabled
//      env latch, echoed at ggml-sycl.cpp's kCompareRelThreshold comment and
//      docs/backend/sycl-env-vars.md's GGML_SYCL_MOE_PP_WOQ_3D row) is clean
//      across every role and block at max_rel <= 0.0258 (WOQ_MAX_REL_TOL
//      below) -- later kernels are held to the same rel_tol bar. A cell only
//      counts as a violation when BOTH its relative error exceeds rel_tol
//      AND its absolute error exceeds abs_floor (default 0.01): a genuine
//      device fp32 epilogue vs this oracle's double reference routinely
//      produces ~1e-6 absolute error that reads as a huge RELATIVE error on
//      a near-zero-magnitude cell -- the exact pathology
//      tests/test-sycl-mxfp4-woq-gemm-bench.cpp's own "FIX CYCLE #4"
//      (its lines ~927-932) documents and fixes with an abs-OR-rel gate;
//      this per-element abs-AND-rel form is the equivalent guard for a
//      per-element violation count rather than a single whole-output
//      pass/fail. A size mismatch between `out` and `ref` is a hard failure
//      (Score.violations == -1, never silently scored via std::min sizes).
//
// Formula sources (read-only; this file reproduces them independently rather
// than including the production headers, which pull in SYCL):
//   - SOA layout ([qs0..qsN][scale0..scaleN], E8M0 scale byte at
//     (ncols/2*nrows) + block_index, block_index = row*n_k_blocks + k_block):
//     ggml/src/ggml-sycl/quants.hpp:190-211
//   - XMX_TILED layout (k-tile-major [tile_k_group][tile_n_group], each group
//     scales[tile_n_total] then qs[tile_n_total][16]) and its inverse map:
//     ggml/src/ggml-sycl/moe-xmx-fused.hpp:118-153,
//     ggml/src/ggml-sycl/moe-tile-convert.cpp:24-173 (the device writer,
//     reorder_mxfp4_aos_to_xmx_tiled, mirrored here by build_xmx_tiled_from_aos),
//     ggml/src/ggml-sycl/convert.cpp:2111-2146 (the documented inverse map
//     comment, mirrored here by decode_xmx_tiled_mxfp4)
//   - E8M0 scale and the e2m1 value table: ggml/src/ggml-impl.h
//     (ggml_e8m0_to_fp32_half), ggml/src/ggml-common.h (kvalues_fp4, aliased
//     kvalues_mxfp4)
//
// Host-only: no SYCL, no device, no model. Runs in a subagent without oneAPI
// sourced. Exit 1 on any failure.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string & name, const std::string & detail) {
    if (ok) {
        std::printf("  PASS  %s%s\n", name.c_str(), detail.empty() ? "" : ("  (" + detail + ")").c_str());
        return;
    }
    std::printf("  FAIL  %s  %s\n", name.c_str(), detail.c_str());
    ++failures;
}

// -----------------------------------------------------------------------------
// Constants shared by both stored layouts.
// -----------------------------------------------------------------------------
constexpr int64_t XMX_K        = QK_MXFP4;   // 32 elements per MXFP4 block
constexpr int64_t PACKED_BYTES = XMX_K / 2;  // 16 nibble-packed bytes per block

// The tolerance every downstream stored-layout MXFP4 kernel is scored
// against: the proven 2-D WOQ GEMM arm's hardware verdict (llama.cpp-sr83
// fix cycle 2, team-lead hardware verdict 2026-08-22) -- clean across every
// role and block at max_rel <= 0.0258. Documented at
// ggml/src/ggml-sycl/gemm.hpp near the pp_woq_3d_enabled env latch, echoed
// at ggml-sycl.cpp's kCompareRelThreshold comment and
// docs/backend/sycl-env-vars.md's GGML_SYCL_MOE_PP_WOQ_3D row (NOT
// tests/test-sycl-mxfp4-woq-gemm-bench.cpp, whose own pass criterion uses
// abs_tol=0.01/rel_tol=0.02 and never states 0.0258 -- an earlier version of
// this file cited that test file for the value; it does not appear there).
constexpr double WOQ_MAX_REL_TOL = 0.0258;

// The absolute-error floor paired with WOQ_MAX_REL_TOL in max_rel_violations:
// a cell only counts as a violation when its relative error exceeds
// WOQ_MAX_REL_TOL AND its absolute error exceeds this floor -- excludes the
// near-zero-cell relative-error artifact fix-cycle-4 documents (see the file
// header and max_rel_violations below).
constexpr double DEFAULT_ABS_FLOOR = 0.01;

// -----------------------------------------------------------------------------
// Layout writers: build each stored byte layout from an AOS block_mxfp4 array
// laid out row-major (block_index = row * n_k_blocks + k_block -- the layout
// ggml_quantize_chunk produces). These mirror, byte for byte, what the SYCL
// device kernels write (moe-tile-convert.cpp's reorder_mxfp4_aos_to_xmx_tiled
// for XMX_TILED; the SOA reorder in convert.cpp for SOA), reproduced here from
// the documented formulas because the device kernels themselves need a queue.
// -----------------------------------------------------------------------------

// SOA: [qs0..qsN][scale0..scaleN] -- quants.hpp:190-211.
std::vector<uint8_t> build_soa_from_aos(const std::vector<block_mxfp4> & aos, int64_t nrows, int64_t ncols) {
    GGML_ASSERT(nrows > 0 && ncols > 0 && ncols % XMX_K == 0);
    const int64_t        n_k_blocks = ncols / XMX_K;
    const int64_t        qs_bytes   = (ncols / 2) * nrows;
    std::vector<uint8_t> out((size_t) (qs_bytes + nrows * n_k_blocks), 0);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
            const int64_t       block_index = row * n_k_blocks + kb;
            const block_mxfp4 & b           = aos[(size_t) block_index];
            std::memcpy(out.data() + block_index * PACKED_BYTES, b.qs, (size_t) PACKED_BYTES);
            out[(size_t) (qs_bytes + block_index)] = b.e;
        }
    }
    return out;
}

// XMX_TILED: k-tile-major [tile_k_group][tile_n_group], each group
// scales[tile_n_total] then qs[tile_n_total][16] -- moe-xmx-fused.hpp:118-153,
// moe-tile-convert.cpp:24-173 (reorder_mxfp4_aos_to_xmx_tiled).
std::vector<uint8_t> build_xmx_tiled_from_aos(const std::vector<block_mxfp4> & aos,
                                              int64_t                          nrows,
                                              int64_t                          ncols,
                                              int64_t                          tile_n_total) {
    GGML_ASSERT(nrows > 0 && ncols > 0 && ncols % XMX_K == 0 && tile_n_total > 0);
    const int64_t        n_k_blocks      = ncols / XMX_K;
    const int64_t        n_tile_groups_n = (nrows + tile_n_total - 1) / tile_n_total;
    const int64_t        group_bytes     = tile_n_total * (1 + PACKED_BYTES);
    std::vector<uint8_t> out((size_t) (n_k_blocks * n_tile_groups_n * group_bytes), 0);
    for (int64_t n = 0; n < nrows; ++n) {
        const int64_t tg_n = n / tile_n_total;
        const int64_t tn   = n - tg_n * tile_n_total;
        for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
            const int64_t       block_index   = n * n_k_blocks + kb;
            const block_mxfp4 & b             = aos[(size_t) block_index];
            const int64_t       group_offset  = (kb * n_tile_groups_n + tg_n) * group_bytes;
            out[(size_t) (group_offset + tn)] = b.e;
            std::memcpy(out.data() + group_offset + tile_n_total + tn * PACKED_BYTES, b.qs, (size_t) PACKED_BYTES);
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Layout readers (decoders): independent re-derivations of the same
// documented formulas. These do NOT share code with the builders above -- the
// builder indexes forward (row, k_block) -> byte offset; the reader derives
// (row, k_block) from a byte offset via the inverse map in
// convert.cpp:2111-2146 -- so a formula error in one is unlikely to cancel
// against the other within the same equality check.
// -----------------------------------------------------------------------------

std::vector<float> decode_soa_mxfp4(const uint8_t * buf, int64_t nrows, int64_t ncols) {
    GGML_ASSERT(nrows > 0 && ncols > 0 && ncols % XMX_K == 0);
    const int64_t      n_k_blocks = ncols / XMX_K;
    const int64_t      qs_bytes   = (ncols / 2) * nrows;
    std::vector<float> out((size_t) (nrows * ncols));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
            const int64_t   block_index = row * n_k_blocks + kb;
            const uint8_t * qs          = buf + block_index * PACKED_BYTES;
            const uint8_t   e           = buf[qs_bytes + block_index];
            const float     d           = GGML_E8M0_TO_FP32_HALF(e);
            for (int64_t j = 0; j < PACKED_BYTES; ++j) {
                const int8_t x0                                             = kvalues_mxfp4[qs[j] & 0x0F];
                const int8_t x1                                             = kvalues_mxfp4[qs[j] >> 4];
                out[(size_t) (row * ncols + kb * XMX_K + j)]                = x0 * d;
                out[(size_t) (row * ncols + kb * XMX_K + j + PACKED_BYTES)] = x1 * d;
            }
        }
    }
    return out;
}

std::vector<float> decode_xmx_tiled_mxfp4(const uint8_t * buf, int64_t nrows, int64_t ncols, int64_t tile_n_total) {
    GGML_ASSERT(nrows > 0 && ncols > 0 && ncols % XMX_K == 0 && tile_n_total > 0);
    const int64_t      n_tile_groups_n = (nrows + tile_n_total - 1) / tile_n_total;
    const int64_t      group_bytes     = tile_n_total * (1 + PACKED_BYTES);
    std::vector<float> out((size_t) (nrows * ncols));
    for (int64_t n = 0; n < nrows; ++n) {
        const int64_t tg_n = n / tile_n_total;
        const int64_t tn   = n - tg_n * tile_n_total;
        for (int64_t k = 0; k < ncols; ++k) {
            const int64_t k_block         = k / XMX_K;
            const int64_t k_local         = k - k_block * XMX_K;
            const int64_t group_offset    = (k_block * n_tile_groups_n + tg_n) * group_bytes;
            const uint8_t e               = buf[group_offset + tn];
            const float   d               = GGML_E8M0_TO_FP32_HALF(e);
            const int64_t qs_local        = (k_local < XMX_K / 2) ? k_local : (k_local - XMX_K / 2);
            const uint8_t qs_byte         = buf[group_offset + tile_n_total + tn * PACKED_BYTES + qs_local];
            // nibble_idx is a TABLE INDEX (0-15), not a looked-up value -- unlike
            // x0/x1 in decode_soa_mxfp4 above, which hold the looked-up e2m1 value.
            const int     nibble_idx      = (k_local < XMX_K / 2) ? (qs_byte & 0x0F) : (qs_byte >> 4);
            out[(size_t) (n * ncols + k)] = kvalues_mxfp4[nibble_idx] * d;
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Reference GEMM and the shared scorer.
// -----------------------------------------------------------------------------

enum class Activation { ACT_F16, ACT_Q8_1 };

// Round X through the same lossy path a device consumer would apply, in
// double: fp16 round-trip for ACT_F16, ggml's own q8_1 quantize/dequantize
// for ACT_Q8_1 (quantize_row_q8_1_ref, matching what the int8 DPAS path
// consumes after its int32-accumulate-then-scale epilogue).
std::vector<double> preprocess_activation(const std::vector<float> & X, int64_t M, int64_t K, Activation act) {
    std::vector<double> Xd((size_t) (M * K));
    if (act == Activation::ACT_F16) {
        for (int64_t i = 0; i < M * K; ++i) {
            Xd[(size_t) i] = (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(X[(size_t) i]));
        }
        return Xd;
    }

    GGML_ASSERT(K % QK8_1 == 0);
    const int64_t           nb = K / QK8_1;
    std::vector<block_q8_1> qX((size_t) (M * nb));
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_1_ref(X.data() + m * K, qX.data() + (size_t) (m * nb), K);
    }
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t b = 0; b < nb; ++b) {
            const block_q8_1 & blk = qX[(size_t) (m * nb + b)];
            const double       d   = (double) ggml_fp16_to_fp32(blk.d);
            for (int64_t j = 0; j < QK8_1; ++j) {
                Xd[(size_t) (m * K + b * QK8_1 + j)] = d * (double) blk.qs[j];
            }
        }
    }
    return Xd;
}

// Y[M,N] = X[M,K] . W[N,K]^T in double. Plain loops, no OpenMP -- keeps
// runtime bounded and simple enough that a second bug can't hide behind a
// fused/tiled formulation. `W_decoded` is N*K row-major float, as produced by
// decode_soa_mxfp4 / decode_xmx_tiled_mxfp4.
std::vector<double> reference_gemm(const std::vector<float> & X,
                                   const std::vector<float> & W_decoded,
                                   int64_t                    M,
                                   int64_t                    N,
                                   int64_t                    K,
                                   Activation                 act) {
    const std::vector<double> Xd = preprocess_activation(X, M, K, act);

    std::vector<double> Y((size_t) (M * N), 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const double * xrow = &Xd[(size_t) (m * K)];
        for (int64_t n = 0; n < N; ++n) {
            const float * wrow = &W_decoded[(size_t) (n * K)];
            double        sum  = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += xrow[k] * (double) wrow[k];
            }
            Y[(size_t) (m * N + n)] = sum;
        }
    }
    return Y;
}

// The exact double dot product with NO activation quantization at all -- used
// only as an internal sanity cross-check in this file, not part of the public
// oracle contract (real kernels always quantize or round the activation).
std::vector<double> exact_gemm_double(const std::vector<float> & X,
                                      const std::vector<float> & W_decoded,
                                      int64_t                    M,
                                      int64_t                    N,
                                      int64_t                    K) {
    std::vector<double> Y((size_t) (M * N), 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const float * xrow = &X[(size_t) (m * K)];
        for (int64_t n = 0; n < N; ++n) {
            const float * wrow = &W_decoded[(size_t) (n * K)];
            double        sum  = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += (double) xrow[k] * (double) wrow[k];
            }
            Y[(size_t) (m * N + n)] = sum;
        }
    }
    return Y;
}

struct Score {
    // -1 is the reserved SIZE-MISMATCH sentinel (see max_rel_violations):
    // `out.size() != ref.size()` is a hard failure, never silently scored
    // over the shorter of the two. Any real count is >= 0, so a caller that
    // naively checks `violations == 0` for "pass" still correctly rejects
    // this sentinel; callers that want to distinguish "no violations" from
    // "could not be scored" should check `violations < 0` explicitly. On the
    // sentinel path max_rel and max_abs_diff are +infinity (not 0.0), so a
    // caller that instead checks `max_rel <= tol` also fails closed on a
    // size mismatch rather than reading it as a perfect match.
    //
    // `violations` is the VERDICT (rel_tol AND abs_floor both required, see
    // max_rel_violations below); `max_rel` and `max_abs_diff` are NOT
    // filtered by abs_floor -- they are the whole-output extrema computed
    // over every cell, including ones the floor exempts from `violations`.
    // So `violations == 0 && max_rel == 5.0` is a normal, expected reading
    // (a near-zero-magnitude cell can carry a huge relative error that the
    // floor correctly declined to count): read `violations` as the pass/fail
    // signal, and `max_rel`/`max_abs_diff` only as diagnostic context.
    int64_t violations;
    double  max_rel;
    double  max_abs_diff;
};

// The shared scorer every later G4-G8 numerics test calls: counts elements
// whose relative error against `ref` exceeds `rel_tol` AND whose absolute
// error exceeds `abs_floor`. Both conditions are required (not either alone)
// so a near-zero-magnitude reference cell -- where a genuine, tiny absolute
// error reads as an enormous relative error -- is not flagged; see the
// tolerance-contract comment on WOQ_MAX_REL_TOL and the file header for the
// fix-cycle-4 precedent this mirrors. The returned `max_rel`/`max_abs_diff`
// are the UNFILTERED whole-output extrema (see the Score comment) -- only
// `violations` reflects the abs_floor gate. `out.size() != ref.size()` is a
// hard failure: returns Score{-1, +inf, +inf} rather than silently scoring
// the shorter of the two via std::min; the infinities make a naive
// `max_rel <= tol` check fail closed too, not just a `violations == 0` check.
Score max_rel_violations(const std::vector<double> & out,
                         const std::vector<double> & ref,
                         double                      rel_tol,
                         double                      abs_floor = DEFAULT_ABS_FLOOR) {
    if (out.size() != ref.size()) {
        // Flush stdout first: PASS/FAIL lines from `check()` go to stdout,
        // which is block-buffered when captured/piped while stderr is not --
        // without this the SIZE MISMATCH line can print before earlier PASS
        // lines that logically preceded it (this repo has been bitten by
        // stdout buffering scrambling test-result attribution before).
        std::fflush(stdout);
        std::fprintf(stderr,
                     "max_rel_violations: SIZE MISMATCH out.size()=%zu ref.size()=%zu -- refusing to score, "
                     "returning the -1 sentinel\n",
                     out.size(), ref.size());
        const double inf = std::numeric_limits<double>::infinity();
        return Score{ -1, inf, inf };
    }
    Score s{ 0, 0.0, 0.0 };
    for (size_t i = 0; i < out.size(); ++i) {
        const double diff = std::fabs(out[i] - ref[i]);
        const double rel  = diff / std::max(1e-12, std::fabs(ref[i]));
        s.max_abs_diff    = std::max(s.max_abs_diff, diff);
        s.max_rel         = std::max(s.max_rel, rel);
        if (rel > rel_tol && diff > abs_floor) {
            ++s.violations;
        }
    }
    return s;
}

// -----------------------------------------------------------------------------
// Small helpers shared by the cases below (each was previously duplicated
// 2-3 times inline; factored out so the shape a downstream G4-G8 test copies
// is the helper, not a repeated loop body).
// -----------------------------------------------------------------------------

// Element count where a[i] != b[i]. A size mismatch returns 0 without
// indexing out of bounds rather than crashing -- the caller's own size check
// (run separately, before this is called) is what turns a wrong-sized buffer
// into a visible FAIL; this helper's only job is to not crash on it.
int64_t count_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return 0;
    }
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++n;
        }
    }
    return n;
}

bool all_finite(const std::vector<double> & v) {
    for (double x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// sqrt(sum((a-b)^2) / sum(b^2)): a whole-output relative error dominated by
// the large-magnitude cells, not fooled by near-zero cells the way a
// per-element relative metric is (see max_rel_violations' scoping note and
// case_reference_gemm_gptoss_shape below for why that matters here).
double frobenius_rel(const std::vector<double> & a, const std::vector<double> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        num += diff * diff;
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

// -----------------------------------------------------------------------------
// Cases
// -----------------------------------------------------------------------------

// Cross-decoder equality: three independent decodes of the SAME quantized
// weights must be bit-identical. Positive control at the end proves the check
// is not vacuously true.
bool case_cross_decoder_equality() {
    // Snapshotted rather than re-deriving `soa_mismatches == 0 && ... ` by
    // hand at the return: a fifth check added to this function and forgotten
    // in a hand-written return expression would silently narrow the
    // main-level gate while the PASS line still printed.
    const int entry_failures = failures;

    const int64_t nrows        = 44;   // deliberately not a multiple of tile_n_total (16):
    const int64_t ncols        = 128;  // exercises the XMX_TILED per-group tail
    const int64_t tile_n_total = 16;   // caps.N=16, optimal_tiles_n=1 (convert.cpp:2111-2146 comment)

    std::mt19937                          rng(20260904);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float>                    weight((size_t) (nrows * ncols));
    for (float & v : weight) {
        v = dist(rng);
    }

    const int64_t            n_k_blocks = ncols / XMX_K;
    std::vector<block_mxfp4> aos((size_t) (nrows * n_k_blocks));
    ggml_quantize_chunk(GGML_TYPE_MXFP4, weight.data(), aos.data(), 0, nrows, ncols, nullptr);

    // Third, independent decode: ggml's own dequantize_row_mxfp4 directly on
    // the AOS source, one row at a time.
    std::vector<float> ref((size_t) (nrows * ncols));
    for (int64_t row = 0; row < nrows; ++row) {
        dequantize_row_mxfp4(aos.data() + row * n_k_blocks, ref.data() + row * ncols, ncols);
    }

    const std::vector<uint8_t> soa       = build_soa_from_aos(aos, nrows, ncols);
    const std::vector<uint8_t> xmx_tiled = build_xmx_tiled_from_aos(aos, nrows, ncols, tile_n_total);

    const std::vector<float> soa_decoded       = decode_soa_mxfp4(soa.data(), nrows, ncols);
    const std::vector<float> xmx_tiled_decoded = decode_xmx_tiled_mxfp4(xmx_tiled.data(), nrows, ncols, tile_n_total);

    // Sizes are checked BEFORE any element access below: a decoder that
    // returns the wrong size (e.g. an empty stub) must report a clean FAIL
    // here rather than reading out of bounds two lines later.
    const bool sizes_ok = soa.size() > 0 && xmx_tiled.size() > 0 && soa_decoded.size() == ref.size() &&
                          xmx_tiled_decoded.size() == ref.size();
    check(sizes_ok, "decoders-produce-nonempty-correctly-sized-output",
          "soa_bytes=" + std::to_string(soa.size()) + " xmx_bytes=" + std::to_string(xmx_tiled.size()) +
              " soa_decoded=" + std::to_string(soa_decoded.size()) +
              " xmx_decoded=" + std::to_string(xmx_tiled_decoded.size()) + " expected=" + std::to_string(ref.size()));
    if (!sizes_ok) {
        return failures == entry_failures;
    }

    const int64_t soa_mismatches = count_mismatches(soa_decoded, ref);
    const int64_t xmx_mismatches = count_mismatches(xmx_tiled_decoded, ref);
    check(soa_mismatches == 0, "soa-decode-matches-dequantize_row_mxfp4",
          "mismatches=" + std::to_string(soa_mismatches) + "/" + std::to_string(ref.size()));
    check(xmx_mismatches == 0, "xmx_tiled-decode-matches-dequantize_row_mxfp4",
          "mismatches=" + std::to_string(xmx_mismatches) + "/" + std::to_string(ref.size()));

    // Positive control: corrupt one qs byte in each stored buffer and require
    // the decode to diverge from the AOS reference -- proves the equality
    // checks above actually read the bytes they claim to.
    std::vector<uint8_t> soa_corrupt = soa;
    soa_corrupt[0] ^= 0xFF;
    const std::vector<float> soa_corrupt_decoded = decode_soa_mxfp4(soa_corrupt.data(), nrows, ncols);
    const int64_t            soa_corrupt_diffs   = count_mismatches(soa_corrupt_decoded, ref);
    check(soa_corrupt_diffs > 0, "corrupted-soa-byte-diverges-from-reference (positive control)",
          "diffs=" + std::to_string(soa_corrupt_diffs));

    // nrows (44) is deliberately not a multiple of tile_n_total (16), so the
    // buffer's tail bytes are padding for out-of-range rows that no decode
    // ever reads (build_xmx_tiled_from_aos only writes real data for
    // n < nrows). Corrupting the literal last byte would corrupt padding and
    // this control would vacuously "pass" with zero diffs -- pick a qs byte
    // that belongs to a REAL row instead, via the same group-address formula
    // the builder used, so the corruption is guaranteed to be read back.
    std::vector<uint8_t> xmx_corrupt             = xmx_tiled;
    const int64_t        corrupt_n_tile_groups_n = (nrows + tile_n_total - 1) / tile_n_total;
    const int64_t        corrupt_group_bytes     = tile_n_total * (1 + PACKED_BYTES);
    const int64_t        corrupt_row             = nrows - 1;
    const int64_t        corrupt_kb              = n_k_blocks - 1;
    const int64_t        corrupt_tg_n            = corrupt_row / tile_n_total;
    const int64_t        corrupt_tn              = corrupt_row - corrupt_tg_n * tile_n_total;
    const int64_t corrupt_group_off = (corrupt_kb * corrupt_n_tile_groups_n + corrupt_tg_n) * corrupt_group_bytes;
    const int64_t corrupt_byte_off  = corrupt_group_off + tile_n_total + corrupt_tn * PACKED_BYTES;
    xmx_corrupt[(size_t) corrupt_byte_off] ^= 0xFF;
    const std::vector<float> xmx_corrupt_decoded =
        decode_xmx_tiled_mxfp4(xmx_corrupt.data(), nrows, ncols, tile_n_total);
    const int64_t xmx_corrupt_diffs = count_mismatches(xmx_corrupt_decoded, ref);
    check(xmx_corrupt_diffs > 0, "corrupted-xmx_tiled-byte-diverges-from-reference (positive control)",
          "diffs=" + std::to_string(xmx_corrupt_diffs));

    return failures == entry_failures;
}

// max_rel_violations sanity: identical vectors score zero; a single
// out-of-tolerance element is caught.
void case_scorer_sanity() {
    std::vector<double> ref       = { 1.0, 2.0, -3.0, 4.0, 0.0 };
    std::vector<double> same      = ref;
    const Score         identical = max_rel_violations(same, ref, WOQ_MAX_REL_TOL);
    check(identical.violations == 0 && identical.max_rel == 0.0, "scorer-zero-violations-on-identical-vectors",
          "violations=" + std::to_string(identical.violations) + " max_rel=" + std::to_string(identical.max_rel));

    std::vector<double> perturbed = ref;
    perturbed[2] *= 1.10;  // 10% deviation on a nonzero element, above the 2.58% tolerance
    const Score deviated = max_rel_violations(perturbed, ref, WOQ_MAX_REL_TOL);
    check(deviated.violations == 1, "scorer-catches-out-of-tolerance-element",
          "violations=" + std::to_string(deviated.violations) + " max_rel=" + std::to_string(deviated.max_rel));

    // abs_floor semantics (team-lead finding, fix-cycle-4 precedent): a
    // near-zero reference cell perturbed by a small ABSOLUTE amount (5e-3,
    // under the default 0.01 floor) must NOT be flagged even though its
    // RELATIVE error is enormous -- that is device-epilogue-vs-double-
    // reference noise, not a kernel bug. A normal-magnitude cell (30.0)
    // perturbed by a genuine 5% must still be flagged. One call, one
    // expected violation, so the abs_floor is proven to suppress exactly
    // the near-zero cell and nothing else.
    std::vector<double> ref_floor   = { 0.001, 30.0 };
    std::vector<double> out_floor   = { 0.001 + 5e-3, 30.0 * 1.05 };
    const Score         floor_score = max_rel_violations(out_floor, ref_floor, WOQ_MAX_REL_TOL);
    check(floor_score.violations == 1, "scorer-abs-floor-excludes-near-zero-noise-but-flags-a-real-5pct-deviation",
          "violations=" + std::to_string(floor_score.violations) + " max_rel=" + std::to_string(floor_score.max_rel));

    // Size mismatch is a hard failure, not a silent std::min(out, ref): all
    // three fields must fail closed, not just `violations`, so a caller
    // checking `max_rel <= tol` instead of `violations == 0` also rejects it.
    std::vector<double> short_vec = { 1.0, 2.0 };
    const Score         mismatch  = max_rel_violations(short_vec, ref, WOQ_MAX_REL_TOL);
    check(mismatch.violations == -1 && std::isinf(mismatch.max_rel) && std::isinf(mismatch.max_abs_diff),
          "scorer-hard-fails-on-size-mismatch-sentinel",
          "violations=" + std::to_string(mismatch.violations) + " max_rel=" + std::to_string(mismatch.max_rel) +
              " max_abs_diff=" + std::to_string(mismatch.max_abs_diff) +
              " (out.size=" + std::to_string(short_vec.size()) + " ref.size=" + std::to_string(ref.size()) + ")");
}

// A hand-checkable GEMM: one MXFP4 block (K=32), weight values chosen from
// the e2m1 table directly (so the decoded float is exact), activation all
// ones so Y == sum(decoded weight row). ACT_F16 must reproduce this exactly
// (1.0 round-trips through fp16 losslessly); ACT_Q8_1's expected tolerance is
// explained where it is checked below, not assumed away here.
void case_reference_gemm_hand_checked() {
    const int64_t            nrows = 1, ncols = XMX_K;  // K=32, one MXFP4 block, N=1
    // e2m1 table indices 0,1,2,3 (values 0,1,2,3) repeated to fill 32 elements,
    // all with E8M0 scale byte 127 (2^(127-127)/2 == 0.5, GGML_E8M0_TO_FP32_HALF's "half" convention).
    std::vector<block_mxfp4> aos(1);
    aos[0].e = 127;
    for (int j = 0; j < (int) PACKED_BYTES; ++j) {
        aos[0].qs[j] = 0x21;  // low nibble=1 (value 1), high nibble=2 (value 2)
    }

    std::vector<float> ref_row(ncols);
    dequantize_row_mxfp4(aos.data(), ref_row.data(), ncols);
    double expected_sum = 0.0;
    for (float v : ref_row) {
        expected_sum += (double) v;
    }

    const std::vector<uint8_t> soa         = build_soa_from_aos(aos, nrows, ncols);
    const std::vector<float>   soa_decoded = decode_soa_mxfp4(soa.data(), nrows, ncols);

    std::vector<float> X((size_t) ncols, 1.0f);  // activation: all ones

    const std::vector<double> y_f16 = reference_gemm(X, soa_decoded, 1, 1, ncols, Activation::ACT_F16);
    check(y_f16.size() == 1 && std::fabs(y_f16[0] - expected_sum) < 1e-9,
          "reference_gemm-f16-matches-hand-computed-sum",
          "got=" + std::to_string(y_f16[0]) + " expected=" + std::to_string(expected_sum));

    const std::vector<double> y_q8_1         = reference_gemm(X, soa_decoded, 1, 1, ncols, Activation::ACT_Q8_1);
    // A constant activation row has EXACT int8 codes under quantize_row_q8_1_ref
    // (id = 127/amax reconstructs qs[j]=127 for every j with zero rounding),
    // but the row's scale `d = amax/127` is stored as GGML_FP32_TO_FP16(d) --
    // ~11 mantissa bits -- so a small FP16-rounding error on the SCALE, not
    // the codes, survives into the reconstructed sum. Bound generously at 3
    // FP16 ULPs relative (~3 * 2^-11 =~ 1.5e-3) rather than assuming exact.
    const double              fp16_scale_tol = 1.5e-3 * std::fabs(expected_sum);
    check(y_q8_1.size() == 1 && std::fabs(y_q8_1[0] - expected_sum) < fp16_scale_tol,
          "reference_gemm-q8_1-matches-hand-computed-sum-within-fp16-scale-rounding",
          "got=" + std::to_string(y_q8_1[0]) + " expected=" + std::to_string(expected_sum) +
              " tol=" + std::to_string(fp16_scale_tol));
}

// reference_gemm at the GPT-OSS expert shape (K=N=2880) for every M this
// track's kernels will be scored at, ACT_Q8_1 (the int8 DPAS path's
// activation format). Checks output shape, finiteness, agreement with the
// no-quantization exact double GEMM within a generous sanity bound, and the
// mandatory <10s bound at M=512.
void case_reference_gemm_gptoss_shape() {
    constexpr int64_t          K = 2880, N = 2880;
    const std::vector<int64_t> Ms = { 1, 2, 4, 8, 32, 128, 512 };

    std::mt19937                          rng(20260904);
    std::uniform_real_distribution<float> wdist(-2.0f, 2.0f);
    std::vector<float>                    weight((size_t) (N * K));
    for (float & v : weight) {
        v = wdist(rng);
    }

    const int64_t            n_k_blocks = K / XMX_K;
    std::vector<block_mxfp4> aos((size_t) (N * n_k_blocks));
    ggml_quantize_chunk(GGML_TYPE_MXFP4, weight.data(), aos.data(), 0, N, K, nullptr);

    const std::vector<uint8_t> soa = build_soa_from_aos(aos, N, K);
    const std::vector<float>   Wd  = decode_soa_mxfp4(soa.data(), N, K);
    // decode_soa_mxfp4 constructs `out` at exactly `nrows * ncols` unconditionally
    // (see :194 above) -- a real assertion on that invariant, not a check() that
    // could never fire and inflate the PASS count.
    GGML_ASSERT(Wd.size() == (size_t) (N * K));

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // The no-quantization exact double GEMM (used only for the Frobenius
    // sanity checks below, never for the timed gate) is run at M in
    // {1, 32, 512} only, not the full sweep: its frob_rel is flat across M
    // (measured ~0.0038 for q8_1, ~0.00018 for f16 at every M tried), so the
    // other four points re-measure a constant while costing real wall time
    // (exact_gemm_double's own M=512 pass is ~2.5s) -- restricting it keeps
    // the binary well clear of the 10s bar that gates reference_gemm at
    // M=512 specifically (unaffected by this restriction; see below).
    auto needs_exact_check = [](int64_t M) {
        return M == 1 || M == 32 || M == 512;
    };

    for (int64_t M : Ms) {
        std::vector<float> X((size_t) (M * K));
        for (float & v : X) {
            v = xdist(rng);
        }

        const auto                t0      = std::chrono::steady_clock::now();
        const std::vector<double> Y       = reference_gemm(X, Wd, M, N, K, Activation::ACT_Q8_1);
        const auto                t1      = std::chrono::steady_clock::now();
        const double              elapsed = std::chrono::duration<double>(t1 - t0).count();

        // reference_gemm always returns exactly M*N elements (:283 above) --
        // a real assertion, not a check() that could never fire.
        GGML_ASSERT(Y.size() == (size_t) (M * N));
        check(all_finite(Y), "reference_gemm-output-finite M=" + std::to_string(M), "");

        if (M == 512) {
            // This is THE criterion: reference_gemm itself, timed, at the
            // GPT-OSS M=512 shape, independent of whether the exact-GEMM
            // sanity cross-check below runs for this M.
            check(elapsed < 10.0, "reference_gemm-M512-K2880-N2880-under-10s (timed criterion arm)",
                  "elapsed=" + std::to_string(elapsed) + "s");
        }

        if (!needs_exact_check(M)) {
            std::printf("    reference_gemm M=%lld N=%lld K=%lld act=q8_1: %.3fs (exact cross-check skipped)\n",
                        (long long) M, (long long) N, (long long) K, elapsed);
            continue;
        }

        // Sanity vs the no-quantization exact double GEMM, scored with a
        // Frobenius-norm relative error rather than max_rel_violations.
        // max_rel_violations is a PER-ELEMENT relative metric and is the
        // wrong tool here on purpose: at K=2880 with random signed terms,
        // many output cells sit near zero from cancellation, and q8_1
        // activation quantization applies a roughly fixed per-row noise
        // vector that dots to a large RELATIVE (but small absolute) error
        // on exactly those near-zero cells -- test-sycl-mxfp4-woq-gemm-bench.cpp
        // hits the identical issue and documents why its own oracle gate is
        // abs_tol-OR-rel_tol, never rel_tol alone, for this reason. A
        // whole-output Frobenius ratio is dominated by the large-magnitude
        // cells and is not fooled by the near-zero ones, so it is what this
        // sanity check uses; max_rel_violations remains reserved, as
        // documented, for comparing two encodings of the SAME activation
        // representation (a real kernel vs this oracle), where it is the
        // right tool.
        const std::vector<double> exact    = exact_gemm_double(X, Wd, M, N, K);
        const double              frob_rel = frobenius_rel(Y, exact);
        check(frob_rel < 0.05, "reference_gemm-q8_1-frobenius-rel-error-vs-exact M=" + std::to_string(M),
              "frob_rel=" + std::to_string(frob_rel));

        std::printf("    reference_gemm M=%lld N=%lld K=%lld act=q8_1: %.3fs frob_rel_vs_exact=%.5f\n", (long long) M,
                    (long long) N, (long long) K, elapsed, frob_rel);

        // ACT_F16 arm at the two shapes the down-stream kernels will be
        // timed at most closely (a mid-size M and the full M=512): the
        // sweep above only exercised the ACT_Q8_1 arm at the GPT-OSS shape,
        // leaving ACT_F16 checked only by the 1x1x32 hand case. Both M=32
        // and M=512 are in the exact-check set above, so `exact` is always
        // available here.
        if (M == 32 || M == 512) {
            const auto                t0f      = std::chrono::steady_clock::now();
            const std::vector<double> Yf16     = reference_gemm(X, Wd, M, N, K, Activation::ACT_F16);
            const auto                t1f      = std::chrono::steady_clock::now();
            const double              elapsedf = std::chrono::duration<double>(t1f - t0f).count();

            // Same invariant as the ACT_Q8_1 arm above -- a real assertion.
            GGML_ASSERT(Yf16.size() == (size_t) (M * N));
            check(all_finite(Yf16), "reference_gemm-f16-output-finite M=" + std::to_string(M), "");

            const double frob_rel_f16 = frobenius_rel(Yf16, exact);
            // fp16 activation rounding is far tighter than q8_1's 8-bit
            // quantization, so this bound is far tighter than the 0.05 used
            // for the ACT_Q8_1 arm above.
            check(frob_rel_f16 < 0.01, "reference_gemm-f16-frobenius-rel-error-vs-exact M=" + std::to_string(M),
                  "frob_rel=" + std::to_string(frob_rel_f16));

            std::printf("    reference_gemm M=%lld N=%lld K=%lld act=f16:  %.3fs frob_rel_vs_exact=%.5f\n",
                        (long long) M, (long long) N, (long long) K, elapsedf, frob_rel_f16);
        }
    }
}

}  // namespace

int main() {
    std::printf("MXFP4 stored-layout GEMM oracle: cross-decoder equality and reference_gemm contract (host-only)\n");

    const bool decoders_ok = case_cross_decoder_equality();
    if (!decoders_ok) {
        // Everything below decodes weights through these same functions;
        // running it against a known-broken decoder would just add noise on
        // top of the failure already reported above.
        std::printf("%d check(s) failed -- cross-decoder equality failed, skipping the rest\n", failures);
        return 1;
    }

    case_scorer_sanity();
    case_reference_gemm_hand_checked();
    case_reference_gemm_gptoss_shape();

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
