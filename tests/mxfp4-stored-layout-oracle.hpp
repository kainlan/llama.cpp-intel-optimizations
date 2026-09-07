// Shared host-only oracle machinery for MXFP4 GEMMs on the two
// device-materialized expert weight layouts (llama.cpp-vtfs, "option C",
// owner ruling 2026-08-23). Factored out of
// tests/test-sycl-mxfp4-stored-layout-gemm-oracle.cpp (Task G3) so downstream
// numerics tests (Task G4 onward) reuse the SAME decoders, reference GEMM and
// scorer rather than re-deriving or copy-pasting them -- a divergence between
// two copies of `reference_gemm`/`max_rel_violations` would silently change
// what "0 violations" means between the oracle's own self-test and a later
// kernel's numerics gate.
//
// Everything here is header-only inside an anonymous namespace: each
// translation unit that includes this file gets its own private copy (no
// linkage/ODR concerns across the two -- or more -- .cpp files that include
// it), exactly mirroring how this content lived directly inside the G3 test
// file before this refactor.
//
// Host-only: no SYCL, no device, no model. See the G3 test file's header
// comment for the full formula-source citations (quants.hpp, moe-xmx-fused.hpp,
// moe-tile-convert.cpp, convert.cpp) and the tolerance-contract rationale
// (WOQ_MAX_REL_TOL / DEFAULT_ABS_FLOOR) -- reproduced here only where the code
// itself needs the comment, not duplicated at length a second time.

#ifndef GGML_TESTS_MXFP4_STORED_LAYOUT_ORACLE_HPP
#define GGML_TESTS_MXFP4_STORED_LAYOUT_ORACLE_HPP

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

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
// abs_tol=0.01/rel_tol=0.02 and never states 0.0258).
constexpr double WOQ_MAX_REL_TOL = 0.0258;

// The absolute-error floor paired with WOQ_MAX_REL_TOL in max_rel_violations:
// a cell only counts as a violation when its relative error exceeds
// WOQ_MAX_REL_TOL AND its absolute error exceeds this floor -- excludes the
// near-zero-cell relative-error artifact fix-cycle-4 documents (see
// max_rel_violations below).
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
            // block_q8_1's `d`/`s` union member names depend on which
            // GGML_COMMON_DECL_* branch won in the INCLUDING translation
            // unit (see ggml-common.h's GGML_COMMON_AGGR_S/_U): this header
            // is shared with GPU test files that pull in SYCL headers
            // (ggml-sycl/common.hpp) ahead of this include, which can
            // already have resolved ggml-common.h under a different branch
            // than this header's own `#define GGML_COMMON_IMPL_C` requests
            // -- ggml-common.h is included-once keyed on GGML_COMMON_DECL,
            // so a later, different macro request is a silent no-op. Read
            // the leading 2 bytes by LAYOUT instead of by member name: `d`
            // (ggml_half, fp16) is always the first 2 bytes of block_q8_1
            // regardless of which branch won -- only the accessor name
            // changes, never the memory layout. Already pinned tree-wide by
            // ggml-common.h:269's own `static_assert(sizeof(block_q8_1) ==
            // 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding")`
            // -- a duplicate assert here would check nothing that assert
            // does not already guarantee build-wide (llama.cpp-6f73 c-py5n
            // nit, correcting round 1's redundant local static_assert).
            ggml_fp16_t        d_bits;
            std::memcpy(&d_bits, &blk, sizeof(d_bits));
            const double d = (double) ggml_fp16_to_fp32(d_bits);
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
// only as an internal sanity cross-check by callers, not part of the public
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

// The shared scorer every stored-layout MXFP4 numerics test calls: counts
// elements whose relative error against `ref` exceeds `rel_tol` AND whose
// absolute error exceeds `abs_floor`. Both conditions are required (not
// either alone) so a near-zero-magnitude reference cell -- where a genuine,
// tiny absolute error reads as an enormous relative error -- is not flagged;
// see the tolerance-contract comment on WOQ_MAX_REL_TOL above. The returned
// `max_rel`/`max_abs_diff` are the UNFILTERED whole-output extrema (see the
// Score comment) -- only `violations` reflects the abs_floor gate.
// `out.size() != ref.size()` is a hard failure: returns Score{-1, +inf, +inf}
// rather than silently scoring the shorter of the two via std::min; the
// infinities make a naive `max_rel <= tol` check fail closed too, not just a
// `violations == 0` check.
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
// Small helpers shared by callers.
// -----------------------------------------------------------------------------

// Element count where a[i] != b[i]. A size mismatch returns -1 (not 0 -- the
// PASS value at the two `== 0` call sites, where 0 would have silently read
// as "no mismatches") without indexing out of bounds rather than crashing;
// the caller's own size check (run separately, before this is called) is
// what normally prevents this path, but the sentinel keeps a mismatched pair
// fail-closed even if that guard were ever skipped.
int64_t count_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return -1;
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
// per-element relative metric is (see max_rel_violations' scoping note).
// Unlike count_mismatches above, this has no in-band sentinel value to fail
// closed with (any double is a plausible ratio), so a size mismatch asserts
// instead.
double frobenius_rel(const std::vector<double> & a, const std::vector<double> & b) {
    GGML_ASSERT(a.size() == b.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        num += diff * diff;
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

}  // namespace

#endif  // GGML_TESTS_MXFP4_STORED_LAYOUT_ORACLE_HPP
