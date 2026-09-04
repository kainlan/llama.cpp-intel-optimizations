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
//   3. `max_rel_violations(out, ref, tol)` is the shared scorer every later
//      G4-G8 numerics test calls. The tolerance contract it is built around:
//      the existing oneDNN WOQ 2-D arm passes
//      tests/test-sycl-mxfp4-woq-gemm-bench.cpp at max_rel <= 0.0258
//      (WOQ_MAX_REL_TOL below) -- later kernels are held to the same bar.
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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
// against, taken from the existing oneDNN WOQ 2-D arm's passing bound
// (tests/test-sycl-mxfp4-woq-gemm-bench.cpp).
constexpr double WOQ_MAX_REL_TOL = 0.0258;

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
            const int8_t  nibble          = (k_local < XMX_K / 2) ? (qs_byte & 0x0F) : (qs_byte >> 4);
            out[(size_t) (n * ncols + k)] = kvalues_mxfp4[nibble] * d;
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
    int64_t violations;
    double  max_rel;
    double  max_abs_diff;
};

// The shared scorer every later G4-G8 numerics test calls: counts elements
// whose relative error against `ref` exceeds `tol`.
Score max_rel_violations(const std::vector<double> & out, const std::vector<double> & ref, double tol) {
    Score        s{ 0, 0.0, 0.0 };
    const size_t n = std::min(out.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const double diff = std::fabs(out[i] - ref[i]);
        const double rel  = diff / std::max(1e-12, std::fabs(ref[i]));
        s.max_abs_diff    = std::max(s.max_abs_diff, diff);
        s.max_rel         = std::max(s.max_rel, rel);
        if (rel > tol) {
            ++s.violations;
        }
    }
    return s;
}

// -----------------------------------------------------------------------------
// Cases
// -----------------------------------------------------------------------------

// Cross-decoder equality: three independent decodes of the SAME quantized
// weights must be bit-identical. Positive control at the end proves the check
// is not vacuously true.
bool case_cross_decoder_equality() {
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
        return false;
    }

    int64_t soa_mismatches = 0, xmx_mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (soa_decoded[i] != ref[i]) {
            ++soa_mismatches;
        }
        if (xmx_tiled_decoded[i] != ref[i]) {
            ++xmx_mismatches;
        }
    }
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
    int64_t                  soa_corrupt_diffs   = 0;
    if (soa_corrupt_decoded.size() == ref.size()) {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (soa_corrupt_decoded[i] != ref[i]) {
                ++soa_corrupt_diffs;
            }
        }
    }
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
    int64_t xmx_corrupt_diffs = 0;
    if (xmx_corrupt_decoded.size() == ref.size()) {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (xmx_corrupt_decoded[i] != ref[i]) {
                ++xmx_corrupt_diffs;
            }
        }
    }
    check(xmx_corrupt_diffs > 0, "corrupted-xmx_tiled-byte-diverges-from-reference (positive control)",
          "diffs=" + std::to_string(xmx_corrupt_diffs));

    return soa_mismatches == 0 && xmx_mismatches == 0 && soa_corrupt_diffs > 0 && xmx_corrupt_diffs > 0;
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
}

// A hand-checkable GEMM: one MXFP4 block (K=32), weight values chosen from
// the e2m1 table directly (so the decoded float is exact), activation all
// ones so Y == sum(decoded weight row). ACT_F16 must reproduce this exactly
// (1.0 round-trips through fp16 losslessly); ACT_Q8_1 must be close (int8
// quantization of a constant row is exact too: id = amax/127, so 1.0 maps to
// round(127/amax * amax)=127... the quantizer's own rounding error is what is
// being sanity-checked here, not assumed away).
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
    check(Wd.size() == (size_t) (N * K), "decoded-weight-has-full-shape", "size=" + std::to_string(Wd.size()));

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    for (int64_t M : Ms) {
        std::vector<float> X((size_t) (M * K));
        for (float & v : X) {
            v = xdist(rng);
        }

        const auto                t0      = std::chrono::steady_clock::now();
        const std::vector<double> Y       = reference_gemm(X, Wd, M, N, K, Activation::ACT_Q8_1);
        const auto                t1      = std::chrono::steady_clock::now();
        const double              elapsed = std::chrono::duration<double>(t1 - t0).count();

        check(Y.size() == (size_t) (M * N), "reference_gemm-output-size M=" + std::to_string(M),
              "size=" + std::to_string(Y.size()));

        bool all_finite = true;
        for (double v : Y) {
            if (!std::isfinite(v)) {
                all_finite = false;
                break;
            }
        }
        check(all_finite, "reference_gemm-output-finite M=" + std::to_string(M), "");

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
        const std::vector<double> exact = exact_gemm_double(X, Wd, M, N, K);
        double                    num = 0.0, den = 0.0;
        for (size_t i = 0; i < Y.size(); ++i) {
            const double diff = Y[i] - exact[i];
            num += diff * diff;
            den += exact[i] * exact[i];
        }
        const double frob_rel = std::sqrt(num / std::max(den, 1e-300));
        check(frob_rel < 0.05, "reference_gemm-q8_1-frobenius-rel-error-vs-exact M=" + std::to_string(M),
              "frob_rel=" + std::to_string(frob_rel));

        if (M == 512) {
            check(elapsed < 10.0, "reference_gemm-M512-K2880-N2880-under-10s",
                  "elapsed=" + std::to_string(elapsed) + "s");
        }
        std::printf("    reference_gemm M=%lld N=%lld K=%lld act=q8_1: %.3fs frob_rel_vs_exact=%.5f\n", (long long) M,
                    (long long) N, (long long) K, elapsed, frob_rel);
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
