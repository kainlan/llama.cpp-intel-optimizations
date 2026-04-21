//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_XMX_F16_V2_HPP
#define GGML_SYCL_FATTN_XMX_F16_V2_HPP

// =============================================================================
// XMX-v2 flash-attention kernel — structurally correct port of CUDA fattn-mma-f16.cuh
//
// Design contract (plan §3.3, ten rules):
//
//   1. No SLM buffer aliasing — tile_Q/tile_K/tile_V/tile_S each occupy a
//      separate, non-overlapping SLM region for the kernel's lifetime.
//   2. All cross-lane softmax state in registers, updated via
//      sycl::reduce_over_group(sg, _, maximum/plus<float>{}).
//      No SLM batch_max_shared, no SLM sum_shared.
//   3. One group_barrier per SLM phase transition. No barrier on aliased regions
//      (there are none).
//   4. Zero overlap between producer/consumer SLM regions.
//   5. No lane-order-dependent computation beyond XMX/sub-group guarantees.
//      QK extracted via joint_matrix_apply (canary-5 pattern), NOT via
//      joint_matrix_store + scalar SLM read.
//   6. Fixed KV tile stride — does not depend on ncols.
//   7. [[sycl::reqd_sub_group_size(XMX_V2_SG)]] on the kernel.
//   8. FTZ rescale via __builtin_memcpy bit-trick (matches fattn-vec-f16.hpp).
//   9. Sinks applied AFTER the KV loop, register-only (CUDA §1027-1082 pattern).
//  10. Mask loaded per-KV-tile with explicit stride — no precomputed per-query-range logic.
//
// Fragment access (canary-5 confirmed): for m8×k16×n16 fp16→fp32, lane l owns
//   column l of the 8×16 tile, rows 0..7 (slot 0=(row 0, col l) … slot 7=(row 7, col l)).
// This means each of the 16 lanes gets 8 fp32 accumulator elements.
// joint_matrix_apply visits them in slot order (row-ascending for the owned column).
// =============================================================================

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>
#include <cstdint>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#  define SYCL_XMX_V2_AVAILABLE 1
#  include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#  define SYCL_XMX_V2_AVAILABLE 0
#endif

#if SYCL_XMX_V2_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// Compile-time configuration
// =============================================================================

// Tile dimensions — m8×k16×n16 fp16→fp32 (confirmed on B580 + B50 via canary).
// K=16 divides every supported head dim (64, 128, 256) evenly.
// N=16 matches SG_SIZE so each lane processes one column of the n-dimension.
static constexpr int XMX_V2_TM = 8;    // rows of the QK accumulator per MAD
static constexpr int XMX_V2_TK = 16;   // reduction dimension (must divide D)
static constexpr int XMX_V2_TN = 16;   // columns (= SG_SIZE for column-striped layout)
static constexpr int XMX_V2_SG = 16;   // sub-group size (Xe2 confirmed)

// Elements owned by each lane in one m8n16k16 accumulator.
// Column-striped: lane l owns all 8 rows of column l → 8 float elements per lane.
static constexpr int XMX_V2_ELEMS_PER_LANE = XMX_V2_TM; // = 8

// Work-group thread count.  Must be a multiple of XMX_V2_SG.
// 512 = 32 sub-groups → enough parallelism for D=256 (16 K-tiles × 2 reduce groups).
static constexpr int XMX_V2_NTHREADS = 512;
static constexpr int XMX_V2_N_SG     = XMX_V2_NTHREADS / XMX_V2_SG; // 32

// KV batch size — number of KV rows processed per outer-loop iteration.
// SLM budget: (ncols*D + 2*BATCH_KV*D + ncols*BATCH_KV) * sizeof(half).
// For ncols=8, D=128, BATCH_KV=32: (8+64)*128*2 + 8*32*2 = 18432 + 512 = 18944 bytes — well within 64 KB.
static constexpr int XMX_V2_BATCH_KV = 32; // must be a multiple of XMX_V2_TK

static_assert(XMX_V2_BATCH_KV % XMX_V2_TK == 0, "BATCH_KV must be divisible by TK");
static_assert(XMX_V2_BATCH_KV % XMX_V2_TN == 0, "BATCH_KV must be divisible by TN");

// =============================================================================
// SLM layout helper — strict, non-aliasing
//
// tile_Q[ncols][D]            — loaded once at kernel start, register-resident Q
// tile_K[BATCH_KV][D]         — K tile per KV batch
// tile_V[BATCH_KV][D]         — V tile per KV batch (separate from K)
// tile_S[ncols][BATCH_KV]     — softmax weights written before S@V MAD
//
// All regions start at WORD-aligned offsets.  The ordering above places each
// region right after the previous one with NO overlap.
// =============================================================================

template <int D, int ncols>
struct fattn_v2_slm {
    static constexpr int Q_ELEMS  = ncols        * D;
    static constexpr int K_ELEMS  = XMX_V2_BATCH_KV * D;
    static constexpr int V_ELEMS  = XMX_V2_BATCH_KV * D;
    static constexpr int S_ELEMS  = ncols        * XMX_V2_BATCH_KV;
    static constexpr int TOTAL    = Q_ELEMS + K_ELEMS + V_ELEMS + S_ELEMS;

    static constexpr int Q_OFFSET = 0;
    static constexpr int K_OFFSET = Q_OFFSET + Q_ELEMS;
    static constexpr int V_OFFSET = K_OFFSET + K_ELEMS;
    static constexpr int S_OFFSET = V_OFFSET + V_ELEMS;
};

// =============================================================================
// FP8 E4M3 → fp16 dequant (identical helper to v1 kernel — needed for kv_is_fp8)
// =============================================================================

inline sycl::half fp8_e4m3_to_half_v2(uint8_t bits) {
    const uint32_t sign = (bits >> 7) & 0x1;
    const uint32_t exp  = (bits >> 3) & 0xF;
    const uint32_t mant = bits & 0x7;
    float result;
    if (exp == 0) {
        result = (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / 512.0f);
    } else if (exp == 15 && mant == 7) {
        result = sycl::nan(0u);
    } else {
        const int32_t fp32_exp  = (int32_t)exp - 7 + 127;
        const uint32_t fp32_bits = (sign << 31) | ((uint32_t)fp32_exp << 23) | (mant << 20);
        union { uint32_t u; float f; } pun;
        pun.u = fp32_bits;
        result = pun.f;
    }
    return sycl::half(result);
}

// =============================================================================
// Kernel
// =============================================================================

template <int D, int ncols, bool use_logit_softcap, typename Q_type, bool kv_is_fp8 = false>
static void flash_attn_xmx_v2_f16_kernel(
        const char * __restrict__ Q_base,
        const char * __restrict__ K_base,
        const char * __restrict__ V_base,
        const char * __restrict__ maskh_base,
        const char * __restrict__ sinks_base,
        float * __restrict__ dst,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap,
        int ne01, int ne02,
        int nb01, int nb02, int nb03,
        int ne11, int ne12,
        int nb11, int nb12, int64_t nb13,
        int nb21, int nb22, int64_t nb23,
        int ne30, int ne32, int ne33,
        int nb31, int nb32, int64_t nb33,
        const sycl::nd_item<3> & item,
        sycl::half * slm) {

    static_assert(D % XMX_V2_TK == 0, "D must be divisible by XMX_V2_TK=16");
    static_assert(ncols % XMX_V2_TM == 0 || ncols < XMX_V2_TM,
                  "ncols must be <= TM or a multiple of TM");

    using slm_layout = fattn_v2_slm<D, ncols>;

    // ------------------------------------------------------------------
    // Sub-group / work-item identifiers
    // ------------------------------------------------------------------
    auto sg       = item.get_sub_group();
    const int sg_id  = static_cast<int>(sg.get_group_linear_id()); // 0..N_SG-1
    const int lane   = static_cast<int>(sg.get_local_id());        // 0..SG-1
    const int tid    = static_cast<int>(item.get_local_linear_id());// 0..NTHREADS-1

    // ------------------------------------------------------------------
    // Work-group → (batch-sequence, kv-head, query-block)
    // ------------------------------------------------------------------
    const int hb_id    = static_cast<int>(item.get_group(0));
    const int sequence = hb_id / ne02;
    const int head     = hb_id % ne02;
    const int ic0      = static_cast<int>(item.get_group(2)) * ncols;

    const int gqa_ratio = ne02 / ne12;
    const int kv_head   = head / gqa_ratio;

    const char * Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char * K_ptr = K_base + nb13         * sequence + (int64_t)nb12 * kv_head;
    const char * V_ptr = V_base + nb23         * sequence + (int64_t)nb22 * kv_head;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Mask pointer — pre-offset to ic0 within the query dimension.
    const int mask_head = (ne32 > 1) ? head % ne32 : 0;
    const sycl::half * maskh = maskh_base ?
        reinterpret_cast<const sycl::half*>(maskh_base
            + (int64_t)nb33 * (sequence % ne33)
            + (int64_t)nb32 * mask_head
            + (int64_t)nb31 * ic0) : nullptr;

    // ------------------------------------------------------------------
    // SLM region pointers (strictly non-overlapping — Rule 1)
    // ------------------------------------------------------------------
    sycl::half * tile_Q = slm + slm_layout::Q_OFFSET;  // [ncols][D]
    sycl::half * tile_K = slm + slm_layout::K_OFFSET;  // [BATCH_KV][D]
    sycl::half * tile_V = slm + slm_layout::V_OFFSET;  // [BATCH_KV][D]
    sycl::half * tile_S = slm + slm_layout::S_OFFSET;  // [ncols][BATCH_KV]

    // ------------------------------------------------------------------
    // Register state (Rule 2: all cross-lane softmax state in registers)
    // ------------------------------------------------------------------
    // Each lane owns column `lane` of each ncols-row output tile.
    // VKQ[j][r]: for query j, accumulated sum for row r of the D/SG strided output.
    // We map D across sub-groups: each SG (TN=16 lanes) collectively owns a D-wide tile.
    // Within a SG, lane l covers V columns that also map to output[d = lane] after S@V.
    // For D=128, TN=16: we iterate (D/TN) = 8 V-tile segments per KV batch.

    static_assert(D % XMX_V2_TN == 0, "D must be divisible by TN=16 for lane-D mapping");
    constexpr int D_TILES = D / XMX_V2_TN; // number of V-tiles in D direction

    // Per-lane VKQ accumulator: [ncols][D_TILES] float.
    // lane l, query j, V-tile t → accumulates row 0..7 of mat_SV for the t-th D-block.
    // After the KV loop we need the full D vector; since lane l owns column l of each SV tile
    // and S is [ncols×BATCH_KV], mat_SV's m8n16k16 accumulator has row=query_slot, col=D_lane.
    //
    // Concretely: for each D tile of V, we run one joint_matrix_mad giving an m8×n16 SV tile.
    // Each lane owns one column (D-slot) of the SV tile for all 8 rows (which correspond to
    // the 8 query rows this sub-group covers: sg_id*TM ... sg_id*TM+7).
    // VKQ[j][t] accumulates the single float lane l owns for query row (sg_id*TM+j) and D index
    // (t*TN + lane).  After the KV loop we scatter-write to dst.
    //
    // Number of SGs per query tile row: ncols/TM.  Each SG handles TM=8 consecutive queries.
    // If ncols < TM, we still instantiate TM rows but only write ncols of them.

    constexpr int SG_ROWS_PER_Q = (ncols + XMX_V2_TM - 1) / XMX_V2_TM; // ceil(ncols/TM)
    const int this_sg_q_tile = sg_id % SG_ROWS_PER_Q; // which Q-tile this SG covers
    // sg_q_base: first query row (within the work-group's ncols block) this SG covers.
    // Hoisted outside the KV loop because it's also used in sinks and output sections.
    const int sg_q_base = this_sg_q_tile * XMX_V2_TM;

    float VKQ[XMX_V2_TM][D_TILES]; // [query_slot_in_tile][D_tile]
    float KQ_max[XMX_V2_TM];
    float KQ_sum[XMX_V2_TM];

    #pragma unroll
    for (int r = 0; r < XMX_V2_TM; ++r) {
        KQ_max[r] = -FLT_MAX / 2.0f;
        KQ_sum[r] = 0.0f;
        #pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[r][t] = 0.0f;
        }
    }

    // ------------------------------------------------------------------
    // Phase 0: Cooperatively load tile_Q from global into SLM (Rule 1)
    // All ncols queries, scaled.
    // ------------------------------------------------------------------
    for (int idx = tid; idx < ncols * D; idx += XMX_V2_NTHREADS) {
        const int j = idx / D;
        const int d = idx % D;
        const int q_idx = ic0 + j;
        if (q_idx < ne01) {
            const Q_type * Q_row_ptr = reinterpret_cast<const Q_type*>(Q_ptr + (int64_t)nb01 * q_idx);
            tile_Q[j * D + d] = sycl::half(static_cast<float>(Q_row_ptr[d]) * scale);
        } else {
            tile_Q[j * D + d] = sycl::half(0.0f);
        }
    }
    sycl::group_barrier(item.get_group()); // barrier: tile_Q ready

    // ------------------------------------------------------------------
    // Main KV loop
    // ------------------------------------------------------------------
    for (int kv_start = 0; kv_start < ne11; kv_start += XMX_V2_BATCH_KV) {
        const int kv_count = sycl::min(XMX_V2_BATCH_KV, ne11 - kv_start);

        // ---- Phase 1: Load tile_K and tile_V from global to SLM ----
        // tile_K[k][d] = K[kv_start+k][d]
        // tile_V[k][d] = V[kv_start+k][d]
        for (int idx = tid; idx < kv_count * D; idx += XMX_V2_NTHREADS) {
            const int k = idx / D;
            const int d = idx % D;
            const int kv_pos = kv_start + k;

            const char * K_row_base = K_ptr + (int64_t)nb11 * kv_pos;
            const char * V_row_base = V_ptr + (int64_t)nb21 * kv_pos;

            sycl::half k_val, v_val;
            if constexpr (kv_is_fp8) {
                k_val = fp8_e4m3_to_half_v2(reinterpret_cast<const uint8_t*>(K_row_base)[d]);
                v_val = fp8_e4m3_to_half_v2(reinterpret_cast<const uint8_t*>(V_row_base)[d]);
            } else {
                k_val = reinterpret_cast<const sycl::half*>(K_row_base)[d];
                v_val = reinterpret_cast<const sycl::half*>(V_row_base)[d];
            }
            tile_K[k * D + d] = k_val;
            tile_V[k * D + d] = v_val;
        }
        // Zero-pad incomplete last batch so XMX never reads uninitialized SLM.
        for (int idx = tid; idx < (XMX_V2_BATCH_KV - kv_count) * D; idx += XMX_V2_NTHREADS) {
            const int k = kv_count + idx / D;
            const int d = idx % D;
            tile_K[k * D + d] = sycl::half(0.0f);
            tile_V[k * D + d] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group()); // barrier: tile_K, tile_V ready

        // ---- Phase 2: Q @ K^T via joint_matrix_mad ----
        // Each sub-group computes one (Q-row-tile) × (K^T) block.
        // Q tile: tile_Q[this_sg_q_tile*TM .. +TM][0..D] as mat_A (m8×k16×TM_tiles)
        // K tile: tile_K[0..BATCH_KV][d..d+TK] as mat_B transposed (k16×n16)
        // Accumulator: mat_QK [m8×n16] → lane l owns (rows 0..7, col l)
        //              = QK score for queries [sg_q_base+0..7] vs KV positions [col_l_mapped..].
        //
        // After all K-dim tiles, mat_QK holds the full Q@K^T block for this sub-group.
        // We then extract per-lane values via joint_matrix_apply (canary-5 pattern).

        // For each KV-position tile of width TN (= BATCH_KV / TN K-tiles):
        constexpr int KV_TILES = XMX_V2_BATCH_KV / XMX_V2_TN;

        for (int kv_tile = 0; kv_tile < KV_TILES; ++kv_tile) {
            const int kv_col = kv_tile * XMX_V2_TN; // first KV position in this tile

            // Accumulate Q @ K^T over D in steps of TK.
            // mat_A: tile_Q[sg_q_base..+TM][d..d+TK] row_major stride D
            // mat_B: tile_K[kv_col..+TN][d..d+TK] row_major stride D
            //        (loaded transposed semantically: K^T means B is [TK×TN])
            //        For row_major layout, joint_matrix_load with use::b gives B[TK][TN].
            //        Reading K[kv_col..+TN][d..d+TK] row-major with stride D gives exactly
            //        the K^T tile we need since the outer loop is over kv, inner over D.

            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator,
                                    XMX_V2_TM, XMX_V2_TN> mat_QK;
            sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

            for (int d = 0; d < D; d += XMX_V2_TK) {
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a,
                                        XMX_V2_TM, XMX_V2_TK, sycl_xmx::layout::row_major> mat_Q;
                // K loaded col_major so that row-major K[kv][d] is transposed:
                // col_major B[TK][TN]: B[k][n] = K[kv_col+n][d+k] = K^T[d+k][kv_col+n].
                // ptr = &tile_K[kv_col*D + d], stride = D → addr = kv_col*D+d + k + D*n = (kv_col+n)*D+(d+k) ✓
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b,
                                        XMX_V2_TK, XMX_V2_TN, sycl_xmx::layout::col_major> mat_K;

                // Load Q slice from SLM: tile_Q[sg_q_base][d], stride = D (row-major in [ncols][D])
                sycl_xmx::joint_matrix_load(
                    sg, mat_Q,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(
                        &tile_Q[sg_q_base * D + d]),
                    D);

                // Load K^T from SLM using col_major: tile_K[kv_col][d], stride = D
                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(
                        &tile_K[kv_col * D + d]),
                    D);

                sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_K, mat_QK);
            }

            // ---- Phase 3: Extract QK values via joint_matrix_apply (Rule 5, canary-5) ----
            // Lane l owns (row 0..7, col l) of mat_QK.
            // slot 0 = (row 0, col lane), ..., slot 7 = (row 7, col lane).
            // For each row r (=query slot within this SG's tile), lane l has the score
            // for query (sg_q_base+r) vs KV position (kv_col + lane).

            // Per-lane stash for the 8 rows this lane owns in this KV-tile column.
            float lane_QK[XMX_V2_TM];
            {
                int slot = 0;
                sycl_xmx::joint_matrix_apply(sg, mat_QK, [&](float & elem) {
                    lane_QK[slot++] = elem;
                });
            }

            // ---- Phase 4: Softcap, mask, per-row max (Rule 8, 10) ----
            #pragma unroll
            for (int r = 0; r < XMX_V2_TM; ++r) {
                const int q_abs = ic0 + sg_q_base + r;
                if (q_abs >= ne01) { lane_QK[r] = -FLT_MAX; continue; }

                // Logit softcap (inside Q@K^T, before softmax — CUDA §2.2 item 7)
                if constexpr (use_logit_softcap) {
                    lane_QK[r] = logit_softcap * sycl::tanh(lane_QK[r]);
                }

                // Mask: lane l contributes the score for KV position (kv_col + lane).
                // maskh is pre-offset to ic0, so query offset = sg_q_base + r.
                // Within the mask row, KV position = kv_start + kv_col + lane.
                if (maskh) {
                    const int kv_abs = kv_start + kv_col + lane;
                    if (kv_abs < ne11) {
                        const float mask_val = static_cast<float>(
                            maskh[(sg_q_base + r) * ne30 + kv_abs]);
                        lane_QK[r] += slope * mask_val;
                    } else {
                        lane_QK[r] = -FLT_MAX;
                    }
                }

                // OOB KV positions get -FLT_MAX so they don't affect softmax.
                if (kv_start + kv_col + lane >= ne11) {
                    lane_QK[r] = -FLT_MAX;
                }
            }

            // Per-row max over this SG's KV-tile (Rule 2: sub-group reduce, no SLM)
            #pragma unroll
            for (int r = 0; r < XMX_V2_TM; ++r) {
                const int q_abs = ic0 + sg_q_base + r;
                if (q_abs >= ne01) continue;

                // All lanes in this SG hold lane_QK[r] for column `lane`.
                // Reduce to get the row-max across all 16 KV columns of this tile.
                const float tile_max =
                    sycl::reduce_over_group(sg, lane_QK[r], sycl::maximum<float>{});

                const float new_max      = sycl::fmax(KQ_max[r], tile_max);
                const float KQ_max_diff  = KQ_max[r] - new_max;

                // FTZ via bit manipulation (Rule 8, matches fattn-vec-f16.hpp)
                float scale_old = sycl::exp(KQ_max_diff);
                uint32_t scale_bits;
                __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
                scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
                __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

                // Rescale running VKQ and KQ_sum for this row.
                #pragma unroll
                for (int t = 0; t < D_TILES; ++t) {
                    VKQ[r][t] *= scale_old;
                }
                KQ_sum[r] *= scale_old;
                KQ_max[r]  = new_max;

                // Accumulate row-sum contribution from this tile.
                const float p = sycl::exp(lane_QK[r] - new_max);
                // Each lane's p is exp(QK[kv_col+lane] - max) for its own KV position.
                // Sum across the SG gives the total softmax-weight sum for this KV tile.
                const float tile_sum =
                    sycl::reduce_over_group(sg, p, sycl::plus<float>{});
                KQ_sum[r] += tile_sum;

                // Store softmax weight for this lane's KV position in tile_S.
                // tile_S layout: [ncols][BATCH_KV] → index = (sg_q_base+r) * BATCH_KV + (kv_col+lane)
                const int s_row = sg_q_base + r;
                const int s_col = kv_col + lane;
                if (s_row < ncols && s_col < XMX_V2_BATCH_KV) {
                    tile_S[s_row * XMX_V2_BATCH_KV + s_col] = sycl::half(p);
                }
            }
        } // end kv_tile loop

        // Barrier: all sub-groups have written their tile_S columns (Rule 3)
        sycl::group_barrier(item.get_group());

        // ---- Phase 5: S @ V via joint_matrix_mad ----
        // mat_S: tile_S[this_sg_q_tile*TM..+TM][0..BATCH_KV]
        // mat_V: tile_V[0..BATCH_KV][d_start..d_start+TN]
        // mat_SV accumulator: [m8×n16] → lane l owns (rows 0..7, col l)
        //        = V contribution for queries [sg_q_base..] and D-slot (d_start + lane)

        for (int d_tile = 0; d_tile < D_TILES; ++d_tile) {
            const int d_start = d_tile * XMX_V2_TN;

            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator,
                                    XMX_V2_TM, XMX_V2_TN> mat_SV;
            sycl_xmx::joint_matrix_fill(sg, mat_SV, 0.0f);

            for (int k = 0; k < XMX_V2_BATCH_KV; k += XMX_V2_TK) {
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a,
                                        XMX_V2_TM, XMX_V2_TK, sycl_xmx::layout::row_major> mat_S;
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b,
                                        XMX_V2_TK, XMX_V2_TN, sycl_xmx::layout::row_major> mat_V;

                // Load S tile: tile_S[sg_q_base][k], stride = BATCH_KV
                sycl_xmx::joint_matrix_load(
                    sg, mat_S,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(
                        &tile_S[sg_q_base * XMX_V2_BATCH_KV + k]),
                    XMX_V2_BATCH_KV);

                // Load V tile: tile_V[k][d_start], stride = D
                sycl_xmx::joint_matrix_load(
                    sg, mat_V,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(
                        &tile_V[k * D + d_start]),
                    D);

                sycl_xmx::joint_matrix_mad(sg, mat_SV, mat_S, mat_V, mat_SV);
            }

            // Extract mat_SV elements via joint_matrix_apply (canary-5 pattern, Rule 5)
            // Lane l gets 8 floats: (row 0..7, col l) = VKQ contribution for
            // query (sg_q_base+row) and D-slot (d_start + lane).
            {
                int slot = 0;
                sycl_xmx::joint_matrix_apply(sg, mat_SV, [&](float & elem) {
                    VKQ[slot][d_tile] += elem;
                    ++slot;
                });
            }
        }

        // Barrier: wait for all SGs to finish reading tile_S and tile_V before next KV load
        sycl::group_barrier(item.get_group()); // Rule 3

    } // end kv_start loop

    // ------------------------------------------------------------------
    // Post-loop: attention sinks (Rule 9 — applied AFTER KV loop, CUDA §1027-1082)
    // ------------------------------------------------------------------
    if (sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        const float   sink    = sinks_f[head];

        #pragma unroll
        for (int r = 0; r < XMX_V2_TM; ++r) {
            const int q_abs = ic0 + sg_q_base + r;
            if (q_abs >= ne01) continue;

            const float new_max     = sycl::fmax(KQ_max[r], sink);
            const float KQ_max_diff = KQ_max[r] - new_max;

            float scale_old = sycl::exp(KQ_max_diff);
            uint32_t scale_bits;
            __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
            scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
            __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

            const float sink_weight = sycl::exp(sink - new_max);
            KQ_sum[r] = KQ_sum[r] * scale_old + sink_weight;
            KQ_max[r] = new_max;

            #pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                VKQ[r][t] *= scale_old;
            }
        }
    }

    // ------------------------------------------------------------------
    // Normalize and write output.
    // dst layout: [D][n_heads][n_queries][batch] (ggml row-major, dim0 fastest).
    // Lane l holds D-slot (d_tile * TN + lane) for all rows in this SG's Q tile.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int r = 0; r < XMX_V2_TM; ++r) {
        const int q_abs = ic0 + sg_q_base + r;
        if (q_abs >= ne01) continue;

        const float inv_sum = (KQ_sum[r] > 0.0f) ? (1.0f / KQ_sum[r]) : 0.0f;
        float * dst_row = dst + (int64_t)D * (head + ne02 * (q_abs + ne01 * sequence));

        #pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int d = t * XMX_V2_TN + lane;
            const float val = VKQ[r][t] * inv_sum;
            dst_row[d] = sycl::isfinite(val) ? val : 0.0f;
        }
    }
}

// =============================================================================
// Launch function
// =============================================================================

template <int D, int ncols, bool use_logit_softcap, typename Q_type, bool kv_is_fp8 = false>
void launch_fattn_xmx_v2_f16(
        const fattn_params & params,
        dpct::queue_ptr      stream) {

    using slm_layout = fattn_v2_slm<D, ncols>;

    // Grid: dim0 = ne02*ne03 (heads×batch), dim1=1, dim2=n_q_blocks
    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, XMX_V2_NTHREADS);

    const char *   Q_ptr  = params.Q;
    const char *   K_ptr  = params.K;
    const char *   V_ptr  = params.V;
    const char *   mask_ptr  = params.mask;
    const char *   sinks_ptr = params.sinks;
    float *        dst_ptr   = params.dst;
    const float    scale_v   = params.scale;
    const float    max_bias  = params.max_bias;
    const float    m0        = params.m0;
    const float    m1        = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;
    const float    logit_sc  = params.logit_softcap;
    const int      ne01 = params.ne01, ne02 = params.ne02;
    const int      nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int      ne11 = params.ne11, ne12 = params.ne12;
    const int      nb11 = params.nb11, nb12 = params.nb12;
    const int64_t  nb13 = params.nb13;
    const int      nb21 = params.nb21, nb22 = params.nb22;
    const int64_t  nb23 = params.nb23;
    const int      ne30 = params.ne30, ne32 = params.ne32, ne33 = params.ne33;
    const int      nb31 = params.nb31, nb32 = params.nb32;
    const int64_t  nb33 = params.nb33;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(slm_layout::TOTAL), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                flash_attn_xmx_v2_f16_kernel<D, ncols, use_logit_softcap, Q_type, kv_is_fp8>(
                    Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr,
                    scale_v, max_bias, m0, m1, n_head_log2, logit_sc,
                    ne01, ne02,
                    nb01, nb02, nb03,
                    ne11, ne12,
                    nb11, nb12, nb13,
                    nb21, nb22, nb23,
                    ne30, ne32, ne33,
                    nb31, nb32, nb33,
                    item,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

// =============================================================================
// Availability check (mirrors fattn_xmx_f16_available)
// =============================================================================

inline bool fattn_xmx_v2_f16_available() {
    return true; // compile-time gate: SYCL_XMX_V2_AVAILABLE == 1
}

#else // !SYCL_XMX_V2_AVAILABLE

template <int D, int ncols, bool use_logit_softcap, typename Q_type, bool kv_is_fp8 = false>
void launch_fattn_xmx_v2_f16(const fattn_params &, dpct::queue_ptr) {
    GGML_ABORT("XMX v2 not available at compile time");
}

inline bool fattn_xmx_v2_f16_available() { return false; }

#endif // SYCL_XMX_V2_AVAILABLE

#endif // GGML_SYCL_FATTN_XMX_F16_V2_HPP
