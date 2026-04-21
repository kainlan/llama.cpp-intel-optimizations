//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_VEC_F16_HPP
#define GGML_SYCL_FATTN_VEC_F16_HPP

#include "fattn-common.hpp"
#include "common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>
#include <cstdint>

// =============================================================================
// VEC flash-attention kernel — TG fast path (ncols == 1).
//
// Design contract (from docs/plans/2026-04-20-fattn-xmx-rewrite.md §3.2):
//   • ZERO SLM — all state is register-private or via sub-group collectives.
//   • Sub-group size 16 via [[sycl::reqd_sub_group_size(16)]].
//   • One sub-group per (query, head) pair.
//   • Online softmax via sycl::reduce_over_group — deterministic per-compile.
//   • FTZ trick via bit manipulation (not branch) to preserve determinism.
//   • Attention sinks applied AFTER the KV loop.
// =============================================================================

static constexpr int FATTN_VEC_SG_SIZE = 16;   // sub-group size (Xe2 confirmed)

// ---------------------------------------------------------------------------
// Kernel implementation
// ---------------------------------------------------------------------------

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
static void flash_attn_vec_f16_kernel(
        const char * __restrict__ Q_base,
        const char * __restrict__ K_base,
        const char * __restrict__ V_base,
        const char * __restrict__ maskh_base,
        const char * __restrict__ sinks_base,
        float      * __restrict__ dst,
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
        int ne32, int ne33,
        int nb31, int nb32, int64_t nb33,
        const sycl::nd_item<3> & item) {

    // Each work-group is exactly one sub-group of 16 lanes.
    auto sg      = item.get_sub_group();
    int  lane_id = static_cast<int>(sg.get_local_id());  // 0..15

    // Work-group identifies (sequence, head, query_block).
    // Grid layout: dim0 = ne02 * ne03 (head * batch),  dim2 = n_q_blocks.
    const int hb_id    = static_cast<int>(item.get_group(0));
    const int sequence = hb_id / ne02;
    const int head     = hb_id % ne02;
    const int ic0      = static_cast<int>(item.get_group(2)) * ncols;

    // Each sub-group computes one (query, head) at a time.
    // For ncols > 1 the outer loop iterates, but for Phase 1 ncols == 1.
    const int gqa_ratio  = ne02 / ne12;
    const int kv_head    = head / gqa_ratio;

    const char * Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char * K_ptr = K_base + nb13 * sequence + (int64_t)nb12 * kv_head;
    const char * V_ptr = V_base + nb23 * sequence + (int64_t)nb22 * kv_head;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Mask pointer for this query block.
    // maskh[query_offset * stride_mask + kv_pos]
    // Critical: stride_mask = nb31/sizeof(half), NOT ne30 — mask tensor may have
    // padding (GGML_KQ_MASK_PAD) so physical stride != logical column count.
    const int mask_head   = (ne32 > 1) ? head % ne32 : 0;
    const int stride_mask = nb31 / static_cast<int>(sizeof(sycl::half));
    const sycl::half * maskh = maskh_base ?
        reinterpret_cast<const sycl::half*>(maskh_base
            + (int64_t)nb33 * (sequence % ne33)
            + (int64_t)nb32 * mask_head
            + (int64_t)nb31 * ic0) : nullptr;

    // -------------------------------------------------------------------------
    // Register state (per lane, private — zero SLM).
    // Each lane owns D/SG_SIZE consecutive elements of Q, VKQ.
    // -------------------------------------------------------------------------
    static_assert(D % FATTN_VEC_SG_SIZE == 0, "D must be divisible by SG_SIZE=16");
    constexpr int D_PER_LANE = D / FATTN_VEC_SG_SIZE;

    // Per-lane slice of Q (loaded once, register-resident).
    float Q_row[D_PER_LANE];

    // Per-lane output accumulator and softmax state.
    float VKQ_partial[D_PER_LANE];
    float KQ_max;
    float KQ_sum;

    // Iterate over ncols query rows (Phase 1: ncols == 1, single iteration).
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const int q_idx = ic0 + j;
        if (q_idx >= ne01) { break; }

        // Load Q row for this lane's D slice (scaled into registers).
        const int d_base = lane_id * D_PER_LANE;
        {
            const Q_type * Q_row_ptr = reinterpret_cast<const Q_type*>(Q_ptr + (int64_t)nb01 * q_idx);
            #pragma unroll
            for (int d = 0; d < D_PER_LANE; ++d) {
                Q_row[d] = static_cast<float>(Q_row_ptr[d_base + d]) * scale;
            }
        }

        // Reset per-query softmax state for each column.
        KQ_max = -FLT_MAX / 2.0f;
        KQ_sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_PER_LANE; ++d) {
            VKQ_partial[d] = 0.0f;
        }

        // -----------------------------------------------------------------------
        // Main KV loop — one KV position per iteration, lanes share Q dot work.
        // -----------------------------------------------------------------------
        for (int kv = 0; kv < ne11; ++kv) {
            const sycl::half * K_row = reinterpret_cast<const sycl::half*>(K_ptr + (int64_t)nb11 * kv);

            // Step 1: lane-local partial dot product Q · K[kv].
            float dot_partial = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_PER_LANE; ++d) {
                dot_partial += Q_row[d] * static_cast<float>(K_row[d_base + d]);
            }

            // Step 2: sub-group reduce to get the full dot product (all lanes get same value).
            float dot = sycl::reduce_over_group(sg, dot_partial, sycl::plus<float>{});

            // Step 3: logit softcap (inside loop, before mask+softmax).
            if constexpr (use_logit_softcap) {
                dot = logit_softcap * sycl::tanh(dot);
            }

            // Step 4: mask.
            // maskh is pre-offset to query block ic0, so use j (0-relative within block).
            // Critical: use stride_mask (nb31/sizeof(half)) not ne30 — mask tensor may have
            // padding (GGML_KQ_MASK_PAD) so physical stride != logical column count.
            if (maskh) {
                const float mask_val = static_cast<float>(maskh[j * stride_mask + kv]);
                dot += slope * mask_val;
            }

            // Step 5: online softmax update (all lanes see the same dot, so symmetric).
            const float new_max = sycl::fmax(KQ_max, dot);
            const float KQ_max_diff = KQ_max - new_max;  // <= 0

            // FTZ trick: multiply scale_old by zero when diff < threshold to avoid
            // calling expf on very negative numbers. Bit-manipulation avoids diverging branch.
            float scale_old = sycl::exp(KQ_max_diff);
            uint32_t scale_old_bits;
            __builtin_memcpy(&scale_old_bits, &scale_old, sizeof(uint32_t));
            scale_old_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
            __builtin_memcpy(&scale_old, &scale_old_bits, sizeof(float));

            // Rescale running state.
            const float p = sycl::exp(dot - new_max);  // exp(dot - new_max) used twice below
            #pragma unroll
            for (int d = 0; d < D_PER_LANE; ++d) {
                VKQ_partial[d] *= scale_old;
            }
            KQ_sum = KQ_sum * scale_old + p;
            KQ_max = new_max;

            // Step 6: accumulate weighted V.
            const sycl::half * V_row = reinterpret_cast<const sycl::half*>(V_ptr + (int64_t)nb21 * kv);
            #pragma unroll
            for (int d = 0; d < D_PER_LANE; ++d) {
                VKQ_partial[d] += p * static_cast<float>(V_row[d_base + d]);
            }
        }

        // -----------------------------------------------------------------------
        // Attention sinks — applied AFTER the KV loop (plan §2.2 item 6).
        // -----------------------------------------------------------------------
        if (sinks_base) {
            const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
            const float   sink    = sinks_f[head];

            const float KQ_max_new   = sycl::fmax(sink, KQ_max);
            const float KQ_max_scale = sycl::exp(KQ_max - KQ_max_new);
            KQ_max = KQ_max_new;

            const float sink_weight = sycl::exp(sink - KQ_max);
            KQ_sum = KQ_sum * KQ_max_scale + sink_weight;

            #pragma unroll
            for (int d = 0; d < D_PER_LANE; ++d) {
                VKQ_partial[d] *= KQ_max_scale;
            }
        }

        // -----------------------------------------------------------------------
        // Normalize and write output.
        // dst layout: [D][n_heads][n_queries][batch] (ggml row-major, dim0 fastest)
        // -----------------------------------------------------------------------
        const float inv_sum = (KQ_sum > 0.0f) ? (1.0f / KQ_sum) : 0.0f;
        float * dst_row = dst + D * (head + ne02 * (q_idx + ne01 * sequence));

        #pragma unroll
        for (int d = 0; d < D_PER_LANE; ++d) {
            const float val = VKQ_partial[d] * inv_sum;
            dst_row[d_base + d] = sycl::isfinite(val) ? val : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Launch function — called from fattn.cpp dispatcher.
// ---------------------------------------------------------------------------

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_vec_f16(
        const fattn_params & params,
        dpct::queue_ptr      stream) {

    // Grid: dim0 = ne02 * ne03 (heads × batch), dim1 = 1, dim2 = n_q_blocks.
    // Work-group: one sub-group = 16 lanes (SG_SIZE).
    // No SLM needed.

    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, FATTN_VEC_SG_SIZE);   // exactly one sub-group per WG

    // Capture params by value to avoid dangling references in lambda.
    const char *     Q_ptr          = params.Q;
    const char *     K_ptr          = params.K;
    const char *     V_ptr          = params.V;
    const char *     mask_ptr       = params.mask;
    const char *     sinks_ptr      = params.sinks;
    float *          dst_ptr        = params.dst;
    const float      scale_val      = params.scale;
    const float      max_bias_val   = params.max_bias;
    const float      m0_val         = params.m0;
    const float      m1_val         = params.m1;
    const uint32_t   n_head_log2    = params.n_head_log2;
    const float      logit_sc       = params.logit_softcap;
    const int        ne01 = params.ne01, ne02 = params.ne02;
    const int        nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int        ne11 = params.ne11, ne12 = params.ne12;
    const int        nb11 = params.nb11, nb12 = params.nb12;
    const int64_t    nb13 = params.nb13;
    const int        nb21 = params.nb21, nb22 = params.nb22;
    const int64_t    nb23 = params.nb23;
    const int        ne32 = params.ne32, ne33 = params.ne33;
    const int        nb31 = params.nb31, nb32 = params.nb32;
    const int64_t    nb33 = params.nb33;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(FATTN_VEC_SG_SIZE)]] {
                flash_attn_vec_f16_kernel<D, ncols, use_logit_softcap, Q_type>(
                    Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr,
                    scale_val, max_bias_val, m0_val, m1_val, n_head_log2, logit_sc,
                    ne01, ne02,
                    nb01, nb02, nb03,
                    ne11, ne12,
                    nb11, nb12, nb13,
                    nb21, nb22, nb23,
                    ne32, ne33,
                    nb31, nb32, nb33,
                    item);
            });
    });
}

#endif // GGML_SYCL_FATTN_VEC_F16_HPP
