//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst,
                                const char *                src0_dd_i,
                                const float *               src1_ddf_i,
                                const char *                src1_ddq_i,
                                float *                     dst_dd_i,
                                const int64_t               row_low,
                                const int64_t               row_high,
                                const int64_t               src1_ncols,
                                const int64_t               src1_padded_row_size,
                                const dpct::queue_ptr &     stream);

struct ggml_sycl_mmvq_fused_add {
    const float * data = nullptr;
    int64_t       ne0  = 0;
    int64_t       nb0  = sizeof(float);

    bool active() const { return data != nullptr && ne0 > 0 && nb0 > 0; }
};

void ggml_sycl_mmvq_set_fused_add(const ggml_sycl_mmvq_fused_add & add);
void ggml_sycl_mmvq_clear_fused_add();

// MoE-aware MUL_MAT_ID dispatch: GPU-side expert routing without host sync
// This is compatible with SYCL command graph recording
// Returns true if the operation was handled, false to fall back to host-side routing
bool ggml_sycl_mul_mat_id_vec_q(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                const ggml_tensor *         ids,
                                ggml_tensor *               dst,
                                const layout_mode *         forced_layout = nullptr);

#ifdef GGML_SYCL_GRAPH
// Pre-allocate Q8_1 buffers for MoE graph recording
// Must be called before graph recording starts (during decode phase)
void ggml_sycl_moe_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);
#endif

// Batched MMVQ dispatch for MoE hybrid GPU path.
// Replaces per-expert ggml_sycl_mul_mat() loop with a single kernel submission.
// expert_ptrs_device: device pointer table (expert_id -> device weight ptr)
// n_experts: total number of experts in the model
// gpu_expert_ids: host array of expert IDs that have cached device pointers
// gpu_iid1s: host array of token indices (parallel to gpu_expert_ids)
// gpu_ids: host array of expert slot indices (parallel to gpu_expert_ids)
// n_gpu_entries: number of GPU-dispatched entries
// Returns true if handled, false to fall back to per-expert loop.
bool mmvq_moe_batched_dispatch(ggml_backend_sycl_context &      ctx,
                               const ggml_tensor *              src0,
                               const ggml_tensor *              src1,
                               ggml_tensor *                    dst,
                               const void * const *             expert_ptrs_device,
                               const int32_t *                  gpu_expert_ids,
                               const int64_t *                  gpu_iid1s,
                               const int64_t *                  gpu_ids,
                               int                              n_gpu_entries,
                               int                              n_experts,
                               int64_t                          n_ids,
                               layout_mode                      layout               = GGML_LAYOUT_AOS,
                               const int32_t *                  direct_ids_device    = nullptr,
                               int64_t                          direct_ids_nb0       = 0,
                               int64_t                          direct_ids_nb1       = 0,
                               const int32_t *                  ids_host             = nullptr,
                               int64_t                          ids_host_count       = 0,
                               sycl::event *                    completion_event     = nullptr,
                               bool *                           completion_event_set = nullptr,
                               const std::vector<sycl::event> * deps                 = nullptr);

bool mmvq_moe_batched_dispatch_pair_mxfp4_soa(ggml_backend_sycl_context & ctx,
                                              const ggml_tensor *         src0_a,
                                              const ggml_tensor *         src0_b,
                                              const ggml_tensor *         src1,
                                              ggml_tensor *               dst_a,
                                              ggml_tensor *               dst_b,
                                              const void * const *        expert_ptrs_a_device,
                                              const void * const *        expert_ptrs_b_device,
                                              const int32_t *             ids_device,
                                              int                         n_gpu_entries,
                                              int64_t                     n_ids,
                                              int64_t                     ids_nb0,
                                              int64_t                     ids_nb1);

bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(ggml_backend_sycl_context &   ctx,
                                                  const ggml_tensor *           gate_weight,
                                                  const ggml_tensor *           up_weight,
                                                  const ggml_tensor *           src1,
                                                  ggml_tensor *                 glu_dst,
                                                  const void * const *          gate_ptrs_device,
                                                  const void * const *          up_ptrs_device,
                                                  const int32_t *               ids_device,
                                                  const float *                 gate_bias_device,
                                                  const float *                 up_bias_device,
                                                  int64_t                       gate_bias_nb1,
                                                  int64_t                       up_bias_nb1,
                                                  int                           n_gpu_entries,
                                                  int64_t                       n_ids,
                                                  int64_t                       ids_nb0,
                                                  int64_t                       ids_nb1,
                                                  int                           glu_op,
                                                  float                         alpha,
                                                  float                         limit,
                                                  ggml_layout_mode              weight_layout,
                                                  const ggml_sycl::mem_handle * glu_dst_handle_override    = nullptr,
                                                  bool                          direct_xmx_eligible        = false,
                                                  bool                          xmx_tiled_grouped_eligible = false,
                                                  ggml_tensor *                 gate_tmp                   = nullptr,
                                                  ggml_tensor *                 up_tmp                     = nullptr,
                                                  const int32_t *               ids_host                   = nullptr,
                                                  int64_t                       ids_host_count             = 0,
                                                  sycl::event *                 completion_event           = nullptr,
                                                  bool *                        completion_event_set       = nullptr);

// Fast all-local decode down projection that consumes the Q8_1 GLU artifact
// published by mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa().  When ids_device
// is null, down_ptrs_device is a selected pointer table in batch order. Ownership
// of weight/output/control memory stays with mem_handles; pointers here are
// transient launch ABI values resolved by the caller from those handles.  The
// XMX_TILED variant is opt-in at the layout planner level and uses the same
// handle-owned expert pointer table; this function only consumes transient ABI
// pointers and never owns allocation state.
bool mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4(ggml_backend_sycl_context &      ctx,
                                                         const ggml_tensor *              down_weight,
                                                         const ggml_tensor *              glu_src,
                                                         ggml_tensor *                    down_dst,
                                                         const void * const *             down_ptrs_device,
                                                         const int32_t *                  ids_device,
                                                         int                              n_gpu_entries,
                                                         int64_t                          n_ids,
                                                         int64_t                          ids_nb0,
                                                         int64_t                          ids_nb1,
                                                         ggml_layout_mode                 down_layout,
                                                         const ggml_sycl::mem_handle *    glu_src_handle_override,
                                                         const ggml_sycl::mem_handle *    down_dst_handle_override,
                                                         const int32_t *                  ids_host         = nullptr,
                                                         int64_t                          ids_host_count   = 0,
                                                         const std::vector<sycl::event> * deps             = nullptr,
                                                         sycl::event *                    completion_event = nullptr,
                                                         bool * completion_event_set                       = nullptr);

struct mxfp4_moe_direct_final_metadata {
    const int32_t * expert_ids_device          = nullptr;
    int64_t         expert_ids_nb0             = 0;
    int64_t         expert_ids_nb1             = 0;
    const int32_t * expert_ids_host            = nullptr;
    int64_t         entries                    = 0;
    int64_t         expected_entries           = 0;
    bool            token_major_deterministic  = false;
    bool            route_weights_device_valid = false;
};

namespace ggml_sycl {

class moe_gateup_prepack_scratch_descriptor;

enum class mxfp4_moe_gateup_prepack_status : uint8_t {
    OK = 0,
    ENV_DISABLED,
    INVALID_ARGUMENT,
    INVALID_DESCRIPTOR,
    INVALID_SHAPE,
    INVALID_SELECTION,
    SCRATCH_TOO_SMALL,
    LAYOUT_OVERFLOW,
};

const char * mxfp4_moe_gateup_prepack_status_name(mxfp4_moe_gateup_prepack_status status);
bool         mxfp4_moe_gateup_prepack_enabled_from_env(const char * env);

struct mxfp4_moe_gateup_prepack_layout {
    int64_t selected_count           = 0;
    int64_t nrows_per_expert         = 0;
    int64_t ncols                    = 0;
    int64_t k_tiles                  = 0;
    int64_t tile_groups_n            = 0;
    size_t  group_bytes              = 0;
    size_t  single_expert_role_bytes = 0;
    size_t  entry_bytes              = 0;
    size_t  total_bytes              = 0;
};

struct mxfp4_moe_gateup_prepack_key {
    int      layer                    = -1;
    int      submit_device            = -1;
    uint64_t route_metadata_signature = 0;
    size_t   gate_identity_hash       = 0;
    size_t   up_identity_hash         = 0;
    size_t   scratch_identity_hash    = 0;
    int64_t  n_experts                = 0;
    int64_t  nrows_per_expert         = 0;
    int64_t  ncols                    = 0;
    int64_t  selected_count           = 0;
    uint64_t selected_experts_hash    = 0;

    size_t stable_hash() const;
};

struct mxfp4_moe_gateup_prepack_request {
    const moe_gateup_prepack_scratch_descriptor * descriptor               = nullptr;
    const uint8_t *                               gate_base                = nullptr;
    const uint8_t *                               up_base                  = nullptr;
    uint8_t *                                     scratch                  = nullptr;
    size_t                                        scratch_bytes            = 0;
    const int32_t *                               selected_experts         = nullptr;
    int64_t                                       selected_count           = 0;
    int64_t                                       n_experts                = 0;
    int64_t                                       nrows_per_expert         = 0;
    int64_t                                       ncols                    = 0;
    int                                           layer                    = -1;
    int                                           submit_device            = -1;
    uint64_t                                      route_metadata_signature = 0;
    size_t                                        gate_identity_hash       = 0;
    size_t                                        up_identity_hash         = 0;
    size_t                                        scratch_identity_hash    = 0;
    bool                                          env_enabled              = false;
};

struct mxfp4_moe_gateup_prepack_result {
    mxfp4_moe_gateup_prepack_status status = mxfp4_moe_gateup_prepack_status::INVALID_ARGUMENT;
    mxfp4_moe_gateup_prepack_layout layout;
    mxfp4_moe_gateup_prepack_key    key;
    sycl::event                     event;
    bool                            event_set = false;

    bool ok() const { return status == mxfp4_moe_gateup_prepack_status::OK; }
};

mxfp4_moe_gateup_prepack_status mxfp4_moe_gateup_prepack_layout_for_shape(int64_t selected_count,
                                                                          int64_t nrows_per_expert,
                                                                          int64_t ncols,
                                                                          mxfp4_moe_gateup_prepack_layout * layout);

mxfp4_moe_gateup_prepack_result mxfp4_moe_gateup_prepack_selected_rows_reference(
    const mxfp4_moe_gateup_prepack_request & request);

mxfp4_moe_gateup_prepack_result mxfp4_moe_gateup_prepack_selected_rows_submit(
    sycl::queue &                            queue,
    const mxfp4_moe_gateup_prepack_request & request,
    const std::vector<sycl::event> *         deps = nullptr);

}  // namespace ggml_sycl

// Prototype fused down projection + router weighted reduce.  Consumes the cached
// Q8 GLU artifact and writes the final MoE output directly, avoiding the
// normal down_dst + ADD_ID/MUL/ADD reduction chain.  SOA can write final
// directly; XMX_TILED uses down_tmp as a weighted projection scratch followed by
// a small top-k reduce.
int          ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env(const char * env);
int          ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env(const char * env,
                                                                             bool         is_down_role,
                                                                             int64_t      n_tokens);
const char * ggml_sycl_moe_down_sum_q8_soa_tg_variant_label(int rows_per_group);
int          ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env(const char * env);
int          ggml_sycl_moe_down_cached_q8_soa_tg_active_variant_from_env(const char * env,
                                                                         bool         is_down_role,
                                                                         bool         is_soa_layout,
                                                                         int64_t      n_tokens);
const char * ggml_sycl_moe_down_cached_q8_soa_tg_variant_label(int variant);
bool         ggml_sycl_moe_down_cached_q8_soa_tg_variant_vector_qs_load(int variant);
bool         ggml_sycl_moe_down_cached_q8_soa_tg_variant_cache_y_local(int variant);

bool mmvq_moe_batched_dispatch_down_sum_from_cached_q8_mxfp4(
    ggml_backend_sycl_context &             ctx,
    const ggml_tensor *                     down_weight,
    const ggml_tensor *                     glu_src,
    ggml_tensor *                           final_dst,
    ggml_tensor *                           down_tmp,
    const void * const *                    down_ptrs_device,
    const int32_t *                         ids_device,
    const ggml_tensor *                     moe_weights,
    const ggml_tensor *                     down_bias,
    int                                     n_gpu_entries,
    int64_t                                 n_ids,
    int64_t                                 ids_nb0,
    int64_t                                 ids_nb1,
    ggml_layout_mode                        down_layout,
    const ggml_sycl::mem_handle *           glu_src_handle_override,
    const ggml_sycl::mem_handle *           final_dst_handle_override,
    const ggml_sycl::mem_handle *           down_tmp_handle_override = nullptr,
    const int32_t *                         ids_host                 = nullptr,
    int64_t                                 ids_host_count           = 0,
    const mxfp4_moe_direct_final_metadata * direct_final_metadata    = nullptr,
    const std::vector<sycl::event> *        deps                     = nullptr,
    sycl::event *                           completion_event         = nullptr,
    bool *                                  completion_event_set     = nullptr);

bool ggml_sycl_moe_down_sum_cached_q8_capacity_ok(const ggml_tensor *           down_weight,
                                                  const ggml_sycl::mem_handle & glu_handle,
                                                  int                           device,
                                                  int64_t                       ncols_y,
                                                  int64_t                       total_batches);

bool ggml_sycl_moe_down_sum_xmx_direct_final_mxfp4(const ggml_tensor *              src0,
                                                   const ggml_tensor *              src1,
                                                   ggml_tensor *                    dst,
                                                   const void * const *             expert_ptrs,
                                                   const void *                     q8_rows,
                                                   float *                          dst_data,
                                                   const int32_t *                  token_ids,
                                                   const int32_t *                  expert_ids,
                                                   const int32_t *                  expert_ids_host,
                                                   const float *                    weights,
                                                   const float *                    bias,
                                                   int64_t                          entries,
                                                   int64_t                          n_ids,
                                                   int64_t                          n_tokens,
                                                   int64_t                          ids_nb0,
                                                   int64_t                          ids_nb1,
                                                   int64_t                          q8_nb11,
                                                   int64_t                          q8_nb12,
                                                   int64_t                          weights_nb1,
                                                   int64_t                          weights_nb2,
                                                   int64_t                          bias_nb1,
                                                   int64_t                          dst_token_stride,
                                                   ggml_layout_mode                 down_layout,
                                                   const std::vector<sycl::event> * deps,
                                                   sycl::event *                    completion_event,
                                                   sycl::queue &                    queue,
                                                   int                              device);

struct mmvq_moe_dispatch_timing {
    double activation_quant_us = 0.0;
    double batch_id_us         = 0.0;
    double kernel_submit_us    = 0.0;
    int    calls               = 0;
    int    entries             = 0;
};

void                     mmvq_moe_dispatch_timing_reset();
mmvq_moe_dispatch_timing mmvq_moe_dispatch_timing_consume();

// Convert reordered tensor to coalesced layout for better memory bandwidth
// Call this AFTER reorder_qw and BEFORE graph recording (at model load time)
// Returns true if conversion was performed, false if not enabled or wrong type
bool ggml_sycl_convert_to_coalesced_q4_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q8_0(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_q6_k(const ggml_tensor * tensor, dpct::queue_ptr stream);
bool ggml_sycl_convert_to_coalesced_mxfp4(const ggml_tensor * tensor, dpct::queue_ptr stream);

// =============================================================================
// Direct MMVQ kernel submission for persistent TG micro-graph
// =============================================================================
// These wrappers submit standalone MMVQ kernels to a queue without requiring
// ggml_tensor metadata.  They are used by the micro-graph persistent TG path
// to replace the generic dp4a compute_matmul_tile with the optimized MMVQ
// kernels that use pre-quantized Q8_1 activations.
//
// Weights must be in SOA (Structure-of-Arrays) layout.
// Activations (vy) must be in SOA Q8_1 format (quants at [0,K), ds at [K,...)).
// Use quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa> to produce vy.

// Q4_0 SOA MMVQ: weights in SOA layout, activations in SOA Q8_1
sycl::event mmvq_submit_q4_0_soa(sycl::queue &                    q,
                                 const void *                     weights_soa,
                                 const void *                     y_q8_soa,
                                 float *                          dst,
                                 int                              ncols,
                                 int                              nrows,
                                 int                              total_nrows,
                                 int                              row_low,
                                 const std::vector<sycl::event> * deps = nullptr);

// Q6_K SOA MMVQ: weights in SOA layout, activations in SOA Q8_1
sycl::event mmvq_submit_q6_k_soa(sycl::queue &                    q,
                                 const void *                     weights_soa,
                                 const void *                     y_q8_soa,
                                 float *                          dst,
                                 int                              ncols,
                                 int                              nrows,
                                 int                              total_nrows,
                                 int                              row_low,
                                 const std::vector<sycl::event> * deps = nullptr);

sycl::event mmvq_submit_mxfp4_soa(sycl::queue &                    q,
                                  const void *                     weights_soa,
                                  const void *                     y_q8_soa,
                                  float *                          dst,
                                  int                              ncols,
                                  int                              nrows,
                                  int                              total_nrows,
                                  int                              row_low,
                                  const std::vector<sycl::event> * deps = nullptr);

sycl::event mmvq_submit_mxfp4_soa_batched(sycl::queue &                    q,
                                          const void * const *             expert_ptrs_device,
                                          const void *                     y_q8_soa,
                                          const int32_t *                  ids_device,
                                          float *                          dst,
                                          int                              ncols,
                                          int                              nrows,
                                          int                              total_batches,
                                          int64_t                          q8_row_stride,
                                          int64_t                          dst_row_stride,
                                          const std::vector<sycl::event> * deps = nullptr);

sycl::event mmvq_submit_mxfp4_soa_pair_glu_batched(sycl::queue &                    q,
                                                   const void * const *             gate_ptrs_device,
                                                   const void * const *             up_ptrs_device,
                                                   const void *                     y_q8_soa,
                                                   const int32_t *                  ids_device,
                                                   const float *                    gate_bias_device,
                                                   const float *                    up_bias_device,
                                                   float *                          dst_glu,
                                                   int                              ncols,
                                                   int                              nrows,
                                                   int                              total_batches,
                                                   int64_t                          q8_row_stride,
                                                   int64_t                          dst_row_stride,
                                                   int64_t                          gate_bias_nb1,
                                                   int64_t                          up_bias_nb1,
                                                   int                              glu_op,
                                                   float                            alpha,
                                                   float                            limit,
                                                   const std::vector<sycl::event> * deps = nullptr);

// Coalesced MXFP4 MMVQ kernel submission
// vx: coalesced-layout weights (quants tiled word-major, then exponents)
// vy: SOA Q8_1 activations (quants[ncols] then ds[ncols/QK8_1])
void mmvq_submit_mxfp4_coalesced(sycl::queue & q, const void * vx, const void * vy, float * dst, int ncols, int nrows);

// Float-to-Q8_1 SOA quantization kernel submission for micro-graph
// Input:  x[ncols] float activations
// Output: y_q8[ncols + ncols/QK8_1 * 4] SOA Q8_1
sycl::event mmvq_submit_quantize_q8_1_soa(sycl::queue &                    q,
                                          const float *                    x,
                                          void *                           y_q8_soa,
                                          int                              ncols,
                                          const std::vector<sycl::event> * deps = nullptr);

// Compute SOA Q8_1 buffer size in bytes for a given K dimension
inline size_t mmvq_q8_1_soa_size(int K) {
    // quants: K bytes, ds: (K/QK8_1) * sizeof(half2) = K/32 * 4
    // Q6_K SOA kernel reads ds[iby + bq8_offset + 2*i] with max index = K/QK8_1 + 7
    // Add 8 half2 entries of overflow padding to prevent out-of-bounds read
    constexpr size_t Q6K_DS_OVERFLOW_PAD = 8 * sizeof(sycl::half2);  // 32 bytes
    return static_cast<size_t>(K) + (K / QK8_1) * sizeof(sycl::half2) + Q6K_DS_OVERFLOW_PAD;
}

#if defined(XMX_TEST_STANDALONE)
namespace ggml_sycl {
inline void test_moe_down_sum_direct_final_reduce_reference(const float * weighted_tmp,
                                                            float *       dst,
                                                            int           nrows,
                                                            int           n_ids,
                                                            int           n_tokens,
                                                            int64_t       tmp_nb1,
                                                            int64_t       tmp_nb2,
                                                            int64_t       dst_token_stride) {
    for (int token = 0; token < n_tokens; ++token) {
        for (int row = 0; row < nrows; ++row) {
            float acc = 0.0f;
            for (int id = 0; id < n_ids; ++id) {
                const float * src = reinterpret_cast<const float *>(reinterpret_cast<const char *>(weighted_tmp) +
                                                                    static_cast<int64_t>(id) * tmp_nb1 +
                                                                    static_cast<int64_t>(token) * tmp_nb2);
                acc += src[row];
            }
            *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + static_cast<int64_t>(row) * sizeof(float) +
                                       static_cast<int64_t>(token) * dst_token_stride) = acc;
        }
    }
}
}  // namespace ggml_sycl
#endif

#endif  // GGML_SYCL_MMVQ_HPP
