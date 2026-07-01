//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// Arguments for benchmarking MMVQ kernels directly (without ggml_tensor overhead)
struct mmvq_bench_args {
    // Required: SYCL queue for kernel submission
    sycl::queue * stream = nullptr;

    // Quantization type of weights (e.g., GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    ggml_type weight_type = GGML_TYPE_Q4_0;

    // Layout mode for weights
    ggml_layout_mode layout = GGML_LAYOUT_AOS;

    // Pointers to data buffers (device memory)
    const void * weights     = nullptr;  // Quantized weight tensor
    const void * layout_base = nullptr;  // Base pointer for SoA/coalesced layouts (nullptr for AOS)
    const void * activations = nullptr;  // Activation tensor (q8_1)
    float *      output      = nullptr;  // Output buffer

    // Matrix dimensions
    int64_t ncols = 0;  // Number of columns (K dimension)
    int64_t nrows = 0;  // Number of rows in weight matrix (output features)
    int64_t batch = 0;  // Batch size (number of input vectors)

    // Row range for partial computation
    int64_t row_low  = 0;  // Starting row (inclusive)
    int64_t row_high = 0;  // Ending row (exclusive)

    // Strides
    int64_t src1_padded_col_size = 0;  // Padded col size for activations
    int64_t dst_row_stride       = 0;  // Output row stride

    // Device ID (-1 for current device)
    int device_id = -1;
};

// Launch MMVQ kernel for benchmarking
// Returns true on success, false on invalid arguments
// If events is non-null, kernel events are appended for timing (if supported)
bool ggml_sycl_mmvq_bench_launch(const mmvq_bench_args & args, std::vector<sycl::event> * events);

// Arguments for benchmarking the production MXFP4 SOA MoE gate/up/GLU kernel
// shape directly.  The pointer tables are transient launch ABI payloads: each
// entry points at one resident SOA expert slice.
struct mxfp4_pair_glu_bench_args {
    sycl::queue * stream = nullptr;

    const void * const * gate_ptrs          = nullptr;
    const void * const * up_ptrs            = nullptr;
    const void *         activations_q8_soa = nullptr;
    float *              output             = nullptr;
    float *              gate_tmp           = nullptr;
    float *              up_tmp             = nullptr;
    void *               down_q8_soa        = nullptr;
    int8_t *             dpas_b_packed      = nullptr;
    float *              dpas_y_scales      = nullptr;
    const int32_t *      ids                = nullptr;
    const int32_t *      grouped_expert_ids = nullptr;
    const int32_t *      grouped_offsets    = nullptr;
    const int32_t *      grouped_row_slots  = nullptr;
    const int32_t *      grouped_chunks     = nullptr;
    const int32_t *      grouped_row_starts = nullptr;
    const float *        gate_bias          = nullptr;
    const float *        up_bias            = nullptr;

    int ncols            = 0;
    int ncols_y          = 0;
    int nrows_per_expert = 0;
    int num_experts      = 0;
    int n_ids            = 0;
    int n_tokens         = 1;
    int ne11             = 1;

    int64_t ids_nb0       = 0;
    int64_t ids_nb1       = 0;
    int64_t nb11          = 0;
    int64_t nb12          = 0;
    int64_t dst_nb1       = 0;
    int64_t dst_nb2       = 0;
    int64_t down_q8_nb11  = 0;
    int64_t gate_bias_nb1 = 0;
    int64_t up_bias_nb1   = 0;

    int   rows_per_wg              = 4;
    bool  cache_y                  = true;
    bool  direct_xmx               = false;
    bool  xmx_tiled                = false;
    bool  xmx_tiled_grouped        = false;
    bool  device_grouped           = false;
    bool  xmx_tiled_pack_q8        = false;
    bool  xmx_tiled_prefetch       = false;
    int   xmx_tiled_m_tiles        = 1;
    bool  xmx_tiled_v2             = false;
    int   xmx_tiled_v2_group_bytes = 320;
    bool  split_gate_up            = false;
    bool  single_column_gateup     = false;
    bool  multi_rhs_gateup         = false;
    int   multi_rhs_cols           = 1;
    bool  predecoded_i8            = false;
    int   xmx_tiles_n              = 1;
    bool  vector_qs_load           = false;
    bool  ignore_weight_scale      = false;
    int   scale_stride_blocks      = 0;
    int   subgroup_size            = 32;
    int   grouped_n_chunks         = 0;
    int   glu_op                   = 0;
    float alpha                    = 1.702f;
    float limit                    = 7.0f;
};

bool ggml_sycl_mxfp4_pair_glu_bench_launch(const mxfp4_pair_glu_bench_args & args);

// Arguments for benchmarking the current one-layer MXFP4 MoE decode sequence:
// gate/up/GLU, GLU->Q8_1 SOA artifact publication, then down projection. This
// is diagnostic-only; production ownership still flows through smart handles.
struct mxfp4_layer_glu_down_bench_args {
    sycl::queue * stream = nullptr;

    const void * const * gate_ptrs          = nullptr;
    const void * const * up_ptrs            = nullptr;
    const void * const * down_ptrs          = nullptr;
    const void *         activations_q8_soa = nullptr;
    float *              glu_f32            = nullptr;
    void *               down_q8_soa        = nullptr;
    float *              output             = nullptr;
    const int32_t *      ids                = nullptr;
    const float *        gate_bias          = nullptr;
    const float *        up_bias            = nullptr;

    int activation_cols        = 0;
    int intermediate_cols      = 0;
    int output_rows_per_expert = 0;
    int num_experts            = 0;
    int n_ids                  = 0;
    int n_tokens               = 1;
    int ne11                   = 1;

    int64_t ids_nb0         = 0;
    int64_t ids_nb1         = 0;
    int64_t activation_nb11 = 0;
    int64_t activation_nb12 = 0;
    int64_t glu_nb1         = 0;
    int64_t glu_nb2         = 0;
    int64_t down_q8_nb11    = 0;
    int64_t down_q8_nb12    = 0;
    int64_t output_nb1      = 0;
    int64_t output_nb2      = 0;
    int64_t gate_bias_nb1   = 0;
    int64_t up_bias_nb1     = 0;

    int   rows_per_wg         = 4;
    bool  cache_y             = true;
    bool  xmx_tiled_gate_up   = false;
    bool  xmx_tiled_grouped   = false;
    bool  xmx_tiled_pack_q8   = false;
    bool  xmx_tiled_prefetch  = false;
    int   xmx_tiled_m_tiles   = 1;
    int   xmx_tiles_n         = 1;
    bool  vector_qs_load      = false;
    bool  ignore_weight_scale = false;
    int   scale_stride_blocks = 0;
    int   subgroup_size       = 32;
    int   grouped_n_chunks    = 0;
    int   glu_op              = 0;
    float alpha               = 1.702f;
    float limit               = 7.0f;

    const int32_t * grouped_expert_ids = nullptr;
    const int32_t * grouped_offsets    = nullptr;
    const int32_t * grouped_row_slots  = nullptr;
    const int32_t * grouped_chunks     = nullptr;
    const int32_t * grouped_row_starts = nullptr;
    int8_t *        dpas_b_packed      = nullptr;
    float *         dpas_y_scales      = nullptr;
};

bool ggml_sycl_mxfp4_layer_glu_down_bench_launch(const mxfp4_layer_glu_down_bench_args & args);

// Arguments for benchmarking the production MXFP4 SOA MoE selected-expert
// single-role kernel, matching the down projection shape.
struct mxfp4_mmv_id_bench_args {
    sycl::queue * stream = nullptr;

    const void * const * expert_ptrs        = nullptr;
    const void *         activations_q8_soa = nullptr;
    float *              output             = nullptr;
    const int32_t *      ids                = nullptr;

    int ncols            = 0;
    int ncols_y          = 0;
    int nrows_per_expert = 0;
    int num_experts      = 0;
    int n_ids            = 0;
    int n_tokens         = 1;
    int ne11             = 1;

    int64_t ids_nb0 = 0;
    int64_t ids_nb1 = 0;
    int64_t nb11    = 0;
    int64_t nb12    = 0;
    int64_t dst_nb1 = 0;
    int64_t dst_nb2 = 0;

    int  rows_per_wg         = 4;
    bool cache_y             = false;
    bool vector_qs_load      = false;
    bool ignore_weight_scale = false;
    int  scale_stride_blocks = 0;
    int  subgroup_size       = 32;
};

bool ggml_sycl_mxfp4_mmv_id_bench_launch(const mxfp4_mmv_id_bench_args & args);
bool ggml_sycl_mxfp4_mmv_id_predecoded_bench_launch(const mxfp4_mmv_id_bench_args & args);

// Arguments for benchmarking a DPAS-ready MXFP4 selected-expert down role.
// Each expert pointer references an XMX-tiled layout materialized outside the
// hot path: per K-block, per output tile, scales followed by packed qs.
struct mxfp4_mmv_id_xmx_tiled_bench_args {
    sycl::queue * stream = nullptr;

    const void * const * expert_ptrs        = nullptr;
    const void *         activations_q8_soa = nullptr;
    float *              output             = nullptr;
    const int32_t *      ids                = nullptr;
    int8_t *             dpas_b_packed      = nullptr;
    float *              dpas_y_scales      = nullptr;
    const int32_t *      grouped_expert_ids = nullptr;
    const int32_t *      grouped_offsets    = nullptr;
    const int32_t *      grouped_row_slots  = nullptr;
    const int32_t *      grouped_chunks     = nullptr;
    const int32_t *      grouped_row_starts = nullptr;

    int  ncols            = 0;
    int  ncols_y          = 0;
    int  nrows_per_expert = 0;
    int  num_experts      = 0;
    int  n_ids            = 0;
    int  n_tokens         = 1;
    int  ne11             = 1;
    int  xmx_tiles_n      = 4;
    bool raw_accum        = false;
    bool i8_rowmajor      = false;
    int  grouped_n_chunks = 0;

    int64_t ids_nb0 = 0;
    int64_t ids_nb1 = 0;
    int64_t nb11    = 0;
    int64_t nb12    = 0;
    int64_t dst_nb1 = 0;
    int64_t dst_nb2 = 0;
};

bool ggml_sycl_mxfp4_mmv_id_xmx_tiled_bench_launch(const mxfp4_mmv_id_xmx_tiled_bench_args & args);

// Arguments for benchmarking MMQ kernels directly (without ggml_tensor overhead)
struct mmq_bench_args {
    // Required: SYCL queue for kernel submission
    sycl::queue * stream = nullptr;

    // Quantization type of weights (e.g., GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    ggml_type weight_type = GGML_TYPE_Q4_0;

    // Layout mode for weights
    ggml_layout_mode layout = GGML_LAYOUT_AOS;

    // Pointers to data buffers (device memory)
    const void * weights     = nullptr;  // Quantized weight tensor
    const void * layout_base = nullptr;  // Base pointer for SoA/coalesced layouts (nullptr for AOS)
    const void * activations = nullptr;  // Activation tensor (f32 or quantized)
    float *      output      = nullptr;  // Output buffer

    // Matrix dimensions
    int64_t ncols = 0;  // Number of columns (K dimension)
    int64_t nrows = 0;  // Number of rows in weight matrix (output features)
    int64_t batch = 0;  // Batch size (number of input vectors)

    // Row range for partial computation
    int64_t row_low  = 0;  // Starting row (inclusive)
    int64_t row_high = 0;  // Ending row (exclusive)

    // Strides
    int64_t src1_padded_row_size = 0;  // Padded row size for activations
    int64_t dst_row_stride       = 0;  // Output row stride

    // Device ID (-1 for current device)
    int device_id = -1;
};

// Launch MMQ kernel for benchmarking
// Returns true on success, false on invalid arguments
// If events is non-null, kernel events are appended for timing
bool ggml_sycl_mmq_bench_launch(const mmq_bench_args & args, std::vector<sycl::event> * events);

}  // namespace ggml_sycl
