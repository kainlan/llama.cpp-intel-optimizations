#pragma once

#include "../../data_generators.hpp"
#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace sycl_bench {

struct ReferenceMetrics {
    double total_us             = 0.0;
    double dequant_us           = 0.0;
    double gemm_us              = 0.0;
    double scale_us             = 0.0;
    double tflops               = 0.0;
    double tops                 = 0.0;
    double bandwidth_gbps       = 0.0;
    double arithmetic_intensity = 0.0;
};

inline bool dequantize_row_fp32(ggml_type type, const void * src, float * dst, int64_t k, std::string & error) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0((const block_q4_0 *) src, dst, k);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_row_q4_1((const block_q4_1 *) src, dst, k);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0((const block_q5_0 *) src, dst, k);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_row_q5_1((const block_q5_1 *) src, dst, k);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0((const block_q8_0 *) src, dst, k);
            break;
        case GGML_TYPE_MXFP4:
            dequantize_row_mxfp4((const block_mxfp4 *) src, dst, k);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K((const block_q2_K *) src, dst, k);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K((const block_q3_K *) src, dst, k);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K((const block_q4_K *) src, dst, k);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K((const block_q5_K *) src, dst, k);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K((const block_q6_K *) src, dst, k);
            break;
        default:
            error = "unsupported quant type for dequantize";
            return false;
    }
    return true;
}

bool run_onednn_fp16_gemm(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n,
                          int64_t                      k,
                          ggml_type                    quant_type,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error);

bool run_onednn_int8_gemm(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n,
                          int64_t                      k,
                          ggml_type                    quant_type,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error);

bool run_onednn_woq_gemm(const GeneratedWeights &     weights,
                         const GeneratedActivations & activations,
                         int64_t                      m,
                         int64_t                      n,
                         int64_t                      k,
                         ggml_type                    quant_type,
                         int                          warmup,
                         int                          iterations,
                         sycl::queue &                queue,
                         ReferenceMetrics &           out,
                         std::string &                error);

bool run_onednn_mxfp4_gemm(const GeneratedWeights &     weights,
                           const GeneratedActivations & activations,
                           int64_t                      m,
                           int64_t                      n,
                           int64_t                      k,
                           int                          warmup,
                           int                          iterations,
                           bool                         validate,
                           int                          scale_mode,
                           sycl::queue &                queue,
                           ReferenceMetrics &           out,
                           std::string &                error);

bool run_memory_bandwidth(size_t             bytes,
                          int                warmup,
                          int                iterations,
                          sycl::queue &      queue,
                          ReferenceMetrics & out,
                          std::string &      error);

bool run_mxfp4_decode_bandwidth(const GeneratedWeights & weights,
                                int64_t                  m,
                                int64_t                  k,
                                ggml_layout_mode         layout,
                                bool                     output_f16,
                                int                      warmup,
                                int                      iterations,
                                sycl::queue &            queue,
                                ReferenceMetrics &       out,
                                std::string &            error);

bool run_mxfp4_inline_dot(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n,
                          int64_t                      k,
                          ggml_layout_mode             layout,
                          bool                         validate,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error);

bool run_mxfp4_selected_read(const GeneratedWeights & weights,
                             int64_t                  m,
                             int64_t                  n_selected,
                             int64_t                  k,
                             ggml_layout_mode         layout,
                             bool                     interleave_rows,
                             bool                     validate,
                             int                      warmup,
                             int                      iterations,
                             sycl::queue &            queue,
                             ReferenceMetrics &       out,
                             std::string &            error);

bool run_mxfp4_selected_kmajor(const GeneratedWeights &     weights,
                               const GeneratedActivations & activations,
                               int64_t                      m,
                               int64_t                      n_selected,
                               int64_t                      k,
                               bool                         pair_glu,
                               bool                         tile_read,
                               bool                         validate,
                               int                          warmup,
                               int                          iterations,
                               sycl::queue &                queue,
                               ReferenceMetrics &           out,
                               std::string &                error);

bool run_mxfp4_selected_xmx_dpas_tile(const GeneratedWeights &     weights,
                                      const GeneratedActivations & activations,
                                      int64_t                      m,
                                      int64_t                      n_selected,
                                      int64_t                      k,
                                      int                          requested_tiles_n,
                                      int                          m_tiles_per_work_item,
                                      bool                         validate,
                                      int                          warmup,
                                      int                          iterations,
                                      sycl::queue &                queue,
                                      ReferenceMetrics &           out,
                                      std::string &                error);

bool run_mxfp4_pair_glu(const GeneratedWeights &     weights,
                        const GeneratedActivations & activations,
                        int64_t                      m,
                        int64_t                      n_selected,
                        int64_t                      k,
                        int64_t                      n_tokens,
                        int                          rows_per_wg,
                        bool                         cache_y,
                        bool                         direct_xmx,
                        bool                         xmx_tiled,
                        bool                         xmx_tiled_grouped,
                        bool                         xmx_tiled_pack_q8,
                        bool                         xmx_tiled_prefetch,
                        bool                         xmx_tiled_loadv2,
                        int                          xmx_tiled_m_tiles,
                        bool                         xmx_tiled_v2,
                        int                          xmx_tiled_v2_group_bytes,
                        bool                         xmx_tiled_bundle4,
                        int                          xmx_tiled_bundle4_group_bytes,
                        bool                         split_gate_up,
                        bool                         single_column_gateup,
                        bool                         multi_rhs_gateup,
                        int                          multi_rhs_cols,
                        bool                         predecoded_i8,
                        int                          xmx_tiles_n,
                        bool                         vector_qs_load,
                        bool                         ignore_weight_scale,
                        int                          scale_stride_blocks,
                        int                          subgroup_size,
                        bool                         sparse_expert_slots,
                        bool                         use_bias,
                        bool                         validate,
                        int                          warmup,
                        int                          iterations,
                        sycl::queue &                queue,
                        ReferenceMetrics &           out,
                        std::string &                error);

inline bool run_mxfp4_pair_glu(const GeneratedWeights &     weights,
                               const GeneratedActivations & activations,
                               int64_t                      m,
                               int64_t                      n_selected,
                               int64_t                      k,
                               int64_t                      n_tokens,
                               int                          rows_per_wg,
                               bool                         cache_y,
                               bool                         direct_xmx,
                               bool                         xmx_tiled,
                               bool                         xmx_tiled_grouped,
                               bool                         xmx_tiled_pack_q8,
                               bool                         xmx_tiled_prefetch,
                               int                          xmx_tiled_m_tiles,
                               bool                         xmx_tiled_v2,
                               int                          xmx_tiled_v2_group_bytes,
                               bool                         xmx_tiled_bundle4,
                               int                          xmx_tiled_bundle4_group_bytes,
                               bool                         split_gate_up,
                               bool                         single_column_gateup,
                               bool                         multi_rhs_gateup,
                               int                          multi_rhs_cols,
                               bool                         predecoded_i8,
                               int                          xmx_tiles_n,
                               bool                         vector_qs_load,
                               bool                         ignore_weight_scale,
                               int                          scale_stride_blocks,
                               int                          subgroup_size,
                               bool                         sparse_expert_slots,
                               bool                         use_bias,
                               bool                         validate,
                               int                          warmup,
                               int                          iterations,
                               sycl::queue &                queue,
                               ReferenceMetrics &           out,
                               std::string &                error) {
    return run_mxfp4_pair_glu(weights, activations, m, n_selected, k, n_tokens, rows_per_wg, cache_y, direct_xmx,
                              xmx_tiled, xmx_tiled_grouped, xmx_tiled_pack_q8, xmx_tiled_prefetch,
                              /*xmx_tiled_loadv2=*/false, xmx_tiled_m_tiles, xmx_tiled_v2,
                              xmx_tiled_v2_group_bytes, xmx_tiled_bundle4, xmx_tiled_bundle4_group_bytes,
                              split_gate_up, single_column_gateup, multi_rhs_gateup, multi_rhs_cols, predecoded_i8,
                              xmx_tiles_n, vector_qs_load, ignore_weight_scale, scale_stride_blocks, subgroup_size,
                              sparse_expert_slots, use_bias, validate, warmup, iterations, queue, out, error);
}

bool run_mxfp4_layer_glu_down(const GeneratedWeights &     weights,
                              const GeneratedActivations & activations,
                              int64_t                      m,
                              int64_t                      n_selected,
                              int64_t                      k,
                              int64_t                      n_tokens,
                              int                          rows_per_wg,
                              bool                         cache_y,
                              bool                         xmx_tiled_gate_up,
                              bool                         xmx_tiled_grouped,
                              bool                         xmx_tiled_pack_q8,
                              bool                         xmx_tiled_prefetch,
                              int                          xmx_tiled_m_tiles,
                              int                          xmx_tiles_n,
                              bool                         vector_qs_load,
                              bool                         ignore_weight_scale,
                              int                          scale_stride_blocks,
                              int                          subgroup_size,
                              bool                         sparse_expert_slots,
                              bool                         use_bias,
                              bool                         validate,
                              int                          warmup,
                              int                          iterations,
                              sycl::queue &                queue,
                              ReferenceMetrics &           out,
                              std::string &                error);

bool run_mxfp4_mmv_id(const GeneratedWeights &     weights,
                      const GeneratedActivations & activations,
                      int64_t                      m,
                      int64_t                      n_selected,
                      int64_t                      k,
                      int64_t                      n_tokens,
                      int                          rows_per_wg,
                      bool                         cache_y,
                      bool                         predecoded_i8,
                      bool                         validate,
                      bool                         vector_qs_load,
                      bool                         ignore_weight_scale,
                      int                          scale_stride_blocks,
                      int                          subgroup_size,
                      bool                         sparse_expert_slots,
                      int                          warmup,
                      int                          iterations,
                      sycl::queue &                queue,
                      ReferenceMetrics &           out,
                      std::string &                error);

bool run_mxfp4_mmv_id_f32(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n_selected,
                          int64_t                      k,
                          int64_t                      n_tokens,
                          bool                         ignore_weight_scale,
                          int                          scale_stride_blocks,
                          int                          subgroup_size,
                          bool                         sparse_expert_slots,
                          bool                         validate,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error);

bool run_mxfp4_mmv_id_xmx_tiled(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                int64_t                      m,
                                int64_t                      n_selected,
                                int64_t                      k,
                                int64_t                      n_tokens,
                                int                          requested_tiles_n,
                                bool                         sparse_expert_slots,
                                bool                         raw_accum,
                                bool                         i8_rowmajor,
                                bool                         validate,
                                int                          warmup,
                                int                          iterations,
                                sycl::queue &                queue,
                                ReferenceMetrics &           out,
                                std::string &                error);

bool run_mxfp4_dpas_grouped_raw(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                int64_t                      m,
                                int64_t                      n,
                                int64_t                      k,
                                int                          n_tile_repeats,
                                bool                         use_prefetch4,
                                bool                         validate,
                                int                          warmup,
                                int                          iterations,
                                sycl::queue &                queue,
                                ReferenceMetrics &           out,
                                std::string &                error);

bool run_mxfp4_dpas_grouped_compact_raw(const GeneratedWeights &     weights,
                                        const GeneratedActivations & activations,
                                        int64_t                      m,
                                        int64_t                      n,
                                        int64_t                      k,
                                        int                          n_tile_repeats,
                                        bool                         validate,
                                        int                          warmup,
                                        int                          iterations,
                                        sycl::queue &                queue,
                                        ReferenceMetrics &           out,
                                        std::string &                error);

bool run_mxfp4_dpas_grouped_compact_scaled(const GeneratedWeights &     weights,
                                           const GeneratedActivations & activations,
                                           int64_t                      m,
                                           int64_t                      n,
                                           int64_t                      k,
                                           int                          n_tile_repeats,
                                           bool                         packed_scales,
                                           bool                         validate,
                                           int                          warmup,
                                           int                          iterations,
                                           sycl::queue &                queue,
                                           ReferenceMetrics &           out,
                                           std::string &                error);

bool run_mxfp4_dpas_grouped_compact_bytescale(const GeneratedWeights &     weights,
                                              const GeneratedActivations & activations,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      k,
                                              int                          n_tile_repeats,
                                              bool                         xmx_layout,
                                              bool                         validate,
                                              int                          warmup,
                                              int                          iterations,
                                              sycl::queue &                queue,
                                              ReferenceMetrics &           out,
                                              std::string &                error);

bool run_mxfp4_dpas_grouped_scaled_predecoded(const GeneratedWeights &     weights,
                                              const GeneratedActivations & activations,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      k,
                                              int                          n_tile_repeats,
                                              bool                         packed_scales,
                                              bool                         validate,
                                              int                          warmup,
                                              int                          iterations,
                                              sycl::queue &                queue,
                                              ReferenceMetrics &           out,
                                              std::string &                error);

bool run_roofline_compute(size_t             elements,
                          int                ops_per_element,
                          int                warmup,
                          int                iterations,
                          sycl::queue &      queue,
                          ReferenceMetrics & out,
                          std::string &      error);

}  // namespace sycl_bench
