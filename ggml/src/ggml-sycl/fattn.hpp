//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"
#include "fattn-common.hpp"

#include <cstddef>

// Check if flash attention is supported for the given tensor configuration
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

// Execute flash attention operation
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

// Pre-allocate V2 partition buffers before SYCL graph recording.
// This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
// Should be called before graph recording starts.
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);

enum class ggml_sycl_fattn_decode_policy_reason {
    OK,
    NOT_SINGLE_QUERY,
    FP8_KV_UNSUPPORTED,
    SINKS_UNSUPPORTED,
    SOFTCAP_UNSUPPORTED,
    ALIBI_UNSUPPORTED,
    PAGED_UNSUPPORTED,
    MULTI_SEQ_UNSUPPORTED,
    SEQ_IDS_UNSUPPORTED,
    MULTI_TOKEN_DECODE_UNSUPPORTED,
    HEAD_DIM_UNSUPPORTED,
    HEAD_RATIO_UNSUPPORTED,
    Q_TYPE_UNSUPPORTED,
    K_TYPE_UNSUPPORTED,
    V_TYPE_UNSUPPORTED,
};

struct ggml_sycl_fattn_decode_policy {
    bool                                 fast_esimd_safe;
    ggml_sycl_fattn_decode_policy_reason reason;
};

// Decide whether single-token FA decode may bypass the conservative safe-decode
// fallback and use the faster ESIMD f16 path. This is intentionally a
// descriptor/capability policy, not a device-name or model-name policy.
ggml_sycl_fattn_decode_policy ggml_sycl_fattn_fast_decode_policy(const fattn_params & params, int D);

#if GGML_SYCL_DNNL
enum class ggml_sycl_onednn_fa_layout_kind {
    DIRECT,
    MATERIALIZE_REQUIRED,
    REJECT,
};

enum class ggml_sycl_onednn_fa_layout_reason {
    OK,
    BELOW_MIN_NCOLS,
    SINKS_UNSUPPORTED,
    SOFTCAP_UNSUPPORTED,
    FP8_KV_UNSUPPORTED,
    MULTI_SEQ_UNSUPPORTED,
    BATCH_UNSUPPORTED,
    PAGED_UNSUPPORTED,
    UNSUPPORTED_D,
    EMPTY_KV,
    KV_NC_STRIDE_MISMATCH,
    HEAD_RATIO_UNSUPPORTED,
    Q_TYPE_UNSUPPORTED,
    K_TYPE_UNSUPPORTED,
    V_TYPE_UNSUPPORTED,
    MASK_TYPE_UNSUPPORTED,
};

struct ggml_sycl_onednn_fa_layout_plan {
    ggml_sycl_onednn_fa_layout_kind   kind;
    ggml_sycl_onednn_fa_layout_reason reason;
};

// Plan whether a flash_attn_ext call can use oneDNN graph SDPA directly,
// needs a planner-owned K/V materialization first, or must fall back.
ggml_sycl_onednn_fa_layout_plan ggml_sycl_flash_attn_ext_onednn_plan(const fattn_params & params,
                                                                     int                  H_q,
                                                                     int                  H_kv,
                                                                     bool                 kv_is_fp8,
                                                                     bool                 multi_seq);

// Planner-owned K/V materialization contract for oneDNN FA:
// - Source: f16 K/V tensors in ggml FA layout [D, n_kv, H_kv, batch] with byte
//   strides from fattn_params::nb11/nb12/nb13 and nb21/nb22/nb23.
// - Target: f16 dense oneDNN-compatible K/V tensors on target_device with
//   token stride D*sizeof(f16), head stride n_kv*D*sizeof(f16), batch=1.
// - Ownership/lifetime: the executable materializer allocates target buffers
//   through unified_alloc/scoped_unified_alloc and exposes them as mem_handle
//   views. The result object must stay alive through oneDNN execute.
// - Graph/replay: oneDNN graph execution is disabled during SYCL graph
//   recording, so materialization is not attempted while recording.
// - Fallback: descriptor creation returns false for unsupported layouts/types;
//   allocation or repack failure returns false and the caller falls back to
//   native FA without executing oneDNN.
struct ggml_sycl_onednn_fa_materialization_desc {
    bool   required         = false;
    int    target_device    = -1;
    int    D                = 0;
    int    n_kv             = 0;
    int    H_q              = 0;
    int    H_kv             = 0;
    int    n_rep            = 1;
    size_t bytes_per_tensor = 0;

    int64_t k_src_nb1 = 0;
    int64_t k_src_nb2 = 0;
    int64_t k_src_nb3 = 0;
    int64_t v_src_nb1 = 0;
    int64_t v_src_nb2 = 0;
    int64_t v_src_nb3 = 0;

    int64_t k_target_nb1 = 0;
    int64_t k_target_nb2 = 0;
    int64_t k_target_nb3 = 0;
    int64_t v_target_nb1 = 0;
    int64_t v_target_nb2 = 0;
    int64_t v_target_nb3 = 0;
};

bool ggml_sycl_flash_attn_ext_onednn_materialization_desc(const fattn_params &                       params,
                                                          int                                        H_q,
                                                          int                                        H_kv,
                                                          int                                        target_device,
                                                          ggml_sycl_onednn_fa_materialization_desc * out);

size_t ggml_sycl_onednn_fa_materialized_src_offset(const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                   bool                                             value_tensor,
                                                   int                                              h,
                                                   int                                              t,
                                                   int                                              d);

size_t ggml_sycl_onednn_fa_materialized_dst_offset(const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                   int                                              h,
                                                   int                                              t,
                                                   int                                              d);

// Check whether a flash_attn_ext call is eligible for the oneDNN graph SDPA path.
bool ggml_sycl_flash_attn_ext_onednn_eligible(const fattn_params & params,
                                              int                  H_q,
                                              int                  H_kv,
                                              bool                 kv_is_fp8,
                                              bool                 multi_seq);
#endif

#endif  // GGML_SYCL_FATTN_HPP
