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
#include <cstdint>

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

enum class ggml_sycl_fattn_xmx_decode_kv_layout_kind {
    PUBLIC_CONTIGUOUS,
    PACKED_K_MEM_HANDLE,
    REJECT,
};

enum class ggml_sycl_fattn_xmx_decode_kv_layout_reason {
    OK,
    NOT_SINGLE_QUERY,
    FP8_KV_UNSUPPORTED,
    PAGED_UNSUPPORTED,
    MULTI_SEQ_UNSUPPORTED,
    MULTI_TOKEN_DECODE_UNSUPPORTED,
    HEAD_DIM_UNSUPPORTED,
    HEAD_RATIO_UNSUPPORTED,
    GQA_RATIO_UNSUPPORTED,
    Q_TYPE_UNSUPPORTED,
    K_TYPE_UNSUPPORTED,
    V_TYPE_UNSUPPORTED,
    KV_NOT_DEVICE_RESIDENT,
    DEVICE_XMX_M1N64_UNSUPPORTED,
    LOCAL_MEM_UNSUPPORTED,
};

struct ggml_sycl_fattn_xmx_decode_kv_caps {
    bool   m1n64_k16_supported = false;
    bool   m1n64_k32_supported = false;
    size_t local_mem_size      = 0;
    bool   k_device_resident   = false;
    bool   v_device_resident   = false;
};

struct ggml_sycl_fattn_xmx_decode_kv_layout_plan {
    ggml_sycl_fattn_xmx_decode_kv_layout_kind   kind;
    ggml_sycl_fattn_xmx_decode_kv_layout_reason reason;
    int                                         D;
    int                                         H_q;
    int                                         H_kv;
    int                                         n_rep;
    int                                         preferred_tk;
    int                                         alternate_tk;
    int                                         kv_block_tokens;
    size_t                                      required_slm_bytes;
    size_t                                      source_k_bytes_per_block;
    size_t                                      packed_k_bytes_per_block;
    size_t                                      packed_k_overhead_per_block;
};

static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_D            = 64;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS       = 64;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_TN           = 64;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_SG           = 16;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_ACTIVE_LANES = 8;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_SLOTS =
    GGML_SYCL_FATTN_XMX_PACKED_K_TN / GGML_SYCL_FATTN_XMX_PACKED_K_SG;
static constexpr int GGML_SYCL_FATTN_XMX_PACKED_K_HALF_TOKENS =
    GGML_SYCL_FATTN_XMX_PACKED_K_ACTIVE_LANES * GGML_SYCL_FATTN_XMX_PACKED_K_SLOTS;
static constexpr size_t GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_HALF =
    (GGML_SYCL_FATTN_XMX_PACKED_K_D / 2) * (GGML_SYCL_FATTN_XMX_PACKED_K_TN * 2);
static constexpr size_t GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_BLOCK = 2 * GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_HALF;
static constexpr size_t GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK =
    GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_BLOCK * sizeof(sycl::half);

static inline size_t ggml_sycl_fattn_xmx_packed_k_element_offset_half(int kv_local, int d) {
    if (kv_local < 0 || kv_local >= GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS || d < 0 ||
        d >= GGML_SYCL_FATTN_XMX_PACKED_K_D) {
        return SIZE_MAX;
    }

    const int half_id     = kv_local / GGML_SYCL_FATTN_XMX_PACKED_K_HALF_TOKENS;
    const int compact_col = kv_local - half_id * GGML_SYCL_FATTN_XMX_PACKED_K_HALF_TOKENS;
    const int active_lane = compact_col % GGML_SYCL_FATTN_XMX_PACKED_K_ACTIVE_LANES;
    const int slot        = compact_col / GGML_SYCL_FATTN_XMX_PACKED_K_ACTIVE_LANES;
    const int active_col  = slot * GGML_SYCL_FATTN_XMX_PACKED_K_SG + active_lane;

    return static_cast<size_t>(half_id) * GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_HALF +
           static_cast<size_t>(d / 2) * (GGML_SYCL_FATTN_XMX_PACKED_K_TN * 2) +
           static_cast<size_t>(active_col * 2 + (d & 1));
}

struct ggml_sycl_fattn_xmx_packed_k_materialization_desc {
    bool required      = false;
    int  target_device = -1;
    int  D             = 0;
    int  n_kv          = 0;
    int  H_kv          = 0;
    int  batch         = 0;
    int  preferred_tk  = 0;
    int  alternate_tk  = 0;
    int  block_tokens  = 0;
    int  n_blocks      = 0;

    size_t source_k_bytes_per_block    = 0;
    size_t packed_k_bytes_per_block    = 0;
    size_t packed_k_overhead_per_block = 0;
    size_t total_blocks                = 0;
    size_t total_packed_bytes          = 0;

    int64_t k_src_nb1 = 0;
    int64_t k_src_nb2 = 0;
    int64_t k_src_nb3 = 0;

    int64_t packed_block_stride = 0;
    int64_t packed_head_stride  = 0;
    int64_t packed_batch_stride = 0;
};

struct ggml_sycl_fattn_xmx_packed_k {
    ggml_sycl::alloc_handle alloc{};
    ggml_sycl::mem_handle   handle{};
    sycl::event             ready_event{};

    int    device      = -1;
    int    D           = 0;
    int    n_kv        = 0;
    int    H_kv        = 0;
    int    batch       = 0;
    int    n_blocks    = 0;
    size_t total_bytes = 0;

    ggml_sycl_fattn_xmx_packed_k() = default;
    ~ggml_sycl_fattn_xmx_packed_k();

    ggml_sycl_fattn_xmx_packed_k(const ggml_sycl_fattn_xmx_packed_k &)             = delete;
    ggml_sycl_fattn_xmx_packed_k & operator=(const ggml_sycl_fattn_xmx_packed_k &) = delete;

    ggml_sycl_fattn_xmx_packed_k(ggml_sycl_fattn_xmx_packed_k && other) noexcept;
    ggml_sycl_fattn_xmx_packed_k & operator=(ggml_sycl_fattn_xmx_packed_k && other) noexcept;

    void reset();
};

// Plan the future XMX decode KV layout using descriptor and device capability
// facts only.  PACKED_K_MEM_HANDLE means the planner must materialize a packed
// K allocation owned by a smart mem_handle before dispatch/graph recording;
// this function does not allocate and default dispatch remains unchanged.
ggml_sycl_fattn_xmx_decode_kv_layout_plan ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(
    const fattn_params &                       params,
    int                                        D,
    const ggml_sycl_fattn_xmx_decode_kv_caps & caps);

ggml_sycl_fattn_xmx_decode_kv_layout_plan ggml_sycl_flash_attn_ext_xmx_decode_kv_layout_plan(
    const fattn_params & params,
    int                  D,
    const sycl::device & dev,
    bool                 k_device_resident,
    bool                 v_device_resident);

bool ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
    const fattn_params &                                params,
    const ggml_sycl_fattn_xmx_decode_kv_layout_plan &   plan,
    int                                                 target_device,
    ggml_sycl_fattn_xmx_packed_k_materialization_desc * out);

size_t ggml_sycl_fattn_xmx_packed_k_block_offset_bytes(const ggml_sycl_fattn_xmx_packed_k_materialization_desc & desc,
                                                       int sequence,
                                                       int kv_head,
                                                       int block);

bool ggml_sycl_fattn_xmx_materialize_packed_k(const fattn_params &                              params,
                                              const ggml_sycl_fattn_xmx_decode_kv_layout_plan & plan,
                                              int                                               target_device,
                                              dpct::queue_ptr                                   stream,
                                              ggml_sycl_fattn_xmx_packed_k *                    out);

ggml_sycl_fattn_xmx_packed_k * ggml_sycl_fattn_xmx_find_packed_k_sidecar(const fattn_params & params,
                                                                         int                  target_device);

bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows(const ggml_tensor * dst,
                                                       const ggml_tensor * src0,
                                                       const ggml_tensor * src1,
                                                       int                 target_device,
                                                       const void *        src0_ptr,
                                                       const void *        index_ptr,
                                                       dpct::queue_ptr     stream,
                                                       const sycl::event & set_rows_event,
                                                       sycl::event *       out_event);

void ggml_sycl_fattn_xmx_unregister_packed_k_range(const void * ptr, size_t size);

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
    MAX_BIAS_UNSUPPORTED,
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
