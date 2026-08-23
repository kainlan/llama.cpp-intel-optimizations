//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONVERT_HPP
#define GGML_SYCL_CONVERT_HPP

#include "common.hpp"

template <typename T>
using to_t_sycl_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float>      to_fp32_sycl_t;
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;

// full_tensor: Set to false when processing row slices to disable SoA-aware kernels
// (SoA kernels compute d_offset from k, which is wrong for row slices)
to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor = true);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor = true);

#ifdef GGML_SYCL_HAS_BF16
typedef to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

typedef to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 inline dst_t ggml_sycl_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
#ifdef GGML_SYCL_HAS_BF16
    } else if constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::bfloat16(float(x));
    } else if constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        return static_cast<float>(x);
#endif
    } else if constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        return x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } else if constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        return {x.x, x.y};
#endif
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}


// =============================================================================
// Q4_0 Coalesced Layout Reorder Functions
// =============================================================================
// These functions convert Q4_0 data between different memory layouts:
//
// AoS (Array of Structures): Original block_q4_0 format
//   [block0: d,qs[16]][block1: d,qs[16]]...
//
// Coalesced: Warp-optimized layout for GPU memory access
//   Tiles of MMVQ_COALESCED_TILE_BLOCKS blocks with interleaved qs for coalesced access:
//   [tile0: qs interleaved, d values][tile1...]
//
// SoA (Structure of Arrays): Intermediate format
//   [all qs bytes contiguous][all d values contiguous]
// =============================================================================

// Convert Q4_0 from AoS (block_q4_0) to SoA layout
void reorder_q4_0_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q8_0 from AoS (block_q8_0) to SoA layout
void reorder_q8_0_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q4_K from AoS (block_q4_K) to SoA layout
void reorder_q4_k_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t nblocks,
    dpct::queue_ptr stream);

// Convert Q6_K from AoS (block_q6_K) to SoA layout
void reorder_q6_k_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t nblocks,
    dpct::queue_ptr stream);

// Convert MXFP4 from AoS (block_mxfp4) to SoA layout
void reorder_mxfp4_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q4_0 from AoS (block_q4_0) to Coalesced layout
void reorder_q4_0_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,  // number of elements per row
    int64_t ne01,  // number of rows
    dpct::queue_ptr stream);

// Convert Q8_0 from AoS (block_q8_0) to Coalesced layout
void reorder_q8_0_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert MXFP4 from AoS (block_mxfp4) to Coalesced layout
void reorder_mxfp4_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q6_K from AoS (block_q6_K) to Coalesced layout
void reorder_q6_k_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Dequantize Q4_0 COALESCED→row-major FP16 (for oneDNN PP path)
void dequantize_row_q4_0_coalesced_to_fp16_rowmajor(
    const void * src,
    sycl::half * dst,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

void dequantize_row_q4_0_soa_to_fp16_rowmajor(
    const void * src,
    sycl::half * dst,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

void dequantize_row_q8_0_coalesced_to_fp16_rowmajor(
    const void * src,
    sycl::half * dst,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

void dequantize_row_q8_0_soa_to_fp16_rowmajor(
    const void * src,
    sycl::half * dst,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

// Dequantize MXFP4 SOA→row-major FP16 (for oneDNN PP path)
void dequantize_row_mxfp4_soa_to_fp16_rowmajor(
    const void * src,
    sycl::half * dst,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

// Repack MXFP4 SOA -> WOQ: de-interleave the intra-block j/j+16 nibble
// packing into sequential nibble order and transpose expert-row-major into
// oneDNN's {K,N} nibble / {K/QK_MXFP4,N} e8m0-scale layout (for the oneDNN
// f4_e2m1 grouped-scale WOQ matmul path; see convert.cpp for the exact index
// derivation). K = blocks_per_row * QK_MXFP4, N = nrows.
void repack_mxfp4_soa_to_woq(
    const void * src,
    uint8_t * dst_nibbles,
    uint8_t * dst_scales,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

// Repack MXFP4 XMX_TILED -> WOQ: inverts the tiled materializer's byte
// layout (ggml_sycl::moe_tile_convert::reorder_mxfp4_to_xmx_tiled /
// reorder_mxfp4_aos_to_xmx_tiled, moe-tile-convert.cpp) into the SAME
// destination layout as repack_mxfp4_soa_to_woq above (see convert.cpp for
// the exact index derivation). K = blocks_per_row * QK_MXFP4, N = nrows.
// tile_n_total is the device's XMX tile width (caps.N * optimal_tiles_n),
// passed by the caller -- not hardcoded here, since it is caps-dependent.
void repack_mxfp4_xmx_tiled_to_woq(
    const void * src_tiled,
    uint8_t * dst_nibbles,
    uint8_t * dst_scales,
    int blocks_per_row,
    int nrows,
    int tile_n_total,
    dpct::queue_ptr stream);

// Upper bound on expert slots batched into one WOQ repack launch by the
// *_batched entry points below (llama.cpp-1lon). The per-slot source
// pointers are passed as a fixed-capacity array captured BY VALUE into the
// kernel lambda -- an ordinary SYCL kernel argument, not a new device
// allocation or H2D buffer upload -- so this bounds that array's size, not
// any allocation. GPT-OSS 20B (the only shipped MXFP4 MoE shape today) uses
// at most 32 active slots per dispatch.
//
// Measured hard ceiling (Battlemage/bmg_g21, oneAPI 2026.1.1): the backend
// compiler rejects a kernel once its TOTAL captured-argument size exceeds
// 2048 bytes ("Total size of kernel arguments exceeds limit"); a 256-slot
// table alone (256 * 8B = 2 KiB) already exceeds it once the destination
// pointer/strides/shape scalars are added (measured 2084-2108 bytes,
// depending on kernel). 128 slots (1 KiB) leaves ~1 KiB of headroom for
// those scalars -- 4x GPT-OSS 20B's current max with margin to spare. The
// ggml-sycl.cpp call site falls back to the pre-1lon per-slot repack loop
// whenever a dispatch's active-slot count exceeds this, so correctness for
// any larger MoE expert count is unaffected -- only the batching speedup is.
constexpr int GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS = 128;

// Batched SOA MXFP4 -> WOQ repack: repacks every active expert slot's
// weights with ONE kernel launch (an added expert-slot grid dimension)
// instead of repack_mxfp4_soa_to_woq's one launch per slot. Per-slot index
// math is identical to the per-slot form; only the source pointer and the
// destination slot offset vary across slots -- everything else
// (blocks_per_row, nrows/N, weight_slot_bytes, woq_nibble_slot_bytes) is a
// per-tensor-dispatch constant shared by every slot. `srcs` is an array of
// `n_slots` per-expert source pointers (n_slots <= GGML_SYCL_MXFP4_WOQ_
// REPACK_MAX_SLOTS); `dst_weight_base` is the batched WOQ scratch base
// (slot s writes at dst_weight_base + s*weight_slot_bytes for nibbles, plus
// woq_nibble_slot_bytes for scales -- same layout repack_mxfp4_soa_to_woq's
// caller already builds per slot).
void repack_mxfp4_soa_to_woq_batched(
    const void * const * srcs,
    int n_slots,
    uint8_t * dst_weight_base,
    size_t weight_slot_bytes,
    size_t woq_nibble_slot_bytes,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

// Batched XMX_TILED MXFP4 -> WOQ repack: same batching as
// repack_mxfp4_soa_to_woq_batched above, for the tiled source layout (see
// repack_mxfp4_xmx_tiled_to_woq for the per-slot index derivation).
void repack_mxfp4_xmx_tiled_to_woq_batched(
    const void * const * srcs,
    int n_slots,
    uint8_t * dst_weight_base,
    size_t weight_slot_bytes,
    size_t woq_nibble_slot_bytes,
    int blocks_per_row,
    int nrows,
    int tile_n_total,
    dpct::queue_ptr stream);

// Coalesced (SLM-tiled transpose) forms of the two repack functions above
// (perf-recovery epic, track D, llama.cpp-0vqt): read/write in each
// direction's native (coalesced) axis, bridged through workgroup-local
// memory, instead of the byte-granularity scattered transpose the
// functions above perform. Byte-for-byte identical output to their
// non-coalesced counterparts in all cases -- each falls back to the
// original kernel whenever its fast-path precondition doesn't hold (N %
// 16 == 0 for the SOA forms; tile_n_total even and N % tile_n_total == 0
// for the XMX_TILED forms), so the speedup is scoped but correctness is
// not. See convert.cpp for the exact tiling derivation.
void repack_mxfp4_soa_to_woq_coalesced(
    const void * src,
    uint8_t * dst_nibbles,
    uint8_t * dst_scales,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

void repack_mxfp4_soa_to_woq_coalesced_batched(
    const void * const * srcs,
    int n_slots,
    uint8_t * dst_weight_base,
    size_t weight_slot_bytes,
    size_t woq_nibble_slot_bytes,
    int blocks_per_row,
    int nrows,
    dpct::queue_ptr stream);

void repack_mxfp4_xmx_tiled_to_woq_coalesced(
    const void * src_tiled,
    uint8_t * dst_nibbles,
    uint8_t * dst_scales,
    int blocks_per_row,
    int nrows,
    int tile_n_total,
    dpct::queue_ptr stream);

void repack_mxfp4_xmx_tiled_to_woq_coalesced_batched(
    const void * const * srcs,
    int n_slots,
    uint8_t * dst_weight_base,
    size_t weight_slot_bytes,
    size_t woq_nibble_slot_bytes,
    int blocks_per_row,
    int nrows,
    int tile_n_total,
    dpct::queue_ptr stream);

// Convert Q4_0 from Coalesced layout to SoA layout (for testing/debugging)
void reorder_q4_0_coalesced_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

#endif  // GGML_SYCL_CONVERT_HPP
