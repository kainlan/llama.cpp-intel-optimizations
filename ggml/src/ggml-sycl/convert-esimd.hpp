//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// =============================================================================
// ESIMD port of Q4_0 row-major dequantization kernels.
//
// Provides SIMD-16 vectorised implementations of:
//   dequantize_row_q4_0_coalesced_to_fp16_rowmajor_esimd  (COALESCED layout)
//   dequantize_row_q4_0_soa_to_fp16_rowmajor_esimd        (SOA layout)
//
// Design (Option B — 1 ESIMD thread per tile = 32 blocks = 1024 weights):
//   Each thread loads 32 × 16 = 512 bytes of qs and 32 scales in a tiled loop,
//   producing 32 × 32 = 1024 FP16 values.  This amortises ESIMD dispatch overhead
//   over a much larger body of work.  Nibble extraction and FP16 conversion are
//   done with explicit simd<uint8_t, 32>  vector ops (no scalar loops in the
//   hot-path per block).
// =============================================================================

#ifndef GGML_SYCL_CONVERT_ESIMD_HPP
#define GGML_SYCL_CONVERT_ESIMD_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define GGML_SYCL_ESIMD_DEQUANT_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
#else
#    define GGML_SYCL_ESIMD_DEQUANT_AVAILABLE 0
#endif

#if GGML_SYCL_ESIMD_DEQUANT_AVAILABLE

// Kernel name tags (needed for VTune / profiler visibility)
class dequant_q4_0_coalesced_rowmajor_esimd_kernel;
class dequant_q4_0_soa_rowmajor_esimd_kernel;

// =============================================================================
// Core dequant helper: process one block with simd<uint8_t, 16> nibble ops.
// Writes 32 FP16 values to dst_ptr.  No scalar loops.
// =============================================================================
SYCL_ESIMD_FUNCTION static inline void
dequant_q4_0_block_to_half_esimd(const esimd::simd<uint8_t, 16> packed,
                                  const sycl::half                d_h,
                                  sycl::half *                    dst_ptr) {
    using namespace esimd;

    // Extract nibbles as SIMD vector ops
    const simd<uint8_t, 16> lo_nib = packed & uint8_t(0x0F);
    const simd<uint8_t, 16> hi_nib = packed >> 4;

    // uint8 → int16 (zero-extend then subtract 8 in 16-bit arithmetic)
    const simd<int16_t, 16> lo_i = simd<int16_t, 16>(lo_nib) - int16_t(8);
    const simd<int16_t, 16> hi_i = simd<int16_t, 16>(hi_nib) - int16_t(8);

    // int16 → half, scaled
    const simd<sycl::half, 16> lo_h = simd<sycl::half, 16>(lo_i) * d_h;
    const simd<sycl::half, 16> hi_h = simd<sycl::half, 16>(hi_i) * d_h;

    // 16-element block stores (32 bytes each)
    block_store<sycl::half, 16>(dst_ptr,      lo_h);
    block_store<sycl::half, 16>(dst_ptr + 16, hi_h);
}

// =============================================================================
// SOA variant — Option B: 1 ESIMD thread per row-tile (TILE_BLOCKS = 32 blocks).
//
// Layout:
//   qs:     [total_blocks * 16 bytes]   (16 bytes per block, contiguous)
//   scales: [total_blocks * 2 bytes]    (sycl::half, contiguous after qs)
//
// Grid: (nrows * tiles_per_row) ESIMD threads, WG_SIZE=32.
// Each thread handles one tile of 32 consecutive blocks in a single row.
// Inner loop is unrolled — the compiler hoists address arithmetic and keeps
// simd regs live across the 32 iterations.
// =============================================================================
SYCL_ESIMD_FUNCTION static inline void
dequant_q4_0_soa_rowmajor_esimd_kernel_body(const void *  vx,
                                            sycl::half *  yy,
                                            int           blocks_per_row,
                                            int           nrows,
                                            sycl::nd_item<1> item) {
    using namespace esimd;

    // Tile = 32 blocks (matches MMVQ_COALESCED_TILE_BLOCKS / WARP_SIZE)
    constexpr int TILE_BLOCKS = 32;

    const int tiles_per_row = (blocks_per_row + TILE_BLOCKS - 1) / TILE_BLOCKS;
    const int total_tiles   = nrows * tiles_per_row;
    const int tile_id       = item.get_global_id(0);

    if (tile_id >= total_tiles) {
        return;
    }

    const int row    = tile_id / tiles_per_row;
    const int tile   = tile_id % tiles_per_row;

    const int64_t total_blocks = (int64_t) nrows * blocks_per_row;

    // Base qs pointer for this tile: qs are contiguous, 16 bytes per block
    const int64_t tile_block0    = (int64_t) row * blocks_per_row + tile * TILE_BLOCKS;
    const auto *  base           = reinterpret_cast<const uint8_t *>(vx);
    const int64_t total_qs_bytes = total_blocks * 16;

    // Process up to TILE_BLOCKS blocks; guard for last partial tile
    const int blocks_this_tile =
        (tile * TILE_BLOCKS + TILE_BLOCKS <= blocks_per_row)
            ? TILE_BLOCKS
            : (blocks_per_row - tile * TILE_BLOCKS);

#pragma unroll 8
    for (int b = 0; b < TILE_BLOCKS; b++) {
        if (b >= blocks_this_tile) {
            break;
        }

        const int64_t global_block = tile_block0 + b;
        const int     block_idx    = tile * TILE_BLOCKS + b;

        // Load 16 packed qs bytes for this block (16-byte aligned)
        const uint8_t * qs_ptr = base + global_block * 16;
        const simd<uint8_t, 16> packed = block_load<uint8_t, 16>(qs_ptr);

        // Scale: after all qs bytes
        const sycl::half d_h = *reinterpret_cast<const sycl::half *>(
            base + total_qs_bytes + global_block * 2);

        // Output: row-major FP16
        sycl::half * dst_ptr = yy + (int64_t) row * blocks_per_row * 32
                                   + (int64_t) block_idx * 32;

        dequant_q4_0_block_to_half_esimd(packed, d_h, dst_ptr);
    }
}

// =============================================================================
// COALESCED variant — Option A: 1 ESIMD thread per block.
// (Tile-interleaved word-plane layout — strided access, can't use block_load
//  for the 4 words since they're 128 bytes apart.  Use scalar loads + simd
//  construction for the nibble body.)
//
// Grid: (nrows * blocks_per_row) ESIMD threads, WG_SIZE=32.
// =============================================================================
SYCL_ESIMD_FUNCTION static inline void
dequant_q4_0_coalesced_rowmajor_esimd_kernel_body(const void *  vx,
                                                  sycl::half *  yy,
                                                  int           blocks_per_row,
                                                  int           nrows,
                                                  sycl::nd_item<1> item) {
    using namespace esimd;

    constexpr int TILE_BLOCKS       = 32;
    constexpr int QS_BYTES_PER_TILE = TILE_BLOCKS * 16;   // 512
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;    // 128

    const int global_id    = item.get_global_id(0);
    const int total_blocks = nrows * blocks_per_row;

    if (global_id >= total_blocks) {
        return;
    }

    const int row         = global_id / blocks_per_row;
    const int block_idx   = global_id % blocks_per_row;
    const int tile        = block_idx / TILE_BLOCKS;
    const int b_in_tile   = block_idx % TILE_BLOCKS;

    const auto * base = reinterpret_cast<const uint8_t *>(vx);

    const int64_t row_quants_bytes   = (int64_t) blocks_per_row * 16;
    const int64_t total_quants_bytes = (int64_t) nrows * row_quants_bytes;
    const int64_t tile_qs_base = (int64_t) row * row_quants_bytes
                                 + (int64_t) tile * QS_BYTES_PER_TILE;

    // Load 4 × uint32 words from strided word-plane offsets
    uint32_t w0 = *reinterpret_cast<const uint32_t *>(base + tile_qs_base + 0 * WORD_PLANE_STRIDE + b_in_tile * 4);
    uint32_t w1 = *reinterpret_cast<const uint32_t *>(base + tile_qs_base + 1 * WORD_PLANE_STRIDE + b_in_tile * 4);
    uint32_t w2 = *reinterpret_cast<const uint32_t *>(base + tile_qs_base + 2 * WORD_PLANE_STRIDE + b_in_tile * 4);
    uint32_t w3 = *reinterpret_cast<const uint32_t *>(base + tile_qs_base + 3 * WORD_PLANE_STRIDE + b_in_tile * 4);

    // Reconstruct 16-byte qs as simd<uint8_t,16> from the 4 uint32 words
    simd<uint8_t, 16> packed;
    packed.template select<4, 1>(0)  = simd<uint8_t, 4>(reinterpret_cast<const uint8_t *>(&w0));
    packed.template select<4, 1>(4)  = simd<uint8_t, 4>(reinterpret_cast<const uint8_t *>(&w1));
    packed.template select<4, 1>(8)  = simd<uint8_t, 4>(reinterpret_cast<const uint8_t *>(&w2));
    packed.template select<4, 1>(12) = simd<uint8_t, 4>(reinterpret_cast<const uint8_t *>(&w3));

    const int64_t global_block = (int64_t) row * blocks_per_row + block_idx;
    const sycl::half d_h = *reinterpret_cast<const sycl::half *>(
        base + total_quants_bytes + global_block * 2);

    sycl::half * dst_ptr = yy + (int64_t) row * blocks_per_row * 32 + (int64_t) block_idx * 32;

    dequant_q4_0_block_to_half_esimd(packed, d_h, dst_ptr);
}

// =============================================================================
// Public launch wrappers
// =============================================================================

// ESIMD kernels on Intel Arc: WG_SIZE ≤ 64 is the hardware limit.  32 matches
// the sub-group size on Battlemage and avoids idle EU lanes.
inline void dequantize_row_q4_0_coalesced_to_fp16_rowmajor_esimd(
    const void * src, sycl::half * dst,
    int blocks_per_row, int nrows, sycl::queue * stream) {

    const int total_blocks = nrows * blocks_per_row;
    constexpr int WG_SIZE  = 32;
    const int     n_wgs    = (total_blocks + WG_SIZE - 1) / WG_SIZE;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<dequant_q4_0_coalesced_rowmajor_esimd_kernel>(
            sycl::nd_range<1>(sycl::range<1>(n_wgs * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                dequant_q4_0_coalesced_rowmajor_esimd_kernel_body(
                    src, dst, blocks_per_row, nrows, item);
            });
    });
}

inline void dequantize_row_q4_0_soa_to_fp16_rowmajor_esimd(
    const void * src, sycl::half * dst,
    int blocks_per_row, int nrows, sycl::queue * stream) {

    // Option B: 1 ESIMD thread per tile (32 blocks).  Grid = nrows * tiles_per_row.
    constexpr int TILE_BLOCKS  = 32;
    const int tiles_per_row    = (blocks_per_row + TILE_BLOCKS - 1) / TILE_BLOCKS;
    const int total_tiles      = nrows * tiles_per_row;
    constexpr int WG_SIZE      = 32;
    const int     n_wgs        = (total_tiles + WG_SIZE - 1) / WG_SIZE;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<dequant_q4_0_soa_rowmajor_esimd_kernel>(
            sycl::nd_range<1>(sycl::range<1>(n_wgs * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                dequant_q4_0_soa_rowmajor_esimd_kernel_body(
                    src, dst, blocks_per_row, nrows, item);
            });
    });
}

#endif  // GGML_SYCL_ESIMD_DEQUANT_AVAILABLE

// =============================================================================
// Stubs when ESIMD unavailable (compile-time fallback guard)
// =============================================================================
#if !GGML_SYCL_ESIMD_DEQUANT_AVAILABLE
inline void dequantize_row_q4_0_coalesced_to_fp16_rowmajor_esimd(
    const void *, sycl::half *, int, int, sycl::queue *) {}

inline void dequantize_row_q4_0_soa_to_fp16_rowmajor_esimd(
    const void *, sycl::half *, int, int, sycl::queue *) {}
#endif

#endif  // GGML_SYCL_CONVERT_ESIMD_HPP
