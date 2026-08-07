//
// Addressing for the Q4_0 warp-coalesced weight layout.
//
// The coalesced layout stores a Q4_0 tensor as two regions:
//
//   [ quants : total_rows * (ncols/2) bytes ][ scales : total_rows * blocks_per_row * fp16 ]
//
// Within a row the quants are grouped into tiles of MMVQ_COALESCED_TILE_BLOCKS
// blocks, and within a tile the bytes are word-major rather than block-major:
// word w of block b sits at `w * (TILE_BLOCKS * 4) + b * 4`, so the 32 lanes of
// a sub-group reading the same word index touch one contiguous cache line
// instead of 32 strided ones. The scales stay block-sequential.
//
// This header exists because that addressing is duplicated across the kernels,
// the reorder writers and the tests, and the two failure modes it produces are
// both silent:
//
//   * the SCALE BASE is an offset into the FULL tensor, so it must be derived
//     from the tensor's total row count -- never from a slice's row count. A
//     sliced dispatch (row_low > 0 or row_diff < ne01) that uses the slice size
//     lands inside the quants region and reads quant bytes as fp16 scales.
//   * the ROW INDEX used for both regions is the row in the FULL tensor
//     (row_low + local_row), not the row within the slice.
//
// Kept deliberately dependency-free -- no SYCL header, no backend header -- so
// the arithmetic can be unit-tested on a host with no GPU and no SYCL runtime,
// exactly like mmvq-launch-geometry.hpp (see
// ggml/src/ggml-sycl/tests/test-dmmv-coalesced-q4-0-oracle.cpp).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

namespace ggml_sycl {

// Q4_0: 32 weights per block, 16 quant bytes per block.
constexpr int DMMV_COALESCED_Q4_0_QK          = 32;
constexpr int DMMV_COALESCED_Q4_0_BLOCK_BYTES = DMMV_COALESCED_Q4_0_QK / 2;

// Blocks per tile. Matches the sub-group width so one lane owns one block.
constexpr int DMMV_COALESCED_TILE_BLOCKS = 32;

constexpr int DMMV_COALESCED_Q4_0_TILE_BYTES = DMMV_COALESCED_TILE_BLOCKS * DMMV_COALESCED_Q4_0_BLOCK_BYTES;

// Distance between word plane w and word plane w+1 inside a tile.
constexpr int DMMV_COALESCED_Q4_0_WORD_STRIDE = DMMV_COALESCED_TILE_BLOCKS * 4;

// Tiles spanned by one row. Rounds up, so a partial trailing tile still gets a
// tile of its own rather than being dropped.
inline int dmmv_coalesced_tile_count(const int blocks_per_row) {
    if (blocks_per_row <= 0) {
        return 0;
    }
    return (blocks_per_row + DMMV_COALESCED_TILE_BLOCKS - 1) / DMMV_COALESCED_TILE_BLOCKS;
}

// Quant bytes per row.
inline int64_t dmmv_coalesced_q4_0_row_quants_bytes(const int blocks_per_row) {
    return (int64_t) blocks_per_row * DMMV_COALESCED_Q4_0_BLOCK_BYTES;
}

// Byte offset, from the start of the tensor, of quant byte `byte_in_block` of
// block `block_in_row` of row `global_row`. `global_row` is the row in the FULL
// tensor -- add row_low before calling from a sliced dispatch.
inline int64_t dmmv_coalesced_q4_0_qs_offset(const int global_row,
                                             const int blocks_per_row,
                                             const int block_in_row,
                                             const int byte_in_block) {
    const int     tile          = block_in_row / DMMV_COALESCED_TILE_BLOCKS;
    const int     block_in_tile = block_in_row % DMMV_COALESCED_TILE_BLOCKS;
    const int     word          = byte_in_block / 4;
    const int     byte_in_word  = byte_in_block % 4;
    const int64_t row_base      = (int64_t) global_row * dmmv_coalesced_q4_0_row_quants_bytes(blocks_per_row);
    const int64_t tile_base     = row_base + (int64_t) tile * DMMV_COALESCED_Q4_0_TILE_BYTES;
    return tile_base + (int64_t) word * DMMV_COALESCED_Q4_0_WORD_STRIDE + (int64_t) block_in_tile * 4 + byte_in_word;
}

// Byte offset, from the start of the tensor, of the fp16 scale region.
// `total_rows` is the FULL tensor row count (ne01). Passing a slice's row count
// here is the silent-corruption case this header exists to name.
inline int64_t dmmv_coalesced_q4_0_scale_base(const int total_rows, const int blocks_per_row) {
    return (int64_t) total_rows * dmmv_coalesced_q4_0_row_quants_bytes(blocks_per_row);
}

// Index into the fp16 scale region. Block-sequential, full-tensor row.
inline int64_t dmmv_coalesced_q4_0_scale_index(const int global_row, const int blocks_per_row, const int block_in_row) {
    return (int64_t) global_row * blocks_per_row + block_in_row;
}

}  // namespace ggml_sycl
