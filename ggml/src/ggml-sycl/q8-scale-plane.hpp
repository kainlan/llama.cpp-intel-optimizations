//
// Q8_0 SOA scale-plane index mapping (llama.cpp-nz1k, prefill L2b phase 1).
//
// Pure C++ -- NO SYCL, NO ggml includes -- so a host-only unit test
// (tests/test-sycl-q8-scale-plane-index.cpp) can check the mapping against a
// scalar reference without a device or a SYCL toolchain.
//
// Layout facts this header encodes (verified against convert.cpp's
// reorder_q8_0_aos_to_soa_sycl and quants.hpp block_q_t<GGML_TYPE_Q8_0>):
//
//   Q8_0 SOA materialization of an [nrows][ncols] weight, blocks_per_row = ncols/32:
//     [qs plane: nrows*ncols int8, block-major, block i = row*blocks_per_row + kb]
//     [d  plane: nrows*blocks_per_row f16, SAME block order -> stored [nrows][K/32]]
//
//   oneDNN 3.11 grouped weight scales (probe llama.cpp-ovkn V4/V5): the scale
//   memory must be logical (K/32, N) with strides {N, 1}, i.e. stored
//   [K/32][nrows]. Binding the SOA order directly with strides {1, K/32} is
//   accepted-but-garbage (probe V5), so phase 1 transposes the d plane into
//   the oneDNN PP scratch per call. Phase 2 (llama.cpp-2zsc) changes the
//   stored order and retires the transpose; the helpers below are the single
//   definition of the mapping so both the kernel and the test agree.
//
#pragma once

#include <cstddef>
#include <cstdint>

// Q8_0 block width; kept local so this header stays free of ggml-common.h.
constexpr int64_t GGML_SYCL_Q8_SCALE_PLANE_QK = 32;

// Byte offset of the d (scale) plane from the SOA base pointer.
constexpr size_t ggml_sycl_q8_0_soa_scale_plane_offset_bytes(int64_t nrows, int64_t blocks_per_row) {
    return static_cast<size_t>(nrows) * static_cast<size_t>(blocks_per_row) *
           static_cast<size_t>(GGML_SYCL_Q8_SCALE_PLANE_QK);
}

// Element index of block (row, kb) in the SOA d plane: stored [nrows][blocks_per_row].
constexpr int64_t ggml_sycl_q8_0_soa_scale_src_index(int64_t row, int64_t kb, int64_t blocks_per_row) {
    return row * blocks_per_row + kb;
}

// Element index of block (row, kb) in the oneDNN [K/32][N] scale plane: stored [blocks_per_row][nrows].
constexpr int64_t ggml_sycl_q8_0_soa_scale_kbn_index(int64_t row, int64_t kb, int64_t nrows) {
    return kb * nrows + row;
}

// For a flat destination index into the [K/32][N] plane, the source index in
// the SOA [N][K/32] plane. The kernel iterates over destination elements so
// its writes are contiguous; this is the one function it calls per element.
constexpr int64_t ggml_sycl_q8_0_soa_scale_src_index_for_kbn(int64_t dst_index, int64_t nrows, int64_t blocks_per_row) {
    const int64_t kb  = dst_index / nrows;
    const int64_t row = dst_index - kb * nrows;
    return ggml_sycl_q8_0_soa_scale_src_index(row, kb, blocks_per_row);
}
