//
// K-quant super-block decode for the SYCL GET_ROWS kernels.
//
// Pure C++ (no SYCL header, no ggml-common block struct) so a host test can run the exact code the device
// kernel runs and compare it with ggml's dequantize_row_q4_K. The kernel (k_get_rows_q4_k_aos, getrows.cpp)
// loads d, dmin, scales and qs from the AoS block and calls ggml_sycl_get_rows_q4_k_elem for each output
// element.
//
// Shape for the other K-quants: ggml_sycl_kquant_scale_min_k4 is the 6-bit scale/min unpack Q4_K and Q5_K
// share, so Q5_K adds only its own element function (the fifth bit from qh) beside the Q4_K one; Q2_K and
// Q3_K carry different scale packings and add their own helpers here. Each new type also needs its arm in
// ggml_sycl_op_get_rows, its case in ggml_sycl_get_rows_type_supported and its layouts in
// ggml_sycl_get_rows_layout_supported (get-rows-support.hpp); the source gate checks they agree.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

// 6-bit scale and min for sub-block j (0..7) of a Q4_K or Q5_K block, unpacked from its 12 scale bytes.
// Same bit layout as ggml's get_scale_min_k4.
inline void ggml_sycl_kquant_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Element idx (0..255) of one Q4_K block. A block is four 64-element chunks; chunk il uses the 32 qs bytes
// at 32 * il, the first 32 elements of the chunk from the low nibbles with sub-block 2 * il and the next 32
// from the high nibbles with sub-block 2 * il + 1. The arithmetic order, (d * sc) * q - (dmin * m), is
// ggml's, so the result matches dequantize_row_q4_K.
inline float ggml_sycl_get_rows_q4_k_elem(float d, float dmin, const uint8_t * scales, const uint8_t * qs, int idx) {
    const int il = idx / 64;
    const int in = idx % 64;
    const int is = 2 * il + (in >= 32 ? 1 : 0);
    const int l  = in & 31;

    uint8_t sc = 0;
    uint8_t m  = 0;
    ggml_sycl_kquant_scale_min_k4(is, scales, sc, m);

    const uint8_t q  = qs[32 * il + l];
    const int     qv = (in >= 32) ? (q >> 4) : (q & 0xF);
    return (d * sc) * qv - (dmin * m);
}
