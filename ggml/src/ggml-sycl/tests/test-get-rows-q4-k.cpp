//
// Test: SYCL GET_ROWS Q4_K element decode and support predicate (llama.cpp-qhfp)
//
// The device kernel k_get_rows_q4_k_aos (getrows.cpp) writes one output element per work-item through
// ggml_sycl_get_rows_q4_k_elem. That function lives in a header with no SYCL dependency so this host test
// can run the exact code the kernel runs and compare it with ggml's own Q4_K dequantisation
// (dequantize_row_q4_K, reached through the type traits), element for element.
//
// It also pins the support predicate supports_op uses: Q4_K is admitted (qwen35's token_embd.weight), the
// types already admitted still are, and a K-quant the kernel set does not cover is not.
//
// Host-only: ggml-base for the reference quantiser; no SYCL header, no device.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "get-rows-kquant.hpp"
#include "get-rows-support.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// block_q4_K: ggml_half d, ggml_half dmin, uint8_t scales[12], uint8_t qs[128].
static constexpr size_t Q4_K_BLOCK_BYTES = 144;
static constexpr size_t Q4_K_OFF_D       = 0;
static constexpr size_t Q4_K_OFF_DMIN    = 2;
static constexpr size_t Q4_K_OFF_SCALES  = 4;
static constexpr size_t Q4_K_OFF_QS      = 16;
static constexpr int    QK_K_ELEMS       = 256;

static float half_at(const uint8_t * p) {
    uint16_t h;
    std::memcpy(&h, p, sizeof(h));
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

int main() {
    CHECK(ggml_type_size(GGML_TYPE_Q4_K) == Q4_K_BLOCK_BYTES, "block_q4_K is 144 bytes");
    CHECK(ggml_blck_size(GGML_TYPE_Q4_K) == QK_K_ELEMS, "block_q4_K holds 256 elements");

    // A few rows of a token-embedding-shaped table: 5120 columns is 20 blocks, qwen35's width.
    const int64_t      n_cols = 5120;
    const int64_t      n_rows = 7;
    std::vector<float> src(static_cast<size_t>(n_cols * n_rows));
    uint32_t           state = 12345u;
    for (auto & v : src) {
        state = state * 1664525u + 1013904223u;
        v     = (static_cast<float>(state >> 8) / static_cast<float>(1 << 24) - 0.5f) * 0.1f;
    }

    const size_t         row_bytes = ggml_row_size(GGML_TYPE_Q4_K, n_cols);
    std::vector<uint8_t> quant(row_bytes * static_cast<size_t>(n_rows));
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), quant.data(), 0, n_rows, n_cols, nullptr);
    CHECK(written == quant.size(), "quantiser wrote every row");

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    CHECK(traits && traits->to_float, "ggml has a Q4_K reference dequantiser");

    std::vector<float> ref(static_cast<size_t>(n_cols));
    for (int64_t r = 0; r < n_rows; ++r) {
        const uint8_t * row = quant.data() + static_cast<size_t>(r) * row_bytes;
        traits->to_float(row, ref.data(), n_cols);
        for (int64_t i = 0; i < n_cols; ++i) {
            const uint8_t * block = row + static_cast<size_t>(i / QK_K_ELEMS) * Q4_K_BLOCK_BYTES;
            const float got = ggml_sycl_get_rows_q4_k_elem(half_at(block + Q4_K_OFF_D), half_at(block + Q4_K_OFF_DMIN),
                                                           block + Q4_K_OFF_SCALES, block + Q4_K_OFF_QS,
                                                           static_cast<int>(i % QK_K_ELEMS));
            // ggml's reference is d1 * q - m1 in float; the shared function does the same arithmetic.
            const float tol = 1e-6f * std::fmax(1.0f, std::fabs(ref[static_cast<size_t>(i)]));
            if (!(std::fabs(got - ref[static_cast<size_t>(i)]) <= tol)) {
                std::fprintf(stderr, "FAIL: row %lld col %lld: got %.9g, ggml reference %.9g\n", (long long) r,
                             (long long) i, got, ref[static_cast<size_t>(i)]);
                return 1;
            }
        }
    }

    // The scale/min unpack has two regimes (j < 4 and j >= 4); both must have been exercised above, which a
    // 256-wide block does by construction (is = 0..7). Pin the pure helper directly as well.
    {
        uint8_t scales[12];
        for (int i = 0; i < 12; ++i) {
            scales[i] = static_cast<uint8_t>(0x40 * (i % 4) + 0x0B + i);
        }
        uint8_t d = 0;
        uint8_t m = 0;
        ggml_sycl_kquant_scale_min_k4(1, scales, d, m);
        CHECK(d == (scales[1] & 63) && m == (scales[5] & 63), "scale/min for j < 4");
        ggml_sycl_kquant_scale_min_k4(6, scales, d, m);
        CHECK(d == ((scales[10] & 0xF) | ((scales[2] >> 6) << 4)) && m == ((scales[10] >> 4) | ((scales[6] >> 6) << 4)),
              "scale/min for j >= 4");
    }

    // Support predicate: what was admitted still is, Q4_K now is, an uncovered K-quant is not.
    CHECK(ggml_sycl_get_rows_type_supported(GGML_TYPE_Q4_K), "Q4_K is admitted");
    CHECK(ggml_sycl_get_rows_type_supported(GGML_TYPE_Q4_0), "Q4_0 stays admitted");
    CHECK(ggml_sycl_get_rows_type_supported(GGML_TYPE_Q6_K), "Q6_K stays admitted");
    CHECK(ggml_sycl_get_rows_type_supported(GGML_TYPE_F16), "F16 stays admitted");
    // (type, layout): the Q4_K kernel reads AoS blocks only, so the pair for any other layout is declined.
    CHECK(ggml_sycl_get_rows_layout_supported(GGML_TYPE_Q4_K, GGML_LAYOUT_AOS), "Q4_K AoS is a supported pair");
    CHECK(!ggml_sycl_get_rows_layout_supported(GGML_TYPE_Q4_K, GGML_LAYOUT_SOA), "Q4_K SoA is declined");
    CHECK(!ggml_sycl_get_rows_layout_supported(GGML_TYPE_Q4_K, GGML_LAYOUT_COALESCED), "Q4_K coalesced is declined");
    CHECK(ggml_sycl_get_rows_layout_supported(GGML_TYPE_Q6_K, GGML_LAYOUT_SOA), "Q6_K keeps its SoA kernel");
    CHECK(ggml_sycl_get_rows_layout_supported(GGML_TYPE_Q4_0, GGML_LAYOUT_COALESCED),
          "Q4_0 keeps its coalesced kernel");

    CHECK(!ggml_sycl_get_rows_type_supported(GGML_TYPE_Q3_K), "Q3_K has no kernel and is not admitted");
    CHECK(!ggml_sycl_get_rows_type_supported(GGML_TYPE_IQ4_NL), "IQ4_NL has no kernel and is not admitted");

    std::printf("PASS: GET_ROWS Q4_K decode matches ggml across %lld rows\n", (long long) n_rows);
    return 0;
}
