//
// Test: MXFP4 vectorized per-block dequantization equivalence
//
// Verifies that dequant_mxfp4_block_half8 (vectorized 32-element
// per-block helper in unified-kernel.hpp) matches the scalar
// reference dequant_mxfp4_half byte-exact, across all 256 qs byte
// patterns x 8 representative E8M0 exponents.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>

#define XMX_TEST_STANDALONE 1

// Pull in the types and SYCL_EXTERNAL inline helpers. All MXFP4 helpers
// (including the new vectorized ones) live in unified-kernel.hpp alongside
// block_mxfp4_unified, kvalues_mxfp4_unified, and e8m0_to_float_half.
// Since they are SYCL_EXTERNAL inline and the test is host-only, the
// inline definitions are directly host-callable.
#include "unified-kernel.hpp"

// Host-side reference: element-by-element scalar dequant that mirrors
// the device-side dequant_mxfp4_half at unified-kernel.cpp:388. We keep
// this inline in the test so any divergence in the canonical nibble
// order is caught locally.
// TODO: cross-check directly against dequant_mxfp4_half
// (unified-kernel.cpp:388) once host-callable; today this test relies on
// ref_scalar mirroring the scalar device helper's logic.
static void ref_scalar(uint8_t e, const uint8_t * qs, sycl::half out[32]) {
    const float scale = ggml_sycl_unified::e8m0_to_float_half(e);
    for (int i = 0; i < 32; ++i) {
        int8_t kval;
        if (i < 16) {
            kval = ggml_sycl_unified::kvalues_mxfp4_unified[qs[i] & 0x0F];
        } else {
            kval = ggml_sycl_unified::kvalues_mxfp4_unified[qs[i - 16] >> 4];
        }
        out[i] = static_cast<sycl::half>(static_cast<float>(kval) * scale);
    }
}

int main() {
    // Representative E8M0 exponents: denormals (0, 1), mid-range
    // negative (0x3F), unity (0x7F), typical model range (0x80, 0x81,
    // 0x9F, 0xBF), max (0xFF).
    const uint8_t e_values[] = { 0x00, 0x3F, 0x7F, 0x80, 0x81, 0x9F, 0xBF, 0xFF };
    const int     n_e        = sizeof(e_values) / sizeof(e_values[0]);

    int failures = 0;

    for (int ei = 0; ei < n_e; ++ei) {
        const uint8_t e = e_values[ei];
        for (int byte_val = 0; byte_val < 256; ++byte_val) {
            uint8_t qs[16];
            // Fill with rotating pattern seeded by byte_val so each
            // iteration exercises a distinct 16-byte block.
            for (int i = 0; i < 16; ++i) {
                qs[i] = static_cast<uint8_t>((byte_val + i * 17) & 0xFF);
            }

            sycl::half ref[32];
            sycl::half vec_core[32];
            sycl::half vec_aos[32];

            ref_scalar(e, qs, ref);
            ggml_sycl_unified::dequant_mxfp4_block_half8(qs, e, vec_core);

            ggml_sycl_unified::block_mxfp4_unified block;
            block.e = e;
            std::memcpy(block.qs, qs, 16);
            ggml_sycl_unified::dequant_mxfp4_block_half8_aos(&block, vec_aos);

            for (int i = 0; i < 32; ++i) {
                uint16_t ref_bits;
                uint16_t core_bits;
                uint16_t aos_bits;
                std::memcpy(&ref_bits, &ref[i], sizeof(uint16_t));
                std::memcpy(&core_bits, &vec_core[i], sizeof(uint16_t));
                std::memcpy(&aos_bits, &vec_aos[i], sizeof(uint16_t));
                if (ref_bits != core_bits || ref_bits != aos_bits) {
                    if (failures < 16) {
                        fprintf(stderr,
                                "mismatch e=0x%02X byte_val=%d elem=%d "
                                "ref=0x%04X (%.6f) core=0x%04X (%.6f) aos=0x%04X (%.6f)\n",
                                e, byte_val, i,
                                ref_bits, static_cast<float>(ref[i]),
                                core_bits, static_cast<float>(vec_core[i]),
                                aos_bits, static_cast<float>(vec_aos[i]));
                    }
                    ++failures;
                }
            }
        }
    }

    if (failures == 0) {
        printf("OK: 256x8 qs x e combinations match byte-exact (core + aos)\n");
        return 0;
    }

    fprintf(stderr, "FAIL: %d element mismatches across 256x8 combinations\n", failures);
    return 1;
}
