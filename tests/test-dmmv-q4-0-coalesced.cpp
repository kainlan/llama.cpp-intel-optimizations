// tests/test-dmmv-q4-0-coalesced.cpp
// Test that Q4_0 DMMV with coalesced layout produces same output as SoA
//
// CORRECTED (llama.cpp-0igs, 2026-08-02): the note this replaced described
// the whole file as "written to fail until the coalesced DMMV kernel is
// properly integrated" -- that describes test_coalesced_vs_soa_dmmv(),
// which main() never calls. main()'s actual Part 2 (run_dmmv_coalesced_case,
// via GGML_SYCL_FORCE_DMMV=1 + a GGML_LAYOUT_COALESCED override) drives the
// real, reachable production dispatch path -- confirmed via
// ggml_backend_graph_compute in its backtrace, not a stub.
//
// CORRECTED AGAIN (llama.cpp-szv8, 2026-08-06): the "~12-31% max_rel, ~35-40%
// of rows" this file then attributed to a wrong-answer bug in the kernel is
// the ORACLE's error, not the kernel's. Part 2 compared the GPU against
// `cpu_out` -- the ggml CPU backend -- and for a Q4_0 weight the CPU backend
// quantizes the ACTIVATION to Q8_0 (vec_dot_q4_0_q8_0). The SYCL DMMV path
// does not: dmmv.cpp converts src1 with to_fp16_sycl and the coalesced kernel
// multiplies in f32 (the Q8_0-activation kernel exists but is off by default,
// GGML_SYCL_DMMV_USE_Q8). So the two sides quantize the activation
// differently, and the gap is the CPU side's Q8_0 noise -- which the GPU does
// not have. The GPU was the more accurate of the two all along.
//
// The decisive check is that an EXACT f64 dot product, scored as if it were
// the GPU, also fails the old comparison: errors=14/37/67 at the three shapes
// (host model, ggml/src/ggml-sycl/tests/test-dmmv-coalesced-q4-0-oracle.cpp).
// No implementation could pass this test as written, correct or not. Part 1's
// layout-integrity pass never contradicted that -- it is CPU-only and checks
// the test's own two converters against each other, so it says nothing about
// the device at all.
//
// Part 2 now compares against dmmv_q4_0_dfloat_reference(), which models what
// the kernel is specified to compute. Tolerances were NOT loosened to achieve
// this; the relative gate was tightened from 1e-2 to 1e-3, because against a
// contract-faithful oracle the residual is f32 reassociation noise (measured
// max_rel 3.8e-5) rather than activation quantization.
//
// CORRECTED A THIRD TIME (llama.cpp-szv8, 2026-08-07): Part 2 was not running
// the coalesced kernel at all, so neither oracle applied to it. Two independent
// reasons, both now fixed:
//
//  1. The weight was named "weight". ggml_sycl::infer_tensor_usage() maps that
//     to tensor_usage::UNKNOWN, and layout_policy::get_optimal() answers SOA for
//     UNKNOWN -- COALESCED is reachable only for ATTENTION_WEIGHT/FFN_WEIGHT.
//     The weights are now named "blk.<n>.attn_q.weight".
//  2. The weight sat in the device buffer with no model-load transaction, so
//     ggml_backend_sycl_register_host_weight_tensor() dropped it and S1-PRELOAD
//     -- the only producer of a dense COALESCED cache entry -- never saw it.
//     Part 2 now stages through the same bracket as
//     tests/test-q8-0-layout-cache-path.cpp (llama.cpp-43uy).
//
// Be precise about which of those two actually drives THIS test, because it is
// not the rename. ggml_sycl_layout_override_active() feeds S1-PRELOAD as well as
// kernel selection (ggml-sycl.cpp:28301 and :28313), and
// ggml_sycl_can_use_layout_for_kernel() returns true for ANY layout on ANY name
// once the weight buffer is host and GGML_SYCL_WEIGHTS_EVICTABLE=1
// (ggml-sycl.cpp:52714). So while the override guard in main() is held, the
// materialized layout is COALESCED whatever the tensor is called. The rename is
// independently sufficient via layout_policy -- which is what makes COALESCED
// production-policy-reachable for this type and shape, and is the fact to reason
// from about production -- but it is not the operative mechanism in this test.
//
// CORRECTED A FOURTH TIME, AND THIS ONE CLOSES IT (llama.cpp-szv8, 2026-08-08):
// Part 2 was STILL not running the coalesced DMMV kernel after the third fix,
// and the numbers say so to seven significant figures. Hardware, all three
// shapes: the GPU matched a q8_1-MMVQ oracle at max_diff 4e-6 / 8e-6 / 5e-4
// while missing the dfloat oracle by 0.187 / 0.268 / 0.500. One row, verbatim:
// gpu=-14.931747, q8_1_mmvq=-14.931746, dfloat=-14.980540.
//
// The mechanism is ggml-sycl.cpp's TG fast-path, which claims every batch=1
// quantized mul_mat whose weight resolved to a non-AoS layout, dispatches MMVQ
// with q8_1-quantized activations against that layout, and returns -- before any
// of the kernel-selection machinery that honours GGML_SYCL_FORCE_DMMV or the
// layout override's kernel choice. Its own comment says it "bypasses
// orchestrator, name parsing, prefetch, TP checks for maximum speed".
//
// This is why the third fix's preconditions could ALL pass while the verdict
// stayed wrong: the override binds on materialization, so the coalesced buffer
// really was built and really was byte-exact -- and then MMVQ read it. Every
// structural check was true and the answer still came from the wrong kernel.
// main() now also sets GGML_SYCL_TG_FAST=0, and the gate no longer takes any
// precondition's word for it: it asserts from the numbers which kernel ran.
//
// Two things follow that are bigger than this test. The observed 12-31% was
// never a wrong answer -- it is the by-design accuracy of MMVQ's 8-bit activation
// quantization scored against a full-precision-activation oracle. And for Q4_0 at
// batch=1 with a reordered layout, production takes that same fast-path, so the
// coalesced DMMV kernel this file is named for is not reached by default.
//
// The symptom was "[LAYOUT-CHECK] readback done (source=tensor-storage-fallback)"
// followed by "Coalesced layout check: NOT RUN" on every shape, while the numeric
// comparison happily produced FAIL rows for whatever kernel the SOA layout
// dispatched. Part 2 now asserts its preconditions -- resolved layout is
// COALESCED, the layout pointer is distinct from the AoS storage, and the
// coalesced bytes are exact -- BEFORE comparing numbers, so a run that misses
// its target says so instead of reporting a verdict about a kernel it never ran.
//
// The coalesced layout reorganizes SoA data into tile-based format for
// better cache line utilization during DMMV operations.
//
// Build: cmake --build build --target test-dmmv-q4-0-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-dmmv-q4-0-coalesced

#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Q4_0 block structure (must match ggml-common.h)
#define QK4_0 32
#define QR4_0 2

// Coalesced tile configuration (from common.hpp)

typedef struct {
    ggml_fp16_t d;
    uint8_t     qs[QK4_0 / 2];
} block_q4_0_test;

static_assert(sizeof(block_q4_0_test) == 18, "block_q4_0 size mismatch");

// =============================================================================
// CPU Reference Implementation
// =============================================================================

static void dequantize_block_q4_0_cpu(const block_q4_0_test * block, float * out) {
    const float d = ggml_fp16_to_fp32(block->d);
    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        const int     lo   = (byte & 0xF);
        const int     hi   = (byte >> 4);
        out[i]             = (lo - 8) * d;
        out[i + QK4_0 / 2] = (hi - 8) * d;
    }
}

static void dmmv_q4_0_cpu_reference(const block_q4_0_test * x, const float * y, float * dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test * block = &x[row * blocks_per_row + b];
            float                   dequant[QK4_0];
            dequantize_block_q4_0_cpu(block, dequant);

            for (int i = 0; i < QK4_0; i++) {
                sum += dequant[i] * y[b * QK4_0 + i];
            }
        }
        dst[row] = sum;
    }
}

// Oracle for the GPU DMMV path. The coalesced kernel dequantizes X to f32 and
// multiplies by the activation as `dfloat` -- sycl::half under GGML_SYCL_F16
// (dmmv.cpp converts src1 with to_fp16_sycl before dispatch), float otherwise --
// accumulating in f32. Rounding y through `dfloat` here is what makes this an
// oracle for the kernel instead of for some other arithmetic; because it is the
// same typedef the kernel's signature uses, it tracks the build automatically.
//
// This is deliberately NOT the ggml CPU backend: that path quantizes the
// activation to Q8_0, which the SYCL DMMV path does not do. See the file header.
static void dmmv_q4_0_dfloat_reference(const block_q4_0_test * x, const float * y, float * dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test * block = &x[row * blocks_per_row + b];
            float                   dequant[QK4_0];
            dequantize_block_q4_0_cpu(block, dequant);

            for (int i = 0; i < QK4_0; i++) {
                sum += dequant[i] * (float) (dfloat) y[b * QK4_0 + i];
            }
        }
        dst[row] = sum;
    }
}

// The same dot product with the ACTIVATION quantized to Q8_0, computed from AoS
// weights. This is not a gate and never becomes one: it exists so a failing run
// can report how much of its gap is explained purely by which activation the
// oracle assumes. The Q4_0 DMMV dispatch has two branches -- src1 as dfloat and
// src1 quantized to Q8_0 (dequantize_mul_mat_vec_q4_0_coalesced_q8_0) -- and the
// dfloat reference models only the first. See llama.cpp-szv8.
static void dmmv_q4_0_q8_activation_reference(const block_q4_0_test * x,
                                              const block_q8_0 *      y,
                                              float *                 dst,
                                              int                     ncols,
                                              int                     nrows) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test * blk  = &x[row * blocks_per_row + b];
            int                     sumi = 0;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                sumi += ((blk->qs[j] & 0xF) - 8) * y[b].qs[j];
                sumi += ((blk->qs[j] >> 4) - 8) * y[b].qs[j + QK4_0 / 2];
            }
            sum += (float) sumi * ggml_fp16_to_fp32(blk->d) * ggml_fp16_to_fp32(y[b].d);
        }
        dst[row] = sum;
    }
}

// The arithmetic an MMVQ dispatch actually performs for a Q4_0 weight, which is
// neither of the two oracles above and is the one no round had modelled
// (llama.cpp-szv8). Three details make it distinct, and all three matter:
//
//   * the activation is quantized to q8_1 -- 8 bits per 32-element block, scale
//     amax/127 (ggml/src/ggml-sycl/quantize.hpp:39-56);
//   * the block also carries `sum`, the sum of the RAW activation floats, and
//     both it and the scale are stored as fp16 (quantize.hpp:121);
//   * vec_dot_q4_0_q8_1_impl folds Q4_0's -8 offset by subtracting that raw
//     `sum` rather than d8*sum(q8). The two differ by the block's quantization
//     residual, which is why a plain Q8_0-activation oracle lands at roughly
//     half the gap and this one lands on it.
//
// On that last point the coefficient below is 8 while the production line reads
// 4, and the difference is real rather than a transcription slip -- two
// re-verifiers have now worked through it, so the derivation belongs here.
// ggml/src/ggml-sycl/vecdotq.hpp:709 literally says
//
//     return d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
//
// with vdr = VDR_Q4_0_Q8_1_MMVQ = 2 (vecdotq.hpp:689) and QI4_0 = QK4_0/(4*QR4_0)
// = 4, so the coefficient is 8*2/4 = 4 -- but that is PER CALL, and each call
// covers only vdr of the block's qi quant groups. mmvq.cpp:2174 (and :2265 for
// the reorder variant) sets block_elements_per_subgroup = qi/vdr_mmvq = 2, so
// two such calls cover one block and their results are summed by the sub-group
// reduction: 2 x 4 = 8. This reference is written per whole block rather than
// per partial group, so 8 is the correct coefficient here. Anyone reconciling
// the two should compare totals per block, not the literal constants.
//
// Modelled on the host, this reproduces the reported GPU numbers to six
// significant figures at two shapes (1024x64: errors 55, max_rel 0.165478 vs
// 0.165479 measured; 2048x128: 116, 0.176323 vs 0.176321), which is an
// identification of the executing path rather than a magnitude-class argument.
static void dmmv_q4_0_q8_1_mmvq_reference(const block_q4_0_test * x,
                                          const float *           y,
                                          float *                 dst,
                                          int                     ncols,
                                          int                     nrows) {
    const int blocks_per_row = ncols / QK4_0;  // QK8_1 == QK4_0 == 32

    std::vector<ggml_fp16_t> d8(blocks_per_row);
    std::vector<ggml_fp16_t> s8(blocks_per_row);
    std::vector<int8_t>      q8((size_t) ncols);

    for (int b = 0; b < blocks_per_row; ++b) {
        float amax = 0.0f;
        float sum  = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = y[b * QK4_0 + j];
            sum += v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float d = (amax == 0.0f) ? 1.0f : amax / 127.0f;
        for (int j = 0; j < QK4_0; ++j) {
            q8[b * QK4_0 + j] = (int8_t) roundf(y[b * QK4_0 + j] / d);
        }
        d8[b] = ggml_fp32_to_fp16((amax == 0.0f) ? 0.0f : d);
        s8[b] = ggml_fp32_to_fp16(sum);
    }

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_0_test * blk = &x[row * blocks_per_row + b];

            // Note the quants are NOT offset by 8 here; the offset is the
            // separate -8*ds.y term below, exactly as the kernel does it.
            int raw = 0;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                raw += (blk->qs[j] & 0xF) * q8[b * QK4_0 + j];
                raw += (blk->qs[j] >> 4) * q8[b * QK4_0 + j + QK4_0 / 2];
            }

            sum +=
                ggml_fp16_to_fp32(blk->d) * ((float) raw * ggml_fp16_to_fp32(d8[b]) - 8.0f * ggml_fp16_to_fp32(s8[b]));
        }
        dst[row] = sum;
    }
}

static void dmmv_q4_0_cpu_from_coalesced(const uint8_t * coalesced, const float * y, float * dst,
                                         int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;
    const int row_bytes = ncols / 2;

    const uint8_t * x_qs = coalesced;
    const ggml_fp16_t * x_d = reinterpret_cast<const ggml_fp16_t *>(coalesced + nrows * row_bytes);

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;

        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int tile_base = row * row_bytes + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int block_in_tile = 0; block_in_tile < TILE_BLOCKS; ++block_in_tile) {
                const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                const float d = ggml_fp16_to_fp32(x_d[block_idx]);
                const int y_base = (tile * TILE_BLOCKS + block_in_tile) * QK4_0;

                uint8_t qs[QK4_0 / 2];
                for (int word = 0; word < 4; ++word) {
                    const int word_offset = word * word_stride + block_in_tile * 4;
                    const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                    memcpy(qs + word * 4, word_ptr, 4);
                }

                float block_sum = 0.0f;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    const float v0 = (float)((byte & 0xF) - 8);
                    const float v1 = (float)((byte >> 4) - 8);
                    block_sum += v0 * y[y_base + j];
                    block_sum += v1 * y[y_base + j + 16];
                }

                sum += d * block_sum;
            }
        }

        dst[row] = sum;
    }
}

static void dmmv_q4_0_cpu_from_coalesced_q8_0(const uint8_t * coalesced, const block_q8_0 * y, float * dst,
                                              int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;
    const int row_bytes = ncols / 2;

    const uint8_t * x_qs = coalesced;
    const ggml_fp16_t * x_d = reinterpret_cast<const ggml_fp16_t *>(coalesced + nrows * row_bytes);

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;

        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int tile_base = row * row_bytes + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int block_in_tile = 0; block_in_tile < TILE_BLOCKS; ++block_in_tile) {
                const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                const float d = ggml_fp16_to_fp32(x_d[block_idx]);

                uint8_t qs[QK4_0 / 2];
                for (int word = 0; word < 4; ++word) {
                    const int word_offset = word * word_stride + block_in_tile * 4;
                    const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                    memcpy(qs + word * 4, word_ptr, 4);
                }

                const block_q8_0 * y_blk = y + (tile * TILE_BLOCKS + block_in_tile);
                int sumi = 0;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    const int v0 = (byte & 0xF) - 8;
                    const int v1 = (byte >> 4) - 8;
                    sumi += v0 * y_blk->qs[j];
                    sumi += v1 * y_blk->qs[j + QK4_0 / 2];
                }

                sum += (float)sumi * d * ggml_fp16_to_fp32(y_blk->d);
            }
        }

        dst[row] = sum;
    }
}

static void reorder_q4_0_aos_bytes_to_coalesced(const uint8_t * aos, uint8_t * coalesced,
                                                int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK = 4;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;
    const int row_quants_bytes = ncols / 2;
    const int total_quants_bytes = nrows * row_quants_bytes;

    for (int ib = 0; ib < total_blocks; ++ib) {
        const int row = ib / blocks_per_row;
        const int col_block = ib % blocks_per_row;
        const int tile = col_block / TILE_BLOCKS;
        const int block_in_tile = col_block % TILE_BLOCKS;

        const int64_t tile_base = (int64_t)row * row_quants_bytes +
                                  (int64_t)tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        const uint8_t * src_qs = aos + (int64_t)ib * sizeof(block_q4_0_test) + sizeof(ggml_fp16_t);
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t dst_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coalesced + dst_offset, src_qs + word * 4, 4);
        }

        const uint8_t * src_d = aos + (int64_t)ib * sizeof(block_q4_0_test);
        memcpy(coalesced + total_quants_bytes + (int64_t)ib * sizeof(ggml_fp16_t), src_d, sizeof(ggml_fp16_t));
    }
}

// Block-major quants: the SoA layout, i.e. the coalesced buffer with the
// word-plane interleave never applied. Named because it is the layout a failing
// check has actually reported (llama.cpp-szv8): comparing SoA bytes against the
// coalesced expectation gives first_bad=4, because byte 4 is block 1's word 0
// under the contract and block 0's word 1 under block-major.
static void reorder_q4_0_aos_bytes_to_block_major(const uint8_t * aos, uint8_t * soa, int ncols, int nrows) {
    const int total_blocks       = nrows * (ncols / QK4_0);
    const int total_quants_bytes = nrows * (ncols / 2);

    for (int ib = 0; ib < total_blocks; ++ib) {
        const uint8_t * block_aos = aos + (int64_t) ib * sizeof(block_q4_0_test);
        memcpy(soa + (int64_t) ib * (QK4_0 / 2), block_aos + sizeof(ggml_fp16_t), QK4_0 / 2);
        memcpy(soa + total_quants_bytes + (int64_t) ib * sizeof(ggml_fp16_t), block_aos, sizeof(ggml_fp16_t));
    }
}

static size_t count_byte_mismatches(const uint8_t * a, const uint8_t * b, size_t n, size_t * first_bad) {
    size_t errors = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (errors == 0 && first_bad) {
                *first_bad = i;
            }
            errors++;
        }
    }
    return errors;
}

// Byte-exact verdict on the device's COALESCED buffer, against the addressing
// contract in dmmv-coalesced-q4-0-layout.hpp. `device_coalesced` must be the
// readback of the tensor's resolved COALESCED layout pointer -- the caller
// establishes that before calling, because a readback of the tensor's own
// storage holds whatever layout the backend staged and a mismatch there says
// nothing about any coalesced buffer.
static bool compare_device_coalesced_layout(const uint8_t * aos,
                                            const uint8_t * device_coalesced,
                                            int             ncols,
                                            int             nrows) {
    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;
    const size_t row_quants_bytes = ncols / 2;
    const size_t total_quants_bytes = (size_t)nrows * row_quants_bytes;
    const size_t expected_bytes = total_quants_bytes + (size_t)total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> expected(expected_bytes);
    reorder_q4_0_aos_bytes_to_coalesced(aos, expected.data(), ncols, nrows);

    size_t       first_bad = 0;
    const size_t errors    = count_byte_mismatches(expected.data(), device_coalesced, expected_bytes, &first_bad);

    if (errors == 0) {
        printf("  Coalesced layout check: PASSED (bytes=%zu)\n", expected_bytes);
        return true;
    }

    // Name the layout the device bytes DO match, when they match a known one.
    // A raw (expected, got) byte pair invites a hunt through the reorder
    // kernels; "the interleave never ran" points at the one thing to check.
    std::vector<uint8_t> block_major(expected_bytes);
    reorder_q4_0_aos_bytes_to_block_major(aos, block_major.data(), ncols, nrows);
    const size_t soa_errors = count_byte_mismatches(block_major.data(), device_coalesced, expected_bytes, nullptr);

    printf("  Coalesced layout check: FAILED (errors=%zu of %zu bytes, first_bad=%zu expected=0x%02x got=0x%02x)\n",
           errors, expected_bytes, first_bad, expected[first_bad], device_coalesced[first_bad]);
    if (soa_errors == 0) {
        printf("    device bytes are EXACTLY block-major (SoA): the word-plane interleave never ran\n");
    } else {
        printf(
            "    device bytes match neither the coalesced contract nor block-major"
            " (block-major mismatches=%zu)\n",
            soa_errors);
    }
    return false;
}

// =============================================================================
// GGML Backend Helpers (DMMV path)
// =============================================================================

static bool run_mul_mat_backend(ggml_backend_t       backend,
                                ggml_type            weight_type,
                                const void *         weight_data,
                                size_t               weight_size,
                                const float *        input_data,
                                int                  n_embd,
                                int                  n_rows,
                                std::vector<float> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    ggml_set_name(weight, "weight");

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_name(input, "input");

    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    ggml_backend_tensor_set(input, input_data, 0, n_embd * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool success = (status == GGML_STATUS_SUCCESS);
    if (success) {
        output.resize(n_rows);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

// =============================================================================
// Staged GPU run (the real coalesced DMMV path)
// =============================================================================

// A staged run has several early exits -- the transaction, the resolve, the
// layout size -- and every one of them must free the ggml context and its three
// buffers. A destructor is the only form of that which stays correct when a new
// early exit is added later.
struct staged_case_resources {
    ggml_context *        ctx        = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_backend_buffer_t input_buf  = nullptr;
    ggml_backend_buffer_t output_buf = nullptr;

    ~staged_case_resources() {
        if (output_buf) {
            ggml_backend_buffer_free(output_buf);
        }
        if (input_buf) {
            ggml_backend_buffer_free(input_buf);
        }
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// The model token must be released on every exit path, the failing ones
// included: an unreleased token leaves the staged weight entry owned by a model
// nothing will ever unload, and the next case's staging then sees a live
// foreign owner instead of a clean cache.
struct staged_model_token {
    ggml_sycl_model_token token{};
    bool                  held = false;

    ~staged_model_token() {
        if (held) {
            (void) ggml_backend_sycl_model_unloaded_token(token);
        }
    }
};

// Everything the preconditions need, not just the numbers: which layout the
// dispatch resolved, whether that pointer is distinct from the AoS storage, and
// the bytes behind that exact pointer.
//
// `outputs` holds one result per probe activation, all computed against the SAME
// staged weight and the SAME resolved layout pointer. Re-staging per probe would
// let a difference between probes be a difference between stagings instead, which
// is the one thing these probes must not be able to confound.
struct staged_gpu_result {
    bool                            ok      = false;
    const char *                    failure = nullptr;
    std::vector<std::vector<float>> outputs;
    std::vector<uint8_t>            layout_bytes;
    ggml_layout_mode                resolved_layout     = GGML_LAYOUT_AOS;
    bool                            layout_ptr_distinct = false;
};

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                                 ggml_tensor *              tensor,
                                                 ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

// Drives the weight through the same lifecycle a model load uses, which is the
// only way a dense COALESCED cache entry gets produced:
// clear registry -> model_load_begin -> register_host_weight_tensor ->
// tensor_set -> model_load_end -> resolve. Registration is honoured only inside
// a transaction and only for a host-resident weight buffer; outside either it
// silently registers nothing and S1-PRELOAD never sees the tensor. Pattern and
// full RCA: tests/test-q8-0-layout-cache-path.cpp (llama.cpp-43uy).
static staged_gpu_result run_dmmv_coalesced_gpu_staged(ggml_backend_t                     backend,
                                                       int                                device_id,
                                                       const char *                       weight_name,
                                                       const void *                       weight_data,
                                                       size_t                             weight_size,
                                                       const std::vector<const float *> & input_probes,
                                                       int                                ncols,
                                                       int                                nrows) {
    staged_gpu_result result{};

    ggml_sycl::test_clear_host_weight_registry();

    // Declared before `res` so it is destroyed after it: buffers go first, then
    // the token, matching the release order test-q8-0-layout-cache-path.cpp uses.
    staged_model_token    model;
    staged_case_resources res;

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    res.ctx = ggml_init(params);
    if (!res.ctx) {
        result.failure = "ggml_init failed";
        return result;
    }

    struct ggml_tensor * weight = ggml_new_tensor_2d(res.ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, weight_name);

    struct ggml_tensor * input = ggml_new_tensor_2d(res.ctx, GGML_TYPE_F32, ncols, 1);
    ggml_set_name(input, "input");

    struct ggml_tensor * out = ggml_mul_mat(res.ctx, weight, input);
    ggml_set_name(out, "output");

    // The weight buffer must be host-resident: ggml_sycl_can_use_layout_for_kernel()
    // admits a non-AoS layout for a weight that has no materialized layout yet
    // only when the buffer is host and weights are evictable, and that is the
    // gate S1-PRELOAD consults before honouring main()'s COALESCED override.
    ggml_backend_buffer_type_t weight_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft    = ggml_backend_get_default_buffer_type(backend);

    res.weight_buf = alloc_tensor_buffer(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    res.input_buf  = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    res.output_buf = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!res.weight_buf || !res.input_buf || !res.output_buf) {
        result.failure = "buffer allocation failed";
        return result;
    }

    ggml_sycl_load_txn load{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) {
        result.failure = "model_load_begin failed";
        return result;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);

    if (ggml_backend_sycl_model_load_end(load, true, &model.token) != GGML_SYCL_LIFECYCLE_OK) {
        result.failure = "model_load_end failed";
        return result;
    }
    model.held = true;

    // This is the same call the mul_mat dispatcher makes to pick a kernel, so
    // its layout is the layout the kernel will run against -- not an adjacent
    // signal about what the cache happens to hold.
    auto resolved          = ggml_sycl_resolve(weight, device_id);
    result.resolved_layout = resolved ? resolved.layout : GGML_LAYOUT_AOS;
    if (!resolved) {
        result.failure = "weight did not resolve";
        return result;
    }
    if (resolved.layout != GGML_LAYOUT_COALESCED) {
        result.failure = "resolved layout is not COALESCED";
        return result;
    }
    result.layout_ptr_distinct = (resolved.ptr != weight->data);
    if (!result.layout_ptr_distinct) {
        result.failure = "COALESCED layout pointer equals the AoS storage pointer";
        return result;
    }

    const size_t layout_size = ggml_sycl::test_layout_bytes(weight, GGML_LAYOUT_COALESCED, device_id);
    const size_t contract_size =
        (size_t) nrows * (size_t) (ncols / 2) + (size_t) nrows * (size_t) (ncols / QK4_0) * sizeof(ggml_fp16_t);
    if (layout_size != contract_size) {
        printf("  FAIL: COALESCED layout bytes=%zu, addressing contract wants %zu\n", layout_size, contract_size);
        result.failure = "COALESCED layout size disagrees with the addressing contract";
        return result;
    }

    fprintf(stderr, "[LAYOUT-CHECK] readback start: source=layout_ptr ptr=%p on_device=%d bytes=%zu (aos ptr=%p)\n",
            resolved.ptr, resolved.on_device ? 1 : 0, layout_size, weight->data);
    result.layout_bytes.resize(layout_size);
    sycl::queue & q = dpct::dev_mgr::instance().get_device(device_id).default_queue();
    q.memcpy(result.layout_bytes.data(), resolved.ptr, layout_size).wait();
    fprintf(stderr, "[LAYOUT-CHECK] readback done (source=layout_ptr)\n");

    struct ggml_cgraph * graph = ggml_new_graph(res.ctx);
    ggml_build_forward_expand(graph, out);

    result.outputs.resize(input_probes.size());
    for (size_t p = 0; p < input_probes.size(); ++p) {
        ggml_backend_tensor_set(input, input_probes[p], 0, (size_t) ncols * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            result.failure = "GPU graph compute failed";
            return result;
        }
        result.outputs[p].resize(nrows);
        ggml_backend_tensor_get(out, result.outputs[p].data(), 0, result.outputs[p].size() * sizeof(float));
    }

    result.ok = true;
    return result;
}

// =============================================================================
// Stage-separating probes (llama.cpp-szv8)
// =============================================================================
//
// The gate compares the GPU against dmmv_q4_0_dfloat_reference(), and a
// disagreement there says only "something upstream of the answer differs". These
// two probes split that into the stages plan Task 9 names, using activations
// chosen so that a whole stage becomes exact rather than merely small:
//
//   * ONE-HOT (row addressing + scale + dequantization). y is 1.0 at a single
//     column and 0 elsewhere, so the answer for row r is exactly the dequantized
//     weight element (r, col) with NO accumulation and NO quantization error:
//     1.0 and 0.0 survive f32, fp16 and 8-bit block quantization unchanged
//     (amax=1 gives q=127 and 127*d back to within one fp16 ulp of the scale).
//     Any disagreement here is structural -- a wrong byte, a wrong scale, or a
//     wrong row -- and cannot be rounding.
//
//   * Q8-SNAPPED (activation quantization). y is the gate's own random vector
//     pushed onto the Q8_0 grid and back, so an 8-bit-activation kernel
//     reproduces it exactly and its quantization error vanishes. If the gate's
//     failure collapses here while the one-hot probe is clean, the causal stage
//     is activation handling and the arithmetic is sound; if it survives, the
//     activation is exonerated and the defect is in the weight path.
//
// The pair is exhaustive over the plan's four candidate stages: materialization
// is already proven byte-exact by compare_device_coalesced_layout(), the one-hot
// probe covers addressing and dequantization, and the Q8-snapped probe covers
// activation handling and accumulation.

// Round-trip through the Q8_0 grid: quantize, then dequantize with the fp16
// scale the block actually stores. The result is a fixed point of that grid, so
// a kernel that re-quantizes it recovers the same integers.
static void snap_activation_to_q8_0(const float * src, float * dst, int ncols) {
    const int               nblocks = ncols / QK4_0;  // QK8_0 == QK4_0 == 32
    std::vector<block_q8_0> q(nblocks);
    quantize_row_q8_0_ref(src, q.data(), ncols);

    for (int b = 0; b < nblocks; ++b) {
        const float d = ggml_fp16_to_fp32(q[b].d);
        for (int j = 0; j < QK4_0; ++j) {
            dst[b * QK4_0 + j] = (float) q[b].qs[j] * d;
        }
    }
}

// The exact answer for a one-hot activation at `col`: the dequantized weight
// element (row, col), read straight out of the AoS blocks.
static float one_hot_expected(const block_q4_0_test * x, int ncols, int row, int col) {
    const int               blocks_per_row = ncols / QK4_0;
    const int               block_in_row   = col / QK4_0;
    const int               elem_in_block  = col % QK4_0;
    const block_q4_0_test * blk            = &x[row * blocks_per_row + block_in_row];

    const uint8_t byte = blk->qs[elem_in_block % (QK4_0 / 2)];
    const int     q    = (elem_in_block < QK4_0 / 2) ? (byte & 0xF) : (byte >> 4);
    return (float) (q - 8) * ggml_fp16_to_fp32(blk->d);
}

struct oracle_score {
    int   errors    = 0;
    float max_diff  = 0.0f;
    float max_rel   = 0.0f;
    int   first_bad = -1;
};

static oracle_score score_against(const std::vector<float> & got, const std::vector<float> & want, float abs_gate) {
    oracle_score s;
    for (size_t i = 0; i < want.size(); ++i) {
        const float diff = fabsf(got[i] - want[i]);
        const float rel  = diff / fmaxf(1.0f, fabsf(want[i]));
        s.max_diff       = fmaxf(s.max_diff, diff);
        s.max_rel        = fmaxf(s.max_rel, rel);
        if (diff > abs_gate) {
            if (s.first_bad < 0) {
                s.first_bad = (int) i;
            }
            s.errors++;
        }
    }
    return s;
}

static bool run_dmmv_coalesced_case(ggml_backend_t gpu_backend,
                                    ggml_backend_t cpu_backend,
                                    int            ncols,
                                    int            nrows,
                                    int            case_index,
                                    bool           verbose) {
    const auto * qfns = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Q4_0 quantization functions not available\n");
        return true;
    }

    std::mt19937 rng(1234 + ncols + nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float(nrows * ncols);
    for (float & v : weight_float) v = dist(rng);

    size_t block_size = qfns->blck_size;
    size_t type_size = qfns->type_size;
    size_t n_elements = weight_float.size();
    size_t nblocks = n_elements / block_size;
    size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_float.data(), weight_quant.data(), n_elements);

    std::vector<float> input_data(ncols);
    for (float & v : input_data) v = dist(rng);

    std::vector<float> cpu_out;
    std::vector<float> cpu_ref;
    const bool debug_coalesced = std::getenv("GGML_SYCL_COALESCED_DMMV_DEBUG") != nullptr;

    // An ATTENTION_WEIGHT name is what makes layout_policy::get_optimal() answer
    // COALESCED for Q4_0 in production; the "weight" this test used maps to
    // tensor_usage::UNKNOWN, whose answer is SOA. Under main()'s override guard
    // the materialized layout would be COALESCED either way, so the name is here
    // to keep the test's premise reachable without a forcing rather than to
    // produce the buffer. Distinct names per case keep the three shapes from
    // colliding on one cache key.
    char weight_name[64];
    snprintf(weight_name, sizeof(weight_name), "blk.%d.attn_q.weight", case_index);

    // Probe 0 is the gate's own activation; the rest are the stage separators
    // described above. All six run against one staging, so a difference between
    // them is a difference in the activation and nothing else.
    std::vector<float> input_q8_snapped(ncols);
    snap_activation_to_q8_0(input_data.data(), input_q8_snapped.data(), ncols);

    // Columns chosen to move every axis of the addressing contract. A one-hot at
    // column c lands in block c/32, byte (c%32)%16, and therefore WORD PLANE
    // ((c%32)%16)/4 -- the plane index is what the coalesced layout interleaves,
    // so covering all four is the point of the set:
    //
    //   0        block 0,  byte 0,  plane 0, low nibble
    //   16       block 0,  byte 0,  plane 0, high nibble  (same byte as col 0)
    //   33       block 1,  byte 1,  plane 0, low nibble   (different BLOCK, same plane)
    //   20       block 0,  byte 4,  plane 1, high nibble
    //   8        block 0,  byte 8,  plane 2, low nibble
    //   ncols-1  last blk, byte 15, plane 3, high nibble  (last element of the row)
    //
    // col=33 is deliberately kept even though it does not add a plane: it is the
    // only column that moves the BLOCK index while holding the plane fixed, which
    // separates a block-stride error from a plane-stride one. (An earlier comment
    // here claimed 33 crossed word planes. It does not -- byte 1 is plane 0 -- and
    // 20 and 8 were added to make the per-plane coverage the comment described
    // actually true.)
    const int                       one_hot_cols[] = { 0, 16, 33, 20, 8, ncols - 1 };
    constexpr int                   n_one_hot      = (int) (sizeof(one_hot_cols) / sizeof(one_hot_cols[0]));
    std::vector<std::vector<float>> one_hot_inputs(n_one_hot, std::vector<float>(ncols, 0.0f));
    for (int p = 0; p < n_one_hot; ++p) {
        one_hot_inputs[p][one_hot_cols[p]] = 1.0f;
    }

    std::vector<const float *> probes;
    probes.push_back(input_data.data());
    probes.push_back(input_q8_snapped.data());
    for (int p = 0; p < n_one_hot; ++p) {
        probes.push_back(one_hot_inputs[p].data());
    }

    staged_gpu_result staged = run_dmmv_coalesced_gpu_staged(gpu_backend, /*device_id=*/0, weight_name,
                                                             weight_quant.data(), weight_bytes, probes, ncols, nrows);
    if (!staged.ok) {
        // Each precondition reports itself, so a run that never reached the
        // coalesced kernel cannot be read as a verdict about that kernel.
        printf("  FAIL: coalesced DMMV path not exercised -- %s (resolved layout=%s)\n",
               staged.failure ? staged.failure : "unknown step", ggml_sycl::test_layout_name(staged.resolved_layout));
        return false;
    }
    printf("  Precondition: resolved layout=%s, layout pointer distinct from AoS storage\n",
           ggml_sycl::test_layout_name(staged.resolved_layout));

    if (!compare_device_coalesced_layout(weight_quant.data(), staged.layout_bytes.data(), ncols, nrows)) {
        printf("  FAIL: the COALESCED buffer the kernel reads does not match the addressing contract\n");
        return false;
    }

    const std::vector<float> &   gpu_out          = staged.outputs[0];
    const std::vector<uint8_t> & weight_coalesced = staged.layout_bytes;

    if (!run_mul_mat_backend(cpu_backend, GGML_TYPE_Q4_0, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, cpu_out)) {
        printf("  FAIL: CPU backend compute failed\n");
        return false;
    }

    cpu_ref.resize(nrows);
    dmmv_q4_0_cpu_reference(reinterpret_cast<const block_q4_0_test *>(weight_quant.data()),
                            input_data.data(), cpu_ref.data(), ncols, nrows);

    if (debug_coalesced && !weight_coalesced.empty()) {
        float max_diff_ref = 0.0f;
        int errors_ref = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_ref[i] - cpu_out[i]);
            if (diff > max_diff_ref) max_diff_ref = diff;
            if (diff > 1e-4f) {
                errors_ref++;
            }
        }
        // The mismatch here is the CPU backend's Q8_0 activation quantization
        // (vec_dot_q4_0_q8_0) against a full-precision activation -- the same
        // effect that made this file's Part 2 read as a kernel bug (szv8).
        printf("  Debug (f32-activation CPU vs CPU backend Q8_0 activation): errors=%d max_diff=%.6f\n", errors_ref,
               max_diff_ref);

        const int blocks_per_row = ncols / QK4_0;
        std::vector<block_q8_0> y_q8(blocks_per_row);
        quantize_row_q8_0_ref(input_data.data(), y_q8.data(), ncols);

        std::vector<float> cpu_from_coalesced(nrows);
        dmmv_q4_0_cpu_from_coalesced_q8_0(weight_coalesced.data(), y_q8.data(), cpu_from_coalesced.data(),
                                          ncols, nrows);

        float max_diff_cpu = 0.0f;
        int errors_cpu = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_from_coalesced[i] - cpu_out[i]);
            if (diff > max_diff_cpu) max_diff_cpu = diff;
            if (diff > 1e-2f) {
                errors_cpu++;
            }
        }
        printf("  Debug (coalesced CPU vs CPU backend): errors=%d max_diff=%.6f\n", errors_cpu, max_diff_cpu);

        float max_diff = 0.0f;
        int errors = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_from_coalesced[i] - gpu_out[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-2f) {
                errors++;
            }
        }
        printf("  Debug (coalesced CPU vs GPU): errors=%d max_diff=%.6f\n", errors, max_diff);
    }

    std::vector<float> ref(nrows);
    dmmv_q4_0_dfloat_reference(reinterpret_cast<const block_q4_0_test *>(weight_quant.data()), input_data.data(),
                               ref.data(), ncols, nrows);

    // Against a contract-faithful oracle the only residual is f32 reassociation
    // (the kernel accumulates per-lane over tiles, then reduces across the
    // sub-group), so the relative gate is 1e-3 -- ~26x above the 3.8e-5 a
    // correct kernel measures in the host model. The absolute gate stays 1e-2.
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int errors = 0;
    for (int i = 0; i < nrows; ++i) {
        float diff = fabsf(gpu_out[i] - ref[i]);
        float denom = fmaxf(1.0f, fabsf(ref[i]));
        float rel = diff / denom;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 1e-2f && rel > 1e-3f) {
            errors++;
        }
    }

    // The CPU backend delta is reported, never gated on: it is dominated by the
    // CPU side's Q8_0 activation quantization, which the GPU path does not have.
    // A value here in the 0.1-0.5 range is expected and is not a defect.
    float cpu_backend_max_diff = 0.0f;
    for (int i = 0; i < nrows; ++i) {
        cpu_backend_max_diff = fmaxf(cpu_backend_max_diff, fabsf(gpu_out[i] - cpu_out[i]));
    }

    // Which kernel actually ran, asserted from the numbers themselves.
    //
    // Three rounds of this ticket reported verdicts about a kernel that never
    // executed, and every structural precondition passed while it happened --
    // the layout resolved COALESCED, the pointer was distinct, the bytes were
    // exact. What separates the two kernels is not their inputs, it is their
    // arithmetic: the dfloat DMMV answer and the q8_1 MMVQ answer differ by
    // ~0.19-0.50 at these shapes, four orders above a correct kernel's residual.
    //
    // So require the dfloat oracle to be the decisively better fit. The margin
    // is a RATIO, not a tolerance, which is what makes it immune to the failure
    // it guards: widening a tolerance cannot satisfy it, because loosening the
    // gate moves both fits together. A correct DMMV run clears it by ~4 orders.
    std::vector<float> ref_mmvq(nrows);
    dmmv_q4_0_q8_1_mmvq_reference(reinterpret_cast<const block_q4_0_test *>(weight_quant.data()), input_data.data(),
                                  ref_mmvq.data(), ncols, nrows);
    const oracle_score fit_dfloat = score_against(gpu_out, ref, 1e-2f);
    const oracle_score fit_mmvq   = score_against(gpu_out, ref_mmvq, 1e-2f);

    const bool kernel_identified = fit_mmvq.max_diff > 100.0f * fmaxf(fit_dfloat.max_diff, 1e-9f);
    if (!kernel_identified) {
        printf(
            "  FAIL: the executing kernel is not the dfloat coalesced DMMV --"
            " max|gpu-dfloat_oracle|=%.6f vs max|gpu-q8_1_MMVQ_oracle|=%.6f\n",
            fit_dfloat.max_diff, fit_mmvq.max_diff);
        if (fit_mmvq.max_diff < fit_dfloat.max_diff) {
            printf(
                "    the q8_1-MMVQ oracle fits BETTER: an MMVQ path took this dispatch."
                " Check that GGML_SYCL_TG_FAST=0 still disables the batch=1 fast-path.\n");
        }
    }

    bool pass = (errors == 0) && kernel_identified;
    if (verbose || !pass) {
        printf(
            "  DMMV coalesced ncols=%d nrows=%d: errors=%d max_diff=%.6f max_rel=%.6f %s"
            " (informational: max|gpu-cpu_backend|=%.6f, Q8_0-activation gap)\n",
            ncols, nrows, errors, max_diff, max_rel, pass ? "PASS" : "FAIL", cpu_backend_max_diff);
    }
    if (!pass) {
        const auto * x_aos = reinterpret_cast<const block_q4_0_test *>(weight_quant.data());

        // How big is the gap between the two oracles alone? If the GPU's spread
        // against the dfloat reference sits at this scale, the dispatch took the
        // Q8_0-activation branch and the reference models the other one; a
        // layout or addressing defect is orders of magnitude larger than this.
        const int               blocks_per_row = ncols / QK4_0;
        std::vector<block_q8_0> y_q8(blocks_per_row);
        quantize_row_q8_0_ref(input_data.data(), y_q8.data(), ncols);

        std::vector<float> ref_q8(nrows);
        dmmv_q4_0_q8_activation_reference(x_aos, y_q8.data(), ref_q8.data(), ncols, nrows);

        float oracle_gap = 0.0f;
        for (int i = 0; i < nrows; ++i) {
            oracle_gap = fmaxf(oracle_gap, fabsf(ref_q8[i] - ref[i]));
        }
        printf(
            "    diagnosis: max|dfloat_oracle - Q8_0_activation_oracle|=%.6f"
            " (the gap the oracle choice alone accounts for)\n",
            oracle_gap);

        // The comparison the previous rounds never made: the GPU scored against
        // the Q8_0-activation oracle DIRECTLY. Scoring the two oracles against
        // each other says how much of the gap that branch COULD explain; scoring
        // the GPU against it says whether it DOES.
        std::vector<float> ref_f32(nrows);
        dmmv_q4_0_cpu_reference(x_aos, input_data.data(), ref_f32.data(), ncols, nrows);

        // Alias, never a second score_against(gpu_out, ref, ...) call: the gate
        // and this diagnostic must report the same fit, and a future threshold
        // edit must not be able to update one and miss the other.
        const oracle_score & s_dfloat = fit_dfloat;
        const oracle_score s_f32    = score_against(gpu_out, ref_f32, 1e-2f);
        const oracle_score s_q8     = score_against(gpu_out, ref_q8, 1e-2f);
        const oracle_score s_mmvq   = fit_mmvq;
        printf("    gpu vs f32-activation    oracle: errors=%d max_diff=%.6f max_rel=%.6f\n", s_f32.errors,
               s_f32.max_diff, s_f32.max_rel);
        printf("    gpu vs dfloat-activation oracle: errors=%d max_diff=%.6f max_rel=%.6f\n", s_dfloat.errors,
               s_dfloat.max_diff, s_dfloat.max_rel);
        printf("    gpu vs Q8_0-activation   oracle: errors=%d max_diff=%.6f max_rel=%.6f\n", s_q8.errors,
               s_q8.max_diff, s_q8.max_rel);
        // Predicted to be the one that lands: errors=0, max_diff at f32
        // reassociation scale (~1e-4). If it does, the executing path is MMVQ
        // with q8_1 activations and the coalesced DMMV kernel never ran.
        printf("    gpu vs q8_1-MMVQ         oracle: errors=%d max_diff=%.6f max_rel=%.6f\n", s_mmvq.errors,
               s_mmvq.max_diff, s_mmvq.max_rel);

        // Same three oracles recomputed from the ACTUAL coalesced bytes the
        // kernel reads, not from the AoS source. Agreement between the two
        // families confirms the addressing contract reproduces the AoS weights;
        // disagreement would move the stage to the layout reader.
        std::vector<float> ref_from_layout(nrows);
        dmmv_q4_0_cpu_from_coalesced(weight_coalesced.data(), input_data.data(), ref_from_layout.data(), ncols, nrows);
        const oracle_score s_layout = score_against(ref_from_layout, ref_f32, 1e-3f);
        printf("    coalesced-bytes oracle vs f32 oracle: errors=%d max_diff=%.6f (contract readback cross-check)\n",
               s_layout.errors, s_layout.max_diff);

        // Stage probe 1: activation quantization. Snapping the activation onto
        // the Q8_0 grid makes an 8-bit-activation kernel exact, so a collapse
        // here names activation handling as the causal stage and a survival
        // exonerates it.
        std::vector<float> ref_snapped(nrows);
        dmmv_q4_0_dfloat_reference(x_aos, input_q8_snapped.data(), ref_snapped.data(), ncols, nrows);
        const oracle_score s_snap = score_against(staged.outputs[1], ref_snapped, 1e-2f);
        printf("    PROBE q8-snapped activation: errors=%d max_diff=%.6f max_rel=%.6f (gate probe was %d / %.6f)\n",
               s_snap.errors, s_snap.max_diff, s_snap.max_rel, errors, max_diff);

        // Stage probe 2: row addressing, scale selection and dequantization.
        // Exact under every activation branch, so any error here is structural.
        for (int p = 0; p < n_one_hot; ++p) {
            const std::vector<float> & got = staged.outputs[2 + p];
            std::vector<float>         want(nrows);
            for (int i = 0; i < nrows; ++i) {
                want[i] = one_hot_expected(x_aos, ncols, i, one_hot_cols[p]);
            }
            const oracle_score s             = score_against(got, want, 1e-3f);
            const int          elem_in_block = one_hot_cols[p] % QK4_0;
            printf("    PROBE one-hot col=%-6d (block=%-4d byte=%-3d plane=%d %s nibble): errors=%d max_diff=%.6f",
                   one_hot_cols[p], one_hot_cols[p] / QK4_0, elem_in_block % (QK4_0 / 2),
                   (elem_in_block % (QK4_0 / 2)) / 4, elem_in_block < QK4_0 / 2 ? "low " : "high", s.errors,
                   s.max_diff);
            if (s.first_bad >= 0) {
                printf(" first_bad_row=%d gpu=%.6f want=%.6f", s.first_bad, got[s.first_bad], want[s.first_bad]);
            }
            printf("\n");
        }

        // The first divergent row, with every oracle side by side. A ratio near
        // one with an additive offset reads as a rounding/quantization stage; a
        // ratio that wanders reads as a structural one.
        if (s_dfloat.first_bad >= 0) {
            const int i = s_dfloat.first_bad;
            printf(
                "    first divergent row=%d: gpu=%.6f f32=%.6f dfloat=%.6f q8=%.6f"
                " q8_1_mmvq=%.6f layout_bytes=%.6f  gpu/dfloat=%.6f\n",
                i, gpu_out[i], ref_f32[i], ref[i], ref_q8[i], ref_mmvq[i], ref_from_layout[i],
                ref[i] != 0.0f ? gpu_out[i] / ref[i] : 0.0f);
        }
    }
    return pass;
}

// =============================================================================
// SoA Layout Conversion (AoS -> SoA)
// =============================================================================

static void convert_aos_to_soa(const block_q4_0_test * aos_data, uint8_t * soa_data, int total_blocks) {
    // SoA layout: all quants first, then all scales
    const size_t qs_bytes = total_blocks * (QK4_0 / 2);

    // Copy quants
    for (int b = 0; b < total_blocks; b++) {
        for (int i = 0; i < QK4_0 / 2; i++) {
            soa_data[b * (QK4_0 / 2) + i] = aos_data[b].qs[i];
        }
    }

    // Copy scales
    ggml_fp16_t * scales = reinterpret_cast<ggml_fp16_t *>(soa_data + qs_bytes);
    for (int b = 0; b < total_blocks; b++) {
        scales[b] = aos_data[b].d;
    }
}

// =============================================================================
// Coalesced Layout Conversion (SoA -> Coalesced)
// =============================================================================

static void convert_soa_to_coalesced(const uint8_t * soa_data, uint8_t * coalesced_data, int ncols, int nrows) {
    const int    blocks_per_row = ncols / QK4_0;
    const int    total_blocks   = nrows * blocks_per_row;
    const size_t qs_bytes       = total_blocks * (QK4_0 / 2);

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;                   // 16 bytes quants per Q4_0 block
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;         // 4 words per block

    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Coalesced row stride: quants only (scales stored globally after all quants)
    const int coalesced_row_stride = blocks_per_row * BYTES_PER_BLOCK;

    // SoA pointers
    const uint8_t *    soa_qs     = soa_data;
    const ggml_fp16_t * soa_scales = reinterpret_cast<const ggml_fp16_t *>(soa_data + qs_bytes);

    for (int row = 0; row < nrows; row++) {
        uint8_t * row_out = coalesced_data + row * coalesced_row_stride;

        for (int tile = 0; tile < tiles_per_row; tile++) {
            uint8_t * tile_out = row_out + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            // Coalesced layout within tile:
            // Word w of block b is at: word_offset = w * (TILE_BLOCKS * 4) + b * 4
            // So all word-0s are together, then all word-1s, etc.

            for (int b = 0; b < TILE_BLOCKS; b++) {
                int             global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;
                const uint8_t * src_block    = soa_qs + global_block * BYTES_PER_BLOCK;

                // Reorder words within tile
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int dst_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(tile_out + dst_offset, src_block + w * 4, 4);
                }
            }
        }

        // Scales are stored globally after all quants
        ggml_fp16_t * global_scales = reinterpret_cast<ggml_fp16_t *>(coalesced_data + qs_bytes);
        for (int b = 0; b < blocks_per_row; b++) {
            global_scales[row * blocks_per_row + b] = soa_scales[row * blocks_per_row + b];
        }
    }
}

// =============================================================================
// Test: Compare SoA DMMV vs Coalesced DMMV
// =============================================================================

bool test_coalesced_vs_soa_dmmv(sycl::queue & q, int ncols, int nrows) {
    printf("\n=== Test: Coalesced vs SoA DMMV (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    // Check if dimensions are compatible with coalesced layout
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;  // Skip, not a failure
    }

    // Generate random test data
    std::mt19937                          rng(42 + ncols + nrows);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int>    byte_dist(0, 255);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = ggml_fp32_to_fp16(scale_dist(rng));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }

    std::vector<float> h_y(ncols);
    for (int i = 0; i < ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    // CPU reference output
    std::vector<float> cpu_output(nrows);
    dmmv_q4_0_cpu_reference(h_aos.data(), h_y.data(), cpu_output.data(), ncols, nrows);

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(ggml_fp16_t);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride + total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // TODO: Once the coalesced DMMV kernel is properly exposed via ggml API,
    // we would run both SoA and coalesced versions and compare.
    //
    // For now, this test fails to drive TDD - the kernel exists in dmmv.cpp
    // but needs proper integration testing.

    printf("  CPU reference computed: first 4 values = ");
    for (int i = 0; i < std::min(4, nrows); i++) {
        printf("%.4f ", cpu_output[i]);
    }
    printf("\n");

    printf("  SoA data size: %zu bytes\n", soa_total_bytes);
    printf("  Coalesced data size: %zu bytes\n", coalesced_total_bytes);
    printf("  Coalesced row stride: %d bytes\n", coalesced_row_stride);

    // This test should fail until proper coalesced DMMV is integrated
    fprintf(stderr, "ERROR: Coalesced DMMV kernel integration test not implemented\n");
    return false;
}

// =============================================================================
// Test: Coalesced Layout Data Integrity
// =============================================================================

bool test_coalesced_layout_integrity(int ncols, int nrows) {
    printf("\n=== Test: Coalesced Layout Data Integrity (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;
    }

    // Generate deterministic test data
    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = ggml_fp32_to_fp16(0.01f * (b % 100 + 1));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>((b + i) % 256);
        }
    }

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(ggml_fp16_t);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride + total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // Verify that all data is preserved (we can dequantize and get same values)
    bool passed = true;
    int  errors = 0;

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;

    const size_t qs_bytes = total_blocks * (QK4_0 / 2);

    for (int row = 0; row < nrows && errors < 10; row++) {
        const uint8_t *    row_data = h_coalesced.data() + row * coalesced_row_stride;
        const ggml_fp16_t * global_scales =
            reinterpret_cast<const ggml_fp16_t *>(h_coalesced.data() + qs_bytes);

        int tiles_per_row = blocks_per_row / TILE_BLOCKS;

        for (int tile = 0; tile < tiles_per_row && errors < 10; tile++) {
            const uint8_t * tile_data = row_data + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int b = 0; b < TILE_BLOCKS && errors < 10; b++) {
                int global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;

                // Extract quants from coalesced layout
                uint8_t extracted_qs[BYTES_PER_BLOCK];
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int src_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(extracted_qs + w * 4, tile_data + src_offset, 4);
                }

                // Compare with original AoS data
                for (int i = 0; i < BYTES_PER_BLOCK; i++) {
                    if (extracted_qs[i] != h_aos[global_block].qs[i]) {
                        fprintf(stderr,
                            "  FAIL: row=%d tile=%d block=%d byte=%d: "
                            "expected=0x%02x got=0x%02x\n",
                            row, tile, b, i, h_aos[global_block].qs[i], extracted_qs[i]);
                        passed = false;
                        errors++;
                    }
                }

                // Check scale
                const float expected_scale = ggml_fp16_to_fp32(h_aos[global_block].d);
                const float actual_scale = ggml_fp16_to_fp32(global_scales[global_block]);
                if (std::fabs(expected_scale - actual_scale) > 1e-6f) {
                    fprintf(stderr, "  FAIL: row=%d block=%d: scale expected=%.4f got=%.4f\n", row,
                           global_block % blocks_per_row, float(expected_scale), float(actual_scale));
                    passed = false;
                    errors++;
                }
            }
        }
    }

    printf("  Layout integrity: %s (errors=%d)\n", passed ? "PASSED" : "FAILED", errors);
    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    printf("=== Q4_0 DMMV Coalesced Layout Tests ===\n");
    printf("Testing coalesced memory layout for DMMV kernel\n");
    printf("Tile configuration: %d blocks per tile, %d bytes per tile\n\n", MMVQ_COALESCED_TILE_BLOCKS,
           MMVQ_COALESCED_TILE_BYTES_Q4_0);

    // Test layout integrity first (CPU-only, no GPU needed)
    printf("=== Part 1: Layout Integrity Tests (CPU) ===\n");

    bool layout_ok = true;
    layout_ok &= test_coalesced_layout_integrity(1024, 16);  // 1024/32 = 32 blocks per row
    layout_ok &= test_coalesced_layout_integrity(2048, 32);  // 2048/32 = 64 blocks per row
    layout_ok &= test_coalesced_layout_integrity(4096, 64);  // 4096/32 = 128 blocks per row (64 rows)

    if (!layout_ok) {
        printf("\nLayout integrity tests FAILED\n");
        return 1;
    }
    printf("\nLayout integrity tests PASSED\n");

    // GPU tests (coalesced DMMV via ggml backend)
    printf("\n=== Part 2: GPU DMMV Tests (Coalesced) ===\n");

    // This override forces BOTH halves of the path, and the second half is easy
    // to misattribute to the ATTENTION_WEIGHT rename:
    //
    //  - Materialization. S1-PRELOAD reads the override at ggml-sycl.cpp:28301
    //    and overwrites its policy answer with it at :28313, gated only on
    //    ggml_sycl_can_use_layout_for_kernel(), whose host-buffer + evictable
    //    early return (:52714) admits any layout for any name. So the COALESCED
    //    buffer here comes from the override, not from the rename -- the rename
    //    would reach the same layout on its own, but never gets the chance while
    //    this guard is held.
    //  - Kernel selection. The orchestrator rejects any preferred kernel whose
    //    layout differs from the override, then calls
    //    pick_kernel_for_layout(COALESCED), which at batch=1 is DMMV_COALESCED.
    //
    // The override must stay for that second reason: DMMV_SOA precedes
    // DMMV_COALESCED in k_mul_mat_priority, and a host + evictable weight is
    // SOA-eligible through the same early return, so without the override the
    // priority list would take DMMV_SOA.
    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_COALESCED);
    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);
    // Without this, NONE of the above binds and the test measures MMVQ
    // (llama.cpp-szv8, hardware-confirmed). ggml-sycl.cpp's TG fast-path claims
    // every batch=1 quantized mul_mat whose weight resolved to a non-AoS layout
    // and dispatches
    //   ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(..., resolved.layout)
    // then returns -- its own comment says it "bypasses orchestrator, name
    // parsing, prefetch, TP checks for maximum speed". GGML_SYCL_FORCE_DMMV and
    // the layout override's KERNEL choice are both consulted downstream of that
    // return, so neither is ever reached. The override still binds on the
    // MATERIALIZATION side, which is why every precondition here passed while
    // the answer came from a different kernel: the coalesced buffer was built,
    // verified byte-exact, and then read by MMVQ.
    //
    // GGML_SYCL_TG_FAST=0 is that path's documented disable and falls through to
    // the unified/orchestrator dispatch, where pick_kernel_for_layout(COALESCED)
    // at batch=1 answers DMMV_COALESCED.
    setenv("GGML_SYCL_TG_FAST", "0", 1);
    // Required by ggml_backend_sycl_register_host_weight_tensor(), which returns
    // early without it -- the tensor would then never reach S1-PRELOAD.
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);

    ggml_backend_sycl_set_unified_cache_budget_pct(90);
    ggml_backend_sycl_set_unified_cache_host_budget_pct(90);

    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return 0;
    }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("SKIP: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 0;
    }

    bool gpu_ok = true;
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 1024, 64, 0, true);
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 2048, 128, 1, true);
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 4096, 256, 2, true);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("\n=== Summary ===\n");
    printf("Layout tests: %s\n", layout_ok ? "PASSED" : "FAILED");
    printf("GPU tests: %s\n", gpu_ok ? "PASSED" : "FAILED");

    return (layout_ok && gpu_ok) ? 0 : 1;
}
