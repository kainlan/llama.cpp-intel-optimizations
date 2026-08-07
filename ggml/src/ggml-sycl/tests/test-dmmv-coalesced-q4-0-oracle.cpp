// Q4_0 coalesced DMMV: layout addressing + the oracle the GPU test compares
// against (llama.cpp-szv8).
//
// Host-only: no SYCL queue, no device, no model. Two things are provable here
// that a GPU run cannot settle, because both are properties of the arithmetic
// rather than of the hardware:
//
//   1. The coalesced addressing in dmmv-coalesced-q4-0-layout.hpp round-trips.
//      A reorder written from the header's offsets, read back through the same
//      offsets, must reconstruct every block -- and must do so for a SLICE of a
//      tensor, which is the case where deriving the scale base from the slice's
//      row count instead of the tensor's silently reads quant bytes as fp16
//      scales.
//
//   2. tests/test-dmmv-q4-0-coalesced.cpp used to compare the GPU against the
//      ggml CPU backend. For a Q4_0 weight the CPU backend quantizes the
//      ACTIVATION to Q8_0; the SYCL DMMV path does not. `oracle-rejects-exact-
//      answer` below scores an EXACT f64 dot product against that old oracle
//      and shows it fails -- so the test could not have been passed by any
//      implementation, and the 12-31% max_rel it reported was the oracle's
//      error, not the kernel's.
//
// `corrupted-kernel-is-caught` is the positive control for the replacement:
// a model with the word-plane interleave dropped must FAIL the new comparison,
// or the new oracle is decorative.

#include "dmmv-coalesced-q4-0-layout.hpp"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using ggml_sycl::dmmv_coalesced_q4_0_qs_offset;
using ggml_sycl::dmmv_coalesced_q4_0_row_quants_bytes;
using ggml_sycl::dmmv_coalesced_q4_0_scale_base;
using ggml_sycl::dmmv_coalesced_q4_0_scale_index;
using ggml_sycl::DMMV_COALESCED_Q4_0_TILE_BYTES;
using ggml_sycl::DMMV_COALESCED_Q4_0_WORD_STRIDE;
using ggml_sycl::DMMV_COALESCED_TILE_BLOCKS;
using ggml_sycl::dmmv_coalesced_tile_count;

// The three shapes tests/test-dmmv-q4-0-coalesced.cpp drives on the GPU.
struct shape {
    int ncols;
    int nrows;
};

static const shape SHAPES[] = {
    { 1024, 64  },
    { 2048, 128 },
    { 4096, 256 },
};

static int failures = 0;

static void check(const bool ok, const std::string & name, const std::string & detail) {
    if (ok) {
        std::cout << "  PASS  " << name << (detail.empty() ? "" : "  (" + detail + ")") << "\n";
        return;
    }
    std::cout << "  FAIL  " << name << "  " << detail << "\n";
    failures++;
}

// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------

// Same generator the GPU test uses, so the numbers below are comparable to the
// numbers it prints.
static void make_inputs(const shape & s, std::vector<float> & weight, std::vector<float> & input) {
    std::mt19937                          rng(1234 + s.ncols + s.nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    weight.resize((size_t) s.nrows * s.ncols);
    for (float & v : weight) {
        v = dist(rng);
    }
    input.resize(s.ncols);
    for (float & v : input) {
        v = dist(rng);
    }
}

// AoS Q4_0 blocks -> coalesced bytes, written entirely through the header's
// offsets. `total_rows` is the row count of the tensor the layout belongs to;
// `rows` may be smaller when only a slice is being written.
static void reorder_to_coalesced(const block_q4_0 *     aos,
                                 std::vector<uint8_t> & out,
                                 int                    blocks_per_row,
                                 int                    total_rows) {
    const int64_t scale_base = dmmv_coalesced_q4_0_scale_base(total_rows, blocks_per_row);
    const int64_t n_blocks   = (int64_t) total_rows * blocks_per_row;
    out.assign((size_t) (scale_base + n_blocks * (int64_t) sizeof(ggml_fp16_t)), 0);

    ggml_fp16_t * scales = (ggml_fp16_t *) (out.data() + scale_base);

    for (int row = 0; row < total_rows; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_0 * blk = &aos[(size_t) row * blocks_per_row + b];
            for (int j = 0; j < QK4_0 / 2; ++j) {
                out[(size_t) dmmv_coalesced_q4_0_qs_offset(row, blocks_per_row, b, j)] = blk->qs[j];
            }
            scales[dmmv_coalesced_q4_0_scale_index(row, blocks_per_row, b)] = blk->d;
        }
    }
}

// -----------------------------------------------------------------------------
// Models of the producers
// -----------------------------------------------------------------------------

// The device AoS -> coalesced reorder (convert.cpp
// reorder_q4_0_aos_to_coalesced_kernel), with the row stride and the scale base
// left as parameters because the row stride is the axis it got wrong: it was
// derived from ggml_sycl_q8_0_coalesced_row_quants_bytes(), which counts 32
// quant bytes per block for Q8_0 where Q4_0 has 16.
//
// `out` is sized by the CONTRACT, which is how the allocation is sized in the
// backend (planner_layout_bytes_coalesced_for_dims). Writes that fall outside
// it are counted rather than performed -- on the device they land in whatever
// follows the allocation.
struct producer_result {
    int64_t oob_writes = 0;
};

static producer_result producer_aos_to_coalesced(const block_q4_0 *     aos,
                                                 std::vector<uint8_t> & out,
                                                 int                    blocks_per_row,
                                                 int                    nrows,
                                                 int64_t                row_quants_bytes,
                                                 int64_t                scale_base) {
    producer_result res;

    const int64_t contract_bytes = dmmv_coalesced_q4_0_scale_base(nrows, blocks_per_row) +
                                   (int64_t) nrows * blocks_per_row * (int64_t) sizeof(ggml_fp16_t);
    out.assign((size_t) contract_bytes, 0);

    const int tiles_per_row = dmmv_coalesced_tile_count(blocks_per_row);

    auto store = [&](int64_t offset, const void * src, size_t bytes) {
        if (offset < 0 || offset + (int64_t) bytes > contract_bytes) {
            res.oob_writes++;
            return;
        }
        std::memcpy(out.data() + offset, src, bytes);
    };

    for (int row = 0; row < nrows; ++row) {
        for (int tile = 0; tile < tiles_per_row; ++tile) {
            for (int block_in_tile = 0; block_in_tile < DMMV_COALESCED_TILE_BLOCKS; ++block_in_tile) {
                const int block_idx = tile * DMMV_COALESCED_TILE_BLOCKS + block_in_tile;
                if (block_idx >= blocks_per_row) {
                    continue;
                }
                const block_q4_0 * blk = &aos[(size_t) row * blocks_per_row + block_idx];

                const int64_t tile_qs_base =
                    (int64_t) row * row_quants_bytes + (int64_t) tile * DMMV_COALESCED_Q4_0_TILE_BYTES;
                for (int word = 0; word < 4; ++word) {
                    const int64_t off =
                        tile_qs_base + (int64_t) word * DMMV_COALESCED_Q4_0_WORD_STRIDE + (int64_t) block_in_tile * 4;
                    store(off, blk->qs + word * 4, 4);
                }
                store(scale_base + ((int64_t) row * blocks_per_row + block_idx) * (int64_t) sizeof(ggml_fp16_t),
                      &blk->d, sizeof(ggml_fp16_t));
            }
        }
    }
    return res;
}

// The SoA -> coalesced reorder (mmvq.cpp convert_q4_0_to_coalesced_kernel):
// one work-item per 4-byte word, block-major in, word-major out, scales left
// where the SoA reorder already put them.
static void producer_soa_to_coalesced(const std::vector<uint8_t> & soa,
                                      std::vector<uint8_t> &       out,
                                      int                          blocks_per_row,
                                      int                          nrows) {
    out = soa;

    const int64_t bytes_per_row = dmmv_coalesced_q4_0_row_quants_bytes(blocks_per_row);
    const int     tiles_per_row = dmmv_coalesced_tile_count(blocks_per_row);

    for (int row = 0; row < nrows; ++row) {
        for (int tile = 0; tile < tiles_per_row; ++tile) {
            for (int block_in_tile = 0; block_in_tile < DMMV_COALESCED_TILE_BLOCKS; ++block_in_tile) {
                for (int word = 0; word < 4; ++word) {
                    const int64_t src = (int64_t) row * bytes_per_row +
                                        (int64_t) tile * DMMV_COALESCED_Q4_0_TILE_BYTES +
                                        (int64_t) block_in_tile * (QK4_0 / 2) + (int64_t) word * 4;
                    const int64_t dst = (int64_t) row * bytes_per_row +
                                        (int64_t) tile * DMMV_COALESCED_Q4_0_TILE_BYTES +
                                        (int64_t) word * DMMV_COALESCED_Q4_0_WORD_STRIDE + (int64_t) block_in_tile * 4;
                    std::memcpy(out.data() + dst, soa.data() + src, 4);
                }
            }
        }
    }
}

// Block-major quants with the scales already in their final place: the SoA
// layout, i.e. a coalesced buffer whose word-plane interleave never ran.
static void build_block_major(const block_q4_0 * aos, std::vector<uint8_t> & out, int blocks_per_row, int nrows) {
    const int64_t scale_base = dmmv_coalesced_q4_0_scale_base(nrows, blocks_per_row);
    const int64_t n_blocks   = (int64_t) nrows * blocks_per_row;
    out.assign((size_t) (scale_base + n_blocks * (int64_t) sizeof(ggml_fp16_t)), 0);

    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        std::memcpy(out.data() + ib * (QK4_0 / 2), aos[ib].qs, QK4_0 / 2);
        std::memcpy(out.data() + scale_base + ib * (int64_t) sizeof(ggml_fp16_t), &aos[ib].d, sizeof(ggml_fp16_t));
    }
}

static int64_t first_byte_mismatch(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return (int64_t) i;
        }
    }
    return a.size() == b.size() ? -1 : (int64_t) n;
}

// -----------------------------------------------------------------------------
// Models of the kernel
// -----------------------------------------------------------------------------

// dequantize_mul_mat_vec_q4_0_coalesced, arithmetic and accumulation order
// preserved: lane `l` owns block_in_tile `l` of every tile, accumulates a
// per-lane partial, then the sub-group butterfly-reduces. `interleaved=false`
// drops the word-plane interleave -- the corruption the positive control uses.
static float kernel_model(const std::vector<uint8_t> & coalesced,
                          const std::vector<float> &   yh,
                          int                          blocks_per_row,
                          int                          total_rows,
                          int                          global_row,
                          bool                         interleaved) {
    const ggml_fp16_t * x_d =
        (const ggml_fp16_t *) (coalesced.data() + dmmv_coalesced_q4_0_scale_base(total_rows, blocks_per_row));
    const int tiles_per_row = dmmv_coalesced_tile_count(blocks_per_row);

    std::vector<float> partial(DMMV_COALESCED_TILE_BLOCKS, 0.0f);

    for (int lane = 0; lane < DMMV_COALESCED_TILE_BLOCKS; ++lane) {
        float partial_sum = 0.0f;
        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int block_idx = tile * DMMV_COALESCED_TILE_BLOCKS + lane;
            if (block_idx >= blocks_per_row) {
                continue;
            }
            const float d =
                ggml_fp16_to_fp32(x_d[dmmv_coalesced_q4_0_scale_index(global_row, blocks_per_row, block_idx)]);

            float block_sum = 0.0f;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const int64_t off     = interleaved ?
                                            dmmv_coalesced_q4_0_qs_offset(global_row, blocks_per_row, block_idx, j) :
                                            (int64_t) global_row * dmmv_coalesced_q4_0_row_quants_bytes(blocks_per_row) +
                                            (int64_t) block_idx * (QK4_0 / 2) + j;
                const uint8_t qs_byte = coalesced[(size_t) off];
                const float   v0      = ((float) (qs_byte & 0xF) - 8.0f) * d;
                const float   v1      = ((float) (qs_byte >> 4) - 8.0f) * d;
                block_sum += v0 * yh[block_idx * QK4_0 + j];
                block_sum += v1 * yh[block_idx * QK4_0 + j + QK4_0 / 2];
            }
            partial_sum += block_sum;
        }
        partial[lane] = partial_sum;
    }

    for (int mask = DMMV_COALESCED_TILE_BLOCKS / 2; mask > 0; mask >>= 1) {
        std::vector<float> next(DMMV_COALESCED_TILE_BLOCKS);
        for (int l = 0; l < DMMV_COALESCED_TILE_BLOCKS; ++l) {
            next[l] = partial[l] + partial[l ^ mask];
        }
        partial.swap(next);
    }
    return partial[0];
}

// -----------------------------------------------------------------------------
// Oracles
// -----------------------------------------------------------------------------

// What the replacement oracle in tests/test-dmmv-q4-0-coalesced.cpp computes:
// dequantize X to f32, multiply by the activation as dfloat, accumulate in f32.
static float dfloat_oracle(const block_q4_0 * x, const std::vector<float> & yh, int blocks_per_row) {
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const float d = ggml_fp16_to_fp32(x[b].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            sum += ((x[b].qs[j] & 0xF) - 8) * d * yh[b * QK4_0 + j];
            sum += ((x[b].qs[j] >> 4) - 8) * d * yh[b * QK4_0 + j + QK4_0 / 2];
        }
    }
    return sum;
}

// What the OLD oracle computed: ggml's CPU backend, i.e. vec_dot_q4_0_q8_0 with
// the activation quantized to Q8_0.
static float cpu_backend_oracle(const block_q4_0 * x, const block_q8_0 * y, int blocks_per_row) {
    float sumf = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        int sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            sumi += ((x[b].qs[j] & 0xF) - 8) * y[b].qs[j];
            sumi += ((x[b].qs[j] >> 4) - 8) * y[b].qs[j + QK4_0 / 2];
        }
        sumf += (float) sumi * ggml_fp16_to_fp32(x[b].d) * ggml_fp16_to_fp32(y[b].d);
    }
    return sumf;
}

// The mathematically exact answer, in f64, from the same quantized weights and
// the unrounded f32 activation.
static double exact_oracle(const block_q4_0 * x, const float * y, int blocks_per_row) {
    double sum = 0.0;
    for (int b = 0; b < blocks_per_row; ++b) {
        const double d = (double) ggml_fp16_to_fp32(x[b].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            sum += d * (double) ((x[b].qs[j] & 0xF) - 8) * (double) y[b * QK4_0 + j];
            sum += d * (double) ((x[b].qs[j] >> 4) - 8) * (double) y[b * QK4_0 + j + QK4_0 / 2];
        }
    }
    return sum;
}

// The GPU test's pass criterion, both thresholds named so a change here is
// visible against the change there.
struct score {
    int   errors;
    float max_diff;
    float max_rel;
};

static score apply_gate(const std::vector<float> & got, const std::vector<float> & ref, float abs_tol, float rel_tol) {
    score s{ 0, 0.0f, 0.0f };
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float rel  = diff / std::fmax(1.0f, std::fabs(ref[i]));
        s.max_diff       = std::fmax(s.max_diff, diff);
        s.max_rel        = std::fmax(s.max_rel, rel);
        if (diff > abs_tol && rel > rel_tol) {
            s.errors++;
        }
    }
    return s;
}

static std::string fmt(const score & s) {
    return "errors=" + std::to_string(s.errors) + " max_diff=" + std::to_string(s.max_diff) +
           " max_rel=" + std::to_string(s.max_rel);
}

// -----------------------------------------------------------------------------
// Cases
// -----------------------------------------------------------------------------

// The header's offsets must round-trip -- and must do so when the rows being
// read are a SLICE, because that is where a scale base taken from the slice's
// row count stops pointing at the scales.
static void case_addressing_round_trips() {
    const int blocks_per_row = 64;
    const int total_rows     = 96;
    const int row_low        = 32;  // deliberately not 0, and not a tile multiple of the row count

    std::vector<block_q4_0> aos((size_t) total_rows * blocks_per_row);
    for (size_t i = 0; i < aos.size(); ++i) {
        aos[i].d = ggml_fp32_to_fp16(0.01f * (float) ((i % 100) + 1));
        for (int j = 0; j < QK4_0 / 2; ++j) {
            aos[i].qs[j] = (uint8_t) ((i * 7 + j * 13) % 256);
        }
    }

    std::vector<uint8_t> coalesced;
    reorder_to_coalesced(aos.data(), coalesced, blocks_per_row, total_rows);

    const int64_t       scale_base = dmmv_coalesced_q4_0_scale_base(total_rows, blocks_per_row);
    const ggml_fp16_t * scales     = (const ggml_fp16_t *) (coalesced.data() + scale_base);

    int mismatches = 0;
    for (int row = row_low; row < total_rows; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_0 & blk = aos[(size_t) row * blocks_per_row + b];
            for (int j = 0; j < QK4_0 / 2; ++j) {
                if (coalesced[(size_t) dmmv_coalesced_q4_0_qs_offset(row, blocks_per_row, b, j)] != blk.qs[j]) {
                    mismatches++;
                }
            }
            if (scales[dmmv_coalesced_q4_0_scale_index(row, blocks_per_row, b)] != blk.d) {
                mismatches++;
            }
        }
    }
    check(mismatches == 0, "addressing-round-trips-over-a-slice", "mismatches=" + std::to_string(mismatches));

    // The quants region must end exactly where the scales begin: an offset that
    // can reach the scale base is the corruption this header is named for.
    const int64_t last_quant_byte =
        dmmv_coalesced_q4_0_qs_offset(total_rows - 1, blocks_per_row, blocks_per_row - 1, QK4_0 / 2 - 1);
    check(last_quant_byte < scale_base, "quants-never-reach-the-scale-base",
          "last=" + std::to_string(last_quant_byte) + " base=" + std::to_string(scale_base));

    // And the defect itself, stated as an inequality rather than a story: a base
    // taken from a slice's row count lands inside the quants.
    const int64_t slice_base = dmmv_coalesced_q4_0_scale_base(total_rows - row_low, blocks_per_row);
    check(slice_base < scale_base, "slice-derived-scale-base-lands-inside-quants",
          "slice_base=" + std::to_string(slice_base) + " correct=" + std::to_string(scale_base));
}

// The kernels were pinned to the addressing contract before the writers were.
// A reader and a writer that disagree produce a buffer no oracle can vouch for,
// and the disagreement is invisible on a host that never runs the writer -- so
// model the writers here, against the same header.
static void case_producers_match_the_contract() {
    const int blocks_per_row = 64;
    const int nrows          = 48;

    std::vector<block_q4_0> aos((size_t) nrows * blocks_per_row);
    for (size_t i = 0; i < aos.size(); ++i) {
        aos[i].d = ggml_fp32_to_fp16(0.01f * (float) ((i % 100) + 1));
        for (int j = 0; j < QK4_0 / 2; ++j) {
            aos[i].qs[j] = (uint8_t) ((i * 11 + j * 5) % 256);
        }
    }

    std::vector<uint8_t> contract;
    reorder_to_coalesced(aos.data(), contract, blocks_per_row, nrows);

    const int64_t contract_row_bytes  = dmmv_coalesced_q4_0_row_quants_bytes(blocks_per_row);
    const int64_t contract_scale_base = dmmv_coalesced_q4_0_scale_base(nrows, blocks_per_row);

    // 1. The device AoS writer, driven from the contract's own numbers.
    std::vector<uint8_t>  produced;
    const producer_result ok =
        producer_aos_to_coalesced(aos.data(), produced, blocks_per_row, nrows, contract_row_bytes, contract_scale_base);
    check(ok.oob_writes == 0 && produced == contract, "device-aos-writer-matches-contract",
          "oob=" + std::to_string(ok.oob_writes) +
              " first_bad=" + std::to_string(first_byte_mismatch(produced, contract)));

    // 2. Positive control, and the defect this case was written for: the Q8_0
    //    row stride. For a tile-aligned row, ggml_sycl_q8_0_coalesced_row_quants_bytes()
    //    is tiles * (32 blocks * 32 bytes) == blocks_per_row * 32 -- exactly
    //    twice the Q4_0 row. It must be caught, and it must be caught as an
    //    overrun and not merely as wrong bytes.
    const int64_t q8_row_bytes  = (int64_t) blocks_per_row * 32;
    const int64_t q8_scale_base = (int64_t) nrows * q8_row_bytes;
    check(q8_row_bytes == 2 * contract_row_bytes, "q8-row-stride-is-twice-the-q4-0-row",
          "q8=" + std::to_string(q8_row_bytes) + " q4_0=" + std::to_string(contract_row_bytes));

    std::vector<uint8_t>  wrong;
    const producer_result bad =
        producer_aos_to_coalesced(aos.data(), wrong, blocks_per_row, nrows, q8_row_bytes, q8_scale_base);
    check(bad.oob_writes > 0, "q8-row-stride-writes-past-the-allocation",
          "oob_writes=" + std::to_string(bad.oob_writes));
    check(wrong != contract, "q8-row-stride-produces-a-different-layout",
          "first_bad=" + std::to_string(first_byte_mismatch(wrong, contract)));

    // 3. The SoA writer: block-major in, contract out.
    std::vector<uint8_t> soa;
    build_block_major(aos.data(), soa, blocks_per_row, nrows);
    std::vector<uint8_t> from_soa;
    producer_soa_to_coalesced(soa, from_soa, blocks_per_row, nrows);
    check(from_soa == contract, "device-soa-writer-matches-contract",
          "first_bad=" + std::to_string(first_byte_mismatch(from_soa, contract)));

    // 4. The fingerprint an un-interleaved buffer leaves, pinned so the next
    //    occurrence is identified instead of re-investigated: byte 4 is block 1
    //    word 0 under the contract and block 0 word 1 under block-major, so a
    //    check comparing the two first disagrees at exactly byte 4.
    check(first_byte_mismatch(soa, contract) == 4, "block-major-buffer-first-differs-at-byte-4",
          "first_bad=" + std::to_string(first_byte_mismatch(soa, contract)));
}

static void case_oracles() {
    for (const shape & s : SHAPES) {
        const int blocks_per_row = s.ncols / QK4_0;

        std::vector<float> weight;
        std::vector<float> input;
        make_inputs(s, weight, input);

        std::vector<block_q4_0> wq((size_t) s.nrows * blocks_per_row);
        quantize_row_q4_0_ref(weight.data(), wq.data(), (int64_t) weight.size());

        std::vector<block_q8_0> yq(blocks_per_row);
        quantize_row_q8_0_ref(input.data(), yq.data(), s.ncols);

        // The activation as the kernel sees it: dfloat == sycl::half in the
        // shipping build, so round through fp16 here too.
        std::vector<float> yh(s.ncols);
        for (int i = 0; i < s.ncols; ++i) {
            yh[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(input[i]));
        }

        std::vector<uint8_t> coalesced;
        reorder_to_coalesced(wq.data(), coalesced, blocks_per_row, s.nrows);

        std::vector<float> gpu(s.nrows), corrupt(s.nrows), ref_dfloat(s.nrows), ref_cpu(s.nrows), ref_exact(s.nrows);
        for (int row = 0; row < s.nrows; ++row) {
            const block_q4_0 * xr = wq.data() + (size_t) row * blocks_per_row;
            gpu[row]              = kernel_model(coalesced, yh, blocks_per_row, s.nrows, row, /*interleaved=*/true);
            corrupt[row]          = kernel_model(coalesced, yh, blocks_per_row, s.nrows, row, /*interleaved=*/false);
            ref_dfloat[row]       = dfloat_oracle(xr, yh, blocks_per_row);
            ref_cpu[row]          = cpu_backend_oracle(xr, yq.data(), blocks_per_row);
            ref_exact[row]        = (float) exact_oracle(xr, input.data(), blocks_per_row);
        }

        const std::string tag = " [" + std::to_string(s.ncols) + "x" + std::to_string(s.nrows) + "]";

        // 1. The replacement oracle accepts a correct kernel, with margin.
        const score kept = apply_gate(gpu, ref_dfloat, 1e-2f, 1e-3f);
        check(kept.errors == 0, "contract-oracle-accepts-correct-kernel" + tag, fmt(kept));

        // 2. The knockout. The exact answer, scored as if it were the GPU,
        //    fails the oracle the test used to gate on. Nothing could pass it.
        const score knockout = apply_gate(ref_exact, ref_cpu, 1e-2f, 1e-2f);
        check(knockout.errors > 0, "old-oracle-rejects-the-exact-answer" + tag, fmt(knockout));

        // 3. Positive control: the replacement oracle is not merely permissive.
        const score caught = apply_gate(corrupt, ref_dfloat, 1e-2f, 1e-3f);
        check(caught.errors > 0, "contract-oracle-catches-corrupted-kernel" + tag, fmt(caught));

        // 4. The replacement oracle's own blind spot, stated as a number rather
        //    than a caveat. ggml_sycl_dmmv_dispatch has two Q4_0 coalesced
        //    branches: src1 as dfloat, and src1 quantized to Q8_0
        //    (dequantize_mul_mat_vec_q4_0_coalesced_q8_0). The dfloat oracle
        //    models the first, so a byte-perfect kernel taking the SECOND fails
        //    this gate. Scored here so a GPU failure at this magnitude is read
        //    as the branch it is, not as a layout defect -- a layout defect is
        //    two orders of magnitude larger (case 3 above).
        const score q8_branch = apply_gate(ref_cpu, ref_dfloat, 1e-2f, 1e-3f);
        check(q8_branch.errors > 0, "dfloat-oracle-rejects-the-q8-activation-branch" + tag, fmt(q8_branch));
    }
}

int main() {
    std::cout << "Q4_0 coalesced DMMV: layout addressing and oracle contract (host-only)\n";
    case_addressing_round_trips();
    case_producers_match_the_contract();
    case_oracles();

    if (failures != 0) {
        std::cout << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
