// Column-split form of the Q8_0 matvec: the dispatch rule and the chunk arithmetic.
//
// Host-only: no SYCL header, no queue, no device, no model. The rule and the index arithmetic live in
// ggml-sycl/mmvq-launch-geometry.hpp, which includes nothing, so this file compiles that production header directly.
//
// What it pins:
//   * which shapes take the column-split form, on the two cards' compute-unit counts, and that every shape with at
//     least as many rows as the card has compute units keeps the row form;
//   * that a chunk count is always a power of two that divides the work-group's sub-groups, never more than the row's
//     sub-group steps;
//   * that the chunks of a row read every quant of every block exactly once, for the SOA lane mapping (four lanes per
//     block, two ints each) and the coalesced one (the shared step walk over tiles, then the per-tile lane loop), the
//     latter at several tile sizes so it holds without the tile matching the sub-group;
//   * that one chunk is the row form's own indexing, and that the work-groups cover the rows with whole work-groups;
//   * of the device test's case table: every case takes the split on both cards, and each layout has a case whose
//     last work-group holds padded rows, so the device run exercises padded sub-groups at the barrier.
// The device test (test-sycl-mmvq-q8-0-column-split) shows the kernels compute the right numbers; this one shows the
// numbers it needs are the right ones on shapes no single device run can sweep.

#include "ggml-sycl/mmvq-launch-geometry.hpp"
#include "test-sycl-mmvq-q8-0-column-split-cases.h"

#include <cstdio>
#include <string>
#include <vector>

using ggml_sycl::mmvq_column_split_block_stride;
using ggml_sycl::mmvq_column_split_chunks;
using ggml_sycl::mmvq_column_split_first_block;
using ggml_sycl::mmvq_column_split_first_step;
using ggml_sycl::mmvq_column_split_rows_per_workgroup;
using ggml_sycl::mmvq_column_split_step_stride;
using ggml_sycl::mmvq_column_split_steps;
using ggml_sycl::MMVQ_COLUMN_SPLIT_SUBGROUPS;
using ggml_sycl::mmvq_column_split_workgroups;

// The row form's constants, transcribed here because this test includes no backend header. WARP_SIZE is 32 on every
// Intel build (presets.hpp, ggml-sycl/CMakeLists.txt); Q8_0 has QI8_0 = 8 ints per block and vdr_mmvq = 2.
static constexpr int WARP               = 32;
static constexpr int QI8_0              = 8;
static constexpr int VDR                = 2;
static constexpr int SOA_STEP           = (VDR * WARP + QI8_0 - 1) / QI8_0;  // blocks one sub-group covers per step: 8
static constexpr int SOA_BLOCK_LANES    = QI8_0 / VDR;                       // lanes per block: 4
// MMVQ_COALESCED_TILE_BLOCKS (common.hpp): the coalesced row form's step is one tile of this many blocks.
static constexpr int TILE_BLOCKS        = 32;
static constexpr int COALESCED_STEP     = TILE_BLOCKS;
// The 16 sub-groups of the row form's work-group (`num_subgroups` at every mmvq.cpp launch site).
static constexpr int ROW_FORM_SUBGROUPS = 16;
// SYCL max_compute_units on the two cards (EUs): Arc Pro B70 and Arc Pro B50.
static constexpr int B70_EUS            = 256;
static constexpr int B50_EUS            = 128;

static int g_failures = 0;
static int g_checks   = 0;

static void expect(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

static void expect_chunks(int rows, int ncols, int step, int eus, int want, const char * what) {
    const int         got   = mmvq_column_split_chunks(rows, ncols / 32, step, eus);
    const std::string label = std::string(what) + ": rows=" + std::to_string(rows) + " ncols=" + std::to_string(ncols) +
                              " step=" + std::to_string(step) + " eus=" + std::to_string(eus) +
                              " -> chunks=" + std::to_string(got) + " (want " + std::to_string(want) + ")";
    expect(got == want, label);
    if (got == want) {
        std::printf("PASS: %s\n", label.c_str());
    }
}

static bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

// The two shapes of the Qwen3.8 decode profile, on both cards and both layouts the kernel exists for.
static void test_task_shapes_take_the_column_split_form() {
    for (const int eus : { B70_EUS, B50_EUS }) {
        // 4 x 10240: 320 blocks, 40 SOA steps, 10 coalesced tiles.
        expect_chunks(4, 10240, SOA_STEP, eus, 16, "hc_inject soa");
        expect_chunks(4, 10240, COALESCED_STEP, eus, 8, "hc_inject coalesced");
        // 48 x 2560: 80 blocks, 10 SOA steps, 3 coalesced tiles (the last one partial).
        expect_chunks(48, 2560, SOA_STEP, eus, 8, "node 48x2560 soa");
        expect_chunks(48, 2560, COALESCED_STEP, eus, 2, "node 48x2560 coalesced");
    }
}

// The row form is kept wherever it already gives every compute unit a sub-group. These shapes must launch exactly
// what they launched before the column-split form existed.
static void test_shapes_with_enough_rows_keep_the_row_form() {
    expect_chunks(B70_EUS, 10240, SOA_STEP, B70_EUS, 1, "rows == B70 EUs");
    expect_chunks(B70_EUS - 1, 10240, SOA_STEP, B70_EUS, 16, "one row fewer than B70 EUs");
    expect_chunks(B50_EUS, 10240, SOA_STEP, B50_EUS, 1, "rows == B50 EUs");
    expect_chunks(B50_EUS - 1, 10240, SOA_STEP, B50_EUS, 16, "one row fewer than B50 EUs");
    // Shapes the existing gates and profiles run: GPT-OSS 20B q8_0 SOA (512 x 2880, 4096 x 2880), Mistral-sized
    // projections, a vocabulary-sized output.
    for (const int eus : { B70_EUS, B50_EUS }) {
        expect_chunks(512, 2880, SOA_STEP, eus, 1, "gpt-oss 512x2880");
        expect_chunks(4096, 2880, SOA_STEP, eus, 1, "gpt-oss 4096x2880");
        expect_chunks(2880, 4096, COALESCED_STEP, eus, 1, "gpt-oss 2880x4096 coalesced");
        expect_chunks(4096, 4096, SOA_STEP, eus, 1, "4096x4096");
        expect_chunks(14336, 4096, COALESCED_STEP, eus, 1, "14336x4096 coalesced");
        expect_chunks(201088, 2880, SOA_STEP, eus, 1, "vocab 201088x2880");
    }
}

// A row too short to give two chunks one step each gains nothing from a split and keeps the row form.
static void test_short_rows_keep_the_row_form() {
    expect_chunks(4, 256, SOA_STEP, B70_EUS, 1, "one SOA step (8 blocks)");
    expect_chunks(4, 32 * 9, SOA_STEP, B70_EUS, 2, "nine blocks round up to two SOA steps");
    expect_chunks(4, 1024, COALESCED_STEP, B70_EUS, 1, "one coalesced tile");
    expect_chunks(4, 1056, COALESCED_STEP, B70_EUS, 2, "33 blocks round up to two tiles");
    expect_chunks(1, 32, SOA_STEP, B70_EUS, 1, "one block");
}

// No answer from the device, an empty slice or a nonsense step must never pick a split.
static void test_unknown_inputs_keep_the_row_form() {
    expect(mmvq_column_split_chunks(4, 320, SOA_STEP, 0) == 1, "compute_units 0 keeps the row form");
    expect(mmvq_column_split_chunks(4, 320, SOA_STEP, -1) == 1, "negative compute_units keeps the row form");
    expect(mmvq_column_split_chunks(0, 320, SOA_STEP, B70_EUS) == 1, "an empty slice keeps the row form");
    expect(mmvq_column_split_chunks(-3, 320, SOA_STEP, B70_EUS) == 1, "a negative slice keeps the row form");
    expect(mmvq_column_split_chunks(4, 0, SOA_STEP, B70_EUS) == 1, "zero blocks keep the row form");
    expect(mmvq_column_split_chunks(4, 320, 0, B70_EUS) == 1, "a zero step keeps the row form");
    expect(mmvq_column_split_steps(320, 0) == 0, "a zero step has no steps");
    expect(mmvq_column_split_steps(0, SOA_STEP) == 0, "no blocks have no steps");
}

// Over a sweep: the chunk count is a power of two, fits the work-group, divides its sub-groups and gives each chunk
// at least one of the row form's steps; and it is 1 exactly when the row form is kept.
static void test_chunk_count_invariants_over_a_sweep() {
    int checked = 0;
    for (const int eus : { B70_EUS, B50_EUS }) {
        for (const int step : { SOA_STEP, COALESCED_STEP }) {
            for (int rows = 1; rows <= 300; ++rows) {
                for (int blocks = 1; blocks <= 400; ++blocks) {
                    const int  chunks = mmvq_column_split_chunks(rows, blocks, step, eus);
                    const int  steps  = mmvq_column_split_steps(blocks, step);
                    const bool ok     = is_power_of_two(chunks) && chunks <= MMVQ_COLUMN_SPLIT_SUBGROUPS &&
                                    MMVQ_COLUMN_SPLIT_SUBGROUPS % chunks == 0 && chunks <= steps &&
                                    (chunks == 1) == (rows >= eus || steps < 2) &&
                                    (chunks == 1 || chunks * 2 > steps || chunks == MMVQ_COLUMN_SPLIT_SUBGROUPS);
                    if (!ok) {
                        expect(false, "invariant: rows=" + std::to_string(rows) + " blocks=" + std::to_string(blocks) +
                                          " step=" + std::to_string(step) + " eus=" + std::to_string(eus) +
                                          " chunks=" + std::to_string(chunks));
                        return;
                    }
                    ++checked;
                }
            }
        }
    }
    expect(checked == 2 * 2 * 300 * 400, "the sweep visited every shape");
    std::printf("PASS: chunk-count invariants over %d shapes\n", checked);
}

// The split's work-group is the row form's: same sub-group count, so the size already known to launch on these cards.
static void test_work_group_matches_the_row_form() {
    expect(MMVQ_COLUMN_SPLIT_SUBGROUPS == ROW_FORM_SUBGROUPS, "the split work-group has the row form's 16 sub-groups");
}

// Every (block, quant int) of a row is read by exactly one lane of exactly one chunk, for the SOA lane mapping:
// lane -> block i = first_block(chunk, lane / 4) stepping by block_stride, quant ints vdr * (lane % 4) .. + vdr - 1.
static bool soa_coverage_is_exact(int blocks, int chunks) {
    std::vector<int> hits(static_cast<size_t>(blocks) * QI8_0, 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int lane = 0; lane < WARP; ++lane) {
            const int stride = mmvq_column_split_block_stride(chunks, SOA_STEP);
            for (int i = mmvq_column_split_first_block(chunk, lane / SOA_BLOCK_LANES, SOA_STEP); i < blocks;
                 i += stride) {
                for (int v = 0; v < VDR; ++v) {
                    ++hits[static_cast<size_t>(i) * QI8_0 + VDR * (lane % SOA_BLOCK_LANES) + v];
                }
            }
        }
    }
    for (const int h : hits) {
        if (h != 1) {
            return false;
        }
    }
    return true;
}

// The coalesced mapping, as mul_mat_vec_q8_0_coalesced_column_split walks it: chunk -> the tiles the shared step walk
// gives it (first_step, step_stride, in tile units), up to the row's tiles; then mmvq_q8_0_coalesced_accumulate_tile's
// lane loop inside each tile (blocks lane, lane + WARP, ... of the tile, skipping blocks past the row).
static bool coalesced_coverage_is_exact(int blocks, int chunks, int tile_blocks) {
    const int        tiles = mmvq_column_split_steps(blocks, tile_blocks);
    std::vector<int> hits(static_cast<size_t>(blocks), 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int tile_stride = mmvq_column_split_step_stride(chunks);
        for (int tile = mmvq_column_split_first_step(chunk); tile < tiles; tile += tile_stride) {
            for (int lane = 0; lane < WARP; ++lane) {
                for (int block_in_tile = lane; block_in_tile < tile_blocks; block_in_tile += WARP) {
                    const int block = tile * tile_blocks + block_in_tile;
                    if (block < blocks) {
                        ++hits[static_cast<size_t>(block)];
                    }
                }
            }
        }
    }
    for (const int h : hits) {
        if (h != 1) {
            return false;
        }
    }
    return true;
}

static void test_chunks_read_every_quant_exactly_once() {
    for (const int chunks : { 1, 2, 4, 8, 16 }) {
        for (int blocks = 1; blocks <= 400; ++blocks) {
            if (!soa_coverage_is_exact(blocks, chunks)) {
                expect(false, "soa coverage: blocks=" + std::to_string(blocks) + " chunks=" + std::to_string(chunks));
                return;
            }
            // The production tile, and tiles smaller and larger than a sub-group: the walk is over tile indices.
            for (const int tile_blocks : { TILE_BLOCKS, 16, 64 }) {
                if (!coalesced_coverage_is_exact(blocks, chunks, tile_blocks)) {
                    expect(false, "coalesced coverage: blocks=" + std::to_string(blocks) +
                                      " chunks=" + std::to_string(chunks) + " tile=" + std::to_string(tile_blocks));
                    return;
                }
            }
        }
    }
    expect(true, "coverage");
    std::printf("PASS: every quant read exactly once, chunks 1..16, blocks 1..400, both layouts, tiles 16/32/64\n");
}

// One chunk is the row form's loop: every step in order (the coalesced row form's tile loop), and in blocks a start at
// lane / 4 stepping by one sub-group step (the SOA row form's loop).
static void test_one_chunk_is_the_row_form_indexing() {
    expect(mmvq_column_split_first_step(0) == 0, "one chunk starts at the row's first step");
    expect(mmvq_column_split_step_stride(1) == 1, "one chunk takes every step");
    for (int lane_block = 0; lane_block < WARP; ++lane_block) {
        expect(mmvq_column_split_first_block(0, lane_block, SOA_STEP) == lane_block,
               "one chunk starts where a row does");
    }
    expect(mmvq_column_split_block_stride(1, SOA_STEP) == SOA_STEP, "one SOA chunk strides one sub-group step");
}

// Rows per work-group times work-groups covers the slice with whole work-groups and no wholly padded work-group.
static void test_work_groups_cover_the_rows() {
    for (const int chunks : { 1, 2, 4, 8, 16 }) {
        const int per_wg = mmvq_column_split_rows_per_workgroup(chunks);
        expect(per_wg * chunks == MMVQ_COLUMN_SPLIT_SUBGROUPS, "a work-group's rows times chunks fill its sub-groups");
        for (int rows = 1; rows <= 600; ++rows) {
            const int wgs = mmvq_column_split_workgroups(rows, chunks);
            if (!(wgs * per_wg >= rows && (wgs - 1) * per_wg < rows)) {
                expect(false, "work-groups: rows=" + std::to_string(rows) + " chunks=" + std::to_string(chunks));
                return;
            }
        }
        expect(mmvq_column_split_workgroups(0, chunks) == 0, "an empty slice launches no work-group");
    }
    expect(mmvq_column_split_workgroups(4, 16) == 4, "hc_inject soa: four work-groups of one row");
    expect(mmvq_column_split_workgroups(48, 8) == 24, "48x2560 soa: 24 work-groups of two rows");
    expect(mmvq_column_split_workgroups(48, 2) == 6, "48x2560 coalesced: six work-groups of eight rows");
    std::printf("PASS: work-groups cover every slice up to 600 rows\n");
}

// The device test's cases, on both cards: each one splits (or the device run would compare the row form with itself),
// and per layout at least one leaves padded rows in its last work-group.
static void test_device_cases_split_and_pad() {
    for (const int eus : { B70_EUS, B50_EUS }) {
        int padded_soa       = 0;
        int padded_coalesced = 0;
        for (const mmvq_q8_0_column_split_case & tc : MMVQ_Q8_0_COLUMN_SPLIT_CASES) {
            const int         step   = tc.coalesced ? COALESCED_STEP : SOA_STEP;
            const int         chunks = mmvq_column_split_chunks(tc.nrows, tc.ncols / 32, step, eus);
            const int         per_wg = mmvq_column_split_rows_per_workgroup(chunks);
            const int         padded = mmvq_column_split_workgroups(tc.nrows, chunks) * per_wg - tc.nrows;
            const std::string label  = std::string("device case ") + tc.name + " eus=" + std::to_string(eus) +
                                      ": chunks=" + std::to_string(chunks) + " padded=" + std::to_string(padded);
            expect(chunks > 1, label + " takes the split");
            expect(tc.row_low >= 0 && tc.row_low + tc.nrows <= tc.total_rows, label + " is a slice of its tensor");
            expect(!tc.coalesced || (tc.row_low == 0 && tc.nrows == tc.total_rows),
                   label + " (the coalesced layout is never sliced)");
            if (padded > 0) {
                ++(tc.coalesced ? padded_coalesced : padded_soa);
            }
            std::printf("%s: %s\n", chunks > 1 ? "PASS" : "FAIL", label.c_str());
        }
        expect(padded_soa > 0, "a SOA device case pads its last work-group on eus=" + std::to_string(eus));
        expect(padded_coalesced > 0, "a coalesced device case pads its last work-group on eus=" + std::to_string(eus));
    }
}

int main() {
    test_task_shapes_take_the_column_split_form();
    test_shapes_with_enough_rows_keep_the_row_form();
    test_short_rows_keep_the_row_form();
    test_unknown_inputs_keep_the_row_form();
    test_chunk_count_invariants_over_a_sweep();
    test_work_group_matches_the_row_form();
    test_chunks_read_every_quant_exactly_once();
    test_one_chunk_is_the_row_form_indexing();
    test_work_groups_cover_the_rows();
    test_device_cases_split_and_pad();

    if (g_failures != 0) {
        std::printf("FAIL: mmvq column-split rule: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("PASS: mmvq column-split rule (%d checks)\n", g_checks);
    return 0;
}
