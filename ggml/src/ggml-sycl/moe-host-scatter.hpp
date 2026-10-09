#pragma once

// How the host-expert MoE results travel back to the device (llama.cpp-cre6).
//
// A MUL_MAT_ID whose experts run on the CPU writes its result rows into one
// compact host-pinned block: row i of the dispatch at src_offset = (first + i)
// * N floats. Each row belongs at its own destination row, dst + i1 * nb1 +
// i2 * nb2, where i1 is the expert slot and i2 the token. The scatter used to
// issue one H2D copy per run of rows whose destinations happen to be adjacent
// (moe_scatter_runs_build), which on a decode token is about one copy per
// host-resident expert: ~12 copies per MoE layer on Qwen3.8.
//
// The compact form (moe_scatter_plan_build) copies the contiguous source block
// to a planned device scratch in as few copies as the sources allow (one, when
// gate and up share a pool), then one small kernel places each scratch row at
// its destination. The row -> destination map travels as kernel arguments, so
// it costs no copy. Rows beyond what the scratch or one kernel's arguments can
// hold go in further chunks of the same shape.
//
// SYCL-free on purpose so tests/test-sycl-moe-host-scatter.cpp can run it
// without a device.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// One result row: `bytes` at `src_offset` in source staging `src`, bound for
// `dst_offset` in destination `dst`. `src` and `dst` are small indices the
// caller maps to its handles.
struct moe_scatter_row {
    int    src        = 0;
    size_t src_offset = 0;
    int    dst        = 0;
    size_t dst_offset = 0;
    size_t bytes      = 0;
};

// The per-run copy: rows merge while source and destination both stay
// contiguous and in the same buffers.
struct moe_scatter_run {
    int    src        = 0;
    size_t src_offset = 0;
    int    dst        = 0;
    size_t dst_offset = 0;
    size_t bytes      = 0;
};

inline void moe_scatter_runs_build(const std::vector<moe_scatter_row> & rows, std::vector<moe_scatter_run> & runs) {
    runs.clear();
    for (const moe_scatter_row & row : rows) {
        if (!runs.empty()) {
            moe_scatter_run & last = runs.back();
            if (last.src == row.src && last.dst == row.dst && last.src_offset + last.bytes == row.src_offset &&
                last.dst_offset + last.bytes == row.dst_offset) {
                last.bytes += row.bytes;
                continue;
            }
        }
        moe_scatter_run run;
        run.src        = row.src;
        run.src_offset = row.src_offset;
        run.dst        = row.dst;
        run.dst_offset = row.dst_offset;
        run.bytes      = row.bytes;
        runs.push_back(run);
    }
}

// Rows one scatter kernel launch places: the row -> destination map is a
// fixed array in the kernel's arguments.
constexpr size_t MOE_SCATTER_MAX_ROWS = 64;
// Destination buffers one compact scatter can write: a layer's gate and up.
constexpr int    MOE_SCATTER_MAX_DSTS = 2;

// One H2D copy into the scratch.
struct moe_scatter_copy {
    int    src            = 0;
    size_t src_offset     = 0;
    size_t scratch_offset = 0;
    size_t bytes          = 0;
};

// One copy-then-kernel step: copies [first_copy, first_copy + n_copies) fill
// scratch rows 0..n_rows-1 with input rows [first_row, first_row + n_rows),
// then one kernel places scratch row r at rows[first_row + r]'s destination.
struct moe_scatter_chunk {
    size_t first_copy = 0;
    size_t n_copies   = 0;
    size_t first_row  = 0;
    size_t n_rows     = 0;
};

struct moe_scatter_plan {
    std::vector<moe_scatter_copy>  copies;
    std::vector<moe_scatter_chunk> chunks;
    size_t                         row_bytes = 0;
};

// Builds the compact plan for `rows` through a scratch of `scratch_bytes`.
// False, with the plan cleared, when the rows cannot take the compact form and
// the caller must use the per-run copies.
inline bool moe_scatter_plan_build(const std::vector<moe_scatter_row> & rows,
                                   size_t                               scratch_bytes,
                                   moe_scatter_plan *                   plan) {
    plan->copies.clear();
    plan->chunks.clear();
    plan->row_bytes = 0;
    if (rows.empty()) {
        return false;
    }
    // The kernel moves whole floats from float-aligned rows, all of one length.
    const size_t row_bytes = rows.front().bytes;
    if (row_bytes == 0 || row_bytes % sizeof(float) != 0) {
        return false;
    }
    const size_t rows_per_chunk = std::min(scratch_bytes / row_bytes, MOE_SCATTER_MAX_ROWS);
    // The kernel indexes a chunk's floats with 32 bits.
    if (rows_per_chunk == 0 || row_bytes / sizeof(float) > UINT32_MAX / rows_per_chunk) {
        return false;
    }
    for (const moe_scatter_row & row : rows) {
        if (row.bytes != row_bytes || row.dst < 0 || row.dst >= MOE_SCATTER_MAX_DSTS ||
            row.dst_offset % sizeof(float) != 0) {
            return false;
        }
    }
    // The per-run copies land in order, so two rows over the same bytes leave the later one; the rows of one kernel
    // launch land in no order. Every row a MUL_MAT_ID dispatch scatters owns its own (slot, token) row, so this only
    // turns away a routing that is already wrong.
    std::vector<const moe_scatter_row *> by_dst;
    by_dst.reserve(rows.size());
    for (const moe_scatter_row & row : rows) {
        by_dst.push_back(&row);
    }
    std::sort(by_dst.begin(), by_dst.end(), [](const moe_scatter_row * a, const moe_scatter_row * b) {
        return a->dst != b->dst ? a->dst < b->dst : a->dst_offset < b->dst_offset;
    });
    for (size_t i = 1; i < by_dst.size(); ++i) {
        if (by_dst[i]->dst == by_dst[i - 1]->dst && by_dst[i - 1]->dst_offset + row_bytes > by_dst[i]->dst_offset) {
            return false;
        }
    }
    for (size_t first = 0; first < rows.size(); first += rows_per_chunk) {
        moe_scatter_chunk chunk;
        chunk.first_row  = first;
        chunk.n_rows     = std::min(rows_per_chunk, rows.size() - first);
        chunk.first_copy = plan->copies.size();
        // Scratch row r holds input row first + r; one copy per run of rows whose sources are contiguous.
        for (size_t r = 0; r < chunk.n_rows; ++r) {
            const moe_scatter_row & row = rows[first + r];
            if (r > 0) {
                moe_scatter_copy & last = plan->copies.back();
                if (last.src == row.src && last.src_offset + last.bytes == row.src_offset) {
                    last.bytes += row_bytes;
                    continue;
                }
            }
            moe_scatter_copy copy;
            copy.src            = row.src;
            copy.src_offset     = row.src_offset;
            copy.scratch_offset = r * row_bytes;
            copy.bytes          = row_bytes;
            plan->copies.push_back(copy);
        }
        chunk.n_copies = plan->copies.size() - chunk.first_copy;
        plan->chunks.push_back(chunk);
    }
    plan->row_bytes = row_bytes;
    return true;
}

// Bytes of the planned scratch: the rows one flush places (both halves of a
// decode gate/up pair when the layer has split gate and up tensors, else one
// op's n_expert_used rows), capped at one kernel launch, times the row bytes.
// False on overflow.
inline bool moe_host_scatter_scratch_bytes(size_t n_expert_used, bool split_gate_up, size_t row_elems, size_t * out) {
    const size_t per_op = split_gate_up ? 2 : 1;
    if (n_expert_used > MOE_SCATTER_MAX_ROWS) {
        n_expert_used = MOE_SCATTER_MAX_ROWS;
    }
    const size_t rows = std::min(per_op * n_expert_used, MOE_SCATTER_MAX_ROWS);
    if (row_elems > SIZE_MAX / sizeof(float) || (rows != 0 && row_elems * sizeof(float) > SIZE_MAX / rows)) {
        return false;
    }
    *out = rows * row_elems * sizeof(float);
    return true;
}

}  // namespace ggml_sycl
