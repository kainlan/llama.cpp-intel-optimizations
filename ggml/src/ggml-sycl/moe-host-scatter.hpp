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
#include <atomic>
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
    // Workspace for the overlap check when the rows are not already in destination order. Kept, like the vectors
    // above, so a plan reused across flushes allocates nothing once it has seen the largest flush.
    std::vector<uint32_t>          by_dst;
};

// Builds the compact plan for `rows` through a scratch of `scratch_bytes`.
// False, with the plan cleared, when the rows cannot take the compact form and
// the caller must use the per-run copies. Clearing keeps every vector's
// capacity, so a caller that reuses one plan allocates nothing per flush once
// the plan has grown to its largest flush.
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
    // turns away a routing that is already wrong. Rows already in destination order (gate before up, slots
    // ascending) are checked in one pass; only others are sorted, by index into the plan's kept workspace.
    if (rows.size() > UINT32_MAX) {
        return false;
    }
    auto before = [&rows](uint32_t a, uint32_t b) {
        return rows[a].dst != rows[b].dst ? rows[a].dst < rows[b].dst : rows[a].dst_offset < rows[b].dst_offset;
    };
    bool in_dst_order = true;
    for (size_t i = 1; i < rows.size() && in_dst_order; ++i) {
        in_dst_order = !before(static_cast<uint32_t>(i), static_cast<uint32_t>(i - 1));
    }
    plan->by_dst.clear();
    plan->by_dst.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        plan->by_dst.push_back(static_cast<uint32_t>(i));
    }
    if (!in_dst_order) {
        std::sort(plan->by_dst.begin(), plan->by_dst.end(), before);
    }
    for (size_t i = 1; i < plan->by_dst.size(); ++i) {
        const moe_scatter_row & prev = rows[plan->by_dst[i - 1]];
        const moe_scatter_row & cur  = rows[plan->by_dst[i]];
        if (cur.dst == prev.dst && prev.dst_offset + row_bytes > cur.dst_offset) {
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

// Why a flush did not take the compact form; each reason is reported once (llama.cpp-cre6).
enum moe_scatter_decline : uint32_t {
    MOE_SCATTER_DECLINE_NO_SCRATCH = 0,
    MOE_SCATTER_DECLINE_QUEUE_NOT_IN_ORDER,
    MOE_SCATTER_DECLINE_GRAPH_RECORDING,
    MOE_SCATTER_DECLINE_TOO_MANY_DSTS,
    MOE_SCATTER_DECLINE_NOT_COMPACT,
    MOE_SCATTER_DECLINE_SCRATCH_UNRESOLVED,
    MOE_SCATTER_DECLINE_DST_UNRESOLVED,
    MOE_SCATTER_DECLINE_DST_OUT_OF_RANGE,
    MOE_SCATTER_DECLINE_COUNT,
};

static_assert(MOE_SCATTER_DECLINE_COUNT <= 32, "one bit per decline reason");

inline const char * moe_scatter_decline_name(moe_scatter_decline why) {
    switch (why) {
        case MOE_SCATTER_DECLINE_NO_SCRATCH:
            return "no planned scratch was claimed for this context";
        case MOE_SCATTER_DECLINE_QUEUE_NOT_IN_ORDER:
            return "the queue is not in order";
        case MOE_SCATTER_DECLINE_GRAPH_RECORDING:
            return "a SYCL graph is recording";
        case MOE_SCATTER_DECLINE_TOO_MANY_DSTS:
            return "the rows have more destination buffers than one kernel takes";
        case MOE_SCATTER_DECLINE_NOT_COMPACT:
            return "the rows do not take the compact form";
        case MOE_SCATTER_DECLINE_SCRATCH_UNRESOLVED:
            return "the scratch does not resolve on the device";
        case MOE_SCATTER_DECLINE_DST_UNRESOLVED:
            return "a destination does not resolve on the device";
        case MOE_SCATTER_DECLINE_DST_OUT_OF_RANGE:
            return "a destination row lies outside its buffer";
        case MOE_SCATTER_DECLINE_COUNT:
            break;
    }
    return "unknown";
}

// True the first time `why` is reported through `seen` and false after: each reason is reported once, and an
// earlier reason does not hide a later, different one.
inline bool moe_scatter_decline_first(std::atomic<uint32_t> & seen, moe_scatter_decline why) {
    const uint32_t bit = 1u << static_cast<uint32_t>(why);
    return (seen.fetch_or(bit, std::memory_order_relaxed) & bit) == 0;
}

// One expert tensor as the scratch plan sees it: which device's MUL_MAT_ID scatters its host-expert rows (-1 when
// its layer has no planned device, which counts for every device), whether its layer splits gate and up, its row
// length, and whether the placement keeps any of its experts on the host.
struct moe_host_scatter_tensor {
    int    device           = -1;
    bool   split_gate_up    = false;
    size_t row_elems        = 0;
    bool   has_host_experts = false;
};

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

// Bytes of the planned scratch on `device`: the largest flush among the tensors that keep experts on the host and
// are scattered by that device. Zero when there are none, as for an all-VRAM placement: the scatter never runs
// there, so nothing is planned, claimed or reported. False on overflow.
inline bool moe_host_scatter_scratch_bytes_for_device(const std::vector<moe_host_scatter_tensor> & tensors,
                                                      int                                          device,
                                                      size_t                                       n_expert_used,
                                                      size_t *                                     out) {
    *out = 0;
    for (const moe_host_scatter_tensor & t : tensors) {
        if (!t.has_host_experts || (t.device >= 0 && t.device != device)) {
            continue;
        }
        size_t bytes = 0;
        if (!moe_host_scatter_scratch_bytes(n_expert_used, t.split_gate_up, t.row_elems, &bytes)) {
            return false;
        }
        *out = std::max(*out, bytes);
    }
    return true;
}

}  // namespace ggml_sycl
