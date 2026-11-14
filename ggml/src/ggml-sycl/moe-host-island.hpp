#pragma once

// How the prompt's CPU island moves its rows across the device.
//
// A MUL_MAT_ID whose experts live on the host runs them on the CPU: each (token, slot) row reads one activation row of
// src1 and writes one result row of dst. Rows that read the same src1 row share one copy of it: for gate and up, whose
// src1 holds one row per token (ne11 == 1), that is every expert of the token. The down projection's src1 holds a row
// per (slot, token), so there every row has its own activation row and nothing is shared. Either way the plan copies
// each distinct activation row once, as one copy per run of adjacent rows (one copy, when those rows are contiguous in
// src1). The result rows are laid out on the host in destination order, so rows adjacent in dst return in one copy: a
// layer whose every selected expert is on the host returns in one.
//
// SYCL-free on purpose so tests/test-sycl-moe-host-island.cpp can run it without a device.

#include "moe-decode-hostpath.hpp"
#include "moe-host-scatter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// One island row: its activation at `src_offset` bytes in src1's storage, its result bound for `dst_offset` bytes in
// dst's storage.
struct moe_island_row {
    size_t src_offset = 0;
    size_t dst_offset = 0;
};

struct moe_island_plan {
    std::vector<uint32_t>        act_index;  // per row: the activation block row it reads
    std::vector<uint32_t>        out_index;  // per row: the result block row it writes
    std::vector<moe_gather_run>  gather;     // device -> host: src1 bytes into the activation block
    std::vector<moe_scatter_run> scatter;    // host -> device: result block bytes into dst
    size_t                       n_act_rows = 0;
};

// Builds the plan for `rows`, in the caller's order. False, with nothing to submit, when there are no rows, a row
// length is zero, two rows are bound for overlapping destinations, or two activation rows overlap without being the
// same row: each means the routing that produced the rows is already wrong.
inline bool moe_island_plan_build(const std::vector<moe_island_row> & rows,
                                  size_t                              act_row_bytes,
                                  size_t                              out_row_bytes,
                                  moe_island_plan *                   plan) {
    plan->act_index.assign(rows.size(), 0);
    plan->out_index.assign(rows.size(), 0);
    plan->gather.clear();
    plan->scatter.clear();
    plan->n_act_rows = 0;
    if (rows.empty() || act_row_bytes == 0 || out_row_bytes == 0 || rows.size() > UINT32_MAX) {
        return false;
    }
    std::vector<uint32_t> order(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        order[i] = static_cast<uint32_t>(i);
    }

    // Activations: the distinct source rows, ascending, so adjacent rows of src1 are adjacent in the block.
    std::sort(order.begin(), order.end(),
              [&rows](uint32_t a, uint32_t b) { return rows[a].src_offset < rows[b].src_offset; });
    std::vector<size_t> unique_src;
    unique_src.reserve(rows.size());
    for (uint32_t i : order) {
        const size_t src = rows[i].src_offset;
        if (unique_src.empty() || unique_src.back() != src) {
            if (!unique_src.empty() && src < unique_src.back() + act_row_bytes) {
                return false;
            }
            unique_src.push_back(src);
        }
        plan->act_index[i] = static_cast<uint32_t>(unique_src.size() - 1);
    }
    moe_gather_runs_build(unique_src, 0, act_row_bytes, plan->gather);
    plan->n_act_rows = unique_src.size();

    // Results: block row r is the r-th row in destination order.
    std::sort(order.begin(), order.end(),
              [&rows](uint32_t a, uint32_t b) { return rows[a].dst_offset < rows[b].dst_offset; });
    std::vector<moe_scatter_row> placed;
    placed.reserve(rows.size());
    for (size_t r = 0; r < order.size(); ++r) {
        const uint32_t i = order[r];
        if (r > 0 && rows[i].dst_offset < rows[order[r - 1]].dst_offset + out_row_bytes) {
            plan->gather.clear();
            plan->n_act_rows = 0;
            return false;
        }
        plan->out_index[i] = static_cast<uint32_t>(r);
        moe_scatter_row row;
        row.src_offset = r * out_row_bytes;
        row.dst_offset = rows[i].dst_offset;
        row.bytes      = out_row_bytes;
        placed.push_back(row);
    }
    moe_scatter_runs_build(placed, plan->scatter);
    return true;
}

}  // namespace ggml_sycl
