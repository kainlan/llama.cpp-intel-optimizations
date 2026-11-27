//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Where each active expert's rows sit in the batched MoE prompt executor's activation and output scratch.
//
// The executor sorts the active experts by row count, largest first, and issues one strided GEMM batch per group of
// experts with similar counts. Each group gets a strip of the scratch: one slot of `n_rows` rows per expert, the
// slots one after another, the strips one after another. A strided batch needs equal slots only inside its own
// group, so no group is padded to another group's row count. The row table (moe_pp_pack_row_table) names each routed
// row's packed row, so the gather and the scatter each run as one launch over the node's rows.
//
// This header has no SYCL or backend dependency, so its test needs no device.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ggml_sycl {

// A group's slot is its head's row count rounded up to this, so GEMM shapes recur across layers and passes. The
// planner's row bound for the scratch ring needs the same number, which is why it lives in this dependency-free
// header.
constexpr size_t moe_pp_gemm_row_align = 64;
static_assert((moe_pp_gemm_row_align & (moe_pp_gemm_row_align - 1)) == 0, "the rounding mask needs a power of two");

// Experts [begin, end) of the sorted order. The expert at sorted position i has rows
// [base_row + (i - begin) * n_rows, + n_rows).
struct moe_pp_gemm_group {
    size_t begin    = 0;
    size_t end      = 0;
    size_t n_rows   = 0;  // the group head's row count, rounded up to moe_pp_gemm_row_align
    size_t base_row = 0;
};

struct moe_pp_packed_layout {
    std::vector<moe_pp_gemm_group> groups;
    // Each expert's first row, indexed by sorted position, not by expert id: the caller's sorted list is what maps a
    // position back to its expert.
    std::vector<size_t>            slot_base_row;
    size_t                         packed_rows = 0;  // rows the scratch must hold
};

// Groups `rows_desc` (each active expert's row count, largest first) and lays the groups out. A new group starts
// where the next expert has under half the rows of the current group's head, so padding to the head at most doubles
// an expert's rows; the head is rounded up to moe_pp_gemm_row_align.
// Returns false, leaving *out untouched, when `out` is null, when the input is empty, holds a zero or is not sorted
// largest first, or when a head rounded up to the alignment, or the total rows, would overflow size_t.
inline bool moe_pp_pack_gemm_groups(const std::vector<size_t> & rows_desc, moe_pp_packed_layout * out) {
    constexpr size_t k_max  = std::numeric_limits<size_t>::max();
    constexpr size_t k_mask = moe_pp_gemm_row_align - 1;
    if (out == nullptr || rows_desc.empty()) {
        return false;
    }
    for (size_t i = 0; i < rows_desc.size(); ++i) {
        if (rows_desc[i] == 0 || (i > 0 && rows_desc[i] > rows_desc[i - 1])) {
            return false;
        }
    }
    moe_pp_packed_layout layout;
    layout.slot_base_row.resize(rows_desc.size());
    size_t begin    = 0;
    size_t next_row = 0;
    for (size_t i = 1; i <= rows_desc.size(); ++i) {
        const size_t head = rows_desc[begin];
        // rows * 2 >= head, written so it cannot overflow (rows <= head): the expert stays in the head's group.
        if (i < rows_desc.size() && rows_desc[i] >= head - rows_desc[i]) {
            continue;
        }
        if (head > k_max - k_mask) {
            return false;
        }
        moe_pp_gemm_group group;
        group.begin        = begin;
        group.end          = i;
        group.n_rows       = (head + k_mask) & ~k_mask;
        group.base_row     = next_row;
        const size_t count = i - begin;
        if (group.n_rows > (k_max - next_row) / count) {
            return false;
        }
        for (size_t j = begin; j < i; ++j) {
            layout.slot_base_row[j] = next_row + (j - begin) * group.n_rows;
        }
        next_row += count * group.n_rows;
        layout.groups.push_back(group);
        begin = i;
    }
    layout.packed_rows = next_row;
    *out               = std::move(layout);
    return true;
}

// One routed row of the batched executor: the row-map entry its gather and scatter read, and the packed scratch
// row they write and read. Padding rows of the scratch have no entry.
struct moe_pp_packed_row {
    int32_t map_row    = 0;  // index into the node's row map
    int32_t packed_row = 0;  // row of the activation and output scratch
};

static_assert(sizeof(moe_pp_packed_row) == 2 * sizeof(int32_t), "the device reads the table as two ints");

// spans[i] is the row-map range [first, second) of the expert at sorted position i (the order `layout` was built
// from). Writes one entry per routed row, group by group, expert by expert, row by row: entry r of expert i is
// { first + r, layout.slot_base_row[i] + r }. Returns false, leaving *out untouched, when out is null, spans is
// empty or not the layout's size, a span is empty or inverted, a span holds more rows than its group's n_rows, the
// groups do not cover every position once, or an index does not fit int32_t.
inline bool moe_pp_pack_row_table(const std::vector<std::pair<size_t, size_t>> & spans,
                                  const moe_pp_packed_layout &                   layout,
                                  std::vector<moe_pp_packed_row> *               out) {
    constexpr size_t k_i32 = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (out == nullptr || spans.empty() || spans.size() != layout.slot_base_row.size()) {
        return false;
    }
    size_t total = 0;
    for (const auto & span : spans) {
        if (span.second <= span.first || span.second > k_i32) {
            return false;
        }
        total += span.second - span.first;
    }
    std::vector<moe_pp_packed_row> table;
    table.reserve(total);
    size_t next = 0;  // the groups must visit positions 0..n-1 in order, each once
    for (const moe_pp_gemm_group & group : layout.groups) {
        if (group.begin != next || group.end <= group.begin || group.end > spans.size()) {
            return false;
        }
        for (size_t i = group.begin; i < group.end; ++i) {
            const size_t rows = spans[i].second - spans[i].first;
            const size_t base = layout.slot_base_row[i];
            if (rows > group.n_rows || base > k_i32 - rows) {
                return false;
            }
            for (size_t r = 0; r < rows; ++r) {
                table.push_back({ static_cast<int32_t>(spans[i].first + r), static_cast<int32_t>(base + r) });
            }
        }
        next = group.end;
    }
    if (next != spans.size()) {
        return false;
    }
    *out = std::move(table);
    return true;
}

}  // namespace ggml_sycl
