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
// group, so no group is padded to another group's row count.
//
// This header has no SYCL or backend dependency, so its test needs no device.

#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace ggml_sycl {

// Experts [begin, end) of the sorted order. Expert i of the group has rows [base_row + (i - begin) * n_rows, + n_rows).
struct moe_pp_gemm_group {
    size_t begin    = 0;
    size_t end      = 0;
    size_t n_rows   = 0;  // the group head's row count, rounded up to a multiple of 64
    size_t base_row = 0;
};

struct moe_pp_packed_layout {
    std::vector<moe_pp_gemm_group> groups;
    std::vector<size_t>            expert_base_row;  // per expert, in the sorted order
    size_t                         packed_rows = 0;  // rows the scratch must hold
};

// Groups `rows_desc` (each active expert's row count, largest first) and lays the groups out. A new group starts
// where the next expert has under half the rows of the current group's head, so padding to the head at most doubles
// an expert's rows; the head is rounded up to a multiple of 64 so GEMM shapes recur across layers and passes.
// Returns false, leaving *out untouched, when the input is empty, holds a zero, is not sorted largest first, or a
// row count overflows.
inline bool moe_pp_pack_gemm_groups(const std::vector<size_t> & rows_desc, moe_pp_packed_layout * out) {
    constexpr size_t k_max = std::numeric_limits<size_t>::max();
    if (out == nullptr || rows_desc.empty()) {
        return false;
    }
    for (size_t i = 0; i < rows_desc.size(); ++i) {
        if (rows_desc[i] == 0 || (i > 0 && rows_desc[i] > rows_desc[i - 1])) {
            return false;
        }
    }
    moe_pp_packed_layout layout;
    layout.expert_base_row.resize(rows_desc.size());
    size_t begin    = 0;
    size_t next_row = 0;
    for (size_t i = 1; i <= rows_desc.size(); ++i) {
        const size_t head = rows_desc[begin];
        // rows * 2 >= head, written so it cannot overflow (rows <= head): the expert stays in the head's group.
        if (i < rows_desc.size() && rows_desc[i] >= head - rows_desc[i]) {
            continue;
        }
        if (head > k_max - 63) {
            return false;
        }
        moe_pp_gemm_group group;
        group.begin        = begin;
        group.end          = i;
        group.n_rows       = (head + 63) & ~static_cast<size_t>(63);
        group.base_row     = next_row;
        const size_t count = i - begin;
        if (group.n_rows > (k_max - next_row) / count) {
            return false;
        }
        for (size_t j = begin; j < i; ++j) {
            layout.expert_base_row[j] = next_row + (j - begin) * group.n_rows;
        }
        next_row += count * group.n_rows;
        layout.groups.push_back(group);
        begin = i;
    }
    layout.packed_rows = next_row;
    *out               = std::move(layout);
    return true;
}

}  // namespace ggml_sycl
