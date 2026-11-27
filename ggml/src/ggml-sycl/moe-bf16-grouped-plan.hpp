//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// The layout of the BF16 grouped prompt arm of MUL_MAT_ID (Inkling 2 Mini plan, Task 24): which experts a prompt node
// routes rows to, where each routed row sits in the activation and output scratch, and each expert's GEMM width.
//
// The arm runs one oneDNN GEMM per active expert, on the expert's BF16 weight where it is loaded. An expert's GEMM
// covers its routed rows rounded up to moe_bf16_grouped_row_align, so GEMM widths recur across layers and passes and
// the primitive cache, keyed on them, stays bounded: a node of n tokens whose picks are distinct experts has at most
// ceil(n / align) widths (with duplicate picks in a token, an expert can hold more rows than there are tokens, and the
// node more widths). The experts' slots sit one after another in ascending expert id, and one expert's rows keep
// the token order of the ids. One batched gather writes every routed row into its packed row and one batched scatter
// writes it back, so each routed row is one entry of `rows`. The packed total is every routed row plus at most
// align - 1 per active expert; with distinct picks that stays within the ring's planned rows
// (pp_moe_onednn_packed_rows_at, moe-scratch-admission.hpp), and the arm admits the packed total before it claims a
// slot, so a node outside that declines.
//
// This header has no SYCL or backend dependency, so its test needs no device.

#pragma once

#include "moe-pp-pack.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml_sycl {

// The row rounding of an expert's GEMM. The batched executor's own (moe-pp-pack.hpp), so the ring's planned rows,
// which are sized from that alignment, hold the arm's layout.
constexpr size_t moe_bf16_grouped_row_align = moe_pp_gemm_row_align;

// The fewest tokens a node needs to take the arm; below it the arm declines as `tokens`, its first check after the
// opt-out, and the direct kernel runs: a decode node, one token, always does. At 64 tokens the arm's GEMMs and their
// host submits beat the direct kernel on both Inkling shapes on the B70 (the step 0 bench,
// docs/backend/data/inkling2/t24-bench-b70-3b132a14ed.txt: gate_up from 32 tokens, down from 64). The bench timed
// those alone, not the gather, the scatter or the two host waits (the ids snapshot and the ring-slot claim).
constexpr int64_t moe_bf16_grouped_min_tokens = 64;

// One routed row: ids slot `slot` of token `token`, the row the gather reads from src1 and the scatter writes to dst,
// and the scratch row it occupies.
struct moe_bf16_grouped_row {
    int32_t slot       = 0;
    int32_t token      = 0;
    int32_t packed_row = 0;
};

// One active expert's slot in the scratch: rows [base_row, base_row + gemm_rows), whose first `rows` are routed rows.
struct moe_bf16_grouped_expert {
    int32_t expert    = 0;
    size_t  rows      = 0;
    size_t  gemm_rows = 0;  // rows rounded up to moe_bf16_grouped_row_align: the GEMM's n
    size_t  base_row  = 0;
};

struct moe_bf16_grouped_plan {
    std::vector<moe_bf16_grouped_expert> experts;      // the active experts, ascending id
    std::vector<moe_bf16_grouped_row>    rows;         // every routed row, by expert in `experts` order, then by token
    std::vector<size_t>                  gemm_widths;  // the distinct gemm_rows, ascending
    size_t                               packed_rows = 0;
};

enum class moe_bf16_grouped_plan_status : uint8_t {
    OK = 0,
    BAD_ARGS,      // a null pointer or a zero count
    BAD_ID,        // an id outside [0, n_experts): the arm declines, and the direct kernel writes that slot's row
    ROW_OVERFLOW,  // a routed row, a token, a slot or a packed row would not fit int32
};

inline const char * moe_bf16_grouped_plan_status_name(moe_bf16_grouped_plan_status status) {
    switch (status) {
        case moe_bf16_grouped_plan_status::OK:
            return "ok";
        case moe_bf16_grouped_plan_status::BAD_ARGS:
            return "bad-args";
        case moe_bf16_grouped_plan_status::BAD_ID:
            return "bad-id";
        case moe_bf16_grouped_plan_status::ROW_OVERFLOW:
            return "row-overflow";
    }
    return "unknown";
}

// The plan for a node from its ids as the host holds them: n_tokens rows of n_slots ids, row-major
// (ids[t * n_slots + s]). Leaves *out untouched unless it returns OK.
inline moe_bf16_grouped_plan_status moe_bf16_grouped_plan_build(const int32_t *         ids,
                                                                size_t                  n_tokens,
                                                                size_t                  n_slots,
                                                                size_t                  n_experts,
                                                                moe_bf16_grouped_plan * out) {
    constexpr size_t k_int32_max = static_cast<size_t>(INT32_MAX);
    if (ids == nullptr || out == nullptr || n_tokens == 0 || n_slots == 0 || n_experts == 0 ||
        n_experts > k_int32_max) {
        return moe_bf16_grouped_plan_status::BAD_ARGS;
    }
    // Before any id is read: every routed row's flat index, token and slot must fit int32.
    if (n_tokens > k_int32_max || n_slots > k_int32_max || n_tokens > k_int32_max / n_slots) {
        return moe_bf16_grouped_plan_status::ROW_OVERFLOW;
    }
    const size_t n_routed = n_tokens * n_slots;

    std::vector<size_t> counts(n_experts, 0);
    for (size_t i = 0; i < n_routed; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= n_experts) {
            return moe_bf16_grouped_plan_status::BAD_ID;
        }
        ++counts[static_cast<size_t>(id)];
    }

    // Each active expert's slot, ascending id, and where its routed rows start in `rows`.
    moe_bf16_grouped_plan plan;
    std::vector<size_t>   next_packed(n_experts, 0);
    std::vector<size_t>   next_entry(n_experts, 0);
    size_t                base    = 0;
    size_t                entries = 0;
    for (size_t e = 0; e < n_experts; ++e) {
        if (counts[e] == 0) {
            continue;
        }
        const size_t gemm_rows =
            (counts[e] + moe_bf16_grouped_row_align - 1) / moe_bf16_grouped_row_align * moe_bf16_grouped_row_align;
        if (gemm_rows > k_int32_max - base) {
            return moe_bf16_grouped_plan_status::ROW_OVERFLOW;
        }
        plan.experts.push_back({ static_cast<int32_t>(e), counts[e], gemm_rows, base });
        plan.gemm_widths.push_back(gemm_rows);
        next_packed[e] = base;
        next_entry[e]  = entries;
        base += gemm_rows;
        entries += counts[e];
    }
    std::sort(plan.gemm_widths.begin(), plan.gemm_widths.end());
    plan.gemm_widths.erase(std::unique(plan.gemm_widths.begin(), plan.gemm_widths.end()), plan.gemm_widths.end());

    // Flat (token, slot) order, so one expert's rows keep the token order of the ids.
    plan.rows.resize(n_routed);
    for (size_t i = 0; i < n_routed; ++i) {
        const size_t e             = static_cast<size_t>(ids[i]);
        plan.rows[next_entry[e]++] = { static_cast<int32_t>(i % n_slots), static_cast<int32_t>(i / n_slots),
                                       static_cast<int32_t>(next_packed[e]++) };
    }
    plan.packed_rows = base;

    *out = std::move(plan);
    return moe_bf16_grouped_plan_status::OK;
}

}  // namespace ggml_sycl
