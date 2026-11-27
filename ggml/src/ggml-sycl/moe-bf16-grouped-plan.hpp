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
// ceil(n / align) widths. The experts' slots sit one after another in ascending expert id, and one expert's rows keep
// the token order of the ids. One batched gather writes every routed row into its packed row and one batched scatter
// writes it back, so each routed row is one entry of `rows`. The packed total stays within the ring's planned rows
// (pp_moe_onednn_packed_rows_at, moe-scratch-admission.hpp): every routed row plus at most align - 1 per active expert.
//
// This header has no SYCL or backend dependency, so its test needs no device.

#pragma once

#include "moe-pp-pack.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// The row rounding of an expert's GEMM. The batched executor's own (moe-pp-pack.hpp), so the ring's planned rows,
// which are sized from that alignment, hold the arm's layout.
constexpr size_t moe_bf16_grouped_row_align = moe_pp_gemm_row_align;

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
inline moe_bf16_grouped_plan_status moe_bf16_grouped_build_plan(const int32_t *         ids,
                                                                size_t                  n_tokens,
                                                                size_t                  n_slots,
                                                                size_t                  n_experts,
                                                                moe_bf16_grouped_plan * out) {
    // RED (Task 24): the plan is not built yet.
    (void) ids;
    (void) n_tokens;
    (void) n_slots;
    (void) n_experts;
    (void) out;
    return moe_bf16_grouped_plan_status::BAD_ARGS;
}

}  // namespace ggml_sycl
