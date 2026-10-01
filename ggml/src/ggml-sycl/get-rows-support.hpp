//
// The source types the SYCL GET_ROWS op computes.
//
// One predicate for supports_op to ask, so the list of types the backend admits cannot drift from the types
// ggml_sycl_op_get_rows has an arm for (tests/test-sycl-get-rows-q4-k-source.py checks every admitted type has
// one). Pure C++ with only ggml.h, so a host test can pin it.
//
// Admitting a type nothing computes hands the scheduler an op the backend then aborts on; declining one the
// device could run sends the op to a CPU split whose output a SYCL op reads out of pinned host memory and which
// cannot be recorded into a command graph (llama.cpp-qhfp: qwen35's q4_K token_embd.weight).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

inline bool ggml_sycl_get_rows_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// The (type, layout) pairs a GET_ROWS kernel covers: the layouts the type's kernels can read. supports_op asks it
// with the layout the placement plan materialises the weight in, so a pair no kernel covers is declined before
// placement routes the op, rather than aborting at dispatch.
//
// Q4_K has an AoS kernel only. layout_policy::get_optimal returns AoS for Q4_K for every usage (its MMQ kernels
// are AoS-only), so AoS is the one layout the planner materialises; a Q4_K table in any other layout is a pair
// this function declines. The Q4_K GET_ROWS non-AoS layouts are a support gap to close with kernels, not a
// routing problem: llama.cpp-qhfp-followup. Every other type keeps the arms it already had.
inline bool ggml_sycl_get_rows_layout_supported(ggml_type type, ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return layout == GGML_LAYOUT_AOS;
        default:
            return true;
    }
}
