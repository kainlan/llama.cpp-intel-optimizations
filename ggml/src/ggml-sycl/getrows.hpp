//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GETROWS_HPP
#define GGML_SYCL_GETROWS_HPP

#include "common.hpp"
#include "ggml-common.h"

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

// llama.cpp-go37: pure index math used by k_get_rows_q8_0_aos_pair
// (getrows.cpp) -- the 2-elements/thread Q8_0 AoS GET_ROWS kernel.  Declared
// here (not static in getrows.cpp) so tests/test-sycl-getrows-q8-0-index-math.cpp
// can call the exact functions the device kernel calls, rather than a
// re-implemented copy of the same formulas that could silently drift.
static inline void q8_0_aos_pair_block_index(int64_t i00, int & ib, int & iqs) {
    ib  = static_cast<int>(i00 / QK8_0);
    iqs = static_cast<int>(i00 % QK8_0);
}

static inline bool q8_0_aos_pair_row_out_of_range(int64_t i01, int64_t ne01) {
    return i01 < 0 || i01 >= ne01;
}

#endif // GGML_SYCL_GETROWS_HPP
