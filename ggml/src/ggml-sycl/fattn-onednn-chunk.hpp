#pragma once
//
// Query-row chunking for the oneDNN Graph SDPA execute (llama.cpp-jwjz).
//
// The decomposed SDPA partition that build_and_compile_sdpa() (fattn-onednn.cpp) compiles
// materializes its score matrix, so one execute asks oneDNN Graph's allocator for
//
//   H_q * rows * ne11 * (sizeof(f32) score + sizeof(Q type) probabilities)
//
// bytes of scratch. Q is always f16 at execute time (an f32 Q is converted first), which makes it
// 6 bytes per score cell. Measured on Mistral 7B (H_q = 32): 768 MB at rows = 2048, ne11 = 2048 and
// 1536 MB at ne11 = 4096 (llama.cpp-jwjz captures), plus the five llama.cpp-0oxf anchors at
// rows = 128..512 that onednn_graph_scratch_zone_floor_bytes_swa() (unified-cache.cpp) reproduces.
//
// A request above the DIRECT Graph-scratch cap is allocated anyway, outside the arena, once per
// layer. At ubatch 2048 and n_kv >= 4096 that overcommitted the device: the B70 ran into the SYCL
// watchdog, and the B50 took a copy-engine CAT fault (llama.cpp-jwjz).
//
// Softmax and the probability-times-V product are independent per query row. So running the op as
// consecutive row ranges, each its own execute over the same K/V, gives the same result while
// bounding every execute's scratch. The planner below picks the fewest such chunks that fit a byte
// budget, balanced so the last chunk is not a sliver.
//
// Pure C++ with standard headers only, so a host test pins the arithmetic.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// Graph scratch one SDPA execute requests for `rows` query rows of `H_q` heads against `ne11` keys,
// with probabilities stored in the Q type (`q_elem_size` bytes). Zero for any non-positive extent.
inline uint64_t onednn_fa_graph_scratch_bytes(int64_t H_q, int64_t rows, int64_t ne11, size_t q_elem_size) {
    if (H_q <= 0 || rows <= 0 || ne11 <= 0) {
        return 0;
    }
    const uint64_t cell_bytes = sizeof(float) + static_cast<uint64_t>(q_elem_size);
    return static_cast<uint64_t>(H_q) * static_cast<uint64_t>(rows) * static_cast<uint64_t>(ne11) * cell_bytes;
}

// How one op's `ncols` query rows split into executes. Every chunk but the last has `rows` rows; the
// last has `tail_rows`, which lies in (0, rows]. All zero for an op with no rows.
struct onednn_fa_row_chunks {
    int64_t rows      = 0;
    int64_t n_chunks  = 0;
    int64_t tail_rows = 0;
};

// The fewest chunks whose scratch each fits `budget_bytes`, with balanced row counts.
//
// A budget of zero is no bound, and the op runs as one execute. When even one row does not fit the
// budget, chunks are one row each: that bounds the request as far as row splitting can, and the
// allocator's DIRECT path serves the rest.
inline onednn_fa_row_chunks onednn_fa_plan_row_chunks(int64_t ncols,
                                                      int64_t ne11,
                                                      int64_t H_q,
                                                      size_t  q_elem_size,
                                                      size_t  budget_bytes) {
    onednn_fa_row_chunks plan;
    if (ncols <= 0) {
        return plan;
    }
    plan.rows      = ncols;
    plan.n_chunks  = 1;
    plan.tail_rows = ncols;

    const uint64_t per_row = onednn_fa_graph_scratch_bytes(H_q, 1, ne11, q_elem_size);
    if (budget_bytes == 0 || per_row == 0 || per_row * static_cast<uint64_t>(ncols) <= budget_bytes) {
        return plan;
    }

    int64_t rows_max = static_cast<int64_t>(budget_bytes / per_row);
    if (rows_max < 1) {
        rows_max = 1;
    }
    // The fewest chunks that fit, then the most even row count for that many chunks. Recounting the
    // chunks from the evened rows keeps the last chunk in (0, rows].
    int64_t       n_chunks = (ncols + rows_max - 1) / rows_max;
    const int64_t rows     = (ncols + n_chunks - 1) / n_chunks;
    n_chunks               = (ncols + rows - 1) / rows;

    plan.rows      = rows;
    plan.n_chunks  = n_chunks;
    plan.tail_rows = ncols - (n_chunks - 1) * rows;
    return plan;
}

}  // namespace ggml_sycl
