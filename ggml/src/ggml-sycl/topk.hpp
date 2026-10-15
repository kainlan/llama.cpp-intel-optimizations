#pragma once

#include "common.hpp"
#include "sycl-kernel-profiler.hpp"

#include <cstddef>
#include <cstdint>

// GGML_OP_TOP_K kernel selection (llama.cpp-1b6u).
//
// The shape decides the kernel, and supports_op and dispatch ask this one
// function with the same inputs, so an op the backend accepts always has a
// kernel and dispatch never re-derives the choice:
//
//   GATING   k <= ggml_sycl_top_k_gating_max_k: the MoE-gating kernel in
//            ggml-sycl.cpp (per-thread insertion lists merged in SLM).
//   BITONIC  the padded row's keys and indices fit in SLM: one work-group per
//            row sorts the whole row and writes the first k (topk.cpp).
//   RADIX    anything larger: a multi-work-group radix select on the ordered
//            float bits, scratch from the op pool (topk.cpp).
//
// BITONIC and RADIX return the same index set for the same row: the k largest
// values, ties broken by the lower column. CPU top_k's order within the k is
// unspecified, and its consumers (the sparse-attention indexer unmasks the
// selected cells with SET_ROWS) do not depend on it.
//
// BITONIC is not the ARGSORT kernel reused: that one keeps only indices in SLM
// and compares x reloaded from global memory with no index tiebreak, so its
// tie order is the network's, not the lower column, and it writes all ncols,
// which would need an ncols-wide scratch row plus a copy of the first k. This
// one keeps keys and indices in SLM under a total order and writes only k.

constexpr int ggml_sycl_top_k_gating_max_k = 32;

enum ggml_sycl_top_k_path {
    GGML_SYCL_TOP_K_PATH_NONE    = 0,
    GGML_SYCL_TOP_K_PATH_GATING  = 1,
    GGML_SYCL_TOP_K_PATH_BITONIC = 2,
    GGML_SYCL_TOP_K_PATH_RADIX   = 3,
};

// SLM the bitonic kernel holds for one row: an ordered key and a column index
// per padded (power-of-two) column.
inline size_t ggml_sycl_top_k_bitonic_slm_bytes(int64_t ncols) {
    size_t ncols_pad = 1;
    while (ncols_pad < (size_t) ncols) {
        ncols_pad *= 2;
    }
    return ncols_pad * (sizeof(uint32_t) + sizeof(int32_t));
}

inline ggml_sycl_top_k_path ggml_sycl_top_k_select_path(int64_t ncols, int64_t k, size_t smpbo) {
    // ggml_top_k requires ne00 >= k; the kernels index columns with int32.
    if (k < 1 || k > ncols || ncols > INT32_MAX) {
        return GGML_SYCL_TOP_K_PATH_NONE;
    }
    if (k <= ggml_sycl_top_k_gating_max_k) {
        return GGML_SYCL_TOP_K_PATH_GATING;
    }
    if (ggml_sycl_top_k_bitonic_slm_bytes(ncols) <= smpbo) {
        return GGML_SYCL_TOP_K_PATH_BITONIC;
    }
    return GGML_SYCL_TOP_K_PATH_RADIX;
}

// The profiler label for a TOP_K kernel submitted on q: category "top_k",
// empty while the kernel profiler is off. The gating kernel in ggml-sycl.cpp
// and the kernels in topk.cpp all take their label from here.
ggml_sycl_profile_label ggml_sycl_top_k_profile_label(sycl::queue & q,
                                                      const char *  name,
                                                      const char *  metadata,
                                                      size_t        bytes);

// BITONIC and RADIX paths. src is nrows contiguous rows of ncols floats, dst is
// nrows contiguous rows of k indices; both resolved for ctx.device.
void ggml_sycl_top_k_large(ggml_backend_sycl_context & ctx,
                           const float *               src,
                           int32_t *                   dst,
                           int64_t                     ncols,
                           int64_t                     nrows,
                           int64_t                     k,
                           ggml_sycl_top_k_path        path);
