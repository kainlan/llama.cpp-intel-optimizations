//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <sycl/sycl.hpp>

namespace ggml_sycl {

// Ring-buffered staging buffers for CPU expert dispatch.
// Allocates via unified_alloc() with must_host_pinned constraint for zero-copy PCIe access.
class PinnedBufferPool {
  public:
    PinnedBufferPool() = default;
    ~PinnedBufferPool();

    // Non-copyable, non-movable (owns unified_alloc-backed mem_handles)
    PinnedBufferPool(const PinnedBufferPool &)             = delete;
    PinnedBufferPool & operator=(const PinnedBufferPool &) = delete;
    PinnedBufferPool(PinnedBufferPool &&)                  = delete;
    PinnedBufferPool & operator=(PinnedBufferPool &&)      = delete;

    // Initialize the pool. Allocates via unified_alloc() with must_host_pinned.
    //   q:            SYCL queue for allocation context
    //   device_id:    SYCL device ordinal
    //   max_experts:  max CPU experts per MUL_MAT_ID (top-K)
    //   act_dim:      activation dimension (K = ne00) in floats
    //   out_dim:      output dimension (N = ne01) in floats
    void init(sycl::queue & q, int device_id, size_t max_experts, size_t act_dim, size_t out_dim);

    // Release all buffers via unified_free().
    void shutdown();

    // Acquire buffers for n_experts.
    // act: n_experts * act_stride_ floats for activation staging (D2H target).
    // out: n_experts * out_stride_ floats for CPU output (H2D source).
    struct BufferPair {
        float * act = nullptr;
        float * out = nullptr;
    };

    BufferPair acquire(size_t n_experts);

    // Whether acquire(n_experts) would be served. The pool's capacity is fixed
    // at init() and the buffers really are max_experts_ * dim floats, so an
    // over-capacity request cannot be served at all -- acquire() asserts on it.
    // Callers must ask FIRST and take their own correctly-sized path when this
    // returns false (llama.cpp-sfal: a dispatch entry is one (token, slot)
    // pair, so a MUL_MAT_ID needs up to top-K * n_ubatch of them, while the
    // pool is sized from the 2-token warmup graph).
    bool can_serve(size_t n_experts) const { return is_initialized() && n_experts <= max_experts_; }

    // Release buffers back to pool (no zeroing -- CPU kernels write all read elements).
    void release(BufferPair);

    mem_handle act_handle() const;
    mem_handle out_handle() const;

    bool is_initialized() const { return act_pool_ != nullptr && out_pool_ != nullptr; }

  private:
    float *    act_pool_    = nullptr;
    float *    out_pool_    = nullptr;
    size_t     act_stride_  = 0;  // floats per expert (K)
    size_t     out_stride_  = 0;  // floats per expert (N)
    size_t     max_experts_ = 0;
    int        device_id_   = -1;
    mem_handle act_handle_;
    mem_handle out_handle_;
};

}  // namespace ggml_sycl
