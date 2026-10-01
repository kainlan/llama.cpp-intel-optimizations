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

    // Acquire the pool's buffer pair for n_experts entries.
    // act: base of the activation staging region (D2H target), K floats per entry.
    // out: base of the CPU output region (H2D source), N floats per entry.
    // The pair is always the pool BASE: it is the same pair on every call. Which entries of it a
    // dispatch owns is decided by reserve(); address them as act + first * K and out + first * N.
    struct BufferPair {
        float * act = nullptr;
        float * out = nullptr;
    };

    BufferPair acquire(size_t n_experts);

    // Reserve n_experts consecutive entries and return the index of the first one.
    // The pool is a ring over its max_experts_ entries: successive reservations advance, and one
    // that would run past the end restarts at entry 0.  Requires can_serve(n).
    //
    // WHAT THE RING GUARANTEES, AND WHAT IT DOES NOT.  It makes two dispatches of ONE
    // MUL_MAT_ID disjoint: the hot and cold groups are slices of a single reservation, so the
    // cold dispatch cannot overwrite the region the hot scatter's H2D is still reading, with no
    // host wait (llama.cpp-4hg7).  It does NOT keep different ops disjoint.  Entry offsets scale
    // with each op's own K and N (gate/up and down differ), so spans of different ops alias
    // byte-wise without any wrap; and a pool sized to the top-K (GPT-OSS: 4) restarts at entry 0
    // on every op that uses all of it.  Never rely on the ring for cross-op safety.
    //
    // CROSS-OP SAFETY comes from ordering, which the CALLER must keep true:
    //   (1) the earlier op's scatter was flushed -- its H2D enqueued on the in-order compute
    //       queue -- BEFORE this op's activation D2H was enqueued (the hybrid MUL_MAT_ID flushes
    //       any pending scatter at op entry, consumed or not, ahead of that D2H); and
    //   (2) the caller does not write the region (zero it, or let the CPU kernels fill it)
    //       until that activation D2H has completed.  Completing an event on an in-order queue
    //       completes every earlier command on it, the H2D included.
    // (1) covers a scatter left pending by an op that nothing consumed (up's, with gate next)
    // (llama.cpp-3bww).  The same flush also keeps the shared activation staging buffer safe: the
    // pending CPU workers read it directly and the next op's D2H rewrites it.  A caller that kept
    // a scatter pending ACROSS a following reserve() would be outside this argument and would have
    // to retain its slice and its staging until the scatter event instead (llama.cpp-8k68).
    //
    // Threading: one MUL_MAT_ID at a time, joined before the next.  It is not main-thread-only:
    // the cpu_async_safe path calls it from the async CPU thread, which the main thread joins
    // before the next op touches the pool.
    size_t reserve(size_t n_experts);

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
    size_t     next_entry_  = 0;  // ring cursor for reserve()
    int        device_id_   = -1;
    mem_handle act_handle_;
    mem_handle out_handle_;
};

}  // namespace ggml_sycl
