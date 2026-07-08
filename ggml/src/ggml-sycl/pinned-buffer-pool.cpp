//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

// PinnedBufferPool implementation.
// Extracted from expert-cache.cpp during ExpertCache removal (Task 7).
// Provides ring-buffered host-pinned staging buffers for CPU expert dispatch.

#include "pinned-buffer-pool.hpp"

#include "common.hpp"
#include "unified-cache.hpp"

#include <cassert>

namespace ggml_sycl {

PinnedBufferPool::~PinnedBufferPool() {
    // Skip cleanup during static destruction — unified cache statics
    // (g_runtime_alloc_registry etc.) may already be destroyed.
    if (!ggml_sycl_is_shutting_down()) {
        shutdown();
    }
}

void PinnedBufferPool::init(sycl::queue & q, int device_id, size_t max_experts, size_t act_dim, size_t out_dim) {
    if (is_initialized()) {
        return;
    }

    device_id_   = device_id;
    act_stride_  = act_dim;
    out_stride_  = out_dim;
    max_experts_ = max_experts;

    const size_t act_bytes = max_experts * act_dim * sizeof(float);
    const size_t out_bytes = max_experts * out_dim * sizeof(float);

    // Allocate activation pool via unified_alloc with pinned host constraint
    alloc_request req_act;
    req_act.queue                               = &q;
    req_act.device                              = device_id;
    req_act.size                                = act_bytes;
    req_act.intent.role                         = alloc_role::EXPERT_STAGING;
    req_act.intent.category                     = runtime_category::EXPERT_CACHE;
    req_act.intent.cohort_id                    = "moe_act_pool";
    req_act.intent.constraints.must_host_pinned = true;

    act_handle_ = unified_allocate(req_act);
    if (!act_handle_.valid()) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate activation pool (%zu bytes)\n", act_bytes);
        return;
    }
    auto act_resolved = act_handle_.resolve(device_id);
    if (!act_resolved.ptr) {
        GGML_LOG_WARN("[MOE-POOL] Activation pool allocation did not resolve on device %d (%zu bytes)\n", device_id,
                      act_bytes);
        act_handle_ = {};
        return;
    }
    act_pool_ = static_cast<float *>(act_resolved.ptr);

    // Allocate output pool
    alloc_request req_out    = req_act;
    req_out.size             = out_bytes;
    req_out.intent.cohort_id = "moe_out_pool";

    out_handle_ = unified_allocate(req_out);
    if (!out_handle_.valid()) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate output pool (%zu bytes)\n", out_bytes);
        act_handle_ = {};
        act_pool_   = nullptr;
        return;
    }
    auto out_resolved = out_handle_.resolve(device_id);
    if (!out_resolved.ptr) {
        GGML_LOG_WARN("[MOE-POOL] Output pool allocation did not resolve on device %d (%zu bytes)\n", device_id,
                      out_bytes);
        out_handle_ = {};
        act_handle_ = {};
        act_pool_   = nullptr;
        return;
    }
    out_pool_ = static_cast<float *>(out_resolved.ptr);

    GGML_LOG_INFO("[MOE-POOL] Pinned buffer pool: act=%zu KB, out=%zu KB, max_experts=%zu\n", act_bytes / 1024,
                  out_bytes / 1024, max_experts);
}

void PinnedBufferPool::shutdown() {
    act_handle_ = {};
    out_handle_ = {};
    act_pool_   = nullptr;
    out_pool_   = nullptr;
}

PinnedBufferPool::BufferPair PinnedBufferPool::acquire(size_t n_experts) {
    GGML_ASSERT(n_experts <= max_experts_ && "Expert count exceeds pool capacity");
    GGML_ASSERT(act_pool_ && out_pool_ && "Pool not initialized");
    return { act_pool_, out_pool_ };
}

void PinnedBufferPool::release(BufferPair) {
    // No-op: CPU vec_dot kernels write every output element that the scatter
    // loop reads back (n_cpu * N floats), so zeroing is unnecessary.
    // Stale data in unused pool slots is never accessed.
}

mem_handle PinnedBufferPool::act_handle() const {
    return act_handle_;
}

mem_handle PinnedBufferPool::out_handle() const {
    return out_handle_;
}

}  // namespace ggml_sycl
