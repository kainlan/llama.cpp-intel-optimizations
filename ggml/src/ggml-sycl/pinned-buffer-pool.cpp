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

    // Allocate activation pool via unified_alloc with pinned host constraint.
    //
    // role=COMPUTE, not EXPERT_STAGING (llama.cpp-cg8j): this pool is a
    // lazy-once singleton allocated on the first MUL_MAT_ID graph and held
    // for the process lifetime (see shutdown() -- released only at module
    // shutdown), not a per-graph ephemeral staging buffer. alloc_role::
    // EXPERT_STAGING is force-routed to the host SCRATCH zone by
    // select_zone() specifically so that genuinely short-lived EXPERT_STAGING
    // leaks stay visible to host_zone_reset()'s per-graph live scan (that
    // routing exists on purpose -- see 9a0670712 / llama.cpp-0igs /
    // llama.cpp-7f2e, which fixed the opposite bug of persistent allocations
    // being silently hidden in the unswept WEIGHT zone). A handle that is
    // deliberately retained forever is not what that scan is meant to catch:
    // every visit finds it live, host_zone_reset(SCRATCH) refuses forever,
    // and the Phase-0 audit logs it as a permanent "NEW-ESCAPE" even though
    // there is no leak (single alloc_id, freed once at shutdown via the
    // WEIGHT zone's own per-allocation host_zone_free() TLSF reclaim).
    // role=COMPUTE + category=EXPERT_CACHE mirrors the existing
    // g_retained_scratch precedent in cpu-dispatch.cpp (role=COMPUTE,
    // category=HOST_COMPUTE): both hit select_zone()'s category-based WEIGHT
    // branch, which is never swept by host_zone_reset(), while leaving the
    // EXPERT_STAGING-role-first SCRATCH routing that 0igs/7f2e protect
    // untouched for every other (genuinely ephemeral) EXPERT_STAGING caller.
    alloc_request req_act;
    req_act.queue                               = &q;
    req_act.device                              = device_id;
    req_act.size                                = act_bytes;
    req_act.intent.role                         = alloc_role::COMPUTE;
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
