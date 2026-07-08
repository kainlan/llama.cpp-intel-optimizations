//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "dense-scheduler.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "mem-ops.hpp"

namespace ggml_sycl {

static void * dense_scheduler_alloc_slot(size_t size, sycl::queue & queue, int device_id, mem_handle & owner) {
    owner = {};
    if (size == 0) {
        return nullptr;
    }

    alloc_request req{};
    req.queue                               = &queue;
    req.device                              = device_id;
    req.size                                = size;
    req.intent.role                         = alloc_role::WEIGHT;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.cohort_id                    = "dense_scheduler_slot";
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;
    req.suppress_failure_log                = true;

    owner = unified_allocate(req);
    if (!owner.valid()) {
        return nullptr;
    }

    auto resolved = owner.resolve(device_id);
    if (!resolved || !resolved.ptr || !resolved.on_device) {
        owner = {};
        return nullptr;
    }

    return resolved.ptr;
}

dense_layer_scheduler::dense_layer_scheduler(sycl::queue & compute_queue, size_t max_layer_size) :
    compute_queue_(compute_queue),
    copy_queue_(compute_queue.get_context(), compute_queue.get_device()),
    device_id_(ggml_sycl_get_device_id_from_queue(compute_queue)),
    slot_size_(max_layer_size) {
    // Allocate two VRAM slots for double buffering
    vram_slot_[0] = dense_scheduler_alloc_slot(slot_size_, compute_queue_, device_id_, vram_slot_handle_[0]);
    vram_slot_[1] = dense_scheduler_alloc_slot(slot_size_, compute_queue_, device_id_, vram_slot_handle_[1]);

    if (!vram_slot_[0] || !vram_slot_[1]) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate dense scheduler VRAM slots (%.1f MB each)\n",
                       slot_size_ / (1024.0 * 1024.0));
        vram_slot_handle_[0] = {};
        vram_slot_handle_[1] = {};
        vram_slot_[0] = vram_slot_[1] = nullptr;
        return;
    }

    GGML_LOG_INFO("[SYCL] Dense layer scheduler created with 2x%.1f MB VRAM slots\n", slot_size_ / (1024.0 * 1024.0));
}

dense_layer_scheduler::~dense_layer_scheduler() {
    // Wait for any pending prefetch
    if (has_pending_prefetch_) {
        try {
            pending_prefetch_.wait();
        } catch (...) {
            // Ignore exceptions during destruction
        }
    }

    // Free VRAM slots
    vram_slot_[0]        = nullptr;
    vram_slot_[1]        = nullptr;
    vram_slot_handle_[0] = {};
    vram_slot_handle_[1] = {};

    GGML_LOG_INFO("[SYCL] Dense scheduler destroyed (prefetch=%zu, sync_load=%zu)\n", prefetch_count_,
                  sync_load_count_);
}

void dense_layer_scheduler::set_host_ptr_callback(host_ptr_callback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    get_host_ptr_ = std::move(cb);
}

void * dense_layer_scheduler::get_dense_layer(int layer_id, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!vram_slot_[current_slot_]) {
        GGML_LOG_ERROR("[SYCL] Dense scheduler not initialized\n");
        return nullptr;
    }

    if (size > slot_size_) {
        GGML_LOG_ERROR("[SYCL] Layer size (%.1f MB) exceeds slot size (%.1f MB)\n", size / (1024.0 * 1024.0),
                       slot_size_ / (1024.0 * 1024.0));
        return nullptr;
    }

    // Check if layer is already in current slot
    if (layer_in_slot_[current_slot_] == layer_id) {
        return vram_slot_[current_slot_];
    }

    // Wait for any pending prefetch to this slot
    if (has_pending_prefetch_) {
        pending_prefetch_.wait();
        has_pending_prefetch_ = false;
    }

    // Check again after waiting (prefetch might have loaded it into the other slot)
    // If the prefetch loaded the layer into the alternate slot, we need to use that
    int prefetch_slot = 1 - current_slot_;
    if (layer_in_slot_[prefetch_slot] == layer_id) {
        // The layer we need is in the prefetch slot - swap to it
        current_slot_ = prefetch_slot;
        return vram_slot_[current_slot_];
    }

    // Check current slot again (might have been loaded by a different path)
    if (layer_in_slot_[current_slot_] == layer_id) {
        return vram_slot_[current_slot_];
    }

    // Need synchronous load
    if (!get_host_ptr_) {
        GGML_LOG_ERROR("[SYCL] No host pointer callback set for dense scheduler\n");
        return nullptr;
    }

    void * host_ptr = get_host_ptr_(layer_id);
    if (!host_ptr) {
        GGML_LOG_ERROR("[SYCL] Failed to get host pointer for layer %d\n", layer_id);
        return nullptr;
    }

    mem_handle dst_handle = vram_slot_handle_[current_slot_];
    mem_handle src_handle = ::ggml_sycl_memcpy_handle_for_raw_ptr(host_ptr, device_id_);
    mem_copy(dst_handle, src_handle, size, compute_queue_);
    layer_in_slot_[current_slot_] = layer_id;
    sync_load_count_++;

    return vram_slot_[current_slot_];
}

void dense_layer_scheduler::prefetch_next(int next_layer_id, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!vram_slot_[0] || !vram_slot_[1]) {
        return;  // Not initialized
    }

    if (size > slot_size_) {
        return;  // Size exceeds slot capacity
    }

    int prefetch_slot = 1 - current_slot_;

    // Already in prefetch slot
    if (layer_in_slot_[prefetch_slot] == next_layer_id) {
        return;
    }

    // Also check if it's in current slot (no need to prefetch)
    if (layer_in_slot_[current_slot_] == next_layer_id) {
        return;
    }

    // Wait for any previous prefetch to complete
    if (has_pending_prefetch_) {
        pending_prefetch_.wait();
        has_pending_prefetch_ = false;
    }

    if (!get_host_ptr_) {
        return;
    }

    void * host_ptr = get_host_ptr_(next_layer_id);
    if (!host_ptr) {
        return;
    }

    // Start async copy to prefetch slot through the canonical mem_handle copy helper.
    mem_handle dst_handle         = vram_slot_handle_[prefetch_slot];
    mem_handle src_handle         = ::ggml_sycl_memcpy_handle_for_raw_ptr(host_ptr, device_id_);
    pending_prefetch_             = mem_copy_async(dst_handle, src_handle, size, copy_queue_);
    has_pending_prefetch_         = true;
    layer_in_slot_[prefetch_slot] = next_layer_id;
    prefetch_count_++;
}

void dense_layer_scheduler::wait_prefetch() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_pending_prefetch_) {
        pending_prefetch_.wait();
        has_pending_prefetch_ = false;
    }
}

void dense_layer_scheduler::advance_slot() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Wait for prefetch before advancing
    if (has_pending_prefetch_) {
        pending_prefetch_.wait();
        has_pending_prefetch_ = false;
    }

    current_slot_ = 1 - current_slot_;
}

}  // namespace ggml_sycl
