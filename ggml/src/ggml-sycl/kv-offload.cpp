//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "kv-offload.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "mem-ops.hpp"

#include <algorithm>
#include <cstring>

namespace ggml_sycl {

static mem_handle kv_tensor_handle_for_device(const ggml_tensor * tensor, int device, void * ptr) {
    if (!ptr || device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return mem_handle{};
    }

    if (tensor && tensor->extra) {
        const auto * extra  = static_cast<const ggml_tensor_extra_gpu *>(tensor->extra);
        const auto & handle = extra->data_handle[device];
        if (handle.valid()) {
            auto resolved = handle.resolve(device);
            if (resolved && resolved.ptr == ptr) {
                return handle;
            }
        }
    }

    return mem_handle::from_chunk_ptr(ptr, device, GGML_LAYOUT_AOS, true);
}

kv_offload_manager::kv_offload_manager(sycl::queue & queue, const kv_offload_config & config) :
    queue_(queue),
    config_(config) {
    GGML_LOG_INFO("[KV-OFFLOAD] Initialized: threshold=%d tokens, budget=%.1f MB, block_size=%d tokens\n",
                  config_.offload_threshold, config_.gpu_kv_budget / (1024.0f * 1024.0f), config_.block_size);
}

kv_offload_manager::~kv_offload_manager() {
    // Wait for all pending transfers
    for (auto & [key, block] : blocks_) {
        try {
            block.transfer_event.wait();
        } catch (...) {
        }

        free_cpu_block(block);
    }
}

void kv_offload_manager::register_layer(int layer_id, ggml_tensor * k_cache, ggml_tensor * v_cache) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!k_cache || !v_cache) {
        GGML_LOG_WARN("[KV-OFFLOAD] Cannot register layer %d: null KV tensors\n", layer_id);
        return;
    }

    kv_layer_info info;
    info.layer_id = layer_id;

    // Calculate per-token sizes from tensor dimensions
    // K cache shape is typically [n_ctx, n_kv_heads, head_dim] or similar
    // For simplicity, we compute total bytes / max_tokens
    size_t k_total = ggml_nbytes(k_cache);
    size_t v_total = ggml_nbytes(v_cache);

    // Get max context from tensor shape (usually first dimension)
    int32_t max_tokens = static_cast<int32_t>(k_cache->ne[0]);
    if (max_tokens <= 0) {
        // Try second dimension
        max_tokens = static_cast<int32_t>(k_cache->ne[1]);
    }
    if (max_tokens <= 0) {
        GGML_LOG_WARN("[KV-OFFLOAD] Cannot determine max_tokens for layer %d\n", layer_id);
        max_tokens = 4096;  // Default fallback
    }

    info.k_size_per_token = k_total / max_tokens;
    info.v_size_per_token = v_total / max_tokens;
    const int device      = ggml_sycl_get_device_id_from_queue(queue_);
    info.device           = device;
    info.k_base_gpu       = ggml_sycl_resolve_tensor_ptr(k_cache, device);
    info.v_base_gpu       = ggml_sycl_resolve_tensor_ptr(v_cache, device);
    info.k_handle         = kv_tensor_handle_for_device(k_cache, device, info.k_base_gpu);
    info.v_handle         = kv_tensor_handle_for_device(v_cache, device, info.v_base_gpu);
    info.max_tokens       = max_tokens;

    if (!info.k_handle.valid() || !info.v_handle.valid()) {
        GGML_LOG_WARN("[KV-OFFLOAD] Layer %d registered without valid KV handles on device %d\n", layer_id, device);
    }

    layers_[layer_id] = info;

    GGML_SYCL_DEBUG("[KV-OFFLOAD] Registered layer %d: K=%zu B/tok, V=%zu B/tok, max=%d tokens\n", layer_id,
                    info.k_size_per_token, info.v_size_per_token, max_tokens);
}

bool kv_offload_manager::is_layer_registered(int layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layers_.find(layer_id) != layers_.end();
}

size_t kv_offload_manager::registered_layer_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layers_.size();
}

sycl::event kv_offload_manager::ensure_on_gpu(int layer_id, int32_t pos) {
    std::lock_guard<std::mutex> lock(mutex_);

    kv_block * block = get_block(layer_id, pos);
    if (!block) {
        // Block doesn't exist yet - KV hasn't been computed for this position
        return sycl::event();
    }

    if (block->on_gpu) {
        // Already on GPU - update access time and return
        prefetch_hits_++;
        update_access(block);
        return sycl::event();
    }

    // Need to transfer from CPU to GPU
    prefetch_misses_++;
    return transfer_to_gpu(block);
}

sycl::event kv_offload_manager::ensure_range_on_gpu(int layer_id, int32_t start_pos, int32_t end_pos) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<sycl::event> events;

    int32_t start_block = pos_to_block_idx(start_pos);
    int32_t end_block   = pos_to_block_idx(end_pos - 1);

    for (int32_t block_idx = start_block; block_idx <= end_block; block_idx++) {
        kv_block_key key{ layer_id, block_idx };
        auto         it = blocks_.find(key);
        if (it == blocks_.end()) {
            continue;  // Block doesn't exist
        }

        kv_block & block = it->second;
        if (!block.on_gpu) {
            events.push_back(transfer_to_gpu(&block));
            prefetch_misses_++;
        } else {
            prefetch_hits_++;
            update_access(&block);
        }
    }

    if (events.empty()) {
        return sycl::event();
    }

    // Return a barrier event that waits for all transfers
    // Note: SYCL doesn't have a direct barrier API, so we wait inline
    for (auto & evt : events) {
        evt.wait();
    }
    return sycl::event();
}

void kv_offload_manager::prefetch_blocks(int layer_id, int32_t start_pos, int32_t end_pos) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Expand range by prefetch ratio
    int32_t range          = end_pos - start_pos;
    int32_t prefetch_extra = static_cast<int32_t>(range * (config_.prefetch_ratio - 1.0f));
    end_pos                = std::min(end_pos + prefetch_extra, context_length_);

    int32_t start_block = pos_to_block_idx(start_pos);
    int32_t end_block   = pos_to_block_idx(end_pos - 1);

    for (int32_t block_idx = start_block; block_idx <= end_block; block_idx++) {
        kv_block_key key{ layer_id, block_idx };
        auto         it = blocks_.find(key);
        if (it == blocks_.end()) {
            continue;
        }

        kv_block & block = it->second;
        if (!block.on_gpu && block.cpu_ptr) {
            // Schedule async transfer - don't wait
            transfer_to_gpu(&block);
        }
        update_access(&block);
    }
}

void kv_offload_manager::prefetch_all_layers(int32_t start_pos, int32_t end_pos) {
    for (const auto & [layer_id, info] : layers_) {
        prefetch_blocks(layer_id, start_pos, end_pos);
    }
}

size_t kv_offload_manager::offload_oldest(size_t bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect all GPU-resident blocks with their access times
    std::vector<std::pair<int64_t, kv_block_key>> gpu_blocks;
    for (auto & [key, block] : blocks_) {
        if (block.on_gpu) {
            gpu_blocks.push_back({ block.last_access, key });
        }
    }

    // Sort by access time (oldest first) - only compare access time, not the key
    std::sort(gpu_blocks.begin(), gpu_blocks.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    size_t freed = 0;
    for (const auto & [access_time, key] : gpu_blocks) {
        if (freed >= bytes_needed) {
            break;
        }

        kv_block &  block = blocks_[key];
        sycl::event evt   = transfer_to_cpu(&block);
        evt.wait();  // Must wait to ensure memory is freed

        freed += block.size;

        GGML_SYCL_DEBUG("[KV-OFFLOAD] Offloaded block L%d:B%d (%.2f MB) to CPU\n", key.layer_id, key.block_idx,
                        block.size / (1024.0f * 1024.0f));
    }

    return freed;
}

void kv_offload_manager::offload_beyond(int32_t pos) {
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t beyond_block = pos_to_block_idx(pos);

    for (auto & [key, block] : blocks_) {
        if (key.block_idx > beyond_block && block.on_gpu) {
            sycl::event evt = transfer_to_cpu(&block);
            evt.wait();
        }
    }
}

void kv_offload_manager::on_context_extended(int32_t new_length) {
    std::lock_guard<std::mutex> lock(mutex_);

    context_length_ = new_length;

    // Check if we need to start offloading
    if (new_length > config_.offload_threshold) {
        // Calculate how much memory we're over budget
        size_t current_gpu = gpu_memory_used_;
        if (current_gpu > config_.gpu_kv_budget) {
            size_t need_to_free = current_gpu - config_.gpu_kv_budget;
            // Release lock temporarily for offload_oldest
            mutex_.unlock();
            offload_oldest(need_to_free);
            mutex_.lock();
        }
    }
}

bool kv_offload_manager::is_offloading_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return context_length_ > config_.offload_threshold;
}

size_t kv_offload_manager::gpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_memory_used_;
}

size_t kv_offload_manager::cpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_memory_used_;
}

size_t kv_offload_manager::blocks_on_gpu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto & [key, block] : blocks_) {
        if (block.on_gpu) {
            count++;
        }
    }
    return count;
}

size_t kv_offload_manager::blocks_on_cpu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto & [key, block] : blocks_) {
        if (!block.on_gpu && block.cpu_ptr) {
            count++;
        }
    }
    return count;
}

void kv_offload_manager::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = prefetch_hits_ + prefetch_misses_;
    float  rate  = total > 0 ? 100.0f * prefetch_hits_ / total : 0.0f;

    size_t gpu_blocks = 0, cpu_blocks = 0;
    for (const auto & [key, block] : blocks_) {
        if (block.on_gpu) {
            gpu_blocks++;
        } else if (block.cpu_ptr) {
            cpu_blocks++;
        }
    }

    GGML_LOG_INFO("[KV-OFFLOAD] Stats: %zu prefetch hits, %zu misses (%.1f%% hit rate)\n", prefetch_hits_,
                  prefetch_misses_, rate);
    GGML_LOG_INFO("[KV-OFFLOAD] Memory: %.1f MB GPU (%zu blocks), %.1f MB CPU (%zu blocks)\n",
                  gpu_memory_used_ / (1024.0f * 1024.0f), gpu_blocks, cpu_memory_used_ / (1024.0f * 1024.0f),
                  cpu_blocks);
}

void kv_offload_manager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetch_hits_   = 0;
    prefetch_misses_ = 0;
}

kv_block * kv_offload_manager::get_block(int layer_id, int32_t pos) {
    int32_t      block_idx = pos_to_block_idx(pos);
    kv_block_key key{ layer_id, block_idx };

    auto it = blocks_.find(key);
    if (it != blocks_.end()) {
        return &it->second;
    }

    // Block doesn't exist - create it
    auto layer_it = layers_.find(layer_id);
    if (layer_it == layers_.end()) {
        return nullptr;  // Layer not registered
    }

    const kv_layer_info & layer = layer_it->second;

    // Calculate block boundaries
    int32_t start_pos = block_idx * config_.block_size;
    int32_t end_pos   = std::min(start_pos + config_.block_size, layer.max_tokens);

    // Create new block
    kv_block block;
    block.layer_id       = layer_id;
    block.start_pos      = start_pos;
    block.end_pos        = end_pos;
    block.on_gpu         = true;     // Initially on GPU (KV is computed there)
    block.gpu_ptr        = nullptr;  // Points into KV cache tensor
    block.cpu_ptr        = nullptr;  // Allocated on first offload
    block.size           = (layer.k_size_per_token + layer.v_size_per_token) * (end_pos - start_pos);
    block.last_access    = current_time_++;
    block.transfer_event = sycl::event();

    // Calculate GPU pointer offset into KV cache
    // Note: actual pointer arithmetic depends on KV cache layout
    // This is simplified - real implementation needs to account for tensor strides

    blocks_[key] = block;
    gpu_memory_used_ += block.size;

    return &blocks_[key];
}

int32_t kv_offload_manager::pos_to_block_idx(int32_t pos) const {
    return pos / config_.block_size;
}

sycl::event kv_offload_manager::transfer_to_gpu(kv_block * block) {
    if (!block || block->on_gpu || !block->cpu_ptr) {
        return sycl::event();
    }

    auto layer_it = layers_.find(block->layer_id);
    if (layer_it == layers_.end()) {
        return sycl::event();
    }

    const kv_layer_info & layer = layer_it->second;

    // Wait for any previous transfer to complete
    try {
        block->transfer_event.wait();
    } catch (...) {
    }

    // Calculate destination offset in KV cache
    size_t k_offset = block->start_pos * layer.k_size_per_token;
    size_t v_offset = block->start_pos * layer.v_size_per_token;
    size_t k_size   = (block->end_pos - block->start_pos) * layer.k_size_per_token;
    size_t v_size   = (block->end_pos - block->start_pos) * layer.v_size_per_token;

    if (!layer.k_handle.valid() || !layer.v_handle.valid() || !block->cpu_handle.valid()) {
        GGML_LOG_ERROR("[KV-OFFLOAD] Cannot transfer layer %d block [%d,%d) to GPU: invalid KV/CPU handle\n",
                       block->layer_id, block->start_pos, block->end_pos);
        return sycl::event();
    }

    // Transfer K and V
    sycl::event k_evt = mem_copy_async(layer.k_handle, k_offset, block->cpu_handle, 0, k_size, queue_);
    sycl::event v_evt = mem_copy_async(layer.v_handle, v_offset, block->cpu_handle, k_size, v_size, queue_, { k_evt });

    block->transfer_event = v_evt;
    block->on_gpu         = true;

    gpu_memory_used_ += block->size;
    update_access(block);

    return v_evt;
}

sycl::event kv_offload_manager::transfer_to_cpu(kv_block * block) {
    if (!block || !block->on_gpu) {
        return sycl::event();
    }

    auto layer_it = layers_.find(block->layer_id);
    if (layer_it == layers_.end()) {
        return sycl::event();
    }

    const kv_layer_info & layer = layer_it->second;

    // Allocate CPU buffer if needed
    if (!block->cpu_ptr) {
        if (!allocate_cpu_block(*block)) {
            GGML_LOG_ERROR("[KV-OFFLOAD] Failed to allocate CPU buffer for block\n");
            return sycl::event();
        }
    }

    // Wait for any previous transfer to complete
    try {
        block->transfer_event.wait();
    } catch (...) {
    }

    // Calculate source offset in KV cache
    size_t k_offset = block->start_pos * layer.k_size_per_token;
    size_t v_offset = block->start_pos * layer.v_size_per_token;
    size_t k_size   = (block->end_pos - block->start_pos) * layer.k_size_per_token;
    size_t v_size   = (block->end_pos - block->start_pos) * layer.v_size_per_token;

    if (!layer.k_handle.valid() || !layer.v_handle.valid() || !block->cpu_handle.valid()) {
        GGML_LOG_ERROR("[KV-OFFLOAD] Cannot transfer layer %d block [%d,%d) to CPU: invalid KV/CPU handle\n",
                       block->layer_id, block->start_pos, block->end_pos);
        return sycl::event();
    }

    // Transfer K and V
    sycl::event k_evt = mem_copy_async(block->cpu_handle, 0, layer.k_handle, k_offset, k_size, queue_);
    sycl::event v_evt = mem_copy_async(block->cpu_handle, k_size, layer.v_handle, v_offset, v_size, queue_, { k_evt });

    block->transfer_event = v_evt;
    block->on_gpu         = false;

    gpu_memory_used_ -= block->size;

    return v_evt;
}

bool kv_offload_manager::allocate_cpu_block(kv_block & block) {
    if (block.cpu_ptr) {
        return true;
    }

    const int device = ggml_sycl_get_device_id_from_queue(queue_);

    alloc_request req{};
    req.queue                                    = &queue_;
    req.device                                   = device;
    req.size                                     = block.size;
    req.suppress_failure_log                     = true;
    req.intent.role                              = alloc_role::KV;
    req.intent.category                          = runtime_category::KV_CACHE;
    req.intent.cohort_id                         = "kv_offload:cpu_block";
    req.intent.constraints.must_host_pinned      = true;
    req.intent.constraints.use_pinned_pool       = true;
    req.intent.constraints.require_host_usm_base = true;

    alloc_handle cpu_block_owner{};
    if (!unified_alloc(req, &cpu_block_owner) || !cpu_block_owner.ptr) {
        return false;
    }

    mem_handle owner    = mem_handle::from_owned_alloc(std::move(cpu_block_owner), GGML_LAYOUT_AOS);
    auto       resolved = owner.resolve(device);
    if (!resolved.ptr || resolved.on_device) {
        return false;
    }

    block.cpu_handle = std::move(owner);
    block.cpu_ptr    = resolved.ptr;
    cpu_memory_used_ += block.size;
    return true;
}

void kv_offload_manager::free_cpu_block(kv_block & block) {
    if (block.cpu_ptr || block.cpu_handle.valid()) {
        if (cpu_memory_used_ >= block.size) {
            cpu_memory_used_ -= block.size;
        } else {
            cpu_memory_used_ = 0;
        }
        block.cpu_ptr    = nullptr;
        block.cpu_handle = mem_handle{};
    }
}

void kv_offload_manager::update_access(kv_block * block) {
    block->last_access = current_time_++;
}

}  // namespace ggml_sycl
