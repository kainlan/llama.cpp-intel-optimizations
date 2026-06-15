//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "unified-cache-key.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace ggml_sycl {

// Prefetch status for a tensor
enum class PrefetchStatus { NOT_STARTED, IN_PROGRESS, COMPLETED, CANCELLED };

// Prefetch request
struct PrefetchRequest {
    const void *       tensor_data;
    ggml_sycl_cache_id cache_id{};
    size_t             size_bytes;
    int                priority;     // Higher = more urgent
    int                layer_index;  // Which layer this belongs to
    PrefetchStatus     status = PrefetchStatus::NOT_STARTED;
};

// PrefetchScheduler: Manages predictive prefetching of weight tensors during inference.
//
// This scheduler analyzes the computation graph to identify weight tensors and their
// layer associations, then triggers prefetch operations for upcoming layers during
// kernel execution. The goal is to overlap data transfer with computation to hide
// memory latency.
//
// Usage pattern:
//   1. Call analyze_graph() at the start of inference with weight tensor information
//   2. Call on_kernel_start(layer) when each layer begins execution
//   3. Use returned PrefetchRequest list to issue actual prefetch commands
//   4. Call mark_completed() when prefetch finishes
//   5. Call reset() between inference runs
//
// The scheduler uses a lookahead window to prefetch weights 2-3 layers ahead
// of the current computation, balancing between memory bandwidth utilization
// and memory pressure.
class PrefetchScheduler {
  public:
    // Configuration
    static constexpr int    DEFAULT_LOOKAHEAD = 3;     // Prefetch 3 layers ahead
    static constexpr size_t MIN_PREFETCH_SIZE = 1024;  // Don't bother with tiny tensors

    PrefetchScheduler() = default;

    // Set lookahead distance (number of layers to prefetch ahead)
    void set_lookahead(int layers) { lookahead_ = layers; }

    int get_lookahead() const { return lookahead_; }

    // Analyze graph and build prefetch schedule
    // Returns number of prefetchable tensors found
    int analyze_graph(const std::vector<void *> & weight_tensors,
                      const std::vector<size_t> & tensor_sizes,
                      const std::vector<int> &    layer_indices) {
        return analyze_graph(weight_tensors, tensor_sizes, layer_indices, nullptr);
    }

    int analyze_graph(const std::vector<void *> &             weight_tensors,
                      const std::vector<size_t> &             tensor_sizes,
                      const std::vector<int> &                layer_indices,
                      const std::vector<ggml_sycl_cache_id> * cache_ids) {
        prefetch_queue_.clear();
        active_prefetches_.clear();

        for (size_t i = 0; i < weight_tensors.size(); i++) {
            if (tensor_sizes[i] >= MIN_PREFETCH_SIZE) {
                PrefetchRequest req;
                req.tensor_data = weight_tensors[i];
                if (cache_ids && i < cache_ids->size()) {
                    req.cache_id = (*cache_ids)[i];
                }
                req.size_bytes  = tensor_sizes[i];
                req.layer_index = layer_indices[i];
                req.priority    = layer_indices[i];  // Earlier layers = lower priority initially
                prefetch_queue_.push_back(req);
            }
        }

        return static_cast<int>(prefetch_queue_.size());
    }

    // Called when kernel starts - trigger prefetch for upcoming layers
    // Returns list of tensors that should be prefetched now
    std::vector<PrefetchRequest> on_kernel_start(int current_layer) {
        std::vector<PrefetchRequest> to_prefetch;

        for (size_t i = 0; i < prefetch_queue_.size(); ++i) {
            auto & req      = prefetch_queue_[i];
            // Prefetch layers within lookahead window
            int    distance = req.layer_index - current_layer;
            if (distance > 0 && distance <= lookahead_) {
                if (req.status == PrefetchStatus::NOT_STARTED) {
                    req.status   = PrefetchStatus::IN_PROGRESS;
                    req.priority = lookahead_ - distance + 1;  // Closer = higher priority
                    to_prefetch.push_back(req);
                    active_prefetches_.insert(i);
                }
            }
        }

        return to_prefetch;
    }

    // Mark prefetch as completed
    void mark_completed(const void * tensor_data) {
        for (size_t i = 0; i < prefetch_queue_.size(); ++i) {
            auto & req = prefetch_queue_[i];
            if (req.tensor_data == tensor_data) {
                req.status = PrefetchStatus::COMPLETED;
                active_prefetches_.erase(i);
                prefetch_hits_++;
                break;
            }
        }
    }

    void mark_completed(const ggml_sycl_cache_id & cache_id) {
        for (size_t i = 0; i < prefetch_queue_.size(); ++i) {
            auto & req = prefetch_queue_[i];
            if (req.cache_id.valid && ggml_sycl::detail::cache_id_equal(req.cache_id, cache_id)) {
                req.status = PrefetchStatus::COMPLETED;
                active_prefetches_.erase(i);
                prefetch_hits_++;
                break;
            }
        }
    }

    // Cancel all pending prefetches (e.g., under memory pressure)
    void cancel_all_pending() {
        for (auto & req : prefetch_queue_) {
            if (req.status == PrefetchStatus::IN_PROGRESS) {
                req.status = PrefetchStatus::CANCELLED;
            }
        }
        active_prefetches_.clear();
        prefetch_cancels_++;
    }

    // Check if tensor is being prefetched
    bool is_prefetching(const void * tensor_data) const {
        for (size_t i = 0; i < prefetch_queue_.size(); ++i) {
            if (prefetch_queue_[i].tensor_data == tensor_data) {
                return active_prefetches_.count(i) > 0;
            }
        }
        return false;
    }

    bool is_prefetching(const ggml_sycl_cache_id & cache_id) const {
        for (size_t i = 0; i < prefetch_queue_.size(); ++i) {
            const auto & req = prefetch_queue_[i];
            if (req.cache_id.valid && ggml_sycl::detail::cache_id_equal(req.cache_id, cache_id)) {
                return active_prefetches_.count(i) > 0;
            }
        }
        return false;
    }

    // Check if tensor needs prefetch (not started and not completed)
    bool needs_prefetch(const void * tensor_data, int layer_index, int current_layer) const {
        int distance = layer_index - current_layer;
        if (distance <= 0 || distance > lookahead_) {
            return false;
        }

        for (const auto & req : prefetch_queue_) {
            if (req.tensor_data == tensor_data) {
                return req.status == PrefetchStatus::NOT_STARTED;
            }
        }
        return true;  // Not in queue, could benefit from prefetch
    }

    bool needs_prefetch(const ggml_sycl_cache_id & cache_id, int layer_index, int current_layer) const {
        int distance = layer_index - current_layer;
        if (distance <= 0 || distance > lookahead_) {
            return false;
        }

        for (const auto & req : prefetch_queue_) {
            if (req.cache_id.valid && ggml_sycl::detail::cache_id_equal(req.cache_id, cache_id)) {
                return req.status == PrefetchStatus::NOT_STARTED;
            }
        }
        return true;
    }

    // Get statistics
    int get_prefetch_hits() const { return prefetch_hits_; }

    int get_prefetch_cancels() const { return prefetch_cancels_; }

    int get_active_count() const { return static_cast<int>(active_prefetches_.size()); }

    float get_hit_rate() const {
        int total = prefetch_hits_ + prefetch_cancels_;
        return total > 0 ? static_cast<float>(prefetch_hits_) / total : 0.0f;
    }

    // Reset statistics
    void reset_stats() {
        prefetch_hits_    = 0;
        prefetch_cancels_ = 0;
    }

    // Reset scheduler for new inference (keeps graph analysis)
    void reset() {
        for (auto & req : prefetch_queue_) {
            req.status = PrefetchStatus::NOT_STARTED;
        }
        active_prefetches_.clear();
    }

    // Get queue size for testing
    size_t get_queue_size() const { return prefetch_queue_.size(); }

  private:
    int                          lookahead_ = DEFAULT_LOOKAHEAD;
    std::vector<PrefetchRequest> prefetch_queue_;
    std::unordered_set<size_t>   active_prefetches_;
    int                          prefetch_hits_    = 0;
    int                          prefetch_cancels_ = 0;
};

}  // namespace ggml_sycl
