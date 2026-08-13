//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_LAYER_STREAMING_HPP
#define GGML_SYCL_LAYER_STREAMING_HPP

#include "ggml.h"
#include "layer-stream-owner.hpp"
#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Double-buffered layer streaming manager.
// Manages two device buffers that alternate holding transformer layer weights.
// While GPU computes on buffer A (layer N), DMA loads layer N+1 into buffer B.
//
// Usage:
//   1. build_layer_map(inventory, count) — once after tensor inventory is set
//   2. allocate_buffers(queue) — once to create device buffers
//   3. ensure_layer(layer_id) — before each layer's first kernel
//   4. get_weight_device_ptr(name) — to get device pointer for a specific weight
//   5. prefetch_next_layer(layer_id+1) — after layer N's first kernel launches
//   6. shutdown() — cleanup
//
// Lifetime (llama.cpp-vbeb): the manager is a per-device singleton, so the
// state above belongs to whichever model built it. That owner is tracked by
// layer_stream_owner_gate and the two entry points a model load can reach —
// build_layer_map() and register_host_ptr() — consult it first. A different
// model arriving releases the previous owner's whole working set through
// release_model_state(), which is the single place any of it is cleared. See
// layer-stream-owner.hpp for why displacement, not preservation, is correct
// here.

class layer_stream_manager {
  public:
    layer_stream_manager() = default;
    ~layer_stream_manager();

    // Non-copyable, non-movable (owns device memory)
    layer_stream_manager(const layer_stream_manager &)             = delete;
    layer_stream_manager & operator=(const layer_stream_manager &) = delete;

    // Build the layer weight map from tensor inventory.
    // inventory: array of (name, size) pairs
    // count: number of entries
    // Filters to only include "blk.N.*" tensors (layer tensors).
    void build_layer_map(const std::pair<std::string, size_t> * inventory, size_t count);

    // Allocate two device buffers sized to max_layer_size.
    // Returns false if allocation fails.
    // queue: the SYCL queue for device allocation.
    bool allocate_buffers(sycl::queue & queue);

    // Free device buffers and reset state.
    void shutdown();

    // Synchronously ensure a layer's weights are loaded into a device buffer.
    // If the layer is already loaded (from a previous prefetch or ensure call), this is a no-op.
    // If not loaded, performs synchronous DMA from host pointers to the buffer.
    // Returns true if the layer is now loaded.
    bool ensure_layer(int layer_id, sycl::queue & queue);

    // Start async DMA of a layer into the alternate buffer.
    // Non-blocking — the DMA runs on copy_queue (if provided) or main queue.
    // Call await_prefetch() before accessing the layer's data.
    void prefetch_next_layer(int layer_id, sycl::queue & queue);

    // Block until the most recent async prefetch completes.
    void await_prefetch();

    // Get the device pointer for a specific weight tensor.
    // Returns nullptr if the weight's layer is not currently loaded.
    // The pointer is valid until the layer is evicted (overwritten by a new layer load).
    void * get_weight_device_ptr(const char * tensor_name) const;

    // Check if a layer is currently loaded in a buffer.
    bool is_layer_loaded(int layer_id) const;

    // Get which buffer index (0 or 1) a layer is in. Returns -1 if not loaded.
    int buffer_for_layer(int layer_id) const;

    // Check if the manager is active (has buffers allocated).
    bool is_active() const { return buffers_[0] != nullptr; }

    // Total bytes allocated for the two buffers.
    size_t allocated_bytes() const { return buffer_size_ * 2; }

    // Max layer size across all layers.
    size_t max_layer_size() const { return max_layer_size_; }

    // Number of layers in the model.
    int n_layers() const { return static_cast<int>(layers_.size()); }

    // Register the host pointer for a tensor (called during model load/cache setup).
    // This is needed because the tensor inventory only stores names and sizes,
    // not the actual host pointers. The pointers are registered as tensors are
    // encountered during inference.
    void register_host_ptr(const char * tensor_name, const void * host_ptr, size_t size);

    // The model that currently owns this manager's working set, or a zeroed
    // token when nobody does.
    layer_stream_owner owner() const { return owner_gate_.current(); }

    // Release the working set if and only if `owner` still owns it, as one
    // atomic step. Returns true when it released.
    //
    // This exists because the teardown path (llama.cpp-y36c) otherwise has to
    // ask owner() and then call shutdown(), and those are two operations: a
    // model load can land between them, take ownership, and have its brand new
    // working set torn down by the previous model's destructor. The window is
    // narrow and the lifecycle registry APPEARS to serialise load against
    // teardown transactions -- but that serialisation is not something this
    // class can see or assert, so it is not something it should depend on.
    // Checking and releasing under one lock removes the question instead of
    // answering it.
    //
    // A model that is NOT the current owner releases nothing: its state was
    // already displaced by whoever owns it now, and tearing down A must never
    // touch B's working set.
    bool release_if_owner(const layer_stream_owner & owner);

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // Private host-only installer for the direct-source carrier fixture.
    void test_install_loaded_buffers(int    device,
                                     size_t buffer_size,
                                     void * buffer0,
                                     int    loaded_layer0,
                                     void * buffer1,
                                     int    loaded_layer1);
#endif

  private:
    // Per-weight entry within a layer
    struct weight_entry {
        std::string  name;
        size_t       size            = 0;
        size_t       offset_in_layer = 0;        // Byte offset within layer buffer
        const void * host_ptr        = nullptr;  // Host-resident data pointer (registered later)
    };

    // Per-layer info
    struct layer_info {
        std::vector<weight_entry> weights;
        size_t                    total_size = 0;
    };

    // Layer data
    std::vector<layer_info>                                 layers_;            // Indexed by layer_id
    size_t                                                  max_layer_size_ = 0;
    std::unordered_map<std::string, std::pair<int, size_t>> name_to_location_;  // name -> (layer_id, weight_idx)

    // Double buffers
    void *     buffers_[2]        = { nullptr, nullptr };  // Cached raw ABI views; handles own lifetime.
    mem_handle buffer_handles_[2] = {};
    size_t     buffer_size_       = 0;
    int        loaded_layers_[2]  = { -1, -1 };
    int        device_id_         = -1;

    // Async prefetch state
    int                prefetch_target_layer_ = -1;
    int                prefetch_buffer_       = -1;
    sycl::event        prefetch_event_;
    bool               prefetch_pending_ = false;
    mutable std::mutex prefetch_mutex_;

    // Host pointer registration
    mutable std::mutex host_ptr_mutex_;

    // Which model the whole cluster above belongs to. owner_transition_mutex_
    // makes "decide the owner" and "release the outgoing owner's state" one
    // step: the gate's own mutex protects only the gate's fields, so without
    // this a load and a teardown can each pass their own check and then both
    // act. Lock order is owner_transition_mutex_ -> prefetch_mutex_ ->
    // host_ptr_mutex_; nothing takes them in the other direction.
    layer_stream_owner_gate owner_gate_;
    std::mutex              owner_transition_mutex_;

    // Internal helpers
    int  pick_buffer_for_layer(int layer_id) const;
    bool load_layer_sync(int layer_id, int buffer_idx, sycl::queue & queue);

    // Return every member above to its pristine value, draining any prefetch
    // that is still writing into the buffers first. This is the ONLY place
    // model-scoped state is released; build_layer_map(), allocate_buffers() and
    // shutdown() all go through it rather than clearing fields themselves.
    void release_model_state();

    // Read the model whose load transaction is active and apply the gate's
    // verdict, releasing the outgoing owner's state on DISPLACE. Called at the
    // top of both entry points a model load can reach.
    void adopt_current_owner();

    // Drop any pending prefetch and forget which layers a buffer holds, without
    // releasing the buffers themselves. Used where the buffers survive but
    // their contents stop being meaningful.
    void drain_and_invalidate_buffers();
};

// Global accessor — returns the singleton layer stream manager for a device.
// Creates on first access. Thread-safe.
layer_stream_manager & get_layer_stream_manager(int device_id);

// Free function API for use from ggml-sycl.cpp
bool   layer_streaming_active(int device_id);
void * layer_streaming_get_weight_ptr(int device_id, const char * name);
void   layer_streaming_ensure_layer(int device_id, int layer_id, sycl::queue & queue);
void   layer_streaming_prefetch_next(int device_id, int layer_id, sycl::queue & queue);
void   layer_streaming_await_prefetch(int device_id);
void   layer_streaming_register_host_ptr(int device_id, const char * name, const void * ptr, size_t size);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYER_STREAMING_HPP
