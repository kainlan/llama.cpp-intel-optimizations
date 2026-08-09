//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_LAYER_STREAM_OWNER_HPP
#define GGML_SYCL_LAYER_STREAM_OWNER_HPP

// Owner-keyed ownership for the layer-streaming working set (llama.cpp-vbeb,
// child of the k7b0 cross-model survival audit).
//
// layer_stream_manager is a per-device singleton holding a model's layer map,
// its name -> host-pointer registry and two device staging buffers. Nothing in
// its lifecycle ever asked whose model that state belonged to: the map was
// rebuilt only when a model activated streaming, and a model that did NOT
// activate streaming never touched the manager at all. So the first streaming
// model in a process owned the buffers until module shutdown, and every later
// model resolved its own blk.N.* weights -- a name set essentially every dense
// architecture shares -- into the first model's buffers.
//
// This gate gives that working set an owner. The rule is structural: the
// working set belongs to exactly one model token at a time, and the arrival of
// a DIFFERENT token is what releases the previous one's state. There is no
// load-boundary sweep and no "clear everything on any load", because either
// would let one model wipe state another model is using.
//
// Unlike moe_discovery_registry there is no park/restore half. The manager owns
// two device staging buffers, not host discovery, and a displaced model's
// weights still live in host memory and the unified cache -- releasing its
// buffers degrades it to the ordinary non-streamed path, which is correct
// output more slowly. Serving the incoming model from the outgoing model's
// buffers is silent corruption. Only one of those is recoverable.
//
// The gate is host-only and depends on nothing from SYCL, so the ownership
// rules can be proven without a device.

#include <cstdint>
#include <mutex>

namespace ggml_sycl {

// Slot value that proves the absence of an owner. Mirrors
// GGML_SYCL_MODEL_SLOT_NONE (and lifecycle::no_model_slot) without pulling the
// backend header in here.
static constexpr uint32_t layer_stream_owner_slot_none = 0xFFFFFFFFu;

// The identity a layer-streaming working set belongs to. It is the full model
// token: a reused slot carrying a newer generation is a DIFFERENT owner.
struct layer_stream_owner {
    uint64_t model_id        = 0;
    uint64_t load_txn_id     = 0;
    uint32_t slot            = layer_stream_owner_slot_none;
    uint64_t slot_generation = 0;
};

// Zero ids and the none-slot fail closed: a caller with no active load
// transaction is unattributed, so it can neither claim the working set nor
// displace whoever holds it.
inline bool layer_stream_owner_valid(const layer_stream_owner & owner) {
    return owner.model_id != 0 && owner.load_txn_id != 0 && owner.slot != layer_stream_owner_slot_none &&
           owner.slot_generation != 0;
}

inline bool layer_stream_owner_same(const layer_stream_owner & a, const layer_stream_owner & b) {
    return a.model_id == b.model_id && a.load_txn_id == b.load_txn_id && a.slot == b.slot &&
           a.slot_generation == b.slot_generation;
}

enum class layer_stream_transition {
    // The caller cannot name an owner. Nothing was claimed and nothing was
    // released; the working set keeps whatever owner it had.
    NONE,
    // An unowned working set was claimed. There is nothing to release.
    ADOPT,
    // The owner that already holds the working set asked again. Deliberately
    // not a release: a model must not lose the map it just built.
    KEEP,
    // A different owner arrived. The caller must release the whole working set
    // before writing anything for the incoming model.
    DISPLACE,
};

struct layer_stream_transition_result {
    layer_stream_transition kind = layer_stream_transition::NONE;
    // The owner whose state must be released. Meaningful only for DISPLACE.
    layer_stream_owner      displaced{};
};

class layer_stream_owner_gate {
  public:
    layer_stream_owner_gate()                                            = default;
    layer_stream_owner_gate(const layer_stream_owner_gate &)             = delete;
    layer_stream_owner_gate & operator=(const layer_stream_owner_gate &) = delete;

    // Record `incoming` as the owner and report what the caller owes the
    // outgoing one. The owner is switched here rather than after the release so
    // that a second observe() racing the first cannot also decide to displace.
    layer_stream_transition_result observe(const layer_stream_owner & incoming) noexcept {
        layer_stream_transition_result result;
        if (!layer_stream_owner_valid(incoming)) {
            return result;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!owned_) {
            owner_      = incoming;
            owned_      = true;
            result.kind = layer_stream_transition::ADOPT;
            return result;
        }
        if (layer_stream_owner_same(owner_, incoming)) {
            result.kind = layer_stream_transition::KEEP;
            return result;
        }
        result.kind      = layer_stream_transition::DISPLACE;
        result.displaced = owner_;
        owner_           = incoming;
        return result;
    }

    layer_stream_owner current() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return owned_ ? owner_ : layer_stream_owner{};
    }

    bool owned() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return owned_;
    }

    bool is_owner(const layer_stream_owner & owner) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return owned_ && layer_stream_owner_same(owner_, owner);
    }

    // Drop the ownership record without releasing anything. The caller releases
    // the working set; this only stops the gate claiming it is still held.
    void forget() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        owner_ = {};
        owned_ = false;
    }

  private:
    mutable std::mutex mutex_;
    layer_stream_owner owner_{};
    bool               owned_ = false;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYER_STREAM_OWNER_HPP
