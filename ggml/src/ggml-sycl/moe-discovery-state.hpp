#ifndef GGML_SYCL_MOE_DISCOVERY_STATE_HPP
#define GGML_SYCL_MOE_DISCOVERY_STATE_HPP

// Owner-keyed ownership for the MoE discovery/popularity state cluster
// (llama.cpp-nn6z, canonical §12 lifecycle contract).
//
// The SYCL backend discovers a model's MoE shape once, from the first graph
// that carries MUL_MAT_ID nodes: the hash -> sequential layer map, the expert
// metadata and group registries, the warmup counters, and the popularity ranks
// that drive prestage and cache eviction. All of it used to live in file-scope
// state that no model-lifecycle event ever touched, so the FIRST MoE model to
// reach moe_hybrid_init_once() in a process owned it until module shutdown and
// every later model silently inherited it.
//
// This registry gives that cluster an owner. The rule it implements is
// structural, not a load-boundary sweep: the working set belongs to exactly one
// model token at a time, an owner that stops being active has its half PARKED
// under its own token, and a model teardown removes only that token's state.
// A model that is not the active owner never has the live working set cleared
// out from under it.
//
// The registry itself is host-only and depends on nothing from SYCL, so the
// ownership rules can be proven without a device. ggml-sycl.cpp supplies the
// moe_discovery_live_state that reads and writes the real process globals.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace ggml_sycl {

// Slot value that proves the absence of an owner. Mirrors
// GGML_SYCL_MODEL_SLOT_NONE without pulling the backend header in here.
static constexpr uint32_t moe_owner_slot_none = 0xFFFFFFFFu;

// At most one parked owner per model slot, so the bookkeeping is a fixed array
// and no registry operation allocates while its mutex is held.
static constexpr size_t moe_discovery_max_owners = 32;

// The identity a MoE working set belongs to. It is the full model token: a
// reused slot carrying a newer generation is a DIFFERENT owner, and must not
// resolve to the dead model's parked state.
struct moe_owner_key {
    uint64_t model_id        = 0;
    uint64_t load_txn_id     = 0;
    uint32_t slot            = moe_owner_slot_none;
    uint64_t slot_generation = 0;
};

// Zero ids and the none-slot fail closed: an unattributed caller can never
// become an owner, and can never reach another owner's parked state.
inline bool moe_owner_key_valid(const moe_owner_key & key) {
    return key.model_id != 0 && key.load_txn_id != 0 && key.slot != moe_owner_slot_none &&
           key.slot < moe_discovery_max_owners && key.slot_generation != 0;
}

inline bool moe_owner_key_same(const moe_owner_key & a, const moe_owner_key & b) {
    return a.model_id == b.model_id && a.load_txn_id == b.load_txn_id && a.slot == b.slot &&
           a.slot_generation == b.slot_generation;
}

enum moe_discovery_result {
    MOE_DISCOVERY_RESULT_OK = 0,
    // The key cannot name an owner. Nothing was read, parked or cleared.
    MOE_DISCOVERY_RESULT_INVALID_OWNER,
    // Release of a token that owns no state here. Nothing was cleared.
    MOE_DISCOVERY_RESULT_NOT_FOUND,
    // Activation of the owner that is already active. Deliberately a no-op:
    // re-activating must not discard the discovery the owner just performed.
    MOE_DISCOVERY_RESULT_UNCHANGED,
    // Another transition is mid-flight. No side effects; the caller retries or
    // reports, and the live working set keeps its current owner.
    MOE_DISCOVERY_RESULT_BUSY,
    // The transition completed but state that could not be parked was dropped
    // instead of being left where a later owner might read it.
    MOE_DISCOVERY_RESULT_DEGRADED,
    // No free parked slot. The outgoing owner's state is dropped, not leaked
    // into the incoming owner's working set.
    MOE_DISCOVERY_RESULT_SLOT_EXHAUSTED,
};

// The parked half of one owner's working set. The registry only owns its
// lifetime; ggml-sycl.cpp derives the concrete type and is the only code that
// knows what is inside.
struct moe_discovery_parked_state {
    virtual ~moe_discovery_parked_state() = default;
};

// The live working set. Exactly one exists per process because MoE dispatch
// reads it through file-scope state that a graph submission touches directly.
// The registry never mutates it; it asks its owner to park, restore or
// reinitialize it, always with no registry lock held.
struct moe_discovery_live_state {
    virtual ~moe_discovery_live_state() = default;

    // Copy the parkable half out of the live set. A null return is the
    // fail-closed answer: the caller drops the state rather than leaving it
    // reachable by whichever owner comes next.
    virtual std::unique_ptr<moe_discovery_parked_state> park() noexcept = 0;

    // Install a previously parked set. Implementations must install a fresh
    // device-bound half first: only host discovery survives an owner switch.
    virtual void restore(std::unique_ptr<moe_discovery_parked_state> parked) noexcept = 0;

    // Install a pristine working set for an owner that has nothing parked.
    virtual void install_fresh() noexcept = 0;
};

class moe_discovery_registry {
  public:
    explicit moe_discovery_registry(moe_discovery_live_state & live) : live_(live) {}

    moe_discovery_registry(const moe_discovery_registry &)             = delete;
    moe_discovery_registry & operator=(const moe_discovery_registry &) = delete;

    // Make `owner` the owner of the live working set. The outgoing owner's
    // host discovery is parked under its own token first, so an A -> B -> A
    // sequence gives A back its own popularity ranks and warmup counts rather
    // than B's, and B never starts from A's.
    moe_discovery_result activate(const moe_owner_key & owner) noexcept {
        if (!moe_owner_key_valid(owner)) {
            return MOE_DISCOVERY_RESULT_INVALID_OWNER;
        }

        moe_owner_key                               outgoing{};
        bool                                        outgoing_valid = false;
        std::unique_ptr<moe_discovery_parked_state> incoming;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (transition_) {
                return MOE_DISCOVERY_RESULT_BUSY;
            }
            if (active_valid_ && moe_owner_key_same(active_, owner)) {
                return MOE_DISCOVERY_RESULT_UNCHANGED;
            }
            transition_    = true;
            outgoing       = active_;
            outgoing_valid = active_valid_;
            // Detach rather than read: the incoming owner's parked state must
            // not stay reachable through the registry while it is live again.
            incoming       = detach_locked(owner);
        }

        // Everything that allocates, and every destructor, runs here with no
        // registry lock held.
        std::unique_ptr<moe_discovery_parked_state> parked;
        bool                                        park_failed = false;
        if (outgoing_valid) {
            parked      = live_.park();
            park_failed = !parked;
        }
        if (incoming) {
            live_.restore(std::move(incoming));
        } else {
            live_.install_fresh();
        }

        std::unique_ptr<moe_discovery_parked_state> dropped;
        bool                                        exhausted = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (parked) {
                moe_discovery_slot * slot = free_slot_locked();
                if (slot) {
                    slot->used   = true;
                    slot->owner  = outgoing;
                    slot->parked = std::move(parked);
                } else {
                    dropped   = std::move(parked);
                    exhausted = true;
                }
            }
            active_       = owner;
            active_valid_ = true;
            transition_   = false;
        }
        dropped.reset();

        if (exhausted) {
            return MOE_DISCOVERY_RESULT_SLOT_EXHAUSTED;
        }
        return park_failed ? MOE_DISCOVERY_RESULT_DEGRADED : MOE_DISCOVERY_RESULT_OK;
    }

    // Retire exactly one owner. A model that is not the active owner loses only
    // its parked state: the live working set belongs to whoever is active and is
    // left untouched, which is what makes unloading A safe while B is live.
    moe_discovery_result release(const moe_owner_key & owner) noexcept {
        if (!moe_owner_key_valid(owner)) {
            return MOE_DISCOVERY_RESULT_INVALID_OWNER;
        }

        bool                                        clear_live = false;
        std::unique_ptr<moe_discovery_parked_state> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (transition_) {
                return MOE_DISCOVERY_RESULT_BUSY;
            }
            dropped = detach_locked(owner);
            if (active_valid_ && moe_owner_key_same(active_, owner)) {
                // The live half holds observer pointers into this model's
                // tensors, so it cannot outlive it. This is the dying owner's
                // own state, never a bystander's.
                clear_live    = true;
                active_valid_ = false;
                active_       = {};
                transition_   = true;
            } else if (!dropped) {
                return MOE_DISCOVERY_RESULT_NOT_FOUND;
            }
        }

        dropped.reset();
        if (clear_live) {
            live_.install_fresh();
            std::lock_guard<std::mutex> lock(mutex_);
            transition_ = false;
        }
        return MOE_DISCOVERY_RESULT_OK;
    }

    // Module teardown only. It drops the registry's bookkeeping; the caller
    // owns the process-global clear, because at that point there is no owner
    // left for whom the live half could be preserved.
    void release_all_for_module_shutdown() noexcept {
        moe_discovery_slot released[moe_discovery_max_owners];
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
                released[i].used   = slots_[i].used;
                released[i].owner  = slots_[i].owner;
                released[i].parked = std::move(slots_[i].parked);
                slots_[i].used     = false;
                slots_[i].owner    = {};
            }
            active_valid_ = false;
            active_       = {};
            transition_   = false;
        }
        for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
            released[i].parked.reset();
        }
    }

    bool has_active_owner() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_valid_;
    }

    moe_owner_key active_owner() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_valid_ ? active_ : moe_owner_key{};
    }

    bool is_active_owner(const moe_owner_key & owner) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_valid_ && moe_owner_key_same(active_, owner);
    }

    bool has_parked_owner(const moe_owner_key & owner) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
            if (slots_[i].used && moe_owner_key_same(slots_[i].owner, owner)) {
                return true;
            }
        }
        return false;
    }

    size_t parked_owner_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t                      count = 0;
        for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
            if (slots_[i].used) {
                ++count;
            }
        }
        return count;
    }

  private:
    struct moe_discovery_slot {
        bool                                        used = false;
        moe_owner_key                               owner{};
        std::unique_ptr<moe_discovery_parked_state> parked;
    };

    // Both helpers are called with mutex_ held and neither allocates.
    std::unique_ptr<moe_discovery_parked_state> detach_locked(const moe_owner_key & owner) noexcept {
        for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
            if (slots_[i].used && moe_owner_key_same(slots_[i].owner, owner)) {
                std::unique_ptr<moe_discovery_parked_state> parked = std::move(slots_[i].parked);
                slots_[i].used                                     = false;
                slots_[i].owner                                    = {};
                return parked;
            }
        }
        return nullptr;
    }

    moe_discovery_slot * free_slot_locked() noexcept {
        for (size_t i = 0; i < moe_discovery_max_owners; ++i) {
            if (!slots_[i].used) {
                return &slots_[i];
            }
        }
        return nullptr;
    }

    mutable std::mutex         mutex_;
    moe_discovery_live_state & live_;
    moe_discovery_slot         slots_[moe_discovery_max_owners];
    moe_owner_key              active_{};
    bool                       active_valid_ = false;
    bool                       transition_   = false;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MOE_DISCOVERY_STATE_HPP
