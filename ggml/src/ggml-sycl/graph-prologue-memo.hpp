#pragma once

// The decode prologue memo of one backend context: the graph signature, the keyed segment slot key and the input
// prestage's effect list of the graph the context last computed, served again while llama hands it the same graph.
//
// The signal is ggml_cgraph::uid. The scheduler gives every split a fresh process-wide uid when it splits a graph,
// and llama's graph-reuse branch skips the scheduler reset, the graph build and the allocation, so a split that
// comes back with the same uid has the same node array, tensor structs, shapes, names, buffers and data pointers. A
// rebuilt graph gets a new uid. A graph from ggml_new_graph, a graph view and the cb_eval path have uid 0, which
// means "not set" and never matches.
//
// That premise is about llama and the scheduler; the backend itself rewrites a graph tensor's data in place, under
// an unchanged uid, buffer and publish generation, in two statements: the existing-storage publish (the cross-card
// MoE scatter, the layer-block executors' planned outputs) and the f16 attention destination publish (the
// cross-device routes). Those need the cross-card expert path or another routable device, so the memo depends on
// the backend's answer to "may it arm for this key": begin() asks it when the key changes, and the backend answers
// yes only where neither statement can run for the context.
//
// key = (uid, n_nodes, n_leafs, device, decode phase, tenant publish generation). n_nodes is in it because the
// GPU-prefix mode truncates a graph in place; the publish generation, because a plan publication or a teardown
// release may change where a tensor's bytes live.
//
// Each part is recorded by the full function on the first call under a key and served on later calls under it:
//   signature - the structural hash, served as recorded;
//   slot key  - served as recorded while every distinct buffer the full walk met keeps the identity and the base it
//               had then. Each tensor's buffer and offset are fixed while the uid matches, so the storage half is
//               then bit-equal to a full walk; a changed identity or base sends the caller to the full walk, so the
//               key changes exactly as it would without the memo;
//   prestage  - the tensors, in walk order and each once, that reached a stage action of a pass that staged every
//               one of them; the caller runs exactly those actions again. The pass's other arms mark a tensor seen
//               or count it, and building the mark's cache key registers a host weight's extra and gives it a cache
//               uuid the first time. The full pass every miss runs has done that, and doing it again changes
//               nothing, so a replay that skips those arms leaves the same state. Every tensor keeps its class while
//               the uid matches. A pass that did not stage everything is never recorded.
//
// The memo owns nothing: a tensor or buffer pointer in it is a view, valid only while the key matches. Every point
// that retires recorded work for the context clears it, and it is a member of the context, so it dies with it.
//
// Host-only: no SYCL and no ggml types (the tensor and buffer types are template parameters), so the switch, the
// key, the parts and the verify bookkeeping are unit-tested without a device (tests/test-graph-prologue-memo.cpp).

#include "graph-segment-cache.hpp"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace ggml_sycl {
namespace graph_prologue_memo {

// GGML_SYCL_GRAPH_PROLOGUE_MEMO, read once by the backend.
enum class mode {
    OFF,     // the full functions on every call; the memo is never touched
    ON,      // serve hits
    VERIFY,  // on a hit run the full function as well, compare, and on a difference use the full result and clear
};

// Unset or empty is ON. Any other value is read with atoi: 0 is OFF, and so is a word such as "on" or "off", which
// atoi reads as 0; 2 is VERIFY; any other number is ON.
inline mode mode_parse(const char * value) {
    if (value == nullptr || value[0] == '\0') {
        return mode::ON;
    }
    switch (std::atoi(value)) {
        case 0:
            return mode::OFF;
        case 2:
            return mode::VERIFY;
        default:
            return mode::ON;
    }
}

struct key {
    uint64_t uid        = 0;  // ggml_cgraph::uid; 0 is "not set" and never matches
    int32_t  n_nodes    = 0;
    int32_t  n_leafs    = 0;
    int32_t  device     = -1;
    bool     is_decode  = false;
    uint64_t tenant_gen = 0;  // ggml_sycl_tenant_publish_gen() when the call began
};

inline bool operator==(const key & a, const key & b) {
    return a.uid == b.uid && a.n_nodes == b.n_nodes && a.n_leafs == b.n_leafs && a.device == b.device &&
           a.is_decode == b.is_decode && a.tenant_gen == b.tenant_gen;
}

// The three parts, as the verify mismatch line names them.
enum class part {
    SIGNATURE,
    SLOT_KEY,
    PRESTAGE,
};

inline const char * part_name(part p) {
    switch (p) {
        case part::SIGNATURE:
            return "signature";
        case part::SLOT_KEY:
            return "slot key";
        case part::PRESTAGE:
            return "prestage effect list and answer";
    }
    return "?";
}

// A distinct buffer the slot key's full walk met, with the owner identity and the base it had then.
template <typename Buffer> struct buffer_identity {
    Buffer       buffer{};
    uint64_t     identity = 0;
    const char * base     = nullptr;
};

// The stage action a tensor reached in the prestage pass.
enum class prestage_arm : uint8_t {
    INPUT,  // a named INPUT tensor: its stable per-tensor staging buffer
    CACHE,  // any other host tensor: the unified cache, then the staging cache when the cache cannot take it
};

template <typename Tensor> struct prestage_effect {
    const Tensor * tensor = nullptr;
    prestage_arm   arm    = prestage_arm::CACHE;
    const char *   source = "";  // where the walk met it ("leaf", "node_src"), for the failure log only
};

// Equal by tensor and action. The source label does not take part: it names where the walk met the tensor first.
template <typename Tensor>
inline bool operator==(const prestage_effect<Tensor> & a, const prestage_effect<Tensor> & b) {
    return a.tensor == b.tensor && a.arm == b.arm;
}

// Summed over the three parts.
struct counters {
    uint64_t hits       = 0;  // a part served from the memo (in verify mode, a part that was then compared)
    uint64_t misses     = 0;  // a part the full function computed, recorded when the memo was armed
    uint64_t verified   = 0;  // verify mode: a served part compared with its full function
    uint64_t mismatches = 0;  // verify mode: a comparison that differed
};

template <typename Tensor, typename Buffer> class memo {
  public:
    using effect   = prestage_effect<Tensor>;
    using identity = buffer_identity<Buffer>;

    // Starts a call under k. The parts recorded under the same key stay, and so does the answer to may_arm(). Any
    // other key drops them and, when k has a uid, asks may_arm() whether the memo may arm for k; a no leaves it
    // unarmed, serving and recording nothing, until the key changes. The answer is asked once per key because the
    // caller's answer depends only on what the key pins. A key with uid 0 drops the parts, arms nothing and does not
    // ask, so no part is served or recorded until a call whose graph has a uid.
    template <typename MayArm> void begin(const key & k, MayArm && may_arm) {
        if (k.uid != 0 && key_ == k) {
            return;
        }
        clear();
        if (k.uid != 0) {
            key_   = k;
            armed_ = may_arm();
        }
    }

    // True when the current key may record and serve parts.
    bool armed() const { return armed_; }

    // Drops every part and forgets the key, so nothing is recorded again before the next begin(), which asks afresh.
    // The counters stay.
    void clear() {
        armed_ = false;
        key_   = key{};
        drop_parts();
    }

    // Drops every part and keeps the key and its answer: the caller records afresh under the same key. It serves a
    // check that found something outside the key moved, which may have moved what the other parts were taken from.
    void drop_parts() {
        has_signature_ = false;
        has_slot_key_  = false;
        has_prestage_  = false;
        buffers_.clear();
        effects_.clear();
    }

    // The recorded signature, or null.
    const uint64_t * signature() const { return armed_ && has_signature_ ? &signature_ : nullptr; }

    void record_signature(uint64_t signature) {
        if (!armed_) {
            return;
        }
        signature_     = signature;
        has_signature_ = true;
    }

    // The recorded slot key, or null. The caller serves it only when slot_key_buffers_unchanged() holds.
    const graph_segment_cache::key * slot_key() const { return armed_ && has_slot_key_ ? &slot_key_ : nullptr; }

    const std::vector<identity> & slot_key_buffers() const { return buffers_; }

    void record_slot_key(const graph_segment_cache::key & k, std::vector<identity> && buffers) {
        if (!armed_) {
            return;
        }
        slot_key_     = k;
        buffers_      = std::move(buffers);
        has_slot_key_ = true;
    }

    // True when every recorded buffer still has the identity and the base it had when the slot key was recorded.
    template <typename IdentityOf, typename BaseOf>
    bool slot_key_buffers_unchanged(IdentityOf && identity_of, BaseOf && base_of) const {
        for (const identity & b : buffers_) {
            if (identity_of(b.buffer) != b.identity || base_of(b.buffer) != b.base) {
                return false;
            }
        }
        return true;
    }

    // The recorded prestage effect list, or null.
    const std::vector<effect> * prestage_effects() const { return armed_ && has_prestage_ ? &effects_ : nullptr; }

    // Records a full pass's effect list when the pass staged everything. A pass that did not is never recorded, and
    // drops a list held from an earlier pass.
    void record_prestage(bool all_staged, std::vector<effect> && effects) {
        if (!all_staged) {
            drop_prestage();
            return;
        }
        if (!armed_) {
            return;
        }
        effects_      = std::move(effects);
        has_prestage_ = true;
    }

    // Drops the prestage list alone: a replay of it that failed to stage a tensor answered "not all staged".
    void drop_prestage() {
        has_prestage_ = false;
        effects_.clear();
    }

    void note_hit() { stats_.hits++; }

    void note_miss() { stats_.misses++; }

    // A verify-mode comparison of a served part with its full function. A difference counts a mismatch and clears
    // the memo, so the caller uses the full result and the next call records afresh. Returns `same`.
    bool note_verified(bool same) {
        stats_.verified++;
        if (!same) {
            stats_.mismatches++;
            clear();
        }
        return same;
    }

    const counters & stats() const { return stats_; }

  private:
    bool                     armed_ = false;
    key                      key_{};
    bool                     has_signature_ = false;
    uint64_t                 signature_     = 0;
    bool                     has_slot_key_  = false;
    graph_segment_cache::key slot_key_{};
    std::vector<identity>    buffers_;
    bool                     has_prestage_ = false;
    std::vector<effect>      effects_;
    counters                 stats_;
};

}  // namespace graph_prologue_memo
}  // namespace ggml_sycl
