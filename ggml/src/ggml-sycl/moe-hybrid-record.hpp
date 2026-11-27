#pragma once

// Host-side bookkeeping of the event-chained hybrid decode route's per-tensor record
// (ggml_sycl_moe_hybrid_decode_record_current in ggml-sycl.cpp).
//
// The route looks its tensor's record up at every decode op. The record is held under the stamp of the decision that
// built it (moe_decode_direct_stamp: replan epoch, the weight's expert storage generation, selected rows), and a lookup
// whose stamp misses builds the record again, resolving every expert of the tensor. This header says why a stamp
// misses, counts per token what the lookups cost, for GGML_SYCL_MOE_HYBRID_RECORD_TRACE, and keeps the weight's log of
// which expert each storage move changed, so a miss on the generation alone can re-check only those experts.
//
// SYCL-free on purpose so tests/test-sycl-moe-hybrid-record.cpp runs it without a device.

#include "moe-decode-hostpath.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ggml_sycl {

// Which expert each recent move of a weight's expert storage generation changed. A holder of a decision made at an
// older generation (the hybrid decode record) asks which experts changed since then and re-checks only those. The log
// keeps the last `capacity` moves: it answers only for a window it saw whole, in which every move named one expert, so
// a move older than the log, one made without a note, or one that changed every expert leaves the caller to decide
// again from scratch.
struct moe_storage_change_log {
    static constexpr int32_t ALL      = -1;  // a move that changed every expert, or named none
    static constexpr size_t  capacity = 64;

    struct entry {
        uint64_t generation = 0;
        int32_t  expert     = ALL;
    };

    entry  ring[capacity] = {};
    size_t next           = 0;  // where the next move is logged
    size_t held           = 0;  // moves logged, up to capacity

    // The generation moved to `generation`, changing `expert` (ALL: every expert).
    void note(uint64_t generation, int32_t expert) {
        ring[next] = { generation, expert };
        next       = (next + 1) % capacity;
        held       = held < capacity ? held + 1 : capacity;
    }

    // The experts whose storage changed after generation `since` up to `now`, ascending and each once, into *out.
    // False when the log cannot say: the window is longer than the log holds, a move in it was not noted, or one
    // changed every expert.
    bool changed_since(uint64_t since, uint64_t now, std::vector<int32_t> * out) const {
        out->clear();
        if (now < since || now - since > held) {
            return false;
        }
        size_t at = next;
        for (uint64_t g = now; g > since; --g) {
            at              = (at + capacity - 1) % capacity;
            const entry & e = ring[at];
            if (e.generation != g || e.expert < 0) {
                out->clear();
                return false;
            }
            out->push_back(e.expert);
        }
        std::sort(out->begin(), out->end());
        out->erase(std::unique(out->begin(), out->end()), out->end());
        return true;
    }
};

// Why a lookup's stamp does not answer for the op. A stamp that misses on several inputs names the first in this
// order: the epoch before the rows, the rows before the generation.
enum moe_hybrid_record_miss : int {
    MOE_HYBRID_RECORD_MISS_NONE = 0,    // the stamp is current: the held decision stands
    MOE_HYBRID_RECORD_MISS_UNDECIDED,   // no decision is held: never built, or dropped with its record
    MOE_HYBRID_RECORD_MISS_RETRY,       // the last build asked to be retried, so no decision is held
    MOE_HYBRID_RECORD_MISS_EPOCH,       // a mid-load replan moved the replan epoch
    MOE_HYBRID_RECORD_MISS_ROWS,        // the op selects another number of rows
    MOE_HYBRID_RECORD_MISS_GENERATION,  // only the weight's expert storage generation moved
    MOE_HYBRID_RECORD_MISS_COUNT,
};

// NONE exactly when moe_decode_direct_stamp_current() holds for the same inputs.
inline moe_hybrid_record_miss moe_hybrid_record_miss_classify(const moe_decode_direct_stamp & s,
                                                              uint64_t                        replan_epoch,
                                                              uint64_t                        storage_generation,
                                                              int64_t                         selected_rows) {
    if (!s.valid) {
        return s.retries > 0 ? MOE_HYBRID_RECORD_MISS_RETRY : MOE_HYBRID_RECORD_MISS_UNDECIDED;
    }
    if (s.plan_generation != replan_epoch) {
        return MOE_HYBRID_RECORD_MISS_EPOCH;
    }
    if (s.selected_rows != selected_rows) {
        return MOE_HYBRID_RECORD_MISS_ROWS;
    }
    if (s.expert_storage_generation != storage_generation) {
        return MOE_HYBRID_RECORD_MISS_GENERATION;
    }
    return MOE_HYBRID_RECORD_MISS_NONE;
}

inline const char * moe_hybrid_record_miss_name(moe_hybrid_record_miss miss) {
    switch (miss) {
        case MOE_HYBRID_RECORD_MISS_NONE:
            return "none";
        case MOE_HYBRID_RECORD_MISS_UNDECIDED:
            return "undecided";
        case MOE_HYBRID_RECORD_MISS_RETRY:
            return "retry";
        case MOE_HYBRID_RECORD_MISS_EPOCH:
            return "epoch";
        case MOE_HYBRID_RECORD_MISS_ROWS:
            return "rows";
        case MOE_HYBRID_RECORD_MISS_GENERATION:
            return "generation";
        case MOE_HYBRID_RECORD_MISS_COUNT:
            break;
    }
    return "?";
}

inline const char * moe_hybrid_record_outcome_name(moe_decode_direct_outcome outcome) {
    switch (outcome) {
        case MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE:
            return "eligible";
        case MOE_DECODE_DIRECT_OUTCOME_REFUSED:
            return "refused";
        case MOE_DECODE_DIRECT_OUTCOME_RETRY:
            return "retry";
    }
    return "?";
}

// Whether a lookup whose stamp missed can patch the held record, and over which experts.
enum moe_hybrid_record_patch_plan : int {
    MOE_HYBRID_RECORD_PATCH_NONE = 0,  // not a miss a patch answers: the record is built again
    MOE_HYBRID_RECORD_PATCH_UNLOGGED,  // only the generation moved, but the log cannot name the experts that did
    MOE_HYBRID_RECORD_PATCH_EXPERTS,   // re-check the experts the log names
};

// A patch keeps everything the held decision was made under except the moved experts' storage, so only a miss on
// the generation alone, over a decision that built a record, qualifies. A replan, a new row count, a retry or a
// refusal is decided again from scratch. The experts to re-check are the ones the weight's log names since the
// decision's generation, into *changed.
inline moe_hybrid_record_patch_plan moe_hybrid_record_plan_patch(moe_hybrid_record_miss          miss,
                                                                 const moe_decode_direct_stamp & stamp,
                                                                 bool                            held,
                                                                 const moe_storage_change_log &  log,
                                                                 uint64_t                        storage_generation,
                                                                 std::vector<int32_t> *          changed) {
    changed->clear();
    if (miss != MOE_HYBRID_RECORD_MISS_GENERATION || !stamp.eligible || !held) {
        return MOE_HYBRID_RECORD_PATCH_NONE;
    }
    if (!log.changed_since(stamp.expert_storage_generation, storage_generation, changed) || changed->empty()) {
        changed->clear();
        return MOE_HYBRID_RECORD_PATCH_UNLOGGED;
    }
    return MOE_HYBRID_RECORD_PATCH_EXPERTS;
}

// One expert's slot in a hybrid decode record: the weight its executor reads, on the record's device or in host
// memory, never both, and the lease that keeps that storage while the record names it. The record keeps its slots
// as parallel arrays (the CPU job reads host_weight as one), so a slot is this triple only on its way in.
template <typename Lease> struct moe_hybrid_record_slot {
    const void * device_weight = nullptr;
    const void * host_weight   = nullptr;
    Lease        lease;
};

// Installs fresh[i] as the slot of expert changed[i] in a record's slot arrays and moves each lease it replaces into
// *replaced, in slot order, for the caller to keep until the queue passes the record the patch publishes. Every other
// slot and lease stays as it was. Returns false, having changed nothing, when an id is outside the record or `fresh`
// does not pair with `changed`.
template <typename Lease>
bool moe_hybrid_record_install(std::vector<const void *> &                  device_weight,
                               std::vector<const void *> &                  host_weight,
                               std::vector<Lease> &                         lease,
                               const std::vector<int32_t> &                 changed,
                               std::vector<moe_hybrid_record_slot<Lease>> & fresh,
                               std::vector<Lease> *                         replaced) {
    const size_t n_as = lease.size();
    if (fresh.size() != changed.size() || device_weight.size() != n_as || host_weight.size() != n_as) {
        return false;
    }
    for (int32_t e : changed) {
        if (e < 0 || static_cast<size_t>(e) >= n_as) {
            return false;
        }
    }
    for (size_t i = 0; i < changed.size(); ++i) {
        const size_t e = static_cast<size_t>(changed[i]);
        if (lease[e].valid()) {
            replaced->push_back(std::move(lease[e]));
        }
        device_weight[e] = fresh[i].device_weight;
        host_weight[e]   = fresh[i].host_weight;
        lease[e]         = std::move(fresh[i].lease);
    }
    return true;
}

// Empties, in a copy of a record's slot arrays, every slot whose lease `matches` (the device entry an expert loan or
// evict retired): its weights and its lease. Moves each emptied slot's lease into *withdrawn, for the caller to keep
// until the queue passes, and appends its expert id to *experts, in id order. Every other slot and lease stays.
// Returns how many slots it emptied.
template <typename Lease, typename Match>
size_t moe_hybrid_record_withdraw(std::vector<const void *> & device_weight,
                                  std::vector<const void *> & host_weight,
                                  std::vector<Lease> &        lease,
                                  const Match &               matches,
                                  std::vector<Lease> *        withdrawn,
                                  std::vector<int32_t> *      experts) {
    size_t emptied = 0;
    for (size_t e = 0; e < lease.size(); ++e) {
        if (!lease[e].valid() || !matches(lease[e])) {
            continue;
        }
        withdrawn->push_back(std::move(lease[e]));
        lease[e] = Lease{};
        if (e < device_weight.size()) {
            device_weight[e] = nullptr;
        }
        if (e < host_weight.size()) {
            host_weight[e] = nullptr;
        }
        experts->push_back(static_cast<int32_t>(e));
        ++emptied;
    }
    return emptied;
}

// What a record's slots add up to. A slot stands when it names exactly one weight and holds a lease on it.
struct moe_hybrid_record_shape {
    size_t       n_device   = 0;
    size_t       n_host     = 0;
    size_t       n_empty    = 0;        // slots naming no weight or both, or holding no lease
    const void * substitute = nullptr;  // the first device slot's weight: what a host slot's table entry names
};

template <typename Lease>
moe_hybrid_record_shape moe_hybrid_record_shape_of(const std::vector<const void *> & device_weight,
                                                   const std::vector<const void *> & host_weight,
                                                   const std::vector<Lease> &        lease) {
    moe_hybrid_record_shape s;
    if (device_weight.size() != lease.size() || host_weight.size() != lease.size()) {
        s.n_empty = lease.size() + 1;
        return s;
    }
    for (size_t e = 0; e < lease.size(); ++e) {
        const bool on_device = device_weight[e] != nullptr;
        const bool on_host   = host_weight[e] != nullptr;
        if (on_device == on_host || !lease[e].valid()) {
            ++s.n_empty;
        } else if (on_device) {
            ++s.n_device;
            s.substitute = s.substitute ? s.substitute : device_weight[e];
        } else {
            ++s.n_host;
        }
    }
    return s;
}

// Why slots of shape `s` cannot stand as an all-host (`all_host`) or a split record, or nullptr when they can. These
// are the build's own refusals over its slots, so a patch never publishes a record the build would refuse.
inline const char * moe_hybrid_record_shape_refusal(bool all_host, const moe_hybrid_record_shape & s) {
    if (s.n_empty > 0) {
        return "an expert's slot names no weight it holds";
    }
    if (s.n_host == 0) {
        return "no expert runs on the CPU";
    }
    if (all_host && s.n_device > 0) {
        return "the plan puts no expert on a device, yet one resolves there";
    }
    if (!all_host && s.n_device == 0) {
        return "no device expert pointer stands in for the host slots";
    }
    return nullptr;
}

// One token's lookups and what they cost.
struct moe_hybrid_record_counts {
    uint64_t lookups                              = 0;   // one per op the route was asked to take
    uint64_t misses[MOE_HYBRID_RECORD_MISS_COUNT] = {};  // by why the stamp missed; [NONE] is the lookups it answered
    uint64_t generation_steps                     = 0;   // over the GENERATION misses, how far the generation moved
    uint64_t no_plan                              = 0;   // misses that found no published plan: nothing built or held
    uint64_t builds                               = 0;   // record builds, whatever they ended in
    uint64_t built_experts                        = 0;   // experts the builds resolved
    uint64_t outcomes[MOE_DECODE_DIRECT_OUTCOME_RETRY + 1] = {};  // the builds, by how each ended
    uint64_t patches                                       = 0;   // patches published in place of a build
    uint64_t patch_experts                                 = 0;   // experts the patch attempts asked for
    uint64_t patch_declined                                = 0;   // patches that could not stand: a build followed
    uint64_t patch_unlogged                                = 0;   // generation misses the log could not name: built
    uint64_t withdrawn = 0;  // slots loans and evicts withdrew from the records the lookups replaced
};

// The submitting thread's count. A token starts when a lookup's block number is lower than the last lookup's, the rule
// the host-expert wait census uses for its own lines; a run's last token is therefore never handed back.
struct moe_hybrid_record_trace {
    uint64_t                 token      = 0;
    int                      last_layer = -1;
    moe_hybrid_record_counts counts;

    // Counts a lookup at block `layer`. When it starts a token, the token that ended is handed back in *done_token and
    // *done, the count restarts, and the lookup counts into the new token; returns true then.
    bool lookup(int layer, uint64_t * done_token, moe_hybrid_record_counts * done) {
        bool started = false;
        if (last_layer >= 0 && layer < last_layer) {
            *done_token = token;
            *done       = counts;
            counts      = {};
            ++token;
            started = true;
        }
        last_layer = layer;
        ++counts.lookups;
        return started;
    }

    // Why the lookup's stamp missed; `generation_steps` is how far the generation moved, for a GENERATION miss.
    void miss(moe_hybrid_record_miss m, uint64_t generation_steps) {
        ++counts.misses[m];
        if (m == MOE_HYBRID_RECORD_MISS_GENERATION) {
            counts.generation_steps += generation_steps;
        }
    }

    void no_plan() { ++counts.no_plan; }

    // A build resolved `experts` experts.
    void resolved(size_t experts) { counts.built_experts += experts; }

    void build_outcome(moe_decode_direct_outcome outcome) {
        ++counts.builds;
        ++counts.outcomes[outcome];
    }

    // What a lookup's patch plan came to: a patch over `experts` experts that stood or not, or a log that could not
    // name them.
    void patch(moe_hybrid_record_patch_plan plan, bool patched, size_t experts) {
        if (plan == MOE_HYBRID_RECORD_PATCH_UNLOGGED) {
            ++counts.patch_unlogged;
            return;
        }
        if (plan != MOE_HYBRID_RECORD_PATCH_EXPERTS) {
            return;
        }
        counts.patch_experts += experts;
        ++(patched ? counts.patches : counts.patch_declined);
    }

    // A lookup is about to replace a record a loan or an evict withdrew `slots` slots from.
    void withdrawn(size_t slots) { counts.withdrawn += slots; }
};

// The per-token line's fields, after its tag. Returns what snprintf returns. The widest values fit 768 bytes.
inline int moe_hybrid_record_line(char * buf, size_t n, uint64_t token, const moe_hybrid_record_counts & c) {
    return std::snprintf(
        buf, n,
        "token=%llu lookups=%llu current=%llu no_plan=%llu undecided=%llu retry=%llu epoch=%llu rows=%llu "
        "generation=%llu generation_steps=%llu builds=%llu built_experts=%llu eligible=%llu refused=%llu "
        "retried=%llu patches=%llu patch_experts=%llu patch_declined=%llu patch_unlogged=%llu withdrawn=%llu",
        (unsigned long long) token, (unsigned long long) c.lookups,
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_NONE], (unsigned long long) c.no_plan,
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_UNDECIDED],
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_RETRY],
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_EPOCH],
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_ROWS],
        (unsigned long long) c.misses[MOE_HYBRID_RECORD_MISS_GENERATION], (unsigned long long) c.generation_steps,
        (unsigned long long) c.builds, (unsigned long long) c.built_experts,
        (unsigned long long) c.outcomes[MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE],
        (unsigned long long) c.outcomes[MOE_DECODE_DIRECT_OUTCOME_REFUSED],
        (unsigned long long) c.outcomes[MOE_DECODE_DIRECT_OUTCOME_RETRY], (unsigned long long) c.patches,
        (unsigned long long) c.patch_experts, (unsigned long long) c.patch_declined,
        (unsigned long long) c.patch_unlogged, (unsigned long long) c.withdrawn);
}

}  // namespace ggml_sycl
