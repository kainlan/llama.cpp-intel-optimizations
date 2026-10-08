#pragma once

// Memo of graph_preload_moe_experts() outcomes, one stamp per expert tensor
// and device.
//
// The preload walks every MUL_MAT_ID of a split: it sizes compact lists,
// ensures the expert pointer table, stages ids and resolves every expert's
// route. Its answer for a tensor depends on where that tensor's experts live,
// not on the token, so re-deciding it per token repeats hundreds of route
// resolutions per tensor for an answer that cannot have changed. The stamp
// holds what can change it:
//   - the replan epoch (a MID_LOAD_REPLAN bumps it),
//   - the tensor's expert storage generation (publishing or forgetting expert
//     storage bumps it),
//   - the routed-token count (it feeds the layout choice),
//   - whether an all-host tensor may stay a direct boundary node,
//   - the device.
// Structural refusals are stamped as well as successes, so a refusal is
// decided once per residency state instead of once per token, and a residency
// change re-opens it. The stamp lives on the tensor's weight extension, like
// the decode direct-dispatch stamp, so it is keyed by the weight itself and
// never by an address that a later tensor could reuse.
//
// What the key does NOT cover: the layout the preload probes. Layout follows
// the stamped inputs, but also the decode down-i8 hotset (the ids of the
// moment) and the prompt-down transient-SoA state, neither of which is keyed.
// Each outcome stays safe without it:
//   - HOST_TIER_BOUNDARY counts host routes, which the probe accepts at any
//     layout, so it does not depend on layout at all.
//   - REFUSED can be stale against a layout the hotset would have chosen. It
//     fails closed: the split runs direct, which is always correct, until a
//     residency change re-opens it.
//   - PREPARED only lets the PP->TG refresh skip its residency check. The
//     graph path re-runs the preload on every call with the layout of that
//     call, so a recorded graph never relies on a stamped table.
//
// SYCL-free on purpose so tests/test-sycl-moe-graph-preload-stamp.cpp can run
// it without a device.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// What the preload concluded for one tensor.
enum class moe_graph_preload_outcome {
    PREPARED,            // one device pointer table represents it; the preload prepared it
    HOST_TIER_BOUNDARY,  // every expert is host-planned: a direct CPU node, nothing to prepare
    REFUSED,             // mixed or missing experts: no single table represents it
};

struct moe_graph_preload_stamp {
    uint64_t                  replan_epoch       = 0;
    uint64_t                  storage_generation = 0;
    int64_t                   n_tokens           = 0;
    int                       device             = -1;
    bool                      host_tier_boundary = false;
    bool                      valid              = false;
    moe_graph_preload_outcome outcome            = moe_graph_preload_outcome::PREPARED;
};

struct moe_graph_preload_inputs {
    uint64_t replan_epoch       = 0;
    uint64_t storage_generation = 0;
    int64_t  n_tokens           = 0;
    int      device             = -1;
    bool     host_tier_boundary = false;
};

inline bool moe_graph_preload_stamp_current(const moe_graph_preload_stamp & s, const moe_graph_preload_inputs & in) {
    return s.valid && s.replan_epoch == in.replan_epoch && s.storage_generation == in.storage_generation &&
           s.n_tokens == in.n_tokens && s.device == in.device && s.host_tier_boundary == in.host_tier_boundary;
}

inline void moe_graph_preload_stamp_record(moe_graph_preload_stamp &        s,
                                           const moe_graph_preload_inputs & in,
                                           moe_graph_preload_outcome        outcome) {
    s.replan_epoch       = in.replan_epoch;
    s.storage_generation = in.storage_generation;
    s.n_tokens           = in.n_tokens;
    s.device             = in.device;
    s.host_tier_boundary = in.host_tier_boundary;
    s.valid              = true;
    s.outcome            = outcome;
}

// The preload checks this first for every tensor, before any layout selection or route probe: a tensor already
// known to be all host-planned under the current inputs needs none of that work again.
inline bool moe_graph_preload_stamp_skips_tensor(const moe_graph_preload_stamp &  s,
                                                 const moe_graph_preload_inputs & in) {
    return moe_graph_preload_stamp_current(s, in) && s.outcome == moe_graph_preload_outcome::HOST_TIER_BOUNDARY;
}

// Why a preload stopped. Only a STRUCTURAL failure (the route probe found mixed or missing experts) is a fact about
// residency, so only it is stamped: the stamp lives on the weight and is shared by every context using it. A
// TRANSIENT failure (ids readback, a staging allocation, a pointer-table update, invalid geometry, a disabled
// preload mode) refuses the current call only and leaves the stamp as it was.
enum class moe_graph_preload_failure {
    STRUCTURAL,
    TRANSIENT,
};

inline void moe_graph_preload_stamp_failure(moe_graph_preload_stamp &        s,
                                            const moe_graph_preload_inputs & in,
                                            moe_graph_preload_failure        failure) {
    if (failure == moe_graph_preload_failure::STRUCTURAL) {
        moe_graph_preload_stamp_record(s, in, moe_graph_preload_outcome::REFUSED);
    }
}

// What a split's stamps say about running the preload for it.
enum class moe_graph_preload_split_decision {
    NO_MOE,    // no MUL_MAT_ID: nothing to prepare
    REFUSED,   // a tensor's current stamp is a refusal: graphs stay off for this split, nothing to run
    KNOWN_OK,  // every tensor has a current success stamp: the residency check need not run again
    RUN,       // some tensor has no current stamp: run the preload
};

struct moe_graph_preload_split_scan {
    int  n_tensors = 0;
    int  n_ok      = 0;
    bool refused   = false;
};

inline void moe_graph_preload_split_add(moe_graph_preload_split_scan &   scan,
                                        const moe_graph_preload_stamp &  stamp,
                                        const moe_graph_preload_inputs & in) {
    scan.n_tensors++;
    if (!moe_graph_preload_stamp_current(stamp, in)) {
        return;
    }
    if (stamp.outcome == moe_graph_preload_outcome::REFUSED) {
        scan.refused = true;
    } else {
        scan.n_ok++;
    }
}

inline moe_graph_preload_split_decision moe_graph_preload_split_decide(const moe_graph_preload_split_scan & scan) {
    if (scan.n_tensors == 0) {
        return moe_graph_preload_split_decision::NO_MOE;
    }
    if (scan.refused) {
        return moe_graph_preload_split_decision::REFUSED;
    }
    return scan.n_ok == scan.n_tensors ? moe_graph_preload_split_decision::KNOWN_OK :
                                         moe_graph_preload_split_decision::RUN;
}

// What the preload does with one planner-owned tensor, from its route probe.
enum class moe_graph_preload_tensor_verdict {
    TABLE,               // every expert is local to the device: one pointer table represents it
    HOST_TIER_BOUNDARY,  // every expert is host-planned: it runs on the CPU as a direct node outside the graph
    REFUSE,              // mixed or missing: no single table represents it
};

inline moe_graph_preload_tensor_verdict moe_graph_preload_classify(size_t  local,
                                                                   size_t  secondary,
                                                                   size_t  host,
                                                                   size_t  missing,
                                                                   int64_t n_experts,
                                                                   bool    host_tier_boundary) {
    const size_t all = n_experts > 0 ? static_cast<size_t>(n_experts) : 0;
    if (local == all) {
        return moe_graph_preload_tensor_verdict::TABLE;
    }
    // Placement decides the executor: host-planned experts run on the CPU, so an all-host tensor needs no
    // device table. It is only safe where the graph path keeps MUL_MAT_ID nodes out of recorded graphs.
    if (host_tier_boundary && all > 0 && host == all && local == 0 && secondary == 0 && missing == 0) {
        return moe_graph_preload_tensor_verdict::HOST_TIER_BOUNDARY;
    }
    return moe_graph_preload_tensor_verdict::REFUSE;
}

// Post-prompt work: after a prompt, some per-split work (releasing prompt-only down layouts, materializing decode
// down layouts, the residency check) must run once on each split's first decode occurrence, never per token. A
// global prompt epoch advances on every prompt split; each expert tensor records, per device, the epoch it last
// handled. A decode split is due when any of its tensors is behind.
inline bool moe_post_prompt_work_due(uint64_t handled_epoch, uint64_t prompt_epoch) {
    return handled_epoch < prompt_epoch;
}

}  // namespace ggml_sycl
