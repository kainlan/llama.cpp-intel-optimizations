//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MOE_RING_CONSUMER_HPP
#define GGML_SYCL_MOE_RING_CONSUMER_HPP

// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it.
//
// The loader sizes the ring for a model with an expert weight of a type a ring consumer runs
// (ggml_sycl_inventory_ring_has_consumer in ggml-sycl.cpp), and the ring's room comes out of the device's weight
// capacity. The rule is by use, not by model or type name. A ring is dropped only when every weight that uses it (a)
// reaches the batched executor's resident claim alone (the types moe_pp_ring_uses_of_type answers with
// MOE_PP_RING_USE_RESIDENT alone: today NVFP4, IQ1_S, IQ1_M and IQ4_XS), and (b) has another device route for a
// resident prompt node (today the MMID direct route, moe_mmid_direct_route_runs_type), and (c) the plan made with the
// ring leaves at least one expert of those weights on the host, and (d) the drop returns room to the weights: the
// RUNTIME zone needs less without the ring, and the arena may be rebuilt (moe_pp_ring_decide). Such a ring serves a
// layer only when every expert of it is planned on the device and stages no host expert, so where (c) holds it has
// traded prompt room for a decode loss on every token, and measured on the B50 with Inkling 2 Mini it lost on the
// prompt as well. Where (d) holds too the load drops the ring, its RUNTIME zone is rebuilt smaller, and the device is
// planned once more (ggml_sycl_stage_drop_displacing_ring in ggml-sycl.cpp): a drop always means that. Where (d) fails,
// a drop would only take the batched executor from the resident layers' prompts and give the experts nothing: a zone
// another consumer sizes (Qwen3.6 NVFP4 on the B50, measured) needs as much without the ring, and an arena with live
// allocations is not rebuilt in this load. The ring then stays published as it was and the plan made with it stands.
// Today the drop set is NVFP4 alone, and the plans that change are those whose NVFP4 experts overflow a card and whose
// RUNTIME zone the ring enlarges: Inkling 2 Mini on the B50 is the one measured. (b) exists because the ring is the
// only device prompt route of a resident IQ1_S, IQ1_M or IQ4_XS layer (decode stays refused for them): a drop would
// leave a node the executor's claim admits with no kernel to run it. A model whose experts of those types overflow a
// card therefore still pays the ring's room. A ring with any STAGED or SOA_EXPERT use, or with any weight that has no
// other device route, is sized, published and kept exactly as before.
//
// Host-only: the plan is data, so tests/test-sycl-moe-ring-consumer.cpp checks the answer on synthetic plans.

#include "moe-mmvq-tables.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// An expert weight of an inventory, as the ring decision reads it: its name, its expert count (ne[2]), its bytes (every
// expert), the ring uses its type reaches (moe_pp_ring_uses_of_type), and whether a resident prompt node of it has a
// device route besides the ring (moe_mmid_direct_route_runs_type). The stage and the plan dry run build the list from
// the inventory with the filter the ring's sizing reads.
struct moe_pp_ring_weight {
    const char * name                      = nullptr;
    int64_t      n_experts                 = 0;
    size_t       bytes                     = 0;
    uint32_t     uses                      = MOE_PP_RING_USE_NONE;
    bool         other_device_prompt_route = false;
};

// What the plan made with a ring says about it.
struct moe_pp_ring_resident_check {
    size_t  n_weights                 = 0;      // the weights with a ring use
    bool    resident_only             = false;  // at least one, and every one reaches MOE_PP_RING_USE_RESIDENT alone
    bool    other_device_prompt_route = false;  // resident_only, and every one has a device prompt route beside it
    int64_t host_experts              = 0;      // when both: the experts of those weights the plan leaves on the host

    // (a) to (c): the ring serves resident layers only, each of which keeps a device prompt route without it, and the
    // plan made with it left an expert of those layers' weights on the host. Whether that drops the ring is (d)'s to
    // say (moe_pp_ring_decide); a load reads no zone figure for a ring this keeps.
    bool displaces_experts() const { return resident_only && other_device_prompt_route && host_experts > 0; }
};

// The check of `weights` under `plan`. A weight with no ring use is not a ring weight and is skipped. Any weight with a
// use beside the resident claim, or with no device prompt route besides the ring, makes the ring one the load keeps as
// before, and nothing more is counted. Otherwise every expert in [0, n_experts) of each weight that the plan does not
// place on a device is counted, on any device's plan, by the plan's own convention (placement_plan::expert_on_device):
// an expert with no entry, or with a name-keyed fallback entry on the host, is on the host, never implicitly on a
// device.
inline moe_pp_ring_resident_check moe_pp_ring_check_resident_only(const placement_plan &     plan,
                                                                  const moe_pp_ring_weight * weights,
                                                                  size_t                     n_weights) {
    moe_pp_ring_resident_check check;
    if (weights == nullptr) {
        return check;
    }
    bool resident_only = true;
    bool other_route   = true;
    for (size_t i = 0; i < n_weights; ++i) {
        if (weights[i].uses == MOE_PP_RING_USE_NONE) {
            continue;
        }
        ++check.n_weights;
        if (weights[i].uses != MOE_PP_RING_USE_RESIDENT) {
            resident_only = false;
        }
        if (!weights[i].other_device_prompt_route) {
            other_route = false;
        }
    }
    check.resident_only             = resident_only && check.n_weights > 0;
    check.other_device_prompt_route = check.resident_only && other_route;
    if (!check.other_device_prompt_route) {
        return check;
    }
    for (size_t i = 0; i < n_weights; ++i) {
        if (weights[i].uses == MOE_PP_RING_USE_NONE || weights[i].name == nullptr || weights[i].n_experts <= 0) {
            continue;
        }
        check.host_experts +=
            weights[i].n_experts - plan.count_experts_on_device(weights[i].name, weights[i].n_experts, -1);
    }
    return check;
}

// The RUNTIME zone figures (d) reads, each as the arena reserves the zone (zone_runtime_reserved_bytes over the floor
// and the planned requirement): the zone the plan made with the ring needs, no more than the arena holds of it; the zone
// the device needs without the ring, at the load's planning shape and with the planner terms from before that plan; and
// whether the arena may be rebuilt (zone_arena_rebuild_allowed: nothing points into it). A device with no arena has no
// zone: both figures are 0.
struct moe_pp_ring_zone_answer {
    size_t runtime_with_ring    = 0;
    size_t runtime_without_ring = 0;
    bool   rebuild_allowed      = false;
};

// What a load does with a device's ring after the plan made with it, and why.
enum class moe_pp_ring_decision {
    KEEP_NO_RING_WEIGHT,         // no ring is published, or no weight uses it: nothing to decide
    KEEP_STAGED_OR_SOA_USE,      // (a): a weight reaches the staged claim or the SOA per-expert staging
    KEEP_NO_OTHER_DEVICE_ROUTE,  // (b): a weight's resident prompt node has no device route but the ring
    KEEP_NO_HOST_EXPERT,         // (c): the plan left no expert of the ring's weights on the host
    KEEP_NO_ROOM_RETURNS,        // (d): the zone needs no less without the ring (another consumer sizes it, or none)
    KEEP_HELD,                   // (d): the zone would need less, but the arena may not be rebuilt in this load
    DROP,                        // all four hold: the ring dropped, the zone rebuilt smaller, one more plan
};

// The decision on a device's ring from the plan's check and the zone's figures, (a) to (d) in that order. Room is asked
// before the arena: where none would return, nothing needs rebuilding. Pure: the load stage and the plan dry run ask
// it, each with its own zone figures.
inline moe_pp_ring_decision moe_pp_ring_decide(const moe_pp_ring_resident_check & check,
                                               const moe_pp_ring_zone_answer &    zone) {
    if (check.n_weights == 0) {
        return moe_pp_ring_decision::KEEP_NO_RING_WEIGHT;
    }
    if (!check.resident_only) {
        return moe_pp_ring_decision::KEEP_STAGED_OR_SOA_USE;
    }
    if (!check.other_device_prompt_route) {
        return moe_pp_ring_decision::KEEP_NO_OTHER_DEVICE_ROUTE;
    }
    if (check.host_experts <= 0) {
        return moe_pp_ring_decision::KEEP_NO_HOST_EXPERT;
    }
    if (zone.runtime_without_ring >= zone.runtime_with_ring) {
        return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS;
    }
    if (!zone.rebuild_allowed) {
        return moe_pp_ring_decision::KEEP_HELD;
    }
    return moe_pp_ring_decision::DROP;
}

// How many experts of the ring weights in `weights` `room` bytes hold at their mean size (their bytes over their
// experts, rounded down): the experts' worth of room a ring keeps from them where the arena holds it (KEEP_HELD). A
// weight with no ring use is skipped. 0 when no ring weight has an expert of at least one byte.
inline int64_t moe_pp_ring_experts_worth(const moe_pp_ring_weight * weights, size_t n_weights, size_t room) {
    if (weights == nullptr) {
        return 0;
    }
    uint64_t bytes   = 0;
    uint64_t experts = 0;
    for (size_t i = 0; i < n_weights; ++i) {
        if (weights[i].uses == MOE_PP_RING_USE_NONE || weights[i].n_experts <= 0) {
            continue;
        }
        bytes += weights[i].bytes;
        experts += static_cast<uint64_t>(weights[i].n_experts);
    }
    const uint64_t expert_bytes = experts == 0 ? 0 : bytes / experts;
    return expert_bytes == 0 ? 0 : static_cast<int64_t>(room / expert_bytes);
}

// The load that dropped each device's ring, one slot per device: the load transaction of the pass that dropped it, 0
// for none. A load plans a device at its early pass and again at its late one; a later pass of the load that dropped
// the ring plans without it again (dropped()), so it does not size the ring's zone back in. A pass that keeps the ring
// marks nothing, and the next pass decides afresh on its own plan. A direct planner entry has
// no load transaction (0) and marks nothing, so it never erases a staged load's mark, and asks afresh. One slot per
// device means a second load interleaved with the first can overwrite the first's mark: the first's next pass then
// decides afresh with the same check, which costs at most one more arena rebuild and never a wrong plan.
struct moe_pp_ring_drop_marks {
    uint64_t load_txn[GGML_SYCL_MAX_DEVICES] = {};

    static bool device_in_range(int device) { return device >= 0 && device < GGML_SYCL_MAX_DEVICES; }

    // Records that load `txn` dropped `device`'s ring. Nothing for txn 0 or a device out of range.
    void mark(int device, uint64_t txn) {
        if (txn != 0 && device_in_range(device)) {
            load_txn[device] = txn;
        }
    }

    // Whether load `txn` dropped `device`'s ring at an earlier pass. Never for txn 0.
    bool dropped(int device, uint64_t txn) const {
        return txn != 0 && device_in_range(device) && load_txn[device] == txn;
    }
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MOE_RING_CONSUMER_HPP
