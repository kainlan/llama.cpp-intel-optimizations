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
// capacity. The rule is by use, not by model or type name. A ring is dropped only when all four of these hold: (a)
// every weight that uses it reaches the batched executor's resident claim alone (the types moe_pp_ring_uses_of_type
// answers with MOE_PP_RING_USE_RESIDENT alone: today NVFP4, IQ1_S, IQ1_M and IQ4_XS); (b) every one has another device
// route for a resident prompt node (today the MMID direct route, moe_mmid_direct_route_runs_type); (c) the plan made
// with the ring leaves at least one expert of those weights on the host; and (d) the drop returns room to the weights:
// the arena's WEIGHT zone, the one both planners pack the weights against, grows once the ring is gone
// (moe_pp_ring_decide). Such a ring serves a layer only when every expert of it is planned on the device and stages no
// host expert, so where (c) holds it has traded prompt room for a decode loss on every token, and measured on the B50
// with Inkling 2 Mini it lost on the prompt as well. Where (d) holds too the load drops the ring, the arena is rebuilt
// with a smaller RUNTIME zone and a larger WEIGHT zone, and the device is planned once more
// (ggml_sycl_stage_drop_displacing_ring in ggml-sycl.cpp): a drop always means that. Where (d) fails, a drop would only
// take the batched executor from the resident layers' prompts and give the experts nothing: a RUNTIME zone another
// consumer sizes needs as much without the ring, a plan without the ring that would grow another zone (the SCRATCH
// zone's MoE MMVQ bound is held to the room RUNTIME leaves beside it) would hand that zone the room, and an arena with
// live allocations is not rebuilt in this load. The ring then stays published as it was and the plan made with it
// stands.
//
// Today the drop set is NVFP4 alone, and a plan changes only where all four hold: every weight of its ring is NVFP4 (one
// IQ1_S, IQ1_M or IQ4_XS weight beside them keeps the ring, for want of another device prompt route), its NVFP4 experts
// overflow the card, the ring enlarges the arena's zones (its total less the WEIGHT zone) by room the WEIGHT zone takes
// back once the ring is gone, and nothing points into the arena (a held arena keeps the ring, with a WARN). Inkling 2
// Mini on the B50 is the one measured. (b) exists because the ring is the only device prompt route of a resident IQ1_S,
// IQ1_M or IQ4_XS layer (decode stays refused for them): a drop would leave a node the executor's claim admits with no
// kernel to run it. A model whose experts of those types overflow a card therefore still pays the ring's room. A ring
// with any STAGED or SOA_EXPERT use, or with any weight that has no other device route, is sized, published and kept
// exactly as before.
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

// The RUNTIME zone figures (d) asks first, each as the arena reserves the zone (zone_runtime_reserved_bytes over the
// floor and the planned requirement), SIZE_MAX where the requirement is unanswerable. Where the ring takes no room in
// the RUNTIME zone a drop cannot give the weights any, and the load asks the arena nothing.
//
// The RUNTIME zone the plan made with the ring holds: `planned`, the zone the terms published with the ring ask for, no
// more than `held`, the arena's RUNTIME zone, where an arena is active, so the room never counts more than the arena
// has; with no arena, `planned`, the zone a load's entry ensure reserves for those terms, an unanswerable one read as
// `floor`, the zone the ensure keeps for it. The load stage and the plan dry run read it alike
// (ggml_sycl_ring_runtime_with_ring_bytes in ggml-sycl.cpp), so neither reads an unanswerable figure as room.
inline size_t moe_pp_ring_runtime_with_ring(size_t planned, bool arena_active, size_t held, size_t floor) {
    if (arena_active) {
        return planned < held ? planned : held;
    }
    return planned == SIZE_MAX ? floor : planned;
}

// The room the ring takes in the RUNTIME zone: `with_ring` (moe_pp_ring_runtime_with_ring) less `without_ring`, the zone
// the device needs with no ring published; 0 where it needs no less, an unanswerable figure without the ring included.
inline size_t moe_pp_ring_runtime_room(size_t with_ring, size_t without_ring) {
    return without_ring < with_ring ? with_ring - without_ring : 0;
}

// The figures (d) decides on: the room the weights have, read as both planners read it (the arena's WEIGHT zone,
// unified_cache_kv_weight_capacity), as the plan made with the ring found it and once the zones were asked for without
// the ring with the RUNTIME zone's surplus given back (zone_arena_ensure_options::give_back); and whether the arena had
// that surplus to give but something pointed into it, so it was kept as built. The WEIGHT zone is the measure, not the
// RUNTIME zone: a rebuild that grows another zone with the room RUNTIME gives up leaves the WEIGHT zone as it was, and
// so does an N-chunk arena whose tail chunk is at its cap, where the KV share at the head of that chunk takes the room.
// A device with no arena has no zone: every figure is 0.
struct moe_pp_ring_zone_answer {
    size_t weight_with_ring    = 0;
    size_t weight_without_ring = 0;
    bool   held                = false;
};

// The answer the plan dry run decides on, from `runtime_room` (moe_pp_ring_runtime_room) alone. The dry run rebuilds no
// arena, so it answers as a give-back on an arena nothing points into whose WEIGHT zone takes the whole RUNTIME room, as
// a single chunk's does (its WEIGHT zone is the chunk less the ONEDNN, RUNTIME and SCRATCH zones), counted from 0. It
// cannot see a plan without the ring that would grow another zone, for which a load keeps the ring.
inline moe_pp_ring_zone_answer moe_pp_ring_dry_run_answer(size_t runtime_room) {
    moe_pp_ring_zone_answer zone;
    zone.weight_without_ring = runtime_room;
    return zone;
}

// What a load does with a device's ring after the plan made with it, and why.
enum class moe_pp_ring_decision {
    KEEP_NO_RING_WEIGHT,         // no ring is published, or no weight uses it: nothing to decide
    KEEP_STAGED_OR_SOA_USE,      // (a): a weight reaches the staged claim or the SOA per-expert staging
    KEEP_NO_OTHER_DEVICE_ROUTE,  // (b): a weight's resident prompt node has no device route but the ring
    KEEP_NO_HOST_EXPERT,         // (c): the plan left no expert of the ring's weights on the host
    KEEP_NO_ROOM_RETURNS,        // (d): the WEIGHT zone would not grow without the ring
    KEEP_HELD,                   // (d): the RUNTIME zone had room to give back, but something points into the arena
    DROP,                        // all four hold: the ring dropped, the WEIGHT zone grown, one more plan
};

// The decision on a device's ring from the plan's check and the arena's answer, (a) to (d) in that order. A held arena
// is asked before the room: the ensure keeps a held arena as built, so its WEIGHT zone reads as no room, and the keep is
// the held one. Pure: the load stage and the plan dry run ask it, each with its own answer.
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
    if (zone.held) {
        return moe_pp_ring_decision::KEEP_HELD;
    }
    if (zone.weight_without_ring <= zone.weight_with_ring) {
        return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS;
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
// marks nothing, and the next pass decides afresh on its own plan. A direct planner entry has no load transaction (0)
// and marks nothing, so it never erases a staged load's mark, and asks afresh. A mark is keyed by its load's
// transaction, and every load's is new, so a mark a failed or finished load left is never read as a later load's. One
// slot per device is enough because one load transaction is open at a time: the lifecycle registry refuses a second
// begin while one is open (LOAD_BUSY, Registry::begin_outer), so a second load is refused, never interleaved, and no
// mark of an open load is overwritten.
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
