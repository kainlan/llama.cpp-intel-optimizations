//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MOE_RING_CONSUMER_HPP
#define GGML_SYCL_MOE_RING_CONSUMER_HPP

// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it. This comment is the one
// statement of the rule; the code that applies it, its tests and its docs point here.
//
// The loader sizes the ring for a model with an expert weight of a type a ring consumer runs
// (ggml_sycl_inventory_ring_has_consumer in ggml-sycl.cpp), and the ring's room comes out of the device's weight
// capacity. The rule is by use, not by model or type name. A ring is dropped only when all four of these hold:
//   (a) every weight that uses it reaches the batched executor's resident claim alone (the types
//       moe_pp_ring_uses_of_type answers with MOE_PP_RING_USE_RESIDENT alone: today NVFP4, IQ1_S, IQ1_M and IQ4_XS);
//   (b) every one has another device route for a resident prompt node (today the MMID direct route,
//       moe_mmid_direct_route_runs_type);
//   (c) the plan made with the ring leaves at least one expert of those weights on the host;
//   (d) the drop returns room to the weights: the WEIGHT room the plan's planner packs against grows once the ring is
//       gone.
// (a) to (c) are one function, moe_pp_ring_resident_check::displaces_experts, asked before any zone figure is read;
// (d) is moe_pp_ring_decide, on the arena's answer. Such a ring serves a layer only when every expert of it is planned
// on the device and stages no host expert, so where (c) holds it has traded prompt room for a decode loss on every
// token (measured: docs/backend/sycl-memory-design.md, "A resident-only ring that displaces experts is dropped where
// that returns room"). Where (d) holds too the load drops the ring, the arena is rebuilt with a smaller RUNTIME zone
// and a larger WEIGHT zone, and the device is planned once more (ggml_sycl_stage_drop_displacing_ring in
// ggml-sycl.cpp): a drop always means that.
//
// Where (d) fails, a drop would only take the batched executor from the resident layers' prompts and give the experts
// nothing, and the ring stays published as it was. (d) fails where:
//   * the plan's planner reads no WEIGHT room on the device: the device has no active arena, or the plan is a
//     multi-device one in GLOBAL cache mode, and either packs the device to its budget alone;
//   * the ring takes no room in the RUNTIME zone: another consumer sizes the zone and needs as much without it;
//   * a plan without the ring would grow another zone (the SCRATCH zone's MoE MMVQ bound is held to the room RUNTIME
//     leaves beside it), which would take the room;
//   * the give-back's rebuild falls back from one chunk to several, which carves a KV share out of the WEIGHT zone;
//   * something points into the arena, so it is not rebuilt in this load.
// The first two are moe_pp_ring_drop_can_return_room, which the load stage and the plan dry run both ask before either
// reads the WEIGHT room a drop would leave. The plan made with a kept ring stands, unless a give-back rebuilt the arena
// and the zones asked for again with the ring leave less WEIGHT room than that plan was packed against: the device is
// then planned once more with the ring (moe_pp_ring_kept_plan_again).
//
// Today the drop set is NVFP4 alone: one IQ1_S, IQ1_M or IQ4_XS weight beside NVFP4 ones keeps the ring, because the
// ring is the only device prompt route of a resident layer of those types (decode stays refused for them), and a drop
// would leave a node the executor's claim admits with no kernel to run it. A model whose experts of those types
// overflow a card therefore still pays the ring's room. A ring with any STAGED or SOA_EXPERT use, or with any weight
// that has no other device route, is sized, published and kept exactly as before.
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

    // (a) to (c), the one function for them: the ring serves resident layers only, each of which keeps a device prompt
    // route without it, and the plan made with it left an expert of those layers' weights on the host. False keeps the
    // ring, and a load and the plan dry run read no zone figure for it; true leaves the ring to (d)
    // (moe_pp_ring_decide).
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

// The room the ring takes in the RUNTIME zone: `with_ring` (moe_pp_ring_runtime_with_ring) less `without_ring`, the
// zone the device needs with no ring published; 0 where it needs no less, an unanswerable figure without the ring
// included.
inline size_t moe_pp_ring_runtime_room(size_t with_ring, size_t without_ring) {
    return without_ring < with_ring ? with_ring - without_ring : 0;
}

// (d)'s first question, the one the load stage asks before it asks the arena anything and the plan dry run asks of its
// own answer (moe_pp_ring_dry_run_answer): whether a drop can return room the plan's planner sees at all. It can only
// where that planner reads WEIGHT room on the device (`weight_room`, the room the plan made with the ring was packed
// against: ggml_sycl_ring_plan_weight_room in ggml-sycl.cpp, 0 where the planner reads no WEIGHT zone or the device has
// no active arena, either of which packs the device to its budget) and the ring takes room in the RUNTIME zone
// (`runtime_room`, moe_pp_ring_runtime_room). Where it cannot, the ring is kept.
inline bool moe_pp_ring_drop_can_return_room(size_t weight_room, size_t runtime_room) {
    return weight_room > 0 && runtime_room > 0;
}

// The figures (d) decides on: the room the weights have, read as the plan's planner reads it (the arena's WEIGHT zone
// no more than the budget the plan packed the device to, unified_cache_kv_weight_capacity), as the plan made with the
// ring found it and once the zones were asked for without the ring with the RUNTIME zone's surplus given back
// (zone_arena_ensure_options::give_back); and whether the arena had that surplus to give but something pointed into
// it, so it was kept as built. The WEIGHT room is the measure, not the RUNTIME zone: a rebuild that grows another zone
// with the room RUNTIME gives up leaves the WEIGHT zone as it was, and a rebuild that falls back from one chunk to
// several carves the tail chunk's KV share out of it. Where a drop can return no room
// (moe_pp_ring_drop_can_return_room) the arena is asked nothing, and the two WEIGHT figures are the same.
struct moe_pp_ring_zone_answer {
    size_t weight_with_ring    = 0;
    size_t weight_without_ring = 0;
    bool   held                = false;
};

// The answer the plan dry run decides on. `can_return_room` is moe_pp_ring_drop_can_return_room's answer, the question
// the load stage asks before it asks the arena: where it is false the answer reads no room, as a load's does. Otherwise
// the dry run, which rebuilds no arena, answers from `runtime_room` (moe_pp_ring_runtime_room) alone, as a give-back on
// an arena nothing points into whose WEIGHT zone takes the whole RUNTIME room, as an arena whose layout the give-back
// keeps does (its WEIGHT zone is the arena less the zones beside it, in one chunk or several), counted from 0. It
// cannot see a plan without the ring that would grow another zone, or a rebuild that changes the arena's layout, for
// either of which a load keeps the ring.
inline moe_pp_ring_zone_answer moe_pp_ring_dry_run_answer(bool can_return_room, size_t runtime_room) {
    moe_pp_ring_zone_answer zone;
    zone.weight_without_ring = can_return_room ? runtime_room : 0;
    return zone;
}

// Whether the plan made with a ring the load keeps is made once more. A give-back rebuilt the arena without the ring
// and the WEIGHT room did not grow, so the zones were asked for again with the ring put back: an ordinary rebuild,
// whose layout is the reserve's to choose. `packed_against` is the WEIGHT room the plan made with the ring was packed
// against and `weight_after` the room after that second ensure, both read as that plan's planner reads them
// (ggml_sycl_ring_plan_weight_room in ggml-sycl.cpp). A plan never stands against a WEIGHT zone smaller than the one it
// was packed for: materialization refuses a planned expert the zone cannot hold and aborts the load. Where the second
// ensure lands in more chunks than the first, the WEIGHT room is the tail chunk's KV share smaller. The same room or
// more, and the plan stands.
inline bool moe_pp_ring_kept_plan_again(size_t packed_against, size_t weight_after) {
    return weight_after < packed_against;
}

// What (d) says of a ring whose plan displaced experts (moe_pp_ring_resident_check::displaces_experts).
enum class moe_pp_ring_decision {
    KEEP_NO_ROOM_RETURNS,  // the WEIGHT room would not grow without the ring
    KEEP_HELD,             // the RUNTIME zone had room to give back, but something points into the arena
    DROP,                  // the ring dropped, the WEIGHT zone grown, one more plan
};

// (d) on the arena's answer, asked only for a ring whose plan displaced experts. A held arena is asked before the room:
// the ensure keeps a held arena as built, so its WEIGHT zone reads as no room, and the keep is the held one. Pure: the
// load stage and the plan dry run ask it, each with its own answer.
inline moe_pp_ring_decision moe_pp_ring_decide(const moe_pp_ring_zone_answer & zone) {
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
// the ring plans without it again (dropped()), so it does not size the ring's zone back in. The load stage records
// every decision it makes (record()): a pass that keeps the ring marks nothing, and the next pass decides afresh on its
// own plan. A direct planner entry has no load transaction (0) and marks nothing, so it never erases a staged load's
// mark, and asks afresh. A mark is keyed by its load's transaction, and every load's is new, so a mark a failed or
// finished load left is never read as a later load's. One slot per device is enough because one load transaction is
// open at a time: the lifecycle registry refuses a second begin while one is open (LOAD_BUSY, Registry::begin_outer),
// so a second load is refused, never interleaved, and no mark of an open load is overwritten.
struct moe_pp_ring_drop_marks {
    // Records what a pass of load `txn` decided on `device`'s ring: a drop marks the device, a keep marks nothing. The
    // load stage calls it with every decision it makes; it is the only way to mark a device.
    void record(int device, uint64_t txn, moe_pp_ring_decision decision) {
        if (decision == moe_pp_ring_decision::DROP) {
            mark(device, txn);
        }
    }

    // Whether load `txn` dropped `device`'s ring at an earlier pass. Never for txn 0.
    bool dropped(int device, uint64_t txn) const {
        return txn != 0 && device_in_range(device) && load_txn[device] == txn;
    }

  private:
    uint64_t load_txn[GGML_SYCL_MAX_DEVICES] = {};

    static bool device_in_range(int device) { return device >= 0 && device < GGML_SYCL_MAX_DEVICES; }

    // Records that load `txn` dropped `device`'s ring, for record() alone. Nothing for txn 0 or a device out of range.
    void mark(int device, uint64_t txn) {
        if (txn != 0 && device_in_range(device)) {
            load_txn[device] = txn;
        }
    }
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MOE_RING_CONSUMER_HPP
