//
// Planner-owned admission for the PP MoE oneDNN device scratch ring.
// See moe-scratch-admission.hpp for the contract and for why this unit carries
// no SYCL or backend dependency.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "moe-scratch-admission.hpp"

#include "zone-sizing.hpp"

#include <limits>

namespace ggml_sycl {

bool pp_moe_onednn_checked_align_slot_bytes(size_t value, size_t * out) {
    constexpr size_t alignment = PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    if (out != nullptr) {
        *out = (value + alignment - 1) / alignment * alignment;
    }
    return true;
}

bool pp_moe_onednn_align_scratch_shape(const pp_moe_onednn_scratch_shape & shape, pp_moe_onednn_scratch_shape * out) {
    pp_moe_onednn_scratch_shape aligned = shape;
    if (!pp_moe_onednn_checked_align_slot_bytes(shape.weight_slot_bytes, &aligned.weight_slot_bytes) ||
        !pp_moe_onednn_checked_align_slot_bytes(shape.activation_slot_bytes, &aligned.activation_slot_bytes) ||
        !pp_moe_onednn_checked_align_slot_bytes(shape.output_slot_bytes, &aligned.output_slot_bytes)) {
        return false;
    }
    if (out != nullptr) {
        *out = aligned;
    }
    return true;
}

pp_moe_onednn_scratch_admission pp_moe_onednn_preflight_scratch(const pp_moe_onednn_scratch_shape & planned,
                                                                const pp_moe_onednn_scratch_shape & required,
                                                                pp_moe_onednn_scratch_shape *       aligned_required) {
    // A zero anywhere in the plan means the planner published nothing for this
    // device. That is distinct from a plan that is merely too small: nothing
    // was budgeted, so there is no ceiling to compare against and no amount of
    // shrinking the request would make it admissible.
    if (planned.weight_slot_bytes == 0 || planned.activation_slot_bytes == 0 || planned.output_slot_bytes == 0 ||
        planned.ring_depth == 0) {
        return { false, pp_moe_onednn_scratch_admission_reason::MISSING_PLAN };
    }
    if (required.weight_slot_bytes == 0 || required.activation_slot_bytes == 0 || required.output_slot_bytes == 0 ||
        required.ring_depth == 0) {
        return { false, pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST };
    }

    pp_moe_onednn_scratch_shape aligned = required;
    if (!pp_moe_onednn_align_scratch_shape(required, &aligned)) {
        return { false, pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST };
    }

    // The request is checked before the plan so a caller asking for an
    // unrepresentable size is told that, rather than being told its device was
    // never planned for.
    pp_moe_onednn_scratch_shape cap = planned;
    if (!pp_moe_onednn_align_scratch_shape(planned, &cap)) {
        return { false, pp_moe_onednn_scratch_admission_reason::MISSING_PLAN };
    }

    if (aligned.weight_slot_bytes > cap.weight_slot_bytes) {
        return { false, pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP };
    }
    if (aligned.activation_slot_bytes > cap.activation_slot_bytes) {
        return { false, pp_moe_onednn_scratch_admission_reason::ACTIVATION_CAP };
    }
    if (aligned.output_slot_bytes > cap.output_slot_bytes) {
        return { false, pp_moe_onednn_scratch_admission_reason::OUTPUT_CAP };
    }
    if (aligned.ring_depth > cap.ring_depth) {
        return { false, pp_moe_onednn_scratch_admission_reason::RING_CAP };
    }

    if (aligned_required != nullptr) {
        *aligned_required = aligned;
    }
    return { true, pp_moe_onednn_scratch_admission_reason::ALLOWED };
}

pp_moe_onednn_scratch_admission pp_moe_onednn_admit_scratch(const pp_moe_onednn_scratch_shape & planned,
                                                            const pp_moe_onednn_scratch_shape & required) {
    pp_moe_onednn_scratch_shape aligned_required;
    return pp_moe_onednn_preflight_scratch(planned, required, &aligned_required);
}

const char * pp_moe_onednn_scratch_admission_reason_name(pp_moe_onednn_scratch_admission_reason reason) {
    switch (reason) {
        case pp_moe_onednn_scratch_admission_reason::ALLOWED:
            return "allowed";
        case pp_moe_onednn_scratch_admission_reason::MISSING_PLAN:
            return "missing-plan";
        case pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST:
            return "invalid-request";
        case pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP:
            return "weight-cap";
        case pp_moe_onednn_scratch_admission_reason::ACTIVATION_CAP:
            return "activation-cap";
        case pp_moe_onednn_scratch_admission_reason::OUTPUT_CAP:
            return "output-cap";
        case pp_moe_onednn_scratch_admission_reason::RING_CAP:
            return "ring-cap";
    }
    return "unknown";
}

// Places and admits the ring at `n_ubatch` into *out. Every multiply and add is
// checked, so an overflowing slot is refused rather than wrapped into a small
// one that would be admitted.
static bool pp_moe_onednn_admit_ring_at(const pp_moe_onednn_ring_admission_inputs & in,
                                        uint32_t                                    n_ubatch,
                                        pp_moe_onednn_ring_admission *              out) {
    constexpr size_t kMax  = std::numeric_limits<size_t>::max();
    const size_t     depth = in.ring_depth;
    const size_t     rows  = n_ubatch;
    out->compute_reserve_bytes =
        in.compute_reserve_bytes_per_row > kMax / rows ? kMax : rows * in.compute_reserve_bytes_per_row;
    if (in.activation_bytes_per_row > kMax / rows || in.output_bytes_per_row > kMax / rows) {
        return false;
    }
    size_t act_slot = 0;
    size_t out_slot = 0;
    if (!pp_moe_onednn_checked_align_slot_bytes(rows * in.activation_bytes_per_row, &act_slot) ||
        !pp_moe_onednn_checked_align_slot_bytes(rows * in.output_bytes_per_row, &out_slot)) {
        return false;
    }
    out->activation_slot_bytes = act_slot;
    out->output_slot_bytes     = out_slot;
    if (in.weight_slot_bytes > kMax / depth || act_slot > kMax / depth || out_slot > kMax / depth) {
        return false;
    }
    const size_t weight_total = in.weight_slot_bytes * depth;
    const size_t act_total    = act_slot * depth;
    const size_t out_total    = out_slot * depth;

    if (weight_total > in.runtime_available_bytes) {
        return false;
    }

    // Larger kind first: with two kinds this keeps the most bytes in the
    // RUNTIME zone, so the least lands in the KV zone.
    size_t       runtime_left = in.runtime_available_bytes - weight_total;
    const bool   out_first    = out_total >= act_total;
    const size_t first_total  = out_first ? out_total : act_total;
    const size_t second_total = out_first ? act_total : out_total;
    const bool   first_in_kv  = first_total > runtime_left;
    if (!first_in_kv) {
        runtime_left -= first_total;
    }
    const bool second_in_kv = second_total > runtime_left;

    out->output_in_kv_zone     = out_first ? first_in_kv : second_in_kv;
    out->activation_in_kv_zone = out_first ? second_in_kv : first_in_kv;

    size_t kv_zone_bytes = first_in_kv ? first_total : 0;
    if (second_in_kv) {
        if (second_total > kMax - kv_zone_bytes) {
            return false;
        }
        kv_zone_bytes += second_total;
    }
    out->kv_zone_bytes = kv_zone_bytes;
    if (kv_zone_bytes == 0) {
        return true;
    }
    // The largest-block term is an estimate (see the header); a failed reserve is the backstop.
    return out->compute_reserve_bytes <= out->kv_zone_headroom_bytes &&
           kv_zone_bytes <= out->kv_zone_headroom_bytes - out->compute_reserve_bytes &&
           kv_zone_bytes <= in.kv_zone_largest_block_bytes && kv_zone_bytes <= in.vram_budget_room_bytes;
}

pp_moe_onednn_ring_admission pp_moe_onednn_admit_ring(const pp_moe_onednn_ring_admission_inputs & in) {
    pp_moe_onednn_ring_admission result;
    result.kv_zone_headroom_bytes =
        in.kv_zone_available_bytes > in.kv_admitted_bytes ? in.kv_zone_available_bytes - in.kv_admitted_bytes : 0;
    if (in.ring_depth == 0 || in.n_ubatch == 0 || (in.activation_bytes_per_row == 0 && in.output_bytes_per_row == 0)) {
        return result;
    }
    if (pp_moe_onednn_admit_ring_at(in, in.n_ubatch, &result)) {
        result.admit                    = true;
        result.largest_fitting_n_ubatch = in.n_ubatch;
        return result;
    }

    // Binary search over multiples of 32 below n_ubatch. Admission is monotone
    // in n_ubatch: every slot and the reserve only grow, so the placements that
    // fit only shrink and the bytes left for the KV zone only grow.
    uint32_t lo = 0;                           // count of 32-row steps known to fit (0: none)
    uint32_t hi = (in.n_ubatch - 1) / 32 + 1;  // count known not to fit
    while (hi - lo > 1) {
        const uint32_t               mid = lo + (hi - lo) / 2;
        pp_moe_onednn_ring_admission probe;
        probe.kv_zone_headroom_bytes = result.kv_zone_headroom_bytes;
        if (pp_moe_onednn_admit_ring_at(in, mid * 32, &probe)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    result.largest_fitting_n_ubatch = lo * 32;
    return result;
}

// The ring at `n_ubatch`, sized as pp_moe_onednn_admit_ring_at() sizes it. False on overflow.
static bool pp_moe_onednn_ring_bytes_at(const pp_moe_onednn_explicit_ubatch_inputs & in,
                                        uint32_t                                     n_ubatch,
                                        size_t *                                     out) {
    constexpr size_t kMax  = std::numeric_limits<size_t>::max();
    const size_t     depth = in.ring_depth;
    const size_t     rows  = n_ubatch;
    if (in.activation_bytes_per_row > kMax / rows || in.output_bytes_per_row > kMax / rows) {
        return false;
    }
    size_t act_slot = 0;
    size_t out_slot = 0;
    if (!pp_moe_onednn_checked_align_slot_bytes(rows * in.activation_bytes_per_row, &act_slot) ||
        !pp_moe_onednn_checked_align_slot_bytes(rows * in.output_bytes_per_row, &out_slot)) {
        return false;
    }
    if (in.weight_slot_bytes > kMax - act_slot || in.weight_slot_bytes + act_slot > kMax - out_slot) {
        return false;
    }
    const size_t per_ring = in.weight_slot_bytes + act_slot + out_slot;
    if (per_ring > kMax / depth) {
        return false;
    }
    *out = per_ring * depth;
    return true;
}

// The RUNTIME zone for a requirement of fixed + dense + ring + compute growth, as the arena reserves it: at least the
// floor, rounded up to 256 bytes. False on overflow.
static bool pp_moe_onednn_runtime_zone_bytes(const pp_moe_onednn_explicit_ubatch_inputs & in,
                                             size_t                                       dense_bytes,
                                             size_t                                       ring_bytes,
                                             size_t                                       compute_growth_bytes,
                                             size_t *                                     out) {
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    if (in.runtime_fixed_bytes > kMax - dense_bytes || in.runtime_fixed_bytes + dense_bytes > kMax - ring_bytes ||
        in.runtime_fixed_bytes + dense_bytes + ring_bytes > kMax - compute_growth_bytes) {
        return false;
    }
    const size_t requirement = in.runtime_fixed_bytes + dense_bytes + ring_bytes + compute_growth_bytes;
    return pp_moe_onednn_checked_align_slot_bytes(
        requirement > in.runtime_floor_bytes ? requirement : in.runtime_floor_bytes, out);
}

size_t pp_moe_onednn_explicit_ubatch_compute_growth(size_t   compute_default_bytes,
                                                    uint32_t n_ubatch_default,
                                                    uint32_t n_ubatch) {
    if (n_ubatch_default == 0 || n_ubatch <= n_ubatch_default) {
        return 0;
    }
    const size_t scaled = zone_hold_scale_request(compute_default_bytes, n_ubatch_default, n_ubatch);
    return scaled == std::numeric_limits<size_t>::max() ? scaled : scaled - compute_default_bytes;
}

pp_moe_onednn_explicit_ubatch_room pp_moe_onednn_explicit_ubatch_plan_room(
    const pp_moe_onednn_explicit_ubatch_inputs & in) {
    constexpr size_t                   kMax = std::numeric_limits<size_t>::max();
    pp_moe_onednn_explicit_ubatch_room room;
    room.in = in;
    if (in.n_ubatch_default == 0 || in.n_ubatch <= in.n_ubatch_default || in.weight_slot_bytes == 0 ||
        in.ring_depth == 0 || (in.activation_bytes_per_row == 0 && in.output_bytes_per_row == 0)) {
        return room;
    }
    room.valid = true;

    // The default micro-batch's ring and zone were already reserved once, so they cannot overflow; if they do anyway,
    // nothing can be planned against them.
    size_t ring_default = 0;
    if (!pp_moe_onednn_ring_bytes_at(in, in.n_ubatch_default, &ring_default) ||
        !pp_moe_onednn_runtime_zone_bytes(in, in.dense_scratch_default_bytes, ring_default, 0,
                                          &room.zone_default_bytes)) {
        room.valid = false;
        return room;
    }
    room.compute_growth_bytes =
        pp_moe_onednn_explicit_ubatch_compute_growth(in.compute_default_bytes, in.n_ubatch_default, in.n_ubatch);
    room.compute_bytes = room.compute_growth_bytes > kMax - in.compute_default_bytes ?
                             kMax :
                             in.compute_default_bytes + room.compute_growth_bytes;
    // An overflowing ring or compute growth for the explicit micro-batch is a room no budget holds, not a wrapped small
    // one.
    if (room.compute_growth_bytes == kMax || !pp_moe_onednn_ring_bytes_at(in, in.n_ubatch, &room.ring_bytes) ||
        !pp_moe_onednn_runtime_zone_bytes(in, in.dense_scratch_bytes, room.ring_bytes, room.compute_growth_bytes,
                                          &room.zone_bytes)) {
        room.ring_bytes = room.ring_bytes == 0 ? kMax : room.ring_bytes;
        room.zone_bytes = kMax;
        room.room_bytes = kMax;
    } else {
        room.room_bytes = room.zone_bytes > room.zone_default_bytes ? room.zone_bytes - room.zone_default_bytes : 0;
    }
    const size_t grown =
        in.runtime_zone_bytes > room.zone_default_bytes ? in.runtime_zone_bytes - room.zone_default_bytes : 0;
    room.room_reserved_bytes = grown < room.room_bytes ? grown : room.room_bytes;
    return room;
}

pp_moe_onednn_explicit_ubatch_decision pp_moe_onednn_explicit_ubatch_decide(
    const pp_moe_onednn_explicit_ubatch_room & room,
    size_t                                     expert_budget_bytes) {
    constexpr size_t                       kMax = std::numeric_limits<size_t>::max();
    pp_moe_onednn_explicit_ubatch_decision decision;
    if (!room.valid || room.room_bytes == 0) {
        return decision;
    }
    if (room.room_reserved_bytes >= room.room_bytes) {
        decision.action = pp_moe_onednn_explicit_ubatch_action::PLACE;
        return decision;
    }
    if (room.room_bytes - room.room_reserved_bytes <= expert_budget_bytes) {
        decision.action = pp_moe_onednn_explicit_ubatch_action::GROW_ZONE;
        return decision;
    }
    decision.action                                 = pp_moe_onednn_explicit_ubatch_action::REFUSE;
    // With every device-resident routed expert on the host tier the zone could be as large as it is now plus their
    // bytes; the fixed rest, the dense scratch at the explicit micro-batch and the compute growth at the micro-batch
    // asked about still take their share of it.
    const pp_moe_onednn_explicit_ubatch_inputs & in = room.in;
    const size_t                                 zone =
        in.runtime_zone_bytes > kMax - expert_budget_bytes ? kMax : in.runtime_zone_bytes + expert_budget_bytes;
    const size_t fixed =
        in.runtime_fixed_bytes > kMax - in.dense_scratch_bytes ? kMax : in.runtime_fixed_bytes + in.dense_scratch_bytes;
    auto ring_room_at = [&](uint32_t n_ubatch) -> size_t {
        const size_t growth =
            pp_moe_onednn_explicit_ubatch_compute_growth(in.compute_default_bytes, in.n_ubatch_default, n_ubatch);
        const size_t taken = growth > kMax - fixed ? kMax : fixed + growth;
        return zone > taken ? zone - taken : 0;
    };
    decision.max_ring_runtime_bytes = ring_room_at(in.n_ubatch);
    // The largest micro-batch whose whole ring fits beside its own compute growth, by the ring admission's own slot
    // arithmetic: no KV zone part is offered, so a ring that does not fit them entirely is not admitted. The growth
    // shrinks with the micro-batch, so each candidate is asked with its own room; the ring and the growth only grow
    // with it, so admission is monotone and a binary search over multiples of 32 below in.n_ubatch finds the largest.
    pp_moe_onednn_ring_admission_inputs fit;
    fit.weight_slot_bytes        = in.weight_slot_bytes;
    fit.activation_bytes_per_row = in.activation_bytes_per_row;
    fit.output_bytes_per_row     = in.output_bytes_per_row;
    fit.ring_depth               = in.ring_depth;
    uint32_t lo                  = 0;                           // count of 32-row steps known to fit (0: none)
    uint32_t hi                  = (in.n_ubatch - 1) / 32 + 1;  // count known not to fit
    while (hi - lo > 1) {
        const uint32_t mid          = lo + (hi - lo) / 2;
        fit.n_ubatch                = mid * 32;
        fit.runtime_available_bytes = ring_room_at(fit.n_ubatch);
        if (pp_moe_onednn_admit_ring(fit).admit) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    decision.largest_fitting_n_ubatch = lo * 32;
    return decision;
}

const char * pp_moe_onednn_explicit_ubatch_action_name(pp_moe_onednn_explicit_ubatch_action action) {
    switch (action) {
        case pp_moe_onednn_explicit_ubatch_action::NONE:
            return "none";
        case pp_moe_onednn_explicit_ubatch_action::PLACE:
            return "place";
        case pp_moe_onednn_explicit_ubatch_action::GROW_ZONE:
            return "grow-zone";
        case pp_moe_onednn_explicit_ubatch_action::REFUSE:
            return "refuse";
    }
    return "unknown";
}

}  // namespace ggml_sycl
