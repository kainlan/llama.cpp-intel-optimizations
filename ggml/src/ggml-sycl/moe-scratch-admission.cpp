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

}  // namespace ggml_sycl
