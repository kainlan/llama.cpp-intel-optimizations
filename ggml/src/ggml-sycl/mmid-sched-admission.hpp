// Scratch admission for the device-built MUL_MAT_ID schedule (upstream 18c5a95e2, fused-gemm.cpp).
//
// The schedule writes four tables per call: the routed-row map, the tile table, the per-expert row offsets, and the
// src1 rows packed for the XMX B operand. They are scratch, so they come from the planner (owner decision 2026-11-21):
// the planner names a ceiling for each table on the submitting device, a request above any ceiling is refused with a
// typed reason, and the caller takes the MoE route it already planned. Nothing here allocates; the caller hands the
// admitted memory to ggml_sycl_grouped_dequant_gemm_device().
//
// Pure: every input is passed in, so a host test can drive it without a device.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

constexpr size_t MMID_SCHED_SCRATCH_ALIGNMENT = 256;

// Bytes of each table. Each table is one allocation of at least this many bytes.
struct mmid_sched_scratch_shape {
    size_t row_mapping_bytes   = 0;
    size_t tile_bytes          = 0;
    size_t expert_offset_bytes = 0;
    size_t packed_b_bytes      = 0;
};

enum class mmid_sched_scratch_reason : uint8_t {
    ADMITTED = 0,
    MISSING_PLAN,       // the planner names no ceiling for these tables on this device
    INVALID_REQUEST,    // a required size is 0, or does not round up to the alignment
    ROW_MAPPING_CAP,    // the routed-row map exceeds its ceiling
    TILE_CAP,           // the tile table exceeds its ceiling
    EXPERT_OFFSET_CAP,  // the per-expert offsets exceed their ceiling
    PACKED_B_CAP,       // the packed B operand exceeds its ceiling
    COUNT,
};

struct mmid_sched_scratch_admission {
    bool                      allowed = false;
    mmid_sched_scratch_reason reason  = mmid_sched_scratch_reason::MISSING_PLAN;
    mmid_sched_scratch_shape  required;  // aligned to MMID_SCHED_SCRATCH_ALIGNMENT
    mmid_sched_scratch_shape  planned;   // all 0 when there is no plan
};

inline bool mmid_sched_scratch_align(size_t bytes, size_t * out) noexcept {
    if (bytes == 0 || bytes > SIZE_MAX - (MMID_SCHED_SCRATCH_ALIGNMENT - 1)) {
        return false;
    }
    *out = (bytes + MMID_SCHED_SCRATCH_ALIGNMENT - 1) / MMID_SCHED_SCRATCH_ALIGNMENT * MMID_SCHED_SCRATCH_ALIGNMENT;
    return true;
}

// `planned` is the planner's ceiling for the submitting device, or nullptr when it has none. The planned sizes are hard
// ceilings: there is no growth path, and a refusal touches no allocator.
inline mmid_sched_scratch_admission mmid_sched_preflight_scratch(const mmid_sched_scratch_shape * planned,
                                                                 const mmid_sched_scratch_shape & required) noexcept {
    mmid_sched_scratch_admission admission;
    if (!mmid_sched_scratch_align(required.row_mapping_bytes, &admission.required.row_mapping_bytes) ||
        !mmid_sched_scratch_align(required.tile_bytes, &admission.required.tile_bytes) ||
        !mmid_sched_scratch_align(required.expert_offset_bytes, &admission.required.expert_offset_bytes) ||
        !mmid_sched_scratch_align(required.packed_b_bytes, &admission.required.packed_b_bytes)) {
        admission.reason = mmid_sched_scratch_reason::INVALID_REQUEST;
        return admission;
    }
    if (planned == nullptr) {
        admission.reason = mmid_sched_scratch_reason::MISSING_PLAN;
        return admission;
    }
    admission.planned = *planned;
    if (admission.required.row_mapping_bytes > planned->row_mapping_bytes) {
        admission.reason = mmid_sched_scratch_reason::ROW_MAPPING_CAP;
    } else if (admission.required.tile_bytes > planned->tile_bytes) {
        admission.reason = mmid_sched_scratch_reason::TILE_CAP;
    } else if (admission.required.expert_offset_bytes > planned->expert_offset_bytes) {
        admission.reason = mmid_sched_scratch_reason::EXPERT_OFFSET_CAP;
    } else if (admission.required.packed_b_bytes > planned->packed_b_bytes) {
        admission.reason = mmid_sched_scratch_reason::PACKED_B_CAP;
    } else {
        admission.allowed = true;
        admission.reason  = mmid_sched_scratch_reason::ADMITTED;
    }
    return admission;
}

inline const char * mmid_sched_scratch_reason_name(mmid_sched_scratch_reason reason) noexcept {
    switch (reason) {
        case mmid_sched_scratch_reason::ADMITTED:
            return "admitted";
        case mmid_sched_scratch_reason::MISSING_PLAN:
            return "missing-plan";
        case mmid_sched_scratch_reason::INVALID_REQUEST:
            return "invalid-request";
        case mmid_sched_scratch_reason::ROW_MAPPING_CAP:
            return "row-mapping-cap";
        case mmid_sched_scratch_reason::TILE_CAP:
            return "tile-cap";
        case mmid_sched_scratch_reason::EXPERT_OFFSET_CAP:
            return "expert-offset-cap";
        case mmid_sched_scratch_reason::PACKED_B_CAP:
            return "packed-b-cap";
        case mmid_sched_scratch_reason::COUNT:
            break;
    }
    return "unknown";
}

// Reports a refusal at WARN the first time each reason occurs in the process (mmid-sched.cpp).
void mmid_sched_warn_refusal_once(const mmid_sched_scratch_admission & admission) noexcept;

}  // namespace ggml_sycl
