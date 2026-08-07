//
// Planner-owned admission for the PP MoE oneDNN device scratch ring.
//
// The planner publishes per-slot byte sizes and a ring depth for the PP MoE
// oneDNN staging ring, and the RUNTIME zone is sized from exactly those
// numbers. Until this module existed the executor asked the unified cache for
// `max(planned, required)` -- so a batch wider than the plan silently enlarged
// a zone that had already been budgeted, and the plan stopped being a cap at
// the first shape nobody predicted. On GPT-OSS layer 0 that is one 15.8 MiB
// FP16 expert matrix planned against a 32-expert request of 506.25 MiB.
//
// Planned dimensions are HARD CEILINGS. An over-plan request is REFUSED with a
// stable typed reason; it is never clamped, never rounded down, and never
// satisfied by growing the reservation. The caller's job on refusal is to pick
// an already-planned route, which is why every reason has a name that reaches
// a log.
//
// This header is deliberately dependency-free, for the same reason
// moe-control-plan.hpp and zone-sizing.hpp are: the policy is pure, so it can
// be unit-tested on a host with no GPU, no SYCL runtime and no backend library
// (see tests/test-moe-scratch-admission.cpp). unified-cache.hpp includes it so
// the backend and the executor share one definition of the cap.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// Why a scratch request was refused. The names are part of the contract: they
// are what `reject_batched` reports and what the source-contract gate keys off,
// so a refusal is always attributable to one predicate rather than to "the
// batch did not run".
enum class pp_moe_onednn_scratch_admission_reason : uint8_t {
    ALLOWED = 0,
    MISSING_PLAN,     // the planner published nothing for this device
    INVALID_REQUEST,  // a zero dimension, or a size that cannot be aligned
    WEIGHT_CAP,
    ACTIVATION_CAP,
    OUTPUT_CAP,
    RING_CAP,
};

struct pp_moe_onednn_scratch_shape {
    size_t   weight_slot_bytes     = 0;
    size_t   activation_slot_bytes = 0;
    size_t   output_slot_bytes     = 0;
    uint32_t ring_depth            = 0;
};

// `allowed == false` defaults to MISSING_PLAN rather than ALLOWED so a caller
// that forgets to run the preflight and reads the struct anyway gets a refusal,
// not an admission.
struct pp_moe_onednn_scratch_admission {
    bool                                   allowed = false;
    pp_moe_onednn_scratch_admission_reason reason  = pp_moe_onednn_scratch_admission_reason::MISSING_PLAN;
};

// Slot sizes are rounded up to this granularity before the ring is built, so it
// is also the unit the cap is expressed in: a request that overruns the plan by
// one byte overruns it by one slot. `reserve_pp_moe_onednn_scratch` allocates
// the shape this module hands back, so the two cannot drift apart.
static constexpr size_t PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT = 256;

// Round up to the slot alignment with the addition checked. Returns false and
// leaves *out untouched on overflow, so a caller that ignores the result cannot
// pick up a wrapped -- and therefore admissible-looking -- size.
bool pp_moe_onednn_checked_align_slot_bytes(size_t value, size_t * out);

// Aligns every byte dimension of `shape` into *out. Returns false on overflow.
bool pp_moe_onednn_align_scratch_shape(const pp_moe_onednn_scratch_shape & shape, pp_moe_onednn_scratch_shape * out);

// The full policy. `aligned_required` receives the shape that must actually be
// allocated when the answer is ALLOWED; it is untouched otherwise.
//
// BOTH shapes are aligned before they are compared, and the aligned PLAN is the
// ceiling. That is a deliberate correction to the branch this was salvaged
// from, which compared the aligned request against the RAW plan: because the
// ring physically allocates aligned slots, an unaligned planned size made
// `reserve_pp_moe_onednn_scratch` refuse its own plan -- the cap rejecting the
// shape it exists to authorise, which fails the whole PP MoE oneDNN path
// closed for a rounding artifact. `admit(planned, planned)` is ALLOWED for
// every non-degenerate plan, aligned or not, and the host test pins that.
pp_moe_onednn_scratch_admission pp_moe_onednn_preflight_scratch(const pp_moe_onednn_scratch_shape & planned,
                                                                const pp_moe_onednn_scratch_shape & required,
                                                                pp_moe_onednn_scratch_shape *       aligned_required);

// The same policy for callers that only need the verdict.
pp_moe_onednn_scratch_admission pp_moe_onednn_admit_scratch(const pp_moe_onednn_scratch_shape & planned,
                                                            const pp_moe_onednn_scratch_shape & required);

const char * pp_moe_onednn_scratch_admission_reason_name(pp_moe_onednn_scratch_admission_reason reason);

// BEHAVIORAL CONTROL SEAM -- TEST-ONLY BY DESIGN. DO NOT DELETE AS DEAD CODE.
//
// Production runs the preflight straight-line at the top of
// unified_cache::reserve_pp_moe_onednn_scratch and at both executor call sites,
// which is what tests/test-sycl-pp-moe-scratch-admission-contract.py gates by
// source order. Source order proves where the call sits; it cannot prove that a
// REFUSAL reaches no lock and no allocator. This seam is how that is proven
// behaviorally: `on_admitted` stands in for every side effect the real
// reservation performs, and a refused shape must leave its counter at zero.
template <typename OnAdmitted>
inline bool pp_moe_onednn_reserve_if_admitted(const pp_moe_onednn_scratch_shape & planned,
                                              const pp_moe_onednn_scratch_shape & requested,
                                              pp_moe_onednn_scratch_admission &   admission,
                                              OnAdmitted &&                       on_admitted) {
    pp_moe_onednn_scratch_shape aligned_required;
    admission = pp_moe_onednn_preflight_scratch(planned, requested, &aligned_required);
    if (!admission.allowed) {
        return false;
    }
    return on_admitted(aligned_required);
}

}  // namespace ggml_sycl
