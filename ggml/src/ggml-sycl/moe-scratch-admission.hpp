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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

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

// One-shot gate for the unconditional refusal warning.
//
// A refusal silently downgrades routing from the batched oneDNN fast path to a
// slower one, so it has to be visible at DEFAULT verbosity -- GGML_LOG_INFO is
// dropped below the default threshold, which is why the warning is WARN. But a
// misconfigured plan refuses on every dispatch of every layer of every token,
// so an unlatched WARN would bury the run in its own diagnostic. Each call site
// owns a function-local static latch: the first refusal is reported in full and
// the rest are silent, while the debug-gated per-dispatch traces still carry the
// detailed view.
class pp_moe_onednn_scratch_refusal_latch {
  public:
    pp_moe_onednn_scratch_refusal_latch()                                                        = default;
    pp_moe_onednn_scratch_refusal_latch(const pp_moe_onednn_scratch_refusal_latch &)             = delete;
    pp_moe_onednn_scratch_refusal_latch & operator=(const pp_moe_onednn_scratch_refusal_latch &) = delete;

    // True for the first caller only. Relaxed is sufficient: this orders
    // nothing but itself, and the exchange is already atomic.
    bool claim() { return !reported_.exchange(true, std::memory_order_relaxed); }

  private:
    std::atomic<bool> reported_{ false };
};

// Whether THIS refusal should be reported at WARN. The short circuit is the
// load-bearing part: an ADMITTED request must not touch the latch, or the first
// successful dispatch burns it and the first real refusal -- the one anybody
// needs to see -- goes unreported.
inline bool pp_moe_onednn_should_report_refusal(const pp_moe_onednn_scratch_admission & admission,
                                                pp_moe_onednn_scratch_refusal_latch &   latch) {
    return !admission.allowed && latch.claim();
}

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

// Where the ring for a runtime n_ubatch goes, and whether it is admitted. The
// runtime-context transaction decides this after it has admitted KV.
//
// The weight slot(s) stay in the RUNTIME zone. Each ubatch-scaled slot kind
// (activation, output; ring_depth copies of each) stays in the RUNTIME zone if
// it still fits there, larger kind first, and otherwise goes to the shared
// KV/weight zone. The KV-zone part is admitted only against what the zone has
// left once the admitted KV is placed, less a compute-buffer reserve. KV is
// admitted before this runs and nothing here moves it, so a larger micro-batch
// can never be the reason KV leaves VRAM.
//
// The reserve stands in for runtime consumers of the KV zone that no plan
// counts yet: compute buffers that miss the RUNTIME zone at graph_reserve,
// which runs after the transaction, and the flash-attention K/V conversion
// buffers. It applies only when some of the ring goes to the KV zone; a ring
// that fits the RUNTIME zone competes with none of them.
//
// Each KV-zone slot is one allocation, so free space in pieces cannot hold it.
// The whole KV-zone part must fit the zone's largest free block. That is an
// estimate: the allocator reports the head of its largest size class, not a
// scanned maximum, and with two or more KV-zone slots the later allocations
// can still miss. It covers the single-allocation case; for the rest, a failed
// reserve refuses the ring, which an automatic -ub steps down from.
struct pp_moe_onednn_ring_admission_inputs {
    size_t   kv_zone_available_bytes       = 0;  // KV bytes the device can still hold (live allocator figure)
    size_t   kv_admitted_bytes             = 0;  // KV the transaction admitted on this device
    size_t   compute_reserve_bytes_per_row = 0;  // KV-zone bytes held back per n_ubatch row
    size_t   runtime_available_bytes       = 0;  // RUNTIME zone free once the old ring is released
    size_t   kv_zone_largest_block_bytes   = std::numeric_limits<size_t>::max();  // largest free KV-zone block
    size_t   weight_slot_bytes             = 0;
    size_t   activation_bytes_per_row      = 0;
    size_t   output_bytes_per_row          = 0;
    uint32_t ring_depth                    = 0;
    uint32_t n_ubatch                      = 0;
};

struct pp_moe_onednn_ring_admission {
    bool     admit                    = false;
    bool     activation_in_kv_zone    = false;
    bool     output_in_kv_zone        = false;
    size_t   activation_slot_bytes    = 0;  // align256(n_ubatch * activation_bytes_per_row)
    size_t   output_slot_bytes        = 0;  // align256(n_ubatch * output_bytes_per_row)
    size_t   kv_zone_bytes            = 0;  // ring bytes placed in the KV zone; 0 when it all fits RUNTIME
    size_t   kv_zone_headroom_bytes   = 0;  // available - admitted, clamped at 0
    size_t   compute_reserve_bytes    = 0;  // n_ubatch * compute_reserve_bytes_per_row, saturating
    // n_ubatch when admitted; otherwise the largest multiple of 32 below it
    // that would be, or 0 when none would (the weight slot alone does not fit
    // the RUNTIME zone, or the inputs are degenerate).
    uint32_t largest_fitting_n_ubatch = 0;
};

// Pure: every input is passed in. Not admitted on degenerate input (ring_depth
// or n_ubatch 0, or both per-row terms 0) or when a slot size overflows size_t.
pp_moe_onednn_ring_admission pp_moe_onednn_admit_ring(const pp_moe_onednn_ring_admission_inputs & in);

}  // namespace ggml_sycl
