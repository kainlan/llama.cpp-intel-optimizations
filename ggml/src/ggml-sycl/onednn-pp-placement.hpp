#pragma once

#include <cstdint>

// Placement answers for oneDNN prompt processing (PP): which devices may run
// oneDNN PP, and whether the plan may add dense oneDNN WOQ second copies.
//
// SYCL-free and header-only, like layer-streaming-gate.hpp. The decision is
// pure logic over plain values the caller adapts from the placement plan, so
// test-onednn-pp-placement can exercise it without a device.
//
// There are two questions here, and they used to share one boolean
// (llama.cpp-1d0n):
//
//   1. May the plan charge and stage a dense ONEDNN_WOQ alternate (a second
//      copy of the weight)? On a multi-device split, no: the copies do not
//      fit the budget that produced the split.
//   2. May a device run oneDNN PP for a weight? Placement decides the
//      executor, so a device may do it for any weight resident on it. It
//      dequantizes the resident layout into planned scratch, which is layout
//      work and not weight streaming.
//
// A dense two-card split answered "no" to both. The B70's 30 layers then ran
// mmq_generic, at 9.2x the matmul time of oneDNN, which is why pp512 was ~790
// against ~3315 on the B70 alone.
//
// Before question 2, ggml_sycl_onednn_pp_candidate asks whether the op
// is admitted at all (onednn_pp_admission_decide below). Every oneDNN PP arm
// asks the candidate; none re-derives a partial answer (llama.cpp-je3b).

namespace ggml_sycl {

enum class onednn_pp_placement {
    ALLOWED,                 // single routable device, or a plan with no secondary devices
    REFUSED_MOE_MULTI_GPU,   // MoE multi-GPU is active on more than one routable device
    SPLIT_RESIDENT_WEIGHTS,  // dense multi-device split: only for weights resident on the executing device
};

struct onednn_pp_placement_inputs {
    bool multiple_routable_devices = false;
    bool moe_multi_gpu_active      = false;
    bool plan_needs_secondaries    = false;  // plan->multi_device && it places work on a secondary device
};

inline onednn_pp_placement onednn_pp_placement_decide(const onednn_pp_placement_inputs & in) {
    if (!in.multiple_routable_devices) {
        return onednn_pp_placement::ALLOWED;
    }
    if (in.moe_multi_gpu_active) {
        return onednn_pp_placement::REFUSED_MOE_MULTI_GPU;
    }
    if (in.plan_needs_secondaries) {
        return onednn_pp_placement::SPLIT_RESIDENT_WEIGHTS;
    }
    return onednn_pp_placement::ALLOWED;
}

// Question 1: may the plan add a dense ONEDNN_WOQ second copy?
inline bool onednn_pp_woq_alternates_allowed(onednn_pp_placement p) {
    return p == onednn_pp_placement::ALLOWED;
}

// Question 2: may the executing device run oneDNN PP for this weight?
// weight_resident_on_device is only consulted for a split.
inline bool onednn_pp_executable(onednn_pp_placement p, bool weight_resident_on_device) {
    switch (p) {
        case onednn_pp_placement::ALLOWED:
            return true;
        case onednn_pp_placement::REFUSED_MOE_MULTI_GPU:
            return false;
        case onednn_pp_placement::SPLIT_RESIDENT_WEIGHTS:
            return weight_resident_on_device;
    }
    return false;
}

// Which oneDNN PP arm is asking (llama.cpp-je3b). Every arm gets the same
// admission checks from ggml_sycl_onednn_pp_candidate; only the batch floor
// differs, because a floor is the break-even against the arm's fallback:
//   DENSE_DISPATCH -- the unified dispatcher, which falls back to MMQ and
//     takes GGML_SYCL_ONEDNN_PP_MIN_BATCH (default 16).
//   MXFP4_DIRECT   -- the MXFP4 per-expert direct dispatch in
//     ggml_sycl_mul_mat, which falls back to one MMVQ launch per activation
//     row (SOA) or the unified kernel (AOS), not MMQ. Its arms have admitted
//     every non-decode batch (M >= 2) since they were written, so that floor
//     is kept here rather than moved to 16 without a measurement.
enum class onednn_pp_route {
    DENSE_DISPATCH,
    MXFP4_DIRECT,
};

// The MXFP4_DIRECT floor: every batch that is not decode.
constexpr int64_t onednn_pp_mxfp4_direct_min_batch = 2;

inline int64_t onednn_pp_min_batch_for(onednn_pp_route route, int64_t dense_min_batch) {
    switch (route) {
        case onednn_pp_route::DENSE_DISPATCH:
            return dense_min_batch;
        case onednn_pp_route::MXFP4_DIRECT:
            return onednn_pp_mxfp4_direct_min_batch;
    }
    return dense_min_batch;
}

// Why an op may not take oneDNN PP, before placement is asked. The names are
// the [ONEDNN-PP-TRACE] candidate= reasons.
enum class onednn_pp_refusal {
    NONE,
    DISABLED_OR_SKIP_TYPE,  // GGML_SYCL_ONEDNN_PP=0, or GGML_SYCL_SKIP_ONEDNN_Q4_0 for a Q4_0 weight
    BATCH_UNDER_THRESHOLD,  // batch below the route's floor
    OPERAND_TYPE,           // activations or destination not F32
    NOT_CONTIGUOUS_QUANT,   // weight not quantized, or not contiguous
};

struct onednn_pp_admission_inputs {
    bool    enabled                     = false;
    bool    skip_type                   = false;
    int64_t batch                       = 0;  // src1->ne[1]
    int64_t min_batch                   = 0;  // onednn_pp_min_batch_for(route, ...)
    bool    f32_operands                = false;
    bool    contiguous_quantized_weight = false;
};

inline onednn_pp_refusal onednn_pp_admission_decide(const onednn_pp_admission_inputs & in) {
    if (!in.enabled || in.skip_type) {
        return onednn_pp_refusal::DISABLED_OR_SKIP_TYPE;
    }
    if (in.batch < in.min_batch) {
        return onednn_pp_refusal::BATCH_UNDER_THRESHOLD;
    }
    if (!in.f32_operands) {
        return onednn_pp_refusal::OPERAND_TYPE;
    }
    if (!in.contiguous_quantized_weight) {
        return onednn_pp_refusal::NOT_CONTIGUOUS_QUANT;
    }
    return onednn_pp_refusal::NONE;
}

inline const char * onednn_pp_refusal_name(onednn_pp_refusal r) {
    switch (r) {
        case onednn_pp_refusal::NONE:
            return "none";
        case onednn_pp_refusal::DISABLED_OR_SKIP_TYPE:
            return "disabled-or-skip-type";
        case onednn_pp_refusal::BATCH_UNDER_THRESHOLD:
            return "batch-under-threshold";
        case onednn_pp_refusal::OPERAND_TYPE:
            return "type";
        case onednn_pp_refusal::NOT_CONTIGUOUS_QUANT:
            return "not-contiguous-quant";
    }
    return "unknown";
}

inline const char * onednn_pp_placement_name(onednn_pp_placement p) {
    switch (p) {
        case onednn_pp_placement::ALLOWED:
            return "allowed";
        case onednn_pp_placement::REFUSED_MOE_MULTI_GPU:
            return "refused-moe-multi-gpu";
        case onednn_pp_placement::SPLIT_RESIDENT_WEIGHTS:
            return "split-resident-weights";
    }
    return "unknown";
}

}  // namespace ggml_sycl
