#pragma once

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
