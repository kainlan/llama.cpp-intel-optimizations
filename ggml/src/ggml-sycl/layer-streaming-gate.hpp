#pragma once

// Auto-activation gate for double-buffered layer streaming
// (layer_stream_manager, layer-streaming.hpp).
//
// Deliberately SYCL-free and header-only, like kv-runtime-demotion.hpp: the
// decision is pure logic over plain values the caller adapts from the
// placement plan, so test-layer-streaming-gate can exercise it host-only.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// How the plan placed the model's per-layer tensors, counted over the tensor
// inventory. A tensor with no layer id (embeddings, output) is not counted.
struct plan_layer_residency {
    size_t layer_tensors      = 0;  // inventory tensors that belong to a layer
    size_t unresident_tensors = 0;  // of those: layer on the host, or absent from the plan
};

// layer_ids: extract_layer_id() of each inventory tensor (-1 = no layer).
// layer_device: placement_plan::layer_device (layer -> device id, <0 = host).
inline plan_layer_residency plan_layer_residency_count(const std::vector<int> &             layer_ids,
                                                       const std::unordered_map<int, int> & layer_device) {
    plan_layer_residency r;
    for (const int layer_id : layer_ids) {
        if (layer_id < 0) {
            continue;
        }
        r.layer_tensors++;
        const auto it = layer_device.find(layer_id);
        if (it == layer_device.end() || it->second < 0) {
            r.unresident_tensors++;
        }
    }
    return r;
}

enum class layer_streaming_gate {
    OFF_FITS,           // the whole model fits this device's weight budget
    OFF_MOE,            // MoE models never auto-stream
    OFF_PLAN_RESIDENT,  // multi-device plan put every layer on a device owner
    ON_OVER_BUDGET,     // model exceeds the budget and the plan does not hold it resident
    ON_FORCED,          // GGML_SYCL_FORCE_STREAMING=1
};

struct layer_streaming_gate_inputs {
    bool                 forced            = false;
    bool                 is_moe            = false;
    size_t               model_bytes       = 0;  // whole tensor inventory
    size_t               weight_budget     = 0;  // this device's budget after arena zones
    bool                 plan_multi_device = false;
    plan_layer_residency residency;
};

inline layer_streaming_gate layer_streaming_gate_decide(const layer_streaming_gate_inputs & in) {
    if (in.forced) {
        return layer_streaming_gate::ON_FORCED;
    }
    if (in.model_bytes <= in.weight_budget) {
        return layer_streaming_gate::OFF_FITS;
    }
    if (in.is_moe) {
        return layer_streaming_gate::OFF_MOE;
    }
    // The whole-model-vs-this-device comparison above is the wrong question
    // for a multi-device plan: it charges dev0 for layers the plan gave to
    // another card. When the plan put every layer on a device owner, each
    // owner executes its layers from resident weights and a staging buffer
    // would never be read. Single-device plans keep the legacy answer.
    if (in.plan_multi_device && in.residency.layer_tensors > 0 && in.residency.unresident_tensors == 0) {
        return layer_streaming_gate::OFF_PLAN_RESIDENT;
    }
    return layer_streaming_gate::ON_OVER_BUDGET;
}

inline bool layer_streaming_gate_enabled(layer_streaming_gate g) {
    return g == layer_streaming_gate::ON_OVER_BUDGET || g == layer_streaming_gate::ON_FORCED;
}

inline const char * layer_streaming_gate_name(layer_streaming_gate g) {
    switch (g) {
        case layer_streaming_gate::OFF_FITS:
            return "off-fits";
        case layer_streaming_gate::OFF_MOE:
            return "off-moe";
        case layer_streaming_gate::OFF_PLAN_RESIDENT:
            return "off-plan-resident";
        case layer_streaming_gate::ON_OVER_BUDGET:
            return "on-over-budget";
        case layer_streaming_gate::ON_FORCED:
            return "on-forced";
    }
    return "unknown";
}

}  // namespace ggml_sycl
