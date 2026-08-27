#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// Input snapshot of the placement state relevant to runtime KV demotion.
// Mirrors placement_plan fields (unified-cache.hpp: kv_device / swa_layer_mask /
// kv_per_layer / kv_per_swa_layer / vram_bytes / vram_budget) as plain values so
// this TU stays host-linkable with no unified-cache dependency.
struct kv_demotion_input {
    size_t               vram_budget      = 0;
    size_t               vram_bytes       = 0;  // current total incl. device-resident KV
    size_t               kv_per_layer     = 0;  // full-attn per-layer KV bytes at requested n_ctx
    size_t               kv_per_swa_layer = 0;
    std::vector<int>     kv_device;             // index = layer id; >=0 device, -1 host
    std::vector<uint8_t> swa_layer_mask;        // 1 = SWA layer (never demoted)
};

struct kv_demotion_result {
    std::vector<int> demoted_layers;  // in demotion order (latest full-attn first)
    size_t           vram_bytes_after    = 0;
    size_t           host_kv_bytes_added = 0;
    bool             fits                = false;  // vram_bytes_after <= vram_budget
};

// Pure decision: which device-resident full-attention KV layers must move to the
// host tier so vram_bytes fits vram_budget. Latest layers first. SWA layers and
// already-host layers are never touched. Does NOT mutate any plan -- the caller
// (ggml-sycl.cpp runtime update) applies the result to placement_plan::kv_device.
kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in);

}  // namespace ggml_sycl
