#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// Input snapshot of the placement state relevant to runtime KV demotion.
// Adapted from placement_plan (unified-cache.hpp) as plain values so this TU
// stays host-linkable with no unified-cache dependency.
//
// llama.cpp-3aos (round 1 F3): kv_bytes_per_layer replaces what used to be a
// single kv_per_layer/kv_per_swa_layer scalar pair -- a uniform "one
// representative full-attention layer's bytes" figure applied to every
// full-attention layer regardless of its REAL per-layer width
// (placement_kv_info::kv_bytes_for_layer(), unified-cache.hpp) disagrees
// with refresh_kv_byte_totals() on the very same plan for a heterogeneous
// model (Gemma 4 E4B: full-attention layers are wider than its SWA layers).
// The caller (ggml_sycl_try_demote_runtime_kv(), ggml-sycl.cpp) now fills
// this per layer from that same shared formula, so the demotion decision
// and the byte-total refresh can no longer disagree about what any one
// layer costs.
struct kv_demotion_input {
    size_t               vram_budget = 0;
    size_t               vram_bytes  = 0;  // current total incl. device-resident KV
    // Per-layer device-resident KV bytes for THIS layer specifically (not a
    // uniform figure) -- 0 for a layer that holds no independent KV of its
    // own (e.g. a SHARED layer) or that this input does not track. Sized to
    // n_layers; kv_device and swa_layer_mask must be the same size.
    std::vector<size_t>  kv_bytes_per_layer;
    // Index = layer id; must be sized to n_layers. >=0 means device-resident on
    // that device id; -1 means already on the host tier. Layers absent from
    // placement_plan::kv_device are represented here as -1 (the caller adapts
    // the plan's sparse map into this dense vector before calling).
    std::vector<int>     kv_device;
    std::vector<uint8_t> swa_layer_mask;  // 1 = SWA layer (never demoted)
};

struct kv_demotion_result {
    std::vector<int> demoted_layers;  // in demotion order (latest full-attn first)
    size_t           vram_bytes_after    = 0;
    size_t           host_kv_bytes_added = 0;
    bool             fits                = false;  // vram_bytes_after <= vram_budget
};

// Pure decision: which device-resident full-attention KV layers must move to the
// host tier so vram_bytes fits vram_budget. Latest layers first. SWA layers,
// already-host layers, and layers with 0 recorded bytes (nothing to move) are
// never touched. Does NOT mutate any plan -- the caller (ggml-sycl.cpp runtime
// update) applies the result to placement_plan::kv_device.
kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in);

}  // namespace ggml_sycl
