//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_CACHE_HPP
#define GGML_SYCL_UNIFIED_CACHE_HPP

#include "device-pool.hpp"
#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "mem-handle.hpp"
#include "pinned-pool.hpp"
#include "residency-plan.hpp"
#include "tlsf-allocator.hpp"
#include "unified-cache-key.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace ggml_sycl {

// Forward declaration — needed by unified_cache::process_deferred_frees_public()
bool unified_cache_is_graph_compute_active();

// Check if VRAM arena mode is enabled (GGML_SYCL_VRAM_ARENA=1).
bool vram_arena_enabled();

// === VRAM Arena: zone-based sub-allocator for a single pre-allocated VRAM block ===
//
// Layout (within arena):
//   [KV (bump->) ...free... (<-bump WEIGHT)] [ONEDNN] [RUNTIME] [SCRATCH (bump->)]
//
// KV zone: bump-right, grows as context fills. Shared region with Weight.
// Weight zone: bump-LEFT from shared region end, with free-list for eviction reclaim.
// ONEDNN zone: persistent oneDNN scratchpad workspace.
// RUNTIME zone: persistent ggml compute buffers (allocated at context creation).
// SCRATCH zone: ephemeral pool_leg per-op scratch, reset between graph_compute calls.

enum class vram_zone_id : uint8_t {
    KV      = 0,  // KV cache, bump-right, grows with context
    WEIGHT  = 1,  // Weight cache, bump-left from end, free-list reclaim
    ONEDNN  = 2,  // oneDNN FP16 scratch, fixed region
    RUNTIME = 3,  // Runtime allocations (KV buffers, staging, MoE pools)
    SCRATCH = 4,  // Per-token scratch, bump-right, reset between tokens
    COUNT   = 5,
};

struct vram_zone {
    size_t              start = 0;  // Offset from arena base
    size_t              size  = 0;  // Total zone capacity
    std::atomic<size_t> used{ 0 };  // Bytes currently allocated (from TLSF)

    // TLSF O(1) sub-allocator for this zone.
    // Nullptr for WEIGHT zone in single-chunk mode (delegates to KV allocator).
    std::unique_ptr<tlsf_allocator> allocator;
    std::mutex                      alloc_mutex;  // Serialize concurrent alloc/free
};

// vram_arena functionality is merged into unified_cache — see arena_* methods.

// === Priority-based static placement planner (P4) ===
//
// At model load time, assigns each weight to VRAM (device) or host based on
// priority.  Weights are sorted by (priority, layer_id, byte_size descending)
// and packed into VRAM greedily.  Weights that don't fit spill to host.
//
// Priority ordering (Unsloth-informed):
//   0 = NORM / EMBEDDING   — tiny, always fit, needed for every token
//   1 = ATTENTION           — Q/K/V/O projections, critical for quality
//   2 = FFN                 — feed-forward weights
//   3 = MOE_GATE            — routing gate (small but latency-sensitive)
//   4 = MOE_DOWN            — expert down projections (Unsloth: most impactful)
//   5 = MOE_UP              — expert up projections
//   6 = MOE_GATE_PROJ       — expert gate projections (least impactful per Unsloth)

enum class placement_priority : uint8_t {
    NORM_EMBED    = 0,  // Norms + embeddings (tiny, always on device)
    ATTENTION     = 1,  // Attention projections
    FFN           = 2,  // FFN weights (non-MoE)
    MOE_GATE      = 3,  // MoE routing gate
    MOE_DOWN      = 4,  // MoE expert down_proj (most impactful per Unsloth)
    MOE_UP        = 5,  // MoE expert up_proj
    MOE_GATE_PROJ = 6,  // MoE expert gate_proj (least impactful per Unsloth)
    COUNT         = 7,
};

enum class expert_tensor_role : uint8_t {
    UNKNOWN = 0,
    GATE    = 1,
    UP      = 2,
    DOWN    = 3,
    COUNT   = 4,
};

inline const char * expert_tensor_role_name(expert_tensor_role role) {
    switch (role) {
        case expert_tensor_role::GATE:
            return "gate";
        case expert_tensor_role::UP:
            return "up";
        case expert_tensor_role::DOWN:
            return "down";
        case expert_tensor_role::UNKNOWN:
        default:
            return "unknown";
    }
}

inline expert_tensor_role expert_tensor_role_from_tensor_name(const char * name) {
    if (!name) {
        return expert_tensor_role::UNKNOWN;
    }
    if (std::strstr(name, "ffn_gate_exps")) {
        return expert_tensor_role::GATE;
    }
    if (std::strstr(name, "ffn_up_exps")) {
        return expert_tensor_role::UP;
    }
    if (std::strstr(name, "ffn_down_exps")) {
        return expert_tensor_role::DOWN;
    }
    return expert_tensor_role::UNKNOWN;
}

struct mxfp4_moe_single_gateup_layout_policy_input {
    const char *       env_value        = nullptr;
    ggml_type          type             = GGML_TYPE_COUNT;
    expert_tensor_role role             = expert_tensor_role::UNKNOWN;
    ggml_layout_mode   requested_layout = GGML_LAYOUT_AOS;
    bool               device_resident  = false;
    bool               xmx_int8_ok      = false;
    bool               shape_aligned    = false;
    size_t             pp_rows          = 0;
    bool               pp_supported     = false;
    bool               tg_supported     = false;
};

struct mxfp4_moe_single_gateup_layout_policy_result {
    bool             accepted               = false;
    bool             requires_soa_alternate = true;
    ggml_layout_mode layout                 = GGML_LAYOUT_SOA;
    const char *     reason                 = "env";
    const char *     route_label            = "fallback";
};

static inline const char * mxfp4_moe_single_gateup_route_label() {
    return "xmx-tiled-single-gateup";
}

static inline bool mxfp4_moe_single_gateup_layout_env_enabled(const char * env_value) {
    return env_value != nullptr && std::atoi(env_value) != 0;
}

static inline bool mxfp4_moe_gateup_role(expert_tensor_role role) {
    return role == expert_tensor_role::GATE || role == expert_tensor_role::UP;
}

static inline mxfp4_moe_single_gateup_layout_policy_result mxfp4_moe_single_gateup_layout_policy(
    const mxfp4_moe_single_gateup_layout_policy_input & in) {
    mxfp4_moe_single_gateup_layout_policy_result out{};
    if (!mxfp4_moe_single_gateup_layout_env_enabled(in.env_value)) {
        out.reason = "env";
        return out;
    }
    if (in.type != GGML_TYPE_MXFP4) {
        out.reason = "type";
        return out;
    }
    if (!mxfp4_moe_gateup_role(in.role)) {
        out.reason = "role";
        return out;
    }
    if (in.requested_layout != GGML_LAYOUT_XMX_TILED) {
        out.reason = "layout";
        return out;
    }
    if (!in.device_resident || !in.xmx_int8_ok) {
        out.reason = "device";
        return out;
    }
    if (!in.shape_aligned) {
        out.reason = "shape";
        return out;
    }
    if (in.pp_rows > 1 && !in.pp_supported) {
        out.reason = "pp";
        return out;
    }
    if (!in.tg_supported) {
        out.reason = "tg";
        return out;
    }
    out.accepted               = true;
    out.requires_soa_alternate = false;
    out.layout                 = GGML_LAYOUT_XMX_TILED;
    out.reason                 = "none";
    out.route_label            = mxfp4_moe_single_gateup_route_label();
    return out;
}

inline int expert_layer_from_tensor_name(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = std::strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    return std::atoi(blk + 4);
}

struct expert_placement_key {
    int                layer_id  = -1;
    int                expert_id = -1;
    expert_tensor_role role      = expert_tensor_role::UNKNOWN;

    bool operator==(const expert_placement_key & other) const {
        return layer_id == other.layer_id && expert_id == other.expert_id && role == other.role;
    }
};

struct expert_placement_key_hash {
    size_t operator()(const expert_placement_key & key) const {
        size_t h = std::hash<int>()(key.layer_id);
        h ^= std::hash<int>()(key.expert_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(static_cast<int>(key.role)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

enum class expert_placement_status : uint8_t {
    FOUND   = 0,
    MISSING = 1,
};

struct placement_alternate_layout {
    ggml_layout_mode layout           = GGML_LAYOUT_AOS;
    size_t           dst_size         = 0;
    size_t           vram_charge_size = 0;
    int              target_device    = -1;  // -1 means same device as the owning placement_entry
};

struct expert_placement_result {
    expert_placement_status                 status    = expert_placement_status::MISSING;
    int                                     layer_id  = -1;
    int                                     expert_id = -1;
    expert_tensor_role                      role      = expert_tensor_role::UNKNOWN;
    std::string                             tensor_name;
    bool                                    on_device        = false;
    int                                     target_device    = -1;
    size_t                                  src_size         = 0;
    size_t                                  dst_size         = 0;
    size_t                                  vram_charge_size = 0;
    ggml_layout_mode                        layout           = GGML_LAYOUT_AOS;
    std::vector<placement_alternate_layout> alternate_layouts;
    placement_priority                      priority = placement_priority::COUNT;

    bool found() const { return status == expert_placement_status::FOUND; }
};

struct expert_placement_summary {
    size_t                                                             planned      = 0;
    size_t                                                             expected     = 0;
    size_t                                                             missing      = 0;
    size_t                                                             duplicates   = 0;
    size_t                                                             unclassified = 0;
    std::array<size_t, static_cast<size_t>(expert_tensor_role::COUNT)> role_counts{};
    std::unordered_map<int, size_t>                                    target_counts;
};

inline size_t placement_vram_charge_bytes(size_t bytes, size_t alignment = 256) {
    if (bytes == 0) {
        return 0;
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        alignment = 256;
    }
    size_t charged = ((bytes + alignment - 1) / alignment) * alignment;
    return std::max<size_t>(charged, 256);
}

// Single entry in the placement plan: one weight tensor's placement decision.
struct placement_entry {
    std::string        name;  // Tensor name (for logging and S1-PRELOAD lookup)
    ggml_type          type              = GGML_TYPE_COUNT;
    int64_t            ne[GGML_MAX_DIMS] = {};
    size_t             src_size          = 0;  // AOS source bytes
    size_t             dst_size          = 0;  // Layout-converted destination bytes (SOA/COALESCED)
    size_t             kv_size           = 0;  // KV bytes charged by this entry (usually 0)
    placement_priority priority          = placement_priority::COUNT;  // Sort key
    int                layer_id          = -1;  // Layer number (earlier = higher priority within same level)
    int                expert_id         = -1;  // -1 for dense weights, >=0 for individual MoE experts
    expert_tensor_role expert_role       = expert_tensor_role::UNKNOWN;  // gate/up/down for MoE expert tensors
    ggml_layout_mode   layout            = GGML_LAYOUT_AOS;         // Materialized device layout planned for this entry
    std::vector<placement_alternate_layout> alternate_layouts;      // Additional cache-owned executable layouts
    bool                                    on_device     = false;  // true = VRAM (any device), false = host
    int                                     target_device = -1;     // Target GPU device_id (-1 = host/CPU)
    size_t vram_charge_size = 0;  // Bytes consumed by zone_alloc(WEIGHT) after allocator rounding

    placement_entry() = default;

    placement_entry(std::string        tensor_name,
                    size_t             source_size,
                    size_t             destination_size,
                    size_t             kv_bytes,
                    placement_priority placement_prio,
                    int                layer,
                    int                expert,
                    expert_tensor_role role,
                    bool               device_resident,
                    int                target) :
        name(std::move(tensor_name)),
        src_size(source_size),
        dst_size(destination_size),
        kv_size(kv_bytes),
        priority(placement_prio),
        layer_id(layer),
        expert_id(expert),
        expert_role(role),
        on_device(device_resident),
        target_device(target),
        vram_charge_size(placement_vram_charge_bytes(destination_size)) {}
};

struct placement_tensor_info {
    std::string name;
    size_t      size              = 0;
    ggml_type   type              = GGML_TYPE_COUNT;
    int64_t     ne[GGML_MAX_DIMS] = {};

    placement_tensor_info() = default;

    placement_tensor_info(std::string tensor_name, size_t tensor_size) :
        name(std::move(tensor_name)),
        size(tensor_size) {}

    bool has_shape() const { return type >= 0 && type < GGML_TYPE_COUNT && ne[0] > 0 && ne[1] > 0; }
};

// Per-device VRAM budget for multi-GPU placement planning.
struct device_budget {
    int    device_id       = 0;     // SYCL device index
    size_t vram_budget     = 0;     // Available VRAM bytes for weights on this device
    size_t total_vram      = 0;     // Total VRAM on this device (for proportional split)
    double dense_score     = 1.0;   // Queried dense-execution capability score
    bool   dense_supported = true;  // False when dense dispatch cannot run on this device
};

// Multi-GPU parallelism mode (GGML_SYCL_MULTI_GPU_MODE env var).
enum class multi_gpu_mode : uint8_t {
    LAYER  = 0,  // Pure layer parallelism (no cross-device expert sharing)
    EXPERT = 1,  // Pure expert parallelism (all layers replicated, experts split)
    HYBRID = 2,  // Layer parallelism for dense + expert parallelism for MoE (default)
};

// Explicit planner inputs used for KV sizing and placement.
struct placement_kv_info {
    uint32_t          n_layer          = 0;
    uint32_t          n_embd_k_gqa     = 0;
    uint32_t          n_embd_v_gqa     = 0;
    uint32_t          n_ctx            = 0;
    uint32_t          n_ubatch         = 512;  // Physical batch size (for SWA KV sizing)
    bool              n_ctx_is_runtime = false;
    // MoE hyperparameters (0 for dense models)
    int               n_expert_used    = 0;  // Top-k experts selected per token
    // SWA (Sliding Window Attention) — 0 means all layers use full attention
    uint32_t          n_swa            = 0;
    uint32_t          n_swa_layers     = 0;
    // Per-layer SWA flag: swa_layer_mask[il] == true means layer il uses SWA.
    // Empty when n_swa_layers == 0 (all layers use full attention).
    std::vector<bool> swa_layer_mask;

    bool valid() const { return n_layer > 0 && n_embd_k_gqa > 0 && n_embd_v_gqa > 0 && n_ctx > 0; }

    uint32_t n_full_attn_layers() const { return n_layer > n_swa_layers ? n_layer - n_swa_layers : 0; }

    // Returns true if the given layer uses SWA. Falls back to false when
    // the per-layer mask is not populated.
    bool is_swa_layer(int layer_id) const {
        if (layer_id < 0 || swa_layer_mask.empty()) {
            return false;
        }
        if (static_cast<size_t>(layer_id) >= swa_layer_mask.size()) {
            return false;
        }
        return swa_layer_mask[layer_id];
    }

    size_t kv_bytes_per_layer() const {
        if (!valid()) {
            return 0;
        }
        return static_cast<size_t>(n_ctx) * static_cast<size_t>(n_embd_k_gqa + n_embd_v_gqa) * sizeof(ggml_fp16_t);
    }

    size_t kv_bytes_per_swa_layer() const {
        if (!valid() || n_swa == 0) {
            return 0;
        }
        // Must match the actual SWA KV size from llama_kv_cache_iswa:
        //   size_swa = GGML_PAD(min(kv_size, n_swa * n_seq_max + n_ubatch), 256)
        // with n_seq_max=1. n_ubatch from model params (runtime hint).
        // Tensor per layer: K=[n_embd_k_gqa, size_swa] + V=[n_embd_v_gqa, size_swa], both fp16.
        const uint32_t swa_cells = ((std::min(n_ctx, n_swa + n_ubatch) + 255) / 256) * 256;
        return static_cast<size_t>(swa_cells) * static_cast<size_t>(n_embd_k_gqa + n_embd_v_gqa) * sizeof(ggml_fp16_t);
    }
};

// Complete placement plan for all model weights.
// Supports single-device (P4) and multi-device (P4.5) planning.
struct placement_plan {
    std::vector<placement_entry> entries;            // All weights, sorted by priority
    size_t                       vram_bytes;         // Total planned bytes on device(s), including KV
    size_t                       host_bytes;         // Total planned bytes on host, including KV
    size_t                       weight_vram_bytes;  // Weight bytes on device(s)
    size_t                       weight_host_bytes;  // Weight bytes on host
    size_t                       kv_vram_bytes;      // KV bytes on device(s)
    size_t                       kv_host_bytes;      // KV bytes on host
    size_t                       vram_budget;        // VRAM budget used for planning (primary device)
    int                          device_id;          // Primary device (single-device) or -1 (multi-device)
    bool                         multi_device;       // True if plan spans multiple GPUs
    size_t                       kv_per_layer     = 0;
    size_t                       kv_per_swa_layer = 0;
    std::vector<bool>            swa_layer_mask;  // swa_layer_mask[l] == true → SWA layer
    uint32_t                     planner_n_ctx            = 0;
    bool                         planner_n_ctx_is_runtime = false;
    // True when XMX_TILED gate/up primaries were rewritten to PP-safe SOA
    // because the prompt path cannot consume tiled. Decode should then
    // rematerialize XMX_TILED through the phase-layout machinery.
    bool                         moe_pp_soa_promoted      = false;
    size_t                       host_zone_weight_bytes   = 0;
    size_t                       host_zone_kv_bytes       = 0;
    size_t                       host_zone_staging_bytes  = 0;
    size_t                       host_zone_scratch_bytes  = 0;
    size_t                       max_tensor_bytes         = 0;
    size_t                       max_staging_pair_bytes   = 0;

    // --- Inference memory categories (computed by populate_host_zone_sizing) ---
    // oneDNN reorder: one temp buffer (max weight tensor size) reused per layer.
    size_t   onednn_reorder_bytes                = 0;  // Zone: SCRATCH (device VRAM)
    // MoE Q8_1 workspace: quantized activations for batched expert dispatch.
    size_t   moe_q8_workspace_bytes              = 0;  // Zone: SCRATCH (device VRAM)
    // CPU expert fallback act/out staging. EXPERT_STAGING HOST_COMPUTE uses
    // smart-handle scoped ownership and routes to host SCRATCH.
    size_t   moe_cpu_expert_staging_bytes        = 0;  // Zone: SCRATCH (host pinned)
    // Expert bias D2H copy: float bias tensors staged to host for gate computation.
    size_t   expert_bias_bytes                   = 0;  // Zone: STAGING (host pinned)
    // MoE routing: per-batch expert ID control buffer (n_expert * max_batch_tokens * sizeof(int32_t)).
    size_t   moe_routing_ids_bytes               = 0;  // Zone: HOST_CONTROL (shared-context host pinned)
    // MoE expert pointer tables: void* per expert per MoE layer (MAX_EXPERTS * n_moe_layers * sizeof(void*)).
    size_t   moe_expert_ptrs_bytes               = 0;  // Zone: RUNTIME (device VRAM)
    // Combined VRAM RUNTIME reservation for MoE device routing buffers (expert_ptrs only;
    // routing IDs are HOST_CONTROL and are accounted in host pinned planning).
    // 0 when model has no MoE layers.
    size_t   moe_vram_runtime_bytes              = 0;  // Zone: RUNTIME (device VRAM)
    // DMA staging pool: device-resident double-buffer for host→device weight streaming.
    // Sized as max_tensor_bytes × k_dma_pipeline_depth (2 buffers). 0 when streaming disabled.
    size_t   dma_staging_pool_bytes              = 0;  // Zone: RUNTIME (device VRAM)
    // oneDNN scratchpad: workspace for matmul weight reorder (weights) + activation buffer.
    // Sized as max_tensor_bytes × 2. Must fit within the ONEDNN zone (default 256 MB).
    size_t   onednn_scratchpad_bytes             = 0;  // Zone: ONEDNN (device VRAM)
    // PP pipeline scratch: double-buffered FP16 weight staging for prompt-processing
    // dequant prefetch. Computed from the largest quantized weight tensor as
    // 2 x (n_elements * sizeof(fp16)). Lives in the RUNTIME zone.
    size_t   pp_pipeline_scratch_bytes           = 0;  // Zone: RUNTIME (device VRAM)
    // PP MoE oneDNN scratch ring for the MXFP4 SOA prompt path. The current
    // prompt path serializes slot use, so the default plan budgets one reusable
    // slot; future pipelining must raise this only when it actually overlaps use.
    size_t   pp_moe_onednn_weight_slot_bytes     = 0;  // Zone: RUNTIME (device VRAM)
    size_t   pp_moe_onednn_activation_slot_bytes = 0;  // Zone: RUNTIME (device VRAM)
    size_t   pp_moe_onednn_output_slot_bytes     = 0;  // Zone: RUNTIME (device VRAM)
    size_t   pp_moe_onednn_scratch_bytes         = 0;  // Zone: RUNTIME (device VRAM)
    uint32_t pp_moe_onednn_ring_depth            = 0;
    // CPU quantization temp buffers: pre-allocated by T1 cpu_dispatch_buffers.
    size_t   cpu_quant_buffer_bytes              = 0;  // Zone: HOST (system heap)
    // Graph metadata: layer classification vectors and MoE routing tables.
    size_t   graph_metadata_bytes                = 0;  // Zone: HOST (system heap)

    // --- Tensor Parallelism buffer estimates (0 when TP is disabled) ---
    // Per-layer FFN compute buffers on secondary TP devices (device VRAM).
    // Sized from n_embd / n_ff heuristics derived from FFN tensor inventory.
    size_t tp_ffn_buffer_bytes     = 0;  // Zone: RUNTIME (device VRAM, per secondary device)
    // Per-layer attention compute buffers on secondary TP devices (device VRAM).
    size_t tp_attn_buffer_bytes    = 0;  // Zone: RUNTIME (device VRAM, per secondary device)
    // Host staging for TP input copies (D2H of activations before H2D to secondary device).
    size_t tp_staging_buffer_bytes = 0;  // Zone: STAGING (host pinned)
    // Combined VRAM RUNTIME estimate for all TP compute buffers (tp_ffn + tp_attn).
    // Folded into host_zone_scratch_bytes for zone budgeting; also available to
    // secondary-device planners as a sizing reference. 0 when TP is disabled.
    size_t tp_vram_runtime_bytes   = 0;  // Zone: RUNTIME (device VRAM)

    // Per-device VRAM usage (multi-device only, indexed by position in devices vector)
    std::vector<int>    devices;          // Device IDs participating in this plan
    std::vector<size_t> per_device_vram;  // VRAM bytes used per device

    // --- Multi-device runtime query maps (populated by compute_multi_device_plan) ---

    // device_layers[device_id] = {layer_start, layer_end} (inclusive range).
    // Dense layers in [start, end] are owned by this device.
    struct layer_range {
        int start = -1;
        int end   = -1;
    };

    // Contiguous transformer-layer block selected by the planner.  This is the
    // contract the multi-device executor consumes: complete layer ranges, their
    // dense/KV owner, and the MoE residency footprint that may require routed
    // expert dispatch inside the block.  `kv_device == -2` means the block has
    // mixed KV ownership.
    struct layer_block {
        int         start_layer             = -1;
        int         end_layer               = -1;
        int         execution_device        = -1;
        int         dense_device            = -1;
        int         kv_device               = -1;
        size_t      dense_weight_bytes      = 0;
        size_t      dense_vram_charge_bytes = 0;
        size_t      kv_bytes                = 0;
        size_t      moe_device_weight_bytes = 0;
        size_t      moe_host_weight_bytes   = 0;
        double      dense_score             = 0.0;
        bool        dense_on_fastest_device = false;
        std::string dense_policy_reason;

        // Estimated activation transfer required at contiguous layer-block
        // boundaries.  The executor still owns the actual smart-handle staging
        // allocations; these fields are planner cost/telemetry inputs.
        int      previous_execution_device      = -1;
        int      next_execution_device          = -1;
        size_t   boundary_from_prev_bytes       = 0;
        size_t   boundary_to_next_bytes         = 0;
        double   boundary_from_prev_est_us      = 0.0;
        double   boundary_to_next_est_us        = 0.0;
        bool     boundary_from_prev_direct      = false;
        bool     boundary_to_next_direct        = false;
        uint32_t boundary_batch_tokens          = 0;
        int64_t  boundary_activation_hidden_dim = 0;
    };

    std::unordered_map<int, layer_range> device_layers;
    std::vector<layer_block>             layer_blocks;
    std::vector<layer_block>             candidate_layer_blocks;
    std::string                          candidate_layer_blocks_reason;
    int                                  fastest_dense_device = -1;
    double                               fastest_dense_score  = 0.0;

    // Compatibility summary: expert_device[layer_id][expert_id] = device_id (-1 = host/CPU).
    // This collapses gate/up/down and is not authoritative for role-specific MoE placement.
    // Use lookup_expert_placement(layer, expert, role) for planner decisions.
    std::unordered_map<int, std::unordered_map<int, int>> expert_device;

    // kv_location[layer_id] = device_id for KV cache co-location.
    // KV cache for a layer lives on the same device as its dense weights.
    std::unordered_map<int, int> kv_device;

    // layer_device[layer_id] = device_id (-1 = host/CPU) for the dense
    // execution unit of the layer. This is the authoritative placement for the
    // layer's shared dense weights; MoE experts remain separately placeable.
    std::unordered_map<int, int> layer_device;

    int get_kv_device(int layer_id) const {
        auto it = kv_device.find(layer_id);
        return it == kv_device.end() ? -1 : it->second;
    }

    int get_layer_device(int layer_id) const {
        auto it = layer_device.find(layer_id);
        return it == layer_device.end() ? -1 : it->second;
    }

    size_t kv_size_for_layer(uint32_t layer_id) const {
        if (kv_per_swa_layer > 0 && layer_id < swa_layer_mask.size() && swa_layer_mask[layer_id]) {
            return kv_per_swa_layer;
        }
        return kv_per_layer;
    }

    size_t kv_layer_count() const {
        size_t n = swa_layer_mask.size();
        for (const auto & [layer_id, _] : kv_device) {
            if (layer_id >= 0) {
                n = std::max(n, static_cast<size_t>(layer_id) + 1);
            }
        }
        for (const auto & [layer_id, _] : layer_device) {
            if (layer_id >= 0) {
                n = std::max(n, static_cast<size_t>(layer_id) + 1);
            }
        }
        return n;
    }

    void refresh_kv_byte_totals() {
        kv_vram_bytes = 0;
        kv_host_bytes = 0;

        const size_t n_layers = kv_layer_count();
        for (uint32_t l = 0; l < n_layers; ++l) {
            const size_t kv_bytes = kv_size_for_layer(l);
            if (kv_bytes == 0) {
                continue;
            }
            if (get_kv_device(static_cast<int>(l)) >= 0) {
                kv_vram_bytes += kv_bytes;
            } else {
                kv_host_bytes += kv_bytes;
            }
        }

        vram_bytes = weight_vram_bytes + kv_vram_bytes;
        host_bytes = weight_host_bytes + kv_host_bytes;
        if (kv_host_bytes > host_zone_kv_bytes) {
            host_zone_kv_bytes = kv_host_bytes;
        }
    }

    void update_runtime_kv_sizes(uint32_t n_ctx, size_t full_kv_per_layer, size_t swa_kv_per_layer) {
        planner_n_ctx            = n_ctx;
        planner_n_ctx_is_runtime = true;
        kv_per_layer             = full_kv_per_layer;
        kv_per_swa_layer         = swa_kv_per_layer;
        refresh_kv_byte_totals();
    }

    expert_placement_result lookup_expert_placement(int layer_id, int expert_id, expert_tensor_role role) const {
        expert_placement_result result;
        result.layer_id  = layer_id;
        result.expert_id = expert_id;
        result.role      = role;

        const expert_placement_key key{ layer_id, expert_id, role };
        auto                       it = expert_placement_index_.find(key);
        if (it == expert_placement_index_.end()) {
            return result;
        }

        const auto & e           = entries[it->second];
        result.status            = expert_placement_status::FOUND;
        result.tensor_name       = e.name;
        result.on_device         = e.on_device;
        result.target_device     = e.target_device;
        result.src_size          = e.src_size;
        result.dst_size          = e.dst_size;
        result.vram_charge_size  = e.vram_charge_size;
        result.layout            = e.layout;
        result.alternate_layouts = e.alternate_layouts;
        result.priority          = e.priority;
        return result;
    }

    expert_placement_result lookup_expert_placement(const std::string & tensor_name, int expert_id) const {
        const int                layer_id = expert_layer_from_tensor_name(tensor_name.c_str());
        const expert_tensor_role role     = expert_tensor_role_from_tensor_name(tensor_name.c_str());
        return lookup_expert_placement(layer_id, expert_id, role);
    }

    bool update_expert_placement(int layer_id, int expert_id, expert_tensor_role role, bool on_device, int target) {
        const expert_placement_key key{ layer_id, expert_id, role };
        auto                       it = expert_placement_index_.find(key);
        if (it == expert_placement_index_.end()) {
            return false;
        }

        placement_entry & e = entries[it->second];
        if (e.on_device == on_device && e.target_device == target) {
            return true;
        }

        auto saturating_sub = [](size_t & value, size_t delta) {
            value = value > delta ? value - delta : 0;
        };

        const size_t charge = e.vram_charge_size != 0 ? e.vram_charge_size : placement_vram_charge_bytes(e.dst_size);

        if (e.on_device && !on_device) {
            saturating_sub(weight_vram_bytes, charge);
            saturating_sub(vram_bytes, charge + e.kv_size);
            weight_host_bytes += e.dst_size;
            host_bytes += e.dst_size + e.kv_size;
        } else if (!e.on_device && on_device) {
            saturating_sub(weight_host_bytes, e.dst_size);
            saturating_sub(host_bytes, e.dst_size + e.kv_size);
            weight_vram_bytes += charge;
            vram_bytes += charge + e.kv_size;
        }

        e.on_device     = on_device;
        e.target_device = target;

        if (!on_device) {
            expert_device[layer_id][expert_id] = -1;
        } else {
            expert_device[layer_id][expert_id] = target;
        }
        return true;
    }

    bool demote_expert_placement_to_host(const std::string & tensor_name, int expert_id) {
        const int                layer_id = expert_layer_from_tensor_name(tensor_name.c_str());
        const expert_tensor_role role     = expert_tensor_role_from_tensor_name(tensor_name.c_str());
        return update_expert_placement(layer_id, expert_id, role, false, -1);
    }

    expert_placement_summary summarize_expert_placements(int expected_layers = -1, int expected_experts = -1) const {
        expert_placement_summary summary;
        summary.planned      = expert_placement_index_.size();
        summary.duplicates   = expert_placement_duplicate_count_;
        summary.unclassified = expert_placement_unclassified_count_;
        for (const auto & e : entries) {
            if (e.expert_id < 0) {
                continue;
            }
            const size_t role_idx = static_cast<size_t>(e.expert_role);
            if (role_idx < summary.role_counts.size()) {
                summary.role_counts[role_idx]++;
            }
            auto target_it = summary.target_counts.find(e.target_device);
            if (target_it == summary.target_counts.end()) {
                summary.target_counts.emplace(e.target_device, 1);
            } else {
                target_it->second++;
            }
        }

        if (expected_layers > 0 && expected_experts > 0) {
            summary.expected = static_cast<size_t>(expected_layers) * static_cast<size_t>(expected_experts) * 3u;
        } else {
            summary.expected = summary.planned + summary.unclassified + summary.duplicates;
        }
        summary.missing = summary.expected > summary.planned ? summary.expected - summary.planned : 0;
        return summary;
    }

    // Quick lookup: is this tensor on any device?
    // Returns true if tensor with given name is planned for VRAM (any GPU).
    // For dense weights (expert_id == -1) only; MoE expert entries are not
    // indexed by name alone — use expert_on_device() for per-expert queries.
    bool is_on_device(const std::string & name) const {
        auto it = name_index_.find(name);
        return it != name_index_.end() && entries[it->second].on_device;
    }

    bool has_dense_entry(const std::string & name) const { return name_index_.find(name) != name_index_.end(); }

    // Multi-device query: is this tensor on a specific device?
    // Returns true if the tensor is assigned to device_id.
    bool is_on_device(const std::string & name, int dev_id) const {
        auto it = name_index_.find(name);
        if (it == name_index_.end()) {
            return false;
        }
        const auto & e = entries[it->second];
        return e.on_device && e.target_device == dev_id;
    }

    // Get the target device for a tensor (-1 if host or not found).
    int get_target_device(const std::string & name) const {
        auto it = name_index_.find(name);
        if (it == name_index_.end()) {
            return -1;
        }
        return entries[it->second].target_device;
    }

    bool dense_alternate_on_device(const std::string &          name,
                                   int                          dev_id,
                                   placement_alternate_layout * out = nullptr) const {
        auto it = name_index_.find(name);
        if (it == name_index_.end()) {
            return false;
        }
        const placement_entry & e = entries[it->second];
        if (e.expert_id >= 0 || dev_id < 0) {
            return false;
        }
        for (const placement_alternate_layout & alt : e.alternate_layouts) {
            const int alt_target = alt.target_device >= 0 ? alt.target_device : e.target_device;
            if (alt_target == dev_id) {
                if (out != nullptr) {
                    *out = alt;
                }
                return true;
            }
        }
        return false;
    }

    // MoE expert query: is a specific expert of a tensor planned for VRAM?
    // tensor_name: the composite MoE tensor name (e.g. "blk.0.ffn_down_exps")
    // expert_id: individual expert index (0..n_experts-1)
    // device_id: if >= 0, checks assignment to a specific device; -1 checks any device
    bool expert_on_device(const std::string & tensor_name, int expert_id, int device_id = -1) const {
        const auto placement = lookup_expert_placement(tensor_name, expert_id);
        if (placement.found()) {
            if (!placement.on_device) {
                return false;
            }
            if (device_id >= 0) {
                return placement.target_device == device_id;
            }
            return true;
        }

        const std::string key = tensor_name + ":e" + std::to_string(expert_id);
        auto              it  = expert_index_.find(key);
        if (it == expert_index_.end()) {
            return false;  // Missing/unclassified entries are explicit host/miss, never implicit device.
        }
        const auto & e = entries[it->second];
        if (!e.on_device) {
            return false;
        }
        if (device_id >= 0) {
            return e.target_device == device_id;
        }
        return true;
    }

    // Returns true if any expert of the given MoE tensor is planned for host (not device).
    // Replaces the static plan_has_cpu_cache map in ggml_sycl_mul_mat_id: the expert_index_
    // already provides O(1) per-expert lookup, so a separate memoization layer is redundant.
    // device_id: if >= 0, checks residency for that specific device; -1 checks any device.
    bool has_host_experts(const std::string & tensor_name, int64_t n_experts, int device_id = -1) const {
        (void) device_id;
        for (int64_t e = 0; e < n_experts; ++e) {
            const auto placement = lookup_expert_placement(tensor_name, static_cast<int>(e));
            if (placement.found()) {
                if (!placement.on_device) {
                    return true;
                }
                continue;
            }

            const std::string key = tensor_name + ":e" + std::to_string(e);
            auto              it  = expert_index_.find(key);
            if (it != expert_index_.end() && !entries[it->second].on_device) {
                return true;
            }
        }
        return false;
    }

    // Build the name->index lookup after entries are populated.
    void build_index() {
        name_index_.clear();
        expert_index_.clear();
        expert_placement_index_.clear();
        expert_placement_duplicate_count_    = 0;
        expert_placement_unclassified_count_ = 0;
        name_index_.reserve(entries.size());
        expert_index_.reserve(entries.size());
        expert_placement_index_.reserve(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto & e = entries[i];
            if (e.expert_id >= 0) {
                // MoE expert entry: index by "tensor_name:eN"
                const std::string key = e.name + ":e" + std::to_string(e.expert_id);
                expert_index_[key]    = i;

                if (e.expert_role == expert_tensor_role::UNKNOWN || e.layer_id < 0) {
                    expert_placement_unclassified_count_++;
                    continue;
                }

                const expert_placement_key semantic_key{ e.layer_id, e.expert_id, e.expert_role };
                auto                       inserted = expert_placement_index_.emplace(semantic_key, i);
                if (!inserted.second) {
                    expert_placement_duplicate_count_++;
                }
            } else {
                // Dense weight: index by name
                name_index_[e.name] = i;
            }
        }
    }

  private:
    std::unordered_map<std::string, size_t> name_index_;    // Dense weights: name -> index
    std::unordered_map<std::string, size_t> expert_index_;  // MoE experts: "name:eN" -> index
    std::unordered_map<expert_placement_key, size_t, expert_placement_key_hash>
           expert_placement_index_;                         // MoE experts: (layer, expert, role) -> index
    size_t expert_placement_duplicate_count_    = 0;
    size_t expert_placement_unclassified_count_ = 0;
};

// Compute placement plan for all model weights given a VRAM budget.
// tensor_inventory: vector of (name, src_size) pairs from model header.
// vram_budget: available VRAM bytes for weights.
// device_id: target device (for layout size calculation).
// n_experts: experts per MoE layer (0 for dense models).  When > 0,
//            MoE tensors (identified by _exps suffix) are split into
//            per-expert entries so each expert competes individually
//            for VRAM placement.
// Sorts by (priority ASC, layer_id ASC, dst_size DESC, expert_id ASC)
// and greedily packs into VRAM.  The mapping from tensor_usage to
// placement_priority is in unified-cache.cpp (requires common.hpp types).
// `envelope` is the authoritative source of the four declared
// llama_model_params capacity inputs (n_ctx / n_ubatch / n_seq_max /
// flash_attn_type), populated by the llama layer at model load via
// ggml_backend_sycl_set_placement_envelope().  May be NULL when the
// caller never set an envelope (legacy / test paths); the planner then
// falls back to inventory-derived sizing through `kv_info` only.
// Numeric KV cost still flows through `kv_info`, but `envelope` lets the
// planner annotate its log output with explicit source attribution and
// is the entry point for future planner decisions that need the raw
// envelope (n_seq_max, flash_attn_type) rather than the derived snapshot.
placement_plan compute_placement_plan(const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
                                      size_t                                              vram_budget,
                                      int                                                 device_id,
                                      const placement_kv_info &                           kv_info,
                                      const ggml_sycl_placement_envelope *                envelope,
                                      int                                                 n_experts = 0);
placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,
                                      size_t                                     vram_budget,
                                      int                                        device_id,
                                      const placement_kv_info &                  kv_info,
                                      const ggml_sycl_placement_envelope *       envelope,
                                      int                                        n_experts = 0);

// P4.5: Compute multi-device placement plan for hybrid parallelism.
// Dense layers use layer parallelism (contiguous ranges per device, proportional to VRAM).
// MoE experts use expert parallelism (distributed across devices by Unsloth priority).
// Falls back to single-device compute_placement_plan() when only 1 device is provided.
//
// device_budgets: per-device VRAM budgets, sorted by device_id.
// tensor_inventory: vector of (name, src_size) pairs from model header.
// n_layers: total number of transformer layers in the model.
// mode: parallelism strategy (LAYER, EXPERT, or HYBRID).
// n_experts: experts per MoE layer (0 for dense models).  When > 0,
//            MoE tensors are split into per-expert entries for fine-grained
//            per-expert device assignment.
placement_plan compute_multi_device_plan(const std::vector<device_budget> &                  device_budgets,
                                         const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
                                         int                                                 n_layers,
                                         multi_gpu_mode                                      mode,
                                         const placement_kv_info &                           kv_info,
                                         const ggml_sycl_placement_envelope *                envelope,
                                         int                                                 n_experts = 0);
placement_plan compute_multi_device_plan(const std::vector<device_budget> &         device_budgets,
                                         const std::vector<placement_tensor_info> & tensor_inventory,
                                         int                                        n_layers,
                                         multi_gpu_mode                             mode,
                                         const placement_kv_info &                  kv_info,
                                         const ggml_sycl_placement_envelope *       envelope,
                                         int                                        n_experts = 0);

void     unified_cache_set_planned_pp_pipeline_scratch_bytes(int device_id, size_t bytes);
size_t   unified_cache_get_planned_pp_pipeline_scratch_bytes(int device_id);
void     unified_cache_set_planned_onednn_scratchpad_bytes(int device_id, size_t bytes);
size_t   unified_cache_get_planned_onednn_scratchpad_bytes(int device_id);
void     unified_cache_set_planned_pp_moe_onednn_scratch(int      device_id,
                                                         size_t   weight_slot_bytes,
                                                         size_t   activation_slot_bytes,
                                                         size_t   output_slot_bytes,
                                                         uint32_t ring_depth);
size_t   unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(int device_id);
size_t   unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(int device_id);
size_t   unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(int device_id);
size_t   unified_cache_get_planned_pp_moe_onednn_scratch_bytes(int device_id);
uint32_t unified_cache_get_planned_pp_moe_onednn_ring_depth(int device_id);

// Parse GGML_SYCL_MULTI_GPU_MODE env var.
// Returns HYBRID for MoE models, LAYER for dense-only, unless overridden.
multi_gpu_mode get_multi_gpu_mode(bool is_moe);

namespace detail {

static constexpr uint64_t k_cache_guard_magic = 0xC0DECA5EC0DECA5EULL;

struct alignas(16) cache_guard_header {
    uint64_t magic        = k_cache_guard_magic;
    size_t   size         = 0;
    size_t   mapping_size = 0;
    void *   mapping_base = nullptr;
};

inline bool cache_guard_pages_enabled() {
    const char * env = std::getenv("GGML_SYCL_CACHE_GUARD_PAGES");
    if (!env || env[0] == '\0') {
        env = std::getenv("GGML_SYCL_UNIFIED_CACHE_GUARD_PAGES");
    }
    return env && std::atoi(env) != 0;
}

inline size_t cache_guard_page_size() {
#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
    const long page_size_long = sysconf(_SC_PAGESIZE);
    return page_size_long > 0 ? static_cast<size_t>(page_size_long) : 4096;
#else
    return 4096;
#endif
}

template <typename T> struct cache_guard_allocator {
    using value_type = T;

    cache_guard_allocator() noexcept = default;

    template <typename U> cache_guard_allocator(const cache_guard_allocator<U> &) noexcept {}

    T * allocate(std::size_t n) {
        if (!cache_guard_pages_enabled()) {
            return std::allocator<T>{}.allocate(n);
        }

        const size_t size_bytes = n * sizeof(T);
        if (size_bytes == 0 || (size_bytes % alignof(T)) != 0) {
            return std::allocator<T>{}.allocate(n);
        }

#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
        const size_t page_size    = cache_guard_page_size();
        const size_t usable       = ((sizeof(cache_guard_header) + size_bytes + page_size - 1) / page_size) * page_size;
        const size_t mapping_size = usable + page_size;
        void *       mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED || !mapping) {
            throw std::bad_alloc();
        }
        if (mprotect(static_cast<uint8_t *>(mapping) + usable, page_size, PROT_NONE) != 0) {
            munmap(mapping, mapping_size);
            throw std::bad_alloc();
        }

        uint8_t * data_end   = static_cast<uint8_t *>(mapping) + usable;
        uint8_t * data       = data_end - size_bytes;
        auto *    header     = reinterpret_cast<cache_guard_header *>(data - sizeof(cache_guard_header));
        header->magic        = k_cache_guard_magic;
        header->size         = size_bytes;
        header->mapping_size = mapping_size;
        header->mapping_base = mapping;

        return reinterpret_cast<T *>(data);
#else
        return std::allocator<T>{}.allocate(n);
#endif
    }

    void deallocate(T * ptr, std::size_t n) noexcept {
        if (!ptr) {
            return;
        }
        if (!cache_guard_pages_enabled()) {
            std::allocator<T>{}.deallocate(ptr, n);
            return;
        }

        auto * header =
            reinterpret_cast<cache_guard_header *>(reinterpret_cast<uint8_t *>(ptr) - sizeof(cache_guard_header));
        if (!header || header->magic != k_cache_guard_magic || header->mapping_size == 0) {
            std::allocator<T>{}.deallocate(ptr, n);
            return;
        }

#if !defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
        if (header->mapping_base && header->mapping_size > 0) {
            munmap(header->mapping_base, header->mapping_size);
        }
#else
        (void) header;
#endif
    }

    template <typename U> struct rebind {
        using other = cache_guard_allocator<U>;
    };
};

template <typename T, typename U>
inline bool operator==(const cache_guard_allocator<T> &, const cache_guard_allocator<U> &) noexcept {
    return true;
}

template <typename T, typename U>
inline bool operator!=(const cache_guard_allocator<T> &, const cache_guard_allocator<U> &) noexcept {
    return false;
}

}  // namespace detail

// Expert tensor group: cache keys for all 3 tensors (gate, up, down) of one expert.
// Built during moe_hybrid_init_once() and used for atomic staging/eviction.
struct expert_tensor_group {
    ggml_sycl_cache_id gate_key;  // cache key for ffn_gate_exps expert slice
    ggml_sycl_cache_id up_key;    // cache key for ffn_up_exps expert slice
    ggml_sycl_cache_id down_key;  // cache key for ffn_down_exps expert slice
    bool               has_gate = false;
    bool               has_up   = false;
    bool               has_down = false;
};

// Key helper: (block_id << 16) | expert_id -- supports up to 65536 experts/block
inline int64_t expert_group_key(int block_id, int expert_id) {
    return (static_cast<int64_t>(block_id) << 16) | static_cast<int64_t>(expert_id & 0xFFFF);
}

// Data needed to stage one tensor of an expert group.
// fill_fn/fill_ctx allow per-tensor reorder (GPU or CPU).
// When fill_fn is nullptr, a raw DMA memcpy is used.
struct staging_tensor_data {
    const void * src_ptr   = nullptr;  // host-accessible source data (AOS)
    size_t       src_size  = 0;        // source bytes (AOS)
    size_t       dst_size  = 0;        // destination bytes (SOA)
    int          layer_id  = -1;
    int          expert_id = -1;
};

// Cache entry readiness
enum class cache_entry_state {
    READY,
    IN_PROGRESS,
    FAILED,
    EVICTING,  // Async D2H eviction in flight — VRAM still occupied, host copy pending
};

// Memory location for cached pointers/buffers
enum class cache_location {
    DEVICE,
    HOST_PINNED,
    HOST_MMAP,
    UNKNOWN,  // Sentinel: entry exists but location is unresolved (cache miss)
};

// XMX layout metadata carried with cache entries/results
struct cache_layout_xmx_info {
    int64_t tile_n        = 0;
    int64_t tile_k        = 0;
    int64_t n_tile_groups = 0;
};

struct cache_ptr_view {
    void *                ptr             = nullptr;
    size_t                size            = 0;
    ggml_layout_mode      layout          = GGML_LAYOUT_AOS;
    int64_t               onednn_pack_m   = 0;
    cache_location        location        = cache_location::DEVICE;
    cache_entry_type      type            = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id        = -1;
    int                   expert_id       = -1;
    cache_layout_xmx_info xmx_info        = {};
    bool                  has_ready_event = false;
    sycl::event           ready_event;
};

struct cache_layout_request;
using cache_layout_fill_fn = sycl::event (*)(sycl::queue &                    queue,
                                             void *                           dst,
                                             size_t                           dst_size,
                                             const void *                     src,
                                             size_t                           src_size,
                                             const void *                     ctx,
                                             const std::vector<sycl::event> & deps);

struct cache_layout_request {
    ggml_sycl_cache_id    key              = {};
    const void *          src_ptr          = nullptr;
    size_t                src_size         = 0;
    size_t                dst_size         = 0;
    cache_entry_type      type             = cache_entry_type::DENSE_WEIGHT;
    int                   layer_id         = -1;
    int                   expert_id        = -1;
    ggml_layout_mode      layout           = GGML_LAYOUT_AOS;
    int64_t               onednn_pack_m    = 0;
    bool                  validate_content = false;
    bool                  prefer_host      = false;
    bool                  force_pool = false;  // When true, use pool even in S1 mode (for S1-PRELOAD pinned weights)
    cache_layout_xmx_info xmx_info   = {};
    cache_layout_fill_fn  fill_fn    = nullptr;
    const void *          fill_ctx   = nullptr;
};

// --- Direct Staging API ---
// Simple structs for direct arena allocation + memcpy + reorder.
// No state machine, no IN_PROGRESS/READY transitions.

struct weight_entry {
    void *                      ptr             = nullptr;
    size_t                      size            = 0;
    ggml_layout_mode            layout          = GGML_LAYOUT_AOS;
    cache_location              location        = cache_location::DEVICE;
    bool                        has_ready_event = false;
    sycl::event                 ready_event;
    std::shared_ptr<mem_handle> handle;
    // Transient direct lookup mirror only. The handle is the canonical lifetime
    // token; ptr is a resolved mirror for diagnostics and legacy ABI boundaries.
};

struct direct_stage_result {
    void *      ptr = nullptr;        // Cache pointer: device VRAM or planned host-pinned
    sycl::event event;                // Completion event
    bool        ok          = false;
    bool        fill_failed = false;  // A submitted fill/copy/reorder threw; caller must not continue inference.
};

enum class expert_resolve_device_policy : uint8_t {
    ANY            = 0,  // Accept any owning device or allowed host tier
    PREFER_CURRENT = 1,  // Prefer current_device, but do not reject other devices
    CURRENT_ONLY   = 2,  // Reject DEVICE entries not owned by current_device
    PREFERRED_ONLY = 3,  // Reject DEVICE entries not owned by preferred_device
};

enum class expert_resolve_tier : uint8_t {
    DEVICE_VRAM = 0,
    HOST_PINNED = 1,
    HOST_MMAP   = 2,
    UNAVAILABLE = 3,
};

enum class expert_resolve_reason : uint8_t {
    FOUND                = 0,
    INVALID_REQUEST      = 1,
    NOT_FOUND            = 2,
    LAYOUT_MISMATCH      = 3,
    NOT_READY            = 4,
    DEVICE_MISMATCH      = 5,
    HOST_DISALLOWED      = 6,
    MMAP_HOST_DISALLOWED = 7,
};

struct expert_resolve_request {
    ggml_sycl_cache_id           key              = {};
    ggml_layout_mode             requested_layout = GGML_LAYOUT_AOS;
    int                          layer_id         = -1;
    int                          expert_id        = -1;
    int                          current_device   = -1;
    int                          preferred_device = -1;
    expert_resolve_device_policy device_policy    = expert_resolve_device_policy::ANY;
    bool                         allow_host       = true;
    bool                         allow_mmap_host  = true;
    bool                         require_ready    = false;
};

struct expert_resolve_result {
    void *                      ptr             = nullptr;
    size_t                      size            = 0;
    expert_resolve_tier         tier            = expert_resolve_tier::UNAVAILABLE;
    cache_location              location        = cache_location::UNKNOWN;
    int                         owning_device   = -1;
    ggml_layout_mode            actual_layout   = GGML_LAYOUT_AOS;
    bool                        cpu_accessible  = false;
    bool                        has_ready_event = false;
    sycl::event                 ready_event;
    expert_resolve_reason       reason = expert_resolve_reason::NOT_FOUND;
    std::unique_ptr<mem_handle> lifetime;

    explicit operator bool() const { return ptr != nullptr && reason == expert_resolve_reason::FOUND; }
};

// --- Expert Popularity Ranking ---
// Lightweight popularity tracking: the cache IS the placement.
// These functions replace the former ExpertPlacementTable.
// Residency is queried via is_expert_resident() / get_expert_device_ptr()
// which check the unified cache directly.

// Get popularity rank for a specific expert (0 = most popular, -1 = unranked).
// Called by cache eviction scoring to boost popular experts.
int get_expert_popularity_rank(int layer_id, int expert_id);

// Set popularity rank for a specific expert (called after warmup profiling).
void set_expert_popularity_rank(int layer_id, int expert_id, int rank);

// Returns true if any popularity ranks have been set.
bool is_expert_popularity_initialized();

// std::atomic<uint32_t> is non-copyable, but `unified_cache_entry` is used
// throughout the cache via copy-assignment into `entries_[key] = entry`.  Wrap
// the lease refcount in a copy-preserving adapter so the surrounding struct
// stays copyable.  The value of the counter is carried across copies (which
// only ever happen at fresh-insert sites where the count is 0).  Lock-free
// reads/updates use acquire/release so a writer racing with an evictor's
// `load()` from under the unique rw_mutex_ sees the reader's bump.
//
// Invariant (mem_handle lifetime): as long as any mem_handle's `leased_entry_`
// points at an entry, its `in_use_count > 0`.  Eviction paths MUST check
// `in_use_count.load(acquire) == 0` before erasing the entry; otherwise the
// handle's pointer dangles and subsequent resolve() / DNNL dispatch faults
// (llama.cpp-vtf7f / a7l5w crash signature).
struct copyable_atomic_u32 {
    std::atomic<uint32_t> v{ 0 };

    copyable_atomic_u32() = default;

    copyable_atomic_u32(const copyable_atomic_u32 & o) : v(o.v.load(std::memory_order_relaxed)) {}

    copyable_atomic_u32(copyable_atomic_u32 && o) noexcept : v(o.v.load(std::memory_order_relaxed)) {}

    copyable_atomic_u32 & operator=(const copyable_atomic_u32 & o) {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    copyable_atomic_u32 & operator=(copyable_atomic_u32 && o) noexcept {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    uint32_t fetch_add(uint32_t n) { return v.fetch_add(n, std::memory_order_acq_rel); }

    uint32_t fetch_sub(uint32_t n) { return v.fetch_sub(n, std::memory_order_acq_rel); }

    uint32_t load() const { return v.load(std::memory_order_acquire); }

    void store(uint32_t x) { v.store(x, std::memory_order_release); }
};

// Forward declarations needed by cache entry owner fields and friend function
// signatures inside unified_cache.
struct alloc_handle;
struct alloc_request;

// Metadata for a cached entry
struct unified_cache_entry {
    void *                device_ptr;       // GPU memory pointer (or host memory if host_resident)
    const void *          src_ptr;          // Source data pointer (for change detection)
    uint64_t              content_hash;     // Simple hash of content (first/last bytes)
    size_t                size;             // Size in bytes
    cache_entry_type      type;             // Dense or MoE
    int                   layer_id;         // Layer ID
    int                   expert_id;        // Expert ID (-1 for dense)
    ggml_layout_mode      layout;           // Target layout for this entry
    int64_t               onednn_pack_m;    // M dimension used for ONEDNN_PACKED/ONEDNN_WOQ (0 when unused)
    cache_layout_xmx_info xmx_info;         // XMX metadata (when applicable)
    uint32_t              access_count;     // Access frequency for LFU
    int64_t               last_access;      // Timestamp for recency
    bool                  pinned;           // Protected from eviction
    bool                  hot;              // Hot-set hint for MoE experts
    cache_entry_state     state;            // READY vs IN_PROGRESS
    bool                  has_ready_event;  // True if ready_event is valid
    sycl::event           ready_event;      // Completion event for IN_PROGRESS entries
    bool                  host_resident;    // Entry lives in host memory, not device (fallback when VRAM full)
    cache_location        location;         // DEVICE/HOST_PINNED/HOST_MMAP
    bool                  pool_allocated;   // True if device_ptr was sub-allocated from layout_pool_
    bool                  cache_budget_charged = false;  // True if this entry incremented cache used_
    bool                  retired              = false;  // Hidden from routing; erased after in_use_count reaches zero.
    sycl::event           last_write_event;              // Event from last fill/reorder that wrote to device_ptr
    bool                  has_write_event = false;       // Whether last_write_event is valid
    // Optional shared allocation owner for entries that are views into a
    // larger cache-owned allocation, such as tensor-bulk MoE materialization.
    // When non-null, individual entries must not free device_ptr directly; the
    // shared owner releases the allocation base after the last view is erased.
    std::shared_ptr<void> storage_owner;
    // Optional allocation owner for single-entry direct materializations that
    // are not arena or pool suballocations.
    mem_handle            direct_alloc_owner;
    // Async eviction state (P7): set when state == EVICTING
    sycl::event           eviction_event;                // D2H copy completion event
    bool                  has_eviction_event = false;    // True if eviction_event is valid
    void *                eviction_host_ptr  = nullptr;  // Host-pinned destination for D2H copy
    // Lifetime refcount (llama.cpp-vtf7f): bumped by mem_handle lease,
    // decremented by mem_handle release.  Eviction paths skip entries with
    // `in_use_count > 0`.  The counter is reset to 0 when the entry is
    // overwritten at fresh-insert sites (always safe because those sites
    // replace a just-erased / never-existed key; any attempt to overwrite a
    // leased entry would be a bug, asserted separately in eviction paths).
    copyable_atomic_u32   in_use_count;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here
};

// Weight set for a transformer layer (for bulk pinning)
// Supports standard dense transformer architecture with attention + FFN blocks.
// For MoE models, use pin/unpin directly with expert_id.
struct layer_weight_set {
    ggml_sycl_cache_id attn_norm;                       // Attention layer norm
    ggml_sycl_cache_id q_proj, k_proj, v_proj, o_proj;  // Attention projections
    ggml_sycl_cache_id ffn_norm;                        // FFN layer norm
    ggml_sycl_cache_id gate_proj, up_proj, down_proj;   // FFN projections (SwiGLU)
    // Optional: Some architectures have additional weights
    ggml_sycl_cache_id attn_qkv_proj;     // Fused QKV for some models
    ggml_sycl_cache_id ffn_gate_up_proj;  // Fused gate+up for some models
};

// Prefetch priority for async layer prefetch queue.
// Defined locally to avoid circular dependency with unified-kernel.hpp.
enum class prefetch_priority { LOW, NORMAL, HIGH };

// Result of await_layer - contains device pointers to all layer weights.
// Populated by looking up each weight in the cache after prefetch completes.
struct layer_weight_pointers {
    void * attn_norm        = nullptr;
    void * q_proj           = nullptr;
    void * k_proj           = nullptr;
    void * v_proj           = nullptr;
    void * o_proj           = nullptr;
    void * ffn_norm         = nullptr;
    void * gate_proj        = nullptr;
    void * up_proj          = nullptr;
    void * down_proj        = nullptr;
    void * attn_qkv_proj    = nullptr;  // Fused QKV (optional)
    void * ffn_gate_up_proj = nullptr;  // Fused gate+up (optional)
};

// Unified GPU cache for both dense weights and MoE experts
//
// Design principles:
// 1. Single memory budget - avoids OOM from competing caches
// 2. Type-aware eviction - dense weights and experts have equal priority
// 3. Automatic partitioning - no manual budget splitting needed
// 4. LFU+LRU hybrid eviction - keeps frequently used entries
//
// Usage:
// - call ensure_cached() for any weight (dense or expert)
// - cache automatically evicts lowest-scoring entries when full
// - for MoE: call with layer_id and expert_id
// - for dense: call with layer_id=-1, expert_id=-1
class unified_cache {
  public:
    // Initialize with SYCL queue and memory budget
    // budget_bytes: total GPU memory for caching (dense + MoE combined)
    // staging_size: pinned host staging buffer size (default 64MB)
    // device_total_vram: total VRAM bytes for the device, forwarded directly
    //   to arena_reserve() to compute the runtime headroom (see
    //   llama.cpp-w2ptt).  Passed in by the caller because
    //   `ggml_backend_sycl_get_device_memory()` dereferences
    //   `ggml_sycl_info()`, which deadlocks if invoked while the cache is
    //   being constructed mid `ggml_sycl_init()` static init.  Pass 0 to opt
    //   out of the headroom cap entirely (the arena will be reserved at the
    //   caller's full budget).  Not stored as a member: the value is only
    //   meaningful at arena-reservation time and storing it would silently
    //   capture a stale value if arena_reserve were ever called post-init.
    unified_cache(sycl::queue & queue,
                  size_t        budget_bytes,
                  size_t        staging_size       = 64 * 1024 * 1024,
                  size_t        dma_reserved_bytes = 0,
                  size_t        device_total_vram  = 0);
    ~unified_cache();

    // Non-copyable, non-movable
    unified_cache(const unified_cache &)             = delete;
    unified_cache & operator=(const unified_cache &) = delete;
    unified_cache(unified_cache &&)                  = delete;
    unified_cache & operator=(unified_cache &&)      = delete;

    // === Primary API ===

    // Result of a layout-agnostic weight pointer lookup.
    struct weight_ptr_result {
        void *           ptr             = nullptr;
        ggml_layout_mode layout          = GGML_LAYOUT_AOS;
        bool             on_device       = false;
        bool             has_ready_event = false;
        sycl::event      ready_event;

        explicit operator bool() const { return ptr != nullptr; }
    };

    // Fast O(1) weight lookup.  Tries the entry for this key regardless of layout.
    // Returns the first READY entry found.  Does NOT create entries or trigger staging.
    weight_ptr_result get_weight_ptr(const ggml_sycl_cache_id & key);

    // Layout-agnostic weight pointer lookup that ALSO pins the lease refcount
    // on the underlying cache entry.  Returns entry pointer in `entry` for the
    // caller (mem_handle) to release on destruction.  The caller MUST release
    // by calling `entry->in_use_count.fetch_sub(1)` exactly once — otherwise
    // the entry cannot be evicted, memory pressure will grow, and eviction
    // will start failing.  If the result is falsy (ptr == nullptr), no lease
    // was acquired and `entry == nullptr`.
    //
    // This is the refcount-safe entry point for mem_handle::resolve_slow().
    // See llama.cpp-vtf7f rootcause for the lifetime contract.
    struct weight_ptr_lease_result {
        void *                ptr             = nullptr;
        ggml_layout_mode      layout          = GGML_LAYOUT_AOS;
        bool                  on_device       = false;
        unified_cache_entry * entry           = nullptr;  // opaque handle for lease release
        bool                  has_ready_event = false;
        sycl::event           ready_event;

        explicit operator bool() const { return ptr != nullptr; }
    };

    weight_ptr_lease_result acquire_weight_lease(const ggml_sycl_cache_id & key);
    weight_ptr_lease_result acquire_entry_lease(const unified_cache_key & key);

    // --- Decomposed cache operations (no queue ops during inference) ---

    // Allocate a VRAM slot for a cache entry. May evict LRU entries.
    // Returns device pointer or nullptr if VRAM is full.
    // NO queue operations. NO DMA. NO fill. Just memory management.
    // If entry already exists with matching layout, returns existing pointer.
    void * allocate_slot(const ggml_sycl_cache_id & key,
                         size_t                     size,
                         ggml_layout_mode           layout,
                         cache_entry_type           type      = cache_entry_type::DENSE_WEIGHT,
                         int                        layer_id  = -1,
                         int                        expert_id = -1);

    // Mark a previously allocated slot as READY.
    // After this call, lookup() will find the entry.
    // Pure metadata update — NO queue operations.
    void register_ready(const ggml_sycl_cache_id & key,
                        void *                     device_ptr,
                        ggml_layout_mode           layout,
                        size_t                     size,
                        cache_entry_type           type          = cache_entry_type::DENSE_WEIGHT,
                        int                        layer_id      = -1,
                        int                        expert_id     = -1,
                        const void *               src_ptr       = nullptr,
                        int64_t                    onednn_pack_m = 0);

    // Lock-free read lookup. Returns device pointer if entry is READY with
    // matching layout. Returns nullptr on miss.
    // NO allocation, NO fill, NO blocking. Same semantics as try_get_cached_fast.
    // NOTE: Excludes HOST_MMAP entries (raw mmap pointers, not GPU-accessible).
    //       Includes HOST_PINNED entries (sycl::malloc_host, GPU-accessible via PCIe).
    void * lookup(const ggml_sycl_cache_id & key, ggml_layout_mode layout);

    // Like lookup(), but only returns device-resident (VRAM) pointers.
    // Excludes both HOST_PINNED and HOST_MMAP entries.
    // Use when the pointer MUST be in device VRAM (e.g., for expert dispatch
    // where host-pinned zero-copy would be too slow or where the caller name
    // implies device residency).
    void * lookup_device_only(const ggml_sycl_cache_id & key, ggml_layout_mode layout);

    // --- DMA queue for cache operations (separate from compute) ---
    sycl::queue & get_dma_queue();

    // --- BCS queue for copy-only H2D transfers (targets copy engine) ---
    // Returns the BCS (copy-only) queue if available, otherwise falls back to
    // the DMA queue.  BCS targets queue group ordinal 1 (COPY engine) on Intel
    // GPUs, keeping H2D memcpy off the CCS (compute) engine so SOA reorder
    // kernels can run concurrently without monopolizing CCS preempt budget.
    sycl::queue & get_bcs_queue();

    // Ensure a weight is cached, loading from src_ptr if needed
    // Returns device pointer, or nullptr if cache is full and eviction failed
    //
    // key: Stable identifier for cache lookup (no pointers, no layout)
    //   - For dense weights: GGUF data identity + tensor metadata (model_id scoped)
    //   - For MoE experts: model_id + aux_id (cache_uuid) + expert_id/layer_id
    // src_ptr: Source data to copy from (may change for same key)
    // If src_ptr differs from cached entry's src_ptr, data is re-uploaded
    //
    // For dense weights: layer_id=-1, expert_id=-1
    // For MoE experts: layer_id=N, expert_id=M
    void * ensure_cached(const ggml_sycl_cache_id & key,
                         const void *               src_ptr,
                         size_t                     size,
                         cache_entry_type           type,
                         int                        layer_id         = -1,
                         int                        expert_id        = -1,
                         ggml_layout_mode           layout           = GGML_LAYOUT_AOS,
                         bool                       validate_content = false);
    // === Direct Staging API ===
    // Simple arena allocate + fill + register.  No state machine.

    // Stage a dense weight: allocate from WEIGHT zone, fill (reorder or memcpy),
    // store in lookup table for inference-time resolution.
    // fill_fn: if non-null, called for AOS->SOA reorder; otherwise plain memcpy.
    // queue: BCS or CCS queue for the DMA/fill submission.
    direct_stage_result direct_stage_weight(ggml_sycl_cache_id   key,
                                            const void *         src_ptr,
                                            size_t               src_size,
                                            size_t               dst_size,
                                            ggml_layout_mode     layout,
                                            cache_layout_fill_fn fill_fn,
                                            const void *         fill_ctx,
                                            sycl::queue *        queue,
                                            mem_handle *         out_handle = nullptr);

    // Stage an expert tensor: same semantics as direct_stage_weight but uses
    // a separate lookup table for MoE experts.
    direct_stage_result direct_stage_expert(ggml_sycl_cache_id   key,
                                            const void *         src_ptr,
                                            size_t               src_size,
                                            size_t               dst_size,
                                            ggml_layout_mode     layout,
                                            cache_layout_fill_fn fill_fn,
                                            const void *         fill_ctx,
                                            sycl::queue *        queue,
                                            mem_handle *         out_handle = nullptr);

    // Stage a contiguous run of experts from one MoE tensor into one
    // cache-owned WEIGHT allocation, then publish per-expert smart handles as
    // views into that allocation. This reduces model-load command count while
    // preserving per-expert resolution and refcount semantics.
    direct_stage_result direct_stage_expert_tensor(const std::vector<ggml_sycl_cache_id> & keys,
                                                   const void *                            src_ptr,
                                                   size_t                                  src_size,
                                                   size_t                                  expert_dst_size,
                                                   ggml_layout_mode                        layout,
                                                   cache_layout_fill_fn                    fill_fn,
                                                   const void *                            fill_ctx,
                                                   sycl::queue *                           queue,
                                                   std::vector<mem_handle> *               out_handles = nullptr);

    // Drop one cache-owned expert entry after all smart-handle leases have
    // been released. This is used for planner-owned PP/TG layout phase
    // switches; it refuses to free live handles and erases only transient raw
    // pointer mirrors after the canonical entry is gone.
    bool drop_expert_entry(ggml_sycl_cache_id key, const char * reason = nullptr);

    // Drop the given experts' entries whose materialized layout matches,
    // releasing the direct-table insert handles that pin them first.
    // Returns the number of cache entries dropped. Used by the phase-layout
    // swap to release the SOA residency after a successful XMX_TILED
    // rematerialization.
    size_t drop_expert_entries_for_tensor_layout(const std::vector<ggml_sycl_cache_id> & expert_keys,
                                                 ggml_layout_mode                        layout,
                                                 const char *                            reason = nullptr);

    // Fast O(1) lookup for inference-time weight resolution.
    // Returns nullptr if not staged.  No allocation, no state machine.
    const weight_entry * lookup_weight(ggml_sycl_cache_id key) const;
    const weight_entry * lookup_expert(ggml_sycl_cache_id key) const;

    // Resolve an MoE expert tensor across both unified cache entries_ and
    // direct_expert_entries_.  The result gives callers explicit tier/device
    // metadata so dispatch code does not need SYCL pointer-type probes.
    expert_resolve_result resolve_expert(const expert_resolve_request & req);

    // Register a host-arena pointer directly as a HOST_PINNED expert entry.
    // ptr must be host-pinned memory (typically from host_zone_alloc(WEIGHT)).
    // No zone_alloc, no device copy.  Used by S1-PRELOAD for host-planned experts.
    void register_host_expert(ggml_sycl_cache_id key,
                              void *             ptr,
                              size_t             size,
                              ggml_layout_mode   layout,
                              mem_handle *       out_handle = nullptr);

    // Register a host-arena pointer directly as a HOST_PINNED dense weight entry.
    // ptr must be host-pinned memory (typically from host_zone_alloc(WEIGHT)).
    // No zone_alloc, no device copy.  Used by S1-PRELOAD for host-planned dense weights.
    void register_host_weight(ggml_sycl_cache_id key,
                              void *             ptr,
                              size_t             size,
                              ggml_layout_mode   layout,
                              mem_handle *       out_handle = nullptr);

    // Placement-plan materialization authority.  In planned mode direct staging
    // and host registration are only legal while model load or an explicit
    // planner-owned migration holds this token; inference must resolve existing
    // mem_handles and route by their actual residency.
    void begin_planned_materialization(const char * reason);
    void end_planned_materialization();
    bool planned_materialization_active() const;

    // === Multi-Device Partial Row Loading ===

    // Load a contiguous row range of a weight tensor to this device with SOA reorder.
    // Used by 3-device tensor split: each device stores only its assigned rows.
    // The source data is AOS from host (mmap-backed), and the result is SOA on device.
    // Returns device pointer to the SOA-reordered partial tensor, or nullptr on failure.
    //
    // tensor_name: tensor name for cache key generation
    // src_host:    host pointer to the START of the row range (already offset by caller)
    // type:        quantized type (Q4_0, Q8_0, etc.)
    // ncols:       number of columns (ne[0]) in the tensor
    // row_start:   first row in the original full tensor
    // row_count:   number of rows in this partial range
    // device_idx:  resolved ggml SYCL device index
    void * load_partial_rows(const char *               tensor_name,
                             const ggml_sycl_cache_id & tensor_id,
                             const void *               src_host,
                             ggml_type                  type,
                             int64_t                    ncols,
                             int64_t                    row_start,
                             int64_t                    row_count,
                             int                        device_idx);

    // Look up a previously loaded partial weight for a given tensor/device/range.
    // Returns the device pointer, or nullptr if not loaded yet.
    void * get_split_weight_ptr(const char *               tensor_name,
                                const ggml_sycl_cache_id & tensor_id,
                                ggml_type                  type,
                                int64_t                    ncols,
                                int64_t                    row_start,
                                int64_t                    row_count,
                                int                        device_idx);

    // Free all partial row entries (called during cache shutdown or device cleanup).
    void free_partial_entries();

    // Check if entry is cached (without loading)
    bool is_cached(const ggml_sycl_cache_id & key, ggml_layout_mode layout) const;
    bool is_cached_any(const ggml_sycl_cache_id & key) const;

    // Get device pointer for cached entry (returns nullptr if not cached)
    void *         get(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    cache_ptr_view get_view(const ggml_sycl_cache_id & key, ggml_layout_mode layout);

    // Get device pointer, waiting for IN_PROGRESS entries to complete.
    // Fast path for cache lookup using shared_lock (reader-writer lock).
    // Returns device_ptr if entry exists, is READY, and layout matches.
    // Otherwise returns nullptr. Does not update LRU/LFU stats.
    void * try_get_cached_fast(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout);

    // Like try_get_cached_fast, but also returns device_ptr for IN_PROGRESS
    // entries whose fill is still running.  When the entry is IN_PROGRESS and
    // has a ready_event, *out_event is set and *out_has_event is true so the
    // caller can chain on it.
    // Returns nullptr only when the entry does not exist or layout mismatches.
    void * try_get_cached_with_event(const ggml_sycl_cache_id & key_id,
                                     ggml_layout_mode           layout,
                                     sycl::event *              out_event,
                                     bool *                     out_has_event);

    // get_by_data_ptr REMOVED — O(N) scan with zero callers (dead code)

    // Remove a cache entry (free device memory).
    void remove(const ggml_sycl_cache_id & key,
                cache_entry_type           type,
                int                        layer_id,
                int                        expert_id,
                ggml_layout_mode           layout);

    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache

    // === Atomic Expert Group Staging/Eviction ===
    // All 3 tensors (gate, up, down) stage or evict together -- never partial.

    // Stage all 3 tensors for an expert atomically.
    // If VRAM is insufficient, evicts cold expert groups first.
    // If any allocation fails after eviction, rolls back partial allocations.
    // Each staging_tensor_data provides source pointer and sizes.
    // Each cache_layout_request provides fill_fn/fill_ctx for reorder.
    // When request pointers are non-null, fill_fn is called per tensor.
    // When null, raw DMA memcpy via get_dma_queue() is used.
    // Returns true if all 3 tensors were staged successfully.
    bool stage_expert_group(int                          block_id,
                            int                          expert_id,
                            const expert_tensor_group &  keys,
                            const staging_tensor_data &  gate_data,
                            const staging_tensor_data &  up_data,
                            const staging_tensor_data &  down_data,
                            ggml_layout_mode             layout,
                            const cache_layout_request * gate_req = nullptr,
                            const cache_layout_request * up_req   = nullptr,
                            const cache_layout_request * down_req = nullptr);

    // Evict all 3 tensors for an expert atomically.
    // Removes gate, up, and down from the cache together.
    void evict_expert_group(const expert_tensor_group & keys, ggml_layout_mode layout);

    // Evict the coldest expert group to free VRAM.
    // Uses combined access frequency across all 3 tensors as coldness metric.
    // expert_groups: the global registry mapping (block,expert) -> keys.
    // Returns bytes freed, or 0 if no evictable expert group was found.
    size_t evict_coldest_expert_group(const std::unordered_map<int64_t, expert_tensor_group> & expert_groups,
                                      ggml_layout_mode                                         layout);

    // === Pinning for Graphs ===

    void pin(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    void unpin(const ggml_sycl_cache_id & key, ggml_layout_mode layout);
    void unpin_experts();
    void unpin_all();
    bool is_pinned(const ggml_sycl_cache_id & key, ggml_layout_mode layout) const;

    // === Bulk Pinning for Persistent Kernels ===
    // Pin all weights for a layer at once. Returns count of successfully pinned entries.
    // Only pins entries that exist in cache with matching layout.
    int pin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout);

    // Unpin all weights for a layer. Requires the same weight set used for pinning.
    void unpin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout);

    // Pin entire model weights (all layers). Returns total count of pinned entries.
    // layers: vector of layer_weight_set for each layer (index = layer_id)
    int pin_model_weights(int n_layers, const std::vector<layer_weight_set> & layers, ggml_layout_mode layout);

    // === Async Layer Prefetch for Persistent Kernels ===
    // Queue a layer for background prefetch (pins weights so they won't be evicted).
    // The prefetch worker thread will pin all weights for the layer and mark it ready.
    // Priority HIGH requests are placed at the front of the queue.
    void queue_layer_prefetch(int                      layer_id,
                              const layer_weight_set & weights,
                              ggml_layout_mode         layout,
                              prefetch_priority        priority = prefetch_priority::NORMAL);

    // Block until the specified layer is prefetched and ready.
    // Returns pointers to all cached weights for the layer.
    layer_weight_pointers await_layer(int layer_id);

    // Get the weight set (cache IDs) for a layer that has been prefetched.
    // Returns true and fills `out` if found, false otherwise.
    bool get_layer_weight_set(int layer_id, layer_weight_set & out) const;

    // Check if a layer has been prefetched and is ready (non-blocking).
    bool is_layer_ready(int layer_id) const;

    // Release a prefetched layer (unpins weights, allows eviction).
    void release_layer(int layer_id);

    // === Memory Management ===

    // Current memory used
    size_t used() const { return used_.load(); }

    // Total budget
    size_t budget() const { return budget_; }

    // Check if budget is exceeded (used > budget after eviction)
    bool is_budget_exceeded() const { return budget_exceeded_; }

    // Returns true if any weight has been evicted from device memory.
    // Once true, never resets. Used to disable graph replay with stale baked pointers.
    bool has_evictions() const { return has_evictions_.load(std::memory_order_acquire); }

    // Available memory
    size_t available() const {
        const size_t used = used_.load();
        return budget_ > used ? budget_ - used : 0;
    }

    // Fraction of budget currently used (0.0 to 1.0+)
    float budget_utilization() const {
        return budget_ > 0 ? static_cast<float>(used_.load(std::memory_order_relaxed)) / static_cast<float>(budget_) :
                             0.0f;
    }

    // Validate an optimized-path residency request against current WEIGHT-zone
    // capacity/fragmentation.  Rejection is structured and side-effect free:
    // no partial materialization, no forced eviction, and no forced free of live
    // handles.  Accepted plans retain any handles supplied by the request so
    // callers can attach the plan to descriptor/pointer-table lifetime.
    residency_plan reserve_residency(const residency_request & req);

    // Drain pending deferred frees at safe sync points (public accessor).
    // NEVER call during graph_compute — sycl::free() unmaps GPU page table
    // entries while in-flight kernels may still reference those addresses.
    // The ONLY safe time is after queue.wait() in ggml_backend_sycl_synchronize()
    // or during prestage yield loops (outside graph_compute_impl).
    void process_deferred_frees_public() {
        if (unified_cache_is_graph_compute_active()) {
            return;  // Kernels in-flight — defer until synchronize()
        }
        process_deferred_frees();
    }

    // Check if there are any pending deferred frees (device or host)
    bool has_pending_deferred_frees() const;

    // Raw VRAM budget before runtime reservations
    size_t base_budget() const { return base_budget_; }

    // Bytes currently occupied by cached weights
    size_t weight_bytes() const { return used_.load(); }

    // Available VRAM for non-weight allocations (KV, compute, staging).
    // This is the budget headroom after weights + current runtime reservations.
    // Higher-level code uses this to size KV cache and compute buffers.
    size_t available_for_compute() const {
        return available();  // budget_ - used_, already accounts for reserved_
    }

    // Force eviction to free at least bytes_needed
    // Returns actual bytes freed
    size_t evict(size_t bytes_needed);

    // Evict cache entries and synchronously flush deferred frees so that
    // used_ is decremented and VRAM is truly released.  Unlike evict(),
    // which defers actual sycl::free behind barrier events, this method
    // waits on the queue and processes deferred frees before returning.
    // Used by unified_alloc() to make room for runtime allocations.
    size_t evict_and_flush(size_t bytes_needed);

    // --- Async DMA eviction (P7) ---

    // Poll in-flight D2H evictions and finalize completed ones.
    // Reclaims arena/device space, removes device entry.
    // Call at safe sync points
    // (between graph_compute calls, after queue drain).
    // Returns number of entries finalized.
    size_t finalize_evictions();
    size_t finalize_evictions_locked();  // Caller must hold rw_mutex_ (unique)

    // Re-promote a weight back to device VRAM.
    // With plan-driven placement, re-promotion is handled by the
    // normal ensure_cached path (re-reads from mmap).  Returns nullptr.
    void * promote_to_device(const unified_cache_key & key, size_t size);

    // === Stats ===

    size_t hits() const { return hits_.load(); }

    size_t misses() const { return misses_.load(); }

    float hit_rate() const {
        size_t total = hits_.load() + misses_.load();
        return total > 0 ? float(hits_.load()) / total : 0.0f;
    }

    size_t dense_count() const;
    size_t expert_count() const;
    size_t used_bytes(cache_entry_type type) const;

    // Number of entries in the cache (for diagnostics).
    size_t entry_count() const {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        return entries_.size();
    }

    // Returns total VRAM bytes used by weight entries for a specific layer.
    // Used by KV tier manager to co-locate KV with layer weights.
    size_t get_layer_vram_bytes(int layer_id) const;

    // Returns total VRAM bytes held by unpinned MoE expert entries that
    // could be evicted to make room for KV layer promotion.
    size_t evictable_expert_bytes() const;

    void print_stats() const;

    // Seal the layout pool to prevent new chunk allocations after S1-PRELOAD.
    // All dense weights are loaded by then; further layout requests (e.g. late
    // MoE experts) that don't fit in existing chunks will fall back to
    // individual allocations instead of wasting a full 256 MB chunk.
    // Pre-allocate device pool chunks for `total_bytes` of layout allocations.
    // Must be called BEFORE any BCS H2D copies to avoid sycl::malloc_device
    // stalling the BCS engine.  Returns number of new chunks allocated.
    // Charges each chunk incrementally to the VRAM budget so that
    // available-space queries reflect the reduced capacity.
    size_t pre_grow_layout_pool(size_t total_bytes) {
        if (!layout_pool_) {
            return 0;
        }
        const size_t chunk_size = layout_pool_->get_default_chunk_size();
        // Subtract existing free space in pool
        size_t       pool_free  = 0;
        {
            size_t total_chunk = layout_pool_->total_chunk_bytes();
            size_t total_used  = layout_pool_->total_used_bytes();
            pool_free          = (total_chunk > total_used) ? (total_chunk - total_used) : 0;
        }
        if (pool_free >= total_bytes) {
            return 0;
        }
        const size_t deficit       = total_bytes - pool_free;
        const size_t chunks_needed = (deficit + chunk_size - 1) / chunk_size;
        size_t       chunks_grown  = 0;
        for (size_t i = 0; i < chunks_needed; i++) {
            // Check budget before each chunk.
            size_t avail =
                (budget_ > used_.load(std::memory_order_relaxed)) ? budget_ - used_.load(std::memory_order_relaxed) : 0;
            if (chunk_size > avail) {
                GGML_LOG_WARN("[UNIFIED-CACHE] pre_grow: budget exhausted after %zu/%zu chunks\n", chunks_grown,
                              chunks_needed);
                break;
            }
            size_t physical = layout_pool_->grow_one_chunk();
            if (physical == 0) {
                break;
            }
            // Charge immediately so the next iteration sees reduced capacity.
            used_.fetch_add(physical, std::memory_order_relaxed);
            chunks_grown++;
        }
        if (chunks_grown > 0) {
            GGML_LOG_INFO("[DEVICE-POOL] pre_grow: %zu/%zu chunks (%.1f MB) for %.1f MB working set\n", chunks_grown,
                          chunks_needed, chunks_grown * chunk_size / (1024.0 * 1024.0),
                          total_bytes / (1024.0 * 1024.0));
        }
        return chunks_grown;
    }

    void seal_layout_pool() {
        if (layout_pool_) {
            layout_pool_->seal();
        }
    }

    // === Placement Plan (P4) ===
    // Priority-based static weight placement computed at model load time.
    // When active, S1-PRELOAD uses the plan to decide VRAM vs host placement
    // instead of first-come-first-served.  Gated by GGML_SYCL_VRAM_ARENA=1.

    void set_placement_plan(placement_plan && plan) {
        placement_plan_     = std::move(plan);
        has_placement_plan_ = true;
    }

    void update_placement_plan_runtime_kv(uint32_t n_ctx, size_t kv_per_layer, size_t kv_per_swa_layer) {
        if (!has_placement_plan_) {
            return;
        }
        placement_plan_.update_runtime_kv_sizes(n_ctx, kv_per_layer, kv_per_swa_layer);
    }

    bool plan_on_device(const std::string & tensor_name) const {
        if (!has_placement_plan_) {
            return true;
        }
        return placement_plan_.is_on_device(tensor_name);
    }

    // P4.5: Check if tensor is assigned to THIS device in a multi-device plan.
    bool plan_on_this_device(const std::string & tensor_name, int this_device) const {
        if (!has_placement_plan_) {
            return true;
        }
        if (!placement_plan_.multi_device) {
            // Single-device plan: original behavior
            return placement_plan_.is_on_device(tensor_name);
        }
        return placement_plan_.is_on_device(tensor_name, this_device);
    }

    bool has_placement_plan() const { return has_placement_plan_; }

    // Returns true if any expert of a MoE tensor is planned for host memory on device_id.
    // Replaces ad-hoc static cache maps in dispatch code: placement_plan::expert_index_ is
    // already O(1) per lookup, so no caller-side memoization is needed.
    bool moe_tensor_has_host_experts(const std::string & tensor_name, int64_t n_experts, int device_id = -1) const {
        if (!has_placement_plan_) {
            return false;  // No plan → all experts are device-resident (default)
        }
        return placement_plan_.has_host_experts(tensor_name, n_experts, device_id);
    }

    const placement_plan & get_placement_plan() const { return placement_plan_; }

    bool demote_expert_placement_to_host(const std::string & tensor_name, int expert_id) {
        if (!has_placement_plan_) {
            return false;
        }
        return placement_plan_.demote_expert_placement_to_host(tensor_name, expert_id);
    }

    // Access the internal SYCL queue (for deferred free of temp allocations
    // made on this queue's context, e.g. GPU-side reorder temp buffers).
    sycl::queue & get_queue() { return queue_; }

    // Memset/memcpy operations that dispatch to GPU or CPU based on handle location.
    void memset(const mem_handle & h, int value, size_t size, sycl::queue & stream);
    void memcpy(const mem_handle & dst, const mem_handle & src, size_t size, sycl::queue & stream);

    // === Arena methods (merged from vram_arena) ===

    // Reserve VRAM arena using 1- or N-chunk allocation.  device_total_vram
    // is passed by the caller because the ctor may run inside ggml_sycl_init()
    // static init, where dereferencing ggml_sycl_info() to query total VRAM
    // would deadlock.  Pass 0 to fall back to "size by budget alone".
    bool arena_reserve(sycl::queue & queue,
                       size_t        budget_bytes,
                       size_t        max_alloc_size,
                       size_t        safe_max_alloc_size,
                       size_t        scratch_bytes,
                       size_t        onednn_bytes,
                       size_t        runtime_bytes,
                       size_t        device_total_vram);

    // Ensure the VRAM arena's fixed zones match the latest planner inputs.
    // If an early arena exists but has no live zone allocations or chunk
    // leases, it may be rebuilt with larger planned zones. Live arenas are
    // never force-reset or force-freed.
    bool ensure_planned_arena_zones();

    // Is arena active?
    bool arena_active() const { return arena_base_ != nullptr; }

    // Base pointer.
    void * arena_base() const { return arena_base_; }

    // Total arena size.
    size_t arena_total_size() const { return arena_size_; }

    // (zone_alloc / zone_free / zone_reset are private — use
    //  unified_cache_zone_alloc / unified_cache_zone_reset free functions)

    // Check if a pointer belongs to the VRAM arena.
    bool vram_owns(const void * ptr) const;

    // Check if a pointer belongs to a specific zone.
    bool zone_owns(vram_zone_id zone, const void * ptr) const;

    // Convert pointer to arena offset (SIZE_MAX if not owned).
    size_t ptr_to_offset(const void * ptr) const;

    // Convert arena offset to pointer.
    void * offset_to_ptr(size_t offset) const;

    // Zone capacity and usage.
    size_t zone_capacity(vram_zone_id zone) const;
    size_t zone_used(vram_zone_id zone) const;
    size_t zone_available(vram_zone_id zone) const;
    size_t zone_largest_free(vram_zone_id zone) const;
    void   dump_live_zone_allocations(vram_zone_id zone, const char * where, size_t max_entries = 32) const;

    const vram_zone & get_zone(vram_zone_id zone) const { return arena_zones_[static_cast<int>(zone)]; }

    // Number of chunks.  Usually 1 (a single sycl::malloc_device covers the
    // budget); >1 only on systems where the runtime enforces a hard
    // per-allocation cap below the budget and the single-chunk attempt
    // fails (e.g. stock libze on smaller GPUs).
    int chunk_count() const { return static_cast<int>(arena_chunks_.size()); }

    // === Arena chunk lease API (llama.cpp-dyhdl) ===
    // Reference-count VRAM arena chunks while any mem_handle holds a raw
    // pointer derived from them.  Complementary to the vtf7f cache_entry
    // refcount: cache_entry prevents cache-layer invalidation; chunk lease
    // prevents sycl::free of the underlying VRAM allocation.
    //
    // Returns -1 when the pointer is not owned by the arena (safe no-op on
    // release).  Otherwise returns the chunk index (0..N-1).
    int      arena_find_chunk(const void * ptr) const;
    int      arena_acquire_chunk_lease(const void * ptr);
    void     arena_release_chunk_lease(int chunk_idx);
    bool     arena_chunk_has_leases(int chunk_idx) const;
    uint32_t arena_chunk_lease_count(int chunk_idx) const;

    // Host pinned-pool chunk lease API (dyhdl) — routes through host_arena_.
    // Returns pinned_chunk_pool::INVALID_CHUNK_HANDLE (UINT64_MAX) on miss.
    // Safe to call even when host_arena_ is null (returns miss).
    uint64_t host_acquire_chunk_lease(const void * ptr);
    void     host_release_chunk_lease(uint64_t handle);

    // Destroy arena (free all chunks).
    void arena_destroy();

    // Abandon arena without freeing (for shutdown when SYCL context is invalid).
    void arena_abandon();

    // Arena generation: monotonically increasing counter, bumped when the arena
    // is destroyed/recreated.  Used by mem_handle to detect stale arena handles.
    uint64_t arena_generation() const { return arena_generation_; }

    void arena_generation_bump() { ++arena_generation_; }

    // Register the compute queue so deferred frees wait for in-flight kernels.
    // Without this, evicted VRAM pointers can be freed while GPU kernels on the
    // compute queue still reference them → GPU page fault → DEVICE_LOST.
    void set_compute_queue(sycl::queue * q) { compute_queue_ = q; }

    // Called at the start of a new model load.  Model-load IDs are deliberately
    // unique, so stale direct/model weight entries from a previous load cannot
    // be resolved by the new graph and must not keep occupying the WEIGHT arena.
    void reset_model_weight_entries();

    void reset_stats();
    // Debug/testing helper: verify internal maps are consistent.
    bool validate() const;

    // Reserve non-cache runtime buffers (compute, KV, etc.)
    void update_reserved_bytes(size_t reserved_bytes);

    // Hot set control (MoE experts)
    void set_hot(const ggml_sycl_cache_id & key,
                 cache_entry_type           type,
                 int                        layer_id,
                 int                        expert_id,
                 ggml_layout_mode           layout,
                 bool                       hot);
    void clear_hot_experts(int layer_id);

    // Track cache entries pinned for in-flight kernels.
    void unpin_on_event(const ggml_sycl_cache_id & key, ggml_layout_mode layout, const sycl::event & event);

    using dma_stream_slice_fn = sycl::event (*)(sycl::queue &                    queue,
                                                void *                           device_slice,
                                                size_t                           slice_bytes,
                                                size_t                           offset_bytes,
                                                const void *                     ctx,
                                                const std::vector<sycl::event> & deps);
    using dma_stream_copy_fn  = sycl::event (*)(sycl::queue &                    queue,
                                               void *                           device_slice,
                                               size_t                           slice_bytes,
                                               size_t                           offset_bytes,
                                               const void *                     src_ptr,
                                               size_t                           src_size,
                                               const void *                     ctx,
                                               const std::vector<sycl::event> & deps);

    struct dma_staging_buffers {
        void ** buffers     = nullptr;
        size_t  count       = 0;
        size_t  slice_bytes = 0;
    };

    // Device-resident DMA staging buffer pool (for streaming).
    bool get_dma_staging_buffers(size_t slice_bytes, size_t count, dma_staging_buffers & out);

    struct dma_stream_result {
        sycl::event event;
        bool        ok                 = false;
        bool        used_mmap_direct   = false;
        bool        mmap_direct_failed = false;
        size_t      slices             = 0;
        size_t      slice_bytes        = 0;
        size_t      buffer_count       = 0;
    };

    dma_stream_result stream_dma(const cache_ptr_view &           src,
                                 size_t                           total_bytes,
                                 size_t                           slice_bytes,
                                 size_t                           buffer_count,
                                 dma_stream_slice_fn              slice_fn,
                                 const void *                     ctx,
                                 const std::vector<sycl::event> & deps,
                                 dma_stream_copy_fn               copy_fn = nullptr);

    // Defer freeing host allocations until the associated event completes.
    void defer_host_free(void * ptr, size_t size, const sycl::event & event);

    // === OneDNN FP16 Scratch Buffers ===
    // Pre-allocated buffers for dequantized weights and converted activations.
    // These avoid per-op allocations that cause OOM with large KV caches.

    struct onednn_scratch_buffers {
        void * weights          = nullptr;  // Dequantized weights (N*K*2 bytes)
        void * activations      = nullptr;  // Converted activations (M*K*2 bytes)
        size_t weights_size     = 0;
        size_t activations_size = 0;
    };

    // Reserve scratch buffers for oneDNN FP16 path.
    // Call once during model load with max dimensions.
    // weights_size: max(N*K*2) across all layers (usually FFN down: 14336*4096*2)
    // activations_size: max(M*K*2) where M=max_batch, K=max_dim
    bool reserve_onednn_scratch(size_t weights_size, size_t activations_size);

    // Get scratch buffers. Returns false if not reserved or sizes exceed reserved.
    // Caller must hold onednn_scratch_mutex_ via lock_onednn_scratch().
    bool get_onednn_scratch(size_t weights_needed, size_t activations_needed, onednn_scratch_buffers & out);

    // Lock/unlock for scratch buffer access (RAII recommended)
    std::unique_lock<std::mutex> lock_onednn_scratch() { return std::unique_lock<std::mutex>(onednn_scratch_mutex_); }

    // Check if scratch is reserved
    bool has_onednn_scratch() const { return onednn_weights_scratch_ != nullptr; }

    bool onednn_scratch_from_arena() const {
        return arena_active() && onednn_weights_scratch_ && onednn_activations_scratch_ &&
               vram_owns(onednn_weights_scratch_) && vram_owns(onednn_activations_scratch_);
    }

    const char * onednn_scratch_source_name() const {
        if (has_onednn_scratch()) {
            return onednn_scratch_from_arena() ? "arena" : "direct-device";
        }
        return arena_active() ? "arena-lazy" : "direct-device-lazy";
    }

    // Get reserved sizes
    size_t onednn_weights_scratch_size() const { return onednn_weights_scratch_size_; }

    size_t onednn_activations_scratch_size() const { return onednn_activations_scratch_size_; }

    struct pp_moe_onednn_scratch_slot {
        void *     weight          = nullptr;
        void *     activation      = nullptr;
        void *     output          = nullptr;
        size_t     weight_size     = 0;
        size_t     activation_size = 0;
        size_t     output_size     = 0;
        mem_handle weight_owner;
        mem_handle activation_owner;
        mem_handle output_owner;
    };

    bool reserve_pp_moe_onednn_scratch(size_t   weight_slot_bytes,
                                       size_t   activation_slot_bytes,
                                       size_t   output_slot_bytes,
                                       uint32_t ring_depth);
    bool get_pp_moe_onednn_scratch_slot(uint32_t slot, pp_moe_onednn_scratch_slot & out);

    // === GPU Reorder Temp Buffer ===
    // Pre-allocated device VRAM buffer reused by GPU-side AOS→SOA reorder.
    // Avoids per-expert sycl::malloc_device that fails when VRAM is tight.

    // Reserve the reorder temp buffer. Call once before weight loading begins.
    // size_bytes: largest expert tensor size (AOS bytes) with margin.
    bool reserve_reorder_temp(size_t size_bytes);

    // Get pointer and size of the pre-allocated reorder temp buffer.
    void * get_reorder_temp_buffer() const { return reorder_temp_buffer_; }

    size_t get_reorder_temp_size() const { return reorder_temp_size_; }

    // === Persistent Scratch Buffers ===
    // Named scratch buffers for persistent kernels (TG optimization).
    // Unlike oneDNN scratch which is shared, these are dedicated per named buffer.
    // Used for: intermediate activations, work counters, temporary storage.

    // Reserve a persistent scratch buffer by name.
    // If buffer already exists with sufficient size, returns true without reallocating.
    // If buffer exists but is too small, it is freed and reallocated.
    // pin: if true, buffer is protected from eviction (default true)
    bool reserve_persistent_scratch(const std::string & buffer_name, size_t size_bytes, bool pin = true);

    // Get pointer to a persistent scratch buffer.
    // Returns nullptr if buffer doesn't exist.
    void * get_persistent_scratch(const std::string & buffer_name);

    // Release (free) a persistent scratch buffer.
    void release_persistent_scratch(const std::string & buffer_name);

    // Check if a persistent scratch buffer exists
    bool has_persistent_scratch(const std::string & buffer_name) const;

    // Get size of a persistent scratch buffer (0 if not found)
    size_t get_persistent_scratch_size(const std::string & buffer_name) const;

    // === Unified Allocation API (Phase 3) ===
    // ALL VRAM allocations during inference should flow through this API.
    // The cache queries L0 free VRAM before allocating, evicts weights if needed,
    // and falls back to host-pinned memory when device allocation is unsafe.
    //
    // Lifetime categories control eviction eligibility and diagnostics:
    //   PERSISTENT — weights, KV cache — freed only on model/context destruction
    //   SCRATCH    — per-graph-compute scratch — reused between tokens
    //   TEMPORARY  — within a single kernel dispatch

    enum class alloc_lifetime : uint8_t {
        PERSISTENT = 0,  // Weights, KV cache — model/context lifetime
        SCRATCH    = 1,  // Per-graph scratch — reused between tokens
        TEMPORARY  = 2,  // Per-kernel dispatch — freed immediately after
    };

    struct vram_alloc_result {
        void * ptr       = nullptr;
        bool   on_device = false;  // true = VRAM, false = host-pinned USM
        size_t size      = 0;

        explicit operator bool() const { return ptr != nullptr; }
    };

    // Primary allocation entry point.  Checks budget (used_ + runtime_reserved +
    // size <= budget), queries L0 for real free VRAM as safety check, evicts
    // cached weights when possible, and falls back to host-pinned memory.
    // Thread-safe.  tag is a debug label for logging.
    vram_alloc_result allocate(size_t size, alloc_lifetime lifetime, const char * tag = nullptr);

    // Release a buffer obtained from allocate().  Adjusts budget tracking.
    // Accepts both device and host-pinned pointers (determined automatically).
    void deallocate(void * ptr, size_t size, alloc_lifetime lifetime);

    // === Compute Arena ===
    // Pre-reserved VRAM region for compute scratch buffers.
    // Reserved BEFORE S1-PRELOAD so weights cannot consume all VRAM.
    // When the VRAM arena is active, arena_alloc()/arena_free()/arena_reset()
    // route through the SCRATCH TLSF zone. Outside arena mode, the legacy
    // bump allocator is still used as a fallback.
    // pool_leg does NOT cache arena pointers — they are ephemeral per graph.
    //
    // arena_bytes: total VRAM to reserve for compute scratch.
    // Returns true if reservation succeeded.
    bool reserve_compute_arena(size_t arena_bytes);

    // Try to sub-allocate from the compute arena (256-byte aligned).
    // Returns nullptr if arena is not reserved or has insufficient space.
    void * arena_alloc(size_t size);

    // Release a compute-arena allocation.
    // In VRAM arena mode, this returns the block to the SCRATCH TLSF zone.
    // Outside arena mode, the legacy bump allocator only supports LIFO reclaim.
    void arena_free(void * ptr, size_t size);

    // Reset compute scratch state between graph_compute invocations.
    void arena_reset();

    // Check if a pointer belongs to the compute arena.
    bool arena_owns(const void * ptr) const;

    // Current compute arena capacity and usage.
    size_t compute_arena_capacity() const;
    size_t compute_arena_used() const;

    // === Inference Scratch Pool ===
    // Pre-allocated VRAM pool for per-op temporaries (Q8_1, FP16 dequant, etc.).
    // Call reserve_scratch_pool() once at context creation with the max scratch
    // size needed across all ops.  get_scratch()/return_scratch() are lock-free
    // bump allocators during inference — zero malloc, zero free.

    // Reserve the scratch pool.  Must be called before get_scratch().
    // Returns true if allocation succeeded (or pool already large enough).
    bool reserve_scratch_pool(size_t pool_bytes);

    // Get a scratch buffer from the pool.  Returns nullptr if pool exhausted.
    // The returned pointer is valid until return_scratch() or reset_scratch_pool().
    void * get_scratch(size_t size);

    // Return a scratch buffer to the pool.  Adjusts the bump pointer.
    void return_scratch(void * ptr, size_t size);

    // Reset the scratch pool bump pointer (call between graph_compute invocations).
    void reset_scratch_pool();

    // Current scratch pool capacity and high-water mark.
    size_t scratch_pool_capacity() const { return scratch_pool_size_; }

    size_t scratch_pool_hwm() const { return scratch_pool_hwm_; }

    // === Expert Allocation ===
    // Allocates from the expert portion of the cache budget.  Falls back to
    // host-pinned when VRAM is full.  Tracked separately for diagnostics.

    vram_alloc_result allocate_expert(size_t size);

    // === Live VRAM Queries ===
    // Query actual free VRAM from Level Zero (not just budget accounting).
    // Includes a safety margin for driver internals.

    // Actual free VRAM on the device (L0 query minus safety margin).
    size_t available_device() const;

    // Budget headroom: budget_ - used_ (what the cache thinks is available).
    size_t available_budget() const { return available(); }

    void * host_pool_alloc(size_t size, size_t alignment = 64);
    void   host_pool_free(void * ptr, size_t size);
    void   host_zone_reset(host_zone_id zone);
    void   host_zone_free(host_zone_id zone, void * ptr);
    size_t host_zone_used(host_zone_id zone) const;
    size_t host_zone_capacity(host_zone_id zone) const;
    size_t host_zone_reserved_capacity(host_zone_id zone) const;
    // Largest single-chunk contiguous free block in `zone`. Used by callers
    // that must hand out a contiguous pointer (the SYCL host buffer type)
    // and therefore cannot consume a fragmented multi-segment allocation.
    size_t host_zone_largest_free_block(host_zone_id zone) const;
    // Grow the given host zone by `additional_bytes`. Returns false when the
    // phase gate or pool budget blocks growth. Callers should treat a false
    // return as a hard failure and fall back to non-pinned allocation.
    bool   host_zone_grow(host_zone_id zone, size_t additional_bytes);
    void   configure_host_zones(size_t weight_bytes, size_t kv_bytes, size_t staging_bytes, size_t scratch_bytes);
    bool   host_zones_configured() const;
    void   grow_scratch_zone(size_t additional_bytes);

    size_t pinned_pool_budget() const { return host_arena_ ? host_arena_->budget() : 0; }

    size_t pinned_pool_committed() const { return host_arena_ ? host_arena_->allocated() : 0; }

    size_t pinned_pool_chunk_count() const { return host_arena_ ? host_arena_->chunk_count() : 0; }

    bool   contains_pinned(const void * ptr) const;
    size_t pre_allocate_host_pool(size_t total_bytes);

    // Deprecated shim — tests written against the old API. Migrate callers to unified_alloc().
    [[deprecated("use unified_alloc()")]] void * ensure_cached_alloc(const ggml_sycl_cache_id & key,
                                                                     const void *               src_ptr,
                                                                     size_t                     src_size,
                                                                     size_t                     alloc_size,
                                                                     cache_entry_type           type,
                                                                     int                        layer_id,
                                                                     int                        expert_id,
                                                                     ggml_layout_mode           layout,
                                                                     bool                       validate_content,
                                                                     bool *                     needs_fill);
    size_t                                       pre_allocate_all(size_t model_weight_bytes);
    size_t                                       pre_allocate_runtime_chunks(size_t total_bytes);
    // Host zone allocation (owned by unified_cache).
    segmented_buffer host_zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment = 64);
    void *           host_zone_alloc(host_zone_id zone, size_t size, size_t alignment = 64);

  private:
    // Sub-allocate from a zone.
    void * zone_alloc(vram_zone_id zone, size_t size, size_t align = 256);

    // Free a sub-allocation from a zone (TLSF O(1) free with coalescing).
    // ptr must have been returned by zone_alloc. No size parameter —
    // TLSF recovers block size from the inline block_header.
    void zone_free(vram_zone_id zone, void * ptr);

    // Reset a zone (returns all memory to the pool as a single free block).
    void zone_reset(vram_zone_id zone);

    // Friends: internal implementation functions that need zone access.
    // Consumer code must use unified_cache_zone_alloc / unified_allocate instead.
    friend bool   unified_alloc(const alloc_request & req_in, alloc_handle * out);
    friend void * unified_cache_arena_alloc_weight(int device_id, size_t size);
    friend void * unified_cache_kv_arena_alloc(int device_id, size_t size);
    friend void * unified_cache_zone_alloc(int device_id, vram_zone_id zone, size_t size, size_t align);
    friend void   unified_cache_zone_free(int device_id, vram_zone_id zone, void * ptr);
    friend void   unified_cache_zone_reset(int device_id, vram_zone_id zone);
    friend void * device_pool_arena_alloc(unified_cache * cache, size_t size, size_t align);

    // Evict lowest-scoring entry to make room for new_size bytes
    // Returns true if eviction succeeded, false if all entries are pinned
    size_t evict_one(size_t new_size);

    struct managed_alloc_ref {
        void *     ptr      = nullptr;
        size_t     size     = 0;
        int        device   = -1;
        uint64_t   alloc_id = 0;
        mem_handle owner;
    };

    struct deferred_free_entry {
        void *            ptr  = nullptr;
        size_t            size = 0;
        managed_alloc_ref handle{};
        bool              managed      = false;
        bool              zone_managed = false;
        vram_zone_id      zone         = vram_zone_id::COUNT;
        bool              has_event    = false;
        sycl::event       event;
    };

    struct deferred_host_free_entry {
        void *            ptr  = nullptr;
        size_t            size = 0;
        managed_alloc_ref handle{};
        bool              managed   = false;
        bool              has_event = false;
        sycl::event       event;
    };

    // Compute eviction score: higher = more valuable (keep longer)
    // score = access_count * exp(-decay * age)
    float compute_score(const unified_cache_entry & entry) const;

    // Copy data from mmap to device via staging. Returns event for the last transfer.
    sycl::event copy_to_device(void * dst, const void * src, size_t size);
    sycl::event copy_to_device_async(void *                           dst,
                                     const void *                     src,
                                     size_t                           size,
                                     const std::vector<sycl::event> & deps,
                                     sycl::queue *                    override_q = nullptr);
    static bool event_complete(const sycl::event & evt);
    sycl::event submit_barrier(const std::vector<sycl::event> & deps);
    sycl::event submit_barrier_all();
    void        process_deferred_frees();
    size_t      finalize_retired_entries_locked();
    void        remap_or_erase_id_mapping_locked(const ggml_sycl_cache_id & id, const unified_cache_key & removed_key);
    void        enqueue_deferred_free(void * ptr, size_t size);
    void        enqueue_deferred_free(const managed_alloc_ref & handle);
    void        enqueue_deferred_free(const managed_alloc_ref & handle, const sycl::event & event);
    void        enqueue_deferred_zone_free(vram_zone_id zone, void * ptr, size_t size, const sycl::event & event);
    void        enqueue_deferred_host_free(void * ptr, size_t size, const sycl::event & event);
    void release_entry_allocation_locked(unified_cache_entry & entry, vram_zone_id arena_zone = vram_zone_id::WEIGHT);

    // Saturating subtract from used_ to prevent underflow to SIZE_MAX.
    // Logs a warning if underflow is detected, then clamps to 0.
    void saturating_sub_used(size_t bytes) {
        size_t prev = used_.load(std::memory_order_relaxed);
        for (;;) {
            const size_t next = (prev >= bytes) ? (prev - bytes) : 0;
            if (used_.compare_exchange_weak(prev, next, std::memory_order_relaxed)) {
                if (prev < bytes) {
                    fprintf(stderr, "[UNIFIED-CACHE] used_ underflow prevented: prev=%zu sub=%zu clamped to 0\n", prev,
                            bytes);
                }
                return;
            }
        }
    }

    sycl::queue &                queue_;
    sycl::queue *                compute_queue_ = nullptr;    // Inference compute queue (for deferred free barriers)
    std::unique_ptr<sycl::queue> dma_queue_;                  // Separate in-order queue for cache DMA ops (CCS)
    std::unique_ptr<sycl::queue> bcs_queue_;                  // Copy-only queue targeting BCS engine (ordinal 1)
    size_t                       budget_;                     // Total GPU memory budget (after reservations)
    size_t                       base_budget_;                // Raw cache budget before reservations
    size_t                       reserved_;                   // Runtime reservation applied to budget_
    size_t                       device_total_vram_ = 0;      // Total VRAM snapshot passed by backend init
    bool                         budget_exceeded_   = false;  // Set when used > budget after eviction
    std::atomic<size_t>          used_{ 0 };                  // Current usage
    std::atomic<int64_t>         time_{ 0 };                  // Monotonic counter
    // P7: async DMA eviction state
    bool                         async_evict_enabled_ = false;  // Set during init from env var
    std::atomic<int>             evictions_in_flight_{ 0 };     // Count of EVICTING entries

    // Cache storage: (identity, type, layer/expert) -> entry
    std::unordered_map<unified_cache_key,
                       unified_cache_entry,
                       unified_cache_key_hash,
                       std::equal_to<unified_cache_key>,
                       detail::cache_guard_allocator<std::pair<const unified_cache_key, unified_cache_entry>>>
        entries_;
    std::unordered_map<ggml_sycl_cache_id, unified_cache_key, detail::cache_id_hash, detail::cache_id_equal_fn>
        id_to_key_;

    // Layout pool: consolidates many individual layout allocations into
    // a few large contiguous chunks to reduce GPU TLB pressure.
    // All layout allocations are sub-allocated from this pool.
    // The pool is destroyed (freeing all chunks) in the unified_cache destructor.
    std::unique_ptr<ggml_sycl::sycl_device_pool>                 layout_pool_;
    std::array<size_t, static_cast<size_t>(host_zone_id::COUNT)> host_zone_reserved_bytes_{};

    // VRAM arena data members (merged from vram_arena class).
    // Single pre-allocated VRAM block with zone-based sub-allocation.
    // Gated by GGML_SYCL_VRAM_ARENA=1 env var.
    void * arena_base_ = nullptr;
    size_t arena_size_ = 0;

    struct arena_chunk {
        void *              ptr  = nullptr;
        size_t              size = 0;
        copyable_atomic_u32 lease_count;  // llama.cpp-dyhdl: live raw-ptr refs
    };

    // N-chunk arena: each chunk capped at the runtime's per-allocation limit
    // so the patched libze (~1.5 GB on Arc B580) does not reject the alloc.
    std::vector<arena_chunk> arena_chunks_;
    sycl::queue *            arena_queue_ = nullptr;

    // Per-physical-chunk TLSF allocators for the WEIGHT zone in N-chunk mode.
    // A weight tensor must not span a chunk boundary (each chunk is its own
    // sycl::malloc_device USM allocation), so we use one TLSF allocator per
    // weight-bearing chunk and try each in turn on alloc.
    //
    // Empty in single-chunk mode — WEIGHT then has no allocator of its own
    // and delegates to the KV allocator via the shared-zone pattern (see
    // zone_alloc()).
    //
    // Invariant: chunk_idx is always < arena_chunks_.size() - 1; the last
    // chunk holds tail zones (KV/ONEDNN/RUNTIME/SCRATCH), never WEIGHT.
    // chunk_idx is `int` to match arena_find_chunk()'s return type and the
    // arena lease API (which uses -1 as a not-owned sentinel).
    struct weight_chunk_alloc {
        int                             chunk_idx     = 0;  // index into arena_chunks_
        size_t                          logical_start = 0;  // cumulative offset of this chunk
        std::unique_ptr<tlsf_allocator> allocator;
    };

    std::vector<weight_chunk_alloc> weight_chunk_allocators_;
    vram_zone                       arena_zones_[static_cast<int>(vram_zone_id::COUNT)];

    // Arena generation counter: incremented when the arena is destroyed/recreated.
    // mem_handle arena handles store the generation at creation time; on resolve(),
    // a mismatch means the handle is stale (arena was recycled).
    uint64_t arena_generation_ = 0;

    // Compute arena: pre-reserved VRAM for compute scratch buffers.
    // Single unified_alloc-owned device allocation made BEFORE S1-PRELOAD fills VRAM.
    // Bump-allocated during graph_compute, reset between invocations.
    void *              compute_arena_ptr_  = nullptr;
    size_t              compute_arena_size_ = 0;
    mem_handle          compute_arena_owner_;
    std::atomic<size_t> compute_arena_off_{ 0 };

    // Staging buffer for mmap -> device transfers
    void *     staging_      = nullptr;
    size_t     staging_size_ = 0;
    mem_handle staging_owner_;
    std::mutex staging_mutex_;

    // Device-resident DMA staging buffers (for streaming).
    std::vector<managed_alloc_ref> dma_staging_allocs_;
    std::vector<void *>            dma_staging_buffers_;
    size_t                         dma_slice_bytes_    = 0;
    size_t                         dma_buffer_count_   = 0;
    size_t                         dma_reserved_bytes_ = 0;
    std::mutex                     dma_staging_mutex_;

    // Reusable temp VRAM buffer for GPU-side AOS→SOA reorder during MoE prestage.
    // Pre-allocated at cache init to avoid per-expert malloc_device that fails
    // when VRAM is full after S1-PRELOAD fills it with dense weights.
    void *     reorder_temp_buffer_ = nullptr;
    size_t     reorder_temp_size_   = 0;
    mem_handle reorder_temp_owner_;

    // OneDNN FP16 scratch buffers for prompt processing.
    // Pre-allocated to avoid per-op allocations that cause OOM with large contexts.
    // weights_scratch_: holds dequantized weights (max N*K*2 bytes)
    // activations_scratch_: holds converted activations (max M*K*2 bytes)
    void *     onednn_weights_scratch_          = nullptr;
    void *     onednn_activations_scratch_      = nullptr;
    size_t     onednn_weights_scratch_size_     = 0;
    size_t     onednn_activations_scratch_size_ = 0;
    mem_handle onednn_weights_scratch_owner_;
    mem_handle onednn_activations_scratch_owner_;
    std::mutex onednn_scratch_mutex_;

    std::vector<pp_moe_onednn_scratch_slot> pp_moe_onednn_scratch_slots_;
    size_t                                  pp_moe_onednn_weight_slot_size_     = 0;
    size_t                                  pp_moe_onednn_activation_slot_size_ = 0;
    size_t                                  pp_moe_onednn_output_slot_size_     = 0;
    uint32_t                                pp_moe_onednn_ring_depth_           = 0;
    std::mutex                              pp_moe_onednn_scratch_mutex_;

    // Persistent scratch buffers for TG optimization (persistent kernels).
    // Keyed by name for flexibility (e.g., "activations", "work_counter", "temp").
    struct persistent_scratch_entry {
        void *     device_ptr = nullptr;
        size_t     size       = 0;
        bool       pinned     = true;
        mem_handle owner;
    };

    std::unordered_map<std::string, persistent_scratch_entry> persistent_scratches_;
    mutable std::mutex                                        persistent_scratch_mutex_;

    // === Inference Scratch Pool (Phase 3) ===
    // Pre-allocated VRAM block for per-op temporaries.  get_scratch() bumps
    // the offset; return_scratch() decrements it (stack discipline).
    // reset_scratch_pool() resets to zero between graph_compute calls.
    void *              scratch_pool_ptr_  = nullptr;  // Base pointer (device VRAM)
    size_t              scratch_pool_size_ = 0;        // Total pool bytes
    mem_handle          scratch_pool_owner_;           // Non-arena direct allocation owner
    std::atomic<size_t> scratch_pool_off_{ 0 };        // Current bump offset
    size_t              scratch_pool_hwm_ = 0;         // High-water mark for diagnostics

    // === Unified allocate() tracking (Phase 3) ===
    // Track pointers returned by allocate() so deallocate() knows whether
    // the pointer is device or host-pinned and which budget to adjust.
    struct managed_alloc_entry {
        size_t         size             = 0;
        bool           on_device        = false;
        bool           uses_pinned_pool = false;
        host_zone_id   host_zone        = host_zone_id::COUNT;
        alloc_lifetime lifetime         = alloc_lifetime::SCRATCH;
    };

    std::unordered_map<void *, managed_alloc_entry> managed_allocs_;
    std::mutex                                      managed_allocs_mutex_;

    // Safety margin subtracted from L0-reported free VRAM to avoid driver OOM.
    static constexpr size_t VRAM_SAFETY_MARGIN = 128 * 1024 * 1024;  // 128 MB

    // Deferred frees to avoid releasing buffers while in flight.
    std::vector<deferred_free_entry>      deferred_frees_;
    std::vector<deferred_host_free_entry> deferred_host_frees_;

    struct inflight_unpin_entry {
        ggml_sycl_cache_id key       = {};
        ggml_layout_mode   layout    = GGML_LAYOUT_AOS;
        bool               has_event = false;
        sycl::event        event;
    };

    // Entries pinned for in-flight kernels.
    std::list<inflight_unpin_entry> inflight_unpins_;

    // === Async Layer Prefetch State ===
    // Background worker thread pins layer weights ahead of kernel execution.

    struct prefetch_request {
        int               layer_id;
        layer_weight_set  weights;
        ggml_layout_mode  layout;
        prefetch_priority priority;
    };

    // Prefetch queue and worker thread
    std::deque<prefetch_request> prefetch_queue_;
    std::mutex                   prefetch_mutex_;
    std::condition_variable      prefetch_cv_;
    std::thread                  prefetch_worker_;
    std::atomic<bool>            prefetch_shutdown_{ false };
    std::atomic<bool>            prefetch_started_{ false };
    std::mutex                   prefetch_lifecycle_mutex_;  // Guards start/stop of prefetch_worker_

    // Start the background prefetch worker thread (called lazily on first queue_layer_prefetch).
    void start_prefetch_worker();

    // Stop the background prefetch worker thread (called from destructor).
    void stop_prefetch_worker();

    // Per-layer ready tracking
    // Guarded by layer_state_mutex_
    std::unordered_map<int, bool>             layer_ready_;    // layer_id -> ready
    std::unordered_map<int, layer_weight_set> layer_weights_;  // for release_layer unpin
    std::unordered_map<int, ggml_layout_mode> layer_layouts_;  // layout used for pinning
    mutable std::mutex                        layer_state_mutex_;
    std::condition_variable                   layer_ready_cv_;

    // The prefetch worker loop (runs on background thread)
    void prefetch_worker_loop();

    // === Direct Staging Lookup Mirrors ===
    // Non-owning mirrors for legacy/raw-pointer call sites and diagnostics.
    // Canonical allocation lifetime is entries_ + mem_handle leases; these maps
    // must not be treated as independent caches or ownership roots.
    // Keyed by full ggml_sycl_cache_id (uses proven detail::cache_id_hash/equal_fn).
    // Thread-safe via direct_stage_mutex_ (shared for reads, exclusive for writes).
    std::unordered_map<ggml_sycl_cache_id, weight_entry, detail::cache_id_hash, detail::cache_id_equal_fn>
        direct_weight_entries_;
    std::unordered_map<ggml_sycl_cache_id, weight_entry, detail::cache_id_hash, detail::cache_id_equal_fn>
                              direct_expert_entries_;
    mutable std::shared_mutex direct_stage_mutex_;

    // Set to true when any weight has been evicted from device to host.
    // One-way flag (false → true, never reset). Used by graph replay / persistent TG
    // to know that baked pointers may reference freed device memory.
    std::atomic<bool> has_evictions_{ false };

    // === Multi-Device Partial Row Cache ===
    // Stores SOA-reordered partial weight tensors for multi-device tensor split.
    // Key: stable tensor cache identity + device + source row range.
    // Value: device pointer + metadata.
    struct partial_rows_key {
        ggml_sycl_cache_id tensor_id;
        int                device_idx;
        ggml_type          type;
        int64_t            ncols;
        int64_t            row_start;
        int64_t            row_count;

        bool operator==(const partial_rows_key & other) const {
            return device_idx == other.device_idx && type == other.type && ncols == other.ncols &&
                   row_start == other.row_start && row_count == other.row_count &&
                   detail::cache_id_equal(tensor_id, other.tensor_id);
        }
    };

    struct partial_rows_key_hash {
        size_t operator()(const partial_rows_key & key) const {
            size_t h = detail::cache_id_hash{}(key.tensor_id);
            h        = detail::cache_hash_combine(h, std::hash<int>()(key.device_idx));
            h        = detail::cache_hash_combine(h, std::hash<int>()((int) key.type));
            h        = detail::cache_hash_combine(h, std::hash<int64_t>()(key.ncols));
            h        = detail::cache_hash_combine(h, std::hash<int64_t>()(key.row_start));
            h        = detail::cache_hash_combine(h, std::hash<int64_t>()(key.row_count));
            return h;
        }
    };

    struct partial_entry {
        void *     ptr;
        int        device_idx;
        ggml_type  type;
        int64_t    ncols;
        int64_t    row_start;
        int64_t    row_count;
        size_t     bytes;
        mem_handle owner;
    };

    std::unordered_map<partial_rows_key, partial_entry, partial_rows_key_hash> partial_cache_;
    std::mutex                                                                 partial_mutex_;

    // === Placement Plan (P4) ===
    placement_plan   placement_plan_;
    bool             has_placement_plan_ = false;
    std::atomic<int> planned_materialization_depth_{ 0 };

    bool planned_materialization_allowed(const char *               op,
                                         const ggml_sycl_cache_id & key,
                                         ggml_layout_mode           layout,
                                         const char *               caller) const;

    // Stats
    mutable std::atomic<size_t> hits_{ 0 };
    mutable std::atomic<size_t> misses_{ 0 };

    static constexpr float DECAY_ALPHA = 0.01f;

    mutable std::shared_mutex           rw_mutex_;
    mutable std::condition_variable_any entry_cv_;

    // Host memory arena (pinned_chunk_pool).
    // Bypasses Intel Level Zero's ~11GB per-allocation limit via 2GB chunks.
    // Initialized with capped budget (128 GB) to prevent unbounded growth.
    std::unique_ptr<pinned_chunk_pool> host_arena_;
};

class scoped_planned_materialization {
  public:
    scoped_planned_materialization(unified_cache * cache, const char * reason) : cache_(cache) {
        if (cache_) {
            cache_->begin_planned_materialization(reason);
        }
    }

    ~scoped_planned_materialization() {
        if (cache_) {
            cache_->end_planned_materialization();
        }
    }

    scoped_planned_materialization(const scoped_planned_materialization &)             = delete;
    scoped_planned_materialization & operator=(const scoped_planned_materialization &) = delete;

    scoped_planned_materialization(scoped_planned_materialization && other) noexcept : cache_(other.cache_) {
        other.cache_ = nullptr;
    }

    scoped_planned_materialization & operator=(scoped_planned_materialization && other) noexcept {
        if (this != &other) {
            if (cache_) {
                cache_->end_planned_materialization();
            }
            cache_       = other.cache_;
            other.cache_ = nullptr;
        }
        return *this;
    }

  private:
    unified_cache * cache_ = nullptr;
};

// === Cache Mode ===
// Controls whether cache is shared globally or per-device

enum class unified_cache_mode {
    GLOBAL,      // Single cache on primary device (default for single-GPU)
    PER_DEVICE,  // Separate cache per device (better for multi-GPU)
    AUTO         // Auto-detect: per_device if multiple GPUs, global otherwise
};

// Get current cache mode (from env var or auto-detection)
unified_cache_mode get_unified_cache_mode();

// Set cache mode (call before first cache access)
void set_unified_cache_mode(unified_cache_mode mode);

// Set the scheduler-filtered device count (call from ggml_sycl_init after device map filtering)
void set_scheduler_device_count(int count);

// Set the total physical GPU count before cache pre-initialization
// (call from ggml_sycl_init BEFORE get_unified_cache_for_device to avoid
// ggml_sycl_info() re-entry deadlock on the static init guard)
void set_total_gpu_count(int count);

// === Shared-Context Queue Pool ===
//
// On Level Zero, cross-device SYCL events (depends_on, USM transfers) REQUIRE
// matching ze_context_handle_t.  Device default queues (dpct::dev_mgr) each
// have their OWN context.
//
// PRIOR DESIGN (broken): shared device 0's context across all secondary queues.
// This created a multi-device Level Zero context, triggering DEVICE_LOST on
// compute-runtime 26.x (intel/compute-runtime#916, #921).
//
// CURRENT DESIGN: Each secondary queue uses its own single-device SYCL context.
// Cross-device data moves via host-staged memcpy (no shared USM).  Each context
// is isolated — no cross-device depends_on() or USM access.

sycl::queue * get_shared_context_queue(int device);
void          init_shared_context_queues(int total_gpus);

// === Global API ===

// Get unified cache for the device associated with the given queue
// In GLOBAL mode: returns same cache for all devices
// In PER_DEVICE mode: returns device-specific cache
unified_cache * get_unified_cache(sycl::queue & queue);

// Get unified cache for a specific device ID
// Useful when device ID is known but queue isn't available
unified_cache * get_unified_cache_for_device(int device_id);

// Lookup an already-created unified cache without constructing one.
// Use from destructors and shutdown paths where creating a new cache would
// re-enter allocator/static teardown.
unified_cache * get_existing_unified_cache_for_device(int device_id);

// Overload with device memory hints — avoids ggml_sycl_info() reentry
// deadlock when called from within ggml_sycl_init() static initialization.
unified_cache * get_unified_cache_for_device(int    device_id,
                                             size_t free_mem,
                                             size_t total_mem,
                                             size_t free_vram_at_init);

// S1 preload runs during model load and materializes unified-cache weight
// handles. Keep the default serialized: current Level Zero stacks can wedge
// waiting on later host-pinned H2D events when many CPU-reordered expert
// copies are outstanding. Wider windows remain a diagnostic opt-in.
static constexpr size_t GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_DEFAULT = 1;
static constexpr size_t GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MIN     = 1;
static constexpr size_t GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MAX     = 64;

inline size_t s1_preload_max_in_flight_limit() {
    size_t value = GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_DEFAULT;
    if (const char * env = std::getenv("GGML_SYCL_S1_MAX_IN_FLIGHT")) {
        value = static_cast<size_t>(std::atoi(env));
    }
    if (value < GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MIN) {
        value = GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MIN;
    }
    if (value > GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MAX) {
        value = GGML_SYCL_S1_PRELOAD_MAX_IN_FLIGHT_MAX;
    }
    return value;
}

// Set unified cache budget (call before first use)
// In PER_DEVICE mode, this sets budget per device
void set_unified_cache_budget(size_t bytes);
// Set unified cache budget as percentage of free VRAM (call before first use)
void set_unified_cache_budget_pct(int pct);
// Set unified host cache budget as percentage of total system RAM (call before first use)
void set_unified_cache_host_budget_pct(int pct);

// Classification of VRAM allocations for budget tracking and diagnostics
enum class alloc_hint : uint8_t {
    WEIGHT     = 0,  // Evictable model weights (managed by cache LRU)
    COMPUTE    = 1,  // Per-inference scratch (compute buffers, activation staging)
    EPHEMERAL  = 2,  // Per-graph temporaries (freed within graph_compute)
    PERSISTENT = 3,  // Context-lifetime buffers (persistent kernel state, DAG arrays)
    DEBUG      = 4,  // Debug/profiling allocations (env-gated, not production)
};

// Classification of runtime (non-weight) VRAM reservations for diagnostics.
enum class runtime_category : uint8_t {
    KV_CACHE     = 0,  // KV buffer allocations
    COMPUTE      = 1,  // Compute buffer pool + scratch
    STAGING      = 2,  // Expert staging, DMA staging, BLAS fallback
    GRAPH        = 3,  // Persistent TG temp allocs, get_rows_pool
    HOST_COMPUTE = 4,  // Host-pinned scratch for CPU offload layers
    EXPERT_CACHE = 5,  // MoE expert VRAM cache pool
    CONTROL      = 6,  // Small CPU-produced control tensors read by any GPU
    OTHER        = 7,  // Everything else (default for backward compat)
    COUNT        = 8
};

// Unified runtime allocation API.
// Managed runtime allocations must flow through this API so budget accounting,
// placement policy, and pointer registry stay consistent.
enum class alloc_tier : uint8_t {
    DEVICE_VRAM  = 0,
    HOST_PINNED  = 1,
    MMAP_TRACKED = 2,
};

enum class alloc_role : uint8_t {
    WEIGHT         = 0,
    COMPUTE        = 1,
    KV             = 2,
    STAGING        = 3,
    GRAPH_TMP      = 4,
    TP_TMP         = 5,
    EXPERT_STAGING = 6,
    CONTROL        = 7,
    OTHER          = 8,
};

// Location-transparent descriptor for any managed allocation.
// Returned by query_location() — gives callers the current pointer, tier,
// layout, and arena zone without SYCL pointer-type queries at runtime.
// Immutable during inference (allocations do not migrate mid-token).
struct memory_location {
    void *           ptr        = nullptr;  // Current data pointer (device or host)
    alloc_tier       tier       = alloc_tier::MMAP_TRACKED;
    ggml_layout_mode layout     = GGML_LAYOUT_AOS;
    alloc_role       role       = alloc_role::OTHER;
    vram_zone_id     zone       = vram_zone_id::COUNT;  // COUNT = not in arena
    int              device     = -1;                   // Owning device (-1 = host)
    bool             from_arena = false;                // True if sub-allocated from VRAM arena

    explicit operator bool() const { return ptr != nullptr; }

    bool on_device() const { return tier == alloc_tier::DEVICE_VRAM; }

    bool host_accessible() const { return tier != alloc_tier::DEVICE_VRAM; }
};

// Query current location of any pointer.  Uses alloc_registry for tier
// classification and arena zone ownership detection.  O(1) via binary search.
// Does NOT acquire SYCL runtime locks — safe to call at graph build time.
memory_location query_location(const void * ptr, int device_hint = -1);

// Query KV cache location for a specific layer.  Returns tier (DEVICE_VRAM
// or HOST_PINNED) based on kv_tier_manager placement decisions.
// O(1) lookup — safe at graph build time.
memory_location query_kv_location(int layer_id, int device);

enum class offload_buffer_role : uint8_t {
    STAGING_SRC0       = 0,
    STAGING_SRC1       = 1,
    STAGING_DST        = 2,
    RETAINED_SCRATCH   = 3,
    SET_TENSOR_STAGE   = 4,
    SET_TENSOR_REORDER = 5,
    OTHER              = 6,
};

struct alloc_constraints {
    bool         must_device                = false;
    bool         must_host_pinned           = false;
    bool         prefer_same_tier_as_cohort = false;
    bool         use_pinned_pool            = false;
    // Refuse dynamic host-zone growth for planner-bounded staging pools. A
    // miss should become back-pressure/reuse instead of expanding the pinned
    // host arena beyond the plan's reservation.
    bool         forbid_host_zone_growth    = false;
    // Require a standalone sycl::malloc_host allocation instead of a pinned-pool
    // slice.  Some Level Zero copy paths need the host pointer to be a
    // driver-visible USM allocation base, not an interior TLSF suballocation.
    bool         require_host_usm_base      = false;
    // When arena is active and prefer_vram_zone != COUNT, unified_alloc routes
    // through that VRAM zone (zone_alloc) instead of raw device malloc.
    // unified_free then calls zone_free(vram_zone, ptr) for explicit TLSF reclaim.
    vram_zone_id prefer_vram_zone           = vram_zone_id::COUNT;
};

struct alloc_intent {
    alloc_role        role      = alloc_role::OTHER;
    runtime_category  category  = runtime_category::OTHER;
    const char *      cohort_id = nullptr;
    alloc_constraints constraints;
};

struct alloc_request {
    sycl::queue * queue                = nullptr;
    int           device               = -1;
    size_t        size                 = 0;
    bool          suppress_failure_log = false;  // Caller handles nullptr locally (e.g. back-pressure/reuse).
    alloc_intent  intent;
};

struct alloc_handle {
    void *           ptr      = nullptr;  // First segment (caller-facing pointer)
    size_t           size     = 0;        // Total size requested
    int              device   = -1;
    alloc_tier       tier     = alloc_tier::DEVICE_VRAM;
    alloc_role       role     = alloc_role::OTHER;
    runtime_category category = runtime_category::OTHER;
    uint64_t         alloc_id = 0;

    // Zone routing fields — set by unified_alloc when the allocation is routed
    // through a zone sub-allocator instead of a raw sycl::malloc call.
    // unified_free() uses these to dispatch the correct reclaim path:
    //   zone_managed=true, vram_zone!=COUNT → cache->zone_free(vram_zone, ptr) [TLSF reclaim]
    //   zone_managed=true, host_zone==WEIGHT|KV → cache->host_zone_free(zone, ptr) [TLSF reclaim]
    //   zone_managed=true, host_zone==SCRATCH and role==EXPERT_STAGING → scoped TLSF reclaim
    //   zone_managed=true, host_zone==SCRATCH|STAGING otherwise → reset-only (freed by host_zone_reset)
    //   zone_managed=false                  → registry lookup → sycl::free or pinned_pool free
    bool         zone_managed = false;
    vram_zone_id vram_zone    = vram_zone_id::COUNT;
    host_zone_id host_zone    = host_zone_id::COUNT;

    // Internal tracking for segmented allocations.
    // When a single allocation request exceeds the chunk size, multiple segments
    // are allocated internally. The caller only sees ptr (first segment), but
    // unified_free() uses all_segments to release all internal segments.
    std::vector<buffer_segment> all_segments;

    // Returns a DIRECT mem_handle view over this allocation.
    // The handle carries no ownership — unified_free(alloc_handle) must still
    // be called when the memory is no longer needed.
    mem_handle as_mem_handle() const;
};

struct offload_buffer_request {
    sycl::queue *       queue     = nullptr;
    int                 device    = -1;
    size_t              size      = 0;
    size_t              alignment = 64;
    offload_buffer_role role      = offload_buffer_role::OTHER;
    alloc_intent        intent{};
};

struct offload_buffer_lease {
    alloc_handle handle{};
    uint64_t     lease_id = 0;
    bool         valid    = false;
};

bool       unified_alloc(const alloc_request & req, alloc_handle * out);
// New: returns mem_handle (auto-validating smart pointer). Existing callers
// continue to use unified_alloc(); callers migrate incrementally in T4/T5.
mem_handle unified_allocate(const alloc_request & req);
bool       unified_free(const alloc_handle & handle);
bool       unified_free_ptr(void * ptr, int expected_device = -1);
bool       unified_lookup(void * ptr, alloc_handle * out);
bool       unified_lookup_runtime_allocation(const void * ptr, alloc_handle * out, sycl::queue ** queue_out = nullptr);
alloc_tier unified_select_tier(const alloc_request & req);
bool       unified_alloc_validate_registry(int device = -1, const char * where = nullptr);
bool       unified_alloc_strict_mode();

// ============================================================================
// Simplified allocation API — the ONE function all subsystems should call.
//
// Maps alloc_category to the underlying alloc_role / runtime_category / constraints,
// then delegates to unified_alloc().  Provides host fallback for eligible categories.
// ============================================================================

enum class alloc_category : uint8_t {
    WEIGHT          = 0,  // Permanent model weights, highest VRAM priority
    KV_CACHE        = 1,  // Permanent KV cache, high priority
    COMPUTE_SCRATCH = 2,  // Ephemeral per-op scratch, can fall back to host
    STAGING         = 3,  // Small transfer staging buffers, can fall back to host
    CONTROL         = 4,  // Tiny CPU-produced control buffers, host-pinned and device-readable
    EXPERT_CACHE    = 5,  // MoE expert data, evictable from device
};

// Returns eviction priority: lower number = higher VRAM priority (less likely to evict).
// 0 = compute scratch (pre-reserved arena, must stay in VRAM for FP16 attention)
// 1 = KV cache (hot tokens, latency-critical)
// 2 = attention weights / control buffers (used every token)
// 3 = MoE expert cache (evictable via LRU/frequency)
// 4 = staging buffers (always host-ok)
int alloc_category_priority(alloc_category cat);

struct unified_alloc_result {
    void *       ptr  = nullptr;
    alloc_tier   tier = alloc_tier::DEVICE_VRAM;
    size_t       size = 0;
    alloc_handle handle{};
};

// Allocate memory through the unified cache budget system.
// Returns {nullptr} on failure.  Host fallback is attempted for
// COMPUTE_SCRATCH and STAGING categories when VRAM is insufficient.
unified_alloc_result unified_cache_allocate(int device, size_t size, alloc_category category, sycl::queue & queue);

// Free memory previously obtained from unified_cache_allocate().
// Updates budget counters and removes from alloc_registry.
void unified_cache_deallocate(void * ptr, int device);

bool acquire_offload_buffer(const offload_buffer_request & req, offload_buffer_lease * out);
bool release_offload_buffer(const offload_buffer_lease & lease);
void offload_buffer_pool_trim(int device = -1);

struct offload_stats_snapshot {
    uint64_t wait_count                                = 0;
    uint64_t wait_count_forced                         = 0;
    uint64_t wait_count_fallback                       = 0;
    uint64_t alloc_count_host                          = 0;
    uint64_t alloc_count_device                        = 0;
    uint64_t alloc_count_shared                        = 0;
    uint64_t pool_hit_count                            = 0;
    uint64_t pool_miss_count                           = 0;
    uint64_t cross_domain_transfer_count               = 0;
    uint64_t cross_domain_transfer_count_pp            = 0;
    uint64_t cross_domain_transfer_count_tg            = 0;
    uint64_t transfer_bytes_h2d                        = 0;
    uint64_t transfer_bytes_d2h                        = 0;
    uint64_t transfer_bytes_h2d_pp                     = 0;
    uint64_t transfer_bytes_h2d_tg                     = 0;
    uint64_t transfer_bytes_d2h_pp                     = 0;
    uint64_t transfer_bytes_d2h_tg                     = 0;
    uint64_t dispatch_count_cpu                        = 0;
    uint64_t dispatch_count_gpu                        = 0;
    uint64_t dispatch_count_gpu_island                 = 0;
    uint64_t dispatch_count_cpu_pp                     = 0;
    uint64_t dispatch_count_cpu_tg                     = 0;
    uint64_t dispatch_count_gpu_pp                     = 0;
    uint64_t dispatch_count_gpu_tg                     = 0;
    uint64_t dispatch_count_gpu_island_pp              = 0;
    uint64_t dispatch_count_gpu_island_tg              = 0;
    uint64_t transition_wait_count                     = 0;
    uint64_t transition_wait_count_pp                  = 0;
    uint64_t transition_wait_count_tg                  = 0;
    uint64_t transition_wait_elided_count              = 0;
    uint64_t transition_wait_elided_count_pp           = 0;
    uint64_t transition_wait_elided_count_tg           = 0;
    uint64_t host_alloc_call_count                     = 0;
    uint64_t host_alloc_bytes                          = 0;
    uint64_t host_alloc_calls_unified_alloc_host       = 0;
    uint64_t host_alloc_bytes_unified_alloc_host       = 0;
    uint64_t host_alloc_calls_unified_cache_host_chunk = 0;
    uint64_t host_alloc_bytes_unified_cache_host_chunk = 0;
    uint64_t host_alloc_calls_host_malloc              = 0;
    uint64_t host_alloc_bytes_host_malloc              = 0;
    uint64_t host_alloc_calls_other                    = 0;
    uint64_t host_alloc_bytes_other                    = 0;
    uint64_t raw_device_alloc_call_count               = 0;
    uint64_t raw_device_alloc_bytes                    = 0;
    uint64_t host_fallback_attempt_count               = 0;
    uint64_t host_fallback_attempt_bytes               = 0;
};

enum class offload_phase : uint8_t {
    UNKNOWN = 0,
    LOAD    = 1,
    WARMUP  = 2,
    PP      = 3,
    TG      = 4,
};

bool                   offload_stats_enabled();
void                   offload_stats_reset();
void                   offload_stats_set_phase(offload_phase phase);
offload_phase          offload_stats_phase();
const char *           offload_phase_name(offload_phase phase);
void                   offload_stats_note_wait(bool fallback = false);
void                   offload_stats_note_alloc(alloc_tier tier);
void                   offload_stats_note_pool_hit();
void                   offload_stats_note_pool_miss();
void                   offload_stats_note_transfer(bool h2d, size_t bytes);
void                   offload_stats_note_cross_domain_transfer(size_t bytes);
void                   offload_stats_note_dispatch(bool cpu, bool gpu_island = false);
void                   offload_stats_note_transition_wait(bool waited);
void                   offload_stats_note_host_alloc(const char * tag, size_t bytes);
void                   offload_stats_note_raw_device_alloc(size_t bytes);
void                   offload_stats_note_host_fallback_attempt(size_t bytes);
offload_stats_snapshot offload_stats_get();
void                   offload_stats_log_summary(const char * tag, int device);
void                   zero_alloc_check(const char * tag, int device);

bool arena_pp_profile_enabled();
bool arena_pp_profile_active();
bool arena_pp_profile_begin(int device, bool is_prompt_phase);
void arena_pp_profile_end(const char * tag, int device);
void arena_pp_profile_note_onednn_reserve(size_t weights_size,
                                          size_t activations_size,
                                          bool   reused,
                                          bool   arena_attempted,
                                          bool   arena_success,
                                          bool   direct_attempted,
                                          bool   ok,
                                          double elapsed_us);
void arena_pp_profile_note_onednn_get(size_t weights_needed, size_t activations_needed, bool ok, double elapsed_us);

class scoped_unified_alloc {
  public:
    scoped_unified_alloc() = default;

    explicit scoped_unified_alloc(const alloc_request & req) { allocate(req); }

    ~scoped_unified_alloc() { reset(); }

    scoped_unified_alloc(const scoped_unified_alloc &)             = delete;
    scoped_unified_alloc & operator=(const scoped_unified_alloc &) = delete;

    scoped_unified_alloc(scoped_unified_alloc && other) noexcept : handle_(other.release()) {}

    scoped_unified_alloc & operator=(scoped_unified_alloc && other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.release();
        }
        return *this;
    }

    bool allocate(const alloc_request & req) {
        reset();
        return unified_alloc(req, &handle_);
    }

    void reset() {
        if (handle_.ptr != nullptr) {
            (void) unified_free(handle_);
            handle_ = {};
        }
    }

    void * get() const { return handle_.ptr; }

    const alloc_handle & handle() const { return handle_; }

    // Returns a mem_handle view over the current allocation.
    // The returned handle is DIRECT (stable within the current graph).
    mem_handle as_mem_handle() const;

    alloc_handle release() {
        alloc_handle out = handle_;
        handle_          = {};
        return out;
    }

    explicit operator bool() const { return handle_.ptr != nullptr; }

  private:
    alloc_handle handle_{};
};

// Graph compute eviction guard — prevents expert cache eviction during inference
void unified_cache_set_graph_compute_active(bool active);
bool unified_cache_is_graph_compute_active();

// Check if any device has pending deferred frees (cheap pre-flight for flush)
bool unified_cache_has_pending_deferred_frees(int device);

// Query total runtime (non-weight) VRAM usage on a device via arena zones.
size_t unified_cache_get_runtime_bytes(int device);

// Query runtime bytes for a specific category on a device.
size_t unified_cache_get_runtime_bytes_by_category(int device, runtime_category cat);

void   unified_cache_add_runtime_host_bytes(size_t bytes);
void   unified_cache_sub_runtime_host_bytes(size_t bytes);
size_t unified_cache_get_runtime_host_bytes();

// Query available VRAM for non-weight allocations (KV, compute, staging).
size_t unified_cache_available_for_compute(int device);

// Total VRAM committed across BOTH budget channels (weights + runtime).
// This is the single authoritative number for "how much VRAM are we using?"
//   = runtime_reserved_bytes[device] + cache->used() (weight layout bytes)
size_t unified_cache_total_committed_bytes(int device);

// Total VRAM still allocatable = base_budget - total_committed.
// This is the single authoritative number for "how much VRAM can we still use?"
size_t unified_cache_total_available_bytes(int device);

// Raw VRAM budget before reservations (= free VRAM * budget_pct at init)
size_t unified_cache_total_managed(int device);

// Current weight bytes on device
size_t unified_cache_weight_bytes(int device);

// Returns total VRAM bytes used by weight entries for a specific layer on a device.
// Used by KV tier manager to co-locate KV with layer weights.
size_t unified_cache_get_layer_vram_bytes(int device, int layer_id);

// Returns total VRAM bytes held by unpinned MoE expert entries that could be
// evicted to make room for other allocations (e.g., KV layer promotion).
size_t unified_cache_evictable_expert_bytes(int device);

// Log budget summary (weights, runtime, available) for diagnostics
void unified_cache_log_budget_summary(int device);

// Seal the layout pool to prevent new chunk allocations after S1-PRELOAD.
void unified_cache_seal_layout_pool(int device);

// Check if the cache budget is exceeded (eviction exhausted but used > budget)
bool unified_cache_is_budget_exceeded(int device);

// Returns true if any unified cache instance has evicted weights from device memory.
// Thread-safe. One-way flag (once true, never resets).
bool unified_cache_has_evictions();

// Budget information exported for external consumers (e.g., llama_params_fit)
struct unified_budget_info {
    int    device_id;
    size_t total_vram;             // Total device memory
    size_t budget_bytes;           // Managed budget (total * pct - headroom)
    size_t weight_bytes;           // Current weight cache usage
    size_t runtime_bytes;          // KV + compute + staging + graph
    size_t total_committed;        // weight_bytes + runtime_bytes (single VRAM ledger)
    size_t available_for_weights;  // budget - runtime (what can hold weights)
    size_t total_available;        // budget - total_committed (what can still be allocated)
    int    budget_pct;             // GGML_SYCL_VRAM_BUDGET_PCT value used
    // model_exceeds_vram removed — unified non-blocking cache handles all model sizes
    // MoE expert breakdown (non-zero only for MoE models)
    size_t expert_weight_bytes;  // Total bytes for ALL expert tensors
    size_t active_expert_bytes;  // Estimated bytes for active experts only
    int    n_expert_total;       // Total experts per layer (e.g., 8, 128)
    int    n_expert_used;        // Experts per token (e.g., 2, 4)
};

// Get budget info for a device (thread-safe snapshot)
unified_budget_info unified_cache_get_budget_info(int device);

// Get margin in bytes for llama_params_fit (how much free space after weights + runtime)
// Returns 0 if budget is exceeded
size_t unified_cache_get_margin_bytes(int device);

// Check if KV cache should be offloaded to host pinned memory.
// Returns true when VRAM is too tight to hold both weights and KV cache,
// or when GGML_SYCL_HOST_COMPUTE=1 (KV on host avoids GPU island transitions).
// Override: GGML_SYCL_KV_HOST=1 forces host, GGML_SYCL_KV_HOST=0 forces device.
// kv_estimate_bytes: estimated KV cache size (0 = skip margin check)
bool unified_cache_should_offload_kv(int device, size_t kv_estimate_bytes = 0);

// Calculate effective weight bytes accounting for MoE expert sparsity.
// For an 8-expert top-2 model, only ~37.5% (1.5x active ratio) of expert weights are needed.
size_t compute_moe_effective_weight_bytes(size_t total_weight_bytes,
                                          size_t expert_total_bytes,
                                          int    n_expert,
                                          int    n_expert_used);

// === OneDNN FP16 Scratch Buffer API ===
// Reserve pre-allocated scratch buffers for oneDNN FP16 prompt processing.
// This avoids per-op allocations that cause OOM when KV cache is large.
//
// Call during model load with:
//   weights_size: max(N*K*2) across all matmuls (typically FFN: 14336*4096*2 = 117MB)
//   activations_size: max_batch * max_K * 2 (e.g., 512*14336*2 = 14.6MB)
//
// The buffers are reserved from the unified cache budget and reused across all matmuls.
bool unified_cache_reserve_onednn_scratch(int device_id, size_t weights_size, size_t activations_size);

struct pp_moe_onednn_scratch_result {
    void *     weight          = nullptr;
    void *     activation      = nullptr;
    void *     output          = nullptr;
    size_t     weight_size     = 0;
    size_t     activation_size = 0;
    size_t     output_size     = 0;
    mem_handle weight_owner;
    mem_handle activation_owner;
    mem_handle output_owner;
    bool       ok = false;
};

bool                         unified_cache_reserve_pp_moe_onednn_scratch(int      device_id,
                                                                         size_t   weight_slot_bytes,
                                                                         size_t   activation_slot_bytes,
                                                                         size_t   output_slot_bytes,
                                                                         uint32_t ring_depth);
pp_moe_onednn_scratch_result unified_cache_get_pp_moe_onednn_scratch_slot(int device_id, uint32_t slot);

// Get scratch buffers for oneDNN FP16 path. Returns pointers and acquires lock.
// Caller must call unified_cache_release_onednn_scratch() when done.
// Returns false if scratch not reserved or sizes exceed reserved.
struct onednn_scratch_result {
    void * weights     = nullptr;
    void * activations = nullptr;
    bool   ok          = false;
};

onednn_scratch_result unified_cache_get_onednn_scratch(int device_id, size_t weights_needed, size_t activations_needed);

// Release scratch buffers (unlocks mutex for other threads)
void unified_cache_release_onednn_scratch(int device_id);

// Check if scratch buffers are reserved
bool unified_cache_has_onednn_scratch(int device_id);

// Persistent scratch buffer public API
bool   unified_cache_reserve_persistent_scratch(int device_id, const char * buffer_name, size_t size_bytes, bool pin);
void * unified_cache_get_persistent_scratch(int device_id, const char * buffer_name);
void   unified_cache_release_persistent_scratch(int device_id, const char * buffer_name);
bool   unified_cache_has_persistent_scratch(int device_id, const char * buffer_name);
size_t unified_cache_get_persistent_scratch_size(int device_id, const char * buffer_name);

// === Bulk Weight Pinning API ===
// Pin/unpin all weights for a layer or entire model at once.
// Used by persistent kernels to ensure weights are not evicted during kernel lifetime.

// Pin all weights in a layer_weight_set. Returns count of successfully pinned entries.
// weights: pointer to layer_weight_set (opaque in C API, cast to layer_weight_set* internally)
int unified_cache_pin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout);

// Unpin all weights in a layer_weight_set.
void unified_cache_unpin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout);

// Pin entire model weights. Returns total count of pinned entries.
// layers: array of layer_weight_set, n_layers elements
int unified_cache_pin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout);

// Unpin entire model weights.
void unified_cache_unpin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout);

// === MoE Cache Helpers ===
// Unpin all experts
void unpin_all_experts();

// === Routing-Aware Expert Pre-staging ===
// These functions use routing indices from argsort to pre-stage only needed experts

// Result of pre-staging operation
struct prestage_result {
    int                                         n_unique = 0;
    bool                                        success  = false;
    std::vector<sycl::event>                    staging_events;
    int                                         n_gpu  = 0;
    int                                         n_cpu  = 0;
    int                                         n_miss = 0;
    std::unordered_map<int32_t, cache_location> expert_locations;
};

// Pre-stage only the experts identified by routing indices.
// Deduplicates expert IDs, resolves them from the preloaded placement state, and pins
// device-resident entries needed by the current dispatch.
//
// Parameters:
//   queue:           SYCL queue for the device (sycl::queue*)
//   expert_ids:      Routing indices from argsort [n_expert_used * n_tokens]
//   n_expert_used:   Experts per token (typically 4)
//   n_tokens:        Batch size
//   weight_base_ptr: mmap base pointer for expert weights
//   expert_stride:   Bytes between consecutive experts
//   expert_size:     Size of each expert in bytes
//   layer_id:        Layer ID for cache key
//   n_experts_total: Total experts for bounds checking (e.g., 128)
//   device_id:       Device ID for cache lookup
//
// Returns: prestage_result with counts and success status.
// `n_gpu` counts device-ready experts found in cache; `n_cpu` counts host-ready
// experts found in the host expert registry.
prestage_result prestage_routed_experts(void *          queue,
                                        const int32_t * expert_ids,
                                        int             n_expert_used,
                                        int             n_tokens,
                                        const void *    weight_base_ptr,
                                        size_t          expert_stride,
                                        size_t          expert_size,
                                        int             layer_id,
                                        int             n_experts_total,
                                        int             device_id,
                                        const char *    tensor_name,
                                        uint64_t        cache_uuid,
                                        uint32_t        model_id,
                                        ggml_type       tensor_type = GGML_TYPE_COUNT,
                                        int64_t         ne0         = 0,
                                        int64_t         ne1         = 0);

// Unpin routed experts after MoE computation completes.
// Call this after the MoE kernel finishes to allow eviction of these experts.
//
// Parameters:
//   expert_ids:      Same routing indices used in prestage_routed_experts
//   n_expert_used:   Experts per token
//   n_tokens:        Batch size
//   weight_base_ptr: mmap base pointer for expert weights
//   expert_stride:   Bytes between consecutive experts
//   layer_id:        Layer ID for cache key
//   n_experts_total: Total experts for bounds checking
//   device_id:       Device ID for cache lookup
void unpin_routed_experts(const int32_t * expert_ids,
                          int             n_expert_used,
                          int             n_tokens,
                          const void *    weight_base_ptr,
                          size_t          expert_stride,
                          int             layer_id,
                          int             n_experts_total,
                          int             device_id,
                          const char *    tensor_name,
                          uint64_t        cache_uuid,
                          uint32_t        model_id,
                          ggml_type       tensor_type = GGML_TYPE_COUNT,
                          int64_t         ne0         = 0,
                          int64_t         ne1         = 0);

// === Multi-Device Partial Row API (free-standing wrappers) ===

// Load a contiguous row range to a device with SOA reorder.
// Automatically routes to the correct unified_cache instance for the target device.
void * unified_cache_load_partial_rows(const char *               tensor_name,
                                       const ggml_sycl_cache_id & tensor_id,
                                       const void *               src_host,
                                       ggml_type                  type,
                                       int64_t                    ncols,
                                       int64_t                    row_start,
                                       int64_t                    row_count,
                                       int                        target_device);

// Look up a previously loaded partial weight on a specific device/range.
void * unified_cache_get_split_weight_ptr(const char *               tensor_name,
                                          const ggml_sycl_cache_id & tensor_id,
                                          ggml_type                  type,
                                          int64_t                    ncols,
                                          int64_t                    row_start,
                                          int64_t                    row_count,
                                          int                        device);

// Free all partial row entries on a device.
void unified_cache_free_partial_entries(int device);

// Register a unified cache for a device using an externally-managed queue.
// Used by tensor split to create a cache for a secondary GPU which has
// no llama.cpp backend and thus cannot use create_cache_for_device().
// Returns the cache pointer, or nullptr on failure.
unified_cache * unified_cache_register_for_queue(int device_id, sycl::queue & queue);

// === Unified Allocation API — Free-Standing Wrappers (Phase 3) ===
// Route to the unified_cache for the given device.  These are the preferred
// entry points for code outside unified-cache.cpp.

// Allocate VRAM (or host-pinned fallback) through the unified cache.
// Returns result with ptr, on_device flag, and actual size.
unified_cache::vram_alloc_result unified_cache_allocate(int                           device_id,
                                                        size_t                        size,
                                                        unified_cache::alloc_lifetime lifetime,
                                                        const char *                  tag = nullptr);

// Free a buffer previously obtained from unified_cache_allocate().
void unified_cache_deallocate(int device_id, void * ptr, size_t size, unified_cache::alloc_lifetime lifetime);

// Fill an existing unified-cache allocation by copying a host-pinned pattern
// buffer through mem_handle operands.  This avoids post-load raw device memset
// commands while preserving allocation/chunk leases for the destination.
bool unified_cache_fill_with_host_copy(int           device_id,
                                       void *        dst,
                                       size_t        size,
                                       uint8_t       value,
                                       sycl::queue & queue,
                                       const char *  tag = nullptr);

// Copy a host payload into an existing unified-cache allocation by first
// staging the payload in host-pinned memory and then copying through
// mem_handle operands.  The returned event retains the source staging handle
// and destination chunk lease until the copy completes.
bool unified_cache_copy_from_host_async(int                              device_id,
                                        void *                           dst,
                                        const void *                     src,
                                        size_t                           size,
                                        sycl::queue &                    queue,
                                        const std::vector<sycl::event> & deps,
                                        sycl::event *                    out_event,
                                        const char *                     tag = nullptr);

// Reserve the compute arena (pre-reserved VRAM for compute scratch).
// Must be called BEFORE S1-PRELOAD to guarantee VRAM availability.
bool unified_cache_reserve_compute_arena(int device_id, size_t arena_bytes);
bool unified_cache_ensure_planned_arena_zones(int device_id);

// Try to sub-allocate from the compute arena.
// Returns nullptr if arena is not reserved or has insufficient space.
[[deprecated("use unified_allocate() with prefer_vram_zone=SCRATCH instead")]] void * unified_cache_arena_alloc(
    int    device_id,
    size_t size);

// Release a compute-arena allocation.
void unified_cache_arena_free(int device_id, void * ptr, size_t size);

// Sub-allocate from the weight zone (persistent, NOT reset between tokens).
// Use for kernel infrastructure that persists for model lifetime.
void * unified_cache_arena_alloc_weight(int device_id, size_t size);

// Reset compute scratch state (call between graph_compute invocations).
void unified_cache_arena_reset(int device_id);

// Check if a pointer belongs to the compute arena.
bool unified_cache_arena_owns(int device_id, const void * ptr);

// Query compute arena capacity and usage.
size_t unified_cache_compute_arena_capacity(int device_id);
size_t unified_cache_compute_arena_used(int device_id);

// Query KV arena zone capacity and usage.
size_t unified_cache_kv_arena_capacity(int device_id);
size_t unified_cache_kv_arena_used(int device_id);

// Sum of zone_used(KV) + zone_used(ONEDNN) + zone_used(RUNTIME) + zone_used(SCRATCH).
// Returns 0 when arena is inactive.
size_t unified_cache_arena_non_weight_used(int device);

// Diagnostic only: log live unified_alloc records in a VRAM zone without
// reclaiming anything. Use this to find leaked mem_handle owners before
// changing lifetime logic.
void unified_cache_dump_live_zone_allocations(int          device,
                                              vram_zone_id zone,
                                              const char * where,
                                              size_t       max_entries = 32);

// Sub-allocate from the arena's KV zone for per-layer KV cache placement.
// Returns nullptr if arena is inactive or KV zone is exhausted.
// The returned pointer is VRAM (device-local) and must NOT be freed individually —
// it is owned by the arena and released when the arena is destroyed.
void * unified_cache_kv_arena_alloc(int device_id, size_t size);

// Reserve the inference scratch pool on a device.
bool unified_cache_reserve_scratch_pool(int device_id, size_t pool_bytes);

// Get/return scratch from the pool.
void * unified_cache_get_scratch(int device_id, size_t size);
void   unified_cache_return_scratch(int device_id, void * ptr, size_t size);
void   unified_cache_reset_scratch_pool(int device_id);

// MoE graph Q8_1 scratch demand derived from the selected MMVQ layout and
// activation shape.  This is a planning contract: graph callers allocate one
// owned unified-cache handle, then slice stable graph buffers from it.
struct moe_q8_1_scratch_shape {
    ggml_type        weight_type    = GGML_TYPE_COUNT;
    ggml_layout_mode layout         = GGML_LAYOUT_AOS;
    int64_t          input_cols     = 0;
    int64_t          input_rows     = 0;
    int              graph_op_count = 0;
    int              device_id      = -1;
    int              segment_id     = -1;
    int              layer_id       = -1;
    int              n_experts      = 0;
    int              n_expert_used  = 0;
};

struct moe_q8_1_scratch_demand {
    bool             supported                = false;
    ggml_type        weight_type              = GGML_TYPE_COUNT;
    ggml_layout_mode layout                   = GGML_LAYOUT_AOS;
    int64_t          input_cols               = 0;
    int64_t          input_cols_padded        = 0;
    int64_t          input_rows               = 0;
    int              buffer_count             = 0;
    int              device_id                = -1;
    int              segment_id               = -1;
    int              layer_id                 = -1;
    int              n_experts                = 0;
    int              n_expert_used            = 0;
    size_t           bytes_per_buffer         = 0;
    size_t           aligned_bytes_per_buffer = 0;
    size_t           total_bytes              = 0;
};

moe_q8_1_scratch_demand unified_cache_plan_moe_q8_1_scratch(const moe_q8_1_scratch_shape & shape);
bool                    unified_cache_reserve_moe_q8_1_scratch(int device_id, const moe_q8_1_scratch_demand & demand);
bool                    unified_cache_allocate_moe_q8_1_graph_scratch(int                             device_id,
                                                                      const moe_q8_1_scratch_demand & demand,
                                                                      alloc_handle *                  out);

// Grow the host SCRATCH zone to accommodate compute buffer needs.
// Called after the ggml scheduler is created and compute buffer sizes are known.
void unified_cache_grow_host_scratch_zone(size_t additional_bytes);

[[deprecated("use unified_allocate() with must_host_pinned + use_pinned_pool instead")]] void *
     unified_cache_host_zone_alloc(host_zone_id zone, size_t size, size_t alignment = 64);
void unified_cache_host_zone_reset(host_zone_id zone);

// Sub-allocate from a VRAM zone (ONEDNN, RUNTIME, KV scratchpads).
// Returns nullptr if the zone is full or the arena is inactive.
void * unified_cache_zone_alloc(int device_id, vram_zone_id zone, size_t size, size_t align = 256);

// Free a sub-allocation from a VRAM zone (TLSF reclaim).
void unified_cache_zone_free(int device_id, vram_zone_id zone, void * ptr);

// Reset a VRAM zone (TLSF coalescing reset — all sub-allocations become free).
void   unified_cache_zone_reset(int device_id, vram_zone_id zone);
void   unified_cache_reset_model_weight_entries(int device_id);
size_t unified_cache_host_zone_used(host_zone_id zone);
size_t unified_cache_host_zone_capacity(host_zone_id zone);
size_t unified_cache_host_zone_largest_free_block(host_zone_id zone);

// Allocate from expert budget (or host fallback).
unified_cache::vram_alloc_result unified_cache_allocate_expert(int device_id, size_t size);

// Query actual L0 free VRAM (with safety margin).
size_t unified_cache_available_device(int device_id);

// Query budget headroom (budget_ - used_).
size_t unified_cache_available_budget(int device_id);

// === Phase 4: Pre-allocated MoE Inference Buffers ===
// Pre-allocate device buffers for MoE dispatch at model/context init time
// instead of on-demand during inference.  This eliminates malloc_device calls
// during graph_compute_impl, preventing DEVICE_LOST from OOM.

// Model parameters needed to size MoE buffers.
struct moe_buffer_params {
    int     n_experts     = 0;    // Total experts per layer (e.g. 128)
    int     n_expert_used = 0;    // Experts selected per token (e.g. 4)
    int     max_batch     = 512;  // Max tokens per batch (PP max)
    int     n_moe_layers  = 0;    // Number of MoE layers (for sizing)
    int     n_moe_tensors = 0;    // MoE tensors per layer (gate/up/down = 3)
    int64_t n_embd        = 0;    // Embedding dimension (for IDs sizing)
};

// Pre-allocated MoE inference buffers for a single device.
// All stored memory state is held by mem_handles; getters resolve raw pointers
// only for immediate kernel/memcpy ABI use.
struct moe_inference_buffers {
    // Expert pointer tables: one per MoE tensor (gate/up/down * n_layers).
    // Each table is n_experts * sizeof(void*) bytes.
    std::shared_ptr<mem_handle> expert_ptr_tables_handle;
    int                         n_tables         = 0;  // n_moe_layers * n_moe_tensors
    size_t                      table_bytes      = 0;  // Bytes per table (n_experts * sizeof(void*))
    bool                        tables_on_device = false;

    // MoE IDs device staging buffer.
    // Size: max(n_expert_used * max_batch * sizeof(int32_t)) across all layers.
    // A single shared buffer is sufficient because MoE layers execute sequentially.
    std::shared_ptr<mem_handle> ids_staging_handle;
    size_t                      ids_staging_bytes = 0;
    bool                        ids_on_device     = false;

    // Compact selected-row pointer list for MoE MMVQ dispatch.
    // Size: max(n_expert_used * max_batch * sizeof(void*)).
    // A single shared buffer is sufficient because MoE layers execute
    // sequentially on a device queue.
    std::shared_ptr<mem_handle> compact_ptrs_handle;
    size_t                      compact_ptrs_bytes = 0;
    bool                        compact_on_device  = false;

    // Optional device-side missing flag used by validation/fallback builders.
    std::shared_ptr<mem_handle> compact_missing_handle;
    size_t                      compact_missing_bytes     = 0;
    bool                        compact_missing_on_device = false;

    // True if buffers have been allocated.
    bool initialized = false;
};

// Pre-allocate MoE inference buffers for a device.
// Call during moe_hybrid_init_once() after model architecture is known.
// Uses unified_cache_allocate() internally — respects budget and falls back to host.
// Returns true if all buffers were allocated successfully.
bool moe_preallocate_inference_buffers(int device_id, const moe_buffer_params & params);

// Get the pre-allocated MoE buffers for a device.
// Returns nullptr if not initialized.
const moe_inference_buffers * moe_get_inference_buffers(int device_id);

// Get a specific expert pointer table by index.
// Returns the device pointer for table[table_index], or nullptr if invalid.
void * moe_get_expert_ptr_table(int device_id, int table_index);

// Get the shared MoE IDs staging buffer.
// Returns nullptr if not pre-allocated or if needed_bytes exceeds the pre-allocated size.
void * moe_get_ids_staging(int device_id, size_t needed_bytes);

// Get the shared compact selected-row pointer list.
// Returns nullptr if not pre-allocated or if needed_bytes exceeds capacity.
void * moe_get_compact_ptrs(int device_id, size_t needed_bytes);

// Get the shared compact-list missing flag.
int * moe_get_compact_missing_flag(int device_id);

// Free all pre-allocated MoE buffers for a device (called during shutdown).
void moe_free_inference_buffers(int device_id);

// === Direct Staging API — Free-Standing Wrappers ===
// Route to the unified_cache for the given device.

direct_stage_result unified_cache_direct_stage_weight(int                  device_id,
                                                      ggml_sycl_cache_id   key,
                                                      const void *         src,
                                                      size_t               src_size,
                                                      size_t               dst_size,
                                                      ggml_layout_mode     layout,
                                                      cache_layout_fill_fn fill_fn,
                                                      const void *         fill_ctx,
                                                      sycl::queue *        queue,
                                                      mem_handle *         out_handle = nullptr);

direct_stage_result unified_cache_direct_stage_expert(int                  device_id,
                                                      ggml_sycl_cache_id   key,
                                                      const void *         src,
                                                      size_t               src_size,
                                                      size_t               dst_size,
                                                      ggml_layout_mode     layout,
                                                      cache_layout_fill_fn fill_fn,
                                                      const void *         fill_ctx,
                                                      sycl::queue *        queue,
                                                      mem_handle *         out_handle = nullptr);

direct_stage_result unified_cache_direct_stage_expert_tensor(int                                     device_id,
                                                             const std::vector<ggml_sycl_cache_id> & keys,
                                                             const void *                            src,
                                                             size_t                                  src_size,
                                                             size_t                                  expert_dst_size,
                                                             ggml_layout_mode                        layout,
                                                             cache_layout_fill_fn                    fill_fn,
                                                             const void *                            fill_ctx,
                                                             sycl::queue *                           queue,
                                                             std::vector<mem_handle> * out_handles = nullptr);

const weight_entry * unified_cache_lookup_weight(int device_id, ggml_sycl_cache_id key);

const weight_entry * unified_cache_lookup_expert(int device_id, ggml_sycl_cache_id key);

// === Raw allocation primitives (unified-cache.cpp owns all sycl::malloc/free calls) ===
// These are the ONLY functions allowed to call sycl::malloc_device/host or sycl::free.
// All other code must route through these or the higher-level unified_alloc/unified_cache_allocate.
void * unified_cache_raw_malloc_device(size_t size, const sycl::queue & queue);
void * unified_cache_raw_malloc_host(size_t size, const sycl::queue & queue);
void * unified_cache_raw_malloc_host(size_t size, const sycl::context & ctx);
bool   unified_cache_raw_free_device(void * ptr, const sycl::queue & queue);

// === Shutdown API ===

// Shutdown the unified cache system before SYCL runtime destruction
// Call this during ggml_backend_sycl_free() to avoid static destruction order issues
// After calling this, the cache destructors will skip sycl::free() calls
void shutdown_unified_cache();

// Returns true if SYCL runtime teardown has begun (atexit handler fired).
// Used by ExpertCache/ExpertPrefetcher to skip sycl::free() during static destruction.
bool ggml_sycl_is_shutting_down();

// (ExpertPlacementTable removed — the cache IS the placement.
//  Use is_expert_resident() / get_expert_device_ptr() for residency,
//  get_expert_popularity_rank() for eviction scoring.)

}  // namespace ggml_sycl

// === Cross-module Budget Queries ===
// These functions are defined in ggml-sycl.cpp but called from unified-cache.cpp
// to query tensor inventory state for budget calculations.

// Get the total model size from tensor inventory (for budget calculations)
size_t ggml_sycl_get_model_size();

// Get MoE expert memory breakdown (for budget calculations)
void ggml_sycl_get_moe_info(size_t * expert_total_bytes, int * n_expert, int * n_expert_used);

#endif  // GGML_SYCL_UNIFIED_CACHE_HPP
