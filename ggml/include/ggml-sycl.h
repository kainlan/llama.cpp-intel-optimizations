//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <limits.h>
#include <stdint.h>

#define GGML_SYCL_NAME        "SYCL"
#define GGML_SYCL_MAX_DEVICES 48

// Attention Q/accumulator dtype for the SYCL FLASH_ATTN_EXT path.
// Mirrors the `afloat`/`afloat2` typedef block in ggml/src/ggml-sycl/common.hpp:
// when the SYCL backend is built with GGML_SYCL_F16 the attention path expects
// an f16 Q input; otherwise it expects f32. Exposed here (pure C header) so
// src/llama-graph.cpp can cast Q at graph-build time without pulling a SYCL
// translation unit header.
#ifdef GGML_SYCL_F16
#    define GGML_SYCL_FATTN_Q_TYPE GGML_TYPE_F16
#else
#    define GGML_SYCL_FATTN_Q_TYPE GGML_TYPE_F32
#endif

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_sycl_init(int device);

GGML_BACKEND_API bool ggml_backend_is_sycl(ggml_backend_t backend);

// Copy the native UUID of this exact SYCL backend device. Returns false when
// device/uuid is null, device is not owned by this backend, or the compiler or
// runtime does not expose the 16-byte Intel device UUID extension.
GGML_BACKEND_API bool ggml_backend_sycl_get_device_uuid(ggml_backend_dev_t device, uint8_t uuid[16]);

// devide buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// KV cache buffer type (enforces safe per-allocation cap for chunking)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_kv_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);

// tensor parallel buffer type (Megatron-style column/row parallel with all-reduce)
// Initializes TP system on first call. Pass device_ids=NULL for auto-detection.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_tp_buffer_type(int n_devices, const int * device_ids);

// Get the TP world size (number of devices in TP group, 1 if TP not enabled)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_world_size(void);

// Get the TP rank for this process (0 if TP not enabled or single-process mode)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_rank(void);

// Check if running in multi-process TP mode
GGML_BACKEND_API bool ggml_backend_sycl_is_multiprocess_tp(void);
// Dynamic registry unload gate. False means exact LIVE model owners still need
// this module's teardown procedures and devices.
GGML_BACKEND_API bool ggml_backend_sycl_can_unload(void);
// Cancel a successful unload reservation if the generic loader cannot proceed.
GGML_BACKEND_API void ggml_backend_sycl_cancel_unload(void);
// Drain module-owned threads, queues and caches before ggml_backend_unload().
GGML_BACKEND_API void ggml_backend_sycl_shutdown(void);

// KV buffer type for a backend device (falls back to default buffer type if not SYCL)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_kv_buffer_type_from_dev(ggml_backend_dev_t device);

// Whether layer `il`'s KV currently lives on-device per the active placement
// plan. Returns true (today's tiered-device behavior) when no plan is active.
GGML_BACKEND_API bool ggml_backend_sycl_kv_layer_on_device_from_dev(ggml_backend_dev_t dev, int32_t il);

// Get the byte offset for reading this rank's shard from GGUF file
// For column-parallel tensors, this is the offset into the tensor data
// For row-parallel tensors, returns 0 (requires special handling due to interleaved data)
// tensor_name: the tensor name to check TP layer type
// tensor_ne: original tensor dimensions [ne0, ne1, ne2, ne3]
// tensor_type: ggml_type of the tensor
GGML_BACKEND_API size_t ggml_backend_sycl_get_tp_data_offset(const char *    tensor_name,
                                                             const int64_t * tensor_ne,
                                                             enum ggml_type  tensor_type);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);
// Exact current-registry host buffer type for a specific SYCL device. Returns
// NULL for foreign/stale devices; never falls back to a default device.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(ggml_backend_dev_t dev);

// Dedicated pinned-host buffer type for runtime-demoted KV layers (distinct
// identity/name "SYCL_KV_Host" from the generic host buft above, so the
// structural residency decline and diagnostics key on exactly this buft
// without perturbing other pinned-host consumers).
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_kv_host_buffer_type(void);

// Host compute buffer type - uses SYCL host memory (malloc_host) with SYCL buffer interface
// This is used for TP compute buffers to allow cross-device data sharing.
// Unlike host_buffer_type, this uses the SYCL buffer interface so it works with SYCL kernels.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_compute_buffer_type(int device);

// CPU-offload compute buffer: host-pinned memory with SYCL interface and is_host=true.
// Eliminates staging overhead for CPU-dispatched layers in cpu-dispatch.cpp.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_cpu_offload_compute_buffer_type(int device);
GGML_BACKEND_API bool                       ggml_backend_sycl_cpu_offload_available(void);

// Weight residency policy: true when dense weights should remain host-backed and streamed via unified cache.
GGML_BACKEND_API bool ggml_backend_sycl_weights_evictable(void);

// Set unified cache budget as a percentage of free VRAM (clamped 1..100).
GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_budget_pct(int pct);
// Set unified host cache budget as a percentage of total system RAM (clamped 1..100).
GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_host_budget_pct(int pct);

// Set per-tensor oneDNN pack M dimension for ONEDNN_PACKED/ONEDNN_WOQ layouts.
// pack_m <= 0 clears the override (falls back to GGML_SYCL_ONEDNN_PACK_M).
GGML_BACKEND_API void ggml_backend_sycl_set_onednn_pack_m(struct ggml_tensor * tensor, int64_t pack_m);

// Register a host-backed weight tensor for SYCL layout metadata/accessors.
GGML_BACKEND_API void ggml_backend_sycl_register_host_weight_tensor(ggml_backend_dev_t   dev,
                                                                    struct ggml_tensor * tensor);

// Cache identity for weights and MoE experts (no pointers, layout handled separately).
// load_scoped: when true, the complete model load/slot owner is part of identity
// model_id/load_txn_id/model_slot/slot_generation: exact lifecycle owner
// has_gguf/file_id/file_idx/file_offs/nbytes: GGUF-backed weights
// name_hash/type/ne: non-GGUF weights (fallback identity)
// aux_id: reserved for non-GGUF/MoE uniqueness (e.g., cache_uuid)
//
// A GGUF-backed weight is identified by WHERE ITS BYTES LIVE, not by which model
// or graph node happens to name them: (file_id, file_offs, nbytes, type, ne) is
// the whole identity, and model_id/name_hash are deliberately excluded from
// equality for such keys.  Two tensors over the same bytes -- a tied embedding
// and output head, or the same file opened by two models -- are one cache entry.
//
// file_idx is only a split index WITHIN one model, so it cannot separate models
// on its own; file_id is the whole-file identity that does, published through
// ggml_backend_sycl_register_gguf_file_identity().  A weight without GGUF
// identity keeps every logical field in its key and never shares.
struct ggml_sycl_cache_id {
    bool           valid;
    bool           load_scoped;
    uint64_t       model_id;
    uint64_t       load_txn_id;
    uint32_t       model_slot;
    uint64_t       slot_generation;
    bool           has_gguf;
    uint64_t       file_id;
    uint16_t       file_idx;
    size_t         file_offs;
    size_t         nbytes;
    uint64_t       name_hash;
    enum ggml_type type;
    int64_t        ne[GGML_MAX_DIMS];
    bool           tp_sharded;
    int            tp_rank;
    int            tp_world_size;
    int64_t        tp_local_ne[GGML_MAX_DIMS];
    int64_t        tp_offset_ne[GGML_MAX_DIMS];
    uint64_t       aux_id;
};

// Publish the physical identity of one GGUF split file, so that weight
// identities registered for that split carry a file identity that is unique
// across models rather than a per-model split index.  Call it for a split
// before registering its tensors.
// file_idx: GGUF split index, as passed to ggml_backend_sycl_register_weight_identity
// path: the split's file path ("" or NULL when the model was loaded from a
//       already-open handle and has no path)
// file_size: the split's byte size
// A split that is never published still gets a file identity, derived from the
// model instead: weights then never collide across models, but two models over
// the same file no longer share one cached copy.
GGML_BACKEND_API void ggml_backend_sycl_register_gguf_file_identity(uint16_t     file_idx,
                                                                    const char * path,
                                                                    uint64_t     file_size);

// Register GGUF metadata for stable weight identity in the unified cache.
// tensor: GGUF tensor
// model_id: unique per model load
// file_idx: GGUF split index
// file_offs: byte offset in the GGUF file
// tensor_nbytes: GGUF tensor byte size
//
// Normally called from inside a model load transaction, which supplies the
// owning model/load identity and makes model_id==0 the usual argument.  Outside
// one there is no lifecycle owner to attribute the registration to, so the
// caller MUST declare a non-zero model_id; the registration is then readable
// only while the process still has no model loaded.
GGML_BACKEND_API void ggml_backend_sycl_register_weight_identity(const struct ggml_tensor * tensor,
                                                                 uint16_t                   file_idx,
                                                                 size_t                     file_offs,
                                                                 size_t                     tensor_nbytes,
                                                                 uint64_t                   model_id);

// Weight usage categories for layout selection.
enum ggml_backend_sycl_tensor_usage {
    GGML_SYCL_TENSOR_USAGE_UNKNOWN = 0,
    GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_FFN_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_MOE_GATE,
    GGML_SYCL_TENSOR_USAGE_EMBEDDING,
    GGML_SYCL_TENSOR_USAGE_NORM,
};

// ABI-compatible registration entry point. The original API returned void.
GGML_BACKEND_API void ggml_backend_sycl_register_weight_usage(const char *                        tensor_name,
                                                              enum ggml_backend_sycl_tensor_usage usage);
// Status-bearing extension for callers that need transactional admission.
GGML_BACKEND_API bool ggml_backend_sycl_try_register_weight_usage(const char *                        tensor_name,
                                                                  enum ggml_backend_sycl_tensor_usage usage);

// Tensor inventory for tiered memory placement
struct ggml_sycl_tensor_info {
    const char *   name;
    size_t         size;
    enum ggml_type type;
    int64_t        ne[GGML_MAX_DIMS];
};

struct ggml_sycl_tensor_inventory {
    struct ggml_sycl_tensor_info * tensors;
    size_t                         count;
    size_t                         total_size;
    // Double-buffered FP16 weight staging needed by the PP dequant prefetch pipeline.
    // Computed from the largest quantized weight tensor as 2 x (n_elements * sizeof(fp16)).
    size_t                         pp_pipeline_scratch_bytes;
    // Ring-backed PP MoE oneDNN staging planned from the model inventory.
    // Slots hold one dequantized expert weight, one f16 activation tile, and one f32 output tile.
    // The current PP path serializes slot use, so the default plan reserves one reusable slot.
    size_t                         pp_moe_onednn_weight_slot_bytes;
    size_t                         pp_moe_onednn_activation_slot_bytes;
    size_t                         pp_moe_onednn_output_slot_bytes;
    size_t                         pp_moe_onednn_scratch_bytes;
    // llama.cpp-ibj0: per-row bytes behind the two slot sizes above (activation
    // slot = align256(n_ubatch * pp_moe_onednn_activation_bytes_per_row), output
    // slot analogous) -- carried into the backend so the runtime-context
    // transaction can re-plan the ring for the REAL runtime n_ubatch instead of
    // the load-time default this struct's n_ubatch field below is fixed to (512).
    // 0 for a dense model (no MoE PP ring).
    size_t                         pp_moe_onednn_activation_bytes_per_row;
    size_t                         pp_moe_onednn_output_bytes_per_row;
    uint32_t                       pp_moe_onednn_ring_depth;
    int                            n_expert;       // Total experts per layer (0 for dense models)
    int                            n_expert_used;  // Experts activated per token (0 for dense models)
    // Model hparams for KV cache size estimation (used by VRAM budget coordination)
    uint32_t                       n_layer;       // Number of transformer layers
    uint32_t                       n_embd_k_gqa;  // Key embedding dim (GQA-adjusted), per layer
    uint32_t                       n_embd_v_gqa;  // Value embedding dim (GQA-adjusted), per layer
    uint32_t                       n_ctx;         // Context size (tokens)
    uint32_t                       n_ubatch;      // Physical batch size (for SWA KV sizing)
    // SWA (Sliding Window Attention) info for models with heterogeneous attention
    uint32_t                       n_swa;                 // Sliding window size (0 = no SWA)
    uint32_t                       n_swa_layers;          // Number of SWA layers (0 = all full-attn)
    const bool *                   swa_layer_mask;        // Per-layer SWA flag [n_layer], NULL if no SWA
    uint32_t                       swa_layer_mask_count;  // Length of swa_layer_mask (must == n_layer)
    // llama.cpp-o3a0: max query-head count across all layers ELIGIBLE for the
    // oneDNN SDPA route, split by attention window class -- non-SWA layers
    // (effective KV window == n_ctx) vs SWA layers (effective KV window ==
    // min(n_ctx, n_swa + n_ubatch), the span a ubatch of n_ubatch queries
    // scans over the n_swa-key sliding window above, GPU-verified). Feeds
    // the window-aware oneDNN Graph-scratch zone floor (unified-cache.cpp's
    // onednn_graph_scratch_zone_floor_bytes_swa()), replacing the single
    // n_head_max (llama.cpp-0oxf) that assumed every oneDNN-served layer's
    // window was n_ctx -- wrong for SWA models: gemma4 E4B (real GGUF
    // attention.sliding_window=512, n_ubatch=512, n_ctx=8192) computes
    // 24 MiB raw, clamped to 64 MiB, vs. the 192 MB the flat formula
    // predicted. The ticket's original "window=1024" figure was
    // numerically right at n_ubatch=512 (512 + 512 == 1024) and wrong
    // only about WHY -- the true window is n_swa + n_ubatch, not the raw
    // sliding_window value alone. "Eligible" mirrors
    // ggml_sycl_flash_attn_ext_onednn_plan()'s D-based gate
    // (fattn-onednn.cpp) as closely as a llama-layer file can: see
    // llama_model_sycl_onednn_head_dim_eligible() in llama-model.cpp, which
    // cannot include that SYCL-only source and so replicates the rule (kept
    // in sync by test-sycl-onednn-graph-floor-eligibility-source.py). 0 if no
    // eligible layer of that class exists. Added at the end of the struct
    // (not inserted among the existing fields) so every existing
    // `ggml_sycl_tensor_inventory x = {};` zero-init call site stays correct
    // without being touched.
    uint32_t                       n_head_ctx_max;
    uint32_t                       n_head_swa_max;
    // llama.cpp-rqak: max query-head count across ALL attention layers,
    // eligibility ignored -- unlike n_head_ctx_max/n_head_swa_max above,
    // which only cover layers ELIGIBLE for the oneDNN SDPA route. This is
    // the non-FA batched mul_mat attention guard's input
    // (ggml_sycl_check_nonfa_attn_scratch() in ggml-sycl.cpp): that path
    // runs on every attention layer regardless of oneDNN eligibility, so
    // its demand model needs the true all-layers maximum -- the two
    // per-class fields above can both be 0 for a model where every layer
    // is oneDNN-ineligible (e.g. a DeepSeek-V3-class model, D=576), which
    // would leave that model's non-FA scratch demand unguarded.
    uint32_t                       n_head_all_max;
};

// SYCL-side projection of the four placement-envelope fields the llama
// layer carries on llama_model_params.  Populated by the llama layer at
// model load and snapshotted into a SYCL file-static; the planner reads
// it directly so the envelope's source attribution is explicit (no
// derivation through the inventory snapshot).  flash_attn_type mirrors
// the llama_flash_attn_type enum as a stable int32 to keep this header
// free of cross-layer includes.
struct ggml_sycl_placement_envelope {
    uint32_t n_ctx;            // per-slot ctx ceiling; 0 = use model default
    uint32_t n_ubatch;         // ubatch ceiling; 0 = use 512 default
    uint32_t n_seq_max;        // max active sequences / slots; default 1
    int32_t  flash_attn_type;  // -1 = AUTO, 0 = DISABLED, 1 = ENABLED
};

// Set tensor inventory for tiered memory placement.
// Must be called after model metadata parsing, before tensor allocation.
// This enables automatic VRAM/host placement based on tensor priority.
// Direct calls retain legacy planning/late side effects; they do not guarantee
// eager host-zone provisioning. The loader uses stage_inventory_plan(..., false)
// to attempt that provisioning before buffer allocation, under lifecycle guards.
GGML_BACKEND_API void ggml_backend_sycl_set_tensor_inventory(ggml_backend_t                            backend,
                                                             const struct ggml_sycl_tensor_inventory * inventory);

// Compute the placement plan EARLY — before per-tensor create_tensor calls.
// Allows create_tensor's per-tensor buft selection to consult the plan.
// Skips late-only side effects (layer-streaming setup, host-pinned pre-allocate)
// that depend on state populated by create_tensor; those still run when
// ggml_backend_sycl_set_tensor_inventory is called later.  Idempotent.
GGML_BACKEND_API void ggml_backend_sycl_compute_placement_plan_early(
    ggml_backend_t                            backend,
    const struct ggml_sycl_tensor_inventory * inventory);

// Sentinel returned by ggml_backend_sycl_planned_target_device when no plan
// applies to the queried tensor (no plan computed, or no entry for the name).
// Distinct from -1 which is a valid "host" placement.
#define GGML_SYCL_PLANNED_NO_PLAN INT_MIN

// Query the planned target device for a weight tensor by name.
//   - GGML_SYCL_PLANNED_NO_PLAN: no plan applies (caller should fall through
//     to the default buft selection).
//   - -1: planned for host (CPU/host-pinned).
//   - >= 0: planned for SYCL device with that index.
//
// Looks up against the dense-weight index of the active placement_plan.  MoE
// expert entries are not name-indexed and are not returned by this query;
// they continue to use the per-expert dispatch path.
GGML_BACKEND_API int ggml_backend_sycl_planned_target_device(const char * tensor_name);

// Set the placement envelope (snapshot of llama_model_params capacity inputs)
// for this backend.  Must be called at model load alongside set_tensor_inventory.
// The setter copies into a file-static; the caller's struct does not need to
// outlive the call.  Passing NULL clears the envelope (planner falls back to
// inventory-derived sizing).
GGML_BACKEND_API void ggml_backend_sycl_set_placement_envelope(ggml_backend_t                              backend,
                                                               const struct ggml_sycl_placement_envelope * envelope);

// Update the planner metadata with the runtime context shape for the current
// inference context. This does not retroactively re-place already loaded
// weights, but it lets KV/runtime consumers size cache/control allocations from
// the active context instead of the model's training context.
//
// llama.cpp-oyfl: flash_attn_enabled is the caller's cparams.flash_attn,
// not the raw llama_flash_attn_type -- but at the constructor's own call
// (llama_context::llama_context, llama-context.cpp) an AUTO
// llama_flash_attn_type has not been resolved yet and cparams.flash_attn
// reads an optimistic `true` (see its init, same file): "not yet known to
// be off" and "known to be on" are indistinguishable at that point, and
// the guard is written to skip on true, so this call correctly does
// nothing for an AUTO context until it resolves. That resolution and the
// narrow re-check that follows it (ggml_backend_sycl_recheck_runtime
// _context_flash_attn() below) are what actually evaluate an AUTO context
// that turns out to resolve OFF. When flash_attn_enabled is false here (a
// context whose flash attention is definitely off already), this also
// checks whether the non-flash-attention batched mul_mat path's
// scratch demand at (n_ctx, n_ubatch) fits the SCRATCH zone the arena
// already reserved, and
// refuses the update (logging the size arithmetic, same style as the KV
// budget refusal) instead of leaving a shape that would abort mid-prefill.
// This predicate is EMPIRICAL, not a modeled worst case: measured on both
// discrete cards, any demand above the zone aborts regardless of how much
// VRAM is free outside the fixed arena (a ~2-4 GB outside-arena consumer
// specific to this path is unexplained and tracked separately,
// llama.cpp-k1ev) -- an earlier revision of this comment described a
// live-free-VRAM predicate that hardware measurement falsified; do not
// reintroduce it without first closing k1ev.
GGML_BACKEND_API void ggml_backend_sycl_set_runtime_context(ggml_backend_t backend,
                                                            uint32_t       n_ctx,
                                                            uint32_t       n_ubatch,
                                                            uint32_t       n_seq_max,
                                                            bool           flash_attn_enabled);

// llama.cpp-tsfl (round 1 F10; round 4 Q1/Q6): per-device count of
// SUCCESSFUL host-pinned fallbacks for any of the buffer types whose
// alloc_buffer is, or delegates to,
// ggml_backend_sycl_buffer_type_alloc_buffer() (ggml-sycl.cpp) -- the name
// is kept from the plan's own Task 4b read,
// which cares specifically about compute buffers, but is not limited to
// them. Exactly two sites increment it, both only once a
// SUCCESSFUL host-pinned landing is confirmed, never merely attempted: (1)
// a single allocation exceeding the safe device-alloc limit, forced
// host-pinned and counted once that forced attempt lands; and (2) a device
// allocation that failed and was RETRIED host-pinned, counted only once
// that retry itself succeeds -- in both cases, an attempt that also fails
// falls through to the allocation-failure ERROR and is not a "fallback".
// NOT counted: the !vram_arena_enabled() < 512 MB headroom branch (dead in
// this fork's default arena-on configuration), the silent
// must_device=false routing that lets the unified cache choose VRAM or
// host-pinned on its own, or the is_kv_buft branch that routes a KV
// allocation to effective_mem_type=GGML_SYCL_MEM_HOST when it does not fit
// zone_available(KV) (ggml-sycl.cpp) -- none of the three logs or
// increments this counter, so a zero reading does not prove every compute
// buffer or KV allocation actually stayed on the device. This is a DELTA,
// not a lifetime total, and the reset is PER-DEVICE: the runtime-context
// transaction resets only ctx->device's own counter to 0 on its own
// successful completion (after the plan has been published), so on a
// multi-device plan every OTHER device's counter is left as a lifetime
// total until that device's own runtime-context transaction succeeds --
// a caller reading it sees "fallbacks since the last successful
// runtime-context transaction FOR THIS DEVICE". Returns 0 for an
// out-of-range device.
GGML_BACKEND_API uint64_t ggml_backend_sycl_compute_buffer_host_fallbacks(int device);

// llama.cpp-nphx: whether the SYCL auto micro-batch selection trial
// (llama_context::sycl_select_auto_ubatch(), llama.cpp-xojq Task 4b) is
// enabled -- GGML_SYCL_AUTO_UBATCH, default ON.
GGML_BACKEND_API bool ggml_backend_sycl_auto_ubatch_enabled(void);

// llama.cpp-7n6n (wires nphx Task 5): the persisted auto n_ubatch tuning
// cache -- a small on-disk store at
// ~/.cache/llama.cpp/sycl-tuning/<sanitized device name>-ubatch.json (see
// tuning-cache-io.hpp's "Ubatch Cache Entry (v2)" section for the format)
// that lets a repeat start of the SAME device+model+context shape skip
// sycl_select_auto_ubatch()'s ladder entirely. `device` is the same logical
// SYCL device index used throughout this header (indexes
// ggml_sycl_info().devices[]); the device name and driver version making up
// the on-disk key are read from there, never passed in, so callers never
// duplicate that lookup. `model_name`/`model_size`/`model_hash` identify the
// exact set of loaded tensors (GGUF general.name, llama_model::size()'s
// total tensor bytes, and a cheap FNV-1a hash over each tensor's (name,
// byte size)) -- the same model file copied elsewhere hits; a re-quantised
// file changes tensor byte sizes and so misses. `n_seq_max`/`type_k`/
// `type_v` round out the shapes that can change which
// ladder candidates fit without changing anything the rest of the key
// tracks. `device_set_hash` (same finding) is a hash over every SYCL
// device's dev_index this context actually uses, in order -- `device`
// alone names only the FIRST one, so a single-GPU and a multi-GPU run that
// both start with the same device 0 would otherwise share one entry even
// though the real demand differs. All pointer fields are borrowed: valid
// only for the duration of the call, never retained.
struct ggml_sycl_ubatch_cache_key {
    int          device;
    const char * model_name;
    uint64_t     model_size;
    uint64_t     model_hash;
    uint32_t     n_ctx;
    uint32_t     n_batch;
    bool         flash_attn;
    uint32_t     n_seq_max;
    int32_t      type_k;
    int32_t      type_v;
    uint32_t     device_set_hash;
};

// Whether the persisted auto n_ubatch cache is enabled -- GGML_SYCL_TUNING_CACHE,
// default ON. "0" disables BOTH lookup and store below (every call then
// returns false without touching the filesystem); any other non-empty value
// is treated as enabled, with one WARN.
GGML_BACKEND_API bool ggml_backend_sycl_ubatch_cache_enabled(void);

// Resolve the on-disk path `device`'s cache file would use (honours
// GGML_SYCL_TUNING_CACHE_DIR; falls back to XDG per tuning-cache-io.hpp's
// get_cache_dir()), for diagnostics -- resolved unconditionally, even when
// the cache is disabled, so a WARN can still name where it would have
// written. Returns false (buf left untouched) for an out-of-range device or
// when `path` would not fit in `buf_size` bytes including the terminating
// NUL (this used to truncate silently).
GGML_BACKEND_API bool ggml_backend_sycl_ubatch_cache_path(int device, char * buf, size_t buf_size);

// Look up a previously-persisted auto n_ubatch for this exact key, along
// with the REASON it was stored with, copied into `reason_buf` and
// truncated to fit `reason_buf_size` including the terminating NUL
// (unlike `ggml_backend_sycl_ubatch_cache_path()`, which refuses instead:
// a truncated reason fails safe to non-terminal, a truncated path would
// not); `reason_buf`/`reason_buf_size` may be null/0 to skip it. The caller uses
// the reason to tell a TERMINAL outcome ("ladder exhausted", "MoE GPU
// routing ceiling") from one that merely lost a transient race, and decides
// from that whether to trust the cached value outright or resume searching
// above it. Returns false (leaving *n_ubatch and reason_buf untouched) on a
// cache miss, a disabled cache, an out-of-range device, or an
// unreadable/corrupt/wrong-version file -- the caller's ladder trial
// tolerates every one of those identically (a cold cache), so this never
// throws and never distinguishes them.
GGML_BACKEND_API bool ggml_backend_sycl_ubatch_cache_lookup(const struct ggml_sycl_ubatch_cache_key * key,
                                                            uint32_t *                                n_ubatch,
                                                            char *                                    reason_buf,
                                                            size_t                                    reason_buf_size);

// Persist the chosen n_ubatch for this exact key (atomic write; see
// tuning-cache-io.hpp). `reason` is recorded verbatim -- llama.cpp-7n6n
// made the only caller (llama_context::sycl_select_auto_ubatch()) pass
// its OWN actual stop reason here (one of the eight-string vocabulary that
// function documents), not a fixed "ladder" literal: doing so is what lets
// a future lookup on this entry (above) tell a terminal outcome from a
// transient one. That caller never passes "cached" (a hit that produces an
// unchanged outcome does not re-store at all) or "transaction busy"/"not
// the published model" (pure races it explicitly skips storing). Returns
// false (never throws) on a disabled cache, an out-of-range device, or a
// write failure -- the caller logs one WARN and continues; a failed store
// never blocks inference.
GGML_BACKEND_API bool ggml_backend_sycl_ubatch_cache_store(const struct ggml_sycl_ubatch_cache_key * key,
                                                           uint32_t                                  n_ubatch,
                                                           const char *                              reason);

// Provide the actual layer membership for the next KV buffer allocation on a
// SYCL device. llama_kv_cache may create multiple same-sized KV buffers for
// heterogeneous attention (for example non-SWA and SWA layers); the SYCL
// unified-cache planner uses this mask instead of inferring membership from
// byte size.
//
// The staged mask is correlated with the model load that staged it, so a mask
// left behind by an aborted or skipped allocation can never be applied to a
// different model. Returns a handoff id (0 if nothing was staged) to be passed
// to ggml_backend_sycl_cancel_kv_layer_mask_from_dev once the allocation window
// closes; the caller is expected to cancel unconditionally, since a mask that
// survives its own allocation is orphaned whatever the outcome was.
GGML_BACKEND_API uint64_t ggml_backend_sycl_push_kv_layer_mask_from_dev(ggml_backend_dev_t dev,
                                                                        const uint8_t *    layer_mask,
                                                                        uint32_t           layer_count);

// Drop a mask staged by ggml_backend_sycl_push_kv_layer_mask_from_dev that was
// never consumed. A no-op when the allocation already consumed it.
GGML_BACKEND_API void ggml_backend_sycl_cancel_kv_layer_mask_from_dev(ggml_backend_dev_t dev, uint64_t handoff_id);

// Backward-compatible helper for callers that only know n_ctx.
GGML_BACKEND_API void ggml_backend_sycl_set_runtime_n_ctx(ggml_backend_t backend, uint32_t n_ctx);

// Report actual tiered placement for the model currently being loaded/loaded.
// Returns true only when its completed placement_plan has host_bytes > 0.
// This is separate from unified-cache availability and resets at each outer
// model-load boundary.
GGML_BACKEND_API bool ggml_backend_sycl_is_tiered_enabled(ggml_backend_t backend);

// ggml_backend_sycl_model_exceeds_vram removed — unified non-blocking cache
// handles all model sizes without model-size branching.

// Get VRAM budget available for weights (budget minus runtime reservations).
// Returns 0 if backend is NULL or cache is not initialized.
GGML_BACKEND_API size_t ggml_backend_sycl_get_vram_budget(ggml_backend_t backend);

// Get free margin in bytes after weights + runtime allocations.
// Returns 0 if budget is exceeded. Used by llama_params_fit.
GGML_BACKEND_API size_t ggml_backend_sycl_get_vram_margin(ggml_backend_t backend);

// Check whether the unified tensor cache/dispatch gate is enabled. This is a
// cache-capability query, independent of the current model's planner placement.
GGML_BACKEND_API bool ggml_backend_sycl_has_tensor_cache(ggml_backend_t backend);

// Feed actual compute buffer sizes (from ggml_backend_sched_get_buffer_size) back to
// the unified cache zone planner.  Call once per context, after graph_reserve() completes
// and backend_buf_exp_size[] has been populated, but BEFORE the first graph_compute call.
// VRAM arena zones are a fixed pre-allocated block — they cannot grow after creation.
// This function pre-sizes the compute arena and host SCRATCH zone using the true
// scheduler-derived sizes, replacing the heuristic defaults set at model load time.
// - Updates the VRAM compute arena reservation with the actual scheduler requirement.
// - Grows the host pinned SCRATCH zone for host-side compute buffers (oneDNN reorder, etc.).
// sizes[i] is the compute buffer size for backend i; n_sizes is the length of sizes[].
// NULL sizes or n_sizes == 0 is a no-op.
GGML_BACKEND_API void ggml_backend_sycl_notify_compute_buffer_sizes(ggml_backend_t backend,
                                                                    const size_t * sizes,
                                                                    int            n_sizes);

// Get cache hit/miss statistics (stub — returns zeros).
// hits/misses may be NULL if caller doesn't need that stat.
GGML_BACKEND_API void ggml_backend_sycl_get_cache_stats(ggml_backend_t backend, uint64_t * hits, uint64_t * misses);

// Get a stable cache identity for a weight tensor on the specified device.
GGML_BACKEND_API struct ggml_sycl_cache_id ggml_backend_sycl_get_weight_cache_key(const struct ggml_tensor * tensor,
                                                                                  int                        device);

// Get a cache identity for ANY tensor (weight or non-weight) on the specified device.
// Uses data pointer as unique identifier for non-GGUF tensors. Suitable for caching
// mmap'd non-weight tensors like MoE ids, get_rows indices, etc.
GGML_BACKEND_API struct ggml_sycl_cache_id ggml_backend_sycl_get_tensor_cache_key(const struct ggml_tensor * tensor,
                                                                                  int                        device);

GGML_BACKEND_API void ggml_backend_sycl_print_sycl_devices(void);
GGML_BACKEND_API void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len);
GGML_BACKEND_API void ggml_backend_sycl_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API int  ggml_backend_sycl_get_device_count();
GGML_BACKEND_API void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total);

// Check if MoE multi-GPU mode is requested via GGML_SYCL_MOE_MULTI_GPU=1.
// Disabled by default while multi-device Level Zero execution is unstable.
// When true, the SYCL backend handles multi-device MoE dispatch internally,
// and secondary GPUs should NOT be exposed to the backend scheduler.
GGML_BACKEND_API bool ggml_backend_sycl_moe_multi_gpu_requested(void);
GGML_BACKEND_API void ggml_backend_sycl_set_debug(int level);

// llama.cpp-xojq (nphx Task 4b): the GPU MoE routing ceiling
// (MOE_GPU_UBATCH_MAX, ggml-sycl/moe-control-plan.hpp) exposed for
// llama_context's auto micro-batch ladder. Above this micro-batch the
// routing-id/compaction table stops being a control-sized allocation and
// moe_build_context_control_layout() refuses the GPU route
// (exceeds_gpu_ubatch), so the trial must not try a larger candidate on a
// MoE model (hparams.n_expert > 0) until llama.cpp-ohkx lifts the ceiling.
GGML_BACKEND_API uint32_t ggml_backend_sycl_moe_gpu_ubatch_max(void);

// Device-to-host memcpy using the SYCL backend queue for the tensor's buffer.
// This avoids mixing queues/contexts in tests.
GGML_BACKEND_API void ggml_backend_sycl_memcpy_d2h(const struct ggml_tensor * tensor, void * dst, size_t size);

// SYCL doesn't support registering host memory, keep here for reference
// GGML_BACKEND_API bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
// GGML_BACKEND_API void ggml_backend_sycl_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_sycl_reg(void);

// Flash attention sequence IDs for multi-sequence batching
// Set host pointers for seq_ids arrays (called from llama layer before graph execution)
// These are stored in thread-local storage and used by fattn kernel
// The pointers must be valid USM host memory (allocated by SYCL_Host buffer)
GGML_BACKEND_API void ggml_backend_sycl_set_seq_ids_host(const int32_t * q_seq_ids,
                                                         size_t          q_count,
                                                         const int32_t * kv_seq_ids,
                                                         size_t          kv_count);

// Clear the seq_ids host pointers (called after graph execution)
GGML_BACKEND_API void ggml_backend_sycl_clear_seq_ids_host(void);

// Pipeline parallelism (vLLM-style layer split with chunked prefill)
// Initialize PP with specified devices and layer count. layers_per_stage can be NULL for even distribution.
GGML_BACKEND_API void ggml_backend_sycl_pp_init(const int * device_ids,
                                                int         n_devices,
                                                int         total_layers,
                                                const int * layers_per_stage);

// Clean up PP resources
GGML_BACKEND_API void ggml_backend_sycl_pp_free(void);

// Check if PP is enabled
GGML_BACKEND_API bool ggml_backend_sycl_pp_enabled(void);

// Get number of PP stages
GGML_BACKEND_API int ggml_backend_sycl_pp_num_stages(void);

// Get device ID for a given layer
GGML_BACKEND_API int ggml_backend_sycl_pp_get_device_for_layer(int layer);

// Set chunked prefill configuration
GGML_BACKEND_API void ggml_backend_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled);

// ===========================================================================
// GPU Sampling API (Multi-step decode support)
// These functions run sampling entirely on GPU, avoiding CPU sync overhead
// ===========================================================================

// GPU sampler handle (opaque pointer)
typedef struct ggml_sycl_sampler * ggml_sycl_sampler_t;

// Create a GPU sampler for the given backend
// n_vocab: vocabulary size (determines work buffer sizes)
// seed: RNG seed for probabilistic sampling
GGML_BACKEND_API ggml_sycl_sampler_t ggml_backend_sycl_sampler_create(ggml_backend_t backend,
                                                                      int            n_vocab,
                                                                      uint32_t       seed);

// Free GPU sampler resources
GGML_BACKEND_API void ggml_backend_sycl_sampler_free(ggml_sycl_sampler_t sampler);

// Sample a token from logits tensor on GPU
// logits_tensor: must be a SYCL tensor of shape [n_batch, n_vocab]
// temp: temperature (0 = greedy, 1 = no scaling)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token(ggml_sycl_sampler_t sampler,
                                                        ggml_tensor *       logits_tensor,
                                                        float               temp);

// Sample from a specific index in a batched logits tensor
// logits_tensor: SYCL tensor of shape [n_batch, n_vocab]
// idx: batch index to sample from (0 to n_batch-1)
// temp: temperature (0 = greedy, 1 = no scaling)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_idx(ggml_sycl_sampler_t sampler,
                                                            ggml_tensor *       logits_tensor,
                                                            int                 idx,
                                                            float               temp);

// Sample from a specific index with full sampling parameters
// logits_tensor: SYCL tensor of shape [n_batch, n_vocab]
// idx: batch index to sample from (0 to n_batch-1)
// temp: temperature (0 = greedy, >0 = probabilistic)
// top_k: top-k filtering (0 = disabled)
// top_p: top-p/nucleus filtering (1.0 = disabled)
// min_p: min-p filtering (0.0 = disabled)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_full(ggml_sycl_sampler_t sampler,
                                                             ggml_tensor *       logits_tensor,
                                                             int                 idx,
                                                             float               temp,
                                                             int                 top_k,
                                                             float               top_p,
                                                             float               min_p);

// Async version - submit sampling kernels without waiting
// Use ggml_backend_sycl_sample_token_get() to retrieve result later
GGML_BACKEND_API void ggml_backend_sycl_sample_token_async(ggml_sycl_sampler_t sampler,
                                                           ggml_tensor *       logits_tensor,
                                                           float               temp);

// Get result from async sampling (blocks until complete)
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_get(ggml_sycl_sampler_t sampler);

// ===========================================================================
// Multi-step GPU Sampling API
// For generating multiple tokens without CPU sync between steps
// Tokens are stored in a device-side ring buffer
// ===========================================================================

// Reset the token buffer before starting a new multi-step generation
// Call this at the beginning of each multi-step batch
GGML_BACKEND_API void ggml_backend_sycl_sampler_reset_buffer(ggml_sycl_sampler_t sampler);

// Sample a token and store it in the device-side ring buffer
// This does NOT sync to host - use for multi-step decode
// Returns the index in the buffer where the token was written
GGML_BACKEND_API int ggml_backend_sycl_sample_token_to_device(ggml_sycl_sampler_t sampler,
                                                              ggml_tensor *       logits_tensor,
                                                              float               temp);

// Sample a token with full parameters and store in device-side ring buffer
// Supports top-k, top-p, min-p filtering for multi-step decode
// Returns the index in the buffer where the token was written
GGML_BACKEND_API int ggml_backend_sycl_sample_token_to_device_full(ggml_sycl_sampler_t sampler,
                                                                   ggml_tensor *       logits_tensor,
                                                                   float               temp,
                                                                   int                 top_k,
                                                                   float               top_p,
                                                                   float               min_p);

// Get device pointer to token at specific buffer index (for embedding lookup)
// Returns NULL if index is out of range or buffer not allocated
GGML_BACKEND_API int32_t * ggml_backend_sycl_get_sampled_token_ptr(ggml_sycl_sampler_t sampler, int index);

// Get device pointer to the most recently sampled token
// Convenience function for feeding back to next decode step
GGML_BACKEND_API int32_t * ggml_backend_sycl_get_current_token_ptr(ggml_sycl_sampler_t sampler);

// Copy tokens from device buffer to host (sync point)
// tokens: host buffer to receive tokens
// max_tokens: size of host buffer
// Returns: number of tokens copied (min of token_count and max_tokens)
GGML_BACKEND_API int ggml_backend_sycl_get_sampled_tokens(ggml_sycl_sampler_t sampler,
                                                          int32_t *           tokens,
                                                          int                 max_tokens);

// Get the number of tokens currently in the buffer (since last reset)
GGML_BACKEND_API int ggml_backend_sycl_get_token_count(ggml_sycl_sampler_t sampler);

// Get the maximum buffer size (number of tokens that can be stored)
GGML_BACKEND_API int ggml_backend_sycl_get_token_buffer_size(void);

// ===========================================================================
// Speculative Decoding Verification API
// For verifying draft tokens against model logits entirely on GPU
// ===========================================================================

// Verify draft tokens against batched logits from multi-token decode
// all_logits: SYCL tensor of shape [n_outputs, n_vocab] - logits for all batch positions
// draft_tokens: array of draft token IDs [n_draft] (on host)
// n_draft: number of draft tokens to verify
// logits_offset: index into all_logits to start verification (typically 1, since logits[0] is for the base token)
// Returns: number of accepted tokens (longest matching prefix where argmax == draft)
// Note: Compares draft[i] with argmax(logits[logits_offset + i]) for i = 0..n_draft-1
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative(ggml_sycl_sampler_t sampler,
                                                          ggml_tensor *       all_logits,
                                                          const int32_t *     draft_tokens,
                                                          int                 n_draft,
                                                          int                 logits_offset);

// Extended version that also returns the sampled tokens (argmax at each position)
// This is needed for llama-lookup and other speculative decoding tools
// sampled_tokens_out: [n_draft] host array to receive the argmax token at each position
// Returns: number of accepted tokens (same as ggml_backend_sycl_verify_speculative)
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_with_tokens(ggml_sycl_sampler_t sampler,
                                                                      ggml_tensor *       all_logits,
                                                                      const int32_t *     draft_tokens,
                                                                      int32_t *           sampled_tokens_out,
                                                                      int                 n_draft,
                                                                      int                 logits_offset);

// Version that takes raw GPU pointer - for use with persistent GPU logits buffer
// gpu_logits: GPU pointer to logits data [n_outputs, n_vocab] in row-major order
// n_vocab: vocabulary size (inner dimension)
// n_outputs: number of output positions (outer dimension)
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_from_ptr(ggml_sycl_sampler_t sampler,
                                                                   const float *       gpu_logits,
                                                                   int                 n_vocab,
                                                                   int                 n_outputs,
                                                                   const int32_t *     draft_tokens,
                                                                   int32_t *           sampled_tokens_out,
                                                                   int                 n_draft,
                                                                   int                 logits_offset);

// Version that takes HOST logits - copies to GPU internally (safest API)
// host_logits: HOST pointer to logits data [n_outputs, n_vocab] in row-major order
// draft_tokens: HOST pointer to draft token IDs [n_draft]
// sampled_tokens_out: HOST pointer to receive sampled tokens [n_draft]
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_from_host(ggml_sycl_sampler_t sampler,
                                                                    const float *       host_logits,
                                                                    int                 n_vocab,
                                                                    int                 n_outputs,
                                                                    const int32_t *     draft_tokens,
                                                                    int32_t *           sampled_tokens_out,
                                                                    int                 n_draft,
                                                                    int                 logits_offset);

// ===========================================================================
// Device Memory Utilities
// ===========================================================================

// Check if a buffer is a SYCL buffer
GGML_BACKEND_API bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer);

// Copy data from device memory to a tensor's device memory
// Used for multi-step decode where token IDs are kept on GPU
// src_device_ptr: SYCL device pointer (source)
// tensor: destination tensor (must be on SYCL backend)
// size: number of bytes to copy
GGML_BACKEND_API void ggml_backend_sycl_copy_device_to_tensor(void * src_device_ptr, ggml_tensor * tensor, size_t size);

// Copy data from tensor to GPU buffer with offset (device-to-device)
// Used for multi-ubatch logits accumulation on GPU
// backend: SYCL backend
// src_tensor: source tensor on SYCL backend
// dst_buffer: destination GPU buffer (allocated via ggml_backend_buft_alloc_buffer)
// dst_offset: byte offset into dst_buffer
// size: number of bytes to copy
GGML_BACKEND_API void ggml_backend_sycl_copy_tensor_to_buffer(ggml_backend_t        backend,
                                                              ggml_tensor *         src_tensor,
                                                              ggml_backend_buffer_t dst_buffer,
                                                              size_t                dst_offset,
                                                              size_t                size);

// Synced version: synchronizes compute before copying
// Use this when the tensor data may still be computing asynchronously
GGML_BACKEND_API void ggml_backend_sycl_copy_tensor_to_buffer_sync(ggml_backend_t        backend,
                                                                   ggml_tensor *         src_tensor,
                                                                   ggml_backend_buffer_t dst_buffer,
                                                                   size_t                dst_offset,
                                                                   size_t                size);

// Get device pointer from a SYCL buffer
// Returns the base pointer that can be used for GPU operations
GGML_BACKEND_API void * ggml_backend_sycl_buffer_get_ptr(ggml_backend_buffer_t buffer);

// Async copy from GPU buffer to host memory (device-to-host)
// Used for deferred logits copy to reduce sync overhead
// buffer: source SYCL GPU buffer
// src_ptr: pointer within the buffer (from ggml_backend_buffer_get_base or similar)
// dst: destination host memory
// offset: byte offset in src_ptr
// size: number of bytes to copy
// The copy is async and will complete when ggml_backend_synchronize is called
GGML_BACKEND_API void ggml_backend_sycl_buffer_get_async(ggml_backend_buffer_t buffer,
                                                         const void *          src_ptr,
                                                         void *                dst,
                                                         size_t                offset,
                                                         size_t                size);

// ===========================================================================
// Pending Device Token API (for multi-step GPU decode)
// ===========================================================================

// Set a pending device token to be used by the next decode operation
// The token_ptr must point to device memory containing n_tokens int32_t tokens
// Call this before llama_decode() when using multi-step GPU decode
GGML_BACKEND_API void ggml_backend_sycl_set_pending_device_token(void * token_ptr, size_t n_tokens);

// Clear the pending device token after decode completes
GGML_BACKEND_API void ggml_backend_sycl_clear_pending_device_token(void);

// ===========================================================================
// Continuous Batching API (Multi-sequence GPU Sampling)
// For processing multiple sequences in parallel with all operations on GPU
// ===========================================================================

// Multi-sequence sampler handle (opaque pointer)
typedef struct ggml_sycl_multi_seq_sampler * ggml_sycl_multi_seq_sampler_t;

// Create a multi-sequence GPU sampler
// backend: SYCL backend to use
// max_seqs: maximum number of concurrent sequences (up to 64)
// n_vocab: vocabulary size
// seed: base RNG seed (each sequence gets seed + seq_id)
GGML_BACKEND_API ggml_sycl_multi_seq_sampler_t ggml_backend_sycl_multi_seq_sampler_create(ggml_backend_t backend,
                                                                                          int            max_seqs,
                                                                                          int            n_vocab,
                                                                                          uint32_t       seed);

// Free multi-sequence sampler resources
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_sampler_free(ggml_sycl_multi_seq_sampler_t sampler);

// Add a sequence to the sampler
// seq_id: unique sequence identifier (0 to max_seqs-1)
// temp: temperature for this sequence
// Returns: true if added successfully, false if seq_id out of range or already active
GGML_BACKEND_API bool ggml_backend_sycl_multi_seq_add(ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp);

// Remove a sequence from the sampler
// seq_id: sequence to remove
// Returns: true if removed, false if not active
GGML_BACKEND_API bool ggml_backend_sycl_multi_seq_remove(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Update temperature for a sequence
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_set_temp(ggml_sycl_multi_seq_sampler_t sampler,
                                                           int                           seq_id,
                                                           float                         temp);

// Update all sampling parameters for a sequence
// seq_id: sequence identifier
// temp: temperature (0 = greedy)
// top_k: top-k filtering (0 = disabled)
// top_p: nucleus sampling threshold (1.0 = disabled)
// min_p: minimum probability threshold (0.0 = disabled)
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_set_params(ggml_sycl_multi_seq_sampler_t sampler,
                                                             int                           seq_id,
                                                             float                         temp,
                                                             int                           top_k,
                                                             float                         top_p,
                                                             float                         min_p);

// Get number of active sequences
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_active_count(ggml_sycl_multi_seq_sampler_t sampler);

// Sample tokens for all active sequences from batched logits
// batched_logits: device pointer to [n_active, n_vocab] float array
//                 rows must be in same order as seq_ids from get_active_seq_ids()
// greedy: if true, use argmax; if false, use probabilistic sampling
// Returns: number of tokens sampled (equals n_active)
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_sample(ggml_sycl_multi_seq_sampler_t sampler,
                                                        float *                       batched_logits,
                                                        bool                          greedy);

// Sample tokens for specific sequences from a logits tensor with batch indices
// logits_base: device pointer to logits tensor base [n_batch, n_vocab]
// seq_ids: array of sequence IDs to sample for
// batch_indices: array of batch indices (logits row for each seq_id)
// n_seqs: number of sequences to sample
// Uses per-sequence parameters (temp, top_k, top_p, min_p) set via set_params()
// Returns: number of tokens sampled
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_sample_indexed(ggml_sycl_multi_seq_sampler_t sampler,
                                                                float *                       logits_base,
                                                                const int *                   seq_ids,
                                                                const int *                   batch_indices,
                                                                int                           n_seqs);

// Get sampled tokens to host
// tokens_out: host buffer to receive tokens [n_active]
// seq_ids_out: optional host buffer to receive corresponding seq_ids [n_active]
// max_tokens: size of output buffers
// Returns: number of tokens copied
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                                            int32_t *                     tokens_out,
                                                            int *                         seq_ids_out,
                                                            int                           max_tokens);

// Get device pointer to sampled token for a sequence (for token ring buffer)
// seq_id: sequence ID
// Returns: device pointer to int32_t, or NULL if seq_id not active
GGML_BACKEND_API int32_t * ggml_backend_sycl_multi_seq_get_token_ptr(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Get list of active sequence IDs (for ordering batched logits)
// seq_ids_out: host buffer to receive sequence IDs
// max_seqs: size of output buffer
// Returns: number of active sequences
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_active_seq_ids(ggml_sycl_multi_seq_sampler_t sampler,
                                                                    int *                         seq_ids_out,
                                                                    int                           max_seqs);

// Reset token ring buffer for a sequence (call before starting generation)
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_reset_buffer(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Get tokens from ring buffer for a sequence
// seq_id: sequence ID
// tokens_out: host buffer to receive tokens
// max_tokens: size of output buffer
// Returns: number of tokens in buffer
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_ring_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                                                 int                           seq_id,
                                                                 int32_t *                     tokens_out,
                                                                 int                           max_tokens);

// ===========================================================================
// Batched Logits Management
// For extracting logits from batched decode without per-sequence D2H copies
// ===========================================================================

// Extract logits pointer for a specific sequence from batched output
// This returns a device pointer - NO copy is performed
// ctx: llama context (needs access to logits tensor)
// batch_idx: index of sequence in batch (0 to n_tokens-1 for sequences needing logits)
// Returns: device pointer to float[n_vocab], or NULL on error
GGML_BACKEND_API float * ggml_backend_sycl_get_batch_logits_ptr(void * ctx, int batch_idx);

// Get number of sequences with logits in current batch
GGML_BACKEND_API int ggml_backend_sycl_get_batch_logits_count(void * ctx);

// ===========================================================================
// KV Cache Synchronization API
// For multi-ubatch processing where KV cache writes must complete before reads
// ===========================================================================

// Submit a barrier after graph execution and return immediately.
// The barrier ensures all prior commands complete before subsequent ones.
// This is lighter-weight than full queue sync - it just ensures ordering.
// Call this AFTER each ubatch's graph_compute.
GGML_BACKEND_API void ggml_backend_sycl_submit_barrier(ggml_backend_t backend);

// Wait for the previously submitted barrier to complete.
// Call this BEFORE the next ubatch's graph_compute if needed.
// Returns immediately if no barrier was submitted.
GGML_BACKEND_API void ggml_backend_sycl_wait_barrier(ggml_backend_t backend);

// ===========================================================================
// Weight Streaming Control API
// ===========================================================================

// Stable model/load identity. A numeric slot is valid only together with its
// generation; zero IDs and GGML_SYCL_MODEL_SLOT_NONE fail closed.
struct ggml_sycl_model_token {
    uint64_t model_id;
    uint64_t load_txn_id;
    uint32_t slot;
    uint64_t slot_generation;
};

struct ggml_sycl_load_txn { uint64_t id; };

struct ggml_sycl_exec_context_id { uint64_t value; };
struct ggml_sycl_exec_session_id { uint64_t value; };
struct ggml_sycl_exec_session_reset_epoch { uint64_t value; };
struct ggml_sycl_exec_graph_epoch { uint64_t value; };
struct ggml_sycl_exec_invocation_id { uint64_t value; };
struct ggml_sycl_exec_control_host_alloc_batch {
    void *   opaque;
    uint32_t count;
};
struct ggml_sycl_exec_drain_ticket {
    struct ggml_sycl_exec_context_id context_id;
    struct ggml_sycl_exec_session_id session_id;
    struct ggml_sycl_exec_session_reset_epoch reset_epoch;
    uint64_t serial;
    uint32_t extracted_control_host_allocs;
};
struct ggml_sycl_exec_reset_ticket {
    struct ggml_sycl_exec_context_id context_id;
    struct ggml_sycl_exec_session_id session_id;
    struct ggml_sycl_exec_session_reset_epoch expected_reset_epoch;
    uint64_t serial;
};

enum ggml_sycl_execution_result {
    GGML_SYCL_EXECUTION_OK = 0,
    GGML_SYCL_EXECUTION_STALE,
    GGML_SYCL_EXECUTION_MISMATCH,
    GGML_SYCL_EXECUTION_OVERFLOW,
    GGML_SYCL_EXECUTION_DEVICE_BUSY,
    GGML_SYCL_EXECUTION_BUSY,
    GGML_SYCL_EXECUTION_NULL_OUTPUT,
    GGML_SYCL_EXECUTION_FOREIGN_BACKEND,
    GGML_SYCL_EXECUTION_NOT_FOUND,
    GGML_SYCL_EXECUTION_ALLOCATION_FAILURE,
    GGML_SYCL_EXECUTION_INTERNAL_ERROR,
};

enum ggml_sycl_execution_context_state {
    GGML_SYCL_EXECUTION_CONTEXT_OPEN = 0,
    GGML_SYCL_EXECUTION_CONTEXT_DRAINING,
    GGML_SYCL_EXECUTION_CONTEXT_RESETTING,
    GGML_SYCL_EXECUTION_CONTEXT_CLOSED,
};

enum ggml_sycl_execution_session_state {
    GGML_SYCL_EXECUTION_SESSION_IDLE = 0,
    GGML_SYCL_EXECUTION_SESSION_OPEN,
    GGML_SYCL_EXECUTION_SESSION_RESETTING,
    GGML_SYCL_EXECUTION_SESSION_DRAINING,
    GGML_SYCL_EXECUTION_SESSION_CLOSED,
};

enum ggml_sycl_execution_graph_state {
    GGML_SYCL_EXECUTION_GRAPH_IDLE = 0,
    GGML_SYCL_EXECUTION_GRAPH_OPEN,
    GGML_SYCL_EXECUTION_GRAPH_SEALED,
    GGML_SYCL_EXECUTION_GRAPH_COMPLETE,
    GGML_SYCL_EXECUTION_GRAPH_QUARANTINED,
    GGML_SYCL_EXECUTION_GRAPH_RETIRED,
};

enum ggml_sycl_execution_token_root_state {
    GGML_SYCL_EXECUTION_TOKEN_ROOT_OPEN = 0,
    GGML_SYCL_EXECUTION_TOKEN_ROOT_SEALED,
    GGML_SYCL_EXECUTION_TOKEN_ROOT_COMPLETE,
    GGML_SYCL_EXECUTION_TOKEN_ROOT_QUARANTINED,
};

struct ggml_sycl_execution_snapshot {
    struct ggml_sycl_exec_context_id         context_id;
    struct ggml_sycl_exec_session_id         session_id;
    struct ggml_sycl_exec_session_reset_epoch reset_epoch;
    struct ggml_sycl_exec_graph_epoch        graph_epoch;
    struct ggml_sycl_exec_invocation_id      invocation_id;
    struct ggml_sycl_model_token             token_root;
    enum ggml_sycl_execution_context_state   context_state;
    enum ggml_sycl_execution_session_state   session_state;
    enum ggml_sycl_execution_graph_state     graph_state;
    enum ggml_sycl_execution_token_root_state token_root_state;
    uint32_t                                 bound_device_count;
    uint32_t                                 busy_device_count;
};

enum ggml_sycl_lifecycle_result {
    GGML_SYCL_LIFECYCLE_OK = 0,
    GGML_SYCL_LIFECYCLE_NESTED,
    GGML_SYCL_LIFECYCLE_ABORTED,
    GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD,
    GGML_SYCL_LIFECYCLE_SLOT_EXHAUSTED,
    GGML_SYCL_LIFECYCLE_ID_EXHAUSTED,
    GGML_SYCL_LIFECYCLE_LOAD_BUSY,
    GGML_SYCL_LIFECYCLE_WRONG_TRANSACTION,
    GGML_SYCL_LIFECYCLE_DEPTH_UNDERFLOW,
    GGML_SYCL_LIFECYCLE_DEPTH_OVERFLOW,
    GGML_SYCL_LIFECYCLE_MISSING_SUCCESS,
    GGML_SYCL_LIFECYCLE_POISONED,
    GGML_SYCL_LIFECYCLE_NOT_FOUND,
    GGML_SYCL_LIFECYCLE_STALE_IDENTITY,
    GGML_SYCL_LIFECYCLE_NULL_OUTPUT,
    GGML_SYCL_LIFECYCLE_ALLOCATION_FAILED,
    GGML_SYCL_LIFECYCLE_EFFECT_FAILED,
    GGML_SYCL_LIFECYCLE_BUSY,
    GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND,
    // A deterministic refusal of a runtime plan update (budget or accounting),
    // as opposed to the transient BUSY above. Callers must NOT retry it: the
    // inputs cannot change between attempts, so a retry loop only multiplies
    // the error output. Appended last on purpose -- these values reach users as
    // `result=%d` and are quoted in tickets and logs (BUSY is 17), so inserting
    // mid-enum would silently renumber every recorded value.
    GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
};

// Registry-resolved static/DL parity entry point. Applies the exact inventory
// and placement envelope to every current SYCL device and stages the same
// immutable load candidate used by direct static calls.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_stage_inventory_plan(
    const struct ggml_sycl_tensor_inventory *   inventory,
    const struct ggml_sycl_placement_envelope * envelope,
    bool                                        early);

// Explicit transaction API. begin reserves slot+generation and both IDs as one
// side-effect-free operation. Nested calls must carry the outer transaction.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_begin(
    struct ggml_sycl_load_txn * txn);
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_enter_nested(
    struct ggml_sycl_load_txn txn);
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_end(
    struct ggml_sycl_load_txn txn, bool explicit_success, struct ggml_sycl_model_token * model);
// Select exact immutable LIVE model authority for explicit A/B/A routing.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_activate_model_plan(
    struct ggml_sycl_model_token model);
// Foundation model-bound runtime update. Context/graph code will call this
// automatically in 1q72; callers currently bind explicitly.
//
// llama.cpp-oyfl: flash_attn_enabled forwards to
// ggml_backend_sycl_set_runtime_context() -- see that declaration's comment.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_set_runtime_context_for_model(
    ggml_backend_t               backend,
    struct ggml_sycl_model_token model,
    uint32_t                     n_ctx,
    uint32_t                     n_ubatch,
    uint32_t                     n_seq_max,
    bool                         flash_attn_enabled);

// llama.cpp-tsfl (nphx comment c-wgxn): result of
// ggml_backend_sycl_probe_runtime_context_for_model() below -- a
// NON-PUBLISHING dry run of the same admission logic
// ggml_backend_sycl_set_runtime_context_for_model() uses to decide whether a
// candidate (n_ctx, n_ubatch, n_seq_max, flash_attn_enabled) fits, without
// publishing a new plan or materializing MoE MMID workspaces. `reason` is
// always a static string literal (a moe_mmid_runtime_reason name, or one of
// the probe's own refusal tags) -- never owned by the caller, and never
// NULL.
//
// llama.cpp-tsfl (round 1 F1): the "leaves no planned zone/ring changed"
// guarantee holds whenever `accepted` is true, OR whenever `reason` names a
// CANDIDATE refusal (the fit-or-not decision itself, unmet by this n_ctx/
// n_ubatch) -- both cases roll their own transient PP MoE oneDNN scratch
// ring re-plan back before returning. There are two exceptions; in each
// the ANOMALY line is logged at GGML_LOG_ERROR unconditionally (probe or
// not), even where the refusal the probe then returns still follows the
// probe's INFO policy:
//   1. reason=="probe rollback failed: ring left at candidate size": the
//      probe's OWN rollback attempt (rolling the ring back to the
//      pre-transaction n_ubatch) itself failed, and the ring is left at the
//      candidate size. Names both the candidate and the pre-transaction
//      n_ubatch it could not be restored to (llama.cpp-jumy: raised from
//      GGML_LOG_WARN to GGML_LOG_ERROR to match this policy).
//   2. A candidate refusal whose own internal restore also fails:
//      ggml_sycl_replan_pp_moe_onednn_ring()'s refuse_and_restore() closure
//      re-reserves the OLD ring on any refusal, and that re-reserve can
//      itself fail ("... restore FAILED ...", ggml-sycl.cpp) -- handled
//      (that line logged at GGML_LOG_ERROR), not asserted, but the probe
//      then returns the plain candidate refusal (logged at INFO in probe
//      mode like any candidate refusal) with the ring left unbacked by
//      any physical allocation rather than restored to its pre-candidate
//      state.
struct ggml_sycl_runtime_context_probe {
    bool         accepted;
    bool         would_demote_kv;
    size_t       host_kv_bytes;
    const char * reason;
};

// See ggml_sycl_runtime_context_probe above. `out` must not be NULL; it is
// zero-initialized on entry. Candidate refusals log at GGML_LOG_INFO, not
// the publishing path's GGML_LOG_ERROR, because a probe exists to be tried
// repeatedly and rejected quietly (Task 4b's ascending micro-batch trial).
// Two return values are argument-validation failures rather than a decision
// about the candidate itself: GGML_SYCL_LIFECYCLE_NULL_OUTPUT when `out` is
// NULL, or when `backend`/`backend->context` is NULL or n_ctx==0; and
// GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND when `backend` is not a SYCL backend
// (ggml_backend_is_sycl() false, no device, or registered against a
// different backend registry).
// Refuses with GGML_SYCL_LIFECYCLE_STALE_IDENTITY when `model` does not
// identify the CURRENTLY PUBLISHED plan AT THIS FUNCTION'S OWN ENTRY CHECK
// -- unlike ggml_backend_sycl_set_runtime_context_for_model(), this probe
// does not itself select or publish a different model's plan; it only
// evaluates candidates against whichever plan is already current. The SAME
// condition, detected instead under the transaction's own lock (a race
// between this entry check and that in-lock re-check), returns
// GGML_SYCL_LIFECYCLE_BUSY, not STALE_IDENTITY -- it is a race a caller's
// retry can resolve, unlike the entry check's own refusal.
// GGML_SYCL_LIFECYCLE_BUSY (round 1 F6; round 4 Q3) means the caller MAY
// retry: a live-update lease could not be acquired, the plan changed while
// acquiring the transaction lock, the module mutation guard refused, the
// published plan's identity changed between this probe's entry check and
// the transaction's in-lock re-check (the STALE_IDENTITY-shaped race just
// above), or the PP MoE oneDNN scratch ring's release refused because it is
// still claimed by an in-flight dispatch (llama.cpp-jumy: transient, unlike
// a ring re-plan that refuses because the requested size does not fit,
// which stays PLAN_REJECTED below). It is distinct from
// GGML_SYCL_LIFECYCLE_PLAN_REJECTED, which callers must NOT retry (see that
// enum value's own comment).
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_probe_runtime_context_for_model(
    ggml_backend_t                           backend,
    struct ggml_sycl_model_token             model,
    uint32_t                                 n_ctx,
    uint32_t                                 n_ubatch,
    uint32_t                                 n_seq_max,
    bool                                     flash_attn_enabled,
    struct ggml_sycl_runtime_context_probe * out);

// llama.cpp-oyfl: re-evaluates ONLY the non-FA attention scratch guard,
// against the CURRENTLY PUBLISHED plan's shape --
// no KV replan, no MoE MMID reaccount/materialize, no plan republish, no
// BUSY retry. For a caller whose n_ctx/n_ubatch have not changed and only
// flash_attn_enabled has (an AUTO llama_flash_attn_type resolving after
// ggml_backend_sycl_set_runtime_context_for_model()'s own initial call
// above already ran with an unresolved, optimistic `true`): re-running the
// full transaction would touch KV/MMID state that has no reason to change
// and would retry the same deterministic decision under BUSY backoff for
// no benefit. GGML_SYCL_LIFECYCLE_STALE_IDENTITY if the model token does
// not match the currently published plan; GGML_SYCL_LIFECYCLE_PLAN_REJECTED
// if the guard refuses (same message and arithmetic as the full
// transaction's own check).
//
// llama.cpp-rqak: an asymmetry worth knowing before touching either path.
// An explicit -fa 0 context goes through the FULL transaction above, which
// records its real runtime shape via
// unified_cache_set_planned_nonfa_attn_scratch_shape() (and restores the
// previous shape if the guard refuses); an AUTO context that resolves OFF
// goes through THIS narrow re-check instead, which deliberately does not
// record anything (see ggml_sycl_check_nonfa_attn_scratch()'s allow_replan
// parameter in ggml-sycl.cpp). So an AUTO-resolved-OFF context is checked
// against the shape recorded at load time (or by an earlier explicit -fa 0
// context), never its own. This has no practical effect today: the
// plan-time raise this shape feeds is a no-op once weights hold live
// leases (see the "Where this can and cannot help" discussion in
// docs/backend/sycl-memory-design.md).
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn(
    ggml_backend_t               backend,
    struct ggml_sycl_model_token model,
    bool                         flash_attn_enabled);

// Execution-lifecycle context identity is separate from the model lifecycle.
// One ContextId is allocated per llama_context and then bound to each SYCL
// backend created for it.
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_create(
    struct ggml_sycl_exec_context_id * context);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_bind_backend(
    ggml_backend_t backend,
    struct ggml_sycl_exec_context_id context);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_extract(
    struct ggml_sycl_exec_context_id      context,
    struct ggml_sycl_execution_snapshot * out);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_close_if_idle(
    struct ggml_sycl_exec_context_id context);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_begin_drain(
    struct ggml_sycl_exec_context_id context,
    struct ggml_sycl_exec_drain_ticket * ticket);
// Owner-targeted terminal wait for exactly one context: drains the queues of
// the backends bound to it, releases that context's invocation and retires its
// terminal graph epoch. Every wait runs with no binding, execution-state or
// registry lock held. An unknown or already-closed context returns STALE and
// touches nothing; teardown never sweeps other contexts or devices.
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_drain_terminal_events(
    struct ggml_sycl_exec_context_id context);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_extract_control_host_allocs(
    struct ggml_sycl_exec_drain_ticket * ticket,
    struct ggml_sycl_exec_control_host_alloc_batch * batch);
// Destroys the extracted batch's mem_handles outside every lock. It must run
// between extract_control_host_allocs and finish_drain; the emptied batch shell
// stays valid so finish_drain still checks the ticket identity that produced it.
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_release_control_host_allocs(
    struct ggml_sycl_exec_drain_ticket ticket,
    struct ggml_sycl_exec_control_host_alloc_batch * batch);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_context_finish_drain(
    struct ggml_sycl_exec_drain_ticket ticket,
    struct ggml_sycl_exec_control_host_alloc_batch * batch);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_session_begin_reset(
    struct ggml_sycl_exec_context_id context,
    struct ggml_sycl_exec_session_id session,
    struct ggml_sycl_exec_session_reset_epoch expected_epoch,
    struct ggml_sycl_exec_reset_ticket * ticket);
GGML_BACKEND_API enum ggml_sycl_execution_result ggml_backend_sycl_execution_session_finish_reset(
    struct ggml_sycl_exec_reset_ticket ticket,
    struct ggml_sycl_exec_session_reset_epoch * next_reset_epoch);

// Deprecated bool compatibility boundary. It is abort-default and cannot be
// used to obtain ownership identity; migrated callers use the APIs above.
GGML_BACKEND_API void ggml_backend_sycl_set_model_loading(bool loading);

// === Model ownership (llama.cpp-0qlw) ===
// The backend caches, pins and evicts weights on behalf of a llama_model, but
// had no way to learn that one had gone away: set_model_loading() is load-only.
// Without that, "is this weight still owned" could only be approximated by
// in_use_count, which is zero for a LIVE model's idle weights between graphs --
// so the model-load boundary freed them and inference faulted on the stale
// handle.  A slot is the model-lifetime token that answers it directly.
#define GGML_SYCL_MODEL_SLOT_NONE 0xFFFFFFFFu

// Deprecated reporting snapshot only. It is not ownership authority and may
// return GGML_SYCL_MODEL_SLOT_NONE; use the token returned by load_end.
GGML_BACKEND_API uint32_t ggml_backend_sycl_model_slot_current(void);

// Deprecated bare-slot teardown fails closed because it cannot prove the slot
// generation. Migrated owners must use model_unloaded_token after dropping
// their tensors/buffers.
GGML_BACKEND_API void ggml_backend_sycl_model_unloaded(uint32_t slot);
// Generation-safe teardown used by migrated model owners.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_model_unloaded_token(
    struct ggml_sycl_model_token model);
// Transfer a quarantined full owner token to the backend reaper when RAII can
// no longer retain it. Retries occur at safe lifecycle entry and shutdown.
GGML_BACKEND_API enum ggml_sycl_lifecycle_result ggml_backend_sycl_model_quarantine_token(
    struct ggml_sycl_model_token model);

// Release all host-backed weight extras (layout metadata, accessors, etc.)
// Call this when unloading a model to free SYCL resources associated with tensors.
// This is safe to call multiple times. Tensor->extra pointers are cleared before
// freeing to prevent use-after-free.
// Note: Extras are NOT automatically released when backends are freed, because
// tensors may outlive backends (e.g., temporary backends during model loading).
GGML_BACKEND_API void ggml_backend_sycl_release_host_weight_extras(void);

// Wire the SYCL placement plan into the ggml scheduler.  When multi-GPU
// SPLIT_RATIO is active, this tells the scheduler which backend (device)
// should process each tensor, enabling proper graph splitting during warmup
// and inference.  Must be called after compute_multi_device_plan and after
// the scheduler is created with all GPU backends.
GGML_BACKEND_API void ggml_backend_sycl_set_sched_placement_plan(ggml_backend_sched_t sched);
GGML_BACKEND_API bool ggml_backend_sycl_has_active_placement_plan(void);

// === Test-only debug accessors (llama.cpp-dfo0, plan task L2) ===
#if defined(GGML_SYCL_PRIVATE_TESTING)
// Process-global by design: ggml_backend_sched does not hand test code the
// compute buffer it allocates internally, so this reads the last COMPUTE-usage
// ggml_backend_sycl_buffer_reset's post-release tensor_extras vector size,
// across whichever compute buffer (or, for a multi-chunk buffer, whichever
// single chunk) it last reset. Not meaningful with more than one compute
// buffer/chunk resetting concurrently -- single-threaded test use only.
GGML_BACKEND_API size_t ggml_backend_sycl_debug_last_compute_buffer_extra_count(void);

// llama.cpp-asdt, plan task L2b: analogous accessor for the tiered KV
// buffer's per-view-tensor extras. Process-global by design (same reason as
// above); reads the current view_extras vector size of whichever tiered KV
// buffer last processed a view tensor's init_tensor call. Single-threaded
// test use only.
GGML_BACKEND_API size_t ggml_backend_sycl_debug_last_kv_view_extra_count(void);

// llama.cpp-asdt, plan task L2b (jemalloc-profile bug fix): counts extras
// actually still allocated (marked debug_is_kv_view_extra, not yet deleted),
// unlike the container-membership accessor above -- see its own comment in
// ggml-sycl.cpp for why that distinction matters.
GGML_BACKEND_API size_t ggml_backend_sycl_debug_live_kv_view_extra_count(void);
#endif

#ifdef __cplusplus
}
#endif
