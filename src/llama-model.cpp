#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-lazy-mode.h"
#include "llama-lazy-storage.h"
#include "llama-mmap.h"
#include "llama-n-ctx-hint.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"
#include "llama-load-measure.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include "llama.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
#    include "ggml-sycl.h"
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
static bool llama_model_buffer_is_sycl(ggml_backend_buffer_t buf) {
    if (buf == nullptr) {
        return false;
    }
    const char * name = ggml_backend_buffer_name(buf);
    return name != nullptr && std::strncmp(name, GGML_SYCL_NAME, std::strlen(GGML_SYCL_NAME)) == 0;
}

static bool llama_model_buft_is_sycl(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }
    const char * name = ggml_backend_buft_name(buft);
    return name != nullptr && std::strncmp(name, GGML_SYCL_NAME, std::strlen(GGML_SYCL_NAME)) == 0;
}

struct llama_model_sycl_lifecycle_hooks {
    ggml_backend_reg_t                                   reg        = nullptr;
    decltype(&ggml_backend_sycl_model_load_begin)        begin      = nullptr;
    decltype(&ggml_backend_sycl_model_load_enter_nested) nested     = nullptr;
    decltype(&ggml_backend_sycl_model_load_end)          end        = nullptr;
    decltype(&ggml_backend_sycl_model_unloaded_token)    unload     = nullptr;
    decltype(&ggml_backend_sycl_model_quarantine_token)  quarantine = nullptr;
    decltype(&ggml_backend_sycl_activate_model_plan)           activate        = nullptr;
    decltype(&ggml_backend_sycl_set_runtime_context_for_model) runtime_context               = nullptr;
    decltype(&ggml_backend_sycl_weights_evictable)             weights_evictable           = nullptr;
    decltype(&ggml_backend_sycl_host_buffer_type_for_device)   host_buffer_type_for_device = nullptr;
    // The staging proc reads this libllama's inventory layout (its resolved storage inputs, design 5efe §4.3), and the
    // loader binds SYCL_Mapped buffers from the plan it stages. Not part of llama_model_sycl_hooks_enabled: a backend
    // that lacks them is refused by name in the loading guard rather than silently treated as a non-SYCL load.
    decltype(&ggml_backend_sycl_stage_load_inventory_plan)     stage_inventory               = nullptr;
    decltype(&ggml_backend_sycl_mapped_buffer_type_for_device) mapped_buffer_type_for_device = nullptr;
    decltype(&ggml_backend_sycl_mapped_buffer_from_file)       mapped_buffer_from_file       = nullptr;
};

static llama_model_sycl_lifecycle_hooks llama_model_sycl_hooks() {
    llama_model_sycl_lifecycle_hooks result;
    const size_t backend_dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < backend_dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg || std::strcmp(ggml_backend_reg_name(reg), "SYCL") != 0) {
            continue;
        }
        result.reg   = reg;
        result.begin = reinterpret_cast<decltype(result.begin)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_begin"));
        result.nested = reinterpret_cast<decltype(result.nested)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_enter_nested"));
        result.end = reinterpret_cast<decltype(result.end)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_end"));
        result.unload = reinterpret_cast<decltype(result.unload)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_unloaded_token"));
        result.quarantine = reinterpret_cast<decltype(result.quarantine)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_quarantine_token"));
        result.activate = reinterpret_cast<decltype(result.activate)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_activate_model_plan"));
        result.runtime_context = reinterpret_cast<decltype(result.runtime_context)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_set_runtime_context_for_model"));
        result.weights_evictable = reinterpret_cast<decltype(result.weights_evictable)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_weights_evictable"));
        result.host_buffer_type_for_device = reinterpret_cast<decltype(result.host_buffer_type_for_device)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_host_buffer_type_for_device"));
        result.stage_inventory = reinterpret_cast<decltype(result.stage_inventory)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_stage_load_inventory_plan"));
        result.mapped_buffer_type_for_device = reinterpret_cast<decltype(result.mapped_buffer_type_for_device)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_mapped_buffer_type_for_device"));
        result.mapped_buffer_from_file = reinterpret_cast<decltype(result.mapped_buffer_from_file)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_mapped_buffer_from_file"));
        break;
    }
    return result;
}

static bool llama_model_sycl_hooks_enabled(const llama_model_sycl_lifecycle_hooks & hooks) {
    return !ggml_backend_device_backends_disabled() && hooks.begin && hooks.nested && hooks.end && hooks.unload &&
           hooks.quarantine && hooks.activate && hooks.runtime_context && hooks.weights_evictable &&
           hooks.host_buffer_type_for_device;
}

static bool llama_model_sycl_hooks_enabled() {
    return llama_model_sycl_hooks_enabled(llama_model_sycl_hooks());
}

// llama.cpp-30h4: gate for the dev_layer()/dev_output() staleness fix below.
// Default ON (the fix is the shipping behaviour). Only the EXPLICIT literal
// "0" or "false" disables it and restores today's stale-dev_layer behaviour --
// mirrors the established default-ON/opt-out idiom this codebase already uses
// for the same shape of switch (e.g. ggml_sycl_fa_onednn_d512_enabled(),
// fattn.cpp: "return !(env && (strcmp(env, \"0\") == 0 || strcmp(env, \"false\") == 0));").
// Deliberately not atoi()-based: atoi() folds any unparseable or unexpected
// value ("xyz", a typo, empty string) to 0 as well, which would silently pick
// the disabled branch nobody asked for. That is the exact failure shape
// llama.cpp-80w9 recorded for GGML_SYCL_KV_HOST, so it is not repeated here.
static bool llama_model_sycl_dev_layer_sync_enabled() {
    const char * env = std::getenv("GGML_SYCL_DEV_LAYER_SYNC");
    return !(env != nullptr && (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0));
}

static bool llama_model_dev_is_sycl(ggml_backend_dev_t dev) {
    const auto hooks = llama_model_sycl_hooks();
    return llama_model_sycl_hooks_enabled(hooks) && dev != nullptr && ggml_backend_dev_backend_reg(dev) == hooks.reg;
}

struct llama_model_sycl_loading_guard {
    llama_model_sycl_lifecycle_hooks hooks{};
    bool                             active    = false;
    bool outer = false;
    ggml_sycl_load_txn txn = {};
    llama_sycl_model_token * out_model = nullptr;

    llama_model_sycl_loading_guard(const llama_model_sycl_loading_guard &) = delete;
    llama_model_sycl_loading_guard & operator=(const llama_model_sycl_loading_guard &) = delete;
    llama_model_sycl_loading_guard(llama_model_sycl_loading_guard &&) = delete;
    llama_model_sycl_loading_guard & operator=(llama_model_sycl_loading_guard &&) = delete;

    explicit llama_model_sycl_loading_guard(bool enabled, llama_sycl_model_token * out = nullptr) :
        hooks(llama_model_sycl_hooks()),
        active(enabled && llama_model_sycl_hooks_enabled(hooks)),
        outer(true),
        out_model(out) {
        if (active) {
            require_host_backing_hooks();
            const auto rc = hooks.begin(&txn);
            if (rc != GGML_SYCL_LIFECYCLE_OK) throw std::runtime_error(format("SYCL model lifecycle begin failed: result=%d", (int) rc));
        }
    }

    llama_model_sycl_loading_guard(bool enabled, ggml_sycl_load_txn parent) :
        hooks(llama_model_sycl_hooks()),
        active(enabled && llama_model_sycl_hooks_enabled(hooks)),
        txn(parent) {
        if (active) {
            require_host_backing_hooks();
            const auto rc = hooks.nested(txn);
            if (rc != GGML_SYCL_LIFECYCLE_OK) throw std::runtime_error(format("SYCL nested model lifecycle begin failed: txn=%llu result=%d", (unsigned long long) txn.id, (int) rc));
        }
    }

    // Runs before begin(), so a refusal leaves no transaction to roll back. The two libraries share the
    // placement inventory layout and the host-backing protocol, so a libggml-sycl without these procs
    // predates this libllama and cannot be driven by it.
    void require_host_backing_hooks() const {
        const char * missing = nullptr;
        if (!hooks.stage_inventory) {
            missing = "ggml_backend_sycl_stage_load_inventory_plan";
        } else if (!hooks.mapped_buffer_type_for_device) {
            missing = "ggml_backend_sycl_mapped_buffer_type_for_device";
        } else if (!hooks.mapped_buffer_from_file) {
            missing = "ggml_backend_sycl_mapped_buffer_from_file";
        }
        if (missing != nullptr) {
            throw std::runtime_error(
                format("the SYCL backend does not export %s, so it predates this libllama: "
                       "rebuild libllama and libggml-sycl together",
                       missing));
        }
    }

    ~llama_model_sycl_loading_guard() {
        if (!active) return;
        ggml_sycl_model_token token = {};
        const auto            rc    = hooks.end(txn, false, outer ? &token : nullptr);
        if (outer && out_model) {
            const bool retain_cleanup = rc == GGML_SYCL_LIFECYCLE_EFFECT_FAILED && token.model_id != 0;
            *out_model = retain_cleanup ? llama_sycl_model_token{ token.model_id, token.load_txn_id, token.slot,
                                                                  token.slot_generation } :
                                          llama_sycl_model_token{};
        }
        if (rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && rc != GGML_SYCL_LIFECYCLE_POISONED &&
            rc != GGML_SYCL_LIFECYCLE_NESTED) {
            LLAMA_LOG_ERROR("SYCL model lifecycle abort failed: txn=%llu result=%d\n", (unsigned long long) txn.id, (int) rc);
        }
    }

    void cancel() {
        if (!active) {
            return;
        }
        ggml_sycl_model_token rollback_token{};
        const auto            rc = hooks.end(txn, false, outer ? &rollback_token : nullptr);
        active                   = false;
        if (outer && out_model) {
            // Normal abort statuses are terminal and own nothing. Retain only
            // an exact owner returned for failed/quarantined cleanup.
            *out_model = rc == GGML_SYCL_LIFECYCLE_EFFECT_FAILED && rollback_token.model_id != 0 ?
                             llama_sycl_model_token{ rollback_token.model_id, rollback_token.load_txn_id,
                                                     rollback_token.slot, rollback_token.slot_generation } :
                             llama_sycl_model_token{};
        }
        if (rollback_token.model_id == 0 && rc != GGML_SYCL_LIFECYCLE_OK && rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS &&
            rc != GGML_SYCL_LIFECYCLE_POISONED && rc != GGML_SYCL_LIFECYCLE_NESTED) {
            throw std::runtime_error(format("SYCL model lifecycle cancel failed: txn=%llu result=%d",
                                            (unsigned long long) txn.id, (int) rc));
        }
    }

    void finish(bool explicit_success) {
        if (!active) return;
        ggml_sycl_model_token token = {};
        const auto            rc    = hooks.end(txn, explicit_success, outer ? &token : nullptr);
        active = false;
        if (outer && out_model && token.model_id != 0) {
            *out_model = { token.model_id, token.load_txn_id, token.slot, token.slot_generation };
        }
        const bool expected = rc == GGML_SYCL_LIFECYCLE_OK || (!outer && rc == GGML_SYCL_LIFECYCLE_NESTED);
        if (!expected) {
            throw std::runtime_error(format("SYCL model lifecycle end failed: txn=%llu outer=%d success=%d result=%d",
                                            (unsigned long long) txn.id, outer, explicit_success, (int) rc));
        }
    }
};

static bool llama_model_buft_backend_is_sycl(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    return llama_model_dev_is_sycl(dev);
}

// The shape every load-time measure of this model runs at: the context and micro-batch the caller told the load
// (llama_model_params::n_ctx_hint, n_ubatch_hint). The measures and the inventory's n_ubatch_measure read the same
// resolver over the same hints, so the backend credits the term at the micro-batch it was measured at.
static llama_load_measure_shape llama_model_load_measure_shape(const llama_model & model) {
    return llama_load_measure_shape_resolve(model.get_n_ctx_hint(), model.get_n_ubatch_hint(),
                                            model.hparams.n_ctx_train);
}

static ggml_sycl_placement_envelope llama_model_sycl_make_placement_envelope() {
    ggml_sycl_placement_envelope envelope = {};
    envelope.n_ctx                        = 0;
    envelope.n_ubatch                     = 0;
    envelope.n_seq_max                    = 1;
    envelope.flash_attn_type              = -1;
    return envelope;
}

static bool llama_model_sycl_is_moe_expert_weight(const ggml_sycl_tensor_info & tensor) {
    if (tensor.name == nullptr) {
        return false;
    }
    return std::strstr(tensor.name, "ffn_gate_exps") != nullptr || std::strstr(tensor.name, "ffn_up_exps") != nullptr ||
           std::strstr(tensor.name, "ffn_down_exps") != nullptr;
}

static size_t llama_model_sycl_align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

// llama.cpp-o3a0: replicates
// ggml_sycl_flash_attn_ext_onednn_plan()'s D-based eligibility gate
// (ggml/src/ggml-sycl/fattn-onednn.cpp: the `ne00 > 512` UNSUPPORTED_D
// reject, and the `ne00 > 256` scale-or-hatch gate) for the sole purpose of
// the oneDNN Graph-scratch zone floor -- this is a llama-layer file and
// cannot include the SYCL-only fattn-onednn.cpp/.hpp to share the real
// predicate directly.
//
// D <= 256 is unconditionally eligible: the GENERALIZED scale fix
// (llama.cpp-p0f5) writes the model's real runtime divisor into the
// compiled partition instead of assuming sqrt(D), so the real gate's only
// remaining requirement at D<=256 (a finite, nonzero scale) holds for
// every in-tree model.
//
// D in (256, 512] mirrors the real gate's FULL disjunction: eligible when
// EITHER GGML_SYCL_FA_ONEDNN_D512_SCALE is set (same env var, same live
// getenv parse -- deliberately NOT cached in a function-local static,
// matching ggml_sycl_fa_onednn_d512_scale_relaxed()'s own uncached form;
// this runs once per layer at load, not per dispatch, so there is no hot
// path to cache for) OR the layer's attention scale is already the
// canonical 1/sqrt(D) -- the real gate's own
// `fabs(1/scale - sqrt(D)) < 1e-3` tolerance check. An earlier version of
// this helper required the hatch unconditionally for D>256, which is
// STRICTER than the real gate and would silently under-count eligibility
// (and so under-provision the floor) for any D>256 layer whose scale
// already happens to be canonical.
//
// The layer's scale is derived the way MOST in-tree build_*.cpp derive
// kq_scale from hparams.f_attention_scale: 0.0f (the default, meaning "no
// override was set") means canonical 1/sqrt(D), which trivially satisfies
// the tolerance check; any other value is used verbatim, matching
// gemma4/gemma3n/gemma4-assistant's f_attention_scale=1.0f pre-scaled-Q
// layers, whose D=512 global layers correctly stay ineligible without the
// hatch. EXCEPTION: gemma2/gemma3/gemma-embedding also pre-scale Q, but
// pass a LITERAL 1.0f to build_attn (src/models/gemma3.cpp) instead of
// hparams.f_attention_scale itself, so this derivation OVER-COUNTS
// eligibility for them at D>256: their nonzero, already-canonical
// f_attention_scale trivially passes the tolerance check here, while
// their real runtime scale of 1.0f would not. Currently unreachable --
// every in-tree instance of this family has D<=256 and is already
// unconditionally eligible before this branch is ever reached -- but a
// future D>256 model in this family would be silently over-provisioned
// (the safe direction) rather than under-provisioned by this gap.
// Separately, if some future architecture computes kq_scale by a
// mechanism OTHER than f_attention_scale while leaving that field at its
// 0.0f default, this reads it as canonical and marks the layer eligible
// even if the real runtime scale would fail the strict check -- the SAFE
// direction: a layer this wrongly counts eligible only ever makes the
// floor bigger than the real oneDNN demand, never smaller, and the real
// dispatch-time gate in fattn-onednn.cpp is entirely unaffected by this
// file either way.
//
// Kept identical to the SYCL-side predicate by
// tests/test-sycl-onednn-graph-floor-eligibility-source.py.
static bool llama_model_sycl_onednn_head_dim_eligible(uint32_t head_dim, float f_attention_scale) {
    if (head_dim > 512) {
        return false;
    }
    if (head_dim <= 256) {
        return true;
    }
    const char * env = std::getenv("GGML_SYCL_FA_ONEDNN_D512_SCALE");
    if (env && std::atoi(env) != 0) {
        return true;
    }
    const float kq_scale = f_attention_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(head_dim)) : f_attention_scale;
    return std::fabs(1.0f / kq_scale - sqrtf(static_cast<float>(head_dim))) < 1e-3f;
}

// llama.cpp-3aos: owning storage for the three per-layer arrays below --
// grouped into one struct so the two llama_model_sycl_populate_inventory
// callers hold one object instead of three separate unique_ptrs.
struct llama_model_sycl_kv_layer_arrays {
    std::unique_ptr<uint32_t[]> k_width;
    std::unique_ptr<uint32_t[]> v_width;
    std::unique_ptr<uint8_t[]>  kind;
    std::unique_ptr<uint32_t[]> idx_k_width;  // indexer key width per layer (llama_kv_layer_owners::idx_k_width)
};

// llama.cpp-3aos: per-layer (attention kind, K width, V width) for the SYCL
// KV planner (ggml_sycl_tensor_inventory::kv_layer_kind/kv_k_width_per_layer/
// kv_v_width_per_layer). Owned by the caller of
// llama_model_sycl_populate_inventory (same lifetime requirement as
// swa_layer_mask just below -- the inventory only stores raw pointers, and
// must remain valid through the caller's llama_model_sycl_apply_inventory
// call). Kind and width must agree with what the tensor loader actually
// created: SHARED (every layer the KV cache does not own -- Gemma 4/3n's
// trailing reused-KV layers, a hybrid model's recurrent layers, see the
// paragraph below) gets width 0; SWA/FULL both use
// hparams.n_embd_{k,v}_gqa(il), which is already per-layer (switches on
// is_swa(il) internally -- see llama-hparams.cpp).
// llama.cpp-8ecj: whether a layer owns K/V comes from the memory the default
// context builds (llama_kv_layer_owners_default), not from has_kv() alone: a
// hybrid model's recurrent layers pass has_kv() but the KV cache holds none of
// them, and charging them KV made the planner reserve and demote 4x the KV
// Qwen3-Next allocates. A memory kind the shapes do not model keeps has_kv().
// The indexer key cache of a llama_memory_hybrid_idx (qwen4exp) is a second
// buffer over the same cells: its key width goes out per layer beside the K/V
// widths, so the planner budgets both buffers and the allocator sizes each one
// from its own cache. A layer whose only KV is indexer keys is FULL with zero
// K/V widths, so it is still charged.
static llama_model_sycl_kv_layer_arrays llama_model_sycl_build_kv_layer_arrays(const llama_model & model,
                                                                               uint32_t            n_layer) {
    const llama_hparams &            hparams = model.hparams;
    const llama_kv_layer_owners      owners  = llama_kv_layer_owners_default(model);
    llama_model_sycl_kv_layer_arrays out;
    out.k_width.reset(new uint32_t[n_layer]);
    out.v_width.reset(new uint32_t[n_layer]);
    out.kind.reset(new uint8_t[n_layer]);
    out.idx_k_width.reset(new uint32_t[n_layer]);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const bool owns_kv = owners.modelled ? il < owners.owns.size() && owners.owns[il] : hparams.has_kv(il);
        const uint32_t idx_k_width = il < owners.idx_k_width.size() ? owners.idx_k_width[il] : 0;
        out.idx_k_width[il]        = idx_k_width;
        if (!owns_kv) {
            out.kind[il]    = idx_k_width > 0 ? GGML_SYCL_KV_LAYER_FULL : GGML_SYCL_KV_LAYER_SHARED;
            out.k_width[il] = 0;
            out.v_width[il] = 0;
        } else {
            out.kind[il]    = hparams.is_swa(il) ? GGML_SYCL_KV_LAYER_SWA : GGML_SYCL_KV_LAYER_FULL;
            out.k_width[il] = hparams.n_embd_k_gqa(il);
            out.v_width[il] = hparams.n_embd_v_gqa(il);
        }
    }
    return out;
}

// Which layers carry attention sinks. The oneDNN SDPA path refuses an op with sinks, so those layers never draw
// oneDNN Graph scratch and the floor's head maxima leave them out. Read from the file's tensor list, the same source
// for the early and the late inventory.
static std::unique_ptr<bool[]> llama_model_sycl_attn_sinks_layers(const llama_model_loader & ml, uint32_t n_layer) {
    std::unique_ptr<bool[]> sinks(new bool[n_layer]());
    const LLM_TN            tn(ml.get_arch());
    for (uint32_t il = 0; il < n_layer; ++il) {
        sinks[il] = ml.weights_map.find(tn(LLM_TENSOR_ATTN_SINKS, "weight", il).str()) != ml.weights_map.end();
    }
    return sinks;
}

static void llama_model_sycl_populate_inventory(ggml_sycl_tensor_inventory &         inventory,
                                                std::vector<ggml_sycl_tensor_info> & tensors,
                                                const bool *                         swa_layer_mask,
                                                const bool *                         sinks_layer_mask,
                                                const uint32_t *                     kv_k_width_per_layer,
                                                const uint32_t *                     kv_v_width_per_layer,
                                                const uint8_t *                      kv_layer_kind,
                                                const uint32_t *                     kv_idx_k_width_per_layer,
                                                size_t                               total_size,
                                                size_t                               max_pp_pipeline_weight_bytes,
                                                const llama_hparams &                hparams,
                                                uint32_t                             n_ctx_hint,
                                                uint32_t                             n_ubatch_hint) {
    const uint32_t n_layer             = hparams.n_layer();
    inventory                         = {};
    inventory.tensors                 = tensors.data();
    inventory.count                   = tensors.size();
    inventory.total_size              = total_size;
    inventory.pp_pipeline_scratch_bytes = max_pp_pipeline_weight_bytes * 2;
    inventory.n_expert                = hparams.n_expert;
    inventory.n_expert_used           = hparams.n_expert_used_max();
    inventory.n_layer                 = n_layer;
    // llama.cpp-3aos: layer 0 may be a sliding layer whose width is
    // narrower than the full-attention layers', so the fallback width
    // here must be the per-layer MAXIMUM, not layer 0's own width. On
    // Gemma 4 E4B layer 0 is a SWA layer (width 512), so the il=0
    // accessor (no explicit layer argument) silently under-widens the
    // FULL-attention-width fields this struct's own field comment
    // (ggml-sycl.h) and unified-cache.hpp's kv_bytes_per_layer()/
    // kv_bytes_per_swa_layer() comments describe -- kv_bytes_per_layer()
    // would compute n_ctx * (512+512) * 2 = 8,388,608 for Gemma 4 E4B
    // instead of the real full-attention n_ctx * (1024+1024) * 2 =
    // 16,777,216, which is plan.kv_per_layer, consumed by
    // ggml_sycl_largest_fitting_n_ctx() and the KV-buffer-kind heuristic
    // in ggml-sycl.cpp. _max() is the correct accessor: the maximum
    // per-layer width, which for a model whose full-attention layers are
    // its widest (true of every SWA architecture in this codebase) is
    // genuinely the FULL-attention width.
    inventory.n_embd_k_gqa              = hparams.n_embd_k_gqa_max();
    inventory.n_embd_v_gqa              = hparams.n_embd_v_gqa_max();
    // Model loading no longer has runtime context params. Do not reserve
    // train-context KV here; actual KV allocations are placed later by the
    // unified cache once llama_context provides the real n_ctx.
    inventory.n_ubatch                = 512;
    inventory.n_ctx                   = inventory.n_ubatch;
    // llama.cpp-8ecj: ...except the room for it. The planner holds the KV room of the context the model opens with on
    // the device ahead of the routed experts, so the experts fill only what the dense weights and that KV leave.
    // llama.cpp-ak0p: that context is the one the caller is about to create (llama_model_params::n_ctx_hint), padded
    // as llama_context pads n_ctx for a single sequence (src/llama-context.cpp: cparams.n_ctx = GGML_PAD(cparams.n_ctx,
    // 256)). A non-unified multi-slot context can exceed it: llama_context pads each sequence's share, and SWA cells
    // grow with the streams, while the load plans for one sequence. The runtime transaction re-places that overflow.
    // llama.cpp-7fmt: a caller that names no n_ctx (common without -c) asks for the model's default context, which
    // llama_context creates as n_ctx_train, so the room is held for that context; the inventory says it was the
    // default, so the room's line can explain a room the size of the training context. llama_n_ctx_hint_resolve() is
    // the one place the hint is resolved. With no hint at all (0, a caller that never sets the field) no room is held,
    // and the KV beyond the planning shape is placed when the context is created. A hint above n_ctx_train (a
    // rope-extended -c) is honoured: the caller asked for that context, and the room is already capped at what the
    // dense weights and their KV leave. The room, the oneDNN Graph scratch floor (sized for the context the model opens
    // with) and the load measures read the hint; the planning shape above and the placement envelope's n_ctx (still 0)
    // keep their inputs.
    const llama_n_ctx_hint_resolution ctx_room = llama_n_ctx_hint_resolve(n_ctx_hint, hparams.n_ctx_train);
    inventory.n_ctx_context                    = ctx_room.n_ctx_context;
    inventory.n_ctx_context_is_default         = ctx_room.is_default;
    // The micro-batch the caller named (llama_model_params::n_ubatch_hint, 0 for none): the planner plans the PP MoE
    // scratch ring in the RUNTIME zone for it and moves routed experts to the host tier for the room. Like the context
    // above, that decision and the oneDNN Graph scratch floor read it; n_ubatch stays the load's planning shape.
    inventory.n_ubatch_context                 = n_ubatch_hint;
    // The micro-batch the load measures its compute term at, from the resolver the measures use
    // (llama_model_load_measure_shape), so the backend credits the term at the rung it was measured at. The early and
    // the late inventory are both built here from the same hints, so both carry the same value.
    inventory.n_ubatch_measure =
        llama_load_measure_shape_resolve(n_ctx_hint, n_ubatch_hint, hparams.n_ctx_train).n_ubatch;
    if (hparams.n_expert > 0 && hparams.n_expert_used_max() > 0) {
        // llama.cpp-sr83 (C3): the batched executor repacks straight to a
        // {nibbles,e8m0-scales} WOQ shape by default and deletes the f16
        // dequant (GGML_SYCL_MOE_PP_WOQ default ON, mirrors the same env
        // read in ggml-sycl.cpp's try_pp_onednn_f16_batched) --
        // size the plan for whichever arm will actually run, or admission
        // refuses every real dispatch (f2bdfbffe's fail-closed contract,
        // cited below). The two arms are NOT summed: only one of them
        // allocates at a time, and sizing for both would erase the ~4x VRAM
        // win that is the entire point of C3.
        static const bool pp_woq_enabled_at_plan = []() {
            const char * env = std::getenv("GGML_SYCL_MOE_PP_WOQ");
            return env == nullptr || std::atoi(env) != 0;
        }();
        // MXFP4's quantization block size (QK_MXFP4 in the SYCL backend's
        // headers) -- not re-included here to avoid pulling ggml-common.h
        // into this TU; grouped scales are one e8m0 byte per 32-element block.
        constexpr size_t kMxfp4BlockSize = 32;
        size_t           max_weight_slot = 0;
        size_t           max_k           = 0;
        size_t           max_n           = 0;
        for (const ggml_sycl_tensor_info & tensor : tensors) {
            if (!llama_model_sycl_is_moe_expert_weight(tensor)) {
                continue;
            }
            if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0) {
                continue;
            }
            const size_t k = static_cast<size_t>(tensor.ne[0]);
            const size_t n = static_cast<size_t>(tensor.ne[1]);
            max_k          = std::max(max_k, k);
            max_n          = std::max(max_n, n);
            if (pp_woq_enabled_at_plan) {
                // Must match try_pp_onednn_f16_batched's WOQ-arm
                // slot layout byte-for-byte: nibbles (k*n/2 bytes) then e8m0
                // scales ((k/kMxfp4BlockSize)*n bytes), each 256-aligned.
                const size_t nibble_bytes = llama_model_sycl_align_up(n * (k / 2), 256);
                const size_t scale_bytes  = llama_model_sycl_align_up((k / kMxfp4BlockSize) * n, 256);
                max_weight_slot           = std::max(max_weight_slot, nibble_bytes + scale_bytes);
            } else {
                max_weight_slot =
                    std::max(max_weight_slot, llama_model_sycl_align_up(k * n * sizeof(ggml_fp16_t), 256));
            }
        }
        if (max_weight_slot > 0 && max_k > 0 && max_n > 0) {
            constexpr uint32_t pp_moe_onednn_ring_depth     = 1;
            // The batched PP MoE oneDNN executor (try_pp_onednn_f16_batched,
            // ggml-sycl.cpp) requests scratch sized for every expert active in a
            // single dispatch, not one -- active_count is bounded by n_expert, not
            // n_expert_used (llama.cpp-dboi; f2bdfbffe made admission fail-closed,
            // so an under-sized plan now refuses every real dispatch instead of
            // silently growing the zone; moe-scratch-admission.hpp's own docstring
            // cites the exact GPT-OSS case this fixes). Size the weight slot for
            // the worst case: every expert resident at once -- ~n_expert x
            // per-expert WOQ bytes by default (~130 MB for GPT-OSS 20B, vs the
            // f16 arm's ~506 MB when GGML_SYCL_MOE_PP_WOQ=0 opts out).
            const size_t       per_expert_weight_slot_bytes = max_weight_slot;
            const size_t       worst_case_active_experts    = static_cast<size_t>(hparams.n_expert);
            inventory.pp_moe_onednn_weight_slot_bytes       = per_expert_weight_slot_bytes * worst_case_active_experts;

            // One scratch row of the widest expert: its input width in half floats, its output width in floats. The
            // backend sizes the ring's activation and output slots from these and the expert counts, for the
            // micro-batch a context actually runs, so this loader sizes neither slot.
            inventory.pp_moe_onednn_activation_bytes_per_row = max_k * sizeof(ggml_fp16_t);
            inventory.pp_moe_onednn_output_bytes_per_row     = max_n * sizeof(float);
            inventory.pp_moe_onednn_activation_slot_bytes    = 0;
            inventory.pp_moe_onednn_output_slot_bytes        = 0;
            inventory.pp_moe_onednn_ring_depth               = pp_moe_onednn_ring_depth;
            // The weight slots only. The backend adds the two row-scaled slots when it plans the ring.
            inventory.pp_moe_onednn_scratch_bytes =
                static_cast<size_t>(pp_moe_onednn_ring_depth) * inventory.pp_moe_onednn_weight_slot_bytes;
        }
    }
    // llama.cpp-o3a0: window-aware refinement of the query-head count that
    // feeds the oneDNN Graph-scratch zone floor (see the field comment in
    // ggml-sycl.h), merged into this same loop over [0, n_layer) rather
    // than a second pass over the identical range. Split the max by
    // attention window class -- non-SWA (effective KV window == n_ctx) vs
    // SWA (effective KV window == min(n_ctx, n_swa + n_ubatch); n_swa is
    // already captured above) -- and restrict to layers eligible for the
    // oneDNN SDPA route: a layer that can never reach oneDNN must not
    // inflate a floor sized for oneDNN's own scratch demand. Max rather
    // than layer 0 alone because a handful of architectures vary head
    // count (and head dim) by layer. hparams.n_embd_head_k(il) is used
    // UNCONDITIONALLY, including for MLA architectures (deepseek2,
    // glm-dsa, kimi-k3, kimi-linear): on the absorbed MLA path Q is built
    // as ggml_concat(q_nope_absorbed, q_pe, 0) (src/models/deepseek2.cpp),
    // giving ne0 == kv_lora_rank + n_rot -- exactly the key_length the
    // conversion scripts already write into n_embd_head_k, and exactly
    // the ne00 the real oneDNN gate reads (params.ne00 = Q->ne[0],
    // fattn.cpp). hparams.n_embd_head_k_mla() is the model-global
    // decompressed KEY head size (qk_nope + qk_rope, the key_length_mla
    // the conversion scripts write) that only the non-absorbed MHA path
    // would use; on the absorbed path K stays at kv_lora_rank + n_rot and
    // only V is decompressed, by v_mla AFTER the attention op returns
    // (llama-graph.cpp) -- so it is a size the flash-attention op itself
    // never sees, and branching on it here would screen the wrong
    // dimension and wrongly accept layers oneDNN will never serve
    // (DeepSeek-V3: n_embd_head_k_mla()=192, <=256 and so eligible, vs.
    // the real ne00=576, >512 and so correctly ineligible at
    // fattn-onednn.cpp).
    // f_attention_scale is model-global too, so hoisted out of the loop
    // once rather than re-derived per layer.
    inventory.n_swa                = hparams.n_swa;
    inventory.n_swa_layers         = 0;
    inventory.swa_layer_mask       = swa_layer_mask;
    inventory.swa_layer_mask_count = n_layer;
    // llama.cpp-3aos: per-layer KV width/kind truth -- see the field
    // comments in ggml-sycl.h and llama_model_sycl_build_kv_layer_arrays()
    // above for why a single global width/mask pair cannot express Gemma
    // 4's heterogeneous layers.
    inventory.kv_k_width_per_layer = kv_k_width_per_layer;
    inventory.kv_v_width_per_layer = kv_v_width_per_layer;
    inventory.kv_layer_kind        = kv_layer_kind;
    inventory.kv_layer_count       = n_layer;
    inventory.kv_idx_k_width_per_layer = kv_idx_k_width_per_layer;

    uint32_t    n_head_ctx_max        = 0;
    uint32_t    n_head_swa_max        = 0;
    uint32_t    n_head_all_max        = 0;
    const float model_attention_scale = hparams.f_attention_scale;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const bool is_swa_layer = hparams.is_swa(il);
        if (is_swa_layer) {
            inventory.n_swa_layers++;
        }
        // llama.cpp-rqak: the all-layers maximum must be taken BEFORE the
        // eligibility `continue` below -- the non-FA attention guard this
        // feeds runs on every attention layer, not just the oneDNN-eligible
        // ones the two per-class maxima below are scoped to.
        n_head_all_max = std::max(n_head_all_max, hparams.n_head(il));
        // The oneDNN SDPA path refuses attention with sinks, whatever the head size.
        if (sinks_layer_mask && sinks_layer_mask[il]) {
            continue;
        }
        if (!llama_model_sycl_onednn_head_dim_eligible(hparams.n_embd_head_k(il), model_attention_scale)) {
            continue;
        }
        if (is_swa_layer) {
            n_head_swa_max = std::max(n_head_swa_max, hparams.n_head(il));
        } else {
            n_head_ctx_max = std::max(n_head_ctx_max, hparams.n_head(il));
        }
    }
    inventory.n_head_ctx_max = n_head_ctx_max;
    inventory.n_head_swa_max = n_head_swa_max;
    inventory.n_head_all_max = n_head_all_max;
}

static void llama_model_sycl_apply_inventory(const ggml_sycl_tensor_inventory &   inventory,
                                             const ggml_sycl_placement_envelope & envelope,
                                             bool                                 early) {
    const auto hooks = llama_model_sycl_hooks();
    if (!llama_model_sycl_hooks_enabled(hooks)) {
        return;
    }
    // Both callers run inside an active loading guard, whose require_host_backing_hooks() refused a backend without
    // the staging proc before begin().
    const auto rc = hooks.stage_inventory(&inventory, &envelope, early);
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        throw std::runtime_error(format("SYCL inventory planning failed: early=%d result=%d", early, (int) rc));
    }
}

static void llama_model_sycl_add_tensor_info(std::vector<ggml_sycl_tensor_info> & tensors,
                                             size_t &                             total_size,
                                             size_t &                             max_pp_pipeline_weight_bytes,
                                             ggml_tensor *                        tensor) {
    if (tensor == nullptr) {
        return;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) {
        return;
    }

    ggml_sycl_tensor_info info = {};
    info.name                  = ggml_get_name(tensor);
    info.size                  = nbytes;
    info.type                  = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        info.ne[i] = tensor->ne[i];
    }
    tensors.push_back(info);
    total_size += nbytes;
    if (ggml_is_quantized(tensor->type)) {
        const size_t fp16_bytes      = static_cast<size_t>(ggml_nelements(tensor)) * sizeof(ggml_fp16_t);
        max_pp_pipeline_weight_bytes = std::max(max_pp_pipeline_weight_bytes, fp16_bytes);
    }
}

// llama.cpp-8ony: the tensors the planner must not charge MUL_MAT scratch for. The loader's own op table
// (llm_tensor_info_for) says which input-layer tensors are consumed by a row gather (GET_ROWS); a tied token embedding
// is the one case where such a tensor is ALSO a MUL_MAT operand, and the loader's rule for that is the same one used
// here: an output head the file does not carry is created as TENSOR_DUPLICATED token_embd and treated as
// LLM_TENSOR_OUTPUT (llama_model_loader::create_tensor). The file's tensor set (weights_map) is the loader's fact for
// "the file carries the head", and it is known before any tensor is created, so the early plan sees it too.
// A false answer is the safe direction: it only keeps the tensor in the plan.
static void llama_model_sycl_mark_get_rows_only(std::vector<ggml_sycl_tensor_info> & tensors,
                                                const llama_model_loader &           ml) {
    const LLM_TN tn(ml.get_arch());
    const bool   file_carries_head = ml.weights_map.find(tn(LLM_TENSOR_OUTPUT, "weight").str()) != ml.weights_map.end();
    static const llm_tensor input_tensors[] = {
        LLM_TENSOR_TOKEN_EMBD,
        LLM_TENSOR_POS_EMBD,
        LLM_TENSOR_TOKEN_TYPES,
        LLM_TENSOR_PER_LAYER_TOKEN_EMBD,
    };
    for (ggml_sycl_tensor_info & tensor : tensors) {
        tensor.get_rows_only = false;
        if (tensor.name == nullptr) {
            continue;
        }
        for (const llm_tensor input : input_tensors) {
            const llm_tensor_info & info = llm_tensor_info_for(input);
            if (info.layer != LLM_TENSOR_LAYER_INPUT || info.op != GGML_OP_GET_ROWS ||
                tn(input, "weight").str() != tensor.name) {
                continue;
            }
            const bool tied_head = input == LLM_TENSOR_TOKEN_EMBD && !file_carries_head;
            tensor.get_rows_only = !tied_head;
        }
    }
}

// Both inventories carry libllama's resolved storage inputs (design 5efe §4.3): use_mmap after the caps
// loop, the load mode's mlock, and the host weight layout. The staging proc derives the host-backing fact from them
// and keeps it on the plan, which every later reader asks; libllama never derives it.
static void llama_model_sycl_set_inventory_host_backing(ggml_sycl_tensor_inventory & inventory,
                                                        const llama_model_loader &   ml,
                                                        bool                         use_mlock) {
    inventory.use_mmap_resolved      = ml.use_mmap;
    inventory.use_mlock_resolved     = use_mlock;
    inventory.host_weight_layout_cpu = llama_model_loader::host_weight_layout_cpu();
}

// The STORAGE rows an inventory's lazy tables describe, and the infos that point at them. Each info's name points into
// tables[i].name, so the owner is never copied (a copy's infos would point into the source), and `tables` is not
// touched once the infos are filled. It must live as long as the inventory that carries the infos. Moving it is safe: a
// vector move keeps its elements where they are.
struct llama_model_sycl_lazy_tables {
    std::vector<llama_lazy_storage_table>  tables;
    std::vector<ggml_sycl_lazy_table_info> infos;

    llama_model_sycl_lazy_tables()                                                 = default;
    llama_model_sycl_lazy_tables(const llama_model_sycl_lazy_tables &)             = delete;
    llama_model_sycl_lazy_tables & operator=(const llama_model_sycl_lazy_tables &) = delete;
    llama_model_sycl_lazy_tables(llama_model_sycl_lazy_tables &&)                  = default;
    llama_model_sycl_lazy_tables & operator=(llama_model_sycl_lazy_tables &&)      = default;
};

// Hands the planner the record's STORAGE rows as the inventory's lazy tables, owned by `owner`.
static void llama_model_sycl_set_inventory_lazy_tables(ggml_sycl_tensor_inventory &   inventory,
                                                       const llama_model_loader &     ml,
                                                       llama_model_sycl_lazy_tables & owner) {
    owner.tables =
        llama_lazy_range_storage_tables(ml.lazy_record, [&](const std::string & name) -> const ggml_tensor * {
            const auto it = ml.weights_map.find(name);
            return it != ml.weights_map.end() ? it->second.tensor : nullptr;
        });
    owner.infos.clear();
    for (const llama_lazy_storage_table & t : owner.tables) {
        owner.infos.push_back(llama_lazy_storage_table_info(t));
    }
    inventory.lazy_tables      = owner.infos.empty() ? nullptr : owner.infos.data();
    inventory.lazy_table_count = static_cast<uint32_t>(owner.infos.size());
}

// The early plan's inventory, built from weights_map before create_tensor() runs, with the storage its raw pointers
// point into. The tensor names point into the loader's metadata, so the loader must outlive it. The load's early plan
// and the plan dry run's header read build it here, so the dry run plans what the load plans.
struct llama_model_sycl_early_inventory {
    std::vector<ggml_sycl_tensor_info> tensors;
    std::unique_ptr<bool[]>            swa_layer_mask;
    llama_model_sycl_kv_layer_arrays   kv_layer_arrays;
    ggml_sycl_tensor_inventory         inventory  = {};
    size_t                             total_size = 0;
    llama_model_sycl_lazy_tables       lazy;
};

// False when no weight remains for the plan.
static bool llama_model_sycl_build_early_inventory(llama_model_loader &               ml,
                                                   const llama_model &                model,
                                                   bool                               use_mlock,
                                                   llama_model_sycl_early_inventory & out) {
    const llama_hparams & hparams = model.hparams;

    out.tensors.reserve(ml.weights_map.size());
    size_t max_pp_pipeline_weight_bytes = 0;
    for (auto & weight : ml.weights_map) {
        // a lazy table the record defers to the SYCL storage tier is not an ordinary weight:
        // it is not placed, staged, or counted in the maxima and the host total. An
        // ineligible row (OFF, or AUTO at or below 4 GiB) is the ordinary weight it is.
        if (llama_lazy_early_plan_excludes(ml.lazy_record, weight.first)) {
            continue;
        }
        llama_model_sycl_add_tensor_info(out.tensors, out.total_size, max_pp_pipeline_weight_bytes,
                                         weight.second.tensor);
    }
    if (out.tensors.empty()) {
        return false;
    }

    llama_model_sycl_mark_get_rows_only(out.tensors, ml);

    const uint32_t n_layer = hparams.n_layer();
    out.swa_layer_mask.reset(new bool[n_layer]);
    for (uint32_t il = 0; il < n_layer; ++il) {
        out.swa_layer_mask[il] = hparams.is_swa(il);
    }
    out.kv_layer_arrays = llama_model_sycl_build_kv_layer_arrays(model, n_layer);

    const std::unique_ptr<bool[]> sinks_layer_mask = llama_model_sycl_attn_sinks_layers(ml, n_layer);

    ggml_sycl_tensor_inventory & inventory = out.inventory;
    llama_model_sycl_populate_inventory(
        inventory, out.tensors, out.swa_layer_mask.get(), sinks_layer_mask.get(), out.kv_layer_arrays.k_width.get(),
        out.kv_layer_arrays.v_width.get(), out.kv_layer_arrays.kind.get(), out.kv_layer_arrays.idx_k_width.get(),
        out.total_size, max_pp_pipeline_weight_bytes, hparams, model.get_n_ctx_hint(), model.get_n_ubatch_hint());
    llama_model_sycl_set_inventory_host_backing(inventory, ml, use_mlock);
    llama_model_sycl_set_inventory_lazy_tables(inventory, ml, out.lazy);
    return true;
}

static void llama_model_sycl_compute_early_plan(llama_model_loader & ml,
                                                const llama_model &  model,
                                                bool                 use_mlock,
                                                const char *         log_func) {
    if (!llama_model_sycl_hooks_enabled()) {
        return;
    }
    if (ml.no_alloc || ml.weights_map.empty()) {
        return;
    }

    llama_model_sycl_early_inventory early;
    if (!llama_model_sycl_build_early_inventory(ml, model, use_mlock, early)) {
        return;
    }
    const ggml_sycl_placement_envelope envelope = llama_model_sycl_make_placement_envelope();
    llama_model_sycl_apply_inventory(early.inventory, envelope, true);
    // From here the bound load has a staged plan, and the loader's buffer-type choices ask it for the fact.
    ml.sycl_load_plan_staged = true;

    LLAMA_LOG_INFO("%s: SYCL early placement plan: %zu weights, %.2f GiB (computed pre-create_tensor)\n", log_func,
                   early.tensors.size(), early.total_size / (1024.0 * 1024.0 * 1024.0));
}

static void llama_model_sycl_set_late_inventory(llama_model_loader & ml,
                                                const llama_model &  model,
                                                bool                 use_mlock,
                                                const char *         log_func) {
    const llama_hparams & hparams = model.hparams;
    if (!llama_model_sycl_hooks_enabled()) {
        return;
    }
    if (ml.no_alloc) {
        return;
    }

    LLAMA_LOG_INFO("%s: [SYCL] Starting tensor inventory collection (ctx_map size: %zu)\n", log_func,
                   ml.ctx_map.size());

    std::vector<ggml_sycl_tensor_info> tensors;
    size_t                             total_size                   = 0;
    size_t                             max_pp_pipeline_weight_bytes = 0;

    for (auto & it : ml.ctx_map) {
        ggml_backend_buffer_type_t buft = it.first.buft;
        if (!llama_model_buft_backend_is_sycl(buft)) {
            continue;
        }
        ggml_context * ctx = it.second.get();
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            llama_model_sycl_add_tensor_info(tensors, total_size, max_pp_pipeline_weight_bytes, t);
        }
    }

    LLAMA_LOG_INFO("%s: [SYCL] Collected %zu tensors, %.2f GiB total\n", log_func, tensors.size(),
                   total_size / (1024.0 * 1024.0 * 1024.0));
    if (tensors.empty()) {
        return;
    }

    llama_model_sycl_mark_get_rows_only(tensors, ml);

    const uint32_t n_layer = hparams.n_layer();
    std::unique_ptr<bool[]> swa_layer_mask(new bool[n_layer]);
    for (uint32_t il = 0; il < n_layer; ++il) {
        swa_layer_mask[il] = hparams.is_swa(il);
    }
    llama_model_sycl_kv_layer_arrays kv_layer_arrays = llama_model_sycl_build_kv_layer_arrays(model, n_layer);

    const std::unique_ptr<bool[]> sinks_layer_mask = llama_model_sycl_attn_sinks_layers(ml, n_layer);

    ggml_sycl_tensor_inventory inventory = {};
    llama_model_sycl_populate_inventory(
        inventory, tensors, swa_layer_mask.get(), sinks_layer_mask.get(), kv_layer_arrays.k_width.get(),
        kv_layer_arrays.v_width.get(), kv_layer_arrays.kind.get(), kv_layer_arrays.idx_k_width.get(), total_size,
        max_pp_pipeline_weight_bytes, hparams, model.get_n_ctx_hint(), model.get_n_ubatch_hint());
    llama_model_sycl_set_inventory_host_backing(inventory, ml, use_mlock);
    llama_model_sycl_lazy_tables lazy;
    llama_model_sycl_set_inventory_lazy_tables(inventory, ml, lazy);

    const ggml_sycl_placement_envelope envelope = llama_model_sycl_make_placement_envelope();
    llama_model_sycl_apply_inventory(inventory, envelope, false);

    LLAMA_LOG_INFO("%s: SYCL tensor inventory: %zu tensors, %.2f GiB (enables unified placement before allocation)\n",
                   log_func, tensors.size(), total_size / (1024.0 * 1024.0 * 1024.0));
}

// The weights a load stage's measure (probe, admitted or late, llama.cpp-p6i0) puts its stand-ins on: every
// (buffer type, context) of the loader, whose tensors are not allocated yet when each stage measures.
static std::vector<llama_measure_dummy_entry> llama_model_measure_weights(const llama_model_loader & ml) {
    std::vector<llama_measure_dummy_entry> weights;
    for (const auto & [ctx_key, ctx_ptr] : ml.ctx_map) {
        weights.push_back({ ctx_key.buft, ctx_ptr.get() });
    }
    return weights;
}
#endif  // GGML_USE_SYCL || GGML_BACKEND_DL

const std::vector<std::string> & llama_model_lazy_tensor_names(llm_arch arch) {
    // the two architectures that create a table with TENSOR_READ_LAZY: qwen4exp
    // (src/models/qwen4exp.cpp) and gemma4 (src/models/gemma4.cpp)
    static const std::vector<std::string> qwen4exp = { LLM_TN(LLM_ARCH_QWEN4EXP)(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str() };
    static const std::vector<std::string> gemma4   = { LLM_TN(LLM_ARCH_GEMMA4)(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str() };
    static const std::vector<std::string> none;

    switch (arch) {
        case LLM_ARCH_QWEN4EXP: return qwen4exp;
        case LLM_ARCH_GEMMA4:   return gemma4;
        default:                return none;
    }
}

// The lazy-range record: one row per TENSOR_READ_LAZY tensor this arch declares and the
// file carries, filled before the early plan, which builds its inventory from weights_map
// and so cannot see a row that create_tensor() would fill later. It runs the same way
// under no_alloc, so the memory-fit pass and the plan dry run plan what the real load plans.
static void llama_model_prefill_lazy_record(llama_model_loader &  ml,
                                            llm_arch              arch,
                                            const llama_hparams & hparams,
                                            enum llama_lazy_mode  lazy_mode_pre,
                                            bool                  lazy_taker) {
    llama_lazy_range_record_prefill(
        ml.lazy_record, arch, hparams, llama_model_lazy_tensor_names(arch),
        [&ml](const std::string & name, llama_lazy_range_source & out) {
            const llama_model_loader::llama_tensor_weight * w = ml.get_weight(name.c_str());
            if (w == nullptr) {
                return false;
            }
            out = { w->idx, w->offs, ggml_nbytes(w->tensor) };
            return true;
        },
        lazy_mode_pre, ml.lazy.mode, lazy_taker, llama_mmap::SUPPORTED);
}

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
bool llama_model_sycl_header_inventory(const char *                         fname,
                                       const llama_model_params &           params,
                                       llama_model_sycl_header_inventory_cb callback,
                                       void *                               user_data) {
    if (fname == nullptr || callback == nullptr) {
        return false;
    }
    try {
        // The header only: no_alloc, so no tensor data is read and no buffer is allocated, and no load transaction
        // is opened, so nothing is staged.
        std::vector<std::string> splits;
        llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr, fname,
                              splits, /*file*/ nullptr, params.load_mode, /*check_tensors*/ false,
                              /*no_alloc*/ true, params.load_mtp, params.kv_overrides,
                              /*param_tensor_buft_overrides_p*/ nullptr);
        ml.lazy.mode = params.lazy_mode;

        // the model reads no_alloc from its own params, so the copy it is built from carries it
        llama_model_params mparams = params;
        mparams.no_alloc           = true;

        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));
        auto *                       model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            return false;
        }
        model->hparams.vocab_only = false;
        model->load_hparams(ml);

        // The load resolves AUTO against its devices, keeping it wherever SYCL owns placement; the dry run plans for
        // SYCL-owned weights, so AUTO stays AUTO, and the model is the lazy taker as a SYCL-owned load is.
        llama_model_prefill_lazy_record(ml, model->arch, model->hparams, ml.lazy.mode, /*lazy_taker*/ true);

        // The storage inputs the load's inventory carries, resolved as the load resolves them: mlock from the load
        // mode, use_mmap as the loader set it. The load's caps loop, which turns use_mmap off when one of its devices
        // cannot map the file, is not run; it matters only where the load would plan on an integrated GPU alone, which
        // it adds to its devices only when no discrete card is found.
        const bool use_mlock =
            params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        llama_model_sycl_early_inventory early;
        if (!llama_model_sycl_build_early_inventory(ml, *model, use_mlock, early)) {
            return false;
        }
        const ggml_sycl_placement_envelope envelope = llama_model_sycl_make_placement_envelope();
        callback(&early.inventory, &envelope, user_data);
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}
#endif  // GGML_USE_SYCL || GGML_BACKEND_DL

llama_lazy_tensor_attrs llama_model_lazy_tensor_attrs(llm_arch arch, const llama_hparams & hparams, const std::string & name) {
    const llama_lazy_tensor_attrs unknown = { 0, -1, false };

    switch (arch) {
        case LLM_ARCH_QWEN4EXP:
            if (name == LLM_TN(arch)(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str()) {
                // the one layer that consumes the table comes from hparams, never a literal:
                // the loader refuses more than one PLE layer (src/models/qwen4exp.cpp)
                int32_t consumer_layer = -1;
                if (hparams.is_ple_impl.count() == 1) {
                    for (uint32_t il = 0; il < LLAMA_MAX_LAYERS; ++il) {
                        if (hparams.is_ple_impl.test(il)) {
                            consumer_layer = (int32_t) il;
                        }
                    }
                }
                return { hparams.ple_n_heads, consumer_layer, true };
            }
            return unknown;
        case LLM_ARCH_GEMMA4:
            if (name == LLM_TN(arch)(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str()) {
                return { 1, 0, false };
            }
            return unknown;
        default:
            return unknown;
    }
}

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_CLIP:
            return new llama_model_clip(params);
        case LLM_ARCH_LLAMA:
            return new llama_model_llama(params);
        case LLM_ARCH_LLAMA4:
            return new llama_model_llama4(params);
        case LLM_ARCH_LLAMA_EMBED:
            return new llama_model_llama_embed(params);
        case LLM_ARCH_MAINCODER:
            return new llama_model_maincoder(params);
        case LLM_ARCH_MAION_CODER:
            return new llama_model_maion_coder(params);
        case LLM_ARCH_TALKIE:
            return new llama_model_talkie(params);
        case LLM_ARCH_DECI:
            return new llama_model_deci(params);
        case LLM_ARCH_BAICHUAN:
            return new llama_model_baichuan(params);
        case LLM_ARCH_FALCON:
            return new llama_model_falcon(params);
        case LLM_ARCH_GROK:
            return new llama_model_grok(params);
        case LLM_ARCH_STARCODER:
            return new llama_model_starcoder(params);
        case LLM_ARCH_REFACT:
            return new llama_model_refact(params);
        case LLM_ARCH_BERT:
            return new llama_model_bert(params);
        case LLM_ARCH_JINA_BERT_V2:
            return new llama_model_jina_bert_v2(params);
        case LLM_ARCH_JINA_BERT_V3:
            return new llama_model_jina_bert_v3(params);
        case LLM_ARCH_NOMIC_BERT:
            return new llama_model_nomic_bert(params);
        case LLM_ARCH_NOMIC_BERT_MOE:
            return new llama_model_nomic_bert_moe(params);
        case LLM_ARCH_MODERN_BERT:
            return new llama_model_modern_bert(params);
        case LLM_ARCH_NEO_BERT:
            return new llama_model_neo_bert(params);
        case LLM_ARCH_EUROBERT:
            return new llama_model_eurobert(params);
        case LLM_ARCH_BLOOM:
            return new llama_model_bloom(params);
        case LLM_ARCH_MPT:
            return new llama_model_mpt(params);
        case LLM_ARCH_STABLELM:
            return new llama_model_stablelm(params);
        case LLM_ARCH_MELLUM:
            return new llama_model_mellum(params);
        case LLM_ARCH_NANBEIGE:
            return new llama_model_nanbeige(params);
        case LLM_ARCH_QWEN:
            return new llama_model_qwen(params);
        case LLM_ARCH_QWEN2:
            return new llama_model_qwen2(params);
        case LLM_ARCH_DREAM:
            return new llama_model_dream(params);
        case LLM_ARCH_LLADA:
            return new llama_model_llada(params);
        case LLM_ARCH_LLADA_MOE:
            return new llama_model_llada_moe(params);
        case LLM_ARCH_RND1:
            return new llama_model_rnd1(params);
        case LLM_ARCH_QWEN2VL:
            return new llama_model_qwen2vl(params);
        case LLM_ARCH_QWEN2MOE:
            return new llama_model_qwen2moe(params);
        case LLM_ARCH_SENSENOVA_U1:
            return new llama_model_sensenova_u1(params);
        case LLM_ARCH_QWEN3:
            return new llama_model_qwen3(params);
        case LLM_ARCH_FLINT:
            return new llama_model_flint(params);
        case LLM_ARCH_QWEN3MOE:
            return new llama_model_qwen3moe(params);
        case LLM_ARCH_QWEN3VL:
            return new llama_model_qwen3vl(params);
        case LLM_ARCH_QWEN3VLMOE:
            return new llama_model_qwen3vlmoe(params);
        case LLM_ARCH_QWEN3TTS:
            return new llama_model_qwen3tts(params);
        case LLM_ARCH_POCKETTTS:
            return new llama_model_pockettts(params);
        case LLM_ARCH_MOSHI:
            return new llama_model_moshi(params);
        case LLM_ARCH_ZONOS3:
            return new llama_model_zonos3(params);
        case LLM_ARCH_KOKORO:
            return new llama_model_kokoro(params);
        case LLM_ARCH_TINY_VOCA:
            return new llama_model_tiny_voca(params);
        case LLM_ARCH_PHI2:
            return new llama_model_phi2(params);
        case LLM_ARCH_PHI3:
            return new llama_model_phi3(params);
        case LLM_ARCH_PHIMOE:
            return new llama_model_phimoe(params);
        case LLM_ARCH_PLAMO:
            return new llama_model_plamo(params);
        case LLM_ARCH_PLAMO2:
            return new llama_model_plamo2(params);
        case LLM_ARCH_PLAMO3:
            return new llama_model_plamo3(params);
        case LLM_ARCH_GPT2:
            return new llama_model_gpt2(params);
        case LLM_ARCH_CODESHELL:
            return new llama_model_codeshell(params);
        case LLM_ARCH_ORION:
            return new llama_model_orion(params);
        case LLM_ARCH_INTERNLM2:
            return new llama_model_internlm2(params);
        case LLM_ARCH_MINICPM3:
            return new llama_model_minicpm3(params);
        case LLM_ARCH_GEMMA:
            return new llama_model_gemma(params);
        case LLM_ARCH_GEMMA2:
            return new llama_model_gemma2(params);
        case LLM_ARCH_GEMMA3:
            return new llama_model_gemma3(params);
        case LLM_ARCH_GEMMA3N:
            return new llama_model_gemma3n(params);
        case LLM_ARCH_GEMMA4:
            return new llama_model_gemma4(params);
        case LLM_ARCH_GEMMA4_ASSISTANT:
            return new llama_model_gemma4_assistant(params);
        case LLM_ARCH_GEMMA_EMBEDDING:
            return new llama_model_gemma_embedding(params);
        case LLM_ARCH_GEMMA_EMBEDDING2:
            return new llama_model_gemma_embedding2(params);
        case LLM_ARCH_STARCODER2:
            return new llama_model_starcoder2(params);
        case LLM_ARCH_MAMBA:
            return new llama_model_mamba(params);
        case LLM_ARCH_MAMBA2:
            return new llama_model_mamba2(params);
        case LLM_ARCH_MAPLE:
            return new llama_model_maple(params);
        case LLM_ARCH_JAMBA:
            return new llama_model_jamba(params);
        case LLM_ARCH_XVERSE:
            return new llama_model_xverse(params);
        case LLM_ARCH_COMMAND_R:
            return new llama_model_command_r(params);
        case LLM_ARCH_COHERE2:
            return new llama_model_cohere2(params);
        case LLM_ARCH_COHERE2MOE:
            return new llama_model_cohere2moe(params);
        case LLM_ARCH_DBRX:
            return new llama_model_dbrx(params);
        case LLM_ARCH_OLMO:
            return new llama_model_olmo(params);
        case LLM_ARCH_OLMO2:
            return new llama_model_olmo2(params);
        case LLM_ARCH_OLMOE:
            return new llama_model_olmoe(params);
        case LLM_ARCH_MUSE_GLIMMER:
            return new llama_model_muse_glimmer(params);
        case LLM_ARCH_MUSE_SPARK:
            return new llama_model_muse_spark(params);
        case LLM_ARCH_OPENELM:
            return new llama_model_openelm(params);
        case LLM_ARCH_GPTNEOX:
            return new llama_model_gptneox(params);
        case LLM_ARCH_ARCTIC:
            return new llama_model_arctic(params);
        case LLM_ARCH_DEEPSEEK:
            return new llama_model_deepseek(params);
        case LLM_ARCH_DEEPSEEK2:
            return new llama_model_deepseek2(params);
        case LLM_ARCH_DEEPSEEK2OCR:
            return new llama_model_deepseek2ocr(params);
        case LLM_ARCH_DEEPSEEK32:
            return new llama_model_deepseek32(params);
        case LLM_ARCH_DOTS3NOTE:
            return new llama_model_dots3note(params);
        case LLM_ARCH_DEEPSEEK4:
            return new llama_model_deepseek4(params);
        case LLM_ARCH_DEEPSEEK41:
            return new llama_model_deepseek41(params);
        case LLM_ARCH_XING4_0:
            return new llama_model_xing4_0(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);
        case LLM_ARCH_MISTRAL4:
            return new llama_model_mistral4(params);
        case LLM_ARCH_MISTRAL_LARGE4:
            return new llama_model_mistral_large4(params);
        case LLM_ARCH_CHATGLM:
            return new llama_model_chatglm(params);
        case LLM_ARCH_GLM4:
            return new llama_model_glm4(params);
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_BITNET:
            return new llama_model_bitnet(params);
        case LLM_ARCH_T5:
            return new llama_model_t5(params);
        case LLM_ARCH_T5ENCODER:
            return new llama_model_t5encoder(params);
        case LLM_ARCH_JAIS:
            return new llama_model_jais(params);
        case LLM_ARCH_JAIS2:
            return new llama_model_jais2(params);
        case LLM_ARCH_NEMOTRON:
            return new llama_model_nemotron(params);
        case LLM_ARCH_NEMOTRON_H:
            return new llama_model_nemotron_h(params);
        case LLM_ARCH_NEMOTRON_H_MOE:
            return new llama_model_nemotron_h_moe(params);
        case LLM_ARCH_METIS16:
            return new llama_model_metis16(params);
        case LLM_ARCH_EXAONE:
            return new llama_model_exaone(params);
        case LLM_ARCH_EXAONE4:
            return new llama_model_exaone4(params);
        case LLM_ARCH_EXAONE_MOE:
            return new llama_model_exaone_moe(params);
        case LLM_ARCH_RWKV6:
            return new llama_model_rwkv6(params);
        case LLM_ARCH_RWKV6QWEN2:
            return new llama_model_rwkv6qwen2(params);
        case LLM_ARCH_RWKV7:
            return new llama_model_rwkv7(params);
        case LLM_ARCH_ARWKV7:
            return new llama_model_arwkv7(params);
        case LLM_ARCH_GRANITE:
            return new llama_model_granite(params);
        case LLM_ARCH_GRANITE_MOE:
            return new llama_model_granite_moe(params);
        case LLM_ARCH_GRANITE_SWITCH:
            return new llama_model_granite_switch(params);
        case LLM_ARCH_MINICPM:
            return new llama_model_minicpm(params);
        case LLM_ARCH_GRANITE_HYBRID:
            return new llama_model_granite_hybrid(params);
        case LLM_ARCH_GRANITE_SWA:
            return new llama_model_granite_swa(params);
        case LLM_ARCH_CHAMELEON:
            return new llama_model_chameleon(params);
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return new llama_model_wavtokenizer_dec(params);
        case LLM_ARCH_PLM:
            return new llama_model_plm(params);
        case LLM_ARCH_BAILINGMOE:
            return new llama_model_bailingmoe(params);
        case LLM_ARCH_BAILINGMOE2:
            return new llama_model_bailingmoe2(params);
        case LLM_ARCH_BAILINGMOE3:
            return new llama_model_bailingmoe3(params);
        case LLM_ARCH_SEED_OSS:
            return new llama_model_seed_oss(params);
        case LLM_ARCH_DOTS1:
            return new llama_model_dots1(params);
        case LLM_ARCH_ARCEE:
            return new llama_model_arcee(params);
        case LLM_ARCH_AFMOE:
            return new llama_model_afmoe(params);
        case LLM_ARCH_LAGUNA:
            return new llama_model_laguna(params);
        case LLM_ARCH_RICURSIVE:
            return new llama_model_ricursive(params);
        case LLM_ARCH_ERNIE4_5:
            return new llama_model_ernie4_5(params);
        case LLM_ARCH_ERNIE4_5_MOE:
            return new llama_model_ernie4_5_moe(params);
        case LLM_ARCH_PADDLEOCR:
            return new llama_model_paddleocr(params);
        case LLM_ARCH_HUNYUAN_MOE:
            return new llama_model_hunyuan_moe(params);
        case LLM_ARCH_HUNYUAN_VL:
            return new llama_model_hunyuan_vl(params);
        case LLM_ARCH_HUNYUAN_DENSE:
            return new llama_model_hunyuan_dense(params);
        case LLM_ARCH_HY_V3:
            return new llama_model_hy_v3(params);
        case LLM_ARCH_HY_V4:
            return new llama_model_hy_v4(params);
        case LLM_ARCH_SMOLLM3:
            return new llama_model_smollm3(params);
        case LLM_ARCH_OPENAI_MOE:
            return new llama_model_openai_moe(params);
        case LLM_ARCH_FALCON_H1:
            return new llama_model_falcon_h1(params);
        case LLM_ARCH_INKLING2:
            return new llama_model_inkling2(params);
        case LLM_ARCH_INKLING2_LOOP:
            return new llama_model_inkling2_loop(params);
        case LLM_ARCH_LFM2:
            return new llama_model_lfm2(params);
        case LLM_ARCH_LFM2MOE:
            return new llama_model_lfm2moe(params);
        case LLM_ARCH_LFM3:
            return new llama_model_lfm3(params);
        case LLM_ARCH_ZAYA1:
            return new llama_model_zaya1(params);
        case LLM_ARCH_SMALLTHINKER:
            return new llama_model_smallthinker(params);
        case LLM_ARCH_GROVEMOE:
            return new llama_model_grovemoe(params);
        case LLM_ARCH_APERTUS:
            return new llama_model_apertus(params);
        case LLM_ARCH_MINIMAX_01:
            return new llama_model_minimax_01(params);
        case LLM_ARCH_MINIMAX_M2:
            return new llama_model_minimax_m2(params);
        case LLM_ARCH_MINIMAX_M3:
            return new llama_model_minimax_m3(params);
        case LLM_ARCH_HRM_TEXT:
            return new llama_model_hrm_text(params);
        case LLM_ARCH_BEAM:
            return new llama_model_beam(params);
        case LLM_ARCH_LIMITE:
            return new llama_model_limite(params);
        case LLM_ARCH_COGVLM:
            return new llama_model_cogvlm(params);
        case LLM_ARCH_PANGU_EMBED:
            return new llama_model_pangu_embed(params);
        case LLM_ARCH_QWEN3NEXT:
            return new llama_model_qwen3next(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        case LLM_ARCH_CLEF:
            return new llama_model_clef(params);
        case LLM_ARCH_VLLM_DECISION2:
            return new llama_model_vllm_decision2(params);
        case LLM_ARCH_CLOVER_MOE:
            return new llama_model_clover_moe(params);
        case LLM_ARCH_QWEN4EXP:
            return new llama_model_qwen4exp(params);
        case LLM_ARCH_MISTRAL3:
            return new llama_model_mistral3(params);
        case LLM_ARCH_EAGLE3:
            return new llama_model_eagle3(params);
        case LLM_ARCH_DFLASH:
            return new llama_model_dflash(params);
        case LLM_ARCH_MIMO2:
            return new llama_model_mimo2(params);
        case LLM_ARCH_KIMI_LINEAR:
            return new llama_model_kimi_linear(params);
        case LLM_ARCH_KIMI_K3:
            return new llama_model_kimi_k3(params);
        case LLM_ARCH_KIMI_K31:
            return new llama_model_kimi_k31(params);
        case LLM_ARCH_GLM5_NEXT:
            return new llama_model_glm5_next(params);
        case LLM_ARCH_STEP35:
            return new llama_model_step35(params);
        case LLM_ARCH_SPARK2_5:
            return new llama_model_spark2_5(params);
        case LLM_ARCH_K2_HORIZON:
            return new llama_model_k2_horizon(params);
        case LLM_ARCH_KOLIBRI1:
            return new llama_model_kolibri1(params);
        case LLM_ARCH_DOMYN_EDGE:
            return new llama_model_domyn_edge(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    const bool is_dsv4 = ud->model->arch == LLM_ARCH_DEEPSEEK4 || ud->model->arch == LLM_ARCH_DEEPSEEK41 ||
        (ud->model->arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0);
    const bool is_glm5_next = ud->model->arch == LLM_ARCH_GLM5_NEXT;

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_v_exps_weight   ("blk\\.\\d*\\.attn_v_exps.weight"); // K2 Horizon MoVA
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_qkv_gate_weight ("blk\\.\\d*\\.attn_qkv_gate.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_idx_cache       ("cache_idx_(k|v)_l\\d*");
    static const std::regex pattern_dsv4_state      ("dsv4_(csa|hca|lid)_state_(kv|score)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_gate_bias  ("blk\\.\\d*\\.attn_gate\\.bias");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_out_a_weight("blk\\.\\d*\\.attn_output_a\\.weight");
    static const std::regex pattern_attn_out_b_weight("blk\\.\\d*\\.attn_output_b\\.weight");
    static const std::regex pattern_attn_q_b_weight ("blk\\.\\d*\\.attn_q_b\\.weight");
    static const std::regex pattern_attn_kv_b_weight("blk\\.\\d*\\.attn_(k|v)_b\\.weight");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_ssm_beta_alpha_cat("blk\\.\\d*\\.ssm_beta_alpha.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_ple_r_cache     ("cache_ple_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");
    static const std::regex pattern_kda_conv1d      ("blk\\.\\d*\\.ssm_conv1d_(q|k|v)\\.weight");
    static const std::regex pattern_kda_gate_b      ("blk\\.\\d*\\.ssm_(f|g)_b\\.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias         ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias    ("blk\\.\\d*\\.ffn_down_exps.bias");
    static const std::regex pattern_ffn_up_shexp_weight   ("blk\\.\\d*\\.ffn_up_shexp.weight");
    static const std::regex pattern_ffn_gate_shexp_weight ("blk\\.\\d*\\.ffn_gate_shexp.weight");
    static const std::regex pattern_ffn_down_shexp_weight ("blk\\.\\d*\\.ffn_down_shexp.weight");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            const bool is_attn = tensor_name.compare(prefix.size(), 5, "attn_") == 0;
            // attention heads must be on the same device as the KV of the source layer
            rotation = get_il_eff(is_attn ? hparams.kv_source(il) : il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        if (ud->model->arch == LLM_ARCH_HRM_TEXT || ud->model->arch == LLM_ARCH_METIS16) {
            // aliased cache slots cannot satisfy the meta-split invariants, so replicate all tensors
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, tensor, 0, 0};
        }
        if (is_dsv4) {
            if (std::regex_match(tensor_name, pattern_kv_cache) ||
                    std::regex_match(tensor_name, pattern_dsv4_state)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2);
            }
            if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
        }
        if (is_glm5_next) {
            // split the KDA and MLA heads, the MLA latent cache and the indexer stay mirrored
            if (std::regex_match(tensor_name, pattern_kv_cache)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight) || std::regex_match(tensor_name, pattern_kda_gate_b) ||
                    std::regex_match(tensor_name, pattern_ssm_beta)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_kv_b_weight) || std::regex_match(tensor_name, pattern_kda_conv1d)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2, "attn_output.weight");
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
            }
        }
        if (is_dsv4 || is_glm5_next) {
            if (std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
            }
        }

        // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
        if (std::regex_match(tensor_name, pattern_idx_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole history
        if (std::regex_match(tensor_name, pattern_ple_r_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        // routed value experts {n_embd, n_embd_v_gqa, n_expert} produce V, so they split like attn_v
        if (std::regex_match(tensor_name, pattern_v_exps_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks) ||
                std::regex_match(tensor_name, pattern_attn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha) || std::regex_match(tensor_name, pattern_ssm_beta_alpha_cat)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            if (ud->model->arch == LLM_ARCH_LFM2 || ud->model->arch == LLM_ARCH_LFM2MOE || ud->model->arch == LLM_ARCH_LFM3) {
                // the LFM2 shortconv block runs fully mirrored, so its conv state must be mirrored too
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED, "");
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL, "ffn_down_exps.weight");
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            if (is_dsv4) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    // if a model has fused tensors they need to be separated into "segments", see the comments on ggml_backend_meta_split_state struct
    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (is_glm5_next && std::regex_match(tensor_name, pattern_r_cache)) {
            // the conv state holds the q, k and v windows one after the other
            const int64_t conv_state_size = (hparams.ssm_d_conv - 1) * hparams.n_head(il) * hparams.n_embd_head_kda;
            GGML_ASSERT(tensor->ne[axis] == 3*conv_state_size);
            return {{conv_state_size, 3}};
        }
        // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
        // zonos3 has the qwen35 GDN layout, its attention layers have no fused qkv
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE || ud->model->arch == LLM_ARCH_CLEF || ud->model->arch == LLM_ARCH_CLOVER_MOE ||
                ud->model->arch == LLM_ARCH_QWEN4EXP || ud->model->arch == LLM_ARCH_ZONOS3) {

            // fused full attention layers with Q gate tensors that need n_embd doubled:
            if (!hparams.is_recr(il) && (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias))) {
                const int64_t n_embd      = hparams.n_head(il) * hparams.n_embd_head_k(il) * 2;
                const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
                GGML_ASSERT(hparams.n_embd_k_gqa(il) == n_embd_gqa);
                GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
                return {{n_embd, 1}, {n_embd_gqa, 2}};
            }

            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2}, {value_dim * (hparams.ssm_d_conv - 1), 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_gate_weight)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + 2*value_dim);
                    return {{key_dim, 2 + 2*head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_beta_alpha_cat)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*n_v_heads);
                    return {{n_k_heads, 2*head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if ((hparams.is_recr(il) && std::regex_match(tensor_name, pattern_attn_gate_weight)) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp(il);
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd_q = hparams.n_head(il) * hparams.n_embd_head_k(il);
            const int64_t n_embd_k = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(tensor->ne[axis] == n_embd_q + n_embd_k + n_embd_v);
            if (n_embd_k == n_embd_v) {
                return {{n_embd_q, 1}, {n_embd_k, 2}};
            }
            // uneven K/V head sizes (e.g. MiMo d_k=192 d_v=128): split K and V as separate
            // segments so each device gets whole heads of both
            return {{n_embd_q, 1}, {n_embd_k, 1}, {n_embd_v, 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp(il);
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (is_glm5_next && hparams.is_recr(il)) {
            // KDA, each device holds whole heads
            const int64_t head_dim        = hparams.n_embd_head_kda;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight) ||
                    std::regex_match(tensor_name, pattern_kda_conv1d) || std::regex_match(tensor_name, pattern_kda_gate_b) ||
                    std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_attn_out_weight)) {
                return {granularity_qkv};
            }
            if (std::regex_match(tensor_name, pattern_ssm_a) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return {granularity_qkv / head_dim};
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return {granularity_qkv * head_dim};
            }
        } else if (is_glm5_next) {
            // MLA, each device holds whole heads
            // an even number of heads per device, the HIP tile FA kernel for D=512 needs an even GQA ratio
            const int64_t granularity_head = std::lcm(std::lcm(hparams.n_embd_head_v_mla(), blck_size) / hparams.n_embd_head_v_mla(), int64_t(2));
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return {granularity_head * hparams.n_embd_head_k_mla()};
            }
            if (std::regex_match(tensor_name, pattern_attn_kv_b_weight)) {
                return {granularity_head};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                return {granularity_head * hparams.n_embd_head_v_mla()};
            }
        } else if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight) ||
                    std::regex_match(tensor_name, pattern_qkv_gate_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                    std::regex_match(tensor_name, pattern_ssm_beta_alpha_cat)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            const int64_t granularity_q    = std::lcm(n_embd_q, blck_size_perf);
            const int64_t granularity_head = granularity_q / hparams.n_embd_head_k(il); // for tensors with one value per head
            if (std::regex_match(tensor_name, pattern_attn_gate_bias)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_head};
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                if (is_dsv4) {
                    return {hparams.n_head(il) / hparams.dsv4_o_group_count};
                }
                return {granularity_head};
            }

            if (is_dsv4) {
                if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the grouped output projection requires each device to hold whole groups of heads
                    const int64_t n_head_group = hparams.n_head(il) / hparams.dsv4_o_group_count;
                    return {n_head_group * hparams.n_embd_head_k(il)};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    return {1};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the boundaries must align with wo_a's per-group split, so quant blocks must not straddle groups
                    GGML_ASSERT(hparams.dsv4_o_lora_rank % blck_size == 0);
                    return {hparams.dsv4_o_lora_rank};
                }
            }
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE || ud->model->arch == LLM_ARCH_CLEF || ud->model->arch == LLM_ARCH_CLOVER_MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_head * hparams.n_embd_head_v(il)};
            }
            if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
                GGML_ASSERT(segments.size() == 1);
                if (tensor->ne[1] == hparams.n_head(il)) {
                    return {granularity_head};
                }
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            // the V head size can differ from the K head size (e.g. MiMo d_k=192 d_v=128):
            // align V tensors to whole V heads at the same head-index scale as Q and K so all
            // three stay in lockstep per device
            const int64_t granularity_v  = (granularity_kv / hparams.n_embd_head_k(il)) * hparams.n_embd_head_v(il);
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_v_exps_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                const bool is_v = tensor_name.find("attn_v") != std::string::npos || tensor_name.find("cache_v") != std::string::npos;
                return {is_v ? granularity_v : granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                // fused full attention layers need Q gate tensors handled like above:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE || ud->model->arch == LLM_ARCH_CLEF || ud->model->arch == LLM_ARCH_CLOVER_MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf), granularity_kv};
                }
                if (segments.size() == 3) {
                    // uneven K/V head sizes: per-segment granularity, V aligned to whole V heads
                    return {granularity_q, granularity_kv, granularity_v};
                }
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_weight) ||
                std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        if (ud->model->arch == LLM_ARCH_STEP35 || ud->model->arch == LLM_ARCH_LAGUNA) {
            GGML_ASSERT(segments.size() == 1);
            if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
                return {hparams.n_head(il) / hparams.n_head_kv(il)};
            }
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            GGML_ASSERT(tc.tensor_axis_0 != tensor);
            const ggml_backend_meta_split_state source_split_state = llama_meta_device_get_split_state(tc.tensor_axis_0, userdata);
            GGML_ASSERT(source_split_state.axis >= 0 && source_split_state.axis < GGML_MAX_DIMS);
            for (size_t j = 0; j < ud->n_devices; j++) {
                for (size_t is = 0; is < source_split_state.n_segments; is++) {
                    split_state.ne[j] += source_split_state.ne[is*ud->n_devices + j] * source_split_state.nr[is];
                }
            }
        }
    }
    return split_state;
    GGML_UNUSED(userdata);
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_230M:          return "230M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_10B:           return "10B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_456B:          return "456B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_816B_A42B:     return "816B.A42B";
        case LLM_TYPE_78B_A3_5B:     return "78B.A3.5B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_1B_A400M:      return "1B.A400M";
        case LLM_TYPE_3B_A800M:      return "3B.A800M";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_7_9B_A1_3B:    return "7.9B.A1.3B";
        case LLM_TYPE_12B_A2_5B:     return "12B.A2.5B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_32B_A9B:       return "32B.A9B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_75B_A9B:       return "75B.A9B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_A3B:           return "A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_118B_A8B:      return "118B.A8B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_124B_A5_1B:    return "124B.A5.1B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_428B_A23B:     return "428B.A23B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_288B_A19B:     return "288B.A19B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_685B_A37B:     return "685B.A37B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_2_8T_A50B:     return "2.8T.A50B";
        case LLM_TYPE_320B_A18B:     return "320B.A18B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

static const std::map<llama_rope_section_order, const char *> LLAMA_ROPE_SECTION_ORDERS = {
    { LLAMA_ROPE_SECTION_ORDER_TYXZ, "tyxz" },
    { LLAMA_ROPE_SECTION_ORDER_ZYXT, "zyxt" },
};

std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order) {
    return LLAMA_ROPE_SECTION_ORDERS.at(rope_section_order);
}

static llama_rope_section_order llama_rope_section_order_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SECTION_ORDERS) {
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED;
}

// Maps GGUF activation names to the FFN op type used by the graph builders.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",              LLM_FFN_GEGLU_ERF },
    { "gelu_python",       LLM_FFN_GEGLU_ERF },
    { "gelu_pytorch_tanh", LLM_FFN_GEGLU     },
    { "gelu_new",          LLM_FFN_GEGLU     },
    { "gelu_fast",         LLM_FFN_GEGLU     },
    { "gelu_accurate",     LLM_FFN_GEGLU     },
    { "gelu_python_tanh",  LLM_FFN_GEGLU     },
    { "geglu",             LLM_FFN_GEGLU     },
    { "silu",              LLM_FFN_SWIGLU    },
    { "swish",             LLM_FFN_SWIGLU    },
    { "swiglu",            LLM_FFN_SWIGLU    },
    { "relu",              LLM_FFN_RELU      },
    { "reglu",             LLM_FFN_REGLU     },
};

// transformers names, "gelu" is the exact (erf) variant
static const std::map<std::string, ggml_unary_op> LLM_CLS_ACT_TYPES_FROM_STRING = {
    { "gelu", GGML_UNARY_OP_GELU_ERF },
    { "silu", GGML_UNARY_OP_SILU     },
    { "tanh", GGML_UNARY_OP_TANH     },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> SYCL mapped (a load that maps its files) -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices,
                                      bool                              use_extra_bufts,
                                      bool                              no_host,
                                      bool                              sycl_maps_files) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    const size_t backend_dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < backend_dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        // A load that maps its files serves a host-resident weight from the file's pages in place, ahead of the copy a
        // pinned host buffer would take (design 5efe §4.3).
        if (sycl_maps_files) {
            const auto hooks = llama_model_sycl_hooks();
            for (const auto & dev : devices) {
                if (hooks.mapped_buffer_type_for_device && llama_model_dev_is_sycl(dev.dev)) {
                    ggml_backend_buffer_type_t buft = hooks.mapped_buffer_type_for_device(dev.dev);
                    if (buft) {
                        buft_list.emplace_back(dev.dev, buft);
                    }
                    break;
                }
            }
        }
#else
        GGML_UNUSED(sycl_maps_files);
#endif
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < backend_dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        } else {
            throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(dev)));
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    const auto sycl_hooks = llama_model_sycl_hooks();
    if (llama_model_sycl_hooks_enabled(sycl_hooks) && dev != nullptr &&
        ggml_backend_dev_backend_reg(dev) == sycl_hooks.reg && sycl_hooks.weights_evictable()) {
        ggml_backend_buffer_type_t host_buft = sycl_hooks.host_buffer_type_for_device(dev);
        if (host_buft != nullptr) {
            buft_list.emplace_back(dev, host_buft);
        }
    }
#endif

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;

    std::vector<float> tensor_split_owned;

    // The STORAGE tables' tensors, in no buffer and never in ctxs_bufs (memory_breakdown expects a buffer for every
    // context there), and the handle each table is read through. The handles go with pimpl: when the model has a
    // token, ~llama_model destroys pimpl before its unload hook runs, so every open table closes while its plan still
    // exists; a model that never got a token never opened a table, so it has nothing to close.
    ggml_context_ptr                                 storage_ctx;
    std::vector<std::unique_ptr<llama_lazy_storage>> lazy_storage;
};

bool llama_prec_policy::apply(ggml_tensor * res) const {
    if (!res || !res->src[0]) {
        return false;
    }

    bool applied = false;

    const auto it_acc = prec_acc.find(res->src[0]);
    if (it_acc != prec_acc.end()) {
        applied |= ggml_prec_set_acc(res, it_acc->second);
    }

    const auto it_src1 = prec_src1.find(res->src[0]);
    if (it_src1 != prec_src1.end()) {
        applied |= ggml_prec_set_src(res, it_src1->second, 1);
    }

    return applied;
}

void llama_prec_policy::load(llama_model_loader & ml, const llama_model & model) {
    std::vector<std::string> tensor_names;
    if (!ml.get_arr(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names, false)) {
        return;
    }

    const gguf_context * ctx = ml.metadata;
    const std::string key = ml.llm_kv(LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4);
    const int kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(format("%s must be a bool array", key.c_str()));
    }

    const size_t n_values = gguf_get_arr_n(ctx, kid);
    if (n_values != tensor_names.size()) {
        throw std::runtime_error(format(
            "%s tensor/value length mismatch (%zu vs %zu)",
            key.c_str(), tensor_names.size(), n_values));
    }

    // tensors that can not use 4-bit activations, keep src1 at higher precision
    const int8_t * values = (const int8_t *) gguf_get_arr_data(ctx, kid);
    std::unordered_set<std::string> want;
    for (size_t i = 0; i < n_values; ++i) {
        if (values[i] == 0) {
            want.insert(tensor_names[i]);
        }
    }

    // resolve names to tensor pointers
    for (const auto & [name, w] : model.tensors_by_name) {
        if (want.count(name)) {
            prec_src1.emplace(w, GGML_PREC_Q8);
        }
    }
}

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    if (params.tensor_split != nullptr) {
        // llama_model_params stores tensor_split as a borrowed pointer, but the model
        // may need it later for tensor-parallel KV-cache split metadata.
        pimpl->tensor_split_owned.assign(params.tensor_split, params.tensor_split + llama_max_devices());
        this->params.tensor_split = pimpl->tensor_split_owned.data();
    }
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    // Model teardown is the SYCL backend's only chance to reclaim this model's
    // cached weights (llama.cpp-0qlw).  Nothing else can: preload pins every
    // dense weight and eviction skips pinned entries, so without this hook a
    // dead model's VRAM was reachable only by another model's load boundary --
    // which is exactly the sweep that could not tell a dead model's weights from
    // a live one's idle ones.
    //
    // pimpl owns the ggml contexts and buffers, so it must be destroyed FIRST:
    // the hook reports any surviving weight lease as a leak, and this model's own
    // tensors hold leases until their extras go.
    if (sycl_model_token.model_id != 0) {
        const ggml_sycl_model_token token = { sycl_model_token.model_id, sycl_model_token.load_txn_id,
                                              sycl_model_token.slot, sycl_model_token.slot_generation };
        pimpl.reset();
        const auto hooks = llama_model_sycl_hooks();
        if (llama_model_sycl_hooks_enabled(hooks)) {
            const auto rc = hooks.unload(token);
            if (rc == GGML_SYCL_LIFECYCLE_OK || rc == GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD) {
                sycl_model_token = {};
            } else {
                const auto queued = hooks.quarantine(token);
                LLAMA_LOG_ERROR(
                    "SYCL model teardown failed: model=%llu txn=%llu slot=%u generation=%llu result=%d quarantine=%d\n",
                    (unsigned long long) token.model_id, (unsigned long long) token.load_txn_id, token.slot,
                    (unsigned long long) token.slot_generation, (int) rc, (int) queued);
            }
        }
    }
#endif
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            // only the string arrays listed here are kept, see llama_model_meta_arr_str()
            static const char * const kept_prefixes[] = { "tokenizer.tts_dict" };
            const std::string arr_name = gguf_get_key(ctx, i);
            const bool is_kept = std::any_of(std::begin(kept_prefixes), std::end(kept_prefixes), [&](const char * prefix) {
                return arr_name.rfind(prefix, 0) == 0;
            });
            if (is_kept && gguf_get_arr_type(ctx, i) == GGUF_TYPE_STRING) {
                auto & arr = gguf_kv_arr[arr_name];
                const size_t n = gguf_get_arr_n(ctx, i);
                arr.reserve(n);
                for (size_t j = 0; j < n; j++) {
                    arr.emplace_back(gguf_get_arr_str(ctx, i, j));
                }
            }
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    load_hparams_hadamard(ml);

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_CLASSIFIER_POOLING_TYPE, hparams.pooling_type_cls, false);
    ml.get_key(LLM_KV_CLASSIFIER_FREE_GUIDANCE, hparams.classifier_free_guidance, false);
    std::string act_cls;
    if (ml.get_key(LLM_KV_CLASSIFIER_ACTIVATION, act_cls, false)) {
        const auto it = LLM_CLS_ACT_TYPES_FROM_STRING.find(act_cls);
        GGML_ASSERT(it != LLM_CLS_ACT_TYPES_FROM_STRING.end() && "unsupported classifier activation");
        hparams.act_cls = it->second;
    }
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,    hparams.n_layer_nextn,   false);
    GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert = 0;
            std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);

        GGML_ASSERT(hparams.posnet.n_layer   <= hparams.n_layer_all);
        GGML_ASSERT(hparams.convnext.n_layer <= hparams.n_layer_all);
    }

    // models may route a different number of experts per layer, so validate the maximum
    uint32_t n_expert_used_max = hparams.n_expert_used_max();

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(n_expert_used_max == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),       hparams.n_head_arr.end(),       0);
    std::fill(hparams.n_head_kv_arr.begin(),    hparams.n_head_kv_arr.end(),    0);
    std::fill(hparams.n_ff_arr.begin(),         hparams.n_ff_arr.end(),         0);
    std::fill(hparams.n_ff_exp_arr.begin(),     hparams.n_ff_exp_arr.end(),     0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.rope_pattern.begin(),  hparams.rope_pattern.end(), 1);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 0);
    std::iota(hparams.kv_source_impl.begin(), hparams.kv_source_impl.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(),    hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(),     hparams.xielu_eps.end(), 0.0f);

    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);
    std::fill(hparams.swiglu_clamp.begin(),       hparams.swiglu_clamp.end(),       0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer_all, false);

    // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer_all, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    std::string rope_section_order("tyxz");
    ml.get_key(LLM_KV_ROPE_SECTION_ORDER, rope_section_order, false);
    hparams.rope_section_order = llama_rope_section_order_from_string(rope_section_order);
    if (hparams.rope_section_order == LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED) {
        throw std::runtime_error("unknown rope section order: " + rope_section_order);
    }

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
            if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // for decision models
    ml.get_arr(LLM_KV_DECISION_LABELS, decision_labels, false);

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    pimpl->ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);

    if (hparams.rope_section_order != LLAMA_ROPE_SECTION_ORDER_TYXZ && hparams.n_pos_per_embd() != 4) {
        throw std::runtime_error("rope section order " + llama_rope_section_order_name(hparams.rope_section_order) + " requires M-RoPE");
    }
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    // the model's weights are SYCL-owned and the SYCL hooks resolve: the fact the early
    // plan is gated on, not the process-level "a SYCL backend is registered" (a `-ngl 0`
    // load in a SYCL build registers the hooks without taking the weights)
    bool lazy_taker = false;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    const bool sycl_layer_assignment =
        this->n_gpu_layers() > 0 && std::any_of(devices.begin(), devices.end(), [](const llama_device & device) {
            return llama_model_dev_is_sycl(device.dev);
        });
    bool sycl_tensor_override = false;
    if (params.tensor_buft_overrides) {
        for (const auto * override = params.tensor_buft_overrides; override->pattern != nullptr; ++override) {
            if (llama_model_buft_is_sycl(override->buft) || llama_model_buft_backend_is_sycl(override->buft)) {
                sycl_tensor_override = true;
                break;
            }
        }
    }
    const bool                     resolved_sycl_weight_owner = sycl_layer_assignment || sycl_tensor_override;
    llama_model_sycl_loading_guard sycl_model_loading_guard(resolved_sycl_weight_owner, &sycl_model_token);
    lazy_taker = sycl_model_loading_guard.active;
#endif

    const auto & split_mode   = params.split_mode;
    const bool use_mlock      = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.use_mmap = false;
                break;
            }
        }
    }

    // the lazy mode as requested, before the fallback below: the SYCL storage tier reads
    // through its own O_DIRECT fd and must not be disengaged by a device that cannot mmap
    const enum llama_lazy_mode lazy_mode_pre = ml.lazy.mode;

    // resolve AUTO: a device without mmap support (e.g. an iGPU) falls back to OFF unless mmap is in use; see #28160,
    // #31943. The exception is a device whose backend owns placement (SYCL): its planner decides where
    // each weight lives and reads the tensors itself, so lazy reads stay available there
    // (llama_lazy_auto_enabled, src/llama-lazy-mode.h).
    if (ml.lazy.mode == LLAMA_LAZY_MODE_AUTO && !ml.use_mmap) {
        std::vector<llama_lazy_device_caps> lazy_caps;
        lazy_caps.reserve(devices.size());
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
            const bool planner_owns_placement = llama_model_dev_is_sycl(dev.dev);
#else
            const bool planner_owns_placement = false;
#endif
            lazy_caps.push_back({ props.caps.mmap_support, planner_owns_placement });
        }
        if (!llama_lazy_auto_enabled(lazy_caps.data(), lazy_caps.size())) {
            ml.lazy.mode = LLAMA_LAZY_MODE_OFF;
        }
    }

    llama_model_prefill_lazy_record(ml, arch, hparams, lazy_mode_pre, lazy_taker);

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    if (sycl_model_loading_guard.active) {
        llama_model_sycl_compute_early_plan(ml, *this, use_mlock, __func__);
    }
#endif

    const char * load_mode_name = params.load_mode == LLAMA_LOAD_MODE_AUTO
        ? llama_load_mode_name(ml.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE)
        : llama_load_mode_name(params.load_mode);

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n",
        __func__, load_mode_name);

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list =
        make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host, ml.sycl_load_maps_files());
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] = split_sum != 0.0f ? splits[i] / split_sum : float(i + 1) / n_devices();
    }

    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert = hparams.n_expert;

        if (n_expert > 0 && hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            if (hparams.has_kv(i)) {
                layer.k_cache_s = create_tensor(tn(LLM_TENSOR_ATTN_K, "k_scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.v_cache_s = create_tensor(tn(LLM_TENSOR_ATTN_V, "v_scale", i), {1}, TENSOR_NOT_REQUIRED);

                if ((layer.k_cache_s && layer.k_cache_s->type != GGML_TYPE_F32) ||
                    (layer.v_cache_s && layer.v_cache_s->type != GGML_TYPE_F32)) {
                    throw std::runtime_error(format("KV cache scales for layer %d must be F32", i));
                }
            }

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
                if (layer.ffn_up_exps_s && layer.ffn_gate_up_exps) {
                    throw std::runtime_error(format("ffn_up_exps.scale and ffn_gate_up_exps.weight both exist for layer %d", i));
                }
            }
            if (!layer.ffn_up_exps_s && layer.ffn_gate_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }

        // The loop above asks for a sidecar only when its weight was loaded. The MTP block's weights are skipped
        // (by the caller's choice, or always by some architectures), so their pointers are null and its sidecars,
        // which are in the file, would never be counted and done_getting_tensors() would find the count short.
        // Skip them with their weights: a sidecar of a loaded weight is never asked for here. A virtual load has no
        // file and ignores TENSOR_SKIP, so it would build an orphan tensor for each ask; TENSOR_SKIP_IF_VIRTUAL
        // makes it return nullptr there, as before.
        const auto skip_scales = [&](int il, llm_tensor tensor, const ggml_tensor * weight, int64_t n) {
            if (weight == nullptr) {
                const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP | TENSOR_SKIP_IF_VIRTUAL;
                create_tensor(tn(tensor, "scale",       il), {n}, skip);
                create_tensor(tn(tensor, "input_scale", il), {n}, skip);
            }
        };
        for (int i = hparams.n_layer(); i < n_layer_all; ++i) {
            const auto & layer = layers[i];

            skip_scales(i, LLM_TENSOR_ATTN_Q,                 layer.wq,                     1);
            skip_scales(i, LLM_TENSOR_ATTN_K,                 layer.wk,                     1);
            skip_scales(i, LLM_TENSOR_ATTN_V,                 layer.wv,                     1);
            skip_scales(i, LLM_TENSOR_ATTN_OUT,               layer.wo,                     1);
            skip_scales(i, LLM_TENSOR_ATTN_QKV,               layer.wqkv,                   1);
            skip_scales(i, LLM_TENSOR_ATTN_GATE,              layer.wqkv_gate,              1);
            skip_scales(i, LLM_TENSOR_FFN_GATE,               layer.ffn_gate,               1);
            skip_scales(i, LLM_TENSOR_FFN_DOWN,               layer.ffn_down,               1);
            skip_scales(i, LLM_TENSOR_FFN_UP,                 layer.ffn_up,                 1);
            skip_scales(i, LLM_TENSOR_FFN_GATE_SHEXP,         layer.ffn_gate_shexp,         1);
            skip_scales(i, LLM_TENSOR_FFN_DOWN_SHEXP,         layer.ffn_down_shexp,         1);
            skip_scales(i, LLM_TENSOR_FFN_UP_SHEXP,           layer.ffn_up_shexp,           1);
            skip_scales(i, LLM_TENSOR_FFN_GATE_EXPS,          layer.ffn_gate_exps,          n_expert);
            skip_scales(i, LLM_TENSOR_FFN_DOWN_EXPS,          layer.ffn_down_exps,          n_expert);
            skip_scales(i, LLM_TENSOR_FFN_UP_EXPS,            layer.ffn_up_exps,            n_expert);
            skip_scales(i, LLM_TENSOR_FFN_GATE_UP_EXPS,       layer.ffn_gate_up_exps,       n_expert);
            skip_scales(i, LLM_TENSOR_SSM_IN,                 layer.ssm_in,                 1);
            skip_scales(i, LLM_TENSOR_SSM_OUT,                layer.ssm_out,                1);
            skip_scales(i, LLM_TENSOR_SSM_ALPHA,              layer.ssm_alpha,              1);
            skip_scales(i, LLM_TENSOR_SSM_BETA,               layer.ssm_beta,               1);
            skip_scales(i, LLM_TENSOR_NEXTN_EH_PROJ,          layer.nextn.eh_proj,          1);
            skip_scales(i, LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, layer.nextn.shared_head_head, 1);
        }

        // output scales
        if (output) {
            // weight scale
            if (!output_s && (ml.get_weight(tn(LLM_TENSOR_OUTPUT, "scale").str().c_str()) ||
                              gguf_find_tensor(ml.metadata, tn(LLM_TENSOR_OUTPUT, "scale").str().c_str()) >= 0)) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s && (ml.get_weight(tn(LLM_TENSOR_OUTPUT, "input_scale").str().c_str()) ||
                                 gguf_find_tensor(ml.metadata, tn(LLM_TENSOR_OUTPUT, "input_scale").str().c_str()) >= 0)) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        post_arch_tensors(ml);
    }

    ml.done_getting_tensors();

    if (per_layer_tok_embd && ml.lazy.has(per_layer_tok_embd)) {
        LLAMA_LOG_INFO("%s: enabling prefetch for '%s'\n", __func__, per_layer_tok_embd->name);

        can_prefetch.insert(per_layer_tok_embd);
    }

    if (output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            (output_s || output_in_s)) {
        throw std::runtime_error("output scales require a separate output weight");
    }
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // the STORAGE tables, before the measure below builds a graph: their consumer builds its F32 input from them, and
    // the table tensor itself must never enter a graph, tensors_by_name or ctxs_bufs
    pimpl->storage_ctx = std::move(ml.storage_ctx);
    // Driven by the record's STORAGE rows, not by whether storage_ctx exists: a STORAGE row whose tensor was never
    // created then aborts in llama_lazy_range_storage_tables() instead of leaving the table silently unread.
    const std::vector<llama_lazy_storage_table> storage_tables =
        llama_lazy_range_storage_tables(ml.lazy_record, [&](const std::string & name) -> const ggml_tensor * {
            return pimpl->storage_ctx ? ggml_get_tensor(pimpl->storage_ctx.get(), name.c_str()) : nullptr;
        });
    // a load with no STORAGE table does not touch the backend registry
    const llama_lazy_storage_procs procs =
        storage_tables.empty() ? llama_lazy_storage_procs{} : llama_lazy_storage_sycl_procs();
    for (const llama_lazy_storage_table & t : storage_tables) {
        // a file loaded from an open handle has no path, and the table is read through its own O_DIRECT
        // descriptor, which needs one
        if (t.value.file_idx >= ml.file_paths.size() || ml.file_paths[t.value.file_idx].empty()) {
            throw std::runtime_error(
                format("STORAGE table %s: its file was opened from a handle and has no path "
                       "to read the table from; load the model from a file path",
                       t.name.c_str()));
        }
        pimpl->lazy_storage.push_back(std::make_unique<llama_lazy_storage>(t, ml.file_paths[t.value.file_idx], procs));
    }

    // map each weight to its "<name>.scale" tensor: shape {1} for a 2D weight, {n_expert} for a 3D weight
    {
        auto ends_with = [](const std::string & str, const std::string & suffix) {
            return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        std::unordered_map<std::string, ggml_tensor *> weights;
        for (const auto & [name, cur] : tensors_by_name) {
            if (ends_with(name, ".weight")) {
                weights.emplace(name.substr(0, name.size() - 7), cur);
            }
        }
        for (const auto & [name, s] : tensors_by_name) {
            if (!ends_with(name, ".scale")) {
                continue;
            }
            const auto it = weights.find(name.substr(0, name.size() - 6));
            if (it == weights.end()) {
                continue;
            }
            ggml_tensor * w = it->second;
            const bool ok = w->ne[3] == 1 && ggml_nelements(s) == w->ne[2] && s->ne[0] == w->ne[2];
            if (ok) {
                w_scales.emplace(w, s);
            } else {
                LLAMA_LOG_DEBUG("%s: '%s' is not a weight scale for '%s', skipping\n", __func__, name.c_str(), w->name);
            }
        }
    }

    // the reference computes the router in F32: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L565
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS || hparams.expert_router_f32) {
        for (const auto & layer : layers) {
            prec_policy.set_acc(layer.ffn_gate_inp, GGML_PREC_F32);
        }
    }

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    bool has_sycl_weight_buft = false;
    for (const auto & [ctx_key, _] : ml.ctx_map) {
        if (llama_model_buft_is_sycl(ctx_key.buft) || llama_model_buft_backend_is_sycl(ctx_key.buft)) {
            has_sycl_weight_buft = true;
            break;
        }
    }
    const bool sycl_model_backend = sycl_model_loading_guard.txn.id != 0 && has_sycl_weight_buft;
    if (sycl_model_backend) {
        // (a) the probe measure: the probe placement's compute term C-hat, before the late plan packs the weights,
        // handed to the backend as a planned RUNTIME term so the pack leaves the compute buffer its room. It is
        // measured at the context and micro-batch the caller told the load (the training context and the auto
        // ladder's bottom rung where it told none); the admitted measure below runs at the same shape. A dense
        // model's auto micro-batch climbs above the bottom rung into whatever room is left; that part is not reserved.
        const llama_load_measure_shape               measure_shape = llama_model_load_measure_shape(*this);
        const std::vector<llama_measure_dummy_entry> probe_weights = llama_model_measure_weights(ml);

        const int64_t                 probe_t0 = ggml_time_us();
        const llama_load_probe_result probe =
            llama_load_probe_bound(*this, measure_shape, sycl_model_loading_guard.txn, probe_weights);
        const int64_t probe_ms = (ggml_time_us() - probe_t0) / 1000;
        if (!probe.unsupported.empty()) {
            // not a refusal of the load: the probe cannot be measured (a model the measure cannot walk, or no plan
            // staged for the probe), so no compute term is reserved; a late plan, where one is staged, still places
            // the weights
            LLAMA_LOG_WARN("%s: probe compute-slot measure skipped, the compute slot is not planned at load: %s\n",
                           __func__, probe.unsupported.c_str());
        }
        if (!probe.refusal.empty()) {
            throw std::runtime_error(probe.refusal);
        }

        llama_model_sycl_set_late_inventory(ml, *this, use_mlock, __func__);

        // (b) the admitted measure, llama.cpp-p6i0: c(P) at the placement the late plan just packed, before the
        // dev_layer sync, so the late check below compares the final placement with it. Refused when it outgrew
        // the probe bound the pack reserved; recorded otherwise. Only a model the probe measured gets here.
        if (probe.measured) {
            const std::vector<llama_measure_dummy_entry> admitted_weights = llama_model_measure_weights(ml);

            const int64_t                     admitted_t0 = ggml_time_us();
            const llama_admitted_check_result admitted =
                llama_load_admitted_check(*this, measure_shape, sycl_model_loading_guard.txn, admitted_weights, probe);
            const int64_t admitted_ms = (ggml_time_us() - admitted_t0) / 1000;
            if (!admitted.unsupported.empty()) {
                LLAMA_LOG_WARN("%s: admitted compute-slot measure skipped, the load continues: %s\n", __func__,
                               admitted.unsupported.c_str());
            }
            if (!admitted.refusal.empty()) {
                throw std::runtime_error(admitted.refusal);
            }
            for (const llama_admitted_term & t : admitted.terms) {
                LLAMA_LOG_WARN(
                    "%s: [LOAD-PLAN] compute slot on device %d: probe bound %.1f MiB %s, admitted %.1f MiB (%zu of "
                    "%zu recorded); measured at n_ctx %u ubatch %u, the context and micro-batch this load was told "
                    "(the training context and ubatch %u where it was told none); measure %lld ms probe, %lld ms "
                    "admitted\n",
                    __func__, (int) t.device, t.probe_term / 1024.0 / 1024.0,
                    t.reserved ? "reserved in RUNTIME" : "NOT reserved (the backend said why), not compared",
                    t.admitted_term / 1024.0 / 1024.0, admitted.n_recorded, admitted.terms.size(), admitted.n_ctx,
                    admitted.n_ubatch, llama_auto_ubatch_ladder[0], (long long) probe_ms, (long long) admitted_ms);
                if (t.kv_excess != 0) {
                    LLAMA_LOG_WARN(
                        "%s: [LOAD-PLAN] compute slot on device %d: admitted %.1f MiB exceeds the probe bound by %.1f "
                        "MiB because the KV residency moved between the measures (probe %u device / %u host / %u "
                        "CPU layers, admitted %u / %u / %u); the excess is not reserved and can land outside the "
                        "RUNTIME zone\n",
                        __func__, (int) t.device, t.admitted_term / 1024.0 / 1024.0, t.kv_excess / 1024.0 / 1024.0,
                        admitted.probe_kv.n_device, admitted.probe_kv.n_host, admitted.probe_kv.n_cpu,
                        admitted.admitted_kv.n_device, admitted.admitted_kv.n_host, admitted.admitted_kv.n_cpu);
                }
                if (t.state_bytes != 0) {
                    // the state term, on its own line: the compute line above never includes it
                    LLAMA_LOG_WARN(
                        "%s: [LOAD-PLAN] state term on device %d: %.1f MiB of recurrent state %s, %s; measured at "
                        "n_seq_max 1, so a context with more sequences allocates more state than this holds\n",
                        __func__, (int) t.device, t.state_bytes / 1024.0 / 1024.0,
                        t.state_reserved ? "reserved in RUNTIME as its own term" :
                                           "NOT reserved (the backend said why)",
                        t.state_recorded ? "recorded for the late check" : "not recorded");
                }
            }
        }

        // llama.cpp-30h4: dev_layer(il)/dev_output() go stale the moment a
        // layer is tiered to host by any route OTHER than -ngl. get_layer_buft_list()
        // above only ever marks the CPU-side prefix [0, i_gpu_start) -- it has no
        // way to see a decision the SYCL planner makes later, inside the LATE
        // replan that llama_model_sycl_set_late_inventory() just triggered
        // (budget-percent tiering, -ot, -nkvo, ...). Two real consumers then act
        // on the stale value as though the weight were still device-resident:
        // the FA executor placement read at llama-context.cpp:1236 and the KV
        // buft selection at llama-kv-cache.cpp:57-75/287.
        //
        // [REVISED, llama.cpp-30h4 Gate 0 retry] This originally asked the
        // planner -- ggml_backend_sycl_planned_target_device(), the
        // weight-side mirror of the KV-side
        // ggml_backend_sycl_kv_layer_on_device_from_dev() -- which device
        // each layer's tensors landed on, name-indexed against the plan's own
        // dense-entry table. That query is void for exactly the case this fix
        // exists for: a layer the planner tiers to host by DEFAULT (since
        // llama.cpp-ir18) gets create_tensor'd straight onto CPU_REPACK, a
        // ggml-cpu buft that is not SYCL-backend-owned and is therefore
        // excluded from the LATE inventory (llama_model_sycl_set_late_inventory()
        // above filters ml.ctx_map on !llama_model_buft_backend_is_sycl()).
        // Its tensor names never entered the plan's dense-entry index at all,
        // so the query returned GGML_SYCL_PLANNED_NO_PLAN for every one of
        // them -- never -1 -- and the -1-only check, correct against the
        // query's own contract, was structurally unreachable. Measured: 0
        // corrections in every arm, including default-on (c-pj9b).
        //
        // Ask the artifact instead of the planner: by this point every
        // tensor in this model has already been create_tensor'd into a
        // specific ggml_context, and ml.ctx_map buckets each context by the
        // buft it was actually given -- that assignment is a settled fact,
        // not a plan to look up. Walk it directly rather than asking a second
        // source that has no entry for a tensor already excluded from its
        // own bookkeeping.
        if (llama_model_sycl_dev_layer_sync_enabled()) {
            // llama_model_buft_is_sycl()/llama_model_buft_backend_is_sycl()
            // (the same OR this file already uses just above to decide
            // has_sycl_weight_buft) test the buft's own name and its
            // device's backend registration -- both TRUE for SYCL0/SYCL1
            // (device-resident) and for SYCL_Host (host-pinned staging
            // memory the SYCL backend still owns, where the GPU still
            // executes the op -- NOT a host-execution marker). Only a truly
            // foreign, ggml-cpu-owned buft such as CPU_REPACK fails both,
            // which is exactly the "this layer now runs on the host" signal
            // this fix needs, and it cannot conflate SYCL_Host with that
            // because SYCL_Host passes the same test a device buft does.
            //
            // Skip "_exps" tensor names specifically: MoE per-expert weights
            // are placed by a separate per-expert decision (the live
            // CpuExpertPool path), not by the per-LAYER decision dev_layer(il)
            // records -- a layer with GPU-resident attention/dense weights
            // but host-resident experts must not be marked host here, or
            // this would silently break FA/KV routing on every MoE model
            // that legitimately spills experts to host.
            std::vector<int> host_layer_ids;
            ggml_tensor *    host_output_tensor = nullptr;
            for (const auto & it : ml.ctx_map) {
                ggml_backend_buffer_type_t buft = it.first.buft;
                if (llama_model_buft_is_sycl(buft) || llama_model_buft_backend_is_sycl(buft)) {
                    continue;  // still SYCL-owned (device or SYCL_Host) -- GPU still executes this
                }
                ggml_context * ctx = it.second.get();
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    if (output != nullptr && t == output) {
                        host_output_tensor = t;
                        continue;
                    }
                    const char * name = ggml_get_name(t);
                    if (name == nullptr || std::strncmp(name, "blk.", 4) != 0 ||
                        std::strstr(name, "_exps") != nullptr) {
                        continue;
                    }
                    const std::string name_str(name);
                    const size_t      dot = name_str.find('.', 4);
                    if (dot == std::string::npos) {
                        continue;
                    }
                    int il = -1;
                    try {
                        il = std::stoi(name_str.substr(4, dot - 4));
                    } catch (const std::exception &) {
                        continue;
                    }
                    if (il >= 0 && il < n_layer_all) {
                        host_layer_ids.push_back(il);
                    }
                }
            }

            int corrected_layers = 0;
            for (int il : host_layer_ids) {
                if (pimpl->dev_layer[il].dev != cpu_dev) {
                    // Mutate .dev ONLY, never .buft_list. This syncs the device
                    // record the scheduler and KV path actually read
                    // (llama-context.cpp:1236, llama-kv-cache.cpp:57-75/287) to
                    // where the weight really landed; it deliberately does not
                    // touch the buft CANDIDATE list, because the weight is
                    // already materialized and .buft_list is no longer the
                    // authority over where it lives -- create_tensor's own
                    // buft selection (llama-model-loader.cpp) already ran for
                    // every tensor in this model, above, and the tensor's own
                    // buft (just read from ml.ctx_map, right here) already
                    // correctly reflects where the bytes really are.
                    // Rewriting .buft_list here would not change an
                    // already-created tensor's buffer -- it would just move
                    // today's disagreement from (dev vs buft_list) to
                    // (buft_list vs actual allocation), which is worse: it
                    // would misinform the one other reader of buft_list
                    // (control-vector attachment) about where the tensor's
                    // bytes really are. So this deliberately leaves a second,
                    // narrower one-fact-two-sources site: dev_layer[il].dev now
                    // says CPU while .buft_list still names the original GPU
                    // list. Re-check this if .buft_list grows a new reader.
                    pimpl->dev_layer[il].dev = cpu_dev;
                    corrected_layers++;
                }
            }

            // dev_output: output.weight may not exist under that literal name --
            // tied embeddings reuse tok_embd's tensor and name -- so this
            // matches the tensor OBJECT actually created (above, inside the
            // ctx_map walk) rather than reconstructing a name.
            bool corrected_output = false;
            if (host_output_tensor != nullptr && pimpl->dev_output.dev != cpu_dev) {
                pimpl->dev_output.dev = cpu_dev;
                corrected_output      = true;
            }

            // WARN, not INFO/DEBUG: dev_layer's own per-layer assignment log
            // (get_layer_buft_list() above) is LLAMA_LOG_DEBUG, and INFO is
            // already dropped at default verbosity in every tool on this tree
            // (CLAUDE.md, "llama-bench traps"). Without a WARN-level line, a
            // default-verbosity capture of this fix flipping cannot be told
            // apart from the fix never running -- fires only when it actually
            // changed something, so it doubles as the positive control for
            // GGML_SYCL_DEV_LAYER_SYNC=0 (which skips this whole block and so
            // can never print it).
            if (corrected_layers > 0 || corrected_output) {
                LLAMA_LOG_WARN(
                    "%s: [SYCL] dev_layer sync: corrected %d layer(s)%s from a stale device record to CPU "
                    "(the SYCL planner tiered them to host after create_tensor; GGML_SYCL_DEV_LAYER_SYNC=0 "
                    "restores the pre-fix stale behaviour)\n",
                    __func__, corrected_layers, corrected_output ? " + output" : "");
            }
        }
    }
    if (sycl_model_backend) {
        // (c) the late measure: the final placement's compute term, over stand-ins for the weights
        // that are not yet allocated, handed to the backend to compare with the term it admitted.
        // Inert until the backend exports the L4 entry points; a refusal is the load's.
        // It runs at the shape the admitted term was measured and recorded at, so the two compare like for like.
        const llama_load_measure_shape               measure_shape = llama_model_load_measure_shape(*this);
        const std::vector<llama_measure_dummy_entry> late_weights  = llama_model_measure_weights(ml);
        const llama_late_check_result                late =
            llama_load_late_check(*this, measure_shape, sycl_model_loading_guard.txn, late_weights);
        if (!late.unsupported.empty()) {
            // not a refusal of the load: the late check cannot be measured (a model the measure cannot walk, or no
            // plan staged for it), so the final placement's compute term is compared with nothing
            LLAMA_LOG_WARN("%s: late compute-slot check skipped, the compute slot is not checked at load: %s\n",
                           __func__, late.unsupported.c_str());
        }
        for (size_t i = 0; i < late.not_recorded.size(); ++i) {
            // nothing was compared for this device, which is not a pass; the text carries the measured term
            const std::string text =
                llama_late_check_not_recorded_text(late.not_recorded[i], late.n_ubatch, late.not_recorded_bytes[i]);
            LLAMA_LOG_WARN("%s: %s\n", __func__, text.c_str());
        }
        for (size_t i = 0; i < late.state_not_recorded.size(); ++i) {
            const std::string text =
                llama_late_check_state_not_recorded_text(late.state_not_recorded[i], late.state_not_recorded_bytes[i]);
            LLAMA_LOG_WARN("%s: %s\n", __func__, text.c_str());
        }
        if (!late.refusal.empty()) {
            throw std::runtime_error(late.refusal);
        }
    }
    if (sycl_model_backend && ml.use_mmap) {
        // A load whose staged plan maps its files keeps the mapping: host-resident weights bind SYCL_Mapped in place
        // and device weights are staged from it (design 5efe §4.3). Only a load without the tier still
        // uploads every weight through the loader's own reads.
        if (!ml.sycl_load_maps_files()) {
            LLAMA_LOG_INFO("%s: disabling mmap for SYCL weight layout upload\n", __func__);
            ml.use_mmap = false;
        }
    }
#endif

    // per-tensor activation precision policy
    prec_policy.load(ml, *this);

    // A SYCL load maps with prefetch 0: its weights are placed by the unified cache, the mapping is only a source for
    // them, and WILLNEED/SEQUENTIAL over the whole file would read the lazy tables and the device-only weights into
    // the page cache for nothing. A load whose plan maps its files keeps use_mmap, and the host-executed ranges are
    // advised by the SYCL_Mapped buffers themselves, so the loader's mapping prefetches nothing. Off SYCL the mapping
    // prefetches as before.
    bool prefetch_mappings = true;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    if (sycl_model_backend) {
        prefetch_mappings = false;
    }
#endif
    ml.init_mappings(prefetch_mappings, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    // note: the buffers are owned by pimpl->ctxs_bufs, this is a raw-pointer view used to load the data
    std::vector<std::pair<ggml_context *, std::vector<ggml_backend_buffer_t>>> ctx_bufs;
    ctx_bufs.reserve(ml.ctx_map.size());

    // reserve hint: each context creates one buffer per file, or more when its tensors are spread
    //     across disjoint ranges of a file (see get_mapping_ranges)
    pimpl->ctxs_bufs.reserve(ml.ctx_map.size() * ml.files.size());

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    bool has_sycl_weight_buffer = false;
#endif

    for (auto & [ctx_key, ctx_ptr] : ml.ctx_map) {
        ggml_backend_buffer_type_t buft = ctx_key.buft;
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        const auto & sycl_hooks     = sycl_model_loading_guard.hooks;
        const bool   is_sycl_mapped = sycl_hooks.mapped_buffer_type_for_device && llama_model_dev_is_sycl(dev) &&
                                    buft == sycl_hooks.mapped_buffer_type_for_device(dev);
#else
        const bool is_sycl_mapped = false;
#endif

        std::vector<ggml_backend_buffer_ptr> bufs;

        // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
        const bool is_lazy_mapped = ctx_key.lazy && !ml.no_alloc;

        if (is_sycl_mapped && !ml.no_alloc) {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
            // One buffer per file over the file's own mapping, built by the unified cache from the tensors' ranges
            // (design 5efe §4.3). The lazy ranges stay out of every advised and locked segment; the
            // tensors are addressed in load_all_data from the base offset the cache returns.
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                std::vector<ggml_sycl_mapped_tensor_range> tensor_ranges;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    const auto * w = ml.get_weight(ggml_get_name(t));
                    if (w == nullptr || w->idx != idx) {
                        continue;
                    }
                    tensor_ranges.push_back({ ggml_get_name(t), (uint64_t) w->offs, (uint64_t) ggml_nbytes(t) });
                }
                if (tensor_ranges.empty()) {
                    continue;
                }
                const llama_mmap::ranges          excluded = ml.lazy_record.excluded_for_file(idx);
                std::vector<ggml_sycl_file_range> lazy_ranges;
                lazy_ranges.reserve(excluded.size());
                for (const auto & range : excluded) {
                    lazy_ranges.push_back({ (uint64_t) range.first, (uint64_t) (range.second - range.first) });
                }
                const auto &          file             = ml.files.at(idx);
                uint64_t              base_file_offset = 0;
                ggml_backend_buffer_t buf              = sycl_hooks.mapped_buffer_from_file(
                    dev, idx, file->file_id(), (uint64_t) file->size(), tensor_ranges.data(), tensor_ranges.size(),
                    lazy_ranges.data(), lazy_ranges.size(), &base_file_offset);
                if (buf == nullptr) {
                    throw std::runtime_error(
                        format("unable to map %s buffer for model file %u", ggml_backend_buft_name(buft), idx));
                }
                ml.sycl_mapped_base_offsets[buf] = { idx, base_file_offset };
                bufs.emplace_back(buf);
            }
#endif
        } else if ((ml.use_mmap || is_lazy_mapped) && use_mmap_buffer && buffer_from_host_ptr_supported &&
                   is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            const size_t max_size = ggml_get_max_tensor_size(ctx);

            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // map the mmap regions that contain the tensors of this context
                // this is important for metal with apple silicon: a backend buffer pins everything it
                //     wraps, so the regions owned by the other contexts (lazy tensors read on demand
                //     from disk, MoE tensors kept on the CPU, etc) are left out and stay out of RAM
                // this also allows using partial offloading when the model size exceeds the metal buffer
                //     size, but not the RAM size
                void * addr = ml.mappings.at(idx)->addr();

                for (const auto & range : ml.get_mapping_ranges(idx, ctx)) {
                    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + range.first, range.last - range.first, max_size);
                    if (buf == nullptr) {
                        throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                    }
                    bufs.emplace_back(buf);
                }
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            // lock the buffers that hold the tensors: a multi_buffer has no base, and its parts are pageable memory when pinning fails
            // a SYCL_Mapped buffer is the file's pages, which the tier locks itself under -lm mmap+mlock
            if (use_mlock && !ml.no_alloc && ggml_backend_buffer_is_host(buf) && !is_sycl_mapped) {
                std::vector<ggml_backend_buffer_t> lock_bufs;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    if (t->buffer != nullptr && std::find(lock_bufs.begin(), lock_bufs.end(), t->buffer) == lock_bufs.end()) {
                        lock_bufs.push_back(t->buffer);
                    }
                }
                for (ggml_backend_buffer_t lock_buf : lock_bufs) {
                    pimpl->mlock_bufs.emplace_back(new llama_mlock);
                    auto & mlock_buf = pimpl->mlock_bufs.back();
                    mlock_buf->init   (ggml_backend_buffer_get_base(lock_buf));
                    mlock_buf->grow_to(ggml_backend_buffer_get_size(lock_buf));
                }
            }
            bufs.emplace_back(buf);
        }

        std::vector<ggml_backend_buffer_t> buf_ptrs;
        buf_ptrs.reserve(bufs.size());

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
            if (llama_model_buffer_is_sycl(buf.get())) {
                has_sycl_weight_buffer = true;
            }
#endif

            buf_ptrs.emplace_back(buf.get());
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_bufs.emplace_back(ctx, std::move(buf_ptrs));
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        if (sycl_model_backend) {
            sycl_model_loading_guard.finish(true);
        } else {
            sycl_model_loading_guard.cancel();
        }
#endif
        return true;
    }

    // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
    if (!ml.use_mmap) {
        std::stable_partition(ctx_bufs.begin(), ctx_bufs.end(), [](const auto & x) {
            const auto & bufs = x.second;
            return !bufs.empty() && !ggml_backend_buffer_is_host(bufs[0]);
        });
    }

    // load tensor data
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    llama_model_sycl_loading_guard sycl_loading_guard(sycl_model_backend, sycl_model_loading_guard.txn);
#endif
    for (auto & [ctx, bufs] : ctx_bufs) {
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    sycl_loading_guard.finish(true);
#endif

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    if (per_layer_tok_embd_s) {
        ggml_backend_tensor_get(per_layer_tok_embd_s, &per_layer_tok_embd_scale, 0, sizeof(per_layer_tok_embd_scale));
    }

    // inside the SYCL model transaction, so the rotation tables it allocates belong to this model
    load_tensors_hadamard();

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    if (has_sycl_weight_buffer) {
        sycl_model_loading_guard.finish(true);
        // the open reads the published plan by the model token, which exists only from here
        for (auto & s : pimpl->lazy_storage) {
            if (!s->open(sycl_model_token)) {
                throw std::runtime_error(format("failed to open STORAGE table %s", s->name().c_str()));
            }
        }
    } else {
        sycl_model_loading_guard.cancel();
    }
#endif
    // a STORAGE table has no data in memory, so a load that could not open one has no way to read it
    for (const auto & s : pimpl->lazy_storage) {
        if (!s->is_open()) {
            throw std::runtime_error(format(
                "STORAGE table %s cannot be read: this load put no weight in a SYCL buffer, so no placement plan was "
                "published to read the table through; let the SYCL backend hold the weights (remove any tensor "
                "override that sends them elsewhere), or load with --lazy-mode off to read the whole table into memory",
                s->name().c_str()));
        }
    }

    return true;
}

// read and check the prism.hadamard metadata, and record the folded weights (load_tensors_hadamard makes the tensors)
void llama_model_base::load_hparams_hadamard(llama_model_loader & ml) {
    uint32_t hadamard_version = 0;
    ml.get_key(LLM_KV_PRISM_HADAMARD_TIED_OUTPUT, hdmd.tied_output, false);
    if (ml.get_key(LLM_KV_PRISM_HADAMARD_VERSION, hadamard_version, false)) {
        if (hadamard_version != 1 && hadamard_version != 2) {
            throw std::runtime_error(format("unsupported prism.hadamard.version: %u", hadamard_version));
        }

        if ((hadamard_version == 2) != hdmd.tied_output) {
            throw std::runtime_error("prism.hadamard version 2 requires tied_output=true; version 1 forbids it");
        }
        if (hdmd.tied_output && ml.get_weight("output.weight")) {
            throw std::runtime_error("prism.hadamard.tied_output requires output.weight to be absent");
        }

        uint32_t block_size = 0;
        std::string transform;
        std::string axis;
        std::string sign_mode;
        std::vector<std::string> weight_names;

        ml.get_key(LLM_KV_PRISM_HADAMARD_BLOCK_SIZE, block_size);
        ml.get_key(LLM_KV_PRISM_HADAMARD_TRANSFORM, transform);
        ml.get_key(LLM_KV_PRISM_HADAMARD_AXIS, axis);
        ml.get_key(LLM_KV_PRISM_HADAMARD_SIGN_MODE, sign_mode);
        ml.get_arr(LLM_KV_PRISM_HADAMARD_WEIGHT_NAMES, weight_names);

        if (block_size == 0 || (block_size & (block_size - 1)) != 0) {
            throw std::runtime_error(format("invalid prism.hadamard.block_size: %u", block_size));
        }
        if (transform != "normalized-sylvester-walsh-hadamard") {
            throw std::runtime_error(format("unsupported prism.hadamard.transform: %s", transform.c_str()));
        }
        if (axis != "input-last-dimension") {
            throw std::runtime_error(format("unsupported prism.hadamard.axis: %s", axis.c_str()));
        }
        if (sign_mode != "identity" && sign_mode != "explicit") {
            throw std::runtime_error(format("unsupported prism.hadamard.sign_mode: %s", sign_mode.c_str()));
        }
        if (weight_names.empty()) {
            throw std::runtime_error("prism.hadamard.weight_names is empty");
        }

        if (sign_mode == "explicit") {
            std::vector<int32_t> sign_widths;
            std::vector<int32_t> sign_values;
            ml.get_arr(LLM_KV_PRISM_HADAMARD_SIGN_WIDTHS, sign_widths);
            ml.get_arr(LLM_KV_PRISM_HADAMARD_SIGN_VALUES, sign_values);
            // explicit mode with no widths gives an empty sign table, which acts as identity and changes the model
            if (sign_widths.empty()) {
                throw std::runtime_error("prism.hadamard.sign_mode is explicit but sign_widths is empty");
            }
            size_t off = 0;
            for (const int32_t width : sign_widths) {
                if (width <= 0 || (uint32_t) width % block_size != 0 || off + width > sign_values.size()) {
                    throw std::runtime_error(format("invalid prism.hadamard sign width: %d", width));
                }
                if (hdmd.sign_data.count(width)) {
                    throw std::runtime_error(format("duplicate prism.hadamard sign width: %d", width));
                }
                auto & vec = hdmd.sign_data[width];
                vec.assign(sign_values.begin() + off, sign_values.begin() + off + width);
                for (const int32_t v : vec) {
                    if (v != 1 && v != -1) {
                        throw std::runtime_error("prism.hadamard sign values must be +/-1");
                    }
                }
                off += width;
            }
            if (off != sign_values.size()) {
                throw std::runtime_error("prism.hadamard.sign_values length mismatch");
            }
        }

        ml.get_key(LLM_KV_PRISM_HADAMARD_GDN_V_GROUPED, hdmd.gdn_v_grouped, false);

        // only build_lora_mm/build_lora_mm_id apply the transform: refuse archs and tensor kinds that can skip them
        switch (arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_QWEN3:
            case LLM_ARCH_QWEN3MOE:
            case LLM_ARCH_QWEN35:
            case LLM_ARCH_QWEN35MOE:
            case LLM_ARCH_QWEN3NEXT:
                break;
            default:
                throw std::runtime_error(format(
                    "prism.hadamard: arch '%s' is not verified to apply the activation transform to all folded weights",
                    llm_arch_name(arch)));
        }

        // a folded weight W_f = W*D*H is the weight W with the +1/-1 signs D and the normalized block Hadamard H folded in
        // H*H = I and D*D = I, so W*x = W_f*(H*(D*x)): the graph applies D, then H, to the matmul input
        const auto is_foldable_weight = [](const std::string & name) {
            static const char * kinds[] = {
                "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
                "ffn_gate", "ffn_up", "ffn_down",
                "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
                "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                "ssm_out",
            };
            if (name == "output.weight") {
                return true; // the output head is built through build_lora_mm in every arch
            }
            if (name.compare(0, 4, "blk.") != 0) {
                return false;
            }
            size_t pos = 4;
            while (pos < name.size() && isdigit((unsigned char) name[pos])) {
                pos++;
            }
            if (pos == 4 || pos >= name.size() || name[pos] != '.') {
                return false;
            }
            pos++;
            for (const char * kind : kinds) {
                const std::string suffix = std::string(kind) + ".weight";
                if (name.compare(pos, std::string::npos, suffix) == 0) {
                    return true;
                }
            }
            return false;
        };

        for (const auto & weight_name : weight_names) {
            if (!is_foldable_weight(weight_name)) {
                throw std::runtime_error(format(
                    "prism.hadamard: weight '%s' is not on a verified Hadamard-aware matmul path", weight_name.c_str()));
            }
            if (!hdmd.weight_blocks.emplace(weight_name, block_size).second) {
                throw std::runtime_error(format("duplicate prism.hadamard weight: %s", weight_name.c_str()));
            }
        }

        // tables read by row lookup store latent rows: the inverse transform goes on the lookup result
        std::vector<std::string> inverse_names;
        ml.get_arr(LLM_KV_PRISM_HADAMARD_INVERSE_WEIGHT_NAMES, inverse_names, false);
        for (const auto & name : inverse_names) {
            // the graph applies the inverse only after the token-embedding lookup, other latent tables stay rotated
            if (name != "token_embd.weight") {
                throw std::runtime_error(format(
                    "prism.hadamard: weight '%s' is not a verified inverse-after-lookup table", name.c_str()));
            }
            if (hdmd.weight_blocks.count(name) || !hdmd.inverse_blocks.emplace(name, block_size).second) {
                throw std::runtime_error(format("duplicate prism.hadamard inverse weight: %s", name.c_str()));
            }
        }
    }

    if (hdmd.tied_output) {
        if (hadamard_version != 2) {
            throw std::runtime_error("prism.hadamard.tied_output requires version 2");
        }
        const auto it = hdmd.inverse_blocks.find("token_embd.weight");
        if (it == hdmd.inverse_blocks.end()) {
            throw std::runtime_error("prism.hadamard.tied_output requires a latent token embedding");
        }
        hdmd.weight_blocks.emplace("token_embd.weight", it->second);
    } else if (hdmd.inverse_blocks.count("token_embd.weight") && !ml.get_weight("output.weight")) {
        throw std::runtime_error("a tied Hadamard output requires version 2 and tied_output=true");
    }
}

// make one Hadamard matrix per block size and one sign vector per width (the GGUF does not contain them)
// each tensor goes on the buffer type of its weights (never a CPU extra type), and each transform goes in hdmd.rot or hdmd.inv
void llama_model_base::load_tensors_hadamard() {
    if (hdmd.weight_blocks.empty() && hdmd.inverse_blocks.empty()) {
        return;
    }

    struct hadamard_rotation {
        uint32_t block_size;
        ggml_backend_buffer_type_t buft;
        ggml_tensor * tensor;
    };

    std::vector<hadamard_rotation> rotations;
    std::map<std::pair<uint32_t, ggml_backend_buffer_type_t>, ggml_tensor *> sign_tensors;

    const std::pair<const std::unordered_map<std::string, uint32_t> *, llama_hadamard_rotations *> groups[] = {
        { &hdmd.weight_blocks,  &hdmd.rot },
        { &hdmd.inverse_blocks, &hdmd.inv  },
    };
    // inverse transforms use the buffer type of the forward rotations, not the host type of a CPU-mapped table (no PCIe round trip per token)
    ggml_backend_buffer_type_t preferred_buft = nullptr;

    for (const auto & [blocks, target] : groups) {
        for (const auto & entry : *blocks) {
            const std::string & weight_name = entry.first;
            const uint32_t block_size = entry.second;
            const ggml_tensor * weight = get_tensor(weight_name.c_str());
            if (hdmd.tied_output && weight_name == "token_embd.weight") {
                weight = target == &hdmd.rot ? output : tok_embd;
                if (!weight || strcmp(weight->name, "token_embd.weight") != 0) {
                    throw std::runtime_error("prism.hadamard.tied_output is not bound to the token embedding");
                }
            }
            if (weight == nullptr) {
                throw std::runtime_error(format("prism.hadamard weight not found: %s", weight_name.c_str()));
            }
            if (weight->ne[0] % block_size != 0) {
                throw std::runtime_error(format(
                    "prism.hadamard block size %u does not divide input dimension %lld for %s",
                    block_size, (long long) weight->ne[0], weight_name.c_str()));
            }
            if (weight->buffer == nullptr) {
                throw std::runtime_error(format("prism.hadamard weight has no buffer: %s", weight_name.c_str()));
            }

            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(weight->buffer);
            // CPU extra buffer types (e.g. CPU_REPACK) only accept tensors they can repack
            if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    buft = ggml_backend_dev_buffer_type(dev);
                }
            }
            if (target == &hdmd.rot) {
                preferred_buft = buft;
            } else if (preferred_buft) {
                buft = preferred_buft;
            }
            auto it = std::find_if(rotations.begin(), rotations.end(),
                    [block_size, buft](const hadamard_rotation & rotation) {
                        return rotation.block_size == block_size && rotation.buft == buft;
                    });

            if (it == rotations.end()) {
                ggml_init_params params = {
                    /*.mem_size   =*/ ggml_tensor_overhead(),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context_ptr ctx { ggml_init(params) };
                if (!ctx) {
                    throw std::runtime_error("failed to create Hadamard rotation context");
                }

                ggml_tensor * rotation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, block_size, block_size);
                char rotation_name[GGML_MAX_NAME];
                snprintf(rotation_name, sizeof(rotation_name), "prism.hadamard.%u", block_size);
                ggml_set_name(rotation, rotation_name);

                ggml_backend_buffer_ptr buffer { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
                if (!buffer) {
                    throw std::runtime_error(format("unable to allocate %s Hadamard rotation buffer", ggml_backend_buft_name(buft)));
                }
                ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                std::vector<float> data((size_t) block_size * block_size);
                const float scale = 1.0f / sqrtf((float) block_size);
                for (uint32_t row = 0; row < block_size; ++row) {
                    for (uint32_t col = 0; col < block_size; ++col) {
                        uint32_t parity = row & col;
                        parity ^= parity >> 16;
                        parity ^= parity >> 8;
                        parity ^= parity >> 4;
                        parity ^= parity >> 2;
                        parity ^= parity >> 1;
                        data[(size_t) row * block_size + col] = (parity & 1) ? -scale : scale;
                    }
                }
                ggml_backend_tensor_set(rotation, data.data(), 0, data.size() * sizeof(float));

                std::vector<ggml_backend_buffer_ptr> buffers;
                buffers.emplace_back(std::move(buffer));
                pimpl->ctxs_bufs.emplace_back(std::move(ctx), std::move(buffers));
                rotations.push_back({ block_size, buft, rotation });
                it = std::prev(rotations.end());
            }

            ggml_tensor * sign_tensor = nullptr;
            if (!hdmd.sign_data.empty()) {
                const uint32_t width = (uint32_t) weight->ne[0];
                const auto sd = hdmd.sign_data.find(width);
                if (sd == hdmd.sign_data.end()) {
                    throw std::runtime_error(format(
                        "prism.hadamard has no sign vector for width %u (%s)", width, weight_name.c_str()));
                }
                const auto key = std::make_pair(width, buft);
                auto st = sign_tensors.find(key);
                if (st == sign_tensors.end()) {
                    ggml_init_params params = {
                        /*.mem_size   =*/ ggml_tensor_overhead(),
                        /*.mem_buffer =*/ NULL,
                        /*.no_alloc   =*/ true,
                    };
                    ggml_context_ptr ctx { ggml_init(params) };
                    if (!ctx) {
                        throw std::runtime_error("failed to create Hadamard sign context");
                    }

                    ggml_tensor * signs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
                    char sign_name[GGML_MAX_NAME];
                    snprintf(sign_name, sizeof(sign_name), "prism.hadamard.signs.%u", width);
                    ggml_set_name(signs, sign_name);

                    ggml_backend_buffer_ptr buffer { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
                    if (!buffer) {
                        throw std::runtime_error(format("unable to allocate %s Hadamard sign buffer", ggml_backend_buft_name(buft)));
                    }
                    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                    std::vector<float> data(width);
                    for (uint32_t i = 0; i < width; ++i) {
                        data[i] = (float) sd->second[i];
                    }
                    ggml_backend_tensor_set(signs, data.data(), 0, data.size() * sizeof(float));

                    std::vector<ggml_backend_buffer_ptr> buffers;
                    buffers.emplace_back(std::move(buffer));
                    pimpl->ctxs_bufs.emplace_back(std::move(ctx), std::move(buffers));
                    st = sign_tensors.emplace(key, signs).first;
                }
                sign_tensor = st->second;
            }

            llama_hadamard_transform transform { it->tensor, sign_tensor };
            // the GDN output projection reads its value heads in tiled order, see set_gdn_v_perm
            if (hdmd.gdn_v_grouped && weight_name.find(".ssm_out.") != std::string::npos &&
                !transform.set_gdn_v_perm(weight->ne[0], hparams.ssm_dt_rank, hparams.ssm_n_group)) {
                throw std::runtime_error(format("prism.hadamard: bad GDN head geometry for %s", weight_name.c_str()));
            }
            target->emplace(weight, transform);
        }
    }

    LLAMA_LOG_INFO("%s: loaded %zu Hadamard-folded weight(s) (%zu inverse-lookup) using %zu rotation(s) and %zu sign vector(s)\n",
            __func__, hdmd.rot.size() + hdmd.inv.size(), hdmd.inv.size(), rotations.size(), sign_tensors.size());
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = nullptr;
    if (tn.bid != -1) {
        // blocks that are not model layers (e.g. the blocks of a head) go with the output
        buft_list_layer = (size_t) tn.bid < pimpl->dev_layer.size() ? pimpl->dev_layer.at(tn.bid).buft_list : pimpl->dev_output.buft_list;
    }
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

llama_ftype llama_model::ftype() const {
    return pimpl->ftype;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

ggml_tensor * llama_model::get_w_scale(const ggml_tensor * w) const {
    const auto it = w_scales.find(w);
    return it == w_scales.end() ? nullptr : it->second;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

bool llama_model::no_alloc() const {
    return params.no_alloc;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (no_alloc()) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, no_alloc());

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_out            = %u\n",     __func__, hparams.n_embd_out());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer());
        LLAMA_LOG_INFO("%s: n_layer_all           = %u\n",     __func__, hparams.n_layer_all);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: non_causal_type       = %d\n",     __func__, hparams.non_causal_type);
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used());
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (arch == LLM_ARCH_GRANITE &&
            std::any_of(hparams.deepstack_mapping_arr.begin(),
                        hparams.deepstack_mapping_arr.end(),
                        [](const auto & entry) { return entry >= 0; })) {
            LLAMA_LOG_INFO("%s: deepstack_mapping_arr = %s\n", __func__,
                           print_f([&](uint32_t il) { return hparams.deepstack_mapping_arr[il]; },
                           hparams.n_layer_all).c_str());
        }
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (hparams.rope_section_order != LLAMA_ROPE_SECTION_ORDER_TYXZ) {
            LLAMA_LOG_INFO("%s: rope section order    = %s\n", __func__, llama_rope_section_order_name(hparams.rope_section_order).c_str());
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_MAMBA ||
                arch == LLM_ARCH_MAMBA2 ||
                arch == LLM_ARCH_JAMBA ||
                arch == LLM_ARCH_FALCON_H1 ||
                arch == LLM_ARCH_PLAMO2 ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_QWEN3NEXT ||
                arch == LLM_ARCH_QWEN35 ||
                arch == LLM_ARCH_QWEN35MOE ||
                arch == LLM_ARCH_CLOVER_MOE ||
                arch == LLM_ARCH_ZONOS3 ||
                arch == LLM_ARCH_NEMOTRON_H ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
        }

        if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK2OCR ||
                arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA ||
                arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_MISTRAL4 ||
                arch == LLM_ARCH_MISTRAL_LARGE4 || arch == LLM_ARCH_HY_V4) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n",     __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n",     __func__, hparams.n_embd_head_k_mla());
            LLAMA_LOG_INFO("%s: n_embd_head_v_mla     = %d\n",     __func__, hparams.n_embd_head_v_mla());
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_QWEN2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_MELLUM ||
                arch == LLM_ARCH_COHERE2MOE ||
                arch == LLM_ARCH_QWEN3MOE ||
                arch == LLM_ARCH_OPENAI_MOE ||
                arch == LLM_ARCH_QWEN3VLMOE ||
                arch == LLM_ARCH_RND1) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
        }

        if (arch == LLM_ARCH_MINICPM ||
                arch == LLM_ARCH_GRANITE ||
                arch == LLM_ARCH_GRANITE_MOE ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_GRANITE_SWITCH ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.f_embedding_scale);
            LLAMA_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.f_residual_scale);
            LLAMA_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.f_attention_scale);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_BAILINGMOE) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
        }

        if (arch == LLM_ARCH_BAILINGMOE2 || arch == LLM_ARCH_BAILINGMOE3) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %d\n",     __func__, hparams.n_layer_nextn);
        }

        if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_GROVEMOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_chexp            = %d\n",     __func__, hparams.n_ff_chexp);
            LLAMA_LOG_INFO("%s: n_group_experts       = %d\n",     __func__, hparams.n_group_experts);
            LLAMA_LOG_INFO("%s: expert_group_scale    = %.2f\n",   __func__, hparams.expert_group_scale);
        }
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

bool llama_model::dev_layer_is_sycl(int il) const {
    return llama_model_dev_is_sycl(dev_layer(il));
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

const llama_lazy_storage * llama_model::lazy_storage(const std::string & name) const {
    for (const auto & s : pimpl->lazy_storage) {
        if (s->name() == name) {
            return s.get();
        }
    }
    return nullptr;
}

llama_rope_params llama_model::get_rope_params(const llama_cparams & cparams, int il) const {
    llama_rope_params res;

    res.n_rot       = hparams.has_rope(il) ? hparams.n_rot(il) : 0;
    res.n_offs      = hparams.n_lora_kv > 0 ? hparams.n_embd_head_k(il) - hparams.n_rot(il) : 0;
    res.type        = hparams.rope_type;
    res.n_ctx_orig  = cparams.n_ctx_orig_yarn;
    res.freq_base   = get_rope_freq_base (cparams, il);
    res.freq_scale  = get_rope_freq_scale(cparams, il);
    res.ext_factor  = cparams.yarn_ext_factor;
    res.attn_factor = cparams.yarn_attn_factor;
    res.beta_fast   = cparams.yarn_beta_fast;
    res.beta_slow   = cparams.yarn_beta_slow;
    res.factors     = get_rope_factors(cparams, il);

    return res;
}

llama_rope_params llama_model::get_rope_params_indexer(const llama_cparams & cparams, int il) const {
    llama_rope_params res = get_rope_params(cparams, il);

    res.n_offs     = 0;
    res.type       = LLAMA_ROPE_TYPE_NEOX;
    res.freq_base  = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;
    res.factors    = nullptr;

    return res;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_policy llama_model::memory_policy(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_policy pol;

    // Every callback below captures the model (`this`) and nothing of the arguments, so the policy
    // outlives this call.

    pol.swa_type = hparams.swa_type;

    if (arch == LLM_ARCH_DFLASH && (hparams.dsv4_hc_mult > 0 || hparams.n_lora_kv > 0)) {
        // DSV4 and ML4 DSpark stages store a single MLA-style K per position (window = the draft ring)
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

        pol.kind = LLAMA_MEMORY_KIND_ISWA;
        return pol;
    }

    // the non-causal LFM2 and LFM3 decision graphs read the whole prompt in one batch, nothing is kept
    if ((arch == LLM_ARCH_LFM2 && !hparams.causal_attn && hparams.n_layer_decision > 0) ||
        (arch == LLM_ARCH_LFM3 && !hparams.causal_attn)) {
        pol.kind = LLAMA_MEMORY_KIND_NONE;
        return pol;
    }

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_GEMMA_EMBEDDING2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_CLEF:
        case LLM_ARCH_KOKORO:
        case LLM_ARCH_TINY_VOCA:
            {
                pol.kind = LLAMA_MEMORY_KIND_NONE;
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                // only sparse (MSA) layers carry an indexer key cache
                pol.kind       = LLAMA_MEMORY_KIND_MSA;
                pol.filter_idx = [this](int32_t il) {
                    return (uint32_t) il < hparams.n_layer() && hparams.is_indexer_full(il);
                };
            } break;
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
                    // MTP context uses a plain attention KV cache holding only the
                    // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
                    pol.kind   = LLAMA_MEMORY_KIND_KV;
                    pol.filter = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                } else {
                    // Main context: DSA cache for the trunk layers only - the nextn
                    // layer(s) are never attended by the trunk graph.
                    pol.kind = LLAMA_MEMORY_KIND_DSA;
                    if (hparams.n_layer_nextn > 0) {
                        pol.filter = [this](int32_t il) { return (uint32_t) il < hparams.n_layer(); };
                    }
                    pol.filter_aux = [this](int32_t il) {
                        return (uint32_t) il < hparams.n_layer() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il));
                    };
                }
            } break;
        case LLM_ARCH_MISTRAL_LARGE4:
            {
                // only "full" layers own an indexer, so the shared layers need no indexer cache
                pol.kind       = LLAMA_MEMORY_KIND_DSA;
                pol.filter_aux = [this](int32_t il) { return hparams.is_indexer_full(il); };

                // layers without own KV read the MLA cache of their source layer
                pol.reuse = [this](int32_t il) -> int32_t {
                    return hparams.has_kv(il) ? -1 : (int32_t) hparams.kv_source(il);
                };
            } break;
        case LLM_ARCH_GLM5_NEXT:
            {
                // KDA layers are recurrent, the DSA layers use a K-only MLA cache plus an indexer cache.
                // The NextN block is never attended by the trunk graph
                pol.kind       = LLAMA_MEMORY_KIND_HYBRID_IDX;
                pol.filter     = [this](int32_t il) {
                    return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il);
                };
                pol.filter_aux = [this](int32_t il) {
                    return (uint32_t) il < hparams.n_layer() && hparams.is_recr(il);
                };
                pol.filter_idx = [this](int32_t il) {
                    return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il) && hparams.is_indexer_full(il);
                };

                // the draft head is a single DSA layer
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    if (hparams.n_layer_nextn == 0) {
                        throw std::runtime_error("GLM5-Next MTP requires the NextN block, convert without --no-mtp");
                    }
                    pol.filter     = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                    pol.filter_idx = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                    pol.filter_aux = [](int32_t) { return false; };
                }
            } break;
        case LLM_ARCH_MOSHI:
            {
                // every layer attends to the same sliding window: one cache that masks it, the cells past the window are reused
                pol.kind     = LLAMA_MEMORY_KIND_KV;
                pol.swa_type = LLAMA_SWA_TYPE_STANDARD;
            } break;
        case LLM_ARCH_HY_V4:
            {
                if (hparams.indexer_top_k == 0) {
                    // full-attention checkpoint: no indexer, so no indexer key cache
                    pol.kind = LLAMA_MEMORY_KIND_KV;
                } else {
                    // only "full" layers own an indexer, so the shared layers need no indexer cache
                    pol.kind       = LLAMA_MEMORY_KIND_DSA;
                    pol.filter_aux = [this](int32_t il) { return hparams.is_indexer_full(il); };
                }
            } break;
        case LLM_ARCH_DOTS3NOTE:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // MTP draft context: plain attention KV cache holding only the nextn layer
                    pol.kind   = LLAMA_MEMORY_KIND_KV;
                    pol.filter = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                } else {
                    // main context: DSA cache for the trunk full-attention layers plus a window-sized SWA cache
                    pol.kind = LLAMA_MEMORY_KIND_DSA_ISWA;
                    if (hparams.n_layer_nextn > 0) {
                        pol.filter = [this](int32_t il) { return (uint32_t) il < hparams.n_layer(); };
                    }
                    pol.filter_aux = [this](int32_t il) {
                        return (uint32_t) il < hparams.n_layer() && hparams.is_indexer_full(il);
                    };
                }
            } break;
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_DEEPSEEK4:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    pol.kind   = LLAMA_MEMORY_KIND_ISWA;
                    pol.filter = [this](int32_t il) { return il >= (int32_t) hparams.n_layer(); };
                } else {
                    pol.kind = LLAMA_MEMORY_KIND_DSV4;
                }
            } break;
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
                     arch == LLM_ARCH_BAILINGMOE3);

                const bool mtp_on_hybrid_nemotron =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_NEMOTRON_H_MOE;

                if (llm_arch_is_recurrent(arch)) {
                    pol.kind = LLAMA_MEMORY_KIND_RECURRENT;
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_i::layer_filter_cb filter_attn = [this](int32_t il) { return !hparams.is_recr(il); };
                    llama_memory_i::layer_filter_cb filter_recr = [this](int32_t il) { return hparams.is_recr(il); };
                    // only the sparse-attention architectures use llama_memory_hybrid_idx
                    // a null filter_idx means the GGUF has no indexer tensors
                    llama_memory_i::layer_filter_cb filter_idx  = nullptr;
                    const bool needs_mem_idx = (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_KIMI_K31);
                    if (arch == LLM_ARCH_KIMI_K31) {
                        // DSA layers keep a K-only MLA cache plus a per-token indexer key cache,
                        // the KDA layers stay recurrent
                        filter_attn = [this](int32_t il) {
                            return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [this](int32_t il) {
                            return (uint32_t) il < hparams.n_layer() && hparams.is_recr(il);
                        };
                        filter_idx = [this](int32_t il) {
                            return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il) && hparams.is_indexer_full(il);
                        };
                    } else if (arch == LLM_ARCH_FALCON_H1 || arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP || arch == LLM_ARCH_ZAYA1) {
                        filter_attn = [](int32_t) { return true; };
                        filter_recr = [](int32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [this](int32_t il) {
                            return !hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [this](int32_t il) {
                            return hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_CLOVER_MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_MINIMAX_01 ||
                               arch == LLM_ARCH_ZONOS3) {
                        filter_attn = [this](int32_t il) {
                            return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [this](int32_t il) {
                            return (uint32_t) il < hparams.n_layer() && hparams.is_recr(il);
                        };

                        if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0) {
                            // QSA runs on the dense-attention layers only
                            filter_idx = [this](int32_t il) {
                                return (uint32_t) il < hparams.n_layer() && !hparams.is_recr(il);
                            };
                        }

                        // the MTP draft context holds the MTP block alone: its attention and indexer, no recurrent layer
                        if (arch == LLM_ARCH_QWEN4EXP && params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter_attn = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                            filter_recr = [](int32_t)        { return false; };
                            if (filter_idx) {
                                filter_idx = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                            }
                        }
                    }

                    pol.kind       = hparams.swa_type != LLAMA_SWA_TYPE_NONE ? LLAMA_MEMORY_KIND_HYBRID_ISWA :
                                     needs_mem_idx                           ? LLAMA_MEMORY_KIND_HYBRID_IDX :
                                                                               LLAMA_MEMORY_KIND_HYBRID;
                    pol.filter     = std::move(filter_attn);
                    pol.filter_aux = std::move(filter_recr);
                    pol.filter_idx = std::move(filter_idx);
                } else {
                    llama_memory_i::layer_filter_cb filter = nullptr;
                    llama_memory_i::layer_reuse_cb  reuse  = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [this](int32_t il) -> int32_t {
                            return hparams.has_kv(il) ? -1 : (int32_t) hparams.kv_source(il);
                        };
                    }

                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                    }

                    // don't filter when n_layer_nextn is repurposed for a router layer the trunk attends
                    // or when a model is entirely n_layer_nextn layers and has no trunk
                    if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0) {
                        if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter = [this](int32_t il) { return (uint32_t) il >= hparams.n_layer(); };
                        } else {
                            filter = [this](int32_t il) { return (uint32_t) il <  hparams.n_layer(); };
                        }
                    }

                    pol.filter = std::move(filter);

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        pol.kind  = LLAMA_MEMORY_KIND_ISWA;
                        pol.reuse = std::move(reuse);

                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            // the target model's KV cache is the source this one shares from
                            llama_context * ctx_other = cparams.ctx_other;

                            pol.mem_other = llama_get_memory(ctx_other);
                            pol.share     = [this, ctx_other](int32_t il) {
                                const llama_model * model_other = llama_get_model(ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };
                        }
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        pol.kind = LLAMA_MEMORY_KIND_KV;
                    }
                }
            }
    }

    return pol;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams,
                                            bool measure_no_alloc) const {
    const llama_memory_policy pol = memory_policy(params, cparams);

    // the measure form refuses a kind its shape structs cannot describe; params.no_alloc (a model loaded with
    // llama_model_params::no_alloc) builds every kind on dummy buffers, as upstream does
    if (measure_no_alloc) {
        if (const char * name = llama_memory_kind_unsupported(pol.kind)) {
            throw llama_measure_unsupported(
                format("memory kind %s has no no_alloc form (arch %s; refused)", name, arch_name().c_str()));
        }
    }

    const bool alloc_none = measure_no_alloc || params.no_alloc;

    llama_memory_i * res = nullptr;

    switch (pol.kind) {
        case LLAMA_MEMORY_KIND_NONE:
            {
                res = nullptr;
            } break;
        case LLAMA_MEMORY_KIND_MSA:
            {
                res = new llama_kv_cache_msa(
                        *this,
                        alloc_none,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        pol.filter_idx,
                        nullptr);
            } break;
        case LLAMA_MEMORY_KIND_DSA:
            {
                res = new llama_kv_cache_dsa(
                        *this,
                        alloc_none,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        hparams.n_swa,
                        hparams.swa_type,
                        pol.filter,
                        pol.filter_aux,
                        pol.reuse);
            } break;
        case LLAMA_MEMORY_KIND_DSA_ISWA:
            {
                res = new llama_kv_cache_dsa_iswa(
                        *this,
                        alloc_none,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        params.swa_full,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        cparams.n_ubatch,
                        pol.filter,
                        pol.filter_aux,
                        nullptr);
            } break;
        case LLAMA_MEMORY_KIND_DSV4:
            {
                res = new llama_kv_cache_dsv4(
                        *this,
                        alloc_none,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        params.swa_full,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        cparams.n_ubatch,
                        cparams.n_rs_seq,
                        nullptr,
                        nullptr);
            } break;
        case LLAMA_MEMORY_KIND_RECURRENT:
            {
                res = new llama_memory_recurrent(
                        *this,
                        alloc_none,
                        GGML_TYPE_F32,
                        GGML_TYPE_F32,
                        cparams.offload_kqv,
                        std::max((uint32_t) 1, cparams.n_seq_max),
                        cparams.n_seq_max,
                        cparams.n_rs_seq,
                        nullptr);
            } break;
        case LLAMA_MEMORY_KIND_HYBRID_ISWA:
            {
                // Use hybrid-iswa for hybrid models with SWA
                res = new llama_memory_hybrid_iswa(
                    /* model             */ *this,
                    /* no_alloc          */ alloc_none,
                    /* attn_type_k       */ params.type_k,
                    /* attn_type_v       */ params.type_v,
                    /* attn_v_trans      */ !cparams.flash_attn,
                    /* attn_swa_full     */ params.swa_full,
                    /* attn_kv_size      */ cparams.n_ctx_seq,
                    /* attn_n_ubatch     */ cparams.n_ubatch,
                    /* recurrent_type_r  */ GGML_TYPE_F32,
                    /* recurrent_type_s  */ GGML_TYPE_F32,
                    /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                    /* n_seq_max         */ cparams.n_seq_max,
                    /* n_rs_seq          */ cparams.n_rs_seq,
                    /* offload           */ cparams.offload_kqv,
                    /* unified           */ cparams.kv_unified,
                    /* filter_attn       */ pol.filter,
                    /* filter_recr       */ pol.filter_aux);
            } break;
        case LLAMA_MEMORY_KIND_HYBRID_IDX:
            {
                // sparse attention over a per-token indexer cache, in its own memory type
                res = new llama_memory_hybrid_idx(
                    /* model             */ *this,
                    /* no_alloc          */ alloc_none,
                    /* attn_type_k       */ params.type_k,
                    /* attn_type_v       */ params.type_v,
                    /* attn_v_trans      */ !cparams.flash_attn,
                    /* attn_kv_size      */ cparams.n_ctx_seq,
                    /* attn_n_swa        */ hparams.n_swa,
                    /* attn_swa_type     */ hparams.swa_type,
                    /* recurrent_type_k  */ GGML_TYPE_F32,
                    /* recurrent_type_v  */ GGML_TYPE_F32,
                    /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                    /* n_seq_max         */ cparams.n_seq_max,
                    /* n_rs_seq          */ cparams.n_rs_seq,
                    /* offload           */ cparams.offload_kqv,
                    /* unified           */ cparams.kv_unified,
                    /* filter_attn       */ pol.filter,
                    /* filter_recr       */ pol.filter_aux,
                    /* filter_idx        */ pol.filter_idx);
            } break;
        case LLAMA_MEMORY_KIND_HYBRID:
            {
                res = new llama_memory_hybrid(
                    /* model             */ *this,
                    /* no_alloc          */ alloc_none,
                    /* attn_type_k       */ params.type_k,
                    /* attn_type_v       */ params.type_v,
                    /* attn_v_trans      */ !cparams.flash_attn,
                    /* attn_kv_size      */ cparams.n_ctx_seq,
                    /* attn_n_swa        */ hparams.n_swa,
                    /* attn_swa_type     */ hparams.swa_type,
                    /* recurrent_type_k  */ GGML_TYPE_F32,
                    /* recurrent_type_v  */ GGML_TYPE_F32,
                    /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                    /* n_seq_max         */ cparams.n_seq_max,
                    /* n_rs_seq          */ cparams.n_rs_seq,
                    /* offload           */ cparams.offload_kqv,
                    /* unified           */ cparams.kv_unified,
                    /* filter_attn       */ pol.filter,
                    /* filter_recr       */ pol.filter_aux);
            } break;
        case LLAMA_MEMORY_KIND_ISWA:
            {
                res = new llama_kv_cache_iswa(
                        *this,
                        alloc_none,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        params.swa_full,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        cparams.n_ubatch,
                        pol.mem_other,
                        pol.filter,
                        pol.reuse,
                        pol.share);
            } break;
        case LLAMA_MEMORY_KIND_KV:
            {
                res = new llama_kv_cache(
                        *this,
                        alloc_none,
                        hparams,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        hparams.n_swa,
                        pol.swa_type,
                        nullptr,
                        pol.filter,
                        nullptr,
                        nullptr);
            } break;
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add the LM head (if the context outputs logits)
    llm->build_logits();

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->build_layer_inp_dev();

    llm->build_h_nextn_dev();

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    // clang-format off
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.load_mode                   =*/ LLAMA_LOAD_MODE_AUTO,
        /*.lazy_mode                   =*/ LLAMA_LAZY_MODE_AUTO,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
        /*.load_mtp                    =*/ false,
        /*.n_ctx_hint                  =*/ 0,
        /*.n_ubatch_hint               =*/ 0,
    };
    // clang-format on

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer();
}

int32_t llama_model_n_layer_nextn(const llama_model * model) {
    return model->hparams.n_layer_nextn;
}

llama_dflash_variant llama_model_dflash_variant(const llama_model * model) {
    const auto * dflash = dynamic_cast<const llama_model_dflash *>(model);
    if (dflash == nullptr) {
        return LLAMA_DFLASH_VARIANT_NONE;
    }

    switch (dflash->variant) {
        case llama_model_dflash::VARIANT_DFLASH:   return LLAMA_DFLASH_VARIANT_DFLASH;
        case llama_model_dflash::VARIANT_DFLASH2:  return LLAMA_DFLASH_VARIANT_DFLASH2;
        case llama_model_dflash::VARIANT_DSPARK:   return LLAMA_DFLASH_VARIANT_DSPARK;
        case llama_model_dflash::VARIANT_LILICORR: return LLAMA_DFLASH_VARIANT_LILICORR;
    }

    return LLAMA_DFLASH_VARIANT_NONE;
}

int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) {
        return 0;
    }
    return model->hparams.n_swa;
}


uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

const char * llama_model_decision_label(const struct llama_model * model, uint32_t i) {
    if (i < model->decision_labels.size()) {
        return model->decision_labels[i].c_str();
    }

    return nullptr;
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_METIS16:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
        case LLM_ARCH_KOKORO:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK2OCR:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_MUSE_GLIMMER:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_GRANITE_SWITCH:
        case LLM_ARCH_GRANITE_SWA:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_NANBEIGE:
        case LLM_ARCH_POCKETTTS:
        case LLM_ARCH_MOSHI:
        case LLM_ARCH_ZONOS3:
        case LLM_ARCH_LIMITE:
        case LLM_ARCH_TINY_VOCA:
            return LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_BAILINGMOE3:
            // VL files carry mrope sections; text-only files keep NORM rope
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        // HY_V4 rotates consecutive pairs, matching the reference implementation
        case LLM_ARCH_HY_V4:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_VLLM_DECISION2:
        case LLM_ARCH_FLINT:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_RICURSIVE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_GEMMA4_ASSISTANT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_GEMMA_EMBEDDING2:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_LFM3:
        case LLM_ARCH_ZAYA1:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_SPARK2_5:
        case LLM_ARCH_TALKIE:
        case LLM_ARCH_K2_HORIZON:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_HRM_TEXT:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_DOMYN_EDGE:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_DFLASH:
            // drafts for M-RoPE targets carry rope sections and follow the target's temporal dim
            if (const auto & s = model->hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
                return LLAMA_ROPE_TYPE_MROPE;
            }
            // DSV4 and ML4 DSpark drafters use normal RoPE; legacy DFlash backbones are NeoX
            return model->hparams.dsv4_hc_mult > 0 || model->hparams.n_lora_kv > 0 ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_SENSENOVA_U1:
        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLEF:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_QWEN3TTS:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_HUNYUAN_VL:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_arr_n(const llama_model * model, const char * key) {
    const auto it = model->gguf_kv_arr.find(key);
    if (it == model->gguf_kv_arr.end() || it->second.size() > (size_t) INT32_MAX) {
        return -1;
    }
    return (int32_t) it->second.size();
}

const char * llama_model_meta_arr_str(const llama_model * model, const char * key, int32_t i) {
    const auto it = model->gguf_kv_arr.find(key);
    if (it == model->gguf_kv_arr.end() || i < 0 || (size_t) i >= it->second.size()) {
        return nullptr;
    }
    return it->second[i].c_str();
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype();
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_DFLASH:    return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

void llama_model_set_attn_rot(struct llama_model * model, bool value) {
    model->attn_rot = value;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
    TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE),
    TENSOR_READ_LAZY      (llama_model_loader::TENSOR_READ_LAZY) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,    "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,      "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        return;
    }

    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
        return;
    }

    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
        if (!layer.wqkv_b) {
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
        }
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

llama_model_base::nextn_flags_t llama_model_base::nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe) const {
    int trunk = 0;
    int mtp   = 0;

    // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
    if (hparams.n_layer_nextn > 0) {
        if (ml.get_weight(tn(trunk_probe, "weight", 0).str().c_str()) == nullptr) {
            trunk = TENSOR_NOT_REQUIRED;
        }
        if (ml.get_weight(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", hparams.n_layer()).str().c_str()) == nullptr) {
            mtp = TENSOR_NOT_REQUIRED;
        }
    }

    if (!ml.load_mtp) {
        mtp |= TENSOR_SKIP;
    }

    return { trunk, mtp };
}

void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
        return;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false);
    hparams.set_swa_pattern(n_pattern, dense_first);
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

// read rows [row0, row0 + n_rows) of a 2D tensor as F32
static void llama_tensor_rows_to_f32(const ggml_tensor * tensor, int64_t row0, int64_t n_rows, float * out) {
    const size_t   nelements = (size_t) n_rows * tensor->ne[0];
    const size_t   offset    = (size_t) row0 * tensor->nb[1];
    const size_t   nbytes    = ggml_row_size(tensor->type, tensor->ne[0]) * n_rows;

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, offset, nbytes);
        return;
    }

    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(tensor, buf.data(), offset, nbytes);

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    const size_t nelements = ggml_nelements(tensor);
    GGML_ASSERT(nelements <= UINT32_MAX); // for the return type

    if (out == nullptr) {
        return (uint32_t) nelements;
    }

    llama_tensor_rows_to_f32(tensor, 0, tensor->ne[1], out);

    return (uint32_t) nelements;
}

uint32_t llama_model_n_tokens_delay(const struct llama_model * model) {
    return model->hparams.n_tokens_delay;
}

bool llama_model_classifier_free_guidance(const struct llama_model * model) {
    return model->hparams.classifier_free_guidance;
}

uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr || tokens == nullptr || out == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= tensor->ne[1]) {
            return 0;
        }
        llama_tensor_rows_to_f32(tensor, tokens[i], 1, out + (size_t) i * tensor->ne[0]);
    }

    return n_tokens * (uint32_t) tensor->ne[0];
}
