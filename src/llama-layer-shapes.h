#pragma once

#include "ggml.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-memory.h"

#include <cstdint>
#include <string>
#include <vector>

// The one source of the memory's layer layout (zhcn C7g, moua L4 "On the llama side, one function
// produces the shape").
//
// Which memory a model gets, and which layers each of its caches holds, is decided once, by
// llama_model::memory_policy(). llama_model::create_memory builds from that policy and the caches decide
// each layer through llama_kv_layer_decide(), and llama_kv_layer_shapes() / llama_rs_layer_shapes() read
// the same two, so the shapes handed to the SYCL backend at publish cannot differ from what the memory
// creates. tests/test-layer-shapes.cpp compares them against the tensors of every kind it can build.

struct llama_model;

// The memory classes create_memory chooses between.
enum llama_memory_kind {
    LLAMA_MEMORY_KIND_NONE,         // no memory (encoder-only and diffusion architectures)
    LLAMA_MEMORY_KIND_KV,           // llama_kv_cache
    LLAMA_MEMORY_KIND_ISWA,         // llama_kv_cache_iswa
    LLAMA_MEMORY_KIND_MSA,          // llama_kv_cache_msa
    LLAMA_MEMORY_KIND_DSA,          // llama_kv_cache_dsa
    LLAMA_MEMORY_KIND_DSA_ISWA,     // llama_kv_cache_dsa_iswa
    LLAMA_MEMORY_KIND_DSV4,         // llama_kv_cache_dsv4
    LLAMA_MEMORY_KIND_RECURRENT,    // llama_memory_recurrent
    LLAMA_MEMORY_KIND_HYBRID,       // llama_memory_hybrid
    LLAMA_MEMORY_KIND_HYBRID_ISWA,  // llama_memory_hybrid_iswa
    LLAMA_MEMORY_KIND_HYBRID_IDX,   // llama_memory_hybrid_idx
};

// What create_memory builds: the memory class, and the callbacks that class is constructed with. The
// callbacks capture the model and values copied out of the caller's arguments, never the arguments
// themselves, so a policy outlives the call that made it.
//
//   KV, ISWA          filter, and for ISWA also reuse, share and mem_other
//   MSA               filter_idx
//   DSA, DSA_ISWA     filter (the MLA cache) and filter_aux (the indexer cache)
//   RECURRENT         none: every layer
//   HYBRID*           filter (attention), filter_aux (recurrent), and for HYBRID_IDX filter_idx
struct llama_memory_policy {
    llama_memory_kind kind = LLAMA_MEMORY_KIND_NONE;

    llama_memory_i::layer_filter_cb filter;
    llama_memory_i::layer_filter_cb filter_aux;
    llama_memory_i::layer_filter_cb filter_idx;
    llama_memory_i::layer_reuse_cb  reuse;
    llama_memory_i::layer_share_cb  share;
    llama_memory_t                  mem_other = nullptr;
};

// One layer of one llama_kv_cache.
struct llama_kv_layer_shape {
    uint32_t n_embd_k_gqa  = 0;
    uint32_t n_embd_v_gqa  = 0;  // after the V-cache padding; 0 for an MLA layer, which has no V
    uint32_t n_head_kv     = 0;
    uint32_t n_embd_head_k = 0;
    bool     has_kv        = false;  // false: filtered, shared, reused, or the model has no KV there
    bool     is_swa        = false;
};

// What llama_kv_cache's constructor does with one layer.
enum llama_kv_layer_role {
    LLAMA_KV_LAYER_NO_KV,     // hparams.has_kv(il) is false
    LLAMA_KV_LAYER_FILTERED,  // the cache's filter refuses it
    LLAMA_KV_LAYER_SHARED,    // it views the tensors of the layer `il_share` of another cache
    LLAMA_KV_LAYER_OWN,       // the cache creates K (and V) for it
};

struct llama_kv_layer_decision {
    llama_kv_layer_role  role     = LLAMA_KV_LAYER_NO_KV;
    int32_t              il_share = -1;
    llama_kv_layer_shape shape;  // meaningful for OWN
};

// The decision llama_kv_cache's constructor takes for layer `il`, and the widths it creates the tensors
// with. `has_other` is whether the cache has a source cache (mem_other) to share from.
llama_kv_layer_decision llama_kv_layer_decide(const llama_hparams &                   hparams,
                                              uint32_t                                il,
                                              bool                                    v_trans,
                                              const llama_memory_i::layer_filter_cb & filter,
                                              const llama_memory_i::layer_share_cb &  share,
                                              bool                                    has_other);

struct llama_kv_layer_shapes_result {
    ggml_type type_k   = GGML_TYPE_F16;
    ggml_type type_v   = GGML_TYPE_F16;
    bool      v_trans  = false;
    bool      no_alloc = false;
    bool      sidecar  = false;  // the backend, not llama, knows whether it keeps a packed-K sidecar
    uint32_t  n_stream = 1;

    // Non-empty: the memory kind has tensors these shapes cannot describe, and `layers` is empty. A
    // publisher must refuse by this name, never publish a guess.
    std::string unsupported;

    // indexed by the model's layer index, size n_layer_all (empty when there is no KV cache)
    std::vector<llama_kv_layer_shape> layers;
};

// The KV layers of the memory create_memory builds for these arguments. Pure: it reads the model's
// hyperparameters and the policy, creates nothing and touches no device.
llama_kv_layer_shapes_result llama_kv_layer_shapes(const llama_model &         model,
                                                   const llama_memory_params & params_mem,
                                                   const llama_cparams &       cparams);

// One recurrent-state layer: exactly the arguments llama_memory_recurrent passes to ggml_new_tensor_2d
// for r_l and s_l.
struct llama_rs_layer_shape {
    uint32_t il       = 0;
    int32_t  type_r   = 0;
    int32_t  type_s   = 0;
    uint32_t n_embd_r = 0;
    uint32_t n_embd_s = 0;
    uint32_t n_rows   = 0;  // mem_size * (1 + n_rs_seq)
};

struct llama_rs_layer_shapes_result {
    std::string                       unsupported;  // as for the KV shapes
    std::vector<llama_rs_layer_shape> layers;
};

// The recurrent-state layers of the memory create_memory builds: those the policy keeps. `on_arena`
// says whether a layer's state lives on an arena device (a SYCL device); a layer that does not is left
// out, and with `offload` false the state lives on the CPU and the list is empty.
llama_rs_layer_shapes_result llama_rs_layer_shapes_for(const llama_model &         model,
                                                       const llama_memory_params & params_mem,
                                                       const llama_cparams &       cparams,
                                                       bool                        offload,
                                                       bool (*on_arena)(const llama_model &, uint32_t il));

// The same with the SYCL arena rule: the layer's device is a SYCL device.
llama_rs_layer_shapes_result llama_rs_layer_shapes(const llama_model &         model,
                                                   const llama_memory_params & params_mem,
                                                   const llama_cparams &       cparams,
                                                   bool                        offload);
