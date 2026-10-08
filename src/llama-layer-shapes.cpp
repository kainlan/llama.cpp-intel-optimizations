#include "llama-layer-shapes.h"

#include "llama-model.h"

#include <algorithm>

llama_kv_layer_decision llama_kv_layer_decide(const llama_hparams &                   hparams,
                                              uint32_t                                il,
                                              bool                                    v_trans,
                                              const llama_memory_i::layer_filter_cb & filter,
                                              const llama_memory_i::layer_share_cb &  share,
                                              bool                                    has_other) {
    llama_kv_layer_decision dec;

    if (!hparams.has_kv(il)) {
        dec.role = LLAMA_KV_LAYER_NO_KV;
        return dec;
    }

    if (filter && !filter((int32_t) il)) {
        dec.role = LLAMA_KV_LAYER_FILTERED;
        return dec;
    }

    if (share && has_other) {
        const int32_t il_share = share((int32_t) il);

        if (il_share >= 0) {
            dec.role     = LLAMA_KV_LAYER_SHARED;
            dec.il_share = il_share;
            return dec;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    dec.role                = LLAMA_KV_LAYER_OWN;
    dec.shape.has_kv        = true;
    dec.shape.is_swa        = hparams.is_swa(il);
    dec.shape.n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    dec.shape.n_embd_v_gqa  = hparams.is_mla() ? 0 : (!v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max());
    dec.shape.n_head_kv     = hparams.n_head_kv(il);
    dec.shape.n_embd_head_k = hparams.n_embd_head_k(il);

    return dec;
}

llama_hparams llama_kv_idx_hparams(const llama_hparams & hparams) {
    llama_hparams hp = hparams;

    // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
    std::fill(hp.n_head_kv_arr.begin(), hp.n_head_kv_arr.end(), 1);
    hp.n_embd_head_k_full = hparams.indexer_head_size;

    // the cached indexer keys are raw, rotation happens after pooling at read time, so a
    // K-shift must not rotate them while the stream copies in the same update still apply
    hp.rope_type = LLAMA_ROPE_TYPE_NONE;

    // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
    hp.n_embd_head_k_mla_impl = hparams.indexer_head_size;
    hp.n_embd_head_v_mla_impl = hparams.indexer_head_size;

    return hp;
}

// The kinds whose caches hold tensors the shape structs have no place for (an indexer key cache, the DSV4
// compressor state). Named, so a publisher refuses by the name instead of publishing a guess.
const char * llama_memory_kind_unsupported(llama_memory_kind kind) {
    switch (kind) {
        case LLAMA_MEMORY_KIND_MSA:
            return "llama_kv_cache_msa (indexer key cache)";
        case LLAMA_MEMORY_KIND_DSA:
            return "llama_kv_cache_dsa (indexer key cache)";
        case LLAMA_MEMORY_KIND_DSA_ISWA:
            return "llama_kv_cache_dsa_iswa (indexer key cache)";
        case LLAMA_MEMORY_KIND_DSV4:
            return "llama_kv_cache_dsv4 (compressor state)";
        case LLAMA_MEMORY_KIND_NONE:
        case LLAMA_MEMORY_KIND_KV:
        case LLAMA_MEMORY_KIND_ISWA:
        case LLAMA_MEMORY_KIND_RECURRENT:
        case LLAMA_MEMORY_KIND_HYBRID:
        case LLAMA_MEMORY_KIND_HYBRID_ISWA:
        case LLAMA_MEMORY_KIND_HYBRID_IDX:
            break;
    }
    return nullptr;
}

llama_kv_layer_shapes_result llama_kv_layer_shapes(const llama_model &         model,
                                                   const llama_memory_params & params_mem,
                                                   const llama_cparams &       cparams) {
    llama_kv_layer_shapes_result res;

    const llama_hparams &     hparams = model.hparams;
    const llama_memory_policy pol     = model.memory_policy(params_mem, cparams);

    if (const char * name = llama_memory_kind_unsupported(pol.kind)) {
        res.unsupported = name;
        return res;
    }

    // an iSWA pair chains is_swa onto the filter (the base cache takes the layers that are not SWA, the SWA cache
    // the layers that are), so exactly one of the two can own a layer and the policy's filter alone decides
    // whether it is owned; the shape's is_swa says which cache
    switch (pol.kind) {
        case LLAMA_MEMORY_KIND_KV:
        case LLAMA_MEMORY_KIND_HYBRID:
        case LLAMA_MEMORY_KIND_ISWA:
        case LLAMA_MEMORY_KIND_HYBRID_ISWA:
        case LLAMA_MEMORY_KIND_HYBRID_IDX:
            break;
        default:
            return res;
    }

    res.type_k   = params_mem.type_k;
    res.type_v   = params_mem.type_v;
    res.v_trans  = !cparams.flash_attn;
    res.no_alloc = hparams.no_alloc;
    res.n_stream = cparams.kv_unified ? 1 : cparams.n_seq_max;

    const bool has_other = pol.mem_other != nullptr;

    res.layers.resize(hparams.n_layer_all);

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const llama_kv_layer_decision dec =
            llama_kv_layer_decide(hparams, il, res.v_trans, pol.filter, pol.share, has_other);
        if (dec.role == LLAMA_KV_LAYER_OWN) {
            res.layers[il] = dec.shape;
        }
    }

    // the indexer key cache: its own hparams and filter, no source cache to share from (llama_memory_hybrid_idx)
    if (pol.kind == LLAMA_MEMORY_KIND_HYBRID_IDX && pol.filter_idx != nullptr) {
        const llama_hparams hparams_idx = llama_kv_idx_hparams(hparams);

        res.layers_idx.resize(hparams.n_layer_all);

        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            const llama_kv_layer_decision dec =
                llama_kv_layer_decide(hparams_idx, il, res.v_trans, pol.filter_idx, nullptr, false);
            if (dec.role == LLAMA_KV_LAYER_OWN) {
                res.layers_idx[il] = dec.shape;
            }
        }
    }

    return res;
}

llama_rs_layer_shapes_result llama_rs_layer_shapes_for(const llama_model &         model,
                                                       const llama_memory_params & params_mem,
                                                       const llama_cparams &       cparams,
                                                       bool                        offload,
                                                       bool (*on_arena)(const llama_model &, uint32_t il)) {
    llama_rs_layer_shapes_result res;

    const llama_hparams &     hparams = model.hparams;
    const llama_memory_policy pol     = model.memory_policy(params_mem, cparams);

    if (const char * name = llama_memory_kind_unsupported(pol.kind)) {
        res.unsupported = name;
        return res;
    }

    llama_memory_i::layer_filter_cb filter;

    switch (pol.kind) {
        case LLAMA_MEMORY_KIND_RECURRENT:
            break;
        case LLAMA_MEMORY_KIND_HYBRID:
        case LLAMA_MEMORY_KIND_HYBRID_ISWA:
        case LLAMA_MEMORY_KIND_HYBRID_IDX:
            filter = pol.filter_aux;
            break;
        default:
            return res;
    }

    if (hparams.ple_conv_state() > 0) {
        res.unsupported = "llama_memory_recurrent (PLE conv-state row)";
        return res;
    }

    if (!offload) {
        // the state lives on the CPU
        return res;
    }

    const uint32_t n_rows = std::max((uint32_t) 1, cparams.n_seq_max) * (1 + cparams.n_rs_seq);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter((int32_t) il)) {
            continue;
        }

        if (on_arena != nullptr && !on_arena(model, il)) {
            continue;
        }

        llama_rs_layer_shape sh;
        sh.il       = il;
        sh.type_r   = (int32_t) GGML_TYPE_F32;
        sh.type_s   = (int32_t) GGML_TYPE_F32;
        sh.n_embd_r = hparams.n_embd_r();
        sh.n_embd_s = hparams.n_embd_s();
        sh.n_rows   = n_rows;
        res.layers.push_back(sh);
    }

    return res;
}

static bool llama_rs_layer_on_sycl(const llama_model & model, uint32_t il) {
    return model.dev_layer_is_sycl((int) il);
}

llama_rs_layer_shapes_result llama_rs_layer_shapes(const llama_model &         model,
                                                   const llama_memory_params & params_mem,
                                                   const llama_cparams &       cparams,
                                                   bool                        offload) {
    return llama_rs_layer_shapes_for(model, params_mem, cparams, offload, &llama_rs_layer_on_sycl);
}
