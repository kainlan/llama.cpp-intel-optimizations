// L6 (zhcn C7g shapes): the layer shapes llama publishes to the backend equal what the memory
// actually creates. CPU only: tiny synthetic models on no device, so nothing touches a GPU.
//
// For each (architecture, configuration) the test builds a context, walks its memory down to the
// llama_kv_cache / llama_memory_recurrent objects, and reads, for every model layer, the tensors the
// memory itself created. It then compares them with llama_kv_layer_shapes() / llama_rs_layer_shapes():
//
//   - a layer the shapes call has_kv owns K (and V unless MLA) of exactly the published widths, element
//     types, stream count and byte size, in the cache the shape's is_swa names;
//   - a layer the shapes do not call has_kv owns no tensor in any cache (filtered, reused, shared);
//   - the indexer key cache of a hybrid_idx memory: a layer the published layers_idx call has_kv owns an
//     indexer K of exactly that width (one head of indexer_head_size), element type, stream count and byte
//     size, and no V; every other layer owns none;
//   - the recurrent layers are exactly the offloaded layers, with the published row widths and rows;
//   - the memory kinds llama does not model (MSA, DSA, DSA_ISWA, DSV4) report themselves unsupported by
//     name, and publish nothing; create_memory(no_alloc) refuses each of them with llama_measure_unsupported,
//     which the load-time measure maps to "unsupported" (a WARN and the unplanned path, never a failed load);
//   - every other kind's create_memory(no_alloc) builds the tensors of the real memory, the indexer cache
//     included, on size-0 dummy buffers;
//   - the K-shift sub-caches a memory reports (get_shift_caches) are exactly the leaf caches that can shift
//     under a memory that can shift, kind by kind.
//
// LLAMA_LAYER_SHAPES_DUMP=<path> writes the realised per-layer tensors as text, so a refactor of how the
// memory is built can be diffed against the tree before it.

#include "../src/llama-context.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-layer-shapes.h"
#include "../src/llama-load-measure.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "test-tiny-model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

struct config {
    const char * name;
    bool         flash_attn;
    ggml_type    type_kv;
    uint32_t     n_seq_max;
    bool         kv_unified;
};

static const config k_configs[] = {
    { "fa-off-f16",            false, GGML_TYPE_F16,  1, true  },
    { "fa-on-q8_0",            true,  GGML_TYPE_Q8_0, 1, true  },
    { "fa-off-f16-2seq-split", false, GGML_TYPE_F16,  2, false },
};

struct arch_case {
    llm_arch arch;
    bool     must;  // the case must build: a fixture that cannot is a failure, not a skip
};

static const arch_case k_archs[] = {
    { LLM_ARCH_LLAMA,          true  },
    { LLM_ARCH_GEMMA3,         true  },
    { LLM_ARCH_GEMMA3N,        true  },
    { LLM_ARCH_DEEPSEEK2,      true  },
    { LLM_ARCH_MAMBA,          true  },
    { LLM_ARCH_QWEN35,         true  },
    { LLM_ARCH_NEMOTRON_H,     true  },
    { LLM_ARCH_OPENAI_MOE,     false },
    { LLM_ARCH_QWEN3NEXT,      false },
    { LLM_ARCH_GRANITE_HYBRID, false },
    { LLM_ARCH_LFM2,           false },
    { LLM_ARCH_OPENELM,        false },
    { LLM_ARCH_DECI,           false },
    { LLM_ARCH_DEEPSEEK32,     false },
    { LLM_ARCH_DEEPSEEK4,      false },
    { LLM_ARCH_MINIMAX_M3,     false },
    { LLM_ARCH_QWEN4EXP,       false },
    { LLM_ARCH_DOTS3NOTE,      false },
    { LLM_ARCH_HY_V4,          false },
    { LLM_ARCH_GLM_DSA,        false },
};

struct fixture {
    llama_model_ptr   model;
    llama_context_ptr ctx;
};

static bool build_fixture(fixture & fx, llm_arch arch, const config & cfg) {
    gguf_context_ptr gguf = get_gguf_ctx(arch, moe_mandatory(arch));

    llama_model_params mp                   = llama_model_default_params();
    mp.progress_callback                    = silent_model_load_progress;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices                              = no_devices;
    mp.n_gpu_layers                         = 0;
    mp.load_mode                            = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts                      = false;

    tensor_data_params tp = { 1234, 0.1f };
    fx.model.reset(llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp));
    if (!fx.model) {
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 512;
    cp.flash_attn_type      = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k               = cfg.type_kv;
    cp.type_v               = cfg.type_kv;
    cp.n_batch              = 64;
    cp.n_ubatch             = 64;
    cp.n_seq_max            = cfg.n_seq_max;
    cp.kv_unified           = cfg.kv_unified;
    cp.n_threads            = 2;
    cp.n_threads_batch      = 2;
    try {
        fx.ctx.reset(llama_init_from_model(fx.model.get(), cp));
    } catch (...) {
        return false;
    }
    return fx.ctx != nullptr;
}

// The memory objects that hold tensors, by the role the shapes give them.
struct kv_part {
    const llama_kv_cache * kv;
    bool                   swa;  // the SWA half of an iSWA pair
};

struct memory_view {
    std::string                    kind = "none";
    std::vector<kv_part>           kvs;
    const llama_memory_recurrent * rs  = nullptr;
    const llama_kv_cache *         idx = nullptr;  // the indexer key cache of a hybrid_idx memory
};

static memory_view view_of(llama_memory_t mem) {
    memory_view v;
    if (mem == nullptr) {
        return v;
    }
    if (auto * m = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        v.kind = "iswa";
        v.kvs.push_back({ m->get_base(), false });
        v.kvs.push_back({ m->get_swa(), true });
    } else if (auto * m = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        v.kind = "hybrid_iswa";
        v.kvs.push_back({ m->get_mem_attn()->get_base(), false });
        v.kvs.push_back({ m->get_mem_attn()->get_swa(), true });
        v.rs = m->get_mem_recr();
    } else if (auto * m = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
        // the indexer cache spans the attention layers too, so it is kept apart from the attention caches
        v.kind = "hybrid_idx";
        v.kvs.push_back({ m->get_mem_attn(), false });
        v.rs  = m->get_mem_recr();
        v.idx = m->get_mem_idx();
    } else if (auto * m = dynamic_cast<llama_memory_hybrid *>(mem)) {
        v.kind = "hybrid";
        v.kvs.push_back({ m->get_mem_attn(), false });
        v.rs = m->get_mem_recr();
    } else if (auto * m = dynamic_cast<llama_memory_recurrent *>(mem)) {
        v.kind = "recurrent";
        v.rs   = m;
    } else if (auto * m = dynamic_cast<llama_kv_cache *>(mem)) {
        v.kind = "kv";
        v.kvs.push_back({ m, false });
    } else {
        v.kind = "other";
    }
    return v;
}

static const ggml_tensor * rs_r(const llama_memory_recurrent * rs, int il) {
    return (size_t) il < rs->r_l.size() ? rs->r_l[il] : nullptr;
}

static const ggml_tensor * rs_s(const llama_memory_recurrent * rs, int il) {
    return (size_t) il < rs->s_l.size() ? rs->s_l[il] : nullptr;
}

// a PLE layer's conv history, the third state tensor; null on every other layer
static const ggml_tensor * rs_p(const llama_memory_recurrent * rs, int il) {
    return (size_t) il < rs->p_l.size() ? rs->p_l[il] : nullptr;
}

// The model layers a memory holds K/V for, from the memory itself: each layer's K tensor and
// which cache owns it.
struct realised_layer {
    bool                   owned = false;
    bool                   swa   = false;
    const ggml_tensor *    k     = nullptr;
    const ggml_tensor *    v     = nullptr;
    const llama_kv_cache * owner = nullptr;
};

static realised_layer realise_kv(const memory_view & mv, int32_t il, bool & ambiguous) {
    realised_layer out;
    for (const auto & part : mv.kvs) {
        const ggml_tensor * k = nullptr;
        const ggml_tensor * v = nullptr;
        if (part.kv != nullptr && part.kv->get_layer_tensors(il, &k, &v)) {
            if (out.owned) {
                ambiguous = true;
            }
            out.owned = true;
            out.swa   = part.swa;
            out.k     = k;
            out.v     = v;
            out.owner = part.kv;
        }
    }
    return out;
}

static void dump_arch(FILE * dump, const char * arch_name, const config & cfg, const memory_view & mv, int n_layer) {
    if (dump == nullptr) {
        return;
    }
    fprintf(dump, "arch=%s cfg=%s kind=%s\n", arch_name, cfg.name, mv.kind.c_str());
    for (int il = 0; il < n_layer; ++il) {
        bool                 amb = false;
        const realised_layer r   = realise_kv(mv, il, amb);
        fprintf(dump, "  il=%d kv=%d swa=%d", il, r.owned ? 1 : 0, r.swa ? 1 : 0);
        if (r.owned) {
            fprintf(dump, " k=%lld/%d/%lld v=%lld/%d/%lld cells=%u", (long long) r.k->ne[0], (int) r.k->type,
                    (long long) r.k->ne[2], r.v ? (long long) r.v->ne[0] : -1LL, r.v ? (int) r.v->type : -1,
                    r.v ? (long long) r.v->ne[2] : -1LL, r.owner->get_size());
        }
        if (mv.rs != nullptr) {
            const ggml_tensor * rt = rs_r(mv.rs, il);
            const ggml_tensor * st = rs_s(mv.rs, il);
            if (rt != nullptr && st != nullptr) {
                fprintf(dump, " r=%lld/%d/%lld s=%lld/%d/%lld", (long long) rt->ne[0], (int) rt->type,
                        (long long) rt->ne[1], (long long) st->ne[0], (int) st->type, (long long) st->ne[1]);
            }
        }
        fprintf(dump, "\n");
    }
}

static const char * expected_kind(const std::string & kind) {
    return kind.c_str();
}

// The fixtures sit on no device, so the SYCL arena rule would drop every layer: the comparison with the
// created tensors runs with every layer on an arena, and a control checks the real rule drops them all.
static bool all_on_arena(const llama_model &, uint32_t) {
    return true;
}

// The memory kinds llama_kv_layer_shapes() models: the rest must report themselves unsupported.
static bool kind_is_modelled(const memory_view & mv) {
    return mv.kind == "kv" || mv.kind == "iswa" || mv.kind == "hybrid" || mv.kind == "hybrid_iswa" ||
           mv.kind == "hybrid_idx" || mv.kind == "recurrent" || mv.kind == "none";
}

// llama.cpp-8ecj: the SYCL placement inventory charges KV for the layers llama_kv_layer_owners_default() says own
// K/V. They must be exactly the layers the context's memory created K/V for, in every config: a hybrid model's
// recurrent layers pass hparams.has_kv() but own none, which is how Qwen3-Next came to be charged 48 KV layers
// where its cache holds 12.
static int n_owner_layers_equal    = 0;
static int n_owner_overcount_cases = 0;  // memories where has_kv() would charge a layer the cache does not hold
static int n_owner_idx_layers_equal = 0;  // indexer layers whose owners width matched the created indexer K

static void check_kv_owners(const char * arch_name, const config & cfg, llama_context * ctx, const memory_view & mv) {
    const llama_model &         model   = ctx->get_model();
    const llama_kv_layer_owners owners  = llama_kv_layer_owners_default(model);
    const int                   n_layer = (int) model.hparams.n_layer_all;

    const bool modelled = kind_is_modelled(mv);
    CHECK(owners.modelled == modelled, "%s/%s: kind '%s' owners modelled=%d", arch_name, cfg.name, mv.kind.c_str(),
          (int) owners.modelled);
    if (!owners.modelled) {
        CHECK(owners.owns.empty(), "%s/%s: an unmodelled kind published owners", arch_name, cfg.name);
        return;
    }
    CHECK((int) owners.owns.size() == n_layer, "%s/%s: %zu owners for %d layers", arch_name, cfg.name,
          owners.owns.size(), n_layer);
    bool overcount = false;
    for (int il = 0; il < n_layer && il < (int) owners.owns.size(); ++il) {
        bool                 amb = false;
        const realised_layer r   = realise_kv(mv, il, amb);
        CHECK(owners.owns[il] == r.owned, "%s/%s: layer %d owner=%d but the memory %s K/V", arch_name, cfg.name, il,
              (int) owners.owns[il], r.owned ? "created" : "created no");
        n_owner_layers_equal += owners.owns[il] == r.owned ? 1 : 0;
        overcount = overcount || (model.hparams.has_kv((uint32_t) il) && !r.owned);
    }
    n_owner_overcount_cases += overcount ? 1 : 0;

    // the indexer key width each layer's second buffer was created with (llama.cpp-8ecj), 0 where there is none
    CHECK((int) owners.idx_k_width.size() == n_layer, "%s/%s: %zu indexer widths for %d layers", arch_name, cfg.name,
          owners.idx_k_width.size(), n_layer);
    for (int il = 0; il < n_layer && il < (int) owners.idx_k_width.size(); ++il) {
        const ggml_tensor * k     = nullptr;
        const ggml_tensor * v     = nullptr;
        const bool          owned = mv.idx != nullptr && mv.idx->get_layer_tensors(il, &k, &v);
        const uint32_t      want  = owned ? (uint32_t) k->ne[0] : 0;
        CHECK(owners.idx_k_width[il] == want, "%s/%s: layer %d indexer width %u but the memory created %u", arch_name,
              cfg.name, il, owners.idx_k_width[il], want);
        n_owner_idx_layers_equal += owned && owners.idx_k_width[il] == want ? 1 : 0;
    }
}

static int n_rs_layers_equal  = 0;
static int n_idx_layers_equal = 0;

static void check_shapes(const char * arch_name, const config & cfg, llama_context * ctx, const memory_view & mv) {
    const llama_model & model = ctx->get_model();
    llama_memory_params pm    = {
        /*.type_k    =*/cfg.type_kv,
        /*.type_v    =*/cfg.type_kv,
        /*.swa_full  =*/ctx->get_cparams().swa_full,
        /*.ctx_type  =*/ctx->get_cparams().ctx_type,
        /*.mem_other =*/llama_get_memory(ctx->get_cparams().ctx_other),
    };
    const llama_kv_layer_shapes_result kv = llama_kv_layer_shapes(model, pm, ctx->get_cparams());
    const llama_rs_layer_shapes_result rs =
        llama_rs_layer_shapes_for(model, pm, ctx->get_cparams(), ctx->get_cparams().offload_kqv, &all_on_arena);
    const llama_rs_layer_shapes_result rs_sycl =
        llama_rs_layer_shapes(model, pm, ctx->get_cparams(), ctx->get_cparams().offload_kqv);
    CHECK(rs_sycl.layers.empty(), "%s/%s: the SYCL arena rule kept %zu layers of a model on no SYCL device", arch_name,
          cfg.name, rs_sycl.layers.size());
    const int n_layer = (int) model.hparams.n_layer_all;

    const bool modelled = kind_is_modelled(mv);
    if (!modelled) {
        CHECK(!kv.unsupported.empty(), "%s/%s: kind '%s' is not modelled and must say so", arch_name, cfg.name,
              expected_kind(mv.kind));
        CHECK(kv.layers.empty(), "%s/%s: an unsupported kind publishes no layers", arch_name, cfg.name);
        return;
    }
    CHECK(kv.unsupported.empty(), "%s/%s: kind '%s' is modelled but reported unsupported: %s", arch_name, cfg.name,
          mv.kind.c_str(), kv.unsupported.c_str());

    if (mv.kvs.empty()) {
        for (const auto & l : kv.layers) {
            CHECK(!l.has_kv, "%s/%s: a memory with no KV cache published a KV layer", arch_name, cfg.name);
        }
    } else {
        CHECK((int) kv.layers.size() == n_layer, "%s/%s: %zu shapes for %d layers", arch_name, cfg.name,
              kv.layers.size(), n_layer);
        CHECK(kv.type_k == cfg.type_kv && kv.type_v == cfg.type_kv, "%s/%s: element types", arch_name, cfg.name);
        const uint32_t n_stream = ctx->get_cparams().kv_unified ? 1 : ctx->get_cparams().n_seq_max;
        CHECK(kv.n_stream == n_stream, "%s/%s: n_stream %u, want %u", arch_name, cfg.name, kv.n_stream, n_stream);
        CHECK(kv.v_trans == !ctx->get_cparams().flash_attn, "%s/%s: v_trans", arch_name, cfg.name);
        CHECK(kv.no_alloc == model.hparams.no_alloc, "%s/%s: no_alloc", arch_name, cfg.name);

        int n_owned = 0;
        for (int il = 0; il < (int) kv.layers.size() && il < n_layer; ++il) {
            const llama_kv_layer_shape & sh  = kv.layers[il];
            bool                         amb = false;
            const realised_layer         r   = realise_kv(mv, il, amb);
            CHECK(!amb, "%s/%s: layer %d is owned by two caches", arch_name, cfg.name, il);
            CHECK(sh.has_kv == r.owned, "%s/%s: layer %d has_kv=%d but the memory %s a tensor", arch_name, cfg.name, il,
                  (int) sh.has_kv, r.owned ? "created" : "created no");
            if (!r.owned) {
                continue;
            }
            n_owned++;
            CHECK(sh.is_swa == r.swa, "%s/%s: layer %d is_swa=%d but lives in the %s cache", arch_name, cfg.name, il,
                  (int) sh.is_swa, r.swa ? "SWA" : "base");
            const uint32_t cells = r.owner->get_size();
            CHECK((int64_t) sh.n_embd_k_gqa == r.k->ne[0], "%s/%s: layer %d n_embd_k_gqa %u vs tensor %lld", arch_name,
                  cfg.name, il, sh.n_embd_k_gqa, (long long) r.k->ne[0]);
            CHECK(r.k->ne[2] == (int64_t) kv.n_stream && r.k->ne[1] == (int64_t) cells, "%s/%s: layer %d K extent",
                  arch_name, cfg.name, il);
            CHECK(ggml_nbytes(r.k) == ggml_row_size(kv.type_k, sh.n_embd_k_gqa) * cells * kv.n_stream,
                  "%s/%s: layer %d K bytes", arch_name, cfg.name, il);
            if (r.v == nullptr) {
                CHECK(sh.n_embd_v_gqa == 0, "%s/%s: layer %d is MLA but n_embd_v_gqa=%u", arch_name, cfg.name, il,
                      sh.n_embd_v_gqa);
            } else {
                CHECK((int64_t) sh.n_embd_v_gqa == r.v->ne[0], "%s/%s: layer %d n_embd_v_gqa %u vs tensor %lld",
                      arch_name, cfg.name, il, sh.n_embd_v_gqa, (long long) r.v->ne[0]);
                CHECK(ggml_nbytes(r.v) == ggml_row_size(kv.type_v, sh.n_embd_v_gqa) * cells * kv.n_stream,
                      "%s/%s: layer %d V bytes", arch_name, cfg.name, il);
            }
            CHECK(sh.n_head_kv == model.hparams.n_head_kv(il) && sh.n_embd_head_k == model.hparams.n_embd_head_k(il),
                  "%s/%s: layer %d head geometry", arch_name, cfg.name, il);
        }
        fprintf(stderr, "  %s/%s kind=%s: %d of %d layers hold KV\n", arch_name, cfg.name, mv.kind.c_str(), n_owned,
                n_layer);
    }

    // indexer keys: exactly the layers the indexer cache created K for, of the published width, and no V
    if (mv.idx == nullptr) {
        CHECK(kv.layers_idx.empty(), "%s/%s: no indexer cache but %zu indexer shapes", arch_name, cfg.name,
              kv.layers_idx.size());
    } else {
        CHECK((int) kv.layers_idx.size() == n_layer, "%s/%s: %zu indexer shapes for %d layers", arch_name, cfg.name,
              kv.layers_idx.size(), n_layer);
        const uint32_t cells = mv.idx->get_size();
        int            n_idx = 0;
        for (int il = 0; il < (int) kv.layers_idx.size() && il < n_layer; ++il) {
            const llama_kv_layer_shape & sh    = kv.layers_idx[il];
            const ggml_tensor *          k     = nullptr;
            const ggml_tensor *          v     = nullptr;
            const bool                   owned = mv.idx->get_layer_tensors(il, &k, &v);
            CHECK(sh.has_kv == owned, "%s/%s: layer %d indexer has_kv=%d but the memory %s an indexer K", arch_name,
                  cfg.name, il, (int) sh.has_kv, owned ? "created" : "created no");
            if (!owned) {
                continue;
            }
            n_idx++;
            CHECK((int64_t) sh.n_embd_k_gqa == k->ne[0], "%s/%s: layer %d indexer n_embd_k_gqa %u vs tensor %lld",
                  arch_name, cfg.name, il, sh.n_embd_k_gqa, (long long) k->ne[0]);
            CHECK(k->ne[1] == (int64_t) cells && k->ne[2] == (int64_t) kv.n_stream && k->type == kv.type_k,
                  "%s/%s: layer %d indexer K extent or type (%d, want %d)", arch_name, cfg.name, il, (int) k->type,
                  (int) kv.type_k);
            CHECK(ggml_nbytes(k) == ggml_row_size(kv.type_k, sh.n_embd_k_gqa) * cells * kv.n_stream,
                  "%s/%s: layer %d indexer K bytes", arch_name, cfg.name, il);
            CHECK(v == nullptr && sh.n_embd_v_gqa == 0, "%s/%s: layer %d: the indexer holds a V (n_embd_v_gqa=%u)",
                  arch_name, cfg.name, il, sh.n_embd_v_gqa);
            // from the model, not from the indexer hparams the shapes are derived with
            CHECK(sh.n_head_kv == 1 && sh.n_embd_head_k == model.hparams.indexer_head_size &&
                      sh.n_embd_k_gqa == model.hparams.indexer_head_size,
                  "%s/%s: layer %d indexer head geometry %u x %u, want 1 x %u", arch_name, cfg.name, il, sh.n_head_kv,
                  sh.n_embd_head_k, model.hparams.indexer_head_size);
            n_idx_layers_equal++;
        }
        fprintf(stderr, "  %s/%s kind=%s: %d of %d layers hold an indexer K\n", arch_name, cfg.name, mv.kind.c_str(),
                n_idx, n_layer);
    }

    // recurrent state: exactly the offloaded layers the memory created r/s for
    if (mv.rs == nullptr) {
        CHECK(rs.layers.empty(), "%s/%s: no recurrent memory but %zu RS layers", arch_name, cfg.name, rs.layers.size());
    } else if (std::any_of(mv.rs->p_l.begin(), mv.rs->p_l.end(), [](const ggml_tensor * p) { return p != nullptr; })) {
        // the memory created a PLE conv history, a third state tensor (p_l) the RS shape has no place for: the
        // publisher refuses the model by that name and publishes nothing. Keyed on the realised memory, so the
        // publisher's own predicate is checked against it rather than shared with it.
        CHECK(model.hparams.ple_conv_state() > 0, "%s/%s: the memory holds a PLE row but ple_conv_state() is 0",
              arch_name, cfg.name);
        CHECK(rs.unsupported == "llama_memory_recurrent (PLE conv-state row)" && rs.layers.empty(),
              "%s/%s: a PLE model's RS shapes must be the named refusal, got '%s' and %zu layers", arch_name, cfg.name,
              rs.unsupported.c_str(), rs.layers.size());
    } else {
        CHECK(model.hparams.ple_conv_state() == 0, "%s/%s: ple_conv_state() is %u but the memory holds no PLE row",
              arch_name, cfg.name, model.hparams.ple_conv_state());
        CHECK(rs.unsupported.empty(), "%s/%s: RS reported unsupported: %s", arch_name, cfg.name,
              rs.unsupported.c_str());
        size_t n_made = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (rs_r(mv.rs, il) != nullptr) {
                n_made++;
            }
        }
        CHECK(rs.layers.size() == n_made, "%s/%s: %zu RS shapes but the memory made %zu", arch_name, cfg.name,
              rs.layers.size(), n_made);
        for (const auto & l : rs.layers) {
            const ggml_tensor * rt = rs_r(mv.rs, (int) l.il);
            const ggml_tensor * st = rs_s(mv.rs, (int) l.il);
            CHECK(rt != nullptr && st != nullptr, "%s/%s: RS layer %u published but not created", arch_name, cfg.name,
                  l.il);
            if (rt == nullptr || st == nullptr) {
                continue;
            }
            CHECK(
                (int64_t) l.n_embd_r == rt->ne[0] && (int64_t) l.n_rows == rt->ne[1] && l.type_r == (int32_t) rt->type,
                "%s/%s: RS layer %u r", arch_name, cfg.name, l.il);
            CHECK(
                (int64_t) l.n_embd_s == st->ne[0] && (int64_t) l.n_rows == st->ne[1] && l.type_s == (int32_t) st->type,
                "%s/%s: RS layer %u s", arch_name, cfg.name, l.il);
            n_rs_layers_equal++;
        }
    }
}

// create_memory(no_alloc) builds the same tensors as the real memory, on size-0 dummy buffers: nothing
// is allocated, which is what a load-time measure needs. A kind with no such form throws, naming it.
static int           n_no_alloc_cases      = 0;
static int           n_no_alloc_refused    = 0;
static int           n_no_alloc_idx_layers = 0;
static std::set<int> refused_kinds;

static void check_no_alloc(const char * arch_name, const config & cfg, llama_context * ctx, const memory_view & mv) {
    const llama_model &   model = ctx->get_model();
    const llama_cparams & cp    = ctx->get_cparams();
    llama_memory_params   pm    = {
        /*.type_k    =*/cfg.type_kv,
        /*.type_v    =*/cfg.type_kv,
        /*.swa_full  =*/ctx->get_cparams().swa_full,
        /*.ctx_type  =*/cp.ctx_type,
        /*.mem_other =*/llama_get_memory(cp.ctx_other),
    };
    const llama_memory_policy pol = model.memory_policy(pm, cp);

    if (llama_memory_kind_unsupported(pol.kind) != nullptr) {
        bool        threw       = false;
        bool        unsupported = false;
        std::string what;
        try {
            delete model.create_memory(pm, cp, true);
        } catch (const llama_measure_unsupported & e) {
            threw       = true;
            unsupported = true;
            what        = e.what();
        } catch (const std::exception & e) {
            threw = true;
            what  = e.what();
        }
        CHECK(threw && what.find("no no_alloc form") != std::string::npos,
              "%s/%s: kind '%s' must refuse no_alloc by name (%s)", arch_name, cfg.name, mv.kind.c_str(),
              threw ? what.c_str() : "it built");
        // the load-time measure maps this type, and only this type, to "unsupported": a plain runtime_error
        // would be a failed measure and a refused load
        CHECK(unsupported, "%s/%s: the refusal is not a llama_measure_unsupported (%s)", arch_name, cfg.name,
              what.c_str());
        CHECK(what.find(arch_name) != std::string::npos, "%s/%s: the refusal does not name the architecture (%s)",
              arch_name, cfg.name, what.c_str());
        n_no_alloc_refused++;
        refused_kinds.insert((int) pol.kind);
        return;
    }

    std::unique_ptr<llama_memory_i> dummy(model.create_memory(pm, cp, true));
    if (mv.kind == "none") {
        // not counted in n_no_alloc_cases: a kind-none fixture would lower the floor in main() (none exists today)
        CHECK(dummy == nullptr, "%s/%s: no memory, yet no_alloc built one", arch_name, cfg.name);
        return;
    }
    CHECK(dummy != nullptr, "%s/%s: no_alloc built no memory", arch_name, cfg.name);
    if (!dummy) {
        return;
    }
    n_no_alloc_cases++;
    const memory_view dv = view_of(dummy.get());
    CHECK(dv.kind == mv.kind, "%s/%s: no_alloc built a '%s', the real memory is a '%s'", arch_name, cfg.name,
          dv.kind.c_str(), mv.kind.c_str());

    const int n_layer = (int) model.hparams.n_layer_all;
    for (int il = 0; il < n_layer; ++il) {
        bool                 amb  = false;
        const realised_layer real = realise_kv(mv, il, amb);
        const realised_layer dumb = realise_kv(dv, il, amb);
        CHECK(real.owned == dumb.owned, "%s/%s: layer %d owned %d (real) vs %d (no_alloc)", arch_name, cfg.name, il,
              (int) real.owned, (int) dumb.owned);
        if (real.owned && dumb.owned) {
            CHECK(
                real.k->ne[0] == dumb.k->ne[0] && real.k->ne[1] == dumb.k->ne[1] && real.k->ne[2] == dumb.k->ne[2] &&
                    real.k->type == dumb.k->type && real.swa == dumb.swa,
                "%s/%s: layer %d K differs: real %lld/%lld/%lld type %d swa %d, no_alloc %lld/%lld/%lld type %d swa %d",
                arch_name, cfg.name, il, (long long) real.k->ne[0], (long long) real.k->ne[1],
                (long long) real.k->ne[2], (int) real.k->type, (int) real.swa, (long long) dumb.k->ne[0],
                (long long) dumb.k->ne[1], (long long) dumb.k->ne[2], (int) dumb.k->type, (int) dumb.swa);
            CHECK(real.k->buffer != nullptr && ggml_backend_buffer_get_size(real.k->buffer) > 0,
                  "%s/%s: layer %d: the real memory has no allocated K", arch_name, cfg.name, il);
            CHECK(dumb.k->buffer != nullptr && ggml_backend_buffer_get_size(dumb.k->buffer) == 0,
                  "%s/%s: layer %d: no_alloc K is not on a size-0 buffer", arch_name, cfg.name, il);
            CHECK((real.v == nullptr) == (dumb.v == nullptr), "%s/%s: layer %d V presence", arch_name, cfg.name, il);
            if (real.v != nullptr && dumb.v != nullptr) {
                CHECK(real.v->ne[0] == dumb.v->ne[0] && real.v->ne[1] == dumb.v->ne[1] &&
                          real.v->type == dumb.v->type && dumb.v->buffer != nullptr &&
                          ggml_backend_buffer_get_size(dumb.v->buffer) == 0,
                      "%s/%s: layer %d V differs or is allocated", arch_name, cfg.name, il);
            }
        }
        if (mv.rs != nullptr && dv.rs != nullptr) {
            const ggml_tensor * rr = rs_r(mv.rs, il);
            const ggml_tensor * dr = rs_r(dv.rs, il);
            const ggml_tensor * rs = rs_s(mv.rs, il);
            const ggml_tensor * ds = rs_s(dv.rs, il);
            CHECK((rr == nullptr) == (dr == nullptr) && (rs == nullptr) == (ds == nullptr),
                  "%s/%s: layer %d RS presence", arch_name, cfg.name, il);
            if (rr && dr && rs && ds) {
                CHECK(rr->ne[0] == dr->ne[0] && rr->ne[1] == dr->ne[1] && rs->ne[0] == ds->ne[0] &&
                          rs->ne[1] == ds->ne[1] && dr->buffer != nullptr &&
                          ggml_backend_buffer_get_size(dr->buffer) == 0 && ds->buffer != nullptr &&
                          ggml_backend_buffer_get_size(ds->buffer) == 0,
                      "%s/%s: layer %d RS differs or is allocated", arch_name, cfg.name, il);
            }
            const ggml_tensor * rp = rs_p(mv.rs, il);
            const ggml_tensor * dp = rs_p(dv.rs, il);
            CHECK((rp == nullptr) == (dp == nullptr), "%s/%s: layer %d PLE row presence %d (real) vs %d (no_alloc)",
                  arch_name, cfg.name, il, (int) (rp != nullptr), (int) (dp != nullptr));
            if (rp && dp) {
                CHECK(rp->ne[0] == dp->ne[0] && rp->ne[1] == dp->ne[1] && rp->type == dp->type &&
                          dp->buffer != nullptr && ggml_backend_buffer_get_size(dp->buffer) == 0,
                      "%s/%s: layer %d PLE row differs or is allocated", arch_name, cfg.name, il);
            }
        }
    }
    CHECK((mv.rs == nullptr) == (dv.rs == nullptr), "%s/%s: recurrent half present in one memory only", arch_name,
          cfg.name);

    CHECK((mv.idx == nullptr) == (dv.idx == nullptr), "%s/%s: indexer cache present in one memory only", arch_name,
          cfg.name);
    if (mv.idx != nullptr && dv.idx != nullptr) {
        CHECK(mv.idx->get_size() == dv.idx->get_size(), "%s/%s: indexer cells %u (real) vs %u (no_alloc)", arch_name,
              cfg.name, mv.idx->get_size(), dv.idx->get_size());
        for (int il = 0; il < n_layer; ++il) {
            const ggml_tensor * real_k    = nullptr;
            const ggml_tensor * real_v    = nullptr;
            const ggml_tensor * dumb_k    = nullptr;
            const ggml_tensor * dumb_v    = nullptr;
            const bool          real_owns = mv.idx->get_layer_tensors(il, &real_k, &real_v);
            const bool          dumb_owns = dv.idx->get_layer_tensors(il, &dumb_k, &dumb_v);
            CHECK(real_owns == dumb_owns, "%s/%s: layer %d indexer owned %d (real) vs %d (no_alloc)", arch_name,
                  cfg.name, il, (int) real_owns, (int) dumb_owns);
            if (!real_owns || !dumb_owns) {
                continue;
            }
            CHECK(
                real_k->ne[0] == dumb_k->ne[0] && real_k->ne[1] == dumb_k->ne[1] && real_k->ne[2] == dumb_k->ne[2] &&
                    real_k->type == dumb_k->type && strcmp(real_k->name, dumb_k->name) == 0,
                "%s/%s: layer %d indexer K differs: real %s %lld/%lld/%lld type %d, no_alloc %s %lld/%lld/%lld type %d",
                arch_name, cfg.name, il, real_k->name, (long long) real_k->ne[0], (long long) real_k->ne[1],
                (long long) real_k->ne[2], (int) real_k->type, dumb_k->name, (long long) dumb_k->ne[0],
                (long long) dumb_k->ne[1], (long long) dumb_k->ne[2], (int) dumb_k->type);
            CHECK(real_k->buffer != nullptr && ggml_backend_buffer_get_size(real_k->buffer) > 0,
                  "%s/%s: layer %d: the real memory has no allocated indexer K", arch_name, cfg.name, il);
            CHECK(dumb_k->buffer != nullptr && ggml_backend_buffer_get_size(dumb_k->buffer) == 0,
                  "%s/%s: layer %d: no_alloc indexer K is not on a size-0 buffer", arch_name, cfg.name, il);
            CHECK(real_v == nullptr && dumb_v == nullptr, "%s/%s: layer %d: an indexer V exists", arch_name, cfg.name,
                  il);
            n_no_alloc_idx_layers++;
        }
    }
}

// The K-shift sub-caches. Each memory kind reports the llama_kv_cache objects whose K the context re-ropes on
// a shift: a leaf cache that can shift and has a rope, under a memory that can shift. The expectation is
// derived here from the realised caches and the memory's own get_can_shift(), not from get_shift_caches():
// a composite that forgets its own gate, a leaf that forgets the rope test and a kind that reports a cache it
// does not hold each fail it. (A mirror cache, `other != nullptr`, is not among the caches view_of reaches.)
static std::set<std::string> shift_kinds_nonempty;
static std::set<std::string> shift_kinds_checked;
static int                   n_shift_gated_off = 0;

static void check_shift_caches(const char *        arch_name,
                               const config &      cfg,
                               llama_context *     ctx,
                               const memory_view & mv) {
    llama_memory_t mem = ctx->get_memory();
    if (mem == nullptr || mv.kind == "other") {
        return;  // no memory, or a kind view_of does not walk (the unsupported ones)
    }
    std::vector<const llama_kv_cache *> got;
    mem->get_shift_caches(got);

    std::vector<const llama_kv_cache *> want;
    if (mem->get_can_shift()) {
        for (const auto & part : mv.kvs) {
            if (part.kv != nullptr && part.kv->get_can_shift() &&
                ctx->get_model().hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
                want.push_back(part.kv);
            }
        }
    } else {
        n_shift_gated_off++;
    }

    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    CHECK(got == want, "%s/%s kind=%s: get_shift_caches reports %zu caches, the realised memory has %zu that shift",
          arch_name, cfg.name, mv.kind.c_str(), got.size(), want.size());
    for (const llama_kv_cache * kv : got) {
        const bool held = std::any_of(mv.kvs.begin(), mv.kvs.end(), [&](const kv_part & p) { return p.kv == kv; });
        CHECK(held, "%s/%s: a shift cache is not one of the memory's caches", arch_name, cfg.name);
    }
    if (mv.kind == "recurrent") {
        CHECK(got.empty(), "%s/%s: a pure recurrent memory reports %zu shift caches", arch_name, cfg.name, got.size());
    }
    shift_kinds_checked.insert(mv.kind);
    if (!got.empty()) {
        shift_kinds_nonempty.insert(mv.kind);
    }
}

// Which kinds have no no_alloc form is one fact, in llama_memory_kind_unsupported. This is its table by kind:
// the four kinds whose caches hold tensors the shape structs have no place for, and every other kind.
static void check_unsupported_table() {
    static const struct {
        llama_memory_kind kind;
        bool              unsupported;
    } k_table[] = {
        { LLAMA_MEMORY_KIND_NONE,        false },
        { LLAMA_MEMORY_KIND_KV,          false },
        { LLAMA_MEMORY_KIND_ISWA,        false },
        { LLAMA_MEMORY_KIND_MSA,         true  },
        { LLAMA_MEMORY_KIND_DSA,         true  },
        { LLAMA_MEMORY_KIND_DSA_ISWA,    true  },
        { LLAMA_MEMORY_KIND_DSV4,        true  },
        { LLAMA_MEMORY_KIND_RECURRENT,   false },
        { LLAMA_MEMORY_KIND_HYBRID,      false },
        { LLAMA_MEMORY_KIND_HYBRID_ISWA, false },
        { LLAMA_MEMORY_KIND_HYBRID_IDX,  false },
    };

    static_assert(sizeof(k_table) / sizeof(k_table[0]) == 11, "a new memory kind needs a row here and a decision");
    std::set<std::string> names;
    for (const auto & row : k_table) {
        const char * name = llama_memory_kind_unsupported(row.kind);
        CHECK((name != nullptr) == row.unsupported, "kind %d: unsupported=%d, expected %d", (int) row.kind,
              name != nullptr, (int) row.unsupported);
        if (name != nullptr) {
            CHECK(names.insert(name).second, "kind %d: the name '%s' repeats", (int) row.kind, name);
        }
    }
}

int main() {
    FILE *       dump      = nullptr;
    const char * dump_path = getenv("LLAMA_LAYER_SHAPES_DUMP");
    if (dump_path != nullptr && dump_path[0] != '\0') {
        dump = fopen(dump_path, "w");
    }

    check_unsupported_table();
    int n_built    = 0;
    int n_kv_cases = 0;
    int n_rs_cases = 0;
    for (const auto & ac : k_archs) {
        const char * arch_name = llm_arch_name(ac.arch);
        for (const auto & cfg : k_configs) {
            fixture    fx;
            const bool built = build_fixture(fx, ac.arch, cfg);
            if (!built) {
                // a fixture the generator cannot make is a skip, except for the architectures the gate rests on
                if (ac.must) {
                    CHECK(false, "%s/%s: the fixture did not build", arch_name, cfg.name);
                } else {
                    fprintf(stderr, "  SKIP %s/%s: fixture did not build\n", arch_name, cfg.name);
                }
                continue;
            }
            n_built++;
            llama_context *   ctx = fx.ctx.get();
            const memory_view mv  = view_of(ctx->get_memory());
            n_kv_cases += mv.kvs.empty() ? 0 : 1;
            n_rs_cases += mv.rs != nullptr ? 1 : 0;
            dump_arch(dump, arch_name, cfg, mv, (int) ctx->get_model().hparams.n_layer_all);
            check_shapes(arch_name, cfg, ctx, mv);
            check_no_alloc(arch_name, cfg, ctx, mv);
            check_kv_owners(arch_name, cfg, ctx, mv);
            check_shift_caches(arch_name, cfg, ctx, mv);
        }
    }
    if (dump != nullptr) {
        fclose(dump);
    }

    // a run that built nothing checked nothing
    CHECK(n_built >= 21, "only %d (arch, config) cases built; the must-cover set alone is 21", n_built);
    // 42 is the full count, not a margin: 20 archs x 3 configs = 60 cases; the 6 archs that reach a kind with no
    // no_alloc form (deepseek32, glm_dsa, hy_v4: DSA; dots3note: DSA_ISWA; minimax_m3: MSA; deepseek4: DSV4) refuse
    // in all 3 configs, 18 cases; 60 - 18 = 42. A SKIP of any non-refused fixture, must=false or not, trips this
    // VOID by design.
    // The count also excludes kind-none fixtures (check_no_alloc returns before counting them), and there are none:
    // llama_model::memory_policy sets LLAMA_MEMORY_KIND_NONE only for the encoder and diffusion archs of its first
    // case list (bert family, wavtokenizer-dec, gemma-embedding, dream, llada, llada-moe, rnd1), and no k_archs entry
    // is among them. Adding one lowers the full count by 3 and must change this floor with it.
    CHECK(n_no_alloc_cases >= 42 && n_no_alloc_refused > 0, "VOID: %d no_alloc builds and %d refusals",
          n_no_alloc_cases, n_no_alloc_refused);
    CHECK(n_kv_cases > 0 && n_rs_cases > 0, "VOID: %d KV and %d recurrent cases", n_kv_cases, n_rs_cases);
    // the indexer comparisons ran on real layers (qwen4exp is the fixture that reaches llama_memory_hybrid_idx)
    CHECK(n_idx_layers_equal > 0 && n_no_alloc_idx_layers > 0,
          "VOID: %d indexer layers compared with the shapes, %d with the no_alloc memory", n_idx_layers_equal,
          n_no_alloc_idx_layers);
    // the recurrent equality compared real layers, and the shift check saw a memory that shifts
    CHECK(n_rs_layers_equal >= 20, "VOID: only %d recurrent layers were compared with the realised r/s tensors",
          n_rs_layers_equal);
    // the owner comparison ran on real layers, and on a memory where has_kv() over-counts (a hybrid or a
    // recurrent model), which is the case the owners exist for
    CHECK(n_owner_layers_equal > 0 && n_owner_overcount_cases > 0,
          "VOID: %d layers compared with the KV owners, %d memories where has_kv() over-counts", n_owner_layers_equal,
          n_owner_overcount_cases);
    // the owners' indexer widths were compared on real indexer layers (qwen4exp reaches llama_memory_hybrid_idx)
    CHECK(n_owner_idx_layers_equal > 0, "VOID: %d indexer layers compared with the KV owners' widths",
          n_owner_idx_layers_equal);
    CHECK(shift_kinds_nonempty.count("kv") == 1 && shift_kinds_nonempty.count("iswa") == 1,
          "VOID: the shift check found no shifting cache in a plain or an iSWA memory");
    CHECK(shift_kinds_checked.count("recurrent") == 1 && shift_kinds_checked.count("hybrid") == 1,
          "VOID: the shift check never saw a recurrent and a hybrid memory");
    fprintf(stderr, "  %zu kinds refused no_alloc through create_memory; %zu kinds with shift caches; %d gated off\n",
            refused_kinds.size(), shift_kinds_nonempty.size(), n_shift_gated_off);
    fprintf(stderr, "  %d indexer layers matched the KV owners' indexer widths\n", n_owner_idx_layers_equal);

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: %d cases (%d with KV, %d with recurrent state)\n", n_built, n_kv_cases, n_rs_cases);
    return 0;
}
