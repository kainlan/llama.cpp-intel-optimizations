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
//   - the recurrent layers are exactly the offloaded layers, with the published row widths and rows;
//   - the memory kinds llama does not model report themselves unsupported by name, and publish nothing.
//
// LLAMA_LAYER_SHAPES_DUMP=<path> writes the realised per-layer tensors as text, so a refactor of how the
// memory is built can be diffed against the tree before it.

#include "../src/llama-context.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-layer-shapes.h"
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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
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
    const llama_memory_recurrent * rs = nullptr;
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
        // the indexer cache holds K the shapes have no place for: this kind is not modelled
        v.kind = "hybrid_idx";
        v.kvs.push_back({ m->get_mem_attn(), false });
        v.rs = m->get_mem_recr();
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

    const bool modelled = mv.kind == "kv" || mv.kind == "iswa" || mv.kind == "hybrid" || mv.kind == "hybrid_iswa" ||
                          mv.kind == "recurrent" || mv.kind == "none";
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

    // recurrent state: exactly the offloaded layers the memory created r/s for
    if (mv.rs == nullptr) {
        CHECK(rs.layers.empty(), "%s/%s: no recurrent memory but %zu RS layers", arch_name, cfg.name, rs.layers.size());
    } else {
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
        }
    }
}

// create_memory(no_alloc) builds the same tensors as the real memory, on size-0 dummy buffers: nothing
// is allocated, which is what a load-time measure needs. A kind with no such form throws, naming it.
static int n_no_alloc_cases   = 0;
static int n_no_alloc_refused = 0;

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
        bool        threw = false;
        std::string what;
        try {
            delete model.create_memory(pm, cp, true);
        } catch (const std::exception & e) {
            threw = true;
            what  = e.what();
        }
        CHECK(threw && what.find("no no_alloc form") != std::string::npos,
              "%s/%s: kind '%s' must refuse no_alloc by name (%s)", arch_name, cfg.name, mv.kind.c_str(),
              threw ? what.c_str() : "it built");
        n_no_alloc_refused++;
        return;
    }

    std::unique_ptr<llama_memory_i> dummy(model.create_memory(pm, cp, true));
    if (mv.kind == "none") {
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
        }
    }
    CHECK((mv.rs == nullptr) == (dv.rs == nullptr), "%s/%s: recurrent half present in one memory only", arch_name,
          cfg.name);
}

int main() {
    FILE *       dump      = nullptr;
    const char * dump_path = getenv("LLAMA_LAYER_SHAPES_DUMP");
    if (dump_path != nullptr && dump_path[0] != '\0') {
        dump = fopen(dump_path, "w");
    }

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
        }
    }
    if (dump != nullptr) {
        fclose(dump);
    }

    // a run that built nothing checked nothing
    CHECK(n_built >= 21, "only %d (arch, config) cases built; the must-cover set alone is 21", n_built);
    CHECK(n_no_alloc_cases >= 39 && n_no_alloc_refused > 0, "VOID: %d no_alloc builds and %d refusals",
          n_no_alloc_cases, n_no_alloc_refused);
    CHECK(n_kv_cases > 0 && n_rs_cases > 0, "VOID: %d KV and %d recurrent cases", n_kv_cases, n_rs_cases);

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: %d cases (%d with KV, %d with recurrent state)\n", n_built, n_kv_cases, n_rs_cases);
    return 0;
}
