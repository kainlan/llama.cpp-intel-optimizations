// H5 (structural arm), zhcn C4: init_reserve(s) must build the memory context a worst-case graph over
// s streams reads, for every memory class.
//
// A context with kv_unified=false and n_seq_max=4 owns four streams. A reserve ubatch of s sequences takes
// the mask's ne[3] from ubatch.n_seqs_unq (= s), while the K and V views take theirs from the memory
// context's slot_info. init_full() spans all four streams, so for s < 4 the mask says s and the views say
// 4: a graph no decode ever builds, measured as if it were one. init_reserve(s) spans exactly s streams,
// and init_reserve(n_stream) must equal init_full().
//
// For every arch below (one per memory class, built by the test-llama-archs tiny-model builder, CPU only)
// and every s in [1, n_seq_max] the test checks, on the reserved graph:
//   - the memory context reports success;
//   - every FLASH_ATTN_EXT node has its mask's, K's and V's ne[3] equal to s (MSA's own FA call excepted, see
//     check_attention_streams; an MSA memory's cache K views are checked instead);
//   - the logits have n_outputs rows, n_outputs = min(s * floor(n_ubatch / s), n_outputs_max);
// and, for s == n_seq_max, that the graph equals the one init_full() builds;
// and for dsv4 also that the compressor plans' n_stream is s and that the csa, hca and lid K views have
// ne[3] == s (dsv4's arbiter: when this fails, init_reserve(s < n_stream) is instead the named refusal).
//
// A class counts only if some arch actually instantiated it (dynamic_cast on the memory), so a builder
// change that silently turns one arch into another memory class fails the coverage line instead of
// shrinking the test. This test loads models; it is lead-run, pinned to the discrete cards:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-context-measure-h5

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "test-tiny-model.h"

#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-graph.h"
#include "../src/llama-kv-cache-dsa-iswa.h"
#include "../src/llama-kv-cache-dsa.h"
#include "../src/llama-kv-cache-dsv4.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-kv-cache-msa.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-memory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                                \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            n_failed++;                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);           \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
        }                                                                               \
    } while (0)

static constexpr uint32_t n_seq_max = 4;
static constexpr uint32_t n_ubatch  = 64;
// below n_ubatch, so the output cap of the n_outputs rule is exercised (it defaults to n_batch, which would make
// min(n_tokens, n_outputs_max) always n_tokens)
static constexpr uint32_t n_outputs_max = 24;

enum mem_class {
    MEM_KV,
    MEM_ISWA,
    MEM_DSA,
    MEM_DSA_ISWA,
    MEM_MSA,
    MEM_DSV4,
    MEM_RECURRENT,
    MEM_HYBRID,
    MEM_HYBRID_ISWA,
    MEM_HYBRID_IDX,
    MEM_NONE,
};

static const char * mem_class_name(mem_class c) {
    switch (c) {
        case MEM_KV:           return "kv";
        case MEM_ISWA:         return "iswa";
        case MEM_DSA:          return "dsa";
        case MEM_DSA_ISWA:     return "dsa_iswa";
        case MEM_MSA:          return "msa";
        case MEM_DSV4:         return "dsv4";
        case MEM_RECURRENT:    return "recurrent";
        case MEM_HYBRID:       return "hybrid";
        case MEM_HYBRID_ISWA:  return "hybrid_iswa";
        case MEM_HYBRID_IDX:   return "hybrid_idx";
        case MEM_NONE:         return "none";
    }
    return "?";
}

static mem_class classify(llama_memory_i * mem) {
    if (dynamic_cast<llama_kv_cache_dsv4 *>(mem))        { return MEM_DSV4; }
    if (dynamic_cast<llama_kv_cache_dsa_iswa *>(mem))    { return MEM_DSA_ISWA; }
    if (dynamic_cast<llama_kv_cache_dsa *>(mem))         { return MEM_DSA; }
    if (dynamic_cast<llama_kv_cache_msa *>(mem))         { return MEM_MSA; }
    if (dynamic_cast<llama_memory_hybrid_idx *>(mem))    { return MEM_HYBRID_IDX; }
    if (dynamic_cast<llama_memory_hybrid_iswa *>(mem))   { return MEM_HYBRID_ISWA; }
    if (dynamic_cast<llama_memory_hybrid *>(mem))        { return MEM_HYBRID; }
    if (dynamic_cast<llama_memory_recurrent *>(mem))     { return MEM_RECURRENT; }
    if (dynamic_cast<llama_kv_cache_iswa *>(mem))        { return MEM_ISWA; }
    if (dynamic_cast<llama_kv_cache *>(mem))             { return MEM_KV; }
    return MEM_NONE;
}

// the classes every run must have instantiated; hybrid_iswa has no tiny-model arch that reaches it, and
// gate 24 (textual) is what covers its init_reserve
static const mem_class required_classes[] = {
    MEM_KV, MEM_ISWA, MEM_DSA, MEM_DSA_ISWA, MEM_MSA, MEM_DSV4, MEM_RECURRENT, MEM_HYBRID, MEM_HYBRID_IDX,
};

struct arch_case {
    llm_arch arch;
    const char * name;
};

static const arch_case arch_cases[] = {
    { LLM_ARCH_LLAMA,       "llama"       },
    { LLM_ARCH_GEMMA3,      "gemma3"      },
    { LLM_ARCH_DEEPSEEK32,  "deepseek32"  },
    { LLM_ARCH_DOTS3NOTE,   "dots3note"   },
    { LLM_ARCH_MINIMAX_M3,  "minimax_m3"  },
    { LLM_ARCH_DEEPSEEK4,   "deepseek4"   },
    { LLM_ARCH_MAMBA,       "mamba"       },
    { LLM_ARCH_QWEN3NEXT,   "qwen3next"   },
    { LLM_ARCH_QWEN4EXP,    "qwen4exp"    },
};

// the graph's shape, as a comparable string: node count, then op and extents of every node
static std::string graph_signature(ggml_cgraph * gf) {
    std::string sig;
    const int n = ggml_graph_n_nodes(gf);
    sig += std::to_string(n) + ":";
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        char buf[96];
        snprintf(buf, sizeof(buf), "%d,%lld,%lld,%lld,%lld;", (int) t->op,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        sig += buf;
    }
    return sig;
}

// every FLASH_ATTN_EXT must agree on one stream count across mask, K and V; returns the count of such nodes.
//
// MSA's own FA call (build_attn_msa_fa, node name "msa_fattn") is not a stream-axis node: it maps the GQA groups
// (and, at decode, the streams) onto the FA sequence dim, so its K, V and mask ne[3] is HKV (batch) or HKV * ns
// (decode), whatever s is. The stream count of an MSA layer is the cache views' (checked in check_msa, and by
// the graph's own GGML_ASSERT(k->ne[3] == ns)), so those nodes are only checked for a consistent channel axis
// and counted into *n_msa_fa.
static int check_attention_streams(ggml_cgraph * gf, uint32_t s, const char * what, int n_head_kv, int * n_msa_fa) {
    int n_fa = 0;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        if (strncmp(t->name, "msa_fattn", 9) == 0) {
            (*n_msa_fa)++;
            const int64_t c = t->src[1]->ne[3];
            CHECK(t->src[2]->ne[3] == c && (t->src[3] == nullptr || t->src[3]->ne[3] == c),
                    "%s: node %d (%s) K, V and mask disagree on the channel axis", what, i, t->name);
            CHECK(c == n_head_kv || c == (int64_t) n_head_kv * s,
                    "%s: node %d (%s) channel axis %lld is neither HKV (%d) nor HKV * s", what, i, t->name, (long long) c, n_head_kv);
            continue;
        }
        n_fa++;
        const ggml_tensor * k    = t->src[1];
        const ggml_tensor * v    = t->src[2];
        const ggml_tensor * mask = t->src[3];
        CHECK(k->ne[3] == (int64_t) s, "%s: node %d (%s) K ne[3] = %lld, want %u", what, i, t->name, (long long) k->ne[3], s);
        CHECK(v->ne[3] == (int64_t) s, "%s: node %d (%s) V ne[3] = %lld, want %u", what, i, t->name, (long long) v->ne[3], s);
        if (mask != nullptr) {
            CHECK(mask->ne[3] == (int64_t) s, "%s: node %d (%s) mask ne[3] = %lld, want %u", what, i, t->name,
                    (long long) mask->ne[3], s);
        }
    }
    return n_fa;
}

// msa: the base and indexer caches' K views carry the stream count
static void check_msa(llama_memory_i * mem, llama_memory_context_i * mctx, uint32_t s) {
    auto * msa_ctx = dynamic_cast<llama_kv_cache_msa_context *>(mctx);
    auto * msa_mem = dynamic_cast<llama_kv_cache_msa *>(mem);
    CHECK(msa_ctx != nullptr && msa_mem != nullptr, "s=%u: the reserve context of an msa memory is not an msa context", s);
    if (msa_ctx == nullptr || msa_mem == nullptr) {
        return;
    }

    ggml_init_params ip = { /*.mem_size =*/ 1u << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * gctx = ggml_init(ip);

    struct part {
        const char *                    name;
        const llama_kv_cache_context *  ctx;
        std::vector<uint32_t>           layers;
    };
    const part parts[] = {
        { "base", msa_ctx->get_base(), msa_mem->get_base()->get_layer_ids() },
        { "idx",  msa_ctx->get_idx(),  msa_mem->get_idx()->get_layer_ids()  },
    };
    for (const part & p : parts) {
        CHECK(p.ctx != nullptr && !p.layers.empty(), "s=%u: the msa %s cache has no context or layer, its K view cannot be checked", s, p.name);
        if (p.ctx == nullptr || p.layers.empty()) {
            continue;
        }
        const ggml_tensor * k = p.ctx->get_k(gctx, (int32_t) p.layers[0]);
        CHECK(k->ne[3] == (int64_t) s, "s=%u: msa %s K view ne[3] = %lld", s, p.name, (long long) k->ne[3]);
    }
    ggml_free(gctx);
}

static llama_model_ptr load_model(const arch_case & ac) {
    gguf_context_ptr gguf = get_gguf_ctx(ac.arch, moe_mandatory(ac.arch));

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback  = silent_model_load_progress;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices         = no_devices; // CPU only: the llama-archs builder's empty device list
    mp.n_gpu_layers    = 0;
    mp.load_mode       = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;

    tensor_data_params tp = { 1234, 0.1f };
    return llama_model_ptr(llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp));
}

static llama_context_ptr make_context(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256 * n_seq_max;
    cp.n_batch         = n_ubatch;
    cp.n_ubatch        = n_ubatch;
    cp.n_outputs_max   = n_outputs_max;
    cp.n_seq_max       = n_seq_max;
    cp.kv_unified      = false;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return llama_context_ptr(llama_init_from_model(model, cp));
}

static void check_dsv4(llama_context * lctx, llama_memory_context_i * mctx, uint32_t s, uint32_t n_tokens) {
    auto * dctx = dynamic_cast<llama_kv_cache_dsv4_context *>(mctx);
    CHECK(dctx != nullptr, "s=%u: the reserve context of a dsv4 memory is not a dsv4 context", s);
    if (dctx == nullptr) {
        return;
    }

    llama_batch_allocr balloc(1);
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens / s, s);

    CHECK(dctx->get_csa_plan(ubatch).n_stream == (int64_t) s, "s=%u: csa plan n_stream = %lld", s,
            (long long) dctx->get_csa_plan(ubatch).n_stream);
    CHECK(dctx->get_hca_plan(ubatch).n_stream == (int64_t) s, "s=%u: hca plan n_stream = %lld", s,
            (long long) dctx->get_hca_plan(ubatch).n_stream);
    CHECK(dctx->get_lid_plan(ubatch).n_stream == (int64_t) s, "s=%u: lid plan n_stream = %lld", s,
            (long long) dctx->get_lid_plan(ubatch).n_stream);

    ggml_init_params ip = { /*.mem_size =*/ 1u << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * gctx = ggml_init(ip);

    struct part {
        const char *                              name;
        const llama_kv_cache_dsv4_comp_context *  ctx;
        std::vector<uint32_t>                     layers;
    };
    auto * mem = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(lctx));
    const part parts[] = {
        { "csa", dctx->get_csa(), mem->get_csa()->get_layer_ids() },
        { "hca", dctx->get_hca(), mem->get_hca()->get_layer_ids() },
        { "lid", dctx->get_lid(), mem->get_lid()->get_layer_ids() },
    };
    for (const part & p : parts) {
        CHECK(p.ctx != nullptr, "s=%u: no %s comp context", s, p.name);
        CHECK(!p.layers.empty(), "s=%u: the %s cache has no layer, so its K view cannot be checked", s, p.name);
        if (p.ctx == nullptr || p.layers.empty()) {
            continue;
        }
        const ggml_tensor * k = p.ctx->get_k(gctx, (int32_t) p.layers[0]);
        CHECK(k->ne[3] == (int64_t) s, "s=%u: %s K view ne[3] = %lld", s, p.name, (long long) k->ne[3]);
    }

    // the raw (SWA) context's K view: read by dsv4's raw attention, which need not be a FLASH_ATTN_EXT that the
    // node loop would see
    const llama_kv_cache_dsv4_raw_context * raw = dctx->get_raw();
    const std::vector<uint32_t> raw_layers = mem->get_raw()->get_swa()->get_layer_ids();
    CHECK(raw != nullptr && !raw_layers.empty(), "s=%u: no raw context or no raw SWA layer, its K view cannot be checked", s);
    if (raw != nullptr && !raw_layers.empty()) {
        const ggml_tensor * k = raw->get_k(gctx, (int32_t) raw_layers[0]);
        CHECK(k->ne[3] == (int64_t) s, "s=%u: raw K view ne[3] = %lld", s, (long long) k->ne[3]);
    }
    ggml_free(gctx);
}

static void run_arch(const arch_case & ac, std::map<mem_class, int> & covered) {
    fprintf(stderr, "--- %s\n", ac.name);

    llama_model_ptr model = load_model(ac);
    CHECK(model != nullptr, "%s: the tiny model did not load", ac.name);
    if (!model) {
        return;
    }
    llama_context_ptr lctx_ptr = make_context(model.get());
    CHECK(lctx_ptr != nullptr, "%s: the context did not build", ac.name);
    if (!lctx_ptr) {
        return;
    }
    llama_context * lctx = lctx_ptr.get();
    llama_memory_i * mem = llama_get_memory(lctx);
    CHECK(mem != nullptr, "%s: no memory", ac.name);
    if (mem == nullptr) {
        return;
    }

    const mem_class cls = classify(mem);
    covered[cls]++;
    fprintf(stderr, "%s: memory class %s\n", ac.name, mem_class_name(cls));

    std::string sig_reserve_full_streams;
    for (uint32_t s = 1; s <= n_seq_max; s++) {
        char what[64];
        snprintf(what, sizeof(what), "%s s=%u", ac.name, s);

        llama_memory_context_ptr mctx = mem->init_reserve(s);
        CHECK(mctx != nullptr, "%s: init_reserve returned null", what);
        if (!mctx) {
            continue;
        }
        CHECK(mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "%s: status %d", what, (int) mctx->get_status());

        // the caller's rule: floor(n_ubatch / s) tokens per sequence, outputs capped at n_outputs_max
        const uint32_t n_tokens  = s * (n_ubatch / s);
        const uint32_t n_outputs = n_tokens < n_outputs_max ? n_tokens : n_outputs_max;

        ggml_cgraph * gf = lctx->graph_reserve(n_tokens, s, n_outputs, mctx.get());
        CHECK(gf != nullptr, "%s: graph_reserve refused", what);
        if (gf == nullptr) {
            continue;
        }

        int n_msa_fa = 0;
        const int n_fa = check_attention_streams(gf, s, what, llama_model_n_head_kv(model.get()), &n_msa_fa);
        const bool has_attention = cls != MEM_RECURRENT;
        CHECK(!has_attention || n_fa + n_msa_fa > 0, "%s: no FLASH_ATTN_EXT node, the stream check is vacuous", what);
        if (!has_attention && s == 1) {
            // a recurrent memory ignores s (its init_reserve is init_full), so nothing here is stream-sensitive: this
            // arm only proves init_reserve(s) builds a graph at every s. It is not evidence about stream counts.
            fprintf(stderr, "%s: NOTE recurrent memory, the stream-axis checks are vacuous by design (init_reserve ignores s)\n", ac.name);
        }
        CHECK(cls != MEM_MSA || n_msa_fa > 0, "%s: no msa_fattn node, the msa graph was not the sparse one", what);

        const ggml_tensor * logits = lctx->get_gf_res_reserve()->get_logits();
        CHECK(logits != nullptr && logits->ne[1] == (int64_t) n_outputs, "%s: logits rows = %lld, want %u", what,
                logits ? (long long) logits->ne[1] : -1LL, n_outputs);

        if (cls == MEM_DSV4) {
            check_dsv4(lctx, mctx.get(), s, n_tokens);
        }
        if (cls == MEM_MSA) {
            check_msa(mem, mctx.get(), s);
        }

        if (s == n_seq_max) {
            sig_reserve_full_streams = graph_signature(gf);
        }
    }

    // init_reserve(n_stream) is init_full(), as far as the graph can tell
    {
        llama_memory_context_ptr mctx = mem->init_full();
        CHECK(mctx != nullptr, "%s: init_full returned null", ac.name);
        if (mctx) {
            ggml_cgraph * gf = lctx->graph_reserve(n_ubatch, n_seq_max, n_outputs_max, mctx.get());
            CHECK(gf != nullptr, "%s: graph_reserve over init_full refused", ac.name);
            if (gf) {
                CHECK(!sig_reserve_full_streams.empty(), "%s: no init_reserve(n_stream) signature, the comparison is vacuous", ac.name);
                CHECK(graph_signature(gf) == sig_reserve_full_streams,
                        "%s: init_reserve(%u) and init_full() build different graphs", ac.name, n_seq_max);
            }
        }
    }
}

int main() {
    std::map<mem_class, int> covered;
    for (const arch_case & ac : arch_cases) {
        run_arch(ac, covered);
    }

    for (mem_class c : required_classes) {
        CHECK(covered[c] > 0, "no arch instantiated the %s memory class: its init_reserve was not exercised", mem_class_name(c));
    }
    fprintf(stderr, "coverage:");
    for (const auto & kv : covered) {
        fprintf(stderr, " %s=%d", mem_class_name(kv.first), kv.second);
    }
    fprintf(stderr, "\n");

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: init_reserve(s) spans s streams for every memory class reached\n");
    return 0;
}
