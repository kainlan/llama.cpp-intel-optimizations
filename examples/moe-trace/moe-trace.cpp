// llama-moe-trace: records which experts the MoE router selects, per evaluated token and layer.
//
// The ggml eval callback is asked about every node of the graph; this one says yes only to the
// router's top-k output (`ffn_moe_topk-<layer>`, named in llm_graph_context::build_moe_ffn) and
// copies just that I32 tensor. It reads the graph and never writes it, so the logits are those of
// an untraced run. Expectation, not measured: with a callback set the scheduler splits the graph
// after each top-k node, so a backend cannot fuse across it; that changes speed and may in
// principle change float summation order, so compare logits against an untraced run before
// relying on bit-identity.
//
// The output is a compact binary trace read by scripts/moe-cache-sim.py (format documented there):
//     "MOETRC01", u32 header_len, header JSON,
//     records of { u32 step, u16 layer, u16 k, u32 n_tokens, u8 phase, u8 flags, u16 pad },
//     each followed by n_tokens * k u16 expert ids.
// A new step starts when the layer index does not increase (one step = one ubatch evaluation).
// Within one layer, tokens arrive in order, so the n-th row seen for a layer is that layer's n-th
// token. The last layer of a prompt ubatch may carry only the rows that produce output.
//
//   llama-moe-trace -m model.gguf -f prompt.txt -n 256 -c 9216
//       --trace-out code-0.moetrace --trace-set code --trace-id code-0
//
// The prompt is tokenized with special tokens parsed (so a chat-templated prompt file works as
// written) and decoded in chunks of n_batch (phase 0); then up to -n tokens are generated greedily,
// one decode each (phase 1), stopping early at an end-of-generation token. Nothing is sampled
// randomly: two runs give the same trace.
//
// --selftest-write FILE writes a synthetic trace through the same callback with no model, so the
// record layout and the strided-view handling are checkable without loading anything.

#include "arg.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char         TRACE_MAGIC[8] = { 'M', 'O', 'E', 'T', 'R', 'C', '0', '1' };
constexpr uint8_t      PHASE_PROMPT   = 0;
constexpr uint8_t      PHASE_DECODE   = 1;
constexpr const char * TOPK_PREFIX    = "ffn_moe_topk-";

#pragma pack(push, 1)

struct trace_record {
    uint32_t step;
    uint16_t layer;
    uint16_t k;
    uint32_t n_tokens;
    uint8_t  phase;
    uint8_t  flags;
    uint16_t pad;
};

#pragma pack(pop)
static_assert(sizeof(trace_record) == 16, "trace record layout");

struct trace_state {
    FILE *                file       = nullptr;
    uint8_t               phase      = PHASE_PROMPT;
    bool                  have_last  = false;
    int                   last_layer = -1;
    uint32_t              step       = 0;
    uint64_t              records    = 0;
    std::vector<uint8_t>  raw;
    std::vector<uint16_t> ids;
    bool                  failed     = false;
    bool                  force_copy = false;  // selftest: read through ggml_backend_tensor_get even on host buffers
};

std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if ((unsigned char) c < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

bool write_header(FILE * f, const std::string & json) {
    const uint32_t len = (uint32_t) json.size();
    return fwrite(TRACE_MAGIC, 1, sizeof(TRACE_MAGIC), f) == sizeof(TRACE_MAGIC) &&
           fwrite(&len, sizeof(len), 1, f) == 1 && fwrite(json.data(), 1, json.size(), f) == json.size();
}

// -1 when `name` is not the router's top-k tensor
int topk_layer(const char * name) {
    const size_t n = strlen(TOPK_PREFIX);
    if (strncmp(name, TOPK_PREFIX, n) != 0) {
        return -1;
    }
    char *     end = nullptr;
    const long il  = strtol(name + n, &end, 10);
    if (end == name + n || *end != '\0' || il < 0 || il > 65535) {
        return -1;
    }
    return (int) il;
}

// ggml_backend_sched_eval_callback
bool trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    const int il = topk_layer(t->name);
    if (ask) {
        // Asking about a node makes the scheduler split the graph after it, so only the top-k
        // nodes are claimed; every other node is evaluated without interruption.
        return il >= 0;
    }
    if (il < 0) {
        return true;  // not ours (never reached through the scheduler): keep computing
    }

    auto * st = (trace_state *) user_data;
    if (st->failed) {
        return true;
    }
    if (t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(int32_t) || t->ne[0] <= 0 ||
        t->ne[0] > 65535 || t->ne[1] <= 0) {
        LOG_ERR("%s: unexpected top-k tensor %s (type %d, ne %lld %lld %lld)\n", __func__, t->name, (int) t->type,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
        st->failed = true;
        return true;
    }

    const size_t k    = (size_t) t->ne[0];
    const size_t nt   = (size_t) t->ne[1];
    // rows may be a strided view of the full argsort result: copy the covering span once
    const size_t span = t->nb[1] * (nt - 1) + k * sizeof(int32_t);

    const uint8_t * data = nullptr;
    if (t->buffer == nullptr || (ggml_backend_buffer_is_host(t->buffer) && !st->force_copy)) {
        data = (const uint8_t *) t->data;
    } else {
        st->raw.resize(span);
        ggml_backend_tensor_get(t, st->raw.data(), 0, span);
        data = st->raw.data();
    }

    st->ids.resize(k * nt);
    for (size_t r = 0; r < nt; ++r) {
        const int32_t * row = (const int32_t *) (data + r * t->nb[1]);
        for (size_t c = 0; c < k; ++c) {
            if (row[c] < 0 || row[c] > 65535) {
                LOG_ERR("%s: expert id %d out of u16 range in %s\n", __func__, (int) row[c], t->name);
                st->failed = true;
                return true;
            }
            st->ids[r * k + c] = (uint16_t) row[c];
        }
    }

    if (st->have_last && il <= st->last_layer) {
        st->step++;
    }
    st->have_last  = true;
    st->last_layer = il;

    trace_record rec = {};
    rec.step         = st->step;
    rec.layer        = (uint16_t) il;
    rec.k            = (uint16_t) k;
    rec.n_tokens     = (uint32_t) nt;
    rec.phase        = st->phase;
    if (fwrite(&rec, sizeof(rec), 1, st->file) != 1 ||
        fwrite(st->ids.data(), sizeof(uint16_t), st->ids.size(), st->file) != st->ids.size()) {
        LOG_ERR("%s: write failed\n", __func__);
        st->failed = true;
        return true;
    }
    st->records++;
    return true;
}

// Writes a synthetic trace through trace_cb_eval: layer 7 as a strided [8, 3] view of a
// [256, 3] I32 tensor (value r * 11 + c), layer 9 as a contiguous [4, 2] tensor (value r * 5 + c),
// then layer 7 again as [8, 1] (value c), which must start a second step; then layer 11 as a
// strided [6, 2] view (value r * 13 + c) of a [64, 2] tensor held in a real ggml buffer and read
// through ggml_backend_tensor_get (st.force_copy), the path a device-resident tensor takes.
//
// What this cannot reach: a buffer that is NOT host memory. The only way to build one is the
// private ggml-backend-impl.h (ggml_backend_buffer_init with a custom interface), which an
// example must not include, and no public CPU buffer type reports is_host == false. So the
// device's own get_tensor is not run here; the span arithmetic, the view offset handling and the
// copy-then-decode path are, and the first real SYCL capture is the check for the rest.
int selftest_write(const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    if (!write_header(f, "{\"selftest\": true, \"set\": \"selftest\", \"n_expert\": 256}")) {
        fclose(f);
        return 1;
    }

    ggml_init_params ip  = { 4u * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        fclose(f);
        return 1;
    }

    ggml_tensor * parent = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 256, 3);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 256; ++c) {
            ((int32_t *) parent->data)[r * 256 + c] = r * 11 + c;
        }
    }
    ggml_tensor * view = ggml_view_2d(ctx, parent, 8, 3, parent->nb[1], 0);
    ggml_set_name(view, "ffn_moe_topk-7");

    ggml_tensor * flat = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 4; ++c) {
            ((int32_t *) flat->data)[r * 4 + c] = r * 5 + c;
        }
    }
    ggml_set_name(flat, "ffn_moe_topk-9");

    ggml_tensor * one = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
    for (int c = 0; c < 8; ++c) {
        ((int32_t *) one->data)[c] = c;
    }
    ggml_set_name(one, "ffn_moe_topk-7");

    ggml_tensor * other = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(other, "ffn_moe_out-7");

    // a second context whose tensors live in a real (CPU) ggml buffer
    ggml_init_params      ip2     = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *        ctx2    = ggml_init(ip2);
    ggml_tensor *         parent2 = ctx2 ? ggml_new_tensor_2d(ctx2, GGML_TYPE_I32, 64, 2) : nullptr;
    ggml_tensor *         view2   = parent2 ? ggml_view_2d(ctx2, parent2, 6, 2, parent2->nb[1], 0) : nullptr;
    ggml_backend_buffer_t buf2 =
        ctx2 ? ggml_backend_alloc_ctx_tensors_from_buft(ctx2, ggml_backend_cpu_buffer_type()) : nullptr;
    if (!buf2) {
        fprintf(stderr, "selftest: cannot allocate the CPU buffer\n");
        if (ctx2) {
            ggml_free(ctx2);
        }
        ggml_free(ctx);
        fclose(f);
        return 1;
    }
    {
        std::vector<int32_t> vals(64 * 2);
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 64; ++c) {
                vals[r * 64 + c] = r * 13 + c;
            }
        }
        ggml_backend_tensor_set(parent2, vals.data(), 0, vals.size() * sizeof(int32_t));
    }
    ggml_set_name(view2, "ffn_moe_topk-11");

    // names that must not be claimed: a suffix after the layer number, or no number at all
    ggml_tensor * suffixed = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
    ggml_set_name(suffixed, "ffn_moe_topk-7 (copy)");
    ggml_tensor * bare = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
    ggml_set_name(bare, "ffn_moe_topk-");

    trace_state st;
    st.file             = f;
    st.phase            = PHASE_DECODE;
    bool          ok    = true;
    // the scheduler asks first, then hands over the data
    ggml_tensor * seq[] = { view, flat, one };
    for (ggml_tensor * t : seq) {
        ok = ok && trace_cb_eval(t, true, &st) && trace_cb_eval(t, false, &st);
    }
    ok = ok && !trace_cb_eval(other, true, &st) && !trace_cb_eval(suffixed, true, &st) &&
         !trace_cb_eval(bare, true, &st) && st.records == 3 && !st.failed;

    st.force_copy = true;
    ok = ok && trace_cb_eval(view2, true, &st) && trace_cb_eval(view2, false, &st) && st.records == 4 && !st.failed;

    ggml_backend_buffer_free(buf2);
    ggml_free(ctx2);

    ggml_free(ctx);
    fclose(f);
    return ok ? 0 : 1;
}

struct trace_args {
    std::string out;
    std::string set = "unset";
    std::string id  = "unset";
    std::string selftest;
};

// the example's own flags are removed from argv before common_params_parse sees it
trace_args take_trace_args(int & argc, char ** argv) {
    trace_args a;
    int        w = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string *     dst = nullptr;
        if (arg == "--trace-out") {
            dst = &a.out;
        } else if (arg == "--trace-set") {
            dst = &a.set;
        } else if (arg == "--trace-id") {
            dst = &a.id;
        } else if (arg == "--selftest-write") {
            dst = &a.selftest;
        }
        if (dst && i + 1 < argc) {
            *dst = argv[++i];
        } else {
            argv[w++] = argv[i];
        }
    }
    argv[w] = nullptr;
    argc    = w;
    return a;
}

std::string meta_str(const llama_model * model, const char * key) {
    char          buf[256];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n >= 0 ? std::string(buf) : std::string();
}

// frees the backend on every return path after llama_backend_init, and after llama_init (declared
// later, so destroyed earlier) has released the model and context
struct backend_guard {
    backend_guard() { llama_backend_init(); }

    ~backend_guard() { llama_backend_free(); }

    backend_guard(const backend_guard &)             = delete;
    backend_guard & operator=(const backend_guard &) = delete;
};

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    trace_args ta = take_trace_args(argc, argv);
    if (!ta.selftest.empty()) {
        return selftest_write(ta.selftest.c_str());
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (ta.out.empty()) {
        LOG_ERR("%s: --trace-out FILE is required\n", __func__);
        return 1;
    }

    trace_state st;
    st.file = fopen(ta.out.c_str(), "wb");
    if (!st.file) {
        LOG_ERR("%s: cannot open %s\n", __func__, ta.out.c_str());
        return 1;
    }

    backend_guard guard;
    llama_numa_init(params.numa);

    params.cb_eval           = trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        fclose(st.file);
        return 1;
    }

    const llama_vocab *      vocab   = llama_model_get_vocab(model);
    const bool               add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens (give -p or -f)\n", __func__);
        fclose(st.file);
        return 1;
    }
    const int n_predict = params.n_predict < 0 ? 0 : params.n_predict;
    const int n_ctx     = (int) llama_n_ctx(ctx);
    if ((int) tokens.size() + n_predict > n_ctx) {
        LOG_ERR("%s: %zu prompt tokens + %d predicted exceed n_ctx %d (raise -c)\n", __func__, tokens.size(), n_predict,
                n_ctx);
        fclose(st.file);
        return 1;
    }

    const std::string arch   = meta_str(model, "general.architecture");
    std::string       header = "{\"format\": \"moetrace\", \"version\": 1";
    header += ", \"set\": \"" + json_escape(ta.set) + "\"";
    header += ", \"id\": \"" + json_escape(ta.id) + "\"";
    header += ", \"model\": \"" + json_escape(params.model.path) + "\"";
    header += ", \"arch\": \"" + json_escape(arch) + "\"";
    header += ", \"n_layer\": " + std::to_string(llama_model_n_layer(model));
    const std::string n_expert = meta_str(model, (arch + ".expert_count").c_str());
    const std::string n_used   = meta_str(model, (arch + ".expert_used_count").c_str());
    header += ", \"n_expert\": " + (n_expert.empty() ? std::string("0") : n_expert);
    header += ", \"n_expert_used\": " + (n_used.empty() ? std::string("0") : n_used);
    header += ", \"n_prompt\": " + std::to_string(tokens.size());
    header += ", \"n_predict\": " + std::to_string(n_predict);
    header += ", \"n_ctx\": " + std::to_string(n_ctx);
    header += ", \"n_batch\": " + std::to_string(llama_n_batch(ctx));
    header += ", \"n_ubatch\": " + std::to_string(llama_n_ubatch(ctx)) + "}";
    if (!write_header(st.file, header)) {
        LOG_ERR("%s: cannot write header\n", __func__);
        fclose(st.file);
        return 1;
    }

    // prompt, in chunks of n_batch
    st.phase          = PHASE_PROMPT;
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < (int) tokens.size(); i += n_batch) {
        const int n = std::min(n_batch, (int) tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("%s: prompt decode failed at token %d\n", __func__, i);
            fclose(st.file);
            return 1;
        }
    }

    // greedy generation
    st.phase             = PHASE_DECODE;
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    int n_gen = 0;
    for (; n_gen < n_predict; ++n_gen) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("%s: decode failed at generated token %d\n", __func__, n_gen);
            llama_sampler_free(smpl);
            fclose(st.file);
            return 1;
        }
    }
    llama_sampler_free(smpl);

    const bool wrote = fclose(st.file) == 0;
    LOG("moe-trace: %zu prompt tokens, %d generated, %llu records, steps %u -> %s\n", tokens.size(), n_gen,
        (unsigned long long) st.records, st.step + 1, ta.out.c_str());
    if (st.failed || !wrote || st.records == 0) {
        LOG_ERR("%s: trace is incomplete or empty (records=%llu): a model without ffn_moe_topk nodes?\n", __func__,
                (unsigned long long) st.records);
        return 1;
    }
    return 0;
}
