#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "test-skip.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Kept in its own include block: they belong to no upstream group, and adding them to one
// above makes clang-format re-sort a block this change has no business touching.
#include "test-archs-exclude.h"
#include "test-archs-table.h"
#include "test-tiny-model.h"

// The one threshold both result columns judge by. The NMSE column and the Roundtrip column
// must not disagree about what "wrong" means, and until this existed that agreement was
// maintained by five separate literals happening to match -- a property asserted in a comment
// and enforced by nothing. Editing one of them would have left the test disagreeing with
// itself about what constitutes a failure. Every site that tests or prints it reads it here.
static constexpr double nmse_gate = 1e-4;

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// Collect the distinct token positions (rows of n_vocab logits) that contain at least one
// non-finite value, formatted for a log line. A count alone cannot distinguish "the whole
// output collapsed" from "two specific positions did", and those have completely different
// causes -- the latter points at a boundary condition (first/last position of a ubatch, a
// particular routing decision), which is the single most useful thing to know before
// touching any code.
static std::string nonfinite_rows(const std::vector<float> & v, const size_t n_row) {
    if (n_row == 0 || v.size() % n_row != 0) {
        return "?";
    }
    const size_t n_col = v.size() / n_row;

    std::vector<size_t> rows;
    for (size_t i = 0; i < n_row; i++) {
        for (size_t j = 0; j < n_col; j++) {
            if (!std::isfinite(v[i*n_col + j])) {
                rows.push_back(i);
                break;
            }
        }
    }
    if (rows.empty()) {
        return "none";
    }

    const size_t n_show = std::min<size_t>(rows.size(), 16);
    std::string  ret;
    for (size_t i = 0; i < n_show; i++) {
        ret += (i == 0 ? "" : ",") + std::to_string(rows[i]);
    }
    if (rows.size() > n_show) {
        ret += ",... (" + std::to_string(rows.size()) + " rows total)";
    }
    return ret;
}

// A non-finite NMSE means the comparison itself degenerated, not that the result is "slightly off".
// The two causes need different fixes and must not be confused, so say which one it was:
//   - non-finite values in the logits => the backend produced NaN/Inf output;
//   - zero reference energy           => the CPU reference is all-zero, so mse_a_0 is 0 and the ratio is 0/0.
static std::string nmse_diagnosis(const std::vector<float> & a, const std::vector<float> & b, const size_t n_row) {
    size_t   nonfinite_a = 0;
    size_t   nonfinite_b = 0;
    double   energy_a    = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        nonfinite_a += !std::isfinite(a[i]);
        nonfinite_b += !std::isfinite(b[i]);
        energy_a    += double(a[i]) * double(a[i]);
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
        "n=%zu, non-finite CPU logits=%zu, non-finite device logits=%zu, CPU reference energy=%g",
        a.size(), nonfinite_a, nonfinite_b, energy_a);
    std::string ret(buf);
    if (nonfinite_a > 0) {
        ret += "\n  CPU    non-finite token positions: " + nonfinite_rows(a, n_row);
    }
    if (nonfinite_b > 0) {
        ret += "\n  device non-finite token positions: " + nonfinite_rows(b, n_row);
    }
    return ret;
}

// The Roundtrip column is a BIT-EXACT comparison, so it reports a single FAIL for two
// causes that need completely different fixes and are trivially told apart by magnitude:
//
//   - the weights did not survive the GGUF save/reload, so the reloaded model computes
//     something else entirely -- large, structured differences, and the reloaded model is
//     also wrong against the CPU reference;
//   - the same correct computation took a different path (different kernel, different
//     residency, non-reproducible accumulation order) -- last-bit differences, and the
//     reloaded model is still correct against the CPU reference.
//
// A bare FAIL loses exactly that distinction, which is the one thing needed before
// touching any code, so measure it.
// Report differing positions as (token, vocab) and not as a flat index. The logits vector is
// n_token rows of n_vocab, and a flat index invites the wrong decomposition: 16384 differing
// logits "first at 8192" reads as the midpoint of the VOCABULARY, when these fixtures have
// n_vocab=128 and 8192 is token 64 of 128 -- the first token of the second ubatch. Those two
// readings point at completely different subsystems, so do the division here, once.
struct logits_diff {
    size_t n_diff      = 0;
    size_t first_diff  = 0;  // flat index
    size_t first_row   = 0;  // token position
    size_t first_col   = 0;  // vocab index
    size_t n_rows_diff = 0;  // token positions with at least one differing logit
    size_t last_row    = 0;
    double max_abs     = 0.0;
    double max_rel     = 0.0;
};

static logits_diff logits_compare(const std::vector<float> & a, const std::vector<float> & b, const size_t n_col) {
    GGML_ASSERT(a.size() == b.size());
    GGML_ASSERT(n_col > 0 && a.size() % n_col == 0);
    logits_diff  ret;
    const size_t n_row = a.size() / n_col;
    for (size_t row = 0; row < n_row; row++) {
        bool row_differs = false;
        for (size_t col = 0; col < n_col; col++) {
            const size_t i = row*n_col + col;
            if (a[i] == b[i]) {
                continue;
            }
            if (ret.n_diff == 0) {
                ret.first_diff = i;
                ret.first_row  = row;
                ret.first_col  = col;
            }
            ret.n_diff++;
            row_differs = true;
            const double abs_diff = std::fabs(double(a[i]) - double(b[i]));
            const double scale    = std::max(std::fabs(double(a[i])), std::fabs(double(b[i])));
            ret.max_abs = std::max(ret.max_abs, abs_diff);
            ret.max_rel = std::max(ret.max_rel, scale > 0.0 ? abs_diff/scale : abs_diff);
        }
        if (row_differs) {
            ret.n_rows_diff++;
            ret.last_row = row;
        }
    }
    return ret;
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("                           A name or regex that matches no architecture is an error.\n");
    LOG("  -x, --exclude <arch>     Skip this architecture; repeatable. It is reported as EXCLUDED, never omitted.\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v, --verbose [N]        Show backend INFO logs; -v N sets the log verbosity level N instead\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  --nan-trace              CPU backend only: name the first graph tensors that go non-finite (needs -a <exact name>)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
    LOG("  %s -x gemma2\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;
    // Both models in the Roundtrip comparison must be loaded the same way. They were not:
    // llama_model_init_from_user() forces these two off, llama_model_load_from_file_ptr()
    // passes llama_model_default_params() straight through, where both are on. So the model
    // written by the saver came back mmap-backed and eligible for weight-repacking extra
    // buffer types, and the model it is compared against was neither.
    //
    // That is not a difference in what was saved, it is a difference in where the weights
    // ended up: llama-model-loader.cpp demotes a host buffer type to plain CPU under mmap,
    // and allocates host-side tensors directly over the mapping instead of going through
    // ggml_backend_tensor_set.
    model_params.load_mode       = LLAMA_LOAD_MODE_NONE;
    model_params.use_extra_bufts = false;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.cb_eval           = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

// speculative decoding: each step decodes a token and 0 to n_draft_max drafts, then rejects the drafts
// the logits of the token must match a decode without drafts
static double check_draft_rollback(llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & logits_ref) {
    const uint32_t n_vocab     = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens    = tokens.size();
    const uint32_t n_draft_max = 2;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_rs_seq        = n_draft_max;

    llama_context_ptr lctx(llama_init_from_model(model, ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }

    common_batch batch(lctx.get());
    for (uint32_t pos = 0; pos < n_tokens/2; pos++) {
        batch.add(tokens[pos], pos, 0, false);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> logits;
    std::vector<float> ref;
    for (uint32_t pos = n_tokens/2; pos < n_tokens; pos++) {
        batch.clear();
        batch.add(tokens[pos], pos, 0, true);
        const uint32_t n_draft = pos % (n_draft_max + 1);
        for (uint32_t i = 1; i <= n_draft; i++) {
            batch.add((tokens[pos] + i) % n_vocab, pos + i, 0, true);
        }
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode batch");
        }

        const float * logits_ith = llama_get_logits_ith(lctx.get(), 0);
        logits.insert(logits.end(), logits_ith, logits_ith + n_vocab);
        ref.insert(ref.end(), logits_ref.begin() + (size_t) pos*n_vocab, logits_ref.begin() + (size_t) (pos + 1)*n_vocab);

        if (!llama_memory_seq_rm(llama_get_memory(lctx.get()), 0, pos + 1, -1)) {
            throw std::runtime_error("failed to reject the drafts");
        }
    }

    return nmse(ref, logits);
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_KOKORO || arch == LLM_ARCH_TINY_VOCA) {
        return false; // TODO encoder without memory, a sequence must fit in one ubatch
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

// --- NaN origin tracing ---------------------------------------------------------------------
//
// A non-finite logit only says that something upstream degenerated; it does not say where. This
// walks the graph in execution order and names the FIRST tensors holding a non-finite value.
// Execution order is topological, so the first tensor named is where the non-finiteness was
// created rather than merely propagated -- its sources are printed so that can be confirmed.
//
// Deliberately CPU-only: it needs no device, so it can be iterated without a GPU and without
// the memory hazard of a full sweep.
struct nan_trace_state {
    size_t n_reported = 0;
    size_t n_scanned  = 0;
    size_t max_report = 12;
};

static bool nan_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    nan_trace_state * st = (nan_trace_state *) user_data;

    if (ask) {
        // only contiguous float tensors can be scanned as a flat array
        return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && ggml_is_contiguous(t);
    }
    if (st->n_reported >= st->max_report) {
        return true;
    }

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    st->n_scanned++;

    const int64_t ne          = ggml_nelements(t);
    int64_t       n_nonfinite = 0;
    int64_t       first_idx   = -1;
    for (int64_t i = 0; i < ne; i++) {
        const float v = t->type == GGML_TYPE_F32 ?
            ((const float *) buf.data())[i] : ggml_fp16_to_fp32(((const ggml_fp16_t *) buf.data())[i]);
        if (!std::isfinite(v)) {
            n_nonfinite++;
            if (first_idx < 0) {
                first_idx = i;
            }
        }
    }
    if (n_nonfinite == 0) {
        return true;
    }

    st->n_reported++;
    fprintf(stderr, "nan-trace: %-28s op=%-10s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
        " non-finite=%" PRId64 "/%" PRId64 " first=[%" PRId64 ",%" PRId64 ",%" PRId64 "]",
        t->name, ggml_op_name(t->op), t->ne[0], t->ne[1], t->ne[2], t->ne[3], n_nonfinite, ne,
        first_idx % t->ne[0], (first_idx / t->ne[0]) % t->ne[1], first_idx / (t->ne[0]*t->ne[1]));
    for (int i = 0; i < GGML_MAX_SRC && t->src[i] != nullptr; i++) {
        fprintf(stderr, " src%d=%s", i, t->src[i]->name);
    }
    fprintf(stderr, "\n");
    return true;
}

static int trace_nan(const llm_arch target_arch, const size_t seed, const float stdev) {
    if (target_arch == LLM_ARCH_UNKNOWN) {
        fprintf(stderr, "%s: --nan-trace requires -a/--arch with an exact architecture name\n", __func__);
        return 1;
    }

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    const bool encode = target_arch == LLM_ARCH_T5 || target_arch == LLM_ARCH_DREAM ||
        target_arch == LLM_ARCH_LLADA || target_arch == LLM_ARCH_LLADA_MOE || target_arch == LLM_ARCH_RND1;

    for (bool moe : {false, true}) {
        if (moe && !moe_implemented(target_arch)) {
            continue;
        }
        if (!moe && moe_mandatory(target_arch)) {
            continue;
        }
        fprintf(stderr, "nan-trace: %s (%s), seed %zu, CPU backend only\n",
            llama_arch_name(target_arch), moe ? "MoE" : "Dense", seed);

        gguf_context_ptr gguf_ctx = get_gguf_ctx(target_arch, moe);
        nan_trace_state  st;
        auto model_and_ctx = get_model_and_ctx(
            gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode, nullptr, 0, nan_trace_eval_cb, &st);
        const std::vector<float> logits = get_logits(
            model_and_ctx.first.get(), model_and_ctx.second.get(), tokens, encode);

        fprintf(stderr, "nan-trace: %zu tensors reported of %zu scanned; non-finite logit rows: %s\n",
            st.n_reported, st.n_scanned, nonfinite_rows(logits, tokens.size()).c_str());
    }
    return 0;
}

static int save_models(const std::string &           arch_filter,
                       const std::vector<llm_arch> & excluded_archs,
                       const size_t                  seed,
                       const float                   stdev,
                       const int                     verbosity,
                       const std::string &           dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (archs_exclude::contains(excluded_archs, arch)) {
            // stderr, not LOG_INF: the surrounding LOG_INF lines are filtered out at this
            // path's default log level, so an exclusion announced through them would be
            // invisible -- an exclude that silently excludes is the exact defect `-x` exists to
            // avoid. -o writes files rather than a table, so there is no row to carry it.
            fprintf(stderr, "%s: %s excluded by --exclude, not saved\n", __func__, llama_arch_name(arch));
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string &           arch_filter,
                         const std::vector<llm_arch> & excluded_archs,
                         const size_t                  seed,
                         const float                   stdev,
                         const int                     verbosity,
                         const char *                  target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (!dev) {
                    continue;
                }
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    // The table's layout, its emission discipline and the parser that gates it all live in
    // tests/test-archs-table.h; test-archs-table verifies THIS emitter rather than a
    // second copy of the format. The mixed-batch, causal-toggle and draft-rollback checks fold
    // into the NMSE column as "FAIL (mixed|toggle|rollback)", with their figures on stderr.
    const archs_table table = { max_arch_name_length, max_device_label_length };

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    // Rows that produced an actual NMSE comparison. A run that measures nothing must not
    // report success: the harness skips several archs outright and arch_supported() turns
    // others into an all-SKIP table, and in both cases the process used to exit 0, which
    // reads as "verified" to anything checking the status. See llama.cpp-k208.
    size_t n_measured = 0;
    // Rows that round-tripped within tolerance but not bit-for-bit. Counted so the closing
    // line can state it whether or not any row hit it -- see the legend in the header.
    size_t n_bitdiff = 0;
    common_log_flush(common_log_main());
    archs_table::emit(table.header(nmse_gate));
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (archs_exclude::contains(excluded_archs, arch)) {
            // Ordered AFTER the -a filter and BEFORE the hard skips below, and both halves of
            // that are deliberate. After -a, because an architecture the run never had in scope
            // was not "excluded" by anything the user asked for. Before the hard skips, because
            // `-x <arch>` should say EXCLUDED rather than inherit their silence: those
            // `continue`s emit no row at all.
            //
            // Emitted through archs_table::row() like every other status, whole and in one
            // stdio call, with the log drained first (llama.cpp-to9m).
            common_log_flush(common_log_main());
            archs_table::emit(archs_exclude::row(table, llama_arch_name(arch)));
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // Which test case is running, so that a crash during model loading or inference is
                // still attributable. Not printed into the table: stdout and stderr are read merged,
                // so every log line the load emitted landed inside a half-written row (see
                // tests/test-archs-table.h). The row is assembled whole and emitted after the work.
                fprintf(stderr, "test case: %s (%s, %s)\n", llama_arch_name(arch), dc.label.c_str(), config_name.c_str());

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
#if defined(GGML_USE_SYCL)
                // [llama.cpp-l6wj] Provisional: the Meta (tensor-parallel) config cannot run on a SYCL build.
                //
                // Mechanism: ggml_backend_sycl_device_supports_op() consults this fork's unified-cache
                // placement planner, which holds no residency record for weights living in Meta buffers and
                // therefore reports every dense MUL_MAT/MUL/ADD as host-executing. The scheduler splits the
                // graph mid-layer, and each split boundary re-enters the Meta backend as a bare GGML_OP_NONE
                // tensor which ggml_backend_meta_get_split_state() retypes MIRRORED -- destroying the
                // tensor-parallel shard identity and tripping GGML_ASSERT(split_states_equal(...)) in
                // handle_set_rows. That assertion is correct; llama.cpp-dy1r owns removal of this skip.
                skip = skip || dc.split_mode == LLAMA_SPLIT_MODE_TENSOR;
                // The MoE cache rows stream host-resident experts into a device cache. This fork's
                // placement rule is that an op runs where its weights already are (no weight streaming,
                // CLAUDE.md "SYCL Memory Ownership"), so the rows are reported SKIP rather than measured
                // against a path the SYCL backend does not take.
                skip = skip || dc.moe_cache_size > 0;
#endif  // GGML_USE_SYCL
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        n_measured++;
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (!std::isfinite(nmse_val)) {
                            // NaN > 1e-4 is false, so a threshold test alone reports total numerical
                            // collapse -- the worst outcome -- as a pass. Reject non-finite explicitly and
                            // label it distinctly from a large finite NMSE: NaN means the output degenerated,
                            // a large finite value means the kernel is wrong but alive.
                            snprintf(nmse_str, sizeof(nmse_str), "(%s)",
                                std::isnan(nmse_val) ? "nan" : (nmse_val > 0.0 ? "+inf" : "-inf"));
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                            fprintf(stderr, "\n%s (%s, %s): non-finite NMSE: %s\n",
                                llama_arch_name(arch), dc.label.c_str(), config_name.c_str(),
                                nmse_diagnosis(logits_cpu, logits_dev, tokens.size()).c_str());
                        } else {
                            snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                            if (nmse_val > nmse_gate) {
                                test_ok = false;
                                status_nmse = "\033[1;31mFAIL\033[0m";
                            }
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                if (!(nmse_mixed <= nmse_gate)) {
                                    if (test_ok) {
                                        status_nmse = "\033[1;31mFAIL\033[0m (mixed)";
                                    }
                                    test_ok = false;
                                    fprintf(stderr, "\n%s (%s, %s): mixed batch vs chunks NMSE %.3e > %.1e\n",
                                        llama_arch_name(arch), dc.label.c_str(), config_name.c_str(), nmse_mixed, nmse_gate);
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    if (test_ok) {
                                        status_nmse = "\033[1;31mFAIL\033[0m (mixed)";
                                    }
                                    test_ok = false;
                                    fprintf(stderr, "\n%s (%s, %s): a mixed batch was not rejected (err %d, want -1)\n",
                                        llama_arch_name(arch), dc.label.c_str(), config_name.c_str(), err);
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }

                        // the k-pool layout is updated in place when drafts are rejected
                        // TODO: the meta backend aborts with n_rs_seq > 0 (view of permuted tensor not implemented)
                        if ((arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                            const double nmse_rollback = check_draft_rollback(model_and_ctx_dev.first.get(), tokens, logits_dev);
                            if (!(nmse_rollback <= nmse_gate)) {
                                if (test_ok) {
                                    status_nmse = "\033[1;31mFAIL\033[0m (rollback)";
                                }
                                test_ok = false;
                            }
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_and_ctx_dev.first.get()));
                            const size_t n_ubatch = llama_n_ubatch(model_and_ctx_dev.second.get());
                            const logits_diff rt = logits_compare(logits_dev, logits_roundtrip, n_vocab);
                            const double nmse_rt = nmse(logits_dev, logits_roundtrip);

                            // Gate on NMSE, not on bit-equality.
                            //
                            // The column's job is to catch a save/reload that lost weights, and that
                            // failure is O(1) -- the reloaded model computes something else entirely.
                            // Bit-equality was only ever a proxy for it, and on this backend the proxy
                            // is broken: the device path is not reproducible run to run, so two decodes
                            // of the same weights differ in the last bits by an amount that varies with
                            // scheduling. Measured 2026-08-01: same seed, same binary, same selector,
                            // two processes -- 438 vs 1493 of 16384 logits differing, a 3.4x change,
                            // with the save/reload path byte-identical between them. A bit-exact
                            // comparison there is asking whether two non-deterministic decodes happened
                            // to land identically; it fails at random and, worse, PASSES at random, so
                            // a green column was never evidence either.
                            //
                            // Judged by nmse_gate, the same constant the NMSE column tests against --
                            // deliberately, because the two columns must not disagree about what
                            // "wrong" means. Sharing the constant is what makes that a constraint
                            // rather than a comment.
                            const bool roundtrip_broken = !std::isfinite(nmse_rt) || nmse_rt > nmse_gate;
                            if (roundtrip_broken) {
                                test_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                            } else if (rt.n_diff > 0) {
                                // Cyan, deliberately NOT the yellow this table uses for SKIP. The
                                // column's vocabulary is green=OK / red=FAIL / yellow=SKIP, so reusing
                                // yellow would render "measured, within tolerance, not bit-exact"
                                // identically to "not measured at all" -- and BITDIFF exists precisely
                                // so a non-bit-exact row can never be mistaken for silence.
                                status_roundtrip = "\033[1;36mBITDIFF\033[0m";
                                n_bitdiff++;
                            }

                            if (rt.n_diff > 0) {
                                const size_t n_tok = logits_dev.size()/n_vocab;

                                // Report magnitudes only -- nothing here may touch a device.
                                //
                                // Two probes did, on 2026-08-01, and the run segfaulted: a re-decode
                                // of the already-loaded device model, and a third load of the same
                                // file. Which of the two crashed was not isolated. What the log shows
                                // immediately before the fault is
                                // "[UNIFIED-CACHE] planned-mode runtime materialization rejected
                                // op=direct_stage_weight ... graph_active=1", i.e. inference asking to
                                // stage a weight the placement plan has no entry for. The plan is
                                // global per device and each load replaces it, so both probes run
                                // against state the single-pass original never creates. Whatever the
                                // exact mechanism, a diagnostic that costs the run it is diagnosing is
                                // worth less than no diagnostic: keep this block pure arithmetic over
                                // logits that have already been computed.
                                // The movable quantities come FIRST, and that ordering is deliberate.
                                // NMSE is the right gate -- it catches the O(1) corruption this column
                                // exists for -- but it is not a sensitivity measure and must not be read
                                // as one. At these magnitudes the whole divergence contributes ~6e-17 to
                                // ~2e-16 to NMSE, against a value printed to four significant figures
                                // whose last digit is worth ~1e-12: four orders of magnitude below what
                                // the format can show. It stayed pinned at 2.201e-09 across the two
                                // replicates whose differing-element counts were 438 and 1493. Anyone
                                // diagnosing from NMSE alone is reading a quantity that cannot move.
                                // The count, the first difference and the affected rows can.
                                fprintf(stderr,
                                    "\n%s (%s, %s): roundtrip not bit-exact [%s]\n"
                                    "  save/reload vs device: %zu/%zu logits differ, max abs %.3e, max rel %.3e\n"
                                    "  first difference:      token %zu of %zu, vocab %zu of %zu (flat index %zu)\n"
                                    "  token rows affected:   %zu of %zu, from %zu to %zu%s\n"
                                    "  gate NMSE(dev,reload): %.3e vs threshold %.1e -- %s\n"
                                    "  NMSE vs CPU:           device %.3e, reloaded %.3e\n"
                                    "  (the gate is NMSE; the three lines above it are the ones with the\n"
                                    "   resolution to move between runs -- diagnose from those, not from NMSE)\n",
                                    llama_arch_name(arch), dc.label.c_str(), config_name.c_str(),
                                    roundtrip_broken ? "FAIL" : "BITDIFF",
                                    rt.n_diff, logits_dev.size(), rt.max_abs, rt.max_rel,
                                    rt.first_row, n_tok, rt.first_col, n_vocab, rt.first_diff,
                                    rt.n_rows_diff, n_tok, rt.first_row, rt.last_row,
                                    rt.first_row == n_ubatch ?
                                        " -- exactly n_ubatch, so every token of the FIRST ubatch is bit-identical"
                                        " and divergence begins at the first token of the SECOND" : "",
                                    nmse_rt, nmse_gate, roundtrip_broken ?
                                        "EXCEEDED: the reloaded model computes something else; the save/reload lost data" :
                                        "within tolerance: both models agree to far better than the gate",
                                    nmse(logits_cpu, logits_dev), nmse(logits_cpu, logits_roundtrip));
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // The row for this test case, assembled whole and written with one stdio call.
                // Draining the log first keeps anything this case logged ahead of its row rather
                // than after it; the single write is what keeps it out of the row.
                common_log_flush(common_log_main());
                archs_table::emit(table.row(llama_arch_name(arch), dc.label.c_str(), config_name.c_str(),
                                            status_nmse.c_str(), nmse_str, status_roundtrip.c_str()));
            }
        }
    }

    common_log_flush(common_log_main());
    archs_table::emit(table.footer(nmse_gate, n_bitdiff));
    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    if (n_measured == 0) {
        // Exit skip (LLAMA_TEST_EXIT_SKIP -- see tests/test-skip.h) rather than 0: this
        // harness compared nothing, so there is no result to report either way. `all_ok`
        // is initialised true, so without this the run would report success it did not earn.
        //
        // Unconditional, deliberately. This used to be guarded by `target_arch !=
        // LLM_ARCH_UNKNOWN`, on the reasoning that "77 is unreachable from the registered
        // invocation, which passes no `-a` and always measures something" -- an assumption
        // about the sweep, stated in a comment and enforced by nothing, and the no-`-a` sweep
        // is precisely what CI runs. So the one path that was left unguarded was the one that
        // mattered. See llama.cpp-to9m.
        //
        // --exclude reaches here too, and 77 is the right answer for it as well: `-x` on
        // everything the run would have measured (`-a llama -x llama` is the small case)
        // leaves n_measured at 0, and a run that measured nothing must score as skipped, not
        // passed. What it must NOT do is let the reader diagnose the wrong thing, so the note
        // below names the exclusions -- otherwise the sweep branch sends someone hunting for a
        // missing backend when the cause is on their own command line. The note is empty when
        // no -x was given, which keeps that path's output byte-identical.
        std::string excluded_note;
        if (!excluded_archs.empty()) {
            std::string names;
            for (const llm_arch & excluded : excluded_archs) {
                names += names.empty() ? "" : ", ";
                names += llama_arch_name(excluded);
            }
            excluded_note = string_format(
                "  %zu architecture(s) were excluded by --exclude (%s). If that covers "
                "everything this run would otherwise have measured then 77 is the honest "
                "answer, and the environment causes listed above do not apply.\n",
                excluded_archs.size(), names.c_str());
        }
        if (!arch_filter.empty()) {
            fprintf(stderr,
                    "\n%s: no NMSE comparison was performed for '%s' -- this harness cannot "
                    "measure the matching architectures, so this run proves NOTHING about them.\n"
                    "  Reasons this happens: the arch is skipped outright by test_backends() "
                    "(eagle3, dflash, gemma4-assistant emit no row at all), or "
                    "arch_supported() returns false for it (gemma-embedding, the BERT family, "
                    "RWKV, ...) and every row is SKIP.\n%s",
                    __func__, arch_filter.c_str(), excluded_note.c_str());
        } else {
            fprintf(stderr,
                    "\n%s: the full sweep performed no NMSE comparison at all -- every "
                    "architecture was excluded or skipped, so this run proves NOTHING about "
                    "any of them.\n"
                    "  A sweep reaching this is a harness or environment failure, not a "
                    "property of one architecture: check that a backend was registered and "
                    "that the run was not cut short before the first comparison.\n%s",
                    __func__, excluded_note.c_str());
        }
        return LLAMA_TEST_EXIT_SKIP;
    }
    return all_ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string &           arch_filter,
                         const std::vector<llm_arch> & excluded_archs,
                         const size_t                  seed,
                         const float                   stdev,
                         const int                     verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (archs_exclude::contains(excluded_archs, arch)) {
            LOG(template_row.c_str(), llama_arch_name(arch), "-", "", "", "", "", archs_exclude::status);
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    // the -a argument when it names one architecture exactly; --nan-trace needs it
    llm_arch arch_exact = LLM_ARCH_UNKNOWN;
    std::vector<llm_arch> excluded_archs;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool kv_shift = false;
    bool nan_trace = false;

    int verbosity = LOG_LEVEL_ERROR;

    // Every argument is matched by exactly one branch, and anything else is an error: an
    // unrecognised argument that was silently dropped would let the run go on to load models
    // onto a GPU while the operator believes it was restricted.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch_exact = llama_arch_from_string(arch_name);
                if (arch_exact != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
                // A filter that selects nothing must exit here, in argument parsing, not run an
                // empty sweep: tests/test-archs-exclude-cli.sh relies on a bogus -a exiting 1 to stay
                // off the GPU, and an empty run would otherwise only surface as 77 after backend init.
                bool any = false;
                for (const llm_arch & arch : llama_arch_all()) {
                    any = any || (arch != LLM_ARCH_UNKNOWN && arch_matches(arch_filter, arch));
                }
                if (!any) {
                    LOG_ERR("%s: unknown LLM architecture: %s (matches no architecture)\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "--exclude") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                // Rejecting the name is the whole point of validating it. A typo'd -x that
                // silently excluded nothing would leave the run measuring exactly what it
                // measured before, while the operator believes an architecture was skipped.
                if (!archs_exclude::parse(arch_name.c_str(), excluded_archs)) {
                    LOG_ERR("%s: --exclude: not a usable LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            // Bare -v/--verbose shows every GGML_LOG_INFO line the backend emits:
            // common_get_verbosity() maps GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE, so a filter at
            // INFO would drop them. Upstream's `-v N` sets the level explicitly.
            if (strcmp(argv[i], "-v") == 0 && i + 1 < argc && argv[i + 1][0] != '\0' &&
                strspn(argv[i + 1], "0123456789") == strlen(argv[i + 1])) {
                verbosity = std::stoi(argv[++i]);
            } else {
                verbosity = LOG_LEVEL_TRACE;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else if (strcmp(argv[i], "--nan-trace") == 0) {
            nan_trace = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    // --nan-trace traces the single architecture named by -a; it iterates no arch list, so
    // there is nothing for --exclude to filter there. Rejecting the combination rather than
    // ignoring it: a flag that is quietly inert in one mode is the same silent no-op the
    // unknown-name check above exists to prevent.
    if (nan_trace && !excluded_archs.empty()) {
        LOG_ERR("%s: --exclude has nothing to filter under --nan-trace, which traces only the -a architecture\n",
                __func__);
        return 1;
    }

    init_normal_pool(seed, stdev);

    try {
        if (nan_trace) {
            return trace_nan(arch_exact, seed, stdev);
        }
        if (!out.empty()) {
            return save_models(arch_filter, excluded_archs, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, excluded_archs, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, excluded_archs, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
