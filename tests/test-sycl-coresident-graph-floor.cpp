// Two co-resident loads of one model with different contexts, in either order, on one card.
//
// The ONEDNN arena zone holds the oneDNN primitive pair and the oneDNN Graph scratch floor, which grows with the
// context the load is planned for (n_ctx_hint). It is not the only zone that does: the load cannot know whether its
// contexts will run flash attention, so the RUNTIME zone holds the load measure's compute term, whose prompt graph
// carries the non-flash attention score matrix (n_head x n_ubatch x n_ctx x 4 B), and the SCRATCH zone the planned
// non-flash attention scratch (n_head x n_ubatch x n_ctx x 2 B x 3). The zones grow only by a rebuild, and nothing is
// rebuilt while a live model's allocations point into the arena. So a second load planned for a larger context than
// the first, with a context small enough that RUNTIME and SCRATCH stay at their 512 MiB defaults, finds a zone too
// small for its Graph floor and nothing else short. The floor is a reservation, not a requirement: what the zone
// cannot hold of it is served on the DIRECT path under its cap. The second load keeps the zones as they are, with one
// "[VRAM-ARENA] device N keeps its ONEDNN zone of" WARN, instead of being refused. A larger context that outgrows
// RUNTIME or SCRATCH too is a shortfall the keep does not cover, and that load is refused. In the reverse order the
// first load sizes the zone for the larger floor, and the second takes the sufficient path with no keep.
//
// Each run loads the model twice through the llama API, creates a context on each load at the context its load was
// planned for, and decodes the count prompt on each. The first model's context is created and decoded before the
// second load starts, so its KV and compute buffers hold the arena while the second load sizes the zones. Then the
// first model decodes once more, beside the second. The log lines are counted through the llama log callback, which
// also receives the ggml backends' lines.
//
// --order small-first (default): the small context first, then the large one. Positive controls: both loads and both
//   contexts succeed, every decode counts on correctly, the second load logs exactly one keep WARN, and the run logs
//   no "live allocations prevent rebuild" ERROR (the held rebuild's) and no "model load refused" line (the
//   REBUILD_HELD refusal's). Keep WARNs outside the second load (a context's re-plan, which belongs to no load and
//   warns at every keep) are printed but not scored.
// --order large-first: the large context first. The same, with no keep WARN anywhere in the run.
//
// What it cannot see: the Graph room the kept zone leaves, the stored zone plan, and a failed load's restore of it
// (no load fails here). A keep that stored the plan's floor rather than what the zone holds, or a restore that left a
// refused load's plan, still passes: those are the source pin's and test-zone-sizing's to catch. Exit 0 PASS, 1 FAIL,
// 77 SKIP (no model file, or no GPU device).
//
// It loads a 7B model twice, so it carries `model`, the label the ctest sweep excludes, and the lead runs it by hand,
// once per order, pinned to one discrete card, after sourcing oneAPI's setvars.sh:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 600 ./build/bin/test-sycl-coresident-graph-floor --order small-first
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 600 ./build/bin/test-sycl-coresident-graph-floor --order large-first

#include "ggml-backend.h"
#include "llama.h"
#include "test-skip.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char * k_default_model = "/models/mistral-7b-v0.1.Q4_0.gguf";
constexpr const char * k_kept_warn     = "keeps its ONEDNN zone of";
constexpr const char * k_held_rebuild  = "live allocations prevent rebuild";
constexpr const char * k_load_refused  = "model load refused on device";
constexpr const char * k_prompt        = "1, 2, 3, 4, 5,";
constexpr const char * k_count_on      = " 6, 7, 8, 9, 10";
constexpr int          k_n_predict     = 16;
// Mistral 7B, 32 heads, at the 512-token micro-batch. The Graph floor is the 64 MiB minimum at a 512-token context
// and 384 MiB at 4096, so a ONEDNN zone sized for the small one (256 MiB, the zone's minimum) cannot hold the large
// one's beside the pair. 4096 keeps the other zones at their 512 MiB defaults: the score matrix in the RUNTIME
// compute term is 256 MiB and the non-flash SCRATCH need 384 MiB. At 8192 they are 512 and 768 MiB, so the large
// load would outgrow RUNTIME and SCRATCH as well and be refused rather than keep.
constexpr uint32_t     k_n_ctx_small   = 512;
constexpr uint32_t     k_n_ctx_large   = 4096;

struct log_counts {
    std::mutex mutex;
    int        kept_warn    = 0;
    int        held_rebuild = 0;
    int        refused      = 0;
};

log_counts g_counts;

int kept_warns() {
    std::lock_guard<std::mutex> lock(g_counts.mutex);
    return g_counts.kept_warn;
}

void count_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (text == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_counts.mutex);
        g_counts.kept_warn += std::strstr(text, k_kept_warn) != nullptr;
        g_counts.held_rebuild += std::strstr(text, k_held_rebuild) != nullptr;
        g_counts.refused += std::strstr(text, k_load_refused) != nullptr;
    }
    if (level >= GGML_LOG_LEVEL_WARN || std::strstr(text, "[VRAM-ARENA]") != nullptr) {
        std::fputs(text, stderr);
    }
}

struct loaded {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    ~loaded() {
        if (ctx != nullptr) {
            llama_free(ctx);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

// Loads the model planned for an `n_ctx` context and creates that context; false when either step fails. `kept`
// receives the keep WARNs the load itself logged, before its context exists.
bool load_at(const char * path, uint32_t n_ctx, loaded & out, int & kept) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_ctx_hint         = n_ctx;
    const int kept_before      = kept_warns();
    out.model                  = llama_model_load_from_file(path, mparams);
    kept                       = kept_warns() - kept_before;
    if (out.model == nullptr) {
        std::fprintf(stderr, "FAIL: the load planned for n_ctx=%u failed\n", n_ctx);
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx;
    out.ctx                      = llama_init_from_model(out.model, cparams);
    if (out.ctx == nullptr) {
        std::fprintf(stderr, "FAIL: the context of n_ctx=%u was not created\n", n_ctx);
        return false;
    }
    return true;
}

// Decodes the count prompt greedily from an empty context and checks the text counts on from 6 to 10.
bool counts_on(loaded & l, const char * name) {
    llama_memory_clear(llama_get_memory(l.ctx), true);
    const llama_vocab *      vocab = llama_model_get_vocab(l.model);
    std::vector<llama_token> tokens(64);
    const int32_t            n =
        llama_tokenize(vocab, k_prompt, (int32_t) std::strlen(k_prompt), tokens.data(), (int32_t) tokens.size(),
                       /*add_special=*/true, /*parse_special=*/false);
    if (n <= 0) {
        std::fprintf(stderr, "FAIL: %s: the prompt did not tokenize\n", name);
        return false;
    }
    tokens.resize(n);
    std::string text;
    const int   n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < k_n_predict; ++i) {
        if (llama_decode(l.ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) != 0) {
            std::fprintf(stderr, "FAIL: %s: decode %d failed\n", name, i);
            return false;
        }
        const float * logits = llama_get_logits_ith(l.ctx, -1);
        llama_token   best   = 0;
        for (llama_token t = 1; t < n_vocab; ++t) {
            best = logits[t] > logits[best] ? t : best;
        }
        char          piece[64];
        const int32_t len = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);
        text.append(piece, len > 0 ? (size_t) len : 0);
        tokens.assign(1, best);
    }
    const bool ok = text.rfind(k_count_on, 0) == 0;
    std::fprintf(stderr, "%s %s: \"%s%s\"\n", ok ? "ok" : "FAIL:", name, k_prompt, text.c_str());
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * path        = k_default_model;
    bool         small_first = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            path = argv[++i];
        } else if (std::strcmp(argv[i], "--order") == 0 && i + 1 < argc) {
            const char * order = argv[++i];
            if (std::strcmp(order, "small-first") != 0 && std::strcmp(order, "large-first") != 0) {
                std::fprintf(stderr, "usage: %s [-m model.gguf] [--order small-first|large-first]\n", argv[0]);
                return 1;
            }
            small_first = std::strcmp(order, "small-first") == 0;
        } else {
            std::fprintf(stderr, "usage: %s [-m model.gguf] [--order small-first|large-first]\n", argv[0]);
            return 1;
        }
    }
    if (FILE * f = std::fopen(path, "rb")) {
        std::fclose(f);
    } else {
        std::fprintf(stderr, "SKIP: no model at %s\n", path);
        return LLAMA_TEST_EXIT_SKIP;
    }

    llama_log_set(count_log, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        std::fprintf(stderr, "SKIP: no GPU device, so no arena zones to keep\n");
        llama_backend_free();
        return LLAMA_TEST_EXIT_SKIP;
    }

    const uint32_t first_ctx  = small_first ? k_n_ctx_small : k_n_ctx_large;
    const uint32_t second_ctx = small_first ? k_n_ctx_large : k_n_ctx_small;
    std::fprintf(stderr, "order %s: n_ctx %u, then %u\n", small_first ? "small-first" : "large-first", first_ctx,
                 second_ctx);

    bool ok          = true;
    int  kept_first  = 0;
    int  kept_second = -1;
    {
        loaded first;
        loaded second;
        ok = load_at(path, first_ctx, first, kept_first) && counts_on(first, "first model") &&
             load_at(path, second_ctx, second, kept_second) && counts_on(second, "second model") &&
             counts_on(first, "first model beside the second");
    }

    const int want_kept = small_first ? 1 : 0;
    {
        std::lock_guard<std::mutex> lock(g_counts.mutex);
        std::fprintf(stderr, "keep WARN lines: %d in the first load, %d in the second, %d in the run\n", kept_first,
                     kept_second, g_counts.kept_warn);
        if (kept_first != 0 || kept_second != want_kept || (!small_first && g_counts.kept_warn != 0)) {
            std::fprintf(stderr, "FAIL: the second load logged %d keep WARN line(s), want %d%s\n", kept_second,
                         want_kept, small_first ? "" : ", and none anywhere in the run");
            ok = false;
        }
        if (g_counts.held_rebuild != 0 || g_counts.refused != 0) {
            std::fprintf(stderr, "FAIL: %d \"%s\" line(s) and %d \"%s\" line(s), want none\n", g_counts.held_rebuild,
                         k_held_rebuild, g_counts.refused, k_load_refused);
            ok = false;
        }
    }
    llama_backend_free();
    std::fprintf(stderr, "%s: two co-resident loads, %s\n", ok ? "PASS" : "FAIL",
                 small_first ? "small context first" : "large context first");
    return ok ? 0 : 1;
}
