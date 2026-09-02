// Two-context KV/runtime isolation proof (llama.cpp-32dg8.15.1, epic llama.cpp-rg2ft).
//
// Question this test answers: can ONE loaded model share its process-scoped
// weight/cache state while TWO live llama_context objects (different n_ctx)
// keep their KV / runtime / graph state isolated from each other?
//
// Method (behavioural, backend-agnostic): the KV cache of a context is the
// only state that carries a decode's history into the next decode.  If context
// B's KV allocation aliased, reset, or overwrote context A's, then A's NEXT
// decode would attend over B's tokens and its logits would diverge from a
// single-context reference run of the same token sequence.  So:
//
//   1. Reference: fresh context R_A decodes P_A then c1 then c2; record the
//      logits after each step.  Same for R_B with a DIFFERENT prompt P_B.
//      (Each reference context is freed before the next is created, so the
//      reference itself never has two contexts alive.)
//   2. Proof: create A (n_ctx_a) and B (n_ctx_b) and keep BOTH alive.
//      Interleave: A(P_A) B(P_B) A(c1) B(c1) A(c2) B(c2).  Every step's
//      logits must match the reference step for that context.
//
// The prompts differ in content AND length on purpose: with identical prompts
// a KV overwrite would replace A's history with byte-identical content and the
// test would pass vacuously (CLAUDE.md: "absence of work looks like success").
//
// Match criterion: argmax equal AND max |diff| <= tolerance.  The tolerance
// exists because a different n_ctx can legitimately change attention kernel
// tiling / n_kv padding and hence FP reduction order; a KV clobber produces
// argmax flips and O(1) logit deltas, orders of magnitude above it.  The
// tolerance is printed and can be tightened with --tol.
//
// This is a model-loading test.  Under ctest it is pinned to level_zero:1 via
// the registration's ENVIRONMENT (tests/CMakeLists.txt); a direct invocation
// must set ONEAPI_DEVICE_SELECTOR itself (CLAUDE.md, llama.cpp-403s).
//
// Exit codes: 0 pass, 1 fail, 77 skip (no model).

#include "llama.h"
#include "test-skip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct step_logits {
    std::vector<float> logits;
    int                argmax = -1;
};

static bool decode_tokens(llama_context * ctx, std::vector<llama_token> tokens, const char * tag, step_logits & out) {
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    const int   rc    = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "[TWO-CTX] %s: llama_decode failed rc=%d\n", tag, rc);
        return false;
    }
    const llama_model * model   = llama_get_model(ctx);
    const int           n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float *       logits  = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        fprintf(stderr, "[TWO-CTX] %s: null logits\n", tag);
        return false;
    }
    out.logits.assign(logits, logits + n_vocab);
    out.argmax = (int) (std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin());
    return true;
}

static bool compare_step(const step_logits & got, const step_logits & ref, const char * tag, float tol) {
    if (got.logits.size() != ref.logits.size()) {
        fprintf(stderr, "[TWO-CTX] %s: vocab size mismatch %zu vs %zu\n", tag, got.logits.size(), ref.logits.size());
        return false;
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < got.logits.size(); ++i) {
        const float d = std::fabs(got.logits[i] - ref.logits[i]);
        if (!(d <= max_diff)) {  // also catches NaN
            max_diff = std::isnan(d) ? INFINITY : d;
        }
    }
    const bool ok = got.argmax == ref.argmax && max_diff <= tol;
    fprintf(stderr, "[TWO-CTX] %s: argmax got=%d ref=%d max|diff|=%.6f tol=%.6f -> %s\n", tag, got.argmax, ref.argmax,
            max_diff, tol, ok ? "match" : "MISMATCH");
    return ok;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_batch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx;
    cparams.n_batch              = n_batch;
    cparams.n_ubatch             = n_batch;
    cparams.n_seq_max            = 1;
    cparams.n_threads            = 2;
    cparams.n_threads_batch      = 2;
    return llama_init_from_model(model, cparams);
}

// Decode a whole script (prompt, then single-token continuations) in ONE fresh
// context and record the logits after every step.  Frees the context.
static bool run_reference(llama_model *                   model,
                          uint32_t                        n_ctx,
                          const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & cont,
                          const char *                    tag,
                          std::vector<step_logits> &      out) {
    llama_context * ctx = make_ctx(model, n_ctx, 32);
    if (!ctx) {
        fprintf(stderr, "[TWO-CTX] %s: reference context creation failed (n_ctx=%u)\n", tag, n_ctx);
        return false;
    }
    bool ok = true;
    out.clear();
    out.resize(1 + cont.size());
    ok = ok && decode_tokens(ctx, prompt, tag, out[0]);
    for (size_t i = 0; ok && i < cont.size(); ++i) {
        ok = ok && decode_tokens(ctx, { cont[i] }, tag, out[1 + i]);
    }
    llama_synchronize(ctx);
    llama_free(ctx);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    uint32_t     n_ctx_a    = 256;
    uint32_t     n_ctx_b    = 512;
    float        tol        = 0.05f;
    int          n_gpu      = 99;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ctx-a") == 0 && i + 1 < argc) {
            n_ctx_a = (uint32_t) std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctx-b") == 0 && i + 1 < argc) {
            n_ctx_b = (uint32_t) std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = (float) std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu = std::atoi(argv[++i]);
        } else {
            model_path = argv[i];
        }
    }
    if (!model_path) {
        model_path = std::getenv("LLAMACPP_TEST_MODELFILE");
    }
    if (!model_path || !model_path[0]) {
        test_skip_no_model();
    }
    {
        FILE * f = std::fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "[TWO-CTX] model file not readable: %s\n", model_path);
            test_skip_no_model();
        }
        std::fclose(f);
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu;
    llama_model * model        = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[TWO-CTX] failed to load model %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // Two DIFFERENT prompts, different lengths, plus shared continuations.
    // Token ids are chosen deterministically from the vocab range so the test
    // needs no tokenizer and works with any model file (stories15M included).
    std::vector<llama_token> prompt_a;
    std::vector<llama_token> prompt_b;
    for (int i = 0; i < 12; ++i) {
        prompt_a.push_back((llama_token) (1 + (i * 37 + 11) % (n_vocab - 1)));
    }
    for (int i = 0; i < 20; ++i) {
        prompt_b.push_back((llama_token) (1 + (i * 53 + 101) % (n_vocab - 1)));
    }
    std::vector<llama_token> cont = { (llama_token) (1 + 7 % (n_vocab - 1)), (llama_token) (1 + 199 % (n_vocab - 1)),
                                      (llama_token) (1 + 3001 % (n_vocab - 1)) };

    fprintf(stderr, "[TWO-CTX] model=%s n_ctx_a=%u n_ctx_b=%u |P_A|=%zu |P_B|=%zu cont=%zu tol=%g\n", model_path,
            n_ctx_a, n_ctx_b, prompt_a.size(), prompt_b.size(), cont.size(), tol);

    bool ok = true;

    // 1. References, each in its own single live context.
    std::vector<step_logits> ref_a;
    std::vector<step_logits> ref_b;
    ok = ok && run_reference(model, n_ctx_a, prompt_a, cont, "ref-A", ref_a);
    ok = ok && run_reference(model, n_ctx_b, prompt_b, cont, "ref-B", ref_b);
    if (!ok) {
        fprintf(stderr, "[TWO-CTX] FAIL: reference runs did not complete\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Reference sanity: the two prompts must NOT produce the same first-step
    // logits, otherwise a KV clobber would be invisible (vacuous pass guard).
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < ref_a[0].logits.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(ref_a[0].logits[i] - ref_b[0].logits[i]));
        }
        fprintf(stderr, "[TWO-CTX] discriminator: max|ref_A0 - ref_B0| = %.6f (must exceed tol)\n", max_diff);
        if (max_diff <= tol) {
            fprintf(stderr, "[TWO-CTX] FAIL: prompts are not discriminating; the proof would be vacuous\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    // 2. Two live contexts, interleaved decodes.
    llama_context * ctx_a = make_ctx(model, n_ctx_a, 32);
    llama_context * ctx_b = make_ctx(model, n_ctx_b, 32);
    if (!ctx_a || !ctx_b) {
        fprintf(stderr, "[TWO-CTX] FAIL: could not create both contexts (a=%p b=%p)\n", (void *) ctx_a,
                (void *) ctx_b);
        if (ctx_a) {
            llama_free(ctx_a);
        }
        if (ctx_b) {
            llama_free(ctx_b);
        }
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    step_logits s;
    ok = ok && decode_tokens(ctx_a, prompt_a, "A(P_A)", s) && compare_step(s, ref_a[0], "A step0", tol);
    ok = ok && decode_tokens(ctx_b, prompt_b, "B(P_B)", s) && compare_step(s, ref_b[0], "B step0", tol);
    for (size_t i = 0; i < cont.size(); ++i) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "A step%zu", i + 1);
        ok = ok && decode_tokens(ctx_a, { cont[i] }, tag, s) && compare_step(s, ref_a[1 + i], tag, tol);
        std::snprintf(tag, sizeof(tag), "B step%zu", i + 1);
        ok = ok && decode_tokens(ctx_b, { cont[i] }, tag, s) && compare_step(s, ref_b[1 + i], tag, tol);
    }

    // 3. Clearing B's memory must not disturb A: A continues to match.
    llama_memory_clear(llama_get_memory(ctx_b), true);
    {
        // Re-run A's script in a third fresh context to extend the reference by
        // one more continuation token after the clear.
        std::vector<llama_token> cont_ext = cont;
        cont_ext.push_back((llama_token) (1 + 4242 % (n_vocab - 1)));
        std::vector<step_logits> ref_a_ext;
        ok = ok && run_reference(model, n_ctx_a, prompt_a, cont_ext, "ref-A-ext", ref_a_ext);
        ok = ok && decode_tokens(ctx_a, { cont_ext.back() }, "A after B clear", s) &&
             compare_step(s, ref_a_ext.back(), "A step after B clear", tol);
    }

    llama_synchronize(ctx_a);
    llama_synchronize(ctx_b);
    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_model_free(model);
    llama_backend_free();

    if (!ok) {
        fprintf(stderr, "[TWO-CTX] FAIL: a context observed another context's KV/runtime state\n");
        return 1;
    }
    fprintf(stderr, "[TWO-CTX] PASS: two live contexts (n_ctx %u / %u) kept KV/runtime state isolated\n", n_ctx_a,
            n_ctx_b);
    return 0;
}
