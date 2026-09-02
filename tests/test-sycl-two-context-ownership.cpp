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
//   1. Reference: fresh context R_A decodes P_A then c1, c2, c3, c4 (c4 is the
//      token later replayed on A after B's memory is cleared); record the
//      logits after each step.  Same for R_B with a DIFFERENT prompt P_B and
//      just c1, c2, c3 (B is never replayed post-clear).  Every reference
//      context is created AND freed here, before the two-context proof phase
//      begins, so no reference run ever overlaps with a live A/B pair -- the
//      reference itself never has two (let alone three) contexts alive.
//   2. Proof: create A (n_ctx_a) and B (n_ctx_b) and keep BOTH alive.
//      Interleave: A(P_A) B(P_B) A(c1) B(c1) A(c2) B(c2) A(c3) B(c3).  Then two
//      more parts: (a) clear B's memory and replay c4 on A alone -- A must
//      still match its reference (R_A's c4 step, computed before A/B ever
//      existed), proving B's clear did not disturb A; (b) POSITIVE CONTROL --
//      replay P_B on B after the same clear -- B must reproduce ref_b[0] (B
//      decoding P_B fresh), which holds only if the clear actually emptied
//      B's KV rather than being a silent no-op that (a) alone could not
//      detect.
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
// Exit codes: 0 pass, 1 fail, 77 skip (no model file, or no GPU backend registered).

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
static bool run_reference(llama_model *                    model,
                          uint32_t                         n_ctx,
                          const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & cont,
                          const char *                     tag,
                          std::vector<step_logits> &       out) {
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
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argv[i][0] == '-') {
            // Reject unknown flags rather than silently swallowing them into
            // model_path -- a typo'd flag used to fall through to fopen()
            // failing, which exits 77 (skip) and reads as "no model provided"
            // instead of "bad arguments".
            fprintf(stderr,
                    "[TWO-CTX] usage: %s [-m <model.gguf>] [--ctx-a N] [--ctx-b N] [--tol F] [-ngl N] "
                    "[<model.gguf>]\n[TWO-CTX] unrecognised argument: %s\n",
                    argv[0], argv[i]);
            return 1;
        } else if (model_path) {
            fprintf(stderr,
                    "[TWO-CTX] usage: %s [-m <model.gguf>] [--ctx-a N] [--ctx-b N] [--tol F] [-ngl N] "
                    "[<model.gguf>]\n[TWO-CTX] unexpected extra positional argument: %s\n",
                    argv[0], argv[i]);
            return 1;
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

    if (!llama_supports_gpu_offload()) {
        // A GGML_SYCL build with no device enumerated, or a backend init
        // failure that silently fell back to CPU, must not be allowed to pass
        // by proving nothing: this test's whole point is a GPU-backend memory
        // ownership property.
        fprintf(stderr, "[TWO-CTX] no GPU backend registered; this test proves nothing on CPU\n");
        llama_backend_free();
        return LLAMA_TEST_EXIT_SKIP;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu;
    llama_model * model        = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[TWO-CTX] failed to load model %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
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

    // 1. References, each in its own single live context, computed entirely
    //    before ctx_a/ctx_b exist (see header comment). ref_a's script
    //    includes the extra post-clear continuation token up front so no
    //    later reference run needs to happen while A/B are alive.
    std::vector<llama_token> cont_ext = cont;
    cont_ext.push_back((llama_token) (1 + 4242 % (n_vocab - 1)));

    std::vector<step_logits> ref_a;
    std::vector<step_logits> ref_b;
    ok = ok && run_reference(model, n_ctx_a, prompt_a, cont_ext, "ref-A", ref_a);
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
        fprintf(stderr, "[TWO-CTX] FAIL: could not create both contexts (a=%p b=%p)\n", (void *) ctx_a, (void *) ctx_b);
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

    step_logits got;
    ok = ok && decode_tokens(ctx_a, prompt_a, "A step0", got) && compare_step(got, ref_a[0], "A step0", tol);
    ok = ok && decode_tokens(ctx_b, prompt_b, "B step0", got) && compare_step(got, ref_b[0], "B step0", tol);
    for (size_t i = 0; i < cont.size(); ++i) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "A step%zu", i + 1);
        ok = ok && decode_tokens(ctx_a, { cont[i] }, tag, got) && compare_step(got, ref_a[1 + i], tag, tol);
        std::snprintf(tag, sizeof(tag), "B step%zu", i + 1);
        ok = ok && decode_tokens(ctx_b, { cont[i] }, tag, got) && compare_step(got, ref_b[1 + i], tag, tol);
    }

    // 3. Clearing B's memory must not disturb A: A continues to match. The
    //    reference for this step (ref_a.back(), i.e. ref_a's c4 entry) was
    //    already computed in step 1, before ctx_a/ctx_b existed -- no new
    //    reference context is created here.
    llama_memory_clear(llama_get_memory(ctx_b), true);
    ok = ok && decode_tokens(ctx_a, { cont_ext.back() }, "A after B clear", got) &&
         compare_step(got, ref_a.back(), "A step after B clear", tol);

    // 3b. POSITIVE CONTROL: without this, step 3 above could pass vacuously if
    //     llama_memory_clear() were a silent no-op -- A would look undisturbed
    //     either way, since nothing checks that B's KV was actually emptied.
    //     Replaying P_B on B after the clear must reproduce ref_b[0] (B
    //     decoding P_B fresh); it would NOT match ref_b[0] if the clear did
    //     not happen, since B would then be decoding P_B on top of its
    //     existing (uncleared) KV history instead of from empty.
    ok = ok && decode_tokens(ctx_b, prompt_b, "B step0 replay after clear", got) &&
         compare_step(got, ref_b[0], "B step0 replay after clear", tol);

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
