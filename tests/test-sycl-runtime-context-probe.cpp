// F-lead-1 (llama.cpp-tsfl, spec orphan-deliverable fix): the plan's Task 2
// acceptance criterion --
//   "probe(512) on B50 GPT-OSS returns accepted with no demotion and leaves
//    the published plan byte-identical ...; probe(8192) returns refused
//    with the ring reason and prints no ERROR (lead-run)"
// -- had no caller anywhere in the tree: the probe is not called by any
// binary, test-sycl-runtime-alloc loads no model, and llama_context's own
// `backends` member is private. This harness is that caller, so the lead
// has something to actually run for that criterion.
//
// This is a MODEL-LOADING test (see CLAUDE.md's "never loop a model-loading
// binary" family). It is built here (ninja -C build
// test-sycl-runtime-context-probe) but NOT run by this session -- GPU work
// is lead-run only. Under ctest it is pinned to level_zero:1 and carries the
// "cache" label (loads a 12 GB MoE model) so it is excluded from the
// throttled full sweep, same as test-sycl-two-context-ownership above it in
// tests/CMakeLists.txt.
//
// Exit codes: 0 pass, 1 fail, 77 skip (no model file, or no GPU backend
// registered).

#include "llama.h"
#include "test-skip.h"

// Private header, precedent tests/test-quantize-stats.cpp:6 -- needed for
// llama_model::get_sycl_model_token(), the exact accessor
// src/llama-context.cpp uses to build a ggml_sycl_model_token from a loaded
// model (cited below at the call site). Guarded by NOT GGML_BACKEND_DL in
// the CMakeLists registration, the same way that precedent is guarded there
// (tests/CMakeLists.txt) -- doubly appropriate here, since calling
// ggml_backend_sycl_probe_runtime_context_for_model() directly (not just
// referencing its type, see test-sycl-lifecycle-public-api.cpp's own
// comment on why it does NOT odr-use these symbols) also requires
// ggml-sycl to be linked in statically rather than loaded as a
// GGML_BACKEND_DL module.
#include "../src/llama-model.h"
#include "ggml-sycl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void log_everything(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    // llama.cpp-tsfl: every level, unfiltered, to stderr -- the [SYCL-PLAN]
    // INFO lines (both the transaction's own and the probe's own INFO
    // refusals/predictions) are dropped at default verbosity in every tool
    // (CLAUDE.md), and this harness's whole point is for the lead to be able
    // to read them. This bypasses that default because it installs the raw
    // ggml_log_callback directly (llama_log_set -> ggml_log_set) -- nothing
    // in this program goes through common/log.cpp's separate verbosity
    // threshold, which only gates common_log_add()/LOG_INF()-style callers.
    (void) level;
    fputs(text, stderr);
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : nullptr;
    if (!model_path) {
        model_path = std::getenv("LLAMACPP_TEST_MODELFILE");
    }
    if (!model_path || !model_path[0]) {
        test_skip_no_model();
    }
    {
        FILE * f = std::fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "[PROBE-HARNESS] model file not readable: %s\n", model_path);
            test_skip_no_model();
        }
        std::fclose(f);
    }

    llama_log_set(log_everything, nullptr);

    llama_backend_init();

    if (!llama_supports_gpu_offload()) {
        fprintf(stderr, "[PROBE-HARNESS] no GPU backend registered; this test proves nothing on CPU\n");
        llama_backend_free();
        return LLAMA_TEST_EXIT_SKIP;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;
    llama_model * model        = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[PROBE-HARNESS] failed to load model %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 4096;
    cparams.n_batch              = 2048;
    cparams.n_ubatch             = 512;
    cparams.n_seq_max            = 1;
    llama_context * ctx          = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[PROBE-HARNESS] llama_init_from_model failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    // The constructor's own runtime-context call (inside llama_init_from_model
    // above) is THE publish -- the plan under test is whatever it committed.

    // Step 4: a SYCL backend for the SAME device the context uses. The
    // transaction (ggml_sycl_run_runtime_context_transaction(), ggml-sycl.cpp)
    // reads only `((ggml_backend_sycl_context *) backend->context)->device`,
    // so a SECOND ggml_backend_t instance for the same device is equivalent
    // for this purpose -- it does not need to be the context's own internal
    // backend object (which llama_context does not expose). With the
    // registration's ONEAPI_DEVICE_SELECTOR pinned to a single GPU
    // (level_zero:1), at most one SYCL GPU device is enumerated in this
    // process, so "first device whose reg==sycl_reg && type==GPU" is
    // unambiguously that same device.
    ggml_backend_dev_t sycl_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_backend_reg(dev) == ggml_backend_sycl_reg() &&
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            sycl_dev = dev;
            break;
        }
    }
    if (!sycl_dev) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: no SYCL GPU device enumerated\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(sycl_dev, nullptr);
    if (!backend) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: ggml_backend_dev_init failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Step 5: exactly src/llama-context.cpp's own token construction.
    const auto & owner = model->get_sycl_model_token();
    if (owner.model_id == 0 || owner.load_txn_id == 0) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: no sycl model token\n");
        ggml_backend_free(backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const ggml_sycl_model_token token = { owner.model_id, owner.load_txn_id, owner.slot, owner.slot_generation };

    bool ok = true;

    fputs("PROBE_BEGIN\n", stdout);
    fputs("PROBE_BEGIN\n", stderr);
    fflush(stdout);
    fflush(stderr);

    // No public accessor exposes llama_context's RESOLVED flash-attention
    // state (llama_flash_attn_type only names the requested policy, e.g.
    // AUTO, not what it resolved to) -- pass true, matching this shared
    // body's own AUTO-context comment (ggml-sycl.cpp: "an AUTO
    // llama_flash_attn_type has not been resolved yet ... reads an
    // optimistic `true`").
    const bool flash_attn_enabled = true;

    ggml_sycl_runtime_context_probe  out512{};
    const ggml_sycl_lifecycle_result rc512 =
        ggml_backend_sycl_probe_runtime_context_for_model(backend, token, 4096, 512, 1, flash_attn_enabled, &out512);
    printf("PROBE 512: rc=%d accepted=%d would_demote_kv=%d host_kv_bytes=%zu reason=%s\n", (int) rc512,
           out512.accepted ? 1 : 0, out512.would_demote_kv ? 1 : 0, out512.host_kv_bytes,
           out512.reason ? out512.reason : "-");
    fflush(stdout);

    ggml_sycl_runtime_context_probe  out8192{};
    const ggml_sycl_lifecycle_result rc8192 =
        ggml_backend_sycl_probe_runtime_context_for_model(backend, token, 8192, 8192, 1, flash_attn_enabled, &out8192);
    printf("PROBE 8192: rc=%d accepted=%d would_demote_kv=%d host_kv_bytes=%zu reason=%s\n", (int) rc8192,
           out8192.accepted ? 1 : 0, out8192.would_demote_kv ? 1 : 0, out8192.host_kv_bytes,
           out8192.reason ? out8192.reason : "-");
    fflush(stdout);

    fputs("PROBE_END\n", stdout);
    fputs("PROBE_END\n", stderr);
    fflush(stdout);
    fflush(stderr);

    // Step 10. Device index 0: with the registration's single-GPU selector
    // pin, at most one SYCL device is enumerated in-process (see the
    // sycl_dev search above), and ggml_backend_sycl_compute_buffer_host_
    // fallbacks() indexes SYCL's own in-process device numbering, not the
    // ggml backend-device registry index used above.
    printf("HOST_FALLBACKS_AFTER=%llu\n", (unsigned long long) ggml_backend_sycl_compute_buffer_host_fallbacks(0));
    fflush(stdout);

    // Step 11: prove the published state (at n_ctx=4096, n_ubatch=512 --
    // its ORIGINAL size, unaffected by either probe above) is still usable.
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    std::string         sentence;
    for (int i = 0; i < 40; ++i) {
        sentence += "The quick brown fox jumps over the lazy dog. ";
    }
    std::vector<llama_token> tokens(sentence.size() + 32);
    const int32_t            n_tok =
        llama_tokenize(vocab, sentence.c_str(), (int32_t) sentence.size(), tokens.data(), (int32_t) tokens.size(),
                       /*add_special=*/true, /*parse_special=*/false);
    if (n_tok <= 0) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: tokenize returned %d\n", n_tok);
        ok = false;
    }
    int decode_rc = -1;
    if (ok) {
        tokens.resize((size_t) n_tok);
        // n_batch=2048 (set above) is >= n_tok, so this is ONE llama_batch;
        // llama_decode splits it internally into ubatches of n_ubatch=512.
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
        decode_rc         = llama_decode(ctx, batch);
    }
    printf("DECODE_AFTER_PROBES rc=%d n_tokens=%d\n", decode_rc, n_tok);
    fflush(stdout);

    if (!(rc512 == GGML_SYCL_LIFECYCLE_OK && out512.accepted && !out512.would_demote_kv)) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: probe(512) did not accept cleanly\n");
        ok = false;
    }
    if (!(rc8192 == GGML_SYCL_LIFECYCLE_PLAN_REJECTED && !out8192.accepted && out8192.reason)) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: probe(8192) did not refuse as expected\n");
        ok = false;
    }
    if (decode_rc != 0) {
        fprintf(stderr, "[PROBE-HARNESS] FAIL: decode after probes\n");
        ok = false;
    }

    ggml_backend_free(backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return ok ? 0 : 1;
}
