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
    //
    // llama.cpp-tsfl round 2 F-lead-2: prefix every line with a level
    // letter so the lead's acceptance script can grep `^E ` between
    // PROBE_BEGIN/PROBE_END and score "prints no ERROR" -- without this an
    // ERROR line was indistinguishable from an INFO one in the log. ggml
    // log text already carries its own trailing newline, so the prefix is
    // prepended, not appended.
    char letter = '?';
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            letter = 'E';
            break;
        case GGML_LOG_LEVEL_WARN:
            letter = 'W';
            break;
        case GGML_LOG_LEVEL_INFO:
            letter = 'I';
            break;
        case GGML_LOG_LEVEL_DEBUG:
            letter = 'D';
            break;
        case GGML_LOG_LEVEL_CONT:
            letter = 'C';
            break;
        default:
            letter = '?';
            break;
    }
    fprintf(stderr, "%c %s", letter, text);
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : nullptr;
    if (!model_path) {
        model_path = std::getenv("LLAMACPP_TEST_MODELFILE");
    }
    if (!model_path || !model_path[0]) {
        // llama.cpp-tsfl round 2 G6: print SKIP: first -- test_skip_no_model()
        // (test-skip.h) prints its own "WARNING: No model file provided..."
        // and exits 77 itself, but does not spell the literal "SKIP: " the
        // spec requires callers be able to grep for.
        fprintf(stderr, "SKIP: no model file provided\n");
        test_skip_no_model();
    }
    {
        FILE * f = std::fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "SKIP: model file not readable: %s\n", model_path);
            test_skip_no_model();
        }
        std::fclose(f);
    }

    llama_log_set(log_everything, nullptr);

    llama_backend_init();

    if (!llama_supports_gpu_offload()) {
        fprintf(stderr, "SKIP: no GPU backend registered\n");
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
    // llama.cpp-uajm: llama_context_default_params() defaults swa_full=true
    // (src/llama-context.cpp:4149) but every tool runs with common's
    // default false (common/common.h:571); the SYCL KV plan sizes SWA
    // layers by the window, so the raw default overflows the planned slab
    // at context init on GPT-OSS. Match the tools.
    cparams.swa_full             = false;
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
        if (!dev) {
            // raced-null safety (fork 51d116467): enumeration can return null
            // slots while backends register; skip rather than deref.
            continue;
        }
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
        // llama.cpp-tsfl round 2 G4: llama_free(ctx)/llama_model_free(model)
        // BEFORE ggml_backend_free(backend) -- see the final cleanup's own
        // comment for why the order matters.
        llama_free(ctx);
        llama_model_free(model);
        ggml_backend_free(backend);
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
    // AUTO, not what it resolved to) -- pass true. Cite corrected round 2
    // G5: cparams.flash_attn = params.flash_attn_type !=
    // LLAMA_FLASH_ATTN_TYPE_DISABLED (src/llama-context.cpp:542) is the
    // resolution this would ideally read; llama_context's own
    // sycl_recheck_runtime_context_flash_attn()'s doc comment
    // (src/llama-context.h:279-284) is where the "constructor's own call
    // ... sees an optimistic `true`" reasoning this line paraphrases
    // actually lives, not ggml-sycl.cpp.
    const bool flash_attn_enabled = true;

    ggml_sycl_runtime_context_probe  out512{};
    const ggml_sycl_lifecycle_result rc512 = ggml_backend_sycl_probe_runtime_context_for_model(
        backend, token, 4096, 512, 1, /*kv_unified=*/false, flash_attn_enabled, &out512);
    printf("PROBE 512: rc=%d accepted=%d would_demote_kv=%d host_kv_bytes=%zu reason=%s\n", (int) rc512,
           out512.accepted ? 1 : 0, out512.would_demote_kv ? 1 : 0, out512.host_kv_bytes,
           out512.reason ? out512.reason : "-");
    fflush(stdout);

    ggml_sycl_runtime_context_probe  out8192{};
    const ggml_sycl_lifecycle_result rc8192 = ggml_backend_sycl_probe_runtime_context_for_model(
        backend, token, 8192, 8192, 1, /*kv_unified=*/false, flash_attn_enabled, &out8192);
    printf("PROBE 8192: rc=%d accepted=%d would_demote_kv=%d host_kv_bytes=%zu reason=%s\n", (int) rc8192,
           out8192.accepted ? 1 : 0, out8192.would_demote_kv ? 1 : 0, out8192.host_kv_bytes,
           out8192.reason ? out8192.reason : "-");
    fflush(stdout);

    fputs("PROBE_END\n", stdout);
    fputs("PROBE_END\n", stderr);
    fflush(stdout);
    fflush(stderr);

    // Step 10. Device index 0 here is SYCL's OWN in-process device index
    // after ONEAPI_DEVICE_SELECTOR filtering -- not the ggml backend-device
    // registry index used by the sycl_dev search above, a different
    // numbering entirely. This registration pins ONEAPI_DEVICE_SELECTOR=
    // level_zero:1, so exactly one GPU is enumerated in-process and it is
    // unambiguously index 0. Running this binary UNPINNED (directly, not
    // via ctest) makes 0 whichever GPU the driver enumerates first (the
    // B70, per CLAUDE.md's device-topology table) while the context may
    // have been placed on a different device entirely -- reviewer's ruling,
    // round 2: do not read this figure from a direct/unpinned invocation.
    //
    // llama.cpp-tsfl round 5 R5: the figure is SCORED only when the
    // selector is pinned -- the ctest registration pins it (and the lead's
    // scripts also pin it), so under ctest this is a real assertion, not
    // merely printed and ignored. An unpinned direct invocation cannot
    // attribute index 0 to any particular device (see the reviewer's ruling
    // above), so it prints the value only, unscored.
    // llama.cpp-tsfl round 6 S2: "pinned" means ONE device -- a multi-device
    // selector (level_zero:0,1) sets the variable but leaves index 0 just as
    // unattributable, so only a comma-free value scores the figure.
    const char *   selector             = std::getenv("ONEAPI_DEVICE_SELECTOR");
    const bool     selector_pinned      = selector != nullptr && std::strchr(selector, ',') == nullptr;
    const uint64_t host_fallbacks_after = ggml_backend_sycl_compute_buffer_host_fallbacks(0);
    printf("HOST_FALLBACKS_AFTER=%llu\n", (unsigned long long) host_fallbacks_after);
    if (selector_pinned && host_fallbacks_after != 0) {
        fprintf(stderr,
                "[PROBE-HARNESS] FAIL: host fallbacks after probes = %llu (expected 0 under a pinned "
                "selector)\n",
                (unsigned long long) host_fallbacks_after);
        ok = false;
    }
    fflush(stdout);

    // Step 11: prove the published state (at n_ctx=4096, n_ubatch=512 --
    // its ORIGINAL size, unaffected by either probe above) is still usable.
    // llama.cpp-tsfl round 2 G3: 60 repetitions (was 40, ~400-450 tokens --
    // too few to ever cross the n_ubatch=512 boundary, so the two-ubatch
    // split this step exists to exercise was never actually reached). 60
    // reps measures ~600-660 tokens on this sentence, comfortably over 512.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string         sentence;
    for (int i = 0; i < 60; ++i) {
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
    // llama.cpp-tsfl round 2 G3: harness-side assertion that the prompt
    // actually crosses n_ubatch -- a silent drop back to a single-ubatch
    // prompt (e.g. a future edit that shortens the repeated sentence, or
    // changes cparams.n_ubatch above) would otherwise pass this step
    // vacuously again. llama.cpp-tsfl round 4 Q7: compares against the
    // ACTUAL cparams.n_ubatch, not a hardcoded 512, so the check tracks
    // the real value if that ever changes too.
    if (ok && n_tok <= (int32_t) cparams.n_ubatch) {
        fprintf(stderr,
                "[PROBE-HARNESS] FAIL: prompt is only %d tokens, must exceed n_ubatch=%d to actually "
                "exercise the two-ubatch split\n",
                n_tok, (int32_t) cparams.n_ubatch);
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

    // llama.cpp-tsfl round 2 G4: llama_free(ctx) and llama_model_free(model)
    // -- which tear down the CONTEXT's own SYCL backend -- must run BEFORE
    // ggml_backend_free(backend) frees this harness's auxiliary one.
    // ggml_backend_sycl_free() (ggml-sycl.cpp, the static function of that
    // name; line cites into that file rot every commit) tears down
    // PROCESS-GLOBAL state (the prestage thread, pipeline copy queues,
    // tp_free, the pinned-owner shutdown GGML_ASSERT, the FP16 cache, split
    // rings) regardless of which ggml_backend_t instance triggers it, so
    // freeing the auxiliary backend FIRST would tear that global state down
    // while the context's own SYCL backend is still live.
    llama_free(ctx);
    llama_model_free(model);
    ggml_backend_free(backend);
    llama_backend_free();

    return ok ? 0 : 1;
}
