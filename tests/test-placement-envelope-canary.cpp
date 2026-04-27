#include "llama.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct options {
    std::string model;
    uint32_t    envelope_ctx       = 1024;
    uint32_t    ubatch             = 16;
    uint32_t    seq_max            = 4;
    uint32_t    over_ctx           = 2048;
    int32_t     ngl                = 0;
    int         iterations         = 2;
    int         threads            = 2;
    bool        probe_over         = false;
    bool        expect_over_reject = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL [--ctx N] [--ubatch N] [--seq-max N]\n"
                 "          [--over-ctx N] [--ngl N] [--iterations N] [--threads N]\n"
                 "          [--probe-over] [--expect-over-reject]\n",
                 argv0);
}

static bool parse_u32(const char * s, uint32_t & out) {
    char *              end = nullptr;
    const unsigned long v   = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v > UINT32_MAX) {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

static bool parse_i32(const char * s, int32_t & out) {
    char *     end = nullptr;
    const long v   = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < INT32_MIN || v > INT32_MAX) {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static bool parse_int(const char * s, int & out) {
    int32_t v = 0;
    if (!parse_i32(s, v)) {
        return false;
    }
    out = (int) v;
    return true;
}

static bool parse_args(int argc, char ** argv, options & opt) {
    for (int i = 1; i < argc; ++i) {
        const char * arg        = argv[i];
        auto         need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(arg, "--model") == 0 || std::strcmp(arg, "-m") == 0) {
            const char * v = need_value(arg);
            if (!v) {
                return false;
            }
            opt.model = v;
        } else if (std::strcmp(arg, "--ctx") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_u32(v, opt.envelope_ctx)) {
                return false;
            }
        } else if (std::strcmp(arg, "--ubatch") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_u32(v, opt.ubatch)) {
                return false;
            }
        } else if (std::strcmp(arg, "--seq-max") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_u32(v, opt.seq_max)) {
                return false;
            }
        } else if (std::strcmp(arg, "--over-ctx") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_u32(v, opt.over_ctx)) {
                return false;
            }
        } else if (std::strcmp(arg, "--ngl") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_i32(v, opt.ngl)) {
                return false;
            }
        } else if (std::strcmp(arg, "--iterations") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_int(v, opt.iterations)) {
                return false;
            }
        } else if (std::strcmp(arg, "--threads") == 0) {
            const char * v = need_value(arg);
            if (!v || !parse_int(v, opt.threads)) {
                return false;
            }
        } else if (std::strcmp(arg, "--probe-over") == 0) {
            opt.probe_over = true;
        } else if (std::strcmp(arg, "--expect-over-reject") == 0) {
            opt.probe_over         = true;
            opt.expect_over_reject = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }

    if (opt.model.empty() || opt.envelope_ctx == 0 || opt.ubatch == 0 || opt.seq_max == 0 || opt.iterations < 1 ||
        opt.threads < 1) {
        return false;
    }
    if (opt.over_ctx <= opt.envelope_ctx) {
        opt.over_ctx = opt.envelope_ctx + 256;
    }
    return true;
}

static uint32_t pad_256(uint32_t v) {
    return ((v + 255u) / 256u) * 256u;
}

static llama_context_params make_context_params(uint32_t n_ctx,
                                                uint32_t n_seq_max,
                                                uint32_t n_ubatch,
                                                bool     kv_unified) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx;
    cparams.n_batch              = n_ubatch;
    cparams.n_ubatch             = n_ubatch;
    cparams.n_seq_max            = n_seq_max;
    cparams.kv_unified           = kv_unified;
    cparams.no_perf              = true;
    cparams.offload_kqv          = false;
    return cparams;
}

static bool verify_context_shape(llama_context * ctx,
                                 uint32_t        requested_ctx,
                                 uint32_t        requested_seq_max,
                                 bool            kv_unified) {
    const uint32_t actual_ctx     = llama_n_ctx(ctx);
    const uint32_t actual_ctx_seq = llama_n_ctx_seq(ctx);
    const uint32_t actual_seq_max = llama_n_seq_max(ctx);
    const uint32_t padded_ctx     = pad_256(requested_ctx);
    const uint32_t expected_seq   = kv_unified ? padded_ctx : pad_256(padded_ctx / requested_seq_max);
    const uint32_t expected_ctx   = kv_unified ? padded_ctx : expected_seq * requested_seq_max;

    std::printf(
        "context shape: requested_ctx=%u requested_seq_max=%u kv_unified=%d actual_ctx=%u actual_ctx_seq=%u "
        "actual_seq_max=%u\n",
        requested_ctx, requested_seq_max, kv_unified ? 1 : 0, actual_ctx, actual_ctx_seq, actual_seq_max);

    if (actual_seq_max != requested_seq_max) {
        std::fprintf(stderr, "FAIL: n_seq_max mismatch: got %u expected %u\n", actual_seq_max, requested_seq_max);
        return false;
    }
    if (actual_ctx != expected_ctx || actual_ctx_seq != expected_seq) {
        std::fprintf(stderr, "FAIL: context geometry mismatch: got ctx=%u seq=%u expected ctx=%u seq=%u\n", actual_ctx,
                     actual_ctx_seq, expected_ctx, expected_seq);
        return false;
    }
    return true;
}

static bool create_check_destroy(llama_model * model,
                                 uint32_t      n_ctx,
                                 uint32_t      n_seq_max,
                                 uint32_t      n_ubatch,
                                 bool          kv_unified) {
    llama_context_params cparams = make_context_params(n_ctx, n_seq_max, n_ubatch, kv_unified);
    llama_context *      ctx     = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: llama_init_from_model returned null for ctx=%u seq=%u kv_unified=%d\n", n_ctx,
                     n_seq_max, kv_unified ? 1 : 0);
        return false;
    }
    const bool ok = verify_context_shape(ctx, n_ctx, n_seq_max, kv_unified);
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    options opt;
    if (!parse_args(argc, argv, opt)) {
        usage(argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = opt.ngl;
    mparams.n_ctx              = opt.envelope_ctx;
    mparams.n_ubatch           = opt.ubatch;

    std::printf("placement envelope canary: model=%s envelope_ctx=%u ubatch=%u seq_max=%u ngl=%d\n", opt.model.c_str(),
                opt.envelope_ctx, opt.ubatch, opt.seq_max, opt.ngl);

    llama_model * model = llama_model_load_from_file(opt.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "FAIL: model load failed\n");
        llama_backend_free();
        return 1;
    }

    bool ok = true;

    for (int i = 0; i < opt.iterations; ++i) {
        ok = create_check_destroy(model, opt.envelope_ctx, opt.seq_max, opt.ubatch, false) && ok;
        ok = create_check_destroy(model, opt.envelope_ctx, opt.seq_max, opt.ubatch, true) && ok;

        const uint32_t half_ctx = std::max<uint32_t>(256, opt.envelope_ctx / 2);
        ok                      = create_check_destroy(model, half_ctx, 1, opt.ubatch, false) && ok;
    }

    std::atomic<bool>        thread_failed(false);
    std::vector<std::thread> threads;
    for (int t = 0; t < opt.threads; ++t) {
        threads.emplace_back([&, t]() {
            const bool kv_unified = (t % 2) != 0;
            if (!create_check_destroy(model, opt.envelope_ctx, opt.seq_max, opt.ubatch, kv_unified)) {
                thread_failed.store(true);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    ok = !thread_failed.load() && ok;

    if (opt.probe_over) {
        llama_context_params over = make_context_params(opt.over_ctx, opt.seq_max, opt.ubatch, false);
        llama_context *      ctx  = llama_init_from_model(model, over);
        if (ctx) {
            std::printf(
                "ADMISSION_GAP: over-envelope context succeeded: envelope_ctx=%u requested_ctx=%u actual_ctx=%u\n",
                opt.envelope_ctx, opt.over_ctx, llama_n_ctx(ctx));
            llama_free(ctx);
            if (opt.expect_over_reject) {
                ok = false;
            }
        } else {
            std::printf("over-envelope context rejected: envelope_ctx=%u requested_ctx=%u\n", opt.envelope_ctx,
                        opt.over_ctx);
        }
    }

    llama_model_free(model);
    llama_backend_free();

    if (!ok) {
        std::fprintf(stderr, "placement envelope canary: FAIL\n");
        return 1;
    }

    std::printf("placement envelope canary: PASS\n");
    return 0;
}
