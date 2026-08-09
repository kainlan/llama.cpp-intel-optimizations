#include "ggml-backend-impl.h"
#include "ggml-sycl/fattn-onednn.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::printf("SKIP: GGML SYCL oneDNN was not compiled in.\n");
    return 77;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

static const char * plan_kind_name(ggml_sycl_onednn_fa_layout_kind kind) {
    switch (kind) {
        case ggml_sycl_onednn_fa_layout_kind::DIRECT:
            return "DIRECT";
        case ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED:
            return "MATERIALIZE_REQUIRED";
        case ggml_sycl_onednn_fa_layout_kind::REJECT:
            return "REJECT";
    }
    return "UNKNOWN";
}

static bool configure_bounded_runtime() {
    // Materialized K/V intentionally enter the unified allocator. This tiny
    // descriptor fixture must not initialize the model-scale full-VRAM arena
    // or its default 2 GiB pinned chunks merely to allocate a few KiB.
    return setenv("GGML_SYCL_VRAM_ARENA", "0", 1) == 0 && setenv("GGML_SYCL_PINNED_CHUNK_MB", "16", 1) == 0;
}

// SMALL keeps the original descriptor-fixture constants verbatim. PRODUCTION
// scales the index-proportional terms by the case's own dimensions, because
// those constants do not transfer: at D=128/n_kv=256 the SMALL formula's
// -0.005*(t+1) K term reaches -1.28, which drives the logit spread to ~15 and
// makes softmax dynamic range -- not the K/V strides -- the dominant source of
// error. A production case that failed for that reason would look exactly like
// "DIRECT is wrong at scale", which is the question this fixture exists to
// answer, so the confound has to be designed out rather than tolerated.
//
// PRODUCTION's coefficients are chosen so the OUTPUT magnitude lands in the
// same band as the SMALL cases (~0.01-0.25). That is what lets both profiles
// share the one 5e-3 absolute tolerance below without it meaning something
// different for each -- no per-case threshold, no widening.
enum class fill_profile {
    SMALL,
    PRODUCTION,
};

struct case_shape {
    const char * name;
    int          H_q;
    int          H_kv;
    int          D;
    int          n_q;
    int          n_kv;
    int          k_stride;
    int          v_stride;
    // Causal mask offset: token t is visible to query q when t <= q + kv_offset.
    // 0 reproduces the original prompt-shaped mask. A production decode step has
    // a short query block against a long cached context, so n_kv - n_q makes all
    // n_kv tokens participate; with 0 at n_q=16/n_kv=256, 240 of 256 KV tokens
    // are fully masked and the extra context length is numerically inert.
    int          kv_offset;
    fill_profile profile;
};

static size_t q_idx(const case_shape & sh, int h, int q, int d) {
    return (size_t) h * sh.n_q * sh.D + (size_t) q * sh.D + d;
}

static size_t kv_idx(int stride, int n_kv, int h, int t, int d) {
    return (size_t) h * n_kv * stride + (size_t) t * stride + d;
}

static size_t out_idx(const case_shape & sh, int h, int q, int d) {
    return (size_t) d + (size_t) sh.D * ((size_t) h + (size_t) sh.H_q * q);
}

static size_t mask_idx(const case_shape & sh, int q, int t) {
    return (size_t) q * sh.n_kv + t;
}

static float to_float(sycl::half x) {
    return static_cast<float>(x);
}

static void fill_inputs(const case_shape &        sh,
                        std::vector<sycl::half> & Q,
                        std::vector<sycl::half> & K,
                        std::vector<sycl::half> & V,
                        std::vector<sycl::half> & mask) {
    const bool  prod = (sh.profile == fill_profile::PRODUCTION);
    const float q_h  = prod ? 0.4f / (float) sh.H_q : 0.03f;
    const float q_q  = prod ? 0.1f / (float) sh.n_q : 0.007f;
    const float k_h  = prod ? 0.15f / (float) sh.H_kv : 0.02f;
    const float k_t  = prod ? 0.22f / (float) sh.n_kv : 0.005f;
    const float v_h  = prod ? 0.11f / (float) sh.H_kv : 0.011f;
    const float v_t  = prod ? 0.13f / (float) sh.n_kv : 0.013f;

    for (int h = 0; h < sh.H_q; ++h) {
        for (int q = 0; q < sh.n_q; ++q) {
            for (int d = 0; d < sh.D; ++d) {
                const float v         = q_h * (float) (h + 1) + q_q * (float) (q + 1) + 0.001f * (float) ((d % 7) - 3);
                Q[q_idx(sh, h, q, d)] = sycl::half(v);
            }
        }
    }
    for (int h = 0; h < sh.H_kv; ++h) {
        for (int t = 0; t < sh.n_kv; ++t) {
            for (int d = 0; d < sh.D; ++d) {
                const float k = k_h * (float) (h + 1) - k_t * (float) (t + 1) + 0.0008f * (float) ((d % 11) - 5);
                const float v = v_h * (float) (h + 1) + v_t * (float) (t + 1) + 0.0009f * (float) ((d % 13) - 6);
                K[kv_idx(sh.k_stride, sh.n_kv, h, t, d)] = sycl::half(k);
                V[kv_idx(sh.v_stride, sh.n_kv, h, t, d)] = sycl::half(v);
            }
        }
    }
    for (int q = 0; q < sh.n_q; ++q) {
        for (int t = 0; t < sh.n_kv; ++t) {
            mask[mask_idx(sh, q, t)] = sycl::half((t <= q + sh.kv_offset) ? 0.0f : -10000.0f);
        }
    }
}

static std::vector<float> reference_sdpa(const case_shape &              sh,
                                         const std::vector<sycl::half> & Q,
                                         const std::vector<sycl::half> & K,
                                         const std::vector<sycl::half> & V,
                                         const std::vector<sycl::half> & mask) {
    std::vector<float> out((size_t) sh.H_q * sh.n_q * sh.D, 0.0f);
    const int          n_rep = sh.H_q / sh.H_kv;
    const float        scale = 1.0f / std::sqrt((float) sh.D);

    for (int h = 0; h < sh.H_q; ++h) {
        const int kv_h = h / n_rep;
        for (int q = 0; q < sh.n_q; ++q) {
            std::vector<float> logits(sh.n_kv, 0.0f);
            float              max_logit = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < sh.n_kv; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < sh.D; ++d) {
                    dot += to_float(Q[q_idx(sh, h, q, d)]) * to_float(K[kv_idx(sh.k_stride, sh.n_kv, kv_h, t, d)]);
                }
                logits[t] = dot * scale + to_float(mask[mask_idx(sh, q, t)]);
                max_logit = std::max(max_logit, logits[t]);
            }

            float denom = 0.0f;
            for (int t = 0; t < sh.n_kv; ++t) {
                logits[t] = std::exp(logits[t] - max_logit);
                denom += logits[t];
            }
            for (int d = 0; d < sh.D; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < sh.n_kv; ++t) {
                    acc += (logits[t] / denom) * to_float(V[kv_idx(sh.v_stride, sh.n_kv, kv_h, t, d)]);
                }
                out[out_idx(sh, h, q, d)] = acc;
            }
        }
    }
    return out;
}

template <typename T> static T * malloc_device_copy(sycl::queue & q, const std::vector<T> & host) {
    T * ptr = sycl::malloc_device<T>(host.size(), q);
    if (!ptr) {
        return nullptr;
    }
    try {
        q.memcpy(ptr, host.data(), host.size() * sizeof(T)).wait();
        return ptr;
    } catch (...) {
        sycl::free(ptr, q);
        throw;
    }
}

struct backend_owner {
    ggml_backend_t backend;

    explicit backend_owner(ggml_backend_t backend) : backend(backend) {}

    backend_owner(const backend_owner &)             = delete;
    backend_owner & operator=(const backend_owner &) = delete;
    backend_owner(backend_owner &&)                  = delete;
    backend_owner & operator=(backend_owner &&)      = delete;

    ~backend_owner() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    ggml_backend_sycl_context & context() const { return *static_cast<ggml_backend_sycl_context *>(backend->context); }
};

struct case_device_buffers {
    sycl::queue & q;
    sycl::half *  Q    = nullptr;
    sycl::half *  K    = nullptr;
    sycl::half *  V    = nullptr;
    sycl::half *  mask = nullptr;
    float *       out  = nullptr;

    ~case_device_buffers() { release_noexcept(); }

    void release() {
        q.wait_and_throw();
        free_all();
    }

  private:
    void free_all() {
        if (Q) {
            sycl::free(Q, q);
            Q = nullptr;
        }
        if (K) {
            sycl::free(K, q);
            K = nullptr;
        }
        if (V) {
            sycl::free(V, q);
            V = nullptr;
        }
        if (mask) {
            sycl::free(mask, q);
            mask = nullptr;
        }
        if (out) {
            sycl::free(out, q);
            out = nullptr;
        }
    }

    void release_noexcept() noexcept {
        try {
            q.wait_and_throw();
        } catch (...) {
            // The original exception is reported by main; cleanup must continue.
        }
        try {
            free_all();
        } catch (...) {
            // Never replace the test failure or an in-flight SYCL exception.
        }
    }
};

static bool run_case(ggml_backend_sycl_context & ctx, const case_shape & sh) {
    std::vector<sycl::half> Q((size_t) sh.H_q * sh.n_q * sh.D);
    std::vector<sycl::half> K((size_t) sh.H_kv * sh.n_kv * sh.k_stride, sycl::half(0.0f));
    std::vector<sycl::half> V((size_t) sh.H_kv * sh.n_kv * sh.v_stride, sycl::half(0.0f));
    std::vector<sycl::half> mask((size_t) sh.n_q * sh.n_kv);
    fill_inputs(sh, Q, K, V, mask);
    const std::vector<float> expected = reference_sdpa(sh, Q, K, V, mask);

    sycl::queue *       q       = ctx.stream();
    case_device_buffers buffers = { *q };
    buffers.Q                   = malloc_device_copy(*q, Q);
    buffers.K                   = malloc_device_copy(*q, K);
    buffers.V                   = malloc_device_copy(*q, V);
    buffers.mask                = malloc_device_copy(*q, mask);
    buffers.out                 = sycl::malloc_device<float>(expected.size(), *q);
    TEST_ASSERT(buffers.Q && buffers.K && buffers.V && buffers.mask && buffers.out, "device allocation failed");
    q->memset(buffers.out, 0, expected.size() * sizeof(float)).wait();

    fattn_params params{};
    params.Q         = reinterpret_cast<const char *>(buffers.Q);
    params.K         = reinterpret_cast<const char *>(buffers.K);
    params.V         = reinterpret_cast<const char *>(buffers.V);
    params.mask      = reinterpret_cast<const char *>(buffers.mask);
    params.dst       = buffers.out;
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.scale     = 1.0f / std::sqrt((float) sh.D);
    params.ne00      = sh.D;
    params.ne01      = sh.n_q;
    params.ne02      = sh.H_q;
    params.ne03      = 1;
    params.nb01      = sh.D * (int) sizeof(sycl::half);
    params.nb02      = params.nb01 * sh.n_q;
    params.nb03      = params.nb02 * sh.H_q;
    params.ne10      = sh.D;
    params.ne11      = sh.n_kv;
    params.ne12      = sh.H_kv;
    params.ne13      = 1;
    params.nb11      = sh.k_stride * (int) sizeof(sycl::half);
    params.nb12      = params.nb11 * sh.n_kv;
    params.nb13      = (int64_t) params.nb12 * sh.H_kv;
    params.nb21      = sh.v_stride * (int) sizeof(sycl::half);
    params.nb22      = params.nb21 * sh.n_kv;
    params.nb23      = (int64_t) params.nb22 * sh.H_kv;
    params.ne30      = sh.n_kv;
    params.ne31      = sh.n_q;
    params.ne32      = 1;
    params.ne33      = 1;
    params.nb31      = sh.n_kv * (int) sizeof(sycl::half);
    params.nb32      = params.nb31 * sh.n_q;
    params.nb33      = params.nb32;
    params.prec      = GGML_PREC_F32;

    // Record the routing decision from the planner itself, not from an
    // environment-gated log line. ggml_sycl_flash_attn_ext_onednn calls this
    // same function with these same arguments, so the printed kind is the
    // branch the execute below takes. GGML_SYCL_FA_DISPATCH_DEBUG=1 prints the
    // corresponding production line and is the independent cross-check; if the
    // two ever disagree, that disagreement is the finding.
    const ggml_sycl_onednn_fa_layout_plan plan =
        ggml_sycl_flash_attn_ext_onednn_plan(params, sh.H_q, sh.H_kv, params.kv_is_fp8, params.n_seqs > 1);
    std::printf("%s plan=%s\n", sh.name, plan_kind_name(plan.kind));

    const bool executed = ggml_sycl_flash_attn_ext_onednn(ctx, params);
    q->wait_and_throw();

    std::vector<float> actual(expected.size(), 0.0f);
    if (executed) {
        q->memcpy(actual.data(), buffers.out, actual.size() * sizeof(float)).wait();
    }
    buffers.release();

    TEST_ASSERT(executed, (std::string(sh.name) + " oneDNN dispatch did not execute").c_str());
    TEST_ASSERT(!actual.empty(), (std::string(sh.name) + " produced an empty host result").c_str());

    float max_abs = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    }
    if (max_abs > 5e-3f) {
        std::fprintf(stderr, "FAIL: %s max_abs=%g\n", sh.name, max_abs);
        for (size_t i = 0; i < std::min<size_t>(expected.size(), 16); ++i) {
            std::fprintf(stderr, "  [%zu] actual=%g expected=%g\n", i, actual[i], expected[i]);
        }
        return false;
    }
    std::printf("%s max_abs=%g\n", sh.name, max_abs);
    return true;
}

static int run_descriptor_tests() {
    // Keep progress/final markers deterministic even when CTest redirects output.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }
    if (!configure_bounded_runtime()) {
        std::fprintf(stderr, "FAIL: could not configure bounded SYCL test allocations\n");
        return 1;
    }
    std::printf("Bounded runtime: VRAM arena disabled, pinned chunks capped at 16 MiB\n");

    std::vector<sycl::device> gpus;
    try {
        gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "FAIL: SYCL GPU discovery failed: %s\n", e.what());
        return 1;
    }
    if (gpus.empty()) {
        std::printf("SKIP: no SYCL GPU device is available.\n");
        return 77;
    }

    const sycl::device & device = gpus.front();
    if (!device.has(sycl::aspect::usm_device_allocations) || !device.has(sycl::aspect::fp16)) {
        std::printf("SKIP: SYCL GPU lacks device USM or fp16 required by the oneDNN path.\n");
        return 77;
    }

    bool ok = true;
    try {
        // Use the production backend owner: ggml_backend_free performs the
        // global queue/pool drains before deleting its context. Direct stack
        // construction bypasses that ordering and is not a supported backend
        // lifetime.
        backend_owner owner{ ggml_backend_sycl_init(0) };
        if (!owner.backend) {
            std::fprintf(stderr, "FAIL: could not initialize SYCL backend\n");
            return 1;
        }
        ggml_backend_sycl_context & ctx = owner.context();
        // kv_offset=0 + fill_profile::SMALL reproduce these four cases exactly as
        // they were before those fields existed.
        ok &= run_case(ctx, { "MHA-4D-direct", 2, 2, 16, 8, 8, 16, 16, 0, fill_profile::SMALL });
        ok &= run_case(ctx, { "GQA-5D-direct", 4, 2, 16, 8, 8, 16, 16, 0, fill_profile::SMALL });
        ok &= run_case(ctx, { "GQA-5D-materialized", 4, 2, 16, 8, 8, 19, 21, 0, fill_profile::SMALL });
        ok &= run_case(ctx, { "MQA-5D-materialized", 4, 1, 16, 8, 8, 19, 21, 0, fill_profile::SMALL });

        // Production-scale GQA (llama.cpp-l7rt). Mistral's decode shape, the one
        // CLAUDE.md records from a live run as
        //   [SYCL] fattn: oneDNN MATERIALIZED D=128 ne01=16 ne11=256 H_q=32 H_kv=8
        //
        // These two exist to answer whether the nc!=D materialize gate at
        // fattn-onednn.cpp:182 is a CORRECTNESS requirement or an optimization,
        // at the shape production actually runs. They are a matched pair and
        // neither is useful alone:
        //
        //   -dense       k_stride == v_stride == D, so the gate does not fire and
        //                the planner says DIRECT on an UNMUTATED build. This is
        //                the positive control at scale: it separates "DIRECT is
        //                wrong for STRIDED K/V" from "DIRECT, or this fixture's
        //                tolerance, is wrong at D=128/n_kv=256 generally".
        //                Without it a red -strided result is uninterpretable.
        //   -strided     k_stride=131, v_stride=133 (coprime with D and with each
        //                other, so no accidental alignment rescues a stride bug),
        //                so the gate fires and the planner says
        //                MATERIALIZE_REQUIRED unmutated. Prefixing `false &&` to
        //                the :182 predicate is what routes this same case through
        //                DIRECT, which is the measurement the ticket wants.
        ok &= run_case(ctx, { "GQA-prod-dense", 32, 8, 128, 16, 256, 128, 128, 240, fill_profile::PRODUCTION });
        ok &= run_case(ctx, { "GQA-prod-strided", 32, 8, 128, 16, 256, 131, 133, 240, fill_profile::PRODUCTION });
        ctx.stream()->wait_and_throw();
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "FAIL: SYCL exception: %s\n", e.what());
        ok = false;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        ok = false;
    }
    std::printf("SYCL oneDNN FA descriptor tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    // Run SYCL in a fresh child so waitpid gives an unambiguous mutation proof:
    // the numerical failure must be WIFEXITED with status 1, never a signal.
    const pid_t child = fork();
    if (child < 0) {
        std::fprintf(stderr, "FAIL: fork failed: errno=%d\n", errno);
        return 2;
    }
    if (child == 0) {
        // Return normally from main after run_descriptor_tests has destroyed
        // its backend owner. This also exercises child static/TLS/atexit
        // teardown; _exit would hide failures in exactly that lifetime tail.
        return run_descriptor_tests();
    }

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            std::fprintf(stderr, "FAIL: waitpid failed: errno=%d\n", errno);
            return 2;
        }
    }
    if (WIFSIGNALED(status)) {
        std::fprintf(stderr, "FAIL: descriptor child terminated by signal %d\n", WTERMSIG(status));
        return 2;
    }
    if (!WIFEXITED(status)) {
        std::fprintf(stderr, "FAIL: descriptor child did not exit normally\n");
        return 2;
    }

    const int rc = WEXITSTATUS(status);
    std::fprintf(stderr, "Descriptor child WIFEXITED status=%d\n", rc);
    return rc;
}

#endif
