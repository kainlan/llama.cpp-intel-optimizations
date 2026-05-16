#include "ggml-sycl/fattn-onednn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::printf("GGML SYCL oneDNN not enabled; skipping test.\n");
    return 0;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

struct case_shape {
    const char * name;
    int          H_q;
    int          H_kv;
    int          D;
    int          n_q;
    int          n_kv;
    int          k_stride;
    int          v_stride;
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
    for (int h = 0; h < sh.H_q; ++h) {
        for (int q = 0; q < sh.n_q; ++q) {
            for (int d = 0; d < sh.D; ++d) {
                const float v = 0.03f * (float) (h + 1) + 0.007f * (float) (q + 1) + 0.001f * (float) ((d % 7) - 3);
                Q[q_idx(sh, h, q, d)] = sycl::half(v);
            }
        }
    }
    for (int h = 0; h < sh.H_kv; ++h) {
        for (int t = 0; t < sh.n_kv; ++t) {
            for (int d = 0; d < sh.D; ++d) {
                const float k = 0.02f * (float) (h + 1) - 0.005f * (float) (t + 1) + 0.0008f * (float) ((d % 11) - 5);
                const float v = 0.011f * (float) (h + 1) + 0.013f * (float) (t + 1) + 0.0009f * (float) ((d % 13) - 6);
                K[kv_idx(sh.k_stride, sh.n_kv, h, t, d)] = sycl::half(k);
                V[kv_idx(sh.v_stride, sh.n_kv, h, t, d)] = sycl::half(v);
            }
        }
    }
    for (int q = 0; q < sh.n_q; ++q) {
        for (int t = 0; t < sh.n_kv; ++t) {
            mask[mask_idx(sh, q, t)] = sycl::half((t <= q) ? 0.0f : -10000.0f);
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
    q.memcpy(ptr, host.data(), host.size() * sizeof(T)).wait();
    return ptr;
}

static bool run_case(const case_shape & sh) {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    std::vector<sycl::half> Q((size_t) sh.H_q * sh.n_q * sh.D);
    std::vector<sycl::half> K((size_t) sh.H_kv * sh.n_kv * sh.k_stride, sycl::half(0.0f));
    std::vector<sycl::half> V((size_t) sh.H_kv * sh.n_kv * sh.v_stride, sycl::half(0.0f));
    std::vector<sycl::half> mask((size_t) sh.n_q * sh.n_kv);
    fill_inputs(sh, Q, K, V, mask);
    const std::vector<float> expected = reference_sdpa(sh, Q, K, V, mask);

    ggml_backend_sycl_context ctx(0);
    sycl::queue *             q       = ctx.stream();
    sycl::half *              q_dev   = malloc_device_copy(*q, Q);
    sycl::half *              k_dev   = malloc_device_copy(*q, K);
    sycl::half *              v_dev   = malloc_device_copy(*q, V);
    sycl::half *              m_dev   = malloc_device_copy(*q, mask);
    float *                   out_dev = sycl::malloc_device<float>(expected.size(), *q);
    TEST_ASSERT(q_dev && k_dev && v_dev && m_dev && out_dev, "device allocation failed");
    q->memset(out_dev, 0, expected.size() * sizeof(float)).wait();

    fattn_params params{};
    params.Q         = reinterpret_cast<const char *>(q_dev);
    params.K         = reinterpret_cast<const char *>(k_dev);
    params.V         = reinterpret_cast<const char *>(v_dev);
    params.mask      = reinterpret_cast<const char *>(m_dev);
    params.dst       = out_dev;
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

    const bool executed = ggml_sycl_flash_attn_ext_onednn(ctx, params);
    q->wait_and_throw();

    std::vector<float> actual(expected.size(), 0.0f);
    if (executed) {
        q->memcpy(actual.data(), out_dev, actual.size() * sizeof(float)).wait();
    }

    sycl::free(q_dev, *q);
    sycl::free(k_dev, *q);
    sycl::free(v_dev, *q);
    sycl::free(m_dev, *q);
    sycl::free(out_dev, *q);

    TEST_ASSERT(executed, (std::string(sh.name) + " oneDNN dispatch did not execute").c_str());

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

int main() {
    bool ok = true;
    ok &= run_case({ "MHA-4D-direct", 2, 2, 16, 8, 8, 16, 16 });
    ok &= run_case({ "GQA-5D-direct", 4, 2, 16, 8, 8, 16, 16 });
    ok &= run_case({ "GQA-5D-materialized", 4, 2, 16, 8, 8, 19, 21 });
    ok &= run_case({ "MQA-5D-materialized", 4, 1, 16, 8, 8, 19, 21 });
    std::printf("SYCL oneDNN FA descriptor tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
