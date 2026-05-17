#include "ggml-sycl/fattn-esimd-f16.hpp"
#include "ggml-sycl/fattn-onednn.hpp"
#include "ggml-sycl/fattn-xmx-f16-v2.hpp"
#include "ggml-sycl/fattn-xmx-f16.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    std::printf("GGML SYCL not enabled; skipping bench.\n");
    return 0;
}
#else

static constexpr int D     = 64;
static constexpr int N_Q   = 512;
static constexpr int N_KV  = 512;
static constexpr int H_Q   = 64;
static constexpr int H_KV  = 8;
static constexpr int N_REP = H_Q / H_KV;

struct bench_shape {
    const char * name;
    float        logit_softcap;
};

struct bench_timing {
    double mean_us = 0.0;
    double min_us  = 0.0;
    double max_us  = 0.0;
};

struct sample_ref {
    int   h;
    int   q;
    int   d;
    float value;
};

enum class output_layout {
    public_qhd,
    head_major_hqd,
};

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

static size_t q_idx(int h, int q, int d) {
    return ((size_t) h * N_Q + (size_t) q) * D + (size_t) d;
}

static size_t kv_idx(int h, int t, int d) {
    return ((size_t) h * N_KV + (size_t) t) * D + (size_t) d;
}

static size_t mask_idx(int q, int t) {
    return (size_t) q * N_KV + (size_t) t;
}

static size_t public_out_idx(int h, int q, int d) {
    return (size_t) d + (size_t) D * ((size_t) h + (size_t) H_Q * q);
}

static size_t head_major_out_idx(int h, int q, int d) {
    return ((size_t) h * N_Q + (size_t) q) * D + (size_t) d;
}

static float to_float(sycl::half value) {
    return static_cast<float>(value);
}

static void fill_inputs(std::vector<sycl::half> & Q,
                        std::vector<sycl::half> & K,
                        std::vector<sycl::half> & V,
                        std::vector<sycl::half> & mask,
                        std::vector<float> &      sinks) {
    for (int h = 0; h < H_Q; ++h) {
        for (int q = 0; q < N_Q; ++q) {
            for (int d = 0; d < D; ++d) {
                const float v = 0.17f * std::sin(0.013f * (float) (h + 3) * (float) (d + 1)) +
                                0.11f * std::cos(0.007f * (float) (q + 5) * (float) (d + 2));
                Q[q_idx(h, q, d)] = sycl::half(v);
            }
        }
    }

    for (int h = 0; h < H_KV; ++h) {
        for (int t = 0; t < N_KV; ++t) {
            for (int d = 0; d < D; ++d) {
                const float k = 0.19f * std::cos(0.011f * (float) (h + 1) * (float) (t + 1)) +
                                0.07f * std::sin(0.017f * (float) (d + 1));
                const float v = 0.21f * std::sin(0.009f * (float) (t + 3) * (float) (d + 1)) +
                                0.05f * std::cos(0.031f * (float) (h + 2) * (float) (d + 7));
                K[kv_idx(h, t, d)] = sycl::half(k);
                V[kv_idx(h, t, d)] = sycl::half(v);
            }
        }
    }

    for (int q = 0; q < N_Q; ++q) {
        for (int t = 0; t < N_KV; ++t) {
            mask[mask_idx(q, t)] = sycl::half(t <= q ? 0.0f : -10000.0f);
        }
    }

    for (int h = 0; h < H_Q; ++h) {
        sinks[h] = 0.25f + 0.07f * (float) (h % 13) - 0.11f * (float) ((h / 7) % 3);
    }
}

static float reference_value(const std::vector<sycl::half> & Q,
                             const std::vector<sycl::half> & K,
                             const std::vector<sycl::half> & V,
                             const std::vector<sycl::half> & mask,
                             const std::vector<float> &      sinks,
                             float                           scale,
                             float                           logit_softcap,
                             int                             h,
                             int                             q,
                             int                             d) {
    const int          kv_h = h / N_REP;
    std::vector<float> logits(N_KV);
    float              max_logit = sinks[h];

    for (int t = 0; t < N_KV; ++t) {
        float dot = 0.0f;
        for (int k = 0; k < D; ++k) {
            dot += to_float(Q[q_idx(h, q, k)]) * to_float(K[kv_idx(kv_h, t, k)]);
        }
        dot *= scale;
        if (logit_softcap != 0.0f) {
            dot = logit_softcap * std::tanh(dot);
        }
        logits[t] = dot + to_float(mask[mask_idx(q, t)]);
        max_logit = std::max(max_logit, logits[t]);
    }

    float denom = std::exp(sinks[h] - max_logit);
    float acc   = 0.0f;
    for (int t = 0; t < N_KV; ++t) {
        const float weight = std::exp(logits[t] - max_logit);
        denom += weight;
        acc += weight * to_float(V[kv_idx(kv_h, t, d)]);
    }

    return denom > 0.0f ? acc / denom : 0.0f;
}

static std::vector<sample_ref> build_reference_samples(const std::vector<sycl::half> & Q,
                                                       const std::vector<sycl::half> & K,
                                                       const std::vector<sycl::half> & V,
                                                       const std::vector<sycl::half> & mask,
                                                       const std::vector<float> &      sinks,
                                                       const fattn_params &            params) {
    const int heads[]   = { 0, 7, 8, 31, 63 };
    const int queries[] = { 0, 1, 17, 255, 511 };
    const int dims[]    = { 0, 1, 15, 32, 63 };

    std::vector<sample_ref> refs;
    refs.reserve((sizeof(heads) / sizeof(heads[0])) * (sizeof(queries) / sizeof(queries[0])) *
                 (sizeof(dims) / sizeof(dims[0])));
    for (int h : heads) {
        for (int q : queries) {
            for (int d : dims) {
                refs.push_back(
                    { h, q, d, reference_value(Q, K, V, mask, sinks, params.scale, params.logit_softcap, h, q, d) });
            }
        }
    }
    return refs;
}

static float sample_max_abs(const std::vector<float> &      actual,
                            const std::vector<sample_ref> & refs,
                            output_layout                   layout) {
    float max_abs = 0.0f;
    for (const sample_ref & ref : refs) {
        const size_t idx = layout == output_layout::public_qhd ? public_out_idx(ref.h, ref.q, ref.d) :
                                                                 head_major_out_idx(ref.h, ref.q, ref.d);
        max_abs          = std::max(max_abs, std::fabs(actual[idx] - ref.value));
    }
    return max_abs;
}

template <typename T> static T * malloc_device_copy(sycl::queue & q, const std::vector<T> & host) {
    T * ptr = sycl::malloc_device<T>(host.size(), q);
    if (!ptr) {
        return nullptr;
    }
    q.memcpy(ptr, host.data(), host.size() * sizeof(T)).wait();
    return ptr;
}

#    if GGML_SYCL_DNNL
static const char * onednn_reason_name(ggml_sycl_onednn_fa_layout_reason reason) {
    switch (reason) {
        case ggml_sycl_onednn_fa_layout_reason::OK:
            return "OK";
        case ggml_sycl_onednn_fa_layout_reason::BELOW_MIN_NCOLS:
            return "BELOW_MIN_NCOLS";
        case ggml_sycl_onednn_fa_layout_reason::SINKS_UNSUPPORTED:
            return "SINKS_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::SOFTCAP_UNSUPPORTED:
            return "SOFTCAP_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::FP8_KV_UNSUPPORTED:
            return "FP8_KV_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::MULTI_SEQ_UNSUPPORTED:
            return "MULTI_SEQ_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::BATCH_UNSUPPORTED:
            return "BATCH_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::PAGED_UNSUPPORTED:
            return "PAGED_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D:
            return "UNSUPPORTED_D";
        case ggml_sycl_onednn_fa_layout_reason::EMPTY_KV:
            return "EMPTY_KV";
        case ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH:
            return "KV_NC_STRIDE_MISMATCH";
        case ggml_sycl_onednn_fa_layout_reason::HEAD_RATIO_UNSUPPORTED:
            return "HEAD_RATIO_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::Q_TYPE_UNSUPPORTED:
            return "Q_TYPE_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::K_TYPE_UNSUPPORTED:
            return "K_TYPE_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::V_TYPE_UNSUPPORTED:
            return "V_TYPE_UNSUPPORTED";
        case ggml_sycl_onednn_fa_layout_reason::MASK_TYPE_UNSUPPORTED:
            return "MASK_TYPE_UNSUPPORTED";
    }
    return "UNKNOWN";
}
#    endif

static bench_timing time_path(sycl::queue & q, int warmup, int iters, const std::function<void()> & fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
        q.wait_and_throw();
    }

    std::vector<double> samples;
    samples.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        q.wait_and_throw();
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        q.wait_and_throw();
        const auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    bench_timing result;
    result.min_us = std::numeric_limits<double>::infinity();
    for (double value : samples) {
        result.mean_us += value;
        result.min_us = std::min(result.min_us, value);
        result.max_us = std::max(result.max_us, value);
    }
    result.mean_us /= (double) samples.size();
    return result;
}

static void print_device_caps(sycl::queue & q) {
    const sycl::device dev = q.get_device();
    const auto         sgs = dev.get_info<sycl::info::device::sub_group_sizes>();

    std::printf("device=%s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("  compute_units=%u max_wg=%zu global_mem=%.1f MiB max_alloc=%.1f MiB local_mem=%zu bytes\n",
                (unsigned) dev.get_info<sycl::info::device::max_compute_units>(),
                (size_t) dev.get_info<sycl::info::device::max_work_group_size>(),
                dev.get_info<sycl::info::device::global_mem_size>() / (1024.0 * 1024.0),
                dev.get_info<sycl::info::device::max_mem_alloc_size>() / (1024.0 * 1024.0),
                (size_t) dev.get_info<sycl::info::device::local_mem_size>());
    std::printf("  sub_group_sizes=");
    for (size_t i = 0; i < sgs.size(); ++i) {
        std::printf("%s%zu", i == 0 ? "" : ",", (size_t) sgs[i]);
    }
    std::printf(" xmx_v1=%d xmx_v2=%d esimd=%d\n", (int) fattn_xmx_f16_available(), (int) fattn_xmx_v2_f16_available(),
                (int) fattn_esimd_f16_available());
}

static void run_shape(const bench_shape & shape) {
    std::vector<sycl::half> Q((size_t) H_Q * N_Q * D);
    std::vector<sycl::half> K((size_t) H_KV * N_KV * D);
    std::vector<sycl::half> V((size_t) H_KV * N_KV * D);
    std::vector<sycl::half> mask((size_t) N_Q * N_KV);
    std::vector<float>      sinks(H_Q);
    fill_inputs(Q, K, V, mask, sinks);

    ggml_backend_sycl_context ctx(0);
    sycl::queue *             q = ctx.stream();
    print_device_caps(*q);

    sycl::half * q_dev     = malloc_device_copy(*q, Q);
    sycl::half * k_dev     = malloc_device_copy(*q, K);
    sycl::half * v_dev     = malloc_device_copy(*q, V);
    sycl::half * mask_dev  = malloc_device_copy(*q, mask);
    float *      sinks_dev = malloc_device_copy(*q, sinks);
    float *      out_dev   = sycl::malloc_device<float>((size_t) H_Q * N_Q * D, *q);
    if (!q_dev || !k_dev || !v_dev || !mask_dev || !sinks_dev || !out_dev) {
        throw std::runtime_error("device allocation failed");
    }

    const float base_scale = 1.0f / std::sqrt((float) D);

    fattn_params params{};
    params.Q                  = reinterpret_cast<const char *>(q_dev);
    params.K                  = reinterpret_cast<const char *>(k_dev);
    params.V                  = reinterpret_cast<const char *>(v_dev);
    params.mask               = reinterpret_cast<const char *>(mask_dev);
    params.sinks              = reinterpret_cast<const char *>(sinks_dev);
    params.dst                = out_dev;
    params.Q_type             = GGML_TYPE_F16;
    params.K_type             = GGML_TYPE_F16;
    params.V_type             = GGML_TYPE_F16;
    params.mask_type          = GGML_TYPE_F16;
    params.scale              = shape.logit_softcap == 0.0f ? base_scale : base_scale / shape.logit_softcap;
    params.logit_softcap      = shape.logit_softcap;
    params.prec               = GGML_PREC_F32;
    params.ne00               = D;
    params.ne01               = N_Q;
    params.ne02               = H_Q;
    params.ne03               = 1;
    params.nb01               = D * (int) sizeof(sycl::half);
    params.nb02               = params.nb01 * N_Q;
    params.nb03               = params.nb02 * H_Q;
    params.ne10               = D;
    params.ne11               = N_KV;
    params.ne12               = H_KV;
    params.ne13               = 1;
    params.nb11               = D * (int) sizeof(sycl::half);
    params.nb12               = params.nb11 * N_KV;
    params.nb13               = (int64_t) params.nb12 * H_KV;
    params.nb21               = D * (int) sizeof(sycl::half);
    params.nb22               = params.nb21 * N_KV;
    params.nb23               = (int64_t) params.nb22 * H_KV;
    params.ne30               = N_KV;
    params.ne31               = N_Q;
    params.ne32               = 1;
    params.ne33               = 1;
    params.nb31               = N_KV * (int) sizeof(sycl::half);
    params.nb32               = params.nb31 * N_Q;
    params.nb33               = params.nb32;
    params.n_seqs             = 0;
    params.use_paged_attn     = false;
    params.use_paged_layout   = false;
    params.block_size         = 0;
    params.max_blocks_per_seq = 0;
    params.kv_is_fp8          = false;
    params.multi_token_decode = false;

    const std::vector<sample_ref> refs   = build_reference_samples(Q, K, V, mask, sinks, params);
    const int                     warmup = env_int("GGML_SYCL_FATTN_GPTOSS_WARMUP", 3);
    const int                     iters  = env_int("GGML_SYCL_FATTN_GPTOSS_ITERS", 10);

    std::printf("\nshape=%s D=%d n_q=%d n_kv=%d H_q=%d H_kv=%d sinks=1 logit_softcap=%.1f scale=%.8f\n", shape.name, D,
                N_Q, N_KV, H_Q, H_KV, shape.logit_softcap, params.scale);
#    if GGML_SYCL_DNNL
    const ggml_sycl_onednn_fa_layout_plan onednn_plan =
        ggml_sycl_flash_attn_ext_onednn_plan(params, H_Q, H_KV, false, false);
    std::printf("oneDNN SDPA plan: kind=%d reason=%s\n", (int) onednn_plan.kind,
                onednn_reason_name(onednn_plan.reason));
#    else
    std::printf("oneDNN SDPA plan: disabled at build time\n");
#    endif
    std::printf("%-22s %10s %10s %10s %12s %12s %s\n", "path", "mean_us", "min_us", "max_us", "max_abs", "public_abs",
                "notes");

    auto run_and_report = [&](const char * name, output_layout layout, const std::function<void()> & fn,
                              const char * notes) {
        q->memset(out_dev, 0, (size_t) H_Q * N_Q * D * sizeof(float)).wait();
        const bench_timing timing = time_path(*q, warmup, iters, fn);

        std::vector<float> actual((size_t) H_Q * N_Q * D);
        q->memcpy(actual.data(), out_dev, actual.size() * sizeof(float)).wait();
        const float max_abs    = sample_max_abs(actual, refs, layout);
        const float public_abs = sample_max_abs(actual, refs, output_layout::public_qhd);
        std::printf("%-22s %10.1f %10.1f %10.1f %12.5g %12.5g %s\n", name, timing.mean_us, timing.min_us, timing.max_us,
                    max_abs, public_abs, notes);
    };

    run_and_report(
        "xmx-v2-ncols16", output_layout::public_qhd,
        [&]() {
            const bool ok = params.logit_softcap == 0.0f ?
                                launch_fattn_xmx_v2_f16<D, 16, false, sycl::half, float>(ctx, params, q) :
                                launch_fattn_xmx_v2_f16<D, 16, true, sycl::half, float>(ctx, params, q);
            if (!ok) {
                throw std::runtime_error("xmx-v2 rejected shape");
            }
        },
        "default public layout");

    run_and_report(
        "xmx-v1-ncols8", output_layout::public_qhd,
        [&]() {
            const size_t local_mem = (size_t) q->get_device().get_info<sycl::info::device::local_mem_size>();
            const int    batch_kv  = ggml_sycl_fattn_xmx_v1_select_batch_kv(D, 8, local_mem);
            if (batch_kv == XMX_BATCH_KV_LARGE) {
                if (params.logit_softcap == 0.0f) {
                    launch_fattn_xmx_f16<D, 8, false, sycl::half, XMX_BATCH_KV_LARGE>(params, q);
                } else {
                    launch_fattn_xmx_f16<D, 8, true, sycl::half, XMX_BATCH_KV_LARGE>(params, q);
                }
            } else if (params.logit_softcap == 0.0f) {
                launch_fattn_xmx_f16<D, 8, false, sycl::half>(params, q);
            } else {
                launch_fattn_xmx_f16<D, 8, true, sycl::half>(params, q);
            }
        },
        "diagnostic public layout");

    run_and_report(
        "esimd-batched", output_layout::head_major_hqd, [&]() { fattn_esimd_f16_batched<D, sycl::half>(params, *q); },
        "diagnostic head-major output");

    sycl::free(q_dev, *q);
    sycl::free(k_dev, *q);
    sycl::free(v_dev, *q);
    sycl::free(mask_dev, *q);
    sycl::free(sinks_dev, *q);
    sycl::free(out_dev, *q);
}

int main() {
    try {
        const bench_shape shapes[] = {
            { "gptoss-sinks-only",      0.0f  },
            { "gptoss-sinks-softcap50", 50.0f },
        };
        for (const bench_shape & shape : shapes) {
            run_shape(shape);
        }
    } catch (const sycl::exception & ex) {
        std::fprintf(stderr, "SYCL exception: %s\n", ex.what());
        return 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        return 1;
    }
    return 0;
}

#endif
