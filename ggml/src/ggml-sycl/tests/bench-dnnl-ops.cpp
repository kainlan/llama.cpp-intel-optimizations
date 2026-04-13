//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Standalone microbenchmark: SYCL custom kernels vs oneDNN primitives
// for softmax, binary MUL (row-broadcast), and fused RMSNorm+MUL.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   ninja -C build bench-dnnl-ops
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bench-dnnl-ops
//

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <thread>

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr int WARMUP_ITERS   = 5;
static constexpr int MEASURE_ITERS  = 50;
static constexpr int WARP_SIZE      = 32;  // Intel sub-group size

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
struct BenchResult {
    double mean_us;
    double min_us;
    double max_us;
};

static BenchResult bench(sycl::queue & q, int warmup, int iters,
                         std::function<void()> fn) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        fn();
        q.wait();
    }

    std::vector<double> times(iters);
    for (int i = 0; i < iters; i++) {
        q.wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double sum = 0, mn = 1e18, mx = 0;
    for (auto t : times) {
        sum += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }
    return {sum / iters, mn, mx};
}

// ---------------------------------------------------------------------------
// Fill buffer with random floats
// ---------------------------------------------------------------------------
static void fill_random(sycl::queue & q, float * dev, size_t n) {
    std::vector<float> host(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) host[i] = dist(rng);
    q.memcpy(dev, host.data(), n * sizeof(float)).wait();
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
static void print_header(const char * section) {
    printf("\n");
    printf("=========================================================="
           "============================\n");
    printf("  %s\n", section);
    printf("=========================================================="
           "============================\n");
    printf("%-28s %12s %12s %12s %10s\n",
           "Config", "SYCL(us)", "oneDNN(us)", "Min/Max(us)", "Speedup");
    printf("----------------------------------------------------------"
           "----------------------------\n");
}

static void print_row(const char * label,
                       BenchResult sycl_r, BenchResult dnnl_r) {
    double speedup = sycl_r.mean_us / dnnl_r.mean_us;
    printf("%-28s %12.1f %12.1f  %5.0f/%5.0f   %7.2fx %s\n",
           label,
           sycl_r.mean_us, dnnl_r.mean_us,
           dnnl_r.min_us, dnnl_r.max_us,
           speedup,
           speedup > 1.0 ? "(oneDNN wins)" : "(SYCL wins)");
}

static void print_row_3way(const char * label,
                           BenchResult sycl_r, BenchResult dnnl_r,
                           BenchResult fused_r) {
    double speedup_vs_sycl = sycl_r.mean_us / fused_r.mean_us;
    printf("%-28s %12.1f %12.1f %12.1f  %7.2fx\n",
           label,
           sycl_r.mean_us, dnnl_r.mean_us, fused_r.mean_us,
           speedup_vs_sycl);
}

// =========================================================================
// 1. SOFTMAX BENCHMARKS
// =========================================================================

// SYCL 3-pass softmax: max reduction, exp-sum reduction, normalize
// One work-group per row.
static void sycl_softmax(sycl::queue & q, const float * src, float * dst,
                         int64_t batch, int64_t ncols) {
    int wg_size = WARP_SIZE;
    while (wg_size < ncols && wg_size < 1024) wg_size *= 2;
    if (wg_size > 1024) wg_size = 1024;

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(ncols + WARP_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(batch * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);
                const int bs  = item.get_local_range(0);

                const float * row_src = src + int64_t(row) * ncols;
                float       * row_dst = dst + int64_t(row) * ncols;
                float       * vals    = slm.get_multi_ptr<sycl::access::decorated::no>().get();

                // Pass 1: find max
                float max_val = -INFINITY;
                for (int col = tid; col < ncols; col += bs) {
                    float v = row_src[col];
                    vals[col] = v;
                    max_val = sycl::max(max_val, v);
                }
                max_val = sycl::reduce_over_group(item.get_group(), max_val,
                                                   sycl::maximum<float>());

                // Pass 2: exp and sum
                float sum = 0.0f;
                for (int col = tid; col < ncols; col += bs) {
                    float v = sycl::exp(vals[col] - max_val);
                    vals[col] = v;
                    sum += v;
                }
                sum = sycl::reduce_over_group(item.get_group(), sum,
                                               sycl::plus<float>());
                float inv_sum = 1.0f / sum;

                // Pass 3: normalize
                for (int col = tid; col < ncols; col += bs) {
                    row_dst[col] = vals[col] * inv_sum;
                }
            });
    });
}

#if GGML_SYCL_DNNL
static void bench_dnnl_softmax_op(dnnl::engine & eng, dnnl::stream & strm,
                                  const float * src, float * dst,
                                  int64_t batch, int64_t features) {
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    dnnl::memory::dims dims = {batch, features};
    auto md = dnnl::memory::desc(dims, dt::f32, tag::ab);

    auto softmax_pd = dnnl::softmax_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_accurate, md, md, /*axis=*/1);

    auto src_mem = dnnl::memory(md, eng, const_cast<float*>(src));
    auto dst_mem = dnnl::memory(md, eng, dst);

    auto prim = dnnl::softmax_forward(softmax_pd);
    prim.execute(strm, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
}
#endif

static void bench_softmax(sycl::queue & q) {
    print_header("SOFTMAX  (src -> softmax(src) along last dim)");

#if GGML_SYCL_DNNL
    auto dev = q.get_device();
    auto ctx = q.get_context();
    auto eng  = dnnl::sycl_interop::make_engine(dev, ctx);
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
#endif

    struct Config { int64_t batch; int64_t ncols; };
    std::vector<Config> configs = {
        {32,    128},
        {32,    512},
        {32,   2048},
        {32,   4096},
        {96,    128},
        {96,    512},
        {96,   4096},
        {512,   128},
        {512,  2048},
        {512,  4096},
        {16384, 128},
        {16384, 4096},
    };

    for (auto & c : configs) {
        size_t n = c.batch * c.ncols;
        float * src = sycl::malloc_device<float>(n, q);
        float * dst = sycl::malloc_device<float>(n, q);
        fill_random(q, src, n);

        char label[64];
        snprintf(label, sizeof(label), "B=%lld  N=%lld",
                 (long long)c.batch, (long long)c.ncols);

        auto sycl_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            sycl_softmax(q, src, dst, c.batch, c.ncols);
        });

#if GGML_SYCL_DNNL
        auto dnnl_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            bench_dnnl_softmax_op(eng, strm, src, dst, c.batch, c.ncols);
        });
        print_row(label, sycl_r, dnnl_r);
#else
        printf("%-28s %12.1f %12s\n", label, sycl_r.mean_us, "N/A (no DNNL)");
#endif

        sycl::free(src, q);
        sycl::free(dst, q);
    }
}

// =========================================================================
// 2. BINARY MUL (row broadcast) BENCHMARKS
// =========================================================================

// Specialized SYCL kernel: dst[i] = src0[i] * src1[i % row_size]
static void sycl_binary_mul_broadcast(sycl::queue & q,
                                      const float * src0,
                                      const float * src1,
                                      float * dst,
                                      int64_t total, int64_t row_size) {
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
        dst[i] = src0[i] * src1[i % row_size];
    });
}

#if GGML_SYCL_DNNL
static void dnnl_binary_mul_broadcast(dnnl::engine & eng, dnnl::stream & strm,
                                      const float * src0,
                                      const float * src1,
                                      float * dst,
                                      int64_t rows, int64_t cols) {
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    // src0 is [rows, cols], src1 is [1, cols] with broadcast
    dnnl::memory::dims src0_dims = {rows, cols};
    dnnl::memory::dims src1_dims = {1, cols};
    dnnl::memory::dims dst_dims  = {rows, cols};

    auto src0_md = dnnl::memory::desc(src0_dims, dt::f32, tag::ab);
    auto src1_md = dnnl::memory::desc(src1_dims, dt::f32, tag::ab);
    auto dst_md  = dnnl::memory::desc(dst_dims,  dt::f32, tag::ab);

    auto binary_pd = dnnl::binary::primitive_desc(
        eng, dnnl::algorithm::binary_mul, src0_md, src1_md, dst_md);

    auto src0_mem = dnnl::memory(src0_md, eng, const_cast<float*>(src0));
    auto src1_mem = dnnl::memory(src1_md, eng, const_cast<float*>(src1));
    auto dst_mem  = dnnl::memory(dst_md,  eng, dst);

    auto prim = dnnl::binary(binary_pd);
    prim.execute(strm, {
        {DNNL_ARG_SRC_0, src0_mem},
        {DNNL_ARG_SRC_1, src1_mem},
        {DNNL_ARG_DST,   dst_mem}
    });
}
#endif

static void bench_binary_mul(sycl::queue & q) {
    print_header("BINARY MUL (row broadcast)  dst[i] = src0[i] * src1[i % row]");

#if GGML_SYCL_DNNL
    auto dev = q.get_device();
    auto ctx = q.get_context();
    auto eng  = dnnl::sycl_interop::make_engine(dev, ctx);
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
#endif

    struct Config { int64_t rows; int64_t cols; };
    std::vector<Config> configs = {
        {1,     4096},
        {128,   4096},
        {512,   4096},
        {1,     8192},
        {128,   8192},
        {512,   8192},
    };

    for (auto & c : configs) {
        int64_t total = c.rows * c.cols;
        float * src0 = sycl::malloc_device<float>(total, q);
        float * src1 = sycl::malloc_device<float>(c.cols, q);
        float * dst  = sycl::malloc_device<float>(total, q);
        fill_random(q, src0, total);
        fill_random(q, src1, c.cols);

        char label[64];
        snprintf(label, sizeof(label), "R=%lld  C=%lld",
                 (long long)c.rows, (long long)c.cols);

        auto sycl_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            sycl_binary_mul_broadcast(q, src0, src1, dst, total, c.cols);
        });

#if GGML_SYCL_DNNL
        auto dnnl_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            dnnl_binary_mul_broadcast(eng, strm, src0, src1, dst, c.rows, c.cols);
        });
        print_row(label, sycl_r, dnnl_r);
#else
        printf("%-28s %12.1f %12s\n", label, sycl_r.mean_us, "N/A (no DNNL)");
#endif

        sycl::free(src0, q);
        sycl::free(src1, q);
        sycl::free(dst, q);
    }
}

// =========================================================================
// 3. FUSED RMSNorm + MUL BENCHMARKS
// =========================================================================

// Two-op baseline: RMS norm then element-wise MUL
static void sycl_rms_norm_then_mul(sycl::queue & q,
                                   const float * x,
                                   const float * weight,
                                   float * tmp,   // intermediate
                                   float * dst,
                                   int64_t batch, int64_t hidden,
                                   float eps) {
    int wg_size = WARP_SIZE;
    while (wg_size < hidden && wg_size < 1024) wg_size *= 2;
    if (wg_size > 1024) wg_size = 1024;

    // Op 1: RMS norm  tmp[row] = x[row] / rms(x[row])
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(batch * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);
                const int bs  = item.get_local_range(0);
                const float * row_x = x + int64_t(row) * hidden;
                float * row_tmp     = tmp + int64_t(row) * hidden;

                float sum_sq = 0.0f;
                for (int col = tid; col < hidden; col += bs) {
                    float v = row_x[col];
                    sum_sq += v * v;
                }
                sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq,
                                                  sycl::plus<float>());
                float scale = sycl::rsqrt(sum_sq / float(hidden) + eps);

                for (int col = tid; col < hidden; col += bs) {
                    row_tmp[col] = row_x[col] * scale;
                }
            });
    });

    // Op 2: element-wise MUL  dst[i] = tmp[i] * weight[i % hidden]
    int64_t total = batch * hidden;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
        dst[i] = tmp[i] * weight[i % hidden];
    });
}

// Fused kernel: single pass RMSNorm + MUL
static void sycl_fused_rms_norm_mul(sycl::queue & q,
                                    const float * x,
                                    const float * weight,
                                    float * dst,
                                    int64_t batch, int64_t hidden,
                                    float eps) {
    int wg_size = WARP_SIZE;
    while (wg_size < hidden && wg_size < 1024) wg_size *= 2;
    if (wg_size > 1024) wg_size = 1024;

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(batch * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);
                const int bs  = item.get_local_range(0);

                const float * row_x = x + int64_t(row) * hidden;
                float * row_dst     = dst + int64_t(row) * hidden;

                // Pass 1: sum of squares
                float sum_sq = 0.0f;
                for (int col = tid; col < hidden; col += bs) {
                    float v = row_x[col];
                    sum_sq += v * v;
                }
                sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq,
                                                  sycl::plus<float>());
                float scale = sycl::rsqrt(sum_sq / float(hidden) + eps);

                // Pass 2: normalize and multiply by weight
                for (int col = tid; col < hidden; col += bs) {
                    row_dst[col] = row_x[col] * scale * weight[col];
                }
            });
    });
}

#if GGML_SYCL_DNNL
// oneDNN layer_normalization with use_scale (closest to RMSNorm + MUL)
// Note: oneDNN layer_norm computes (x - mean) / sqrt(var + eps) * gamma + beta
// which is LayerNorm, not RMSNorm. We benchmark it to see the overhead.
static void dnnl_layer_norm_mul(dnnl::engine & eng, dnnl::stream & strm,
                                const float * src, const float * weight,
                                float * dst,
                                int64_t batch, int64_t hidden, float eps) {
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using flags = dnnl::normalization_flags;

    dnnl::memory::dims dims = {batch, hidden};
    auto src_md = dnnl::memory::desc(dims, dt::f32, tag::ab);
    auto dst_md = dnnl::memory::desc(dims, dt::f32, tag::ab);

    // use_scale = gamma (our "weight"), no shift (beta)
    auto ln_pd = dnnl::layer_normalization_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference,
        src_md, dst_md, eps, flags::use_scale);

    auto src_mem    = dnnl::memory(src_md, eng, const_cast<float*>(src));
    auto dst_mem    = dnnl::memory(dst_md, eng, dst);

    // Scale (gamma) is 1D with shape [hidden]
    auto scale_md  = dnnl::memory::desc({hidden}, dt::f32, tag::a);
    auto scale_mem = dnnl::memory(scale_md, eng, const_cast<float*>(weight));

    auto prim = dnnl::layer_normalization_forward(ln_pd);
    prim.execute(strm, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_SCALE, scale_mem}
    });
}
#endif

static void bench_rms_norm_mul(sycl::queue & q) {
    printf("\n");
    printf("=========================================================="
           "============================================\n");
    printf("  RMSNorm + MUL  (fused vs 2-op vs oneDNN LayerNorm)\n");
    printf("=========================================================="
           "============================================\n");
    printf("%-28s %12s %12s %12s %10s\n",
           "Config", "2-op(us)", "oneDNN(us)", "Fused(us)", "Fused/2-op");
    printf("----------------------------------------------------------"
           "--------------------------------------------\n");

#if GGML_SYCL_DNNL
    auto dev = q.get_device();
    auto ctx = q.get_context();
    auto eng  = dnnl::sycl_interop::make_engine(dev, ctx);
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
#endif

    float eps = 1e-5f;

    struct Config { int64_t batch; int64_t hidden; };
    std::vector<Config> configs = {
        {1,    4096},
        {1,    5120},
        {128,  4096},
        {128,  5120},
        {512,  4096},
        {512,  5120},
    };

    for (auto & c : configs) {
        size_t n = c.batch * c.hidden;
        float * x      = sycl::malloc_device<float>(n, q);
        float * weight = sycl::malloc_device<float>(c.hidden, q);
        float * tmp    = sycl::malloc_device<float>(n, q);
        float * dst    = sycl::malloc_device<float>(n, q);
        fill_random(q, x, n);
        fill_random(q, weight, c.hidden);

        char label[64];
        snprintf(label, sizeof(label), "B=%lld  H=%lld",
                 (long long)c.batch, (long long)c.hidden);

        // Two-op baseline
        auto two_op_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            sycl_rms_norm_then_mul(q, x, weight, tmp, dst, c.batch, c.hidden, eps);
        });

        // Fused kernel
        auto fused_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            sycl_fused_rms_norm_mul(q, x, weight, dst, c.batch, c.hidden, eps);
        });

#if GGML_SYCL_DNNL
        // oneDNN LayerNorm (closest match)
        auto dnnl_r = bench(q, WARMUP_ITERS, MEASURE_ITERS, [&]() {
            dnnl_layer_norm_mul(eng, strm, x, weight, dst, c.batch, c.hidden, eps);
        });
        print_row_3way(label, two_op_r, dnnl_r, fused_r);
#else
        printf("%-28s %12.1f %12s %12.1f  %7.2fx\n",
               label, two_op_r.mean_us, "N/A",
               fused_r.mean_us, two_op_r.mean_us / fused_r.mean_us);
#endif

        sycl::free(x, q);
        sycl::free(weight, q);
        sycl::free(tmp, q);
        sycl::free(dst, q);
    }
}

// =========================================================================
// 4. CORRECTNESS VERIFICATION
// =========================================================================
static bool verify_softmax(sycl::queue & q) {
    printf("\n[Verify] Softmax correctness... ");
    const int B = 4, N = 128;
    size_t n = B * N;
    float * src = sycl::malloc_device<float>(n, q);
    float * dst = sycl::malloc_device<float>(n, q);
    fill_random(q, src, n);

    sycl_softmax(q, src, dst, B, N);
    q.wait();

    std::vector<float> h_dst(n);
    q.memcpy(h_dst.data(), dst, n * sizeof(float)).wait();

    bool ok = true;
    for (int b = 0; b < B; b++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            float v = h_dst[b * N + j];
            if (v < 0 || v > 1) { ok = false; break; }
            sum += v;
        }
        if (std::abs(sum - 1.0f) > 1e-4f) ok = false;
    }
    printf("%s (row sums in [0.9999, 1.0001])\n", ok ? "PASS" : "FAIL");

    sycl::free(src, q);
    sycl::free(dst, q);
    return ok;
}

static bool verify_rms_norm(sycl::queue & q) {
    printf("[Verify] Fused RMSNorm+MUL correctness... ");
    const int B = 2, H = 64;
    size_t n = B * H;
    float * x      = sycl::malloc_device<float>(n, q);
    float * weight = sycl::malloc_device<float>(H, q);
    float * dst    = sycl::malloc_device<float>(n, q);

    // Use known input: all 1.0 -> RMS = 1.0, scale = 1/sqrt(1+eps) ~ 1.0
    std::vector<float> ones(n, 1.0f);
    std::vector<float> twos(H, 2.0f);
    q.memcpy(x, ones.data(), n * sizeof(float)).wait();
    q.memcpy(weight, twos.data(), H * sizeof(float)).wait();

    sycl_fused_rms_norm_mul(q, x, weight, dst, B, H, 1e-5f);
    q.wait();

    std::vector<float> h_dst(n);
    q.memcpy(h_dst.data(), dst, n * sizeof(float)).wait();

    // Expected: x=1, rms(x)=1, scale=1/sqrt(1+1e-5) ~ 0.999995, result ~ 2*0.999995
    float expected = 2.0f * sycl::rsqrt(1.0f + 1e-5f);
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        if (std::abs(h_dst[i] - expected) > 1e-3f) {
            ok = false;
            printf("  MISMATCH at %zu: got %.6f expected %.6f\n",
                   i, h_dst[i], expected);
            break;
        }
    }
    printf("%s (expected ~%.4f)\n", ok ? "PASS" : "FAIL", expected);

    sycl::free(x, q);
    sycl::free(weight, q);
    sycl::free(dst, q);
    return ok;
}

// =========================================================================
// MAIN
// =========================================================================
int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v,
                      sycl::property_list{sycl::property::queue::in_order()});

        auto dev = q.get_device();
        printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        printf("Max WG size: %zu\n",
               dev.get_info<sycl::info::device::max_work_group_size>());
        printf("Max compute units: %u\n",
               dev.get_info<sycl::info::device::max_compute_units>());
        printf("Global mem: %.1f MB\n",
               dev.get_info<sycl::info::device::global_mem_size>() / 1048576.0);
        printf("\nWarmup iterations: %d, Measured iterations: %d\n",
               WARMUP_ITERS, MEASURE_ITERS);
#if GGML_SYCL_DNNL
        printf("oneDNN: ENABLED\n");
#else
        printf("oneDNN: DISABLED (GGML_SYCL_DNNL=0)\n");
#endif

        // Correctness checks first
        bool ok1 = verify_softmax(q);
        bool ok2 = verify_rms_norm(q);
        if (!ok1 || !ok2) {
            printf("\nCorrectness verification FAILED - aborting benchmarks.\n");
            return 1;
        }

        // Benchmark groups with cooling pauses
        bench_softmax(q);

        printf("\n[Cooling 5s between benchmark groups...]\n");
        std::this_thread::sleep_for(std::chrono::seconds(5));

        bench_binary_mul(q);

        printf("\n[Cooling 5s between benchmark groups...]\n");
        std::this_thread::sleep_for(std::chrono::seconds(5));

        bench_rms_norm_mul(q);

        printf("\n[Done]\n");
        return 0;

    } catch (sycl::exception & e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    } catch (std::exception & e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
