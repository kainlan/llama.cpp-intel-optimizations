// WOQ-MXFP4 spike: can this device's oneDNN consume f4_e2m1 weights with
// e8m0 group-32 scales directly? (perf-recovery epic, track C gate)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>
using dt = dnnl::memory::data_type;

// Deliberately HALVED vs. ggml's own kvalues_mxfp4 (ggml-common.h:1126:
// { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }), paired below
// with an un-halved e8m0_to_f32. The product is algebraically identical to
// ggml's doubled-int table paired with its own e8m0_to_fp32_half (which
// halves the decoded scale to compensate) -- so a literal element-by-element
// comparison against ggml's table would show a difference despite both
// factorizations producing the same dequantized weight.
static const float kvalues_mxfp4[16] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6, -0, -.5f, -1, -1.5f, -2, -3, -4, -6 };

// Mirrors ggml_e8m0_to_fp32 (ggml-impl.h:439-473), including its e==0
// special case: the exponent field alone can't represent 2^-127 (bits would
// be 0, i.e. positive zero), so e==0 is defined as the denormal bit pattern
// 0x00400000 (0.5 * 2^-126 = 2^-127) rather than falling through to the
// general `e << 23` formula. Current test data never selects e==0 (scales
// start at 124), so this doesn't move today's verdict, but C2's repack test
// inherits this reference and could exercise it.
static float e8m0_to_f32(uint8_t e) {
    union {
        uint32_t u;
        float    f;
    } v;

    v.u = (e == 0) ? 0x00400000u : ((uint32_t) e << 23);
    return v.f;
}

int main() {
    constexpr int        M = 8, N = 64, K = 128, G = 32;  // small but group-aligned
    std::vector<uint8_t> nibbles(K * N / 2);              // sequential packing candidate
    std::vector<uint8_t> scales(K / G * N);
    for (size_t i = 0; i < nibbles.size(); ++i) {
        nibbles[i] = (uint8_t) ((i * 7 + 3) & 0xff);
    }
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = (uint8_t) (124 + (i % 8));  // ~0.06..8.0
    }
    std::vector<float> src(M * K);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = 0.01f * (float) ((int) (i % 17) - 8);
    }

    // CPU reference under SEQUENTIAL nibble order: element (k,n), value index
    // = nibble at position k*N+n; scale = scales[(k/G)*N + n].
    std::vector<float> ref(M * N, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                size_t  el  = (size_t) k * N + n;
                uint8_t nib = (nibbles[el / 2] >> ((el % 2) * 4)) & 0xf;
                float   w   = kvalues_mxfp4[nib] * e8m0_to_f32(scales[(k / G) * N + n]);
                ref[m * N + n] += src[m * K + k] * w;
            }
        }
    }
    try {
        sycl::queue          q{ sycl::gpu_selector_v };
        auto                 eng    = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        auto                 stream = dnnl::sycl_interop::make_stream(eng, q);
        dnnl::memory::desc   a_md({ M, K }, dt::f16, { K, 1 });
        dnnl::memory::desc   b_md({ K, N }, dt::f4_e2m1, { N, 1 });
        dnnl::memory::desc   c_md({ M, N }, dt::f32, { N, 1 });
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { G, 1 }, dt::e8m0);
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
        dnnl::matmul::primitive_desc pd;
        try {
            pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        } catch (const dnnl::error & e) {
            std::printf("VERDICT: unsupported (primitive_desc: %s)\n", e.what());
            return 42;
        }
        // f16 src upload
        std::vector<uint16_t> src_h(M * K);
        for (int i = 0; i < M * K; ++i) {
            src_h[i] = sycl::bit_cast<uint16_t>(sycl::half(src[i]));
        }
        auto dev_alloc = [&](size_t bytes, const void * host) {
            void * p = sycl::malloc_device(bytes, q);
            q.memcpy(p, host, bytes).wait();
            return p;
        };
        void *       a_dev = dev_alloc(src_h.size() * 2, src_h.data());
        void *       b_dev = dev_alloc(nibbles.size(), nibbles.data());
        void *       s_dev = dev_alloc(scales.size(), scales.data());
        void *       c_dev = sycl::malloc_device((size_t) M * N * 4, q);
        dnnl::memory a_m(a_md, eng, a_dev), b_m(b_md, eng, b_dev), c_m(c_md, eng, c_dev);
        dnnl::memory s_m(
            {
                { K / G, N },
                dt::e8m0, { N,     1 }
        },
            eng, s_dev);
        dnnl::matmul                          prim(pd);
        std::unordered_map<int, dnnl::memory> args{
            { DNNL_ARG_SRC,                            a_m },
            { DNNL_ARG_WEIGHTS,                        b_m },
            { DNNL_ARG_DST,                            c_m },
            { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, s_m }
        };
        prim.execute(stream, args);
        stream.wait();
        std::vector<float> out(M * N);
        q.memcpy(out.data(), c_dev, out.size() * 4).wait();
        double max_rel = 0;
        for (int i = 0; i < M * N; ++i) {
            max_rel = std::max(max_rel, (double) (std::fabs(out[i] - ref[i]) / std::max(1.f, std::fabs(ref[i]))));
        }
        std::printf("numerics: max_rel=%.4g (sequential-order reference)\n", max_rel);
        if (max_rel > 2e-2) {
            std::printf("VERDICT: wrong-numerics (try interleaved order next)\n");
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 50; ++r) {
            prim.execute(stream, args);
        }
        stream.wait();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        std::printf("VERDICT: supported order=sequential exec=%.1f us/iter\n", us / 50.0);
        sycl::free(a_dev, q);
        sycl::free(b_dev, q);
        sycl::free(s_dev, q);
        sycl::free(c_dev, q);
        return 0;
    } catch (const dnnl::error & e) {
        // A dnnl::error thrown anywhere past primitive_desc creation (memory
        // construction, prim.execute, ...) is a driver/runtime-side failure,
        // not the "unsupported" verdict the inner try/catch above reports.
        // Without this handler it propagates out of main uncaught -> std::
        // terminate, which on future hardware would look like a harness bug
        // rather than a clean, distinguishable verdict.
        std::printf("VERDICT: dnnl error outside primitive creation (%s)\n", e.what());
        return 43;
    } catch (const sycl::exception & e) {
        std::printf("SKIP: no SYCL GPU (%s)\n", e.what());
        return 77;
    }
}
