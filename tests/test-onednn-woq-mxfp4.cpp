// WOQ-MXFP4 spike: can this device's oneDNN consume f4_e2m1 weights with
// e8m0 group-32 scales directly? (perf-recovery epic, track C gate)
//
// Extended for llama.cpp-4nlr (option-A probe, spike llama.cpp-vyjl): a
// second, GPT-OSS-shaped case comparing the row-major WOQ layout the repack
// manufactures today against the STORED SOA plane read flat -- weights
// {K,N} strides {1,K}, scales {groups,N} strides {1,groups}. Correctness
// alone is NOT the verdict: oneDNN may accept a transposed sub-byte desc
// and quietly select a reference implementation, so each arm reports
// per-iteration time and the ratio is the load-bearing output (a
// green-but-slow strided arm FAILS option A).
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

// Original C1 capability case, unchanged: sequential-order row-major descs
// at a small group-aligned shape. rc: 0 ok, 1 wrong-numerics, 42 unsupported.
static int run_original_case(sycl::queue & q, dnnl::engine & eng, dnnl::stream & stream) {
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
}

// llama.cpp-4nlr option-A probe. One arm of the GPT-OSS-shaped A/B: the same
// nibble/scale/activation bytes bound under either the row-major descs the
// repack manufactures (b {K,N} strides {N,1}, s {G_cnt,N} strides {N,1}) or
// the stored-SOA-plane-read-flat descs (b strides {1,K}, s strides {1,G_cnt}).
// The CPU reference decodes each element THROUGH THE ARM'S OWN indexing, so a
// pass proves oneDNN honored that desc, independent of what the bytes "mean".
struct ab_arm_result {
    bool               supported;
    double             max_rel;
    double             us_per_iter;
    std::vector<float> out;  // GPU output, kept for cross-arm diagnosis
};

// b_mode: 0 = row-major strides {N,1}; 1 = stored-flat strides {1,K};
// 2 = format_tag::ba (formally transposed — same layout as mode 1, declared
// via tag instead of raw strides, in case the implementation keys on tags).
static ab_arm_result run_ab_arm(sycl::queue & q, dnnl::engine & eng, dnnl::stream & stream, const char * name,
                                int b_mode, int M, int N, int K, int G, const std::vector<uint8_t> & nibbles,
                                const std::vector<uint8_t> & scales, const std::vector<float> & src) {
    ab_arm_result res    = { false, 0.0, 0.0, {} };
    const bool    strided = b_mode != 0;
    const int     groups  = K / G;

    // Per-arm CPU reference: decode w(k,n) via this arm's memory indexing.
    // Row-major: nibble offset k*N+n, scale offset (k/G)*N+n.
    // Stored-flat (strides {1,K} / {1,groups}): nibble offset n*K+k, scale
    // offset n*groups + k/G. Sub-byte packing: two consecutive offsets share
    // a byte, even offset in the LOW nibble (gemm.hpp:947-950 convention).
    std::vector<float> wdec((size_t) K * N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const size_t  off = strided ? (size_t) n * K + k : (size_t) k * N + n;
            const uint8_t nib = (nibbles[off / 2] >> ((off % 2) * 4)) & 0xf;
            const size_t  soff =
                strided ? (size_t) n * groups + (size_t) (k / G) : (size_t) (k / G) * N + n;
            wdec[(size_t) k * N + n] = kvalues_mxfp4[nib] * e8m0_to_f32(scales[soff]);
        }
    }
    std::vector<float> ref((size_t) M * N, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float a = src[(size_t) m * K + k];
            if (a == 0.f) {
                continue;
            }
            const float * wrow = wdec.data() + (size_t) k * N;
            float *       out  = ref.data() + (size_t) m * N;
            for (int n = 0; n < N; ++n) {
                out[n] += a * wrow[n];
            }
        }
    }

    dnnl::memory::desc a_md({ M, K }, dt::f16, { K, 1 });
    dnnl::memory::desc b_md = (b_mode == 2)
                                  ? dnnl::memory::desc({ K, N }, dt::f4_e2m1, dnnl::memory::format_tag::ba)
                                  : dnnl::memory::desc({ K, N }, dt::f4_e2m1,
                                                       strided ? dnnl::memory::dims{ 1, K } :
                                                                 dnnl::memory::dims{ N, 1 });
    dnnl::memory::desc c_md({ M, N }, dt::f32, { N, 1 });
    dnnl::memory::desc s_md({ groups, N }, dt::e8m0,
                            strided ? dnnl::memory::dims{ 1, groups } : dnnl::memory::dims{ N, 1 });
    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { G, 1 }, dt::e8m0);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    dnnl::matmul::primitive_desc pd;
    try {
        pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    } catch (const dnnl::error & e) {
        std::printf("[AB] arm=%s supported=0 (primitive_desc: %s)\n", name, e.what());
        return res;
    }
    res.supported = true;

    std::vector<uint16_t> src_h((size_t) M * K);
    for (size_t i = 0; i < src_h.size(); ++i) {
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
    dnnl::memory a_m(a_md, eng, a_dev), b_m(b_md, eng, b_dev), c_m(c_md, eng, c_dev), s_m(s_md, eng, s_dev);
    dnnl::matmul                          prim(pd);
    std::unordered_map<int, dnnl::memory> args{
        { DNNL_ARG_SRC,                            a_m },
        { DNNL_ARG_WEIGHTS,                        b_m },
        { DNNL_ARG_DST,                            c_m },
        { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, s_m }
    };
    prim.execute(stream, args);
    stream.wait();
    res.out.resize((size_t) M * N);
    q.memcpy(res.out.data(), c_dev, res.out.size() * 4).wait();
    for (size_t i = 0; i < res.out.size(); ++i) {
        res.max_rel = std::max(
            res.max_rel, (double) (std::fabs(res.out[i] - ref[i]) / std::max(1.f, std::fabs(ref[i]))));
    }

    constexpr int WARMUP = 3, ITERS = 30;
    for (int r = 0; r < WARMUP; ++r) {
        prim.execute(stream, args);
    }
    stream.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ITERS; ++r) {
        prim.execute(stream, args);
    }
    stream.wait();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    res.us_per_iter = us / (double) ITERS;

    const double weights_gbps = (nibbles.size() + scales.size()) / res.us_per_iter / 1e3;
    std::printf("[AB] arm=%s supported=1 max_rel=%.4g us/iter=%.1f weights_GBps=%.1f\n", name, res.max_rel,
                res.us_per_iter, weights_gbps);
    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(s_dev, q);
    sycl::free(c_dev, q);
    return res;
}

// Role-swap arm (llama.cpp-4nlr round 3): sidestep transposition entirely.
// The stored SOA nibble plane read as {N,K} IS dense row-major f4 — bind it
// as the quantized SRC operand with grouped scales on DNNL_ARG_SRC (the
// stored e8m0 plane is byte-for-byte the {N,K/G} dense scale tensor), and
// carry the intra-block j/j+16 interleave as the activation permutation
// pi(kappa) = kappa/2 + 16*(kappa%2) applied to the f16 WEIGHTS operand
// (fork-owned gather kernel at the real call site; host-side here). Output
// lands transposed ({N,M}); the executor's scatter-out kernel would consume
// that. D[n,m] = sum_kappa S(n,kappa)*A[m,pi(kappa)] = C[m,n] exactly.
static ab_arm_result run_roleswap_arm(sycl::queue & q, dnnl::engine & eng, dnnl::stream & stream, int M, int N,
                                      int K, int G, const std::vector<uint8_t> & nibbles,
                                      const std::vector<uint8_t> & scales, const std::vector<float> & src) {
    ab_arm_result res    = { false, 0.0, 0.0, {} };
    const int     groups = K / G;
    auto          pi     = [](int kappa) {
        return (kappa / 2) % 16 + 16 * (kappa % 2) + (kappa / 32) * 32;  // within-block j/j+16 de-interleave
    };

    // CPU reference: decode stored plane flat as {N,K} dense, scale from the
    // stored {N,groups} plane, activations permuted along kappa.
    std::vector<float> ref((size_t) N * M, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int kappa = 0; kappa < K; ++kappa) {
            const size_t  off = (size_t) n * K + kappa;
            const uint8_t nib = (nibbles[off / 2] >> ((off % 2) * 4)) & 0xf;
            const float   w   = kvalues_mxfp4[nib] * e8m0_to_f32(scales[(size_t) n * groups + kappa / G]);
            if (w == 0.f) {
                continue;
            }
            const int ak = pi(kappa);
            for (int m = 0; m < M; ++m) {
                ref[(size_t) n * M + m] += w * src[(size_t) m * K + ak];
            }
        }
    }

    dnnl::memory::desc   s_src_md({ N, K }, dt::f4_e2m1, { K, 1 });   // dense row-major: the stored plane
    dnnl::memory::desc   w_md({ K, M }, dt::f16, { M, 1 });           // pi-permuted activations, materialized
    dnnl::memory::desc   c_md({ N, M }, dt::f32, { M, 1 });
    dnnl::memory::desc   sc_md({ N, groups }, dt::e8m0, { groups, 1 });  // the stored e8m0 plane, as-is
    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), { 1, G }, dt::e8m0);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    dnnl::matmul::primitive_desc pd;
    try {
        pd = dnnl::matmul::primitive_desc(eng, s_src_md, w_md, c_md, attr);
    } catch (const dnnl::error & e) {
        std::printf("[AB] arm=role-swap supported=0 (primitive_desc: %s)\n", e.what());
        return res;
    }
    res.supported = true;

    std::vector<uint16_t> act_h((size_t) K * M);  // Wt[kappa,m] = A[m, pi(kappa)]
    for (int kappa = 0; kappa < K; ++kappa) {
        const int ak = pi(kappa);
        for (int m = 0; m < M; ++m) {
            act_h[(size_t) kappa * M + m] = sycl::bit_cast<uint16_t>(sycl::half(src[(size_t) m * K + ak]));
        }
    }
    auto dev_alloc = [&](size_t bytes, const void * host) {
        void * p = sycl::malloc_device(bytes, q);
        q.memcpy(p, host, bytes).wait();
        return p;
    };
    void *       s_dev  = dev_alloc(nibbles.size(), nibbles.data());
    void *       w_dev  = dev_alloc(act_h.size() * 2, act_h.data());
    void *       sc_dev = dev_alloc(scales.size(), scales.data());
    void *       c_dev  = sycl::malloc_device((size_t) N * M * 4, q);
    dnnl::memory s_m(s_src_md, eng, s_dev), w_m(w_md, eng, w_dev), c_m(c_md, eng, c_dev), sc_m(sc_md, eng, sc_dev);
    dnnl::matmul                          prim(pd);
    std::unordered_map<int, dnnl::memory> args{
        { DNNL_ARG_SRC,                        s_m  },
        { DNNL_ARG_WEIGHTS,                    w_m  },
        { DNNL_ARG_DST,                        c_m  },
        { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sc_m }
    };
    prim.execute(stream, args);
    stream.wait();
    res.out.resize((size_t) N * M);
    q.memcpy(res.out.data(), c_dev, res.out.size() * 4).wait();
    for (size_t i = 0; i < res.out.size(); ++i) {
        res.max_rel = std::max(
            res.max_rel, (double) (std::fabs(res.out[i] - ref[i]) / std::max(1.f, std::fabs(ref[i]))));
    }

    constexpr int WARMUP = 3, ITERS = 30;
    for (int r = 0; r < WARMUP; ++r) {
        prim.execute(stream, args);
    }
    stream.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ITERS; ++r) {
        prim.execute(stream, args);
    }
    stream.wait();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    res.us_per_iter = us / (double) ITERS;
    std::printf("[AB] arm=role-swap supported=1 max_rel=%.4g us/iter=%.1f weights_GBps=%.1f\n", res.max_rel,
                res.us_per_iter, (nibbles.size() + scales.size()) / res.us_per_iter / 1e3);
    sycl::free(s_dev, q);
    sycl::free(w_dev, q);
    sycl::free(sc_dev, q);
    sycl::free(c_dev, q);
    return res;
}

// rc: 0 both arms supported + numerically clean (ratio is REPORTED, judged by
// the reader); 44 stored-flat unsupported; 45 stored-flat wrong numerics.
static int run_ab_case(sycl::queue & q, dnnl::engine & eng, dnnl::stream & stream) {
    // GPT-OSS gate/up expert shape: K = N = 2880 (blocks_per_row 90); M = 64
    // is the mean rows/expert at pp512 (512 tokens x 4 active / 32 experts).
    constexpr int        M = 64, N = 2880, K = 2880, G = 32;
    std::vector<uint8_t> nibbles((size_t) K * N / 2);
    std::vector<uint8_t> scales((size_t) K / G * N);
    uint32_t             lcg = 0x12345u;
    for (size_t i = 0; i < nibbles.size(); ++i) {
        lcg        = lcg * 1664525u + 1013904223u;
        nibbles[i] = (uint8_t) (lcg >> 24);
    }
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = (uint8_t) (124 + (i % 8));
    }
    std::vector<float> src((size_t) M * K);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = 0.01f * (float) ((int) (i % 17) - 8);
    }

    std::printf("[AB] shape M=%d N=%d K=%d G=%d (GPT-OSS gate/up expert)\n", M, N, K, G);
    ab_arm_result row = run_ab_arm(q, eng, stream, "row-major", /*b_mode=*/0, M, N, K, G, nibbles, scales, src);
    ab_arm_result st  = run_ab_arm(q, eng, stream, "stored-flat", /*b_mode=*/1, M, N, K, G, nibbles, scales, src);
    ab_arm_result ba  = run_ab_arm(q, eng, stream, "tag-ba", /*b_mode=*/2, M, N, K, G, nibbles, scales, src);
    ab_arm_result rs  = run_roleswap_arm(q, eng, stream, M, N, K, G, nibbles, scales, src);

    // Cross-arm diagnosis: if a transposed arm's GPU output matches the
    // ROW-MAJOR arm's GPU output on the same bytes, the implementation
    // silently ignored the transposition (desc accepted, strides dropped).
    auto vs_row = [&](const ab_arm_result & a) -> double {
        if (!a.supported || !row.supported || a.out.size() != row.out.size()) {
            return -1.0;
        }
        double d = 0;
        for (size_t i = 0; i < a.out.size(); ++i) {
            d = std::max(d, (double) (std::fabs(a.out[i] - row.out[i]) / std::max(1.f, std::fabs(row.out[i]))));
        }
        return d;
    };
    std::printf("[AB] diag stored-vs-rowmajor-gpu=%.4g tag-ba-vs-rowmajor-gpu=%.4g (near-0 => transposition "
                "silently ignored)\n",
                vs_row(st), vs_row(ba));

    // f16 accumulation at K=2880 legitimately reaches ~2.6e-2 on this floored
    // relative metric (sqrt(K)*2^-11); 6e-2 separates that from indexing
    // garbage (measured wrong-layout arms read >100).
    // CAPABILITY TRIPWIRE (llama.cpp-4nlr verdict, oneDNN 3.11.4 / 2026.0,
    // measured on B50 2026-08-22): the EXPECTED map is
    //   row-major        -> supported, correct (f16-accum noise only)
    //   stored-flat / ba -> ACCEPTED BUT WRONG (max_rel ~295; identical to
    //                       each other, matching neither the declared layout
    //                       nor a strides-ignored row-major read -- a third,
    //                       internal canonicalization; silent-garbage hazard:
    //                       NEVER bind non-row-major sub-byte weights)
    //   role-swap        -> primitive_desc REFUSED (no grouped scales on
    //                       DNNL_ARG_SRC)
    // rc 0 = map unchanged. rc 45 = row-major itself broke (real regression).
    // rc 46 = a stored-layout form STARTED WORKING (a oneDNN/driver upgrade
    // changed capability) -- good news, but it reopens the llama.cpp-4nlr /
    // llama.cpp-vyjl option-A decision, so it must be loud, not silent.
    constexpr double TOL   = 6e-2;
    const bool       st_ok = st.supported && st.max_rel <= TOL;
    const bool       ba_ok = ba.supported && ba.max_rel <= TOL;
    const bool       rs_ok = rs.supported && rs.max_rel <= TOL;
    if (!row.supported || row.max_rel > TOL) {
        std::printf("VERDICT-AB: row-major arm regressed (supported=%d max_rel=%.4g)\n", row.supported ? 1 : 0,
                    row.max_rel);
        return 45;
    }
    if (st_ok || ba_ok || rs_ok) {
        const ab_arm_result & good      = rs_ok ? rs : (st_ok ? st : ba);
        const char *          good_name = rs_ok ? "role-swap" : (st_ok ? "strides" : "tag-ba");
        std::printf("VERDICT-AB: CAPABILITY CHANGED -- form=%s now works (max_rel=%.4g, %.1f us vs row %.1f us); "
                    "re-open the option-A decision (llama.cpp-4nlr)\n",
                    good_name, good.max_rel, good.us_per_iter, row.us_per_iter);
        return 46;
    }
    std::printf("VERDICT-AB: stored-layout forms remain unusable as expected on this oneDNN "
                "(stored=%.4g tag-ba=%.4g role-swap=%s)\n",
                st.max_rel, ba.max_rel, rs.supported ? "wrong" : "unsupported");
    return 0;
}

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };
        auto        eng    = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        auto        stream = dnnl::sycl_interop::make_stream(eng, q);

        const int rc0 = run_original_case(q, eng, stream);
        if (rc0 != 0) {
            return rc0;
        }
        return run_ab_case(q, eng, stream);
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
