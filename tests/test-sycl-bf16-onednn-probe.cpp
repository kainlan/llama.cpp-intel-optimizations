// oneDNN BF16-weight matmul probe and microbenchmark against the native BF16 kernel.
//
// QUESTION THIS ANSWERS: dense BF16 weights in prompt MUL_MAT nodes run today on ggml_sycl_bf16::mul_mat_bf16_f32
// (mul-mat-bf16.hpp: F32 activations, no XMX), at about 3 TFLOPS on the B70. Can the oneDNN linked into this build
// run the same product on the card's matrix engines, and how fast? Per card it asks:
//   - does bf16 x bf16 -> f32 create, in the orientation DnnlGemmWrapper::gemm (gemm.hpp) uses and in the natural
//     one, and does a MIXED f32 x bf16 -> f32 create in either orientation (no oneDNN document lists it): a mixed
//     type would need no rounded copy of the activations;
//   - which implementation oneDNN picks (impl_info_str) and how many scratchpad bytes it asks for;
//   - its device time against the native kernel on the same operands;
//   - what the F32 -> BF16 activation conversion costs at the same sizes.
// The two fpmath_bf16 arms also ask whether oneDNN takes F32 activations with a licence to round them itself (the
// same rounding as an explicit conversion, without the copy). act_rounding= tells which arithmetic an arm did.
//
// Orientation. ggml computes dst[n][m] = sum_k w[m][k] * x[n][k] (w: M rows of K BF16 values, x: N columns of K F32
// values, dst: N columns of M F32 values). gemm() passes the WEIGHT as oneDNN's source ([m, k], k contiguous), the
// activations as oneDNN's weights ([k, n], k contiguous) and writes [m, n] with m contiguous ("fork"). The natural
// orientation passes the activations as the source ([n, k]) and the weight as oneDNN's weights ([k, m], k
// contiguous). Both write the same bytes: dst[n * M + m].
//
// Shapes: K x M = 2304 x 3072 and 3072 x 2304 (Inkling's Q/gate projection and its output projection), at
// N = 9, 16, 64, 129, 512 (512 and 129 are the prompt's micro-batches; 9 is the first width the tiled kernel takes).
// Attributes are production's: the deterministic attribute (dnnl-deterministic.hpp), a user scratchpad, and
// fpmath_mode strict (what gemm() sets for GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT) unless the arm says fpmath_bf16.
//
// It is STANDALONE: SYCL, oneDNN and the header-only native kernel, no backend, no unified cache. The raw
// sycl::malloc_device below is a probe fixture, not a backend allocation (the unified-cache rule governs backend
// code; this binary never links the backend). About 40 MB of device memory.
//
// Output, one line per arm and case:
//   probe=bf16_onednn case=K<K>xM<M>xN<N> arm=<id> created=<yes|no> [reason=...] impl=<str> scratch=<bytes>
//         span_us=<device span per call> ev_us=<last kernel's event per call> wall_us=<host per call>
//         tflops=<2MNK / span> err_f32x=<max err / sum|w*x| vs double over F32 x>
//         err_bf16x=<same vs double over RNE bf16(x)> act_rounding=<none|bf16|unclear> match=<Y|N>
// span_us brackets the calls with two empty kernels on the in-order profiling queue, so it is the device time of
// every kernel a call launched plus any gap between them; ev_us times only the event the call returned. The native
// kernel and the conversion print the same fields (arm=native, arm=convert_f32_to_bf16). The native kernel is timed
// again after the oneDNN arms (arm=native_after), so a card still warming up at the first timing cannot inflate a
// speedup: every speedup divides the smaller of the two native spans. One summary line per case:
//   summary case=... native_us=<span> native_after_us=<span> convert_us=<span>
//           route_us=<convert + fork.bf16xbf16.strict> route_speedup=<native / route>
//           mixed_strict=<fastest strict mixed arm whose act_rounding is none, or none> mixed_strict_us=<span>
//           mixed_strict_speedup=<native / it>
//           mixed_fpmath_bf16=<fastest fpmath_bf16 arm, or none> mixed_fpmath_bf16_us=<span>
//           mixed_fpmath_bf16_speedup=<native / it>
// Speedups compare span_us. A strict mixed arm keeps the activations in F32 (no rounding); an fpmath_bf16 arm rounds
// them inside oneDNN, so it is the explicit conversion's arithmetic without the copy.
//
// Exit: 77 when it cannot probe (no GPU, no oneDNN build, engine creation failed); 1 when an arm that created failed
// to execute or produced numbers outside its bound, or the native kernel or the conversion did (a defect in the
// probe, oneDNN or the kernel, which must not read as "unsupported"); 0 otherwise. A primitive that does not create
// is a printed "no", never a failure. Bounds: native 1e-4 * sum|w*x| against double over F32 x (the backend device test's bound); a oneDNN arm
// 4e-3 * sum|w*x| against the same reference (BF16 rounding of x is at most 2^-9 per product); the conversion is
// bit-equal to the host's round-to-nearest-even, which is the CPU backend's ggml_compute_fp32_to_bf16.
//
// Build: ./scripts/sycl-build.sh test-sycl-bf16-onednn-probe
// Run (lead only: it touches the GPU; once per card, pinned):
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 300 ./build/bin/test-sycl-bf16-onednn-probe; echo rc=$?
//   ctest --test-dir build -R '^test-sycl-bf16-onednn-probe$' -V    (pins level_zero:1; -V because the numbers are the
//                                                                    result and --output-on-failure hides a pass)
// Env: PROBE_ITERS (calls per timed arm, default 20).
//      PROBE_SMOKE=1 checks the harness itself on a CPU device (cpu_selector_v; run it with
//      ONEAPI_DEVICE_SELECTOR=opencl:cpu): small shapes, N = 9 and 16, two calls per arm. Its timings mean nothing and
//      its header says smoke=1, so it cannot be read as a card's result.

#if GGML_SYCL_DNNL
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_sycl.hpp>
#    include <sycl/sycl.hpp>
// The attribute production builds with, so the answer is about the primitive the backend would run.
#    include "ggml-sycl/dnnl-deterministic.hpp"
#    include "ggml-sycl/mul-mat-bf16.hpp"
#endif

#include "test-skip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int EXIT_SKIP = LLAMA_TEST_EXIT_SKIP;

#if GGML_SYCL_DNNL

namespace {

using dt = dnnl::memory::data_type;

struct probe_shape {
    int64_t K;  // ne00: the reduced dimension
    int64_t M;  // ne01: output rows
};

constexpr probe_shape SHAPES[] = {
    { 2304, 3072 },
    { 3072, 2304 },
};
const std::vector<int64_t> WIDTHS = { 9, 16, 64, 129, 512 };

constexpr probe_shape SMOKE_SHAPES[] = {
    { 256, 192 },
    { 192, 256 },
};
const std::vector<int64_t> SMOKE_WIDTHS = { 9, 16 };

constexpr int SAMPLES = 256;

constexpr double NATIVE_BOUND = 1e-4;
constexpr double ONEDNN_BOUND = 4e-3;

enum class orientation { fork, natural };
enum class act_type { bf16, f32 };
enum class float_math { strict, bf16 };

struct arm {
    const char * id;
    orientation  orient;
    act_type     act;
    float_math   fpmath;
};

// The first arm is the one gemm() would run after an explicit conversion; the summary's route_us is built from it.
constexpr arm ARMS[] = {
    { "fork.bf16xbf16.strict",     orientation::fork,    act_type::bf16, float_math::strict },
    { "natural.bf16xbf16.strict",  orientation::natural, act_type::bf16, float_math::strict },
    { "fork.mixed.strict",         orientation::fork,    act_type::f32,  float_math::strict },
    { "natural.mixed.strict",      orientation::natural, act_type::f32,  float_math::strict },
    { "fork.mixed.fpmath_bf16",    orientation::fork,    act_type::f32,  float_math::bf16   },
    { "natural.mixed.fpmath_bf16", orientation::natural, act_type::f32,  float_math::bf16   },
};

// Round to nearest even, the CPU backend's ggml_compute_fp32_to_bf16 for a finite value (every input here is).
uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

float bf16_to_f32_host(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float          f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::string env_or(const char * name, const char * fallback) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string(fallback);
}

struct device_buffers {
    sycl::queue &       q;
    std::vector<void *> owned;

    explicit device_buffers(sycl::queue & queue) : q(queue) {}

    device_buffers(const device_buffers &)             = delete;
    device_buffers & operator=(const device_buffers &) = delete;

    ~device_buffers() {
        // An exception can leave calls in flight on the queue; none may still read a buffer when it is freed.
        try {
            q.wait();
        } catch (const std::exception &) {
        }
        for (void * p : owned) {
            sycl::free(p, q);
        }
    }

    template <typename T> T * alloc(size_t count) {
        T * p = sycl::malloc_device<T>(count, q);
        if (!p) {
            throw std::runtime_error("malloc_device failed");
        }
        owned.push_back(p);
        return p;
    }
};

struct timing {
    double span_us = -1.0;
    double ev_us   = -1.0;
    double wall_us = -1.0;
};

// One untimed call first (JIT compilation, first-execution setup), then `iters` back-to-back calls bracketed by two
// empty kernels on the in-order profiling queue.
template <typename Submit> timing time_calls(sycl::queue & q, int iters, Submit && submit) {
    auto marker = [&q]() {
        return q.single_task([]() {});
    };
    submit().wait_and_throw();
    marker().wait_and_throw();

    std::vector<sycl::event> events;
    events.reserve(static_cast<size_t>(iters));
    sycl::event pre = marker();
    const auto  t0  = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        events.push_back(submit());
    }
    sycl::event post = marker();
    post.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();

    timing t;
    t.wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    try {
        const auto begin = pre.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto end   = post.get_profiling_info<sycl::info::event_profiling::command_start>();
        t.span_us        = static_cast<double>(end - begin) / 1000.0 / iters;
    } catch (const std::exception &) {
    }
    try {
        double sum_ns = 0.0;
        for (const sycl::event & e : events) {
            const auto st = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto en = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            sum_ns += static_cast<double>(en - st);
        }
        t.ev_us = sum_ns / 1000.0 / iters;
    } catch (const std::exception &) {
    }
    return t;
}

struct sample_ref {
    int64_t n;
    int64_t m;
    double  ref_f32x;   // sum over k of bf16(w) * x, x as stored (F32)
    double  ref_bf16x;  // sum over k of bf16(w) * bf16_rne(x)
    double  mag;        // sum over k of |bf16(w) * x|
};

struct error_pair {
    double f32x  = 0.0;
    double bf16x = 0.0;
};

error_pair measure_errors(const std::vector<float> & y, int64_t M, const std::vector<sample_ref> & refs) {
    error_pair e;
    for (const sample_ref & r : refs) {
        const double got = static_cast<double>(y[r.n * M + r.m]);
        const double ea  = std::fabs(got - r.ref_f32x) / r.mag;
        const double eb  = std::fabs(got - r.ref_bf16x) / r.mag;

        // A NaN (an output the call never wrote: the buffer is filled with NaN first) is the worst error.
        e.f32x  = std::isnan(ea) ? std::numeric_limits<double>::infinity() : std::max(e.f32x, ea);
        e.bf16x = std::isnan(eb) ? std::numeric_limits<double>::infinity() : std::max(e.bf16x, eb);
    }
    return e;
}

// Which arithmetic produced the output: an error ten times smaller against one reference than the other names it.
const char * act_rounding_of(const error_pair & e) {
    if (e.bf16x * 10.0 < e.f32x) {
        return "bf16";
    }
    if (e.f32x * 10.0 < e.bf16x) {
        return "none";
    }
    return "unclear";
}

void print_timing(const timing & t, double flop) {
    std::printf(" span_us=%.1f ev_us=%.1f wall_us=%.1f tflops=%.2f", t.span_us, t.ev_us, t.wall_us,
                t.span_us > 0.0 ? flop / t.span_us / 1e6 : -1.0);
}

struct case_operands {
    const uint16_t * w;   // [M][K] BF16
    const float *    x;   // [N][K] F32
    const uint16_t * xb;  // [N][K] BF16, the conversion's output
    float *          y;   // [N][M] F32
};

struct arm_result {
    bool         created      = false;
    bool         matched      = false;
    double       span_us      = -1.0;
    const char * act_rounding = "unclear";
};

arm_result run_arm(sycl::queue &                   q,
                   dnnl::engine &                  eng,
                   dnnl::stream &                  strm,
                   device_buffers &                bufs,
                   const arm &                     a,
                   int64_t                         K,
                   int64_t                         M,
                   int64_t                         N,
                   const case_operands &           ops,
                   const std::vector<sample_ref> & refs,
                   int                             iters) {
    arm_result result;
    std::printf("probe=bf16_onednn case=K%lldxM%lldxN%lld arm=%s", (long long) K, (long long) M, (long long) N, a.id);

    const dt           act_dt = a.act == act_type::bf16 ? dt::bf16 : dt::f32;
    dnnl::memory::desc src_md;
    dnnl::memory::desc wei_md;
    dnnl::memory::desc dst_md;
    const void *       src_ptr = nullptr;
    const void *       wei_ptr = nullptr;
    const void *       act_ptr = a.act == act_type::bf16 ? static_cast<const void *>(ops.xb) : ops.x;
    if (a.orient == orientation::fork) {
        // gemm.hpp's descriptors for m = M, n = N, k = K, stra0 = 1, stra1 = K, strb0 = 1, strb1 = K, ldc = M.
        src_md  = dnnl::memory::desc({ 1, M, K }, dt::bf16, { M * K, K, 1 });
        wei_md  = dnnl::memory::desc({ 1, K, N }, act_dt, { K * N, 1, K });
        dst_md  = dnnl::memory::desc({ 1, M, N }, dt::f32, { M * N, 1, M });
        src_ptr = ops.w;
        wei_ptr = act_ptr;
    } else {
        src_md  = dnnl::memory::desc({ 1, N, K }, act_dt, { N * K, K, 1 });
        wei_md  = dnnl::memory::desc({ 1, K, M }, dt::bf16, { K * M, 1, K });
        dst_md  = dnnl::memory::desc({ 1, N, M }, dt::f32, { N * M, M, 1 });
        src_ptr = act_ptr;
        wei_ptr = ops.w;
    }

    dnnl::primitive_attr attr;
    ggml_sycl_dnnl_attr_set_reproducible(attr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(a.fpmath == float_math::strict ? dnnl::fpmath_mode::strict : dnnl::fpmath_mode::bf16);

    dnnl::matmul::primitive_desc pd;
    try {
        pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    } catch (const dnnl::error & e) {
        std::printf(" created=no reason=\"%s\"\n", e.what());
        return result;
    } catch (const std::exception & e) {
        std::printf(" created=no reason=\"%s\"\n", e.what());
        return result;
    }
    result.created            = true;
    const size_t scratch_size = pd.scratchpad_desc().get_size();
    std::printf(" created=yes impl=%s scratch=%zu", pd.impl_info_str(), scratch_size);

    try {
        dnnl::matmul                          prim(pd);
        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(src_md, eng, const_cast<void *>(src_ptr)) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(wei_md, eng, const_cast<void *>(wei_ptr)) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(dst_md, eng, ops.y) });
        if (scratch_size > 0) {
            void * scratch = bufs.alloc<uint8_t>(scratch_size);
            args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(pd.scratchpad_desc(), eng, scratch) });
        }

        q.fill(ops.y, std::numeric_limits<float>::quiet_NaN(), static_cast<size_t>(N * M)).wait_and_throw();
        const timing t = time_calls(q, iters, [&]() { return dnnl::sycl_interop::execute(prim, strm, args, {}); });

        std::vector<float> y(static_cast<size_t>(N * M));
        q.memcpy(y.data(), ops.y, y.size() * sizeof(float)).wait_and_throw();
        const error_pair e = measure_errors(y, M, refs);

        print_timing(t, 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K));
        result.matched      = e.f32x <= ONEDNN_BOUND;
        result.span_us      = t.span_us;
        result.act_rounding = act_rounding_of(e);
        std::printf(" err_f32x=%.3g err_bf16x=%.3g act_rounding=%s match=%s\n", e.f32x, e.bf16x, result.act_rounding,
                    result.matched ? "Y" : "N");
    } catch (const dnnl::error & e) {
        std::printf(" exec_failed reason=\"%s\"\n", e.what());
    } catch (const sycl::exception & e) {
        std::printf(" exec_failed reason=\"%s\"\n", e.what());
    }
    return result;
}

int run_shape(sycl::queue &                q,
              dnnl::engine &               eng,
              dnnl::stream &               strm,
              const probe_shape &          s,
              const std::vector<int64_t> & widths,
              size_t                       max_wg,
              int                          iters) {
    const int64_t K  = s.K;
    const int64_t M  = s.M;
    int           rc = 0;

    std::mt19937                          rng(static_cast<unsigned>(1234 + K * 7 + M));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> w_bits(static_cast<size_t>(M * K));
    std::vector<float>    w_dec(w_bits.size());
    for (size_t i = 0; i < w_bits.size(); ++i) {
        w_bits[i] = f32_to_bf16_rne(dist(rng));
        w_dec[i]  = bf16_to_f32_host(w_bits[i]);
    }
    const int64_t         max_width = *std::max_element(widths.begin(), widths.end());
    std::vector<float>    x(static_cast<size_t>(max_width * K));
    std::vector<uint16_t> x_rne(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x[i]     = dist(rng);
        x_rne[i] = f32_to_bf16_rne(x[i]);
    }

    device_buffers bufs(q);
    uint16_t *     w_dev  = bufs.alloc<uint16_t>(w_bits.size());
    float *        x_dev  = bufs.alloc<float>(x.size());
    uint16_t *     xb_dev = bufs.alloc<uint16_t>(x.size());
    float *        y_dev  = bufs.alloc<float>(static_cast<size_t>(max_width * M));
    q.memcpy(w_dev, w_bits.data(), w_bits.size() * sizeof(uint16_t)).wait_and_throw();
    q.memcpy(x_dev, x.data(), x.size() * sizeof(float)).wait_and_throw();

    for (int64_t N : widths) {
        const double flop = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

        // Reference outputs at SAMPLES random (n, m) positions, in double.
        std::vector<sample_ref>                refs;
        std::uniform_int_distribution<int64_t> pick_n(0, N - 1);
        std::uniform_int_distribution<int64_t> pick_m(0, M - 1);
        for (int i = 0; i < SAMPLES; ++i) {
            sample_ref r{ pick_n(rng), pick_m(rng), 0.0, 0.0, 0.0 };
            for (int64_t k = 0; k < K; ++k) {
                const double wk = w_dec[r.m * K + k];
                const double xk = x[r.n * K + k];
                r.ref_f32x += wk * xk;
                r.ref_bf16x += wk * static_cast<double>(bf16_to_f32_host(x_rne[r.n * K + k]));
                r.mag += std::fabs(wk * xk);
            }
            if (r.mag == 0.0) {
                r.mag = 1.0;
            }
            refs.push_back(r);
        }

        // The native kernel, as the backend calls it: ldx = K, ldy = M.
        std::printf("probe=bf16_onednn case=K%lldxM%lldxN%lld arm=native", (long long) K, (long long) M, (long long) N);
        q.fill(y_dev, std::numeric_limits<float>::quiet_NaN(), static_cast<size_t>(N * M)).wait_and_throw();
        auto native_call = [&]() {
            return ggml_sycl_bf16::mul_mat_bf16_f32(q, w_dev, x_dev, y_dev, K, M, N, K, M, max_wg);
        };
        const timing tn = time_calls(q, iters, native_call);

        std::vector<float> y(static_cast<size_t>(N * M));
        q.memcpy(y.data(), y_dev, y.size() * sizeof(float)).wait_and_throw();
        const error_pair en = measure_errors(y, M, refs);
        print_timing(tn, flop);
        const bool native_ok = en.f32x <= NATIVE_BOUND;
        std::printf(" err_f32x=%.3g err_bf16x=%.3g act_rounding=%s match=%s\n", en.f32x, en.bf16x, act_rounding_of(en),
                    native_ok ? "Y" : "N");
        if (!native_ok) {
            rc = 1;
        }

        // The F32 -> BF16 activation conversion: one element per work-item, round to nearest even.
        const size_t n_act = static_cast<size_t>(N * K);
        auto *       xb    = reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(xb_dev);
        std::printf("probe=bf16_onednn case=K%lldxM%lldxN%lld arm=convert_f32_to_bf16 elems=%zu", (long long) K,
                    (long long) M, (long long) N, n_act);
        const float * x_src          = x_dev;
        auto          convert_kernel = [=](sycl::id<1> i) {
            xb[i] = sycl::ext::oneapi::bfloat16(x_src[i]);
        };
        auto convert_call = [&]() {
            return q.parallel_for(sycl::range<1>(n_act), convert_kernel);
        };
        const timing tc = time_calls(q, iters, convert_call);

        std::vector<uint16_t> xb_host(n_act);
        q.memcpy(xb_host.data(), xb_dev, n_act * sizeof(uint16_t)).wait_and_throw();
        size_t mismatches = 0;
        for (size_t i = 0; i < n_act; ++i) {
            mismatches += xb_host[i] != x_rne[i] ? 1 : 0;
        }
        std::printf(" span_us=%.1f ev_us=%.1f wall_us=%.1f gbps=%.1f rne_mismatches=%zu match=%s\n", tc.span_us,
                    tc.ev_us, tc.wall_us, tc.span_us > 0.0 ? 6.0 * static_cast<double>(n_act) / tc.span_us / 1e3 : -1.0,
                    mismatches, mismatches == 0 ? "Y" : "N");
        if (mismatches != 0) {
            rc = 1;
        }

        const case_operands ops{ w_dev, x_dev, xb_dev, y_dev };
        double              route_gemm_us = -1.0;
        const char *        strict_arm    = "none";
        double              strict_us     = -1.0;
        const char *        fpmath_arm    = "none";
        double              fpmath_us     = -1.0;
        for (const arm & a : ARMS) {
            const arm_result r = run_arm(q, eng, strm, bufs, a, K, M, N, ops, refs, iters);
            if (r.created && !r.matched) {
                rc = 1;
            }
            if (!r.created || !r.matched || r.span_us <= 0.0) {
                continue;
            }
            if (&a == &ARMS[0]) {
                route_gemm_us = r.span_us;
            }
            if (a.act != act_type::f32) {
                continue;
            }
            if (a.fpmath == float_math::strict) {
                if (std::strcmp(r.act_rounding, "none") == 0 && (strict_us < 0.0 || r.span_us < strict_us)) {
                    strict_arm = a.id;
                    strict_us  = r.span_us;
                }
            } else if (fpmath_us < 0.0 || r.span_us < fpmath_us) {
                fpmath_arm = a.id;
                fpmath_us  = r.span_us;
            }
        }

        std::printf("probe=bf16_onednn case=K%lldxM%lldxN%lld arm=native_after", (long long) K, (long long) M,
                    (long long) N);
        const timing tn_after = time_calls(q, iters, native_call);
        print_timing(tn_after, flop);
        std::printf("\n");

        double native_us = tn.span_us;
        if (tn_after.span_us > 0.0 && (native_us <= 0.0 || tn_after.span_us < native_us)) {
            native_us = tn_after.span_us;
        }
        auto speedup = [&](double us) {
            return us > 0.0 && native_us > 0.0 ? native_us / us : -1.0;
        };
        const double route_us = route_gemm_us > 0.0 && tc.span_us > 0.0 ? route_gemm_us + tc.span_us : -1.0;
        std::printf(
            "summary case=K%lldxM%lldxN%lld native_us=%.1f native_after_us=%.1f convert_us=%.1f route_us=%.1f "
            "route_speedup=%.2f mixed_strict=%s mixed_strict_us=%.1f mixed_strict_speedup=%.2f mixed_fpmath_bf16=%s "
            "mixed_fpmath_bf16_us=%.1f mixed_fpmath_bf16_speedup=%.2f\n",
            (long long) K, (long long) M, (long long) N, tn.span_us, tn_after.span_us, tc.span_us, route_us,
            speedup(route_us), strict_arm, strict_us, speedup(strict_us), fpmath_arm, fpmath_us, speedup(fpmath_us));
    }
    return rc;
}

}  // namespace
#endif  // GGML_SYCL_DNNL

int main() {
    // Each line is printed in parts around a call that can hang or abort; unbuffered, a killed run still shows which
    // arm it was in.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#if GGML_SYCL_DNNL
    // A failure before setup_done is an absence (no GPU under the selector, no engine): exit 77. A failure after it
    // is a result: exit 1.
    bool       setup_done = false;
    const bool smoke      = env_or("PROBE_SMOKE", "") == "1";
    try {
        sycl::device  dev = smoke ? sycl::device{ sycl::cpu_selector_v } : sycl::device{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        sycl::queue   q{
            ctx, dev, { sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }
        };
        const size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
        dnnl::engine eng    = dnnl::sycl_interop::make_engine(dev, ctx);
        dnnl::stream strm   = dnnl::sycl_interop::make_stream(eng, q);
        std::printf(
            "probe=bf16_onednn device=\"%s\" driver=%s dnnl=%d.%d.%d deterministic=%d max_work_group=%zu smoke=%d\n",
            dev.get_info<sycl::info::device::name>().c_str(),
            dev.get_info<sycl::info::device::driver_version>().c_str(), DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR,
            DNNL_VERSION_PATCH, ggml_sycl_onednn_deterministic_enabled() ? 1 : 0, max_wg, smoke ? 1 : 0);
        if (max_wg < ggml_sycl_bf16::MIN_WORK_GROUP) {
            std::printf("SKIP: bf16_onednn.probe max_work_group %zu below the native kernel's %zu\n", max_wg,
                        ggml_sycl_bf16::MIN_WORK_GROUP);
            return EXIT_SKIP;
        }
        setup_done = true;

        const int iters = smoke ? 2 : std::max(1, std::atoi(env_or("PROBE_ITERS", "20").c_str()));
        int       rc    = 0;
        if (smoke) {
            for (const probe_shape & s : SMOKE_SHAPES) {
                rc |= run_shape(q, eng, strm, s, SMOKE_WIDTHS, max_wg, iters);
            }
        } else {
            for (const probe_shape & s : SHAPES) {
                rc |= run_shape(q, eng, strm, s, WIDTHS, max_wg, iters);
            }
        }
        std::puts(rc == 0 ? "PASS: bf16_onednn.probe completed" : "FAIL: bf16_onednn.probe a result is out of bounds");
        return rc;
    } catch (const std::exception & e) {
        if (!setup_done) {
            std::printf("SKIP: bf16_onednn.probe setup_failed reason=\"%s\"\n", e.what());
            return EXIT_SKIP;
        }
        std::printf("\nFAIL: bf16_onednn.probe run_failed reason=\"%s\"\n", e.what());
        return 1;
    }
#else
    std::puts("SKIP: bf16_onednn.probe GGML_SYCL_DNNL=0");
    return EXIT_SKIP;
#endif
}
