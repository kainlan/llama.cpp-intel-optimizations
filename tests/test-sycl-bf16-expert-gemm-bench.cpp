// Step 0 of Task 24 (Inkling 2 Mini plan): what one oneDNN bf16 x bf16 -> f32 GEMM per active expert costs on the
// card, measured before the BF16 MUL_MAT_ID prompt arm is written.
//
// QUESTION THIS ANSWERS. The arm runs one GEMM per active expert of a prompt node, reading the expert's BF16 weight
// where it is loaded. A long-prompt pass has about 88 such nodes of about 100 active experts, so about 8,800 calls
// inside a device budget of 0.6 s, and the host submits every one of them while the device waits behind the node's
// ids read. Two numbers decide whether that is viable: the device time of one call at the row counts the arm
// produces, and the host time of one submission. A third, the token count from which the arm beats the BF16 direct
// kernel on a whole node, sets the arm's minimum width.
//
// Two parts, on Inkling's expert shapes (gate_up: K 2304, M 4608, one activation row per token; down: K 2304, M 2304,
// one row per slot), with 128 experts whose weights are all distinct, so no call finds another's weight in a cache:
//   - Per call. For n = 16, 32, 64 and 128 rows per expert: one primitive, then PASSES passes of 128 back-to-back
//     calls, one per expert, each on its own weight, activation slot and output slot. Every call builds its memory
//     objects and its argument map and depends on an event, as DnnlGemmWrapper::gemm (gemm.hpp) does for the arm,
//     with the descriptors gemm() builds: the weight as oneDNN's source ([m, k], k contiguous), the activations as its
//     weights ([k, n], k contiguous), dst [m, n] with m contiguous, fpmath strict
//     (GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT), a user scratchpad and the backend's deterministic attribute. The host
//     time per submit leaves out what gemm() adds around that (its mutex, key and cache lookup).
//   - Per node. For T = 16 to 512 tokens routed top-8 to distinct experts drawn uniformly: the BF16 direct kernel
//     (moe-bf16-direct.cpp, compiled into this binary) over the whole node, against the arm's GEMMs alone, each active
//     expert's rows rounded up to moe_pp_gemm_row_align (moe-pp-pack.hpp) and packed one after another. The arm's
//     gather and scatter are not in it. Uniform routing activates nearly every expert at 512 tokens, against about
//     100 on the real node, so the GEMM count is the high side.
//
// It is STANDALONE: SYCL, oneDNN, the backend's deterministic attribute header and the backend-free direct kernel; no
// backend, no unified cache. The raw sycl::malloc_device below is a bench fixture, not a backend allocation (the
// unified-cache rule governs backend code, and this binary never links the backend). Weights and activations are
// filled on the device from an integer hash that the host evaluates again for the reference, so host memory stays
// small. Device memory peaks at about 3.2 GB (gate_up: 2.7 GB of weights, 0.4 GB of activation and output slots).
//
// Output:
//   bench=bf16_expert_gemm device="..." driver=... dnnl=X.Y.Z deterministic=0|1 smoke=0|1 passes=P experts=E top_k=K
//   bench=bf16_expert_gemm shape=<gate_up|down> K= M= n= calls= created=<yes|no> [reason="..."] impl=<oneDNN impl>
//         scratch=<bytes> dev_us_mean= dev_us_median= span_us_per_call= host_us_mean= host_us_median= err= match=Y|N
//   bench=bf16_expert_node shape=... tokens=T active= distinct_n= packed_rows= direct_us= arm_gemm_us= arm_host_us=
//         arm_total_us= arm_wins=Y|N
//   summary n64_span_us= n64_dev_us= n64_host_us= verdict=<PROCEED|MARGINAL|STOP> suggested_min_tokens=<T|none>
// dev_us_* are the event of each call (the last kernel a call launched); span_us_per_call brackets a pass with two
// empty kernels on the in-order profiling queue and divides by its calls, so it holds every kernel of every call and
// the gaps between them. host_us_* is the host time of one submission, memory objects and argument map included.
// direct_us and arm_gemm_us are bracketed spans per node; arm_host_us is the host time to submit the node's GEMMs;
// arm_total_us adds the two, since after the node's ids read the device has nothing queued while the host submits.
// The summary reads n = 64: n64_span_us is the mean over the two shapes of span_us_per_call, n64_dev_us the same of
// dev_us_mean, n64_host_us the larger host_us_mean. The verdict applies the lead's thresholds (Task 24, rulings
// after the checkpoint): STOP when n64_span_us > 60 or n64_host_us > 100, MARGINAL when n64_host_us > 40, else
// PROCEED. suggested_min_tokens is the smallest T from which the arm wins on both shapes at every larger T measured.
//
// Exit: 77 when it cannot measure (no device under the selector, no oneDNN build, engine creation failed); 1 when a
// primitive does not create, a call or the direct kernel fails, or a sampled output is outside its bound; 0
// otherwise. The verdict does not change the exit status: it is a measurement for the lead to rule on. Bound: 1e-4 of
// sum |w * x| against a double reference over the same BF16 values (the f32 sum's own rounding is far below it).
//
// Build: ./scripts/sycl-build.sh test-sycl-bf16-expert-gemm-bench
// Run (lead only: it touches the GPU; once, pinned to the B70). Not a ctest.
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 300 ./build/bin/test-sycl-bf16-expert-gemm-bench; echo rc=$?
// Env: BENCH_PASSES (timed passes per measurement, default 3).
//      BENCH_SMOKE=1 checks the harness on a CPU device (run it with ONEAPI_DEVICE_SELECTOR=opencl:cpu): small shapes,
//      8 experts, top-2, one pass. Its timings mean nothing and its header says smoke=1.

#if GGML_SYCL_DNNL
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_sycl.hpp>
#    include <sycl/sycl.hpp>
// The attribute production builds with, so the numbers are about the primitive the backend would run.
#    include "ggml-sycl/dnnl-deterministic.hpp"
#    include "ggml-sycl/moe-bf16-direct.hpp"
#    include "ggml-sycl/moe-pp-pack.hpp"
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
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int EXIT_SKIP = LLAMA_TEST_EXIT_SKIP;

#if GGML_SYCL_DNNL

namespace {

using dt = dnnl::memory::data_type;

// The lead's thresholds for n = 64 (Task 24, rulings after the checkpoint).
constexpr double  SPAN_STOP_US     = 60.0;
constexpr double  HOST_STOP_US     = 100.0;
constexpr double  HOST_MARGINAL_US = 40.0;
constexpr int64_t SUMMARY_N        = 64;

constexpr int    SAMPLES = 32;
constexpr double BOUND   = 1e-4;

constexpr uint32_t SALT_W    = 0x5eed0001u;
constexpr uint32_t SALT_X    = 0x5eed0002u;
constexpr uint32_t SALT_SRC1 = 0x5eed0003u;

struct bench_shape {
    const char * name;
    int64_t      K;              // ne00: the reduced dimension
    int64_t      M;              // ne01: output rows per expert
    bool         rows_per_slot;  // down: src1 holds one row per slot; gate_up: one row per token
};

constexpr bench_shape SHAPES[] = {
    { "gate_up", 2304, 4608, false },
    { "down",    2304, 2304, true  },
};
constexpr bench_shape SMOKE_SHAPES[] = {
    { "gate_up", 256, 192, false },
    { "down",    256, 128, true  },
};

struct bench_config {
    int64_t              n_experts = 0;
    int64_t              top_k     = 0;
    std::vector<int64_t> widths;
    std::vector<int64_t> tokens;
    int                  passes = 0;
    bool                 smoke  = false;
};

// lowbias32: an integer hash the host and the device evaluate bit for bit alike.
inline uint32_t bench_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A value in [-1, 1) with 24 significant bits, exact in float on either side.
inline float bench_value(uint32_t salt, uint32_t index) {
    const uint32_t h = bench_mix(index ^ bench_mix(salt));
    return static_cast<float>(h >> 8) * 0x1p-23f - 1.0f;
}

// Round to nearest even, the CPU backend's ggml_compute_fp32_to_bf16 for a finite value (every value here is).
inline uint16_t bench_bf16_rne(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

float bf16_to_f32_host(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float          f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

double bf16_value_host(uint32_t salt, uint64_t index) {
    return static_cast<double>(bf16_to_f32_host(bench_bf16_rne(bench_value(salt, static_cast<uint32_t>(index)))));
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

sycl::event fill_bf16(sycl::queue & q, uint16_t * dst, size_t count, uint32_t salt, uint64_t base) {
    const uint32_t b = static_cast<uint32_t>(base);
    return q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
        dst[i] = bench_bf16_rne(bench_value(salt, b + static_cast<uint32_t>(i[0])));
    });
}

sycl::event fill_f32(sycl::queue & q, float * dst, size_t count, uint32_t salt) {
    return q.parallel_for(sycl::range<1>(count),
                          [=](sycl::id<1> i) { dst[i] = bench_value(salt, static_cast<uint32_t>(i[0])); });
}

sycl::event marker(sycl::queue & q) {
    return q.single_task([]() {});
}

double span_us(const sycl::event & pre, const sycl::event & post) {
    const auto begin = pre.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto end   = post.get_profiling_info<sycl::info::event_profiling::command_start>();
    return static_cast<double>(end - begin) / 1000.0;
}

double event_us(const sycl::event & e) {
    const auto st = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto en = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(en - st) / 1000.0;
}

double mean_of(const std::vector<double> & v) {
    double s = 0.0;
    for (double x : v) {
        s += x;
    }
    return v.empty() ? -1.0 : s / static_cast<double>(v.size());
}

double median_of(std::vector<double> v) {
    if (v.empty()) {
        return -1.0;
    }
    std::sort(v.begin(), v.end());
    const size_t h = v.size() / 2;
    return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

double host_us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

// One created primitive for (K, M, n), with the descriptors DnnlGemmWrapper::gemm builds for m = M, n, k = K,
// stra0 = 1, stra1 = K, strb0 = 1, strb1 = K, ldc = M and one batch.
struct gemm_prim {
    dnnl::matmul       prim;
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::memory::desc scratch_md;
    size_t             scratch_size = 0;
    void *             scratch      = nullptr;
    std::string        impl;
};

bool make_gemm(dnnl::engine &   eng,
               device_buffers & bufs,
               int64_t          K,
               int64_t          M,
               int64_t          n,
               gemm_prim *      out,
               std::string *    reason) {
    const dnnl::memory::desc a_md({ 1, M, K }, dt::bf16, { M * K, K, 1 });
    const dnnl::memory::desc b_md({ 1, K, n }, dt::bf16, { K * n, 1, K });
    const dnnl::memory::desc c_md({ 1, M, n }, dt::f32, { M * n, 1, M });
    dnnl::primitive_attr     attr;
    ggml_sycl_dnnl_attr_set_reproducible(attr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    try {
        const dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
        out->a_md         = a_md;
        out->b_md         = b_md;
        out->c_md         = pd.dst_desc();
        out->scratch_md   = pd.scratchpad_desc();
        out->scratch_size = out->scratch_md.get_size();
        out->impl         = pd.impl_info_str();
        out->prim         = dnnl::matmul(pd);
    } catch (const std::exception & e) {
        *reason = e.what();
        return false;
    }
    // One scratchpad per primitive: every call runs on the one in-order queue, so its calls can share it.
    out->scratch = out->scratch_size > 0 ? bufs.alloc<uint8_t>(out->scratch_size) : nullptr;
    return true;
}

// One call as gemm() makes it for the arm: memory objects over this expert's pointers, the argument map, execute
// after `deps`.
sycl::event run_gemm(const gemm_prim &                g,
                     dnnl::engine &                   eng,
                     dnnl::stream &                   strm,
                     const void *                     a,
                     const void *                     b,
                     void *                           c,
                     const std::vector<sycl::event> & deps) {
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, dnnl::memory(g.a_md, eng, const_cast<void *>(a)) });
    args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(g.b_md, eng, const_cast<void *>(b)) });
    args.insert({ DNNL_ARG_DST, dnnl::memory(g.c_md, eng, c) });
    if (g.scratch_size > 0) {
        args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(g.scratch_md, eng, g.scratch) });
    }
    return dnnl::sycl_interop::execute(g.prim, strm, args, deps);
}

struct call_result {
    bool   ok          = false;
    double span_per_us = -1.0;
    double dev_mean_us = -1.0;
    double host_mean   = -1.0;
};

struct node_result {
    int64_t tokens = 0;
    bool    wins   = false;
};

struct shape_buffers {
    uint16_t * w   = nullptr;  // [E][M][K] BF16
    uint16_t * act = nullptr;  // activation slots, BF16 rows of K
    float *    out = nullptr;  // output slots, F32 rows of M
};

// Per call: 128 back-to-back calls per pass at `n` rows per expert, then sampled outputs against the reference.
call_result run_calls(sycl::queue &         q,
                      dnnl::engine &        eng,
                      dnnl::stream &        strm,
                      device_buffers &      bufs,
                      const bench_shape &   s,
                      const bench_config &  cfg,
                      const shape_buffers & sb,
                      int64_t               n,
                      std::mt19937 &        rng,
                      int &                 rc) {
    const int64_t K = s.K;
    const int64_t M = s.M;
    const int64_t E = cfg.n_experts;
    call_result   r;
    std::printf("bench=bf16_expert_gemm shape=%s K=%lld M=%lld n=%lld calls=%lld", s.name, (long long) K, (long long) M,
                (long long) n, (long long) (E * cfg.passes));

    gemm_prim   g;
    std::string reason;
    if (!make_gemm(eng, bufs, K, M, n, &g, &reason)) {
        std::printf(" created=no reason=\"%s\"\n", reason.c_str());
        rc = 1;
        return r;
    }
    std::printf(" created=yes impl=%s scratch=%zu", g.impl.c_str(), g.scratch_size);

    // Expert e's activation slot is act[e * n * K ...], its output slot out[e * n * M ...].
    fill_bf16(q, sb.act, static_cast<size_t>(E * n * K), SALT_X, 0).wait_and_throw();
    q.fill(sb.out, std::numeric_limits<float>::quiet_NaN(), static_cast<size_t>(E * n * M)).wait_and_throw();
    run_gemm(g, eng, strm, sb.w, sb.act, sb.out, {}).wait_and_throw();

    std::vector<double>      host_us;
    std::vector<sycl::event> events;
    host_us.reserve(static_cast<size_t>(E * cfg.passes));
    events.reserve(static_cast<size_t>(E * cfg.passes));
    double span_total = 0.0;
    for (int p = 0; p < cfg.passes; ++p) {
        const sycl::event              pre  = marker(q);
        const std::vector<sycl::event> deps = { pre };
        for (int64_t e = 0; e < E; ++e) {
            const auto t0 = std::chrono::steady_clock::now();
            events.push_back(run_gemm(g, eng, strm, sb.w + e * M * K, sb.act + e * n * K, sb.out + e * n * M, deps));
            host_us.push_back(host_us_since(t0));
        }
        sycl::event post = marker(q);  // wait_and_throw() is not const
        post.wait_and_throw();
        span_total += span_us(pre, post);
    }
    std::vector<double> dev_us;
    dev_us.reserve(events.size());
    for (const sycl::event & e : events) {
        dev_us.push_back(event_us(e));
    }

    std::uniform_int_distribution<int64_t> pick_e(0, E - 1);
    std::uniform_int_distribution<int64_t> pick_j(0, n - 1);
    std::uniform_int_distribution<int64_t> pick_m(0, M - 1);
    double                                 err = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        const int64_t e   = pick_e(rng);
        const int64_t j   = pick_j(rng);
        const int64_t m   = pick_m(rng);
        float         got = 0.0f;
        q.memcpy(&got, sb.out + e * n * M + j * M + m, sizeof(float)).wait_and_throw();
        double ref = 0.0;
        double mag = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            const double w = bf16_value_host(SALT_W, static_cast<uint64_t>(e * M * K + m * K + k));
            const double x = bf16_value_host(SALT_X, static_cast<uint64_t>(e * n * K + j * K + k));
            ref += w * x;
            mag += std::fabs(w * x);
        }
        const double ei = std::fabs(static_cast<double>(got) - ref) / (mag > 0.0 ? mag : 1.0);
        // A NaN is an output the calls never wrote (the slots are filled with NaN first): the worst error.
        err             = std::isnan(ei) ? std::numeric_limits<double>::infinity() : std::max(err, ei);
    }
    const bool matched = err <= BOUND;
    if (!matched) {
        rc = 1;
    }

    r.ok          = matched;
    r.span_per_us = span_total / static_cast<double>(E * cfg.passes);
    r.dev_mean_us = mean_of(dev_us);
    r.host_mean   = mean_of(host_us);
    std::printf(
        " dev_us_mean=%.1f dev_us_median=%.1f span_us_per_call=%.1f host_us_mean=%.1f host_us_median=%.1f err=%.3g "
        "match=%s\n",
        r.dev_mean_us, median_of(dev_us), r.span_per_us, r.host_mean, median_of(host_us), err, matched ? "Y" : "N");
    return r;
}

// Per node: the direct kernel on T tokens against the arm's GEMMs for the same routing.
node_result run_node(sycl::queue &         q,
                     dnnl::engine &        eng,
                     dnnl::stream &        strm,
                     device_buffers &      bufs,
                     const bench_shape &   s,
                     const bench_config &  cfg,
                     const shape_buffers & sb,
                     int64_t               T,
                     const void * const *  table_dev,
                     float *               src1_dev,
                     int32_t *             ids_dev,
                     float *               dst_dev,
                     std::mt19937 &        rng,
                     int &                 rc) {
    const int64_t K         = s.K;
    const int64_t M         = s.M;
    const int64_t E         = cfg.n_experts;
    const int64_t top_k     = cfg.top_k;
    const int64_t src1_rows = s.rows_per_slot ? top_k : 1;
    node_result   r;
    r.tokens = T;

    // Top-k distinct experts per token, drawn uniformly.
    std::vector<int32_t> ids(static_cast<size_t>(T * top_k));
    std::vector<int64_t> rows(static_cast<size_t>(E), 0);
    std::vector<int32_t> pool(static_cast<size_t>(E));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            pool[static_cast<size_t>(e)] = static_cast<int32_t>(e);
        }
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int64_t sl = 0; sl < top_k; ++sl) {
            const int32_t e                          = pool[static_cast<size_t>(sl)];
            ids[static_cast<size_t>(t * top_k + sl)] = e;
            ++rows[static_cast<size_t>(e)];
        }
    }
    q.memcpy(ids_dev, ids.data(), ids.size() * sizeof(int32_t)).wait_and_throw();
    fill_f32(q, src1_dev, static_cast<size_t>(K * src1_rows * T), SALT_SRC1).wait_and_throw();

    ggml_sycl::moe_bf16_direct_shape shape;
    shape.k         = K;
    shape.rows      = M;
    shape.n_experts = E;
    shape.n_slots   = top_k;
    shape.n_tokens  = T;
    shape.src1_rows = src1_rows;
    shape.w_nb1     = static_cast<size_t>(K) * sizeof(uint16_t);
    shape.src1_nb1  = static_cast<size_t>(K) * sizeof(float);
    shape.src1_nb2  = static_cast<size_t>(K * src1_rows) * sizeof(float);
    shape.ids_nb0   = sizeof(int32_t);
    shape.ids_nb1   = static_cast<size_t>(top_k) * sizeof(int32_t);
    shape.dst_nb1   = static_cast<size_t>(M) * sizeof(float);
    shape.dst_nb2   = static_cast<size_t>(M * top_k) * sizeof(float);

    auto direct_call = [&]() {
        sycl::event last;
        if (!ggml_sycl::moe_bf16_direct_submit(q, table_dev, src1_dev, ids_dev, dst_dev, shape, &last)) {
            throw std::runtime_error("the BF16 direct kernel refused the shape");
        }
        return last;
    };
    direct_call().wait_and_throw();
    double direct_total = 0.0;
    for (int p = 0; p < cfg.passes; ++p) {
        const sycl::event pre = marker(q);
        direct_call();
        sycl::event post = marker(q);  // wait_and_throw() is not const
        post.wait_and_throw();
        direct_total += span_us(pre, post);
    }
    const double direct_us = direct_total / cfg.passes;

    // The arm's layout: each active expert's rows rounded up to the GEMM row alignment, packed in expert order.
    constexpr int64_t            align = static_cast<int64_t>(ggml_sycl::moe_pp_gemm_row_align);
    std::vector<int64_t>         active;
    std::vector<int64_t>         slot_rows;
    std::vector<int64_t>         slot_base;
    std::map<int64_t, gemm_prim> prims;
    int64_t                      packed = 0;
    for (int64_t e = 0; e < E; ++e) {
        if (rows[static_cast<size_t>(e)] == 0) {
            continue;
        }
        const int64_t n = (rows[static_cast<size_t>(e)] + align - 1) / align * align;
        active.push_back(e);
        slot_rows.push_back(n);
        slot_base.push_back(packed);
        packed += n;
        if (prims.find(n) == prims.end()) {
            std::string reason;
            gemm_prim   g;
            if (!make_gemm(eng, bufs, K, M, n, &g, &reason)) {
                std::printf("bench=bf16_expert_node shape=%s tokens=%lld n=%lld created=no reason=\"%s\"\n", s.name,
                            (long long) T, (long long) n, reason.c_str());
                rc = 1;
                return r;
            }
            prims.emplace(n, std::move(g));
        }
    }

    auto arm_node = [&](const std::vector<sycl::event> & deps, double * host_us) {
        double host = 0.0;
        for (size_t i = 0; i < active.size(); ++i) {
            const int64_t e  = active[i];
            const auto    t0 = std::chrono::steady_clock::now();
            run_gemm(prims.at(slot_rows[i]), eng, strm, sb.w + e * M * K, sb.act + slot_base[i] * K,
                     sb.out + slot_base[i] * M, deps);
            host += host_us_since(t0);
        }
        if (host_us) {
            *host_us = host;
        }
    };
    arm_node({}, nullptr);
    q.wait_and_throw();
    double arm_total = 0.0;
    double arm_host  = 0.0;
    for (int p = 0; p < cfg.passes; ++p) {
        const sycl::event pre  = marker(q);
        double            host = 0.0;
        arm_node({ pre }, &host);
        sycl::event post = marker(q);  // wait_and_throw() is not const
        post.wait_and_throw();
        arm_total += span_us(pre, post);
        arm_host += host;
    }
    const double arm_gemm_us = arm_total / cfg.passes;
    const double arm_host_us = arm_host / cfg.passes;
    const double arm_all_us  = arm_gemm_us + arm_host_us;
    r.wins                   = arm_all_us < direct_us;
    std::printf(
        "bench=bf16_expert_node shape=%s tokens=%lld active=%zu distinct_n=%zu packed_rows=%lld direct_us=%.1f "
        "arm_gemm_us=%.1f arm_host_us=%.1f arm_total_us=%.1f arm_wins=%s\n",
        s.name, (long long) T, active.size(), prims.size(), (long long) packed, direct_us, arm_gemm_us, arm_host_us,
        arm_all_us, r.wins ? "Y" : "N");
    return r;
}

struct shape_summary {
    call_result              at_summary_n;
    std::vector<node_result> nodes;
};

shape_summary run_shape(sycl::queue &        q,
                        dnnl::engine &       eng,
                        dnnl::stream &       strm,
                        const bench_shape &  s,
                        const bench_config & cfg,
                        int &                rc) {
    const int64_t K          = s.K;
    const int64_t M          = s.M;
    const int64_t E          = cfg.n_experts;
    const int64_t top_k      = cfg.top_k;
    const int64_t n_max      = *std::max_element(cfg.widths.begin(), cfg.widths.end());
    const int64_t T_max      = *std::max_element(cfg.tokens.begin(), cfg.tokens.end());
    // The packed bound of moe_pp_onednn_packed_rows_at's first term for per-expert rounding: every routed row plus
    // align - 1 per active expert.
    const int64_t routed_max = T_max * top_k;
    const int64_t rows_max =
        routed_max + static_cast<int64_t>(ggml_sycl::moe_pp_gemm_row_align - 1) * std::min(E, routed_max);
    const int64_t slot_rows = std::max(E * n_max, rows_max);
    const int64_t src1_rows = s.rows_per_slot ? top_k : 1;

    shape_summary  summary;
    std::mt19937   rng(static_cast<unsigned>(4242 + K * 7 + M));
    device_buffers bufs(q);
    shape_buffers  sb;
    sb.w   = bufs.alloc<uint16_t>(static_cast<size_t>(E * M * K));
    sb.act = bufs.alloc<uint16_t>(static_cast<size_t>(slot_rows * K));
    sb.out = bufs.alloc<float>(static_cast<size_t>(slot_rows * M));
    for (int64_t e = 0; e < E; ++e) {
        fill_bf16(q, sb.w + e * M * K, static_cast<size_t>(M * K), SALT_W, static_cast<uint64_t>(e * M * K));
    }
    q.wait_and_throw();

    for (int64_t n : cfg.widths) {
        const call_result r = run_calls(q, eng, strm, bufs, s, cfg, sb, n, rng, rc);
        if (n == SUMMARY_N) {
            summary.at_summary_n = r;
        }
    }

    std::vector<const void *> table(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        table[static_cast<size_t>(e)] = sb.w + e * M * K;
    }
    auto * table_dev = bufs.alloc<const void *>(table.size());
    q.memcpy(table_dev, table.data(), table.size() * sizeof(const void *)).wait_and_throw();
    float *   src1_dev = bufs.alloc<float>(static_cast<size_t>(K * src1_rows * T_max));
    int32_t * ids_dev  = bufs.alloc<int32_t>(static_cast<size_t>(top_k * T_max));
    float *   dst_dev  = bufs.alloc<float>(static_cast<size_t>(M * top_k * T_max));
    for (int64_t T : cfg.tokens) {
        summary.nodes.push_back(
            run_node(q, eng, strm, bufs, s, cfg, sb, T, table_dev, src1_dev, ids_dev, dst_dev, rng, rc));
    }
    return summary;
}

}  // namespace
#endif  // GGML_SYCL_DNNL

int main() {
    // Each line is printed in parts around a call that can hang or abort; unbuffered, a killed run still shows which
    // measurement it was in.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#if GGML_SYCL_DNNL
    // A failure before setup_done is an absence (no device under the selector, no engine): exit 77. A failure after
    // it is a result: exit 1.
    bool         setup_done = false;
    bench_config cfg;
    cfg.smoke = env_or("BENCH_SMOKE", "") == "1";
    if (cfg.smoke) {
        cfg.n_experts = 8;
        cfg.top_k     = 2;
        cfg.widths    = { 16, 32 };
        cfg.tokens    = { 8, 16 };
        cfg.passes    = 1;
    } else {
        cfg.n_experts = 128;
        cfg.top_k     = 8;
        cfg.widths    = { 16, 32, 64, 128 };
        cfg.tokens    = { 16, 32, 64, 128, 256, 512 };
        cfg.passes    = std::max(1, std::atoi(env_or("BENCH_PASSES", "3").c_str()));
    }
    try {
        sycl::device  dev = cfg.smoke ? sycl::device{ sycl::cpu_selector_v } : sycl::device{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        sycl::queue   q{
            ctx, dev, { sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }
        };
        dnnl::engine eng  = dnnl::sycl_interop::make_engine(dev, ctx);
        dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
        std::printf(
            "bench=bf16_expert_gemm device=\"%s\" driver=%s dnnl=%d.%d.%d deterministic=%d smoke=%d passes=%d "
            "experts=%lld top_k=%lld\n",
            dev.get_info<sycl::info::device::name>().c_str(),
            dev.get_info<sycl::info::device::driver_version>().c_str(), DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR,
            DNNL_VERSION_PATCH, ggml_sycl_onednn_deterministic_enabled() ? 1 : 0, cfg.smoke ? 1 : 0, cfg.passes,
            (long long) cfg.n_experts, (long long) cfg.top_k);
        setup_done = true;

        int                        rc = 0;
        std::vector<shape_summary> summaries;
        if (cfg.smoke) {
            for (const bench_shape & s : SMOKE_SHAPES) {
                summaries.push_back(run_shape(q, eng, strm, s, cfg, rc));
            }
        } else {
            for (const bench_shape & s : SHAPES) {
                summaries.push_back(run_shape(q, eng, strm, s, cfg, rc));
            }
        }

        // n = 64 across the shapes; the smoke widths have no 64, so its summary reads -1.
        double span_sum = 0.0;
        double dev_sum  = 0.0;
        double host_max = -1.0;
        bool   have_n   = true;
        for (const shape_summary & s : summaries) {
            have_n = have_n && s.at_summary_n.ok;
            span_sum += s.at_summary_n.span_per_us;
            dev_sum += s.at_summary_n.dev_mean_us;
            host_max = std::max(host_max, s.at_summary_n.host_mean);
        }
        const double n_shapes  = static_cast<double>(summaries.size());
        const double span_mean = have_n ? span_sum / n_shapes : -1.0;
        const double dev_mean  = have_n ? dev_sum / n_shapes : -1.0;
        const char * verdict   = !have_n                                             ? "none" :
                                 span_mean > SPAN_STOP_US || host_max > HOST_STOP_US ? "STOP" :
                                 host_max > HOST_MARGINAL_US                         ? "MARGINAL" :
                                                                                       "PROCEED";

        // The smallest T from which the arm wins on every shape at every larger T measured.
        int64_t suggested = -1;
        for (size_t i = cfg.tokens.size(); i-- > 0;) {
            bool all = true;
            for (const shape_summary & s : summaries) {
                all = all && i < s.nodes.size() && s.nodes[i].wins;
            }
            if (!all) {
                break;
            }
            suggested = cfg.tokens[i];
        }
        std::printf("summary n64_span_us=%.1f n64_dev_us=%.1f n64_host_us=%.1f verdict=%s suggested_min_tokens=",
                    span_mean, dev_mean, have_n ? host_max : -1.0, verdict);
        if (suggested > 0) {
            std::printf("%lld\n", (long long) suggested);
        } else {
            std::printf("none\n");
        }
        std::puts(rc == 0 ? "PASS: bf16_expert_gemm.bench completed" :
                            "FAIL: bf16_expert_gemm.bench a primitive, a call or an output failed");
        return rc;
    } catch (const std::exception & e) {
        if (!setup_done) {
            std::printf("SKIP: bf16_expert_gemm.bench setup_failed reason=\"%s\"\n", e.what());
            return EXIT_SKIP;
        }
        std::printf("\nFAIL: bf16_expert_gemm.bench run_failed reason=\"%s\"\n", e.what());
        return 1;
    }
#else
    std::puts("SKIP: bf16_expert_gemm.bench GGML_SYCL_DNNL=0");
    return EXIT_SKIP;
#endif
}
