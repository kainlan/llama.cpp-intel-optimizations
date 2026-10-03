// Host-only microbench for the CpuExpertPool expert matvec at batch 1.
//
// `prod` drives the real ggml_sycl_cpu_expert_mul_mat_batched() -- the function
// a CpuExpertPool worker runs -- on expert-shaped weights. The other variants
// run over the SAME cold weight pool with an in-process thread team that
// schedules like bench-host-stream --sched dynamic (atomic row-chunk counter,
// persistent spinning workers), to separate "kernel" from "memory":
//
//   read        AVX2 loads over the weight bytes of each row, no decode: the
//               ceiling for this access pattern, data size and scheduling.
//   vecdot      ggml-cpu's own row vec_dot via the type traits, one row at a
//               time (what `prod` runs for every type without a hand-written
//               kernel -- Q8_0, IQ3_XXS, IQ4_NL, ...; for MXFP4/Q4_0 it is the
//               generic ggml kernel, i.e. what the hand-written 16-row VNNI
//               kernels in cpu-dispatch.cpp replace).
//   q8_4row     prototype: Q8_0 x Q8_0, 4 weight rows per activation block with
//               independent accumulators.
//   q8_4row_pf  q8_4row plus a software prefetch 512 B ahead per row.
//   r8          IQ4_NL / MXFP4 repacked 8 rows interleaved, ggml-cpu's exported
//               ggml_gemv_*_8x8_q8_0 kernel (the repack is done once, untimed).
//   mx8, mx16   MXFP4 8-row / 16-row VNNI kernels in the shape of the
//               cpu-dispatch.cpp hand kernel (16 spills the accumulators).
//   q2          Q2_0 prototype that keeps the weights packed and uses AVX-VNNI
//               on four activation planes; ggml-cpu has only a scalar kernel.
//
// --pin-tbb pins the library-created arena workers (CPUs 0..) from outside, to
// measure what production would gain from pinning CpuExpertPool's workers.
//
// The weight set is rotated so every call reads cold (beyond-LLC) data, as a
// decode token does. Opens no SYCL queue and touches no GPU. The pool's TBB
// arena size is fixed once per process by GGML_SYCL_CPU_THREADS; the own team
// uses the same count, so a thread sweep is one process per count
// (run-expert-sweep.py). Variants are interleaved burst by burst inside the
// process, in a fresh random order each burst, because ambient load on this
// host drifts by 2x between runs: only same-burst comparisons mean anything.
//
// Output: one CSV row per timed call.
//   shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us
// gbps = effective WEIGHT bytes / second (decimal GB).
//
// MIT license
// SPDX-License-Identifier: MIT

#include "cpu-dispatch.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <dirent.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ggml-cpu's AVX2 gemv kernels for the 8-row interleaved ("8x8") repack, exported
// from libggml-cpu. nr=1: one activation row, nc output rows (a multiple of 8).
extern "C" {
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}

// Q8_0 block layout (ggml-common.h block_q8_0): fp16 scale + 32 int8, 34 bytes.
struct __attribute__((packed)) bq8_0 {
    uint16_t d;
    int8_t   qs[32];
};

static_assert(sizeof(bq8_0) == 34, "block_q8_0 is 34 bytes");

struct bench_shape {
    const char * name;
    int          hidden;
    int          inter;
    int          top_k;
};

// Qwen3.8-Flash-Next: hidden 2560, MoE intermediate 640, 512 experts top-10.
// GPT-OSS 20B: hidden 2880, expert intermediate 2880, 32 experts top-4
// (gate/up [2880,2880], down [2880,2880], read from the GGUF tensor table).
static const bench_shape k_shapes[] = {
    { "qwen38", 2560, 640,  10 },
    { "gptoss", 2880, 2880, 4  },
};

// ---------------------------------------------------------------------------
// Thread team: persistent workers, spin then yield, one job at a time.
// ---------------------------------------------------------------------------
// One job: threads pull row chunks off `next` and add the rows they finished to
// `done`. The caller waits for rows, NOT for threads: a worker the scheduler has
// not run yet (this host is permanently oversubscribed) must not hold the call
// hostage, exactly as it cannot hold up the TBB master in the production path.
// State lives in a shared_ptr so a late worker finds `next` exhausted and leaves
// without touching anything the caller has already freed.
struct team_job {
    std::function<void(team_job &)> fn;
    std::atomic<int>                next{ 0 };
    std::atomic<int>                done{ 0 };
    int                             total = 0;
    std::atomic<float>              sink{ 0 };
};

class team {
  public:
    explicit team(int n) : n_(n) {
        for (int t = 1; t < n; t++) {
            th_.emplace_back([this]() { loop(); });
        }
    }

    ~team() {
        stop_.store(true);
        publish(nullptr);
        for (auto & t : th_) {
            t.join();
        }
    }

    void run(const std::shared_ptr<team_job> & job) {
        publish(job);
        job->fn(*job);
        wait_until([&] { return job->done.load(std::memory_order_acquire) >= job->total; });
    }

    int size() const { return n_; }

    std::vector<pid_t> tids() {
        std::lock_guard<std::mutex> lk(tid_mu_);
        return tids_;
    }

  private:
    void publish(std::shared_ptr<team_job> job) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            std::atomic_store(&job_, std::move(job));
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
    }

    template <class F> static void wait_until(F f) {
        int spins = 0;
        while (!f()) {
            if (++spins < 4000) {
                _mm_pause();
            } else {
                sched_yield();
            }
        }
    }

    // Workers spin briefly between back-to-back calls, then SLEEP on the condvar.
    // An idle yield-loop would steal CPU from the TBB workers of the `prod`
    // variant and bias the comparison against it.
    void loop() {
        {
            std::lock_guard<std::mutex> lk(tid_mu_);
            tids_.push_back((pid_t) syscall(SYS_gettid));
        }
        int seen = 0;
        for (;;) {
            for (int i = 0; i < 3000 && gen_.load(std::memory_order_acquire) == seen; i++) {
                _mm_pause();
            }
            if (gen_.load(std::memory_order_acquire) == seen) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_acquire) != seen; });
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_.load()) {
                return;
            }
            std::shared_ptr<team_job> j = std::atomic_load(&job_);
            if (j) {
                j->fn(*j);
            }
        }
    }

    int                       n_;
    std::vector<std::thread>  th_;
    std::atomic<int>          gen_{ 0 };
    std::atomic<bool>         stop_{ false };
    std::shared_ptr<team_job> job_;
    std::mutex                mu_;
    std::condition_variable   cv_;
    std::mutex                tid_mu_;
    std::vector<pid_t>        tids_;
};

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------
static inline float hsum8(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    r        = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r        = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

static inline float f16(uint16_t h) {
    return _cvtsh_ss(h);
}

static float read_row(const uint8_t * row, size_t bytes) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    size_t  i = 0;
    for (; i + 128 <= bytes; i += 128) {
        a0 = _mm256_add_epi32(a0, _mm256_loadu_si256((const __m256i *) (row + i)));
        a1 = _mm256_add_epi32(a1, _mm256_loadu_si256((const __m256i *) (row + i + 32)));
        a2 = _mm256_add_epi32(a2, _mm256_loadu_si256((const __m256i *) (row + i + 64)));
        a3 = _mm256_add_epi32(a3, _mm256_loadu_si256((const __m256i *) (row + i + 96)));
    }
    for (; i + 32 <= bytes; i += 32) {
        a0 = _mm256_add_epi32(a0, _mm256_loadu_si256((const __m256i *) (row + i)));
    }
    a0 = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
    alignas(32) int32_t t[8];
    _mm256_store_si256((__m256i *) t, a0);
    return (float) (t[0] ^ t[1] ^ t[2] ^ t[3] ^ t[4] ^ t[5] ^ t[6] ^ t[7]);
}

// 4 rows of Q8_0 against one Q8_0 activation. Rows are contiguous (row_stride).
template <int PF_DIST>
static void q8_0_4row(int K, const uint8_t * w, size_t row_stride, const bq8_0 * y, float * out) {
    const int     nb = K / 32;
    const bq8_0 * x0 = (const bq8_0 *) (w);
    const bq8_0 * x1 = (const bq8_0 *) (w + row_stride);
    const bq8_0 * x2 = (const bq8_0 *) (w + 2 * row_stride);
    const bq8_0 * x3 = (const bq8_0 *) (w + 3 * row_stride);
    __m256        a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int ib = 0; ib < nb; ib++) {
        if (PF_DIST > 0 && (ib & 1) == 0) {  // 2 blocks = 68 B ~ one line per row per iteration pair
            _mm_prefetch((const char *) x0 + (size_t) ib * 34 + PF_DIST, _MM_HINT_T0);
            _mm_prefetch((const char *) x1 + (size_t) ib * 34 + PF_DIST, _MM_HINT_T0);
            _mm_prefetch((const char *) x2 + (size_t) ib * 34 + PF_DIST, _MM_HINT_T0);
            _mm_prefetch((const char *) x3 + (size_t) ib * 34 + PF_DIST, _MM_HINT_T0);
        }
        const __m256i qy = _mm256_loadu_si256((const __m256i *) y[ib].qs);
        const float   dy = f16(y[ib].d);
#define ROW(xr, acc)                                                                                          \
    {                                                                                                         \
        const __m256i qx = _mm256_loadu_si256((const __m256i *) (xr)[ib].qs);                                 \
        const __m256i p  = _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy);                               \
        acc              = _mm256_fmadd_ps(_mm256_set1_ps(dy * f16((xr)[ib].d)), _mm256_cvtepi32_ps(p), acc); \
    }
        ROW(x0, a0) ROW(x1, a1) ROW(x2, a2) ROW(x3, a3)
#undef ROW
    }
    out[0] = hsum8(a0);
    out[1] = hsum8(a1);
    out[2] = hsum8(a2);
    out[3] = hsum8(a3);
}

// MXFP4 block (ggml-common.h block_mxfp4): e8m0 scale + 32 x 4-bit, 17 bytes.
struct __attribute__((packed)) bmx4 {
    uint8_t e;
    uint8_t qs[16];
};

static_assert(sizeof(bmx4) == 17, "block_mxfp4 is 17 bytes");

static inline float e8m0_half(uint8_t x) {  // GGML_E8M0_TO_FP32_HALF
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t) (x - 1) << 23);
    float          f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Copies of the MXFP4 x Q8_0 row-tile kernels in cpu-dispatch.cpp
// (simd_mxfp4_q8_0_8row / _16row), one activation block per step, so the
// register-pressure effect of 16 accumulators can be isolated. Differences:
// fp16 scale via F16C instead of the ggml table; no 2-block unroll.
template <int R> static void mxfp4_rows(int K, const uint8_t * w, size_t row_stride, const bq8_0 * y, float * out) {
    static const int8_t kv[16]    = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    const __m128i       values128 = _mm_loadu_si128((const __m128i *) kv);
    const __m128i       m4b       = _mm_set1_epi8(0x0f);
    const int           nb        = K / 32;
    const bmx4 *        x[R];
    __m256              acc[R];
    for (int r = 0; r < R; r++) {
        x[r]   = (const bmx4 *) (w + (size_t) r * row_stride);
        acc[r] = _mm256_setzero_ps();
    }
    for (int ib = 0; ib < nb; ib++) {
        const float   dy = f16(y[ib].d);
        const __m256i qy = _mm256_loadu_si256((const __m256i *) y[ib].qs);
        for (int r = 0; r < R; r++) {
            const __m128i q4 = _mm_loadu_si128((const __m128i *) x[r][ib].qs);
            const __m256i qx = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4, 4), m4b)),
                                                _mm_shuffle_epi8(values128, _mm_and_si128(q4, m4b)));
            const __m256i p  = _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dy * e8m0_half(x[r][ib].e)), _mm256_cvtepi32_ps(p), acc[r]);
        }
    }
    for (int r = 0; r < R; r++) {
        out[r] = hsum8(acc[r]);
    }
}

// ---------------------------------------------------------------------------
// Q2_0 (64 weights = fp16 d + 16 B of 2-bit values {0,1,2,3} -> {-1,0,1,2}, byte b
// holds weights 4b..4b+3). ggml-cpu has no x86 kernel for it: the scalar C
// reference runs. This prototype keeps the weights packed in memory (no repack):
//   activation, once per call set: for every Q2_0 block, split the 64 Q8_0
//   activation bytes into 4 planes P_m[b] = y[4b+m] (m=0..3, b=0..15), the
//   int32x4 lane sums of y (the -1 offset), and the per-lane Q8_0 scales;
//   weights, per block per row: one 16 B load, (q>>2m)&3 for m=0..3 as u8, four
//   u8 x s8 dot products (AVX-VNNI vpdpbusd), minus the y sum, scaled.
// ---------------------------------------------------------------------------
struct q2_act_block {
    int8_t  plane[4][16];
    int32_t ysum[4];   // lane l covers weights of bytes 4l..4l+3 across the four planes
    float   scale[4];  // Q8_0 scale of the half the lane belongs to
};

static void q2_prepare_act(int K, const bq8_0 * y, std::vector<q2_act_block> & out) {
    const int nb = K / 64;
    out.resize(nb);
    for (int i = 0; i < nb; i++) {
        q2_act_block & a = out[i];
        for (int b = 0; b < 16; b++) {
            const bq8_0 & yb = y[2 * i + b / 8];
            for (int m = 0; m < 4; m++) {
                a.plane[m][b] = yb.qs[4 * (b % 8) + m];
            }
        }
        for (int l = 0; l < 4; l++) {
            int sum = 0;
            for (int bb = 0; bb < 4; bb++) {
                for (int m = 0; m < 4; m++) {
                    sum += a.plane[m][4 * l + bb];
                }
            }
            a.ysum[l]  = sum;
            a.scale[l] = f16(y[2 * i + l / 2].d);
        }
    }
}

struct __attribute__((packed)) bq2_0 {
    uint16_t d;
    uint8_t  qs[16];
};

static_assert(sizeof(bq2_0) == 18, "block_q2_0 is 18 bytes");

static float q2_0_row(int K, const uint8_t * w, const q2_act_block * a) {
    const int     nb  = K / 64;
    const bq2_0 * x   = (const bq2_0 *) w;
    const __m128i m3  = _mm_set1_epi8(3);
    __m128        acc = _mm_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const __m128i q  = _mm_loadu_si128((const __m128i *) x[i].qs);
        const __m128i t0 = _mm_and_si128(q, m3);
        const __m128i t1 = _mm_and_si128(_mm_srli_epi16(q, 2), m3);
        const __m128i t2 = _mm_and_si128(_mm_srli_epi16(q, 4), m3);
        const __m128i t3 = _mm_and_si128(_mm_srli_epi16(q, 6), m3);
        __m128i v = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), t0, _mm_loadu_si128((const __m128i *) a[i].plane[0]));
        v         = _mm_dpbusd_avx_epi32(v, t1, _mm_loadu_si128((const __m128i *) a[i].plane[1]));
        v         = _mm_dpbusd_avx_epi32(v, t2, _mm_loadu_si128((const __m128i *) a[i].plane[2]));
        v         = _mm_dpbusd_avx_epi32(v, t3, _mm_loadu_si128((const __m128i *) a[i].plane[3]));
        v         = _mm_sub_epi32(v, _mm_loadu_si128((const __m128i *) a[i].ysum));
        acc = _mm_fmadd_ps(_mm_cvtepi32_ps(v), _mm_mul_ps(_mm_loadu_ps(a[i].scale), _mm_set1_ps(f16(x[i].d))), acc);
    }
    acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    acc = _mm_add_ss(acc, _mm_movehdup_ps(acc));
    return _mm_cvtss_f32(acc);
}

// ---------------------------------------------------------------------------
// Process CPU time (all threads). At 1 thread it is the kernel's own time with
// no wait for the scheduler.
static double cpu_us() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    size_t                   i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) {
            j = s.size();
        }
        if (j > i) {
            out.push_back(s.substr(i, j - i));
        }
        i = j + 1;
    }
    return out;
}

static ggml_type parse_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcasecmp(n, s.c_str()) == 0) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

// One expert matrix [N rows, K cols] quantized from random floats.
static bool quantize_expert(ggml_type type, int N, int K, std::vector<uint8_t> & out) {
    if (K % ggml_blck_size(type) != 0) {
        return false;
    }
    std::mt19937                    rng(1234 + (int) type);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              src((size_t) N * K);
    for (auto & v : src) {
        v = dist(rng);
    }
    ggml_quantize_init(type);
    out.resize((size_t) N * ggml_row_size(type, K));
    // A flat importance matrix: IQ2_XXS/IQ2_XS/IQ1_* refuse to quantize without
    // one, and the bytes read per row do not depend on it.
    std::vector<float> imatrix((size_t) K, 1.0f);
    const size_t       n = ggml_quantize_chunk(type, src.data(), out.data(), 0, N, K, imatrix.data());
    return n == out.size();
}

// Repack [N rows x nb blocks] MXFP4 (17 B: e8m0 + 16 qs) or IQ4_NL (18 B: fp16 d +
// 16 qs) into ggml-cpu's 8-row interleaved blocks, mirroring make_block_mxfp4x8 /
// make_block_iq4_nlx8: per 8 rows and block, 8 scales then 128 qs bytes where
// qs chunk i (8 B) comes from row i%8, bytes [(i/8)*8, +8). Group size equals
// 8 x the row size, so the byte layout of an expert keeps the same footprint.
static bool repack_8x8(ggml_type type, int N, int K, const std::vector<uint8_t> & in, std::vector<uint8_t> & out) {
    size_t scale_sz = 0, blk = 0;
    if (type == GGML_TYPE_MXFP4) {
        scale_sz = 1;
        blk      = 17;
    } else if (type == GGML_TYPE_IQ4_NL) {
        scale_sz = 2;
        blk      = 18;
    } else {
        return false;
    }
    if (N % 8) {
        return false;
    }
    const int nb = K / 32;
    out.assign(in.size(), 0);
    for (int g = 0; g < N / 8; g++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *       dst = out.data() + ((size_t) g * nb + b) * (8 * blk);
            const uint8_t * src[8];
            for (int r = 0; r < 8; r++) {
                src[r] = in.data() + ((size_t) (g * 8 + r) * nb + b) * blk;
            }
            for (int r = 0; r < 8; r++) {
                memcpy(dst + r * scale_sz, src[r], scale_sz);
            }
            uint8_t * qs = dst + 8 * scale_sz;
            for (int i = 0; i < 16; i++) {
                memcpy(qs + i * 8, src[i % 8] + scale_sz + (i / 8) * 8, 8);
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string types_arg    = "q8_0,mxfp4,iq3_xxs";
    std::string shapes_arg   = "qwen38,gptoss";
    std::string mats_arg     = "gate,down";
    std::string variants_arg = "prod,read,vecdot";
    int         bursts       = 8;    // interleaved rounds of every variant
    int         burst_calls  = 6;    // timed calls per variant per burst
    int         warmup       = 4;
    size_t      pool_mb      = 384;  // weight bytes per config, >> LLC
    int         top_k_ovr    = 0;
    int         chunk        = 16;   // rows per work item of the own team
    bool        thp          = false;
    bool        pin_tbb      = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto        v = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        if (a == "--types") {
            types_arg = v();
        } else if (a == "--shapes") {
            shapes_arg = v();
        } else if (a == "--mats") {
            mats_arg = v();
        } else if (a == "--variants") {
            variants_arg = v();
        } else if (a == "--bursts") {
            bursts = atoi(v().c_str());
        } else if (a == "--burst-calls") {
            burst_calls = atoi(v().c_str());
        } else if (a == "--warmup") {
            warmup = atoi(v().c_str());
        } else if (a == "--pool-mb") {
            pool_mb = (size_t) atoll(v().c_str());
        } else if (a == "--topk") {
            top_k_ovr = atoi(v().c_str());
        } else if (a == "--chunk") {
            chunk = atoi(v().c_str());
        } else if (a == "--thp") {
            thp = true;
        } else if (a == "--pin-tbb") {
            pin_tbb = true;
        } else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    ggml_cpu_init();  // fills the fp16->fp32 table the vec_dot kernels read (the backend does this in production)
    const char * thr_env = getenv("GGML_SYCL_CPU_THREADS");
    const int    threads = thr_env ? atoi(thr_env) : 22;  // library default: hw-2
    team         tm(threads);
    std::mt19937 order_rng(5);
    bool         pinned_done = false;

    printf("shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us\n");

    for (const auto & sname : split_csv(shapes_arg)) {
        const bench_shape * shp = nullptr;
        for (const auto & s : k_shapes) {
            if (sname == s.name) {
                shp = &s;
            }
        }
        if (!shp) {
            fprintf(stderr, "unknown shape %s\n", sname.c_str());
            return 2;
        }
        const int k = top_k_ovr > 0 ? top_k_ovr : shp->top_k;

        for (const auto & mat : split_csv(mats_arg)) {
            // gate/up: out=inter, in=hidden.  down: out=hidden, in=inter.
            const int N = mat == "down" ? shp->hidden : shp->inter;
            const int K = mat == "down" ? shp->inter : shp->hidden;

            for (const auto & tname : split_csv(types_arg)) {
                const ggml_type type = parse_type(tname);
                if (type == GGML_TYPE_COUNT) {
                    fprintf(stderr, "unknown type %s\n", tname.c_str());
                    return 2;
                }

                std::vector<uint8_t> proto;
                if (!quantize_expert(type, N, K, proto)) {
                    fprintf(stderr, "skip %s %s %s: K=%d not a multiple of block %d (or quantize failed)\n", shp->name,
                            mat.c_str(), tname.c_str(), K, (int) ggml_blck_size(type));
                    continue;
                }
                const size_t expert_bytes = proto.size();
                const size_t row_bytes    = ggml_row_size(type, K);
                // Pool of distinct expert slots; each call reads the next k slots,
                // so a slot is re-read only after the whole pool (> LLC) was streamed.
                size_t       n_pool       = std::max<size_t>((size_t) k * 4, (pool_mb << 20) / expert_bytes);
                n_pool                    = (n_pool / k) * k;

                const size_t pool_bytes = n_pool * expert_bytes;
                const size_t align      = thp ? (2u << 20) : 4096;
                uint8_t *    pool       = (uint8_t *) aligned_alloc(align, (pool_bytes + align - 1) / align * align);
                if (!pool) {
                    fprintf(stderr, "alloc failed\n");
                    return 1;
                }
                if (thp) {
                    madvise(pool, pool_bytes, MADV_HUGEPAGE);
                }
                for (size_t e = 0; e < n_pool; e++) {
                    memcpy(pool + e * expert_bytes, proto.data(), expert_bytes);
                }

                // Repacked twin pool, only when a repack variant is requested.
                uint8_t *            pool_r = nullptr;
                std::vector<uint8_t> proto_r;
                if (variants_arg.find("r8") != std::string::npos && repack_8x8(type, N, K, proto, proto_r)) {
                    pool_r = (uint8_t *) aligned_alloc(align, (pool_bytes + align - 1) / align * align);
                    if (thp) {
                        madvise(pool_r, pool_bytes, MADV_HUGEPAGE);
                    }
                    for (size_t e = 0; e < n_pool; e++) {
                        memcpy(pool_r + e * expert_bytes, proto_r.data(), expert_bytes);
                    }
                }

                // gate/up share one activation across experts; down has one per expert.
                std::vector<float>              act((size_t) k * K);
                std::mt19937                    rng(7);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                for (auto & v : act) {
                    v = dist(rng);
                }
                std::vector<float> outv((size_t) k * N);

                const auto *         tr  = ggml_get_type_traits_cpu(type);
                const auto *         vtr = tr && tr->vec_dot ? ggml_get_type_traits_cpu(tr->vec_dot_type) : nullptr;
                // Own-team variants take one pre-quantized activation (the shared
                // gate/up case); `prod` quantizes per call, as production does.
                std::vector<uint8_t> actq(vtr ? ggml_row_size(tr->vec_dot_type, K) : 0);
                if (vtr) {
                    vtr->from_float(act.data(), actq.data(), K);
                }

                std::vector<q2_act_block> q2act;
                if (type == GGML_TYPE_Q2_0) {
                    q2_prepare_act(K, (const bq8_0 *) actq.data(), q2act);
                }

                std::vector<std::string> variants;
                for (const auto & v : split_csv(variants_arg)) {
                    if (v.rfind("q8_4row", 0) == 0 && type != GGML_TYPE_Q8_0) {
                        continue;
                    }
                    if (v == "vecdot" && !vtr) {
                        continue;
                    }
                    if (v == "r8" && !pool_r) {
                        continue;
                    }
                    if ((v == "mx8" || v == "mx16") && type != GGML_TYPE_MXFP4) {
                        continue;
                    }
                    if (v == "q2" && type != GGML_TYPE_Q2_0) {
                        continue;
                    }
                    variants.push_back(v);
                }

                std::vector<cpu_expert_task> tasks(k);
                const int                    rows_total = k * N;
                size_t                       call_ctr   = 0;

                auto run_one = [&](const std::string & var, int burst, int call, bool record) {
                    const size_t base = (call_ctr++ * k) % n_pool;
                    const double t0   = now_us();
                    const double c0   = cpu_us();
                    if (var == "prod") {
                        for (int j = 0; j < k; j++) {
                            cpu_expert_task & t = tasks[j];
                            t.weight_host       = pool + (base + j) * expert_bytes;
                            t.act_host          = act.data() + (mat == "down" ? (size_t) j * K : 0);
                            t.output_host       = outv.data() + (size_t) j * N;
                            t.bias              = nullptr;
                            t.type              = type;
                            t.K                 = K;
                            t.N                 = N;
                        }
                        ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), k);
                    } else {
                        const int pf  = var == "q8_4row_pf" ? 512 : 0;
                        auto      job = std::make_shared<team_job>();
                        job->total    = rows_total;
                        job->fn       = [&, base, pf](team_job & J) {
                            float local = 0;
                            for (;;) {
                                const int r0 = J.next.fetch_add(chunk, std::memory_order_relaxed);
                                if (r0 >= rows_total) {
                                    break;
                                }
                                const int r1 = std::min(r0 + chunk, rows_total);
                                int       i  = r0;
                                while (i < r1) {
                                    const int       e = i / N, row = i % N;
                                    const uint8_t * w    = pool + (base + e) * expert_bytes + (size_t) row * row_bytes;
                                    float *         o    = outv.data() + i;
                                    const int       left = std::min(r1 - i, N - row);
                                    if (var == "read") {
                                        for (int j = 0; j < left; j++) {
                                            local += read_row(w + (size_t) j * row_bytes, row_bytes);
                                        }
                                    } else if (var == "r8") {
                                        const uint8_t * wr =
                                            pool_r + (base + e) * expert_bytes + (size_t) row * row_bytes;
                                        if (type == GGML_TYPE_MXFP4) {
                                            ggml_gemv_mxfp4_8x8_q8_0(K, o, 0, wr, actq.data(), 1, left);
                                        } else {
                                            ggml_gemv_iq4_nl_8x8_q8_0(K, o, 0, wr, actq.data(), 1, left);
                                        }
                                    } else if (var == "q2") {
                                        for (int j = 0; j < left; j++) {
                                            o[j] = q2_0_row(K, w + (size_t) j * row_bytes, q2act.data());
                                        }
                                    } else if (var == "mx8" || var == "mx16") {
                                        const int R = var == "mx16" ? 16 : 8;
                                        int       j = 0;
                                        for (; j + R <= left; j += R) {
                                            if (R == 16) {
                                                mxfp4_rows<16>(K, w + (size_t) j * row_bytes, row_bytes,
                                                                     (const bq8_0 *) actq.data(), o + j);
                                            } else {
                                                mxfp4_rows<8>(K, w + (size_t) j * row_bytes, row_bytes,
                                                                    (const bq8_0 *) actq.data(), o + j);
                                            }
                                        }
                                        for (; j < left; j++) {
                                            tr->vec_dot(K, o + j, sizeof(float), w + (size_t) j * row_bytes, 0,
                                                              actq.data(), 0, 1);
                                        }
                                    } else if (var == "vecdot") {
                                        for (int j = 0; j < left; j++) {
                                            tr->vec_dot(K, o + j, sizeof(float), w + (size_t) j * row_bytes, 0,
                                                              actq.data(), 0, 1);
                                        }
                                    } else {  // q8_4row, q8_4row_pf
                                        int j = 0;
                                        for (; j + 3 < left; j += 4) {
                                            if (pf) {
                                                q8_0_4row<512>(K, w + (size_t) j * row_bytes, row_bytes,
                                                                     (const bq8_0 *) actq.data(), o + j);
                                            } else {
                                                q8_0_4row<0>(K, w + (size_t) j * row_bytes, row_bytes,
                                                                   (const bq8_0 *) actq.data(), o + j);
                                            }
                                        }
                                        for (; j < left; j++) {
                                            tr->vec_dot(K, o + j, sizeof(float), w + (size_t) j * row_bytes, 0,
                                                              actq.data(), 0, 1);
                                        }
                                    }
                                    i += left;
                                }
                                J.done.fetch_add(r1 - r0, std::memory_order_release);
                            }
                            J.sink.store(J.sink.load(std::memory_order_relaxed) + local, std::memory_order_relaxed);
                        };
                        tm.run(job);
                    }
                    const double us  = now_us() - t0;
                    const double cus = cpu_us() - c0;
                    if (record) {
                        const double bytes = (double) k * expert_bytes;
                        printf("%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,%.1f,%.0f,%.3f,%.1f\n", shp->name, mat.c_str(),
                               ggml_type_name(type), N, K, k, threads, var.c_str(), burst, call, us, bytes,
                               bytes / (us * 1e3), cus);
                    }
                };

                // Numeric check of the own-team variants against ggml's reference
                // vec_dot on expert 0 (they must be the same dot products).
                if (vtr) {
                    std::vector<float> ref(N), got(N);
                    for (int r = 0; r < N; r++) {
                        tr->vec_dot(K, &ref[r], sizeof(float), pool + (size_t) r * row_bytes, 0, actq.data(), 0, 1);
                    }
                    for (const auto & var : variants) {
                        if (var != "r8" && var.rfind("q8_4row", 0) != 0 && var != "mx8" && var != "mx16" &&
                            var != "q2") {
                            continue;
                        }
                        if (var == "r8") {
                            if (type == GGML_TYPE_MXFP4) {
                                ggml_gemv_mxfp4_8x8_q8_0(K, got.data(), 0, pool_r, actq.data(), 1, N);
                            } else {
                                ggml_gemv_iq4_nl_8x8_q8_0(K, got.data(), 0, pool_r, actq.data(), 1, N);
                            }
                        } else if (var == "q2") {
                            for (int j = 0; j < N; j++) {
                                got[j] = q2_0_row(K, pool + (size_t) j * row_bytes, q2act.data());
                            }
                        } else if (var == "mx8" || var == "mx16") {
                            const int R = var == "mx16" ? 16 : 8;
                            for (int j = 0; j + R <= N; j += R) {
                                if (R == 16) {
                                    mxfp4_rows<16>(K, pool + (size_t) j * row_bytes, row_bytes,
                                                   (const bq8_0 *) actq.data(), &got[j]);
                                } else {
                                    mxfp4_rows<8>(K, pool + (size_t) j * row_bytes, row_bytes,
                                                  (const bq8_0 *) actq.data(), &got[j]);
                                }
                            }
                        } else {
                            for (int j = 0; j + 3 < N; j += 4) {
                                q8_0_4row<0>(K, pool + (size_t) j * row_bytes, row_bytes, (const bq8_0 *) actq.data(),
                                             &got[j]);
                            }
                        }
                        double maxd = 0, maxr = 0;
                        for (int r = 0; r < N; r++) {
                            maxd = std::max(maxd, (double) std::fabs(ref[r] - got[r]));
                            maxr = std::max(maxr, (double) std::fabs(ref[r]));
                        }
                        fprintf(stderr, "check %s %s %s %s: max|diff|=%.3g max|ref|=%.3g\n", shp->name, mat.c_str(),
                                tname.c_str(), var.c_str(), maxd, maxr);
                    }
                }
                for (const auto & var : variants) {
                    for (int w = 0; w < warmup; w++) {
                        run_one(var, -1, w, false);
                    }
                }
                if (pin_tbb && !pinned_done) {
                    // The arena's workers are created by the library on first use, so
                    // pin them from outside: this thread (the TBB master, a pool worker
                    // in production) to CPU 0, every other thread that is not one of our
                    // team's to CPUs 1.. in order (CPUs 0-7 are the P-cores).
                    pinned_done                  = true;
                    std::vector<pid_t> team_tids = tm.tids();
                    int                cpu       = 1;
                    cpu_set_t          set;
                    CPU_ZERO(&set);
                    CPU_SET(0, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                    if (DIR * d = opendir("/proc/self/task")) {
                        while (dirent * ent = readdir(d)) {
                            const pid_t tid = (pid_t) atoi(ent->d_name);
                            if (tid <= 0 || tid == (pid_t) syscall(SYS_gettid)) {
                                continue;
                            }
                            if (std::find(team_tids.begin(), team_tids.end(), tid) != team_tids.end()) {
                                continue;
                            }
                            CPU_ZERO(&set);
                            CPU_SET(cpu % 24, &set);
                            if (sched_setaffinity(tid, sizeof(set), &set) == 0) {
                                cpu++;
                            }
                        }
                        closedir(d);
                    }
                    fprintf(stderr, "pin-tbb: pinned %d non-team threads\n", cpu - 1);
                }
                for (int b = 0; b < bursts; b++) {
                    std::vector<std::string> order = variants;
                    std::shuffle(order.begin(), order.end(), order_rng);
                    for (const auto & var : order) {
                        for (int c = 0; c < burst_calls; c++) {
                            run_one(var, b, c, true);
                        }
                    }
                }
                fflush(stdout);
                free(pool);
                free(pool_r);
            }
        }
    }
    return 0;
}
