// Host-only microbench for the CpuExpertPool expert matvec at batch 1.
//
// `prod` drives the real ggml_sycl_cpu_expert_mul_mat_batched() -- the function
// a CpuExpertPool worker runs -- on expert-shaped weights. The other variants
// run over the SAME cold weight pool with an in-process thread team that
// schedules like bench-host-stream --sched dynamic (atomic row-chunk counter,
// workers that sleep when idle), to separate "kernel" from "memory":
//
//   read        AVX2 loads over every weight byte of each row, no decode: the
//               ceiling for this access pattern, data size and scheduling.
//   vecdot      ggml-cpu's own row vec_dot via the type traits, one row at a
//               time (what `prod` runs for every type without a hand-written
//               kernel -- Q8_0, IQ3_XXS, IQ4_NL, ...; for MXFP4/Q4_0 it is the
//               AVX2/VNNI-INT8 path of ggml_vec_dot_mxfp4_q8_0, i.e. the
//               one-row kernel the hand-written 16-row tiles in
//               cpu-dispatch.cpp replace).
//   q8_4row     prototype: Q8_0 x Q8_0, 4 weight rows per activation block with
//               independent accumulators.
//   q8_4row_pf  q8_4row plus a software prefetch 512 B ahead per row.
//   r8          IQ4_NL / MXFP4 repacked 8 rows interleaved, ggml-cpu's exported
//               ggml_gemv_*_8x8_q8_0 kernel (the repack is done once, untimed).
//   mx8, mx16   MXFP4 8-row / 16-row VNNI kernels in the shape of the
//               cpu-dispatch.cpp hand kernels, kept out of line so their
//               disassembly can be inspected. NOT the production kernel: no
//               2-block unroll, F16C instead of the table for the fp16 scale.
//   q2          Q2_0 prototype that keeps the weights packed and uses AVX-VNNI
//               on four activation planes; ggml-cpu has only a scalar kernel.
//               Single row per call: no row tiling.
//
// Pinning is a per-variant suffix, applied inside the process just before that
// variant's calls of a burst and undone for the unsuffixed variants, so a
// pinned and an unpinned arm are paired burst by burst:
//   +pin    main thread on CPU 0, worker i on CPU i+1 (CPUs 0..7 are the P-cores)
//   +pinE   the same, but only on the E-cores (CPUs 8..), a negative control
//   +pinS   a fixed random permutation of all CPUs, a "pinned but not P-first"
//           control
// For `prod` the workers are the arena threads the library created (found by CPU
// time consumed, not by tid order, so idle helper threads take no slot); for the
// other variants they are this team's threads. Every pinned `prod` burst
// samples the CPU each worker last ran on and prints the P/E split to stderr.
//
// What a `prod` call includes that the other variants do not: quantising the
// activation to the vec_dot type and the per-call activation-map lookup, a few
// microseconds against calls of 100 us to several ms.
//
// Before timing, every config is checked: `prod` and each producing variant is
// run once over NaN-filled outputs and compared row by row to ggml-cpu's own
// vec_dot on the same weights; a skipped row (NaN) or a mismatch FAILS the run
// (exit 3) rather than printing a plausibly high GB/s. Other non-zero exits:
// 2 bad arguments, 4 an --oracle-selftest corruption the oracle failed to reject
// (or nothing to exercise), 5 threads > 1 but no arena worker ran (library built
// without TBB, so `prod` was serial), 6 a sched_setaffinity call failed (the
// +pin arms did not run as labelled), 77 the CPU lacks an ISA the kernels need.
// The pool's expert slots hold identical bytes, so a wrong-slot read is not
// detected, only a missing or wrong row.
//
// The weight set is rotated so every call reads cold (beyond-LLC) data, as a
// decode token does. Opens no SYCL queue and touches no GPU. The pool's TBB
// arena size is fixed once per process by GGML_SYCL_CPU_THREADS; the own team
// uses the same count, so a thread sweep is one process per count
// (run_expert_sweep.py). Variants are interleaved burst by burst inside the
// process, in a fresh random order each burst, because ambient load on this
// host drifts by 2x between runs: only same-burst comparisons mean anything.
//
// Output: one CSV row per timed call.
//   shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us
// gbps = effective WEIGHT bytes / second (decimal GB).
//
// Needs AVX2, FMA, F16C, AVX-VNNI and AVX-VNNI-INT8 (it is compiled with them);
// without them it exits 77 instead of faulting on the first VNNI instruction.
//
// MIT license
// SPDX-License-Identifier: MIT

#include "cpu-dispatch.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cpuid.h>
#include <dirent.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

#define NOINLINE __attribute__((noinline))

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
// CPU feature guard: the kernels below use these unconditionally.
// ---------------------------------------------------------------------------
static bool host_has_required_isa() {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return false;
    }
    const bool fma = (c >> 12) & 1, f16c = (c >> 29) & 1;
    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }
    __cpuid_count(7, 0, a, b, c, d);
    const bool avx2 = (b >> 5) & 1;
    __cpuid_count(7, 1, a, b, c, d);
    const bool avxvnni = (a >> 4) & 1, avxvnniint8 = (d >> 4) & 1;
    return fma && f16c && avx2 && avxvnni && avxvnniint8;
}

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

// Touches every byte of the row, including the sub-32-byte tail some row sizes
// have (an MXFP4 row of K=2560 is 1360 B = 42.5 x 32), so the bytes it counts
// are the bytes it reads.
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
    if (i + 16 <= bytes) {
        a0 = _mm256_add_epi32(a0, _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *) (row + i))));
        i += 16;
    }
    uint32_t tail = 0;
    for (; i < bytes; i++) {
        tail ^= (uint32_t) row[i] << ((i & 3) * 8);
    }
    alignas(32) int32_t t[8];
    _mm256_store_si256((__m256i *) t, a0);
    return (float) (t[0] ^ t[1] ^ t[2] ^ t[3] ^ t[4] ^ t[5] ^ t[6] ^ t[7] ^ (int32_t) tail);
}

// 4 rows of Q8_0 against one Q8_0 activation. Rows are contiguous (row_stride).
template <int PF_DIST>
static NOINLINE void q8_0_4row(int K, const uint8_t * w, size_t row_stride, const bq8_0 * y, float * out) {
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

// Row-tile kernels in the shape of simd_mxfp4_q8_0_8row / _16row in
// cpu-dispatch.cpp, one activation block per step. Not byte-for-byte the
// production code (see the header comment). Kept out of line.
template <int R>
static NOINLINE void mxfp4_rows(int K, const uint8_t * w, size_t row_stride, const bq8_0 * y, float * out) {
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

// Q2_0 (64 weights = fp16 d + 16 B of 2-bit values {0,1,2,3} -> {-1,0,1,2}, byte b
// holds weights 4b..4b+3). ggml-cpu has no x86 kernel for it: the scalar C
// reference runs. This prototype keeps the weights packed in memory (no repack):
//   activation, once per call set: for every Q2_0 block, split the 64 Q8_0
//   activation bytes into 4 planes P_m[b] = y[4b+m] (m=0..3, b=0..15), the
//   int32x4 lane sums of y (the -1 offset), and the per-lane Q8_0 scales;
//   weights, per block per row: one 16 B load, (q>>2m)&3 for m=0..3 as u8, four
//   u8 x s8 dot products (AVX-VNNI vpdpbusd), minus the y sum, scaled.
// Per 64 weights and ONE row: 1 weight load, 3 shifts, 4 ands, 4 vpdpbusd, 4
// plane loads + 1 ysum + 1 scale load; the six activation loads are the same
// for every row, so a multi-row tile would amortise them. This prototype has
// no such tile.
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

static NOINLINE float q2_0_row(int K, const uint8_t * w, const q2_act_block * a) {
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Affinity and per-thread CPU accounting (/proc/self/task).
// ---------------------------------------------------------------------------
enum pin_mode { PIN_NONE, PIN_SEQ, PIN_E, PIN_SHUF };

static const char * pin_suffix(pin_mode m) {
    return m == PIN_SEQ ? "+pin" : m == PIN_E ? "+pinE" : m == PIN_SHUF ? "+pinS" : "";
}

// The affinity mask the process started with, captured once in main(): the CPUs
// a pin may use, and the mask PIN_NONE restores (cgroups / taskset / isolcpus
// can make CPUs 0..n-1 not all allowed).
static cpu_set_t g_start_mask;
static bool      g_have_start_mask = false;
static long      g_pin_calls       = 0;
static long      g_pin_failures    = 0;

static bool cpu_allowed(int c) {
    return !g_have_start_mask || CPU_ISSET(c, &g_start_mask);
}

static std::vector<int> cpu_order(pin_mode m, int ncpu, int pcores) {
    std::vector<int> o;
    for (int c = (m == PIN_E ? pcores : 0); c < ncpu; c++) {
        if (cpu_allowed(c)) {
            o.push_back(c);
        }
    }
    if (m == PIN_SHUF) {
        std::mt19937 r(42);
        std::shuffle(o.begin(), o.end(), r);
    }
    return o;
}

// Every affinity change is counted; the totals are reported at exit, and the
// first failure is printed where it happens.
static void set_mask(pid_t tid, const cpu_set_t & s, const char * what) {
    g_pin_calls++;
    if (sched_setaffinity(tid, sizeof(s), &s) != 0) {
        if (g_pin_failures++ == 0) {
            fprintf(stderr, "warning: sched_setaffinity(tid %d, %s) failed: %s (further failures are only counted)\n",
                    (int) tid, what, strerror(errno));
        }
    }
}

static void set_cpu(pid_t tid, int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    set_mask(tid, s, "one CPU");
}

static void restore_start_mask(pid_t tid) {
    if (g_have_start_mask) {
        set_mask(tid, g_start_mask, "start mask");
    }
}

// Main thread to slot 0 of the order, worker i (tid-sorted) to slot i+1.
static void apply_pin(pid_t main_tid, std::vector<pid_t> workers, pin_mode m, int ncpu, int pcores) {
    std::sort(workers.begin(), workers.end());
    if (m == PIN_NONE) {
        restore_start_mask(main_tid);
        for (pid_t t : workers) {
            restore_start_mask(t);
        }
        return;
    }
    const std::vector<int> order = cpu_order(m, ncpu, pcores);
    if (order.empty()) {
        g_pin_failures++;  // no allowed CPU for this mode (e.g. E-only on a mask without E-cores)
        fprintf(stderr, "warning: pin mode %s has no allowed CPU; left unpinned\n", pin_suffix(m));
        return;
    }
    set_cpu(main_tid, order[0]);
    for (size_t i = 0; i < workers.size(); i++) {
        set_cpu(workers[i], order[(i + 1) % order.size()]);
    }
}

struct thread_stat {
    pid_t tid = 0;
    int   cpu = -1;  // field 39, CPU the thread last ran on
};

static bool read_thread_stat(pid_t tid, thread_stat & out) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/task/%d/stat", (int) tid);
    FILE * f = fopen(path, "r");
    if (!f) {
        return false;
    }
    char         buf[1024];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n]         = 0;
    const char * p = strrchr(buf, ')');  // comm may contain spaces
    if (!p) {
        return false;
    }
    // After ')' the first token is field 3 (state).
    int cpu = -1;
    int fld = 3;
    for (p += 1; *p; fld++) {
        while (*p == ' ') {
            p++;
        }
        if (fld == 39) {
            cpu = atoi(p);
            break;
        }
        while (*p && *p != ' ') {
            p++;
        }
    }
    out.tid = tid;
    out.cpu = cpu;
    return true;
}

// Nanoseconds the thread has spent on a CPU (/proc/self/task/<tid>/schedstat),
// which, unlike the 10 ms ticks of the stat file, resolves a few sub-millisecond calls.
static unsigned long long read_oncpu_ns(pid_t tid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/task/%d/schedstat", (int) tid);
    FILE * f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    unsigned long long ns = 0;
    if (fscanf(f, "%llu", &ns) != 1) {
        ns = 0;
    }
    fclose(f);
    return ns;
}

static std::vector<pid_t> list_tids() {
    std::vector<pid_t> v;
    if (DIR * d = opendir("/proc/self/task")) {
        while (dirent * ent = readdir(d)) {
            const pid_t t = (pid_t) atoi(ent->d_name);
            if (t > 0) {
                v.push_back(t);
            }
        }
        closedir(d);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Thread team: persistent workers, one job at a time.
// ---------------------------------------------------------------------------
// Everything a job reads is copied into it. The caller waits for rows, NOT for
// threads: a worker the scheduler has not run yet (this host is permanently
// oversubscribed) must not hold the call hostage, exactly as it cannot hold up
// the TBB master in the production path. Job state lives in a shared_ptr, so a
// late worker finds `next` exhausted and leaves without dereferencing the pool
// or the output the caller may already have freed.
struct job_ctx {
    const uint8_t *              pool         = nullptr;
    const uint8_t *              pool_r       = nullptr;  // repacked twin, r8 only
    float *                      out          = nullptr;
    const uint8_t *              actq         = nullptr;  // activation quantized to the vec_dot type
    const q2_act_block *         q2act        = nullptr;
    const ggml_type_traits_cpu * tr           = nullptr;
    size_t                       expert_bytes = 0;
    size_t                       row_bytes    = 0;
    size_t                       base         = 0;  // first expert slot of this call
    int                          N            = 0;
    int                          K            = 0;
    int                          chunk        = 16;
    int                          rows_total   = 0;
};

typedef float (*row_kernel)(const job_ctx & c, int e, int row, int left, float * o);

struct team_job {
    job_ctx            ctx;
    row_kernel         kernel = nullptr;
    std::atomic<int>   next{ 0 };
    std::atomic<int>   done{ 0 };
    std::atomic<float> sink{ 0 };
};

static void run_rows(team_job & J) {
    const job_ctx & c     = J.ctx;
    float           local = 0;
    for (;;) {
        const int r0 = J.next.fetch_add(c.chunk, std::memory_order_relaxed);
        if (r0 >= c.rows_total) {
            break;
        }
        const int r1 = std::min(r0 + c.chunk, c.rows_total);
        int       i  = r0;
        while (i < r1) {
            const int e = i / c.N, row = i % c.N;
            const int left = std::min(r1 - i, c.N - row);
            local += J.kernel(c, e, row, left, c.out + i);
            i += left;
        }
        J.done.fetch_add(r1 - r0, std::memory_order_release);
    }
    J.sink.store(J.sink.load(std::memory_order_relaxed) + local, std::memory_order_relaxed);
}

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
        run_rows(*job);
        const int total = job->ctx.rows_total;
        int       spins = 0;
        while (job->done.load(std::memory_order_acquire) < total) {
            if (++spins < 4000) {
                _mm_pause();
            } else {
                sched_yield();
            }
        }
    }

    // Tids of the worker threads (not the calling thread), once all have started.
    std::vector<pid_t> tids() {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(tid_mu_);
                if ((int) tids_.size() == n_ - 1) {
                    return tids_;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
                run_rows(*j);
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
// Row kernels of the own-team variants, one function each (a table, not a
// string compare in the timed loop).
// ---------------------------------------------------------------------------
static inline const uint8_t * row_ptr(const job_ctx & c, int e, int row) {
    return c.pool + (c.base + e) * c.expert_bytes + (size_t) row * c.row_bytes;
}

static float kern_read(const job_ctx & c, int e, int row, int left, float *) {
    const uint8_t * w = row_ptr(c, e, row);
    float           s = 0;
    for (int j = 0; j < left; j++) {
        s += read_row(w + (size_t) j * c.row_bytes, c.row_bytes);
    }
    return s;
}

static float kern_vecdot(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * w = row_ptr(c, e, row);
    for (int j = 0; j < left; j++) {
        c.tr->vec_dot(c.K, o + j, sizeof(float), w + (size_t) j * c.row_bytes, 0, c.actq, 0, 1);
    }
    return 0;
}

static float kern_r8_mxfp4(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * wr = c.pool_r + (c.base + e) * c.expert_bytes + (size_t) row * c.row_bytes;
    ggml_gemv_mxfp4_8x8_q8_0(c.K, o, 0, wr, c.actq, 1, left);
    return 0;
}

static float kern_r8_iq4_nl(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * wr = c.pool_r + (c.base + e) * c.expert_bytes + (size_t) row * c.row_bytes;
    ggml_gemv_iq4_nl_8x8_q8_0(c.K, o, 0, wr, c.actq, 1, left);
    return 0;
}

static float kern_q2(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * w = row_ptr(c, e, row);
    for (int j = 0; j < left; j++) {
        o[j] = q2_0_row(c.K, w + (size_t) j * c.row_bytes, c.q2act);
    }
    return 0;
}

template <int R> static float kern_mx(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * w = row_ptr(c, e, row);
    int             j = 0;
    for (; j + R <= left; j += R) {
        mxfp4_rows<R>(c.K, w + (size_t) j * c.row_bytes, c.row_bytes, (const bq8_0 *) c.actq, o + j);
    }
    for (; j < left; j++) {
        c.tr->vec_dot(c.K, o + j, sizeof(float), w + (size_t) j * c.row_bytes, 0, c.actq, 0, 1);
    }
    return 0;
}

template <int PF> static float kern_q84(const job_ctx & c, int e, int row, int left, float * o) {
    const uint8_t * w = row_ptr(c, e, row);
    int             j = 0;
    for (; j + 3 < left; j += 4) {
        q8_0_4row<PF>(c.K, w + (size_t) j * c.row_bytes, c.row_bytes, (const bq8_0 *) c.actq, o + j);
    }
    for (; j < left; j++) {
        c.tr->vec_dot(c.K, o + j, sizeof(float), w + (size_t) j * c.row_bytes, 0, c.actq, 0, 1);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// One benchmark configuration: shape x matrix x type.
// ---------------------------------------------------------------------------
struct config {
    const bench_shape *          shp = nullptr;
    std::string                  mat;
    std::string                  tname;
    ggml_type                    type = GGML_TYPE_COUNT;
    int                          N = 0, K = 0, k = 0;
    size_t                       expert_bytes = 0, row_bytes = 0, n_pool = 0;
    uint8_t *                    pool   = nullptr;
    uint8_t *                    pool_r = nullptr;
    std::vector<uint8_t>         proto;  // one expert's bytes (every pool slot is a copy)
    std::vector<float>           act;    // k activation rows (gate/up use row 0 only)
    std::vector<float>           outv;
    std::vector<uint8_t>         actq;   // act row 0 quantized to the vec_dot type
    std::vector<q2_act_block>    q2act;
    const ggml_type_traits_cpu * tr       = nullptr;
    const ggml_type_traits_cpu * vtr      = nullptr;
    size_t                       call_ctr = 0;

    ~config() {
        free(pool);
        free(pool_r);
    }
};

struct variant_def {
    const char * name;
    row_kernel   kernel;  // nullptr for prod
    bool         team;
    bool         produces_output;
    bool (*applies)(const config &);
};

static bool ap_always(const config &) {
    return true;
}

static bool ap_vecdot(const config & c) {
    return c.vtr != nullptr;
}

static bool ap_q8(const config & c) {
    return c.type == GGML_TYPE_Q8_0;
}

static bool ap_mxfp4(const config & c) {
    return c.type == GGML_TYPE_MXFP4;
}

static bool ap_q2(const config & c) {
    return c.type == GGML_TYPE_Q2_0;
}

static bool ap_r8(const config & c) {
    return c.pool_r != nullptr;
}

static const variant_def k_variants[] = {
    { "prod",       nullptr,       false, true,  ap_always },
    { "read",       kern_read,     true,  false, ap_always },
    { "vecdot",     kern_vecdot,   true,  true,  ap_vecdot },
    { "q8_4row",    kern_q84<0>,   true,  true,  ap_q8     },
    { "q8_4row_pf", kern_q84<512>, true,  true,  ap_q8     },
    { "mx8",        kern_mx<8>,    true,  true,  ap_mxfp4  },
    { "mx16",       kern_mx<16>,   true,  true,  ap_mxfp4  },
    { "q2",         kern_q2,       true,  true,  ap_q2     },
    // r8 picks its kernel from the type below; the table entry is a placeholder.
    { "r8",         nullptr,       true,  true,  ap_r8     },
};

struct variant_inst {
    std::string         name;  // as printed, e.g. "prod+pin"
    const variant_def * def    = nullptr;
    pin_mode            pin    = PIN_NONE;
    row_kernel          kernel = nullptr;
};

static bool parse_variant(const std::string & s, variant_inst & out) {
    std::string base = s;
    out.pin          = PIN_NONE;
    const size_t p   = s.find('+');
    if (p != std::string::npos) {
        base                = s.substr(0, p);
        const std::string m = s.substr(p + 1);
        out.pin             = m == "pin" ? PIN_SEQ : m == "pinE" ? PIN_E : m == "pinS" ? PIN_SHUF : PIN_NONE;
        if (out.pin == PIN_NONE) {
            return false;
        }
    }
    for (const auto & d : k_variants) {
        if (base == d.name) {
            out.def  = &d;
            out.name = s;
            return true;
        }
    }
    return false;
}

struct runner {
    team &                                       tm;
    int                                          threads;
    int                                          chunk;
    int                                          ncpu;
    int                                          pcores;
    pid_t                                        main_tid;
    std::vector<pid_t>                           team_tids;
    std::vector<pid_t>                           tbb_tids;     // arena workers of `prod`, found by CPU time
    std::vector<cpu_expert_task>                 tasks;
    std::map<std::string, std::pair<long, long>> cpu_samples;  // variant -> {on P-core, on E-core}
};

static void make_prod_tasks(config & cfg, size_t base, std::vector<cpu_expert_task> & tasks) {
    tasks.resize(cfg.k);
    for (int j = 0; j < cfg.k; j++) {
        cpu_expert_task & t = tasks[j];
        t.weight_host       = cfg.pool + (base + j) * cfg.expert_bytes;
        t.act_host          = cfg.act.data() + (cfg.mat == "down" ? (size_t) j * cfg.K : 0);
        t.output_host       = cfg.outv.data() + (size_t) j * cfg.N;
        t.bias              = nullptr;
        t.type              = cfg.type;
        t.K                 = cfg.K;
        t.N                 = cfg.N;
    }
}

static void run_call(runner & R, config & cfg, const variant_inst & v) {
    const size_t base = (cfg.call_ctr++ * cfg.k) % cfg.n_pool;
    if (!v.def->kernel && !v.kernel) {  // prod
        make_prod_tasks(cfg, base, R.tasks);
        ggml_sycl_cpu_expert_mul_mat_batched(R.tasks.data(), cfg.k);
        return;
    }
    auto job       = std::make_shared<team_job>();
    job->kernel    = v.kernel ? v.kernel : v.def->kernel;
    job_ctx & c    = job->ctx;
    c.pool         = cfg.pool;
    c.pool_r       = cfg.pool_r;
    c.out          = cfg.outv.data();
    c.actq         = cfg.actq.data();
    c.q2act        = cfg.q2act.data();
    c.tr           = cfg.tr;
    c.expert_bytes = cfg.expert_bytes;
    c.row_bytes    = cfg.row_bytes;
    c.base         = base;
    c.N            = cfg.N;
    c.K            = cfg.K;
    c.chunk        = R.chunk;
    c.rows_total   = cfg.k * cfg.N;
    R.tm.run(job);
}

// Find the arena threads that did work during `calls` prod calls: not the main
// thread, not the team, CPU time advanced. Idle helper threads take no pin slot.
static void discover_tbb_workers(runner & R, config & cfg, const variant_inst & prod, int calls) {
    std::map<pid_t, unsigned long long> before;
    for (pid_t t : list_tids()) {
        before[t] = read_oncpu_ns(t);
    }
    for (int i = 0; i < calls; i++) {
        run_call(R, cfg, prod);
    }
    R.tbb_tids.clear();
    int idle_other = 0;
    for (pid_t t : list_tids()) {
        if (t == R.main_tid || std::find(R.team_tids.begin(), R.team_tids.end(), t) != R.team_tids.end()) {
            continue;
        }
        if (read_oncpu_ns(t) > before[t]) {
            R.tbb_tids.push_back(t);
        } else {
            idle_other++;
        }
    }
    fprintf(stderr, "pin: %s %s %s threads=%d: %d active arena workers found, %d other threads idle (left unpinned)\n",
            cfg.shp->name, cfg.mat.c_str(), cfg.tname.c_str(), R.threads, (int) R.tbb_tids.size(), idle_other);
}

static bool has_prod(const std::vector<variant_inst> & variants) {
    for (const auto & v : variants) {
        if (!v.def->kernel && !v.kernel) {
            return true;
        }
    }
    return false;
}

static void apply_variant_pin(runner & R, const variant_inst & v) {
    const bool is_prod = !v.def->kernel && !v.kernel;
    apply_pin(R.main_tid, is_prod ? R.tbb_tids : R.team_tids, v.pin, R.ncpu, R.pcores);
}

static void sample_cpus(runner & R, const variant_inst & v) {
    if (v.pin == PIN_NONE) {
        return;
    }
    const bool         is_prod = !v.def->kernel && !v.kernel;
    std::vector<pid_t> ts      = is_prod ? R.tbb_tids : R.team_tids;
    ts.push_back(R.main_tid);
    auto & s = R.cpu_samples[v.name];
    for (pid_t t : ts) {
        thread_stat st;
        if (read_thread_stat(t, st) && st.cpu >= 0) {
            (st.cpu < R.pcores ? s.first : s.second)++;
        }
    }
}

// Reference: ggml-cpu's vec_dot of every row of `proto` against the activation
// row quantized to the vec_dot type.
static void reference_rows(const config & cfg, const float * act_row, std::vector<float> & ref) {
    std::vector<uint8_t> q(ggml_row_size(cfg.tr->vec_dot_type, cfg.K));
    cfg.vtr->from_float(act_row, q.data(), cfg.K);
    ref.resize(cfg.N);
    for (int r = 0; r < cfg.N; r++) {
        cfg.tr->vec_dot(cfg.K, &ref[r], sizeof(float), cfg.proto.data() + (size_t) r * cfg.row_bytes, 0, q.data(), 0,
                        1);
    }
}

static bool compare_block(const char *               what,
                          const config &             cfg,
                          const float *              got,
                          const std::vector<float> & ref,
                          int                        expert,
                          bool                       expected_fail) {
    double maxr = 0, maxd = 0;
    int    bad = 0;
    for (int r = 0; r < cfg.N; r++) {
        maxr = std::max(maxr, (double) std::fabs(ref[r]));
    }
    const double tol = 1e-3 * maxr + 1e-6;
    for (int r = 0; r < cfg.N; r++) {
        const double d = std::fabs((double) got[r] - ref[r]);
        if (!(d <= tol)) {  // also true for NaN: an unwritten row fails
            bad++;
        }
        if (d == d) {
            maxd = std::max(maxd, d);
        }
    }
    if (bad) {
        fprintf(stderr, "%s %s %s %s %s expert %d: %d of %d rows differ from vec_dot (max|diff|=%.3g tol=%.3g)\n",
                expected_fail ? "selftest-corrupted" : "FAIL", cfg.shp->name, cfg.mat.c_str(), cfg.tname.c_str(), what,
                expert, bad, cfg.N, maxd, tol);
    }
    return bad == 0;
}

// Runs `v` once over NaN-filled outputs and compares every expert's rows to the
// reference. Returns false on any mismatch or unwritten row.
static bool oracle_check(runner & R, config & cfg, const variant_inst & v, bool corrupt = false) {
    if (!v.def->produces_output) {
        return true;
    }
    std::fill(cfg.outv.begin(), cfg.outv.end(), std::nanf(""));
    cfg.call_ctr = 0;
    run_call(R, cfg, v);
    if (corrupt) {  // --oracle-selftest: a skipped row and a wrong row must both be caught
        cfg.outv[5] = std::nanf("");
        cfg.outv[(size_t) cfg.N * (cfg.k - 1) + 7] += 1.0f;
    }
    bool               ok = true;
    std::vector<float> ref;
    const bool         is_prod = !v.def->kernel && !v.kernel;
    if (!is_prod) {
        reference_rows(cfg, cfg.act.data(), ref);  // team variants share the activation of row 0
    }
    for (int e = 0; e < cfg.k; e++) {
        if (is_prod) {  // down has one activation per expert, gate/up share row 0
            reference_rows(cfg, cfg.act.data() + (cfg.mat == "down" ? (size_t) e * cfg.K : 0), ref);
        }
        ok = compare_block(v.name.c_str(), cfg, cfg.outv.data() + (size_t) e * cfg.N, ref, e, corrupt) && ok;
    }
    cfg.call_ctr = 0;
    fprintf(stderr, "oracle %s %s %s %s: %s\n", cfg.shp->name, cfg.mat.c_str(), cfg.tname.c_str(), v.name.c_str(),
            ok ? "ok" : (corrupt ? "rejected the corrupted run" : "FAIL"));
    return ok;
}

struct options {
    std::string types_arg       = "q8_0,mxfp4,iq3_xxs";
    std::string shapes_arg      = "qwen38,gptoss";
    std::string mats_arg        = "gate,down";
    std::string variants_arg    = "prod,read,vecdot";
    int         bursts          = 8;    // interleaved rounds of every variant
    int         burst_calls     = 6;    // timed calls per variant per burst
    int         warmup          = 4;
    size_t      pool_mb         = 384;  // weight bytes per config, >> LLC
    int         top_k_ovr       = 0;
    int         chunk           = 16;   // rows per work item of the own team (a multiple of 8, for r8)
    int         pcores          = 8;    // CPUs 0..pcores-1 are P-cores
    bool        thp             = false;
    bool        oracle_only     = false;
    bool        oracle_selftest = false;  // corrupt one run's outputs and require the oracle to FAIL it
};

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto              v = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        if (a == "--types") {
            o.types_arg = v();
        } else if (a == "--shapes") {
            o.shapes_arg = v();
        } else if (a == "--mats") {
            o.mats_arg = v();
        } else if (a == "--variants") {
            o.variants_arg = v();
        } else if (a == "--bursts") {
            o.bursts = atoi(v().c_str());
        } else if (a == "--burst-calls") {
            o.burst_calls = atoi(v().c_str());
        } else if (a == "--warmup") {
            o.warmup = atoi(v().c_str());
        } else if (a == "--pool-mb") {
            o.pool_mb = (size_t) atoll(v().c_str());
        } else if (a == "--topk") {
            o.top_k_ovr = atoi(v().c_str());
        } else if (a == "--chunk") {
            o.chunk = atoi(v().c_str());
        } else if (a == "--pcores") {
            o.pcores = atoi(v().c_str());
        } else if (a == "--thp") {
            o.thp = true;
        } else if (a == "--oracle-only") {
            o.oracle_only = true;
        } else if (a == "--oracle-selftest") {
            o.oracle_selftest = true;
        } else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return false;
        }
    }
    if (o.chunk < 8 || o.chunk % 8) {
        fprintf(stderr, "--chunk must be a multiple of 8\n");
        return false;
    }
    return true;
}

// Builds the weight pool(s), activation and outputs for one config. Returns
// false (with a note) when the type cannot run here.
static bool build_config(const options &     o,
                         const bench_shape * shp,
                         const std::string & mat,
                         const std::string & tname,
                         config &            cfg) {
    cfg.shp   = shp;
    cfg.mat   = mat;
    cfg.tname = tname;
    cfg.type  = parse_type(tname);
    if (cfg.type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type %s\n", tname.c_str());
        return false;
    }
    cfg.k   = o.top_k_ovr > 0 ? o.top_k_ovr : shp->top_k;
    // gate/up: out=inter, in=hidden.  down: out=hidden, in=inter.
    cfg.N   = mat == "down" ? shp->hidden : shp->inter;
    cfg.K   = mat == "down" ? shp->inter : shp->hidden;
    cfg.tr  = ggml_get_type_traits_cpu(cfg.type);
    cfg.vtr = cfg.tr && cfg.tr->vec_dot ? ggml_get_type_traits_cpu(cfg.tr->vec_dot_type) : nullptr;
    if (!cfg.vtr) {
        fprintf(stderr, "skip %s %s %s: no ggml-cpu vec_dot, so no oracle\n", shp->name, mat.c_str(), tname.c_str());
        return false;
    }
    if (!quantize_expert(cfg.type, cfg.N, cfg.K, cfg.proto)) {
        fprintf(stderr, "skip %s %s %s: K=%d not a multiple of block %d (or quantize failed)\n", shp->name, mat.c_str(),
                tname.c_str(), cfg.K, (int) ggml_blck_size(cfg.type));
        return false;
    }
    cfg.expert_bytes = cfg.proto.size();
    cfg.row_bytes    = ggml_row_size(cfg.type, cfg.K);
    // Pool of distinct expert slots; each call reads the next k slots, so a slot
    // is re-read only after the whole pool (> LLC) was streamed.
    cfg.n_pool       = std::max<size_t>((size_t) cfg.k * 4, (o.pool_mb << 20) / cfg.expert_bytes);
    cfg.n_pool       = (cfg.n_pool / cfg.k) * cfg.k;

    const size_t pool_bytes = cfg.n_pool * cfg.expert_bytes;
    const size_t align      = o.thp ? (2u << 20) : 4096;
    const size_t alloc      = (pool_bytes + align - 1) / align * align;
    cfg.pool                = (uint8_t *) aligned_alloc(align, alloc);
    if (!cfg.pool) {
        fprintf(stderr, "alloc failed\n");
        return false;
    }
    if (o.thp) {
        madvise(cfg.pool, pool_bytes, MADV_HUGEPAGE);
    }
    for (size_t e = 0; e < cfg.n_pool; e++) {
        memcpy(cfg.pool + e * cfg.expert_bytes, cfg.proto.data(), cfg.expert_bytes);
    }
    std::vector<uint8_t> proto_r;
    if (o.variants_arg.find("r8") != std::string::npos && repack_8x8(cfg.type, cfg.N, cfg.K, cfg.proto, proto_r)) {
        cfg.pool_r = (uint8_t *) aligned_alloc(align, alloc);
        if (o.thp) {
            madvise(cfg.pool_r, pool_bytes, MADV_HUGEPAGE);
        }
        for (size_t e = 0; e < cfg.n_pool; e++) {
            memcpy(cfg.pool_r + e * cfg.expert_bytes, proto_r.data(), cfg.expert_bytes);
        }
    }
    // gate/up share one activation across experts; down has one per expert.
    cfg.act.resize((size_t) cfg.k * cfg.K);
    std::mt19937                    rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : cfg.act) {
        x = dist(rng);
    }
    cfg.outv.assign((size_t) cfg.k * cfg.N, 0.0f);
    // Own-team variants take one pre-quantized activation (the shared gate/up
    // case); `prod` quantizes per call, as production does.
    cfg.actq.resize(ggml_row_size(cfg.tr->vec_dot_type, cfg.K));
    cfg.vtr->from_float(cfg.act.data(), cfg.actq.data(), cfg.K);
    if (cfg.type == GGML_TYPE_Q2_0) {
        q2_prepare_act(cfg.K, (const bq8_0 *) cfg.actq.data(), cfg.q2act);
    }
    return true;
}

static std::vector<variant_inst> select_variants(const options & o, const config & cfg) {
    std::vector<variant_inst> out;
    for (const auto & s : split_csv(o.variants_arg)) {
        variant_inst v;
        if (!parse_variant(s, v)) {
            fprintf(stderr, "unknown variant %s\n", s.c_str());
            exit(2);
        }
        if (!v.def->applies(cfg)) {
            continue;
        }
        if (std::string(v.def->name) == "r8") {
            v.kernel = cfg.type == GGML_TYPE_MXFP4 ? kern_r8_mxfp4 : kern_r8_iq4_nl;
        }
        out.push_back(v);
    }
    return out;
}

int main(int argc, char ** argv) {
    // First thing: nothing may run an AVX-VNNI instruction before this check.
    if (!host_has_required_isa()) {
        fprintf(stderr, "SKIP: needs AVX2, FMA, F16C, AVX-VNNI and AVX-VNNI-INT8; this CPU lacks one\n");
        return 77;
    }
    options opt;
    if (!parse_args(argc, argv, opt)) {
        return 2;
    }
    g_have_start_mask = sched_getaffinity(0, sizeof(g_start_mask), &g_start_mask) == 0;
    ggml_cpu_init();  // fills the fp16->fp32 table the vec_dot kernels read (the backend does this in production)
    // Same rule as ggml_sycl_cpu_threads_hint() (cpu-dispatch.cpp:3133): env value >= 1, default hw-2, capped at
    // 32. The arena is sized from it once per process, so the own team must match or the arms compare unequal teams.
    const char * thr_env = getenv("GGML_SYCL_CPU_THREADS");
    const int    hw      = std::max(1, (int) std::thread::hardware_concurrency());
    const int    threads = std::min(32, thr_env ? std::max(1, atoi(thr_env)) : std::max(1, hw - 2));
    team         tm(threads);

    runner R{ tm, threads, opt.chunk, (int) sysconf(_SC_NPROCESSORS_ONLN), opt.pcores, (pid_t) syscall(SYS_gettid), {},
              {}, {},      {} };
    R.team_tids = tm.tids();
    std::mt19937 order_rng(5);
    int          rc = 0;

    printf("shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us\n");

    for (const auto & sname : split_csv(opt.shapes_arg)) {
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
        for (const auto & mat : split_csv(opt.mats_arg)) {
            for (const auto & tname : split_csv(opt.types_arg)) {
                config cfg;
                if (!build_config(opt, shp, mat, tname, cfg)) {
                    continue;
                }
                const std::vector<variant_inst> variants = select_variants(opt, cfg);
                if (variants.empty()) {
                    fprintf(stderr, "%s %s %s: no variant applies to this config; skipped\n", shp->name, mat.c_str(),
                            tname.c_str());
                    continue;
                }

                if (opt.oracle_selftest) {
                    // The positive control runs on every producing variant: a variant whose
                    // output the oracle cannot reject proves nothing when it passes.
                    int n_selftest = 0;
                    for (const auto & v : variants) {
                        if (!v.def->produces_output) {
                            continue;
                        }
                        n_selftest++;
                        const bool detected = !oracle_check(R, cfg, v, true);
                        fprintf(stderr, "oracle selftest %s %s %s %s: %s\n", shp->name, mat.c_str(), tname.c_str(),
                                v.name.c_str(), detected ? "corruption detected (ok)" : "NOT DETECTED");
                        if (!detected) {
                            rc = 4;
                        }
                    }
                    if (n_selftest == 0) {
                        fprintf(stderr, "oracle selftest %s %s %s: no producing variant selected, nothing exercised\n",
                                shp->name, mat.c_str(), tname.c_str());
                        rc = 4;
                    }
                }
                bool ok = true;
                for (const auto & v : variants) {
                    ok = oracle_check(R, cfg, v) && ok;
                }
                if (!ok) {
                    rc = 3;
                    continue;
                }
                if (opt.oracle_only) {
                    continue;
                }

                for (const auto & v : variants) {
                    for (int w = 0; w < opt.warmup; w++) {
                        run_call(R, cfg, v);
                    }
                }
                // Arena workers are created by the library on first use: find them
                // by the CPU time they burn, once per config (idempotent).
                for (const auto & v : variants) {
                    if (!v.def->kernel && !v.kernel) {
                        discover_tbb_workers(R, cfg, v, 12);
                        break;
                    }
                }
                if (R.threads > 1 && R.tbb_tids.empty() && has_prod(variants)) {
                    // Without TBB the library runs `prod` serially on the calling thread (GGML_SYCL_HAS_TBB == 0):
                    // every prod number from such a build is a single-core number.
                    fprintf(stderr,
                            "ERROR: threads=%d but no arena worker burned CPU during prod: the library was likely "
                            "built without TBB and prod ran serially; config skipped\n",
                            R.threads);
                    rc = 5;
                    continue;
                }

                const double bytes = (double) cfg.k * cfg.expert_bytes;
                for (int b = 0; b < opt.bursts; b++) {
                    std::vector<variant_inst> order = variants;
                    std::shuffle(order.begin(), order.end(), order_rng);
                    for (const auto & v : order) {
                        apply_variant_pin(R, v);
                        for (int c = 0; c < opt.burst_calls; c++) {
                            const double t0 = now_us();
                            const double c0 = cpu_us();
                            run_call(R, cfg, v);
                            const double us  = now_us() - t0;
                            const double cus = cpu_us() - c0;
                            printf("%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,%.1f,%.0f,%.3f,%.1f\n", shp->name, mat.c_str(),
                                   ggml_type_name(cfg.type), cfg.N, cfg.K, cfg.k, threads, v.name.c_str(), b, c, us,
                                   bytes, bytes / (us * 1e3), cus);
                        }
                        sample_cpus(R, v);
                    }
                }
                apply_pin(R.main_tid, R.tbb_tids, PIN_NONE, R.ncpu, R.pcores);
                apply_pin(R.main_tid, R.team_tids, PIN_NONE, R.ncpu, R.pcores);
                for (const auto & kv : R.cpu_samples) {
                    fprintf(stderr, "cpus %s %s %s threads=%d %s: last-run CPU on P-cores=%ld E-cores=%ld\n", shp->name,
                            mat.c_str(), tname.c_str(), threads, kv.first.c_str(), kv.second.first, kv.second.second);
                }
                R.cpu_samples.clear();
                fflush(stdout);
            }
        }
    }
    fprintf(stderr, "pin: %ld affinity calls, %ld failed\n", g_pin_calls, g_pin_failures);
    if (g_pin_failures && rc == 0) {
        rc = 6;  // the +pin arms did not run as labelled
    }
    return rc;
}
