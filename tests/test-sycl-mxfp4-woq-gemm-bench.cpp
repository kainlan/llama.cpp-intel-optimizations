// Custom batched-PP MXFP4 GEMM activation-format microbench (llama.cpp-iikr,
// stage 1). This is the "decide by microbench, not argument" instrument the
// task asked for: it measures, via SYCL DEVICE EVENTS (never host chrono --
// see repo memory host-chrono-cannot-see-past-submission-backpressure), the
// GEMM-only device time of two candidate activation/compute formats for a
// hand-written MXFP4 x activation GEMM that reads the STORED SOA and
// XMX_TILED weight layouts directly (no repack):
//
//   arm f16     -- activations already f16 (the format
//                  k_copy_src1_to_contiguous_f16_mapped already produces for
//                  the current WOQ executor -- zero new activation-prep
//                  cost), weight nibbles dequantized via the kvalues_mxfp4
//                  LUT to sycl::half and FMA'd directly.
//   arm q8dp4a  -- activations quantized to Q8_1 (mmq.cpp's established
//                  idiom for exactly this 4-bit x int8 shape -- see
//                  mmq.cpp's load_tiles_q4_0_soa / vec_dot_q4_0_q8_1_mul_mat_soa
//                  and mmvq.cpp's mxfp4_soa_q8_1_block_dot, the verified
//                  production GEMV analogue this arm's math is copied from),
//                  weight nibbles unpacked to int8 via the same LUT and
//                  combined with the activation via dp4a. The activation
//                  quantization step is timed SEPARATELY and reported next
//                  to the GEMM-only time -- per the task's explicit
//                  instruction, it is not free and must not be hidden.
//
// Both arms are benched against BOTH stored layouts (SOA -- the down-role
// layout; XMX_TILED -- the gate/up-role layout) at the real GPT-OSS 20B
// shape (blocks_per_row=90, nrows=2880 -- hidden_size==intermediate_size==
// 2880 for this model, so K==N here) and a sweep of M (activation rows per
// GEMM group) spanning the range the batched executor's expert-grouping
// actually produces (ggml-sycl.cpp's align_rows_64 padding; see the design
// note on llama.cpp-iikr for the host-side grouping logic this mirrors --
// one microbench call == one "gemm_group" call in the production executor,
// m=group.n_rows, n=ne01, k=ne00, exactly the DnnlGemmWrapper::
// woq_gemm_batch_mxfp4 call shape at ggml-sycl.cpp ~71440).
//
// SCOPE NOTE (read before extrapolating these numbers to a shipped kernel):
// both kernels below are WORK-GROUP TILED for activation reuse only (one
// work-group per output row, activation for that row staged once into SLM
// and reused across a TILE_N-wide strip of output columns) -- they do NOT
// tile the weight side across M rows, and they are not SLM-double-buffered
// or register-blocked. This is deliberate: an untiled (one-thread-per-
// output, no reuse) design would have skewed the f16-vs-q8dp4a comparison
// in q8dp4a's favor for a reason UNRELATED to the arithmetic question this
// bench exists to answer -- redundant re-reads of the activation row would
// cost 2 bytes/elem for f16 vs 1 byte/elem for already-quantized Q8_1,
// manufacturing a "q8dp4a wins" result out of pure memory traffic instead
// of compute throughput. Staging the activation row once in SLM removes
// that skew for both arms alike (weight-side bytes read are IDENTICAL
// between arms -- both read the same nibble+scale bytes once per lane).
// The stage-2 production kernel will add real M-side tiling (weight reuse
// across rows) on top of whichever arm this bench selects; this bench
// answers "which per-element dequant+dot math is faster on this hardware",
// not "what is the fastest possible GEMM".
//
// NOT a ctest: this is a benchmark, not a correctness gate. Build
// explicitly:
//   ./scripts/sycl-build.sh test-sycl-mxfp4-woq-gemm-bench
// Run (source oneAPI first -- see CLAUDE.md "A SYCL test binary run WITHOUT
// sourcing setvars.sh prints SKIP and exits 0"):
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mxfp4-woq-gemm-bench   # B50
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-mxfp4-woq-gemm-bench   # B70
//
// LEAD-HW ONLY: per CLAUDE.md's hard division of labor, this binary must be
// BUILT (compiled) by anyone, but RUN only by the lead session, serially,
// with the standard Shmem/MemAvailable sampling around it.

#include "ggml-common.h"
#include "ggml-sycl/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// joint_matrix (DPAS) arms -- added stage-1 follow-up cycle (team-lead
// directive, task comment c-0xub on llama.cpp-iikr): the plain-SYCL f16/
// q8dp4a arms above are ~14x below the throughput oneDNN's woq_gemm
// achieves (measured effective ~31/~16 TFLOPs on B70/B50 -- it is on the
// XMX systolic arrays), so joint_matrix is the only candidate class that
// can plausibly reach the bar. Same include-guard idiom as moe-xmx.hpp's
// SYCL_XMX_MOE_AVAILABLE.
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define SYCL_XMX_JM_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
namespace sycl_xmx_bench = sycl::ext::oneapi::experimental::matrix;

// joint_matrix_load/store require a decorated local-space pointer, not a
// raw one -- same cast moe-xmx-fused.hpp's fused_xmx_moe_gemm_mxfp4_tiled
// uses for its SLM operands.
template <typename T> static inline auto as_local_ptr(T * p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
#else
#    define SYCL_XMX_JM_AVAILABLE 0
#endif

// ---------------------------------------------------------------------------
// Local dequant table + LUT lookup, duplicated rather than pulling in
// vecdotq.hpp's full dependency chain -- same precedent as
// fused-moe-esimd.hpp's moe_get_int_from_table_16 ("inlined here to avoid
// pulling in the full vecdotq.hpp dependency chain") and
// tests/test-moe-mxfp4-dp4a.cpp's local kvalues_mxfp4. Values verified
// identical to ggml-common.h's GGML_TABLE_BEGIN(int8_t, kvalues_fp4, 16)
// (kvalues_mxfp4 is a #define alias for kvalues_fp4).
static constexpr int8_t kvalues_mxfp4_local[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// Byte-for-byte the production LUT unpack (ggml-sycl/vecdotq.hpp
// get_int_from_table_16(const int&, const int8_t*)), duplicated for the same
// dependency-chain reason as the table above.
static __dpct_inline__ sycl::int2 get_int_from_table_16_local(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t         tmp[2];
    const uint32_t   low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low   = dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high  = dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i]               = dpct::byte_level_permute(low, high, low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// ---------------------------------------------------------------------------
// Shared device-event timing harness (identical pattern to
// test-sycl-mxfp4-woq-repack-bench.cpp -- duplicated per that file's own
// precedent of one self-contained TU per bench).
struct gemm_bench_begin_marker;
struct gemm_bench_end_marker;

static sycl::event submit_begin_marker(sycl::queue & q) {
    return q.submit([&](sycl::handler & cgh) { cgh.single_task<gemm_bench_begin_marker>([] {}); });
}

static sycl::event submit_end_marker(sycl::queue & q) {
    return q.submit([&](sycl::handler & cgh) { cgh.single_task<gemm_bench_end_marker>([] {}); });
}

template <typename F> static double time_device_ms(sycl::queue & q, F && work) {
    sycl::event begin = submit_begin_marker(q);
    work(q);
    sycl::event end = submit_end_marker(q);
    q.wait();
    const uint64_t t0 = begin.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t t1 = end.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(t1 - t0) / 1.0e6;  // ns -> ms
}

struct bench_result {
    double min_ms  = 0.0;
    double mean_ms = 0.0;
};

template <typename F> static bench_result run_bench(sycl::queue & q, int warmup, int iters, F && work) {
    for (int i = 0; i < warmup; ++i) {
        (void) time_device_ms(q, work);
    }
    double sum    = 0.0;
    double min_ms = -1.0;
    for (int i = 0; i < iters; ++i) {
        const double ms = time_device_ms(q, work);
        sum += ms;
        if (min_ms < 0.0 || ms < min_ms) {
            min_ms = ms;
        }
    }
    return bench_result{ min_ms, sum / iters };
}

// ---------------------------------------------------------------------------
// Layout geometry (verified against the STORED forms this task must consume
// directly -- see design note comments on llama.cpp-iikr for the full
// citation trail):
//
//   SOA   (ggml-sycl/quants.hpp block_q_t<GGML_TYPE_MXFP4>): per output row
//   n (0..N-1), block b (0..blocks_per_row-1), block_index = n*blocks_per_row+b:
//     qs @ block_index * 16                      (16 bytes, 32 packed nibbles)
//     e  @ nblocks*16 + block_index               (1 byte E8M0, nblocks=N*blocks_per_row)
//
//   XMX_TILED (ggml-sycl/moe-xmx-fused.hpp MXFPXMXLayoutInfo::compute /
//   reorder_mxfp4_to_xmx_layout -- the SAME struct convert.cpp cites as the
//   layout's authority at convert.cpp:1961-1965): k-tile-major
//   [tile_k_group][tile_n_group], tiles_k_per_group=1 so tile_k_group ==
//   block index b directly; tile_n_group = n / tile_n_total; within a tile
//   group: scales[tile_n_total] (1 byte each) then qs[tile_n_total][16]
//   (16 bytes each, same packed-nibble convention as SOA).
//     bytes_per_tile_group = tile_n_total * (1 + 16)
//     tile_group(b, n) = base + (b * n_tile_groups_n + n/tile_n_total) * bytes_per_tile_group
//     scale @ tile_group + (n % tile_n_total)
//     qs    @ tile_group + tile_n_total + (n % tile_n_total) * 16
//
// Nibble->element mapping (verified against
// ggml-sycl/convert.cpp:dequantize_tile_mxfp4_soa_rowmajor, the production
// SOA->f16 dequant this task's f16 arm must match): for byte position p
// (0..15) within the 16-byte qs block, the LOW nibble is element p and the
// HIGH nibble is element p+16 -- NOT interleaved pairs (p, p+1); this is the
// "j / j+16" split the vyjl spike and vecdotq.hpp's get_int_from_table_16
// both describe. Confirmed identical convention feeds both arms below.

constexpr int TILE_N_TOTAL = 16;  // matches test-sycl-mxfp4-woq-repack-bench.cpp's
                                  // precedent value (its own comment: "caps.N *
                                  // optimal_tiles_n, current device caps") -- kept
                                  // for direct comparability with the accepted
                                  // 0vqt battery artifact's GB/s figures. The
                                  // production default (moe_xmx_fused::
                                  // MXFPXMXConfig, tiles_n=4, XMX_N=16) is 64 on
                                  // this hardware; NOT swept here for time --
                                  // flagged as a follow-up sweep point in the
                                  // design note if TILE_N_TOTAL turns out to
                                  // matter to the arm decision.

static inline int64_t soa_qs_offset(int64_t n, int64_t b, int64_t blocks_per_row) {
    return (n * blocks_per_row + b) * 16;
}

static inline int64_t soa_e_offset(int64_t n, int64_t b, int64_t blocks_per_row, int64_t nblocks) {
    return nblocks * 16 + (n * blocks_per_row + b);
}

static inline int64_t xmx_tile_group_offset(int64_t b, int64_t n, int64_t n_tile_groups_n, int64_t tile_n_total) {
    const int64_t bytes_per_tile_group = tile_n_total * 17;
    return (b * n_tile_groups_n + n / tile_n_total) * bytes_per_tile_group;
}

// ---------------------------------------------------------------------------
// Activation prep kernels (bracket-timed separately from the GEMM itself --
// the design note's explicit instruction for the q8dp4a arm's quant cost).
// f16 arm: matches k_copy_src1_to_contiguous_f16_mapped's OUTPUT format
// exactly (plain row-major f16, K contiguous) -- generated and uploaded
// directly in main() (its host copy is retained there for the CPU
// reference oracle), rather than through a helper here.

// q8dp4a arm quantization kernel: one work-group per row, block_q8_1 SOA
// layout per row -- [90 blocks x 32 int8 qs][90 x half2(d,sum)] -- matching
// mxfp4_soa_q8_1_block_dot's addressing convention exactly (qs at
// row_base + b*QK8_1, ds at row_base + K_bytes + b*sizeof(half2), K_bytes
// == blocks_per_row*QK8_1 == K since QK8_1==32==QK_MXFP4).
static void quantize_q8_1_row_kernel(const sycl::half * __restrict__ act_f16,
                                     int8_t * __restrict__ q8_out,
                                     int                      k,
                                     int                      blocks_per_row,
                                     const sycl::nd_item<1> & item) {
    const int row = static_cast<int>(item.get_group(0));
    const int b   = static_cast<int>(item.get_local_id(0));
    if (b >= blocks_per_row) {
        return;
    }
    const sycl::half * src  = act_f16 + static_cast<int64_t>(row) * k + b * QK_MXFP4;
    float              amax = 0.0f;
    float              vals[QK_MXFP4];
#pragma unroll
    for (int i = 0; i < QK_MXFP4; ++i) {
        vals[i]       = static_cast<float>(src[i]);
        const float a = sycl::fabs(vals[i]);
        amax          = a > amax ? a : amax;
    }
    const float d   = amax == 0.0f ? 1.0f : amax / 127.0f;
    float       sum = 0.0f;
    int8_t *    qs_dst =
        reinterpret_cast<int8_t *>(q8_out) + static_cast<int64_t>(row) * (k + blocks_per_row * 4) + b * QK_MXFP4;
#pragma unroll
    for (int i = 0; i < QK_MXFP4; ++i) {
        const float scaled = vals[i] / d;
        qs_dst[i]          = static_cast<int8_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        sum += vals[i];
    }
    sycl::half * ds_dst = reinterpret_cast<sycl::half *>(reinterpret_cast<int8_t *>(q8_out) +
                                                         static_cast<int64_t>(row) * (k + blocks_per_row * 4) + k) +
                          b * 2;
    ds_dst[0] = sycl::half(amax == 0.0f ? 0.0f : d);
    ds_dst[1] = sycl::half(sum);
}

static sycl::event quantize_q8_1_rows(sycl::queue &      q,
                                      const sycl::half * act_f16,
                                      int8_t *           q8_out,
                                      int64_t            m,
                                      int                k,
                                      int                blocks_per_row) {
    return q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(m * blocks_per_row), sycl::range<1>(blocks_per_row)),
            [=](sycl::nd_item<1> item) { quantize_q8_1_row_kernel(act_f16, q8_out, k, blocks_per_row, item); });
    });
}

// ---------------------------------------------------------------------------
// GEMM kernels. Grid: (m, n_tiles) work-groups, WG_SIZE lanes = one lane per
// output column in the tile. Phase 1: cooperative SLM load of this row's
// activation (whole K). Phase 2: barrier. Phase 3: each lane computes one
// output column's full K-reduction, reading weight bytes from global
// memory (SOA or XMX_TILED per the two kernels below) and the row's
// activation from SLM.
constexpr int WG_SIZE = 64;                                           // == TILE_N: one lane per output column per tile

static void gemm_soa_f16_kernel(const uint8_t * __restrict__ w_qs_e,  // SOA: qs then e, as one buffer
                                int64_t nblocks,
                                const sycl::half * __restrict__ act_row_slm_src,
                                float * __restrict__ out,
                                int64_t k,
                                int64_t n,
                                int     blocks_per_row,
                                sycl::half * __restrict__ slm_act,
                                const sycl::nd_item<2> & item) {
    const int64_t row  = item.get_group(0);
    const int64_t n0   = item.get_group(1) * WG_SIZE;
    const int     lane = item.get_local_id(1);

    // Phase 1: cooperative load of this row's K activations into SLM.
    for (int64_t i = lane; i < k; i += WG_SIZE) {
        slm_act[i] = act_row_slm_src[row * k + i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t col = n0 + lane;
    if (col >= n) {
        return;
    }
    float           acc    = 0.0f;
    const uint8_t * e_base = w_qs_e + nblocks * 16;
    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t      block_index = col * blocks_per_row + b;
        const uint8_t *    qs          = w_qs_e + block_index * 16;
        const float        d           = sycl_e8m0_to_fp32_half(e_base[block_index]);
        const sycl::half * act_blk     = slm_act + b * QK_MXFP4;
#pragma unroll
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            const uint8_t q = qs[j];
            acc += static_cast<float>(act_blk[j]) * (d * kvalues_mxfp4_local[q & 0xf]);
            acc += static_cast<float>(act_blk[j + QK_MXFP4 / 2]) * (d * kvalues_mxfp4_local[q >> 4]);
        }
    }
    out[row * n + col] = acc;
}

static void gemm_tiled_f16_kernel(const uint8_t * __restrict__ w_tiled,
                                  int64_t n_tile_groups_n,
                                  int64_t tile_n_total,
                                  const sycl::half * __restrict__ act_row_src,
                                  float * __restrict__ out,
                                  int64_t k,
                                  int64_t n,
                                  int     blocks_per_row,
                                  sycl::half * __restrict__ slm_act,
                                  const sycl::nd_item<2> & item) {
    const int64_t row  = item.get_group(0);
    const int64_t n0   = item.get_group(1) * WG_SIZE;
    const int     lane = item.get_local_id(1);

    for (int64_t i = lane; i < k; i += WG_SIZE) {
        slm_act[i] = act_row_src[row * k + i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t col = n0 + lane;
    if (col >= n) {
        return;
    }
    float         acc = 0.0f;
    const int64_t tn  = col % tile_n_total;
    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t      tg      = xmx_tile_group_offset(b, col, n_tile_groups_n, tile_n_total);
        const uint8_t      e8m0    = w_tiled[tg + tn];
        const uint8_t *    qs      = w_tiled + tg + tile_n_total + tn * 16;
        const float        d       = sycl_e8m0_to_fp32_half(e8m0);
        const sycl::half * act_blk = slm_act + b * QK_MXFP4;
#pragma unroll
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            const uint8_t q = qs[j];
            acc += static_cast<float>(act_blk[j]) * (d * kvalues_mxfp4_local[q & 0xf]);
            acc += static_cast<float>(act_blk[j + QK_MXFP4 / 2]) * (d * kvalues_mxfp4_local[q >> 4]);
        }
    }
    out[row * n + col] = acc;
}

// q8dp4a arm: SLM holds the row's pre-quantized Q8_1 SOA block (copy-in
// only, no compute -- quantization already happened once per row via
// quantize_q8_1_rows above, matching production's "quantize once, reuse
// across the whole N-wide GEMM" shape).
static void gemm_soa_q8dp4a_kernel(const uint8_t * __restrict__ w_qs_e,
                                   int64_t nblocks,
                                   const int8_t * __restrict__ q8_act,
                                   float * __restrict__ out,
                                   int64_t k,
                                   int64_t n,
                                   int     blocks_per_row,
                                   int8_t * __restrict__ slm_qs,
                                   sycl::half * __restrict__ slm_ds,
                                   const sycl::nd_item<2> & item) {
    const int64_t      row       = item.get_group(0);
    const int64_t      n0        = item.get_group(1) * WG_SIZE;
    const int          lane      = item.get_local_id(1);
    const int64_t      row_bytes = k + blocks_per_row * 4;
    const int8_t *     row_qs    = q8_act + row * row_bytes;
    const sycl::half * row_ds    = reinterpret_cast<const sycl::half *>(reinterpret_cast<const int8_t *>(row_qs) + k);

    for (int64_t i = lane; i < k; i += WG_SIZE) {
        slm_qs[i] = row_qs[i];
    }
    for (int64_t i = lane; i < blocks_per_row * 2; i += WG_SIZE) {
        slm_ds[i] = row_ds[i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t col = n0 + lane;
    if (col >= n) {
        return;
    }
    float           acc    = 0.0f;
    const uint8_t * e_base = w_qs_e + nblocks * 16;
    const int *     q8_qs  = reinterpret_cast<const int *>(slm_qs);
    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t   block_index = col * blocks_per_row + b;
        const uint8_t * qs          = w_qs_e + block_index * 16;
        const float     d           = sycl_e8m0_to_fp32_half(e_base[block_index]);
        const float     d8          = static_cast<float>(slm_ds[b * 2 + 0]);
        int             sumi        = 0;
#pragma unroll
        for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
            const int        aux_q4 = *reinterpret_cast<const int *>(qs + i);
            const sycl::int2 va     = get_int_from_table_16_local(aux_q4, kvalues_mxfp4_local);
            const int        q8_lo  = q8_qs[b * (QK_MXFP4 / 4) + i / 4];
            const int        q8_hi  = q8_qs[b * (QK_MXFP4 / 4) + i / 4 + 4];
            sumi                    = dpct::dp4a(va.x(), q8_lo, sumi);
            sumi                    = dpct::dp4a(va.y(), q8_hi, sumi);
        }
        acc += d * d8 * static_cast<float>(sumi);
    }
    out[row * n + col] = acc;
}

static void gemm_tiled_q8dp4a_kernel(const uint8_t * __restrict__ w_tiled,
                                     int64_t n_tile_groups_n,
                                     int64_t tile_n_total,
                                     const int8_t * __restrict__ q8_act,
                                     float * __restrict__ out,
                                     int64_t k,
                                     int64_t n,
                                     int     blocks_per_row,
                                     int8_t * __restrict__ slm_qs,
                                     sycl::half * __restrict__ slm_ds,
                                     const sycl::nd_item<2> & item) {
    const int64_t      row       = item.get_group(0);
    const int64_t      n0        = item.get_group(1) * WG_SIZE;
    const int          lane      = item.get_local_id(1);
    const int64_t      row_bytes = k + blocks_per_row * 4;
    const int8_t *     row_qs    = q8_act + row * row_bytes;
    const sycl::half * row_ds    = reinterpret_cast<const sycl::half *>(reinterpret_cast<const int8_t *>(row_qs) + k);

    for (int64_t i = lane; i < k; i += WG_SIZE) {
        slm_qs[i] = row_qs[i];
    }
    for (int64_t i = lane; i < blocks_per_row * 2; i += WG_SIZE) {
        slm_ds[i] = row_ds[i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t col = n0 + lane;
    if (col >= n) {
        return;
    }
    float         acc   = 0.0f;
    const int64_t tn    = col % tile_n_total;
    const int *   q8_qs = reinterpret_cast<const int *>(slm_qs);
    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t   tg   = xmx_tile_group_offset(b, col, n_tile_groups_n, tile_n_total);
        const uint8_t   e8m0 = w_tiled[tg + tn];
        const uint8_t * qs   = w_tiled + tg + tile_n_total + tn * 16;
        const float     d    = sycl_e8m0_to_fp32_half(e8m0);
        const float     d8   = static_cast<float>(slm_ds[b * 2 + 0]);
        int             sumi = 0;
#pragma unroll
        for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
            const int        aux_q4 = *reinterpret_cast<const int *>(qs + i);
            const sycl::int2 va     = get_int_from_table_16_local(aux_q4, kvalues_mxfp4_local);
            const int        q8_lo  = q8_qs[b * (QK_MXFP4 / 4) + i / 4];
            const int        q8_hi  = q8_qs[b * (QK_MXFP4 / 4) + i / 4 + 4];
            sumi                    = dpct::dp4a(va.x(), q8_lo, sumi);
            sumi                    = dpct::dp4a(va.y(), q8_hi, sumi);
        }
        acc += d * d8 * static_cast<float>(sumi);
    }
    out[row * n + col] = acc;
}

#if SYCL_XMX_JM_AVAILABLE
// ---------------------------------------------------------------------------
// joint_matrix (DPAS) arms, tiled (XMX_TILED) layout only -- team-lead's
// directive scopes this cycle to the tiled layout, where the existing
// production DPAS precedent (mmvq.cpp's
// mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl) and the dead-code
// precedent (moe-xmx-fused.hpp's fused_xmx_moe_gemm_mxfp4_tiled) both live;
// SOA joint_matrix arms are an easy follow-on with this same structure but
// out of scope here.
//
// Tile shapes are NOT guessed -- both are copied from VERIFIED, ALREADY-
// COMPILING production code in this repo, at the two different K depths
// Intel's DPAS uses for the two element widths:
//   f16:  XMX_TILE_M=8, XMX_TILE_N=16, XMX_TILE_K=16
//         (ggml-sycl/unified-kernel.hpp:575-577 -- the SAME constants an
//         existing production MXFP4-consuming (AOS) f16 joint_matrix GEMM
//         in unified-kernel.cpp already uses successfully on this hardware)
//   int8: XMX_M=8, XMX_N=16, XMX_K=32
//         (ggml-sycl/moe-xmx-fused.hpp's MXFPXMXConfig -- the same constants
//         GGML_SYCL_MXFP4_MOE_XMX_M/N/K in common.hpp use for the LIVE
//         production mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl
//         kernel)
// The load/mad/store call sequence (joint_matrix_load with an explicit
// leading dimension for a [N][K]-flat SLM buffer as the B operand,
// joint_matrix_mad into an accumulator, joint_matrix_store back to SLM for
// scalar extraction rather than joint_matrix_apply) is copied verbatim in
// shape from moe-xmx-fused.hpp's fused_xmx_moe_gemm_mxfp4_tiled (int8) and
// unified-kernel.cpp's XMX MUL_MAT path (f16) -- both ALREADY COMPILE AND
// RUN on this hardware today, which de-risks the API usage even though
// neither one is doing what this bench asks of it (per-token M=1 GEMV for
// the former; AOS-only for the latter). What's NEW here, and therefore NOT
// covered by that precedent, is genuinely batching XMX_TILE_M=8 REAL rows
// (both precedents above only ever have 1 real row in the M dimension) and
// reading XMX_TILED (not AOS). Flagging this precisely so a wrong number
// here is diagnosed as "the batching is new" rather than "the API is
// unverified".
constexpr int XMX_JM_M       = 8;                          // real M dim: both precedents use 8
constexpr int XMX_JM_N       = 16;                         // == TILE_N_TOTAL above, by construction
constexpr int XMX_JM_K_F16   = 16;
constexpr int XMX_JM_K_I8    = 32;                         // == QK_MXFP4: one MXFP4 block per K-step
constexpr int XMX_JM_SG      = 16;                         // sub-group size both precedents require
constexpr int XMX_JM_NUM_SG  = 4;                          // sub-groups per WG == N-tiles per WG
constexpr int XMX_JM_WG_SIZE = XMX_JM_NUM_SG * XMX_JM_SG;  // 64, matches WG_SIZE above

// f16 arm: weight nibbles dequantized AND pre-scaled by the block's e8m0
// factor while staging into SLM, so the joint_matrix accumulator can sum
// RAW mad() results across the WHOLE K loop with no per-block reset --
// unlike the int8 arm below, f16 has no per-row activation scale to apply
// post-hoc, so baking the (single, per-column) weight scale into the SLM
// tile before joint_matrix_load is both correct and simpler. Each MXFP4
// block (32 elements) is fed as TWO K-steps of XMX_JM_K_F16=16 (low
// nibbles -> elements 0..15, high nibbles -> elements 16..31, matching the
// verified p/p+16 mapping documented above) into the SAME accumulator --
// mathematically identical to scaling the two halves separately and
// summing, since both halves of one block share the same scale.
static void gemm_tiled_f16_jm_kernel(const uint8_t * __restrict__ w_tiled,
                                     int64_t n_tile_groups_n,
                                     int64_t tile_n_total,
                                     const sycl::half * __restrict__ act,
                                     float * __restrict__ out,
                                     int64_t k,
                                     int64_t n,
                                     int     blocks_per_row,
                                     sycl::half * __restrict__ slm_act,  // [XMX_JM_M][XMX_JM_K_F16], WG-shared
                                     sycl::half * __restrict__ slm_w,    // [XMX_JM_NUM_SG][XMX_JM_N][XMX_JM_K_F16]
                                     float * __restrict__ slm_out,       // [XMX_JM_NUM_SG][XMX_JM_M][XMX_JM_N]
                                     const sycl::nd_item<2> & item) {
    namespace jm        = sycl_xmx_bench;
    const int64_t m0    = item.get_group(0) * XMX_JM_M;
    const int64_t n_wg0 = item.get_group(1) * XMX_JM_WG_SIZE;
    auto          sg    = item.get_sub_group();
    const int     sg_id = static_cast<int>(sg.get_group_linear_id());
    const int     lane  = static_cast<int>(sg.get_local_linear_id());
    const int64_t n0    = n_wg0 + sg_id * XMX_JM_N;
    // n0 is always a multiple of tile_n_total (== XMX_JM_N == 16 by
    // construction: TILE_N_TOTAL==16 above), so this subgroup's 16-column
    // span is EXACTLY one XMX_TILED tile group -- not one column of it.
    // scales_ptr[col_local] / qs_ptr[col_local*16 + byte] index the WHOLE
    // tile group below, one entry per column in this subgroup's span.

    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, XMX_JM_M, XMX_JM_K_F16, jm::layout::row_major> mat_a;
    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, XMX_JM_K_F16, XMX_JM_N, jm::layout::col_major> mat_b;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, XMX_JM_M, XMX_JM_N>                       acc;
    jm::joint_matrix_fill(sg, acc, 0.0f);

    sycl::half * my_slm_w   = slm_w + sg_id * XMX_JM_N * XMX_JM_K_F16;
    float *      my_slm_out = slm_out + sg_id * XMX_JM_M * XMX_JM_N;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t   tg         = xmx_tile_group_offset(b, n0, n_tile_groups_n, tile_n_total);
        const uint8_t * scales_ptr = w_tiled + tg;
        const uint8_t * qs_ptr     = w_tiled + tg + tile_n_total;
        for (int half = 0; half < 2; ++half) {
            // Shared activation tile: staged once by the whole WG (every
            // subgroup needs the same m0/K-range), not per subgroup.
            for (int i = lane + sg_id * XMX_JM_SG; i < XMX_JM_M * XMX_JM_K_F16; i += XMX_JM_WG_SIZE) {
                const int m_local = i / XMX_JM_K_F16;
                const int k_local = i % XMX_JM_K_F16;
                slm_act[i]        = act[(m0 + m_local) * k + b * QK_MXFP4 + half * XMX_JM_K_F16 + k_local];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Per-subgroup weight tile: ALL 16 columns of this subgroup's
            // own tile group, each with its own e8m0 scale.
            for (int lid = lane; lid < XMX_JM_N * XMX_JM_K_F16; lid += XMX_JM_SG) {
                const int     col_local                      = lid / XMX_JM_K_F16;
                const int     k_local                        = lid % XMX_JM_K_F16;
                const float   d                              = sycl_e8m0_to_fp32_half(scales_ptr[col_local]);
                const uint8_t byte                           = qs_ptr[col_local * 16 + k_local];
                const int8_t  nib                            = half == 0 ? (byte & 0xf) : (byte >> 4);
                my_slm_w[col_local * XMX_JM_K_F16 + k_local] = sycl::half(d * kvalues_mxfp4_local[nib]);
            }
            sycl::group_barrier(sg);

            jm::joint_matrix_load(sg, mat_a, as_local_ptr(slm_act), XMX_JM_K_F16);
            jm::joint_matrix_load(sg, mat_b, as_local_ptr(my_slm_w), XMX_JM_K_F16);
            jm::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            // WG-scoped, not subgroup-scoped: slm_act is WG-SHARED and the
            // next iteration's cooperative load (above) overwrites it --
            // every subgroup must finish reading it via joint_matrix_load
            // before any subgroup starts that overwrite, which a
            // subgroup-only barrier cannot guarantee across subgroups.
            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    jm::joint_matrix_store(sg, acc, as_local_ptr(my_slm_out), XMX_JM_N, jm::layout::row_major);
    sycl::group_barrier(sg);
    for (int i = lane; i < XMX_JM_M * XMX_JM_N; i += XMX_JM_SG) {
        const int m_local                      = i / XMX_JM_N;
        const int n_local                      = i % XMX_JM_N;
        out[(m0 + m_local) * n + n0 + n_local] = my_slm_out[i];
    }
}

// int8/dp4a-via-DPAS arm: mirrors moe-xmx-fused.hpp's
// fused_xmx_moe_gemm_mxfp4_tiled block loop exactly (fresh accumulator per
// MXFP4 block == one XMX_JM_K_I8=32 K-step, extract to SLM, apply BOTH the
// per-row Q8_1 activation scale and the per-column weight scale, add into a
// running float total, reset) -- generalized from that kernel's M=1-real-
// row special case to XMX_JM_M=8 genuinely real rows, which needs the
// per-row scale applied per matrix ELEMENT (not once for the whole tile,
// as the M=1 precedent could get away with).
static void gemm_tiled_q8dp4a_jm_kernel(const uint8_t * __restrict__ w_tiled,
                                        int64_t n_tile_groups_n,
                                        int64_t tile_n_total,
                                        const int8_t * __restrict__ q8_act,  // per-row SOA: [90*32 qs][90*half2 ds]
                                        float * __restrict__ out,
                                        int64_t k,
                                        int64_t n,
                                        int     blocks_per_row,
                                        int8_t * __restrict__ slm_act,   // [XMX_JM_M][XMX_JM_K_I8], WG-shared
                                        float * __restrict__ slm_act_d,  // [XMX_JM_M], WG-shared, this block's scale
                                        int8_t * __restrict__ slm_w,     // [XMX_JM_NUM_SG][XMX_JM_N][XMX_JM_K_I8]
                                        int32_t * __restrict__ slm_raw,  // [XMX_JM_NUM_SG][XMX_JM_M][XMX_JM_N]
                                        const sycl::nd_item<2> & item) {
    namespace jm            = sycl_xmx_bench;
    const int64_t m0        = item.get_group(0) * XMX_JM_M;
    const int64_t n_wg0     = item.get_group(1) * XMX_JM_WG_SIZE;
    auto          sg        = item.get_sub_group();
    const int     sg_id     = static_cast<int>(sg.get_group_linear_id());
    const int     lane      = static_cast<int>(sg.get_local_linear_id());
    const int64_t n0        = n_wg0 + sg_id * XMX_JM_N;
    // As in the f16 arm above: n0 is always tile_n_total-aligned (==
    // XMX_JM_N==16), so this subgroup's span is exactly one XMX_TILED tile
    // group -- scales_ptr/qs_ptr below are indexed per-column across the
    // WHOLE group, not a single "my column".
    const int64_t row_bytes = k + blocks_per_row * 4;

    int8_t *  my_slm_w   = slm_w + sg_id * XMX_JM_N * XMX_JM_K_I8;
    int32_t * my_slm_raw = slm_raw + sg_id * XMX_JM_M * XMX_JM_N;

    // Zero this SUBGROUP's own (m0..m0+8, n0..n0+16) tile of the global out
    // buffer, which doubles as the running float total across blocks
    // (mirroring the scalar arms above). Must be SG-scoped (lane; += SG),
    // not WG-flat (lane+sg_id*SG; += WG_SIZE) -- each subgroup's n0 differs,
    // so a WG-flat distribution over a single-subgroup-sized range would
    // leave 3/4 of each subgroup's own tile unzeroed.
    for (int i = lane; i < XMX_JM_M * XMX_JM_N; i += XMX_JM_SG) {
        const int m_local                      = i / XMX_JM_N;
        const int n_local                      = i % XMX_JM_N;
        out[(m0 + m_local) * n + n0 + n_local] = 0.0f;
    }
    sycl::group_barrier(sg);

    for (int b = 0; b < blocks_per_row; ++b) {
        const int64_t   tg         = xmx_tile_group_offset(b, n0, n_tile_groups_n, tile_n_total);
        const uint8_t * scales_ptr = w_tiled + tg;
        const uint8_t * qs_ptr     = w_tiled + tg + tile_n_total;

        for (int i = lane + sg_id * XMX_JM_SG; i < XMX_JM_M * XMX_JM_K_I8; i += XMX_JM_WG_SIZE) {
            const int m_local = i / XMX_JM_K_I8;
            const int k_local = i % XMX_JM_K_I8;
            slm_act[i]        = q8_act[(m0 + m_local) * row_bytes + b * QK_MXFP4 + k_local];
        }
        for (int m_local = lane + sg_id * XMX_JM_SG; m_local < XMX_JM_M; m_local += XMX_JM_WG_SIZE) {
            const sycl::half * ds = reinterpret_cast<const sycl::half *>(q8_act + (m0 + m_local) * row_bytes + k);
            slm_act_d[m_local]    = static_cast<float>(ds[b * 2 + 0]);
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Per-subgroup weight tile: ALL 16 columns of this subgroup's own
        // tile group (each column's 16-byte qs block covers K=32 nibbles
        // for THIS one MXFP4 block == exactly XMX_JM_K_I8).
        for (int lid = lane; lid < XMX_JM_N * XMX_JM_K_I8; lid += XMX_JM_SG) {
            const int     col_local                     = lid / XMX_JM_K_I8;
            const int     k_local                       = lid % XMX_JM_K_I8;
            const uint8_t byte                          = qs_ptr[col_local * 16 + (k_local & 0xf)];
            const int8_t  nib                           = k_local < 16 ? (byte & 0xf) : (byte >> 4);
            my_slm_w[col_local * XMX_JM_K_I8 + k_local] = kvalues_mxfp4_local[nib];
        }
        sycl::group_barrier(sg);

        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, XMX_JM_M, XMX_JM_K_I8, jm::layout::row_major> mat_a;
        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, XMX_JM_K_I8, XMX_JM_N, jm::layout::col_major> mat_b;
        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, XMX_JM_M, XMX_JM_N>                acc;
        jm::joint_matrix_fill(sg, acc, 0);
        jm::joint_matrix_load(sg, mat_a, as_local_ptr(slm_act), XMX_JM_K_I8);
        jm::joint_matrix_load(sg, mat_b, as_local_ptr(my_slm_w), XMX_JM_K_I8);
        jm::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
        jm::joint_matrix_store(sg, acc, as_local_ptr(my_slm_raw), XMX_JM_N, jm::layout::row_major);
        sycl::group_barrier(sg);

        for (int i = lane; i < XMX_JM_M * XMX_JM_N; i += XMX_JM_SG) {
            const int   m_local = i / XMX_JM_N;
            const int   n_local = i % XMX_JM_N;
            const float d_col   = sycl_e8m0_to_fp32_half(scales_ptr[n_local]);
            out[(m0 + m_local) * n + n0 + n_local] += static_cast<float>(my_slm_raw[i]) * slm_act_d[m_local] * d_col;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
}
#endif  // SYCL_XMX_JM_AVAILABLE

static void report(const char * form, int64_t m, int64_t n, int64_t k, const bench_result & r) {
    const double flops       = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double gflops_mean = (flops / 1.0e9) / (r.mean_ms / 1000.0);
    std::printf("[GEMM-BENCH] form=%-24s m=%lld n=%lld k=%lld mean_ms=%.4f min_ms=%.4f mean_GFLOPs=%.2f\n", form,
                (long long) m, (long long) n, (long long) k, r.mean_ms, r.min_ms, gflops_mean);
}

// ---------------------------------------------------------------------------
// CPU CORRECTNESS ORACLE (team-lead directive, task comment on llama.cpp-iikr
// after the first jm run produced physically-impossible GFLOPs: the marker
// queue was out-of-order with no dependency edges, so t1-t0 could measure
// marker-to-marker latency with the kernel still in flight, AND no form
// validated its own output -- a silently-failed launch (JIT failure, bad
// launch config) would print a fantastic time instead of an error).
//
// This is a coarse SANITY gate, not a precision gate: it exists to catch
// "the kernel did not compute the right thing at all" (wrong indexing,
// uninitialized launch, garbage/stale output), not to characterize fine
// numerical error. Host reference computed in double precision from the
// SAME random host bytes uploaded to the device, using the identical
// dequant formula as the device kernels (kvalues_mxfp4 LUT + the exact
// sycl_e8m0_to_fp32_half bit-manipulation, reproduced here on host since
// it's portable bit arithmetic, not SYCL-specific) -- one reference per
// LAYOUT (not per arm): f16 arms compare against it directly (tight
// tolerance -- they should match almost exactly, mod float accumulation
// order); q8dp4a arms compare against the SAME full-precision reference
// but with a looser tolerance, since Q8_1 quantization is a real, expected,
// already-understood source of error that this gate must not misreport as
// a bug.

static float cpu_e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) {
        bits = 0x00200000u << e;
    } else {
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// out[row][col] for row in [0,max_m), col in [0,N), row-major stride N.
static void cpu_reference_gemm_soa(const std::vector<sycl::half> & act,
                                   const std::vector<uint8_t> &    w_soa,
                                   std::vector<float> &            out,
                                   int64_t                         max_m,
                                   int64_t                         n,
                                   int64_t                         k,
                                   int                             blocks_per_row) {
    const int64_t nblocks = n * blocks_per_row;
    for (int64_t row = 0; row < max_m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int b = 0; b < blocks_per_row; ++b) {
                const int64_t      block_index = col * blocks_per_row + b;
                const uint8_t *    qs          = w_soa.data() + block_index * 16;
                const double       d           = cpu_e8m0_to_fp32_half(w_soa[nblocks * 16 + block_index]);
                const sycl::half * act_blk     = act.data() + row * k + b * QK_MXFP4;
                for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    acc += static_cast<double>(static_cast<float>(act_blk[j])) * (d * kvalues_mxfp4_local[byte & 0xf]);
                    acc += static_cast<double>(static_cast<float>(act_blk[j + QK_MXFP4 / 2])) *
                           (d * kvalues_mxfp4_local[byte >> 4]);
                }
            }
            out[row * n + col] = static_cast<float>(acc);
        }
    }
}

static void cpu_reference_gemm_tiled(const std::vector<sycl::half> & act,
                                     const std::vector<uint8_t> &    w_tiled,
                                     std::vector<float> &            out,
                                     int64_t                         max_m,
                                     int64_t                         n,
                                     int64_t                         k,
                                     int                             blocks_per_row,
                                     int64_t                         n_tile_groups_n,
                                     int64_t                         tile_n_total) {
    for (int64_t row = 0; row < max_m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int b = 0; b < blocks_per_row; ++b) {
                const int64_t      tg      = xmx_tile_group_offset(b, col, n_tile_groups_n, tile_n_total);
                const int64_t      tn      = col % tile_n_total;
                const double       d       = cpu_e8m0_to_fp32_half(w_tiled[tg + tn]);
                const uint8_t *    qs      = w_tiled.data() + tg + tile_n_total + tn * 16;
                const sycl::half * act_blk = act.data() + row * k + b * QK_MXFP4;
                for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    acc += static_cast<double>(static_cast<float>(act_blk[j])) * (d * kvalues_mxfp4_local[byte & 0xf]);
                    acc += static_cast<double>(static_cast<float>(act_blk[j + QK_MXFP4 / 2])) *
                           (d * kvalues_mxfp4_local[byte >> 4]);
                }
            }
            out[row * n + col] = static_cast<float>(acc);
        }
    }
}

// Q8-AWARE references for the q8dp4a arms (FIX CYCLE #4, team-lead
// directive after oracle-v3's live classification, task comment c-y140):
// the q8dp4a forms were comparing against the FULL-PRECISION reference
// above, which is the wrong oracle for them -- their activations are
// genuinely, expectedly lossy (Q8_1 quantization), so a reference computed
// from the unquantized f16 values necessarily disagrees even for a
// perfectly correct kernel. `q8_data` is READ BACK FROM THE DEVICE after
// quantize_q8_1_rows runs (not a shadow CPU requantization that could
// subtly diverge from the device kernel's exact rounding) -- so this
// validates only the GEMM math (dequant + dot + scale) given a common,
// already-quantized input, which is the right thing to test. Math mirrors
// gemm_soa_q8dp4a_kernel/gemm_tiled_q8dp4a_kernel exactly, but expressed
// directly over raw bytes instead of dp4a's int32-packed-4-bytes form
// (mathematically identical -- byte order within the int32 words dp4a
// consumes matches flat array order on this little-endian target, so
// there is no need to replicate the device's SPIR-V-only byte_level_permute
// intrinsic on host): weight nibble low/high -> kvalues, multiplied by
// the corresponding raw Q8_1 int8 byte, summed as a per-block integer,
// then scaled by (weight e8m0) * (activation Q8_1 d) and accumulated in
// double across blocks -- same per-block-scale-then-accumulate structure
// the device kernels use, just double instead of float for a tighter
// reference.
static void cpu_reference_gemm_soa_q8(const std::vector<int8_t> &  q8_data,
                                      int64_t                      row_bytes,
                                      const std::vector<uint8_t> & w_soa,
                                      std::vector<float> &         out,
                                      int64_t                      max_m,
                                      int64_t                      n,
                                      int64_t                      k,
                                      int                          blocks_per_row) {
    const int64_t nblocks = n * blocks_per_row;
    for (int64_t row = 0; row < max_m; ++row) {
        const int8_t *     q8_row = q8_data.data() + row * row_bytes;
        const sycl::half * ds_row = reinterpret_cast<const sycl::half *>(q8_row + k);
        for (int64_t col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int b = 0; b < blocks_per_row; ++b) {
                const int64_t   block_index = col * blocks_per_row + b;
                const uint8_t * qs          = w_soa.data() + block_index * 16;
                const double    d           = cpu_e8m0_to_fp32_half(w_soa[nblocks * 16 + block_index]);
                const double    d8          = static_cast<double>(ds_row[b * 2 + 0]);
                const int8_t *  q8_blk      = q8_row + b * QK_MXFP4;
                int64_t         sumi        = 0;
                for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    sumi += static_cast<int64_t>(q8_blk[j]) * kvalues_mxfp4_local[byte & 0xf];
                    sumi += static_cast<int64_t>(q8_blk[j + QK_MXFP4 / 2]) * kvalues_mxfp4_local[byte >> 4];
                }
                acc += d * d8 * static_cast<double>(sumi);
            }
            out[row * n + col] = static_cast<float>(acc);
        }
    }
}

static void cpu_reference_gemm_tiled_q8(const std::vector<int8_t> &  q8_data,
                                        int64_t                      row_bytes,
                                        const std::vector<uint8_t> & w_tiled,
                                        std::vector<float> &         out,
                                        int64_t                      max_m,
                                        int64_t                      n,
                                        int64_t                      k,
                                        int                          blocks_per_row,
                                        int64_t                      n_tile_groups_n,
                                        int64_t                      tile_n_total) {
    for (int64_t row = 0; row < max_m; ++row) {
        const int8_t *     q8_row = q8_data.data() + row * row_bytes;
        const sycl::half * ds_row = reinterpret_cast<const sycl::half *>(q8_row + k);
        for (int64_t col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int b = 0; b < blocks_per_row; ++b) {
                const int64_t   tg     = xmx_tile_group_offset(b, col, n_tile_groups_n, tile_n_total);
                const int64_t   tn     = col % tile_n_total;
                const double    d      = cpu_e8m0_to_fp32_half(w_tiled[tg + tn]);
                const double    d8     = static_cast<double>(ds_row[b * 2 + 0]);
                const uint8_t * qs     = w_tiled.data() + tg + tile_n_total + tn * 16;
                const int8_t *  q8_blk = q8_row + b * QK_MXFP4;
                int64_t         sumi   = 0;
                for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    sumi += static_cast<int64_t>(q8_blk[j]) * kvalues_mxfp4_local[byte & 0xf];
                    sumi += static_cast<int64_t>(q8_blk[j + QK_MXFP4 / 2]) * kvalues_mxfp4_local[byte >> 4];
                }
                acc += d * d8 * static_cast<double>(sumi);
            }
            out[row * n + col] = static_cast<float>(acc);
        }
    }
}

// Runs `work` ONCE untimed, wait_and_throw's (surfaces async JIT/launch
// failures instead of leaving stale output), copies the device `out` buffer
// back, and compares against `ref`'s first (m x n) rows. Prints PASS/FAIL
// with the observed error. Only calls report() -- i.e. only times it -- on
// PASS; a FAIL form prints no timing, per the directive ("a FAIL form
// prints no timing").
//
// FIX CYCLE #4 (team-lead directive after oracle-v3's live classification,
// task comment c-y140): PASS now requires abs_tol OR rel_tol, not rel_tol
// alone -- soa-f16 was failing on rel-err-on-near-zero-outputs (tiny
// absolute error, e.g. 0.006-0.008, but a large RELATIVE error because the
// reference value itself was close to zero for some (row,col) cells; a
// pure-relative metric is exactly the wrong tool for that case, not a sign
// of a kernel bug). `capture_out`, if non-null, receives a copy of
// host_out regardless of pass/fail, so callers can cross-check two
// independently-run forms' outputs against EACH OTHER (see the
// tiled-q8dp4a vs tiled-q8dp4a-jm cross-check below) -- a corroborating
// signal that is not itself a substitute for comparing against ref (per
// repo memory: adjacent green results can manufacture corroboration), so
// it is only ever printed alongside, never in place of, the oracle verdict.
template <typename F>
static bool validate_and_report(sycl::queue &              q,
                                const char *               form,
                                int64_t                    m,
                                int64_t                    n,
                                int64_t                    k,
                                float *                    dev_out,
                                const std::vector<float> & ref,
                                int64_t                    ref_stride,
                                double                     abs_tol,
                                double                     rel_tol,
                                int                        warmup,
                                int                        iters,
                                F &&                       work,
                                std::vector<float> *       capture_out = nullptr) {
    work(q);
    q.wait_and_throw();
    std::vector<float> host_out(static_cast<size_t>(m * n));
    q.memcpy(host_out.data(), dev_out, host_out.size() * sizeof(float)).wait_and_throw();
    if (capture_out) {
        *capture_out = host_out;
    }

    // NaN-SAFE max: FIX CYCLE #2 -- plain std::max(a, b), implemented as
    // (a<b)?b:a, silently KEEPS `a` whenever `b` is NaN (NaN compares false
    // against everything, so `a<NaN` is false). With max_abs_err/max_rel_err
    // initialized to 0.0, a run where EVERY comparison is NaN (which random
    // e8m0 bytes spanning the full uint8 range could produce -- see the fix
    // above) would leave both trackers at their 0.0 initial value the whole
    // loop, printing a false ORACLE=PASS with max_abs_err=0. `!(b <= a)` is
    // true whenever b is NaN (NaN<=a is false, so !false=true), so this
    // correctly latches NaN into the tracker instead of discarding it --
    // and pass = (max_rel_err <= rel_tol) is then correctly false once
    // max_rel_err is NaN, since NaN<=anything is false.
    auto nan_safe_max = [](double a, double b) {
        return !(b <= a) ? b : a;
    };

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            const double got     = host_out[static_cast<size_t>(row * n + col)];
            const double want    = ref[static_cast<size_t>(row * ref_stride + col)];
            const double abs_err = std::fabs(got - want);
            const double rel_err = abs_err / std::max(1e-6, std::fabs(want));
            max_abs_err          = nan_safe_max(max_abs_err, abs_err);
            max_rel_err          = nan_safe_max(max_rel_err, rel_err);
        }
    }
    // abs OR rel: a cell whose reference value is near zero can have a
    // huge relative error from a tiny, benign absolute error -- rel_tol
    // alone is the wrong metric for that cell, not evidence of a bug.
    const bool pass = (max_abs_err <= abs_tol) || (max_rel_err <= rel_tol);
    std::printf(
        "[GEMM-BENCH] form=%-24s m=%lld ORACLE=%s max_abs_err=%.6g max_rel_err=%.6g abs_tol=%.4g rel_tol=%.4g\n", form,
        (long long) m, pass ? "PASS" : "FAIL", max_abs_err, max_rel_err, abs_tol, rel_tol);
    if (!pass) {
        return false;
    }
    report(form, m, n, k, run_bench(q, warmup, iters, work));
    return true;
}

// Coarse sanity tolerances (see the block comment above this section for
// why these are sanity-level, not precision-level): f16 arms should match
// the double-precision reference almost exactly (float accumulation order
// differences only). GEMM_ORACLE_ABS_TOL covers the near-zero-output cells
// where a pure-relative metric is the wrong tool (oracle-v3 finding,
// task comment c-y140: soa-f16 abs err 0.006-0.008 but rel err 0.05-0.09 --
// a metric artifact, not a kernel defect).
//
// FIX CYCLE #4: the q8dp4a arms now compare against a Q8-AWARE reference
// (cpu_reference_gemm_{soa,tiled}_q8 below, computed from the SAME
// quantized bytes read back from the device -- not a full-precision
// reference), which removes the systematic Q8_1-quantization-vs-exactness
// gap the old GEMM_ORACLE_TOL_Q8=0.10 existed to paper over. With the
// right reference there is no more excusable error margin beyond ordinary
// floating-point reordering, so q8dp4a arms now use the SAME tight
// tolerances as the f16 arms -- a real bug in the q8dp4a kernels' math
// would now actually be caught, which the old (wrong-reference, loose-
// tolerance) combination could not do either way.
constexpr double GEMM_ORACLE_ABS_TOL = 0.01;
constexpr double GEMM_ORACLE_TOL_F16 = 0.02;
constexpr double GEMM_ORACLE_TOL_Q8  = 0.02;

int main() {
    // Queue construction failure (no SYCL GPU device) is a legitimate SKIP
    // (rc=77) -- but nothing AFTER construction should share that catch: a
    // sycl::exception raised later by wait_and_throw (a real launch/JIT
    // failure, or the async_handler's rethrow) is a genuine bug, not the
    // absence of a device, and must be reported as a distinct failure
    // (rc=1) rather than misreported as the same benign "no GPU" skip.
    std::optional<sycl::queue> q_opt;
    try {
        // in_order + an async_handler that rethrows: fixes the defect that
        // produced physically-impossible GFLOPs on the first jm run (task
        // comment on llama.cpp-iikr) -- an out-of-order queue submits the
        // begin marker / work / end marker as three INDEPENDENT command
        // groups with no dependency edge between them, so t1-t0 could
        // measure marker-to-marker latency while the timed work was still
        // in flight (or had failed to launch at all). in_order makes
        // submission order a real ordering guarantee; the async_handler
        // (paired with wait_and_throw calls below) surfaces a silently
        // failed launch as a thrown, printed exception instead of stale
        // output read back as if it were real.
        auto async_handler = [](sycl::exception_list exceptions) {
            for (const std::exception_ptr & e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch (const sycl::exception & ex) {
                    std::fprintf(stderr, "[GEMM-BENCH] ASYNC SYCL EXCEPTION: %s\n", ex.what());
                    std::exit(1);
                }
            }
        };
        q_opt.emplace(
            sycl::gpu_selector_v, async_handler,
            sycl::property_list{ sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{} });
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
    sycl::queue & q = *q_opt;

    // Everything from here on is a real run against a real device: a
    // sycl::exception past this point (from wait_and_throw, or the
    // async_handler's rethrow) is a genuine failure and must be reported
    // as one, not folded into the "no GPU" SKIP path above.
    try {
        // Diagnostic per the directive: print what this device actually
        // supports, so an unsupported-shape refusal is visible in the log
        // rather than inferred. sub_group_sizes is a standard, stable SYCL
        // device query; DPC++ does not expose a stable joint_matrix
        // supported-combination query in this version, so that capability
        // is NOT introspected here -- instead, any actual capability
        // mismatch at a jm kernel's launch now surfaces as a real thrown
        // sycl::exception via the async_handler above (it did not before
        // this fix), which is the property that actually matters.
        {
            const std::vector<size_t> sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
            std::printf("[GEMM-BENCH] device sub_group_sizes=[");
            for (size_t i = 0; i < sg_sizes.size(); ++i) {
                std::printf("%s%zu", i ? "," : "", sg_sizes[i]);
            }
            std::printf("] name=\"%s\"\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        }

        // GPT-OSS 20B expert shape -- see task llama.cpp-0vqt / llama.cpp-iikr
        // for the derivation (hidden_size == intermediate_size == 2880).
        constexpr int     blocks_per_row = 90;
        constexpr int64_t N              = 2880;                                 // ne01 (output dim)
        constexpr int64_t K              = (int64_t) blocks_per_row * QK_MXFP4;  // 2880
        // realistic gemm_group row counts (align_rows_64-padded groups; see
        // ggml-sycl.cpp's pp_gemm_group builder)
        const int         Ms[]           = { 32, 64, 128 };
        constexpr int     WARMUP         = 3;
        constexpr int     ITERS          = 10;

        const int64_t nblocks         = N * blocks_per_row;
        const size_t  soa_bytes       = (size_t) (nblocks * 16 + nblocks);
        const int64_t n_tile_groups_n = (N + TILE_N_TOTAL - 1) / TILE_N_TOTAL;
        const size_t  tiled_bytes     = (size_t) (blocks_per_row * n_tile_groups_n * TILE_N_TOTAL * 17);

        std::mt19937                       rng(1729);
        std::uniform_int_distribution<int> nibble_dist(0, 255);
        // FIX CYCLE #2 (task comment on llama.cpp-iikr, second invalidation):
        // scale (e8m0) bytes were previously drawn from the SAME flat
        // uniform(0,255) as nibble bytes. e8m0=255 maps (via the "halved"
        // convention, cpu_e8m0_to_fp32_half/sycl_e8m0_to_fp32_half) to a
        // FINITE but ~1.7e38 scale -- close enough to float32's ~3.4e38 max
        // that a handful of blocks per (row,col) dot product (blocks_per_row
        // = 90 independent random draws) reliably overflowed the float32
        // accumulator to +/-inf, and summing opposite-signed infs produced
        // NaN. That NaN then poisoned validate_and_report's error tracking
        // (see the NaN-safe rewrite below) -- this is what produced the
        // impossible ORACLE=PASS max_abs_err=0 / ORACLE=FAIL max_abs_err=inf
        // pairing, not a device/reference divergence.
        //
        // FIX CYCLE #3 (team-lead directive after the positive control
        // itself FAILED -- "positive control (soa) found NO oracle
        // sensitivity"): the FIRST narrowing to e in [100,160] fixed
        // OVERFLOW but not a SEPARATE effect -- DYNAMIC-RANGE SWAMPING.
        // [100,160] is still a 60-exponent spread (~2^60, ~1e18x). With 90
        // INDEPENDENT random e8m0 draws per dot product, the sum is
        // overwhelmingly dominated by whichever 1-2 blocks happen to draw
        // the largest scale; a block with a much smaller scale (e.g. the
        // very block a positive control perturbs) can contribute LESS than
        // double precision's ~15-17 significant decimal digits can
        // represent relative to the dominant term -- its true contribution
        // is not wrong, it is genuinely below the sum's representable
        // resolution. This is NOT a comparison-logic bug (unlike the NaN
        // issue above); it is real floating-point swamping from a spread
        // still too wide for THIS synthetic test's purpose. Verified with a
        // standalone host-only probe (perturb + recompute in plain C++, no
        // SYCL/GPU needed, so runnable directly rather than only inferred):
        // sweeping candidate ranges confirmed [100,160] gives an EXACT
        // diff=0 at (0,0) for the soa layout, matching the failure exactly,
        // while [124,132] gives a robust, comfortably nonzero diff checked
        // across 5 independent RNG seeds (smallest observed |diff| was
        // 0.079 against reference values on the order of tens to
        // thousands). e in [124,132] keeps scales in roughly [2^-4, 2^4]
        // (~0.06 to ~16) -- comfortably narrow enough that no block's
        // contribution is lost to another's, while still exercising real
        // per-block scale variation (not a single fixed value) and
        // remaining nowhere near float32 overflow.
        std::uniform_int_distribution<int> scale_dist(124, 132);

        // Host copies are RETAINED (not discarded after upload) -- the CPU
        // reference oracle below needs the exact same random bytes the
        // device kernels read.
        std::vector<uint8_t> host_w_soa(soa_bytes);
        std::vector<uint8_t> host_w_tiled(tiled_bytes);
        {
            // SOA layout: [nblocks*16 nibble bytes][nblocks scale bytes].
            for (int64_t i = 0; i < nblocks * 16; ++i) {
                host_w_soa[static_cast<size_t>(i)] = (uint8_t) nibble_dist(rng);
            }
            for (int64_t i = 0; i < nblocks; ++i) {
                host_w_soa[static_cast<size_t>(nblocks * 16 + i)] = (uint8_t) scale_dist(rng);
            }
            // XMX_TILED layout: n_groups tile groups, each
            // [tile_n_total scale bytes][tile_n_total*16 nibble bytes] --
            // see the layout geometry comment near xmx_tile_group_offset.
            const int64_t n_groups        = (int64_t) blocks_per_row * n_tile_groups_n;
            const int64_t bytes_per_group = TILE_N_TOTAL * 17;
            for (int64_t g = 0; g < n_groups; ++g) {
                uint8_t * grp = host_w_tiled.data() + g * bytes_per_group;
                for (int i = 0; i < TILE_N_TOTAL; ++i) {
                    grp[i] = (uint8_t) scale_dist(rng);
                }
                for (int i = 0; i < TILE_N_TOTAL * 16; ++i) {
                    grp[TILE_N_TOTAL + i] = (uint8_t) nibble_dist(rng);
                }
            }
        }
        auto upload = [&](const std::vector<uint8_t> & host) -> uint8_t * {
            uint8_t * dev = (uint8_t *) sycl::malloc_device(host.size(), q);
            q.memcpy(dev, host.data(), host.size()).wait_and_throw();
            return dev;
        };
        uint8_t * w_soa   = upload(host_w_soa);
        uint8_t * w_tiled = upload(host_w_tiled);

        std::printf(
            "[GEMM-BENCH] shape blocks_per_row=%d N=%lld K=%lld tile_n_total=%d soa_bytes=%zu tiled_bytes=%zu\n",
            blocks_per_row, (long long) N, (long long) K, TILE_N_TOTAL, soa_bytes, tiled_bytes);

        const int64_t           max_m   = 128;
        sycl::half *            act_f16 = (sycl::half *) sycl::malloc_device(sizeof(sycl::half) * max_m * K, q);
        std::vector<sycl::half> host_act_f16(static_cast<size_t>(max_m * K));
        {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto & v : host_act_f16) {
                v = sycl::half(dist(rng));
            }
            q.memcpy(act_f16, host_act_f16.data(), host_act_f16.size() * sizeof(sycl::half)).wait_and_throw();
        }
        const size_t q8_row_bytes = (size_t) (K + blocks_per_row * 4);
        int8_t *     act_q8       = (int8_t *) sycl::malloc_device(q8_row_bytes * max_m, q);
        float *      out          = (float *) sycl::malloc_device(sizeof(float) * max_m * N, q);

        // CPU reference GEMMs, computed ONCE (double precision, from the
        // exact host bytes above) for the FULL max_m=128 rows -- each real
        // M in Ms below just compares against a PREFIX of these, since
        // neither weight buffer depends on M and the activation buffer is
        // shared across all M values too.
        std::printf("[GEMM-BENCH] computing CPU reference GEMMs (max_m=%lld, this takes a few seconds)...\n",
                    (long long) max_m);
        std::vector<float> soa_ref(static_cast<size_t>(max_m * N));
        std::vector<float> tiled_ref(static_cast<size_t>(max_m * N));
        cpu_reference_gemm_soa(host_act_f16, host_w_soa, soa_ref, max_m, N, K, blocks_per_row);
        cpu_reference_gemm_tiled(host_act_f16, host_w_tiled, tiled_ref, max_m, N, K, blocks_per_row, n_tile_groups_n,
                                 TILE_N_TOTAL);
        std::printf("[GEMM-BENCH] CPU reference GEMMs done.\n");

        // POSITIVE CONTROL (team-lead directive, fix cycle #2, strengthened
        // in fix cycle #3 after the ORIGINAL single-element form itself
        // FAILED -- "positive control (soa) found NO oracle sensitivity",
        // diff=0 exactly): prove the oracle can actually DETECT a
        // difference BEFORE trusting any PASS/FAIL verdict below.
        //
        // The single-element check ((row 0, col 0) only) could not tell
        // apart two very different explanations for diff=0: (a) a control
        // bug -- the perturbed byte doesn't feed element (0,0) at all under
        // the true layout, so diff=0 is EXPECTED even with a healthy
        // oracle; or (b) an oracle bug -- the reference's layout decode
        // never reads that byte, so every comparison it makes is against
        // wrong math. Root cause turned out to be neither: it was dynamic-
        // range swamping in the TEST DATA (see the scale_dist comment
        // above) -- but (a) and (b) are real failure modes a future change
        // could reintroduce, and the single-element check is blind to both.
        // Upgraded to team-lead's "perturb-and-scan": recompute the WHOLE
        // row from the perturbed copy and report which columns changed. The
        // perturbed byte is block b=0's first byte for column 0 under BOTH
        // layouts' definitions (SOA: block_index=0 => col=0,b=0's first
        // nibble byte; XMX_TILED: tile group (b=0,tg_n=0)'s first byte,
        // which is a SCALE byte since tiled groups are scales-then-nibbles)
        // -- so a healthy oracle on healthy data must report EXACTLY ONE
        // changed column, and it must be column 0. Any other outcome
        // (zero changed columns => oracle bug per (b); more than one, or a
        // different column => control-siting bug per (a), or a genuine
        // layout-math error) aborts loudly instead of silently passing.
        {
            auto positive_control = [&](const char * layout_name, const std::vector<uint8_t> & host_w, bool is_tiled) {
                std::vector<uint8_t> perturbed = host_w;
                perturbed[0] ^= 0xFF;
                std::vector<float> original_row0(static_cast<size_t>(N));
                std::vector<float> perturbed_row0(static_cast<size_t>(N));
                if (is_tiled) {
                    cpu_reference_gemm_tiled(host_act_f16, host_w, original_row0, 1, N, K, blocks_per_row,
                                             n_tile_groups_n, TILE_N_TOTAL);
                    cpu_reference_gemm_tiled(host_act_f16, perturbed, perturbed_row0, 1, N, K, blocks_per_row,
                                             n_tile_groups_n, TILE_N_TOTAL);
                } else {
                    cpu_reference_gemm_soa(host_act_f16, host_w, original_row0, 1, N, K, blocks_per_row);
                    cpu_reference_gemm_soa(host_act_f16, perturbed, perturbed_row0, 1, N, K, blocks_per_row);
                }
                int64_t changed_count = 0;
                int64_t first_changed = -1;
                for (int64_t col = 0; col < N; ++col) {
                    if (original_row0[col] != perturbed_row0[col]) {
                        ++changed_count;
                        if (first_changed < 0) {
                            first_changed = col;
                        }
                    }
                }
                const double diff0 =
                    std::fabs(static_cast<double>(perturbed_row0[0]) - static_cast<double>(original_row0[0]));
                std::printf(
                    "[GEMM-BENCH] positive control (%s): changed_cols=%lld first_changed=%lld diff[0]=%.6g "
                    "original[0]=%.6g perturbed[0]=%.6g\n",
                    layout_name, (long long) changed_count, (long long) first_changed, diff0, original_row0[0],
                    perturbed_row0[0]);
                if (changed_count != 1 || first_changed != 0) {
                    std::fprintf(stderr,
                                 "[GEMM-BENCH] FATAL: positive control (%s) expected EXACTLY column 0 to change, "
                                 "got changed_cols=%lld first_changed=%lld -- changed_cols=0 means the reference "
                                 "never reads the perturbed byte (oracle/layout-decode bug); any other count or "
                                 "column means the perturbation site or the layout math is wrong. Refusing to "
                                 "trust any PASS/FAIL below.\n",
                                 layout_name, (long long) changed_count, (long long) first_changed);
                    std::exit(1);
                }
            };
            positive_control("soa", host_w_soa, false);
            positive_control("tiled", host_w_tiled, true);
        }

        for (int m : Ms) {
            // Activation quant step, timed separately -- q8dp4a arm's cost
            // that must not be hidden.
            const bench_result quant_r = run_bench(q, WARMUP, ITERS, [&](sycl::queue & qq) {
                quantize_q8_1_rows(qq, act_f16, act_q8, m, K, blocks_per_row);
            });
            std::printf("[GEMM-BENCH] form=%-24s m=%d quant_only_mean_ms=%.4f quant_only_min_ms=%.4f\n",
                        "q8_1-quantize", m, quant_r.mean_ms, quant_r.min_ms);
            // Materialize the quantized activation once for the GEMM-only
            // timing loop below (mirrors production: quantize once, GEMM
            // many times against the same activation).
            quantize_q8_1_rows(q, act_f16, act_q8, m, K, blocks_per_row).wait_and_throw();

            // FIX CYCLE #4: read back the DEVICE's own quantized bytes for
            // rows [0,m) and build this m's Q8-aware references from them
            // (see cpu_reference_gemm_{soa,tiled}_q8 above for why this is
            // the right oracle for the q8dp4a arms). Readback+reference
            // cost here (~1e8-1e9 double-precision ops) is a diagnostic
            // expense, not a hot path -- fine to pay once per m.
            std::vector<int8_t> host_act_q8(static_cast<size_t>(m) * q8_row_bytes);
            q.memcpy(host_act_q8.data(), act_q8, host_act_q8.size()).wait_and_throw();
            std::vector<float> soa_ref_q8(static_cast<size_t>(m * N));
            std::vector<float> tiled_ref_q8(static_cast<size_t>(m * N));
            cpu_reference_gemm_soa_q8(host_act_q8, static_cast<int64_t>(q8_row_bytes), host_w_soa, soa_ref_q8, m, N, K,
                                      blocks_per_row);
            cpu_reference_gemm_tiled_q8(host_act_q8, static_cast<int64_t>(q8_row_bytes), host_w_tiled, tiled_ref_q8, m,
                                        N, K, blocks_per_row, n_tile_groups_n, TILE_N_TOTAL);

            // Populated by the plain and jm tiled-q8dp4a forms below for
            // the cross-check (declared unconditionally since the plain
            // form exists outside the SYCL_XMX_JM_AVAILABLE guard).
            std::vector<float> tiled_q8dp4a_plain_out;
            std::vector<float> tiled_q8dp4a_jm_out;

            const int64_t n_tiles = (N + WG_SIZE - 1) / WG_SIZE;

            {
                auto work = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(K), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<2>(sycl::range<2>(m, n_tiles * WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                            [=](sycl::nd_item<2> item) {
                                gemm_soa_f16_kernel(w_soa, nblocks, act_f16, out, K, N, blocks_per_row,
                                                    get_pointer(slm), item);
                            });
                    });
                };
                validate_and_report(q, "soa-f16", m, N, K, out, soa_ref, N, GEMM_ORACLE_ABS_TOL, GEMM_ORACLE_TOL_F16,
                                    WARMUP, ITERS, work);
            }
            {
                auto work = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<int8_t, 1>     slm_qs(sycl::range<1>(K), cgh);
                        sycl::local_accessor<sycl::half, 1> slm_ds(sycl::range<1>(blocks_per_row * 2), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<2>(sycl::range<2>(m, n_tiles * WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                            [=](sycl::nd_item<2> item) {
                                gemm_soa_q8dp4a_kernel(w_soa, nblocks, act_q8, out, K, N, blocks_per_row,
                                                       get_pointer(slm_qs), get_pointer(slm_ds), item);
                            });
                    });
                };
                validate_and_report(q, "soa-q8dp4a-gemm-only", m, N, K, out, soa_ref_q8, N, GEMM_ORACLE_ABS_TOL,
                                    GEMM_ORACLE_TOL_Q8, WARMUP, ITERS, work);
            }
            {
                auto work = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(K), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<2>(sycl::range<2>(m, n_tiles * WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                            [=](sycl::nd_item<2> item) {
                                gemm_tiled_f16_kernel(w_tiled, n_tile_groups_n, TILE_N_TOTAL, act_f16, out, K, N,
                                                      blocks_per_row, get_pointer(slm), item);
                            });
                    });
                };
                validate_and_report(q, "tiled-f16", m, N, K, out, tiled_ref, N, GEMM_ORACLE_ABS_TOL,
                                    GEMM_ORACLE_TOL_F16, WARMUP, ITERS, work);
            }
            {
                auto work = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<int8_t, 1>     slm_qs(sycl::range<1>(K), cgh);
                        sycl::local_accessor<sycl::half, 1> slm_ds(sycl::range<1>(blocks_per_row * 2), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<2>(sycl::range<2>(m, n_tiles * WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                            [=](sycl::nd_item<2> item) {
                                gemm_tiled_q8dp4a_kernel(w_tiled, n_tile_groups_n, TILE_N_TOTAL, act_q8, out, K, N,
                                                         blocks_per_row, get_pointer(slm_qs), get_pointer(slm_ds),
                                                         item);
                            });
                    });
                };
                // Captured for the plain-vs-jm cross-check below (team-lead
                // ask, c-y140): the byte-identical error maxima team-lead
                // observed between this form and tiled-q8dp4a-jm against
                // the WRONG reference is corroborating, not conclusive on
                // its own -- comparing their actual outputs directly is the
                // stronger check, done once both forms have run.
                validate_and_report(q, "tiled-q8dp4a-gemm-only", m, N, K, out, tiled_ref_q8, N, GEMM_ORACLE_ABS_TOL,
                                    GEMM_ORACLE_TOL_Q8, WARMUP, ITERS, work, &tiled_q8dp4a_plain_out);
            }
#if SYCL_XMX_JM_AVAILABLE
            // joint_matrix (DPAS) arms -- team-lead directive, task comment
            // c-0xub. m must be a multiple of XMX_JM_M=8; all three Ms above
            // (32/64/128) are.
            {
                const int64_t n_tiles_jm = N / XMX_JM_WG_SIZE;  // 2880/64 = 45, exact
                auto          work       = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<sycl::half, 1> slm_act(sycl::range<1>(XMX_JM_M * XMX_JM_K_F16), cgh);
                        sycl::local_accessor<sycl::half, 1> slm_w(
                            sycl::range<1>(XMX_JM_NUM_SG * XMX_JM_N * XMX_JM_K_F16), cgh);
                        sycl::local_accessor<float, 1> slm_out(sycl::range<1>(XMX_JM_NUM_SG * XMX_JM_M * XMX_JM_N),
                                                                              cgh);
                        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(m / XMX_JM_M, n_tiles_jm * XMX_JM_WG_SIZE),
                                                                          sycl::range<2>(1, XMX_JM_WG_SIZE)),
                                                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_JM_SG)]] {
                                             gemm_tiled_f16_jm_kernel(w_tiled, n_tile_groups_n, TILE_N_TOTAL, act_f16,
                                                                                     out, K, N, blocks_per_row, get_pointer(slm_act),
                                                                                     get_pointer(slm_w), get_pointer(slm_out), item);
                                         });
                    });
                };
                validate_and_report(q, "tiled-f16-jm", m, N, K, out, tiled_ref, N, GEMM_ORACLE_ABS_TOL,
                                    GEMM_ORACLE_TOL_F16, WARMUP, ITERS, work);
            }
            {
                const int64_t n_tiles_jm = N / XMX_JM_WG_SIZE;
                auto          work       = [&](sycl::queue & qq) {
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<int8_t, 1>  slm_act(sycl::range<1>(XMX_JM_M * XMX_JM_K_I8), cgh);
                        sycl::local_accessor<float, 1>   slm_act_d(sycl::range<1>(XMX_JM_M), cgh);
                        sycl::local_accessor<int8_t, 1>  slm_w(sycl::range<1>(XMX_JM_NUM_SG * XMX_JM_N * XMX_JM_K_I8),
                                                                              cgh);
                        sycl::local_accessor<int32_t, 1> slm_raw(sycl::range<1>(XMX_JM_NUM_SG * XMX_JM_M * XMX_JM_N),
                                                                                cgh);
                        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(m / XMX_JM_M, n_tiles_jm * XMX_JM_WG_SIZE),
                                                                          sycl::range<2>(1, XMX_JM_WG_SIZE)),
                                                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_JM_SG)]] {
                                             gemm_tiled_q8dp4a_jm_kernel(
                                                 w_tiled, n_tile_groups_n, TILE_N_TOTAL, act_q8, out, K, N,
                                                 blocks_per_row, get_pointer(slm_act), get_pointer(slm_act_d),
                                                 get_pointer(slm_w), get_pointer(slm_raw), item);
                                         });
                    });
                };
                validate_and_report(q, "tiled-q8dp4a-jm-gemm-only", m, N, K, out, tiled_ref_q8, N, GEMM_ORACLE_ABS_TOL,
                                    GEMM_ORACLE_TOL_Q8, WARMUP, ITERS, work, &tiled_q8dp4a_jm_out);
            }

            // Cross-check (team-lead ask, c-y140): compare the plain and jm
            // tiled-q8dp4a forms' ACTUAL outputs directly, not just their
            // errors against the reference -- this is a stronger version of
            // the byte-identical-error-maxima signature that first pointed
            // at shared quantization error rather than independent bugs.
            // Runs whenever both captured (i.e. both forms actually
            // produced output, regardless of PASS/FAIL against the
            // reference -- capture_out is filled unconditionally).
            if (!tiled_q8dp4a_plain_out.empty() && !tiled_q8dp4a_jm_out.empty()) {
                double max_cross_diff = 0.0;
                for (size_t i = 0; i < tiled_q8dp4a_plain_out.size(); ++i) {
                    const double d = std::fabs(static_cast<double>(tiled_q8dp4a_plain_out[i]) -
                                               static_cast<double>(tiled_q8dp4a_jm_out[i]));
                    if (d > max_cross_diff) {
                        max_cross_diff = d;
                    }
                }
                std::printf("[GEMM-BENCH] cross-check tiled-q8dp4a plain-vs-jm m=%d max_diff=%.6g (%s)\n", m,
                            max_cross_diff, max_cross_diff < GEMM_ORACLE_ABS_TOL ? "AGREE" : "DISAGREE");
            }
#endif
        }

        sycl::free(w_soa, q);
        sycl::free(w_tiled, q);
        sycl::free(act_f16, q);
        sycl::free(act_q8, q);
        sycl::free(out, q);
        return 0;
    } catch (const sycl::exception & ex) {
        std::fprintf(stderr, "[GEMM-BENCH] FATAL SYCL EXCEPTION (device acquired, run failed): %s\n", ex.what());
        return 1;
    }
}
