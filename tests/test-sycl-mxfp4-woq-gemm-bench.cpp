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

#include <cstdint>
#include <cstdio>
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
// exactly (plain row-major f16, K contiguous) -- this bench synthesizes
// that format directly (random f16) rather than re-deriving the f32->f16
// copy kernel, since this bench is scoped to the GEMM's own compute, not
// the (already free, already-shipped) activation copy step.
static void fill_random_f16_act(sycl::queue & q, sycl::half * act, int64_t m, int64_t k, std::mt19937 & rng) {
    std::vector<sycl::half>               host(static_cast<size_t>(m * k));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : host) {
        v = sycl::half(dist(rng));
    }
    q.memcpy(act, host.data(), host.size() * sizeof(sycl::half)).wait();
}

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

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

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
        std::uniform_int_distribution<int> byte_dist(0, 255);

        auto fill_random_device = [&](size_t bytes) -> uint8_t * {
            std::vector<uint8_t> host(bytes);
            for (auto & b : host) {
                b = (uint8_t) byte_dist(rng);
            }
            uint8_t * dev = (uint8_t *) sycl::malloc_device(bytes, q);
            q.memcpy(dev, host.data(), bytes).wait();
            return dev;
        };

        uint8_t * w_soa   = fill_random_device(soa_bytes);
        uint8_t * w_tiled = fill_random_device(tiled_bytes);

        std::printf(
            "[GEMM-BENCH] shape blocks_per_row=%d N=%lld K=%lld tile_n_total=%d soa_bytes=%zu tiled_bytes=%zu\n",
            blocks_per_row, (long long) N, (long long) K, TILE_N_TOTAL, soa_bytes, tiled_bytes);

        const int64_t max_m   = 128;
        sycl::half *  act_f16 = (sycl::half *) sycl::malloc_device(sizeof(sycl::half) * max_m * K, q);
        fill_random_f16_act(q, act_f16, max_m, K, rng);
        const size_t q8_row_bytes = (size_t) (K + blocks_per_row * 4);
        int8_t *     act_q8       = (int8_t *) sycl::malloc_device(q8_row_bytes * max_m, q);
        float *      out          = (float *) sycl::malloc_device(sizeof(float) * max_m * N, q);

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
            quantize_q8_1_rows(q, act_f16, act_q8, m, K, blocks_per_row).wait();

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
                report("soa-f16", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
                report("soa-q8dp4a-gemm-only", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
                report("tiled-f16", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
                report("tiled-q8dp4a-gemm-only", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
                report("tiled-f16-jm", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
                report("tiled-q8dp4a-jm-gemm-only", m, N, K, run_bench(q, WARMUP, ITERS, work));
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
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
