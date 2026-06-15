//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_XMX_F16_V2_HPP
#define GGML_SYCL_FATTN_XMX_F16_V2_HPP

// =============================================================================
// XMX-v2 flash-attention kernel — structurally correct port of CUDA fattn-mma-f16.cuh
//
// Design contract (plan §3.3, ten rules):
//
//   1. No SLM buffer aliasing — tile_Q/tile_K/tile_V/tile_S each occupy a
//      separate, non-overlapping SLM region for the kernel's lifetime.
//   2. All cross-lane softmax state in registers, updated via
//      sycl::reduce_over_group(sg, _, maximum/plus<float>{}).
//      No SLM batch_max_shared, no SLM sum_shared.
//   3. One group_barrier per SLM phase transition. No barrier on aliased regions
//      (there are none).
//   4. Zero overlap between producer/consumer SLM regions.
//   5. No lane-order-dependent computation beyond XMX/sub-group guarantees.
//      QK extracted via joint_matrix_apply (canary-5 pattern), NOT via
//      joint_matrix_store + scalar SLM read.
//   6. Fixed KV tile stride — does not depend on ncols.
//   7. [[sycl::reqd_sub_group_size(XMX_V2_SG)]] on the kernel.
//   8. FTZ rescale via __builtin_memcpy bit-trick (matches fattn-vec-f16.hpp).
//   9. Sinks applied AFTER the KV loop, register-only (CUDA §1027-1082 pattern).
//  10. Mask loaded per-KV-tile with explicit stride — no precomputed per-query-range logic.
//
// Rule 10 of plan §3.3 — tile shape IS hardware-discovered: matrix_combinations
// is queried at runtime, and the picker dispatches to a compile-time specialization
// whose (M, K, N) matches what the device actually supports. If the query fails or
// no variant matches, we fall back to the (M=8, K=16, N=16) leaf which is validated
// on B580 and B50 today.
//
// Implemented by llama.cpp-1a038 (closed): runtime matrix_combinations query
// with a single (8,16,16) leaf active today. The variant array is sized for
// N>1 so adding future leaves (e.g. TM=1/TK=64/TN=64 TG fast path) is an
// explicit array entry plus a matching leaf specialization. See
// `fattn_xmx_v2_variants[]` below.
// =============================================================================

#include "fattn-common.hpp"
#include "fattn.hpp"

#include <array>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define SYCL_XMX_V2_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define SYCL_XMX_V2_AVAILABLE 0
#endif

#if SYCL_XMX_V2_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// Compile-time configuration (shared by every variant)
// =============================================================================

// Sub-group size — Xe2 confirmed (canary 1). Shared by every variant because
// sub-group collectives are parameterized at kernel level, not per-variant.
static constexpr int XMX_V2_SG = 16;

// Historical maximum work-group size. The leaf now launches only as many
// sub-groups as own unique query rows for the compile-time (ncols, TM) shape.
// Keeping the bound makes the intended resource ceiling explicit.
static constexpr int XMX_V2_MAX_NTHREADS = 512;

// KV batch size — number of KV rows processed per outer-loop iteration.
// SLM budget: (ncols*D + 2*BATCH_KV*D + 2*ncols*BATCH_KV) * sizeof(half).
// For ncols=8, D=128, BATCH_KV=32: (8+64)*128*2 + 2*8*32*2 = 18432 + 1024 = 19456 bytes — within 64 KB.
// llama.cpp-0sres (closed) added the runtime SLM-fit check in
// `fattn_xmx_v2_pick_variant_cached`: if the fixed BATCH_KV overflows the device's
// reported local_mem_size for the requested shape, the picker returns use_xmx=false
// and the dispatcher falls back to the TILE kernel. BATCH_KV itself remains a
// compile-time constant; only the gate is runtime.
static constexpr int XMX_V2_BATCH_KV = 32;  // must be divisible by every variant's TK and TN
static constexpr int XMX_V2_FLOAT_AS_HALF_ELEMS =
    static_cast<int>((sizeof(float) + sizeof(sycl::half) - 1) / sizeof(sycl::half));

// Decode prototype shape for GPT-OSS D=64 TG.  Xe2 reports fp16
// M=1,K=16,N=64, but with SG16 + ext_intel_packed B only lanes 0..7 expose
// useful accumulator values.  The decode leaf therefore uses two packed MADs
// per 64-token KV tile, each covering a compact 32-token half.
static constexpr int XMX_V2_DECODE_TK           = 16;
static constexpr int XMX_V2_DECODE_TN           = 64;
static constexpr int XMX_V2_DECODE_BATCH_KV     = 64;
static constexpr int XMX_V2_DECODE_ACTIVE_LANES = 8;
static constexpr int XMX_V2_DECODE_SLOTS        = XMX_V2_DECODE_TN / XMX_V2_SG;
static constexpr int XMX_V2_DECODE_HALF_KV      = XMX_V2_DECODE_ACTIVE_LANES * XMX_V2_DECODE_SLOTS;
static constexpr int XMX_V2_DECODE_GQA_MAX      = 8;
static_assert(XMX_V2_DECODE_BATCH_KV == 2 * XMX_V2_DECODE_HALF_KV,
              "Decode M=1,N=64 prototype expects two compact 32-token halves");

// =============================================================================
// Variant catalogue — the compile-time-instantiated universe of (TM, TK, TN)
// leaf kernels. Today only ONE leaf is active; the array is sized for growth.
//
// The leaf that runs is chosen at runtime by `fattn_xmx_v2_pick_variant()` from
// the device's `matrix_combinations` list. If the query fails or no entry in the
// array matches any reported combination, the picker returns index 0 (the
// always-compiled fallback leaf `(TM=8, TK=16, TN=16)` — validated on B580/B50
// via canary 1).
//
// Adding a new leaf:
//   1. Append a `variant_info{...}` entry to `fattn_xmx_v2_variants[]`.
//   2. Add a matching `case` branch to `launch_fattn_xmx_v2_f16()` that
//      invokes `launch_fattn_xmx_v2_f16_leaf<D, ncols, ..., TM, TK, TN>`
//      with the concrete tile constants.
//   3. Add the kernel-body specialization if (TM, TK, TN) differs from (8,16,16)
//      — the current body's lane-ownership math is tied to the fallback shape.
// =============================================================================

struct variant_info {
    int tm;
    int tk;
    int tn;
    // elems_per_lane is how many accumulator floats each sub-group lane owns
    // after one M×N joint_matrix_mad. For the canonical column-striped layout
    // (lane l owns column l, rows 0..TM-1) this equals TM. Kept explicit so a
    // future variant with a different accumulator fragment layout (e.g. a
    // row-striped TM=1/TK=64 leaf) can override it without confusing readers.
    int elems_per_lane;
};

// Leaf 0: (8, 16, 16) — always available, validated on Xe2 (B580, B50).
// Matches matrix_combinations entry `0x16x16 max=8x0x0` with fp16 A/B and fp32 C/D.
//
// Leaf 1: (16, 16, 16) — ncols>=16 path, matches `16x16x16 fp16→fp32`
//   SLM for D=128, nc=16: Q=4096 + K=4096 + V=4096 + S=512 = 12800 halves = 25 KB
//   Fits B580's 128 KB SLM for nc<=8; for nc>8 uses nc=16 SLM budget.
//
// Triple-brace init: std::array<T, N> wraps a single-element C-array; the outer
// pair opens the array, the middle opens the C-array, and the innermost opens
// the variant_info aggregate.
static constexpr std::array<variant_info, 2> fattn_xmx_v2_variants = {
    { { 8, 16, 16, 8 }, { 16, 16, 16, 16 } }
};

static_assert(fattn_xmx_v2_variants[0].tm == 8 && fattn_xmx_v2_variants[0].tk == 16 &&
                  fattn_xmx_v2_variants[0].tn == 16,
              "Fallback leaf must remain at index 0 as (8,16,16)");

// Backwards-compat constants that name the fallback leaf's dimensions.
// These are the ONLY hardcoded tile values left in the kernel body (per Rule 10
// of the plan: fallback tile shape must remain concrete and deterministic).
static constexpr int XMX_V2_TM             = fattn_xmx_v2_variants[0].tm;
static constexpr int XMX_V2_TK             = fattn_xmx_v2_variants[0].tk;
static constexpr int XMX_V2_TN             = fattn_xmx_v2_variants[0].tn;
static constexpr int XMX_V2_ELEMS_PER_LANE = fattn_xmx_v2_variants[0].elems_per_lane;

static_assert(XMX_V2_BATCH_KV % XMX_V2_TK == 0, "BATCH_KV must be divisible by fallback TK");
static_assert(XMX_V2_BATCH_KV % XMX_V2_TN == 0, "BATCH_KV must be divisible by fallback TN");

// Pin the column-striped lane-ownership invariant. If a future variant switches
// to a different accumulator fragment layout (e.g. row-striped for TM=1), its
// `elems_per_lane` will differ and this assert should be relaxed per-variant.
static_assert(XMX_V2_ELEMS_PER_LANE == 8, "Fallback leaf's column-striped layout owns 8 floats per lane (TM=8)");

// =============================================================================
// Runtime matrix_combinations query + variant picker
//
// Called on first use per ggml_backend_sycl_context via `fattn_xmx_v2_pick_variant_cached`
// (per-context cache wired in llama.cpp-0sres, closed — one cache entry per device
// since each sycl context binds to a single device). Queries the device's reported
// `matrix_combinations`, scores each variant in `fattn_xmx_v2_variants` against
// every reported combo, and returns the winning variant index. Returns 0 (the
// fallback leaf) if:
//   - the device query throws
//   - no combo matches any variant (zero score)
//   - the matrix extension is not available at compile time
//
// llama.cpp-0sres (closed) promoted the single static cache to a per-context
// `fattn_xmx_v2_device_cache` (see struct below). `fattn_xmx_v2_pick_variant_cached`
// keys on ggml_backend_sycl_context, so heterogeneous configs (B580 + iGPU Xe-LPG)
// resolve independently per device.
// =============================================================================

// Query the matrix_combinations list in a noexcept-ish wrapper. Returns empty
// vector on any failure path (non-XMX device, driver bug, exception). oneAPI
// 2025.3 exposes `matrix_combinations` but does not define the old
// SYCL_EXT_ONEAPI_MATRIX_VERSION macro, so header presence plus the runtime
// try/catch is the capability gate here.
inline std::vector<sycl_xmx::combination> fattn_xmx_v2_query_combinations(const sycl::device & dev) noexcept {
    std::vector<sycl_xmx::combination> out;
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return out;
    }
    try {
        out = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    } catch (...) {
        out.clear();
    }
    return out;
}

// Score a (variant, combo) pair. Higher score = better fit. 0 = no match.
//
// Match requirements (all must hold):
//   - combo is fp16→fp32 (a=fp16, b=fp16, c=fp32, d=fp32 — what the kernel uses)
//   - combo N size == variant.tn (fixed sub-group striping)
//   - combo K size == variant.tk (fixed D-loop step)
//   - combo M range covers variant.tm (either fixed M == tm, OR variable-M combo
//     with max_msize >= tm so the lane-ownership fragment exists at that M)
//
// Scoring: larger TM wins (better XMX utilization). Extendable later.
inline int fattn_xmx_v2_score_variant_for_combo(const variant_info & v, const sycl_xmx::combination & c) noexcept {
    using sycl_xmx::matrix_type;
    if (c.atype != matrix_type::fp16 || c.btype != matrix_type::fp16 || c.ctype != matrix_type::fp32 ||
        c.dtype != matrix_type::fp32) {
        return 0;
    }
    if ((int) c.nsize != v.tn) {
        return 0;
    }
    if ((int) c.ksize != v.tk) {
        return 0;
    }

    // msize == 0 means the combo reports a *range* [1..max_msize]; otherwise it's fixed.
    const bool m_is_range = (c.msize == 0);
    const int  m_fixed    = (int) c.msize;
    const int  m_max      = (int) c.max_msize;

    if (m_is_range) {
        if (m_max < v.tm) {
            return 0;
        }
    } else {
        if (m_fixed != v.tm) {
            return 0;
        }
    }

    return v.tm * 100;
}

inline int fattn_xmx_v2_parse_forced_variant(const char * env) noexcept {
    if (!env || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
        return -1;
    }
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "tm8") == 0 || std::strcmp(env, "TM8") == 0) {
        return 0;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "tm16") == 0 || std::strcmp(env, "TM16") == 0) {
        return 1;
    }
    return -1;
}

inline bool fattn_xmx_v2_variant_supported(const sycl::device & dev, std::size_t variant_idx) noexcept {
    if (variant_idx == 0) {
        return true;
    }
    if (variant_idx >= fattn_xmx_v2_variants.size()) {
        return false;
    }
    const auto combos = fattn_xmx_v2_query_combinations(dev);
    for (const auto & c : combos) {
        if (fattn_xmx_v2_score_variant_for_combo(fattn_xmx_v2_variants[variant_idx], c) > 0) {
            return true;
        }
    }
    return false;
}

inline bool fattn_xmx_v2_decode_m1n64_supported(const sycl::device & dev, int tk = XMX_V2_DECODE_TK) noexcept {
    const auto combos = fattn_xmx_v2_query_combinations(dev);
    for (const auto & c : combos) {
        using sycl_xmx::matrix_type;
        if (c.atype == matrix_type::fp16 && c.btype == matrix_type::fp16 && c.ctype == matrix_type::fp32 &&
            c.dtype == matrix_type::fp32 && (int) c.msize == 1 && (int) c.ksize == tk &&
            (int) c.nsize == XMX_V2_DECODE_TN) {
            return true;
        }
    }
    return false;
}

// Shape-aware variant selector for XMX-v2.
// Returns 0 (fallback TM=8) for shapes that should not use TM=16:
//   - TG/small ncols: ne01 <= 1 or ncols <= 8 (TM=16 over-provisioned)
//   - small D: D < 128 (not enough elements per tile for TM=16 benefit)
// Returns 1 (TM=16) for PP shapes with ncols >= 16 and D >= 128,
//   only when the device matrix extension supports (16,16,16).
//
// Env override: GGML_SYCL_FA_XMX_V2_VARIANT=0/tm8 forces TM=8,
// =1/tm16 forces TM=16, =auto (default) uses shape-aware selection.
inline std::size_t fattn_xmx_v2_pick_variant_for_shape(const sycl::device & dev, int D, int ncols, int ne01) noexcept {
    const int forced_variant = fattn_xmx_v2_parse_forced_variant(std::getenv("GGML_SYCL_FA_XMX_V2_VARIANT"));
    if (forced_variant >= 0 && fattn_xmx_v2_variant_supported(dev, static_cast<std::size_t>(forced_variant))) {
        return static_cast<std::size_t>(forced_variant);
    }

    // Shape guard: don't use TM=16 for small shapes.
    // TG (ncol=1) or small batches (< 16 queries) → TM=8 is strictly better.
    // Small D (< 128) doesn't benefit from wider tiles.
    if (ne01 <= 1 || ncols < 16 || D < 128) {
        return 0;
    }

    // Delegate to device-wide matrix_combinations query for TM=16 support.
    const auto combos = fattn_xmx_v2_query_combinations(dev);
    for (const auto & c : combos) {
        using sycl_xmx::matrix_type;
        if (c.atype == matrix_type::fp16 && c.btype == matrix_type::fp16 && c.ctype == matrix_type::fp32 &&
            c.dtype == matrix_type::fp32 && (int) c.nsize == 16 && (int) c.ksize == 16) {
            const bool m_is_range = (c.msize == 0);
            const int  m_fixed    = (int) c.msize;
            const int  m_max      = (int) c.max_msize;
            if (m_is_range ? m_max >= 16 : m_fixed == 16) {
                return 1;
            }
        }
    }
    return 0;
}

// Pick the best variant index for this device. Returns 0 (fallback) on any
// ambiguous / query-failed path.
inline std::size_t fattn_xmx_v2_pick_variant_for_device(const sycl::device & dev) noexcept {
    const auto combos = fattn_xmx_v2_query_combinations(dev);
    if (combos.empty()) {
        return 0;  // fallback leaf
    }

    std::size_t best_idx   = 0;
    int         best_score = 0;
    for (std::size_t vi = 0; vi < fattn_xmx_v2_variants.size(); ++vi) {
        const auto & v = fattn_xmx_v2_variants[vi];
        for (const auto & c : combos) {
            const int s = fattn_xmx_v2_score_variant_for_combo(v, c);
            if (s > best_score) {
                best_score = s;
                best_idx   = vi;
            }
        }
    }

    // best_score==0 means no combo matched any variant → fallback leaf.
    return (best_score > 0) ? best_idx : 0;
}

// Per-context cache: variant pick + SLM-fit decision, cached per ggml_backend_sycl_context.
// Mirrors the sdpa_cache pattern in fattn-onednn.cpp — opaque void* in ctx, freed via
// ggml_sycl_fattn_xmx_v2_cache_destroy() in ~ggml_backend_sycl_context().
//
// "Per-context" is functionally "per-device" today because llama.cpp creates one
// context per device. If that ever changes, each context still gets its own cache
// (correct but duplicative).
struct fattn_xmx_v2_device_cache {
    std::once_flag once;
    std::size_t    variant_idx    = 0;
    size_t         slm_size_bytes = 0;
    bool           slm_ok         = false;

    ggml_sycl_fattn_xmx_packed_k forced_split_packed_k;
    ggml_sycl::mem_handle        split_partial_max_handle;
    ggml_sycl::mem_handle        split_partial_sum_handle;
    ggml_sycl::mem_handle        split_partial_out_handle;
    size_t                       split_partial_elems     = 0;
    size_t                       split_partial_out_elems = 0;
    int                          split_partial_device    = -1;
};

// Bytes of SLM required for the fallback leaf's worst case (D=256, NCOLS=32).
// fattn_v2_slm<D, ncols>::TOTAL is in sycl::half units, so multiply by sizeof(half).
// NCOLS=32 covers the D=64 batched-PP fast path that reuses K/V across wider query blocks.
static constexpr size_t fattn_xmx_v2_required_slm_bytes() {
    constexpr size_t D_max     = 256;
    constexpr size_t ncols_max = 32;
    constexpr size_t halves =
        ncols_max * D_max + 2 * XMX_V2_BATCH_KV * D_max + ncols_max * XMX_V2_BATCH_KV * XMX_V2_FLOAT_AS_HALF_ELEMS;
    return halves * sizeof(sycl::half);
}

inline fattn_xmx_v2_device_cache * fattn_xmx_v2_get_or_create_cache(ggml_backend_sycl_context & ctx) {
    static std::mutex           cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!ctx.fattn_xmx_v2_cache) {
        ctx.fattn_xmx_v2_cache = new fattn_xmx_v2_device_cache();
    }
    return static_cast<fattn_xmx_v2_device_cache *>(ctx.fattn_xmx_v2_cache);
}

// Inline destroy helper. Called by the non-inline ggml_sycl_fattn_xmx_v2_cache_destroy
// in fattn.cpp, which is forward-declared from ggml-sycl.cpp for the ctx destructor.
inline void fattn_xmx_v2_cache_destroy_inline(void * ptr) {
    delete static_cast<fattn_xmx_v2_device_cache *>(ptr);
}

// Cached picker — one-shot init per context. Queries matrix_combinations + SLM size,
// logs one INFO line, and stores the decision. Subsequent calls are lock-free reads.
inline std::size_t fattn_xmx_v2_pick_variant_cached(ggml_backend_sycl_context & ctx, const sycl::device & dev) {
    auto * cache = fattn_xmx_v2_get_or_create_cache(ctx);
    std::call_once(cache->once, [&]() {
        cache->variant_idx    = fattn_xmx_v2_pick_variant_for_device(dev);
        cache->slm_size_bytes = dev.get_info<sycl::info::device::local_mem_size>();

        const size_t required_bytes = fattn_xmx_v2_required_slm_bytes();
        cache->slm_ok               = (cache->slm_size_bytes >= required_bytes);

        std::fprintf(
            stderr,
            "[fattn-xmx-v2] device %d: local_mem=%zu KB, required=%zu B for (D=256,ncols=32), variant=%zu, slm_ok=%d\n",
            ctx.device, cache->slm_size_bytes / 1024, required_bytes, cache->variant_idx, cache->slm_ok ? 1 : 0);

        if (!cache->slm_ok) {
            std::fprintf(stderr,
                         "[fattn-xmx-v2] device %d: ERROR — worst-case SLM %zu B exceeds device local_mem %zu B; "
                         "XMX-v2 will fall back to the (8,16,16) leaf with best-effort correctness.\n",
                         ctx.device, required_bytes, cache->slm_size_bytes);
        }
    });
    return cache->variant_idx;
}

// =============================================================================
// SLM layout helper — strict, non-aliasing
//
// tile_Q[ncols][D]            — loaded once at kernel start, register-resident Q
// tile_K[BATCH_KV][D]         — K tile per KV batch
// tile_V[BATCH_KV][D]         — V tile per KV batch (separate from K)
// tile_S[ncols][BATCH_KV]     — softmax weights written before S@V. The S
//                                region is sized for float weights so
//                                GGML_PREC_F32 can avoid half probability
//                                quantization; PREC_DEFAULT uses the first
//                                half-sized portion for the XMX S@V path.
//
// All regions start at WORD-aligned offsets.  The ordering above places each
// region right after the previous one with NO overlap.
// =============================================================================

template <int D, int ncols> struct fattn_v2_slm {
    static constexpr int Q_ELEMS         = ncols * D;
    static constexpr int K_ELEMS         = XMX_V2_BATCH_KV * D;
    static constexpr int V_ELEMS         = XMX_V2_BATCH_KV * D;
    static constexpr int S_HALF_ELEMS    = ncols * XMX_V2_BATCH_KV;
    static constexpr int S_STORAGE_ELEMS = S_HALF_ELEMS * XMX_V2_FLOAT_AS_HALF_ELEMS;
    static constexpr int TOTAL           = Q_ELEMS + K_ELEMS + V_ELEMS + S_STORAGE_ELEMS;

    static constexpr int Q_OFFSET = 0;
    static constexpr int K_OFFSET = Q_OFFSET + Q_ELEMS;
    static constexpr int V_OFFSET = K_OFFSET + K_ELEMS;
    static constexpr int S_OFFSET = V_OFFSET + V_ELEMS;
};

template <int D> struct fattn_v2_decode_slm {
    static constexpr int Q_ELEMS           = D;
    static constexpr int K_PACKED_PER_HALF = (D / 2) * (XMX_V2_DECODE_TN * 2);
    static constexpr int K_PACKED_ELEMS    = 2 * K_PACKED_PER_HALF;
    static constexpr int V_ELEMS           = XMX_V2_DECODE_BATCH_KV * D;
    static constexpr int S_STORAGE_ELEMS   = XMX_V2_DECODE_BATCH_KV * XMX_V2_FLOAT_AS_HALF_ELEMS;
    static constexpr int TOTAL             = Q_ELEMS + K_PACKED_ELEMS + V_ELEMS + S_STORAGE_ELEMS;

    static constexpr int Q_OFFSET        = 0;
    static constexpr int K_PACKED_OFFSET = Q_OFFSET + Q_ELEMS;
    static constexpr int V_OFFSET        = K_PACKED_OFFSET + K_PACKED_ELEMS;
    static constexpr int S_OFFSET        = V_OFFSET + V_ELEMS;
};

template <int D> struct fattn_v2_decode_gqa_slm {
    static constexpr int Q_ELEMS           = XMX_V2_DECODE_GQA_MAX * D;
    static constexpr int K_PACKED_PER_HALF = (D / 2) * (XMX_V2_DECODE_TN * 2);
    static constexpr int K_PACKED_ELEMS    = 2 * K_PACKED_PER_HALF;
    static constexpr int V_ELEMS           = XMX_V2_DECODE_BATCH_KV * D;
    static constexpr int S_STORAGE_ELEMS = XMX_V2_DECODE_GQA_MAX * XMX_V2_DECODE_BATCH_KV * XMX_V2_FLOAT_AS_HALF_ELEMS;
    static constexpr int TOTAL           = Q_ELEMS + K_PACKED_ELEMS + V_ELEMS + S_STORAGE_ELEMS;

    static constexpr int Q_OFFSET        = 0;
    static constexpr int K_PACKED_OFFSET = Q_OFFSET + Q_ELEMS;
    static constexpr int V_OFFSET        = K_PACKED_OFFSET + K_PACKED_ELEMS;
    static constexpr int S_OFFSET        = V_OFFSET + V_ELEMS;
};

// =============================================================================
// FP8 E4M3 → fp16 dequant (identical helper to v1 kernel — needed for kv_is_fp8)
// =============================================================================

inline sycl::half fp8_e4m3_to_half_v2(uint8_t bits) {
    const uint32_t sign = (bits >> 7) & 0x1;
    const uint32_t exp  = (bits >> 3) & 0xF;
    const uint32_t mant = bits & 0x7;
    float          result;
    if (exp == 0) {
        result = (sign ? -1.0f : 1.0f) * (float) mant * (1.0f / 512.0f);
    } else if (exp == 15 && mant == 7) {
        result = sycl::nan(0u);
    } else {
        const int32_t  fp32_exp  = (int32_t) exp - 7 + 127;
        const uint32_t fp32_bits = (sign << 31) | ((uint32_t) fp32_exp << 23) | (mant << 20);

        union {
            uint32_t u;
            float    f;
        } pun;

        pun.u  = fp32_bits;
        result = pun.f;
    }
    return sycl::half(result);
}

// =============================================================================
// Kernel leaf — parameterised on (TM, TK, TN). Today only the (8, 16, 16)
// instantiation is reachable; the static_assert enforces that until future
// leaves land with matching lane-ownership math.
// =============================================================================

template <int  D,
          int  ncols,
          bool use_logit_softcap,
          typename Q_type,
          typename Acc_t,
          bool kv_is_fp8,
          int  TM,
          int  TK,
          int  TN>
static void flash_attn_xmx_v2_f16_kernel_leaf(const char * __restrict__ Q_base,
                                              const char * __restrict__ K_base,
                                              const char * __restrict__ V_base,
                                              const char * __restrict__ maskh_base,
                                              const char * __restrict__ sinks_base,
                                              float * __restrict__ dst,
                                              float                    scale,
                                              float                    max_bias,
                                              float                    m0,
                                              float                    m1,
                                              uint32_t                 n_head_log2,
                                              float                    logit_softcap,
                                              int                      ne01,
                                              int                      ne02,
                                              int                      nb01,
                                              int                      nb02,
                                              int                      nb03,
                                              int                      ne11,
                                              int                      ne12,
                                              int                      nb11,
                                              int                      nb12,
                                              int64_t                  nb13,
                                              int                      nb21,
                                              int                      nb22,
                                              int64_t                  nb23,
                                              int                      ne30,
                                              int                      ne32,
                                              int                      ne33,
                                              int                      nb31,
                                              int                      nb32,
                                              int64_t                  nb33,
                                              const sycl::nd_item<3> & item,
                                              sycl::half *             slm) {
    static_assert(
        (TM == XMX_V2_TM && TK == XMX_V2_TK && TN == XMX_V2_TN) || (TM == 16 && TK == XMX_V2_TK && TN == XMX_V2_TN),
        "Only (8,16,16) and (16,16,16) fallback leaf bodies are implemented. "
        "Follow-up tasks must add kernel specializations for other leaves.");
    static_assert(D % TK == 0, "D must be divisible by TK");
    static_assert(ncols % TM == 0 || ncols < TM, "ncols must be <= TM or a multiple of TM");

    using slm_layout            = fattn_v2_slm<D, ncols>;
    constexpr int SG_ROWS_PER_Q = (ncols + TM - 1) / TM;  // ceil(ncols/TM)
    constexpr int NTHREADS      = SG_ROWS_PER_Q * XMX_V2_SG;
    static_assert(NTHREADS <= XMX_V2_MAX_NTHREADS, "XMX-v2 active workgroup exceeds historical max");

    // ------------------------------------------------------------------
    // Sub-group / work-item identifiers
    // ------------------------------------------------------------------
    auto      sg    = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());    // 0..active_subgroups-1
    const int lane  = static_cast<int>(sg.get_local_id());           // 0..SG-1
    const int tid   = static_cast<int>(item.get_local_linear_id());  // 0..NTHREADS-1

    // ------------------------------------------------------------------
    // Work-group → (batch-sequence, kv-head, query-block)
    // ------------------------------------------------------------------
    const int hb_id    = static_cast<int>(item.get_group(0));
    const int sequence = hb_id / ne02;
    const int head     = hb_id % ne02;
    const int ic0      = static_cast<int>(item.get_group(2)) * ncols;

    const int gqa_ratio = ne02 / ne12;
    const int kv_head   = head / gqa_ratio;

    const char * Q_ptr = Q_base + (int64_t) nb03 * sequence + (int64_t) nb02 * head;
    const char * K_ptr = K_base + nb13 * sequence + (int64_t) nb12 * kv_head;
    const char * V_ptr = V_base + nb23 * sequence + (int64_t) nb22 * kv_head;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Mask pointer — pre-offset to ic0 within the query dimension.
    const int          mask_head = (ne32 > 1) ? head % ne32 : 0;
    const sycl::half * maskh =
        maskh_base ? reinterpret_cast<const sycl::half *>(maskh_base + (int64_t) nb33 * (sequence % ne33) +
                                                          (int64_t) nb32 * mask_head + (int64_t) nb31 * ic0) :
                     nullptr;

    // ------------------------------------------------------------------
    // SLM region pointers (strictly non-overlapping — Rule 1)
    // ------------------------------------------------------------------
    sycl::half * tile_Q   = slm + slm_layout::Q_OFFSET;  // [ncols][D]
    sycl::half * tile_K   = slm + slm_layout::K_OFFSET;  // [BATCH_KV][D]
    sycl::half * tile_V   = slm + slm_layout::V_OFFSET;  // [BATCH_KV][D]
    sycl::half * tile_S   = slm + slm_layout::S_OFFSET;  // [ncols][BATCH_KV]
    float *      tile_S_f = reinterpret_cast<float *>(slm + slm_layout::S_OFFSET);

    // ------------------------------------------------------------------
    // Register state (Rule 2: all cross-lane softmax state in registers)
    // ------------------------------------------------------------------
    static_assert(D % TN == 0, "D must be divisible by TN for lane-D mapping");
    constexpr int D_TILES = D / TN;                    // number of V-tiles in D direction

    const int this_sg_q_tile = sg_id % SG_ROWS_PER_Q;  // which Q-tile this SG covers
    const int sg_q_base      = this_sg_q_tile * TM;

    // Per-lane accumulator. `Acc_t` is a template parameter set by the
    // dispatcher: `float` on GGML_PREC_F32 (matches CUDA's KQ_acc_t=float
    // branch) and `afloat` on GGML_PREC_DEFAULT (picks up the SYCL-F16 build
    // flag — see ggml-sycl/common.hpp). `KQ_max` / `KQ_sum` stay in full
    // float: online-softmax state is precision-critical (the
    // exp(max_old - max_new) rescale cancels rapidly if the running max is
    // rounded to half), and the FTZ bit manipulation further down assumes a
    // 32-bit scale_old.
    Acc_t VKQ[TM][D_TILES];  // [query_slot_in_tile][D_tile]
    float KQ_max[TM];
    float KQ_sum[TM];

#    pragma unroll
    for (int r = 0; r < TM; ++r) {
        KQ_max[r] = -FLT_MAX / 2.0f;
        KQ_sum[r] = 0.0f;
#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[r][t] = Acc_t(0.0f);
        }
    }

    // ------------------------------------------------------------------
    // Phase 0: Cooperatively load tile_Q from global into SLM (Rule 1)
    // ------------------------------------------------------------------
    for (int idx = tid; idx < ncols * D; idx += NTHREADS) {
        const int j     = idx / D;
        const int d     = idx % D;
        const int q_idx = ic0 + j;
        if (q_idx < ne01) {
            const Q_type * Q_row_ptr = reinterpret_cast<const Q_type *>(Q_ptr + (int64_t) nb01 * q_idx);
            tile_Q[j * D + d]        = sycl::half(static_cast<float>(Q_row_ptr[d]) * scale);
        } else {
            tile_Q[j * D + d] = sycl::half(0.0f);
        }
    }
    sycl::group_barrier(item.get_group());  // barrier: tile_Q ready

    // ------------------------------------------------------------------
    // Main KV loop
    // ------------------------------------------------------------------
    for (int kv_start = 0; kv_start < ne11; kv_start += XMX_V2_BATCH_KV) {
        const int kv_count = sycl::min(XMX_V2_BATCH_KV, ne11 - kv_start);

        // ---- Phase 1: Load tile_K and tile_V from global to SLM ----
        for (int idx = tid; idx < kv_count * D; idx += NTHREADS) {
            const int k      = idx / D;
            const int d      = idx % D;
            const int kv_pos = kv_start + k;

            const char * K_row_base = K_ptr + (int64_t) nb11 * kv_pos;
            const char * V_row_base = V_ptr + (int64_t) nb21 * kv_pos;

            sycl::half k_val, v_val;
            if constexpr (kv_is_fp8) {
                k_val = fp8_e4m3_to_half_v2(reinterpret_cast<const uint8_t *>(K_row_base)[d]);
                v_val = fp8_e4m3_to_half_v2(reinterpret_cast<const uint8_t *>(V_row_base)[d]);
            } else {
                k_val = reinterpret_cast<const sycl::half *>(K_row_base)[d];
                v_val = reinterpret_cast<const sycl::half *>(V_row_base)[d];
            }
            tile_K[k * D + d] = k_val;
            tile_V[k * D + d] = v_val;
        }
        // Zero-pad incomplete last batch so XMX never reads uninitialized SLM.
        for (int idx = tid; idx < (XMX_V2_BATCH_KV - kv_count) * D; idx += NTHREADS) {
            const int k       = kv_count + idx / D;
            const int d       = idx % D;
            tile_K[k * D + d] = sycl::half(0.0f);
            tile_V[k * D + d] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group());  // barrier: tile_K, tile_V ready

        // ---- Phase 2: Q @ K^T via joint_matrix_mad ----
        constexpr int KV_TILES = XMX_V2_BATCH_KV / TN;

        for (int kv_tile = 0; kv_tile < KV_TILES; ++kv_tile) {
            const int kv_col = kv_tile * TN;  // first KV position in this tile

            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, TM, TN> mat_QK;
            sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

            for (int d = 0; d < D; d += TK) {
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, TM, TK,
                                       sycl_xmx::layout::row_major>
                    mat_Q;
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, TK, TN,
                                       sycl_xmx::layout::col_major>
                    mat_K;

                sycl_xmx::joint_matrix_load(
                    sg, mat_Q,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_Q[sg_q_base * D + d]),
                    D);

                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_K[kv_col * D + d]),
                    D);

                sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_K, mat_QK);
            }

            // ---- Phase 3: Extract QK values via joint_matrix_apply (Rule 5) ----
            float lane_QK[TM];
            {
                int slot = 0;
                sycl_xmx::joint_matrix_apply(sg, mat_QK, [&](float & elem) { lane_QK[slot++] = elem; });
            }

// ---- Phase 4: Softcap, mask, per-row max (Rule 8, 10) ----
#    pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int q_abs = ic0 + sg_q_base + r;
                if (q_abs >= ne01) {
                    lane_QK[r] = -FLT_MAX;
                    continue;
                }

                if constexpr (use_logit_softcap) {
                    lane_QK[r] = logit_softcap * sycl::tanh(lane_QK[r]);
                }

                if (maskh) {
                    const int kv_abs = kv_start + kv_col + lane;
                    if (kv_abs < ne11) {
                        const float mask_val = static_cast<float>(maskh[(sg_q_base + r) * ne30 + kv_abs]);
                        lane_QK[r] += slope * mask_val;
                    } else {
                        lane_QK[r] = -FLT_MAX;
                    }
                }

                if (kv_start + kv_col + lane >= ne11) {
                    lane_QK[r] = -FLT_MAX;
                }
            }

// Per-row max over this SG's KV-tile (Rule 2: sub-group reduce, no SLM)
#    pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int q_abs = ic0 + sg_q_base + r;
                if (q_abs >= ne01) {
                    continue;
                }

                const float tile_max = sycl::reduce_over_group(sg, lane_QK[r], sycl::maximum<float>{});

                const float new_max     = sycl::fmax(KQ_max[r], tile_max);
                const float KQ_max_diff = KQ_max[r] - new_max;

                // FTZ via bit manipulation (Rule 8)
                float    scale_old = sycl::exp(KQ_max_diff);
                uint32_t scale_bits;
                __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
                scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
                __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

                // VKQ lives in `Acc_t` (half under GGML_SYCL_F16+PREC_DEFAULT,
                // float on PREC_F32). Apply the rescale through an explicit
                // narrowing cast so the mixed-mode multiply stays readable.
                const Acc_t scale_old_a = Acc_t(scale_old);
#    pragma unroll
                for (int t = 0; t < D_TILES; ++t) {
                    VKQ[r][t] *= scale_old_a;
                }
                KQ_sum[r] *= scale_old;
                KQ_max[r] = new_max;

                const float p        = sycl::exp(lane_QK[r] - new_max);
                const float tile_sum = sycl::reduce_over_group(sg, p, sycl::plus<float>{});
                KQ_sum[r] += tile_sum;

                const int s_row = sg_q_base + r;
                const int s_col = kv_col + lane;
                if (s_row < ncols && s_col < XMX_V2_BATCH_KV) {
                    if constexpr (std::is_same_v<Acc_t, float>) {
                        tile_S_f[s_row * XMX_V2_BATCH_KV + s_col] = p;
                    } else {
                        tile_S[s_row * XMX_V2_BATCH_KV + s_col] = sycl::half(p);
                    }
                }
            }

            // ---- Phase 5: S @ V for this KV tile ----
            // Online softmax may rescale VKQ when a later KV tile raises the
            // row max, so the current tile's probabilities must be consumed
            // before the next tile can update KQ_max.
            sycl::group_barrier(item.get_group());  // Rule 3

            if constexpr (std::is_same_v<Acc_t, float>) {
                for (int d_tile = 0; d_tile < D_TILES; ++d_tile) {
                    const int d = d_tile * TN + lane;

#    pragma unroll
                    for (int r = 0; r < TM; ++r) {
                        const int q_abs = ic0 + sg_q_base + r;
                        if (q_abs >= ne01) {
                            continue;
                        }

                        float acc = 0.0f;
#    pragma unroll
                        for (int k = 0; k < TN; ++k) {
                            const int kv_idx = kv_col + k;
                            acc += tile_S_f[(sg_q_base + r) * XMX_V2_BATCH_KV + kv_idx] *
                                   static_cast<float>(tile_V[kv_idx * D + d]);
                        }
                        VKQ[r][d_tile] += acc;
                    }
                }
            } else {
                for (int d_tile = 0; d_tile < D_TILES; ++d_tile) {
                    const int d_start = d_tile * TN;

                    sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, TM, TN> mat_SV;
                    sycl_xmx::joint_matrix_fill(sg, mat_SV, 0.0f);

                    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, TM, TK,
                                           sycl_xmx::layout::row_major>
                        mat_S;
                    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, TK, TN,
                                           sycl_xmx::layout::row_major>
                        mat_V;

                    sycl_xmx::joint_matrix_load(
                        sg, mat_S,
                        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            &tile_S[sg_q_base * XMX_V2_BATCH_KV + kv_col]),
                        XMX_V2_BATCH_KV);

                    sycl_xmx::joint_matrix_load(
                        sg, mat_V,
                        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            &tile_V[kv_col * D + d_start]),
                        D);

                    sycl_xmx::joint_matrix_mad(sg, mat_SV, mat_S, mat_V, mat_SV);

                    {
                        int slot = 0;
                        sycl_xmx::joint_matrix_apply(sg, mat_SV, [&](float & elem) {
                            VKQ[slot][d_tile] += elem;
                            ++slot;
                        });
                    }
                }
            }
        }  // end kv_tile loop

        sycl::group_barrier(item.get_group());  // Rule 3

    }  // end kv_start loop

    // ------------------------------------------------------------------
    // Post-loop: attention sinks (Rule 9 — applied AFTER KV loop)
    // ------------------------------------------------------------------
    if (sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        const float   sink    = sinks_f[head];

#    pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int q_abs = ic0 + sg_q_base + r;
            if (q_abs >= ne01) {
                continue;
            }

            const float new_max     = sycl::fmax(KQ_max[r], sink);
            const float KQ_max_diff = KQ_max[r] - new_max;

            float    scale_old = sycl::exp(KQ_max_diff);
            uint32_t scale_bits;
            __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
            scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
            __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

            const float sink_weight = sycl::exp(sink - new_max);
            KQ_sum[r]               = KQ_sum[r] * scale_old + sink_weight;
            KQ_max[r]               = new_max;

            const Acc_t scale_old_a = Acc_t(scale_old);
#    pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                VKQ[r][t] *= scale_old_a;
            }
        }
    }

// ------------------------------------------------------------------
// Normalize and write output.
// ------------------------------------------------------------------
#    pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int q_abs = ic0 + sg_q_base + r;
        if (q_abs >= ne01) {
            continue;
        }

        // KQ_sum stays in float (softmax denominator precision); VKQ is `Acc_t`
        // (half under GGML_SYCL_F16+PREC_DEFAULT, float under PREC_F32), so the
        // final multiply promotes to float for the f32 dst_row write expected
        // by the public FA contract.
        const float inv_sum = (KQ_sum[r] > 0.0f) ? (1.0f / KQ_sum[r]) : 0.0f;
        float *     dst_row = dst + (int64_t) D * (head + ne02 * (q_abs + ne01 * sequence));

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int   d   = t * TN + lane;
            const float val = (float) VKQ[r][t] * inv_sum;
            dst_row[d]      = sycl::isfinite(val) ? val : 0.0f;
        }
    }
}

// =============================================================================
// Leaf launch helper — one instantiation per compile-time variant.
// =============================================================================

template <int  D,
          int  ncols,
          bool use_logit_softcap,
          typename Q_type,
          typename Acc_t,
          bool kv_is_fp8,
          int  TM,
          int  TK,
          int  TN>
static void launch_fattn_xmx_v2_f16_leaf(const fattn_params & params, dpct::queue_ptr stream) {
    using slm_layout            = fattn_v2_slm<D, ncols>;
    constexpr int sg_rows_per_q = (ncols + TM - 1) / TM;
    constexpr int nthreads      = sg_rows_per_q * XMX_V2_SG;
    static_assert(nthreads <= XMX_V2_MAX_NTHREADS, "XMX-v2 active workgroup exceeds historical max");

    const int      n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    const char *   Q_ptr       = params.Q;
    const char *   K_ptr       = params.K;
    const char *   V_ptr       = params.V;
    const char *   mask_ptr    = params.mask;
    const char *   sinks_ptr   = params.sinks;
    float *        dst_ptr     = params.dst;
    const float    scale_v     = params.scale;
    const float    max_bias    = params.max_bias;
    const float    m0          = params.m0;
    const float    m1          = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;
    const float    logit_sc    = params.logit_softcap;
    const int      ne01 = params.ne01, ne02 = params.ne02;
    const int      nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int      ne11 = params.ne11, ne12 = params.ne12;
    const int      nb11 = params.nb11, nb12 = params.nb12;
    const int64_t  nb13 = params.nb13;
    const int      nb21 = params.nb21, nb22 = params.nb22;
    const int64_t  nb23 = params.nb23;
    const int      ne30 = params.ne30, ne32 = params.ne32, ne33 = params.ne33;
    const int      nb31 = params.nb31, nb32 = params.nb32;
    const int64_t  nb33 = params.nb33;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(slm_layout::TOTAL), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                flash_attn_xmx_v2_f16_kernel_leaf<D, ncols, use_logit_softcap, Q_type, Acc_t, kv_is_fp8, TM, TK, TN>(
                    Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, scale_v, max_bias, m0, m1, n_head_log2, logit_sc,
                    ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, ne30, ne32, ne33,
                    nb31, nb32, nb33, item, slm.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

template <int D, bool use_logit_softcap, typename Q_type>
static void flash_attn_xmx_v2_decode_m1n64_kernel(const char * __restrict__ Q_base,
                                                  const char * __restrict__ K_base,
                                                  const char * __restrict__ V_base,
                                                  const char * __restrict__ maskh_base,
                                                  const char * __restrict__ sinks_base,
                                                  float * __restrict__ dst,
                                                  float                    scale,
                                                  float                    max_bias,
                                                  float                    m0,
                                                  float                    m1,
                                                  uint32_t                 n_head_log2,
                                                  float                    logit_softcap,
                                                  int                      ne01,
                                                  int                      ne02,
                                                  int                      nb01,
                                                  int                      nb02,
                                                  int                      nb03,
                                                  int                      ne11,
                                                  int                      ne12,
                                                  int                      nb11,
                                                  int                      nb12,
                                                  int64_t                  nb13,
                                                  int                      nb21,
                                                  int                      nb22,
                                                  int64_t                  nb23,
                                                  int                      ne30,
                                                  int                      ne32,
                                                  int                      ne33,
                                                  int                      nb31,
                                                  int                      nb32,
                                                  int64_t                  nb33,
                                                  const sycl::nd_item<3> & item,
                                                  sycl::half *             slm) {
    static_assert(D == 64, "M=1,N=64 decode prototype is only implemented for GPT-OSS D=64");
    static_assert(D % XMX_V2_DECODE_TK == 0, "D must be divisible by decode TK");
    (void) ne30;

    using slm_layout       = fattn_v2_decode_slm<D>;
    constexpr int NTHREADS = XMX_V2_SG;
    constexpr int D_TILES  = D / XMX_V2_SG;

    auto      sg   = item.get_sub_group();
    const int lane = static_cast<int>(sg.get_local_id());
    const int tid  = static_cast<int>(item.get_local_linear_id());

    const int hb_id    = static_cast<int>(item.get_group(0));
    const int sequence = hb_id / ne02;
    const int head     = hb_id % ne02;
    const int q_abs    = static_cast<int>(item.get_group(2));

    const int gqa_ratio = ne02 / ne12;
    const int kv_head   = head / gqa_ratio;

    const char * Q_ptr = Q_base + (int64_t) nb03 * sequence + (int64_t) nb02 * head;
    const char * K_ptr = K_base + nb13 * sequence + (int64_t) nb12 * kv_head;
    const char * V_ptr = V_base + nb23 * sequence + (int64_t) nb22 * kv_head;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    const int          mask_head = (ne32 > 1) ? head % ne32 : 0;
    const sycl::half * maskh =
        maskh_base ? reinterpret_cast<const sycl::half *>(maskh_base + (int64_t) nb33 * (sequence % ne33) +
                                                          (int64_t) nb32 * mask_head + (int64_t) nb31 * q_abs) :
                     nullptr;

    sycl::half * tile_Q        = slm + slm_layout::Q_OFFSET;
    sycl::half * tile_K_packed = slm + slm_layout::K_PACKED_OFFSET;
    sycl::half * tile_V        = slm + slm_layout::V_OFFSET;
    float *      tile_S_f      = reinterpret_cast<float *>(slm + slm_layout::S_OFFSET);

    float VKQ[D_TILES];
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        VKQ[t] = 0.0f;
    }

    for (int d = tid; d < D; d += NTHREADS) {
        const Q_type * Q_row_ptr = reinterpret_cast<const Q_type *>(Q_ptr + (int64_t) nb01 * q_abs);
        tile_Q[d]                = sycl::half(static_cast<float>(Q_row_ptr[d]) * scale);
    }
    sycl::group_barrier(item.get_group());

    for (int kv_start = 0; kv_start < ne11; kv_start += XMX_V2_DECODE_BATCH_KV) {
        for (int idx = tid; idx < slm_layout::K_PACKED_ELEMS; idx += NTHREADS) {
            tile_K_packed[idx] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group());

        for (int idx = tid; idx < XMX_V2_DECODE_BATCH_KV * D; idx += NTHREADS) {
            const int kv_local = idx / D;
            const int d        = idx - kv_local * D;
            const int kv_abs   = kv_start + kv_local;

            sycl::half k_val = sycl::half(0.0f);
            sycl::half v_val = sycl::half(0.0f);
            if (kv_abs < ne11) {
                k_val = reinterpret_cast<const sycl::half *>(K_ptr + (int64_t) nb11 * kv_abs)[d];
                v_val = reinterpret_cast<const sycl::half *>(V_ptr + (int64_t) nb21 * kv_abs)[d];
            }
            tile_V[kv_local * D + d] = v_val;

            const int half_id     = kv_local / XMX_V2_DECODE_HALF_KV;
            const int compact_col = kv_local - half_id * XMX_V2_DECODE_HALF_KV;
            const int active_lane = compact_col % XMX_V2_DECODE_ACTIVE_LANES;
            const int slot        = compact_col / XMX_V2_DECODE_ACTIVE_LANES;
            const int active_col  = slot * XMX_V2_SG + active_lane;
            const int packed_idx =
                half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2) + active_col * 2 + (d & 1);
            tile_K_packed[packed_idx] = k_val;
        }
        sycl::group_barrier(item.get_group());

        float lane_scores[2 * XMX_V2_DECODE_SLOTS];
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            lane_scores[i] = -FLT_MAX;
        }

#    pragma unroll
        for (int half_id = 0; half_id < 2; ++half_id) {
            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, 1, XMX_V2_DECODE_TN> mat_QK;
            sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

#    pragma unroll
            for (int d = 0; d < D; d += XMX_V2_DECODE_TK) {
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, 1, XMX_V2_DECODE_TK,
                                       sycl_xmx::layout::row_major>
                    mat_Q;
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, XMX_V2_DECODE_TK,
                                       XMX_V2_DECODE_TN, sycl_xmx::layout::ext_intel_packed>
                    mat_K;

                sycl_xmx::joint_matrix_load(
                    sg, mat_Q,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_Q[d]),
                    D);
                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_K_packed[half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2)]),
                    XMX_V2_DECODE_TN * 2);
                sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_K, mat_QK);
            }

            int slot = 0;
            sycl_xmx::joint_matrix_apply(sg, mat_QK, [&](float & elem) {
                if (lane < XMX_V2_DECODE_ACTIVE_LANES && slot < XMX_V2_DECODE_SLOTS) {
                    lane_scores[half_id * XMX_V2_DECODE_SLOTS + slot] = elem;
                }
                ++slot;
            });
        }

        float local_max = -FLT_MAX;
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            const int half_id  = i / XMX_V2_DECODE_SLOTS;
            const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
            const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
            const int kv_abs   = kv_start + kv_local;

            float score = lane_scores[i];
            if (lane >= XMX_V2_DECODE_ACTIVE_LANES || kv_abs >= ne11 || q_abs >= ne01) {
                score = -FLT_MAX;
            } else {
                if constexpr (use_logit_softcap) {
                    score = logit_softcap * sycl::tanh(score);
                }
                if (maskh) {
                    score += slope * static_cast<float>(maskh[kv_abs]);
                }
            }
            lane_scores[i] = score;
            local_max      = sycl::fmax(local_max, score);
        }

        const float tile_max    = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});
        const float new_max     = sycl::fmax(KQ_max, tile_max);
        const float KQ_max_diff = KQ_max - new_max;
        float       scale_old   = sycl::exp(KQ_max_diff);
        uint32_t    scale_bits;
        __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
        scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
        __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[t] *= scale_old;
        }
        KQ_sum *= scale_old;
        KQ_max = new_max;

        float local_sum = 0.0f;
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            const int half_id  = i / XMX_V2_DECODE_SLOTS;
            const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
            const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
            if (lane < XMX_V2_DECODE_ACTIVE_LANES) {
                const float p      = sycl::exp(lane_scores[i] - new_max);
                tile_S_f[kv_local] = sycl::isfinite(p) ? p : 0.0f;
                local_sum += tile_S_f[kv_local];
            }
        }
        const float tile_sum = sycl::reduce_over_group(sg, local_sum, sycl::plus<float>{});
        KQ_sum += tile_sum;

        sycl::group_barrier(item.get_group());

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int d   = t * XMX_V2_SG + lane;
            float     acc = 0.0f;
#    pragma unroll
            for (int k = 0; k < XMX_V2_DECODE_BATCH_KV; ++k) {
                if (kv_start + k < ne11) {
                    acc += tile_S_f[k] * static_cast<float>(tile_V[k * D + d]);
                }
            }
            VKQ[t] += acc;
        }

        sycl::group_barrier(item.get_group());
    }

    if (sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        const float   sink    = sinks_f[head];

        const float new_max     = sycl::fmax(KQ_max, sink);
        const float KQ_max_diff = KQ_max - new_max;

        float    scale_old = sycl::exp(KQ_max_diff);
        uint32_t scale_bits;
        __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
        scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
        __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

        const float sink_weight = sycl::exp(sink - new_max);
        KQ_sum                  = KQ_sum * scale_old + sink_weight;
        KQ_max                  = new_max;

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[t] *= scale_old;
        }
    }

    const float inv_sum = (KQ_sum > 0.0f) ? (1.0f / KQ_sum) : 0.0f;
    float *     dst_row = dst + (int64_t) D * (head + ne02 * (q_abs + ne01 * sequence));
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        const int   d   = t * XMX_V2_SG + lane;
        const float val = VKQ[t] * inv_sum;
        dst_row[d]      = sycl::isfinite(val) ? val : 0.0f;
    }
}

template <int D, bool use_logit_softcap, typename Q_type>
static void launch_fattn_xmx_v2_decode_m1n64_leaf(const fattn_params & params, dpct::queue_ptr stream) {
    using slm_layout       = fattn_v2_decode_slm<D>;
    constexpr int nthreads = XMX_V2_SG;

    const int      n_query_blocks = params.ne01;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    const char *   Q_ptr       = params.Q;
    const char *   K_ptr       = params.K;
    const char *   V_ptr       = params.V;
    const char *   mask_ptr    = params.mask;
    const char *   sinks_ptr   = params.sinks;
    float *        dst_ptr     = params.dst;
    const float    scale_v     = params.scale;
    const float    max_bias    = params.max_bias;
    const float    m0          = params.m0;
    const float    m1          = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;
    const float    logit_sc    = params.logit_softcap;
    const int      ne01 = params.ne01, ne02 = params.ne02;
    const int      nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int      ne11 = params.ne11, ne12 = params.ne12;
    const int      nb11 = params.nb11, nb12 = params.nb12;
    const int64_t  nb13 = params.nb13;
    const int      nb21 = params.nb21, nb22 = params.nb22;
    const int64_t  nb23 = params.nb23;
    const int      ne30 = params.ne30, ne32 = params.ne32, ne33 = params.ne33;
    const int      nb31 = params.nb31, nb32 = params.nb32;
    const int64_t  nb33 = params.nb33;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(slm_layout::TOTAL), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                             flash_attn_xmx_v2_decode_m1n64_kernel<D, use_logit_softcap, Q_type>(
                                 Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, scale_v, max_bias, m0, m1,
                                 n_head_log2, logit_sc, ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13,
                                 nb21, nb22, nb23, ne30, ne32, ne33, nb31, nb32, nb33, item,
                                 slm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <int D, bool use_logit_softcap, typename Q_type>
static void flash_attn_xmx_v2_decode_gqa_kernel(const char * __restrict__ Q_base,
                                                const char * __restrict__ K_base,
                                                const char * __restrict__ V_base,
                                                const char * __restrict__ maskh_base,
                                                const char * __restrict__ sinks_base,
                                                float * __restrict__ dst,
                                                float                    scale,
                                                float                    max_bias,
                                                float                    m0,
                                                float                    m1,
                                                uint32_t                 n_head_log2,
                                                float                    logit_softcap,
                                                int                      ne01,
                                                int                      ne02,
                                                int                      nb01,
                                                int                      nb02,
                                                int                      nb03,
                                                int                      ne11,
                                                int                      ne12,
                                                int                      nb11,
                                                int                      nb12,
                                                int64_t                  nb13,
                                                int                      nb21,
                                                int                      nb22,
                                                int64_t                  nb23,
                                                int                      ne30,
                                                int                      ne32,
                                                int                      ne33,
                                                int                      nb31,
                                                int                      nb32,
                                                int64_t                  nb33,
                                                const sycl::nd_item<3> & item,
                                                sycl::half *             slm) {
    static_assert(D == 64, "GQA-shared M=1,N=64 decode prototype is only implemented for GPT-OSS D=64");
    static_assert(D % XMX_V2_DECODE_TK == 0, "D must be divisible by decode TK");
    (void) ne30;

    using slm_layout       = fattn_v2_decode_gqa_slm<D>;
    constexpr int NTHREADS = XMX_V2_DECODE_GQA_MAX * XMX_V2_SG;
    constexpr int D_TILES  = D / XMX_V2_SG;

    auto      sg    = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int lane  = static_cast<int>(sg.get_local_id());
    const int tid   = static_cast<int>(item.get_local_linear_id());

    const int  hb_id     = static_cast<int>(item.get_group(0));
    const int  sequence  = hb_id / ne12;
    const int  kv_head   = hb_id % ne12;
    const int  q_abs     = static_cast<int>(item.get_group(2));
    const int  gqa_ratio = ne02 / ne12;
    const int  q_rel     = sg_id;
    const int  head      = kv_head * gqa_ratio + q_rel;
    const bool active    = q_rel < gqa_ratio && head < ne02 && q_abs < ne01;

    const char * Q_ptr = Q_base + (int64_t) nb03 * sequence;
    const char * K_ptr = K_base + nb13 * sequence + (int64_t) nb12 * kv_head;
    const char * V_ptr = V_base + nb23 * sequence + (int64_t) nb22 * kv_head;

    const float slope = active ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 0.0f;

    const int          mask_head = active && ne32 > 1 ? head % ne32 : 0;
    const sycl::half * maskh =
        (active && maskh_base) ?
            reinterpret_cast<const sycl::half *>(maskh_base + (int64_t) nb33 * (sequence % ne33) +
                                                 (int64_t) nb32 * mask_head + (int64_t) nb31 * q_abs) :
            nullptr;

    sycl::half * tile_Q        = slm + slm_layout::Q_OFFSET;
    sycl::half * tile_K_packed = slm + slm_layout::K_PACKED_OFFSET;
    sycl::half * tile_V        = slm + slm_layout::V_OFFSET;
    float *      tile_S_f      = reinterpret_cast<float *>(slm + slm_layout::S_OFFSET);

    float VKQ[D_TILES];
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        VKQ[t] = 0.0f;
    }

    for (int idx = tid; idx < XMX_V2_DECODE_GQA_MAX * D; idx += NTHREADS) {
        const int q_slot = idx / D;
        const int d      = idx - q_slot * D;
        const int h      = kv_head * gqa_ratio + q_slot;
        if (q_slot < gqa_ratio && h < ne02 && q_abs < ne01) {
            const Q_type * Q_row_ptr =
                reinterpret_cast<const Q_type *>(Q_ptr + (int64_t) nb02 * h + (int64_t) nb01 * q_abs);
            tile_Q[q_slot * D + d] = sycl::half(static_cast<float>(Q_row_ptr[d]) * scale);
        } else {
            tile_Q[q_slot * D + d] = sycl::half(0.0f);
        }
    }
    sycl::group_barrier(item.get_group());

    for (int kv_start = 0; kv_start < ne11; kv_start += XMX_V2_DECODE_BATCH_KV) {
        for (int idx = tid; idx < slm_layout::K_PACKED_ELEMS; idx += NTHREADS) {
            tile_K_packed[idx] = sycl::half(0.0f);
        }
        sycl::group_barrier(item.get_group());

        for (int idx = tid; idx < XMX_V2_DECODE_BATCH_KV * D; idx += NTHREADS) {
            const int kv_local = idx / D;
            const int d        = idx - kv_local * D;
            const int kv_abs   = kv_start + kv_local;

            sycl::half k_val = sycl::half(0.0f);
            sycl::half v_val = sycl::half(0.0f);
            if (kv_abs < ne11) {
                k_val = reinterpret_cast<const sycl::half *>(K_ptr + (int64_t) nb11 * kv_abs)[d];
                v_val = reinterpret_cast<const sycl::half *>(V_ptr + (int64_t) nb21 * kv_abs)[d];
            }
            tile_V[kv_local * D + d] = v_val;

            const int half_id     = kv_local / XMX_V2_DECODE_HALF_KV;
            const int compact_col = kv_local - half_id * XMX_V2_DECODE_HALF_KV;
            const int active_lane = compact_col % XMX_V2_DECODE_ACTIVE_LANES;
            const int slot        = compact_col / XMX_V2_DECODE_ACTIVE_LANES;
            const int active_col  = slot * XMX_V2_SG + active_lane;
            const int packed_idx =
                half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2) + active_col * 2 + (d & 1);
            tile_K_packed[packed_idx] = k_val;
        }
        sycl::group_barrier(item.get_group());

        float lane_scores[2 * XMX_V2_DECODE_SLOTS];
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            lane_scores[i] = -FLT_MAX;
        }

#    pragma unroll
        for (int half_id = 0; half_id < 2; ++half_id) {
            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, 1, XMX_V2_DECODE_TN> mat_QK;
            sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

#    pragma unroll
            for (int d = 0; d < D; d += XMX_V2_DECODE_TK) {
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, 1, XMX_V2_DECODE_TK,
                                       sycl_xmx::layout::row_major>
                    mat_Q;
                sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, XMX_V2_DECODE_TK,
                                       XMX_V2_DECODE_TN, sycl_xmx::layout::ext_intel_packed>
                    mat_K;

                sycl_xmx::joint_matrix_load(
                    sg, mat_Q,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_Q[q_rel * D + d]),
                    D);
                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_K_packed[half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2)]),
                    XMX_V2_DECODE_TN * 2);
                sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_K, mat_QK);
            }

            int slot = 0;
            sycl_xmx::joint_matrix_apply(sg, mat_QK, [&](float & elem) {
                if (lane < XMX_V2_DECODE_ACTIVE_LANES && slot < XMX_V2_DECODE_SLOTS) {
                    lane_scores[half_id * XMX_V2_DECODE_SLOTS + slot] = elem;
                }
                ++slot;
            });
        }

        float local_max = -FLT_MAX;
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            const int half_id  = i / XMX_V2_DECODE_SLOTS;
            const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
            const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
            const int kv_abs   = kv_start + kv_local;

            float score = lane_scores[i];
            if (!active || lane >= XMX_V2_DECODE_ACTIVE_LANES || kv_abs >= ne11) {
                score = -FLT_MAX;
            } else {
                if constexpr (use_logit_softcap) {
                    score = logit_softcap * sycl::tanh(score);
                }
                if (maskh) {
                    score += slope * static_cast<float>(maskh[kv_abs]);
                }
            }
            lane_scores[i] = score;
            local_max      = sycl::fmax(local_max, score);
        }

        const float tile_max    = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});
        const float new_max     = sycl::fmax(KQ_max, tile_max);
        const float KQ_max_diff = KQ_max - new_max;
        float       scale_old   = sycl::exp(KQ_max_diff);
        uint32_t    scale_bits;
        __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
        scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
        __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[t] *= scale_old;
        }
        KQ_sum *= scale_old;
        KQ_max = new_max;

        float local_sum = 0.0f;
#    pragma unroll
        for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
            const int half_id  = i / XMX_V2_DECODE_SLOTS;
            const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
            const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
            if (active && lane < XMX_V2_DECODE_ACTIVE_LANES) {
                const float p                                       = sycl::exp(lane_scores[i] - new_max);
                tile_S_f[q_rel * XMX_V2_DECODE_BATCH_KV + kv_local] = sycl::isfinite(p) ? p : 0.0f;
                local_sum += tile_S_f[q_rel * XMX_V2_DECODE_BATCH_KV + kv_local];
            }
        }
        const float tile_sum = sycl::reduce_over_group(sg, local_sum, sycl::plus<float>{});
        KQ_sum += tile_sum;

        sycl::group_barrier(item.get_group());

        if (active) {
#    pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                const int d   = t * XMX_V2_SG + lane;
                float     acc = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_V2_DECODE_BATCH_KV; ++k) {
                    if (kv_start + k < ne11) {
                        acc += tile_S_f[q_rel * XMX_V2_DECODE_BATCH_KV + k] * static_cast<float>(tile_V[k * D + d]);
                    }
                }
                VKQ[t] += acc;
            }
        }

        sycl::group_barrier(item.get_group());
    }

    if (active && sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        const float   sink    = sinks_f[head];

        const float new_max     = sycl::fmax(KQ_max, sink);
        const float KQ_max_diff = KQ_max - new_max;

        float    scale_old = sycl::exp(KQ_max_diff);
        uint32_t scale_bits;
        __builtin_memcpy(&scale_bits, &scale_old, sizeof(uint32_t));
        scale_bits *= static_cast<uint32_t>(KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD);
        __builtin_memcpy(&scale_old, &scale_bits, sizeof(float));

        const float sink_weight = sycl::exp(sink - new_max);
        KQ_sum                  = KQ_sum * scale_old + sink_weight;
        KQ_max                  = new_max;

#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            VKQ[t] *= scale_old;
        }
    }

    if (active) {
        const float inv_sum = (KQ_sum > 0.0f) ? (1.0f / KQ_sum) : 0.0f;
        float *     dst_row = dst + (int64_t) D * (head + ne02 * (q_abs + ne01 * sequence));
#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int   d   = t * XMX_V2_SG + lane;
            const float val = VKQ[t] * inv_sum;
            dst_row[d]      = sycl::isfinite(val) ? val : 0.0f;
        }
    }
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false, bool PACKED_K = false>
static void flash_attn_xmx_v2_decode_gqa_split_first_kernel(const char * __restrict__ Q_base,
                                                            const char * __restrict__ K_base,
                                                            const sycl::half * __restrict__ K_packed_base,
                                                            const char * __restrict__ V_base,
                                                            const char * __restrict__ maskh_base,
                                                            float * __restrict__ partial_max,
                                                            float * __restrict__ partial_sum,
                                                            float * __restrict__ partial_out,
                                                            float                    scale,
                                                            float                    max_bias,
                                                            float                    m0,
                                                            float                    m1,
                                                            uint32_t                 n_head_log2,
                                                            float                    logit_softcap,
                                                            int                      ne01,
                                                            int                      ne02,
                                                            int                      nb01,
                                                            int                      nb02,
                                                            int                      nb03,
                                                            int                      ne11,
                                                            int                      ne12,
                                                            int                      nb11,
                                                            int                      nb12,
                                                            int64_t                  nb13,
                                                            int                      nb21,
                                                            int                      nb22,
                                                            int64_t                  nb23,
                                                            int                      ne30,
                                                            int                      ne32,
                                                            int                      ne33,
                                                            int                      nb31,
                                                            int                      nb32,
                                                            int64_t                  nb33,
                                                            int64_t                  packed_batch_stride,
                                                            int64_t                  packed_head_stride,
                                                            int64_t                  packed_block_stride,
                                                            int                      n_partitions,
                                                            const sycl::nd_item<3> & item,
                                                            sycl::half *             slm) {
    static_assert(D == 64, "Split-KV GQA decode prototype is only implemented for GPT-OSS D=64");
    static_assert(TK == 16 || TK == 32, "Split-KV decode currently supports TK=16 or TK=32");
    static_assert(D % TK == 0, "D must be divisible by decode TK");
    (void) ne30;

    using slm_layout       = fattn_v2_decode_gqa_slm<D>;
    constexpr int NTHREADS = XMX_V2_DECODE_GQA_MAX * XMX_V2_SG;
    constexpr int D_TILES  = D / XMX_V2_SG;

    auto      sg    = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int lane  = static_cast<int>(sg.get_local_id());
    const int tid   = static_cast<int>(item.get_local_linear_id());

    const int  hb_id     = static_cast<int>(item.get_group(0));
    const int  sequence  = hb_id / ne12;
    const int  kv_head   = hb_id % ne12;
    const int  partition = static_cast<int>(item.get_group(1));
    const int  q_abs     = static_cast<int>(item.get_group(2));
    const int  kv_start  = partition * XMX_V2_DECODE_BATCH_KV;
    const int  gqa_ratio = ne02 / ne12;
    const int  q_rel     = sg_id;
    const int  head      = kv_head * gqa_ratio + q_rel;
    const bool active    = q_rel < gqa_ratio && head < ne02 && q_abs < ne01 && partition < n_partitions;

    const char * Q_ptr = Q_base + (int64_t) nb03 * sequence;
    const char * K_ptr = nullptr;
    if constexpr (!PACKED_K) {
        K_ptr = K_base + nb13 * sequence + (int64_t) nb12 * kv_head;
    }
    const sycl::half * K_packed_block = nullptr;
    if constexpr (PACKED_K) {
        const int64_t packed_block_offset_half =
            ((int64_t) sequence * packed_batch_stride + (int64_t) kv_head * packed_head_stride +
             (int64_t) partition * packed_block_stride) /
            (int64_t) sizeof(sycl::half);
        K_packed_block = K_packed_base + packed_block_offset_half;
    }
    const char * V_ptr = V_base + nb23 * sequence + (int64_t) nb22 * kv_head;

    const float slope = active ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 0.0f;

    const int          mask_head = active && ne32 > 1 ? head % ne32 : 0;
    const sycl::half * maskh =
        (active && maskh_base) ?
            reinterpret_cast<const sycl::half *>(maskh_base + (int64_t) nb33 * (sequence % ne33) +
                                                 (int64_t) nb32 * mask_head + (int64_t) nb31 * q_abs) :
            nullptr;

    sycl::half * tile_Q        = slm + slm_layout::Q_OFFSET;
    sycl::half * tile_K_packed = slm + slm_layout::K_PACKED_OFFSET;
    sycl::half * tile_V        = slm + slm_layout::V_OFFSET;
    float *      tile_S_f      = reinterpret_cast<float *>(slm + slm_layout::S_OFFSET);

    float VKQ[D_TILES];
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        VKQ[t] = 0.0f;
    }

    for (int idx = tid; idx < XMX_V2_DECODE_GQA_MAX * D; idx += NTHREADS) {
        const int q_slot = idx / D;
        const int d      = idx - q_slot * D;
        const int h      = kv_head * gqa_ratio + q_slot;
        if (q_slot < gqa_ratio && h < ne02 && q_abs < ne01) {
            const Q_type * Q_row_ptr =
                reinterpret_cast<const Q_type *>(Q_ptr + (int64_t) nb02 * h + (int64_t) nb01 * q_abs);
            tile_Q[q_slot * D + d] = sycl::half(static_cast<float>(Q_row_ptr[d]) * scale);
        } else {
            tile_Q[q_slot * D + d] = sycl::half(0.0f);
        }
    }
    sycl::group_barrier(item.get_group());

    for (int idx = tid; idx < XMX_V2_DECODE_BATCH_KV * D; idx += NTHREADS) {
        const int kv_local = idx / D;
        const int d        = idx - kv_local * D;
        const int kv_abs   = kv_start + kv_local;

        sycl::half v_val = sycl::half(0.0f);
        if (kv_abs < ne11) {
            v_val = reinterpret_cast<const sycl::half *>(V_ptr + (int64_t) nb21 * kv_abs)[d];
        }
        tile_V[kv_local * D + d] = v_val;

        if constexpr (!PACKED_K) {
            sycl::half k_val = sycl::half(0.0f);
            if (kv_abs < ne11) {
                k_val = reinterpret_cast<const sycl::half *>(K_ptr + (int64_t) nb11 * kv_abs)[d];
            }
            const int half_id     = kv_local / XMX_V2_DECODE_HALF_KV;
            const int compact_col = kv_local - half_id * XMX_V2_DECODE_HALF_KV;
            const int active_lane = compact_col % XMX_V2_DECODE_ACTIVE_LANES;
            const int slot        = compact_col / XMX_V2_DECODE_ACTIVE_LANES;
            const int active_col  = slot * XMX_V2_SG + active_lane;
            const int packed_idx =
                half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2) + active_col * 2 + (d & 1);
            tile_K_packed[packed_idx] = k_val;
        }
    }
    sycl::group_barrier(item.get_group());

    float lane_scores[2 * XMX_V2_DECODE_SLOTS];
#    pragma unroll
    for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
        lane_scores[i] = -FLT_MAX;
    }

#    pragma unroll
    for (int half_id = 0; half_id < 2; ++half_id) {
        sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, 1, XMX_V2_DECODE_TN> mat_QK;
        sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

#    pragma unroll
        for (int d = 0; d < D; d += TK) {
            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, 1, TK, sycl_xmx::layout::row_major>
                mat_Q;
            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, TK, XMX_V2_DECODE_TN,
                                   sycl_xmx::layout::ext_intel_packed>
                mat_K;

            sycl_xmx::joint_matrix_load(
                sg, mat_Q,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    &tile_Q[q_rel * D + d]),
                D);
            if constexpr (PACKED_K) {
                const sycl::half * K_packed_tile = K_packed_block + half_id * slm_layout::K_PACKED_PER_HALF +
                                                   (d / 2) * (XMX_V2_DECODE_TN * 2);
                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        K_packed_tile),
                    XMX_V2_DECODE_TN * 2);
            } else {
                sycl_xmx::joint_matrix_load(
                    sg, mat_K,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &tile_K_packed[half_id * slm_layout::K_PACKED_PER_HALF + (d / 2) * (XMX_V2_DECODE_TN * 2)]),
                    XMX_V2_DECODE_TN * 2);
            }
            sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_K, mat_QK);
        }

        int slot = 0;
        sycl_xmx::joint_matrix_apply(sg, mat_QK, [&](float & elem) {
            if (lane < XMX_V2_DECODE_ACTIVE_LANES && slot < XMX_V2_DECODE_SLOTS) {
                lane_scores[half_id * XMX_V2_DECODE_SLOTS + slot] = elem;
            }
            ++slot;
        });
    }

    float local_max = -FLT_MAX;
#    pragma unroll
    for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
        const int half_id  = i / XMX_V2_DECODE_SLOTS;
        const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
        const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
        const int kv_abs   = kv_start + kv_local;

        float score = lane_scores[i];
        if (!active || lane >= XMX_V2_DECODE_ACTIVE_LANES || kv_abs >= ne11) {
            score = -FLT_MAX;
        } else {
            if constexpr (use_logit_softcap) {
                score = logit_softcap * sycl::tanh(score);
            }
            if (maskh) {
                score += slope * static_cast<float>(maskh[kv_abs]);
            }
        }
        lane_scores[i] = score;
        local_max      = sycl::fmax(local_max, score);
    }

    KQ_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});

    float lane_probs[2 * XMX_V2_DECODE_SLOTS];
    float local_sum = 0.0f;
#    pragma unroll
    for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
        const int half_id  = i / XMX_V2_DECODE_SLOTS;
        const int slot     = i - half_id * XMX_V2_DECODE_SLOTS;
        const int kv_local = half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + lane;
        lane_probs[i]      = 0.0f;
        if (active && lane < XMX_V2_DECODE_ACTIVE_LANES) {
            const float p = sycl::exp(lane_scores[i] - KQ_max);
            lane_probs[i] = sycl::isfinite(p) ? p : 0.0f;
            if constexpr (!DIRECT_PV) {
                tile_S_f[q_rel * XMX_V2_DECODE_BATCH_KV + kv_local] = lane_probs[i];
            }
            local_sum += lane_probs[i];
        }
    }
    KQ_sum = sycl::reduce_over_group(sg, local_sum, sycl::plus<float>{});

    if constexpr (!DIRECT_PV) {
        sycl::group_barrier(item.get_group());
    }

    if (active) {
#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int d   = t * XMX_V2_SG + lane;
            float     acc = 0.0f;
            if constexpr (DIRECT_PV) {
#    pragma unroll
                for (int i = 0; i < 2 * XMX_V2_DECODE_SLOTS; ++i) {
                    const int half_id = i / XMX_V2_DECODE_SLOTS;
                    const int slot    = i - half_id * XMX_V2_DECODE_SLOTS;
#    pragma unroll
                    for (int src_lane = 0; src_lane < XMX_V2_DECODE_ACTIVE_LANES; ++src_lane) {
                        const int kv_local =
                            half_id * XMX_V2_DECODE_HALF_KV + slot * XMX_V2_DECODE_ACTIVE_LANES + src_lane;
                        if (kv_start + kv_local < ne11) {
                            const float p = sycl::select_from_group(sg, lane_probs[i], src_lane);
                            acc += p * static_cast<float>(tile_V[kv_local * D + d]);
                        }
                    }
                }
            } else {
#    pragma unroll
                for (int k = 0; k < XMX_V2_DECODE_BATCH_KV; ++k) {
                    if (kv_start + k < ne11) {
                        acc += tile_S_f[q_rel * XMX_V2_DECODE_BATCH_KV + k] * static_cast<float>(tile_V[k * D + d]);
                    }
                }
            }
            VKQ[t] = acc;
        }

        const size_t partial_idx =
            ((((size_t) sequence * (size_t) ne02 + (size_t) head) * (size_t) ne01 + (size_t) q_abs) *
                 (size_t) n_partitions +
             (size_t) partition);
        if (lane == 0) {
            partial_max[partial_idx] = KQ_max;
            partial_sum[partial_idx] = KQ_sum;
        }
#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int d                               = t * XMX_V2_SG + lane;
            partial_out[partial_idx * D + (size_t) d] = VKQ[t];
        }
    }

    if constexpr (!DIRECT_PV) {
        sycl::group_barrier(item.get_group());
    }
}

template <int D>
static void flash_attn_xmx_v2_decode_gqa_split_merge_kernel(const char * __restrict__ sinks_base,
                                                            const float * __restrict__ partial_max,
                                                            const float * __restrict__ partial_sum,
                                                            const float * __restrict__ partial_out,
                                                            float * __restrict__ dst,
                                                            int                      ne01,
                                                            int                      ne02,
                                                            int                      ne03,
                                                            int                      n_partitions,
                                                            const sycl::nd_item<3> & item) {
    static_assert(D == 64, "Split-KV GQA merge prototype is only implemented for GPT-OSS D=64");
    constexpr int D_TILES = D / XMX_V2_SG;

    auto      sg       = item.get_sub_group();
    const int lane     = static_cast<int>(sg.get_local_id());
    const int hb_id    = static_cast<int>(item.get_group(0));
    const int q_abs    = static_cast<int>(item.get_group(2));
    const int sequence = hb_id / ne02;
    const int head     = hb_id % ne02;

    if (sequence >= ne03 || q_abs >= ne01) {
        return;
    }

    const size_t partial_base =
        (((size_t) sequence * (size_t) ne02 + (size_t) head) * (size_t) ne01 + (size_t) q_abs) * (size_t) n_partitions;

    float max_val = -FLT_MAX / 2.0f;
    for (int p = 0; p < n_partitions; ++p) {
        max_val = sycl::fmax(max_val, partial_max[partial_base + (size_t) p]);
    }
    if (sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        max_val               = sycl::fmax(max_val, sinks_f[head]);
    }

    float denom = 0.0f;
    if (sinks_base) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks_base);
        denom += sycl::exp(sinks_f[head] - max_val);
    }
    for (int p = 0; p < n_partitions; ++p) {
        const float weight = sycl::exp(partial_max[partial_base + (size_t) p] - max_val);
        denom += partial_sum[partial_base + (size_t) p] * weight;
    }

    const float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    float *     dst_row = dst + (int64_t) D * (head + ne02 * (q_abs + ne01 * sequence));
    float acc[D_TILES];
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        acc[t] = 0.0f;
    }
    for (int p = 0; p < n_partitions; ++p) {
        const float weight = sycl::exp(partial_max[partial_base + (size_t) p] - max_val);
#    pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            const int d = t * XMX_V2_SG + lane;
            acc[t] += partial_out[(partial_base + (size_t) p) * D + (size_t) d] * weight;
        }
    }
#    pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
        const int   d   = t * XMX_V2_SG + lane;
        const float val = acc[t] * inv_sum;
        dst_row[d]      = sycl::isfinite(val) ? val : 0.0f;
    }
}

template <int D, bool use_logit_softcap, typename Q_type>
static void launch_fattn_xmx_v2_decode_gqa_leaf(const fattn_params & params, dpct::queue_ptr stream) {
    using slm_layout       = fattn_v2_decode_gqa_slm<D>;
    constexpr int nthreads = XMX_V2_DECODE_GQA_MAX * XMX_V2_SG;

    const int      n_query_blocks = params.ne01;
    sycl::range<3> grid(params.ne12 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, nthreads);

    const char *   Q_ptr       = params.Q;
    const char *   K_ptr       = params.K;
    const char *   V_ptr       = params.V;
    const char *   mask_ptr    = params.mask;
    const char *   sinks_ptr   = params.sinks;
    float *        dst_ptr     = params.dst;
    const float    scale_v     = params.scale;
    const float    max_bias    = params.max_bias;
    const float    m0          = params.m0;
    const float    m1          = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;
    const float    logit_sc    = params.logit_softcap;
    const int      ne01 = params.ne01, ne02 = params.ne02;
    const int      nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int      ne11 = params.ne11, ne12 = params.ne12;
    const int      nb11 = params.nb11, nb12 = params.nb12;
    const int64_t  nb13 = params.nb13;
    const int      nb21 = params.nb21, nb22 = params.nb22;
    const int64_t  nb23 = params.nb23;
    const int      ne30 = params.ne30, ne32 = params.ne32, ne33 = params.ne33;
    const int      nb31 = params.nb31, nb32 = params.nb32;
    const int64_t  nb33 = params.nb33;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(slm_layout::TOTAL), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                             flash_attn_xmx_v2_decode_gqa_kernel<D, use_logit_softcap, Q_type>(
                                 Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, scale_v, max_bias, m0, m1,
                                 n_head_log2, logit_sc, ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13,
                                 nb21, nb22, nb23, ne30, ne32, ne33, nb31, nb32, nb33, item,
                                 slm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false, bool PACKED_K = false>
static sycl::event launch_fattn_xmx_v2_decode_gqa_split_leaf(const fattn_params & params,
                                                             dpct::queue_ptr      stream,
                                                             float *              partial_max,
                                                             float *              partial_sum,
                                                             float *              partial_out,
                                                             int                  n_partitions,
                                                             const sycl::half *   packed_k_ptr         = nullptr,
                                                             int64_t              packed_batch_stride  = 0,
                                                             int64_t              packed_head_stride   = 0,
                                                             int64_t              packed_block_stride  = 0,
                                                             const sycl::event *  packed_k_ready_event = nullptr) {
    using slm_layout       = fattn_v2_decode_gqa_slm<D>;
    constexpr int nthreads = XMX_V2_DECODE_GQA_MAX * XMX_V2_SG;

    const int n_query_blocks = params.ne01;

    const char *      Q_ptr       = params.Q;
    const char *      K_ptr       = params.K;
    const char *      V_ptr       = params.V;
    const char *      mask_ptr    = params.mask;
    const char *      sinks_ptr   = params.sinks;
    float *           dst_ptr     = params.dst;
    const float       scale_v     = params.scale;
    const float       max_bias    = params.max_bias;
    const float       m0          = params.m0;
    const float       m1          = params.m1;
    const uint32_t    n_head_log2 = params.n_head_log2;
    const float       logit_sc    = params.logit_softcap;
    const int         ne01 = params.ne01, ne02 = params.ne02, ne03 = params.ne03;
    const int         nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int         ne11 = params.ne11, ne12 = params.ne12;
    const int         nb11 = params.nb11, nb12 = params.nb12;
    const int64_t     nb13 = params.nb13;
    const int         nb21 = params.nb21, nb22 = params.nb22;
    const int64_t     nb23 = params.nb23;
    const int         ne30 = params.ne30, ne32 = params.ne32, ne33 = params.ne33;
    const int         nb31 = params.nb31, nb32 = params.nb32;
    const int64_t     nb33               = params.nb33;
    const sycl::event packed_ready_event = packed_k_ready_event ? *packed_k_ready_event : sycl::event{};
    const bool        add_packed_dep =
        PACKED_K && packed_k_ready_event != nullptr && ggml_sycl_should_add_dependency(packed_ready_event);

    sycl::range<3> first_grid(params.ne12 * params.ne03, n_partitions, n_query_blocks);
    sycl::range<3> first_block(1, 1, nthreads);

    sycl::event first_event = stream->submit([&](sycl::handler & cgh) {
        if constexpr (PACKED_K) {
            if (add_packed_dep) {
                cgh.depends_on(*packed_k_ready_event);
            }
        }
        sycl::local_accessor<sycl::half, 1> slm(sycl::range<1>(slm_layout::TOTAL), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(first_grid * first_block, first_block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                flash_attn_xmx_v2_decode_gqa_split_first_kernel<D, use_logit_softcap, Q_type, TK, DIRECT_PV, PACKED_K>(
                    Q_ptr, K_ptr, packed_k_ptr, V_ptr, mask_ptr, partial_max, partial_sum, partial_out, scale_v,
                    max_bias, m0, m1, n_head_log2, logit_sc, ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13,
                    nb21, nb22, nb23, ne30, ne32, ne33, nb31, nb32, nb33, packed_batch_stride, packed_head_stride,
                    packed_block_stride, n_partitions, item, slm.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });

    sycl::range<3> merge_grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> merge_block(1, 1, XMX_V2_SG);

    sycl::event merge_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(first_event);
        cgh.parallel_for(sycl::nd_range<3>(merge_grid * merge_block, merge_block),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_V2_SG)]] {
                             flash_attn_xmx_v2_decode_gqa_split_merge_kernel<D>(sinks_ptr, partial_max, partial_sum,
                                                                                partial_out, dst_ptr, ne01, ne02, ne03,
                                                                                n_partitions, item);
                         });
    });
    return merge_event;
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_m1n64(ggml_backend_sycl_context & ctx,
                                      const fattn_params &        params,
                                      dpct::queue_ptr             stream) {
    (void) ctx;
    if constexpr (D != 64) {
        return false;
    } else {
        if (params.ne01 <= 0 || params.kv_is_fp8 || params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
            return false;
        }
        if (params.mask != nullptr && params.mask_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
            params.seq_lens != nullptr) {
            return false;
        }
        if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
            params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
            return false;
        }
        if (params.multi_token_decode || params.q_positions != nullptr) {
            return false;
        }
        if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0) {
            return false;
        }

        const sycl::device dev = stream->get_device();
        if (!fattn_xmx_v2_decode_m1n64_supported(dev)) {
            return false;
        }
        const size_t required_bytes = fattn_v2_decode_slm<D>::TOTAL * sizeof(sycl::half);
        const size_t local_mem      = dev.get_info<sycl::info::device::local_mem_size>();
        if (local_mem < required_bytes) {
            return false;
        }

        launch_fattn_xmx_v2_decode_m1n64_leaf<D, use_logit_softcap, Q_type>(params, stream);
        return true;
    }
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa(ggml_backend_sycl_context & ctx,
                                    const fattn_params &        params,
                                    dpct::queue_ptr             stream) {
    (void) ctx;
    if constexpr (D != 64) {
        return false;
    } else {
        if (params.ne01 != 1 || params.kv_is_fp8 || params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
            return false;
        }
        if (params.mask != nullptr && params.mask_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
            params.seq_lens != nullptr) {
            return false;
        }
        if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
            params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
            return false;
        }
        if (params.multi_token_decode || params.q_positions != nullptr) {
            return false;
        }
        if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0) {
            return false;
        }
        const int gqa_ratio = params.ne02 / params.ne12;
        if (gqa_ratio <= 0 || gqa_ratio > XMX_V2_DECODE_GQA_MAX) {
            return false;
        }

        const sycl::device dev = stream->get_device();
        if (!fattn_xmx_v2_decode_m1n64_supported(dev)) {
            return false;
        }
        const size_t required_bytes = fattn_v2_decode_gqa_slm<D>::TOTAL * sizeof(sycl::half);
        const size_t local_mem      = dev.get_info<sycl::info::device::local_mem_size>();
        if (local_mem < required_bytes) {
            return false;
        }

        launch_fattn_xmx_v2_decode_gqa_leaf<D, use_logit_softcap, Q_type>(params, stream);
        return true;
    }
}

template <int D, int TK>
static bool fattn_xmx_v2_decode_gqa_split_supported(const fattn_params & params,
                                                    dpct::queue_ptr      stream,
                                                    float *              partial_max,
                                                    float *              partial_sum,
                                                    float *              partial_out,
                                                    int                  n_partitions) {
    if constexpr (D != 64) {
        return false;
    } else {
        static_assert(TK == 16 || TK == 32, "Split-KV decode currently supports TK=16 or TK=32");
        if (!partial_max || !partial_sum || !partial_out || n_partitions <= 0) {
            return false;
        }
        if (params.ne01 != 1 || params.kv_is_fp8 || params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
            return false;
        }
        if (params.mask != nullptr && params.mask_type != GGML_TYPE_F16) {
            return false;
        }
        if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
            params.seq_lens != nullptr) {
            return false;
        }
        if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
            params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
            return false;
        }
        if (params.multi_token_decode || params.q_positions != nullptr) {
            return false;
        }
        if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0 || params.ne11 <= 0) {
            return false;
        }
        const int gqa_ratio = params.ne02 / params.ne12;
        if (gqa_ratio <= 0 || gqa_ratio > XMX_V2_DECODE_GQA_MAX) {
            return false;
        }
        const int expected_partitions = (params.ne11 + XMX_V2_DECODE_BATCH_KV - 1) / XMX_V2_DECODE_BATCH_KV;
        if (n_partitions != expected_partitions) {
            return false;
        }

        const sycl::device dev = stream->get_device();
        if (!fattn_xmx_v2_decode_m1n64_supported(dev, TK)) {
            return false;
        }
        const size_t required_bytes = fattn_v2_decode_gqa_slm<D>::TOTAL * sizeof(sycl::half);
        const size_t local_mem      = dev.get_info<sycl::info::device::local_mem_size>();
        if (local_mem < required_bytes) {
            return false;
        }

        return true;
    }
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false>
bool launch_fattn_xmx_v2_decode_gqa_split_tk(ggml_backend_sycl_context & ctx,
                                             const fattn_params &        params,
                                             dpct::queue_ptr             stream,
                                             float *                     partial_max,
                                             float *                     partial_sum,
                                             float *                     partial_out,
                                             int                         n_partitions) {
    (void) ctx;
    if constexpr (D != 64) {
        return false;
    } else {
        if (!fattn_xmx_v2_decode_gqa_split_supported<D, TK>(params, stream, partial_max, partial_sum, partial_out,
                                                            n_partitions)) {
            return false;
        }

        (void) launch_fattn_xmx_v2_decode_gqa_split_leaf<D, use_logit_softcap, Q_type, TK, DIRECT_PV, false>(
            params, stream, partial_max, partial_sum, partial_out, n_partitions);
        return true;
    }
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false>
bool launch_fattn_xmx_v2_decode_gqa_split_packed_tk(ggml_backend_sycl_context &    ctx,
                                                    const fattn_params &           params,
                                                    dpct::queue_ptr                stream,
                                                    ggml_sycl_fattn_xmx_packed_k * packed_k,
                                                    float *                        partial_max,
                                                    float *                        partial_sum,
                                                    float *                        partial_out,
                                                    int                            n_partitions) {
    if constexpr (D != 64) {
        return false;
    } else {
        if (!fattn_xmx_v2_decode_gqa_split_supported<D, TK>(params, stream, partial_max, partial_sum, partial_out,
                                                            n_partitions)) {
            return false;
        }
        if (packed_k == nullptr || packed_k->device != ctx.device || packed_k->D != D ||
            packed_k->n_kv < params.ne11 || packed_k->H_kv != params.ne12 || packed_k->batch != params.ne03 ||
            packed_k->n_blocks < n_partitions) {
            return false;
        }

        const size_t block_stride = GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK;
        if (packed_k->n_blocks <= 0 || packed_k->H_kv <= 0 || packed_k->batch <= 0) {
            return false;
        }
        if ((size_t) packed_k->n_blocks > std::numeric_limits<size_t>::max() / block_stride) {
            return false;
        }
        const size_t head_stride = (size_t) packed_k->n_blocks * block_stride;
        if ((size_t) packed_k->H_kv > std::numeric_limits<size_t>::max() / head_stride) {
            return false;
        }
        const size_t batch_stride = (size_t) packed_k->H_kv * head_stride;
        if ((size_t) packed_k->batch > std::numeric_limits<size_t>::max() / batch_stride) {
            return false;
        }
        const size_t required_bytes = (size_t) packed_k->batch * batch_stride;
        if (packed_k->total_bytes < required_bytes) {
            return false;
        }

        const ggml_sycl::resolved_ptr resolved = packed_k->handle.resolve(ctx.device);
        if (!resolved || !resolved.on_device) {
            return false;
        }

        sycl::event merge_event =
            launch_fattn_xmx_v2_decode_gqa_split_leaf<D, use_logit_softcap, Q_type, TK, DIRECT_PV, true>(
                params, stream, partial_max, partial_sum, partial_out, n_partitions,
                static_cast<const sycl::half *>(resolved.ptr), (int64_t) batch_stride, (int64_t) head_stride,
                (int64_t) block_stride, &packed_k->ready_event);
        packed_k->ready_event = merge_event;
        return true;
    }
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa_split_packed(ggml_backend_sycl_context &    ctx,
                                                 const fattn_params &           params,
                                                 dpct::queue_ptr                stream,
                                                 ggml_sycl_fattn_xmx_packed_k * packed_k,
                                                 float *                        partial_max,
                                                 float *                        partial_sum,
                                                 float *                        partial_out,
                                                 int                            n_partitions) {
    return launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, use_logit_softcap, Q_type, XMX_V2_DECODE_TK>(
        ctx, params, stream, packed_k, partial_max, partial_sum, partial_out, n_partitions);
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa_split(ggml_backend_sycl_context & ctx,
                                          const fattn_params &        params,
                                          dpct::queue_ptr             stream,
                                          float *                     partial_max,
                                          float *                     partial_sum,
                                          float *                     partial_out,
                                          int                         n_partitions) {
    return launch_fattn_xmx_v2_decode_gqa_split_tk<D, use_logit_softcap, Q_type, XMX_V2_DECODE_TK>(
        ctx, params, stream, partial_max, partial_sum, partial_out, n_partitions);
}

// =============================================================================
// Public launcher — resolves the runtime-picked variant index to a compile-time
// leaf specialization and dispatches. Returns true if the kernel was launched,
// false if the caller must fall back (e.g. device SLM cannot fit the fallback
// leaf's worst case). Mirrors the bool-return pattern of
// ggml_sycl_flash_attn_ext_onednn() so fattn.cpp can use the same
// `if (launcher(...)) return;` idiom.
// =============================================================================

template <int D, int ncols, bool use_logit_softcap, typename Q_type, typename Acc_t, bool kv_is_fp8 = false>
bool launch_fattn_xmx_v2_f16_variant(ggml_backend_sycl_context & ctx,
                                     const fattn_params &        params,
                                     dpct::queue_ptr             stream,
                                     int                         forced_variant) {
    const sycl::device dev = stream->get_device();
    (void) fattn_xmx_v2_pick_variant_cached(ctx, dev);

    std::size_t variant_idx = 0;
    if (forced_variant >= 0) {
        variant_idx = static_cast<std::size_t>(forced_variant);
        if (!fattn_xmx_v2_variant_supported(dev, variant_idx)) {
            return false;
        }
    } else {
        variant_idx = fattn_xmx_v2_pick_variant_for_shape(dev, D, ncols, params.ne01);
    }

    // SLM-fit gate: if the fallback leaf's worst case doesn't fit this device's
    // local_mem_size, bail to caller-level fallback (TILE path). Logged once at
    // cache init so there's a paper trail in stderr.
    const auto * cache = static_cast<const fattn_xmx_v2_device_cache *>(ctx.fattn_xmx_v2_cache);
    if (cache && !cache->slm_ok) {
        return false;
    }

    switch (variant_idx) {
        // variant 1: (16, 16, 16) — larger tile for better XMX utilization
        case 1:
            launch_fattn_xmx_v2_f16_leaf<D, ncols, use_logit_softcap, Q_type, Acc_t, kv_is_fp8,
                                         /*TM=*/16, /*TK=*/16, /*TN=*/16>(params, stream);
            break;
        default:
            launch_fattn_xmx_v2_f16_leaf<D, ncols, use_logit_softcap, Q_type, Acc_t, kv_is_fp8,
                                         /*TM=*/8, /*TK=*/16, /*TN=*/16>(params, stream);
            break;
    }
    return true;
}

template <int D, int ncols, bool use_logit_softcap, typename Q_type, typename Acc_t, bool kv_is_fp8 = false>
bool launch_fattn_xmx_v2_f16(ggml_backend_sycl_context & ctx, const fattn_params & params, dpct::queue_ptr stream) {
    return launch_fattn_xmx_v2_f16_variant<D, ncols, use_logit_softcap, Q_type, Acc_t, kv_is_fp8>(ctx, params, stream,
                                                                                                  -1);
}

// =============================================================================
// Availability check (mirrors fattn_xmx_f16_available)
// =============================================================================

inline bool fattn_xmx_v2_f16_available() {
    return true;  // compile-time gate: SYCL_XMX_V2_AVAILABLE == 1
}

#else   // !SYCL_XMX_V2_AVAILABLE

template <int D, int ncols, bool use_logit_softcap, typename Q_type, typename Acc_t, bool kv_is_fp8 = false>
bool launch_fattn_xmx_v2_f16_variant(ggml_backend_sycl_context &, const fattn_params &, dpct::queue_ptr, int) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, int ncols, bool use_logit_softcap, typename Q_type, typename Acc_t, bool kv_is_fp8 = false>
bool launch_fattn_xmx_v2_f16(ggml_backend_sycl_context &, const fattn_params &, dpct::queue_ptr) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_m1n64(ggml_backend_sycl_context &, const fattn_params &, dpct::queue_ptr) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa(ggml_backend_sycl_context &, const fattn_params &, dpct::queue_ptr) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false>
bool launch_fattn_xmx_v2_decode_gqa_split_tk(ggml_backend_sycl_context &,
                                             const fattn_params &,
                                             dpct::queue_ptr,
                                             float *,
                                             float *,
                                             float *,
                                             int) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, bool use_logit_softcap, typename Q_type, int TK, bool DIRECT_PV = false>
bool launch_fattn_xmx_v2_decode_gqa_split_packed_tk(ggml_backend_sycl_context &,
                                                    const fattn_params &,
                                                    dpct::queue_ptr,
                                                    ggml_sycl_fattn_xmx_packed_k *,
                                                    float *,
                                                    float *,
                                                    float *,
                                                    int) {
    GGML_ABORT("XMX v2 not available at compile time");
    return false;
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa_split_packed(ggml_backend_sycl_context &    ctx,
                                                 const fattn_params &           params,
                                                 dpct::queue_ptr                stream,
                                                 ggml_sycl_fattn_xmx_packed_k * packed_k,
                                                 float *                        partial_max,
                                                 float *                        partial_sum,
                                                 float *                        partial_out,
                                                 int                            n_partitions) {
    return launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, use_logit_softcap, Q_type, XMX_V2_DECODE_TK>(
        ctx, params, stream, packed_k, partial_max, partial_sum, partial_out, n_partitions);
}

template <int D, bool use_logit_softcap, typename Q_type>
bool launch_fattn_xmx_v2_decode_gqa_split(ggml_backend_sycl_context & ctx,
                                          const fattn_params &        params,
                                          dpct::queue_ptr             stream,
                                          float *                     partial_max,
                                          float *                     partial_sum,
                                          float *                     partial_out,
                                          int                         n_partitions) {
    return launch_fattn_xmx_v2_decode_gqa_split_tk<D, use_logit_softcap, Q_type, XMX_V2_DECODE_TK>(
        ctx, params, stream, partial_max, partial_sum, partial_out, n_partitions);
}

inline bool fattn_xmx_v2_f16_available() {
    return false;
}

// No-op destroy for the compile-disabled path — the ctx destructor still calls
// ggml_sycl_fattn_xmx_v2_cache_destroy unconditionally.
inline void fattn_xmx_v2_cache_destroy_inline(void *) {}

#endif  // SYCL_XMX_V2_AVAILABLE

#endif  // GGML_SYCL_FATTN_XMX_F16_V2_HPP
