// This file previously had no include guard because only fattn-tile.cpp
// included it. llama.cpp-dtpk adds a second includer (fattn.cpp, for the
// D=512 launcher below), so guard it now against any future diamond include.
#ifndef GGML_SYCL_FATTN_TILE_HPP
#define GGML_SYCL_FATTN_TILE_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"

#include <cmath>
#include <float.h>

namespace syclex = sycl::ext::oneapi::experimental;

// SYCL_FLASH_ATTN / SYCL_FAST_FP16 (llama.cpp-dtpk): this file was dpct-
// migrated from an upstream CUDA kernel that gated its body behind
// capability macros neither of which this fork ever defined anywhere
// (verified: a whole-tree grep finds only this guard's own #ifdef/#endif
// pairs below). Left undefined, every instantiation of flash_attn_tile<>
// compiled to a silent no-op -- GGML_UNUSED_VARS on every parameter, dst
// never written, no crash, no abort. Defined here because this file's
// kernel is getting a real caller for the first time (D=512 tile-path FA,
// the gemma-viable route jahv's oneDNN wiring cannot serve). FAST_FP16 is
// unconditional -- not tied to GGML_SYCL_F16 -- to stay consistent with
// fast_fp16_available()'s own unconditional `return true` (common.hpp):
// that function drives the RUNTIME (cc-taking) overload of
// ggml_sycl_fattn_tile_get_nthreads() that any launcher uses to size its
// nd_range, while this macro drives the COMPILE-TIME-only overload the
// kernel body uses internally to size its own local-memory layout: tying
// FAST_FP16 to a build flag fast_fp16_available() ignores would let those
// two disagree in a GGML_SYCL_F16=OFF build, mismatching the launched
// work-group shape against the kernel's actual SLM allocation.
#define SYCL_FLASH_ATTN 1
#define SYCL_FAST_FP16 1

// Missing dpct-migration utilities (llama.cpp-dtpk): flash_attn_tile<>'s
// body called three CUDA-intrinsic-named helpers (make_half2, make_float2,
// ggml_sycl_mad) that existed nowhere in this fork under any name
// (confirmed by a whole-tree grep) -- another consequence of this file
// never having compiled before (see the SYCL_FLASH_ATTN/SYCL_FAST_FP16
// comment above; this was found only once those macros let the compiler
// actually reach these call sites for the first time). Defined here with a
// file-scoped `fattn_tile_` prefix (spec review rev-dtpk-spec nit) rather
// than the original CUDA names, both to read as this file's own utilities
// and to avoid ever colliding with a same-named helper landing elsewhere in
// this translation unit (fattn.cpp now includes this header too).
//
// fattn_tile_make_half2/fattn_tile_make_float2 are exactly what this same
// file already spells out inline elsewhere as
// sycl::half2(a,b)/sycl::float2(a,b) constructor calls (e.g. the
// `KQ_max_scale_h2 = sycl::half2(...)` pattern further down) -- these are
// thin aliases for that constructor, the one call site that was never
// converted off its CUDA name.
//
// fattn_tile_mad is a half2-dot-into-float / float-FMA accumulate, inferred
// from its two call shapes at the single site that uses it (K_k/Q_k are
// sycl::half2 under SYCL_FAST_FP16, plain float otherwise) and
// cross-checked two ways: against this file's own already-working,
// already-compiling scalar half2-dot-product pattern in
// vec_dot_fattn_vec_KQ_f16 (fattn-common.hpp: `sum += x.x()*y.x() +
// x.y()*y.y()`), and against the canonical in-tree reference for the same
// primitive under its original name, ggml_cuda_mad
// (ggml/src/ggml-cuda/common.cuh:743-769). Attribution is per branch, not
// per type name: this file's non-FAST_FP16 float overload matches
// ggml_cuda_mad's plain `(float, float, float)` overload (`acc += v*u`,
// common.cuh:744-746); this file's SYCL_FAST_FP16 half2 overload matches
// ggml_cuda_mad's `(float, float2, float2)` overload's FORMULA (`acc +=
// v.x*u.x; acc += v.y*u.y`, common.cuh:748-750) rather than its `(float,
// half2, half2)` overload (common.cuh:757-769) -- that one additionally
// branches on hardware dot-product availability (AMD `v_dot2_f32_f16` /
// `FAST_FP16_AVAILABLE`), which this file's fattn_tile_mad does not
// replicate and does not need to: this fork always resolves the
// SYCL_FAST_FP16 config table per fast_fp16_available()'s unconditional
// `true`, so there is no hardware-capability branch to make here.
static __dpct_inline__ sycl::half2 fattn_tile_make_half2(float x, float y) {
    return sycl::half2(x, y);
}

static __dpct_inline__ sycl::float2 fattn_tile_make_float2(float x, float y) {
    return sycl::float2(x, y);
}

static __dpct_inline__ void fattn_tile_mad(float & acc, const sycl::half2 & a, const sycl::half2 & b) {
    acc += static_cast<float>(a.x()) * static_cast<float>(b.x()) +
           static_cast<float>(a.y()) * static_cast<float>(b.y());
}

static __dpct_inline__ void fattn_tile_mad(float & acc, const float & a, const float & b) {
    acc += a * b;
}

#define GGML_SYCL_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

static constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp16(const int DKQ, const int DV, const int ncols) {
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  64,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2,  64,  64)

    return 0;
}

static constexpr uint32_t ggml_sycl_fattn_tile_get_config_fp32(const int DKQ, const int DV, const int ncols) {
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 3,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 3,  64, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  2, 128, 3,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 3,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  32, 256)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  2, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2,  64,  64)

    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  32,  64)
    GGML_SYCL_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  32,  64)

    return 0;
}

static constexpr uint32_t ggml_sycl_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if(fast_fp16_available(cc))
        return ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
    else
        return ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
}

static constexpr uint32_t ggml_sycl_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef SYCL_FAST_FP16
    return ggml_sycl_fattn_tile_get_config_fp16(DKQ, DV, ncols);
#else
    return ggml_sycl_fattn_tile_get_config_fp32(DKQ, DV, ncols);
#endif // SYCL_FAST_FP16
}

static int ggml_sycl_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

static constexpr int ggml_sycl_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static int ggml_sycl_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

static constexpr int ggml_sycl_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

static int ggml_sycl_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

static constexpr int ggml_sycl_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static int ggml_sycl_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

static constexpr int ggml_sycl_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
    return (ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

template <int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __dpct_inline__ void flash_attn_tile_load_tile(const sycl::half2 * const __restrict__ KV,
                                                      sycl::half2 * const __restrict__ tile_KV,
                                                      const int stride_KV,
                                                      const int i_sup) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_size ? 0 : item_ct1.get_local_id(2) / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0 * cpy_ne + (stride_j == warp_size ? item_ct1.get_local_id(2) :
                                                                         item_ct1.get_local_id(2) % stride_j) *
                                                    cpy_ne;

                    const __dpct_align__(16) sycl::half2 zero[cpy_ne] = {
                        { 0.0f, 0.0f }
                    };
                    ggml_sycl_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };
    // 1: max 64*16=512 bytes, 512 half
    // 2: max 32*16=512 bytes, 256 half
    // 3: max 16*16=256 bytes, 128 half
    // 4: max  8*16=128 bytes,  64 half
    // 5: max  4*16= 64 bytes,  32 half
    // 6: max  2*16= 32 bytes,  16 half
    // 7: max  1*16= 16 bytes,   8 half
    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_sycl_unroll<7>{}(load);
}

template <int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __dpct_inline__ void flash_attn_tile_load_tile(const sycl::half2 * const __restrict__ KV,
                                                      float * const __restrict__ tile_KV,
                                                      const int stride_KV,
                                                      const int i_sup) {
    constexpr int cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] (const int n) {
        auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + item_ct1.get_local_id(1) * stride_i +
                          (stride_j == warp_size ? 0 : item_ct1.get_local_id(2) / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0 * (cpy_ne / 2) + (stride_j == warp_size ? item_ct1.get_local_id(2) :
                                                                               item_ct1.get_local_id(2) % stride_j) *
                                                          (cpy_ne / 2);

                    const sycl::half2 zero[cpy_ne / 2] = {
                        { 0.0f, 0.0f }
                    };
                    __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne / 2];
                    ggml_sycl_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __dpct_align__(16) sycl::float2 tmp_f2[cpy_ne / 2];
#pragma unroll
                    for (int l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = tmp_h2[l].template convert<float, sycl::rounding_mode::automatic>();
                    }
                    ggml_sycl_memcpy_1<sizeof(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                }
            }
        }
    };
    // 1: max 32*16=512 bytes, 128 float
    // 2: max 16*16=256 bytes,  64 float
    // 3: max  8*16=128 bytes,  32 float
    // 4: max  4*16= 64 bytes,  16 float
    // 5: max  2*16= 32 bytes,   8 float
    static_assert(J % 8 == 0, "bad J");
    static_assert(J % cpy_ne == 0, "bad J");
    ggml_sycl_unroll<5>{}(load);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int  warp_size,
          int  nwarps,
          int  ncols1,
          int  ncols2,
          int  DKQ,
          int  nbatch_fa,
          int  nbatch_K,
          bool use_logit_softcap,
          bool oob_check,
          typename T_vec_dot>
static __dpct_inline__ void flash_attn_tile_iter_KQ(T_vec_dot * const Q_tmp,
                                                    const sycl::half2 * const __restrict__ K_h2,
                                                    T_vec_dot * const KV_tmp,
                                                    const int         stride_K2,
                                                    const int         k_VKQ_0,
                                                    const int         k_VKQ_sup,
                                                    const int         k_KQ_0,
                                                    float *           KQ_acc) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) sycl::half2 K_k[nbatch_fa / (np * warp_size)][cpy_ne];
        __dpct_align__(16) sycl::half2 Q_k[cpw][cpy_ne];
#else
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __dpct_align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __dpct_align__(16) float Q_k[cpw][cpy_ne];
#endif // SYCL_FAST_FP16

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#else
            ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

#ifdef SYCL_FAST_FP16
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#else
            ggml_sycl_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    fattn_tile_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        item_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int  warp_size,
          int  nwarps,
          int  ncols1,
          int  ncols2,
          int  DKQ,
          int  DV,
          int  nbatch_fa,
          int  nbatch_K,
          bool use_logit_softcap,
          bool oob_check,
          typename T_vec_dot,
          typename T_KQ,
          typename T_acc>
/*
The total declared local variable size in device function flash_attn_tile_iter exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void flash_attn_tile_iter(T_vec_dot * const Q_tmp,
                                                 const sycl::half2 * const __restrict__ K_h2,
                                                 const sycl::half2 * const __restrict__ V_h2,
                                                 const sycl::half * const __restrict__ mask,
                                                 const sycl::uint3 ne01,
                                                 const float       logit_softcap,
                                                 const float       slope,
                                                 T_KQ * const      KQ,
                                                 T_vec_dot * const KV_tmp,
                                                 const int         stride_K2,
                                                 const int         stride_V2,
                                                 const int         stride_mask,
                                                 float * const     KQ_max,
                                                 float * const     KQ_sum,
                                                 T_acc * const     VKQ,
                                                 const int         k_VKQ_0,
                                                 const int         k_VKQ_max,
                                                 const int         col_Q_0,
                                                 float *           KQ_max_new_shared) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int cpy_nb   = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

#ifdef SYCL_FAST_FP16
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // SYCL_FAST_FP16
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // Per KQ_acc element: its cell's mask is -inf. Dead is decided where the
    // mask is read, not from the score later (see fattn_weight_mark_dead).
    bool KQ_dead[nbatch_fa / (np * warp_size) * cpw] = {};

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int j = fastmodulo(col_Q_0 + (jc0 + (item_ct1.get_local_id(1) / np) * cpw) / ncols2, ne01);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

#if defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0] =
                    logit_softcap * sycl::tanh((float) KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                if (ncols2 > 1 || mask) {
                    // Select, not add: a dead cell's K may be non-finite and
                    // NaN + -inf is NaN (see fattn_mask_is_dead).
                    const sycl::half mask_val = mask[j * stride_mask + k_VKQ_0 + i_KQ];
                    const int        i_acc    = (i_KQ_0 / (np * warp_size)) * cpw + jc0;
                    KQ_dead[i_acc]            = fattn_mask_is_dead(mask_val);
                    KQ_acc[i_acc]             = fattn_mask_apply(KQ_acc[i_acc], slope, mask_val);
                }

                KQ_max_new[jc0] =
                    sycl::fmax((float) KQ_max_new[jc0],
                               (float) (KQ_acc[(i_KQ_0 / (np * warp_size)) * cpw + jc0] + FATTN_KQ_MAX_OFFSET));
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    if constexpr (np == 1) {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } else {
        static_assert(cpw == 1, "bad cpw");

        if (item_ct1.get_local_id(2) == 0) {
            KQ_max_new_shared[item_ct1.get_local_id(1)] = KQ_max_new[0];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        KQ_max_new[0] = KQ_max_new_shared[(item_ct1.get_local_id(1) & ~(np - 1)) + item_ct1.get_local_id(2) % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef SYCL_FAST_FP16
        __dpct_align__(16) sycl::half tmp[nbatch_fa / (np * warp_size)][KQ_cs];
#else
        __dpct_align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // SYCL_FAST_FP16

#pragma unroll
        for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[jc] - KQ_max_new[jc]));
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float KQ_val = (float) KQ_acc[(i0 / (np * warp_size)) * cpw + jc];
                const float val =
                    !oob_check || i0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2) <
                                      static_cast<uint32_t>(k_VKQ_sup) ?
                        fattn_weight_mark_dead(KQ_dead[(i0 / (np * warp_size)) * cpw + jc],
                                               sycl::native::exp(KQ_val - (float) KQ_max[jc])) :
                        0.0f;
                KQ_sum_add += fattn_weight_sum_term(val);
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#ifdef SYCL_FAST_FP16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale_h2.x();
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale_h2.y();
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
            const int i = i0 + (item_ct1.get_local_id(1) % np) * warp_size + item_ct1.get_local_id(2);

            ggml_sycl_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs)) * (nbatch_fa * KQ_cs) + i * KQ_cs,
                tmp[i0 / (np * warp_size)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef SYCL_FAST_FP16
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::half2 V_k[(DVp / 2) / warp_size];
            __dpct_align__(16) sycl::half2 KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_size],
                                                 &KV_tmp[(k1 + item_ct1.get_local_id(1) % np) * (DV / 2) + i0 +
                                                         item_ct1.get_local_id(2) * cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0 / KQ_cs + (item_ct1.get_local_id(1) / np) * (cpw / KQ_cs);

                __dpct_align__(16) sycl::half tmp[KQ_cs];
                ggml_sycl_memcpy_1<KQ_cs * sizeof(sycl::half)>(
                    &tmp, KQ + jc_KQ * (nbatch_fa * KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np) * KQ_cs);
#pragma unroll
                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = sycl::half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                // A dead cell's V may be non-finite; skip it rather than add
                // 0 * V (see fattn_weight_mark_dead). KQ_k is the same across
                // the warp, and the test is made once per column.
                if (fattn_weight_is_dead(static_cast<float>(KQ_k[jc_VKQ_0].x()))) {
                    continue;
                }
#pragma unroll
                for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x() +=
                        V_k[i0/warp_size].x()*KQ_k[jc_VKQ_0].x();
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y() +=
                        V_k[i0/warp_size].y()*KQ_k[jc_VKQ_0].y();
                }
            }
        }
#else
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __dpct_align__(16) sycl::float2 V_k[(DVp/2)/warp_size];
            __dpct_align__(16) float  KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + item_ct1.get_local_id(1) % np)*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (item_ct1.get_local_id(1) / np)*(cpw/KQ_cs);

                ggml_sycl_memcpy_1<KQ_cs*sizeof(float)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + item_ct1.get_local_id(1) % np)*KQ_cs);
            }

#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                // A dead cell's V may be non-finite; skip it, once per column
                // (see fattn_weight_mark_dead).
                if (fattn_weight_is_dead(KQ_k[jc_VKQ_0])) {
                    continue;
                }
#pragma unroll
                for (int i0 = 0; i0 < DVp / 2; i0 += warp_size) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x() += V_k[i0/warp_size].x()*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y() += V_k[i0/warp_size].y()*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // SYCL_FAST_FP16
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, int warp_size>  // D == head size
/*
The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
static void flash_attn_tile(const char *  Q,
                            const char *  K,
                            const char *  V,
                            const char *  mask,
                            const char *  sinks,
                            const int *  KV_max,
                            float *  dst,
                            sycl::float2 *  dst_meta,
                            const float          scale,
                            const float          max_bias,
                            const float          m0,
                            const float          m1,
                            const uint32_t       n_head_log2,
                            const float          logit_softcap,
                            const int32_t        ne00,
                            const sycl::uint3    ne01,
                            const int32_t        ne02,
                            const int32_t        ne03,
                            const int32_t        nb01,
                            const int32_t        nb02,
                            const int32_t        nb03,
                            const int32_t        ne10,
                            const int32_t        ne11,
                            const int32_t        ne12,
                            const int32_t        ne13,
                            const int32_t        nb11,
                            const int32_t        nb12,
                            const int64_t        nb13,
                            const int32_t        nb21,
                            const int32_t        nb22,
                            const int64_t        nb23,
                            const int32_t        ne31,
                            const int32_t        ne32,
                            const int32_t        ne33,
                            const int32_t        nb31,
                            const int32_t        nb32,
                            const int64_t        nb33) {
#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if ((use_logit_softcap && !(DV == 128 || DV == 256))) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        return;
    }

    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;

    constexpr int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_sycl_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int col_Q_0 = item_ct1.get_group(2) * ncols1;  // Index of the first Q column for this SYCL block to work on.

    const int           sequence  = item_ct1.get_group(0) / (ne02 / ncols2);
    const int           head0     = item_ct1.get_group(0) * ncols2 - sequence * ne02;  // == item_ct1.get_group(0) % (ne02/ncols2)
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const sycl::half2 * K_h2      = (const sycl::half2 *) (K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
    const sycl::half2 * V_h2 =
        (const sycl::half2 *) (V + nb23 * sequence + nb22 * (head0 / gqa_ratio));  // K and V have same shape

    const sycl::half * maskh = mask ? (const sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;

    const int stride_K2   = nb11 / sizeof(sycl::half2);
    const int stride_V2   = nb21 / sizeof(sycl::half2);
    const int stride_mask = nb31 / sizeof(sycl::half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.

    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.


#ifdef SYCL_FAST_FP16
    constexpr size_t lsm_size1 = ncols * DKQ/2 ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV ;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_size = lsm_size1 * sizeof(sycl::half2) +
                                            lsm_size2 * sizeof(sycl::half2) +
                                            lsm_size3 * sizeof(sycl::half) +
                                            lsm_size4 * sizeof(float);

    syclex::work_group_static<char[local_share_mem_size]> lsm;

    sycl::half2 *Q_tmp = (sycl::half2 *)&lsm;
    sycl::half2 *KV_tmp = (sycl::half2*)(Q_tmp +lsm_size1);
    sycl::half *KQ = (sycl::half *)(KV_tmp+lsm_size2);
    float *KQ_max_new_shared = (float *)(KQ+lsm_size3);

    __dpct_align__(16) sycl::half2 VKQ[cpw * ((DVp / 2) / warp_size)] = {
        { 0.0f, 0.0f }
    };
#else
    constexpr size_t lsm_size1 = ncols * DKQ ;
    constexpr size_t lsm_size2 = nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV;
    constexpr size_t lsm_size3 = ncols * nbatch_fa;
    constexpr size_t lsm_size4 = nwarps;

    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 +lsm_size3 + lsm_size4) * sizeof(float);

    syclex::work_group_static<char[local_share_mem_size]> lsm;

    float *Q_tmp = (float *)&lsm;
    float *KV_tmp = Q_tmp +lsm_size1;
    float *KQ = KV_tmp+lsm_size2;
    float *KQ_max_new_shared = KQ+lsm_size3;

    __dpct_align__(16) sycl::float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};


#endif // SYCL_FAST_FP16

    float KQ_max[cpw] = {};

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np * warp_size * cpy_ne_D <= DKQ ||
                i0 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D) + item_ct1.get_local_id(2) * cpy_ne_D <
                    DKQ) {
                __dpct_align__(16) float tmp_f[cpy_ne_D] = { 0.0f };
                ggml_sycl_memcpy_1<sizeof(tmp_f)>(
                    tmp_f, &Q_f[c * (nb02 / sizeof(float)) + fastmodulo(col_Q_0 + j, ne01) * (nb01 / sizeof(float)) +
                                i0 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D) +
                                item_ct1.get_local_id(2) * cpy_ne_D]);

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef SYCL_FAST_FP16
                __dpct_align__(16) sycl::half2 tmp_h2[cpy_ne_D / 2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = fattn_tile_make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#if defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1 / 2] *= sycl::half2(0.25f, 0.25f);
#endif // defined(SYCL_FAST_FP16) && !defined(GGML_SYCL_F16)
                }
                ggml_sycl_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[jc * (DKQ / 2) + i0 / 2 + (item_ct1.get_local_id(1) % np) * (warp_size * cpy_ne_D / 2) +
                           item_ct1.get_local_id(2) * (cpy_ne_D / 2)],
                    tmp_h2);
#else
                ggml_sycl_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (item_ct1.get_local_id(1) % np)*(warp_size*cpy_ne_D)   + item_ct1.get_local_id(2)* cpy_ne_D],
                    tmp_f);
#endif // SYCL_FAST_FP16
            }
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    if (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa;
        while (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
            k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = item_ct1.get_group(1) * nbatch_fa; k_VKQ_0 < k_VKQ_max;
             k_VKQ_0 += item_ct1.get_group_range(1) * nbatch_fa) {

            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap,
                                 oob_check>(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp, stride_K2,
                                            stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0,
                                            KQ_max_new_shared);
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef SYCL_FAST_FP16
        sycl::half2 * VKQ_combine = (sycl::half2 *) KV_tmp;
#else
        float * VKQ_combine    = (float *) KV_tmp;
#endif // SYCL_FAST_FP16

        float * KQ_sum_combine = (float *) Q_tmp;

        if (item_ct1.get_local_id(1) % np != 0) {

#ifdef SYCL_FAST_FP16
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                    &VKQ_combine[item_ct1.get_local_id(1) * (DVp / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D],
                    &VKQ[i0 / warp_size]);
            }
#else

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;

#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[item_ct1.get_local_id(1)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
            }
#endif // SYCL_FAST_FP16

            if (item_ct1.get_local_id(2) == 0) {
                KQ_sum_combine[item_ct1.get_local_id(1)] = KQ_sum[0];
            }
            return;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
#ifdef SYCL_FAST_FP16
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __dpct_align__(16) sycl::half2 tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D * 4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip) * (DVp / 2) + i0 +
                                                                   item_ct1.get_local_id(2) * cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __dpct_align__(16) float tmp[cpy_ne_D];
                ggml_sycl_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(item_ct1.get_local_id(1) + ip)*DVp + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // SYCL_FAST_FP16

            KQ_sum[0] += KQ_sum_combine[item_ct1.get_local_id(1) + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && item_ct1.get_group(1) == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int   jc   = jc0 + (item_ct1.get_local_id(1) / np) * cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float       KQ_max_new_j = sycl::fmax((float) KQ_max[jc0], sink);
            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[jc0] - KQ_max_new_j));
            KQ_max[jc0] = KQ_max_new_j;

            const float val = sycl::native::exp((float) (sink - KQ_max[jc0]));
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef SYCL_FAST_FP16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x() *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y() *= KQ_max_scale;
            }
#endif // SYCL_FAST_FP16
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (item_ct1.get_local_id(1) / np) * cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z())) {
            return;
        }

        // A row with no visible cell (S == 0) is written as 0, like the CPU
        // reference, instead of 0 * (1/0) = NaN.
        const float scale = item_ct1.get_group_range(1) == 1 ? (KQ_sum[jc0] == 0.0f ? 0.0f : 1.0f / KQ_sum[jc0]) : 1.0f;

        const int j_dst_unrolled =
            ((sequence * int(ne01.z()) + col_Q_0 + j) * ne02 + head0 + c) * item_ct1.get_group_range(1) +
            item_ct1.get_group(1);

#ifdef SYCL_FAST_FP16
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __dpct_align__(16) sycl::float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[jc0 * ((DVp / 2) / warp_size) + i0 / warp_size + i1]
                              .template convert<float, sycl::rounding_mode::automatic>();
                tmp[i1].x() *= scale;
                tmp[i1].y() *= scale;
            }
            if (i0 + warp_size * cpy_ne_D <= DV / 2 || i0 + item_ct1.get_local_id(2) * cpy_ne_D < DV / 2) {
                ggml_sycl_memcpy_1<sizeof(tmp)>(
                    &dst[j_dst_unrolled * DV + 2 * i0 + item_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + item_ct1.get_local_id(2)*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x() *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y() *= scale;
                }
                ggml_sycl_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + item_ct1.get_local_id(2)*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // SYCL_FAST_FP16

        if (item_ct1.get_group_range(1) != 1 && item_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = fattn_tile_make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN
}

// =============================================================================
// Fork-native D=512 launcher (llama.cpp-dtpk)
// =============================================================================
// NOT part of the launch_fattn_tile_switch_ncols1/ncols2/
// ggml_sycl_flash_attn_ext_tile_case scaffolding below, which depends on a
// generic launch_fattn<D,ncols1,ncols2,KERNEL,warp_size> template that was
// never ported into this fork (see the llama.cpp-dtpk design note). This
// calls flash_attn_tile<> directly and replicates only what gemma's
// DKQ==DV==512 shape needs:
//  - no KV_max/dst_meta split-KV partitioning -- this fork's other kernel
//    families (xmx-v2, esimd, tile-f16) don't use it either; single
//    partition only (grid dim1 == 1, KV_max == nullptr, dst_meta == nullptr,
//    matching how flash_attn_tile itself falls back to k_VKQ_max = ne11 and
//    a direct 1/KQ_sum write when item.get_group_range(1) == 1).
//  - no DKQ != DV (576,512) MLA support.
//  - the GQA-grouping ncols1/ncols2 selection ported as plain runtime logic,
//    reproducing exactly what launch_fattn_tile_switch_ncols2/ncols1's
//    DV==512 branch below would select (traced by hand -- for DV==512 the
//    DV<512 tier in switch_ncols1 is dead and DKQ==DV rules out the
//    576-only tiers, so only the ncols2<=4 / ncols2<=2 / fallback tiers
//    there, plus switch_ncols2's own gqa_limit==INT_MAX-unconditionally
//    branch for DV>256, are reachable -- both are folded in below).
//  - Q is F32 at this call site by construction: llama.cpp-dtpk also
//    patched llama-graph.cpp's build_attn_mha() to skip the blanket
//    GGML_SYCL_FATTN_Q_TYPE(=F16) cast specifically for D=512, since this
//    kernel (unlike every sibling family) has no Q_type template parameter
//    and reads Q as raw F32 unconditionally. Do not call this with F16 Q.
template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, int warp_size>
static void submit_fattn_tile_d512(const fattn_params & params, dpct::queue_ptr stream) {
    static_assert(DKQ == DV, "submit_fattn_tile_d512 only covers the DKQ==DV config rows");
    constexpr int ncols  = ncols1 * ncols2;
    constexpr int nwarps = ggml_sycl_fattn_tile_get_nthreads(DKQ, DV, ncols) / warp_size;
    static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols) != 0, "kernel config not defined");

    // Grid dim0 selects (sequence, head-group); dim1 is the KV-partition
    // index (fixed at 1 -- no split-KV, see above); dim2 selects the
    // Q-column block. Matches flash_attn_tile's own indexing exactly:
    // sequence = get_group(0) / (ne02/ncols2), head0 = get_group(0)*ncols2 - sequence*ne02.
    const int n_query_blocks = (params.ne01 + ncols1 - 1) / ncols1;
    sycl::range<3> block(1, nwarps, warp_size);
    sycl::range<3> grid(params.ne03 * (params.ne02 / ncols2), 1, n_query_blocks);

    const char *   Q_ptr         = params.Q;
    const char *   K_ptr         = params.K;
    const char *   V_ptr         = params.V;
    const char *   mask_ptr      = params.mask;
    const char *   sinks_ptr     = params.sinks;
    float *        dst_ptr       = params.dst;
    const float    scale_v       = params.scale;
    const float    max_bias_v    = params.max_bias;
    const float    m0_v          = params.m0;
    const float    m1_v          = params.m1;
    const uint32_t n_head_log2_v = params.n_head_log2;
    const float    logit_sc_v    = params.logit_softcap;
    const int32_t  ne00_v = params.ne00, ne02_v = params.ne02, ne03_v = params.ne03;
    const sycl::uint3 ne01_fd = init_fastdiv_values((uint32_t) params.ne01);
    const int32_t  nb01_v = params.nb01, nb02_v = params.nb02, nb03_v = params.nb03;
    const int32_t  ne10_v = params.ne10, ne11_v = params.ne11, ne12_v = params.ne12, ne13_v = params.ne13;
    const int32_t  nb11_v = params.nb11;
    const int32_t  nb12_v = params.nb12;
    const int64_t  nb13_v = params.nb13;
    const int32_t  nb21_v = params.nb21, nb22_v = params.nb22;
    const int64_t  nb23_v = params.nb23;
    const int32_t  ne31_v = params.ne31, ne32_v = params.ne32, ne33_v = params.ne33;
    const int32_t  nb31_v = params.nb31, nb32_v = params.nb32;
    const int64_t  nb33_v = params.nb33;

    // Wrapped for llama.cpp-86a7: this is the tile_d512 decode kernel
    // (gemma4's D=512 global-attention layers), previously dark to the
    // kernel profiler entirely -- a full GGML_SYCL_KERNEL_PROFILE census of
    // gemma4 showed zero D=512 rows even though ~7 dispatches/token reach
    // this function, making it impossible to confirm on hardware whether
    // these kernels are actually recorded into and replayed from a SYCL
    // command graph rather than staying eager. Matches the
    // esimd_partitioned/onednn_sdpa_graph wrapping pattern
    // (fattn-esimd-f16.hpp, fattn-onednn.cpp): a distinct label name from
    // onednn_sdpa_graph so the two D=512 routes are never conflated, even
    // though oneDNN's own D=512 attempt already carries D in its metadata.
    ggml_sycl_profile_label profile_label{};
    profile_label.name                 = "fattn.decode.tile_d512";
    profile_label.category             = "fattn";
    profile_label.queue_kind           = "compute";
    const std::string profile_metadata = "D=" + std::to_string(DV) + ";ncols1=" + std::to_string(ncols1) +
                                         ";ncols2=" + std::to_string(ncols2) + ";ne01=" + std::to_string(params.ne01) +
                                         ";ne02=" + std::to_string(ne02_v) + ";ne03=" + std::to_string(ne03_v);
    profile_label.metadata = profile_metadata.c_str();
    profile_label.device   = ggml_sycl_get_device_id_from_queue(*stream);

    (void) ggml_sycl_profile_submit(*stream, profile_label, [&](sycl::queue & profiled_queue) {
        return profiled_queue.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(warp_size)]] {
                    flash_attn_tile<DKQ, DV, ncols1, ncols2, use_logit_softcap, warp_size>(
                        Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, /*KV_max=*/nullptr, dst_ptr,
                        /*dst_meta=*/nullptr, scale_v, max_bias_v, m0_v, m1_v, n_head_log2_v, logit_sc_v, ne00_v,
                        ne01_fd, ne02_v, ne03_v, nb01_v, nb02_v, nb03_v, ne10_v, ne11_v, ne12_v, ne13_v, nb11_v, nb12_v,
                        nb13_v, nb21_v, nb22_v, nb23_v, ne31_v, ne32_v, ne33_v, nb31_v, nb32_v, nb33_v);
                });
        });
    });
}

// Entry point: selects (ncols1, ncols2) at runtime -- see the derivation in
// the block comment above.
template <bool use_logit_softcap>
static void launch_fattn_tile_d512(const fattn_params & params, dpct::queue_ptr stream) {
    constexpr int warp_size = WARP_32_SIZE;  // can't support WARP_16_SIZE, matches switch_ncols1's own comment
    constexpr int DKQ = 512;
    constexpr int DV  = 512;

    GGML_ASSERT(params.ne02 % params.ne12 == 0);
    const int gqa_ratio = params.ne02 / params.ne12;
    const int ne01      = params.ne01;
    // DV==512 forces launch_fattn_tile_switch_ncols2's own
    // `gqa_ratio<=4 && DV<=256 ? 16 : INT_MAX` ternary to INT_MAX
    // unconditionally (the DV<=256 half is always false here) -- so unlike
    // the D<=256 kernels, ne01 never gates GQA-grouping eligibility at D=512.
    const bool use_gqa_opt =
        params.mask != nullptr && params.max_bias == 0.0f && (params.ne11 % FATTN_KQ_STRIDE == 0);

    if (use_gqa_opt && gqa_ratio % 16 == 0) {
        submit_fattn_tile_d512<DKQ, DV, 2, 16, use_logit_softcap, warp_size>(params, stream);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        submit_fattn_tile_d512<DKQ, DV, 1, 4, use_logit_softcap, warp_size>(params, stream);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        if (ne01 >= 2) {
            submit_fattn_tile_d512<DKQ, DV, 2, 2, use_logit_softcap, warp_size>(params, stream);
        } else {
            submit_fattn_tile_d512<DKQ, DV, 1, 2, use_logit_softcap, warp_size>(params, stream);
        }
        return;
    }
    // ncols2 == 1 fallback -- always reachable when the above don't apply,
    // matching launch_fattn_tile_switch_ncols2's unconditional final branch
    // for DKQ==DV.
    if (ne01 >= 3) {
        submit_fattn_tile_d512<DKQ, DV, 4, 1, use_logit_softcap, warp_size>(params, stream);
    } else {
        submit_fattn_tile_d512<DKQ, DV, 2, 1, use_logit_softcap, warp_size>(params, stream);
    }
}

template <int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    const int id        = ggml_sycl_get_device();
    const int cc        = ggml_sycl_info().devices[id].cc;
    const int warp_size = WARP_32_SIZE; //can't support WARP_16_SIZE

    constexpr size_t nbytes_shared = 0;

    if (DV < 512 && Q->ne[1] < 32) {
        if constexpr (ncols2 <= 32) {
            if (Q->ne[1] > 16/ncols2) {
                constexpr int cols_per_block = 32;
                const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                return;
            }
        }
        if constexpr (ncols2 <= 16) {
            if (Q->ne[1] > 8/ncols2) {
                constexpr int cols_per_block = 16;
                const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                return;
            }
        }
        if constexpr (ncols2 <= 8) {
            if (Q->ne[1] > 4/ncols2) {
                constexpr int cols_per_block = 8;
                const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
                const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
                launch_fattn<DV, cols_per_block/ncols2, ncols2,
                    flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                    (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
                return;
            }
        }
    }

    if constexpr (ncols2 <= 4) {
        if (Q->ne[1] > 2/ncols2) {
            constexpr int cols_per_block = 4;
            const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            launch_fattn<DV, cols_per_block/ncols2, ncols2,
                flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
                (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
            return;
        }
    }

    if constexpr (ncols2 <= 2) {
        constexpr int cols_per_block = 2;
        const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
        const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
        return;
    }

    {
        constexpr int cols_per_block = ncols2*2;
        const int nwarps    = ggml_sycl_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
        const int nbatch_fa = ggml_sycl_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        launch_fattn<DV, cols_per_block/ncols2, ncols2,
            flash_attn_tile<DKQ, DV, cols_per_block / ncols2, ncols2, use_logit_softcap, warp_size>, warp_size>
            (ctx, dst, nwarps, nbytes_shared, nbatch_fa, true, true, false);
        return;
    }

    GGML_ABORT("fatal error");
}

template <int DKQ, int DV, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    //const bool nvidia = GGML_SYCL_CC_IS_NVIDIA(ggml_sycl_info().devices[ggml_sycl_get_device()].cc);
    const int gqa_limit = gqa_ratio <= 4 && DV <= 256 ? 16 : INT_MAX;
    const bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    if constexpr (DV == 512) {
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            return;
        }
        // ncols2=2 and ncols2=1 fallbacks only for cases where ncols=2 config exists (DKQ == DV).
        // For DKQ == 576, DV == 512 only GQA-optimized variants are implemented.
        if constexpr (DKQ == DV) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
                return;
            }
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            return;
        }
    }

    if constexpr (DV <= 256) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            return;
        }

        launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
        return;
    }
    GGML_ABORT("fatal error");
}

template <int DKQ, int DV>
void ggml_sycl_flash_attn_ext_tile_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template void ggml_sycl_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_TILE_CASE( 40,  40);
extern DECL_FATTN_TILE_CASE( 64,  64);
extern DECL_FATTN_TILE_CASE( 72,  72);
extern DECL_FATTN_TILE_CASE( 80,  80);
extern DECL_FATTN_TILE_CASE( 96,  96);
extern DECL_FATTN_TILE_CASE(112, 112);
extern DECL_FATTN_TILE_CASE(128, 128);
extern DECL_FATTN_TILE_CASE(256, 256);
extern DECL_FATTN_TILE_CASE(512, 512);
extern DECL_FATTN_TILE_CASE(576, 512);

// Scope SYCL_FLASH_ATTN/SYCL_FAST_FP16 to this header (spec review
// rev-dtpk-spec2, N1): fattn.cpp now includes this file in addition to
// fattn-tile.cpp, and these macros have no reason to stay defined for the
// several thousand lines of unrelated code that follow this include in
// that translation unit -- undef them here rather than let them leak,
// completing the hygiene pair with the file-scoped helper renames above.
#undef SYCL_FLASH_ATTN
#undef SYCL_FAST_FP16

#endif // GGML_SYCL_FATTN_TILE_HPP
