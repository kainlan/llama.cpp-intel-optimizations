#pragma once

// llama.cpp-9qjy: native BF16-weight x F32-activation MUL_MAT.
//
// The weight is consumed in the layout the planner and the unified cache
// already materialized it in -- raw BF16, row-major, K contiguous -- so there
// is no second (F32) copy: the planner's byte count for a BF16 weight
// (ggml_nbytes) is exactly what lives on the device, and each token reads half
// the bytes the F32 route read.
//
// Precision: BF16 -> F32 is an exact 16-bit shift (the BF16 bits are the top
// half of the F32 bits), and accumulation is F32 FMA. BF16 -> F16 would NOT be
// exact (F16 has a 5-bit exponent against BF16's 8), so no F16 intermediate
// exists anywhere on this path.
//
// Header-only and free of backend state on purpose: tests/test-sycl-bf16-mul-mat.cpp
// compiles it against any SYCL device (opencl:cpu on a host without a GPU) and
// checks the numbers against a double-precision reference, so the kernel the
// backend runs is the kernel the host test checks.

#include "ggml.h"

#include <cstdint>
#include <sycl/sycl.hpp>

namespace ggml_sycl_bf16 {

// Columns handled by the one-pass skinny kernel (decode, speculative batches).
// Above this the tiled kernel takes over so the weight is read once per
// 32-column tile instead of once per skinny pass.
constexpr int64_t SKINNY_MAX_COLS = 8;

// True when the n_cols = ne1*ne2*ne3 columns of an F32 tensor sit at one uniform
// stride, i.e. the tensor is indexable as t[col * ld + row]. ld is in elements.
inline bool f32_columns_uniform(const ggml_tensor * t, int64_t * ld) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t n_cols = t->ne[1] * t->ne[2] * t->ne[3];
    if (n_cols <= 0 || t->ne[0] <= 0) {
        return false;
    }
    if (n_cols == 1) {
        *ld = t->ne[0];
        return true;
    }
    if (t->nb[1] % sizeof(float) != 0 || t->nb[1] < t->nb[0] * static_cast<size_t>(t->ne[0])) {
        return false;
    }
    if (t->ne[2] > 1 && t->nb[2] != t->nb[1] * static_cast<size_t>(t->ne[1])) {
        return false;
    }
    if (t->ne[3] > 1 && t->nb[3] != t->nb[1] * static_cast<size_t>(t->ne[1] * t->ne[2])) {
        return false;
    }
    *ld = static_cast<int64_t>(t->nb[1] / sizeof(float));
    return true;
}

// The shapes the native executor computes. Pure: no allocation, no device.
// ggml_backend_sycl_device_supports_op() and the dispatcher both consult it
// (through ggml_sycl_bf16_weight_native_route_available), so admission and
// execution cannot drift apart.
//
//   src0  BF16, one contiguous 2-D matrix [K x M]  (a dense weight)
//   src1  F32, columns at one uniform stride        (the activations)
//   dst   F32, columns at one uniform stride        (ne0 == M)
inline bool mul_mat_shape_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_BF16 || !ggml_is_contiguous(src0) || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1]) {
        return false;
    }
    if (src1->ne[1] != dst->ne[1] || src1->ne[2] != dst->ne[2] || src1->ne[3] != dst->ne[3]) {
        return false;
    }
    int64_t ldx = 0;
    int64_t ldy = 0;
    return f32_columns_uniform(src1, &ldx) && f32_columns_uniform(dst, &ldy);
}

inline float bf16_to_f32(uint16_t bits) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

// Sub-groups per output row for the skinny kernel. Decode runs hundreds of these ops per
// token on one in-order queue, and the Qwen3.8 BF16 matrices are short and wide (M = 1, 4,
// 48, 320 against K up to 10240): one sub-group per row would leave a handful of sub-groups
// walking K serially, so the op would be latency-bound rather than bandwidth-bound. Split K
// across a power-of-two number of sub-groups of one work-group, growing it until the op has
// about SKINNY_TARGET_SUBGROUPS sub-groups, but never so far that a sub-group's share of K
// drops below one 128-element pass. The partial sums meet in local memory in a fixed order
// (no atomics), so the result does not depend on scheduling.
constexpr int64_t SKINNY_TARGET_SUBGROUPS = 2048;
constexpr int     SKINNY_MAX_SPLIT        = 32;  // sub-groups per row; 32 x 16 lanes = a 512-thread work-group
constexpr int     SKINNY_PASS_ELEMS       = 128;

inline int skinny_split(int64_t K, int64_t M) {
    int64_t cap = (K + SKINNY_PASS_ELEMS - 1) / SKINNY_PASS_ELEMS;
    if (cap > SKINNY_MAX_SPLIT) {
        cap = SKINNY_MAX_SPLIT;
    }
    int split = 1;
    while (split * 2 <= cap && M * split < SKINNY_TARGET_SUBGROUPS) {
        split *= 2;
    }
    return split;
}

// Decode shape (N <= 8 columns), bandwidth-bound at 2 bytes per weight element. Each row is
// owned by `split` sub-groups of one work-group; the N activation columns accumulate in
// registers so a weight element is read exactly once. With vec4 the lanes load 4 BF16 (8
// bytes) per step and keep 4 loads in flight; with xvec the activations load as float4.
template <int N>
inline sycl::event launch_skinny(sycl::queue &    q,
                                 const uint16_t * w,
                                 const float *    x,
                                 float *          y,
                                 int64_t          K,
                                 int64_t          M,
                                 int64_t          ldx,
                                 int64_t          ldy,
                                 bool             vec4,
                                 bool             xvec,
                                 int              split) {
    constexpr int        SG          = 16;
    constexpr int        UNROLL      = 4;
    // At least 8 sub-groups per work-group so short splits still fill a work-group with rows.
    const int            rows_per_wg = split >= 8 ? 1 : 8 / split;
    const int            sg_per_wg   = rows_per_wg * split;
    const int64_t        n_wg        = (M + rows_per_wg - 1) / rows_per_wg;
    const sycl::range<1> global(static_cast<size_t>(n_wg * sg_per_wg * SG));
    const sycl::range<1> local(static_cast<size_t>(sg_per_wg * SG));

    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(static_cast<size_t>(sg_per_wg * N)), h);

        h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            sycl::sub_group sg    = it.get_sub_group();
            const int       sg_id = static_cast<int>(sg.get_group_linear_id());  // within the work-group
            const int       r     = sg_id / split;                               // row within the work-group
            const int       s     = sg_id % split;                               // K slice within the row
            const int       lane  = static_cast<int>(sg.get_local_linear_id());
            const int64_t   row   = static_cast<int64_t>(it.get_group(0)) * rows_per_wg + r;

            float acc[N];
            for (int n = 0; n < N; ++n) {
                acc[n] = 0.0f;
            }

            // Every sub-group reaches the barrier below; only the accumulation is guarded.
            if (row < M) {
                const uint16_t * wr = w + row * K;
                if (vec4) {
                    const int64_t k4_count = K / 4;
                    const int64_t stride   = static_cast<int64_t>(split) * SG;  // in 4-element groups
                    const auto *  wr4      = reinterpret_cast<const uint64_t *>(wr);
                    for (int64_t k4 = s * SG + lane; k4 < k4_count; k4 += stride * UNROLL) {
                        uint64_t packed[UNROLL];
                        for (int u = 0; u < UNROLL; ++u) {
                            const int64_t idx = k4 + u * stride;
                            packed[u]         = idx < k4_count ? wr4[idx] : 0;
                        }
                        for (int u = 0; u < UNROLL; ++u) {
                            const int64_t idx = k4 + u * stride;
                            if (idx >= k4_count) {
                                break;
                            }
                            const float   w0 = bf16_to_f32(static_cast<uint16_t>(packed[u]));
                            const float   w1 = bf16_to_f32(static_cast<uint16_t>(packed[u] >> 16));
                            const float   w2 = bf16_to_f32(static_cast<uint16_t>(packed[u] >> 32));
                            const float   w3 = bf16_to_f32(static_cast<uint16_t>(packed[u] >> 48));
                            const int64_t k  = idx * 4;
                            for (int n = 0; n < N; ++n) {
                                const float * xn = x + n * ldx + k;
                                float         x0, x1, x2, x3;
                                if (xvec) {
                                    const sycl::float4 xv = *reinterpret_cast<const sycl::float4 *>(xn);
                                    x0                    = xv.x();
                                    x1                    = xv.y();
                                    x2                    = xv.z();
                                    x3                    = xv.w();
                                } else {
                                    x0 = xn[0];
                                    x1 = xn[1];
                                    x2 = xn[2];
                                    x3 = xn[3];
                                }
                                acc[n] = sycl::fma(w0, x0, acc[n]);
                                acc[n] = sycl::fma(w1, x1, acc[n]);
                                acc[n] = sycl::fma(w2, x2, acc[n]);
                                acc[n] = sycl::fma(w3, x3, acc[n]);
                            }
                        }
                    }
                } else {
                    const int64_t stride = static_cast<int64_t>(split) * SG;
                    for (int64_t k = s * SG + lane; k < K; k += stride) {
                        const float wk = bf16_to_f32(wr[k]);
                        for (int n = 0; n < N; ++n) {
                            acc[n] = sycl::fma(wk, x[n * ldx + k], acc[n]);
                        }
                    }
                }
            }

            // Stage 1: reduce the 16 lanes of this sub-group, park the partial sum in local memory.
            for (int n = 0; n < N; ++n) {
                const float sum = sycl::reduce_over_group(sg, acc[n], sycl::plus<float>());
                if (lane == 0) {
                    partial[sg_id * N + n] = sum;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            // Stage 2: one work-item per (row, column) adds the `split` partials in a fixed order.
            const int t = static_cast<int>(it.get_local_linear_id());
            if (t < rows_per_wg * N) {
                const int     rr   = t / N;
                const int     n    = t % N;
                const int64_t row2 = static_cast<int64_t>(it.get_group(0)) * rows_per_wg + rr;
                if (row2 < M) {
                    float total = 0.0f;
                    for (int s2 = 0; s2 < split; ++s2) {
                        total += partial[(rr * split + s2) * N + n];
                    }
                    y[n * ldy + row2] = total;
                }
            }
        });
    });
}

// 32 x 32 output tile per 16 x 16 work-group, 2 x 2 outputs per work-item, K staged
// through local memory in steps of 32. Prompt-processing shape (N > 8); it does not
// use XMX, which is the follow-up if these projections ever show up in a profile.
inline sycl::event launch_tiled(sycl::queue &    q,
                                const uint16_t * w,
                                const float *    x,
                                float *          y,
                                int64_t          K,
                                int64_t          M,
                                int64_t          N,
                                int64_t          ldx,
                                int64_t          ldy) {
    constexpr int TILE = 32;
    constexpr int WG   = 16;
    constexpr int PAD  = TILE + 1;  // odd row stride: conflict-free column reads of the local tiles

    const int64_t        n_tiles_m = (M + TILE - 1) / TILE;
    const int64_t        n_tiles_n = (N + TILE - 1) / TILE;
    // dim 1 is the fast one and maps to output rows, which are contiguous in y.
    const sycl::range<2> global(static_cast<size_t>(n_tiles_n * WG), static_cast<size_t>(n_tiles_m * WG));
    const sycl::range<2> local(WG, WG);

    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> w_tile(sycl::range<1>(TILE * PAD), h);
        sycl::local_accessor<float, 1> x_tile(sycl::range<1>(TILE * PAD), h);

        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            const int     tx  = static_cast<int>(it.get_local_id(0));  // column within the tile
            const int     ty  = static_cast<int>(it.get_local_id(1));  // row within the tile
            const int64_t m0  = static_cast<int64_t>(it.get_group(1)) * TILE;
            const int64_t n0  = static_cast<int64_t>(it.get_group(0)) * TILE;
            const int     lid = ty + tx * WG;

            float acc[2][2] = {
                { 0.0f, 0.0f },
                { 0.0f, 0.0f }
            };

            for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                // 1024 elements per tile over 256 work-items: 4 each, consecutive lanes along k.
                for (int i = 0; i < (TILE * TILE) / (WG * WG); ++i) {
                    const int     e     = lid + i * (WG * WG);
                    const int     r     = e / TILE;
                    const int     c     = e % TILE;
                    const int64_t k     = k0 + c;
                    const int64_t mr    = m0 + r;
                    const int64_t nc    = n0 + r;
                    w_tile[r * PAD + c] = (mr < M && k < K) ? bf16_to_f32(w[mr * K + k]) : 0.0f;
                    x_tile[r * PAD + c] = (nc < N && k < K) ? x[nc * ldx + k] : 0.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int kk = 0; kk < TILE; ++kk) {
                    const float a0 = w_tile[ty * PAD + kk];
                    const float a1 = w_tile[(ty + WG) * PAD + kk];
                    const float b0 = x_tile[tx * PAD + kk];
                    const float b1 = x_tile[(tx + WG) * PAD + kk];
                    acc[0][0]      = sycl::fma(a0, b0, acc[0][0]);
                    acc[0][1]      = sycl::fma(a0, b1, acc[0][1]);
                    acc[1][0]      = sycl::fma(a1, b0, acc[1][0]);
                    acc[1][1]      = sycl::fma(a1, b1, acc[1][1]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }

            for (int j = 0; j < 2; ++j) {
                const int64_t n = n0 + tx + j * WG;
                for (int i = 0; i < 2; ++i) {
                    const int64_t m = m0 + ty + i * WG;
                    if (m < M && n < N) {
                        y[n * ldy + m] = acc[i][j];
                    }
                }
            }
        });
    });
}

// y[col * ldy + m] = sum_k bf16(w[m * K + k]) * x[col * ldx + k], for m < M, col < N.
// w is device memory holding M rows of K packed BF16 values. Asynchronous: the
// returned event completes when y is written; nothing here waits on the host.
// Ordering against the producers of w and x is the caller's: graph_compute already chains the
// source ready events, so this takes no event of its own.
inline sycl::event mul_mat_bf16_f32(sycl::queue &    q,
                                    const uint16_t * w,
                                    const float *    x,
                                    float *          y,
                                    int64_t          K,
                                    int64_t          M,
                                    int64_t          N,
                                    int64_t          ldx,
                                    int64_t          ldy) {
    if (K <= 0 || M <= 0 || N <= 0) {
        return sycl::event();
    }
    if (N > SKINNY_MAX_COLS) {
        return launch_tiled(q, w, x, y, K, M, N, ldx, ldy);
    }
    // The 8-byte weight loads need the row start 8-byte aligned: base aligned and K % 4 == 0.
    const bool vec4  = (K % 4 == 0) && (reinterpret_cast<uintptr_t>(w) % 8 == 0);
    // float4 activation loads additionally need 16-byte aligned column starts.
    const bool xvec  = vec4 && (reinterpret_cast<uintptr_t>(x) % 16 == 0) && (ldx % 4 == 0);
    const int  split = skinny_split(K, M);
    switch (N) {
        case 1:
            return launch_skinny<1>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 2:
            return launch_skinny<2>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 3:
            return launch_skinny<3>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 4:
            return launch_skinny<4>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 5:
            return launch_skinny<5>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 6:
            return launch_skinny<6>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        case 7:
            return launch_skinny<7>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
        default:
            return launch_skinny<8>(q, w, x, y, K, M, ldx, ldy, vec4, xvec, split);
    }
}

}  // namespace ggml_sycl_bf16
