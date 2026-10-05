#ifndef GGML_SYCL_LIGHTNING_INDEXER_KERNEL_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_KERNEL_HPP

// GGML_OP_LIGHTNING_INDEXER kernel.
//
// Header-only and free of every backend header on purpose, like dsv4-hc-kernels.hpp: a launcher takes
// a plain sycl::queue and raw pointers, so tests/test-sycl-dsv4-hc-kernels can compile the SAME kernel
// and run it on the opencl:cpu device against an independent oracle. lightning-indexer.cpp is the only
// production caller.
//
// Semantics of the CPU reference (ggml-cpu/ops.cpp ggml_compute_forward_lightning_indexer):
//   dst[ik, t, 0, s] = mask[ik, t, 0, s % nem3] + sum_h relu(q[:, h, t, s] . k[:, 0, ik, s]) * w[h, t, 0, s]
//
// One sub-group of LANES work-items owns one (token, stream, kv) row: each lane holds n_embd / LANES
// consecutive elements of K, so the dot product is a sub-group reduction per head. K is dequantized in
// FLOAT here, never through the backend's dfloat (sycl::half under GGML_SYCL_F16), which would put 3
// mantissa bits of half-precision error into every score of a quantized K cache.

#include "ggml.h"
#include "lightning-indexer-predicate.hpp"

#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>

#ifndef GGML_COMMON_DECL
#    define GGML_COMMON_DECL_SYCL
#    define GGML_COMMON_IMPL_SYCL
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnested-anon-types"
#    include "ggml-common.h"
#    pragma clang diagnostic pop
#endif

namespace ggml_sycl_lightning_indexer {

// What the kernel's casts of a K row read, per K type: one float, half or bfloat16 element, or one quantized block.
// A type added to GGML_SYCL_LIGHTNING_INDEXER_K_TYPES with no specialization here fails to compile (incomplete type)
// in the static_assert below, so no K type can be admitted with the default alignment.
template <ggml_type KT> struct lightning_indexer_k_storage;

// clang-format off
template <> struct lightning_indexer_k_storage<GGML_TYPE_F32>    { using type = float; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_F16>    { using type = sycl::half; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_BF16>   { using type = sycl::ext::oneapi::bfloat16; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_Q8_0>   { using type = block_q8_0; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_Q4_0>   { using type = block_q4_0; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_Q4_1>   { using type = block_q4_1; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_Q5_0>   { using type = block_q5_0; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_Q5_1>   { using type = block_q5_1; };
template <> struct lightning_indexer_k_storage<GGML_TYPE_IQ4_NL> { using type = block_iq4_nl; };
// clang-format on

// k_align() (the predicate's stride check) states the alignment this kernel's casts of a K row need. The predicate
// header is pure ggml and cannot see the block types, so the kernel -- which can -- pins the two together for every
// type in the table: a block layout change breaks the build here instead of silently letting the predicate admit a
// misaligned view.
#define X(T)                                                                            \
    static_assert(k_align(T) == alignof(typename lightning_indexer_k_storage<T>::type), \
                  "k_align disagrees with the alignment of the K storage of " #T);
GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X)
#undef X

constexpr int64_t LIGHTNING_INDEXER_ROWS_PER_BLOCK = 4;

// Element `i` of one K row, as float. All the quantized types here use 32-element blocks. The type is a template
// parameter so the launcher picks the instantiation once per launch and the kernel body carries no per-element
// switch on it.
template <ggml_type KT> inline float lightning_indexer_k_elem(const char * row, int64_t i) {
    if constexpr (KT == GGML_TYPE_F32) {
        return ((const float *) row)[i];
    } else if constexpr (KT == GGML_TYPE_F16) {
        return (float) ((const sycl::half *) row)[i];
    } else if constexpr (KT == GGML_TYPE_BF16) {
        return (float) ((const sycl::ext::oneapi::bfloat16 *) row)[i];
    } else if constexpr (KT == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row + i / 32;
        return (float) b->d * (float) b->qs[i % 32];
    } else if constexpr (KT == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) row + i / 32;
        const int          j = (int) (i % 32);
        const int          q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
        return (float) b->d * (float) (q - 8);
    } else if constexpr (KT == GGML_TYPE_Q4_1) {
        const block_q4_1 * b = (const block_q4_1 *) row + i / 32;
        const int          j = (int) (i % 32);
        const int          q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
        return (float) b->dm[0] * (float) q + (float) b->dm[1];
    } else if constexpr (KT == GGML_TYPE_Q5_0) {
        const block_q5_0 * b = (const block_q5_0 *) row + i / 32;
        const int          j = (int) (i % 32);
        uint32_t           qh;
        std::memcpy(&qh, b->qh, sizeof(qh));
        const int q = j < 16 ? ((b->qs[j] & 0xF) | (int) (((qh >> j) << 4) & 0x10)) :
                               ((b->qs[j - 16] >> 4) | (int) ((qh >> (j - 16 + 12)) & 0x10));
        return (float) b->d * (float) (q - 16);
    } else if constexpr (KT == GGML_TYPE_Q5_1) {
        const block_q5_1 * b = (const block_q5_1 *) row + i / 32;
        const int          j = (int) (i % 32);
        uint32_t           qh;
        std::memcpy(&qh, b->qh, sizeof(qh));
        const int q = j < 16 ? ((b->qs[j] & 0xF) | (int) (((qh >> j) << 4) & 0x10)) :
                               ((b->qs[j - 16] >> 4) | (int) ((qh >> (j - 16 + 12)) & 0x10));
        return (float) b->dm[0] * (float) q + (float) b->dm[1];
    } else if constexpr (KT == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * b = (const block_iq4_nl *) row + i / 32;
        const int            j = (int) (i % 32);
        const int            q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
        return (float) b->d * (float) kvalues_iq4nl[q];
    } else {
        static_assert(KT == GGML_TYPE_F32,
                      "lightning_indexer_k_elem: a K type in GGML_SYCL_LIGHTNING_INDEXER_K_TYPES has no dequant");
        return 0.0f;
    }
}

struct lightning_indexer_args {
    const char * q;    // F32 [n_embd, n_head, n_batch, n_stream]
    const char * k;    // k_type [n_embd, 1, n_kv, n_stream]
    const char * w;    // F32 [n_head, n_batch, 1, n_stream]
    const char * m;    // F16 [n_kv, n_batch, 1, nem3]
    float *      dst;  // F32 [n_kv, n_batch, 1, n_stream]
    ggml_type    k_type;
    int64_t      n_embd;
    int64_t      n_head;
    int64_t      n_batch;
    int64_t      n_stream;
    int64_t      n_kv;
    int64_t      nem3;
    // byte strides; dst strides are bytes too and nb0 == sizeof(float) is a precondition of the op
    size_t       nbq1, nbq2, nbq3;
    size_t       nbk2, nbk3;
    size_t       nbw1, nbw3;
    size_t       nbm1, nbm3;
    size_t       nb1, nb3;
};

// The launch grid is two-dimensional on purpose. One sub-group owns one (token, stream, kv) row, so the total
// work-item count is n_batch * n_stream * n_kv * LANES, and a 1D range of that size overflows the 32-bit id range
// the compiler assumes for work-item ids once n_batch * n_stream * n_kv reaches 2^31 / LANES, e.g. ub 512 at
// n_kv 131072 or ub 2048 at n_kv 32768. A grid of groups_y x groups_x work-groups keeps every dimension far
// below that for any shape the op can have, so no shape is declined to the CPU.
constexpr int64_t LIGHTNING_INDEXER_MAX_GROUPS_X = int64_t(1) << 20;

struct lightning_indexer_dims {
    int64_t groups_x;
    int64_t groups_y;
};

// The smallest grid of at most `max_groups_x` columns (0: LIGHTNING_INDEXER_MAX_GROUPS_X) holding every work-group
// the rows need.
inline lightning_indexer_dims lightning_indexer_launch_dims(int64_t n_rows, int64_t max_groups_x) {
    const int64_t n_blocks = (n_rows + LIGHTNING_INDEXER_ROWS_PER_BLOCK - 1) / LIGHTNING_INDEXER_ROWS_PER_BLOCK;
    const int64_t cap      = max_groups_x > 0 ? max_groups_x : LIGHTNING_INDEXER_MAX_GROUPS_X;
    if (n_blocks <= 0) {
        return { 1, 0 };
    }
    const int64_t gx = n_blocks < cap ? n_blocks : cap;
    return { gx, (n_blocks + gx - 1) / gx };
}

// Cost note: every sub-group re-reads the whole Q tile (n_head * n_embd floats) for its (token, kv) row, with no
// tiling across kv, so prefill at long n_kv is bound by those Q re-reads rather than by the K dequant.
template <int LANES, int EPL, ggml_type KT>
inline void lightning_indexer_launch_impl(sycl::queue & queue, const lightning_indexer_args & a, int64_t max_groups_x) {
    constexpr int64_t BLOCK_SIZE = LIGHTNING_INDEXER_ROWS_PER_BLOCK * LANES;

    const int64_t n_rows = a.n_batch * a.n_stream * a.n_kv;
    const auto    dims   = lightning_indexer_launch_dims(n_rows, max_groups_x);
    if (dims.groups_y == 0) {
        return;
    }
    const int64_t groups_x = dims.groups_x;

    queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(dims.groups_y, groups_x * BLOCK_SIZE), sycl::range<2>(1, BLOCK_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(LANES)]] {
            const int64_t lid   = item.get_local_id(1);
            const int64_t lane  = lid % LANES;
            const int64_t block = (int64_t) item.get_group(0) * groups_x + (int64_t) item.get_group(1);
            const int64_t row   = block * LIGHTNING_INDEXER_ROWS_PER_BLOCK + lid / LANES;
            // `row` is uniform across the sub-group (a sub-group never spans two rows), so this return cannot
            // split the reduction below
            if (row >= n_rows) {
                return;
            }

            const int64_t i_bs     = row / a.n_kv;
            const int64_t i_kv     = row % a.n_kv;
            const int64_t i_batch  = i_bs / a.n_stream;
            const int64_t i_stream = i_bs % a.n_stream;

            const char * k_base = a.k + i_kv * a.nbk2 + i_stream * a.nbk3;
            float        k_local[EPL];
#pragma unroll
            for (int j = 0; j < EPL; ++j) {
                k_local[j] = lightning_indexer_k_elem<KT>(k_base, lane * EPL + j);
            }

            const char *  q_base = a.q + i_batch * a.nbq2 + i_stream * a.nbq3;
            const float * w_base = (const float *) (a.w + i_batch * a.nbw1 + i_stream * a.nbw3);

            float score = 0.0f;
            for (int64_t h = 0; h < a.n_head; ++h) {
                const float * q_row = (const float *) (q_base + h * a.nbq1);
                float         dot   = 0.0f;
#pragma unroll
                for (int j = 0; j < EPL; ++j) {
                    dot += q_row[lane * EPL + j] * k_local[j];
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<float>());
                if (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            if (lane == 0) {
                const sycl::half * m_base =
                    (const sycl::half *) (a.m + i_batch * a.nbm1 + (i_stream % a.nem3) * a.nbm3);
                // flat-index store: storing through a strided base pointer hangs/misroutes writes on this
                // stack when n_batch * n_stream > 1
                const int64_t dst_idx = i_kv + i_batch * (a.nb1 / sizeof(float)) + i_stream * (a.nb3 / sizeof(float));
                a.dst[dst_idx]        = score + (float) m_base[i_kv];
            }
        });
}

template <int LANES, ggml_type KT>
inline bool lightning_indexer_launch_epl(sycl::queue & queue, const lightning_indexer_args & a, int64_t max_groups_x) {
    switch (a.n_embd / LANES) {
#define X(E)                                                                 \
    case E:                                                                  \
        lightning_indexer_launch_impl<LANES, E, KT>(queue, a, max_groups_x); \
        return true;
        GGML_SYCL_LIGHTNING_INDEXER_EPLS(X)
#undef X
        default:
            return false;
    }
}

// Submits one kernel and returns true, or returns false and submits nothing when n_embd is not LANES * an
// instantiated elements-per-lane count or K's type is not one the kernel reads. `max_groups_x` caps the launch
// grid's width (0: LIGHTNING_INDEXER_MAX_GROUPS_X, which is what production uses); tests pass a small value to force
// a tall grid.
template <int LANES>
inline bool lightning_indexer_launch(sycl::queue & queue, const lightning_indexer_args & a, int64_t max_groups_x = 0) {
    if (a.n_embd % LANES != 0) {
        return false;
    }
    switch (a.k_type) {
#define X(T) \
    case T:  \
        return lightning_indexer_launch_epl<LANES, T>(queue, a, max_groups_x);
        GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X)
#undef X
        default:
            return false;
    }
}

}  // namespace ggml_sycl_lightning_indexer

#endif  // GGML_SYCL_LIGHTNING_INDEXER_KERNEL_HPP
