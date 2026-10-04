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

namespace ggml_sycl_dsv4 {

constexpr int64_t LID_ROWS_PER_BLOCK = 4;

// K row types the kernel can read, and the one place that says so (the supports_op predicate asks it).
inline bool lightning_indexer_k_type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// Elements per lane the kernel is instantiated for. n_embd must equal LANES * EPL for one of these.
inline bool lightning_indexer_epl_supported(int64_t epl) {
    return epl == 2 || epl == 4 || epl == 8 || epl == 16;
}

// Element `i` of one K row, as float. All the quantized types here use 32-element blocks.
inline float lightning_indexer_k_elem(const char * row, ggml_type type, int64_t i) {
    switch (type) {
        case GGML_TYPE_F32:
            return ((const float *) row)[i];
        case GGML_TYPE_F16:
            return (float) ((const sycl::half *) row)[i];
        case GGML_TYPE_BF16:
            return (float) ((const sycl::ext::oneapi::bfloat16 *) row)[i];
        case GGML_TYPE_Q8_0:
            {
                const block_q8_0 * b = (const block_q8_0 *) row + i / 32;
                return (float) b->d * (float) b->qs[i % 32];
            }
        case GGML_TYPE_Q4_0:
            {
                const block_q4_0 * b = (const block_q4_0 *) row + i / 32;
                const int          j = (int) (i % 32);
                const int          q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
                return (float) b->d * (float) (q - 8);
            }
        case GGML_TYPE_Q4_1:
            {
                const block_q4_1 * b = (const block_q4_1 *) row + i / 32;
                const int          j = (int) (i % 32);
                const int          q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
                return (float) b->dm[0] * (float) q + (float) b->dm[1];
            }
        case GGML_TYPE_Q5_0:
            {
                const block_q5_0 * b = (const block_q5_0 *) row + i / 32;
                const int          j = (int) (i % 32);
                uint32_t           qh;
                std::memcpy(&qh, b->qh, sizeof(qh));
                const int q = j < 16 ? ((b->qs[j] & 0xF) | (int) (((qh >> j) << 4) & 0x10)) :
                                       ((b->qs[j - 16] >> 4) | (int) ((qh >> (j - 16 + 12)) & 0x10));
                return (float) b->d * (float) (q - 16);
            }
        case GGML_TYPE_Q5_1:
            {
                const block_q5_1 * b = (const block_q5_1 *) row + i / 32;
                const int          j = (int) (i % 32);
                uint32_t           qh;
                std::memcpy(&qh, b->qh, sizeof(qh));
                const int q = j < 16 ? ((b->qs[j] & 0xF) | (int) (((qh >> j) << 4) & 0x10)) :
                                       ((b->qs[j - 16] >> 4) | (int) ((qh >> (j - 16 + 12)) & 0x10));
                return (float) b->dm[0] * (float) q + (float) b->dm[1];
            }
        case GGML_TYPE_IQ4_NL:
            {
                const block_iq4_nl * b = (const block_iq4_nl *) row + i / 32;
                const int            j = (int) (i % 32);
                const int            q = j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4);
                return (float) b->d * (float) kvalues_iq4nl[q];
            }
        default:
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

template <int LANES, int EPL>
inline void lightning_indexer_launch_impl(sycl::queue & queue, const lightning_indexer_args & a) {
    constexpr int64_t BLOCK_SIZE = LID_ROWS_PER_BLOCK * LANES;

    const int64_t n_rows   = a.n_batch * a.n_stream * a.n_kv;
    const int64_t n_blocks = (n_rows + LID_ROWS_PER_BLOCK - 1) / LID_ROWS_PER_BLOCK;

    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(LANES)]] {
                           const int64_t ir   = item.get_global_id(0);
                           const int64_t lane = ir % LANES;
                           const int64_t row  = ir / LANES;
                           // `row` is uniform across the sub-group, so this return cannot split the reduction below
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
                               k_local[j] = lightning_indexer_k_elem(k_base, a.k_type, lane * EPL + j);
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
                               const int64_t dst_idx =
                                   i_kv + i_batch * (a.nb1 / sizeof(float)) + i_stream * (a.nb3 / sizeof(float));
                               a.dst[dst_idx] = score + (float) m_base[i_kv];
                           }
                       });
}

// Returns false, submitting nothing, when n_embd is not LANES * {2, 4, 8, 16} or K's type is not readable.
template <int LANES> inline bool lightning_indexer_launch(sycl::queue & queue, const lightning_indexer_args & a) {
    if (!lightning_indexer_k_type_supported(a.k_type) || a.n_embd % LANES != 0) {
        return false;
    }
    switch (a.n_embd / LANES) {
        case 2:
            lightning_indexer_launch_impl<LANES, 2>(queue, a);
            return true;
        case 4:
            lightning_indexer_launch_impl<LANES, 4>(queue, a);
            return true;
        case 8:
            lightning_indexer_launch_impl<LANES, 8>(queue, a);
            return true;
        case 16:
            lightning_indexer_launch_impl<LANES, 16>(queue, a);
            return true;
        default:
            return false;
    }
}

}  // namespace ggml_sycl_dsv4

#endif  // GGML_SYCL_LIGHTNING_INDEXER_KERNEL_HPP
