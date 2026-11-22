#ifndef GGML_SYCL_TOP_K_MASK_KERNEL_HPP
#define GGML_SYCL_TOP_K_MASK_KERNEL_HPP

// GGML_OP_TOP_K_MASK: the top-k sparse attention mask.
//
// Header-only and free of every backend header on purpose: the launcher takes a plain sycl::queue and raw pointers,
// so tests/test-sycl-dsv4-hc-kernels compiles the SAME kernel and runs it on the opencl:cpu device against an
// independent oracle, with no model, no GPU and no unified cache. top-k-mask.cpp is the only production caller; it
// resolves the ggml tensors into the argument struct.
//
// The semantics are those of the CPU reference (ggml-cpu/ops.cpp ggml_compute_forward_top_k_mask):
//   dst[c, i] = mask[c, i % m] when c is a cell of a live pick of row i, or an extra cell of row i; else -inf.
//   A pick p is live when score is NULL or score[p, i] > -inf; its cells are blocks[:, p], or p itself when blocks is
//   NULL. Picks outside [0, n_pick) and cells outside [0, n_kv) are skipped.
//
// The kernel only copies cells and writes -inf, so it moves raw 16- or 32-bit patterns and never does half-precision
// arithmetic: the result is bit-exact whatever the build's half type. Every stride is in ELEMENTS of its operand.
// Nothing is allocated and nothing blocks the host: the launcher submits one kernel on the caller's in-order queue,
// so it is recordable into a SYCL graph.

#include "top-k-mask-predicate.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace ggml_sycl_top_k_mask {

// One work-group fills one dst row at a time.
constexpr int64_t TOP_K_MASK_BLOCK_SIZE = 256;
// At most this many work-groups, each walking rows group-strided, so the global range stays far inside the 32-bit ids
// the device compiler assumes whatever the row count.
constexpr int64_t TOP_K_MASK_MAX_GROUPS = 65536;

// -inf as the raw cell pattern of each dst type
constexpr uint16_t TOP_K_MASK_NINF_F16 = 0xFC00;
constexpr uint32_t TOP_K_MASK_NINF_F32 = 0xFF800000;

struct top_k_mask_args {
    const void *    mask;    // [n_kv, m1, m2, m3]
    const int32_t * top_k;   // [n_top, ne1, ne2, ne3]
    const float *   score;   // [n_pick, ne1, ne2, ne3], or nullptr: every pick is live
    const int32_t * blocks;  // [n_blk, n_pick], or nullptr: a pick is a cell
    const int32_t * extra;   // [n_extra, ne1, ne2, ne3], or nullptr
    void *          dst;     // [n_kv, ne1, ne2, ne3]
    ggml_type       type;    // of mask and dst
    int64_t         n_kv, ne1, ne2, ne3;
    int64_t         m1, m2, m3;
    int64_t         n_top, n_pick, n_blk, n_extra;
    int64_t         sm1, sm2, sm3;
    int64_t         sd1, sd2, sd3;
    int64_t         sk0, sk1, sk2, sk3;
    int64_t         ss0, ss1, ss2, ss3;
    int64_t         sb0, sb1;
    int64_t         se0, se1, se2, se3;
};

// The number of work-groups the launch uses for n_rows rows.
inline int64_t top_k_mask_groups(int64_t n_rows) {
    return std::min(n_rows, TOP_K_MASK_MAX_GROUPS);
}

// The kernel's event, or a default (complete) event when there are no rows and nothing is submitted.
template <typename T, T NINF> inline sycl::event top_k_mask_submit(sycl::queue & q, const top_k_mask_args & a) {
    const int64_t n_rows = a.ne1 * a.ne2 * a.ne3;
    if (n_rows <= 0) {
        return sycl::event{};
    }
    const int64_t n_groups   = top_k_mask_groups(n_rows);
    const T *     mask       = static_cast<const T *>(a.mask);
    T *           dst        = static_cast<T *>(a.dst);
    const float   ninf_score = -std::numeric_limits<float>::infinity();

    // Every work-item of a group runs the same rows and reaches every barrier: the row loop depends on the group
    // alone and nothing returns early.
    return q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(n_groups * TOP_K_MASK_BLOCK_SIZE), sycl::range<1>(TOP_K_MASK_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
            const int64_t lid = item.get_local_id(0);
            for (int64_t ir = item.get_group(0); ir < n_rows; ir += n_groups) {
                const int64_t i1 = ir % a.ne1;
                const int64_t i2 = (ir / a.ne1) % a.ne2;
                const int64_t i3 = ir / (a.ne1 * a.ne2);

                const T *       mask_row  = mask + (i1 % a.m1) * a.sm1 + (i2 % a.m2) * a.sm2 + (i3 % a.m3) * a.sm3;
                T *             dst_row   = dst + i1 * a.sd1 + i2 * a.sd2 + i3 * a.sd3;
                const int32_t * top_k_row = a.top_k + i1 * a.sk1 + i2 * a.sk2 + i3 * a.sk3;

                for (int64_t c = lid; c < a.n_kv; c += TOP_K_MASK_BLOCK_SIZE) {
                    dst_row[c] = NINF;
                }

                // A kept cell written by one work-item must land after the -inf another work-item wrote to the same
                // cell. The barrier orders global memory within the group.
                sycl::group_barrier(item.get_group());

                // one work-item per pick or extra cell; a cell kept twice gets the same value
                for (int64_t i = lid; i < a.n_top + a.n_extra; i += TOP_K_MASK_BLOCK_SIZE) {
                    if (i >= a.n_top) {
                        const int32_t c = a.extra[(i - a.n_top) * a.se0 + i1 * a.se1 + i2 * a.se2 + i3 * a.se3];
                        if (c >= 0 && c < a.n_kv) {
                            dst_row[c] = mask_row[c];
                        }
                        continue;
                    }

                    const int32_t p = top_k_row[i * a.sk0];
                    if (p < 0 || p >= a.n_pick) {
                        continue;
                    }
                    // a NaN score is not above -inf either, so it is not live, as in the CPU reference
                    if (a.score != nullptr &&
                        !(a.score[p * a.ss0 + i1 * a.ss1 + i2 * a.ss2 + i3 * a.ss3] > ninf_score)) {
                        continue;
                    }
                    if (a.blocks == nullptr) {
                        if (p < a.n_kv) {
                            dst_row[p] = mask_row[p];
                        }
                        continue;
                    }
                    for (int64_t k = 0; k < a.n_blk; ++k) {
                        const int32_t c = a.blocks[k * a.sb0 + p * a.sb1];
                        if (c >= 0 && c < a.n_kv) {
                            dst_row[c] = mask_row[c];
                        }
                    }
                }
            }
        });
}

// Submits the kernel for the cell type in a.type, its event in *out_event when given. Returns false, submitting
// nothing, for a type it has no instantiation for (the predicate admits none).
inline bool top_k_mask_launch(sycl::queue & q, const top_k_mask_args & a, sycl::event * out_event = nullptr) {
    sycl::event ev;
    switch (a.type) {
        case GGML_TYPE_F16:
            ev = top_k_mask_submit<uint16_t, TOP_K_MASK_NINF_F16>(q, a);
            break;
        case GGML_TYPE_F32:
            ev = top_k_mask_submit<uint32_t, TOP_K_MASK_NINF_F32>(q, a);
            break;
        default:
            return false;
    }
    if (out_event != nullptr) {
        *out_event = ev;
    }
    return true;
}

}  // namespace ggml_sycl_top_k_mask

#endif  // GGML_SYCL_TOP_K_MASK_KERNEL_HPP
