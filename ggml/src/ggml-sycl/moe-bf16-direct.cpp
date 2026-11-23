#include "moe-bf16-direct.hpp"

#include "mul-mat-bf16.hpp"

#include <limits>

namespace ggml_sycl {

namespace {

// One 16-lane sub-group per output row, four rows per work-group. The lanes stride along k, so a sub-group reads one
// weight row in contiguous 128-byte steps (vec4) and the 16 partial sums meet in one reduce_over_group.
constexpr int MOE_BF16_DIRECT_SG      = 16;
constexpr int MOE_BF16_DIRECT_ROWS_WG = 4;
static_assert(MOE_BF16_DIRECT_SG == 16, "the kernel attribute below spells the sub-group size out");

template <bool VEC4, bool XVEC>
sycl::event moe_bf16_direct_launch(sycl::queue &                 q,
                                   const void * const *          expert_table,
                                   const float *                 src1,
                                   const int32_t *               ids,
                                   float *                       dst,
                                   const moe_bf16_direct_shape & shape) {
    constexpr int SG      = MOE_BF16_DIRECT_SG;
    constexpr int ROWS_WG = MOE_BF16_DIRECT_ROWS_WG;

    const int64_t k         = shape.k;
    const int64_t rows      = shape.rows;
    const int64_t n_experts = shape.n_experts;
    const int64_t src1_rows = shape.src1_rows;
    const size_t  w_nb1     = shape.w_nb1;
    const size_t  src1_nb1  = shape.src1_nb1;
    const size_t  src1_nb2  = shape.src1_nb2;
    const size_t  ids_nb0   = shape.ids_nb0;
    const size_t  ids_nb1   = shape.ids_nb1;
    const size_t  dst_nb1   = shape.dst_nb1;
    const size_t  dst_nb2   = shape.dst_nb2;

    const size_t         row_groups = static_cast<size_t>((rows + ROWS_WG - 1) / ROWS_WG);
    const sycl::range<3> global(static_cast<size_t>(shape.n_tokens), static_cast<size_t>(shape.n_slots),
                                row_groups * ROWS_WG * SG);
    const sycl::range<3> local(1, 1, ROWS_WG * SG);

    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg   = it.get_sub_group();
            const int             lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t         row =
                static_cast<int64_t>(it.get_group(2)) * ROWS_WG + static_cast<int64_t>(sg.get_group_linear_id());
            if (row >= rows) {
                return;
            }
            const int64_t t = static_cast<int64_t>(it.get_group(0));
            const int64_t s = static_cast<int64_t>(it.get_group(1));

            float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + t * dst_nb2 + s * dst_nb1) + row;
            const int32_t id =
                *reinterpret_cast<const int32_t *>(reinterpret_cast<const char *>(ids) + t * ids_nb1 + s * ids_nb0);

            // Every branch below is uniform across the sub-group (row, id and the slot are), so either the whole
            // sub-group reaches the reduction or none of it does.
            if (id == -1) {
                if (lane == 0) {
                    *out = 0.0f;
                }
                return;
            }
            const void * expert = (id >= 0 && id < n_experts) ? expert_table[id] : nullptr;
            if (expert == nullptr) {
                if (lane == 0) {
                    *out = std::numeric_limits<float>::quiet_NaN();
                }
                return;
            }

            const char *  w_row = static_cast<const char *>(expert) + row * w_nb1;
            const float * x     = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src1) + t * src1_nb2 +
                                                                  (s % src1_rows) * src1_nb1);
            float         acc   = 0.0f;
            if (VEC4 && (reinterpret_cast<uintptr_t>(w_row) & 7) == 0) {
                const uint64_t * w4 = reinterpret_cast<const uint64_t *>(w_row);
                const int64_t    k4 = k / 4;
                for (int64_t i = lane; i < k4; i += SG) {
                    const uint64_t packed = w4[i];
                    const float    w0     = ggml_sycl_bf16::bf16_to_f32(static_cast<uint16_t>(packed));
                    const float    w1     = ggml_sycl_bf16::bf16_to_f32(static_cast<uint16_t>(packed >> 16));
                    const float    w2     = ggml_sycl_bf16::bf16_to_f32(static_cast<uint16_t>(packed >> 32));
                    const float    w3     = ggml_sycl_bf16::bf16_to_f32(static_cast<uint16_t>(packed >> 48));
                    float          x0, x1, x2, x3;
                    if (XVEC) {
                        const sycl::float4 xv = reinterpret_cast<const sycl::float4 *>(x)[i];
                        x0                    = xv.x();
                        x1                    = xv.y();
                        x2                    = xv.z();
                        x3                    = xv.w();
                    } else {
                        x0 = x[4 * i + 0];
                        x1 = x[4 * i + 1];
                        x2 = x[4 * i + 2];
                        x3 = x[4 * i + 3];
                    }
                    acc = sycl::fma(w0, x0, acc);
                    acc = sycl::fma(w1, x1, acc);
                    acc = sycl::fma(w2, x2, acc);
                    acc = sycl::fma(w3, x3, acc);
                }
            } else {
                const uint16_t * w = reinterpret_cast<const uint16_t *>(w_row);
                for (int64_t i = lane; i < k; i += SG) {
                    acc = sycl::fma(ggml_sycl_bf16::bf16_to_f32(w[i]), x[i], acc);
                }
            }
            const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) {
                *out = sum;
            }
        });
    });
}

}  // namespace

bool moe_bf16_direct_vec4(const moe_bf16_direct_shape & shape) {
    return shape.k > 0 && shape.k % 4 == 0 && shape.w_nb1 % 8 == 0;
}

bool moe_bf16_direct_submit(sycl::queue &                 q,
                            const void * const *          expert_table,
                            const float *                 src1,
                            const int32_t *               ids,
                            float *                       dst,
                            const moe_bf16_direct_shape & shape,
                            sycl::event *                 event) {
    if (expert_table == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr || event == nullptr) {
        return false;
    }
    if (shape.k <= 0 || shape.rows <= 0 || shape.n_experts <= 0 || shape.n_slots <= 0 || shape.n_tokens <= 0 ||
        shape.src1_rows <= 0 || shape.w_nb1 < static_cast<size_t>(shape.k) * sizeof(uint16_t)) {
        return false;
    }

    // The float4 activation load needs every activation row 16-byte aligned: the base and both row strides.
    const bool vec4 = moe_bf16_direct_vec4(shape);
    const bool xvec =
        vec4 && reinterpret_cast<uintptr_t>(src1) % 16 == 0 && shape.src1_nb1 % 16 == 0 && shape.src1_nb2 % 16 == 0;
    if (xvec) {
        *event = moe_bf16_direct_launch<true, true>(q, expert_table, src1, ids, dst, shape);
    } else if (vec4) {
        *event = moe_bf16_direct_launch<true, false>(q, expert_table, src1, ids, dst, shape);
    } else {
        *event = moe_bf16_direct_launch<false, false>(q, expert_table, src1, ids, dst, shape);
    }
    return true;
}

}  // namespace ggml_sycl
