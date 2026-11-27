#include "moe-bf16-grouped.hpp"

#include "mul-mat-bf16.hpp"

#include <limits>

namespace ggml_sycl {

bool moe_bf16_grouped_io_shape_ok(const moe_bf16_grouped_io_shape & shape) {
    if (shape.k <= 0 || shape.m <= 0 || shape.src1_rows <= 0 || shape.n_rows == 0) {
        return false;
    }
    if (shape.src1_nb1 % sizeof(float) != 0 || shape.src1_nb2 % sizeof(float) != 0 ||
        shape.dst_nb1 % sizeof(float) != 0 || shape.dst_nb2 % sizeof(float) != 0) {
        return false;
    }
    const size_t max_items = static_cast<size_t>(std::numeric_limits<int>::max());
    const size_t widest    = static_cast<size_t>(shape.k > shape.m ? shape.k : shape.m);
    return shape.n_rows <= max_items / widest;
}

bool moe_bf16_grouped_gather_submit(sycl::queue &                     q,
                                    const moe_bf16_grouped_row *      rows,
                                    const float *                     src1,
                                    uint16_t *                        act,
                                    const moe_bf16_grouped_io_shape & shape,
                                    const std::vector<sycl::event> &  deps,
                                    sycl::event *                     event) {
    if (rows == nullptr || src1 == nullptr || act == nullptr || event == nullptr ||
        !moe_bf16_grouped_io_shape_ok(shape)) {
        return false;
    }
    const int64_t k         = shape.k;
    const int64_t src1_rows = shape.src1_rows;
    const size_t  nb1       = shape.src1_nb1 / sizeof(float);
    const size_t  nb2       = shape.src1_nb2 / sizeof(float);
    *event                  = q.submit([&](sycl::handler & h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(shape.n_rows, static_cast<size_t>(k)), [=](sycl::id<2> i) {
            const moe_bf16_grouped_row row = rows[i[0]];
            const size_t               col = i[1];
            const float *              x =
                src1 + static_cast<size_t>(row.slot % src1_rows) * nb1 + static_cast<size_t>(row.token) * nb2;
            act[static_cast<size_t>(row.packed_row) * static_cast<size_t>(k) + col] =
                ggml_sycl_bf16::f32_to_bf16_rne(x[col]);
        });
    });
    return true;
}

bool moe_bf16_grouped_scatter_submit(sycl::queue &                     q,
                                     const moe_bf16_grouped_row *      rows,
                                     const float *                     out,
                                     float *                           dst,
                                     const moe_bf16_grouped_io_shape & shape,
                                     const std::vector<sycl::event> &  deps,
                                     sycl::event *                     event) {
    if (rows == nullptr || out == nullptr || dst == nullptr || event == nullptr ||
        !moe_bf16_grouped_io_shape_ok(shape)) {
        return false;
    }
    const int64_t m   = shape.m;
    const size_t  nb1 = shape.dst_nb1 / sizeof(float);
    const size_t  nb2 = shape.dst_nb2 / sizeof(float);
    *event            = q.submit([&](sycl::handler & h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(shape.n_rows, static_cast<size_t>(m)), [=](sycl::id<2> i) {
            const moe_bf16_grouped_row row = rows[i[0]];
            const size_t               col = i[1];
            float * y = dst + static_cast<size_t>(row.slot) * nb1 + static_cast<size_t>(row.token) * nb2;
            y[col]    = out[static_cast<size_t>(row.packed_row) * static_cast<size_t>(m) + col];
        });
    });
    return true;
}

}  // namespace ggml_sycl
