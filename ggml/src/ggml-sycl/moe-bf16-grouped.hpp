#pragma once

// The gather and the scatter of the BF16 grouped prompt arm of MUL_MAT_ID (Inkling 2 Mini plan, Task 24).
//
// The arm runs one oneDNN bf16 x bf16 -> f32 GEMM per active expert on the expert's BF16 weight where it is loaded,
// over that expert's rows packed one after another in a ring slot (moe-bf16-grouped-plan.hpp lays them out). These
// two kernels move the rows: the gather writes every routed row of src1 into its packed row as bf16, rounded to
// nearest even (ggml_sycl_bf16::f32_to_bf16_rne, as the dense BF16 oneDNN route rounds), and the scatter writes every
// packed f32 output row back to its (slot, token) row of dst. Each is one launch over every routed row, so a node of
// any number of experts costs two submissions here; a packed row the plan leaves as padding is neither written nor
// read.
//
// The submitters allocate nothing, free nothing and wait on nothing: each enqueues one kernel on the caller's queue
// after `deps` and hands back its event. The caller owns the row map, src1, the scratch and dst until that event
// completes. The file is free of backend state, so a test compiles it against any SYCL device.

#include "moe-bf16-grouped-plan.hpp"

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// The row map the device reads is the plan's rows as the host holds them.
static_assert(sizeof(moe_bf16_grouped_row) == 3 * sizeof(int32_t), "the device reads the plan's rows as three ints");

// One node's rows and their byte strides, in ggml's terms.
struct moe_bf16_grouped_io_shape {
    int64_t k         = 0;  // src0->ne[0]: an activation row's width, the GEMM's reduction length
    int64_t m         = 0;  // src0->ne[1]: an output row's width
    int64_t src1_rows = 0;  // src1->ne[1]: 1 broadcasts one activation row to every slot, else one row per slot
    size_t  src1_nb1  = 0;  // src1->nb[1], src1->nb[2]
    size_t  src1_nb2  = 0;
    size_t  dst_nb1   = 0;  // dst->nb[1], dst->nb[2]
    size_t  dst_nb2   = 0;
    size_t  n_rows    = 0;  // the routed rows: the plan's rows.size()
};

// Whether the shape is one the two kernels take: nonzero widths and rows, float-aligned strides, and each launch's
// range within an int (a SYCL global range must fit one).
bool moe_bf16_grouped_io_shape_ok(const moe_bf16_grouped_io_shape & shape);

// act[rows[r].packed_row * k + j] = bf16(src1 row (rows[r].slot % src1_rows, rows[r].token))[j] for every routed row r.
// Returns false, and enqueues nothing, when the shape is not one the kernels take or a pointer is null.
bool moe_bf16_grouped_gather_submit(sycl::queue &                     q,
                                    const moe_bf16_grouped_row *      rows,
                                    const float *                     src1,
                                    uint16_t *                        act,
                                    const moe_bf16_grouped_io_shape & shape,
                                    const std::vector<sycl::event> &  deps,
                                    sycl::event *                     event);

// dst row (rows[r].slot, rows[r].token)[i] = out[rows[r].packed_row * m + i] for every routed row r. Returns false,
// and enqueues nothing, when the shape is not one the kernels take or a pointer is null.
bool moe_bf16_grouped_scatter_submit(sycl::queue &                     q,
                                     const moe_bf16_grouped_row *      rows,
                                     const float *                     out,
                                     float *                           dst,
                                     const moe_bf16_grouped_io_shape & shape,
                                     const std::vector<sycl::event> &  deps,
                                     sycl::event *                     event);

}  // namespace ggml_sycl
