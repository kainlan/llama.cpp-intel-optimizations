#pragma once

// The BF16 expert kernel of MUL_MAT_ID (Inkling 2 Mini plan, Task 8g).
//
// dst[:, s, t] = W[ids[s, t]] * src1[:, s % src1_rows, t] for every slot s and token t, where W[e] is expert e's
// BF16 matrix read in place through a device table of per-expert base pointers (the unified cache's full-local
// expert table, AOS layout), and src1, dst and ids are read and written in place with their own byte strides.
// BF16 -> F32 is an exact 16-bit shift and the sum is F32 (mul-mat-bf16.hpp), so no F16 or Q8_1 copy exists.
//
// Slot semantics match the CPU backend's MUL_MAT_ID: an id of -1 (an unused slot) writes a zero row, and any other id
// outside [0, n_experts), or an expert whose table entry is null, writes a row of quiet NaN, so a bad slot shows in
// the output instead of reading through a pointer nobody checked.
//
// The submitter allocates nothing, frees nothing and waits on nothing: it enqueues the kernel on the caller's queue,
// in token chunks when one launch's global range would not fit (moe_bf16_direct_tokens_per_launch), each chunk after
// the one before it, and hands back the last chunk's event. The caller owns the lifetime of the table, the expert
// storage behind it and the activations until that event completes. The file is free of backend state, so a test
// compiles it against any SYCL device (opencl:cpu on a host without a GPU) and checks the kernel the backend runs.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace ggml_sycl {

// Shape and byte strides of one dispatch, in ggml's terms.
struct moe_bf16_direct_shape {
    int64_t k         = 0;  // src0->ne[0]: elements per weight row, the reduction length
    int64_t rows      = 0;  // src0->ne[1]: output rows per expert
    int64_t n_experts = 0;  // src0->ne[2]: entries of the expert table
    int64_t n_slots   = 0;  // ids->ne[0]: experts used per token
    int64_t n_tokens  = 0;  // ids->ne[1]
    int64_t src1_rows = 0;  // src1->ne[1]: 1 broadcasts one activation row to every slot, else one row per slot
    size_t  w_nb1     = 0;  // src0->nb[1]: bytes between the rows of one expert
    size_t  src1_nb1  = 0;  // src1->nb[1], src1->nb[2]
    size_t  src1_nb2  = 0;
    size_t  ids_nb0   = 0;  // ids->nb[0], ids->nb[1]
    size_t  ids_nb1   = 0;
    size_t  dst_nb1   = 0;  // dst->nb[1], dst->nb[2]
    size_t  dst_nb2   = 0;

    // The most work-items one launch may have. A launch covers its tokens x n_slots x ceil(rows / 4) * 64 work-items
    // (one 16-lane sub-group per output row, four rows per work-group), and a SYCL global range must fit an int, so
    // the backend keeps the default; the kernel test lowers it to force token chunks on a small shape.
    size_t max_launch_items = static_cast<size_t>(std::numeric_limits<int>::max());
};

// Whether the shape takes the vec4 form: four BF16 weights per 8-byte load. It needs k a multiple of 4 and 8-byte
// aligned weight rows; a row whose expert base is not 8-byte aligned still takes the scalar loop inside the kernel.
bool moe_bf16_direct_vec4(const moe_bf16_direct_shape & shape);

// The most tokens one launch takes within shape.max_launch_items: n_tokens when the whole dispatch fits, fewer when
// it runs in chunks, and 0 when the shape is empty or inconsistent or one token alone does not fit.
int64_t moe_bf16_direct_tokens_per_launch(const moe_bf16_direct_shape & shape);

// Enqueues the kernel on q and stores the event of its last launch in *event. Returns false, and enqueues nothing,
// when the shape is empty or inconsistent, one token alone does not fit a launch, or a pointer is null. expert_table,
// src1, ids and dst are device-visible pointers.
bool moe_bf16_direct_submit(sycl::queue &                 q,
                            const void * const *          expert_table,
                            const float *                 src1,
                            const int32_t *               ids,
                            float *                       dst,
                            const moe_bf16_direct_shape & shape,
                            sycl::event *                 event);

}  // namespace ggml_sycl
