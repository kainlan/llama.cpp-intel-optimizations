#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

struct ggml_sycl_set_rows_plan {
    int          owner_device        = 0;
    int          src0_device         = ggml_sycl::mem_handle::HOST_DEVICE;
    int          index_device        = ggml_sycl::mem_handle::HOST_DEVICE;
    const void * src0_ptr            = nullptr;
    const void * index_ptr           = nullptr;
    void *       dst_ptr             = nullptr;
    bool         src0_needs_staging  = false;
    bool         index_needs_staging = false;
};

bool ggml_sycl_plan_set_rows(const ggml_tensor * dst, int fallback_device, ggml_sycl_set_rows_plan * plan);

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

// True while the dense block executor runs a node range on that range's own
// device (llama.cpp-tf8m). Defined in ggml-sycl.cpp; SET_ROWS refuses to stage
// inside such a range.
bool ggml_sycl_block_exec_dense_active();

#endif  // GGML_SYCL_SET_ROWS_HPP
