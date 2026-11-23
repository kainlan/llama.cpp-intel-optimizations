#ifndef GGML_SYCL_TOP_K_MASK_HPP
#define GGML_SYCL_TOP_K_MASK_HPP

#include "common.hpp"
#include "top-k-mask-predicate.hpp"

// GGML_OP_TOP_K_MASK. The predicate (top-k-mask-predicate.hpp, pure ggml so a host test runs it) is the one statement
// of which (type, shape, stride) combinations the kernel implements: ggml_backend_sycl_device_supports_op asks it and
// the executor asserts the same one, so supports_op cannot admit a node the executor aborts on. The kernel itself
// lives in top-k-mask-kernel.hpp.
inline bool ggml_sycl_top_k_mask_supported(const ggml_tensor * op) {
    return ggml_sycl_top_k_mask::op_supported(op);
}

void ggml_sycl_op_top_k_mask(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

#endif  // GGML_SYCL_TOP_K_MASK_HPP
