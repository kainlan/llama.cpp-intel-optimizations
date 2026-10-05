#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"
#include "dsv4-hc-predicates.hpp"

// GGML_OP_DSV4_HC_PRE / _COMB / _POST. The *_supported predicates (dsv4-hc-predicates.hpp, pure ggml so a host
// test runs them) are the one statement of which (type, shape, op-param) combinations the kernels implement:
// ggml_backend_sycl_device_supports_op asks them and each executor asserts the same one, so supports_op cannot
// admit an op the executor aborts on. The kernels themselves live in dsv4-hc-kernels.hpp.

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);
void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

#endif  // GGML_SYCL_DSV4_HC_HPP
