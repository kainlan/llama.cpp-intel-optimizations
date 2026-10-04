#ifndef GGML_SYCL_LIGHTNING_INDEXER_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_HPP

#include "common.hpp"

// GGML_OP_LIGHTNING_INDEXER. The predicate is the one statement of which (K type, head size, layout)
// combinations the kernel implements: ggml_backend_sycl_device_supports_op asks it and the executor asserts
// the same one. The kernel itself lives in lightning-indexer-kernel.hpp.
bool ggml_sycl_lightning_indexer_supported(const ggml_tensor * op);

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

#endif  // GGML_SYCL_LIGHTNING_INDEXER_HPP
