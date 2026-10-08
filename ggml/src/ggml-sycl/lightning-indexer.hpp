#ifndef GGML_SYCL_LIGHTNING_INDEXER_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_HPP

#include "common.hpp"
#include "lightning-indexer-predicate.hpp"

// GGML_OP_LIGHTNING_INDEXER. The predicate (lightning-indexer-predicate.hpp, pure ggml so a host test runs it) is the
// one statement of which (K type, head size, layout) combinations the kernel implements:
// ggml_backend_sycl_device_supports_op asks it and the executor asserts the same one. One sub-group of WARP_SIZE
// lanes holds a K row. The kernel itself lives in lightning-indexer-kernel.hpp.
inline bool ggml_sycl_lightning_indexer_supported(const ggml_tensor * op) {
    return ggml_sycl_lightning_indexer::op_supported(op, WARP_SIZE);
}

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

#endif  // GGML_SYCL_LIGHTNING_INDEXER_HPP
