#ifndef GGML_SYCL_TOP_K_MASK_PREDICATE_HPP
#define GGML_SYCL_TOP_K_MASK_PREDICATE_HPP

// Which GGML_OP_TOP_K_MASK nodes the SYCL kernel implements.
//
// Pure ggml on purpose -- no SYCL, no backend header -- so tests/test-sycl-dsv4-hc-kernels runs the SAME predicate the
// backend ships over real ggml op nodes, on a host with no device. ggml_backend_sycl_device_supports_op asks it (via
// ggml_sycl_top_k_mask_supported in top-k-mask.hpp) and the executor asserts the same one.
//
// It reads types, shapes and strides, never data or buffers: llama's fused-op resolver asks the capability query
// (ggml_backend_sycl_supports_op_capability) about a node that has no storage yet.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl_top_k_mask {

// The stride of `t` along `dim` in elements of its own type. The predicate admits only strides that are whole
// elements, so the division is exact.
inline int64_t elem_stride(const ggml_tensor * t, int dim) {
    return (int64_t) (t->nb[dim] / ggml_type_size(t->type));
}

// Every byte stride of `t` over dims [0, n_dims) is a whole number of its elements: the kernel indexes each operand
// through element strides, and a stride that split an element would silently be truncated.
inline bool strides_whole(const ggml_tensor * t, int n_dims) {
    const size_t ts = ggml_type_size(t->type);
    for (int d = 0; d < n_dims; ++d) {
        if (t->nb[d] % ts != 0) {
            return false;
        }
    }
    return true;
}

// mask   [n_kv, m1, m2, m3]           f16 or f32, row i reads mask row (i1 % m1, i2 % m2, i3 % m3)
// top_k  [n_top, ne1, ne2, ne3]       i32
// score  [n_pick, ne1, ne2, ne3]      f32, may be NULL
// blocks [n_blk, n_pick]              i32, may be NULL
// extra  [n_extra, ne1, ne2, ne3]     i32, may be NULL
// dst    [n_kv, ne1, ne2, ne3]        mask's type
inline bool op_supported(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_TOP_K_MASK) {
        return false;
    }
    const ggml_tensor * mask   = op->src[0];
    const ggml_tensor * top_k  = op->src[1];
    const ggml_tensor * score  = op->src[2];
    const ggml_tensor * blocks = op->src[3];
    const ggml_tensor * extra  = op->src[4];
    if (mask == nullptr || top_k == nullptr) {
        return false;
    }

    // dst holds copies of mask cells and -inf, so it is mask's type; the kernel copies 16- and 32-bit cells
    if ((op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32) || mask->type != op->type) {
        return false;
    }
    if (top_k->type != GGML_TYPE_I32 || (score != nullptr && score->type != GGML_TYPE_F32) ||
        (blocks != nullptr && blocks->type != GGML_TYPE_I32) || (extra != nullptr && extra->type != GGML_TYPE_I32)) {
        return false;
    }

    // a row is one run of cells, and every other stride is walked in whole elements
    if (op->nb[0] != ggml_type_size(op->type) || mask->nb[0] != ggml_type_size(mask->type)) {
        return false;
    }
    if (!strides_whole(op, GGML_MAX_DIMS) || !strides_whole(mask, GGML_MAX_DIMS) ||
        !strides_whole(top_k, GGML_MAX_DIMS) || (score != nullptr && !strides_whole(score, GGML_MAX_DIMS)) ||
        (blocks != nullptr && !strides_whole(blocks, 2)) ||
        (extra != nullptr && !strides_whole(extra, GGML_MAX_DIMS))) {
        return false;
    }

    // one dst row per row of top_k, n_kv cells wide; the mask row is the row index modulo the mask's extent, which
    // must therefore be positive
    if (op->ne[0] != mask->ne[0]) {
        return false;
    }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (op->ne[d] != top_k->ne[d] || mask->ne[d] <= 0) {
            return false;
        }
        if ((score != nullptr && score->ne[d] != top_k->ne[d]) || (extra != nullptr && extra->ne[d] != top_k->ne[d])) {
            return false;
        }
    }

    // blocks is read through its first two dims only, and a pick indexes both blocks and score
    if (blocks != nullptr && (blocks->ne[2] != 1 || blocks->ne[3] != 1)) {
        return false;
    }
    if (blocks != nullptr && score != nullptr && score->ne[0] != blocks->ne[1]) {
        return false;
    }
    return true;
}

}  // namespace ggml_sycl_top_k_mask

#endif  // GGML_SYCL_TOP_K_MASK_PREDICATE_HPP
