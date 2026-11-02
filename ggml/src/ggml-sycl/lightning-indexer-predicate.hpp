#ifndef GGML_SYCL_LIGHTNING_INDEXER_PREDICATE_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_PREDICATE_HPP

// Which (K type, head size, layout) combinations the GGML_OP_LIGHTNING_INDEXER kernel implements, and the tables the
// kernel's dispatch is generated from.
//
// Pure ggml on purpose -- no SYCL, no backend header -- so tests/test-sycl-dsv4-hc-kernels runs the SAME predicate the
// backend ships over real ggml op nodes, on a host with no device. ggml_backend_sycl_device_supports_op asks it (via
// ggml_sycl_lightning_indexer_supported in lightning-indexer.hpp) and the executor asserts the same one.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

// K row types the kernel reads, and the elements-per-lane counts it is instantiated for. These two lists are the
// only place either is written: the predicate's switches, the launcher's dispatch and the host test's enumeration
// are all generated from them, so they cannot disagree. Adding a K type means adding it here AND, in
// lightning-indexer-kernel.hpp, a lightning_indexer_k_storage specialization (the block or element type its
// alignment is checked against) and a dequant in lightning_indexer_k_elem; the build fails without either.
// clang-format off
#define GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X) \
    X(GGML_TYPE_F32)                           \
    X(GGML_TYPE_F16)                           \
    X(GGML_TYPE_BF16)                          \
    X(GGML_TYPE_Q8_0)                          \
    X(GGML_TYPE_Q5_1)                          \
    X(GGML_TYPE_Q5_0)                          \
    X(GGML_TYPE_Q4_1)                          \
    X(GGML_TYPE_Q4_0)                          \
    X(GGML_TYPE_IQ4_NL)

#define GGML_SYCL_LIGHTNING_INDEXER_EPLS(X) X(2) X(4) X(8) X(16)
// clang-format on

namespace ggml_sycl_lightning_indexer {

inline bool k_type_supported(ggml_type t) {
    switch (t) {
#define X(T) \
    case T:  \
        return true;
        GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X)
#undef X
        default:
            return false;
    }
}

// Elements per lane the kernel is instantiated for. n_embd must equal lanes * epl for one of these.
inline bool epl_supported(int64_t epl) {
    switch (epl) {
#define X(E) \
    case E:  \
        return true;
        GGML_SYCL_LIGHTNING_INDEXER_EPLS(X)
#undef X
        default:
            return false;
    }
}

// The alignment the kernel's casts of a K row need: the 32-bit-aligned block types carry a half2, the rest are
// 16-bit-aligned. lightning-indexer-kernel.hpp static_asserts this against the block types themselves.
constexpr size_t k_align(ggml_type t) {
    return (t == GGML_TYPE_F32 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_1) ? 4 : 2;
}

// An operand whose strides are whole elements of the width the kernel casts to, so every access it makes is aligned.
inline bool strides_aligned(const ggml_tensor * t, std::initializer_list<int> dims, size_t align) {
    for (int d : dims) {
        if (t->nb[d] % align != 0) {
            return false;
        }
    }
    return true;
}

// `lanes` is the sub-group size the kernel is launched with: one sub-group holds a K row, n_embd / lanes elements
// per lane.
inline bool op_supported(const ggml_tensor * op, int lanes) {
    if (op == nullptr || op->op != GGML_OP_LIGHTNING_INDEXER) {
        return false;
    }
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];
    if (q == nullptr || k == nullptr || w == nullptr || m == nullptr) {
        return false;
    }

    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F16 ||
        !k_type_supported(k->type)) {
        return false;
    }

    // the kernel reads each row as one contiguous run
    if (op->nb[0] != ggml_type_size(op->type) || q->nb[0] != ggml_type_size(q->type) ||
        k->nb[0] != ggml_type_size(k->type) || w->nb[0] != ggml_type_size(w->type) ||
        m->nb[0] != ggml_type_size(m->type)) {
        return false;
    }

    // and indexes every operand through casts that need the strides it walks to keep them aligned
    if (!strides_aligned(op, { 1, 3 }, sizeof(float)) || !strides_aligned(q, { 1, 2, 3 }, sizeof(float)) ||
        !strides_aligned(w, { 1, 3 }, sizeof(float)) || !strides_aligned(m, { 1, 3 }, sizeof(uint16_t)) ||
        !strides_aligned(k, { 2, 3 }, k_align(k->type))) {
        return false;
    }

    // [n_embd, n_head, n_batch, n_stream] x [n_embd, 1, n_kv, n_stream] -> [n_kv, n_batch, 1, n_stream]
    const int64_t n_embd   = q->ne[0];
    const int64_t n_head   = q->ne[1];
    const int64_t n_batch  = q->ne[2];
    const int64_t n_stream = q->ne[3];
    const int64_t n_kv     = k->ne[2];
    if (k->ne[0] != n_embd || k->ne[1] != 1 || k->ne[3] != n_stream) {
        return false;
    }
    if (w->ne[0] != n_head || w->ne[1] != n_batch || w->ne[2] != 1 || w->ne[3] != n_stream) {
        return false;
    }
    if (m->ne[0] != n_kv || m->ne[1] != n_batch || m->ne[2] != 1 || m->ne[3] <= 0 || n_stream % m->ne[3] != 0) {
        return false;
    }
    if (op->ne[0] != n_kv || op->ne[1] != n_batch || op->ne[2] != 1 || op->ne[3] != n_stream) {
        return false;
    }

    return lanes > 0 && n_embd % lanes == 0 && epl_supported(n_embd / lanes);
}

}  // namespace ggml_sycl_lightning_indexer

#endif  // GGML_SYCL_LIGHTNING_INDEXER_PREDICATE_HPP
