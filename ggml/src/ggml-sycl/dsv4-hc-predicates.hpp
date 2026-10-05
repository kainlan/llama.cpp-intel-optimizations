#ifndef GGML_SYCL_DSV4_HC_PREDICATES_HPP
#define GGML_SYCL_DSV4_HC_PREDICATES_HPP

// Which (type, shape, op-param) combinations the DSV4_HC_* kernels implement, and the layout constants they share.
//
// Pure ggml on purpose -- no SYCL, no backend header -- so tests/test-sycl-dsv4-hc-kernels can run the SAME predicates
// the backend ships over real ggml op nodes, on a host with no device. ggml_backend_sycl_device_supports_op asks them
// and each executor (dsv4-hc.cpp) asserts the same one, so supports_op cannot admit an op the executor aborts on.

#include "ggml.h"

#include <cstdint>
#include <cstring>

namespace ggml_sycl_dsv4 {

// The comb op is defined for four streams only (ggml_dsv4_hc_comb asserts hc == 4).
constexpr int HC_COMB_STREAMS = 4;

// mixes[:, t] holds the pre mixes, then the post mixes, then the hc x hc comb matrix; base[] has the same layout.
constexpr int64_t HC_COMB_COMB_OFFSET = 2 * HC_COMB_STREAMS;
constexpr int64_t HC_COMB_MIX_DIM     = (2 + HC_COMB_STREAMS) * HC_COMB_STREAMS;

// scale[0] and scale[1] scale the pre and post mixes; scale[HC_COMB_SCALE_COMB_IDX] scales the comb matrix's mixes.
constexpr int64_t HC_COMB_SCALE_COMB_IDX = 2;

inline int32_t hc_op_param_i32(const ggml_tensor * op, int i) {
    int32_t v;
    std::memcpy(&v, (const int32_t *) op->op_params + i, sizeof(v));
    return v;
}

// An F32 tensor whose byte strides are whole floats, so the kernels can index it in elements.
inline bool hc_f32_tensor(const ggml_tensor * t) {
    if (t == nullptr || t->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

inline int64_t hc_stride(const ggml_tensor * t, int dim) {
    return (int64_t) (t->nb[dim] / sizeof(float));
}

}  // namespace ggml_sycl_dsv4

inline bool ggml_sycl_dsv4_hc_pre_supported(const ggml_tensor * op) {
    using namespace ggml_sycl_dsv4;
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_PRE) {
        return false;
    }
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * w = op->src[1];
    if (!hc_f32_tensor(op) || !hc_f32_tensor(x) || !hc_f32_tensor(w)) {
        return false;
    }

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    if (hc <= 0 || x->ne[3] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (op->ne[0] != n_embd || op->ne[1] != n_tokens || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }

    // gated: w holds gate logits shaped like x; otherwise per-stream weights [hc, n_tokens]
    const bool gated = hc_op_param_i32(op, 1) != 0;
    if (gated) {
        return w->ne[0] == n_embd && w->ne[1] == hc && w->ne[2] == n_tokens;
    }
    return w->ne[0] == hc && w->ne[1] == n_tokens && w->ne[2] == 1;
}

inline bool ggml_sycl_dsv4_hc_comb_supported(const ggml_tensor * op) {
    using namespace ggml_sycl_dsv4;
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_COMB) {
        return false;
    }
    const ggml_tensor * mixes = op->src[0];
    const ggml_tensor * scale = op->src[1];
    const ggml_tensor * base  = op->src[2];
    if (!hc_f32_tensor(op) || !hc_f32_tensor(mixes) || !hc_f32_tensor(scale) || !hc_f32_tensor(base)) {
        return false;
    }

    constexpr int64_t hc = HC_COMB_STREAMS;
    if (mixes->ne[0] != HC_COMB_MIX_DIM || mixes->ne[2] != 1 || mixes->ne[3] != 1) {
        return false;
    }
    if (scale->ne[0] <= HC_COMB_SCALE_COMB_IDX || base->ne[0] != HC_COMB_MIX_DIM) {
        return false;
    }
    if (op->ne[0] != hc || op->ne[1] != hc || op->ne[2] != mixes->ne[1] || op->ne[3] != 1) {
        return false;
    }
    return hc_op_param_i32(op, 1) > 0;
}

inline bool ggml_sycl_dsv4_hc_post_supported(const ggml_tensor * op) {
    using namespace ggml_sycl_dsv4;
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_POST) {
        return false;
    }
    const ggml_tensor * x        = op->src[0];
    const ggml_tensor * residual = op->src[1];
    const ggml_tensor * post     = op->src[2];
    const ggml_tensor * comb     = op->src[3];  // nullptr: identity mixing, each stream keeps its own residual
    if (!hc_f32_tensor(op) || !hc_f32_tensor(x) || !hc_f32_tensor(residual) || !hc_f32_tensor(post)) {
        return false;
    }

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    if (hc <= 0 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    if (residual->ne[0] != n_embd || residual->ne[2] != n_tokens || residual->ne[3] != 1) {
        return false;
    }
    if (post->ne[0] != hc || post->ne[1] != n_tokens || post->ne[2] != 1 || post->ne[3] != 1) {
        return false;
    }
    if (op->ne[0] != n_embd || op->ne[1] != hc || op->ne[2] != n_tokens || op->ne[3] != 1) {
        return false;
    }
    if (comb != nullptr) {
        return hc_f32_tensor(comb) && comb->ne[0] == hc && comb->ne[1] == hc && comb->ne[2] == n_tokens &&
               comb->ne[3] == 1;
    }
    return true;
}

#endif  // GGML_SYCL_DSV4_HC_PREDICATES_HPP
