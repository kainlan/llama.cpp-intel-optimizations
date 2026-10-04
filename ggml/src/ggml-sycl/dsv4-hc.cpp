#include "dsv4-hc.hpp"

#include "dsv4-hc-kernels.hpp"

// An F32 tensor whose byte strides are whole floats, so the kernels can index it in elements.
static bool dsv4_hc_f32_tensor(const ggml_tensor * t) {
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

static int64_t dsv4_hc_stride(const ggml_tensor * t, int dim) {
    return (int64_t) (t->nb[dim] / sizeof(float));
}

bool ggml_sycl_dsv4_hc_pre_supported(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_PRE) {
        return false;
    }
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * w = op->src[1];
    if (!dsv4_hc_f32_tensor(op) || !dsv4_hc_f32_tensor(x) || !dsv4_hc_f32_tensor(w)) {
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
    const bool gated = ggml_get_op_params_i32(op, 1) != 0;
    if (gated) {
        return w->ne[0] == n_embd && w->ne[1] == hc && w->ne[2] == n_tokens;
    }
    return w->ne[0] == hc && w->ne[1] == n_tokens && w->ne[2] == 1;
}

bool ggml_sycl_dsv4_hc_comb_supported(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_COMB) {
        return false;
    }
    const ggml_tensor * mixes = op->src[0];
    const ggml_tensor * scale = op->src[1];
    const ggml_tensor * base  = op->src[2];
    if (!dsv4_hc_f32_tensor(op) || !dsv4_hc_f32_tensor(mixes) || !dsv4_hc_f32_tensor(scale) ||
        !dsv4_hc_f32_tensor(base)) {
        return false;
    }

    // the comb op is defined for four streams: 2 * 4 pre/post mixes plus the 4 x 4 comb matrix
    constexpr int64_t hc         = ggml_sycl_dsv4::HC_COMB_STREAMS;
    constexpr int64_t hc_mix_dim = (2 + hc) * hc;
    if (mixes->ne[0] != hc_mix_dim || mixes->ne[2] != 1 || mixes->ne[3] != 1) {
        return false;
    }
    if (scale->ne[0] < 3 || base->ne[0] != hc_mix_dim) {
        return false;
    }
    if (op->ne[0] != hc || op->ne[1] != hc || op->ne[2] != mixes->ne[1] || op->ne[3] != 1) {
        return false;
    }
    return ggml_get_op_params_i32(op, 1) > 0;
}

bool ggml_sycl_dsv4_hc_post_supported(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_DSV4_HC_POST) {
        return false;
    }
    const ggml_tensor * x        = op->src[0];
    const ggml_tensor * residual = op->src[1];
    const ggml_tensor * post     = op->src[2];
    const ggml_tensor * comb     = op->src[3];  // nullptr: identity mixing, each stream keeps its own residual
    if (!dsv4_hc_f32_tensor(op) || !dsv4_hc_f32_tensor(x) || !dsv4_hc_f32_tensor(residual) ||
        !dsv4_hc_f32_tensor(post)) {
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
        return dsv4_hc_f32_tensor(comb) && comb->ne[0] == hc && comb->ne[1] == hc && comb->ne[2] == n_tokens &&
               comb->ne[3] == 1;
    }
    return true;
}

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/2);
    GGML_ASSERT(ggml_sycl_dsv4_hc_pre_supported(dst.raw()));

    const ggml_tensor * op    = dst.raw();
    const ggml_tensor * x     = op->src[0];
    const ggml_tensor * w     = op->src[1];
    const bool          gated = ggml_get_op_params_i32(op, 1) != 0;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_dsv4::hc_pre_args args = {};
    args.x                           = dst.src(0).resolve_as<const float>();
    args.w                           = dst.src(1).resolve_as<const float>();
    args.dst                         = dst.resolve_as<float>();
    args.n_embd                      = x->ne[0];
    args.hc                          = x->ne[1];
    args.n_tokens                    = x->ne[2];
    args.sx0                         = dsv4_hc_stride(x, 0);
    args.sx1                         = dsv4_hc_stride(x, 1);
    args.sx2                         = dsv4_hc_stride(x, 2);
    args.sw0                         = dsv4_hc_stride(w, 0);
    args.sw1                         = dsv4_hc_stride(w, 1);
    args.sw2                         = gated ? dsv4_hc_stride(w, 2) : 0;
    args.sd0                         = dsv4_hc_stride(op, 0);
    args.sd1                         = dsv4_hc_stride(op, 1);
    args.scale                       = ggml_get_op_params_f32(op, 0);
    args.gated                       = gated;

    ggml_sycl_dsv4::hc_pre_launch(*ctx.stream(), args);
}

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/3);
    GGML_ASSERT(ggml_sycl_dsv4_hc_comb_supported(dst.raw()));

    const ggml_tensor * op    = dst.raw();
    const ggml_tensor * mixes = op->src[0];
    const ggml_tensor * scale = op->src[1];
    const ggml_tensor * base  = op->src[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_dsv4::hc_comb_args args = {};
    args.mixes                        = dst.src(0).resolve_as<const float>();
    args.scale                        = dst.src(1).resolve_as<const float>();
    args.base                         = dst.src(2).resolve_as<const float>();
    args.dst                          = dst.resolve_as<float>();
    args.n_tokens                     = mixes->ne[1];
    args.sm0                          = dsv4_hc_stride(mixes, 0);
    args.sm1                          = dsv4_hc_stride(mixes, 1);
    args.ss0                          = dsv4_hc_stride(scale, 0);
    args.sb0                          = dsv4_hc_stride(base, 0);
    args.sd0                          = dsv4_hc_stride(op, 0);
    args.sd1                          = dsv4_hc_stride(op, 1);
    args.sd2                          = dsv4_hc_stride(op, 2);
    args.eps                          = ggml_get_op_params_f32(op, 0);
    args.n_iter                       = ggml_get_op_params_i32(op, 1);

    ggml_sycl_dsv4::hc_comb_launch(*ctx.stream(), args);
}

void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/dst.raw()->src[3] ? 4 : 3);
    GGML_ASSERT(ggml_sycl_dsv4_hc_post_supported(dst.raw()));

    const ggml_tensor * op       = dst.raw();
    const ggml_tensor * x        = op->src[0];
    const ggml_tensor * residual = op->src[1];
    const ggml_tensor * post     = op->src[2];
    const ggml_tensor * comb     = op->src[3];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_dsv4::hc_post_args args = {};
    args.x                            = dst.src(0).resolve_as<const float>();
    args.residual                     = dst.src(1).resolve_as<const float>();
    args.post                         = dst.src(2).resolve_as<const float>();
    args.comb                         = comb != nullptr ? dst.src(3).resolve_as<const float>() : nullptr;
    args.dst                          = dst.resolve_as<float>();
    args.n_embd                       = x->ne[0];
    args.hc                           = residual->ne[1];
    args.n_tokens                     = x->ne[1];
    args.sx0                          = dsv4_hc_stride(x, 0);
    args.sx1                          = dsv4_hc_stride(x, 1);
    args.sr0                          = dsv4_hc_stride(residual, 0);
    args.sr1                          = dsv4_hc_stride(residual, 1);
    args.sr2                          = dsv4_hc_stride(residual, 2);
    args.sp0                          = dsv4_hc_stride(post, 0);
    args.sp1                          = dsv4_hc_stride(post, 1);
    args.sc0                          = comb != nullptr ? dsv4_hc_stride(comb, 0) : 0;
    args.sc1                          = comb != nullptr ? dsv4_hc_stride(comb, 1) : 0;
    args.sc2                          = comb != nullptr ? dsv4_hc_stride(comb, 2) : 0;
    args.sd0                          = dsv4_hc_stride(op, 0);
    args.sd1                          = dsv4_hc_stride(op, 1);
    args.sd2                          = dsv4_hc_stride(op, 2);

    ggml_sycl_dsv4::hc_post_launch(*ctx.stream(), args);
}
