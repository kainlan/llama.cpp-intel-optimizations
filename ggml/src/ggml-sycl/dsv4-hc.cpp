#include "dsv4-hc.hpp"

#include "dsv4-hc-kernels.hpp"

using ggml_sycl_dsv4::hc_stride;

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/2);
    GGML_ASSERT(ggml_sycl_dsv4_hc_pre_supported(dst.raw()));

    const ggml_tensor * op    = dst.raw();
    const ggml_tensor * x     = op->src[0];
    const ggml_tensor * w     = op->src[1];
    const bool          gated = ggml_sycl_dsv4::hc_op_param_i32(op, ggml_sycl_dsv4::HC_OP_PARAM_I32_SLOT) != 0;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_dsv4::hc_pre_args args = {};
    args.x                           = dst.src(0).resolve_as<const float>();
    args.w                           = dst.src(1).resolve_as<const float>();
    args.dst                         = dst.resolve_as<float>();
    args.n_embd                      = x->ne[0];
    args.hc                          = x->ne[1];
    args.n_tokens                    = x->ne[2];
    args.sx0                         = hc_stride(x, 0);
    args.sx1                         = hc_stride(x, 1);
    args.sx2                         = hc_stride(x, 2);
    args.sw0                         = hc_stride(w, 0);
    args.sw1                         = hc_stride(w, 1);
    args.sw2                         = gated ? hc_stride(w, 2) : 0;
    args.sd0                         = hc_stride(op, 0);
    args.sd1                         = hc_stride(op, 1);
    args.scale                       = ggml_get_op_params_f32(op, ggml_sycl_dsv4::HC_OP_PARAM_F32_SLOT);
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
    args.sm0                          = hc_stride(mixes, 0);
    args.sm1                          = hc_stride(mixes, 1);
    args.ss0                          = hc_stride(scale, 0);
    args.sb0                          = hc_stride(base, 0);
    args.sd0                          = hc_stride(op, 0);
    args.sd1                          = hc_stride(op, 1);
    args.sd2                          = hc_stride(op, 2);
    args.eps                          = ggml_get_op_params_f32(op, ggml_sycl_dsv4::HC_OP_PARAM_F32_SLOT);
    args.n_iter                       = ggml_sycl_dsv4::hc_op_param_i32(op, ggml_sycl_dsv4::HC_OP_PARAM_I32_SLOT);

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
    args.sx0                          = hc_stride(x, 0);
    args.sx1                          = hc_stride(x, 1);
    args.sr0                          = hc_stride(residual, 0);
    args.sr1                          = hc_stride(residual, 1);
    args.sr2                          = hc_stride(residual, 2);
    args.sp0                          = hc_stride(post, 0);
    args.sp1                          = hc_stride(post, 1);
    args.sc0                          = comb != nullptr ? hc_stride(comb, 0) : 0;
    args.sc1                          = comb != nullptr ? hc_stride(comb, 1) : 0;
    args.sc2                          = comb != nullptr ? hc_stride(comb, 2) : 0;
    args.sd0                          = hc_stride(op, 0);
    args.sd1                          = hc_stride(op, 1);
    args.sd2                          = hc_stride(op, 2);

    ggml_sycl_dsv4::hc_post_launch(*ctx.stream(), args);
}
