#include "top-k-mask.hpp"

#include "prestage-class.hpp"
#include "top-k-mask-kernel.hpp"

using ggml_sycl_top_k_mask::elem_stride;

// The device pointer the kernel reads `src` through, or nullptr for an absent optional operand.
//
// The kpool member cells and tail cells (blocks, extra) are graph inputs llama writes on the host every ubatch, and the
// DSA form's mask is llama's KQ mask input. A recorded kernel bakes the pointers it is given, so while this thread
// records, such a source is read from the device copy the graph pre-stage made of it for this token
// (graph_input_staging, keyed by tensor identity and refreshed before every replay), by the pre-stage's own view-aware
// INPUT rule, as bin-broadcast does. The lookup neither allocates, copies nor waits, so recording never falls back to a
// direct run for this op. The pre-stage copied ggml_nbytes(src) from the tensor's own data, which is the span the
// kernel addresses. The staging map owns that copy: a size change that replaces it retires every recorder that baked
// the old pointer and holds the old buffer until the work submitted before the swap has completed.
//
// A miss (an input already on the device, which the pre-stage does not copy, or a second tensor over a range it staged
// once) and every run outside a recording resolve the tensor as any other op does.
template <typename T> static T * top_k_mask_src_ptr(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor src) {
    const ggml_tensor * t = src.raw();
    if (t == nullptr) {
        return nullptr;
    }
    if (ggml_sycl_graph_recording_this_thread() && ggml_sycl_prestage_is_input(t)) {
        void * staged = nullptr;
        if (ctx.graph_input_stage_lookup(t, ggml_nbytes(t), ctx.device, nullptr, &staged) && staged != nullptr) {
            return static_cast<T *>(staged);
        }
    }
    return src.resolve_as<T>();
}

void ggml_sycl_op_top_k_mask(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/5);
    GGML_ASSERT(ggml_sycl_top_k_mask_supported(dst.raw()));

    const ggml_tensor * op     = dst.raw();
    const ggml_tensor * mask   = op->src[0];
    const ggml_tensor * top_k  = op->src[1];
    const ggml_tensor * score  = op->src[2];
    const ggml_tensor * blocks = op->src[3];
    const ggml_tensor * extra  = op->src[4];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_top_k_mask::top_k_mask_args args = {};
    args.mask                                  = top_k_mask_src_ptr<const void>(ctx, dst.src(0));
    args.top_k                                 = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(1));
    args.score                                 = top_k_mask_src_ptr<const float>(ctx, dst.src(2));
    args.blocks                                = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(3));
    args.extra                                 = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(4));
    args.dst                                   = dst.resolve_as<void>();
    args.type                                  = op->type;
    args.n_kv                                  = mask->ne[0];
    args.ne1                                   = op->ne[1];
    args.ne2                                   = op->ne[2];
    args.ne3                                   = op->ne[3];
    args.m1                                    = mask->ne[1];
    args.m2                                    = mask->ne[2];
    args.m3                                    = mask->ne[3];
    args.n_top                                 = top_k->ne[0];
    args.n_pick                                = blocks ? blocks->ne[1] : score ? score->ne[0] : mask->ne[0];
    args.n_blk                                 = blocks ? blocks->ne[0] : 1;
    args.n_extra                               = extra ? extra->ne[0] : 0;
    args.sm1                                   = elem_stride(mask, 1);
    args.sm2                                   = elem_stride(mask, 2);
    args.sm3                                   = elem_stride(mask, 3);
    args.sd1                                   = elem_stride(op, 1);
    args.sd2                                   = elem_stride(op, 2);
    args.sd3                                   = elem_stride(op, 3);
    args.sk0                                   = elem_stride(top_k, 0);
    args.sk1                                   = elem_stride(top_k, 1);
    args.sk2                                   = elem_stride(top_k, 2);
    args.sk3                                   = elem_stride(top_k, 3);
    if (score) {
        args.ss0 = elem_stride(score, 0);
        args.ss1 = elem_stride(score, 1);
        args.ss2 = elem_stride(score, 2);
        args.ss3 = elem_stride(score, 3);
    }
    if (blocks) {
        args.sb0 = elem_stride(blocks, 0);
        args.sb1 = elem_stride(blocks, 1);
    }
    if (extra) {
        args.se0 = elem_stride(extra, 0);
        args.se1 = elem_stride(extra, 1);
        args.se2 = elem_stride(extra, 2);
        args.se3 = elem_stride(extra, 3);
    }

    const bool launched = ggml_sycl_top_k_mask::top_k_mask_launch(*ctx.stream(), args);
    GGML_ASSERT(launched);
}
