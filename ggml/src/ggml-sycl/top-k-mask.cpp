#include "top-k-mask.hpp"

#include "graph-input-staged.hpp"
#include "top-k-mask-kernel.hpp"

#include <vector>

using ggml_sycl_top_k_mask::elem_stride;

// The device pointer the kernel reads `src` through, or nullptr for an absent optional operand.
//
// The kpool member cells and tail cells (blocks, extra) are graph inputs llama writes on the host every ubatch, and the
// DSA form's mask is llama's KQ mask input. A recorded kernel bakes the pointers it is given, so while this thread
// records, a source is read from the device copy the graph pre-stage made of it for this token, over the
// ggml_nbytes(src) the pre-stage copied from the tensor's own data, which is the span the kernel addresses; the
// staging handle joins `owners`, the op's terminal owners. A source with no such copy must be device memory already,
// or the lookup throws before anything is submitted and the segment runs directly instead (graph-input-staged.hpp).
// Outside a recording the tensor resolves as it does for any other op.
template <typename T>
static T * top_k_mask_src_ptr(ggml_backend_sycl_context &          ctx,
                              ggml_sycl::sycl_tensor               src,
                              std::vector<ggml_sycl::mem_handle> & owners) {
    const ggml_tensor * t = src.raw();
    if (t == nullptr) {
        return nullptr;
    }
    ggml_sycl::mem_handle staged;
    void * ptr = ggml_sycl_graph_input_recordable_ptr(ctx, t, ggml_nbytes(t), ctx.device, staged, "top-k mask");
    if (staged.valid()) {
        owners.push_back(std::move(staged));
    }
    GGML_ASSERT(ptr != nullptr && "top-k mask: source data not resolved");
    return static_cast<T *>(ptr);
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

    // every source resolves, and may throw, before the kernel is submitted
    std::vector<ggml_sycl::mem_handle>    owners;
    ggml_sycl_top_k_mask::top_k_mask_args args = {};
    args.mask                                  = top_k_mask_src_ptr<const void>(ctx, dst.src(0), owners);
    args.top_k                                 = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(1), owners);
    args.score                                 = top_k_mask_src_ptr<const float>(ctx, dst.src(2), owners);
    args.blocks                                = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(3), owners);
    args.extra                                 = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(4), owners);
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

    // A recorded kernel can read a staged copy for the recording's lifetime, a direct run until its event completes;
    // the ticket keeps the staging handles that long.
    auto        ticket = ggml_sycl::terminal_retention_ticket::prepare({}, std::move(owners));
    sycl::event done;
    const bool  launched = ggml_sycl_top_k_mask::top_k_mask_launch(*ctx.stream(), args, &done);
    GGML_ASSERT(launched);
    ticket.mark_submitted(*ctx.stream());
    ticket.commit(done);
}
