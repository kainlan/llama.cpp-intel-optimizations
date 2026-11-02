#include "lightning-indexer.hpp"

#include "lightning-indexer-kernel.hpp"

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/4);
    GGML_ASSERT(ggml_sycl_lightning_indexer_supported(dst.raw()));

    const ggml_tensor * op = dst.raw();
    const ggml_tensor * q  = op->src[0];
    const ggml_tensor * k  = op->src[1];
    const ggml_tensor * w  = op->src[2];
    const ggml_tensor * m  = op->src[3];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_lightning_indexer::lightning_indexer_args args = {};
    args.q                                                   = dst.src(0).resolve_as<const char>();
    args.k                                                   = dst.src(1).resolve_as<const char>();
    args.w                                                   = dst.src(2).resolve_as<const char>();
    args.m                                                   = dst.src(3).resolve_as<const char>();
    args.dst                                                 = dst.resolve_as<float>();
    args.k_type                                              = k->type;
    args.n_embd                                              = q->ne[0];
    args.n_head                                              = q->ne[1];
    args.n_batch                                             = q->ne[2];
    args.n_stream                                            = q->ne[3];
    args.n_kv                                                = k->ne[2];
    args.nem3                                                = m->ne[3];
    args.nbq1                                                = q->nb[1];
    args.nbq2                                                = q->nb[2];
    args.nbq3                                                = q->nb[3];
    args.nbk2                                                = k->nb[2];
    args.nbk3                                                = k->nb[3];
    args.nbw1                                                = w->nb[1];
    args.nbw3                                                = w->nb[3];
    args.nbm1                                                = m->nb[1];
    args.nbm3                                                = m->nb[3];
    args.nb1                                                 = op->nb[1];
    args.nb3                                                 = op->nb[3];

    const bool launched = ggml_sycl_lightning_indexer::lightning_indexer_launch<WARP_SIZE>(*ctx.stream(), args);
    GGML_ASSERT(launched);
}
