#include "lightning-indexer.hpp"

#include "lightning-indexer-kernel.hpp"

bool ggml_sycl_lightning_indexer_supported(const ggml_tensor * op) {
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
        !ggml_sycl_dsv4::lightning_indexer_k_type_supported(k->type)) {
        return false;
    }

    // the kernel reads each row as one contiguous run
    if (op->nb[0] != ggml_type_size(op->type) || q->nb[0] != ggml_type_size(q->type) ||
        k->nb[0] != ggml_type_size(k->type) || w->nb[0] != ggml_type_size(w->type) ||
        m->nb[0] != ggml_type_size(m->type)) {
        return false;
    }
    if (op->nb[1] % sizeof(float) != 0 || op->nb[3] % sizeof(float) != 0) {
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

    // one sub-group of WARP_SIZE lanes holds a K row, n_embd / WARP_SIZE elements per lane
    return n_embd % WARP_SIZE == 0 && ggml_sycl_dsv4::lightning_indexer_epl_supported(n_embd / WARP_SIZE);
}

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/4);
    GGML_ASSERT(ggml_sycl_lightning_indexer_supported(dst.raw()));

    const ggml_tensor * op = dst.raw();
    const ggml_tensor * q  = op->src[0];
    const ggml_tensor * k  = op->src[1];
    const ggml_tensor * w  = op->src[2];
    const ggml_tensor * m  = op->src[3];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ggml_sycl_dsv4::lightning_indexer_args args = {};
    args.q                                      = dst.src(0).resolve_as<const char>();
    args.k                                      = dst.src(1).resolve_as<const char>();
    args.w                                      = dst.src(2).resolve_as<const char>();
    args.m                                      = dst.src(3).resolve_as<const char>();
    args.dst                                    = dst.resolve_as<float>();
    args.k_type                                 = k->type;
    args.n_embd                                 = q->ne[0];
    args.n_head                                 = q->ne[1];
    args.n_batch                                = q->ne[2];
    args.n_stream                               = q->ne[3];
    args.n_kv                                   = k->ne[2];
    args.nem3                                   = m->ne[3];
    args.nbq1                                   = q->nb[1];
    args.nbq2                                   = q->nb[2];
    args.nbq3                                   = q->nb[3];
    args.nbk2                                   = k->nb[2];
    args.nbk3                                   = k->nb[3];
    args.nbw1                                   = w->nb[1];
    args.nbw3                                   = w->nb[3];
    args.nbm1                                   = m->nb[1];
    args.nbm3                                   = m->nb[3];
    args.nb1                                    = op->nb[1];
    args.nb3                                    = op->nb[3];

    const bool launched = ggml_sycl_dsv4::lightning_indexer_launch<WARP_SIZE>(*ctx.stream(), args);
    GGML_ASSERT(launched);
}
