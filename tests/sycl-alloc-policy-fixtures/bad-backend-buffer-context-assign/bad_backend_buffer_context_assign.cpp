struct ctx_t {
    int managed_alloc;
    int tp_allocs[4];
};

void bad(ctx_t * ctx, int alloc, int dev) {
    ctx->managed_alloc  = alloc;
    ctx->tp_allocs[dev] = alloc;
}
