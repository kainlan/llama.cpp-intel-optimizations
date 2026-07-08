struct alloc_t {
    int as_mem_handle() const;
};

struct ctx_t {
    alloc_t managed_alloc;
    alloc_t tp_allocs[4];
};

int bad(ctx_t * ctx, int dev) {
    return ctx->managed_alloc.as_mem_handle() + ctx->tp_allocs[dev].as_mem_handle();
}
