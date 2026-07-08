namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct sycl_host_buf_ctx {
    void *                  ptr = nullptr;
    ggml_sycl::alloc_handle alloc;
};

void release(sycl_host_buf_ctx * ctx) {
    if (ctx->alloc.ptr) {
        (void) ggml_sycl::unified_free(ctx->alloc);
    }
}
