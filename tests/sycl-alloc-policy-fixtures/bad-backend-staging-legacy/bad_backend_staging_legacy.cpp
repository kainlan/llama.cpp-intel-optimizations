struct bad_backend_context {
    void * readback_staging_alloc;
    void * mmvq_host_staging_alloc;
    void * staging_buffer_alloc_;
};

namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_backend_staging_legacy_fixture(bad_backend_context * ctx, void * ptr) {
    ctx->readback_staging_alloc  = ptr;
    ctx->mmvq_host_staging_alloc = ptr;
    ctx->staging_buffer_alloc_   = ptr;

    ggml_sycl::alloc_handle device_staging_alloc;
    ggml_sycl::alloc_handle host_staging_alloc;
    ggml_sycl::alloc_handle readback_alloc;
    (void) device_staging_alloc;
    (void) host_staging_alloc;
    (void) readback_alloc;
}
