namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle & handle);
}  // namespace ggml_sycl

struct mxfp4_moe_tg_reuse_cache {
    ggml_sycl::alloc_handle q8_alloc     = {};
    ggml_sycl::alloc_handle dpas_b_alloc = {};
    ggml_sycl::alloc_handle dpas_y_alloc = {};
};

void use_legacy_mxfp4_tg_reuse_owner(mxfp4_moe_tg_reuse_cache & cache) {
    if (cache.q8_alloc.ptr) {
        (void) ggml_sycl::unified_free(cache.q8_alloc);
    }
}
