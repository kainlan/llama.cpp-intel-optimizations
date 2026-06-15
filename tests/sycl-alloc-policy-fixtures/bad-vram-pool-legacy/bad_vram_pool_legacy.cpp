namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct allocation {
    ggml_sycl::alloc_handle handle;
    unsigned long           size;
};

void release_vram_pool_legacy(allocation & alloc) {
    (void) ggml_sycl::unified_free(alloc.handle);
}
