namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(const alloc_handle &);
}  // namespace ggml_sycl

static void ggml_sycl_release_materialized_after_event(ggml_sycl::alloc_handle q,
                                                       ggml_sycl::alloc_handle k,
                                                       ggml_sycl::alloc_handle v) {
    if (q.ptr) {
        (void) ggml_sycl::unified_free(q);
    }
    if (k.ptr) {
        (void) ggml_sycl::unified_free(k);
    }
    if (v.ptr) {
        (void) ggml_sycl::unified_free(v);
    }
}
