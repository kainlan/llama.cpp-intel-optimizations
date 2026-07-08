namespace ggml_sycl {
struct scoped_unified_alloc {
    bool allocate(int) { return true; }

    void * get() { return nullptr; }

    int as_mem_handle() { return 0; }
};
}  // namespace ggml_sycl

void bad_readback_fallback_scoped(int req) {
    ggml_sycl::scoped_unified_alloc fallback_alloc;
    (void) fallback_alloc.allocate(req);
    (void) fallback_alloc.get();
    (void) fallback_alloc.as_mem_handle();

    ggml_sycl::scoped_unified_alloc host_alloc;
    (void) host_alloc.allocate(req);
    (void) host_alloc.get();
    (void) host_alloc.as_mem_handle();
}
