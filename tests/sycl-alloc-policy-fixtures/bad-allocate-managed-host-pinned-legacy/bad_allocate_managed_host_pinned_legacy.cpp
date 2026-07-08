namespace ggml_sycl {
struct mem_handle {};

struct alloc_handle {
    void * ptr;

    mem_handle as_mem_handle() const;
};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

bool allocate_managed_host_pinned_legacy(ggml_sycl::alloc_handle * out) {
    auto mh = out->as_mem_handle();
    (void) mh;
    return true;
}

void cpu_dispatch_legacy_cleanup() {
    ggml_sycl::alloc_handle act_owner{};
    ggml_sycl::alloc_handle out_owner{};
    (void) ggml_sycl::unified_free(act_owner);
    (void) ggml_sycl::unified_free(out_owner);
}
