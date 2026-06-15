namespace ggml_sycl {
struct mem_handle {};

struct alloc_handle {
    mem_handle as_mem_handle() const;
};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct managed_host_pinned_buffer {
    ggml_sycl::alloc_handle handle{};

    bool zero() {
        auto mh = handle.as_mem_handle();
        (void) mh;
        return true;
    }

    void reset() { (void) ggml_sycl::unified_free(handle); }

    ggml_sycl::mem_handle as_mem_handle() const { return handle.as_mem_handle(); }
};
