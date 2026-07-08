namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct tp_host_staging_buffer {
    ggml_sycl::alloc_handle alloc;
};

bool bad_tp_host_staging_legacy(tp_host_staging_buffer & host_staging) {
    return host_staging.alloc.ptr != nullptr;
}
