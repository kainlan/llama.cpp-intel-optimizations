namespace ggml_sycl {
struct mem_handle {};

struct scoped_unified_alloc {
    void *     get() const;
    mem_handle as_mem_handle() const;
};
}  // namespace ggml_sycl

void bad_common_host_staging_scoped() {
    int                             host_req = 0;
    ggml_sycl::scoped_unified_alloc host_alloc(host_req);
    void *                          host_buf    = host_alloc.get();
    ggml_sycl::mem_handle           host_handle = host_alloc.as_mem_handle();
    (void) host_buf;
    (void) host_handle;
}
