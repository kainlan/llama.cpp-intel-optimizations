namespace ggml_sycl {
struct mem_handle {};

struct scoped_unified_alloc {
    void *     get() const;
    mem_handle as_mem_handle() const;
};
}  // namespace ggml_sycl

void bad_get_rows_host_stage_scoped() {
    int                             host_req = 0;
    ggml_sycl::scoped_unified_alloc host_stage(host_req);
    auto                            host_handle = host_stage.as_mem_handle();
    (void) host_handle;
}
