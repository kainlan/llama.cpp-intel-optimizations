namespace ggml_sycl {
struct scoped_unified_alloc {
    scoped_unified_alloc() = default;

    explicit scoped_unified_alloc(int) {}

    void allocate(int) {}

    void * get() { return nullptr; }

    int as_mem_handle() { return 0; }
};
}  // namespace ggml_sycl

void bad_onednn_xmx_fill_scoped_stage(int tmp_req, int dev_req, int host_req) {
    ggml_sycl::scoped_unified_alloc tmp_alloc;
    tmp_alloc.allocate(tmp_req);
    (void) tmp_alloc.get();

    ggml_sycl::scoped_unified_alloc dev_alloc(dev_req);
    (void) dev_alloc.as_mem_handle();

    ggml_sycl::scoped_unified_alloc host_alloc(host_req);
    (void) host_alloc.get();
}
