namespace ggml_sycl {
struct mem_handle {};

struct scoped_unified_alloc {
    bool       allocate(int);
    void *     get() const;
    mem_handle as_mem_handle() const;
};
}  // namespace ggml_sycl

void bad_moe_phase2_d2h_scoped() {
    int                             req = 0;
    ggml_sycl::scoped_unified_alloc d2h_staging;
    d2h_staging.allocate(req);
    void *                staged     = d2h_staging.get();
    ggml_sycl::mem_handle dst_handle = d2h_staging.as_mem_handle();
    (void) staged;
    (void) dst_handle;
}
