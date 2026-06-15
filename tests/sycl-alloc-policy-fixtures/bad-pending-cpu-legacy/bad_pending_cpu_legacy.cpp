namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct pending_cpu_scatter {
    ggml_sycl::alloc_handle out_alloc;
    ggml_sycl::alloc_handle act_alloc;
    ggml_sycl::alloc_handle weight_alloc;
};

void assign_legacy(pending_cpu_scatter & g_pending_scatter, pending_cpu_scatter & result) {
    g_pending_scatter.out_alloc = result.out_alloc;
    g_pending_scatter.act_alloc = result.act_alloc;
}
