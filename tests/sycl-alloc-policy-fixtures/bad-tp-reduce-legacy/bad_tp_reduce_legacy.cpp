namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

static ggml_sycl::alloc_handle g_tp_shared_reduce_alloc;
static ggml_sycl::alloc_handle g_tp_host_buf0_alloc;
static ggml_sycl::alloc_handle g_tp_host_buf1_alloc;

void bad_tp_reduce_legacy() {
    (void) g_tp_shared_reduce_alloc;
    (void) g_tp_host_buf0_alloc;
    (void) g_tp_host_buf1_alloc;
}
