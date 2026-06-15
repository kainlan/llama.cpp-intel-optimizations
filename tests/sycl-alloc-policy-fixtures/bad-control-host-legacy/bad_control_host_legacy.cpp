namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_control_host_legacy() {
    ggml_sycl::alloc_handle control_alloc;
    (void) control_alloc;
}
