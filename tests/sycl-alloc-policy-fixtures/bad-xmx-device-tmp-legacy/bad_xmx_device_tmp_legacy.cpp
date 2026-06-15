namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_xmx_device_tmp_legacy() {
    ggml_sycl::alloc_handle tmp_alloc;
    (void) tmp_alloc;
}
