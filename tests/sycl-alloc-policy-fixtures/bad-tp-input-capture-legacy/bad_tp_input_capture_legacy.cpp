namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void capture_input() {
    ggml_sycl::alloc_handle input_dev1_alloc;
    (void) input_dev1_alloc;
}
