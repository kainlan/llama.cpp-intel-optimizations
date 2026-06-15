namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void set_tensor_reorder() {
    ggml_sycl::alloc_handle reorder_fallback_alloc;
    (void) reorder_fallback_alloc;
}
