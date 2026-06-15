namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class UnifiedKernel {
    ggml_sycl::alloc_handle get_rows_alloc_;
};
