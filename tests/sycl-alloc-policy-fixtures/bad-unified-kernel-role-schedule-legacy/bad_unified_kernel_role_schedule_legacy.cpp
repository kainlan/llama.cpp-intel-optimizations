namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class UnifiedKernel {
    ggml_sycl::alloc_handle role_sync_alloc_;
    ggml_sycl::alloc_handle role_elem_alloc_;
    ggml_sycl::alloc_handle role_matmul_alloc_;
};
