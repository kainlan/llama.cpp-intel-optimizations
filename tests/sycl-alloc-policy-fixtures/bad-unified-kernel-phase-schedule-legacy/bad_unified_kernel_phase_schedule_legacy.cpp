namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class UnifiedKernel {
    ggml_sycl::alloc_handle phase_entries_alloc_;
    ggml_sycl::alloc_handle phase_offset_alloc_;
    ggml_sycl::alloc_handle phase_tiles_alloc_;
    ggml_sycl::alloc_handle phase_type_alloc_;
};
