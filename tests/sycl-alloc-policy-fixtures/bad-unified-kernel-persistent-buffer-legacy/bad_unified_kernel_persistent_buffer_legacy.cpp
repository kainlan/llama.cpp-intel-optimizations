namespace ggml_sycl {
struct alloc_handle {};
}

class UnifiedKernel {
    ggml_sycl::alloc_handle persistent_buf_allocs_[4];
};
