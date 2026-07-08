namespace ggml_sycl {
struct alloc_handle {};
}

class UnifiedKernel {
    ggml_sycl::alloc_handle sync_block_alloc_;
};
