namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class PinnedBufferPool {
    ggml_sycl::alloc_handle act_alloc_;
    ggml_sycl::alloc_handle out_alloc_;
};
