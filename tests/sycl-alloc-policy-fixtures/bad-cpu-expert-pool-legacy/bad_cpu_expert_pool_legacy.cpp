namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class CpuExpertPool {
    ggml_sycl::alloc_handle ring_alloc_;
};
