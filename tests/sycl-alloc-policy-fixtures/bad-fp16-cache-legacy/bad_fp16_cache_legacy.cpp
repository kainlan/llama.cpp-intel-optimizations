namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct fp16_weight_cache {
    ggml_sycl::alloc_handle slab_alloc;
};
