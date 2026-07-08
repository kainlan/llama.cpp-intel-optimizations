namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct sec_cpu_item {
    ggml_sycl::alloc_handle reorder_alloc;
};
