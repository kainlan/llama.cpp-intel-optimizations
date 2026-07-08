namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct split_persistent_resources {
    ggml_sycl::alloc_handle progress_counter_alloc;
    ggml_sycl::alloc_handle merge_complete_alloc;
};
