namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct bad_backend_buffer_context {
    ggml_sycl::alloc_handle managed_alloc{};
    ggml_sycl::alloc_handle tp_allocs[4] = {};
};
