namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_runtime_lookup_owner_alloc() {
    ggml_sycl::alloc_handle owner_alloc{};
    (void) owner_alloc;
}
