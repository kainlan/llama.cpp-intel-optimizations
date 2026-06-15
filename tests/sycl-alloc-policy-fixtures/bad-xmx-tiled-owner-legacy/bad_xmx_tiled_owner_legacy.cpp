namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_xmx_tiled_owner_legacy() {
    ggml_sycl::alloc_handle tiled_alloc;
    (void) tiled_alloc;
}
