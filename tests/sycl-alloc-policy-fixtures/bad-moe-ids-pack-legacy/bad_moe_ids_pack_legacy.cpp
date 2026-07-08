namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void pack_ids() {
    ggml_sycl::alloc_handle ids_pack_alloc;
    (void) ids_pack_alloc;
}
