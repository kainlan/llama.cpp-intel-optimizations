namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void ensure_moe_ptr_table() {
    ggml_sycl::alloc_handle table_alloc;
    (void) table_alloc;
}
