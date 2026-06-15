namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void reorder_temp() {
    ggml_sycl::alloc_handle temp_vram_h;
    (void) temp_vram_h;
}
