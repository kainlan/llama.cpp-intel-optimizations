namespace ggml_sycl {
struct mem_handle {
    static mem_handle from_chunk_ptr(void *, int, int, bool);
};
}  // namespace ggml_sycl

static constexpr int GGML_LAYOUT_AOS = 0;

void bad_pending_device_token_rewrap(void * src_device) {
    auto src_handle =
        ggml_sycl::mem_handle::from_chunk_ptr(src_device, 0, GGML_LAYOUT_AOS, true);
    (void) src_handle;
}
