namespace ggml_sycl {
struct mem_handle {
    static mem_handle from_chunk_ptr(void *, int, int, bool);
};
}  // namespace ggml_sycl

struct secondary_matmul_info {
    const void * input_device_ptr;
};

void bad_persistent_split_input_rewrap(const secondary_matmul_info & info, int primary_device) {
    auto input_handle = ggml_sycl::mem_handle::from_chunk_ptr(
        const_cast<void *>(info.input_device_ptr), primary_device, 0, true);
    (void) input_handle;
}
