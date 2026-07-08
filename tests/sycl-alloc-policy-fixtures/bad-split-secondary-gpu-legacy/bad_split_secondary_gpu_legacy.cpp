namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

static struct {
    char *                  q8_dev  = nullptr;
    float *                 f32_dev = nullptr;
    ggml_sycl::alloc_handle q8_alloc;
    ggml_sycl::alloc_handle f32_alloc;
} g_split_secondary_gpu;

void use_legacy_split_secondary_gpu_owner() {
    (void) g_split_secondary_gpu.q8_alloc;
    (void) g_split_secondary_gpu.f32_alloc;
}
