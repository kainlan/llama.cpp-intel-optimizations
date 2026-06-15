namespace ggml_sycl {
enum class vram_zone_id {
    RUNTIME,
};

void unified_cache_zone_reset(int device_id, vram_zone_id zone);
}  // namespace ggml_sycl

void bad_backend_buffer_runtime_zone_reset(int device) {
    ggml_sycl::unified_cache_zone_reset(device, ggml_sycl::vram_zone_id::RUNTIME);
}
