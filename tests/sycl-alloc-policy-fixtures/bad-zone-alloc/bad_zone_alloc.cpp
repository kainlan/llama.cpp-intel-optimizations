namespace ggml_sycl {
enum class vram_zone_id {
    SCRATCH,
};

void * unified_cache_zone_alloc(int device_id, vram_zone_id zone, unsigned long size, unsigned long align = 256);
void   unified_cache_zone_free(int device_id, vram_zone_id zone, void * ptr);
}  // namespace ggml_sycl

void bad_zone_alloc_free() {
    void * ptr = ggml_sycl::unified_cache_zone_alloc(0, ggml_sycl::vram_zone_id::SCRATCH, 4096, 256);
    ggml_sycl::unified_cache_zone_free(0, ggml_sycl::vram_zone_id::SCRATCH, ptr);
}
