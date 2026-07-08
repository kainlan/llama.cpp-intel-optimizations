namespace ggml_sycl {
enum class vram_zone_id {
    SCRATCH,
};

struct alloc_handle {
    bool         zone_managed = false;
    vram_zone_id vram_zone    = vram_zone_id::SCRATCH;
};
}  // namespace ggml_sycl

void bad_synthetic_zone_handle() {
    ggml_sycl::alloc_handle h{};
    h.zone_managed = true;
    h.vram_zone    = ggml_sycl::vram_zone_id::SCRATCH;
}
