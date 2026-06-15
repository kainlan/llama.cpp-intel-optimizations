namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};
}  // namespace ggml_sycl

void bad_mmvq_rmsnorm_scales_legacy_owner() {
    ggml_sycl::alloc_handle scales_alloc{};
    auto                    scales_owner = ggml_sycl::mem_handle::from_owned_alloc(scales_alloc, 0);
    (void) scales_owner;
}
