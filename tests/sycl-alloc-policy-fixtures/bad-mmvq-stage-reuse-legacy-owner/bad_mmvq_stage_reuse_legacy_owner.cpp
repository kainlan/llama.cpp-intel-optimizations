namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

void bad_mmvq_stage_reuse_legacy_owner() {
    ggml_sycl::alloc_handle device_scratch_owner{};
    ggml_sycl::alloc_handle host_stage_owner{};
    ggml_sycl::alloc_handle q8_owner{};
    ggml_sycl::alloc_handle scratch_owner{};
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(device_scratch_owner), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(host_stage_owner), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(q8_owner), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(scratch_owner), 0);
}
