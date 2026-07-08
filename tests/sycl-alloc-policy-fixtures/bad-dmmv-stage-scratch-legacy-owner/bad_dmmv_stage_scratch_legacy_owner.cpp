namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

void bad_dmmv_stage_scratch_legacy_owner() {
    ggml_sycl::alloc_handle host_stage_owner{};
    ggml_sycl::alloc_handle device_scratch_owner{};
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(host_stage_owner), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(device_scratch_owner), 0);
}
