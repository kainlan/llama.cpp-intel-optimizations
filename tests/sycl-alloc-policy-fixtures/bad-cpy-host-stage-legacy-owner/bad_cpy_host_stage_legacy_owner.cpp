namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

void bad_cpy_host_stage_legacy_owner() {
    ggml_sycl::alloc_handle host_stage_owner{};
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(host_stage_owner), 0);
}
