namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

void bad_compute_buffer_manager_legacy_owner() {
    ggml_sycl::alloc_handle buffer_owner{};
    ggml_sycl::alloc_handle scratch_owner{};
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(buffer_owner), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(scratch_owner), 0);
}
