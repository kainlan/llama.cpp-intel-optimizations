namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

namespace ggml_sycl {
void bad_pinned_buffer_pool_legacy_owner() {
    alloc_handle act_owner{};
    alloc_handle out_owner{};
    (void) mem_handle::from_owned_alloc(std::move(act_owner), 0);
    (void) mem_handle::from_owned_alloc(std::move(out_owner), 0);
}
}  // namespace ggml_sycl
