namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

namespace ggml_sycl {
void bad_cpu_expert_pool_legacy_owner() {
    alloc_handle ring_owner{};
    (void) mem_handle::from_owned_alloc(std::move(ring_owner), 0);
}
}  // namespace ggml_sycl
