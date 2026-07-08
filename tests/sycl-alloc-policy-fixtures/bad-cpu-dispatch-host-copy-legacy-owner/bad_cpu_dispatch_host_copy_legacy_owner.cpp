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
void bad_cpu_dispatch_host_copy_legacy_owner() {
    alloc_handle copy_alloc{};
    (void) mem_handle::from_owned_alloc(std::move(copy_alloc), 0);
}
}  // namespace ggml_sycl
