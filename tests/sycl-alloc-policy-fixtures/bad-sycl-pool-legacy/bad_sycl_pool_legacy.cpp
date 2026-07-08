namespace ggml_sycl {
struct alloc_handle;
bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

template <typename K, typename V> struct fake_map {};

namespace std {
template <typename K, typename V> using unordered_map = fake_map<K, V>;
}  // namespace std

struct bad_sycl_pool {
    struct ggml_sycl_buffer {
        void *                  ptr = nullptr;
        ggml_sycl::alloc_handle handle;
    };

    std::unordered_map<void *, ggml_sycl::alloc_handle> active_handles;

    void release(ggml_sycl_buffer & b, ggml_sycl::alloc_handle & active) {
        (void) active_handles;
        (void) ggml_sycl::unified_free(b.handle);
        (void) ggml_sycl::unified_free(active);
    }
};
