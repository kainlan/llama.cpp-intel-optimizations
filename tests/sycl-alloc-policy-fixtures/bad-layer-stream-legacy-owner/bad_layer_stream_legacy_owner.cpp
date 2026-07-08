namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

bool unified_free(alloc_handle &);

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

namespace ggml_sycl {
void bad_layer_stream_legacy_owner() {
    alloc_handle buffer_alloc{};
    if (buffer_alloc.ptr) {
        (void) unified_free(buffer_alloc);
    }
    (void) mem_handle::from_owned_alloc(std::move(buffer_alloc), 0);
}
}  // namespace ggml_sycl
