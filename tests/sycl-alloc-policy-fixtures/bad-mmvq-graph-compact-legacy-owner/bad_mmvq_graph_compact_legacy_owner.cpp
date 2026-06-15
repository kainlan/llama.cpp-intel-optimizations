namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

namespace std {
template <typename T> T && move(T & value);
}  // namespace std

void bad_mmvq_graph_compact_legacy_owner() {
    ggml_sycl::alloc_handle compact_alloc{};
    ggml_sycl::alloc_handle missing_alloc{};
    ggml_sycl::alloc_handle graph_owner{};
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(compact_alloc), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(missing_alloc), 0);
    (void) ggml_sycl::mem_handle::from_owned_alloc(std::move(graph_owner), 0);
}
