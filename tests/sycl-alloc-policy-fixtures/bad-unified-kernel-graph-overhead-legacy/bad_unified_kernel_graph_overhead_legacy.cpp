namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};
}  // namespace ggml_sycl

#include <utility>

void bad_unified_kernel_graph_overhead_legacy() {
    ggml_sycl::alloc_handle dummy_alloc;
    ggml_sycl::alloc_handle dummy_owner;
    auto                    dummy_handle = ggml_sycl::mem_handle::from_owned_alloc(std::move(dummy_owner), 0);
    (void) dummy_alloc;
    (void) dummy_handle;
}
