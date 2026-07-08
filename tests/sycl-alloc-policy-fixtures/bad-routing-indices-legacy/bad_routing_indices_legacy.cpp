#include <cstddef>

namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct routing_indices_cache {
    int32_t *               host_indices;
    size_t                  capacity;
    ggml_sycl::alloc_handle host_alloc;  // Ownership for host_indices
};

routing_indices_cache g_routing_indices_cache;

void clear_legacy_routing_indices_cache() {
    (void) g_routing_indices_cache.host_alloc;
    ggml_sycl::alloc_handle new_alloc;
    (void) new_alloc;
}
