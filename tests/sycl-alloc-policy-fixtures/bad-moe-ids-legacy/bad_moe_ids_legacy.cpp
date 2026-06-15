#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct moe_ids_cache_entry {
    uint64_t                hash = 0;
    std::vector<int32_t>    host_ids;
    void *                  device_ids   = nullptr;
    size_t                  device_bytes = 0;
    ggml_sycl::alloc_handle device_alloc;
    ggml_sycl::alloc_handle staging_alloc;
    void *                  staging_ids   = nullptr;
    size_t                  staging_bytes = 0;

    void * device_ids_ptr() const { return device_alloc.ptr ? device_alloc.ptr : device_ids; }

    void * staging_ids_ptr() const { return staging_alloc.ptr ? staging_alloc.ptr : staging_ids; }
};

void use_legacy_moe_ids_owner(moe_ids_cache_entry & entry) {
    (void) entry.device_alloc;
    (void) entry.staging_alloc;
}
