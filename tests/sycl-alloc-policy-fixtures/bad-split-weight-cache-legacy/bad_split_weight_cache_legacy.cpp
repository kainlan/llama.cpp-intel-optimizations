namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle & handle);
}  // namespace ggml_sycl

struct split_weight_cache_entry {
    void *                  data = nullptr;
    ggml_sycl::alloc_handle handle{};
};

void use_legacy_split_weight_cache_owner(split_weight_cache_entry & entry) {
    if (entry.handle.ptr) {
        (void) ggml_sycl::unified_free(entry.handle);
    }
}
