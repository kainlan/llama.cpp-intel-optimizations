namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct dev1_kv_cache_entry {
    float *                 k_cache;
    float *                 v_cache;
    ggml_sycl::alloc_handle k_alloc;
    ggml_sycl::alloc_handle v_alloc;
};

void use_legacy_dev1_kv_cache_owner(dev1_kv_cache_entry & entry) {
    (void) entry.k_cache;
    (void) entry.v_cache;
    (void) entry.k_alloc;
    (void) entry.v_alloc;
}
