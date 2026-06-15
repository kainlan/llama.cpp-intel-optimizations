namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};

bool unified_free(alloc_handle) {
    return true;
}
}  // namespace ggml_sycl

void bad_unified_kernel_pinned_persistent_owner() {
    ggml_sycl::alloc_handle entries_owner{};
    auto                    entries_handle = ggml_sycl::mem_handle::from_owned_alloc(entries_owner, 0);

    ggml_sycl::alloc_handle ops_pool_owner{};
    auto                    ops_pool_handle = ggml_sycl::mem_handle::from_owned_alloc(ops_pool_owner, 0);

    ggml_sycl::alloc_handle profile_alloc{};
    void *                  raw_profile_ptr = profile_alloc.ptr;
    (void) ggml_sycl::unified_free(profile_alloc);

    (void) entries_handle;
    (void) ops_pool_handle;
    (void) raw_profile_ptr;
}
