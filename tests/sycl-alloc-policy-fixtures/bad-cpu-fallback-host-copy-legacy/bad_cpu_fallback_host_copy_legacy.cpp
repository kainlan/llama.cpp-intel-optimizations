namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct fallback_host_copy_legacy {
    void *                  host_ptr = nullptr;
    ggml_sycl::alloc_handle alloc{};
};

void restore_host_copies_legacy(fallback_host_copy_legacy & entry) {
    if (entry.alloc.ptr) {
        ggml_sycl::unified_free(entry.alloc);
        entry.alloc = {};
    }
}
