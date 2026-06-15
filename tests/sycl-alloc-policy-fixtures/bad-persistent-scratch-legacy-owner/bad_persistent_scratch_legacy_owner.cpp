namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct persistent_scratch_entry {
    void *                                   device_ptr = nullptr;
    std::shared_ptr<ggml_sycl::alloc_handle> owner;
};

void release_bad_persistent_scratch(persistent_scratch_entry & entry) {
    if (entry.owner && entry.owner->ptr) {
        ggml_sycl::unified_free(*entry.owner);
        entry.owner.reset();
    }
}
