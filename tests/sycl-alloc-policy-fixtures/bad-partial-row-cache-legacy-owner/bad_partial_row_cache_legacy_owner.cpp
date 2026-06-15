namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct partial_entry {
    void *                                   ptr;
    std::shared_ptr<ggml_sycl::alloc_handle> owner;
};

void release_bad_partial_row(partial_entry & entry, ggml_sycl::alloc_handle & partial_owner) {
    auto legacy_handle = partial_owner.as_mem_handle();
    (void) legacy_handle;
    if (entry.owner && entry.owner->ptr) {
        ggml_sycl::unified_free(*entry.owner);
    }
}
