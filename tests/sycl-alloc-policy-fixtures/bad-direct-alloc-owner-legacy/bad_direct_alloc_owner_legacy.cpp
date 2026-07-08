namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct cache_entry {
    std::shared_ptr<ggml_sycl::alloc_handle> direct_alloc_owner;
};

struct managed_ref {
    std::shared_ptr<ggml_sycl::alloc_handle> owner;
};

void release_bad_direct_owner(cache_entry & entry, managed_ref & ref, ggml_sycl::alloc_handle owner) {
    std::shared_ptr<ggml_sycl::alloc_handle> new_direct_alloc_owner;
    new_direct_alloc_owner   = std::make_shared<ggml_sycl::alloc_handle>(std::move(owner));
    entry.direct_alloc_owner = new_direct_alloc_owner;
    if (entry.direct_alloc_owner && entry.direct_alloc_owner->ptr) {
        ggml_sycl::unified_free(*entry.direct_alloc_owner);
    }
    if (ref.owner && ref.owner->ptr) {
        ggml_sycl::unified_free(*ref.owner);
    }
}
