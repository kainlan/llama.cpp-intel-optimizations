namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct unified_cache {
    std::shared_ptr<ggml_sycl::alloc_handle> compute_arena_owner_;
};

void release_bad_compute_arena(unified_cache & cache) {
    if (cache.compute_arena_owner_ && cache.compute_arena_owner_->ptr) {
        ggml_sycl::unified_free(*cache.compute_arena_owner_);
        cache.compute_arena_owner_.reset();
    }
}
