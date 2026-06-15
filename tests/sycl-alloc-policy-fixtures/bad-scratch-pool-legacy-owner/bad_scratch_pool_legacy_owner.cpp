namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct unified_cache {
    std::shared_ptr<ggml_sycl::alloc_handle> scratch_pool_owner_;
};

void release_bad_scratch_pool(unified_cache & cache) {
    if (cache.scratch_pool_owner_ && cache.scratch_pool_owner_->ptr) {
        ggml_sycl::unified_free(*cache.scratch_pool_owner_);
        cache.scratch_pool_owner_.reset();
    }
}
