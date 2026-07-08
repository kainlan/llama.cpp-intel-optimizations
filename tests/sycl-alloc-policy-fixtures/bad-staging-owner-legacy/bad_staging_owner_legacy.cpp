namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle &);
}  // namespace ggml_sycl

struct unified_cache {
    std::shared_ptr<ggml_sycl::alloc_handle> staging_owner_;
};

void release_bad_staging_owner(unified_cache & cache) {
    if (cache.staging_owner_ && cache.staging_owner_->ptr) {
        ggml_sycl::unified_free(*cache.staging_owner_);
        cache.staging_owner_.reset();
    }
}
