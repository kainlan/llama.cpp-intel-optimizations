namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct split_persistent_resources {
    ggml_sycl::alloc_handle q8_staging_alloc;
    void *                  q8_staging = nullptr;
};

void use_legacy_split_persistent_q8_owner(split_persistent_resources & r) {
    (void) r.q8_staging_alloc;
}
