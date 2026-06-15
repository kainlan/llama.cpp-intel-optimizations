namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

bool unified_free(alloc_handle &);

struct chunk {
    void *         base  = nullptr;
    alloc_handle * owner = nullptr;
};

void bad_pinned_pool_chunk_legacy_owner(chunk & c) {
    alloc_handle owner{};
    if (c.owner && c.owner->ptr) {
        (void) unified_free(*c.owner);
    }
}
}  // namespace ggml_sycl
