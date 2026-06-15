namespace ggml_sycl {
struct alloc_handle {
    void * ptr = nullptr;
};

struct scoped_unified_alloc {
    void *       get() const;
    alloc_handle release();
};
}  // namespace ggml_sycl

static const void * ggml_sycl_set_rows_stage_ptr() {
    int                             req = 0;
    ggml_sycl::scoped_unified_alloc scoped_alloc(req);
    ggml_sycl::alloc_handle         staged_owner = scoped_alloc.release();
    return scoped_alloc.get() ? scoped_alloc.get() : staged_owner.ptr;
}
