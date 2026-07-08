namespace ggml_sycl {
class scoped_unified_alloc {
  public:
    bool   allocate(int);
    void * get();
    int    as_mem_handle();
};
}  // namespace ggml_sycl

void bad_split_weight_stage_scoped(int req) {
    ggml_sycl::scoped_unified_alloc staging_alloc;
    (void) staging_alloc.allocate(req);
    (void) staging_alloc.get();
    (void) staging_alloc.as_mem_handle();
}
