namespace ggml_sycl {
class scoped_unified_alloc {
  public:
    explicit scoped_unified_alloc(int);
    int as_mem_handle();
};
}  // namespace ggml_sycl

void bad_persistent_set_rows_validate_scoped(int tmp_req) {
    ggml_sycl::scoped_unified_alloc tmp_alloc(tmp_req);
    (void) tmp_alloc.as_mem_handle();
}
