namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_reorder_d2h_stage_legacy() {
    ggml_sycl::alloc_handle src_stage_alloc;
    (void) src_stage_alloc;
}
