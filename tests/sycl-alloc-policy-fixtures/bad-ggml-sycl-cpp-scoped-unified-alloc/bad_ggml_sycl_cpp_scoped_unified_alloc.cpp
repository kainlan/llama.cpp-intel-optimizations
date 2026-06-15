namespace ggml_sycl {
struct scoped_unified_alloc {
    scoped_unified_alloc() = default;
};
}  // namespace ggml_sycl

void bad_ggml_sycl_cpp_scoped_unified_alloc() {
    ggml_sycl::scoped_unified_alloc scoped;
    (void) scoped;
}
