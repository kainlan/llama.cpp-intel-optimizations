namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void allocate_probe() {
    ggml_sycl::alloc_handle probe_alloc;
    (void) probe_alloc;
}
