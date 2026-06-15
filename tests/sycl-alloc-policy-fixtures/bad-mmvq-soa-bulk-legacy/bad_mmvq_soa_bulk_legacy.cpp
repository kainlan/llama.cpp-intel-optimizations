namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_mmvq_soa_bulk_legacy() {
    ggml_sycl::alloc_handle bulk_alloc;
    (void) bulk_alloc;
}
