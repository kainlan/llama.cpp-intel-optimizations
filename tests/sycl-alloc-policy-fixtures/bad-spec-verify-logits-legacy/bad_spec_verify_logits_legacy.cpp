namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_spec_verify_logits_legacy() {
    ggml_sycl::alloc_handle logits_alloc;
    (void) logits_alloc;
}
