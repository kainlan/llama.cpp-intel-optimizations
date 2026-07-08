namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

static bool ggml_sycl_tp_alloc_tmp(int bytes, ggml_sycl::alloc_handle * out) {
    (void) bytes;
    (void) out;
    return true;
}

void use_legacy_tp_alloc_tmp_out() {
    ggml_sycl::alloc_handle owner{};
    (void) ggml_sycl_tp_alloc_tmp(4096, &owner);
}
