namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

static float *                 s_second_out_dev       = nullptr;
static ggml_sycl::alloc_handle s_second_out_dev_alloc = {};

void use_legacy_split_secondary_output_owner() {
    (void) s_second_out_dev;
    (void) s_second_out_dev_alloc;
}
