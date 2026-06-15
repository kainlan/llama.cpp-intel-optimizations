namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct tp_async_ffn_job {
    float *                 result_buf;
    ggml_sycl::alloc_handle result_alloc;
};

void use_legacy_async_result_owner(tp_async_ffn_job & job) {
    (void) job.result_buf;
    (void) job.result_alloc;
}
