namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct ExpertPredictor {
    float *                 scores_dev_ = nullptr;
    ggml_sycl::alloc_handle scores_alloc_{};
};
