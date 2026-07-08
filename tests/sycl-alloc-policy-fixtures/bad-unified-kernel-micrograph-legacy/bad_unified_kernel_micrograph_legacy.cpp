namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

class UnifiedKernel {
    ggml_sycl::alloc_handle micro_tile_counters_alloc_;
    ggml_sycl::alloc_handle micro_gen_alloc_;
    ggml_sycl::alloc_handle mmvq_q8_buf_allocs_[2];
    ggml_sycl::alloc_handle mmvq_gate_scratch_alloc_;
    ggml_sycl::alloc_handle mmvq_up_scratch_alloc_;
};
