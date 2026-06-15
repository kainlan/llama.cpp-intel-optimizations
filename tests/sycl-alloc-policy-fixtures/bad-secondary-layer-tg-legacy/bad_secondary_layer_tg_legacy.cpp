namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct secondary_layer_tg_buffers {
    ggml_sycl::alloc_handle q8_gate_up_alloc;
    ggml_sycl::alloc_handle q8_down_alloc;
    ggml_sycl::alloc_handle act_batch_primary_dev_alloc;
    ggml_sycl::alloc_handle scatter_primary_dev_alloc;
    ggml_sycl::alloc_handle row_indices_primary_alloc;
};
