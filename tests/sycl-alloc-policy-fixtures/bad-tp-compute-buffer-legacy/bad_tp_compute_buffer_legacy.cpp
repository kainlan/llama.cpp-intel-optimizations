namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct tp_ffn_compute_buffers {
    ggml_sycl::alloc_handle input_q8_alloc;
    ggml_sycl::alloc_handle gate_out_alloc;
    ggml_sycl::alloc_handle up_out_alloc;
    ggml_sycl::alloc_handle hidden_out_alloc;
    ggml_sycl::alloc_handle hidden_q8_alloc;
    ggml_sycl::alloc_handle partial_out_alloc;
};

struct tp_attn_compute_buffers {
    ggml_sycl::alloc_handle q_out_alloc;
    ggml_sycl::alloc_handle k_out_alloc;
    ggml_sycl::alloc_handle v_out_alloc;
    ggml_sycl::alloc_handle attn_out_alloc;
    ggml_sycl::alloc_handle attn_q8_alloc;
    ggml_sycl::alloc_handle attn_scores_alloc;
};
