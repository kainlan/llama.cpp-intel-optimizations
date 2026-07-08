namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct bad_quant_comm_buffers {
    ggml_sycl::alloc_handle dev_q_alloc[2];
    ggml_sycl::alloc_handle dev_minmax_alloc[2];
    ggml_sycl::alloc_handle host_q0_alloc;
    ggml_sycl::alloc_handle host_q1_alloc;
    ggml_sycl::alloc_handle host_result_alloc;
};

void bad_tp_quant_comm_legacy(bad_quant_comm_buffers & buffers) {
    (void) buffers.dev_q_alloc;
    (void) buffers.dev_minmax_alloc;
    (void) buffers.host_q0_alloc;
    (void) buffers.host_q1_alloc;
    (void) buffers.host_result_alloc;
}
