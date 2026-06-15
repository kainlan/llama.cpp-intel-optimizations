namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

static constexpr int MERGE_RING_SIZE = 4;

struct secondary_ring_buffers {
    ggml_sycl::alloc_handle dev_q8_1_alloc[MERGE_RING_SIZE];
    ggml_sycl::alloc_handle dev_out_alloc[MERGE_RING_SIZE];
    ggml_sycl::alloc_handle dev_q8_batch_alloc;
    ggml_sycl::alloc_handle dev_ptrs_alloc;
    ggml_sycl::alloc_handle dev_ids_alloc;
    ggml_sycl::alloc_handle dev_agg_alloc;
    ggml_sycl::alloc_handle dev_reduce_alloc;
    ggml_sycl::alloc_handle gate_dev_alloc;
};
