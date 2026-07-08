namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct bad_moe_graph_buffers {
    ggml_sycl::alloc_handle q8_1_owner;
};

struct bad_mmvq_q8_activation_cache {
    ggml_sycl::alloc_handle backing_alloc;
};

bool bad_graph_q8_legacy(bad_moe_graph_buffers & buffers, bad_mmvq_q8_activation_cache & cache) {
    return buffers.q8_1_owner.ptr != nullptr || cache.backing_alloc.ptr != nullptr;
}
