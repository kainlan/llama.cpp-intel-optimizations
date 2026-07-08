namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct moe_fusion_state {
    ggml_sycl::alloc_handle fused_alloc;
};
