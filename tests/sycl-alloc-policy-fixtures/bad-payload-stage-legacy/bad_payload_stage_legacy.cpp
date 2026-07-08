namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int);
};
}  // namespace ggml_sycl

constexpr int GGML_LAYOUT_AOS = 0;

void bad_payload_stage_legacy(ggml_sycl::alloc_handle host_alloc) {
    auto stage = ggml_sycl::mem_handle::from_owned_alloc(std::move(host_alloc), GGML_LAYOUT_AOS);
    (void) stage;
}
