namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&);
};
}  // namespace ggml_sycl

void bad_generic_owned_alloc_handoff() {
    ggml_sycl::alloc_handle alloc{};
    auto                    handle = ggml_sycl::mem_handle::from_owned_alloc(std::move(alloc));
    (void) handle;
}
