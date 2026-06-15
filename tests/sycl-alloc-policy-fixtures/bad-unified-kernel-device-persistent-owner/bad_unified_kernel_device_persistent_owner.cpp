namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};
}  // namespace ggml_sycl

void bad_unified_kernel_device_persistent_owner() {
    ggml_sycl::alloc_handle ready_owner{};
    auto                    ready_handle = ggml_sycl::mem_handle::from_owned_alloc(ready_owner, 0);

    ggml_sycl::alloc_handle flags_owner{};
    auto                    flags_handle = ggml_sycl::mem_handle::from_owned_alloc(flags_owner, 0);

    (void) ready_handle;
    (void) flags_handle;
}
