namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};
}  // namespace ggml_sycl

void bad_unified_kernel_device_scratch_owner() {
    ggml_sycl::alloc_handle get_rows_owner{};
    auto                    get_rows_handle = ggml_sycl::mem_handle::from_owned_alloc(get_rows_owner, 0);

    ggml_sycl::alloc_handle scratch_owner{};
    auto                    scratch_handle = ggml_sycl::mem_handle::from_owned_alloc(scratch_owner, 0);

    ggml_sycl::alloc_handle q8_owner{};
    auto                    q8_handle = ggml_sycl::mem_handle::from_owned_alloc(q8_owner, 0);

    ggml_sycl::alloc_handle gate_owner{};
    auto                    gate_handle = ggml_sycl::mem_handle::from_owned_alloc(gate_owner, 0);

    (void) get_rows_handle;
    (void) scratch_handle;
    (void) q8_handle;
    (void) gate_handle;
}
