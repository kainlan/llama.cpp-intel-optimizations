namespace ggml_sycl {
struct alloc_handle {
    int device;
};

bool unified_lookup(void *, alloc_handle *);
bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct backend_buffer_context_legacy {
    void * dev_ptr = nullptr;
    int    device  = 0;

    void free_missing_owner() {
        ggml_sycl::alloc_handle looked_up{};
        if (ggml_sycl::unified_lookup(dev_ptr, &looked_up) && looked_up.device == device) {
            (void) ggml_sycl::unified_free(looked_up);
        }
    }
};
