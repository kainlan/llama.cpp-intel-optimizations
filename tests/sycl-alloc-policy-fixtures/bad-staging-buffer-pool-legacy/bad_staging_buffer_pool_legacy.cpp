namespace ggml_sycl {
struct alloc_handle {};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct staging_buffer_pool_legacy {
    struct slot {
        bool                    has_unified_handle = false;
        ggml_sycl::alloc_handle unified_handle{};
    };

    void publish(ggml_sycl::alloc_handle unified_handle) {
        slot new_slot{};
        new_slot.unified_handle = static_cast<ggml_sycl::alloc_handle &&>(unified_handle);
    }

    void shutdown(slot & s) {
        if (s.has_unified_handle) {
            (void) ggml_sycl::unified_free(s.unified_handle);
        }
    }
};
