namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(alloc_handle);
}  // namespace ggml_sycl

struct kv_layer_alloc {
    void *                  ptr;
    ggml_sycl::alloc_handle zone_h;
};

void bad_tiered_kv_zone_h_legacy(kv_layer_alloc & la, ggml_sycl::alloc_handle owner) {
    la.ptr    = owner.ptr;
    la.zone_h = owner;
    (void) ggml_sycl::unified_free(la.zone_h);
}
