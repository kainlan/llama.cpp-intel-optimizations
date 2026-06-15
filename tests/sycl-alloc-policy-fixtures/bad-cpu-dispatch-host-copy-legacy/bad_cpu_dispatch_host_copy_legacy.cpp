namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

template <typename K, typename V> struct fake_map {};

void bad_cpu_dispatch_host_copy_legacy() {
    fake_map<const char *, ggml_sycl::alloc_handle> host_ptr_owned_allocs;
    (void) host_ptr_owned_allocs;
}
