#include <sycl/sycl.hpp>

void * ggml_sycl_malloc_device_raw(size_t size, const sycl::queue & q, const char * tag);

void bad_fixture(sycl::queue & q) {
    void * ptr = sycl::malloc_device(128, q);
    sycl::free(ptr, q);
    (void) ggml_sycl_malloc_device_raw(128, q, "not_a_probe");
}
