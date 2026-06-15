#include <sycl/sycl.hpp>

void bad_copy_fixture(sycl::queue & q, void * dst, const void * src) {
    q.memcpy(dst, src, 16);
}

void bad_pointer_copy_fixture(sycl::queue * q, void * dst, const void * src) {
    q->memcpy(dst, src, 16);
}

void bad_dpct_copy_fixture(sycl::queue & q, void * dst, const void * src) {
    dpct::async_dpct_memcpy(dst, src, 16, dpct::device_to_device, q);
}
