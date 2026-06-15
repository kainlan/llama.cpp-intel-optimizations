#include <sycl/sycl.hpp>

void bad_fill_fixture(sycl::queue & q, void * dst) {
    q.memset(dst, 0, 16);
    q.fill(static_cast<int *>(dst), 0, 4);
}

void bad_pointer_fill_fixture(sycl::queue * q, void * dst) {
    q->memset(dst, 0, 16);
    q->fill(static_cast<int *>(dst), 0, 4);
}
