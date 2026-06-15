#include <cstdlib>

void bad_host_alloc_fixture() {
    void * a = std::malloc(16);
    void * b = std::aligned_alloc(16, 16);
    std::free(a);
    std::free(b);
}
