#include <sys/mman.h>

#include <cstdlib>

void bad_vm_alloc_fixture() {
    void * aligned = nullptr;
    (void) posix_memalign(&aligned, 64, 4096);
    void * mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping != MAP_FAILED) {
        munmap(mapping, 4096);
    }
    free(aligned);
}
