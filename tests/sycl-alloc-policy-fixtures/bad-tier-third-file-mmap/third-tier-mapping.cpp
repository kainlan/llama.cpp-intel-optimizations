// Negative control: an mmap in a file the canon's clause (q) does not name.
#include <sys/mman.h>

#include <cstddef>

void * unified_cache_raw_alloc_host_cpu(size_t mapped) {
    return mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}
