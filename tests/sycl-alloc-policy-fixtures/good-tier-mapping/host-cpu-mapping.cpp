// Positive control: exactly the sites the canon's clause (q) pins in host-cpu-mapping.cpp.
#include <sys/mman.h>

#include <cstddef>

static void host_cpu_munmap(void * ptr, size_t mapped_bytes) {
    munmap(ptr, mapped_bytes);
}

void * unified_cache_raw_alloc_host_cpu(size_t mapped) {
    void * p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    host_cpu_munmap(p, mapped);
    return p;
}
