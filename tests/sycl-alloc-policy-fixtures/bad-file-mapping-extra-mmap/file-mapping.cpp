// Negative control: the pinned sites of file-mapping.cpp plus a third mmap in the map function.
#include <sys/mman.h>

#include <cstddef>

static void file_map_munmap(void * base, size_t length) {
    munmap(base, length);
}

int unified_cache_raw_map_file(int dup_fd, size_t length) {
    void * base = mmap(nullptr, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mmap(base, length, PROT_READ, MAP_SHARED | MAP_FIXED, dup_fd, 0) == MAP_FAILED) {
        file_map_munmap(base, length);
        return -1;
    }
    void * third = mmap(nullptr, length, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);  // EXTRA-MMAP
    return third == MAP_FAILED ? -1 : 0;
}
