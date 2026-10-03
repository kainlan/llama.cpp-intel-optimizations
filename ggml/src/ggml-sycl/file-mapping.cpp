#include "file-mapping.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>

namespace ggml_sycl {

#if defined(GGML_SYCL_PRIVATE_TESTING)
static std::atomic<uint64_t> g_munmap_count{ 0 };

uint64_t unified_cache_file_mapping_munmap_count_for_test() {
    return g_munmap_count.load(std::memory_order_acquire);
}
#endif

const char * file_map_status_name(file_map_status status) {
    switch (status) {
        case file_map_status::OK:
            return "ok";
        case file_map_status::INVALID_ARGUMENT:
            return "invalid-argument";
        case file_map_status::DUP_FAILED:
            return "dup-failed";
        case file_map_status::FSTAT_FAILED:
            return "fstat-failed";
        case file_map_status::SIZE_MISMATCH:
            return "size-mismatch";
        case file_map_status::RESERVE_FAILED:
            return "reserve-failed";
        case file_map_status::SEGMENT_MAP_FAILED:
            return "segment-map-failed";
        case file_map_status::LOCK_FAILED:
            return "lock-failed";
    }
    return "?";
}

file_map_status unified_cache_raw_map_file(int                                 fd,
                                           uint64_t                            file_size,
                                           const file_map_range &              reservation,
                                           const std::vector<file_map_range> & segments,
                                           file_mapping *                      out) {
    // RED stub (llama.cpp-5efe C2a): the release below skips the munmap, which the host test must catch.
    if (out == nullptr || fd < 0 || reservation.length == 0 || segments.empty()) {
        return file_map_status::INVALID_ARGUMENT;
    }
    const uint64_t page      = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t res_begin = reservation.file_offset / page * page;
    const uint64_t res_end   = (reservation.file_end() + page - 1) / page * page;

    struct stat st = {};
    if (fstat(fd, &st) != 0) {
        return file_map_status::FSTAT_FAILED;
    }
    if ((uint64_t) st.st_size != file_size) {
        return file_map_status::SIZE_MISMATCH;
    }
    const int dup_fd = dup(fd);
    if (dup_fd < 0) {
        return file_map_status::DUP_FAILED;
    }
    void * base = mmap(nullptr, res_end - res_begin, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        close(dup_fd);
        return file_map_status::RESERVE_FAILED;
    }
    for (const file_map_range & seg : segments) {
        void * at = (char *) base + (seg.file_offset - res_begin);
        if (mmap(at, seg.length, PROT_READ, MAP_SHARED | MAP_FIXED, dup_fd, (off_t) seg.file_offset) == MAP_FAILED) {
            munmap(base, res_end - res_begin);
            close(dup_fd);
            return file_map_status::SEGMENT_MAP_FAILED;
        }
    }
    out->base        = base;
    out->length      = res_end - res_begin;
    out->file_offset = res_begin;
    out->fd          = dup_fd;
    return file_map_status::OK;
}

file_map_status unified_cache_raw_lock_file_segments(const file_mapping &                mapping,
                                                     const std::vector<file_map_range> & lock_segments) {
    (void) mapping;
    (void) lock_segments;
    return file_map_status::OK;
}

void unified_cache_raw_unmap_file(file_mapping * mapping) {
    if (mapping == nullptr || mapping->base == nullptr) {
        return;
    }
    // naive release: forget the mapping without unmapping it
    if (mapping->fd >= 0) {
        close(mapping->fd);
    }
    *mapping = {};
}

}  // namespace ggml_sycl
