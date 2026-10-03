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

static uint64_t file_map_page_size() {
    return (uint64_t) sysconf(_SC_PAGESIZE);
}

static uint64_t page_floor(uint64_t v, uint64_t page) {
    return v / page * page;
}

static uint64_t page_ceil(uint64_t v, uint64_t page) {
    return (v + page - 1) / page * page;
}

file_map_status unified_cache_raw_map_file(int                                 fd,
                                           uint64_t                            file_size,
                                           const file_map_range &              reservation,
                                           const std::vector<file_map_range> & segments,
                                           file_mapping *                      out) {
    if (out == nullptr || fd < 0 || reservation.length == 0 || segments.empty() ||
        reservation.file_end() < reservation.file_offset || reservation.file_end() > file_size) {
        return file_map_status::INVALID_ARGUMENT;
    }
    const uint64_t page      = file_map_page_size();
    const uint64_t res_begin = page_floor(reservation.file_offset, page);
    const uint64_t res_end   = page_ceil(reservation.file_end(), page);

    // Every segment is checked before anything is opened or mapped, so a refusal leaves nothing behind.
    std::vector<file_map_range> sorted = segments;
    std::sort(sorted.begin(), sorted.end(),
              [](const file_map_range & a, const file_map_range & b) { return a.file_offset < b.file_offset; });
    uint64_t prev_end = res_begin;
    for (const file_map_range & seg : sorted) {
        if (seg.length == 0 || seg.file_offset % page != 0 || seg.file_end() < seg.file_offset ||
            seg.file_offset < prev_end || page_ceil(seg.file_end(), page) > res_end) {
            return file_map_status::INVALID_ARGUMENT;
        }
        prev_end = page_ceil(seg.file_end(), page);
    }

    // The loader recorded the file's size when it opened it (register_file_identity); a different size now means the
    // file was replaced or truncated under us.
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
    const size_t length = (size_t) (res_end - res_begin);
    void *       base   = mmap(nullptr, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        close(dup_fd);
        return file_map_status::RESERVE_FAILED;
    }
    for (const file_map_range & seg : sorted) {
        void * at = (char *) base + (seg.file_offset - res_begin);
        if (mmap(at, (size_t) seg.length, PROT_READ, MAP_SHARED | MAP_FIXED, dup_fd, (off_t) seg.file_offset) ==
            MAP_FAILED) {
            const int    err = errno;
            file_mapping partial;
            partial.base        = base;
            partial.length      = length;
            partial.file_offset = res_begin;
            partial.fd          = dup_fd;
            unified_cache_raw_unmap_file(&partial);
            errno = err;
            return file_map_status::SEGMENT_MAP_FAILED;
        }
    }
    out->base        = base;
    out->length      = length;
    out->file_offset = res_begin;
    out->fd          = dup_fd;
    return file_map_status::OK;
}

file_map_status unified_cache_raw_lock_file_segments(const file_mapping &                mapping,
                                                     const std::vector<file_map_range> & lock_segments) {
    if (mapping.base == nullptr) {
        return file_map_status::INVALID_ARGUMENT;
    }
    const uint64_t page = file_map_page_size();
    const uint64_t lo   = mapping.file_offset;
    const uint64_t hi   = mapping.file_offset + mapping.length;

    // Validate all of them first: a refused set locks nothing. A range is never widened here, because mlock rounds a
    // mid-page start down to the page and would lock a neighbouring page (a lazy one) the caller meant to leave out.
    for (const file_map_range & seg : lock_segments) {
        if (seg.length == 0 || seg.file_offset % page != 0 || seg.length % page != 0 || seg.file_offset < lo ||
            seg.file_end() < seg.file_offset || seg.file_end() > hi) {
            return file_map_status::INVALID_ARGUMENT;
        }
    }
    for (const file_map_range & seg : lock_segments) {
        if (mlock((char *) mapping.base + (seg.file_offset - lo), (size_t) seg.length) != 0) {
            return file_map_status::LOCK_FAILED;  // errno names why; the unmap releases whatever was locked
        }
    }
    return file_map_status::OK;
}

void unified_cache_raw_unmap_file(file_mapping * mapping) {
    if (mapping == nullptr || mapping->base == nullptr) {
        return;
    }
    // One munmap of the whole reservation, never one per segment: the segments are not separate allocations.
    munmap(mapping->base, mapping->length);
#if defined(GGML_SYCL_PRIVATE_TESTING)
    g_munmap_count.fetch_add(1, std::memory_order_acq_rel);
#endif
    if (mapping->fd >= 0) {
        close(mapping->fd);
    }
    *mapping = {};
}

}  // namespace ggml_sycl
