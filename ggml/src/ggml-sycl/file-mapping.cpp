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

// Sorts and merges ranges that overlap or touch. Empty ranges are dropped.
static std::vector<file_map_range> merge_ranges(std::vector<file_map_range> in) {
    in.erase(std::remove_if(in.begin(), in.end(), [](const file_map_range & r) { return r.length == 0; }), in.end());
    std::sort(in.begin(), in.end(),
              [](const file_map_range & a, const file_map_range & b) { return a.file_offset < b.file_offset; });
    std::vector<file_map_range> out;
    for (const file_map_range & r : in) {
        if (!out.empty() && r.file_offset <= out.back().file_end()) {
            out.back().length = std::max(out.back().file_end(), r.file_end()) - out.back().file_offset;
        } else {
            out.push_back(r);
        }
    }
    return out;
}

// `a` minus `b`, both sorted and merged.
static std::vector<file_map_range> subtract_ranges(const std::vector<file_map_range> & a,
                                                   const std::vector<file_map_range> & b) {
    std::vector<file_map_range> out;
    for (const file_map_range & r : a) {
        uint64_t at = r.file_offset;
        for (const file_map_range & cut : b) {
            if (cut.file_end() <= at || cut.file_offset >= r.file_end()) {
                continue;
            }
            if (cut.file_offset > at) {
                out.push_back({ at, cut.file_offset - at });
            }
            at = std::max(at, cut.file_end());
        }
        if (at < r.file_end()) {
            out.push_back({ at, r.file_end() - at });
        }
    }
    return out;
}

// `a` intersected with `b`, both sorted and merged.
static std::vector<file_map_range> intersect_ranges(const std::vector<file_map_range> & a,
                                                    const std::vector<file_map_range> & b) {
    std::vector<file_map_range> out;
    for (const file_map_range & x : a) {
        for (const file_map_range & y : b) {
            const uint64_t lo = std::max(x.file_offset, y.file_offset);
            const uint64_t hi = std::min(x.file_end(), y.file_end());
            if (lo < hi) {
                out.push_back({ lo, hi - lo });
            }
        }
    }
    return merge_ranges(out);
}

file_segment_plan file_mapping_segments(const std::vector<file_tensor_range> & tensor_ranges,
                                        const std::vector<file_map_range> &    lazy_ranges,
                                        uint64_t                               page,
                                        uint64_t                               random_window) {
    file_segment_plan plan;
    if (page == 0 || (page & (page - 1)) != 0) {
        plan.error = "the page size is not a power of two";
        return plan;
    }
    const std::vector<file_map_range> lazy = merge_ranges(lazy_ranges);

    std::vector<file_map_range> all;
    std::vector<file_map_range> host;
    for (const file_tensor_range & t : tensor_ranges) {
        if (t.length == 0) {
            continue;
        }
        for (const file_map_range & z : lazy) {
            if (t.file_offset < z.file_end() && z.file_offset < t.file_offset + t.length) {
                plan.error = "a tensor range overlaps a lazy range";
                return plan;
            }
        }
        all.push_back({ t.file_offset, t.length });
        if (t.host_executed) {
            host.push_back({ t.file_offset, t.length });
        }
    }

    // Map segments: outward, merged, never a complement.
    std::vector<file_map_range> outward;
    for (const file_map_range & r : all) {
        const uint64_t lo = page_floor(r.file_offset, page);
        outward.push_back({ lo, page_ceil(r.file_end(), page) - lo });
    }
    plan.map = merge_ranges(outward);

    // WILLNEED / lock segments: host-executed bytes first merged exactly (a page two touching host tensors share
    // holds no lazy byte and may be advised), then rounded inward, then every page holding a lazy byte removed.
    std::vector<file_map_range> inward;
    for (const file_map_range & r : merge_ranges(host)) {
        const uint64_t lo = page_ceil(r.file_offset, page);
        const uint64_t hi = page_floor(r.file_end(), page);
        if (lo < hi) {
            inward.push_back({ lo, hi - lo });
        }
    }
    std::vector<file_map_range> lazy_pages;
    for (const file_map_range & z : lazy) {
        const uint64_t lo = page_floor(z.file_offset, page);
        lazy_pages.push_back({ lo, page_ceil(z.file_end(), page) - lo });
    }
    plan.willneed = subtract_ranges(merge_ranges(inward), merge_ranges(lazy_pages));

    // MADV_RANDOM windows at both edges of every lazy range, the shared pages inside, kept to what is mapped.
    std::vector<file_map_range> windows;
    for (const file_map_range & z : lazy) {
        const uint64_t before = z.file_offset > random_window ? z.file_offset - random_window : 0;
        const uint64_t lo     = page_floor(before, page);
        const uint64_t hi     = page_ceil(z.file_offset, page);
        if (lo < hi) {
            windows.push_back({ lo, hi - lo });
        }
        const uint64_t after_lo = page_floor(z.file_end(), page);
        const uint64_t after_hi = page_ceil(z.file_end() + random_window, page);
        if (after_lo < after_hi) {
            windows.push_back({ after_lo, after_hi - after_lo });
        }
    }
    plan.random_windows = intersect_ranges(merge_ranges(windows), plan.map);
    return plan;
}

uint64_t file_mapping_random_window(uint64_t read_ahead_bytes, uint64_t page) {
    const uint64_t floor_bytes = 2ull * 1024 * 1024;
    const uint64_t w           = std::max(2 * read_ahead_bytes, floor_bytes);
    return page == 0 ? w : page_ceil(w, page);
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
