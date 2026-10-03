#pragma once

// The raw file-mapping primitives of the unified cache's FILE_MAPPED tier
// (llama.cpp-5efe C2, docs/design/sycl-mmap-host-tier.md §4.1).
//
// Device-free on purpose: this header and file-mapping.cpp include no SYCL header, so host tests link the
// translation unit alone. They are the only code in ggml-sycl that maps, locks or unmaps a model file; the
// FILE_MAPPED owner (unified_allocate_file_mapping in unified-cache.cpp) calls them and nothing else does.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// A byte range of the file. Offsets are file offsets, never addresses.
struct file_map_range {
    uint64_t file_offset = 0;
    uint64_t length      = 0;

    uint64_t file_end() const { return file_offset + length; }
};

enum class file_map_status : uint8_t {
    OK = 0,
    INVALID_ARGUMENT,    // a range is empty, unaligned, overlapping or outside the reservation
    DUP_FAILED,          // dup(fd) failed
    FSTAT_FAILED,        // fstat(fd) failed
    SIZE_MISMATCH,       // the file is not the size the loader recorded (a replaced or truncated file)
    RESERVE_FAILED,      // the PROT_NONE address-space reservation failed
    SEGMENT_MAP_FAILED,  // a MAP_FIXED file segment failed
    LOCK_FAILED,         // mlock failed (errno names why)
};

const char * file_map_status_name(file_map_status status);

// One reservation: a single PROT_NONE address range with the file's segments mapped over it. The mapping owns
// `fd`, a dup of the loader's open file, for as long as it is mapped (POSIX_FADV_DONTNEED after staging needs it).
struct file_mapping {
    void *   base        = nullptr;
    size_t   length      = 0;
    uint64_t file_offset = 0;  // the file offset `base` stands for; page aligned
    int      fd          = -1;

    explicit operator bool() const { return base != nullptr; }
};

// Reserves [page_floor(reservation.file_offset), page_ceil(reservation.file_end())) as PROT_NONE anonymous memory,
// then maps every segment over it with PROT_READ, MAP_SHARED | MAP_FIXED. A segment starts on a page and lies inside
// the reservation; segments do not overlap. Bytes between segments stay the reservation's PROT_NONE memory: never
// faulted, and a read of one is a SIGSEGV. `file_size` is the size the loader recorded for the file; the mapping
// refuses a file whose fstat size differs. `fd` is the loader's: the mapping dups it and the caller keeps its own.
// On any failure nothing is left mapped or open and *out is untouched.
file_map_status unified_cache_raw_map_file(int                                 fd,
                                           uint64_t                            file_size,
                                           const file_map_range &              reservation,
                                           const std::vector<file_map_range> & segments,
                                           file_mapping *                      out);

// mlock(2)s each segment, which must start and end on a page and lie inside the reservation. The caller rounds
// inward; this never widens a range, because mlock rounds a mid-page start down to the page.
file_map_status unified_cache_raw_lock_file_segments(const file_mapping &                mapping,
                                                     const std::vector<file_map_range> & lock_segments);

// The only release: ONE munmap of the whole reservation, then the dup is closed. Idempotent.
void unified_cache_raw_unmap_file(file_mapping * mapping);

#if defined(GGML_SYCL_PRIVATE_TESTING)
// Private counter of the munmap calls this translation unit has issued, so a test can tell one whole-reservation
// release from a per-segment one.
uint64_t unified_cache_file_mapping_munmap_count_for_test();
#endif

}  // namespace ggml_sycl
