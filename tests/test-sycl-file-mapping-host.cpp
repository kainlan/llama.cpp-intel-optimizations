// Host test for the FILE_MAPPED tier's raw file-mapping primitives (llama.cpp-5efe C2).
//
// Links ggml-sycl/file-mapping.cpp alone: no SYCL header, no device, no model. It checks the reservation shape (one
// PROT_NONE reservation with MAP_FIXED file segments at the right offsets, the bytes read back equal to the file's),
// dup fd ownership, the size-mismatch refusal, the argument refusals, the ranged lock, and the private munmap counter:
// exactly one munmap, of the whole reservation, at release, and the unwind of a segment that fails to map.

#include "../ggml/src/ggml-sycl/file-mapping.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ggml_sycl;

// The build is -DNDEBUG, so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static uint64_t g_page = 0;

static unsigned char pattern(uint64_t i) {
    return (unsigned char) (i * 131 + 7);
}

struct fixture {
    std::string path;
    int         fd   = -1;
    uint64_t    size = 0;

    explicit fixture(uint64_t pages) {
        char      tmpl[] = "/tmp/file-mapping-host-XXXXXX";
        const int tmp    = mkstemp(tmpl);
        if (tmp < 0) {
            std::perror("mkstemp");
            std::exit(2);
        }
        path = tmpl;
        size = pages * g_page;
        std::vector<unsigned char> bytes(size);
        for (uint64_t i = 0; i < size; ++i) {
            bytes[i] = pattern(i);
        }
        if (write(tmp, bytes.data(), size) != (ssize_t) size) {
            std::perror("write");
            std::exit(2);
        }
        close(tmp);
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::perror("open");
            std::exit(2);
        }
    }

    ~fixture() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path.c_str());
    }
};

static bool fd_is_open(int fd) {
    return fcntl(fd, F_GETFD) != -1;
}

static int open_fd_count() {
    int n = 0;
    for (int fd = 0; fd < 1024; ++fd) {
        n += fd_is_open(fd) ? 1 : 0;
    }
    return n;
}

struct vma {
    uintptr_t   start = 0;
    uintptr_t   end   = 0;
    std::string perms;
    uint64_t    offset = 0;
    std::string file;
};

static std::vector<vma> read_vmas() {
    std::vector<vma> out;
    std::ifstream    in("/proc/self/maps");
    std::string      line;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string        range, perms, off, dev, inode, file;
        s >> range >> perms >> off >> dev >> inode;
        std::getline(s, file);
        vma          v;
        const size_t dash = range.find('-');
        v.start           = std::stoull(range.substr(0, dash), nullptr, 16);
        v.end             = std::stoull(range.substr(dash + 1), nullptr, 16);
        v.perms           = perms;
        v.offset          = std::stoull(off, nullptr, 16);
        const size_t b    = file.find_first_not_of(' ');
        v.file            = b == std::string::npos ? "" : file.substr(b);
        out.push_back(v);
    }
    return out;
}

// the VMAs that overlap [lo, hi)
static std::vector<vma> vmas_in(uintptr_t lo, uintptr_t hi) {
    std::vector<vma> out;
    for (const vma & v : read_vmas()) {
        if (v.end > lo && v.start < hi) {
            out.push_back(v);
        }
    }
    return out;
}

// Locked kB of the VMA that starts at `start`, from /proc/self/smaps; -1 if not found.
static long locked_kb(uintptr_t start) {
    std::ifstream in("/proc/self/smaps");
    std::string   line;
    bool          inside = false;
    while (std::getline(in, line)) {
        unsigned long long lo = 0;
        unsigned long long hi = 0;
        char               perms[8];
        // a VMA header is "lo-hi perms ..."; a field line ("Size:", "Anonymous:") never parses to three items
        if (std::sscanf(line.c_str(), "%llx-%llx %7s", &lo, &hi, perms) == 3) {
            inside = (uintptr_t) lo == start;
            continue;
        }
        long kb = 0;
        if (inside && std::sscanf(line.c_str(), "Locked: %ld kB", &kb) == 1) {
            return kb;
        }
    }
    return -1;
}

// Fixture file of 8 pages. Reservation = pages [1, 7); segments = page 1, and pages 4-5. Pages 2-3 and 6 stay holes.
static int test_reservation_shape() {
    fixture                           f(8);
    const uint64_t                    P = g_page;
    file_mapping                      m;
    const file_map_range              reservation = { 1 * P, 6 * P };
    const std::vector<file_map_range> segments    = {
        { 1 * P, 1 * P },
        { 4 * P, 2 * P }
    };

    CHECK(unified_cache_raw_map_file(f.fd, f.size, reservation, segments, &m) == file_map_status::OK, "map refused");
    CHECK(m.base != nullptr, "no base");
    CHECK(m.length == 6 * P, "reservation length is not 6 pages");
    CHECK(m.file_offset == 1 * P, "reservation does not stand for file offset 1 page");

    const uintptr_t base = (uintptr_t) m.base;

    // bytes read back equal the file's, at the segments
    const unsigned char * bytes = (const unsigned char *) m.base;
    for (uint64_t i = 0; i < P; ++i) {
        CHECK(bytes[i] == pattern(1 * P + i), "segment A byte differs from the file's");
    }
    for (uint64_t i = 0; i < 2 * P; ++i) {
        CHECK(bytes[3 * P + i] == pattern(4 * P + i), "segment B byte differs from the file's");
    }

    // one PROT_NONE reservation with the segments over it, at the right file offsets
    const std::vector<vma> vmas   = vmas_in(base, base + m.length);
    int                    n_file = 0;
    int                    n_none = 0;
    for (const vma & v : vmas) {
        if (v.file.find(f.path) != std::string::npos) {
            ++n_file;
            CHECK(v.perms == "r--s", "a file segment is not PROT_READ | MAP_SHARED");
            if (v.start == base) {
                CHECK(v.end == base + 1 * P && v.offset == 1 * P, "segment A is not at base, file offset 1 page");
            } else if (v.start == base + 3 * P) {
                CHECK(v.end == base + 5 * P && v.offset == 4 * P, "segment B is not at base + 3 pages, offset 4");
            } else {
                CHECK(false, "a file VMA the segments do not account for");
            }
        } else {
            ++n_none;
            CHECK(v.perms == "---p", "a hole of the reservation is not PROT_NONE and private");
            CHECK(v.file.empty(), "a hole of the reservation is file backed");
        }
    }
    CHECK(n_file == 2, "expected exactly two file segments");
    CHECK(n_none >= 1, "no PROT_NONE hole between the segments");

    unified_cache_raw_unmap_file(&m);
    for (const vma & v : vmas_in(base, base + 6 * P)) {
        CHECK(v.file.find(f.path) == std::string::npos, "a file segment is still mapped after release");
    }
    return 0;
}

// The mapping dups the loader's fd: the caller keeps its own, the mapping's is closed at release, and the mapping
// does not depend on the caller's.
static int test_dup_fd_ownership() {
    fixture        f(4);
    const uint64_t P = g_page;
    file_mapping   m;
    const int      before = open_fd_count();
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, 2 * P
    },
                                     { { 0, 2 * P } }, &m) == file_map_status::OK,
          "map");
    CHECK(m.fd >= 0 && m.fd != f.fd, "the mapping does not hold its own dup");
    CHECK(open_fd_count() == before + 1, "mapping did not open exactly one fd");

    const int dup_fd = m.fd;
    unified_cache_raw_unmap_file(&m);
    CHECK(!fd_is_open(dup_fd), "the dup fd is still open after release");
    CHECK(fd_is_open(f.fd), "release closed the caller's own fd");
    CHECK(open_fd_count() == before, "an fd leaked");
    CHECK(m.base == nullptr && m.fd == -1, "release left the mapping non-empty");

    // closing the caller's fd while mapped does not break the mapping or its fd
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, 2 * P
    },
                                     { { 0, 2 * P } }, &m) == file_map_status::OK,
          "map");
    close(f.fd);
    f.fd = -1;
    CHECK(((const unsigned char *) m.base)[5] == pattern(5), "mapping unreadable after the caller closed its fd");
    CHECK(fd_is_open(m.fd), "the mapping's dup died with the caller's fd");
    unified_cache_raw_unmap_file(&m);
    return 0;
}

static int test_size_mismatch_refused() {
    fixture        f(4);
    const uint64_t P = g_page;
    file_mapping   m;
    const int      before  = open_fd_count();
    const uint64_t munmaps = unified_cache_file_mapping_munmap_count_for_test();

    CHECK(unified_cache_raw_map_file(f.fd, f.size + 1,
                                     {
                                         0, 2 * P
    },
                                     { { 0, 2 * P } }, &m) == file_map_status::SIZE_MISMATCH,
          "a larger recorded size was accepted");
    CHECK(unified_cache_raw_map_file(f.fd, f.size - 1,
                                     {
                                         0, 2 * P
    },
                                     { { 0, 2 * P } }, &m) == file_map_status::SIZE_MISMATCH,
          "a smaller recorded size was accepted");
    CHECK(m.base == nullptr && m.fd == -1, "a refused map wrote its output");
    CHECK(open_fd_count() == before, "a refused map leaked an fd");
    CHECK(unified_cache_file_mapping_munmap_count_for_test() == munmaps, "a refusal reached the reservation");
    return 0;
}

static int test_argument_refusals() {
    fixture        f(8);
    const uint64_t P = g_page;
    file_mapping   m;
    const int      before = open_fd_count();
    using R               = file_map_range;
    using S               = std::vector<R>;
    const auto refused    = [&](const R & reservation, const S & segs) {
        file_mapping out;
        const bool   ok =
            unified_cache_raw_map_file(f.fd, f.size, reservation, segs, &out) == file_map_status::INVALID_ARGUMENT &&
            out.base == nullptr;
        return ok;
    };
    CHECK(refused({ 0, 4 * P }, {}), "no segments accepted");
    CHECK(refused(
              {
                  0, 0
    },
              { { 0, P } }),
          "an empty reservation accepted");
    CHECK(refused(
              {
                  0, 4 * P
    },
              { { P / 2, P } }),
          "an unaligned segment start accepted");
    CHECK(refused(
              {
                  P, 2 * P
    },
              { { 0, P } }),
          "a segment before the reservation accepted");
    CHECK(refused(
              {
                  0, 2 * P
    },
              { { P, 2 * P } }),
          "a segment past the reservation accepted");
    CHECK(refused(
              {
                  0, 4 * P
    },
              { { 0, 2 * P }, { P, 2 * P } }),
          "overlapping segments accepted");
    CHECK(refused(
              {
                  0, 4 * P
    },
              { { 0, 0 } }),
          "an empty segment accepted");
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, 4 * P
    },
                                     { { 0, P } }, nullptr) == file_map_status::INVALID_ARGUMENT,
          "a null output accepted");
    CHECK(unified_cache_raw_map_file(-1, f.size,
                                     {
                                         0, 4 * P
    },
                                     { { 0, P } }, &m) == file_map_status::INVALID_ARGUMENT,
          "a bad fd accepted");
    CHECK(open_fd_count() == before, "a refused map leaked an fd");
    return 0;
}

// The release is ONE munmap of the whole reservation, however many segments it holds.
static int test_munmap_counter() {
    fixture        f(8);
    const uint64_t P = g_page;
    file_mapping   m;
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, 8 * P
    },
                                     { { 0, P }, { 2 * P, P }, { 4 * P, P }, { 6 * P, P } }, &m) == file_map_status::OK,
          "map");
    const uint64_t before = unified_cache_file_mapping_munmap_count_for_test();
    unified_cache_raw_unmap_file(&m);
    CHECK(unified_cache_file_mapping_munmap_count_for_test() == before + 1,
          "release did not issue exactly one munmap for a four-segment reservation");

    // idempotent: an already released mapping issues nothing
    unified_cache_raw_unmap_file(&m);
    CHECK(unified_cache_file_mapping_munmap_count_for_test() == before + 1, "a second release issued an munmap");

    // a one-segment reservation too
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, P
    },
                                     { { 0, P } }, &m) == file_map_status::OK,
          "map");
    const uint64_t one = unified_cache_file_mapping_munmap_count_for_test();
    unified_cache_raw_unmap_file(&m);
    CHECK(unified_cache_file_mapping_munmap_count_for_test() == one + 1, "single-segment release is not one munmap");
    return 0;
}

// A segment that cannot be mapped unwinds the reservation: nothing stays mapped, no fd stays open, and the
// output is untouched. A write-only descriptor makes the first PROT_READ segment fail (EACCES) after the dup
// and the PROT_NONE reservation exist, so the unwind has something to release.
static int test_segment_map_failure_unwinds() {
    fixture        f(8);
    const uint64_t P  = g_page;
    const int      wo = open(f.path.c_str(), O_WRONLY);
    CHECK(wo >= 0, "open write-only");

    file_mapping          m;
    const int             fds_before     = open_fd_count();
    const uint64_t        munmaps_before = unified_cache_file_mapping_munmap_count_for_test();
    const file_map_status st             = unified_cache_raw_map_file(wo, f.size,
                                                                      {
                                                              0, 8 * P
    },
                                                                      { { 0, P }, { 4 * P, P } }, &m);
    CHECK(st == file_map_status::SEGMENT_MAP_FAILED, "a write-only descriptor did not fail the segment map");
    CHECK(m.base == nullptr && m.fd == -1, "a failed map wrote its output");
    CHECK(open_fd_count() == fds_before, "the failed map leaked its dup of the fd");
    CHECK(unified_cache_file_mapping_munmap_count_for_test() == munmaps_before + 1,
          "the failed map did not unmap its reservation exactly once");
    close(wo);
    return 0;
}

// The ranged lock locks the segments it is given and nothing else; a range outside the reservation or off a page
// is refused before any mlock. Skips (and says so) where RLIMIT_MEMLOCK is too small to lock.
static int test_ranged_lock() {
    fixture        f(8);
    const uint64_t P = g_page;
    file_mapping   m;
    CHECK(unified_cache_raw_map_file(f.fd, f.size,
                                     {
                                         0, 8 * P
    },
                                     { { 0, P }, { 4 * P, 2 * P } }, &m) == file_map_status::OK,
          "map");
    CHECK(unified_cache_raw_lock_file_segments(m,
                                               {
                                                   { P / 2, P }
    }) == file_map_status::INVALID_ARGUMENT,
          "an unaligned lock segment accepted");
    CHECK(unified_cache_raw_lock_file_segments(m,
                                               {
                                                   { 0, 9 * P }
    }) == file_map_status::INVALID_ARGUMENT,
          "a lock segment past the reservation accepted");

    const file_map_status st = unified_cache_raw_lock_file_segments(m, {
                                                                           { 4 * P, 2 * P }
    });
    if (st == file_map_status::LOCK_FAILED) {
        std::printf("SKIP the lock arm: mlock refused (RLIMIT_MEMLOCK), errno %d\n", errno);
    } else {
        CHECK(st == file_map_status::OK, "the lock was refused for a reason other than the memlock limit");
        long locked_b = -1;
        long locked_a = -1;
        for (const vma & v : vmas_in((uintptr_t) m.base, (uintptr_t) m.base + m.length)) {
            if (v.start == (uintptr_t) m.base + 4 * P) {
                locked_b = locked_kb(v.start);
            } else if (v.start == (uintptr_t) m.base) {
                locked_a = locked_kb(v.start);
            }
        }
        CHECK(locked_b == (long) (2 * P / 1024), "the locked segment is not locked in full");
        CHECK(locked_a == 0, "a segment that was not in the lock set is locked");
    }
    unified_cache_raw_unmap_file(&m);
    return 0;
}

int main() {
    g_page = (uint64_t) sysconf(_SC_PAGESIZE);
    int rc = 0;
    rc |= test_reservation_shape();
    rc |= test_dup_fd_ownership();
    rc |= test_size_mismatch_refused();
    rc |= test_argument_refusals();
    rc |= test_munmap_counter();
    rc |= test_segment_map_failure_unwinds();
    rc |= test_ranged_lock();
    if (rc == 0) {
        std::printf("PASS: test-sycl-file-mapping-host\n");
    }
    return rc;
}
