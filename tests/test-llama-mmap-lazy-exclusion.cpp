// llama_mmap leaves the record's lazy ranges cold (llama.cpp-5efe C4 part d, docs/design/sycl-mmap-host-tier.md §4.6).
//
// The loader maps every model file whole. A lazy (deferred) range of it is read on demand by someone else, so the
// mapping must not bring those bytes into the page cache: no WILLNEED/SEQUENTIAL over a page that holds a lazy byte,
// and no read-around or 2 MiB folio from a neighbouring fault that reaches into the range. The page cache of the FILE
// is read with cachestat(2) (mincore over a test-owned mapping where that is unavailable), never through the mapping
// under test. Both arms drive libllama's llama_mmap with lazy ranges taken from a llama_lazy_range_record, with no
// backend and no device.
//
// Prefetch arm. A 64 MiB fixture under the build directory (not tmpfs): a small tensor A at the head, a 24 MiB lazy
// range, then tensor B, every boundary 32 B aligned and not page aligned. The lazy range starts inside the first
// 512 KiB of the file because a WILLNEED over a long range reads only its head on current kernels (a single
// read-ahead chunk, 512 KiB here), so a mutant that prefetches from offset 0 only reaches a lazy byte if the range
// begins there. llama_mmap(prefetch = -1) over the record's ranges, polled for up to 10 s:
//   * the lazy pages (the lazy range less the two pages it shares with a neighbour) read 0 resident
//   * the two shared pages read 0 resident: WILLNEED rounds inward, so it never reaches a page holding a lazy byte
//   * the neighbours are prefetched up to the shared pages, so the zero is not "nothing was prefetched"
//   * the mutant, an EMPTY record, makes the lazy head resident, so the arm can fail
//
// HUGEPAGE arm. The lazy boundaries sit 1 MiB past a 2 MiB boundary (begin) and 1 MiB before the next one (end), so a
// 2 MiB folio read for a fault next to an edge reaches 1 MiB into the range. The test faults the page next to each
// edge through llama_mmap's own mapping (prefetch 0, the SYCL load, and prefetch -1, where the page next to the begin
// edge faults cold and a window that were SEQUENTIAL would read forward into the range):
//   * the folio band, the lazy bytes from the shared page + 1 page to the edge's 2 MiB boundary, reads 0
//   * the VMAs there carry nh and not hg
//   * the mutant re-advises the mapping's window MADV_HUGEPAGE and faults again: the band must read resident, or the
//     arm exits 77 naming why (THP never, or a kernel whose file path ignores the advice)
//
// Exits 0 on pass, 1 on a fail, 77 on a skip with the reason printed; a fail outranks a skip.

#include "../src/llama-lazy-range-record.h"
#include "../src/llama-mmap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef __NR_cachestat
#    define __NR_cachestat 451
#endif

static const uint64_t MiB = 1024ull * 1024;

static uint64_t g_page = 0;

// the fixture directory: argv[1], the build directory under ctest (never tmpfs)
static const char * g_fixture_dir = ".";

struct cs_range {
    uint64_t off;
    uint64_t len;
};

struct cs_stat {
    uint64_t nr_cache;
    uint64_t nr_dirty;
    uint64_t nr_writeback;
    uint64_t nr_evicted;
    uint64_t nr_recently_evicted;
};

// Resident pages of [off, off + len) of the FILE; -1 if no probe works.
static int64_t resident_pages(int fd, uint64_t file_size, uint64_t off, uint64_t len) {
    if (len == 0) {
        return 0;
    }
    cs_range r  = { off, len };
    cs_stat  cs = {};
    if (syscall(__NR_cachestat, fd, &r, &cs, 0) == 0) {
        return (int64_t) cs.nr_cache;
    }
    if (errno != ENOSYS) {
        return -1;
    }
    void * p = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        return -1;
    }
    const uint64_t             lo = off / g_page * g_page;
    const uint64_t             hi = (off + len + g_page - 1) / g_page * g_page;
    std::vector<unsigned char> vec((hi - lo) / g_page);
    int64_t                    n = -1;
    if (mincore((char *) p + lo, hi - lo, vec.data()) == 0) {
        n = 0;
        for (unsigned char c : vec) {
            n += c & 1;
        }
    }
    munmap(p, file_size);
    return n;
}

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// every mapping of the file must be gone: FADV_DONTNEED skips mapped pages. Large folios of the previous arm can take a
// moment to be released after the unmap, so the eviction is retried.
static bool evict(int fd, uint64_t file_size) {
    for (int attempt = 0; attempt < 20; ++attempt) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        const int64_t left = resident_pages(fd, file_size, 0, file_size);
        if (left >= 0 && left <= 64) {
            return true;
        }
        sleep_ms(100);
    }
    return false;
}

// VmFlags of the VMA that contains `p`, " "-padded, or "" when no VMA or no VmFlags line for it was found
static std::string vm_flags_at(const void * p) {
    const uintptr_t target = (uintptr_t) p;
    std::ifstream   smaps("/proc/self/smaps");
    std::string     line;
    bool            in_vma = false;
    while (std::getline(smaps, line)) {
        uintptr_t beg = 0, end = 0;
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " ", &beg, &end) == 2 &&
            line.find(':') > line.find(' ')) {
            in_vma = target >= beg && target < end;
            continue;
        }
        if (in_vma && line.rfind("VmFlags:", 0) == 0) {
            return line.substr(8) + " ";
        }
    }
    return "";
}

static bool has_flag(const std::string & flags, const char * flag) {
    return flags.find(std::string(" ") + flag + " ") != std::string::npos;
}

// the backing device's read_ahead_kb, or -1 when /sys has no entry for the file's device. A whole disk or a
// filesystem with its own bdi is /sys/class/bdi/MAJ:MIN; a partition shares its disk's, one directory up.
static long read_ahead_kb(int fd) {
    struct stat st = {};
    if (fstat(fd, &st) != 0) {
        return -1;
    }
    char paths[2][160];
    std::snprintf(paths[0], sizeof(paths[0]), "/sys/class/bdi/%u:%u/read_ahead_kb", major(st.st_dev), minor(st.st_dev));
    std::snprintf(paths[1], sizeof(paths[1]), "/sys/dev/block/%u:%u/../bdi/read_ahead_kb", major(st.st_dev),
                  minor(st.st_dev));
    for (const char * path : paths) {
        std::ifstream in(path);
        long          kb = -1;
        if (in >> kb) {
            return kb;
        }
    }
    return -1;
}

static std::string thp_mode() {
    std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string   line;
    std::getline(in, line);
    const size_t l = line.find('[');
    const size_t r = line.find(']');
    return l != std::string::npos && r != std::string::npos && r > l ? line.substr(l + 1, r - l - 1) : "unknown";
}

// Creates and fills an anonymous fixture of `size` bytes in g_fixture_dir and returns its fd, -1 on failure. The file
// has no name: O_TMPFILE where the filesystem has it, else mkstemp unlinked at once. Anything that walks the build
// directory (an indexer, a scanner) could otherwise open a named fixture mid-run and read it into the page cache,
// which fails the arm that wants it cold and satisfies the mutant that wants it warm. The mapping under test opens
// the same inode through fixture_path(), so nothing opens the file by a name it could find.
static int make_fixture(const char * stem, uint64_t size) {
    int fd = open(g_fixture_dir, O_TMPFILE | O_RDWR, 0600);
    if (fd < 0) {
        std::string       path = std::string(g_fixture_dir) + "/" + stem + "-XXXXXX";
        std::vector<char> tmpl(path.begin(), path.end());
        tmpl.push_back('\0');
        fd = mkstemp(tmpl.data());
        if (fd >= 0) {
            unlink(tmpl.data());
        }
    }
    if (fd < 0) {
        return -1;
    }
    std::vector<unsigned char> bytes(1 * MiB, 0x5a);
    for (uint64_t off = 0; off < size; off += bytes.size()) {
        if (write(fd, bytes.data(), bytes.size()) != (ssize_t) bytes.size()) {
            close(fd);
            return -1;
        }
    }
    fsync(fd);
    return fd;
}

static std::string fixture_path(int fd) {
    return "/proc/self/fd/" + std::to_string(fd);
}

// the ranges llama_mmap is given, as the loader gets them: from a filled record
static llama_mmap::ranges record_ranges(const std::vector<std::pair<uint64_t, uint64_t>> & lazy) {
    std::vector<llama_lazy_range_row> rows;
    for (size_t i = 0; i < lazy.size(); ++i) {
        llama_lazy_range_row row;
        row.name          = "lazy." + std::to_string(i);
        row.value         = { 0, (size_t) lazy[i].first, (size_t) (lazy[i].second - lazy[i].first) };
        row.lazy_eligible = true;
        row.deferral      = LLAMA_LAZY_DEFERRAL_LAZY_READ;
        row.attrs         = { 0, -1 };
        rows.push_back(row);
    }
    llama_lazy_range_record record;
    record.fill(rows);
    return record.excluded_for_file(0);
}

static uint64_t page_floor(uint64_t x) {
    return x / g_page * g_page;
}

//
// the prefetch arm
//

static const uint64_t P_A_BEGIN    = 32;
static const uint64_t P_A_END      = 192 * 1024 + 96;
static const uint64_t P_LAZY_BEGIN = P_A_END;
static const uint64_t P_LAZY_END   = P_LAZY_BEGIN + 24 * MiB + 64;
static const uint64_t P_FILE_SIZE  = 64 * MiB;

struct prefetch_result {
    bool    ran                 = false;
    int64_t interior_max        = 0;   // the lazy pages between the two shared pages, the most ever resident
    int64_t shared_begin_max    = 0;   // the page holding A's tail and the lazy range's first byte
    int64_t shared_end_max      = 0;   // the page holding the lazy range's last byte and B's head
    bool    neighbours_ready    = false;
    double  seconds_to_interior = -1;  // when the interior first read resident
};

// Maps with prefetch = -1 over `ranges` and polls the file's residency for up to 10 s. `stop_on_interior` ends the
// poll when the interior first reads resident (the mutant); otherwise the poll runs until the neighbours are
// prefetched and one more second has passed, so that read that arrives late is seen.
static prefetch_result run_prefetch(int fd, const llama_mmap::ranges & ranges, bool stop_on_interior) {
    prefetch_result res;
    if (!evict(fd, P_FILE_SIZE)) {
        return res;
    }
    const uint64_t shared_begin = page_floor(P_LAZY_BEGIN);
    const uint64_t shared_end   = page_floor(P_LAZY_END);
    const uint64_t interior_off = shared_begin + g_page;
    const uint64_t interior_len = shared_end - interior_off;
    if (resident_pages(fd, P_FILE_SIZE, interior_off, interior_len) != 0) {
        return res;
    }
    res.ran = true;
    {
        llama_file f(fixture_path(fd).c_str(), "rb");
        llama_mmap m(&f, (size_t) -1, /*numa =*/false, ranges);

        const auto t0       = std::chrono::steady_clock::now();
        int        ready_ms = -1;
        for (int ms = 0; ms < 10000; ms += 50) {
            sleep_ms(50);
            const int64_t interior = resident_pages(fd, P_FILE_SIZE, interior_off, interior_len);
            res.interior_max       = std::max(res.interior_max, interior);
            res.shared_begin_max =
                std::max(res.shared_begin_max, resident_pages(fd, P_FILE_SIZE, shared_begin, g_page));
            res.shared_end_max = std::max(res.shared_end_max, resident_pages(fd, P_FILE_SIZE, shared_end, g_page));
            if (interior > 0 && res.seconds_to_interior < 0) {
                res.seconds_to_interior = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stop_on_interior) {
                    break;
                }
            }
            // the neighbours are prefetched right up to the shared pages: the page before and the page after them
            const bool ready = resident_pages(fd, P_FILE_SIZE, shared_begin - g_page, g_page) == 1 &&
                               resident_pages(fd, P_FILE_SIZE, shared_end + g_page, g_page) == 1;
            if (ready && ready_ms < 0) {
                ready_ms = ms;
            }
            if (ready_ms >= 0 && ms - ready_ms >= 1000) {
                break;
            }
        }
        res.neighbours_ready = ready_ms >= 0;
    }
    return res;
}

static int prefetch_arm(int * skip) {
    int fail = 0;

    const int fd = make_fixture("llama-mmap-lazy-prefetch", P_FILE_SIZE);
    if (fd < 0) {
        std::fprintf(stderr, "SKIP: prefetch arm: cannot create the fixture under %s\n", g_fixture_dir);
        ++*skip;
        return 0;
    }

    const llama_mmap::ranges ranges = record_ranges({
        { P_LAZY_BEGIN, P_LAZY_END }
    });
    const prefetch_result    prod   = run_prefetch(fd, ranges, false);
    if (!prod.ran) {
        std::fprintf(stderr, "SKIP: prefetch arm: the fixture directory cannot be evicted\n");
        ++*skip;
    } else {
        std::printf(
            "prefetch arm, record ranges: interior max %lld pages resident (want 0), shared pages %lld %lld "
            "(want 0 0), neighbours prefetched %d (want 1)\n",
            (long long) prod.interior_max, (long long) prod.shared_begin_max, (long long) prod.shared_end_max,
            (int) prod.neighbours_ready);
        if (prod.interior_max != 0) {
            std::fprintf(stderr, "FAIL: prefetch arm: the mapping brought the lazy interior into the page cache\n");
            ++fail;
        }
        if (prod.shared_begin_max != 0 || prod.shared_end_max != 0) {
            std::fprintf(stderr, "FAIL: prefetch arm: WILLNEED reached a page that holds a lazy byte\n");
            ++fail;
        }
        if (!prod.neighbours_ready) {
            std::fprintf(stderr,
                         "SKIP: prefetch arm: the neighbouring tensors were not prefetched within 10 s, so a zero "
                         "interior proves nothing\n");
            ++*skip;
        }
    }

    // the mutant: an empty record. The same call must now bring the interior in, or the arm cannot fail.
    const prefetch_result mut = run_prefetch(fd, record_ranges({}), true);
    if (mut.ran) {
        std::printf("prefetch arm, empty-record mutant: interior max %lld pages resident (want > 0)%s\n",
                    (long long) mut.interior_max, mut.seconds_to_interior >= 0 ? "" : ", not within 10 s");
        if (mut.interior_max <= 0) {
            std::fprintf(stderr,
                         "SKIP: prefetch arm: the empty-record mutant did not populate the interior within 10 s, so "
                         "the arm cannot fail\n");
            ++*skip;
        }
    } else {
        std::fprintf(stderr, "SKIP: prefetch arm: the mutant could not run\n");
        ++*skip;
    }

    close(fd);
    return fail;
}

//
// the HUGEPAGE arm
//

static const uint64_t H_LAZY_BEGIN   = 9 * MiB + 64;
static const uint64_t H_LAZY_END     = 25 * MiB + 96;
static const uint64_t H_FILE_SIZE    = 36 * MiB;
// the window, as the design states it: [lazy_begin - W, lazy_end + W) with W = 2 MiB here, widened outward to 2 MiB
static const uint64_t H_WINDOW_BEGIN = 6 * MiB;
static const uint64_t H_WINDOW_END   = 28 * MiB;

struct hugepage_result {
    bool        ran           = false;
    int64_t     band_begin    = -1;  // lazy bytes from the shared page + 1 page up to the 2 MiB boundary at 10 MiB
    int64_t     band_end      = -1;  // lazy bytes from the 2 MiB boundary at 24 MiB up to the shared page
    int64_t     faulted_begin = -1;  // the pages the test read, which must be resident or the probe saw nothing
    int64_t     faulted_end   = -1;
    std::string flags_before;        // VmFlags of the page next to the begin edge
    std::string flags_after;         // VmFlags of the page next to the end edge
};

// Maps with `prefetch`, faults the page next to each edge, and reads the folio bands. `mutant` re-advises the window
// MADV_HUGEPAGE on the mapping first. At prefetch -1 the page next to the begin edge lies past the 512 KiB head that
// WILLNEED reads, so it faults cold: a window that is SEQUENTIAL instead of RANDOM would read forward into the range.
static hugepage_result run_hugepage(int fd, size_t prefetch, bool mutant) {
    hugepage_result res;
    if (!evict(fd, H_FILE_SIZE)) {
        return res;
    }
    res.ran = true;

    const uint64_t           shared_begin = page_floor(H_LAZY_BEGIN);
    const uint64_t           shared_end   = page_floor(H_LAZY_END);
    const uint64_t           fault_begin  = shared_begin - g_page;
    const uint64_t           fault_end    = shared_end + g_page;
    const llama_mmap::ranges ranges       = record_ranges({
        { H_LAZY_BEGIN, H_LAZY_END }
    });
    {
        llama_file f(fixture_path(fd).c_str(), "rb");
        llama_mmap m(&f, prefetch, /*numa =*/false, ranges);
        char *     base = (char *) m.addr();

        if (mutant) {
            if (madvise(base + H_WINDOW_BEGIN, H_WINDOW_END - H_WINDOW_BEGIN, MADV_HUGEPAGE) != 0) {
                std::fprintf(stderr, "mutant: madvise(MADV_HUGEPAGE) failed: %s\n", std::strerror(errno));
                res.ran = false;
            }
        }
        res.flags_before = vm_flags_at(base + fault_begin);
        res.flags_after  = vm_flags_at(base + fault_end);

        volatile char sink = 0;
        sink               = sink + base[fault_begin];
        sink               = sink + base[fault_end];
        (void) sink;

        res.faulted_begin = resident_pages(fd, H_FILE_SIZE, fault_begin, g_page);
        res.faulted_end   = resident_pages(fd, H_FILE_SIZE, fault_end, g_page);
        res.band_begin    = resident_pages(fd, H_FILE_SIZE, shared_begin + g_page, 10 * MiB - (shared_begin + g_page));
        res.band_end      = resident_pages(fd, H_FILE_SIZE, 24 * MiB, shared_end - 24 * MiB);
    }
    return res;
}

static int hugepage_arm(int * skip) {
    int fail = 0;

    const std::string mode = thp_mode();
    if (mode == "never") {
        std::fprintf(stderr, "SKIP: HUGEPAGE arm: transparent hugepages are [never], so no folio can be read\n");
        ++*skip;
        return 0;
    }

    const int fd = make_fixture("llama-mmap-lazy-hugepage", H_FILE_SIZE);
    if (fd < 0) {
        std::fprintf(stderr, "SKIP: HUGEPAGE arm: cannot create the fixture under %s\n", g_fixture_dir);
        ++*skip;
        return 0;
    }

    // prefetch 0 is the SYCL load, prefetch -1 the default load: the window must hold in both
    const size_t prefetches[] = { 0, (size_t) -1 };
    for (const size_t prefetch : prefetches) {
        const char *          label = prefetch == 0 ? "prefetch 0" : "prefetch -1";
        const hugepage_result prod  = run_hugepage(fd, prefetch, false);
        if (!prod.ran) {
            std::fprintf(stderr, "SKIP: HUGEPAGE arm: the fixture directory cannot be evicted\n");
            ++*skip;
            continue;
        }
        std::printf(
            "HUGEPAGE arm, THP %s, %s: folio band at the begin edge %lld pages resident, at the end edge %lld "
            "(want 0 0); the faulted pages %lld %lld (want 1 1)\n",
            mode.c_str(), label, (long long) prod.band_begin, (long long) prod.band_end, (long long) prod.faulted_begin,
            (long long) prod.faulted_end);
        std::printf("HUGEPAGE arm, %s: VmFlags next to the begin edge '%s', next to the end edge '%s'\n", label,
                    prod.flags_before.c_str(), prod.flags_after.c_str());
        if (prod.faulted_begin != 1 || prod.faulted_end != 1) {
            std::fprintf(stderr, "FAIL: HUGEPAGE arm, %s: the faulted page is not resident, so the probe saw nothing\n",
                         label);
            ++fail;
        }
        if (prod.band_begin != 0 || prod.band_end != 0) {
            std::fprintf(stderr, "FAIL: HUGEPAGE arm, %s: a fault next to a lazy edge read lazy bytes\n", label);
            ++fail;
        }
        for (const std::string * flags : { &prod.flags_before, &prod.flags_after }) {
            if (flags->empty() || !has_flag(*flags, "nh") || has_flag(*flags, "hg")) {
                std::fprintf(stderr,
                             "FAIL: HUGEPAGE arm, %s: the window next to a lazy edge is not advised NOHUGEPAGE "
                             "and not hugepage: '%s'\n",
                             label, flags->c_str());
                ++fail;
            }
        }
    }

    const hugepage_result mut = run_hugepage(fd, 0, true);
    if (mut.ran) {
        std::printf("HUGEPAGE arm, window left MADV_HUGEPAGE: folio band %lld %lld pages resident (want > 0 each)\n",
                    (long long) mut.band_begin, (long long) mut.band_end);
        if (mut.band_begin <= 0 || mut.band_end <= 0) {
            std::fprintf(stderr,
                         "SKIP: HUGEPAGE arm: with the window advised MADV_HUGEPAGE the fault did not read the folio "
                         "band (THP %s, or a kernel whose file path ignores the advice), so the arm cannot fail\n",
                         mode.c_str());
            ++*skip;
        }
    } else {
        std::fprintf(stderr, "SKIP: HUGEPAGE arm: the mutant could not run\n");
        ++*skip;
    }

    close(fd);
    return fail;
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        g_fixture_dir = argv[1];
    }
    g_page = (uint64_t) sysconf(_SC_PAGESIZE);

    int skip = 0;

    // the layout assumes W = 2 MiB, which holds while twice the backing device's read-ahead is at most 2 MiB
    const int pfd = make_fixture("llama-mmap-lazy-bdi", 1 * MiB);
    if (pfd < 0) {
        std::fprintf(stderr, "SKIP: cannot create a fixture under %s\n", g_fixture_dir);
        return 77;
    }
    const long ra = read_ahead_kb(pfd);
    close(pfd);
    std::printf("fixture directory %s, backing read_ahead_kb %ld, page %llu\n", g_fixture_dir, ra,
                (unsigned long long) g_page);
    if (ra > 1024) {
        std::fprintf(stderr, "SKIP: read_ahead_kb %ld puts the window past 2 MiB, which the fixtures assume\n", ra);
        return 77;
    }

    int fail = prefetch_arm(&skip);
    fail += hugepage_arm(&skip);

    if (fail != 0) {
        return 1;
    }
    if (skip != 0) {
        return 77;
    }
    std::printf("PASS: test-llama-mmap-lazy-exclusion\n");
    return 0;
}
