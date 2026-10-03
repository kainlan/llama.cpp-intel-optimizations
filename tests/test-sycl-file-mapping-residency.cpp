// The residency gate of the FILE_MAPPED tier (llama.cpp-5efe C3, docs/design/sycl-mmap-host-tier.md §4.6).
//
// The property: a lazy (deferred) range of a model file is never read into the page cache by the tier's mapping of
// its neighbours. It is measured as the page-cache residency of the FILE (cachestat(2), or mincore over a test-owned
// mapping), never as mincore over the tier's own reservation, whose hole is PROT_NONE anonymous memory and reads
// "not resident" whatever the file's pages are doing.
//
// Fixture: a 64 MiB file under the build directory (not tmpfs): tensor A (16 MiB), a lazy range (24 MiB), tensor B
// (16 MiB), every boundary 32 B aligned and deliberately not page aligned, so each lazy edge has a shared page. The
// fd carries POSIX_FADV_SEQUENTIAL, as the loader's does, so the read-around reach matches production.
//
// Every arm evicts first (unmaps, POSIX_FADV_DONTNEED over the file, probe ~0) and reports SKIP if it cannot.
// Arms 1, 2, 5 and 6 must read 0 resident bytes of the hole; arm 3, the mutants and the control twins must read it
// resident, which shows each of them could fail. Arm 7 requires WILLNEED to leave >= 95% of each host segment resident
// (production), and shows the one-call and 1 MiB-chunk advice reading short of it. Arm 4 (a synchronous pread) runs last.
//
// Exit 0 pass, 1 fail, 77 skipped (a fail outranks a skip).

#include "../ggml/src/ggml-sycl/file-mapping.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl;

#ifndef __NR_cachestat
#    define __NR_cachestat 451
#endif

namespace {

constexpr uint64_t MiB = 1024ull * 1024;

uint64_t g_page = 0;

// the fixture directory: argv[1], the build directory under ctest (never tmpfs)
const char * g_fixture_dir = ".";

// ---- results ----

int g_fail = 0;
int g_skip = 0;

void report_fail(const char * arm, const char * why) {
    std::fprintf(stderr, "FAIL: %s: %s\n", arm, why);
    ++g_fail;
}

void report_skip(const char * arm, const char * why) {
    std::fprintf(stderr, "SKIP: %s: %s\n", arm, why);
    ++g_skip;
}

// ---- the fixture ----

constexpr uint64_t A_BEGIN    = 32;
constexpr uint64_t A_END      = A_BEGIN + 16 * MiB + 96;  // 16 MiB + 128
constexpr uint64_t LAZY_BEGIN = A_END;
constexpr uint64_t LAZY_END   = LAZY_BEGIN + 24 * MiB + 64;
constexpr uint64_t B_BEGIN    = LAZY_END;
constexpr uint64_t B_END      = B_BEGIN + 16 * MiB;
constexpr uint64_t FILE_SIZE  = 64 * MiB;
constexpr uint64_t GUARD      = 1 * MiB;

unsigned char pattern(uint64_t i) {
    return (unsigned char) (i * 131 + 7);
}

struct fixture {
    std::string path;
    int         fd               = -1;
    uint64_t    read_ahead_bytes = 0;  // the bdi's read_ahead_kb, in bytes; 0 if unknown

    fixture()                            = default;
    fixture(const fixture &)             = delete;
    fixture & operator=(const fixture &) = delete;

    bool create() {
        path = std::string(g_fixture_dir) + "/file-mapping-residency-XXXXXX";
        std::vector<char> tmpl(path.begin(), path.end());
        tmpl.push_back('\0');
        const int w = mkstemp(tmpl.data());
        if (w < 0) {
            std::perror("mkstemp");
            return false;
        }
        path = tmpl.data();
        std::vector<unsigned char> bytes(1 * MiB);
        for (uint64_t off = 0; off < FILE_SIZE; off += bytes.size()) {
            for (uint64_t i = 0; i < bytes.size(); ++i) {
                bytes[i] = pattern(off + i);
            }
            if (write(w, bytes.data(), bytes.size()) != (ssize_t) bytes.size()) {
                std::perror("write");
                close(w);
                return false;
            }
        }
        fsync(w);  // POSIX_FADV_DONTNEED leaves dirty pages alone
        close(w);
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::perror("open");
            return false;
        }
        // as llama_mmap does (src/llama-mmap.cpp), so the read-around reach matches production
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        read_ahead_bytes = lookup_read_ahead();
        return true;
    }

    // The read_ahead_kb of the bdi of the file's st_dev. A partition's bdi is its disk's.
    uint64_t lookup_read_ahead() const {
        struct stat st = {};
        if (fstat(fd, &st) != 0) {
            return 0;
        }
        char dev[64];
        std::snprintf(dev, sizeof(dev), "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
        char real[4096];
        if (realpath(dev, real) == nullptr) {
            return 0;
        }
        std::string dir = real;
        if (access((dir + "/partition").c_str(), F_OK) == 0) {
            dir = dir.substr(0, dir.rfind('/'));
        }
        std::ifstream in(dir + "/bdi/read_ahead_kb");
        uint64_t      kb = 0;
        in >> kb;
        return kb * 1024;
    }

    ~fixture() {
        if (fd >= 0) {
            close(fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

// ---- the probe ----

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

// Resident pages of [off, off + len) of the FILE; -1 if neither probe works.
int64_t resident_pages(const fixture & f, uint64_t off, uint64_t len) {
    if (len == 0) {
        return 0;
    }
    cs_range r  = { off, len };
    cs_stat  cs = {};
    if (syscall(__NR_cachestat, f.fd, &r, &cs, 0) == 0) {
        return (int64_t) cs.nr_cache;
    }
    if (errno != ENOSYS) {
        return -1;
    }
    // mincore over a test-owned, separate PROT_READ mapping: it reports the page cache of those file pages whichever
    // mapping faulted them
    void * p = mmap(nullptr, FILE_SIZE, PROT_READ, MAP_SHARED, f.fd, 0);
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
    munmap(p, FILE_SIZE);
    return n;
}

// Evicts the whole file. The caller has unmapped every mapping (POSIX_FADV_DONTNEED skips mapped pages). False if the
// directory cannot be evicted (tmpfs, or a filesystem that ignores the advice).
int64_t interior_resident(const fixture & f);

bool evict(const fixture & f) {
    // Readahead the previous arm started can still be landing when its mapping is gone, so the advice is repeated for a
    // few seconds before the file is declared un-evictable.
    for (int i = 0; i < 10; ++i) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_DONTNEED);
        const int64_t left = resident_pages(f, 0, FILE_SIZE);
        // A few pages of metadata-adjacent noise are not the file's data, but they must not sit where an arm measures:
        // the interior is read 0 here, so a later "> 0" is the arm's doing, and the edge arms read their own band
        // before the fault for the same reason.
        if (left >= 0 && left <= 64 && interior_resident(f) == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

// ---- a tier mapping of the fixture ----

struct tier_mapping {
    file_mapping m;

    ~tier_mapping() { unified_cache_raw_unmap_file(&m); }
};

std::vector<file_tensor_range> fixture_tensors() {
    return {
        { A_BEGIN, A_END - A_BEGIN, true },
        { B_BEGIN, B_END - B_BEGIN, true }
    };
}

std::vector<file_map_range> fixture_lazy() {
    return {
        { LAZY_BEGIN, LAZY_END - LAZY_BEGIN }
    };
}

file_map_range reservation() {
    return { A_BEGIN / g_page * g_page, (B_END + g_page - 1) / g_page * g_page - A_BEGIN / g_page * g_page };
}

bool map_plan(const fixture & f, const file_segment_plan & plan, tier_mapping * out) {
    return unified_cache_raw_map_file(f.fd, FILE_SIZE, reservation(), plan.map, &out->m) == file_map_status::OK;
}

uint64_t window_for(const fixture & f) {
    return file_mapping_random_window(f.read_ahead_bytes, g_page);
}

// Touches every byte of [begin, end) of the file through the mapping.
uint64_t read_through(const tier_mapping & t, uint64_t begin, uint64_t end) {
    const volatile unsigned char * p   = (const volatile unsigned char *) t.m.base;
    uint64_t                       sum = 0;
    for (uint64_t i = begin; i < end; ++i) {
        sum += p[i - t.m.file_offset];
    }
    return sum;
}

bool byte_is_the_files(const tier_mapping & t, uint64_t off) {
    return ((const unsigned char *) t.m.base)[off - t.m.file_offset] == pattern(off);
}

constexpr uint64_t INTERIOR_BEGIN = LAZY_BEGIN + GUARD;
constexpr uint64_t INTERIOR_END   = LAZY_END - GUARD;

int64_t interior_resident(const fixture & f) {
    return resident_pages(f, INTERIOR_BEGIN, INTERIOR_END - INTERIOR_BEGIN);
}

bool advised(const char * arm, file_map_status st) {
    if (st != file_map_status::OK) {
        std::fprintf(stderr, "FAIL: %s: the advice was refused (%s), so the arm would pass without running it\n", arm,
                     file_map_status_name(st));
        ++g_fail;
        return false;
    }
    return true;
}

// Waits for readahead the kernel started to land, so a "reads 0" is not a probe taken before the pages arrive.
void settle() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
}

// Polls until `probe` reads above zero, for at most 10 s. For the arms that must show residency.
template <typename F> int64_t poll_resident(F probe) {
    int64_t n = 0;
    for (int i = 0; i < 20; ++i) {
        n = probe();
        if (n > 0) {
            return n;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return n;
}

// ---- the edge bands ----

struct band {
    uint64_t begin;
    uint64_t end;
    bool     usable;  // at least two pages past the shared page
};

// [shared page + 1 page, edge + reach) inward from the lazy start, where reach is read_ahead (the fd's SEQUENTIAL
// doubles the window, the centred read-around halves it).
band forward_band(const fixture & f) {
    const uint64_t reach  = f.read_ahead_bytes;
    const uint64_t shared = LAZY_BEGIN / g_page * g_page;
    band           b      = { shared + g_page, LAZY_BEGIN + reach, false };
    b.usable              = reach != 0 && b.end > b.begin && (b.end - b.begin) >= 2 * g_page && b.end <= INTERIOR_BEGIN;
    return b;
}

// Mirrored at the lazy end: [edge - reach, shared page - 1 page).
band backward_band(const fixture & f) {
    const uint64_t reach  = f.read_ahead_bytes;
    const uint64_t shared = LAZY_END / g_page * g_page;
    band           b      = { LAZY_END - reach, shared - g_page, false };
    b.usable              = reach != 0 && b.end > b.begin && (b.end - b.begin) >= 2 * g_page && b.begin >= INTERIOR_END;
    return b;
}

int64_t band_resident(const fixture & f, const band & b) {
    return resident_pages(f, b.begin, b.end - b.begin);
}

// ---- arms ----

// Arm 1, the property: map with the hole, advise, read every byte of A and B (including the bytes that share a page
// with the hole), and the interior reads 0 resident.
void arm1_property(const fixture & f) {
    const char * arm = "arm 1 (the property)";
    if (!evict(f)) {
        report_skip(
            arm, "the fixture directory cannot be evicted (tmpfs, or a filesystem that ignores POSIX_FADV_DONTNEED)");
        return;
    }
    const file_segment_plan plan = file_mapping_segments(fixture_tensors(), fixture_lazy(), g_page, window_for(f));
    tier_mapping            t;
    if (!plan.ok() || !map_plan(f, plan, &t)) {
        report_fail(arm, "the plan or the mapping was refused");
        return;
    }
    if (!advised(arm, unified_cache_raw_advise_file_segments(t.m, plan.random_windows, file_advice::RANDOM)) ||
        !advised(arm, unified_cache_raw_advise_file_segments(t.m, plan.willneed, file_advice::WILLNEED))) {
        return;
    }
    read_through(t, A_BEGIN, A_END);
    read_through(t, B_BEGIN, B_END);
    if (!byte_is_the_files(t, A_END - 1) || !byte_is_the_files(t, B_BEGIN)) {
        report_fail(arm, "a byte sharing a page with the hole did not read back as the file's");
        return;
    }
    settle();
    const int64_t n = interior_resident(f);
    std::printf("arm 1: interior resident pages = %lld (want 0)\n", (long long) n);
    if (n != 0) {
        report_fail(arm, "the lazy interior is resident");
    }
}

// Arm 2, the mlock case: arm 1 under the lock path, the lazy range BEFORE a locked tensor. Its mutant maps the hole
// and locks the prefix [0, B_end), and must read the interior resident.
void arm2_mlock(const fixture & f) {
    const char *  arm = "arm 2 (mlock)";
    struct rlimit rl  = {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < FILE_SIZE)) {
        report_skip(arm, "RLIMIT_MEMLOCK is below the fixture size");
        return;
    }
    if (!evict(f)) {
        report_skip(arm, "the fixture directory cannot be evicted");
        return;
    }
    {
        const file_segment_plan plan = file_mapping_segments(fixture_tensors(), fixture_lazy(), g_page, window_for(f));
        tier_mapping            t;
        if (!plan.ok() || !map_plan(f, plan, &t)) {
            report_fail(arm, "the plan or the mapping was refused");
            return;
        }
        if (!advised(arm, unified_cache_raw_advise_file_segments(t.m, plan.random_windows, file_advice::RANDOM))) {
            return;
        }
        if (unified_cache_raw_lock_file_segments(t.m, plan.willneed) != file_map_status::OK) {
            report_skip(arm, "mlock was refused (RLIMIT_MEMLOCK or a container limit)");
            return;
        }
        read_through(t, A_BEGIN, A_END);
        read_through(t, B_BEGIN, B_END);
        settle();
        const int64_t n = interior_resident(f);
        std::printf("arm 2: interior resident pages under the lock = %lld (want 0)\n", (long long) n);
        if (n != 0) {
            report_fail(arm, "the lazy interior is resident under the lock");
        }
    }
    // the mutant: the hole mapped as part of one segment and the whole prefix locked
    if (!evict(f)) {
        report_skip(arm, "the fixture directory cannot be evicted before the mutant");
        return;
    }
    {
        file_segment_plan plan;
        plan.map = {
            { A_BEGIN / g_page * g_page, B_END / g_page * g_page - A_BEGIN / g_page * g_page }
        };
        std::vector<file_map_range> lock = {
            { A_BEGIN / g_page * g_page, B_END / g_page * g_page - A_BEGIN / g_page * g_page }
        };
        tier_mapping t;
        if (!map_plan(f, plan, &t) || unified_cache_raw_lock_file_segments(t.m, lock) != file_map_status::OK) {
            report_skip(arm, "the prefix-lock mutant could not run");
            return;
        }
        const int64_t n = interior_resident(f);
        std::printf("arm 2 mutant: interior resident pages under the prefix lock = %lld (want > 0)\n", (long long) n);
        if (n <= 0) {
            report_fail(arm, "the prefix-lock mutant did not populate the interior, so the arm cannot fail");
        }
    }
}

// Arm 3, the negative control: the hole passed as a TENSOR range (host-executed, not lazy), mapped and advised
// WILLNEED. The interior must read resident, or the gate could not fail.
void arm3_negative_control(const fixture & f) {
    const char * arm = "arm 3 (negative control)";
    if (!evict(f)) {
        report_skip(arm, "the fixture directory cannot be evicted");
        return;
    }
    std::vector<file_tensor_range> tensors = fixture_tensors();
    tensors.push_back({ LAZY_BEGIN, LAZY_END - LAZY_BEGIN, true });
    const file_segment_plan plan = file_mapping_segments(tensors, {}, g_page, window_for(f));
    tier_mapping            t;
    if (!plan.ok() || !map_plan(f, plan, &t)) {
        report_fail(arm, "the plan or the mapping was refused");
        return;
    }
    if (!advised(arm, unified_cache_raw_advise_file_segments(t.m, plan.willneed, file_advice::WILLNEED))) {
        return;
    }
    const int64_t n = poll_resident([&] { return interior_resident(f); });
    std::printf("arm 3: interior resident pages with the hole as a tensor = %lld (want > 0)\n", (long long) n);
    if (n <= 0) {
        report_fail(arm, "the hole, advised WILLNEED as a tensor, did not become resident: the gate cannot fail");
    }
}

// Arms 5 and 6: fault the real mapped page adjacent to the hole (the shared page), and measure the edge band. With
// the edge MADV_RANDOM windows the band reads 0; the control twin, with no edge advice, must read it resident.
// `tail_only` is the mutant that advises only the windows before the lazy range.
enum class edge_advice { WINDOWS, NONE, TAIL_ONLY };

// Why a measurement did not happen. Only the first two are reasons to skip: a refused plan, map or advice is a
// failure of the thing under test, and must not read as "cannot evict".
enum class measure { DONE, NOT_EVICTABLE, BAND_NOT_CLEAN, REFUSED };

measure fault_and_measure(const fixture & f, bool forward, edge_advice advice, int64_t * pages) {
    const char * arm = "arms 5 and 6";
    *pages           = -1;
    if (!evict(f)) {
        return measure::NOT_EVICTABLE;
    }
    const file_segment_plan plan = file_mapping_segments(fixture_tensors(), fixture_lazy(), g_page, window_for(f));
    tier_mapping            t;
    if (!plan.ok() || !map_plan(f, plan, &t)) {
        report_fail(arm, "the plan or the mapping was refused");
        return measure::REFUSED;
    }
    std::vector<file_map_range> windows;
    for (const file_map_range & w : plan.random_windows) {
        const bool before_lazy = w.file_end() <= (LAZY_BEGIN + g_page - 1) / g_page * g_page;
        if (advice == edge_advice::WINDOWS || (advice == edge_advice::TAIL_ONLY && before_lazy)) {
            windows.push_back(w);
        }
    }
    if (!advised(arm, unified_cache_raw_advise_file_segments(t.m, windows, file_advice::RANDOM))) {
        return measure::REFUSED;
    }
    if (band_resident(f, forward ? forward_band(f) : backward_band(f)) != 0) {
        return measure::BAND_NOT_CLEAN;  // residue before the fault: a "> 0" after it would not be the fault's doing
    }
    const uint64_t at = forward ? A_END - 1 : B_BEGIN;  // a byte of the shared page
    (void) read_through(t, at, at + 1);
    settle();
    *pages = band_resident(f, forward ? forward_band(f) : backward_band(f));
    return measure::DONE;
}

// true when the measurement ran; a skip is reported here, a failure was reported where it happened
bool measured(const char * arm, measure m) {
    if (m == measure::NOT_EVICTABLE) {
        report_skip(arm, "the fixture directory cannot be evicted");
    } else if (m == measure::BAND_NOT_CLEAN) {
        report_skip(arm, "the edge band held residue before the fault");
    }
    return m == measure::DONE;
}

void arm_read_around(const fixture & f, bool forward) {
    const char * arm = forward ? "arm 5 (forward read-around)" : "arm 6 (backward read-around)";
    const band   b   = forward ? forward_band(f) : backward_band(f);
    if (!b.usable) {
        report_skip(arm, "read_ahead_kb is 0 or the reach does not extend two pages past the shared page");
        return;
    }
    int64_t twin = -1;
    if (!measured(arm, fault_and_measure(f, forward, edge_advice::NONE, &twin))) {
        return;
    }
    std::printf("%s: control twin (no edge advice) band resident pages = %lld (want > 0)\n", arm, (long long) twin);
    if (twin <= 0) {
        report_fail(arm, "the no-advice control twin did not read the band resident: the arm cannot fail");
    }
    int64_t n = -1;
    if (!measured(arm, fault_and_measure(f, forward, edge_advice::WINDOWS, &n))) {
        return;
    }
    std::printf("%s: band resident pages with the edge windows = %lld (want 0)\n", arm, (long long) n);
    if (n != 0) {
        report_fail(arm, "the edge band is resident with the MADV_RANDOM windows applied");
    }
    if (!forward) {
        // a mutant that advises only the segment tail leaves the backward edge unadvised
        int64_t tail = -1;
        if (measured(arm, fault_and_measure(f, false, edge_advice::TAIL_ONLY, &tail))) {
            std::printf("arm 6 mutant: band resident pages with only the tail advised = %lld (want > 0)\n",
                        (long long) tail);
            if (tail <= 0) {
                report_fail(arm,
                            "advising only the segment tail still read 0: the arm cannot see a missing head window");
            }
        }
    }
}

// Arm 7, the positive control of the advice itself: WILLNEED over the host segments must prefetch ALL of each, not
// the first few pages the kernel's per-call cap allows. The kernel clamps one MADV_WILLNEED to max(io_pages, ra_pages)
// of the backing device, so a call (or a chunk) above that cap reads part of a segment, and the rest is left to
// synchronous read-around faults on the CPU decode path. Arms 1 and 2 only read the interior as 0 and would pass if
// the advice never ran. The mutants advise the same segments in ONE call and in 1 MiB chunks; each must read short of
// the bar on a host whose cap is below them (if it does not, the host cannot discriminate that mutant: skip).
constexpr int64_t WILLNEED_BAR_PERCENT = 95;

enum class willneed_how { PRODUCTION, ONE_CALL, MIB_CHUNKS };

// The least resident fraction, in percent, over the segments after at most 10 s (0 when a segment never starts).
int64_t willneed_least_percent(const fixture &           f,
                               const tier_mapping &      t,
                               const file_segment_plan & plan,
                               willneed_how              how,
                               bool *                    advice_ok) {
    *advice_ok = true;
    if (how == willneed_how::PRODUCTION) {
        *advice_ok =
            advised("arm 7", unified_cache_raw_advise_file_segments(t.m, plan.willneed, file_advice::WILLNEED));
    } else {
        const uint64_t chunk = how == willneed_how::ONE_CALL ? ~0ull : 1 * MiB;
        for (const file_map_range & seg : plan.willneed) {
            for (uint64_t done = 0; done < seg.length; done += chunk) {
                const uint64_t n = std::min(chunk, seg.length - done);
                if (madvise((char *) t.m.base + (seg.file_offset - t.m.file_offset + done), (size_t) n,
                            MADV_WILLNEED) != 0) {
                    *advice_ok = false;
                }
            }
        }
    }
    int64_t least = 100;
    for (int i = 0; i < 20; ++i) {
        least = 100;
        for (const file_map_range & seg : plan.willneed) {
            const int64_t pct = 100 * resident_pages(f, seg.file_offset, seg.length) / (int64_t) (seg.length / g_page);
            least             = std::min(least, pct);
        }
        if (least >= WILLNEED_BAR_PERCENT) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return least;
}

void arm7_willneed_prefetches_each_segment(const fixture & f) {
    const char * arm = "arm 7 (WILLNEED reaches the whole segment)";
    for (willneed_how how : { willneed_how::PRODUCTION, willneed_how::ONE_CALL, willneed_how::MIB_CHUNKS }) {
        if (!evict(f)) {
            report_skip(arm, "the fixture directory cannot be evicted");
            return;
        }
        const file_segment_plan plan = file_mapping_segments(fixture_tensors(), fixture_lazy(), g_page, window_for(f));
        tier_mapping            t;
        if (!plan.ok() || plan.willneed.size() != 2 || !map_plan(f, plan, &t)) {
            report_fail(arm, "the plan or the mapping was refused");
            return;
        }
        bool          advice_ok = false;
        const int64_t least     = willneed_least_percent(f, t, plan, how, &advice_ok);
        if (!advice_ok) {
            if (how != willneed_how::PRODUCTION) {
                report_skip(arm, "a mutant's madvise was refused");
            }
            continue;
        }
        const char * name = how == willneed_how::PRODUCTION ? "production" :
                            how == willneed_how::ONE_CALL   ? "one-call mutant" :
                                                              "1 MiB-chunk mutant";
        std::printf("arm 7 %s: least segment resident = %lld%% (%s %d%%)\n", name, (long long) least,
                    how == willneed_how::PRODUCTION ? "want >=" : "want <", (int) WILLNEED_BAR_PERCENT);
        if (how == willneed_how::PRODUCTION) {
            if (least < WILLNEED_BAR_PERCENT) {
                report_fail(arm, "WILLNEED did not prefetch the whole of each host segment");
            }
        } else if (least >= WILLNEED_BAR_PERCENT) {
            report_skip(arm, "this host's per-call WILLNEED cap does not stop that mutant: it cannot discriminate");
        }
    }
}

// Arm 4, the synchronous positive control, last: pread 1 MiB inside the interior and that MiB reads resident.
void arm4_pread_control(const fixture & f) {
    const char * arm = "arm 4 (synchronous positive control)";
    if (!evict(f)) {
        report_skip(arm, "the fixture directory cannot be evicted");
        return;
    }
    const uint64_t             off = INTERIOR_BEGIN + 4 * MiB;
    std::vector<unsigned char> buf(1 * MiB);
    if (pread(f.fd, buf.data(), buf.size(), (off_t) off) != (ssize_t) buf.size()) {
        report_fail(arm, "pread failed");
        return;
    }
    const int64_t n = resident_pages(f, off, buf.size());
    std::printf("arm 4: resident pages of the pread MiB = %lld (want %llu)\n", (long long) n,
                (unsigned long long) (buf.size() / g_page));
    if (n < (int64_t) (buf.size() / g_page)) {
        report_fail(arm, "a pread did not populate the page cache: the probe cannot see residency");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1) {
        g_fixture_dir = argv[1];
    }
    g_page = (uint64_t) sysconf(_SC_PAGESIZE);
    fixture f;
    if (!f.create()) {
        std::fprintf(stderr, "SKIP: cannot create the fixture under %s\n", g_fixture_dir);
        return 77;
    }
    std::printf("fixture %s, read_ahead = %llu KiB, page = %llu\n", f.path.c_str(),
                (unsigned long long) (f.read_ahead_bytes / 1024), (unsigned long long) g_page);
    arm1_property(f);
    arm2_mlock(f);
    arm3_negative_control(f);
    arm_read_around(f, true);
    arm_read_around(f, false);
    arm7_willneed_prefetches_each_segment(f);
    arm4_pread_control(f);  // last: its pread populates the hole
    if (g_fail != 0) {
        return 1;
    }
    if (g_skip != 0) {
        return 77;
    }
    std::printf("PASS: test-sycl-file-mapping-residency\n");
    return 0;
}
