// The loader's ranged lock (llama.cpp-5efe C3, docs/design/sycl-mmap-host-tier.md §4.6).
//
// load_all_data used to lock each tensor with grow_to(weight->offs + n_size), which locks contiguously from the
// mapping's base, so a tensor after a lazy (deferred) range locked the range too and faulted all of it in. The fix is
// llama_mlock::lock_range(offs, size, deferred): the tensor's own range, outward to pages except inward at an end whose
// page holds a deferred byte.
//
// Fixture: a 64 MiB file under the build directory (not tmpfs): tensor A (16 MiB), a lazy range (24 MiB), tensor B
// (16 MiB), every boundary 32 B aligned and not page aligned. The lazy range sits BEFORE the locked tensor B, which is
// the layout DeepSeek's shards have. The page cache of the FILE is read with cachestat(2) (mincore over a test-owned
// mapping where that is unavailable), never mincore over the lock's own mapping.
//
//   * lock_range(A) then lock_range(B): the lazy interior reads 0 resident pages, and the tensors read resident
//   * the mutant, grow_to(B_end) (the whole-prefix lock): the interior reads resident, so the test can fail
//
// Exits 77 if RLIMIT_MEMLOCK is below the fixture size or the directory cannot be evicted; a fail outranks a skip.

#include "../src/llama-mmap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef RESIDENCY_FIXTURE_DIR
#    define RESIDENCY_FIXTURE_DIR "."
#endif
#ifndef __NR_cachestat
#    define __NR_cachestat 451
#endif

static const uint64_t MiB = 1024ull * 1024;

static const uint64_t A_BEGIN    = 32;
static const uint64_t A_END      = A_BEGIN + 16 * MiB + 96;
static const uint64_t LAZY_BEGIN = A_END;
static const uint64_t LAZY_END   = LAZY_BEGIN + 24 * MiB + 64;
static const uint64_t B_BEGIN    = LAZY_END;
static const uint64_t B_END      = B_BEGIN + 16 * MiB;
static const uint64_t FILE_SIZE  = 64 * MiB;
static const uint64_t GUARD      = 1 * MiB;

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

static uint64_t g_page = 0;

// Resident pages of [off, off + len) of the FILE; -1 if no probe works.
static int64_t resident_pages(int fd, uint64_t off, uint64_t len) {
    cs_range r  = { off, len };
    cs_stat  cs = {};
    if (syscall(__NR_cachestat, fd, &r, &cs, 0) == 0) {
        return (int64_t) cs.nr_cache;
    }
    if (errno != ENOSYS) {
        return -1;
    }
    void * p = mmap(nullptr, FILE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
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

static bool evict(int fd) {
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    const int64_t left = resident_pages(fd, 0, FILE_SIZE);
    return left >= 0 && left <= 64;
}

struct run_result {
    int64_t interior = -1;
    int64_t a_mid    = -1;
    int64_t b_mid    = -1;
    bool    evicted  = false;
    bool    locked   = false;
};

// Maps the file, locks with `lock`, and reads back the residency of the file's pages. The mapping and the lock are
// gone before the next arm evicts.
template <typename F> static run_result run(int fd, F lock) {
    run_result res;
    res.evicted = evict(fd);
    if (!res.evicted) {
        return res;
    }
    void * addr = mmap(nullptr, FILE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        return res;
    }
    {
        llama_mlock m;
        m.init(addr);
        lock(m);
        res.locked   = true;
        res.interior = resident_pages(fd, LAZY_BEGIN + GUARD, (LAZY_END - GUARD) - (LAZY_BEGIN + GUARD));
        res.a_mid    = resident_pages(fd, A_BEGIN + 4 * MiB, 1 * MiB);
        res.b_mid    = resident_pages(fd, B_BEGIN + 4 * MiB, 1 * MiB);
    }
    munmap(addr, FILE_SIZE);
    return res;
}

int main() {
    g_page = (uint64_t) sysconf(_SC_PAGESIZE);

    struct rlimit rl = {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < FILE_SIZE)) {
        std::fprintf(stderr, "SKIP: RLIMIT_MEMLOCK is below the fixture size\n");
        return 77;
    }

    std::string       path = std::string(RESIDENCY_FIXTURE_DIR) + "/llama-mlock-ranged-XXXXXX";
    std::vector<char> tmpl(path.begin(), path.end());
    tmpl.push_back('\0');
    const int w = mkstemp(tmpl.data());
    if (w < 0) {
        std::fprintf(stderr, "SKIP: cannot create the fixture under %s\n", RESIDENCY_FIXTURE_DIR);
        return 77;
    }
    path = tmpl.data();
    std::vector<unsigned char> bytes(1 * MiB, 0x5a);
    for (uint64_t off = 0; off < FILE_SIZE; off += bytes.size()) {
        if (write(w, bytes.data(), bytes.size()) != (ssize_t) bytes.size()) {
            std::fprintf(stderr, "FAIL: cannot write the fixture\n");
            close(w);
            unlink(path.c_str());
            return 1;
        }
    }
    fsync(w);
    close(w);
    const int fd = open(path.c_str(), O_RDONLY);

    const llama_mmap::ranges deferred = {
        { LAZY_BEGIN, LAZY_END }
    };
    int fail = 0;
    int skip = 0;

    // the fix: each tensor's own range, with the lazy range before B
    const run_result ranged = run(fd, [&](llama_mlock & m) {
        m.lock_range(A_BEGIN, A_END - A_BEGIN, deferred);
        m.lock_range(B_BEGIN, B_END - B_BEGIN, deferred);
    });
    if (!ranged.evicted) {
        std::fprintf(stderr, "SKIP: the fixture directory cannot be evicted\n");
        ++skip;
    } else if (!ranged.locked || ranged.interior < 0) {
        std::fprintf(stderr, "FAIL: the ranged arm could not run\n");
        ++fail;
    } else {
        std::printf("ranged: interior %lld pages resident (want 0), A mid %lld, B mid %lld (want 256 each)\n",
                    (long long) ranged.interior, (long long) ranged.a_mid, (long long) ranged.b_mid);
        if (ranged.interior != 0) {
            std::fprintf(stderr, "FAIL: the ranged lock faulted the lazy interior in\n");
            ++fail;
        }
        if (ranged.a_mid < 256 || ranged.b_mid < 256) {
            std::fprintf(stderr, "FAIL: the ranged lock did not lock the tensors\n");
            ++fail;
        }
    }

    // the mutant: the whole-prefix lock, which must read the interior resident
    const run_result prefix = run(fd, [&](llama_mlock & m) { m.grow_to(B_END); });
    if (prefix.evicted && prefix.locked) {
        std::printf("prefix mutant: interior %lld pages resident (want > 0)\n", (long long) prefix.interior);
        if (prefix.interior <= 0) {
            std::fprintf(stderr, "FAIL: the whole-prefix mutant did not populate the interior: the test cannot fail\n");
            ++fail;
        }
    } else {
        std::fprintf(stderr, "SKIP: the prefix mutant could not run\n");
        ++skip;
    }

    close(fd);
    unlink(path.c_str());
    if (fail != 0) {
        return 1;
    }
    if (skip != 0) {
        return 77;
    }
    std::printf("PASS: test-llama-mlock-ranged\n");
    return 0;
}
