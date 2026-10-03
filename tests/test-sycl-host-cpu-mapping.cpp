// The raw wrapper of the HOST_CPU tier (llama.cpp-th32 T1, docs/design/sycl-storage-tier-lazy-tables.md §3.1).
//
// Links ggml-sycl/host-cpu-mapping.cpp alone: no SYCL header, no device, no model. It checks the one sizing function,
// the typed refusal under RLIMIT_MEMLOCK = 1 page (host-cpu-lock-failed, naming errno, the bytes and the limit, and
// nothing left mapped), the success path (locked in full, advised MADV_HUGEPAGE, writable, one munmap at release) and
// the argument and map refusals.
//
// Exits 77 where the process may lock without limit (CAP_IPC_LOCK), because then the refusal arm cannot be reached.

#include "../ggml/src/ggml-sycl/host-cpu-mapping.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

using namespace ggml_sycl;

// The build is -DNDEBUG, so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static const size_t MiB = 1024 * 1024;

// VmFlags of the VMA that contains `p`, or "" when none was found
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

static bool vma_exists(const void * p) {
    const uintptr_t target = (uintptr_t) p;
    std::ifstream   maps("/proc/self/maps");
    std::string     line;
    while (std::getline(maps, line)) {
        uintptr_t beg = 0, end = 0;
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " ", &beg, &end) == 2 && target >= beg && target < end) {
            return true;
        }
    }
    return false;
}

static int test_sizing() {
    CHECK(host_cpu_mapped_size(0) == 0, "a zero request must size to 0 (refused by the wrapper)");
    CHECK(host_cpu_mapped_size(1) == 2 * MiB, "one byte is not one huge page");
    CHECK(host_cpu_mapped_size(2 * MiB) == 2 * MiB, "exactly one huge page is not one huge page");
    CHECK(host_cpu_mapped_size(2 * MiB + 1) == 4 * MiB, "a byte past a huge page is not two");
    CHECK(host_cpu_mapped_size((size_t) 50 * 1024 * MiB + 17) % (2 * MiB) == 0, "a large size is not whole huge pages");
    CHECK(host_cpu_mapped_size(SIZE_MAX) == 0, "an overflowing request must size to 0");
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    CHECK((2 * MiB) % page == 0, "the huge page is not whole pages on this host");
    return 0;
}

static int test_argument_and_map_refusals() {
    host_cpu_mapping m;
    CHECK(unified_cache_raw_alloc_host_cpu(0, &m) == host_cpu_status::INVALID_ARGUMENT, "a zero request was accepted");
    CHECK(unified_cache_raw_alloc_host_cpu(SIZE_MAX, &m) == host_cpu_status::INVALID_ARGUMENT,
          "an overflowing request was accepted");
    CHECK(unified_cache_raw_alloc_host_cpu(4096, nullptr) == host_cpu_status::INVALID_ARGUMENT,
          "a null out was accepted");
    // the whole user address space cannot be mapped
    const host_cpu_status st = unified_cache_raw_alloc_host_cpu((size_t) 1 << 47, &m);
    CHECK(st == host_cpu_status::MMAP_FAILED, "an unmappable request was not refused as a map failure");
    CHECK(m.ptr == nullptr && m.err != 0, "a map refusal left a pointer or no errno");
    const std::string text = host_cpu_refusal_text(st, m);
    CHECK(text.find("host-cpu-map-failed") != std::string::npos, "the map refusal text does not carry its name");
    return 0;
}

static int test_lock_refusal_under_a_one_page_limit() {
    struct rlimit saved = {};
    CHECK(getrlimit(RLIMIT_MEMLOCK, &saved) == 0, "getrlimit");
    const size_t  page = (size_t) sysconf(_SC_PAGESIZE);
    struct rlimit tiny = { page, saved.rlim_max };
    if (saved.rlim_max != RLIM_INFINITY && saved.rlim_max < page) {
        std::fprintf(stderr, "SKIP: the hard RLIMIT_MEMLOCK is below one page\n");
        return 77;
    }
    CHECK(setrlimit(RLIMIT_MEMLOCK, &tiny) == 0, "could not lower RLIMIT_MEMLOCK");

    const uint64_t        munmaps_before = unified_cache_host_cpu_munmap_count_for_test();
    host_cpu_mapping      m;
    const host_cpu_status st = unified_cache_raw_alloc_host_cpu(4096, &m);
    setrlimit(RLIMIT_MEMLOCK, &saved);
    if (st == host_cpu_status::OK) {
        unified_cache_raw_free_host_cpu(m.ptr, m.mapped_bytes);
        std::fprintf(stderr, "SKIP: mlock succeeded over a one-page RLIMIT_MEMLOCK (CAP_IPC_LOCK?)\n");
        return 77;
    }
    CHECK(st == host_cpu_status::LOCK_FAILED, "the over-limit request was not refused as a lock failure");
    CHECK(m.ptr == nullptr, "a lock refusal left a pointer");
    CHECK(m.err != 0, "a lock refusal carries no errno");
    CHECK(m.requested_bytes == 4096 && m.mapped_bytes == 2 * MiB, "the refusal does not name the bytes");
    CHECK(m.rlimit_memlock == page, "the refusal does not name the limit in force");
    CHECK(unified_cache_host_cpu_munmap_count_for_test() == munmaps_before + 1,
          "a lock refusal did not unmap exactly once");
    const std::string text = host_cpu_refusal_text(st, m);
    char              want_errno[32];
    std::snprintf(want_errno, sizeof(want_errno), "errno=%d", m.err);
    CHECK(text.find("host-cpu-lock-failed") != std::string::npos, "the text does not carry the typed name");
    CHECK(text.find(want_errno) != std::string::npos, "the text does not name errno");
    CHECK(text.find("2097152") != std::string::npos && text.find("4096") != std::string::npos,
          "the text does not name the mapped and requested bytes");
    CHECK(text.find("RLIMIT_MEMLOCK=4096") != std::string::npos, "the text does not name RLIMIT_MEMLOCK");
    return 0;
}

static int test_success_is_locked_hugepage_advised_and_released_once() {
    struct rlimit rl = {};
    CHECK(getrlimit(RLIMIT_MEMLOCK, &rl) == 0, "getrlimit");
    if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < 2 * MiB) {
        std::fprintf(stderr, "SKIP: RLIMIT_MEMLOCK is below one huge page\n");
        return 77;
    }
    const uint64_t   munmaps_before = unified_cache_host_cpu_munmap_count_for_test();
    host_cpu_mapping m;
    CHECK(unified_cache_raw_alloc_host_cpu(4096, &m) == host_cpu_status::OK, "a valid request was refused");
    CHECK(m.ptr != nullptr && m.mapped_bytes == 2 * MiB && m.requested_bytes == 4096, "the mapping is not as sized");
    CHECK(m.huge_advised, "MADV_HUGEPAGE was refused");

    // writable, and zero-filled as anonymous memory is
    unsigned char * p = (unsigned char *) m.ptr;
    CHECK(p[0] == 0 && p[m.mapped_bytes - 1] == 0, "anonymous memory is not zero");
    std::memset(p, 0xa5, m.mapped_bytes);

    // the kernel's own record: locked (lo) and huge-page advised (hg)
    const std::string flags = vm_flags_at(m.ptr);
    CHECK(!flags.empty(), "no VmFlags line for the mapping");
    CHECK(flags.find(" lo ") != std::string::npos, "the mapping is not VM_LOCKED");
    std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string   thp_mode;
    std::getline(thp, thp_mode);
    if (thp_mode.find("[never]") == std::string::npos) {
        CHECK(flags.find(" hg ") != std::string::npos, "the mapping is not VM_HUGEPAGE although THP is available");
    }

    void * const base = m.ptr;
    unified_cache_raw_free_host_cpu(m.ptr, m.mapped_bytes);
    CHECK(unified_cache_host_cpu_munmap_count_for_test() == munmaps_before + 1, "the release was not one munmap");
    CHECK(!vma_exists(base), "the mapping survived its release");
    return 0;
}

int main() {
    int  rc      = 0;
    int  skipped = 0;
    auto run     = [&](int r) {
        if (r == 77) {
            ++skipped;
        } else {
            rc |= r;
        }
    };
    run(test_sizing());
    run(test_argument_and_map_refusals());
    run(test_lock_refusal_under_a_one_page_limit());
    run(test_success_is_locked_hugepage_advised_and_released_once());
    if (rc != 0) {
        return 1;
    }
    if (skipped != 0) {
        return 77;
    }
    std::printf("PASS: test-sycl-host-cpu-mapping\n");
    return 0;
}
