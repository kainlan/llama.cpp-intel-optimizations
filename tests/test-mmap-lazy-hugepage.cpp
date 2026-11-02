// llama_mmap must not request transparent huge pages over the lazily read
// ranges (TENSOR_READ_LAZY, llama.cpp-n77l). Those ranges are read one row at
// a time. On Linux, a fault in a VM_HUGEPAGE area takes the PMD readahead path
// before VM_RAND_READ is honoured, so every row fault would read a whole 2 MiB
// folio, even though the ranges are advised POSIX_MADV_RANDOM.
//
// The kernel reports the advice per VMA in /proc/self/smaps ("hg" in VmFlags),
// so this maps a small file with one unaligned lazy range and checks each
// region: far before and far after the range carry "hg", and the range itself,
// including the pages holding its unaligned ends and the 2 MiB-aligned window
// round it (docs/design/sycl-mmap-host-tier.md §4.6: the MADV_RANDOM window of 2 MiB on each
// side, widened outward to 2 MiB), does not.
//
// Whether THP is available is decided by a positive control that does not go
// through llama_mmap: the test maps the file itself and advises it
// MADV_HUGEPAGE. Exits 77 only when that madvise fails or its VMA still lacks
// "hg". Once the control shows the kernel honours the advice, a llama_mmap
// region without "hg" is a failure, not a skip: deciding the skip from
// llama_mmap's own mapping would turn a regression that stops advising the
// non-lazy ranges into a skip. A VmFlags line that cannot be found is a failure
// too, since "no flags" also reads as "no hg".

#include "../src/llama-mmap.h"
#include "test-skip.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const size_t MiB       = 1024 * 1024;
static const size_t FILE_SIZE = 20 * MiB;

// The backing device's read_ahead_kb, or -1 when /sys has no entry for the
// file's device. The probes below assume the lazy window is 2 MiB, which holds
// while twice the read-ahead is at most 2 MiB. A whole disk or a filesystem
// with its own bdi is /sys/class/bdi/MAJ:MIN; a partition shares its disk's,
// one directory up.
static long read_ahead_kb(int fd) {
    struct stat st = {};
    if (fstat(fd, &st) != 0) {
        return -1;
    }
    char paths[2][160];
    snprintf(paths[0], sizeof(paths[0]), "/sys/class/bdi/%u:%u/read_ahead_kb", major(st.st_dev), minor(st.st_dev));
    snprintf(paths[1], sizeof(paths[1]), "/sys/dev/block/%u:%u/../bdi/read_ahead_kb", major(st.st_dev),
             minor(st.st_dev));
    for (const char * p : paths) {
        std::ifstream in(p);
        long          kb = -1;
        if (in >> kb) {
            return kb;
        }
    }
    return -1;
}

// VmFlags of the VMA that contains `p`, or "" when no VMA or no VmFlags line
// for it was found
static std::string vm_flags_at(const void * p) {
    const uintptr_t target = (uintptr_t) p;
    std::ifstream   smaps("/proc/self/smaps");
    std::string     line;
    bool            in_vma = false;
    while (std::getline(smaps, line)) {
        uintptr_t beg = 0, end = 0;
        if (sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " ", &beg, &end) == 2 && line.find(':') > line.find(' ')) {
            in_vma = target >= beg && target < end;
            continue;
        }
        if (in_vma && line.rfind("VmFlags:", 0) == 0) {
            return line.substr(8) + " ";
        }
    }
    return "";
}

static bool has_hugepage(const std::string & flags) {
    return flags.find(" hg ") != std::string::npos;
}

int main() {
    char      path[] = "/tmp/test-mmap-lazy-hugepage-XXXXXX";
    const int fd     = mkstemp(path);
    if (fd < 0 || ftruncate(fd, FILE_SIZE) != 0) {
        fprintf(stderr, "FAIL: could not create a 20 MiB temp file\n");
        return 1;
    }

    const long ra = read_ahead_kb(fd);
    if (ra > 1024) {
        printf("SKIP: read_ahead_kb %ld puts the lazy window past 2 MiB, which the probes assume\n", ra);
        close(fd);
        unlink(path);
        return LLAMA_TEST_EXIT_SKIP;
    }

    // positive control: can this kernel mark a file mapping VM_HUGEPAGE at all?
    {
        void * ctl = mmap(nullptr, FILE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
        if (ctl == MAP_FAILED) {
            fprintf(stderr, "FAIL: could not map the temp file for the THP control\n");
            close(fd);
            unlink(path);
            return 1;
        }
        if (madvise(ctl, FILE_SIZE, MADV_HUGEPAGE) != 0) {
            printf("SKIP: madvise(MADV_HUGEPAGE) failed on the control mapping (THP unavailable): %s\n",
                   strerror(errno));
            munmap(ctl, FILE_SIZE);
            close(fd);
            unlink(path);
            return LLAMA_TEST_EXIT_SKIP;
        }
        const std::string ctl_flags = vm_flags_at(ctl);
        munmap(ctl, FILE_SIZE);
        if (ctl_flags.empty()) {
            fprintf(stderr, "FAIL: no VmFlags line for the control mapping in /proc/self/smaps\n");
            close(fd);
            unlink(path);
            return 1;
        }
        if (!has_hugepage(ctl_flags)) {
            printf("SKIP: the advised control mapping carries no hg flag (THP unavailable): '%s'\n", ctl_flags.c_str());
            close(fd);
            unlink(path);
            return LLAMA_TEST_EXIT_SKIP;
        }
    }
    close(fd);

    // the window is [4 MiB + 100, 11 MiB + 7) widened to [4 MiB, 12 MiB)
    const size_t lazy_beg = 6 * MiB + 100;
    const size_t lazy_end = 9 * MiB + 7;

    int n_failures = 0;
    {
        const llama_mmap::ranges lazy_ranges = {
            { lazy_beg, lazy_end }
        };

        llama_file   f(path, "rb");
        llama_mmap   m(&f, /*prefetch =*/0, /*numa =*/false, lazy_ranges);
        const char * base = (const char *) m.addr();

        struct probe {
            const char * what;
            size_t       off;
            bool         want_hg;
        };

        const probe probes[] = {
            { "before the window",            0,               true  },
            { "just before the window",       4 * MiB - 4096,  true  },
            { "window start",                 4 * MiB,         false },
            { "next to the lazy start",       lazy_beg - 4096, false },
            { "first byte of the lazy range", lazy_beg,        false },
            { "middle of the lazy range",     7 * MiB,         false },
            { "last byte of the lazy range",  lazy_end - 1,    false },
            { "next to the lazy end",         lazy_end + 4096, false },
            { "last page of the window",      12 * MiB - 4096, false },
            { "just after the window",        12 * MiB,        true  },
            { "after the window",             15 * MiB,        true  },
        };
        for (const probe & pr : probes) {
            const std::string flags = vm_flags_at(base + pr.off);
            if (flags.empty()) {
                fprintf(stderr, "FAIL: %s (offset %zu): no VmFlags line\n", pr.what, pr.off);
                n_failures++;
                continue;
            }
            const bool got = has_hugepage(flags);
            if (got != pr.want_hg) {
                fprintf(stderr, "FAIL: %s (offset %zu): hg %s, want %s; VmFlags '%s'\n", pr.what, pr.off,
                        got ? "present" : "absent", pr.want_hg ? "present" : "absent", flags.c_str());
                n_failures++;
            }
        }
    }
    unlink(path);

    if (n_failures) {
        fprintf(stderr, "%d check(s) failed\n", n_failures);
        return 1;
    }
    printf("OK: MADV_HUGEPAGE covers only the non-lazy ranges\n");
    return 0;
}
