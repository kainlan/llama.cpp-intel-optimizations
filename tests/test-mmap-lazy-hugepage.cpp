// llama_mmap must not request transparent huge pages over the lazily read
// ranges (TENSOR_READ_LAZY, llama.cpp-n77l). Those ranges are read one row at
// a time. On Linux, a fault in a VM_HUGEPAGE area takes the PMD readahead path
// before VM_RAND_READ is honoured, so every row fault would read a whole 2 MiB
// folio, even though the ranges are advised POSIX_MADV_RANDOM.
//
// The kernel reports the advice per VMA in /proc/self/smaps ("hg" in VmFlags),
// so this maps a small file with one unaligned lazy range and checks each
// region: before and after the range carry "hg", and the range itself,
// including the pages holding its unaligned ends, does not.
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

#include <sys/mman.h>
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

static const size_t MiB = 1024 * 1024;

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
    if (fd < 0 || ftruncate(fd, 8 * MiB) != 0) {
        fprintf(stderr, "FAIL: could not create an 8 MiB temp file\n");
        return 1;
    }

    // positive control: can this kernel mark a file mapping VM_HUGEPAGE at all?
    {
        void * ctl = mmap(nullptr, 8 * MiB, PROT_READ, MAP_SHARED, fd, 0);
        if (ctl == MAP_FAILED) {
            fprintf(stderr, "FAIL: could not map the temp file for the THP control\n");
            close(fd);
            unlink(path);
            return 1;
        }
        if (madvise(ctl, 8 * MiB, MADV_HUGEPAGE) != 0) {
            printf("SKIP: madvise(MADV_HUGEPAGE) failed on the control mapping (THP unavailable): %s\n",
                   strerror(errno));
            munmap(ctl, 8 * MiB);
            close(fd);
            unlink(path);
            return 77;
        }
        const std::string ctl_flags = vm_flags_at(ctl);
        munmap(ctl, 8 * MiB);
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
            return 77;
        }
    }
    close(fd);

    const size_t lazy_beg = 2 * MiB + 100;
    const size_t lazy_end = 5 * MiB + 7;

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
            { "before the lazy range",        0,            true  },
            { "first byte of the lazy range", lazy_beg,     false },
            { "middle of the lazy range",     3 * MiB,      false },
            { "last byte of the lazy range",  lazy_end - 1, false },
            { "after the lazy range",         6 * MiB,      true  },
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
