// llama_mmap must not request transparent huge pages over the lazily read
// ranges (TENSOR_READ_LAZY, llama.cpp-n77l). Those ranges are read one row at
// a time. On Linux, a fault in a VM_HUGEPAGE area takes the PMD readahead path
// before VM_RAND_READ is honoured, so every row fault would read a whole 2 MiB
// folio, even though the ranges are advised POSIX_MADV_RANDOM.
//
// The kernel reports the advice per VMA in /proc/self/smaps ("hg" in VmFlags),
// so this maps a small file with one unaligned lazy range and checks each
// region: before and after the range carry "hg", and the range itself,
// including the pages holding its unaligned ends, does not. Exits 77 when the
// non-lazy region's VmFlags were read and carry no "hg" either (THP
// unavailable), since then the check cannot tell anything apart. A VmFlags
// line that cannot be found is a failure, not a skip: "no flags" also reads as
// "no hg", so skipping on it would pass every probe vacuously.

#include "../src/llama-mmap.h"

#include <unistd.h>

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

        const std::string base_flags = vm_flags_at(base);
        if (base_flags.empty()) {
            fprintf(stderr, "FAIL: no VmFlags line for the mapping in /proc/self/smaps\n");
            unlink(path);
            return 1;
        }
        if (!has_hugepage(base_flags)) {
            printf("SKIP: the non-lazy region carries no hg flag (THP unavailable): '%s'\n", base_flags.c_str());
            unlink(path);
            return 77;
        }

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
