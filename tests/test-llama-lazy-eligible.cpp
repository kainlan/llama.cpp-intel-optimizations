// The one eligibility predicate for a TENSOR_READ_LAZY tensor (llama.cpp-th32 T6a;
// th32 design §4.1, DLM (a)). It is a function of the lazy mode, the tensor's
// size and whether mmap is supported:
//
//   - nothing is eligible when mmap is unsupported;
//   - OFF is never eligible;
//   - AUTO is eligible only above 4 GiB (a tensor of exactly 4 GiB is not);
//   - ON is always eligible.
//
// The pre-fill of the per-file lazy-range record (llama.cpp-5efe C1) calls it
// once per entry; nothing else evaluates it. A predicate returning `mode != OFF`
// is the mutant this test exists to kill: it passes the ON rows and the OFF
// rows, and fails AUTO at 4 GiB and every flag-false row that is not OFF.
//
// Host-only: pure function, no backend registry, device or model.

#include "../src/llama-model-loader.h"

#include <cstddef>
#include <cstdio>

static int n_failures = 0;

struct row {
    const char *    name;
    llama_lazy_mode mode;
    size_t          nbytes;
    bool            mmap_supported;
    bool            expected;
};

int main() {
    const size_t gib4 = 4ull * 1024 * 1024 * 1024;

    const row rows[] = {
        // mmap supported
        { "OFF  @ 4GiB      mmap",    LLAMA_LAZY_MODE_OFF,  gib4,     true,  false },
        { "OFF  @ 4GiB+1    mmap",    LLAMA_LAZY_MODE_OFF,  gib4 + 1, true,  false },
        { "AUTO @ 4GiB      mmap",    LLAMA_LAZY_MODE_AUTO, gib4,     true,  false },
        { "AUTO @ 4GiB+1    mmap",    LLAMA_LAZY_MODE_AUTO, gib4 + 1, true,  true  },
        { "ON   @ 4GiB      mmap",    LLAMA_LAZY_MODE_ON,   gib4,     true,  true  },
        { "ON   @ 4GiB+1    mmap",    LLAMA_LAZY_MODE_ON,   gib4 + 1, true,  true  },
        // mmap unsupported: nothing is eligible
        { "OFF  @ 4GiB      no mmap", LLAMA_LAZY_MODE_OFF,  gib4,     false, false },
        { "OFF  @ 4GiB+1    no mmap", LLAMA_LAZY_MODE_OFF,  gib4 + 1, false, false },
        { "AUTO @ 4GiB      no mmap", LLAMA_LAZY_MODE_AUTO, gib4,     false, false },
        { "AUTO @ 4GiB+1    no mmap", LLAMA_LAZY_MODE_AUTO, gib4 + 1, false, false },
        { "ON   @ 4GiB      no mmap", LLAMA_LAZY_MODE_ON,   gib4,     false, false },
        { "ON   @ 4GiB+1    no mmap", LLAMA_LAZY_MODE_ON,   gib4 + 1, false, false },
    };

    for (const row & r : rows) {
        const bool got = llama_lazy_eligible(r.mode, r.nbytes, r.mmap_supported);
        const bool ok  = got == r.expected;
        printf("%-4s %s: got %d expected %d\n", ok ? "ok" : "FAIL", r.name, (int) got, (int) r.expected);
        if (!ok) {
            n_failures++;
        }
    }

    if (n_failures != 0) {
        printf("FAIL: %d row(s) of %zu\n", n_failures, sizeof(rows) / sizeof(rows[0]));
        return 1;
    }
    printf("PASS: %zu rows\n", sizeof(rows) / sizeof(rows[0]));
    return 0;
}
