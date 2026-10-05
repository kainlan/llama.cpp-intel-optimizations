// hostmem-bigalloc-probe.cpp -- LD_PRELOAD probe that logs the size and call
// stack of every large host allocation (llama.cpp-z5fn).
//
// [HOSTMEM] shows WHEN glibc's mmapped total grows; this shows WHO. glibc's own
// large allocations (malloc, calloc, realloc, memalign, posix_memalign,
// aligned_alloc, and so operator new and std::vector) go through the
// interposed entry points below, which forward to glibc's __libc_* functions and
// print one `[BIGALLOC]` block per allocation of at least
// GGML_HOSTMEM_BIGALLOC_MIN_MB (default 64) MiB to stderr.
//
// Pure host code: build with plain g++, no oneAPI.
//
//   g++ -O1 -g -fPIC -shared -o libhostmem_bigalloc.so scripts/hostmem-bigalloc-probe.cpp -ldl
//   LD_PRELOAD=$PWD/libhostmem_bigalloc.so <command> 2> run.err
//   python3 scripts/hostmem-bigalloc-resolve.py run.err
//
// Every frame is printed as `module+0xOFFSET` (offset relative to the module's
// load base, which is what addr2line wants for a shared object); the resolver
// turns those into function names. Nothing in here allocates: output goes
// through write(2), and a thread-local flag stops the unwinder's own first-use
// allocations from recursing.
//
// Caveats (read before trusting a negative result):
//  * Frame skipping. The first two backtrace frames are dropped on the assumption
//    that they are report() and the interposed entry point. That holds only while
//    report() is not inlined into its caller; an -O2 build that inlines it shifts
//    the stacks by one frame. Build with the -O1 line above.
//  * Deadlock risk. The hook takes no lock of its own, but backtrace() may call
//    dl_iterate_phdr, which takes the loader lock. A >= threshold malloc made
//    from inside dlopen() on the same thread could therefore deadlock. Never seen
//    in practice at the 64 MiB default; if a run hangs at library load, raise
//    GGML_HOSTMEM_BIGALLOC_MIN_MB.
//  * Coverage. valloc(), pvalloc() and direct mmap() calls are NOT interposed, so
//    a mapping created by mmap itself (a library's own arena, a file mapping, the
//    driver) never shows up. If the resolved total is well below the growth you
//    are chasing, the remainder is such a mapping: fall back to a debugger break
//    on mmap.
//  * Offsets. Frames are printed as offsets from dladdr's dli_fbase, which is what
//    addr2line wants for a shared object or a PIE executable. For a non-PIE
//    (ET_EXEC) main binary addr2line expects the absolute address, so frames in
//    that module resolve to the wrong symbol unless the base is added back. The
//    llama.cpp tools and libraries here are PIE/shared.

#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" {
void * __libc_malloc(size_t);
void * __libc_calloc(size_t, size_t);
void * __libc_realloc(void *, size_t);
void * __libc_memalign(size_t, size_t);
}

static thread_local int t_in_hook = 0;
static size_t           g_min_bytes = 0;  // 0 = not read yet

static size_t min_bytes() {
    if (g_min_bytes == 0) {
        size_t       mb = 64;
        const char * e  = getenv("GGML_HOSTMEM_BIGALLOC_MIN_MB");
        if (e && *e) {
            const long v = strtol(e, nullptr, 10);
            if (v > 0) {
                mb = (size_t) v;
            }
        }
        g_min_bytes = mb * 1024 * 1024;
    }
    return g_min_bytes;
}

static void emit(const char * s) {
    const size_t n = strlen(s);
    ssize_t      r = write(2, s, n);
    (void) r;
}

static void report(const char * fn, size_t size, size_t align) {
    if (size < min_bytes() || t_in_hook) {
        return;
    }
    t_in_hook = 1;
    void * frames[28];
    const int n = backtrace(frames, 28);
    char line[256];
    snprintf(line, sizeof(line), "[BIGALLOC] fn=%s size=%zu MiB=%.1f align=%zu tid=%ld\n", fn, size,
             size / (1024.0 * 1024.0), align, (long) syscall(SYS_gettid));
    emit(line);
    // frame 0 is this function and 1 the interposed entry point; start at 2.
    for (int i = 2; i < n; ++i) {
        Dl_info info;
        if (dladdr(frames[i], &info) && info.dli_fname) {
            snprintf(line, sizeof(line), "  #%d %s+0x%lx\n", i - 2, info.dli_fname,
                     (unsigned long) ((char *) frames[i] - (char *) info.dli_fbase));
        } else {
            snprintf(line, sizeof(line), "  #%d ?+%p\n", i - 2, frames[i]);
        }
        emit(line);
    }
    emit("[BIGALLOC-END]\n");
    t_in_hook = 0;
}

// Prime backtrace()/dladdr() so their first-use allocations happen before any
// large allocation is reported.
__attribute__((constructor)) static void probe_init() {
    t_in_hook = 1;
    void * f[4];
    (void) backtrace(f, 4);
    Dl_info i;
    (void) dladdr((void *) probe_init, &i);
    t_in_hook = 0;
}

extern "C" {

void * malloc(size_t size) {
    report("malloc", size, 0);
    return __libc_malloc(size);
}

void * calloc(size_t n, size_t size) {
    report("calloc", n * size, 0);
    return __libc_calloc(n, size);
}

void * realloc(void * p, size_t size) {
    report("realloc", size, 0);
    return __libc_realloc(p, size);
}

void * memalign(size_t align, size_t size) {
    report("memalign", size, align);
    return __libc_memalign(align, size);
}

void * aligned_alloc(size_t align, size_t size) {
    report("aligned_alloc", size, align);
    return __libc_memalign(align, size);
}

int posix_memalign(void ** out, size_t align, size_t size) {
    if (align < sizeof(void *) || (align & (align - 1)) != 0) {
        return EINVAL;
    }
    report("posix_memalign", size, align);
    void * p = __libc_memalign(align, size);
    if (!p) {
        return ENOMEM;
    }
    *out = p;
    return 0;
}

}  // extern "C"
