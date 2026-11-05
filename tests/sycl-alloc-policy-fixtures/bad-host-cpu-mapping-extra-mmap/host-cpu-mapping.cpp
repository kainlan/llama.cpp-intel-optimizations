#include "host-cpu-mapping.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#    include <sys/mman.h>
#    include <sys/resource.h>
#endif

namespace ggml_sycl {

#if defined(GGML_SYCL_PRIVATE_TESTING)
static std::atomic<uint64_t> g_munmap_count{ 0 };

uint64_t unified_cache_host_cpu_munmap_count_for_test() {
    return g_munmap_count.load(std::memory_order_acquire);
}
#endif

static const size_t HOST_CPU_HUGE_PAGE = (size_t) 2 * 1024 * 1024;

size_t host_cpu_mapped_size(size_t request_bytes) {
    if (request_bytes == 0 || request_bytes > SIZE_MAX - (HOST_CPU_HUGE_PAGE - 1)) {
        return 0;
    }
    // a 2 MiB multiple is a multiple of every page size this tier runs on (4 KiB, 16 KiB, 64 KiB)
    return (request_bytes + HOST_CPU_HUGE_PAGE - 1) / HOST_CPU_HUGE_PAGE * HOST_CPU_HUGE_PAGE;
}

const char * host_cpu_status_name(host_cpu_status status) {
    switch (status) {
        case host_cpu_status::OK:
            return "ok";
        case host_cpu_status::INVALID_ARGUMENT:
            return "invalid-argument";
        case host_cpu_status::MMAP_FAILED:
            return "host-cpu-map-failed";
        case host_cpu_status::LOCK_FAILED:
            return "host-cpu-lock-failed";
        case host_cpu_status::UNSUPPORTED:
            return "unsupported-platform";
    }
    return "?";
}

std::string host_cpu_refusal_text(host_cpu_status status, const host_cpu_mapping & mapping) {
    char buf[384];
    if (status == host_cpu_status::LOCK_FAILED) {
        std::snprintf(buf, sizeof(buf),
                      "%s: mlock of %zu bytes (requested %zu) failed: errno=%d (%s), RLIMIT_MEMLOCK=%llu",
                      host_cpu_status_name(status), mapping.mapped_bytes, mapping.requested_bytes, mapping.err,
                      std::strerror(mapping.err), (unsigned long long) mapping.rlimit_memlock);
    } else {
        // a refusal that made no syscall (a zero or overflowing request, an absent platform) has no errno to print
        if (mapping.err != 0) {
            std::snprintf(buf, sizeof(buf), "%s: request of %zu bytes failed: errno=%d (%s)",
                          host_cpu_status_name(status), mapping.requested_bytes, mapping.err,
                          std::strerror(mapping.err));
        } else {
            std::snprintf(buf, sizeof(buf), "%s: request of %zu bytes refused", host_cpu_status_name(status),
                          mapping.requested_bytes);
        }
    }
    return buf;
}

#if !defined(_WIN32)

// The one munmap of this translation unit (a gate counts the calls): the lock-failure unwind and the release both come
// through here, so the private counter sits at the syscall and counts only a munmap that returned 0.
static void host_cpu_munmap(void * ptr, size_t mapped_bytes) {
    if (munmap(ptr, mapped_bytes) == 0) {
#    if defined(GGML_SYCL_PRIVATE_TESTING)
        g_munmap_count.fetch_add(1, std::memory_order_acq_rel);
#    endif
    }
}

host_cpu_status unified_cache_raw_alloc_host_cpu(size_t request_bytes, host_cpu_mapping * out) {
    if (out == nullptr) {
        return host_cpu_status::INVALID_ARGUMENT;
    }
    *out                 = host_cpu_mapping{};
    out->requested_bytes = request_bytes;
    const size_t mapped  = host_cpu_mapped_size(request_bytes);
    if (mapped == 0) {
        return host_cpu_status::INVALID_ARGUMENT;
    }

    void * p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        out->err = errno;
        return host_cpu_status::MMAP_FAILED;
    }
    out->mapped_bytes = mapped;
    void * extra = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);  // EXTRA-MMAP
    (void) extra;

    // advisory: a refusal (THP compiled out, or "never") costs speed, not correctness. Before the lock, because mlock
    // faults every page in and a page faulted in small stays small. MADV_HUGEPAGE is Linux-only, so a POSIX kernel
    // without it simply leaves huge_advised false.
#    if defined(MADV_HUGEPAGE)
    out->huge_advised = madvise(p, mapped, MADV_HUGEPAGE) == 0;
#    endif

    // the whole range, faulted in and locked before returning: never MLOCK_ONFAULT, so resident == planned from the
    // first moment and an unlockable request fails here, at load, never mid-run
    if (mlock(p, mapped) != 0) {
        out->err            = errno;
        struct rlimit rl    = {};
        out->rlimit_memlock = getrlimit(RLIMIT_MEMLOCK, &rl) == 0 ? (uint64_t) rl.rlim_cur : 0;
        host_cpu_munmap(p, mapped);
        out->huge_advised = false;  // mapped_bytes stays: the refusal names what was asked of the kernel
        return host_cpu_status::LOCK_FAILED;
    }
    out->ptr = p;
    return host_cpu_status::OK;
}

void unified_cache_raw_free_host_cpu(void * ptr, size_t mapped_bytes) {
    if (ptr == nullptr || mapped_bytes == 0) {
        return;
    }
    host_cpu_munmap(ptr, mapped_bytes);
}

#else

// A build without POSIX has no HOST_CPU tier: the allocation refuses by name and maps nothing, the release is a no-op.
host_cpu_status unified_cache_raw_alloc_host_cpu(size_t request_bytes, host_cpu_mapping * out) {
    if (out == nullptr) {
        return host_cpu_status::INVALID_ARGUMENT;
    }
    *out                 = host_cpu_mapping{};
    out->requested_bytes = request_bytes;
    return host_cpu_status::UNSUPPORTED;
}

void unified_cache_raw_free_host_cpu(void * ptr, size_t mapped_bytes) {
    (void) ptr;
    (void) mapped_bytes;
}

#endif

}  // namespace ggml_sycl
