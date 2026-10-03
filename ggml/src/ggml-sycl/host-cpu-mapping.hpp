#pragma once

// The raw wrapper of the unified cache's HOST_CPU tier (llama.cpp-th32 T1,
// docs/design/sycl-storage-tier-lazy-tables.md §3.1).
//
// HOST_CPU is host memory that only the CPU reads: anonymous, MADV_HUGEPAGE, and mlocked in full before the allocation
// returns, so the resident bytes equal the planned bytes from the first moment and a failure surfaces at load, not
// mid-run (an unlocked table would be swapped, and a "resident" table would silently become disk reads). A request
// that cannot be mapped or locked is refused by name, never served from another tier.
//
// Device-free on purpose, like file-mapping.hpp: this header and host-cpu-mapping.cpp include no SYCL header, so host
// tests link the translation unit alone. Together they are the only code in ggml-sycl that maps or locks anonymous host
// memory for the tier; the owner entry (unified_allocate_owner with constraints.must_host_cpu) calls them and nothing
// else does.
//
// Huge-page backing depends on the kernel: the whole mapping is 2 MiB sized, and it lands in huge pages only when the
// kernel also 2 MiB-aligns it. Kernels from 6.7 on align large anonymous mmaps; an older one may hand an unaligned
// base, and then the first and last huge pages degrade to small pages (correctness is unchanged, the speed is not).
// The wrapper does not over-map and trim to force alignment, because the release is one munmap of the whole mapping.

#include <cstddef>
#include <cstdint>
#include <string>

namespace ggml_sycl {

// The largest alignment a HOST_CPU request may ask for: one 4 KiB page, which is all an anonymous mmap guarantees
// (2 MiB only by the kernel >= 6.7 behaviour above, which no request may count on). The front doors refuse a request
// that asks for more as INVALID_REQUEST rather than hand back a pointer that is not aligned as asked.
constexpr size_t HOST_CPU_MAX_ALIGNMENT = 4096;

// The one sizing function: the bytes the wrapper maps, the owner entry charges and the plan's HOST terms count for a
// request of `request_bytes`. Rounded up to whole 2 MiB huge pages (which are whole pages), at least one. 0 on
// overflow, which no caller may pass on.
size_t host_cpu_mapped_size(size_t request_bytes);

enum class host_cpu_status : uint8_t {
    OK = 0,
    INVALID_ARGUMENT,  // a zero request, or one whose mapped size overflows
    MMAP_FAILED,       // the anonymous mmap failed (errno names why)
    LOCK_FAILED,       // mlock failed: refused as host-cpu-lock-failed (errno names why)
    UNSUPPORTED,       // this build has no POSIX mapping (Windows): the tier is absent, nothing is mapped
};

// Whether this build has the tier. host-cpu-mapping.cpp is compiled on every platform (the backend globs its sources),
// but anonymous mmap, MADV_HUGEPAGE and mlock need POSIX, so on Windows the wrapper is a refusing stub and this is
// false; the caller then refuses a must_host_cpu request by name instead of reaching a missing hook.
constexpr bool host_cpu_supported() {
#if defined(_WIN32)
    return false;
#else
    return true;
#endif
}

// "ok", "invalid-argument", "host-cpu-map-failed", "host-cpu-lock-failed" or "unsupported-platform".
const char * host_cpu_status_name(host_cpu_status status);

struct host_cpu_mapping {
    void *   ptr             = nullptr;
    size_t   mapped_bytes    = 0;      // host_cpu_mapped_size(request): the size asked of the kernel. A live mapping
                                       // only when ptr != null; after LOCK_FAILED it is non-zero though nothing is
                                       // mapped (it was unmapped again), after MMAP_FAILED it is 0.
    bool     huge_advised    = false;  // MADV_HUGEPAGE was accepted (the kernel may still back it with small pages)
    int      err             = 0;      // errno of the failing call, 0 on success
    uint64_t rlimit_memlock  = 0;      // RLIMIT_MEMLOCK soft limit at the failure; ~0 is unlimited
    size_t   requested_bytes = 0;

    explicit operator bool() const { return ptr != nullptr; }
};

// mmap(MAP_PRIVATE | MAP_ANONYMOUS) of host_cpu_mapped_size(request_bytes), madvise(MADV_HUGEPAGE), then mlock over the
// whole range (without MLOCK_ONFAULT, so every page is faulted in and locked before this returns). On a refusal nothing
// is left mapped, *out names the failure (err, rlimit_memlock, the sizes) and ptr is null.
host_cpu_status unified_cache_raw_alloc_host_cpu(size_t request_bytes, host_cpu_mapping * out);

// The only release: one munmap of the whole mapping, which drops its lock. `mapped_bytes` is what the allocation
// returned, never the request. The caller releases exactly once, from the owning handle's destructor: the release is
// not idempotent, a second call with the same range would unmap whatever was mapped there since.
void unified_cache_raw_free_host_cpu(void * ptr, size_t mapped_bytes);

// The refusal as the log line and the typed error carry it: the status name, the mapped and requested bytes, errno and
// the RLIMIT_MEMLOCK in force, e.g.
//   host-cpu-lock-failed: mlock of 2097152 bytes (requested 4096) failed: errno=12 (Cannot allocate memory),
//   RLIMIT_MEMLOCK=4096
std::string host_cpu_refusal_text(host_cpu_status status, const host_cpu_mapping & mapping);

#if defined(GGML_SYCL_PRIVATE_TESTING)
// Private counter of the munmap calls this translation unit has issued, so a test can tell a release from a leak.
uint64_t unified_cache_host_cpu_munmap_count_for_test();
#endif

}  // namespace ggml_sycl
