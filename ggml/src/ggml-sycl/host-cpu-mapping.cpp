#include "host-cpu-mapping.hpp"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace ggml_sycl {

#if defined(GGML_SYCL_PRIVATE_TESTING)
static std::atomic<uint64_t> g_munmap_count{ 0 };

uint64_t unified_cache_host_cpu_munmap_count_for_test() {
    return g_munmap_count.load(std::memory_order_acquire);
}
#endif

// RED stand-in (llama.cpp-th32 T1a): the wrapper maps nothing and locks nothing.
size_t host_cpu_mapped_size(size_t request_bytes) {
    return request_bytes;
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
    }
    return "?";
}

host_cpu_status unified_cache_raw_alloc_host_cpu(size_t request_bytes, host_cpu_mapping * out) {
    (void) request_bytes;
    (void) out;
    return host_cpu_status::OK;
}

void unified_cache_raw_free_host_cpu(void * ptr, size_t mapped_bytes) {
    (void) ptr;
    (void) mapped_bytes;
}

std::string host_cpu_refusal_text(host_cpu_status status, const host_cpu_mapping & mapping) {
    (void) status;
    (void) mapping;
    return "";
}

}  // namespace ggml_sycl
