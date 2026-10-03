// The HOST_CPU tier on a build without it (llama.cpp-th32 T1a; the C2 Windows ruling applied to the second wrapper).
//
// host-cpu-mapping.cpp is compiled on every platform (the backend globs its sources), but anonymous mmap,
// MADV_HUGEPAGE and mlock need POSIX, so its POSIX body is guarded by `#if !defined(_WIN32)` and a refusing stub
// stands in where it is not. CMake builds this target with _WIN32 defined for these two translation units only, so the
// stub path is compiled and run on the Linux host: the allocation answers UNSUPPORTED and maps nothing, release is a
// safe no-op, the capability query says the tier is absent, and the pure sizing function still works.
//
// Host-only: no device, no model, no mapping.

#include "../ggml/src/ggml-sycl/host-cpu-mapping.hpp"

#include <cstdio>
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

int main() {
    CHECK(!host_cpu_supported(), "the tier reports itself present on a build without it");

    host_cpu_mapping m;
    CHECK(unified_cache_raw_alloc_host_cpu(4096, &m) == host_cpu_status::UNSUPPORTED,
          "the allocation is not refused as unsupported");
    CHECK(m.ptr == nullptr && m.mapped_bytes == 0, "a refused allocation wrote a mapping");
    CHECK(unified_cache_raw_alloc_host_cpu(4096, nullptr) == host_cpu_status::INVALID_ARGUMENT,
          "a null out was not refused as an invalid argument");

    // release of a mapping that never existed is a no-op, and a stub never counts a munmap it did not issue
    unified_cache_raw_free_host_cpu(nullptr, 0);
    unified_cache_raw_free_host_cpu(m.ptr, m.mapped_bytes);

    CHECK(std::string(host_cpu_status_name(host_cpu_status::UNSUPPORTED)) == "unsupported-platform",
          "UNSUPPORTED has no name");
    CHECK(host_cpu_refusal_text(host_cpu_status::UNSUPPORTED, m).find("unsupported-platform") != std::string::npos,
          "the refusal text does not carry the name");

    // the plan's HOST terms use the sizing function on every platform, so it is not behind the guard
    CHECK(host_cpu_mapped_size(1) == (size_t) 2 * 1024 * 1024, "the sizing function does not work here");
    CHECK(host_cpu_mapped_size(0) == 0, "a zero request sizes to something");

    std::printf("PASS: test-sycl-host-cpu-unsupported\n");
    return 0;
}
