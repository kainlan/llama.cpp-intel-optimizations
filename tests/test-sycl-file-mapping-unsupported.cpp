// The FILE_MAPPED tier on a build without it (llama.cpp-5efe C2 review M2; design sycl-mmap-host-tier.md §4.3 m-19).
//
// file-mapping.cpp is compiled on every platform (the backend globs its sources), but the reservation and the
// MAP_FIXED segments need POSIX, so its POSIX body is guarded by `#if !defined(_WIN32)` and a refusing stub stands in
// where it is not. CMake builds this target with _WIN32 defined for these two translation units only, so the stub
// path is compiled and run on the Linux host: every primitive answers UNSUPPORTED and touches nothing, release is a
// safe no-op, the capability query says the tier is absent (the answer C4's `load_maps_files` reads), and the pure
// segment plan, which has no POSIX call, still works.
//
// Host-only: no device, no model, no mapping.

#include "../ggml/src/ggml-sycl/file-mapping.hpp"

#include <cstdio>
#include <string>
#include <vector>

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
    CHECK(!file_mapping_supported(), "the tier reports itself present on a build without it");

    // the map refuses, writes nothing, and opens nothing (there is no fd to dup here)
    file_mapping m;
    CHECK(unified_cache_raw_map_file(0, 4096,
                                     {
                                         0, 4096
    },
                                     { { 0, 4096 } }, &m) == file_map_status::UNSUPPORTED,
          "the map is not refused as unsupported");
    CHECK(m.base == nullptr && m.fd == -1 && m.length == 0, "a refused map wrote its output");

    // lock and advise refuse, whatever they are given
    file_mapping live;
    live.base = (void *) 0x1000;  // never dereferenced: the stub touches nothing
    CHECK(unified_cache_raw_lock_file_segments(live,
                                               {
                                                   { 0, 4096 }
    }) == file_map_status::UNSUPPORTED,
          "the lock is not refused as unsupported");
    CHECK(unified_cache_raw_advise_file_segments(live,
                                                 {
                                                     { 0, 4096 }
    },
                                                 file_advice::WILLNEED) == file_map_status::UNSUPPORTED,
          "the advise is not refused as unsupported");

    // release of a mapping that never existed is a no-op, and clears the struct
    unified_cache_raw_unmap_file(&m);
    unified_cache_raw_unmap_file(nullptr);
    CHECK(m.base == nullptr && m.fd == -1, "the release of an empty mapping left it non-empty");

    CHECK(std::string(file_map_status_name(file_map_status::UNSUPPORTED)) == "unsupported-platform",
          "UNSUPPORTED has no name");

    // the plan is pure: it still answers, with the page size as a parameter
    const file_segment_plan plan = file_mapping_segments(
        {
            { 0, 8192, true }
    },
        {}, 4096, 8192);
    CHECK(plan.ok() && plan.map.size() == 1 && plan.map[0].length == 8192, "the pure plan does not work here");

    std::printf("PASS: test-sycl-file-mapping-unsupported\n");
    return 0;
}
