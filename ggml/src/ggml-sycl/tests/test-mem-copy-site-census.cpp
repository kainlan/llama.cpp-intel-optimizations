// Runtime gate for the opt-in per-call-site memcpy census (llama.cpp-th86,
// GGML_SYCL_MEMCPY_SITE_CENSUS).
//
// Drives the host-host branch of mem_copy_direct_submit through the public
// mem_copy / mem_copy_async entry points with two plain host buffers, then
// captures what mem_copy_site_census_flush prints and checks that:
//   - the table has one row per call site in THIS file, each naming this
//     file's line (not mem-ops.cpp's), with the right call and byte counts and
//     direction H2H;
//   - no H2H row names mem-ops.cpp, which is what a site captured inside
//     mem-ops.cpp instead of forwarded from the caller would print;
//   - the copies themselves still happened.
//
// The census reads its env once, so main() sets it before the first copy (the
// ctest registration sets it too). Needs a SYCL queue to call the API, so it
// exits 77 (ctest SKIP_RETURN_CODE) when no GPU is visible: a skip, not a pass.

#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP
#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"

#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

static std::string capture_census(const char * reason) {
    std::fflush(stderr);
    FILE * tmp = std::tmpfile();
    if (tmp == nullptr) {
        return {};
    }
    const int saved = dup(fileno(stderr));
    dup2(fileno(tmp), fileno(stderr));
    ggml_sycl::mem_copy_site_census_flush(reason);
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);

    std::string out;
    std::rewind(tmp);
    char   buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), tmp)) > 0) {
        out.append(buf, n);
    }
    std::fclose(tmp);
    return out;
}

// The one census row whose site is this file at `line`, or "" when absent.
// More than one such row also fails: the caller must see exactly one.
static std::string row_for_line(const std::string & table, int line, int * matches) {
    const std::string needle = "test-mem-copy-site-census.cpp:" + std::to_string(line) + " ";
    std::string       found;
    *matches   = 0;
    size_t pos = 0;
    while (pos < table.size()) {
        size_t end = table.find('\n', pos);
        if (end == std::string::npos) {
            end = table.size();
        }
        const std::string l = table.substr(pos, end - pos);
        if (l.find(needle) != std::string::npos) {
            found = l;
            ++*matches;
        }
        pos = end + 1;
    }
    return found;
}

static bool check_row(const std::string & table, int line, uint64_t calls, uint64_t bytes, const char * label) {
    int               matches = 0;
    const std::string row     = row_for_line(table, line, &matches);
    if (matches != 1) {
        std::fprintf(stderr, "FAIL: %s: expected one census row for this file's line %d, found %d\n", label, line,
                     matches);
        return false;
    }
    char want[160];
    std::snprintf(want, sizeof(want), " calls=%" PRIu64 " bytes=%" PRIu64 " dir=H2H path=sycl.memcpy.mem_ops ", calls,
                  bytes);
    if (row.find(want) == std::string::npos) {
        std::fprintf(stderr, "FAIL: %s: row does not carry '%s':\n  %s\n", label, want, row.c_str());
        return false;
    }
    if (row.find(" fn=main") == std::string::npos) {
        std::fprintf(stderr, "FAIL: %s: row does not name the calling function main:\n  %s\n", label, row.c_str());
        return false;
    }
    return true;
}

int main() {
    // Before the first copy: the census latches its env on first use.
    setenv("GGML_SYCL_MEMCPY_SITE_CENSUS", "1", 1);

    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::fprintf(stderr,
                     "SKIP: no SYCL GPU devices available -- NO CENSUS WAS EXERCISED.\n"
                     "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    sycl::queue q(gpus.front(), sycl::property::queue::in_order{});

    constexpr size_t            size = 256;
    std::vector<uint8_t>        src(size, 0xAB);
    std::vector<uint8_t>        dst(size, 0x00);
    // Plain (non-USM) host memory on both ends: mem_copy_direct_submit's
    // host-host branch, a std::memcpy the kernel profiler never sees.
    const ggml_sycl::mem_handle src_h = ggml_sycl::mem_handle::from_direct(src.data(), GGML_LAYOUT_AOS, false,
                                                                           ggml_sycl::mem_handle::HOST_DEVICE, size);
    const ggml_sycl::mem_handle dst_h = ggml_sycl::mem_handle::from_direct(dst.data(), GGML_LAYOUT_AOS, false,
                                                                           ggml_sycl::mem_handle::HOST_DEVICE, size);

    int failed = 0;

    // Site A: three synchronous copies from one line.
    const int site_a = __LINE__ + 2;
    for (int i = 0; i < 3; ++i) {
        ggml_sycl::mem_copy(dst_h, 0, src_h, 0, 64, q);
    }
    // Site B: one async copy at an offset, from another line.
    const int   site_b = __LINE__ + 1;
    sycl::event ev     = ggml_sycl::mem_copy_async(dst_h, 128, src_h, 128, 32, q);
    ev.wait_and_throw();

    for (size_t i = 0; i < size; ++i) {
        const uint8_t want = (i < 64 || (i >= 128 && i < 160)) ? 0xAB : 0x00;
        if (dst[i] != want) {
            std::fprintf(stderr, "FAIL: byte %zu expected 0x%02x got 0x%02x\n", i, want, dst[i]);
            failed = 1;
            break;
        }
    }

    const std::string table = capture_census("test");
    if (table.find("[SYCL-MEMCPY-SITE-CENSUS] test: sites=") == std::string::npos) {
        std::fprintf(stderr, "FAIL: the flush printed no census header:\n%s\n", table.c_str());
        return 1;
    }
    if (!check_row(table, site_a, 3, 3 * 64, "site A (3 x mem_copy, 64 B)")) {
        failed = 1;
    }
    if (!check_row(table, site_b, 1, 32, "site B (1 x mem_copy_async, 32 B)")) {
        failed = 1;
    }
    // A site captured inside mem-ops.cpp instead of forwarded from the caller.
    if (table.find("dir=H2H path=sycl.memcpy.mem_ops site=ggml/src/ggml-sycl/mem-ops.cpp:") != std::string::npos) {
        std::fprintf(stderr, "FAIL: an H2H row names mem-ops.cpp, not its caller:\n%s\n", table.c_str());
        failed = 1;
    }

    if (failed) {
        std::fprintf(stderr, "--- census ---\n%s", table.c_str());
        return 1;
    }
    std::printf("PASS: memcpy site census names this file's lines %d (3 calls, 192 B) and %d (1 call, 32 B) as H2H\n",
                site_a, site_b);
    return 0;
}
