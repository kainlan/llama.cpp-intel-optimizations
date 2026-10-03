// The segment plan of the FILE_MAPPED tier (llama.cpp-5efe C3, docs/design/sycl-mmap-host-tier.md §4.6): a pure
// function, no I/O and no device. The page size is a parameter, so every case here is the same on every host.
//
// Page 4096 and W = 8192 (two pages) throughout. GGUF aligns tensors to 32 B, so a lazy range and a tensor routinely
// share a page; the cases below are built around that.

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

static const uint64_t P = 4096;
static const uint64_t W = 8192;

using ranges = std::vector<file_map_range>;

static std::string show(const ranges & r) {
    std::string s = "{";
    for (const file_map_range & x : r) {
        s += " [" + std::to_string(x.file_offset) + "," + std::to_string(x.file_end()) + ")";
    }
    return s + " }";
}

static bool same(const ranges & got, const ranges & want, const char * what) {
    bool eq = got.size() == want.size();
    for (size_t i = 0; eq && i < got.size(); ++i) {
        eq = got[i].file_offset == want[i].file_offset && got[i].length == want[i].length;
    }
    if (!eq) {
        std::fprintf(stderr, "FAIL: %s: got %s, want %s\n", what, show(got).c_str(), show(want).c_str());
    }
    return eq;
}

static file_map_range R(uint64_t begin, uint64_t end) {
    return { begin, end - begin };
}

static file_tensor_range T(uint64_t begin, uint64_t end, bool host) {
    return { begin, end - begin, host };
}

// A lazy range between two host tensors, each sharing a page with it.
static int test_interleaved_lazy_range() {
    const file_segment_plan p =
        file_mapping_segments({ T(0, 40000, true), T(100000, 140000, true) }, { R(40000, 100000) }, P, W);
    CHECK(p.ok(), "a valid plan was refused");
    // outward: A [0,40960) takes the shared page at 36864; B [98304,143360) takes the shared page at 98304
    CHECK(same(p.map, { R(0, 40960), R(98304, 143360) }, "map"), "map segments");
    // inward, minus every page holding a lazy byte (pages 36864..102400): the shared pages are not advised
    CHECK(same(p.willneed, { R(0, 36864), R(102400, 139264) }, "willneed"), "willneed segments");
    // W before the start and after the end, shared pages inside, intersected with the map
    CHECK(same(p.random_windows, { R(28672, 40960), R(98304, 110592) }, "random"), "random windows");
    return 0;
}

// Two lazy ranges back to back: the same as one lazy range of their union.
static int test_back_to_back_lazy_ranges() {
    const file_segment_plan one =
        file_mapping_segments({ T(0, 40000, true), T(100000, 140000, true) }, { R(40000, 100000) }, P, W);
    const file_segment_plan two = file_mapping_segments({ T(0, 40000, true), T(100000, 140000, true) },
                                                        { R(40000, 60000), R(60000, 100000) }, P, W);
    CHECK(two.ok(), "back-to-back lazy ranges were refused");
    CHECK(same(two.map, one.map, "back-to-back map"), "back-to-back map");
    CHECK(same(two.willneed, one.willneed, "back-to-back willneed"), "back-to-back willneed");
    // each lazy range has its own windows; the shared inner edge at 60000 is inside the lazy bytes, so it is
    // never inside the map, and the outer edges are the ones of the single range
    CHECK(same(two.random_windows, one.random_windows, "back-to-back random"), "back-to-back random windows");
    return 0;
}

// A lazy range at offset 0, before the first locked tensor.
static int test_lazy_range_at_offset_zero() {
    const file_segment_plan p = file_mapping_segments({ T(5000, 30000, true) }, { R(0, 5000) }, P, W);
    CHECK(p.ok(), "a lazy range at offset 0 was refused");
    CHECK(same(p.map, { R(4096, 32768) }, "map"), "map takes the shared page and nothing before it");
    CHECK(same(p.willneed, { R(8192, 28672) }, "willneed"), "the lock set starts after the shared page");
    CHECK(same(p.random_windows, { R(4096, 16384) }, "random"), "only the window after the lazy range exists");
    return 0;
}

// A lazy range shorter than a page, between two tensors: one shared page holds both neighbours.
static int test_sub_page_lazy_range() {
    const file_segment_plan p =
        file_mapping_segments({ T(0, 5000, true), T(5100, 12000, true) }, { R(5000, 5100) }, P, W);
    CHECK(p.ok(), "a sub-page lazy range was refused");
    CHECK(same(p.map, { R(0, 12288) }, "map"), "the tensors' outward pages merge");
    CHECK(same(p.willneed, { R(0, 4096) }, "willneed"), "only the page wholly before the lazy page is advised");
    return 0;
}

// Bytes that belong to no tensor of the buffer are never mapped, and a device-only source is mapped but not advised.
static int test_bytes_of_no_tensor_are_never_mapped() {
    const file_segment_plan p = file_mapping_segments({ T(0, 10000, true), T(200000, 210000, false) }, {}, P, W);
    CHECK(p.ok(), "no lazy ranges was refused");
    CHECK(same(p.map, { R(0, 12288), R(196608, 212992) }, "map"), "the bytes between the tensors were mapped");
    CHECK(same(p.willneed, { R(0, 8192) }, "willneed"), "a device-only source was advised");
    CHECK(p.random_windows.empty(), "windows without a lazy range");
    return 0;
}

// Tensors that touch: the page they share is advised, because no lazy byte is on it.
static int test_touching_host_tensors_share_a_willneed_page() {
    const file_segment_plan p = file_mapping_segments({ T(0, 6000, true), T(6000, 12288, true) }, {}, P, W);
    CHECK(p.ok(), "touching tensors were refused");
    CHECK(same(p.willneed, { R(0, 12288) }, "willneed"), "a page shared by two host tensors was left unadvised");
    return 0;
}

// A tensor that overlaps a lazy range is a caller bug: refused, and the plan is empty.
static int test_overlap_is_refused() {
    const file_segment_plan p = file_mapping_segments({ T(0, 50000, true) }, { R(40000, 100000) }, P, W);
    CHECK(!p.ok(), "a tensor overlapping a lazy range was accepted");
    CHECK(p.map.empty() && p.willneed.empty() && p.random_windows.empty(), "a refused plan is not empty");
    const file_segment_plan zero_page = file_mapping_segments({ T(0, 100, true) }, {}, 0, W);
    CHECK(!zero_page.ok(), "a zero page size was accepted");
    return 0;
}

static int test_random_window_width() {
    CHECK(file_mapping_random_window(128 * 1024, P) == 2 * 1024 * 1024,
          "W is not max(2 x read_ahead, 2 MiB) at 128 KiB");
    CHECK(file_mapping_random_window(4 * 1024 * 1024, P) == 8 * 1024 * 1024, "W does not follow a large read_ahead");
    CHECK(file_mapping_random_window(0, P) == 2 * 1024 * 1024, "W has no 2 MiB floor");
    CHECK(file_mapping_random_window(1048577, P) % P == 0, "W is not whole pages");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_interleaved_lazy_range();
    rc |= test_back_to_back_lazy_ranges();
    rc |= test_lazy_range_at_offset_zero();
    rc |= test_sub_page_lazy_range();
    rc |= test_bytes_of_no_tensor_are_never_mapped();
    rc |= test_touching_host_tensors_share_a_willneed_page();
    rc |= test_overlap_is_refused();
    rc |= test_random_window_width();
    if (rc == 0) {
        std::printf("PASS: test-sycl-file-mapping-segments\n");
    }
    return rc;
}
