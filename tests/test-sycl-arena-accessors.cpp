// Host test for the two device-level arena accessors, and the two committed-side
// capacity accessors beside zone_capacity():
//
//   ggml_sycl_arena_backing(device)     what backs the device's zones (fixed at
//                                       construction, creating lookup)
//   ggml_sycl_device_has_zones(device)  whether the zones are backed now (a
//                                       lookup that never creates)
//   zone_capacity_committed / zone_capacity_to_commit
//                                       non-VM bodies: a plan-defect line, an
//                                       abort under GGML_SYCL_STRICT_LEASES=1
//
// The cache is a real unified_cache on the OpenCL CPU device, registered through
// unified_cache_register_for_queue() so no GPU is touched (the registration pins
// the selector to the CPU device). It is read at three instants:
//   0  before any cache exists   (has_zones only)
//   1  after the constructor
//   4  inside cache destruction, once arena_destroy() has returned, through the
//      destroy observer, which reads the members arena_backing() and zone_backed()
//
// Expected, as (ggml_sycl_arena_backing, ggml_sycl_device_has_zones):
//   USM mock    0: (-, false)   1: (USM, true)    4: (USM, false)
//   NONE mock   0: (-, false)   1: (NONE, false)  4: (NONE, false)
// The NONE mock runs in its own process: vram_arena_enabled() is read once.
//
// Usage:
//   ./build/bin/test-sycl-arena-accessors                  # every case
//   ./build/bin/test-sycl-arena-accessors none-child       # NONE mock (GGML_SYCL_VRAM_ARENA=0)
//   ./build/bin/test-sycl-arena-accessors strict-backing-child
//   ./build/bin/test-sycl-arena-accessors strict-committed-child
//   ./build/bin/test-sycl-arena-accessors strict-to-commit-child

#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace ggml_sycl;

namespace {

constexpr int    EXIT_SKIP   = 77;
constexpr size_t MOCK_BUDGET = 320ULL * 1024 * 1024;

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

std::vector<std::string> g_lines;

// Records every line the library logs, and still echoes it so a child's parent
// can read it from the pipe.
void capture_log(ggml_log_level, const char * text, void *) {
    g_lines.emplace_back(text);
    std::fputs(text, stderr);
}

size_t count_lines(const char * needle) {
    size_t n = 0;
    for (const auto & line : g_lines) {
        if (line.find(needle) != std::string::npos) {
            ++n;
        }
    }
    return n;
}

struct destroy_record {
    std::atomic<int>             calls{ 0 };
    int                          device      = -1;
    ggml_sycl_arena_backing_type backing     = GGML_SYCL_ARENA_BACKING_TYPE_VM;
    bool                         zone_backed = true;
};

destroy_record g_destroy;

void observe_destroy(int device, ggml_sycl_arena_backing_type backing, bool zone_backed) {
    g_destroy.device      = device;
    g_destroy.backing     = backing;
    g_destroy.zone_backed = zone_backed;
    g_destroy.calls.fetch_add(1);
}

constexpr int MOCK_DEVICE = 0;

// The mock's environment, set before anything reads it. A 1 MiB compute and
// runtime arena keep the zone tail small enough for the 320 MiB budget.
void set_mock_environment() {
    setenv("GGML_SYCL_UNIFIED_CACHE_MODE", "per_device", 1);
    setenv("GGML_SYCL_COMPUTE_ARENA_MB", "1", 1);
    setenv("GGML_SYCL_RUNTIME_ARENA_MB", "1", 1);
    unsetenv("GGML_SYCL_NONFA_ATTN_SCRATCH_MB");
    set_unified_cache_budget(MOCK_BUDGET);
}

// Instant 0: no cache exists, and reading has_zones creates none.
void check_instant_0() {
    CHECK(!unified_cache_test_cache_exists(MOCK_DEVICE), "instant 0: no cache before the read");
    CHECK(!ggml_sycl_device_has_zones(MOCK_DEVICE), "instant 0: no cache has no zones");
    CHECK(!unified_cache_test_cache_exists(MOCK_DEVICE), "instant 0: the lookup created nothing");
}

unified_cache * make_mock(sycl::queue & q) {
    unified_cache * cache = unified_cache_register_for_queue(MOCK_DEVICE, q);
    if (cache == nullptr) {
        std::fprintf(stderr, "FAIL: the mock cache was not created\n");
        std::exit(1);
    }
    return cache;
}

// Instant 4: destroy the cache and read what the observer saw inside its
// destruction.
void check_instant_4(ggml_sycl_arena_backing_type want_backing) {
    unified_cache_test_set_destroy_observer(observe_destroy);
    CHECK(shutdown_unified_cache(), "instant 4: the cache shut down");
    unified_cache_test_set_destroy_observer(nullptr);
    CHECK_EQ(g_destroy.calls.load(), 1, "instant 4: the observer ran once, inside destruction");
    CHECK_EQ(g_destroy.device, MOCK_DEVICE, "instant 4: the observer names the cache's device");
    CHECK_EQ(g_destroy.backing, want_backing, "instant 4: the member backing survives arena_destroy()");
    CHECK(!g_destroy.zone_backed, "instant 4: arena_destroy() left no backed zone");
    CHECK(!unified_cache_test_cache_exists(MOCK_DEVICE), "instant 4: the cache is gone");
    CHECK(!ggml_sycl_device_has_zones(MOCK_DEVICE), "instant 4: a destroyed cache has no zones");
}

int none_child(sycl::queue & q) {
    ggml_log_set(capture_log, nullptr);
    set_mock_environment();
    check_instant_0();
    (void) make_mock(q);
    CHECK_EQ(ggml_sycl_arena_backing(MOCK_DEVICE), GGML_SYCL_ARENA_BACKING_TYPE_NONE, "NONE 1: backing");
    CHECK(!ggml_sycl_device_has_zones(MOCK_DEVICE), "NONE 1: no arena, no zones");
    check_instant_4(GGML_SYCL_ARENA_BACKING_TYPE_NONE);
    if (g_failures != 0) {
        std::fprintf(stderr, "none-child: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("none-child: all ok\n");
    return 0;
}

// A USM mock whose arena was admitted at the budget the arm pre-registers, or
// an exit that says the precondition did not hold (VOID, never FAIL).
unified_cache * make_usm_mock(sycl::queue & q) {
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    std::printf("max_mem_alloc_size=%zu bytes\n", max_alloc);
    // The single-chunk arena needs the per-chunk cap (0.95 x max_alloc) at or
    // above the budget; a smaller cap splits it and the size below differs.
    if (max_alloc / 100 * 95 < MOCK_BUDGET) {
        std::fprintf(stderr, "VOID: max_mem_alloc_size %zu is below the single-chunk precondition\n", max_alloc);
        std::exit(EXIT_SKIP);
    }
    return make_mock(q);
}

// The capacity accessors off VM: the plan-defect line, the zone_capacity value.
void test_capacity_accessors(unified_cache * cache) {
    // A zone the mock sized, so a value read through the wrong path is not 0 == 0.
    const vram_zone_id zone     = vram_zone_id::SCRATCH;
    const size_t       capacity = cache->zone_capacity(zone);
    CHECK(capacity > 0, "capacity: the mock's SCRATCH zone is non-empty");

    const size_t committed_before = count_lines("[VM-PLAN-BUG] zone_capacity_committed read on a non-VM device");
    CHECK_EQ(cache->zone_capacity_committed(zone), capacity, "capacity: committed reads zone_capacity off VM");
    CHECK_EQ(count_lines("[VM-PLAN-BUG] zone_capacity_committed read on a non-VM device") - committed_before, 1,
             "capacity: committed printed its line once");

    const size_t to_commit_before = count_lines("[VM-PLAN-BUG] zone_capacity_to_commit read on a non-VM device");
    CHECK_EQ(cache->zone_capacity_to_commit(zone), capacity, "capacity: to_commit reads zone_capacity off VM");
    CHECK_EQ(count_lines("[VM-PLAN-BUG] zone_capacity_to_commit read on a non-VM device") - to_commit_before, 1,
             "capacity: to_commit printed its line once");
}

// ggml_sycl_arena_backing with no cache: the line, and NONE.
void test_backing_without_cache() {
    const size_t before = count_lines("[VM-PLAN-BUG] ggml_sycl_arena_backing read with no cache for device");
    CHECK_EQ(ggml_sycl_arena_backing(-1), GGML_SYCL_ARENA_BACKING_TYPE_NONE, "no cache: NONE");
    CHECK_EQ(count_lines("[VM-PLAN-BUG] ggml_sycl_arena_backing read with no cache for device") - before, 1,
             "no cache: the line printed once");
}

int usm_main(sycl::queue & q) {
    ggml_log_set(capture_log, nullptr);
    set_mock_environment();
    check_instant_0();

    unified_cache * cache = make_usm_mock(q);
    CHECK_EQ(cache->budget(), MOCK_BUDGET, "USM 1: the budget is the one the arm set");
    CHECK_EQ(count_lines("[VRAM-ARENA] Active on device 0: 1 chunk(s), 320.0 MB total"), 1,
             "USM 1: the early reserve printed its Active line (a failed reserve would WARN instead)");
    CHECK_EQ(cache->arena_total_size(), MOCK_BUDGET, "USM 1: the arena is the whole budget, in one chunk");
    CHECK_EQ(ggml_sycl_arena_backing(MOCK_DEVICE), GGML_SYCL_ARENA_BACKING_TYPE_USM, "USM 1: backing");
    CHECK(ggml_sycl_device_has_zones(MOCK_DEVICE), "USM 1: the reserved arena backs the zones");
    CHECK(unified_cache_test_cache_exists(MOCK_DEVICE), "USM 1: the cache exists");
    CHECK_EQ(cache->arena_backing(), ggml_sycl_arena_backing(MOCK_DEVICE),
             "USM 1: the free function returns the member");
    CHECK_EQ(cache->zone_backed(), ggml_sycl_device_has_zones(MOCK_DEVICE), "USM 1: has_zones reads the member");

    test_capacity_accessors(cache);
    test_backing_without_cache();
    check_instant_4(GGML_SYCL_ARENA_BACKING_TYPE_USM);
    return g_failures;
}

int strict_backing_child() {
    // Without per_device a direct run resolves to GLOBAL, where every id maps to cache 0 and -1 is
    // served instead of refused.
    setenv("GGML_SYCL_UNIFIED_CACHE_MODE", "per_device", 1);
    ggml_log_set(capture_log, nullptr);
    (void) ggml_sycl_arena_backing(-1);
    std::printf("strict-child: returned without aborting\n");
    return 0;
}

int strict_capacity_child(sycl::queue & q, bool to_commit) {
    ggml_log_set(capture_log, nullptr);
    set_mock_environment();
    unified_cache * cache = make_usm_mock(q);
    if (to_commit) {
        (void) cache->zone_capacity_to_commit(vram_zone_id::SCRATCH);
    } else {
        (void) cache->zone_capacity_committed(vram_zone_id::SCRATCH);
    }
    std::printf("strict-child: returned without aborting\n");
    return 0;
}

// Runs this binary again with `env` and `child`, returning its output and status.
std::string run_child(const char * self, const char * env, const char * child, int & status) {
    const std::string cmd = std::string(env) + " " + self + " " + child + " 2>&1";
    FILE *            p   = popen(cmd.c_str(), "r");
    CHECK(p != nullptr, "child started");
    if (!p) {
        status = -1;
        return {};
    }
    std::string out;
    char        buf[512];
    while (std::fgets(buf, sizeof(buf), p)) {
        out += buf;
    }
    status = pclose(p);
    return out;
}

void test_strict_aborts(const char * self, const char * child, const char * line) {
    int               status = 0;
    const std::string out    = run_child(self, "GGML_SYCL_STRICT_LEASES=1", child, status);
    CHECK(status != 0, "strict: the child did not exit 0");
    CHECK(out.find(line) != std::string::npos, "strict: the child printed the plan-bug line");
    CHECK(out.find("returned without aborting") == std::string::npos, "strict: the child aborted at the read");
    if (status == 0 || out.find(line) == std::string::npos) {
        std::fprintf(stderr, "strict: %s printed:\n%s", child, out.c_str());
    }
}

void test_none_process(const char * self) {
    int               status = 0;
    const std::string out    = run_child(self, "GGML_SYCL_VRAM_ARENA=0", "none-child", status);
    CHECK_EQ(status, 0, "NONE: the child exited 0");
    CHECK(out.find("none-child: all ok") != std::string::npos, "NONE: the child passed every row");
    if (status != 0) {
        std::fprintf(stderr, "NONE: child printed:\n%s", out.c_str());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    // A bare run must not reach a GPU: no code here enumerates one, and the
    // selector keeps the backend from doing it on this process's behalf. A plain
    // setenv() here is too late (oneCCL makes libsycl memoize the selector before
    // main()), so this re-execs; under ctest the registration already sets it.
    sycl_test_selector_fallback(argv, "opencl:cpu");
    sycl::queue q{ sycl::cpu_selector_v };

    const char * mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "none-child") == 0) {
        return none_child(q);
    }
    if (std::strcmp(mode, "strict-backing-child") == 0) {
        return strict_backing_child();
    }
    if (std::strcmp(mode, "strict-committed-child") == 0) {
        return strict_capacity_child(q, false);
    }
    if (std::strcmp(mode, "strict-to-commit-child") == 0) {
        return strict_capacity_child(q, true);
    }

    // strict_lease_checks_enabled() latches this variable on first read; an exported value would abort
    // the non-STRICT arms below before the STRICT children run.
    unsetenv("GGML_SYCL_STRICT_LEASES");
    usm_main(q);
    test_none_process(argv[0]);
    test_strict_aborts(argv[0], "strict-backing-child",
                       "[VM-PLAN-BUG] ggml_sycl_arena_backing read with no cache for device");
    test_strict_aborts(argv[0], "strict-committed-child",
                       "[VM-PLAN-BUG] zone_capacity_committed read on a non-VM device");
    test_strict_aborts(argv[0], "strict-to-commit-child",
                       "[VM-PLAN-BUG] zone_capacity_to_commit read on a non-VM device");

    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-arena-accessors: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-arena-accessors: all ok\n");
    return 0;
}
