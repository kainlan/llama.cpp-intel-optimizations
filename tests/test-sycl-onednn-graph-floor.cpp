// Host-only gate for llama.cpp-0oxf: the shape-derived Graph-scratch zone
// floor formula in unified-cache.cpp's onednn_graph_scratch_zone_floor_bytes().
//
// No SYCL device is needed -- this is a pure function of an env var and
// three integers, tested through its exported wrapper
// (ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(); the real function
// has internal linkage, see its declaration in unified-cache.hpp).
//
// WHY THIS EXISTS, AND WHY THE FORMULA TAKES THREE ARGUMENTS NOT ONE. The
// pre-0oxf floor was a flat 64 MiB. A first revision of this fix anchored a
// floor to n_ctx alone, which turned out to be the wrong independent
// variable: measurement (task llama.cpp-0oxf, comment c-xcop) showed the
// oneDNN Graph-scratch request is the PEAK OUTSTANDING size across however
// many compiled SDPA partitions are concurrently alive, proportional to
// n_head x n_ubatch x n_ctx, not n_ctx in isolation --
//   graph_peak = c * n_head * n_ubatch * n_ctx * sizeof(f32), c = 1.5
// (c=1.5, not 1.0, because ~5 SDPA scratch buffers were measured
// concurrently in flight even on an idle host). Five Mistral 7B Q4_0
// (n_head=32) measurements matched this to the exact byte across two
// different axes (n_ubatch AND n_ctx varied independently) -- see the test
// cases below, which reproduce all five.
//
// TWO PROCESSES, NOT TWO MODES IN ONE. onednn_graph_scratch_zone_floor_bytes()
// memoizes GGML_SYCL_ONEDNN_GRAPH_ZONE_MB via a function-local `static const`
// on its FIRST call in the process, so "env unset" and "env set" cannot both
// be exercised in one invocation -- whichever happens first wins for the rest
// of the process. --mode=default (no override in the test's own environment)
// covers the formula itself; --mode=override (registered with
// GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=99 in its ctest ENVIRONMENT) covers the
// escape hatch. See tests/CMakeLists.txt for the two registrations against
// this one binary.

#include "ggml-sycl/unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL/GGML_SYCL_DNNL not both enabled; no gate was evaluated.\n");
    return 77;  // ctest SKIP_RETURN_CODE
}
#else

using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t kMiB = 1024ull * 1024ull;

void test_default_formula() {
    printf("Shape-derived floor (no override):\n");

    // The floor at an all-zero shape (no model planned yet) must still
    // respect the historical 64 MiB minimum -- 0 * anything == 0, which the
    // max(64 MiB, ...) half of the formula must catch.
    const size_t floor_zero = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(0, 0, 0);
    check(floor_zero == 64 * kMiB, "floor(0, 0, 0) is exactly the 64 MiB minimum");

    // The five Mistral 7B Q4_0 (n_head=32) measurements this formula is
    // fit to (task llama.cpp-0oxf, comment c-xcop) -- varying n_ubatch AND
    // n_ctx independently. Three match the RAW formula to the exact byte;
    // the other two (512x512, 128x2048) compute to a raw 48 MiB, BELOW the
    // 64 MiB historical minimum, so their expectation here is the CLAMPED
    // 64 MiB, not the raw 48 -- verified against the ticket's own log
    // (comment c-xcop measured 48 MiB at both, i.e. the clamp already
    // matches the real allocator's behavior, not just this formula in
    // isolation). Keeping both clamped rows (rather than replacing one with
    // an unclamped duplicate) is deliberate: it is what keeps
    // max(64 MiB, ...) itself under test, not just the multiplication. If
    // any of these ever fails, either the measurement was deliberately
    // superseded (update it with a citation) or the formula regressed.
    struct case_t {
        uint32_t     n_head;
        uint32_t     n_ubatch;
        uint32_t     n_ctx;
        size_t       expect_mib;
        const char * what;
    };

    const case_t cases[] = {
        { 32, 512, 512,  64,  "Mistral (n_head=32) @ ubatch=512 ctx=512 -> raw 48 MiB, clamped to 64 MiB"  },
        { 32, 512, 2048, 192, "Mistral (n_head=32) @ ubatch=512 ctx=2048 -> 192 MiB"                       },
        { 32, 512, 8192, 768, "Mistral (n_head=32) @ ubatch=512 ctx=8192 -> 768 MiB"                       },
        { 32, 256, 2048, 96,  "Mistral (n_head=32) @ ubatch=256 ctx=2048 -> 96 MiB"                        },
        { 32, 128, 2048, 64,  "Mistral (n_head=32) @ ubatch=128 ctx=2048 -> raw 48 MiB, clamped to 64 MiB" },
    };
    for (const case_t & c : cases) {
        const size_t got = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(c.n_head, c.n_ubatch, c.n_ctx);
        check(got == c.expect_mib * kMiB, c.what);
    }

    // Monotonic in each axis independently -- a structural property the
    // exact-byte cases above don't directly exercise pairwise.
    const size_t floor_ub256 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 256, 2048);
    const size_t floor_ub512 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 512, 2048);
    check(floor_ub256 < floor_ub512, "the floor increases with n_ubatch at fixed n_head/n_ctx");
    const size_t floor_ctx2k = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 512, 2048);
    const size_t floor_ctx8k = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 512, 8192);
    check(floor_ctx2k < floor_ctx8k, "the floor increases with n_ctx at fixed n_head/n_ubatch");

    // A tiny shape must not undercut the historical 64 MiB floor.
    const size_t floor_tiny = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(1, 1, 1);
    check(floor_tiny >= 64 * kMiB, "floor(1, 1, 1) still respects the 64 MiB minimum");
}

void test_override() {
    printf("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB override:\n");

    // Set by ctest's ENVIRONMENT for the *-override registration; assert it
    // is visible before relying on it, so a misconfigured registration fails
    // loudly here instead of silently testing the default formula twice.
    const char * env = std::getenv("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB");
    check(env != nullptr && std::strcmp(env, "99") == 0,
          "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=99 is set (ctest ENVIRONMENT) before the first call");

    // The override must win regardless of shape -- it is an escape hatch for
    // "the formula is wrong for my workload right now", not a formula input.
    const size_t floor_small = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(0, 0, 0);
    const size_t floor_large = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(128, 4096, 1u << 20);
    check(floor_small == 99 * kMiB, "override wins at an all-zero shape");
    check(floor_large == 99 * kMiB, "override wins at a huge shape too (the formula is bypassed entirely)");
}

}  // namespace

int main(int argc, char ** argv) {
    const char * mode = "default";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        }
    }

    if (std::strcmp(mode, "override") == 0) {
        test_override();
    } else {
        test_default_formula();
    }

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
