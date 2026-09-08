// Host-only gate for llama.cpp-0oxf: the context-derived Graph-scratch zone
// floor formula in unified-cache.cpp's onednn_graph_scratch_zone_floor_bytes().
//
// No SYCL device is needed -- this is a pure function of an env var and an
// integer, tested through its exported wrapper
// (ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(); the real function
// has internal linkage, see its declaration in unified-cache.hpp).
//
// WHY THIS EXISTS. The pre-0oxf floor was a flat 64 MiB, which this ticket's
// bisect showed does not fit a real 144 MB request at n_kv=8192 (that request
// is exactly why the DIRECT path -- and the bug it could trigger -- exists at
// all). The replacement floor is context-derived: floor(n_ctx) = max(64 MiB,
// a + b*n_ctx), anchored so floor(8192) == 144 MiB (the one measured point;
// see the PROVISIONAL constants comment in unified-cache.cpp for why this is
// an anchor, not a fit, and what should replace it once the lead's
// pp2048/4096/8192 sweep exists).
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
    printf("Context-derived floor (no override):\n");

    const size_t floor_0 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(0);
    check(floor_0 == 64 * kMiB, "floor(n_ctx=0) is exactly the 64 MiB minimum");

    // The one anchor this formula is built from (see the PROVISIONAL
    // constants comment in unified-cache.cpp): a real 144 MB request was
    // measured at n_kv=8192. If this ever fails, either the anchor changed
    // (update it deliberately, with a new measurement) or the formula
    // regressed to something that no longer covers the request that
    // motivated this whole ticket.
    const size_t floor_8192 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(8192);
    check(floor_8192 == 144 * kMiB, "floor(n_ctx=8192) matches the measured 144 MB anchor exactly");

    const size_t floor_2048 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(2048);
    const size_t floor_4096 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(4096);
    check(floor_2048 >= 64 * kMiB, "floor(2048) never drops below the 64 MiB minimum");
    check(floor_2048 < floor_4096 && floor_4096 < floor_8192,
          "the floor is strictly increasing with n_ctx between the measured points");

    // A tiny n_ctx must not UNDERCUT the historical 64 MiB floor -- the
    // "max(64 MiB, ...)" half of the formula, independent of the slope.
    const size_t floor_1 = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(1);
    check(floor_1 >= 64 * kMiB, "floor(n_ctx=1) still respects the 64 MiB minimum");
}

void test_override() {
    printf("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB override:\n");

    // Set by ctest's ENVIRONMENT for the *-override registration; assert it
    // is visible before relying on it, so a misconfigured registration fails
    // loudly here instead of silently testing the default formula twice.
    const char * env = std::getenv("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB");
    check(env != nullptr && std::strcmp(env, "99") == 0,
          "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=99 is set (ctest ENVIRONMENT) before the first call");

    // The override must win regardless of n_ctx -- it is an escape hatch for
    // "the formula is wrong for my workload right now", not a formula input.
    const size_t floor_small = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(0);
    const size_t floor_large = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(1u << 20);
    check(floor_small == 99 * kMiB, "override wins at n_ctx=0");
    check(floor_large == 99 * kMiB, "override wins at a huge n_ctx too (the formula's slope is bypassed entirely)");
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
