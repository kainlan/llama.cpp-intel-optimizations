// Host-only gate for llama.cpp-oyfl: the non-flash-attention batched mul_mat
// scratch demand formula in unified-cache.cpp/hpp
// (unified_cache_nonfa_attn_scratch_demand_bytes() and its inverse,
// unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch()).
//
// No SYCL device is needed -- both are pure functions of an env var and a
// few integers. Modeled closely on tests/test-sycl-onednn-graph-floor.cpp,
// which gates the sibling oneDNN Graph-scratch floor this formula was
// derived by analogy from -- but this one is NOT gated behind
// GGML_SYCL_DNNL: the batched mul_mat path it sizes for is the native
// SYCL/oneMath route, unconditional in the backend.
//
// THE REPRO SHAPE THIS IS FIT TO. llama.cpp-oyfl's B50 repro
// (`llama-bench -m mistral-7b-v0.1.Q4_0.gguf -p 8192 -n 0 -r 1 -fa 0 -v`)
// aborted with "[SYCL] batched F16 mul_mat failed -- no recovery path
// available" (ggml-sycl.cpp, formerly line 62541) at n_head=32, n_ubatch=512,
// n_ctx=8192 -- Mistral 7B's query-head count, llama-bench's default ubatch,
// and the prompt length driving the KV/attention shape. The formula:
//   demand = n_head * n_ubatch * n_ctx * sizeof(f16) * 3
// (see the formula's derivation comment in unified-cache.cpp for why the
// KQV term alone is kept, and why c=3 is MEASURED from the repro log's own
// SCRATCH_ZONE occupancy at the moment of failure -- not an analogy to the
// oneDNN Graph-scratch floor's c=1.5, an earlier version of this formula's
// mistake). At the repro shape this computes to an EXACT 768 MiB, which is
// the anchor case below.

#include "ggml-sycl/unified-cache.hpp"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch;
using ggml_sycl::unified_cache_nonfa_attn_scratch_demand_bytes;

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
    printf("Non-FA attention scratch demand (no override):\n");

    // Precondition, same reasoning as test-sycl-onednn-graph-floor.cpp's own
    // check: the formula's env read is memoized on first call, so this mode
    // must run without an override in the environment or every case below
    // tests the escape hatch instead of the formula.
    const char * scratch_mb_env = std::getenv("GGML_SYCL_NONFA_ATTN_SCRATCH_MB");
    check(scratch_mb_env == nullptr || scratch_mb_env[0] == '\0',
          "GGML_SYCL_NONFA_ATTN_SCRATCH_MB is unset (or empty) before the first call -- "
          "otherwise every case below tests the override, not the formula");

    // All-zero shape (nothing planned yet) must still respect the 16 MiB
    // minimum -- 0 * anything == 0, which the max(16 MiB, ...) half must catch.
    const size_t demand_zero = unified_cache_nonfa_attn_scratch_demand_bytes(0, 0, 0);
    check(demand_zero == 16 * kMiB, "demand(0, 0, 0) is exactly the 16 MiB minimum");

    struct case_t {
        uint32_t     n_head;
        uint32_t     n_ubatch;
        uint32_t     n_ctx;
        size_t       expect_mib;
        const char * what;
    };

    const case_t cases[] = {
        // The B50 repro shape (llama.cpp-oyfl): exact.
        { 32, 512, 8192, 768, "Mistral B50 repro (n_head=32, n_ubatch=512, n_ctx=8192) -> 768 MiB" },
        { 32, 512, 2048, 192, "n_head=32, n_ubatch=512, n_ctx=2048 -> 192 MiB"                     },
        { 32, 256, 2048, 96,  "n_head=32, n_ubatch=256, n_ctx=2048 -> 96 MiB"                      },
        // Raw 48 MiB, above the 16 MiB floor -- exercises the formula, not the clamp.
        { 32, 512, 512,  48,  "n_head=32, n_ubatch=512, n_ctx=512 -> 48 MiB"                       },
        // Raw well below 16 MiB -- exercises the clamp.
        { 1,  1,   1,    16,  "n_head=1, n_ubatch=1, n_ctx=1 -> clamped to 16 MiB"                 },
    };
    for (const case_t & c : cases) {
        const size_t got = unified_cache_nonfa_attn_scratch_demand_bytes(c.n_head, c.n_ubatch, c.n_ctx);
        check(got == c.expect_mib * kMiB, c.what);
    }

    // Monotonic in each axis independently.
    const size_t demand_ub256 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 256, 2048);
    const size_t demand_ub512 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 2048);
    check(demand_ub256 < demand_ub512, "demand increases with n_ubatch at fixed n_head/n_ctx");
    const size_t demand_ctx2k = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 2048);
    const size_t demand_ctx8k = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 8192);
    check(demand_ctx2k < demand_ctx8k, "demand increases with n_ctx at fixed n_head/n_ubatch");
    const size_t demand_h8  = unified_cache_nonfa_attn_scratch_demand_bytes(8, 512, 2048);
    const size_t demand_h32 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 2048);
    check(demand_h8 < demand_h32, "demand increases with n_head at fixed n_ubatch/n_ctx");
}

void test_override() {
    printf("GGML_SYCL_NONFA_ATTN_SCRATCH_MB override:\n");

    const char * env = std::getenv("GGML_SYCL_NONFA_ATTN_SCRATCH_MB");
    check(env != nullptr && std::strcmp(env, "77") == 0,
          "GGML_SYCL_NONFA_ATTN_SCRATCH_MB=77 is set (ctest ENVIRONMENT) before the first call");

    const size_t demand_small = unified_cache_nonfa_attn_scratch_demand_bytes(0, 0, 0);
    const size_t demand_large = unified_cache_nonfa_attn_scratch_demand_bytes(128, 4096, 1u << 20);
    check(demand_small == 77 * kMiB, "override wins at an all-zero shape");
    check(demand_large == 77 * kMiB, "override wins at a huge shape too (the formula is bypassed entirely)");
}

void test_largest_fitting_n_ctx() {
    printf("Inverse: largest fitting n_ctx for a SCRATCH zone capacity:\n");

    // Exact round trip against the B50 repro shape's own demand: a zone
    // sized to exactly hold n_ctx=8192's demand must report 8192 back (it is
    // already a multiple of 256, so rounding does not perturb it).
    const size_t   repro_demand = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 8192);
    const uint32_t fits_exact   = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(repro_demand, 32, 512);
    check(fits_exact == 8192, "a zone sized exactly for n_ctx=8192's demand reports 8192 back");

    // The default 512 MiB SCRATCH zone (ensure_planned_arena_zones()'s
    // pre-existing floor) at the repro's n_head/n_ubatch fits LESS than
    // 8192 -- this is the concrete case the runtime-context-update guard
    // must refuse rather than let ggml_sycl_mul_mat_batched_sycl() abort on.
    // Exact value: 512 MiB / (32 * 512 * 6) = 5461.33, rounded down to the
    // nearest 256 = 5376.
    const uint32_t fits_default_zone = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(512 * kMiB, 32, 512);
    check(fits_default_zone == 5376,
          "the default 512 MiB SCRATCH zone fits exactly n_ctx=5376 at n_head=32/n_ubatch=512, below the "
          "repro's n_ctx=8192");

    // A zone capacity of 0 fits nothing.
    check(unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(0, 32, 512) == 0,
          "a zero-capacity zone fits no context");

    // n_head or n_ubatch of 0 is undefined for the formula -- must not divide
    // by zero, and must report "fits nothing" rather than a garbage large
    // value.
    check(unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(512 * kMiB, 0, 512) == 0,
          "n_head=0 reports 0 rather than dividing by zero");
    check(unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(512 * kMiB, 32, 0) == 0,
          "n_ubatch=0 reports 0 rather than dividing by zero");

    // Monotonic in zone capacity.
    const uint32_t fits_small = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(64 * kMiB, 32, 512);
    const uint32_t fits_large = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(256 * kMiB, 32, 512);
    check(fits_small < fits_large, "the largest fitting n_ctx increases with zone capacity");

    // The result is always a multiple of 256 (the same cell-rounding
    // convention the KV-side ggml_sycl_largest_fitting_n_ctx() uses).
    const uint32_t fits_odd = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(300 * kMiB, 32, 512);
    check(fits_odd % 256 == 0, "the largest fitting n_ctx is rounded down to a multiple of 256");
}

}  // namespace

int main(int argc, char ** argv) {
    constexpr char kModeFlag[] = "--mode=";

    const char * mode = "default";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], kModeFlag, sizeof(kModeFlag) - 1) == 0) {
            mode = argv[i] + sizeof(kModeFlag) - 1;
        }
    }

    if (std::strcmp(mode, "override") == 0) {
        test_override();
    } else if (std::strcmp(mode, "default") == 0) {
        test_default_formula();
        test_largest_fitting_n_ctx();
    } else {
        std::fprintf(stderr, "usage: %s [--mode=default|--mode=override] (got --mode=%s)\n", argv[0], mode);
        return 2;
    }

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
