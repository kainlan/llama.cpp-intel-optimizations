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
#include <limits>

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch;
using ggml_sycl::unified_cache_nonfa_attn_outside_arena_reserve_bytes;
using ggml_sycl::unified_cache_nonfa_attn_scratch_demand_bytes;
using ggml_sycl::unified_cache_nonfa_attn_scratch_fits_headroom;

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

    // A shape chosen so that n_head * n_ubatch alone (4294967296) exceeds
    // UINT32_MAX (4294967295) by 1 -- a 32-bit intermediate product would
    // silently wrap to 0 here, not merely produce a large-but-wrong number,
    // which is exactly the kind of bug a "the formula is monotonic" check
    // above cannot catch (0 is not obviously wrong on its own; it just looks
    // like a tiny shape). The implementation promotes every operand to
    // uint64_t before multiplying (unified-cache.cpp), so this must compute
    // the exact product, not silently wrap.
    const size_t demand_huge = unified_cache_nonfa_attn_scratch_demand_bytes(65536, 65536, 1);
    check(demand_huge == 24576ull * kMiB,
          "demand(65536, 65536, 1) is exactly 24576 MiB, not truncated by a 32-bit intermediate "
          "(n_head * n_ubatch alone overflows uint32_t)");
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

void test_override_zero() {
    printf("GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0 (explicit-disable, not \"unset\"):\n");

    // env_mb_override() (unified-cache.cpp) treats "0" as a genuine parsed
    // value (env_mb == 0), distinct from an unset/empty var (env_mb == -1,
    // which falls through to the formula). A caller with this override set
    // gets EXACTLY 0 bytes back -- the 16 MiB kNonfaAttnScratchFloorBytes
    // clamp is part of the FORMULA branch only and is bypassed entirely
    // once an override (any non-negative value, including 0) is in effect,
    // the same way a 77 MiB override above is not itself clamped to 16 MiB.
    const char * env = std::getenv("GGML_SYCL_NONFA_ATTN_SCRATCH_MB");
    check(env != nullptr && std::strcmp(env, "0") == 0,
          "GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0 is set (ctest ENVIRONMENT) before the first call");

    const size_t demand_zero_shape = unified_cache_nonfa_attn_scratch_demand_bytes(0, 0, 0);
    const size_t demand_real_shape = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 8192);
    check(demand_zero_shape == 0, "override=0 returns exactly 0 bytes at an all-zero shape (not the 16 MiB floor)");
    check(demand_real_shape == 0,
          "override=0 returns exactly 0 bytes even at the B50 repro shape "
          "(the override bypasses the formula and its floor entirely)");
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

    // Floor-vs-inverse interaction, scored to the outcome the inverse
    // function's own comment documents: the inverse IGNORES
    // kNonfaAttnScratchFloorBytes, so for a zone capacity small enough that
    // the raw (unfloored) formula at the reported n_ctx is still below the
    // 16 MiB floor, the inverse's own "fits" answer is optimistic --
    // feeding it back into the forward formula (which DOES apply the floor)
    // reports a demand larger than the zone capacity the inverse was asked
    // about. n_head=1, n_ubatch=1 keeps the per-context-cell cost tiny (6
    // bytes), so a capacity of just 2000 bytes still rounds up to a
    // non-zero, 256-aligned n_ctx rather than degenerating to the n_ctx=0
    // corner case the dedicated zero-capacity check above already covers.
    constexpr size_t kTinyZoneBytes = 2000;
    const uint32_t   fits_tiny_zone = unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(kTinyZoneBytes, 1, 1);
    check(fits_tiny_zone == 256,
          "a 2000-byte zone at n_head=1/n_ubatch=1 reports n_ctx=256 (the raw formula's "
          "own floor-free arithmetic), not the degenerate n_ctx=0 case");
    const size_t demand_at_fits_tiny_zone = unified_cache_nonfa_attn_scratch_demand_bytes(1, 1, fits_tiny_zone);
    check(demand_at_fits_tiny_zone == 16 * kMiB,
          "the forward formula's 16 MiB floor dominates at that reported n_ctx (raw modeled demand there is "
          "only 1536 bytes)");
    check(demand_at_fits_tiny_zone > kTinyZoneBytes,
          "the inverse's own \"fits\" answer does not actually fit once the forward formula's floor is applied "
          "-- exactly the caveat unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch()'s comment "
          "documents (it ignores kNonfaAttnScratchFloorBytes), so a caller must not treat its result as exact "
          "for a zone this small");
}

// llama.cpp-pvjr: the headroom predicate that replaced the SCRATCH-zone-
// capacity check -- refuse iff demand + reserve > free. Reserve and the
// four bracket points below are taken directly from task llama.cpp-pvjr's
// comments c-1hv7/c-gnyc (the sweep that decided R = 928 MiB); see this
// file's header comment and unified_cache_nonfa_attn_outside_arena_reserve_
// bytes()'s own derivation comment (unified-cache.cpp) for the full
// narrative.
void test_headroom_predicate() {
    printf("Non-FA attention scratch outside-arena headroom predicate:\n");

    const size_t reserve = unified_cache_nonfa_attn_outside_arena_reserve_bytes();
    check(reserve == 928 * kMiB, "the outside-arena reserve is exactly 928 MiB");

    // Measured live device free bytes at guard time (task c-1hv7/c-gnyc),
    // in the same "MB" == MiB convention the [SYCL-BUDGET] free= log line
    // uses -- NOT decimal megabytes.
    const size_t h_b50 = static_cast<size_t>(1627.3 * static_cast<double>(kMiB));
    const size_t h_b70 = static_cast<size_t>(2045.2 * static_cast<double>(kMiB));

    // B50: p7168 (d=672 MiB) ran clean guard-off; p8192 (d=768 MiB) aborted
    // with the VRAM exhausted (free=29 MB) -- both at n_head=32, n_ubatch=512.
    const size_t d_b50_7168 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 7168);
    const size_t d_b50_8192 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 8192);
    check(d_b50_7168 == 672 * kMiB, "sanity: demand(32, 512, 7168) is 672 MiB");
    check(d_b50_8192 == 768 * kMiB, "sanity: demand(32, 512, 8192) is 768 MiB");
    check(unified_cache_nonfa_attn_scratch_fits_headroom(d_b50_7168, h_b50),
          "B50: n_ctx=7168 (d=672 MiB) fits the measured 1627.3 MB headroom at R=928 MiB");
    check(!unified_cache_nonfa_attn_scratch_fits_headroom(d_b50_8192, h_b50),
          "B50: n_ctx=8192 (d=768 MiB) does NOT fit -- this is the shape that aborted on hardware");

    // B70: p11264 (d=1056 MiB) ran clean; p12288 (d=1152 MiB) exhausted
    // outside-arena VRAM 4 times, surviving only on the scalar fallback --
    // treated as refuse per the decided predicate.
    const size_t d_b70_11264 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 11264);
    const size_t d_b70_12288 = unified_cache_nonfa_attn_scratch_demand_bytes(32, 512, 12288);
    check(d_b70_11264 == 1056 * kMiB, "sanity: demand(32, 512, 11264) is 1056 MiB");
    check(d_b70_12288 == 1152 * kMiB, "sanity: demand(32, 512, 12288) is 1152 MiB");
    check(unified_cache_nonfa_attn_scratch_fits_headroom(d_b70_11264, h_b70),
          "B70: n_ctx=11264 (d=1056 MiB) fits the measured 2045.2 MB headroom at R=928 MiB");
    check(!unified_cache_nonfa_attn_scratch_fits_headroom(d_b70_12288, h_b70),
          "B70: n_ctx=12288 (d=1152 MiB) does NOT fit -- degraded on hardware (scalar fallback), "
          "treated as refuse");

    // Overflow: a demand within [0, SIZE_MAX - reserve] must not overflow;
    // above that, the helper must treat the sum as refuse rather than wrap
    // to a small value that would read as "fits".
    check(!unified_cache_nonfa_attn_scratch_fits_headroom(std::numeric_limits<size_t>::max(),
                                                          std::numeric_limits<size_t>::max()),
          "demand == SIZE_MAX refuses even against a SIZE_MAX free_bytes (demand + reserve would overflow)");
    check(!unified_cache_nonfa_attn_scratch_fits_headroom(std::numeric_limits<size_t>::max() - reserve + 1,
                                                          std::numeric_limits<size_t>::max()),
          "demand one byte past the exact overflow boundary (SIZE_MAX - reserve + 1) refuses even against "
          "a SIZE_MAX free_bytes");
    check(unified_cache_nonfa_attn_scratch_fits_headroom(std::numeric_limits<size_t>::max() - reserve,
                                                         std::numeric_limits<size_t>::max()),
          "demand exactly at the overflow boundary (SIZE_MAX - reserve) does not overflow and fits a "
          "SIZE_MAX free_bytes");

    // A card with less free memory than the reserve alone refuses any
    // positive demand, however small.
    check(!unified_cache_nonfa_attn_scratch_fits_headroom(1, reserve - 1),
          "free_bytes below the reserve alone refuses even a 1-byte demand");
    check(unified_cache_nonfa_attn_scratch_fits_headroom(0, reserve),
          "free_bytes exactly equal to the reserve fits a zero demand");

    // Largest-fitting n_ctx at capacity = H - R on both cards (the
    // headroom-limited remediation the runtime refusal reports). Computed
    // by hand: floor((H - R) / (n_head * n_ubatch * 6)) rounded down to the
    // nearest multiple of 256.
    //   B50: H=1627.3 MiB, R=928 MiB -> capacity ~= 733,269,196 B
    //        (~699.3 MiB); 733269196 / 98304 = 7459.86..., rounded down to
    //        the nearest multiple of 256 (7459 / 256 = 29.14...) -> 7424.
    const size_t capacity_b50 = h_b50 - reserve;
    check(unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(capacity_b50, 32, 512) == 7424,
          "B50 headroom-limited largest fitting n_ctx at R=928 MiB is 7424");

    //   B70: H=2045.2 MiB, R=928 MiB -> capacity ~= 1,171,469,107 B
    //        (~1117.2 MiB); 1171469107 / 98304 = 11916.4..., rounded down
    //        to the nearest multiple of 256 (11916 / 256 = 46.5...) -> 11776.
    const size_t capacity_b70 = h_b70 - reserve;
    check(unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(capacity_b70, 32, 512) == 11776,
          "B70 headroom-limited largest fitting n_ctx at R=928 MiB is 11776");
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
    } else if (std::strcmp(mode, "override-zero") == 0) {
        test_override_zero();
    } else if (std::strcmp(mode, "default") == 0) {
        test_default_formula();
        test_largest_fitting_n_ctx();
        test_headroom_predicate();
    } else {
        std::fprintf(stderr, "usage: %s [--mode=default|--mode=override|--mode=override-zero] (got --mode=%s)\n",
                     argv[0], mode);
        return 2;
    }

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
