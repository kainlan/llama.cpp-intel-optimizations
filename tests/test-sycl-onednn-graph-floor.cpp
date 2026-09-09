// Host-only gate for llama.cpp-0oxf/o3a0: the shape-derived Graph-scratch
// zone floor formula in unified-cache.cpp's
// onednn_graph_scratch_zone_floor_bytes_swa() (and the 3-arg no-SWA overload
// onednn_graph_scratch_zone_floor_bytes() that now delegates to it).
//
// No SYCL device is needed -- this is a pure function of an env var and a
// handful of integers, tested through its exported wrappers
// (ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes() and
// ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa(); the real
// functions have internal linkage, see their declarations in
// unified-cache.hpp).
//
// WHY THIS EXISTS, AND WHY THE FORMULA TAKES THE ARGUMENTS IT DOES. The
// pre-0oxf floor was a flat 64 MiB. A first revision of that fix anchored a
// floor to n_ctx alone, which turned out to be the wrong independent
// variable: measurement (task llama.cpp-0oxf, comment c-xcop) showed the
// oneDNN Graph-scratch request is the PEAK OUTSTANDING size across however
// many compiled SDPA partitions are concurrently alive, proportional to
// n_head x n_ubatch x ne11 (the KV length a partition actually attends
// over), not n_ctx in isolation --
//   graph_peak = c * n_head * n_ubatch * ne11 * sizeof(f32), c = 1.5
// (c=1.5, not 1.0, because ~5 SDPA scratch buffers were measured
// concurrently in flight even on an idle host). Five Mistral 7B Q4_0
// (n_head=32, no SWA so ne11 == n_ctx) measurements matched this to the
// exact byte across two different axes (n_ubatch AND n_ctx varied
// independently) -- see the test cases below, which reproduce all five.
//
// llama.cpp-o3a0 THEN SPLIT ne11 BY ATTENTION CLASS. A second revision
// (0oxf) still used n_ctx as ne11 for every oneDNN-served layer, which
// over-provisions a sliding-window (SWA) model: a SWA layer's real ne11 is
// min(n_ctx, n_swa), not n_ctx. The original hypothesis was gemma4 E4B
// measured 24 MB at n_ctx=8192 because its window was 1024, against the
// flat n_ctx formula's 192 MB prediction -- ⚠️ THAT WINDOW VALUE WAS WRONG
// (GPU-verified on the B50): gemma4 E4B's real GGUF
// attention.sliding_window is 512, at which this formula predicts 12 MiB
// (see test_swa_formula()'s "gemma4 E4B (real)" row below). The swa suite
// (test_swa_formula()) exercises the two-class max(ctx_term, swa_term)
// formula this ticket introduced using both the real gemma4 shape and
// several n_swa=1024 shapes kept purely as formula/arithmetic tests (not
// gemma4's actual value).
//
// TWO PROCESSES, NOT TWO MODES IN ONE. onednn_graph_scratch_zone_floor_bytes_swa()
// memoizes GGML_SYCL_ONEDNN_GRAPH_ZONE_MB via a function-local `static const`
// on its FIRST call in the process (both the 3-arg and 5-arg wrappers share
// this one memoization, since the 3-arg overload delegates straight into the
// 5-arg formula), so "env unset" and "env set" cannot both be exercised in
// one invocation -- whichever happens first wins for the rest of the
// process. --mode=default (no override in the test's own environment)
// covers the formula itself; --mode=override (registered with
// GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=99 in its ctest ENVIRONMENT) covers the
// escape hatch. See tests/CMakeLists.txt for the two registrations against
// this one binary.

#include "ggml-sycl/unified-cache.hpp"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL/GGML_SYCL_DNNL not both enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes;
using ggml_sycl::ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa;

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

    // Precondition, mirroring test_override()'s own check below: this mode
    // must run WITHOUT an override in the environment, or every case in
    // this function is really testing the escape hatch instead of the
    // formula. Empty counts as unset (onednn_graph_scratch_zone_floor_bytes()'s
    // own env == unset test, unified-cache.cpp).
    const char * zone_mb_env = std::getenv("GGML_SYCL_ONEDNN_GRAPH_ZONE_MB");
    check(zone_mb_env == nullptr || zone_mb_env[0] == '\0',
          "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB is unset (or empty) before the first call -- "
          "otherwise every case below tests the override, not the formula");

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

// llama.cpp-o3a0: the window-aware 5-arg formula
// (onednn_graph_scratch_zone_floor_bytes_swa()) that the 3-arg suite above
// now delegates through with an empty SWA class. Shares the same
// GGML_SYCL_ONEDNN_GRAPH_ZONE_MB memoization as test_default_formula() (one
// underlying static per process either way), so it must run in the same
// --mode=default invocation, not a separate one.
void test_swa_formula() {
    printf("Window-aware shape-derived floor (no override):\n");

    struct case_t {
        uint32_t     n_head_ctx_max;
        uint32_t     n_head_swa_max;
        uint32_t     n_swa;
        uint32_t     n_ubatch;
        uint32_t     n_ctx;
        size_t       expect_mib;
        const char * what;
    };

    const case_t cases[] = {
        // gemma4 E4B (REAL shape, GPU-verified on the B50): D=512
        // global layers are not oneDNN-eligible by default
        // (n_head_ctx_max=0); its SWA layers have n_head_swa_max=8 and the
        // model's actual GGUF attention.sliding_window is 512 (NOT the
        // 1024 this ticket originally assumed -- see the row below, kept
        // as a formula/arithmetic test under its honest hypothetical
        // label, and the corrected history in
        // onednn_graph_scratch_zone_floor_bytes_swa()'s own comment,
        // unified-cache.cpp). Raw:
        // 1.5 x 8 x 512 x min(8192, 512) x 4 B == 12 MiB exactly, BELOW
        // the 64 MiB minimum -- clamps to 64 MiB, not the 192 MiB the
        // pre-o3a0 flat n_ctx formula predicted at this shape (see
        // test_default_formula()'s Mistral row at the same n_ubatch/n_ctx
        // for that 192 MiB figure).
        { 0,  8,  512,  512, 8192, 64,
         "gemma4 E4B (real: n_head_swa_max=8, n_swa=512) @ ubatch=512 ctx=8192 -> raw 12 MiB, clamped to 64 MiB"    },
        // A HYPOTHETICAL n_swa=1024 shape -- NOT gemma4's actual value
        // (see the row above and the ⚠️ correction in this file's header
        // comment) -- kept as a formula/arithmetic test: it independently
        // exercises the same min(n_ctx, n_swa) clamp-vs-window arithmetic
        // at a different n_swa, still below the 64 MiB minimum.
        // 1.5 x 8 x 512 x min(8192, 1024) x 4 B == 24 MiB exactly.
        { 0,  8,  1024, 512, 8192, 64,
         "hypothetical (n_head_swa_max=8, n_swa=1024, NOT gemma4's real value) -> raw 24 MiB, clamped to 64 MiB"    },
        // n_ctx < n_swa: the window never binds, so the SWA class's
        // effective ne11 must fall back to n_ctx, not the (larger) n_swa --
        // this must equal the non-SWA formula at the identical (n_head,
        // n_ubatch, n_ctx), matched below via the 3-arg overload.
        { 0,  32, 8192, 512, 2048, 192,
         "n_ctx(2048) < n_swa(8192): SWA class effective window is n_ctx, matching the non-SWA formula"             },
        // Both classes present; the ctx class dominates (32 heads over the
        // full 8192 ctx beats 8 heads over a 1024 window) -- the floor must
        // take the MAX across classes, not their sum (which would double
        // the Mistral 768 MiB figure this matches).
        { 32, 8,  1024, 512, 8192, 768,
         "both classes present, ctx class dominates -> matches the ctx-only Mistral figure, not ctx+swa summed"     },
        // Both classes present, SWA class dominates this time (64 heads over
        // a 1024 window beats 1 head over the full 1024 ctx) -- same MAX
        // requirement, opposite class winning. An earlier version of this
        // row used n_head_swa_max=8, which put BOTH
        // the correct answer (24 MiB raw) and the wrong ctx-only answer
        // (3 MiB raw) below the 64 MiB clamp -- a VOID positive control,
        // since either formula produces the identical clamped 64 MiB and
        // the row could never distinguish them. n_head_swa_max=64 pushes
        // the correct answer to 192 MiB, well above the clamp, so a
        // formula that silently picked the ctx term (3 MiB raw, also
        // clamped to 64 MiB) would visibly diverge from this row's
        // expectation instead of coincidentally matching it.
        { 1,  64, 1024, 512, 1024, 192,
         "both classes present, swa class dominates -> raw 3 MiB from ctx alone would be wrong; correct is 192 MiB" },
    };
    for (const case_t & c : cases) {
        const size_t got = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa(c.n_head_ctx_max, c.n_head_swa_max,
                                                                                    c.n_swa, c.n_ubatch, c.n_ctx);
        check(got == c.expect_mib * kMiB, c.what);
    }

    // Cross-check: n_ctx < n_swa must produce the IDENTICAL result to the
    // no-SWA 3-arg overload at the same (n_head, n_ubatch, n_ctx) -- both
    // describe "this class's effective window is n_ctx".
    const size_t swa_below_window  = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa(0, 32, 8192, 512, 2048);
    const size_t no_swa_equivalent = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 512, 2048);
    check(swa_below_window == no_swa_equivalent,
          "n_ctx < n_swa matches the no-SWA overload exactly (both use n_ctx as the effective window)");

    // An empty SWA class (n_head_swa_max=0, n_swa=0) must reproduce the
    // no-SWA overload exactly, for every Mistral row above -- this is the
    // backward-compatibility contract the 3-arg overload's thin-wrapper
    // delegation depends on.
    const size_t via_swa_empty_class = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa(32, 0, 0, 512, 8192);
    const size_t via_no_swa_overload = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes(32, 512, 8192);
    check(via_swa_empty_class == via_no_swa_overload,
          "an empty SWA class (n_head_swa_max=0, n_swa=0) matches the no-SWA overload exactly");
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

    // llama.cpp-o3a0: the override is one memoized static shared by both the
    // 3-arg and 5-arg formulas (the 3-arg overload delegates into the 5-arg
    // one) -- confirm it also wins through the 5-arg wrapper directly,
    // rather than only through the delegation path exercised above.
    const size_t floor_swa = ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes_swa(128, 64, 4096, 4096, 1u << 20);
    check(floor_swa == 99 * kMiB, "override wins through the 5-arg swa wrapper too, not just via delegation");
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
        test_swa_formula();
    } else {
        // An unrecognised --mode= value used to silently fall through to
        // the default suite, which would mask a typo'd ctest registration
        // ARGS as a passing (but wrong) test run rather than a usage error.
        std::fprintf(stderr, "usage: %s [--mode=default|--mode=override] (got --mode=%s)\n", argv[0], mode);
        return 2;
    }

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL && GGML_SYCL_DNNL
