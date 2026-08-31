// Host-only unit tests for zero_alloc_baseline_tracker (llama.cpp-dcx6). No
// GPU, no allocation, no SYCL runtime -- it drives the pure phase-transition
// state machine backing zero_alloc_check with plain (phase, bytes) sequences.
//
// Why this exists: the check's old design captured its baseline exactly
// once, on the process's first PP/TG call, and never re-established it.
// Every model this was measured against (gemma, Mistral, GPT-OSS) steps its
// non-weight arena footprint once during PP warm-up or the PP->TG
// transition and then holds flat -- but the frozen baseline meant that one
// expected step got reported as a fresh "runtime allocation detected" on
// every subsequent call for the rest of the run. The evidence was that the
// check's own `total=` field was bit-identical across dozens of consecutive
// warnings, proving no repeated allocation was actually happening.
//
// The fix re-baselines on every phase transition (including the very first
// observation) and only compares same-phase calls against the baseline that
// phase already established. These cases assert both halves: a transition
// must never warn by itself, and a real allocation appearing mid-phase must
// still be caught on the very next call.

#include "ggml-sycl/unified-cache.hpp"

#include <cstdio>

#if !defined(GGML_USE_SYCL)
int main() {
    // 77 (ctest SKIP_RETURN_CODE), not 0: no gate was evaluated in this
    // build, so this must not read as a pass.
    std::fprintf(stderr, "SKIP: GGML SYCL not enabled; no gate was evaluated.\n");
    return 77;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

using ggml_sycl::offload_phase;
using ggml_sycl::zero_alloc_baseline_tracker;

// The very first observation, in any phase, must establish the baseline
// silently. This is the "process just started PP" case -- nothing has run
// yet, so there is nothing to warn about.
static bool test_first_observation_never_warns() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;
    const bool                  warn  = t.observe(offload_phase::PP, 1000, delta);
    TEST_ASSERT(!warn, "first-ever observation must not warn");
    TEST_ASSERT(t.baseline == 1000, "first observation must set baseline to the observed bytes");
    return true;
}

// This is the regression this fix exists for: gemma's PP phase peaked at
// 8667.6 MB then settled to 8402.8 MB on entering TG (+6.0 MB over the
// *original* frozen baseline of 8396.8 MB, taken before PP even started).
// A transition into a new phase must re-baseline to the value observed on
// entry, not compare against a stale pre-existing number.
static bool test_phase_transition_rebaselines_without_warning() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;

    // Baseline for PP, then PP grows during warm-up (multiple ubatches).
    TEST_ASSERT(!t.observe(offload_phase::PP, 8396800000ull / 1000, delta), "PP entry must not warn");
    (void) t.observe(offload_phase::PP, 8667600000ull / 1000, delta);  // PP warm-up growth, same phase

    // Transition PP -> TG: total actually DROPS (8667.6 -> 8402.8 MB) but is
    // still above the ORIGINAL PP baseline. The old design would compare
    // against that stale PP baseline and warn forever; the fix must instead
    // treat this as phase entry and re-baseline silently.
    const bool warn_on_transition = t.observe(offload_phase::TG, 8402800, delta);
    TEST_ASSERT(!warn_on_transition, "PP->TG transition must not warn even though bytes exceed the old PP baseline");
    TEST_ASSERT(t.baseline == 8402800, "TG baseline must be the bytes observed on entering TG, not inherited from PP");
    return true;
}

// Once a phase's baseline is established, holding flat for many subsequent
// calls in that same phase must never warn. This is gemma/Mistral/GPT-OSS's
// actual steady-state TG behavior post-fix: the check should go silent.
static bool test_flat_steady_state_never_warns() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;
    (void) t.observe(offload_phase::TG, 8402800, delta);  // establishes baseline
    for (int i = 0; i < 50; ++i) {
        const bool warn = t.observe(offload_phase::TG, 8402800, delta);
        TEST_ASSERT(!warn, "identical bytes within an unchanged phase must never warn");
    }
    return true;
}

// The positive control: a REAL allocation appearing mid-phase (not at a
// transition) must still be caught, on the very next call. This is what
// distinguishes the fix from simply silencing the check -- it must still
// detect the failure mode it was built for.
static bool test_real_midphase_growth_still_warns() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;
    (void) t.observe(offload_phase::TG, 8402800, delta);  // establishes baseline, same phase throughout

    const bool warn = t.observe(offload_phase::TG, 8402800 + 6 * 1024 * 1024, delta);
    TEST_ASSERT(warn, "a genuine allocation within an unchanged phase must warn");
    TEST_ASSERT(delta == 6u * 1024 * 1024, "delta must be the bytes actually gained since this phase's baseline");

    // And it must keep growing call over call if the leak keeps happening --
    // this is the shape a real per-step leak has that the stale-baseline
    // artifact never did (its `total=` field was constant across calls).
    const bool warn2 = t.observe(offload_phase::TG, 8402800 + 12 * 1024 * 1024, delta);
    TEST_ASSERT(warn2, "a second consecutive per-step growth must warn again");
    TEST_ASSERT(delta == 12u * 1024 * 1024, "delta must track cumulative growth since the (unchanged) phase baseline");
    return true;
}

// A drop back to or below baseline within the same phase must not warn --
// only growth over the phase's own baseline is a violation.
static bool test_shrink_within_phase_never_warns() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;
    (void) t.observe(offload_phase::PP, 9000000, delta);
    const bool warn = t.observe(offload_phase::PP, 8000000, delta);
    TEST_ASSERT(!warn, "bytes at or below the phase baseline must never warn");
    return true;
}

// Round-tripping through LOAD/WARMUP and back into PP is still a phase
// transition and must re-baseline like any other -- the fix must not special
// case PP<->TG and miss transitions through the other phases.
static bool test_transition_through_other_phases_rebaselines() {
    zero_alloc_baseline_tracker t;
    size_t                      delta = 0;
    (void) t.observe(offload_phase::PP, 5000000, delta);
    (void) t.observe(offload_phase::PP, 5500000, delta);  // PP warm-up growth
    (void) t.observe(offload_phase::WARMUP, 6000000, delta);
    const bool warn = t.observe(offload_phase::PP, 6200000, delta);
    TEST_ASSERT(!warn, "re-entering PP after WARMUP is a transition and must re-baseline silently");
    TEST_ASSERT(t.baseline == 6200000,
                "re-entry baseline must be the bytes observed on this entry, not the earlier PP run");
    return true;
}

int main() {
    bool ok = true;
    ok &= test_first_observation_never_warns();
    ok &= test_phase_transition_rebaselines_without_warning();
    ok &= test_flat_steady_state_never_warns();
    ok &= test_real_midphase_growth_still_warns();
    ok &= test_shrink_within_phase_never_warns();
    ok &= test_transition_through_other_phases_rebaselines();
    std::printf("SYCL zero-alloc-check baseline tracker tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
