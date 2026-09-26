// Host-only gate for the pure helpers in src/llama-auto-ubatch.h.
//
// llama_auto_ubatch_ladder_has_candidate() is the predicate behind the SYCL
// auto micro-batch trial's early exit before its ladder loop. When it returns false the loop would skip every rung, leave the
// `tried` list empty, and still log the [SYCL-PLAN] auto n_ubatch= outcome and
// persist a terminal tuning-cache entry for a ladder that never ran -- so the
// trial must take the silent pre-trial path instead.
//
// Each case gives the trial's two bounds as the constructor and the trial
// derive them: floor = fallback_ubatch = min(n_batch, n_ubatch), and
// cap = min(n_batch, n_ctx), narrowed further to the GPU MoE routing ceiling
// for a MoE model. A causal context already has n_batch <= n_ctx, so without
// the MoE ceiling cap is n_batch and floor <= cap.
//
// The three cases the narrower "floor above the ladder's largest rung" form
// gets wrong also run that form and require it to DISAGREE, so each is known
// to discriminate between the two.
//
// llama_auto_ubatch_settle_needs_publish() decides whether the trial's settle
// step republishes last_good. A candidate publish that threw may have landed
// on some devices, so it must force a republish even when the last candidate
// tried was fallback_ubatch itself.
//
// No device, no model, no allocation.

#include "../src/llama-auto-ubatch.h"

#include <cstdio>

// The trial's own ladder, so the cases below run on the real rungs.
static const uint32_t * const ladder   = llama_auto_ubatch_ladder;
static const size_t           n_ladder = llama_auto_ubatch_ladder_size;

static int g_failures = 0;

static bool naive_has_candidate(uint32_t floor) {
    return !(floor > ladder[n_ladder - 1]);
}

static void check_case(const char * name, uint32_t floor, uint32_t cap, bool expect, bool naive_is_wrong = false) {
    const bool got = llama_auto_ubatch_ladder_has_candidate(ladder, n_ladder, floor, cap);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: floor=%u cap=%u has_candidate=%d, expected %d\n", name, floor, cap, got, expect);
        g_failures++;
        return;
    }
    if (naive_is_wrong && naive_has_candidate(floor) == expect) {
        std::fprintf(stderr, "FAIL %s: case does not discriminate -- the last-rung form also gives %d\n", name, expect);
        g_failures++;
        return;
    }
    std::printf("ok   %s: floor=%u cap=%u has_candidate=%d\n", name, floor, cap, got);
}

static void check_publish(const char * name,
                          bool         published_any,
                          bool         publish_dirty,
                          uint32_t     n_ubatch,
                          uint32_t     fallback_ubatch,
                          bool         expect) {
    const bool got = llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty, n_ubatch, fallback_ubatch);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: needs_publish=%d, expected %d\n", name, got, expect);
        g_failures++;
        return;
    }
    std::printf("ok   %s: needs_publish=%d\n", name, got);
}

int main() {
    // Defaults: n_ctx=4096, n_batch=2048, n_ubatch=512 -> 512, 1024 and 2048 are candidates.
    check_case("default dense context", 512, 2048, true);

    // Both bounds are inclusive: a single rung equal to floor and cap is a candidate.
    check_case("single rung at floor == cap", 1024, 1024, true);
    check_case("floor between rungs, next rung under cap", 600, 1024, true);

    // n_ctx=1000, n_batch=1000, n_ubatch=600: floor=600, cap=1000. 512 is below
    // the floor and 1024 is above the cap, although 600 is far below 4096.
    check_case("floor and cap between the same two rungs", 600, 1000, false, true);

    // MoE model, n_batch=2048, explicit n_ubatch=1024, routing ceiling 512:
    // cap=512 < floor=1024. Nothing clamps fallback_ubatch to the MoE ceiling.
    check_case("floor above a MoE-narrowed cap", 1024, 512, false, true);

    // Raw-API n_ubatch=8192 with n_batch=n_ctx=8192: floor above every rung.
    check_case("floor above the largest rung", 8192, 8192, false);

    // -c 256 (llama-bench pp128/tg128 rows): cap below the first rung. The
    // trial's own `cap < ladder[0]` exit handles this before the cache
    // lookup; the predicate agrees.
    check_case("cap below the first rung", 256, 256, false, true);

    // Settle publish gate. Nothing published or attempted and the value is
    // the constructor's own: the constructor's publish still describes it.
    check_publish("nothing attempted, value unchanged", false, false, 512, 512, false);
    check_publish("a candidate's publish took effect", true, false, 512, 512, true);
    check_publish("last candidate left a different value", false, false, 1024, 512, true);

    // Two SYCL backends, fallback 512, cached 1024: the cached publish lands
    // on dev0 and dev1 refuses; rung 512's publish then throws on dev0, so
    // cparams.n_ubatch is 512 again and published_any is still false. dev0
    // still holds the 1024 plan, so the settle must republish.
    check_publish("earlier partial publish above fallback, last candidate at fallback", false, true, 512, 512, true);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d case(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all cases passed\n");
    return 0;
}
