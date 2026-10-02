// Host-only gate for the pure helpers in src/llama-auto-ubatch.h.
//
// llama_auto_ubatch_ladder_has_candidate() is the predicate behind the SYCL
// auto micro-batch trial's early exit before its ladder loop. When it returns
// false the loop would skip every rung, leave the `tried` list empty, and
// still log the [SYCL-PLAN] auto n_ubatch= outcome and persist a terminal
// tuning-cache entry for a ladder that never ran -- so the trial must take
// the silent pre-trial path instead.
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
// llama_auto_ubatch_next_lower() / llama_auto_ubatch_descend() are the downward continuation the trial takes
// when the default rung is refused (llama.cpp-kpjw item 0): the same pure walk the trial runs, driven here by a
// fake that accepts what the B50 / Qwen arithmetic accepts.
//
// No device, no model, no allocation.

#include "../src/llama-auto-ubatch.h"

#include <cstdio>
#include <vector>

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

static void check_lower(const char * name, uint32_t from, uint32_t expect) {
    const uint32_t got = llama_auto_ubatch_next_lower(from);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: next_lower(%u)=%u, expected %u\n", name, from, got, expect);
        g_failures++;
        return;
    }
    std::printf("ok   %s: next_lower(%u)=%u\n", name, from, got);
}

static void check_descend(const char *                  name,
                          uint32_t                      got,
                          uint32_t                      expect,
                          const std::vector<uint32_t> & tried,
                          const std::vector<uint32_t> & expect_tried) {
    if (got != expect || tried != expect_tried) {
        std::fprintf(stderr, "FAIL %s: got %u (expected %u), tried %zu rung(s) (expected %zu)\n", name, got, expect,
                     tried.size(), expect_tried.size());
        g_failures++;
        return;
    }
    std::printf("ok   %s: got %u after %zu rung(s)\n", name, got, tried.size());
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
    // The helper's contract, not an input the trial can produce: the trial
    // writes n_ubatch only just before a publish attempt, which always sets
    // one of the two flags. This pins the defensive n_ubatch term on its own.
    check_publish("last candidate left a different value", false, false, 1024, 512, true);

    // Two SYCL backends, fallback 512, cached 1024: the cached publish lands
    // on dev0 and dev1 refuses; rung 512's publish then throws on dev0, so
    // cparams.n_ubatch is 512 again and published_any is still false. dev0
    // still holds the 1024 plan, so the settle must republish.
    check_publish("earlier partial publish above fallback, last candidate at fallback", false, true, 512, 512, true);

    // ---- The downward continuation (llama.cpp-kpjw item 0). When the default rung (and so everything above it) is
    // refused, the trial lowers n_ubatch instead of refusing the context: a smaller -ub is not a smaller context. ----
    check_lower("512 halves to 256", 512, 256);
    check_lower("256 halves to 128", 256, 128);
    check_lower("128 halves to 64", 128, 64);
    check_lower("64 is the floor: nothing below it", 64, 0);
    check_lower("a value whose half is under the floor names nothing", 100, 0);
    check_lower("zero names nothing", 0, 0);
    check_lower("a non-rung value stays a multiple of 32", 600, 288);
    check_lower("1024 halves to 512", 1024, 512);

    // The B50 / Qwen3.6-27B auto case (kpjw-g6): -c 512, n_batch 2048, the default 512 spills a 495 MB compute buffer
    // outside the arena and leaves 107.7 MB against the 256 MB driver headroom. Only the rungs at or under the share
    // the card can take (about 358, so 256) fit, and the continuation lands on the first of them.
    {
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(512, 2048, [&](uint32_t c) {
            tried.push_back(c);
            return c <= 256;
        });
        check_descend("B50 Qwen: 512 refused, 256 fits", won, 256, tried, { 256 });
    }
    {
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(512, 2048, [&](uint32_t c) {
            tried.push_back(c);
            return c <= 64;
        });
        check_descend("only the floor fits", won, 64, tried, { 256, 128, 64 });
    }
    {
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(512, 2048, [&](uint32_t c) {
            tried.push_back(c);
            return false;
        });
        check_descend("nothing fits: 0, every rung down to the floor was tried", won, 0, tried, { 256, 128, 64 });
    }
    {
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(64, 2048, [&](uint32_t c) {
            tried.push_back(c);
            return true;
        });
        check_descend("a default already at the floor has nothing to lower to", won, 0, tried, {});
    }
    {
        // A MoE routing ceiling under the default: rungs above the cap (1024 and 512 here) are not tried, only
        // what the cap allows.
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(2048, 256, [&](uint32_t c) {
            tried.push_back(c);
            return c <= 256;
        });
        check_descend("rungs above the cap are skipped, not tried", won, 256, tried, { 256 });
    }
    {
        // Ascending first: the continuation is the answer to "nothing at or above the default won", so the
        // first accepted rung is returned and nothing smaller is tried after it.
        std::vector<uint32_t> tried;
        const uint32_t        won = llama_auto_ubatch_descend(512, 2048, [&](uint32_t c) {
            tried.push_back(c);
            return true;
        });
        check_descend("the first accepted rung wins; nothing smaller is tried", won, 256, tried, { 256 });
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d case(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all cases passed\n");
    return 0;
}
