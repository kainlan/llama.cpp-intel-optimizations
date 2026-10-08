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
// The planned ladder (zhcn design 1188-1197) adds three more pure helpers:
//   llama_auto_ubatch_trial_runs()   the four-way conjunction deciding whether
//                                    the trial runs at all;
//   llama_auto_ubatch_cap()          min(n_batch, n_ctx), narrowed to the MoE
//                                    routing ceiling when that ceiling binds;
//   llama_auto_ubatch_rung_set()     fallback, the ladder rungs in
//                                    [fallback, cap] and a valid cached value,
//                                    ascending and deduplicated.
// The ladder loop iterates the set's ladder members, so each case below also
// replays the loop it replaced (break above cap, skip at or below the resume
// value, skip below the floor) and requires the same candidate sequence.
//
// llama_auto_ubatch_advice() is the -ub a refusal names: the one hold-spill fit function's answer, capped under the
// lowest rung that already lost, so it never names a refused rung.
//
// llama_auto_ubatch_next_lower() / llama_auto_ubatch_descend() are the downward continuation the trial takes
// when the default rung is refused (llama.cpp-kpjw item 0): the same pure walk the trial runs, driven here by a
// fake that accepts what the B50 / Qwen arithmetic accepts.
//
// No device, no model, no allocation.

#include "../src/llama-auto-ubatch.h"

#include <cstdio>
#include <string>
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

static void check_trial_runs(const char * name, bool a, bool b, bool c, bool d, bool expect) {
    const bool got = llama_auto_ubatch_trial_runs(a, b, c, d);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: trial_runs(%d,%d,%d,%d)=%d, expected %d\n", name, a, b, c, d, got, expect);
        g_failures++;
        return;
    }
    std::printf("ok   %s: trial_runs=%d\n", name, got);
}

static void check_cap(const char * name,
                      uint32_t     n_batch,
                      uint32_t     n_ctx,
                      uint32_t     n_expert,
                      uint32_t     moe_cap,
                      bool         moe_cap_available,
                      uint32_t     expect_cap,
                      bool         expect_moe_bound) {
    bool           moe_bound = !expect_moe_bound;  // a helper that never writes it fails the case
    const uint32_t got       = llama_auto_ubatch_cap(n_batch, n_ctx, n_expert, moe_cap, moe_cap_available, &moe_bound);
    if (got != expect_cap || moe_bound != expect_moe_bound) {
        std::fprintf(stderr, "FAIL %s: cap=%u moe_bound=%d, expected cap=%u moe_bound=%d\n", name, got, moe_bound,
                     expect_cap, expect_moe_bound);
        g_failures++;
        return;
    }
    std::printf("ok   %s: cap=%u moe_bound=%d\n", name, got, moe_bound);
}

static std::vector<uint32_t> rung_set_of(uint32_t fallback, uint32_t cap, uint32_t cached) {
    uint32_t     out[llama_auto_ubatch_rung_set_capacity];
    const size_t n =
        llama_auto_ubatch_rung_set(ladder, n_ladder, fallback, cap, cached, out, llama_auto_ubatch_rung_set_capacity);
    return std::vector<uint32_t>(out, out + n);
}

static std::vector<uint32_t> members_of(const std::vector<uint32_t> & set) {
    uint32_t     out[llama_auto_ubatch_rung_set_capacity];
    const size_t n = llama_auto_ubatch_ladder_members(set.data(), set.size(), ladder, n_ladder, out,
                                                      llama_auto_ubatch_rung_set_capacity);
    return std::vector<uint32_t>(out, out + n);
}

// The loop sycl_select_auto_ubatch ran before the planned ladder, as pure
// integers: the candidates it reaches before its first refusal.
static std::vector<uint32_t> old_loop_candidates(uint32_t fallback, uint32_t cap, uint32_t resume_above) {
    std::vector<uint32_t> r;
    for (size_t i = 0; i < n_ladder; ++i) {
        const uint32_t c = ladder[i];
        if (c > cap) {
            break;
        }
        if (c <= resume_above) {
            continue;
        }
        if (c < fallback) {
            continue;
        }
        r.push_back(c);
    }
    return r;
}

static std::vector<uint32_t> new_loop_candidates(uint32_t fallback,
                                                 uint32_t cap,
                                                 uint32_t cached,
                                                 uint32_t resume_above) {
    std::vector<uint32_t>       r;
    const std::vector<uint32_t> rungs = members_of(rung_set_of(fallback, cap, cached));
    for (size_t i = 0; i < rungs.size(); ++i) {
        if (rungs[i] <= resume_above) {
            continue;
        }
        r.push_back(rungs[i]);
    }
    return r;
}

static std::string fmt(const std::vector<uint32_t> & v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s + "}";
}

static void check_set(const char *                  name,
                      uint32_t                      fallback,
                      uint32_t                      cap,
                      uint32_t                      cached,
                      const std::vector<uint32_t> & expect) {
    const std::vector<uint32_t> got = rung_set_of(fallback, cap, cached);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: rung_set(fallback=%u cap=%u cached=%u)=%s, expected %s\n", name, fallback, cap,
                     cached, fmt(got).c_str(), fmt(expect).c_str());
        g_failures++;
        return;
    }
    std::printf("ok   %s: rung_set=%s\n", name, fmt(got).c_str());
}

// The set's ladder members, run through the loop's remaining skip, must be the
// candidate sequence the old loop reached; cached_valid marks the cache value
// the trial revalidates first, which resumes the ladder above it.
static void check_equivalence(const char * name, uint32_t fallback, uint32_t cap, uint32_t cached, bool quiet = false) {
    const bool                  cached_valid = llama_auto_ubatch_cached_valid(ladder, n_ladder, cached, fallback, cap);
    const uint32_t              resume       = cached_valid ? cached : 0;
    const std::vector<uint32_t> a            = old_loop_candidates(fallback, cap, resume);
    const std::vector<uint32_t> b            = new_loop_candidates(fallback, cap, cached_valid ? cached : 0, resume);
    if (a != b) {
        std::fprintf(stderr, "FAIL %s: old loop %s, planned loop %s\n", name, fmt(a).c_str(), fmt(b).c_str());
        g_failures++;
        return;
    }
    if (!quiet) {
        std::printf("ok   %s: candidates=%s\n", name, fmt(b).c_str());
    }
}

static void run_planned_ladder_cases() {
    check_trial_runs("all four hold", true, true, true, true, true);
    check_trial_runs("pinned -ub", false, true, true, true, false);
    check_trial_runs("non-causal", true, false, true, true, false);
    check_trial_runs("no SYCL backend", true, true, false, true, false);
    check_trial_runs("trial disabled", true, true, true, false, false);

    check_cap("dense, batch under ctx", 2048, 4096, 0, 512, true, 2048, false);
    check_cap("dense, ctx under batch", 2048, 1000, 0, 512, true, 1000, false);
    // A dense model never consults the ceiling, whatever it says.
    check_cap("dense ignores the ceiling", 2048, 4096, 0, 256, true, 2048, false);
    check_cap("MoE, ceiling narrows", 2048, 4096, 32, 512, true, 512, true);
    check_cap("MoE, ceiling equals cap still binds", 512, 4096, 32, 512, true, 512, true);
    check_cap("MoE, ceiling above cap does not bind", 512, 4096, 32, 1024, true, 512, false);
    check_cap("MoE, accessor absent", 2048, 4096, 32, 2048, false, 2048, false);

    check_set("dense three rungs", 512, 2048, 0, { 512, 1024, 2048 });
    check_set("MoE single rung", 512, 512, 0, { 512 });
    check_set("fallback between rungs joins the set", 600, 2048, 0, { 600, 1024, 2048 });
    check_set("fallback above every rung", 8192, 8192, 0, { 8192 });
    check_set("cap below the first rung", 256, 256, 0, { 256 });
    check_set("cached rung is deduplicated", 512, 2048, 1024, { 512, 1024, 2048 });
    check_set("cached non-rung value is added", 512, 2048, 768, { 512, 768, 1024, 2048 });
    check_set("cached below fallback is dropped", 1024, 2048, 600, { 1024, 2048 });
    check_set("cached above cap is dropped", 512, 1024, 2048, { 512, 1024 });
    check_set("cached below the first rung is dropped", 256, 2048, 300, { 256, 512, 1024, 2048 });
    check_set("fallback equal to cap", 1024, 1024, 0, { 1024 });
    check_set("fallback above cap is excluded", 1024, 512, 0, {});

    check_equivalence("equiv default dense", 512, 2048, 0);
    check_equivalence("equiv fallback between rungs", 600, 1000, 0);
    check_equivalence("equiv floor above MoE cap", 1024, 512, 0);
    check_equivalence("equiv floor above the ladder", 8192, 8192, 0);
    check_equivalence("equiv cap below the first rung", 256, 256, 0);
    check_equivalence("equiv resume above a cached rung", 512, 4096, 1024);
    check_equivalence("equiv cached non-rung resumes above it", 512, 4096, 768);
    check_equivalence("equiv cached at the cap", 512, 2048, 2048);
    check_equivalence("equiv cached invalid below floor", 1024, 4096, 512);
    for (uint32_t fallback = 1; fallback <= 8192; fallback = fallback * 2 + (fallback % 3)) {
        for (uint32_t cap = 1; cap <= 8192; cap = cap * 2 + (cap % 5)) {
            for (uint32_t cached = 0; cached <= 8192; cached = cached * 2 + 256) {
                check_equivalence("equiv sweep", fallback, cap, cached, true);
            }
        }
    }
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

static void check_advice(const char * name, uint32_t largest_fit, uint32_t lowest_refused, uint32_t expect) {
    const uint32_t got = llama_auto_ubatch_advice(largest_fit, lowest_refused);
    if (got != expect) {
        std::fprintf(stderr, "FAIL %s: advice(%u, %u)=%u, expected %u\n", name, largest_fit, lowest_refused, got,
                     expect);
        g_failures++;
        return;
    }
    std::printf("ok   %s: advice(%u, %u)=%u\n", name, largest_fit, lowest_refused, got);
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

    run_planned_ladder_cases();

    // ---- The downward continuation (llama.cpp-kpjw item 0). When the default rung (and so everything above it) is
    // refused, the trial lowers n_ubatch instead of refusing the context: a smaller -ub is not a smaller context. ----
    check_lower("512 halves to 256", 512, 256);
    check_lower("256 halves to 128", 256, 128);
    check_lower("128 halves to 64", 128, 64);
    check_lower("64 is the floor: nothing below it", 64, 0);
    check_lower("a value whose half is under the floor names nothing", 100, 0);
    check_lower("zero names nothing", 0, 0);
    check_lower("a non-rung value rounds down to a multiple of 64", 600, 256);
    check_lower("a value under the next multiple of 64 keeps the one below", 700, 320);
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

    // ---- The -ub a refusal names (llama.cpp-kpjw, kpjw-g7 / r6 M4). It comes from the one hold-spill fit function
    // (the largest -ub it accepts) and is capped under the lowest rung this start already saw lose, so advice never
    // names a rung that was asked and refused. There is no settle descent: a refused settle is a named error. ----
    check_advice("nothing refused yet: the function's answer stands", 512, 0, 512);
    check_advice("the function's answer is under the lowest refused rung", 256, 512, 256);
    check_advice("the function accepts a rung that was refused for another reason: one rung under it", 512, 512, 256);
    check_advice("the function accepts more than the refused default: capped under it", 1024, 512, 256);
    check_advice("a refused non-rung default caps at the power of two under it", 2048, 600, 512);
    check_advice("nothing known to fit stays nothing", 0, 512, 0);
    check_advice("the rung under the floor names nothing", 512, 64, 0);
    check_advice("a refused 1024 with the function accepting it: 512", 1024, 1024, 512);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d case(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all cases passed\n");
    return 0;
}
