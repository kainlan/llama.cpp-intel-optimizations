// Host-only gate for the by-name refusal of a scheduler compute buffer that no tier of the SYCL device could place
// (compute-refusal-advice.hpp, llama.cpp-mmi1).
//
// THE SHAPE THIS IS FIT TO (Qwen3.8-Flash-Next IQ3_XXS, Arc Pro B70, -ngl 99 -ub 512, no -c so n_ctx = 262144): the
// planner filled the arena, uize demoted all 48 KV layers to the host, and the scheduler's compute buffer is
// 2133928064 B (2035.07 MiB; 490 MiB at -c 4096, the difference being three f32 [n_kv, n_ub] tensors of the QSA
// indexer path). The RUNTIME zone is the PP MoE ring's (nothing free), the KV zone has 283.6 MiB, and the card has
// 2045.2 MiB free outside the arena against a 256 MiB driver headroom, so no tier holds the buffer whole.
//
// What a refusal must say: the request, the room found per tier, the largest -ub that fits, and the
// GGML_SYCL_VRAM_BUDGET_PCT that leaves enough of the card outside the arena. Never a smaller context.
//
// Host-only: the header is plain C++ with no SYCL or backend dependency.

#include "compute-refusal-advice.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using ggml_sycl::compute_refusal_advice;
using ggml_sycl::compute_refusal_inputs;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t kMiB = 1024ull * 1024ull;
constexpr size_t kMax = std::numeric_limits<size_t>::max();

// The incident, in bytes. 2045.2 MiB is the log's free figure; the budget is the log's 30553.1 MiB at 100%.
constexpr size_t kRequest  = 2133928064ull;
constexpr size_t kKvRoom   = 297373696ull;   // 283.6 MiB
constexpr size_t kRawFree  = 2144499712ull;  // 2045.2 MiB
constexpr size_t kHeadroom = 256ull * kMiB;
constexpr size_t kBase     = 32656ull * kMiB;
constexpr size_t kExt      = 2048ull * kMiB;
constexpr size_t kBudget   = 32601ull * kMiB - kExt;  // min(base * 100%, 32601 MiB free at init) - headroom
constexpr size_t kFixed    = 5036ull * kMiB;          // RUNTIME + SCRATCH + ONEDNN

compute_refusal_inputs incident() {
    compute_refusal_inputs in;
    in.device                   = 0;
    in.n_ubatch                 = 512;
    in.request                  = kRequest;
    in.runtime_room             = 0;
    in.kv_room                  = kKvRoom;
    in.raw_free                 = kRawFree;
    in.headroom_target          = kHeadroom;
    in.budget.pct               = 100;
    in.budget.base_mem          = kBase;
    in.budget.budget_bytes      = kBudget;
    in.budget.external_headroom = kExt;
    in.budget.fixed_zone_bytes  = kFixed;
    return in;
}

bool contains(const std::string & text, const char * needle) {
    return text.find(needle) != std::string::npos;
}

void test_best_room_is_the_largest_whole_block() {
    printf("best room: tiers are blocks, never summed; the raw tier keeps the driver headroom\n");
    compute_refusal_inputs in = incident();
    // raw tier = 2045.2 MiB free - 256 MiB headroom = 1789.2 MiB; it beats the 283.6 MiB KV block.
    check(ggml_sycl::compute_refusal_best_room(in) == kRawFree - kHeadroom, "the raw tier (free less headroom) wins");
    in.runtime_room = 3000ull * kMiB;
    check(ggml_sycl::compute_refusal_best_room(in) == 3000ull * kMiB, "a larger RUNTIME block wins");
    in.runtime_room = 0;
    in.raw_free     = kHeadroom / 2;
    check(ggml_sycl::compute_refusal_best_room(in) == kKvRoom, "a card under its headroom offers no raw room");
    in.kv_room = 0;
    check(ggml_sycl::compute_refusal_best_room(in) == 0, "no tier, no room");
}

void test_scaled_request() {
    printf("scaled request: linear in -ub, rounded up, saturating\n");
    check(ggml_sycl::compute_refusal_scaled_request(1000, 512, 256) == 500, "half the -ub, half the buffer");
    check(ggml_sycl::compute_refusal_scaled_request(1000, 512, 128) == 250, "a quarter");
    check(ggml_sycl::compute_refusal_scaled_request(1001, 512, 256) == 501, "rounded up, never down");
    check(ggml_sycl::compute_refusal_scaled_request(1000, 0, 256) == 1000, "an unknown source -ub scales nothing");
    check(ggml_sycl::compute_refusal_scaled_request(kMax / 2, 1, 4) == kMax, "saturates instead of wrapping");
}

void test_largest_ub() {
    printf("largest -ub: the largest power of two whose scaled buffer fits the best room\n");
    compute_refusal_inputs in = incident();
    check(ggml_sycl::compute_refusal_largest_ub(in) == 256,
          "the incident: 512 needs 2035 MiB of 1789, 256 needs 1018 and fits");
    in.n_ubatch = 600;
    in.request  = kRequest / 512 * 600;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 256, "a non-power-of-two -ub starts at 512, which is refused");
    in = incident();
    in.raw_free = 200ull * kMiB + kHeadroom;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 64, "a small room names a small -ub: 200 MiB holds 64");
    in.raw_free = kHeadroom;
    in.kv_room  = 0;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 0, "no room at all: no -ub is known to fit");
    in = incident();
    in.hold_fit_refused = true;
    in.hold_largest_ub  = 128;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 128,
          "the kpjw hold-spill fit refusing 256 caps the answer: the printed -ub passes both");
    in.hold_largest_ub = 0;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 0, "the hold fit knowing no -ub fits means none is named");
    in = incident();
    in.hold_fit_refused = false;
    in.hold_largest_ub  = 64;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 256, "a hold fit that accepted the rung caps nothing");
}

// budget(pct) the way the authority computes it: min(base * pct / 100, free at init) - headroom.
size_t budget_at(const compute_refusal_inputs & in, int pct) {
    return ggml_sycl::compute_refusal_budget_bytes_at(in.budget, pct);
}

void test_budget_arithmetic_is_the_authoritys() {
    printf("budget at a pct: the authority's own arithmetic, capped by the free memory it started from\n");
    const compute_refusal_inputs in = incident();
    check(budget_at(in, 100) == kBudget, "100% reproduces the published budget (free at init is the cap)");
    // The log: vram_budget=25709.6 MiB at 85%, 24076.8 MiB at 80%.
    const double mib85 = budget_at(in, 85) / (double) kMiB;
    const double mib80 = budget_at(in, 80) / (double) kMiB;
    check(mib85 > 25709.5 && mib85 < 25709.7, "85% gives the logged 25709.6 MiB");
    check(mib80 > 24076.7 && mib80 < 24076.9, "80% gives the logged 24076.8 MiB");
    compute_refusal_inputs unknown = in;
    unknown.budget.base_mem        = 0;
    check(budget_at(unknown, 90) == 0, "an unknown device total gives no budget");
}

void test_budget_pct_frees_enough_and_no_more_than_needed() {
    printf("budget pct: the highest pct that leaves the buffer, the headroom and a margin outside the arena\n");
    const compute_refusal_inputs in  = incident();
    const int                    pct = ggml_sycl::compute_refusal_budget_pct(in);
    check(pct > 0 && pct < 100, "the incident gets a pct under the current 100");
    const auto freed_for = [&](int p) {
        // every byte the budget gives up is a byte outside the arena
        return in.raw_free + (in.budget.budget_bytes - budget_at(in, p));
    };
    const size_t need = in.request + in.headroom_target + in.headroom_target;
    check(freed_for(pct) >= need, "at the named pct the outside-arena room covers buffer + headroom + margin");
    check(freed_for(pct + 1) < need, "one more percent would not: the answer is the highest that works");
    check(budget_at(in, pct) > in.budget.fixed_zone_bytes, "the weights keep room above the fixed zones");
}

void test_budget_pct_declines_what_it_cannot_do() {
    printf("budget pct: 0 when lowering the budget cannot help\n");
    compute_refusal_inputs in = incident();
    in.raw_free               = kRequest + 2 * kHeadroom;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0,
          "the room was already there: another cause, no budget change is advised");
    in = incident();
    in.request = 64ull * 1024ull * kMiB;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0, "a buffer larger than the whole budget: none");
    in = incident();
    in.budget.fixed_zone_bytes = 29000ull * kMiB;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0, "a pct that would starve the fixed zones is not advised");
    in = incident();
    in.budget.base_mem = 0;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0, "no budget authority, no advice");
    in = incident();
    in.budget.pct = 1;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0, "a pct that cannot go lower is not advised");
}

void test_message_names_what_fits() {
    printf("message: request, room per tier, the -ub and the pct; never a smaller context\n");
    const compute_refusal_inputs in  = incident();
    const compute_refusal_advice adv = ggml_sycl::compute_refusal_advise(in);
    check(adv.largest_ub == 256, "advise carries the -ub");
    check(adv.budget_pct > 0 && adv.budget_pct < 100, "advise carries the pct");
    check(adv.best_room == kRawFree - kHeadroom, "advise carries the best room");
    const std::string msg = ggml_sycl::compute_refusal_message(in, adv);
    printf("    %s\n", msg.c_str());
    check(contains(msg, "2035.1 MiB"), "names the refused request");
    check(contains(msg, "-ub 512"), "names the -ub it was refused at");
    check(contains(msg, "283.6 MiB"), "names the KV zone's room");
    check(contains(msg, "1789.2 MiB"), "names the outside-arena room after the headroom");
    check(contains(msg, "-ub 256"), "names the -ub that fits");
    char pct[64];
    snprintf(pct, sizeof(pct), "GGML_SYCL_VRAM_BUDGET_PCT=%d", adv.budget_pct);
    check(contains(msg, pct), "names the computed budget pct");
    check(contains(msg, "device 0"), "names the device");
    check(contains(msg, "host-pinned"), "says why the host fallback is not an answer");
    check(!contains(msg, "-c "), "never advises a smaller context");
    check(!contains(msg, "context size") && !contains(msg, "smaller -c"), "no context-size remedy at all");
}

void test_message_when_nothing_fits() {
    printf("message: says so when no -ub and no pct is known to fit\n");
    compute_refusal_inputs in = incident();
    in.raw_free               = kHeadroom;
    in.kv_room                = 0;
    in.budget.base_mem        = 0;
    const compute_refusal_advice adv = ggml_sycl::compute_refusal_advise(in);
    const std::string            msg = ggml_sycl::compute_refusal_message(in, adv);
    check(adv.largest_ub == 0 && adv.budget_pct == 0, "nothing is named");
    check(contains(msg, "no -ub is known to fit"), "says no -ub is known to fit");
    check(!contains(msg, "GGML_SYCL_VRAM_BUDGET_PCT="), "does not name a pct it does not have");
    check(contains(msg, "free VRAM on the card"), "still names the one thing the user controls");
}

void test_degenerate_inputs() {
    printf("degenerate: zero request or -ub names nothing and never divides by zero\n");
    compute_refusal_inputs in = incident();
    in.n_ubatch               = 0;
    check(ggml_sycl::compute_refusal_largest_ub(in) == 0, "an unknown -ub names no -ub");
    in = incident();
    in.request = 0;
    check(ggml_sycl::compute_refusal_budget_pct(in) == 0, "no request, no pct");
    check(ggml_sycl::compute_refusal_message(in, ggml_sycl::compute_refusal_advise(in)).size() > 0,
          "a message is still produced");
}

}  // namespace

int main() {
    test_best_room_is_the_largest_whole_block();
    test_scaled_request();
    test_largest_ub();
    test_budget_arithmetic_is_the_authoritys();
    test_budget_pct_frees_enough_and_no_more_than_needed();
    test_budget_pct_declines_what_it_cannot_do();
    test_message_names_what_fits();
    test_message_when_nothing_fits();
    test_degenerate_inputs();

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
