// The expert swap policy of the MoE expert cache (ggml-sycl/moe-swap-policy.hpp). SYCL-free: no device is opened, no
// model is loaded, and the decisions are checked on small tables whose answer is worked out by hand, plus a replay at
// the size of a 48 x 512 model.
#include "ggml-sycl/moe-swap-policy.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <set>
#include <utility>
#include <vector>

using ggml_sycl::moe_swap_block_input;
using ggml_sycl::moe_swap_call;
using ggml_sycl::moe_swap_params;
using ggml_sycl::moe_swap_proposal;
using ggml_sycl::moe_swap_state;
using ggml_sycl::moe_swap_status;
using ggml_sycl::moe_swap_step;

static int g_failed = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                \
        }                                                              \
    } while (0)

// Every heap allocation of this binary is counted, so a test can say a call allocated nothing. The test is one thread.
static size_t g_allocs = 0;

void * operator new(std::size_t n) {
    ++g_allocs;
    if (void * p = std::malloc(n != 0 ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, std::size_t) noexcept {
    std::free(p);
}

static bool near(double a, double b) {
    return a > b - 1e-9 && a < b + 1e-9;
}

static uint32_t lcg(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

// The caller's side: cumulative counts, residency and eligibility of every block, and the call built from them.
struct table {
    int                                n_blocks;
    int                                n_experts;
    std::vector<std::vector<uint64_t>> decode;
    std::vector<std::vector<uint64_t>> prompt;
    std::vector<std::vector<uint8_t>>  resident;
    std::vector<uint8_t>               eligible;
    std::vector<moe_swap_block_input>  in;

    table(int nb, int ne) :
        n_blocks(nb),
        n_experts(ne),
        decode(static_cast<size_t>(nb), std::vector<uint64_t>(static_cast<size_t>(ne), 0)),
        prompt(static_cast<size_t>(nb), std::vector<uint64_t>(static_cast<size_t>(ne), 0)),
        resident(static_cast<size_t>(nb), std::vector<uint8_t>(static_cast<size_t>(ne), 0)),
        eligible(static_cast<size_t>(nb), 1),
        in(static_cast<size_t>(nb)) {}

    // Experts 0 .. n-1 of block b are resident, the rest are on the host.
    void resident_prefix(int b, int n) {
        for (int e = 0; e < n_experts; ++e) {
            resident[static_cast<size_t>(b)][static_cast<size_t>(e)] = e < n ? 1 : 0;
        }
    }

    int resident_count(int b) const {
        int n = 0;
        for (uint8_t f : resident[static_cast<size_t>(b)]) {
            n += f != 0 ? 1 : 0;
        }
        return n;
    }

    const moe_swap_block_input * blocks() {
        for (size_t b = 0; b < in.size(); ++b) {
            in[b].decode_counts = decode[b].data();
            in[b].prompt_counts = prompt[b].data();
            in[b].resident      = resident[b].data();
            in[b].eligible      = eligible[b] != 0;
        }
        return in.data();
    }

    // Execute the proposals: each evicted expert leaves, each admitted expert enters.
    void apply(const std::vector<moe_swap_proposal> & out) {
        for (const moe_swap_proposal & p : out) {
            resident[static_cast<size_t>(p.block)][static_cast<size_t>(p.evict_expert)] = 0;
            resident[static_cast<size_t>(p.block)][static_cast<size_t>(p.admit_expert)] = 1;
        }
    }
};

static moe_swap_status step(moe_swap_state &                 s,
                            const moe_swap_params &          p,
                            table &                          t,
                            std::vector<moe_swap_proposal> & out,
                            bool                             in_flight    = false,
                            bool                             after_prompt = false,
                            int                              n_tokens     = 1) {
    moe_swap_call c;
    c.blocks       = t.blocks();
    c.n_blocks     = t.n_blocks;
    c.n_experts    = t.n_experts;
    c.n_tokens     = n_tokens;
    c.in_flight    = in_flight;
    c.after_prompt = after_prompt;
    return moe_swap_step(s, p, c, out);
}

// One period of decode tokens, one call each; the result is the last call's, the period boundary. The state's token
// clock must be at a multiple of the period when this starts.
static moe_swap_status run_period(moe_swap_state &                 s,
                                  const moe_swap_params &          p,
                                  table &                          t,
                                  std::vector<moe_swap_proposal> & out,
                                  bool                             in_flight = false) {
    moe_swap_status st = ggml_sycl::MOE_SWAP_STATUS_WAITING;
    for (int i = 0; i < p.period; ++i) {
        st = step(s, p, t, out, in_flight);
    }
    return st;
}

using ggml_sycl::MOE_SWAP_STATUS_ADAPTED;
using ggml_sycl::MOE_SWAP_STATUS_IN_FLIGHT;
using ggml_sycl::MOE_SWAP_STATUS_REJECTED;
using ggml_sycl::MOE_SWAP_STATUS_SEEDED;
using ggml_sycl::MOE_SWAP_STATUS_WAITING;

// ---------------------------------------------------------------------------------------------------------------------

// The five numbers of the rule are fields with the simulated defaults.
static void test_defaults() {
    moe_swap_params p;
    CHECK(p.period == 4);
    CHECK(near(p.decay, 0.7));
    CHECK(p.max_swaps == 96);
    CHECK(near(p.min_usage, 2.0));
    CHECK(near(p.margin, 1.5));
}

// A host expert whose usage is under min_usage is not proposed, one at min_usage is. Each sub-case says which rule it
// isolates: the margin test (host >= victim + margin) is the other rule that refuses a cold host expert, so a case only
// has power over min_usage when the margin would let the host expert through.
static void test_candidate_below_min_usage_is_not_proposed() {
    const moe_swap_params          p;
    std::vector<moe_swap_proposal> out;
    {
        table t(1, 8);
        t.resident_prefix(0, 4);
        t.decode[0][5] = 1;  // usage 1 < 2.0 (and 1 < 0 + 1.5: both rules refuse it, so this is the plain case)
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
        CHECK(near(s.usage_of(0, 5), 0.7));  // it still counted, and decayed
    }
    {
        // Isolates min_usage: with a margin of 0.5 the margin lets usage 1 through (1 >= 0 + 0.5), only the default
        // min_usage of 2.0 refuses it.
        moe_swap_params loose;
        loose.margin = 0.5;
        table t(1, 8);
        t.resident_prefix(0, 4);
        t.decode[0][5] = 1;
        moe_swap_state s(1, 8);
        CHECK(run_period(s, loose, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
    {
        // Isolates min_usage the other way: at the default margin usage 2 clears 0 + 1.5, only a min_usage of 3
        // refuses it, and usage 3 is at that min_usage and proposed.
        moe_swap_params strict;
        strict.min_usage = 3.0;
        table t2(1, 8);
        t2.resident_prefix(0, 4);
        t2.decode[0][5] = 2;
        moe_swap_state s2(1, 8);
        CHECK(run_period(s2, strict, t2, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
        table t3(1, 8);
        t3.resident_prefix(0, 4);
        t3.decode[0][5] = 3;
        moe_swap_state s3(1, 8);
        CHECK(run_period(s3, strict, t3, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 1);
    }
    {
        table t(1, 8);
        t.resident_prefix(0, 4);
        t.decode[0][5] = 2;  // usage exactly min_usage (and 2 >= 0 + 1.5), resident experts at 0
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 1);
        if (out.size() == 1) {
            CHECK(out[0].block == 0 && out[0].evict_expert == 0 && out[0].admit_expert == 5);
            CHECK(near(out[0].gain, 2.0));
        }
    }
}

// A host expert under the coldest resident expert's usage plus the margin is not proposed, one at it is, and the first
// pair that fails the margin ends the block. The margin is additive, as simulated.
static void test_candidate_below_victim_plus_margin_is_not_proposed() {
    const moe_swap_params          p;
    std::vector<moe_swap_proposal> out;
    // Residents 0..3 route 4, 5, 6, 7 times. The coldest is expert 0 at 4, so a candidate needs 5.5.
    auto                           make = [](uint64_t host4, uint64_t host5) {
        table t(1, 8);
        t.resident_prefix(0, 4);
        t.decode[0][0] = 4;
        t.decode[0][1] = 5;
        t.decode[0][2] = 6;
        t.decode[0][3] = 7;
        t.decode[0][4] = host4;
        t.decode[0][5] = host5;
        return t;
    };
    {
        table          t = make(5, 0);  // 5 < 4 + 1.5
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
    {
        table          t = make(6, 0);  // 6 >= 4 + 1.5
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 1);
        if (out.size() == 1) {
            CHECK(out[0].evict_expert == 0 && out[0].admit_expert == 4 && near(out[0].gain, 2.0));
        }
    }
    {
        // A margin of 2: 6 == 4 + 2 is at the margin and proposed, 5 is not.
        moe_swap_params wide;
        wide.margin      = 2.0;
        table          t = make(6, 0);
        moe_swap_state s(1, 8);
        CHECK(run_period(s, wide, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 1);
        table          t5 = make(5, 0);
        moe_swap_state s5(1, 8);
        CHECK(run_period(s5, wide, t5, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
    {
        // The margin is added, not multiplied: against a victim at 2 a host expert at 3 clears 1.5 times (3.0) but
        // not 2 + 1.5 (3.5).
        table t(1, 8);
        t.resident_prefix(0, 4);
        t.decode[0][0] = 2;
        t.decode[0][1] = 9;
        t.decode[0][2] = 9;
        t.decode[0][3] = 9;
        t.decode[0][4] = 3;
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
    {
        // Host 4 at 20 passes against resident 0 (4). Host 5 at 6 meets resident 1 (5): 6 < 5 + 1.5, so the block
        // ends.
        table          t = make(20, 6);
        moe_swap_state s(1, 8);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 1);
        if (out.size() == 1) {
            CHECK(out[0].evict_expert == 0 && out[0].admit_expert == 4 && near(out[0].gain, 16.0));
        }
    }
}

// No call returns more than max_swaps proposals; the default is 96 over the whole model.
static void test_max_swaps_caps_the_proposals() {
    std::vector<moe_swap_proposal> out;
    {
        // 16 experts, 0..7 resident and cold, 8..15 host and hot: eight pairs could swap, three are allowed.
        moe_swap_params p;
        p.max_swaps = 3;
        table t(1, 16);
        t.resident_prefix(0, 8);
        for (int e = 8; e < 16; ++e) {
            t.decode[0][static_cast<size_t>(e)] = 10;
        }
        moe_swap_state s(1, 16);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 3);
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i].admit_expert == 8 + static_cast<int>(i));
            CHECK(out[i].evict_expert == static_cast<int>(i));
        }
    }
    {
        // 48 blocks with eight possible swaps each is 384; the default cap returns 96.
        const moe_swap_params p;
        table                 t(48, 32);
        for (int b = 0; b < 48; ++b) {
            t.resident_prefix(b, 8);
            for (int e = 8; e < 16; ++e) {
                t.decode[static_cast<size_t>(b)][static_cast<size_t>(e)] = 10;
            }
        }
        moe_swap_state s(48, 32);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 96);
    }
    {
        moe_swap_params p;
        p.max_swaps = 0;
        table t(1, 16);
        t.resident_prefix(0, 8);
        t.decode[0][9] = 10;
        moe_swap_state s(1, 16);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
}

// Proposals come back best gain first, so a cap below the number of candidates keeps the largest gains.
static void test_proposals_are_ordered_by_gain_and_the_cap_keeps_the_best() {
    std::vector<moe_swap_proposal> out;
    // Block b: residents 0..9 cold, host expert 20 routed 10 * (b + 1) times, so its one swap has gain 10 * (b + 1).
    auto                           fill = [](table & t) {
        for (int b = 0; b < t.n_blocks; ++b) {
            t.resident_prefix(b, 10);
            t.decode[static_cast<size_t>(b)][20] = 10 * static_cast<uint64_t>(b + 1);
        }
    };
    {
        const moe_swap_params p;
        table                 t(48, 24);
        fill(t);
        moe_swap_state s(48, 24);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 48);
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i].block == 47 - static_cast<int>(i));
            CHECK(near(out[i].gain, 10.0 * static_cast<double>(48 - i)));
        }
    }
    {
        moe_swap_params p;
        p.max_swaps = 10;
        table t(48, 24);
        fill(t);
        moe_swap_state s(48, 24);
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.size() == 10);
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i].block == 47 - static_cast<int>(i));  // blocks 47 .. 38: the ten largest gains
        }
    }
}

struct random_run_result {
    size_t proposals            = 0;
    bool   counts_invariant     = true;   // every block ends with the resident count it started with
    bool   never_both           = true;   // no (block, expert) is both admitted and evicted in one call
    bool   never_twice          = true;   // no (block, expert) appears twice among a call's admits or evicts
    bool   moves_are_real       = true;   // each admit was on the host and each evict resident when proposed
    bool   skipped_block_chosen = false;  // a block with no slot or all slots got a proposal
};

// 400 decode tokens of skewed routing over six blocks of 32 experts with 0, 32, 8, 3, 8 and 16 resident, the
// proposals of every boundary executed as they come.
static random_run_result random_run() {
    const int                      n_res[6] = { 0, 32, 8, 3, 8, 16 };
    const moe_swap_params          p;
    table                          t(6, 32);
    moe_swap_state                 s(6, 32);
    std::vector<moe_swap_proposal> out;
    random_run_result              r;
    for (int b = 0; b < 6; ++b) {
        for (int e = 0; e < 32; ++e) {
            t.resident[static_cast<size_t>(b)][static_cast<size_t>(e)] = (e * 7 + b * 3) % 32 < n_res[b] ? 1 : 0;
        }
    }
    uint32_t seed = 12345;
    for (int token = 0; token < 400; ++token) {
        for (int b = 0; b < 6; ++b) {
            for (int k = 0; k < 6; ++k) {
                const uint32_t x = lcg(seed) % 32;
                const uint32_t y = lcg(seed) % 32;
                ++t.decode[static_cast<size_t>(b)][x * y / 32];  // skewed to the low experts
            }
        }
        if (step(s, p, t, out) != MOE_SWAP_STATUS_ADAPTED) {
            continue;
        }
        std::set<std::pair<int, int>> admits;
        std::set<std::pair<int, int>> evicts;
        for (const moe_swap_proposal & m : out) {
            r.proposals++;
            const size_t b   = static_cast<size_t>(m.block);
            r.moves_are_real = r.moves_are_real && t.resident[b][static_cast<size_t>(m.admit_expert)] == 0 &&
                               t.resident[b][static_cast<size_t>(m.evict_expert)] != 0;
            r.never_twice = r.never_twice && admits.insert({ m.block, m.admit_expert }).second &&
                            evicts.insert({ m.block, m.evict_expert }).second;
            r.skipped_block_chosen = r.skipped_block_chosen || m.block == 0 || m.block == 1;
        }
        for (const std::pair<int, int> & a : admits) {
            r.never_both = r.never_both && evicts.count(a) == 0;
        }
        t.apply(out);
    }
    for (int b = 0; b < 6; ++b) {
        r.counts_invariant = r.counts_invariant && t.resident_count(b) == n_res[b];
    }
    return r;
}

// The randomized run is deterministic, so the two tests that read it share one.
static const random_run_result & shared_random_run() {
    static const random_run_result r = random_run();
    return r;
}

// A swap exchanges one resident expert for one host expert, so a block's resident count never changes.
static void test_a_blocks_resident_count_is_invariant() {
    const random_run_result & r = shared_random_run();
    CHECK(r.proposals > 0);  // positive control: the run did swap
    CHECK(r.counts_invariant);
    CHECK(r.moves_are_real);
    CHECK(!r.skipped_block_chosen);  // the block with no slot and the block with every slot got no proposal
}

// One call never admits and evicts the same expert, and never proposes an expert twice.
static void test_an_expert_is_never_admitted_and_evicted_or_proposed_twice() {
    const random_run_result & r = shared_random_run();
    CHECK(r.proposals > 0);
    CHECK(r.never_both);
    CHECK(r.never_twice);
}

// While swaps are in flight a boundary proposes nothing, absorbs nothing and does not decay; the next boundary reads
// the counts the skipped one left.
static void test_in_flight_suppresses_the_proposals() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);

    t.decode[0][6] = 10;
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.size() == 1);
    CHECK(near(s.usage_of(0, 6), 7.0));  // 10 * 0.7

    t.decode[0][6] = 20;
    CHECK(run_period(s, p, t, out, true) == MOE_SWAP_STATUS_IN_FLIGHT);
    CHECK(out.empty());
    CHECK(near(s.usage_of(0, 6), 7.0));  // not absorbed, not decayed
    CHECK(s.tokens == 8);                // the clock kept running

    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(!out.empty());
    CHECK(near(s.usage_of(0, 6), 11.9));  // (7 + 10) * 0.7: the skipped period's counts arrive here
}

// A prompt pass absorbs its counts, adds nothing to the period clock, does not decay and proposes nothing; the usage it
// seeded is what the first decode boundary acts on.
static void test_prompt_seeding_changes_usage_and_proposes_nothing() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);

    t.prompt[0][5] = 100;  // a host expert the prompt routed to a lot, while every resident expert saw nothing
    CHECK(step(s, p, t, out, false, true) == MOE_SWAP_STATUS_SEEDED);
    CHECK(out.empty());
    CHECK(near(s.usage_of(0, 5), 100.0));
    CHECK(s.tokens == 0);

    CHECK(step(s, p, t, out, false, true) == MOE_SWAP_STATUS_SEEDED);  // the same cumulative counts: nothing new
    CHECK(near(s.usage_of(0, 5), 100.0));                              // and no decay between prompt passes
    t.prompt[0][5] = 150;
    CHECK(step(s, p, t, out, false, true) == MOE_SWAP_STATUS_SEEDED);
    CHECK(out.empty());
    CHECK(near(s.usage_of(0, 5), 150.0));
    CHECK(s.tokens == 0);  // three prompt passes did not bring the first decode boundary closer

    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.size() == 1);
    if (out.size() == 1) {
        CHECK(out[0].evict_expert == 0 && out[0].admit_expert == 5 && near(out[0].gain, 150.0));
    }
    CHECK(near(s.usage_of(0, 5), 105.0));  // 150 * 0.7
}

// Usage is multiplied by the decay at every boundary, after the boundary's counts are added; a value that decayed to
// nothing is zero, not a denormal number.
static void test_decay_arithmetic_over_several_periods() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);

    t.decode[0][6] = 10;
    t.decode[0][1] = 3;
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 7.0));
    CHECK(near(s.usage_of(0, 1), 2.1));

    t.decode[0][6] = 15;
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 8.4));  // (7 + 5) * 0.7
    CHECK(near(s.usage_of(0, 1), 1.47));

    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 5.88));
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 4.116));

    for (int i = 0; i < 100; ++i) {
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    }
    CHECK(s.usage_of(0, 6) == 0.0);  // 4.1 * 0.7^100 is far under the floor
    CHECK(s.usage_of(0, 1) == 0.0);

    moe_swap_params half;
    half.decay = 0.5;  // the decay is a parameter, not a constant
    table          t2(1, 8);
    moe_swap_state s2(1, 8);
    t2.resident_prefix(0, 4);
    t2.decode[0][6] = 8;
    CHECK(run_period(s2, half, t2, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s2.usage_of(0, 6), 4.0));
}

static bool same(const std::vector<moe_swap_proposal> & a, const std::vector<moe_swap_proposal> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].block != b[i].block || a[i].evict_expert != b[i].evict_expert ||
            a[i].admit_expert != b[i].admit_expert || a[i].gain != b[i].gain) {
            return false;
        }
    }
    return true;
}

// Equal usage goes to the lower expert index, equal gain across blocks to the lower admitted expert and then the lower
// block, and the same inputs always give the same output.
static void test_ties_go_to_the_lower_expert_index_and_the_output_is_reproducible() {
    moe_swap_params p;
    p.max_swaps = 3;
    // Two identical blocks: residents 0..3 at zero, hosts 4..7 all at 5, so every pair has gain 5.
    auto make   = []() {
        table t(2, 8);
        for (int b = 0; b < 2; ++b) {
            t.resident_prefix(b, 4);
            for (int e = 4; e < 8; ++e) {
                t.decode[static_cast<size_t>(b)][static_cast<size_t>(e)] = 5;
            }
        }
        return t;
    };
    std::vector<moe_swap_proposal> out1;
    std::vector<moe_swap_proposal> out2;
    table                          t1 = make();
    table                          t2 = make();
    moe_swap_state                 s1(2, 8);
    moe_swap_state                 s2(2, 8);
    CHECK(run_period(s1, p, t1, out1) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(run_period(s2, p, t2, out2) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(same(out1, out2));
    CHECK(out1.size() == 3);
    if (out1.size() == 3) {
        // The k-th lowest-index host expert meets the k-th lowest-index resident one; then by admitted expert, block.
        CHECK(out1[0].block == 0 && out1[0].admit_expert == 4 && out1[0].evict_expert == 0);
        CHECK(out1[1].block == 1 && out1[1].admit_expert == 4 && out1[1].evict_expert == 0);
        CHECK(out1[2].block == 0 && out1[2].admit_expert == 5 && out1[2].evict_expert == 1);
    }
    // A third state, built and fed alone, gives the same answer again.
    table                          t3 = make();
    moe_swap_state                 s3(2, 8);
    std::vector<moe_swap_proposal> out3;
    CHECK(run_period(s3, p, t3, out3) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(same(out1, out3));
}

// Counts that are all zero propose nothing, however many periods pass, and leave usage at zero.
static void test_a_table_of_zero_counts_proposes_nothing() {
    const moe_swap_params          p;
    table                          t(4, 16);
    moe_swap_state                 s(4, 16);
    std::vector<moe_swap_proposal> out;
    for (int b = 0; b < 4; ++b) {
        t.resident_prefix(b, 6);
    }
    for (int i = 0; i < 10; ++i) {
        CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
        CHECK(out.empty());
    }
    for (int b = 0; b < 4; ++b) {
        for (int e = 0; e < 16; ++e) {
            CHECK(s.usage_of(b, e) == 0.0);
        }
    }
}

// A block with no resident expert has nothing to evict, one with every expert resident has nothing to admit, and a
// block marked not eligible is left alone; all three still follow their counts.
static void test_blocks_with_zero_slots_all_slots_or_not_eligible_are_skipped() {
    const moe_swap_params          p;
    table                          t(4, 8);
    moe_swap_state                 s(4, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 0);  // no slots
    t.resident_prefix(1, 8);  // every expert resident
    t.resident_prefix(2, 4);  // the block that swaps
    t.resident_prefix(3, 4);  // would swap, but the caller says not now
    t.eligible[3] = 0;
    for (int b = 0; b < 4; ++b) {
        for (int e = 0; e < 8; ++e) {
            t.decode[static_cast<size_t>(b)][static_cast<size_t>(e)] = e >= 4 ? 10 : 0;
        }
    }
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.size() == 4);
    for (const moe_swap_proposal & m : out) {
        CHECK(m.block == 2);
    }
    CHECK(near(s.usage_of(0, 5), 7.0));
    CHECK(near(s.usage_of(1, 5), 7.0));
    CHECK(near(s.usage_of(3, 5), 7.0));  // the ineligible block's usage is current for when it becomes eligible
}

// The inputs are cumulative counts: the policy absorbs the difference since its last look, and a counter that restarted
// at a lower value is taken as the difference.
static void test_cumulative_counts_are_absorbed_as_deltas() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);

    for (int i = 0; i < 8; ++i) {  // one routed pair per token for two periods
        ++t.decode[0][6];
        step(s, p, t, out);
    }
    CHECK(near(s.usage_of(0, 6), 4.76));  // ((4 * 0.7) + 4) * 0.7: each period's four counts once, never twice

    t.decode[0][6] = 1;                   // the counter restarted: one pair since
    for (int i = 0; i < 4; ++i) {
        step(s, p, t, out);
    }
    CHECK(near(s.usage_of(0, 6), 4.032));  // (4.76 + 1) * 0.7
}

// A call that reaches no period boundary reads nothing and allocates nothing, and a boundary with no candidate
// allocates nothing either once the state exists.
static void test_a_call_without_a_boundary_is_cheap_and_does_not_allocate() {
    const moe_swap_params p;
    table                 t(48, 512);
    const size_t          at_start = g_allocs;
    moe_swap_state        s(48, 512);
    CHECK(g_allocs > at_start);  // positive control: the replaced operator new does count (the state's vectors)
    std::vector<moe_swap_proposal> out;
    for (int b = 0; b < 48; ++b) {
        t.resident_prefix(b, 92);
    }
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);  // warm: the zero table's own first boundary

    t.decode[3][300]    = 1000;  // counts that change, which a non-boundary call must not read
    const size_t before = g_allocs;
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
    CHECK(g_allocs == before);
    CHECK(s.usage_of(3, 300) == 0.0);  // nothing was read

    t.decode[3][300] = 1;              // a boundary where nothing qualifies: usage 1 is under min_usage
    CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.empty());
    CHECK(g_allocs == before);
}

// A call whose shape does not fit the state is refused and touches nothing, and so is a state with no blocks or no
// experts, which is not a model.
static void test_a_call_of_the_wrong_shape_is_rejected() {
    const moe_swap_params          p;
    std::vector<moe_swap_proposal> out(1);  // a stale entry the refused call must clear
    {
        moe_swap_state s(2, 8);
        table          t(3, 8);
        t.decode[0][6] = 10;           // a count the refused calls must not absorb, as they must not move the clock
        for (int i = 0; i < 4; ++i) {  // a whole period, so a boundary would have absorbed it
            out.resize(1);
            CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_REJECTED);
            CHECK(out.empty());
            CHECK(s.tokens == 0);
            CHECK(s.usage_of(0, 6) == 0.0);
        }

        table t16(2, 16);  // the right number of blocks, the wrong number of experts
        t16.decode[0][6] = 10;
        for (int i = 0; i < 4; ++i) {
            out.resize(1);
            CHECK(step(s, p, t16, out) == MOE_SWAP_STATUS_REJECTED);
            CHECK(out.empty());
            CHECK(s.tokens == 0);
            CHECK(s.usage_of(0, 6) == 0.0);
        }
        out.resize(1);
        CHECK(step(s, p, t16, out, false, true) == MOE_SWAP_STATUS_REJECTED);  // and a prompt call
        CHECK(s.usage_of(0, 6) == 0.0);

        moe_swap_call c;
        c.n_blocks  = 2;
        c.n_experts = 8;  // blocks left null
        out.resize(1);
        CHECK(moe_swap_step(s, p, c, out) == MOE_SWAP_STATUS_REJECTED);
        CHECK(out.empty());
        CHECK(s.tokens == 0);
    }
    {
        // A state with no block, no expert, or a negative dimension (clamped to none) refuses every call, including
        // one that matches its empty shape, and reads nothing.
        const int dims[3][2] = {
            { 0,  8 },
            { 3,  0 },
            { -1, 5 }
        };
        for (const auto & d : dims) {
            const int      nb = d[0] > 0 ? d[0] : 0;
            const int      ne = d[1] > 0 ? d[1] : 0;
            moe_swap_state s(d[0], d[1]);
            table          t(nb, ne);
            CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_REJECTED);
            CHECK(step(s, p, t, out, false, true) == MOE_SWAP_STATUS_REJECTED);
            CHECK(s.tokens == 0);
            CHECK(s.usage.empty());
        }
    }
}

// A parameter outside the contract is refused on every kind of call, and nothing moves: not usage, not the token
// clock, and `out` is left empty.
static void test_parameters_outside_the_contract_are_rejected_and_move_nothing() {
    const moe_swap_params          good;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);
    t.decode[0][6] = 10;
    CHECK(run_period(s, good, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 7.0));
    CHECK(s.tokens == 4);

    const double                 nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<moe_swap_params> bad(6, good);
    bad[0].decay     = nan;
    bad[1].decay     = 1.5;
    bad[2].decay     = -0.1;
    bad[3].margin    = nan;
    bad[4].margin    = -1.0;
    bad[5].min_usage = nan;
    t.decode[0][6]   = 50;             // counts a refused call must not absorb
    for (const moe_swap_params & p : bad) {
        for (int i = 0; i < 4; ++i) {  // a whole period of decode calls
            out.assign(1, moe_swap_proposal());
            CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_REJECTED);
            CHECK(out.empty());
        }
        out.assign(1, moe_swap_proposal());
        CHECK(step(s, p, t, out, false, true) == MOE_SWAP_STATUS_REJECTED);  // and a prompt call
        CHECK(out.empty());
        CHECK(s.tokens == 4);
        CHECK(near(s.usage_of(0, 6), 7.0));
    }
    // The edges of the contract are accepted: decay 0 and 1, margin 0, min_usage 0.
    moe_swap_params edge;
    edge.decay     = 1.0;
    edge.margin    = 0.0;
    edge.min_usage = 0.0;
    CHECK(run_period(s, edge, t, out) == MOE_SWAP_STATUS_ADAPTED);
    edge.decay = 0.0;
    CHECK(run_period(s, edge, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(s.usage_of(0, 6) == 0.0);
}

// A call that stands for several decode tokens advances the clock by all of them, and crossing one or two period
// boundaries is one boundary: one absorb, one proposal pass, one decay.
static void test_a_call_of_several_tokens_is_one_boundary_however_many_it_crosses() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);
    t.decode[0][6] = 10;

    CHECK(step(s, p, t, out, false, false, 3) == MOE_SWAP_STATUS_WAITING);  // clock 3
    CHECK(s.tokens == 3);
    CHECK(step(s, p, t, out, false, false, 2) == MOE_SWAP_STATUS_ADAPTED);  // clock 5 crosses 4
    CHECK(s.tokens == 5);
    CHECK(out.size() == 1);
    CHECK(near(s.usage_of(0, 6), 7.0));

    t.decode[0][6] = 20;
    CHECK(step(s, p, t, out, false, false, 8) == MOE_SWAP_STATUS_ADAPTED);  // clock 13 crosses 8 and 12
    CHECK(s.tokens == 13);
    CHECK(near(s.usage_of(0, 6), 11.9));  // (7 + 10) * 0.7 once; two decays would leave 8.33

    CHECK(step(s, p, t, out, false, false, 2) == MOE_SWAP_STATUS_WAITING);  // clock 15
    CHECK(step(s, p, t, out, false, false, 1) == MOE_SWAP_STATUS_ADAPTED);  // clock 16 lands on a boundary
    CHECK(s.tokens == 16);

    // In flight, the crossing is skipped like any other boundary.
    CHECK(step(s, p, t, out, true, false, 9) == MOE_SWAP_STATUS_IN_FLIGHT);
    CHECK(s.tokens == 25);
}

// A call for no token, or a negative count, moves nothing.
static void test_a_call_of_no_tokens_is_waiting_and_moves_nothing() {
    const moe_swap_params          p;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);
    t.decode[0][6] = 10;
    for (int n : { 0, -1, -1000000 }) {
        CHECK(step(s, p, t, out, false, false, n) == MOE_SWAP_STATUS_WAITING);
        CHECK(s.tokens == 0);
        CHECK(s.usage_of(0, 6) == 0.0);
    }
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);  // the clock was not disturbed
}

// A period below 1 never reaches a boundary: every call waits, the clock still runs, nothing is read.
static void test_a_period_below_one_never_reaches_a_boundary() {
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);
    t.decode[0][6] = 10;
    for (int period : { 0, -3 }) {
        moe_swap_params p;
        p.period = period;
        for (int i = 0; i < 10; ++i) {
            CHECK(step(s, p, t, out) == MOE_SWAP_STATUS_WAITING);
            CHECK(out.empty());
        }
    }
    CHECK(s.tokens == 20);
    CHECK(s.usage_of(0, 6) == 0.0);
}

// A null count array is a block whose counts did not change; a block that is eligible but has no residency array
// cannot swap, and neither reads memory it was not given.
static void test_null_counts_mean_unchanged_and_a_block_without_residency_is_skipped() {
    const moe_swap_params          p;
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    std::vector<uint64_t>          decode(8, 0);
    std::vector<uint64_t>          prompt(8, 0);
    std::vector<uint8_t>           resident = { 1, 1, 1, 1, 0, 0, 0, 0 };
    decode[6]                               = 100;  // never read: the pointer below is null
    prompt[5]                               = 6;

    moe_swap_block_input in;
    in.decode_counts = nullptr;
    in.prompt_counts = prompt.data();
    in.resident      = resident.data();
    in.eligible      = true;
    moe_swap_call c;
    c.blocks    = &in;
    c.n_blocks  = 1;
    c.n_experts = 8;
    for (int i = 0; i < 4; ++i) {
        moe_swap_step(s, p, c, out);
    }
    CHECK(out.size() == 1);  // the prompt counts alone were absorbed, and are enough to swap
    if (out.size() == 1) {
        CHECK(out[0].admit_expert == 5 && near(out[0].gain, 6.0));
    }
    CHECK(near(s.usage_of(0, 5), 4.2));  // 6 * 0.7
    CHECK(s.usage_of(0, 6) == 0.0);

    in.decode_counts = decode.data();  // now the decode array arrives, and the prompt array is the one that is null
    in.prompt_counts = nullptr;
    for (int i = 0; i < 4; ++i) {
        moe_swap_step(s, p, c, out);
    }
    CHECK(near(s.usage_of(0, 6), 70.0));  // 100 * 0.7
    CHECK(near(s.usage_of(0, 5), 2.94));  // 4.2 * 0.7: unchanged counts add nothing

    in.resident = nullptr;                // eligible, but nothing to say which experts are resident
    for (int i = 0; i < 3; ++i) {
        CHECK(moe_swap_step(s, p, c, out) == MOE_SWAP_STATUS_WAITING);
    }
    CHECK(moe_swap_step(s, p, c, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.empty());
    CHECK(near(s.usage_of(0, 6), 49.0));  // it still decayed

    in.resident      = resident.data();   // both count arrays null in the same call: nothing to absorb, usage decays
    in.decode_counts = nullptr;
    in.prompt_counts = nullptr;
    for (int i = 0; i < 3; ++i) {
        CHECK(moe_swap_step(s, p, c, out) == MOE_SWAP_STATUS_WAITING);
    }
    CHECK(moe_swap_step(s, p, c, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(near(s.usage_of(0, 6), 34.3));  // 49 * 0.7, no count added
}

// A max_swaps below zero proposes nothing, like zero, and usage still decays.
static void test_a_negative_max_swaps_proposes_nothing() {
    moe_swap_params p;
    p.max_swaps = -5;
    table                          t(1, 8);
    moe_swap_state                 s(1, 8);
    std::vector<moe_swap_proposal> out;
    t.resident_prefix(0, 4);
    t.decode[0][6] = 10;
    CHECK(run_period(s, p, t, out) == MOE_SWAP_STATUS_ADAPTED);
    CHECK(out.empty());
    CHECK(near(s.usage_of(0, 6), 7.0));
}

// The call sits at a graph boundary on the decode path of a 48 x 512 model, so replay one at that size: skewed routing,
// 92 resident experts per block, every boundary's proposals executed. The cost is a bound on the host, stated here and
// in the header. The time check is loose on purpose (measured about 25 ms, bound two seconds) so a loaded host cannot
// fail it: it catches a gross regression only, such as a quadratic pass or an allocation per call. That a boundary is
// not taken at every token, which would cost about four times as much and still pass a time bound, is held by
// counting the statuses instead: one boundary in four, the rest cheap WAITING calls.
static void test_replay_at_the_size_of_a_48_by_512_model_stays_within_a_time_bound() {
    const moe_swap_params          p;
    table                          t(48, 512);
    moe_swap_state                 s(48, 512);
    std::vector<moe_swap_proposal> out;
    for (int b = 0; b < 48; ++b) {
        for (int e = 0; e < 512; ++e) {
            t.resident[static_cast<size_t>(b)][static_cast<size_t>(e)] = e >= 420 ? 1 : 0;  // the cold tail
        }
    }
    uint32_t  seed        = 777;
    size_t    proposals   = 0;
    size_t    boundaries  = 0;
    size_t    waiting     = 0;
    bool      cap_held    = true;
    const int tokens      = 1024;
    double    boundary_us = 0.0;
    double    other_us    = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int b = 0; b < 48; ++b) {
            for (int k = 0; k < 10; ++k) {
                const uint32_t x = lcg(seed) % 512;
                const uint32_t y = lcg(seed) % 512;
                ++t.decode[static_cast<size_t>(b)][x * y / 512];
            }
        }
        const auto            t0 = std::chrono::steady_clock::now();
        const moe_swap_status st = step(s, p, t, out);
        const auto            t1 = std::chrono::steady_clock::now();
        const double          us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (st == MOE_SWAP_STATUS_ADAPTED) {
            ++boundaries;
            boundary_us += us;
            proposals += out.size();
            cap_held = cap_held && out.size() <= static_cast<size_t>(p.max_swaps);
            t.apply(out);
        } else {
            other_us += us;
            waiting += st == MOE_SWAP_STATUS_WAITING ? 1 : 0;
        }
    }
    CHECK(boundaries == static_cast<size_t>(tokens / p.period));
    CHECK(waiting == static_cast<size_t>(tokens) - boundaries);
    CHECK(proposals > 0);
    CHECK(cap_held);
    for (int b = 0; b < 48; ++b) {
        CHECK(t.resident_count(b) == 92);
    }
    std::printf("replay 48 x 512: %zu boundaries, %.1f us per boundary, %.3f us per other call, %zu swaps\n",
                boundaries, boundary_us / static_cast<double>(boundaries),
                other_us / static_cast<double>(tokens - static_cast<int>(boundaries)), proposals);
    CHECK(boundary_us + other_us < 2.0e6);  // all 1024 calls together under two seconds: a gross regression only
}

int main() {
    test_defaults();
    test_candidate_below_min_usage_is_not_proposed();
    test_candidate_below_victim_plus_margin_is_not_proposed();
    test_max_swaps_caps_the_proposals();
    test_proposals_are_ordered_by_gain_and_the_cap_keeps_the_best();
    test_a_blocks_resident_count_is_invariant();
    test_an_expert_is_never_admitted_and_evicted_or_proposed_twice();
    test_in_flight_suppresses_the_proposals();
    test_prompt_seeding_changes_usage_and_proposes_nothing();
    test_decay_arithmetic_over_several_periods();
    test_ties_go_to_the_lower_expert_index_and_the_output_is_reproducible();
    test_a_table_of_zero_counts_proposes_nothing();
    test_blocks_with_zero_slots_all_slots_or_not_eligible_are_skipped();
    test_cumulative_counts_are_absorbed_as_deltas();
    test_a_call_without_a_boundary_is_cheap_and_does_not_allocate();
    test_a_call_of_the_wrong_shape_is_rejected();
    test_parameters_outside_the_contract_are_rejected_and_move_nothing();
    test_a_call_of_several_tokens_is_one_boundary_however_many_it_crosses();
    test_a_call_of_no_tokens_is_waiting_and_moves_nothing();
    test_a_period_below_one_never_reaches_a_boundary();
    test_null_counts_mean_unchanged_and_a_block_without_residency_is_skipped();
    test_a_negative_max_swaps_proposes_nothing();
    test_replay_at_the_size_of_a_48_by_512_model_stays_within_a_time_bound();

    if (g_failed != 0) {
        std::printf("test-sycl-moe-swap-policy: %d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test-sycl-moe-swap-policy: all checks passed\n");
    return 0;
}
