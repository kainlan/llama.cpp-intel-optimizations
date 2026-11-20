// Which resident expert of a MoE block leaves and which host expert enters, decided at one graph boundary, as a pure
// function of the routed-expert counts. No SYCL, no ggml, no allocation through the cache, no logging, no global or
// static state: the caller owns one moe_swap_state and passes it in. The driver that reads the counters and the
// ADMIT/EVICT transaction that executes the proposals are other code.
//
// The rule. The policy keeps a decayed usage value per (block, expert), fed with the routed pairs the counters saw
// since it last looked. Every `period` decode tokens it proposes swaps and then multiplies every usage by `decay`. A
// swap pairs, inside one block, a host expert with a resident expert: the k-th hottest host expert with the k-th
// coldest resident one, k = 0, 1, ..., and stops at the first pair that fails. A pair is proposed only if the host
// expert's usage is at least `min_usage` and at least `margin` times the resident expert's usage. Its gain is the
// difference of the two usages. All blocks' pairs are ranked by gain and the best `max_swaps` are returned, so the
// cap keeps the best.
//   - A block's slot count is the number of its experts the caller flags resident, and a swap exchanges one resident
//     expert for one host expert, so no swap changes a block's resident count. A block with no resident expert has
//     nothing to evict and a block with no host expert nothing to admit: both are skipped. A block the caller
//     marks not eligible (fully resident, a loan outstanding, ...) is skipped for proposals but still follows its
//     counts and still decays, so its usage is current when it becomes eligible.
//   - The two sets that make a pair are disjoint (host and resident), and each expert is in at most one pair, so an
//     expert is never both admitted and evicted in one call and never proposed twice.
//   - While the caller reports swaps in flight, a period boundary proposes nothing and does not decay, and that
//     boundary is not made up: the next one is `period` tokens later. The counts it did not absorb are absorbed by
//     the next boundary, so nothing is lost.
//   - A prompt pass seeds usage and proposes nothing: its counts are absorbed with no decay, it does not count toward
//     the period, so the first decode boundary after a prompt starts from the prompt's routing.
//
// Inputs are cumulative counts, as the route counters keep them, not deltas: the state remembers the counts it last
// saw and absorbs the difference. A count that is lower than the remembered one means the counter restarted, and
// its new value is taken as the difference. Usage is a sum of differences and decay only happens at a boundary, so
// absorbing the counts at the boundary instead of at every token gives exactly the value an eager absorb would, and
// a call that reaches no boundary has nothing to read.
//
// Determinism. The same state and the same inputs give the same output. Ties in usage go to the lower expert index,
// the proposals are totally ordered by gain (higher first), then by admitted expert (lower first), then by block
// (lower first), then by evicted expert (lower first). Nothing depends on addresses, threads or time.
//
// Cost, for a model of B blocks of E experts (Qwen3.8 Flash Next: 48 x 512 = 24576 cells), R resident experts per
// block and P proposals before the cap (P is at most B * min(R, max_swaps)):
//   - a call that reaches no boundary does O(1) work and allocates nothing;
//   - a prompt call and a boundary absorb the counts: two passes over the B * E counters;
//   - a boundary adds one pass over the usage per block for the candidates, a partial sort of the candidates and of
//     the residents of each block that has a candidate, one pass over the usage to decay, and one partial sort of the
//     P proposals: O(B * E + B * (R + c log R) + P log max_swaps), c the candidates above min_usage of a block;
//   - a boundary that has no candidate allocates nothing. The scratch vectors are sized for E experts when the state
//     is built and the proposal pool grows only the first times a boundary has more proposals than before, so a
//     boundary allocates only until its scratch has been as large as it gets. The caller's output vector is cleared
//     and refilled, so it allocates only when it grows past its own capacity (never above max_swaps entries).
// State is 24 bytes per cell (usage and the two remembered counts).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// The five numbers of the rule. The defaults are the simulated ones. `margin` must not be negative.
struct moe_swap_params {
    int    period    = 4;    // decode tokens between boundaries; a value below 1 never reaches one
    double decay     = 0.7;  // usage is multiplied by this at every boundary that was not skipped
    int    max_swaps = 96;   // proposals per boundary over the whole model
    double min_usage = 2.0;  // a host expert below this is not a candidate
    double margin    = 1.5;  // a host expert must be at least this many times the resident expert it replaces
};

// One proposed swap: evict `evict_expert` from `block` and admit `admit_expert` into it.
struct moe_swap_proposal {
    int    block        = 0;
    int    evict_expert = 0;
    int    admit_expert = 0;
    double gain         = 0.0;  // usage of the admitted expert minus usage of the evicted one
};

// A block as the caller sees it at this boundary. The count arrays hold n_experts cumulative counts each; a null
// array is a block whose counts did not change. A block with a null `resident` is not eligible.
struct moe_swap_block_input {
    const uint64_t * decode_counts = nullptr;  // cumulative routed (token, expert) pairs of the decode route
    const uint64_t * prompt_counts = nullptr;  // the same for the prompt route
    const uint8_t *  resident      = nullptr;  // n_experts flags, nonzero = the expert is resident
    bool             eligible      = false;    // may this block take part in a swap now
};

// One call at a graph boundary.
struct moe_swap_call {
    const moe_swap_block_input * blocks       = nullptr;  // n_blocks entries
    int                          n_blocks     = 0;
    int                          n_tokens     = 1;  // decode tokens this boundary stands for; ignored after a prompt
    bool                         in_flight    = false;  // an earlier swap has not finished
    bool                         after_prompt = false;  // this boundary follows a prompt pass, not a decode token
};

enum moe_swap_status {
    MOE_SWAP_STATUS_WAITING,    // a decode token that did not reach a period boundary: nothing was read
    MOE_SWAP_STATUS_SEEDED,     // a prompt pass: counts absorbed, no decay, no proposal
    MOE_SWAP_STATUS_IN_FLIGHT,  // a period boundary skipped because swaps are in flight: no absorb, no decay
    MOE_SWAP_STATUS_ADAPTED,    // a period boundary taken: proposals (possibly none) returned, usage decayed
    MOE_SWAP_STATUS_REJECTED,   // the call does not fit the state's shape: nothing was touched
};

// A usage value with its expert, for ranking inside one block.
struct moe_swap_ranked {
    double usage  = 0.0;
    int    expert = 0;
};

// Usage below this cannot change any decision (it is far under any min_usage) and would reach denormal numbers after a
// few thousand periods, where a multiply is slow, so a decay flushes it to zero.
constexpr double MOE_SWAP_USAGE_FLOOR = 1e-12;

// Everything the policy remembers between calls. Built once for a model's dimensions and owned by the caller.
struct moe_swap_state {
    moe_swap_state(int n_blocks_, int n_experts_) :
        n_blocks(n_blocks_ > 0 ? n_blocks_ : 0),
        n_experts(n_experts_ > 0 ? n_experts_ : 0) {
        const size_t cells = static_cast<size_t>(n_blocks) * static_cast<size_t>(n_experts);
        usage.assign(cells, 0.0);
        last_decode.assign(cells, 0);
        last_prompt.assign(cells, 0);
        cands.reserve(static_cast<size_t>(n_experts));
        victs.reserve(static_cast<size_t>(n_experts));
    }

    int                            n_blocks  = 0;
    int                            n_experts = 0;
    uint64_t                       tokens    = 0;  // decode tokens seen, the period's clock
    std::vector<double>            usage;          // decayed usage per (block, expert), as of the last absorb
    std::vector<uint64_t>          last_decode;    // cumulative decode counts at the last absorb
    std::vector<uint64_t>          last_prompt;    // cumulative prompt counts at the last absorb
    std::vector<moe_swap_ranked>   cands;          // scratch: host experts of one block above min_usage
    std::vector<moe_swap_ranked>   victs;          // scratch: resident experts of one block
    std::vector<moe_swap_proposal> pool;           // scratch: every block's proposals before the cap

    // The usage of one cell as of the last absorb; zero for a cell the state does not hold.
    double usage_of(int block, int expert) const {
        if (block < 0 || block >= n_blocks || expert < 0 || expert >= n_experts) {
            return 0.0;
        }
        return usage[static_cast<size_t>(block) * static_cast<size_t>(n_experts) + static_cast<size_t>(expert)];
    }
};

// Add what `cur` counted since `last` to `usage` and remember `cur`. A count below the remembered one is a counter
// that restarted: its value is the difference.
inline void moe_swap_absorb_row(double * usage, uint64_t * last, const uint64_t * cur, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint64_t c = cur[i];
        const uint64_t l = last[i];
        usage[i] += static_cast<double>(c >= l ? c - l : c);
        last[i] = c;
    }
}

inline void moe_swap_absorb(moe_swap_state & s, const moe_swap_block_input * blocks) {
    const size_t ne = static_cast<size_t>(s.n_experts);
    for (int b = 0; b < s.n_blocks; ++b) {
        const size_t base = static_cast<size_t>(b) * ne;
        if (blocks[b].decode_counts != nullptr) {
            moe_swap_absorb_row(&s.usage[base], &s.last_decode[base], blocks[b].decode_counts, ne);
        }
        if (blocks[b].prompt_counts != nullptr) {
            moe_swap_absorb_row(&s.usage[base], &s.last_prompt[base], blocks[b].prompt_counts, ne);
        }
    }
}

// The proposals of one block, appended to s.pool. Host candidates are ranked hottest first and residents coldest
// first, both with ties to the lower expert index; the k-th candidate meets the k-th resident. Candidates fall and
// residents rise along the ranking, so the first pair that fails ends the block.
inline void moe_swap_collect_block(moe_swap_state & s, const moe_swap_params & p, int block, const uint8_t * resident) {
    const size_t   ne = static_cast<size_t>(s.n_experts);
    const double * u  = &s.usage[static_cast<size_t>(block) * ne];

    s.cands.clear();
    size_t n_res = 0;
    for (size_t e = 0; e < ne; ++e) {
        if (resident[e] != 0) {
            ++n_res;
        } else if (u[e] >= p.min_usage) {
            s.cands.push_back({ u[e], static_cast<int>(e) });
        }
    }
    if (n_res == 0 || s.cands.empty()) {
        return;
    }

    const size_t k = std::min(std::min(s.cands.size(), n_res), static_cast<size_t>(p.max_swaps));
    std::partial_sort(s.cands.begin(), s.cands.begin() + static_cast<std::ptrdiff_t>(k), s.cands.end(),
                      [](const moe_swap_ranked & a, const moe_swap_ranked & b) {
                          return a.usage != b.usage ? a.usage > b.usage : a.expert < b.expert;
                      });

    s.victs.clear();
    for (size_t e = 0; e < ne; ++e) {
        if (resident[e] != 0) {
            s.victs.push_back({ u[e], static_cast<int>(e) });
        }
    }
    std::partial_sort(s.victs.begin(), s.victs.begin() + static_cast<std::ptrdiff_t>(k), s.victs.end(),
                      [](const moe_swap_ranked & a, const moe_swap_ranked & b) {
                          return a.usage != b.usage ? a.usage < b.usage : a.expert < b.expert;
                      });

    for (size_t i = 0; i < k; ++i) {
        if (!(s.cands[i].usage >= p.margin * s.victs[i].usage)) {
            break;
        }
        s.pool.push_back({ block, s.victs[i].expert, s.cands[i].expert, s.cands[i].usage - s.victs[i].usage });
    }
}

// The order proposals are returned in: best gain first, ties to the lower admitted expert, then the lower block, then
// the lower evicted expert. Two proposals never share (block, admitted expert), so the order is total.
inline bool moe_swap_proposal_before(const moe_swap_proposal & a, const moe_swap_proposal & b) {
    if (a.gain != b.gain) {
        return a.gain > b.gain;
    }
    if (a.admit_expert != b.admit_expert) {
        return a.admit_expert < b.admit_expert;
    }
    if (a.block != b.block) {
        return a.block < b.block;
    }
    return a.evict_expert < b.evict_expert;
}

// One call at a graph boundary. `out` is cleared, and holds the proposals in execution order when the status is
// MOE_SWAP_STATUS_ADAPTED. The caller applies them (a proposal whose block it can no longer swap it may drop; the
// rest are independent of each other).
inline moe_swap_status moe_swap_step(moe_swap_state &                 s,
                                     const moe_swap_params &          p,
                                     const moe_swap_call &            c,
                                     std::vector<moe_swap_proposal> & out) {
    out.clear();
    if (c.n_blocks != s.n_blocks || (c.n_blocks > 0 && c.blocks == nullptr)) {
        return MOE_SWAP_STATUS_REJECTED;
    }

    if (c.after_prompt) {
        moe_swap_absorb(s, c.blocks);
        return MOE_SWAP_STATUS_SEEDED;
    }

    const uint64_t before = s.tokens;
    s.tokens += static_cast<uint64_t>(c.n_tokens > 0 ? c.n_tokens : 0);
    if (p.period < 1 || s.tokens / static_cast<uint64_t>(p.period) == before / static_cast<uint64_t>(p.period)) {
        return MOE_SWAP_STATUS_WAITING;
    }
    if (c.in_flight) {
        return MOE_SWAP_STATUS_IN_FLIGHT;
    }

    moe_swap_absorb(s, c.blocks);

    s.pool.clear();
    if (p.max_swaps > 0) {
        for (int b = 0; b < s.n_blocks; ++b) {
            if (c.blocks[b].eligible && c.blocks[b].resident != nullptr) {
                moe_swap_collect_block(s, p, b, c.blocks[b].resident);
            }
        }
    }

    for (double & x : s.usage) {
        const double v = x * p.decay;
        x              = v < MOE_SWAP_USAGE_FLOOR ? 0.0 : v;
    }

    if (!s.pool.empty()) {
        const size_t k = std::min(s.pool.size(), static_cast<size_t>(p.max_swaps));
        std::partial_sort(s.pool.begin(), s.pool.begin() + static_cast<std::ptrdiff_t>(k), s.pool.end(),
                          moe_swap_proposal_before);
        out.assign(s.pool.begin(), s.pool.begin() + static_cast<std::ptrdiff_t>(k));
    }
    return MOE_SWAP_STATUS_ADAPTED;
}

}  // namespace ggml_sycl
