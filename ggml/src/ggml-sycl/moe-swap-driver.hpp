// The decode-boundary expert swap driver (GGML_SYCL_MOE_SWAP=1): what one synchronize boundary does with the swap
// policy's answer, as a pure function of a plan view, the route counters and the policy state. No SYCL, no ggml, no
// cache, no logging, no environment and no state outside moe_swap_driver_state, which the caller owns, one per route
// table. The caller (ggml_sycl_run_expert_swap_at_boundary in ggml-sycl.cpp) gates the boundary, builds the view from
// the published plan (moe_swap_plan_view_build, moe-stream-plan.hpp), and passes three callables: one that copies
// the route counters, one that reads the host room, one that runs the named move.
//
// One boundary sends one transaction at most, never both directions:
//   - Pending admits first. Each evict that moved leaves its admit pending, and the next boundary sends every pending
//     admit and nothing else, before the policy may propose. While any is pending the policy is in flight.
//   - Otherwise the policy's proposals, as evicts only. An evict's device room comes back only when a deferred-free
//     pass reaps the entries it retired, never to an admit in the same call, so the admit waits a boundary.
//   - A pending admit refused finally becomes a restore: an admit of the triplet its pair evicted, under the same
//     pending rule. A restore refused finally, or dropped, leaves the slot open: counted, and its block takes no new
//     evict, so the block never holds fewer resident experts than the plan sized it for without the summary saying so.
//   - A pending admit that is BUSY at MOE_SWAP_PENDING_DROP_STREAK settled boundaries in a row with one reason is
//     dropped (and becomes a restore, or an open slot if it was one). A boundary is settled when nothing is lent and
//     no deferred free is pending: an unsettled boundary may yet give the room back, so it starts the streak over.
//   - An evict refused finally latches its block only when the transaction marks the refusal as one every evict of
//     the block would meet (moe_swap_move::latches_block: the caller sets it for a device copy that shares a bulk
//     tensor's storage). Any other final refusal drops that proposal alone, counted by its reason.
//   - A transaction that throws leaves its batch unsettled: the moves it was sent wait in the state, and the next
//     boundary, before anything else, settles each from the view it is handed, since the published plan is the
//     record of what moved. A triplet that is where its move takes it moved; an evict that did not move is dropped
//     (BUSY, MOE_SWAP_UNSETTLED_REASON), and an admit that did not move stays pending.
//   - With loans outstanding nothing is sent: the loan's ring owns the room, and the policy is held in flight.
//   - The batch is trimmed to the host room, hottest first: an evict takes pinned host room for its three copies,
//     which it may not grow (the host zone was sized by the plan). What does not fit is deferred and counted apart
//     from refusals. A prompt batch trimmed after part of it fits keeps the prompt boundary for the rest; one with
//     room for no pair spends it, and the decode rule follows.
//   - Off, with nothing read and nothing sent, on a plan that spans cards or holds the card only through experts.
//
// Cost: a boundary that reaches no boundary of the policy and has nothing pending reads, per block, the view's
// eligible flag and admissible count and the state's latch and open slot, reads no route counter and no cell, and
// allocates nothing.
#pragma once

#include "moe-swap-policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ggml_sycl {

// The switch's one spelling, for the accessor and its settings-table row.
constexpr const char MOE_SWAP_ENV[] = "GGML_SYCL_MOE_SWAP";

// Settled boundaries in a row with one BUSY reason after which a pending admit is dropped. A settled boundary has no
// deferred free pending and nothing lent, so the two waits that hand an admit its room back by themselves are over
// there. What can still answer BUSY is a transient the move cannot see ending (a load or a publication in flight, an
// entry changed under it, a submit that threw) or a full WEIGHT zone that only another move frees. One such answer
// can be the first kind met by chance; the same answer at four settled boundaries in a row says it is not passing,
// and four bounds how long an admit that will not land holds the policy in flight. It is a chosen bound, not a
// measured one.
constexpr int MOE_SWAP_PENDING_DROP_STREAK = 4;

// The BUSY reason under which an evict of a batch whose transaction threw, still resident by the next view, is counted.
constexpr const char MOE_SWAP_UNSETTLED_REASON[] = "transaction threw before it answered";

// The published plan as the driver reads it, for one device over the route table's n_blocks x n_experts cells
// (block = blk.N). Built by moe_swap_plan_view_build whenever the plan snapshot changes.
struct moe_swap_plan_view {
    int                   n_blocks  = 0;
    int                   n_experts = 0;
    std::vector<uint8_t>  resident;    // per cell: the whole triplet is on this card
    std::vector<uint8_t>  admissible;  // per cell: the whole triplet is on the host, pinned, in a stageable layout
    std::vector<uint8_t>  eligible;    // per block: every triplet whole on one side, the residents lendable
    std::vector<size_t>   pair_bytes;  // per block: the pinned host bytes one evict takes for its three copies
    std::vector<uint32_t> admissible_count;   // per block: its admissible cells (moe_swap_plan_view_count_admissible)
    uint64_t              file_mapped   = 0;  // host triplets of eligible blocks served from the file mapping
    bool                  device_held   = false;  // something other than an expert holds this card in the plan
    bool                  single_device = false;  // the plan does not span cards
};

// Each block's admissible cells, counted once when the view is built, so a boundary reads one count per block
// instead of every cell.
inline void moe_swap_plan_view_count_admissible(moe_swap_plan_view & v) {
    const size_t nb = v.n_blocks > 0 ? static_cast<size_t>(v.n_blocks) : 0;
    const size_t ne = v.n_experts > 0 ? static_cast<size_t>(v.n_experts) : 0;
    v.admissible_count.assign(nb, 0);
    if (v.admissible.size() != nb * ne) {
        return;
    }
    for (size_t b = 0; b < nb; ++b) {
        uint32_t n = 0;
        for (size_t e = 0; e < ne; ++e) {
            n += v.admissible[b * ne + e] != 0 ? 1 : 0;
        }
        v.admissible_count[b] = n;
    }
}

enum moe_swap_move_kind {
    MOE_SWAP_MOVE_EVICT,
    MOE_SWAP_MOVE_ADMIT,
};

enum moe_swap_move_outcome {
    MOE_SWAP_MOVE_UNSENT,   // not answered yet
    MOE_SWAP_MOVE_MOVED,    // the triplet moved
    MOE_SWAP_MOVE_BUSY,     // a transient refusal: the same move may go later
    MOE_SWAP_MOVE_REFUSED,  // a final refusal: the same move would be refused again
};

// One triplet of a transaction: the driver fills kind, block and expert; the transaction fills the rest.
struct moe_swap_move {
    moe_swap_move_kind    kind    = MOE_SWAP_MOVE_EVICT;
    int                   block   = 0;
    int                   expert  = 0;
    moe_swap_move_outcome outcome = MOE_SWAP_MOVE_UNSENT;
    const char *          reason  = nullptr;  // a string with static storage, as the transaction's reasons are
    size_t                bytes   = 0;        // device bytes taken (admit) or given back (evict)
    // A final evict refusal every evict of this block would meet, so the block takes no evict again. The driver does
    // not read refusal texts for this; the transaction knows which of its refusals hold for the whole block.
    bool                  latches_block = false;
};

// An admit waiting for the boundary after its evict. A restore admits evict_expert back.
struct moe_swap_pending {
    int          block        = 0;
    int          admit_expert = 0;
    int          evict_expert = 0;
    bool         restore      = false;
    bool         prompt       = false;  // its evict came from a prompt boundary
    int          busy_streak  = 0;      // settled boundaries in a row it was BUSY with busy_reason
    const char * busy_reason  = nullptr;
};

// A move whose transaction threw before it answered: an evict's pair, or a pending admit as it was sent.
struct moe_swap_unsettled {
    moe_swap_move_kind kind = MOE_SWAP_MOVE_EVICT;
    moe_swap_pending   pair;
};

struct moe_swap_reason_count {
    const char * reason = nullptr;
    uint64_t     count  = 0;
};

namespace moe_swap_detail {

inline bool moe_swap_same_text(const char * a, const char * b) {
    if (a == b) {
        return true;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

inline void moe_swap_count_reason(std::vector<moe_swap_reason_count> & v, const char * reason) {
    const char * r = reason != nullptr ? reason : "no reason";
    for (moe_swap_reason_count & c : v) {
        if (moe_swap_same_text(c.reason, r)) {
            ++c.count;
            return;
        }
    }
    v.push_back({ r, 1 });
}

}  // namespace moe_swap_detail

// How often `reason` was counted in `v`.
inline uint64_t moe_swap_reason_count_of(const std::vector<moe_swap_reason_count> & v, const char * reason) {
    for (const moe_swap_reason_count & c : v) {
        if (moe_swap_detail::moe_swap_same_text(c.reason, reason)) {
            return c.count;
        }
    }
    return 0;
}

// Everything the summary lines print. Cumulative over the driver's life. The accepted transactions are counted apart
// from the triplets they moved: each one publishes a plan, and a published plan can make a context drop what it
// recorded, so it is the transactions, not the triplets, that can cost decode tokens their replay.
struct moe_swap_driver_stats {
    uint64_t                           boundaries               = 0;
    uint64_t                           boundaries_off           = 0;
    uint64_t                           boundaries_skipped       = 0;
    uint64_t                           boundaries_held          = 0;  // loans outstanding
    uint64_t                           decode_graphs            = 0;
    uint64_t                           graphs_after_move        = 0;  // decode graphs after a boundary that moved
    uint64_t                           proposals                = 0;
    uint64_t                           evicts                   = 0;
    uint64_t                           evict_bytes              = 0;
    uint64_t                           admits                   = 0;
    uint64_t                           admit_bytes              = 0;
    uint64_t                           restores                 = 0;
    uint64_t                           transactions_evict       = 0;  // accepted transactions of evicts
    uint64_t                           transactions_admit       = 0;  // accepted transactions of admits and restores
    uint64_t                           prompt_boundaries        = 0;
    uint64_t                           prompt_proposals         = 0;
    uint64_t                           prompt_admits            = 0;
    uint64_t                           prompt_admit_bytes       = 0;
    uint64_t                           open_slots               = 0;
    uint64_t                           deferred_room            = 0;  // pairs deferred for host room
    uint64_t                           room_reads               = 0;
    size_t                             last_room                = 0;
    uint64_t                           boundaries_no_admissible = 0;
    uint64_t                           file_mapped_passed       = 0;  // file-mapped triplets at the last such boundary
    uint64_t                           boundaries_after_prompt  = 0;  // boundaries the caller saw a prompt graph before
    uint64_t                           settled_batches          = 0;  // batches whose transaction threw, settled later
    uint64_t                           settled_moved            = 0;  // their moves the next view showed had moved
    uint32_t                           pending_high_water       = 0;
    const char *                       off_reason               = nullptr;
    std::vector<moe_swap_reason_count> refused;
    std::vector<moe_swap_reason_count> busy;
    std::vector<moe_swap_reason_count> dropped;
};

// One driver per route table. The vectors are sized once, so a boundary with nothing to do allocates nothing.
struct moe_swap_driver_state {
    moe_swap_driver_state(int n_blocks, int n_experts, const moe_swap_params & p = moe_swap_params{}) :
        params(p),
        policy(n_blocks, n_experts) {
        const size_t nb    = static_cast<size_t>(policy.n_blocks);
        const size_t cells = nb * static_cast<size_t>(policy.n_experts);
        decode.assign(cells, 0);
        prompt.assign(cells, 0);
        blocks.assign(nb, moe_swap_block_input{});
        latched.assign(nb, 0);
        open_slots.assign(nb, 0);
    }

    moe_swap_params                   params;
    moe_swap_state                    policy;
    std::vector<uint64_t>             decode;  // the route counters' snapshot, refreshed only for a call that absorbs
    std::vector<uint64_t>             prompt;
    std::vector<moe_swap_block_input> blocks;
    std::vector<uint8_t>              latched;     // per block: a final admit refusal, or an evict one that latches
    std::vector<uint32_t>             open_slots;  // per block: slots a dropped or refused restore left open
    std::vector<moe_swap_pending>     pending;
    std::vector<moe_swap_unsettled>   unsettled;   // the batch of a transaction that threw, until the next view
    std::vector<moe_swap_proposal>    proposals;
    std::vector<moe_swap_move>        batch;
    bool                              last_boundary_moved = false;
    moe_swap_driver_stats             stats;
};

// K, the prompt boundary's gate (moe-swap-policy.hpp): 2 x the CPU's rate for the routed experts over the
// host-to-device rate, so a pair that clears it costs the CPU more over a generation as long as the prompt than its
// two copies cost once. 0, the decode rule, when either rate is not a positive number (unprobed, or NaN).
inline double moe_swap_prompt_gate_of(double cpu_gbps, double h2d_gbps) {
    if (!(cpu_gbps > 0.0) || !(h2d_gbps > 0.0)) {
        return 0.0;
    }
    return 2.0 * cpu_gbps / h2d_gbps;
}

// What the caller knows of this boundary.
struct moe_swap_driver_call {
    bool   after_prompt      = false;  // a prompt pass ran since the last boundary
    int    n_tokens          = 0;      // decode graphs the context computed since the last boundary
    bool   loans_outstanding = false;  // triplets are lent to a prompt ring
    bool   cache_settled     = true;   // nothing lent and no deferred free pending
    double prompt_gate       = 0.0;    // K for the policy: 0 when the host-to-device rate is not probed
};

enum moe_swap_driver_status {
    MOE_SWAP_DRIVER_IDLE,  // nothing sent
    MOE_SWAP_DRIVER_SENT,  // one transaction sent
    MOE_SWAP_DRIVER_HELD,  // loans outstanding: nothing sent
    MOE_SWAP_DRIVER_OFF,   // the plan does not allow the driver: nothing read, nothing sent
};

// `n` boundaries the caller did not run the driver at, because another graph was in flight or the instance was
// claimed.
inline void moe_swap_driver_note_skipped(moe_swap_driver_state & d, uint64_t n = 1) {
    d.stats.boundaries += n;
    d.stats.boundaries_skipped += n;
}

// `n` boundaries the caller turned the driver off at, with why (multithreaded graphs, say).
inline void moe_swap_driver_note_off(moe_swap_driver_state & d, const char * reason, uint64_t n = 1) {
    if (n == 0) {
        return;
    }
    d.stats.boundaries += n;
    d.stats.boundaries_off += n;
    d.stats.off_reason = reason;
}

namespace moe_swap_detail {

inline bool moe_swap_view_fits(const moe_swap_driver_state & d, const moe_swap_plan_view & v) {
    const size_t nb    = static_cast<size_t>(d.policy.n_blocks);
    const size_t cells = nb * static_cast<size_t>(d.policy.n_experts);
    return d.policy.n_blocks > 0 && v.n_blocks == d.policy.n_blocks && v.n_experts == d.policy.n_experts &&
           v.resident.size() == cells && v.admissible.size() == cells && v.eligible.size() == nb &&
           v.pair_bytes.size() == nb && v.admissible_count.size() == nb;
}

inline void moe_swap_open_slot(moe_swap_driver_state & d, int block) {
    ++d.open_slots[static_cast<size_t>(block)];
    ++d.stats.open_slots;
}

// A pending admit that will not land: an admit becomes a restore of the triplet its pair evicted, and a restore
// leaves its slot open. Returns whether it stays pending.
inline bool moe_swap_pending_give_up(moe_swap_driver_state & d, moe_swap_pending & p) {
    if (p.restore) {
        moe_swap_open_slot(d, p.block);
        return false;
    }
    p.restore     = true;
    p.busy_streak = 0;
    p.busy_reason = nullptr;
    return true;
}

inline void moe_swap_off(moe_swap_driver_state & d, const char * reason) {
    ++d.stats.boundaries_off;
    d.stats.off_reason = reason;
}

// The batch of a transaction that threw, settled from `view`, the published plan the boundary reads: a triplet moved
// when it is where its move takes it. A moved evict leaves its admit pending, and an evict that did not move is
// dropped (BUSY, MOE_SWAP_UNSETTLED_REASON; the policy may propose it again). A moved admit leaves pending, and one
// that did not move goes back to pending with its streak as it was, for this boundary to send again. The outcome list
// that carried the bytes was lost with the throw, so settled moves add none. Returns whether anything had moved.
inline bool moe_swap_settle(moe_swap_driver_state & d, const moe_swap_plan_view & view) {
    moe_swap_driver_stats & st          = d.stats;
    const size_t            ne          = static_cast<size_t>(d.policy.n_experts);
    bool                    evict_moved = false;
    bool                    admit_moved = false;
    ++st.settled_batches;
    for (const moe_swap_unsettled & u : d.unsettled) {
        const moe_swap_pending & p = u.pair;
        if (u.kind == MOE_SWAP_MOVE_EVICT) {
            if (view.resident[static_cast<size_t>(p.block) * ne + static_cast<size_t>(p.evict_expert)] == 0) {
                evict_moved = true;
                ++st.settled_moved;
                ++st.evicts;
                d.pending.push_back(p);
            } else {
                moe_swap_count_reason(st.busy, MOE_SWAP_UNSETTLED_REASON);
            }
            continue;
        }
        const int expert = p.restore ? p.evict_expert : p.admit_expert;
        if (view.resident[static_cast<size_t>(p.block) * ne + static_cast<size_t>(expert)] != 0) {
            admit_moved = true;
            ++st.settled_moved;
            if (p.restore) {
                ++st.restores;
            } else {
                ++st.admits;
                st.prompt_admits += p.prompt ? 1 : 0;
            }
        } else {
            d.pending.push_back(p);
        }
    }
    d.unsettled.clear();
    st.transactions_evict += evict_moved ? 1 : 0;
    st.transactions_admit += admit_moved ? 1 : 0;
    st.pending_high_water = std::max(st.pending_high_water, static_cast<uint32_t>(d.pending.size()));
    return evict_moved || admit_moved;
}

}  // namespace moe_swap_detail

// The route table was replaced while the load the driver ran under stays: `to`, the new table's driver, takes what
// `from` has in flight (pending admits, an unsettled batch, latches and open slots), so nothing it moved is forgotten.
// The policy's usage starts over with the new table's counters. False, with nothing moved, when the shapes differ.
inline bool moe_swap_driver_carry(moe_swap_driver_state & from, moe_swap_driver_state & to) {
    if (from.policy.n_blocks != to.policy.n_blocks || from.policy.n_experts != to.policy.n_experts) {
        return false;
    }
    to.pending.swap(from.pending);
    to.unsettled.swap(from.unsettled);
    to.latched.swap(from.latched);
    to.open_slots.swap(from.open_slots);
    to.stats.pending_high_water = static_cast<uint32_t>(to.pending.size());
    return true;
}

// The route table was replaced under another load, or with another shape: what `d` has in flight will not land, and
// each such move counts as an open slot in d's figures before its report prints. A pending admit is one; so is each
// move of an unsettled batch, an evict among them counted though it may not have moved, since no view settles it now.
// Returns how many were counted.
inline uint64_t moe_swap_driver_drop_in_flight(moe_swap_driver_state & d) {
    const uint64_t n = static_cast<uint64_t>(d.pending.size()) + static_cast<uint64_t>(d.unsettled.size());
    for (const moe_swap_pending & p : d.pending) {
        moe_swap_detail::moe_swap_open_slot(d, p.block);
    }
    for (const moe_swap_unsettled & u : d.unsettled) {
        moe_swap_detail::moe_swap_open_slot(d, u.pair.block);
    }
    d.pending.clear();
    d.unsettled.clear();
    return n;
}

// One synchronize boundary. `counts(decode, prompt)` copies the route counters into two arrays of n_blocks x
// n_experts; it is called only for a policy call that absorbs. `room()` returns the pinned host bytes the evicts may
// take; it is called once, and only at a boundary with proposals to send. `txn(batch)` runs the named move on the
// batch and fills each move's outcome, reason, bytes and block latch. An exception out of `txn` leaves the batch
// unsettled in `d` and propagates.
template <typename Counts, typename Room, typename Txn>
moe_swap_driver_status moe_swap_driver_step(moe_swap_driver_state &      d,
                                            const moe_swap_plan_view &   view,
                                            const moe_swap_driver_call & c,
                                            Counts &&                    counts,
                                            Room &&                      room,
                                            Txn &&                       txn) {
    moe_swap_driver_stats & st     = d.stats;
    const uint64_t          graphs = static_cast<uint64_t>(c.n_tokens > 0 ? c.n_tokens : 0);
    ++st.boundaries;
    st.decode_graphs += graphs;
    st.boundaries_after_prompt += c.after_prompt ? 1 : 0;
    bool after_move       = d.last_boundary_moved;
    d.last_boundary_moved = false;

    if (!moe_swap_detail::moe_swap_view_fits(d, view)) {
        st.graphs_after_move += after_move ? graphs : 0;
        moe_swap_detail::moe_swap_off(d, "plan view does not match the route table");
        return MOE_SWAP_DRIVER_OFF;
    }
    // Before anything is decided: a batch whose transaction threw is settled from this view.
    if (!d.unsettled.empty() && moe_swap_detail::moe_swap_settle(d, view)) {
        after_move = true;
    }
    st.graphs_after_move += after_move ? graphs : 0;
    if (!view.single_device) {
        moe_swap_detail::moe_swap_off(d, "multi-device plan");
        return MOE_SWAP_DRIVER_OFF;
    }
    if (!view.device_held) {
        moe_swap_detail::moe_swap_off(d, "card held only through experts");
        return MOE_SWAP_DRIVER_OFF;
    }

    const size_t nb          = static_cast<size_t>(d.policy.n_blocks);
    const size_t ne          = static_cast<size_t>(d.policy.n_experts);
    uint64_t     n_admitting = 0;
    for (size_t b = 0; b < nb; ++b) {
        moe_swap_block_input & in = d.blocks[b];
        in.decode_counts          = d.decode.data() + b * ne;
        in.prompt_counts          = d.prompt.data() + b * ne;
        in.resident               = view.resident.data() + b * ne;
        in.admissible             = view.admissible.data() + b * ne;
        in.eligible               = view.eligible[b] != 0 && d.latched[b] == 0 && d.open_slots[b] == 0;
        n_admitting += in.eligible ? view.admissible_count[b] : 0u;
    }
    if (n_admitting == 0) {
        ++st.boundaries_no_admissible;
        st.file_mapped_passed = view.file_mapped;
    }

    moe_swap_call pc;
    pc.blocks       = d.blocks.data();
    pc.n_blocks     = d.policy.n_blocks;
    pc.n_experts    = d.policy.n_experts;
    pc.n_tokens     = c.n_tokens;
    pc.in_flight    = !d.pending.empty() || c.loans_outstanding;
    pc.after_prompt = c.after_prompt;
    pc.prompt_gate  = c.prompt_gate;

    bool                  prompt = false;
    const moe_swap_status peek   = moe_swap_peek(d.policy, d.params, pc, &prompt);
    if (peek == MOE_SWAP_STATUS_SEEDED || peek == MOE_SWAP_STATUS_ADAPTED) {
        counts(d.decode.data(), d.prompt.data());
    }
    const moe_swap_status ps = moe_swap_step(d.policy, d.params, pc, d.proposals);
    if (ps == MOE_SWAP_STATUS_REJECTED) {
        moe_swap_detail::moe_swap_off(d, "policy refused its parameters");
        return MOE_SWAP_DRIVER_OFF;
    }

    if (c.loans_outstanding) {
        ++st.boundaries_held;
        return MOE_SWAP_DRIVER_HELD;
    }

    if (!d.pending.empty()) {
        d.batch.clear();
        for (const moe_swap_pending & p : d.pending) {
            moe_swap_move m;
            m.kind   = MOE_SWAP_MOVE_ADMIT;
            m.block  = p.block;
            m.expert = p.restore ? p.evict_expert : p.admit_expert;
            d.batch.push_back(m);
        }
        // Reserved before the call, so holding the batch after a throw allocates nothing.
        d.unsettled.reserve(d.pending.size());
        try {
            txn(d.batch);
        } catch (...) {
            for (const moe_swap_pending & p : d.pending) {
                d.unsettled.push_back({ MOE_SWAP_MOVE_ADMIT, p });
            }
            d.pending.clear();
            throw;
        }
        size_t kept = 0;
        for (size_t i = 0; i < d.pending.size(); ++i) {
            moe_swap_pending &    p    = d.pending[i];
            const moe_swap_move & m    = d.batch[i];
            bool                  stay = false;
            if (m.outcome == MOE_SWAP_MOVE_MOVED) {
                d.last_boundary_moved = true;
                if (p.restore) {
                    ++st.restores;
                } else {
                    ++st.admits;
                    st.admit_bytes += m.bytes;
                    if (p.prompt) {
                        ++st.prompt_admits;
                        st.prompt_admit_bytes += m.bytes;
                    }
                }
            } else if (m.outcome == MOE_SWAP_MOVE_BUSY) {
                moe_swap_detail::moe_swap_count_reason(st.busy, m.reason);
                if (!c.cache_settled) {
                    p.busy_streak = 0;
                } else if (moe_swap_detail::moe_swap_same_text(p.busy_reason, m.reason)) {
                    ++p.busy_streak;
                } else {
                    p.busy_streak = 1;
                }
                p.busy_reason = m.reason;
                stay          = true;
                if (p.busy_streak >= MOE_SWAP_PENDING_DROP_STREAK) {
                    moe_swap_detail::moe_swap_count_reason(st.dropped, m.reason);
                    stay = moe_swap_detail::moe_swap_pending_give_up(d, p);
                }
            } else {
                moe_swap_detail::moe_swap_count_reason(st.refused, m.reason);
                d.latched[static_cast<size_t>(p.block)] = 1;
                stay                                    = moe_swap_detail::moe_swap_pending_give_up(d, p);
            }
            if (stay) {
                d.pending[kept++] = p;
            }
        }
        d.pending.resize(kept);
        if (d.last_boundary_moved) {
            ++st.transactions_admit;
        }
        return MOE_SWAP_DRIVER_SENT;
    }

    if (ps != MOE_SWAP_STATUS_ADAPTED) {
        return MOE_SWAP_DRIVER_IDLE;
    }
    if (prompt) {
        ++st.prompt_boundaries;
    }
    const size_t n = d.proposals.size();
    if (n == 0) {
        return MOE_SWAP_DRIVER_IDLE;
    }
    st.proposals += n;
    if (prompt) {
        st.prompt_proposals += n;
    }

    // The hottest pairs that fit the host room go; the rest wait for a later boundary.
    const size_t free_room = room();
    ++st.room_reads;
    st.last_room = free_room;
    size_t keep  = 0;
    size_t used  = 0;
    while (keep < n) {
        const size_t need = view.pair_bytes[static_cast<size_t>(d.proposals[keep].block)];
        if (need > free_room - used) {
            break;
        }
        used += need;
        ++keep;
    }
    st.deferred_room += n - keep;
    if (keep == 0) {
        // Not even one pair fits: the boundary is spent (its pairs are counted as deferred), and a prompt boundary is
        // not kept, so the next decode boundary follows the decode rule instead of reading every counter again.
        return MOE_SWAP_DRIVER_IDLE;
    }
    if (prompt && keep < n) {
        moe_swap_prompt_boundary_defer(d.policy);  // part moved: the next boundary takes the rest
    }

    d.batch.clear();
    for (size_t i = 0; i < keep; ++i) {
        moe_swap_move m;
        m.kind   = MOE_SWAP_MOVE_EVICT;
        m.block  = d.proposals[i].block;
        m.expert = d.proposals[i].evict_expert;
        d.batch.push_back(m);
    }
    // Reserved before the call, so holding the batch after a throw allocates nothing.
    d.unsettled.reserve(keep);
    try {
        txn(d.batch);
    } catch (...) {
        for (size_t i = 0; i < keep; ++i) {
            moe_swap_unsettled u;
            u.kind              = MOE_SWAP_MOVE_EVICT;
            u.pair.block        = d.proposals[i].block;
            u.pair.admit_expert = d.proposals[i].admit_expert;
            u.pair.evict_expert = d.proposals[i].evict_expert;
            u.pair.prompt       = prompt;
            d.unsettled.push_back(u);
        }
        throw;
    }
    for (size_t i = 0; i < keep; ++i) {
        const moe_swap_move &     m = d.batch[i];
        const moe_swap_proposal & q = d.proposals[i];
        if (m.outcome == MOE_SWAP_MOVE_MOVED) {
            d.last_boundary_moved = true;
            ++st.evicts;
            st.evict_bytes += m.bytes;
            moe_swap_pending p;
            p.block        = q.block;
            p.admit_expert = q.admit_expert;
            p.evict_expert = q.evict_expert;
            p.prompt       = prompt;
            d.pending.push_back(p);
        } else if (m.outcome == MOE_SWAP_MOVE_BUSY) {
            moe_swap_detail::moe_swap_count_reason(st.busy, m.reason);
        } else {
            // A final refusal drops this pair; only one the transaction says every evict of the block would meet
            // keeps the block from evicting again.
            moe_swap_detail::moe_swap_count_reason(st.refused, m.reason);
            if (m.latches_block) {
                d.latched[static_cast<size_t>(q.block)] = 1;
            }
        }
    }
    if (d.last_boundary_moved) {
        ++st.transactions_evict;
    }
    st.pending_high_water = std::max(st.pending_high_water, static_cast<uint32_t>(d.pending.size()));
    return MOE_SWAP_DRIVER_SENT;
}

namespace moe_swap_detail {

inline double moe_swap_mib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// "none", or "reason xN, reason xM" in the order the reasons were first met.
inline std::string moe_swap_reason_list(const std::vector<moe_swap_reason_count> & v) {
    if (v.empty()) {
        return "none";
    }
    std::string out;
    char        n[32];
    for (const moe_swap_reason_count & c : v) {
        if (!out.empty()) {
            out += ", ";
        }
        out += c.reason;
        std::snprintf(n, sizeof(n), " x%llu", static_cast<unsigned long long>(c.count));
        out += n;
    }
    return out;
}

}  // namespace moe_swap_detail

// The first summary line: what the driver saw and moved. `k_text` says the prompt gate in effect.
inline std::string moe_swap_driver_summary_moves(const moe_swap_driver_stats & st, int device, const char * k_text) {
    using ull = unsigned long long;
    char line[640];
    std::snprintf(line, sizeof(line),
                  "[MOE-SWAP] device %d: %llu boundaries (%llu off, %llu skipped, %llu held by loans), %llu decode "
                  "graphs (%llu after a move), %llu proposals, %llu evicts (%.1f MiB), %llu admits (%.1f MiB), %llu "
                  "restores, pending high-water %u; prompt boundaries %llu: %llu proposals, %llu admits (%.1f MiB), "
                  "K %s",
                  device, (ull) st.boundaries, (ull) st.boundaries_off, (ull) st.boundaries_skipped,
                  (ull) st.boundaries_held, (ull) st.decode_graphs, (ull) st.graphs_after_move, (ull) st.proposals,
                  (ull) st.evicts, moe_swap_detail::moe_swap_mib(st.evict_bytes), (ull) st.admits,
                  moe_swap_detail::moe_swap_mib(st.admit_bytes), (ull) st.restores, st.pending_high_water,
                  (ull) st.prompt_boundaries, (ull) st.prompt_proposals, (ull) st.prompt_admits,
                  moe_swap_detail::moe_swap_mib(st.prompt_admit_bytes), k_text != nullptr ? k_text : "unknown");
    return line;
}

// The second summary line: the accepted transactions beside the decode graphs they ran among (a round is two, and
// each can make a context drop what it recorded), then what the driver could not do, and why.
inline std::string moe_swap_driver_summary_faults(const moe_swap_driver_stats & st, int device) {
    using ull = unsigned long long;
    char room[64];
    if (st.room_reads != 0) {
        std::snprintf(room, sizeof(room), "last room %.1f MiB", moe_swap_detail::moe_swap_mib(st.last_room));
    } else {
        std::snprintf(room, sizeof(room), "room never read");
    }
    char off[160];
    if (st.off_reason != nullptr) {
        std::snprintf(off, sizeof(off), "%s (%llu boundaries)", st.off_reason, (ull) st.boundaries_off);
    } else {
        std::snprintf(off, sizeof(off), "none");
    }
    char head[640];
    std::snprintf(head, sizeof(head),
                  "[MOE-SWAP] device %d: accepted transactions %llu (evict %llu, admit %llu) over %llu decode graphs; "
                  "open slots %llu; deferred for host room %llu (%s); no admissible host expert at %llu boundaries "
                  "(%llu file-mapped triplets passed over); off: %s",
                  device, (ull) (st.transactions_evict + st.transactions_admit), (ull) st.transactions_evict,
                  (ull) st.transactions_admit, (ull) st.decode_graphs, (ull) st.open_slots, (ull) st.deferred_room,
                  room, (ull) st.boundaries_no_admissible, (ull) st.file_mapped_passed, off);
    char tail[256];
    std::snprintf(tail, sizeof(tail), "; settled from the plan after a throw: %llu batches (%llu moves moved)",
                  (ull) st.settled_batches, (ull) st.settled_moved);
    std::string out = std::string(head) + tail + "; refused: " + moe_swap_detail::moe_swap_reason_list(st.refused) +
                      "; busy: " + moe_swap_detail::moe_swap_reason_list(st.busy) +
                      "; dropped pending: " + moe_swap_detail::moe_swap_reason_list(st.dropped);
    // The backend counts a graph as decode only when it is one row, so parallel sequences or speculative decoding
    // make every boundary a prompt boundary: the policy is seeded and never swaps.
    if (st.decode_graphs == 0 && st.boundaries_after_prompt != 0) {
        std::snprintf(tail, sizeof(tail),
                      "; no single-row decode graph at %llu boundaries: every one read as a prompt, so nothing swapped",
                      (ull) st.boundaries_after_prompt);
        out += tail;
    }
    return out;
}

}  // namespace ggml_sycl
