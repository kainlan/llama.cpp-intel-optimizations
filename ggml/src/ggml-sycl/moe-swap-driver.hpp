// The decode-boundary expert swap driver (GGML_SYCL_MOE_SWAP=1): what one synchronize boundary does with the swap
// policy's answer, as a pure function of a plan view, the route counters and the policy state. No SYCL, no ggml, no
// cache, no logging, no environment and no state outside moe_swap_driver_state, which the caller owns, one per route
// table. The caller (ggml_sycl_run_expert_swap_at_boundary in ggml-sycl.cpp gates the boundary; its claimed body
// ggml_sycl_run_expert_swap_claimed) builds the view from the published plan (moe_swap_plan_view_build,
// moe-stream-plan.hpp), and passes three callables: one that copies the route counters, one that reads the host room,
// one that runs the named move.
//
// One boundary sends one transaction at most, never both directions:
//   - Pending admits first. Each evict that moved leaves its admit pending, and the next boundary sends every pending
//     admit it does not hold (below) and nothing else, before the policy may propose. While any is pending the policy
//     is in flight.
//   - Otherwise the policy's proposals, as evicts only. An evict's device room comes back only when a deferred-free
//     pass reaps the entries it retired, never to an admit in the same call, so the admit waits a boundary.
//   - A pending admit refused finally becomes a restore: an admit of the triplet its pair evicted, under the same
//     pending rule. A restore refused finally, or dropped, leaves the slot open: counted, and its block takes no new
//     evict, so the block never holds fewer resident experts than the plan sized it for without the summary saying so.
//   - A pending admit that is BUSY at MOE_SWAP_PENDING_DROP_STREAK settled boundaries in a row with one reason is
//     dropped (and becomes a restore, or an open slot if it was one). A boundary is settled when nothing is lent and
//     no deferred free is pending: an unsettled boundary may yet give the room back, so it starts the streak over.
//   - A pending admit that was BUSY for device zone room (moe_swap_move::busy_cause) is held, not sent, while the
//     zone's room epoch (moe_swap_driver_call::room_epoch) is the one that answer's transaction left: the zone has
//     gained no room since, so the same request would be refused again. A transaction moves the epoch itself when it
//     gives back storage it took, which is no room gained, so the step takes those moves out of the epoch's change
//     (moe_swap_txn_room): they neither keep the refused admit from being held nor send a held one again. A boundary
//     it is held at does not count toward its drop streak, and the policy stays in flight. An epoch that moved for
//     any other reason sends it again, and that answer counts toward the streak like any other. It is given up at the
//     MOE_SWAP_PENDING_HOLD_BOUND-th settled boundary at which no outside room arrived (the epoch moved by no more
//     than the boundary's own give-backs), as a dropped admit is (a restore, which goes at that boundary, or an open
//     slot), so room that never comes does not hold the policy in flight for the rest of the generation. An unsettled
//     boundary neither counts nor resets the hold.
//   - An evict refused finally latches its block only when the transaction marks the refusal as one every evict of
//     the block would meet (moe_swap_move::latches_block: the caller sets it for a device copy that shares a bulk
//     tensor's storage). Any other final refusal drops that proposal alone, counted by its reason.
//   - A transaction that throws leaves its batch unsettled: the moves it was sent wait in the state, and the next
//     boundary with no loans outstanding, before anything else, settles each from the view it is handed, since the
//     published plan is the record of what moved (a loan publishes lent triplets off the card, so a boundary with
//     loans outstanding settles nothing). A triplet that is where its move takes it moved; an evict that did not move
//     is dropped (BUSY, MOE_SWAP_UNSETTLED_REASON), and an admit that did not move stays pending. A throw counts as a
//     BUSY answer, MOE_SWAP_UNSETTLED_REASON, for each pending admit it was sent, and advances its streak, so an
//     admit whose every transaction throws is dropped under the streak rule like any other.
//   - With loans outstanding nothing is sent: the loan's ring owns the room, and the policy is held in flight.
//   - The batch is trimmed to the host room, hottest first: an evict takes pinned host room for its three copies,
//     which it may not grow (the host zone was sized by the plan). What does not fit is deferred and counted apart
//     from refusals. A prompt batch trimmed after part of it fits keeps the prompt boundary for the rest; one with
//     room for no pair spends it, and the decode rule follows.
//   - Off, with nothing read and nothing sent, on a plan that spans cards or holds the card only through experts.
//
// Cost: a boundary that reaches no boundary of the policy and has nothing pending reads, per block, the view's
// eligible flag and admissible count and the state's latch and open slot, reads no route counter and no cell, and
// allocates nothing. A boundary whose pending admits are all held sends nothing and allocates nothing either.
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
// entry changed under it, a submit that threw) or a full device zone, whose answer counts here only when it is sent:
// while the zone's room epoch stands the admit is held instead (MOE_SWAP_PENDING_HOLD_BOUND). One such answer can be
// a transient met by chance; the same answer at four settled boundaries in a row says it is not passing, and four
// bounds how long an admit that will not land holds the policy in flight. It is a chosen bound, not a measured one.
constexpr int MOE_SWAP_PENDING_DROP_STREAK = 4;

// A pending admit held for zone room is given up at the 16th settled boundary at which no outside room arrived (the
// epoch moved by no more than the boundary's own give-backs); an unsettled boundary neither counts nor resets the
// hold. It is held at the fifteen before it. An admit refused for zone room is not sent again until the epoch says
// the zone could have gained room; when no room ever comes, the hold alone would keep the policy in flight for the
// rest of the generation. Sixteen is above the case the hold exists for, the few boundaries until a deferred-free
// pass hands an evict's room back. At 43 ms a decode token one hold is about 0.7 s, and a pair that never fits is
// held twice, as the admit and then as the restore it becomes: about 32 settled boundaries, about 1.4 s, with the
// policy in flight before the slot is left open. A held boundary costs the admit nothing and a sent one a lookup in
// the zone's allocator, so the bound trades only how long the policy waits. It is a chosen bound, not a measured one.
constexpr int MOE_SWAP_PENDING_HOLD_BOUND = 16;

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

// Why a move was BUSY, where the driver acts on it: a refusal for device zone room is passing only once the zone
// gains room, which the room epoch (moe_swap_driver_call::room_epoch) says; every other one may pass by itself.
enum moe_swap_busy_cause {
    MOE_SWAP_BUSY_CAUSE_TRANSIENT,
    MOE_SWAP_BUSY_CAUSE_ZONE_ROOM,  // the device zone had no room for the triplet
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
    // For a BUSY admit: what it waits for. The driver does not read refusal texts for this either.
    moe_swap_busy_cause   busy_cause    = MOE_SWAP_BUSY_CAUSE_TRANSIENT;
};

// What one transaction cost the boundary thread, in microseconds: the copy steps' submission, the id handovers' time in
// the cache, the metadata lookups, and the call's whole time from its start, which shows what the three leave out. The
// driver sums them over the transactions it sends (moe_swap_driver_stats), so the move's own per-move line can stay a
// DEBUG line and the run's cost is still in the summary.
struct moe_swap_txn_cost {
    uint64_t copy_us  = 0;
    uint64_t remap_us = 0;
    uint64_t meta_us  = 0;
    uint64_t total_us = 0;
    // The placement plan's entry chunks the transaction's copy of the plan cloned to record its moves: the
    // copy-on-write's cost, so a change that makes the publication clone the whole plan again shows in the summary.
    uint64_t plan_chunks_cloned = 0;
    // The transaction's lent-view drops that walked a whole route table or hybrid decode record because the moved
    // expert's own slot did not hold the retired entry: a view the slot rule missed is still dropped, and counted here.
    uint64_t view_scan_fallbacks = 0;
};

// What a transaction says of the device zone's room epoch once it has returned. A transaction moves the epoch itself
// when it gives back blocks it took: the members a triplet minted before the zone refused a later one, and whatever a
// prepared triplet held when the move did not take it. That is room the zone had before the transaction, not room it
// gained, so the caller measures it: epoch_after is the epoch read after the transaction, and self_delta the sum of
// the epoch's moves across each of the transaction's own give-backs, read right before and right after each. The step
// holds an admit refused for room only when the two account for every move since the boundary's own read
// (moe_swap_driver_call::room_epoch): epoch_after - room_epoch == self_delta. It is measured, not counted, so it
// does not depend on how many moves one give-back makes. A block another thread gives back inside one of those reads
// is taken for the transaction's own, and the admit it would have sent is then held, at most to
// MOE_SWAP_PENDING_HOLD_BOUND.
struct moe_swap_txn_room {
    uint64_t          epoch_after = 0;
    uint64_t          self_delta  = 0;
    moe_swap_txn_cost cost;
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
    bool         room_held    = false;  // its last answer was BUSY for device zone room, and the zone gained none since
    uint64_t     room_epoch   = 0;      // the zone's room epoch it is held under: the one the last transaction left
    int          held_streak  = 0;      // settled boundaries it was held at since that answer; unsettled ones add none
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
    uint64_t                           admits_room_held         = 0;  // pending admits a boundary held, not sent
    uint64_t                           room_reads               = 0;
    size_t                             last_room                = 0;
    uint64_t                           boundaries_no_admissible = 0;
    uint64_t                           file_mapped_passed       = 0;  // file-mapped triplets at the last such boundary
    uint64_t                           boundaries_after_prompt  = 0;  // boundaries the caller saw a prompt graph before
    uint64_t                           settled_batches          = 0;  // batches whose transaction threw, settled later
    uint64_t                           settled_moved            = 0;  // their moves the next view showed had moved
    uint64_t                           unsettled_moves          = 0;  // moves of a thrown batch not settled yet
    uint32_t                           pending_high_water       = 0;
    // Transactions sent, accepted or refused (one that threw is not counted), and their cost to the boundary thread:
    // the copy, remap and meta parts the move times (0 for a call refused before its preparation) and the whole call.
    uint64_t                           txn_sent                 = 0;
    uint64_t                           txn_copy_us              = 0;
    uint64_t                           txn_remap_us             = 0;
    uint64_t                           txn_meta_us              = 0;
    uint64_t                           txn_total_us             = 0;
    uint64_t                           txn_plan_chunks_cloned   = 0;
    uint64_t                           txn_view_scan_fallbacks  = 0;
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
    bool     after_prompt      = false;  // a prompt pass ran since the last boundary
    int      n_tokens          = 0;      // decode graphs the context computed since the last boundary
    bool     loans_outstanding = false;  // triplets are lent to a prompt ring
    bool     cache_settled     = true;   // nothing lent and no deferred free pending
    double   prompt_gate       = 0.0;    // K for the policy: 0 when the host-to-device rate is not probed
    // The device WEIGHT zone's room epoch, read before the transaction runs: a count that moves each time the zone
    // could have gained room, and never for an allocation or a refusal. Read before, so the step can tell, with what
    // the transaction returns (moe_swap_txn_room), whether the zone gained room while it ran.
    uint64_t room_epoch        = 0;
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

// One transaction's cost, summed into the stats whatever its answer was: a refused one still took the time, its whole
// call in total_us and, when it was refused before the preparation, 0 in the three parts. A transaction that threw
// returns nothing to sum, so it is neither reported nor counted.
inline void moe_swap_count_cost(moe_swap_driver_stats & st, const moe_swap_txn_cost & c) {
    ++st.txn_sent;
    st.txn_copy_us += c.copy_us;
    st.txn_remap_us += c.remap_us;
    st.txn_meta_us += c.meta_us;
    st.txn_total_us += c.total_us;
    st.txn_plan_chunks_cloned += c.plan_chunks_cloned;
    st.txn_view_scan_fallbacks += c.view_scan_fallbacks;
}

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
    p.room_held   = false;
    p.held_streak = 0;
    return true;
}

// Whether pending admit `p` is held at a boundary handed room epoch `epoch`: its last answer was BUSY for device zone
// room, and the zone has not gained room since.
inline bool moe_swap_pending_held(const moe_swap_pending & p, uint64_t epoch) {
    return p.room_held && p.room_epoch == epoch;
}

// Whether the zone gained no room while a transaction ran: every move its room epoch made since `before`, the
// boundary's own read, is one the transaction measured across its own give-backs.
inline bool moe_swap_room_stood(uint64_t before, const moe_swap_txn_room & room) {
    return room.epoch_after - before == room.self_delta;
}

// A BUSY answer for pending admit `p`: its streak counts settled boundaries in a row with one reason, and an unsettled
// boundary starts it over.
inline void moe_swap_pending_note_busy(moe_swap_pending & p, const char * reason, bool cache_settled) {
    if (!cache_settled) {
        p.busy_streak = 0;
    } else if (moe_swap_same_text(p.busy_reason, reason)) {
        ++p.busy_streak;
    } else {
        p.busy_streak = 1;
    }
    p.busy_reason = reason;
}

inline void moe_swap_off(moe_swap_driver_state & d, const char * reason) {
    ++d.stats.boundaries_off;
    d.stats.off_reason = reason;
}

// Runs `on_throw` when it leaves scope armed. The step declares one before it calls a transaction and dismisses it on
// the line after the call returns, so `on_throw` runs only while a throw out of the transaction unwinds, and the throw
// reaches the caller as it was thrown: no handler sees it, so none can swallow it. A destructor that throws while a
// throw unwinds ends the process, so `on_throw` must not throw: the step reserves, before the call, every list
// `on_throw` pushes into.
template <typename F> struct moe_swap_unwind_guard {
    explicit moe_swap_unwind_guard(F fn) : on_throw(fn) {}

    moe_swap_unwind_guard(const moe_swap_unwind_guard &)             = delete;
    moe_swap_unwind_guard & operator=(const moe_swap_unwind_guard &) = delete;

    ~moe_swap_unwind_guard() {
        if (armed) {
            on_throw();
        }
    }

    void dismiss() { armed = false; }

    F    on_throw;
    bool armed = true;
};

// The batch of a transaction that threw, settled from `view`, the published plan the boundary reads: a triplet moved
// when it is where its move takes it. A moved evict leaves its admit pending, and an evict that did not move is
// dropped (BUSY, MOE_SWAP_UNSETTLED_REASON; the policy may propose it again). A moved admit leaves pending, and one
// that did not move goes back to pending with its streak as the throw left it, for this boundary to send again,
// unless that streak reached MOE_SWAP_PENDING_DROP_STREAK: then it is dropped as a BUSY admit is. The outcome list
// that carried the bytes was lost with the throw, so settled moves add none. The batch leaves the state before any
// entry is settled, so an allocation that throws part way never settles an entry twice. Returns whether anything had
// moved.
inline bool moe_swap_settle(moe_swap_driver_state & d, const moe_swap_plan_view & view) {
    moe_swap_driver_stats & st          = d.stats;
    const size_t            ne          = static_cast<size_t>(d.policy.n_experts);
    bool                    evict_moved = false;
    bool                    admit_moved = false;
    d.pending.reserve(d.pending.size() + d.unsettled.size());
    std::vector<moe_swap_unsettled> batch;
    batch.swap(d.unsettled);
    st.unsettled_moves = 0;
    ++st.settled_batches;
    for (const moe_swap_unsettled & u : batch) {
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
        } else if (p.busy_streak >= MOE_SWAP_PENDING_DROP_STREAK) {
            moe_swap_count_reason(st.dropped, p.busy_reason);
            moe_swap_pending q = p;
            if (moe_swap_pending_give_up(d, q)) {
                d.pending.push_back(q);
            }
        } else {
            d.pending.push_back(p);
        }
    }
    batch.clear();
    d.unsettled.swap(batch);  // empty, with the capacity kept for the next throw
    st.transactions_evict += evict_moved ? 1 : 0;
    st.transactions_admit += admit_moved ? 1 : 0;
    st.pending_high_water = std::max(st.pending_high_water, static_cast<uint32_t>(d.pending.size()));
    return evict_moved || admit_moved;
}

}  // namespace moe_swap_detail

// The route table was replaced while the load the driver ran under stays: `to`, the new table's driver, takes what
// `from` has in flight (pending admits, an unsettled batch, latches and open slots), so nothing it moved is forgotten.
// The policy's usage starts over with the new table's counters. `to` must be freshly constructed, with nothing of its
// own in flight: the lists are exchanged, so whatever `to` held would land in `from`. False, with nothing moved, when
// the shapes differ.
inline bool moe_swap_driver_carry(moe_swap_driver_state & from, moe_swap_driver_state & to) {
    if (from.policy.n_blocks != to.policy.n_blocks || from.policy.n_experts != to.policy.n_experts) {
        return false;
    }
    to.pending.swap(from.pending);
    to.unsettled.swap(from.unsettled);
    to.latched.swap(from.latched);
    to.open_slots.swap(from.open_slots);
    to.stats.pending_high_water = static_cast<uint32_t>(to.pending.size());
    to.stats.unsettled_moves    = to.unsettled.size();
    from.stats.unsettled_moves  = 0;
    return true;
}

// The route table was replaced under another load, or with another shape: what `d` has in flight will not land, and
// each such move counts as an open slot in d's figures before its report prints. A pending admit is one; so is each
// move of an unsettled batch, since no view settles it now: an evict among them is counted though it may not have
// moved, and an admit though it may have, so the count can be high by the moves the throw had already made.
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
    d.stats.unsettled_moves = 0;
    return n;
}

// One synchronize boundary. `counts(decode, prompt)` copies the route counters into two arrays of n_blocks x
// n_experts; it is called only for a policy call that absorbs. `room()` returns the pinned host bytes the evicts may
// take; it is called once, and only at a boundary with proposals to send. `txn(batch)` runs the named move on the
// batch, fills each move's outcome, reason, bytes, block latch and busy cause, and returns what the device zone's room
// epoch is once it has returned and how far its own give-backs moved it (moe_swap_txn_room). An exception out of `txn`
// leaves the batch unsettled in `d` and propagates.
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
    // Before anything is decided: a batch whose transaction threw is settled from this view, unless loans are
    // outstanding, since a loan's publication shows lent triplets off the card whether a move took them or not.
    if (!d.unsettled.empty() && !c.loans_outstanding && moe_swap_detail::moe_swap_settle(d, view)) {
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
        // The held admits first: each settled boundary one is held at counts toward the hold bound, and at the bound it
        // is given up. An admit given up becomes a restore, no longer held, so it goes in this boundary's batch; a
        // restore given up leaves the list. Each entry is finished before the next is read, so a count that throws
        // leaves the list as it stands.
        for (size_t i = 0; i < d.pending.size();) {
            moe_swap_pending & p = d.pending[i];
            if (moe_swap_detail::moe_swap_pending_held(p, c.room_epoch)) {
                p.held_streak += c.cache_settled ? 1 : 0;
                if (p.held_streak < MOE_SWAP_PENDING_HOLD_BOUND) {
                    ++st.admits_room_held;
                } else {
                    moe_swap_detail::moe_swap_count_reason(st.dropped, p.busy_reason);
                    if (!moe_swap_detail::moe_swap_pending_give_up(d, p)) {
                        d.pending.erase(d.pending.begin() + static_cast<std::ptrdiff_t>(i));
                        continue;
                    }
                }
            }
            ++i;
        }
        d.batch.clear();
        for (const moe_swap_pending & p : d.pending) {
            if (moe_swap_detail::moe_swap_pending_held(p, c.room_epoch)) {
                continue;
            }
            moe_swap_move m;
            m.kind   = MOE_SWAP_MOVE_ADMIT;
            m.block  = p.block;
            m.expert = p.restore ? p.evict_expert : p.admit_expert;
            d.batch.push_back(m);
        }
        if (d.batch.empty()) {
            return MOE_SWAP_DRIVER_IDLE;  // every pending admit is held: nothing to send, and nothing allocated
        }
        // The guard below runs while a throw unwinds, where a throw of its own ends the process. These reserves are what
        // make its pushes safe: room for one unsettled move per admit sent, and for the throw's BUSY reason should it
        // be new. The answers are recorded after the transaction has moved what it moved, and a triplet cannot be
        // taken back, so the lists that record them are reserved here too, one entry per admit sent for each reason
        // list: an allocation that failed in the middle of the recording would leave the driver not knowing what moved.
        d.unsettled.reserve(d.unsettled.size() + d.batch.size());
        st.busy.reserve(st.busy.size() + d.batch.size());
        st.refused.reserve(st.refused.size() + d.batch.size());
        st.dropped.reserve(st.dropped.size() + d.batch.size());
        // If the transaction throws, each admit sent is BUSY for the throw, and its streak advances, so admits whose
        // every transaction throws are dropped under the streak rule (at the settle that finds them not moved). A held
        // admit was not sent, so it stays pending as it is.
        moe_swap_detail::moe_swap_unwind_guard admit_throw([&]() {
            size_t kept = 0;
            for (moe_swap_pending & p : d.pending) {
                if (moe_swap_detail::moe_swap_pending_held(p, c.room_epoch)) {
                    d.pending[kept++] = p;
                    continue;
                }
                moe_swap_detail::moe_swap_count_reason(st.busy, MOE_SWAP_UNSETTLED_REASON);
                moe_swap_detail::moe_swap_pending_note_busy(p, MOE_SWAP_UNSETTLED_REASON, c.cache_settled);
                p.room_held = false;
                d.unsettled.push_back({ MOE_SWAP_MOVE_ADMIT, p });
            }
            d.pending.resize(kept);
            st.unsettled_moves = d.unsettled.size();
        });
        const moe_swap_txn_room                room = txn(d.batch);
        admit_throw.dismiss();
        moe_swap_detail::moe_swap_count_cost(st, room.cost);
        // The transaction leaves the epoch somewhere else whenever it gave storage back. Only when that is all that
        // moved it has the zone gained no room, and only then is an admit held under the epoch the transaction left:
        // one it just refused for room, and one this boundary held and did not send.
        const bool room_stood = moe_swap_detail::moe_swap_room_stood(c.room_epoch, room);

        // The answers are in the order the batch was built in: one per pending admit that was not held, which is asked
        // of each entry before the answer changes it.
        size_t kept = 0;
        size_t sent = 0;
        for (size_t i = 0; i < d.pending.size(); ++i) {
            moe_swap_pending & p = d.pending[i];
            if (moe_swap_detail::moe_swap_pending_held(p, c.room_epoch)) {
                // Not sent. It follows the epoch past the transaction's own give-backs; when the zone gained room it
                // keeps the epoch it had, which the next boundary no longer reads, so that boundary sends it.
                p.room_epoch      = room_stood ? room.epoch_after : p.room_epoch;
                d.pending[kept++] = p;
                continue;
            }
            const moe_swap_move & m    = d.batch[sent++];
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
                moe_swap_detail::moe_swap_pending_note_busy(p, m.reason, c.cache_settled);
                // Refused for zone room with none gained while the transaction ran: held from the next boundary on,
                // under the epoch the transaction left. Room gained meanwhile may be room for it, so it is sent again.
                p.room_held   = m.busy_cause == MOE_SWAP_BUSY_CAUSE_ZONE_ROOM && room_stood;
                p.room_epoch  = room.epoch_after;
                p.held_streak = 0;
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
        moe_swap_prompt_boundary_defer(d.policy);  // part fits: the next boundary takes the rest
    }

    d.batch.clear();
    for (size_t i = 0; i < keep; ++i) {
        moe_swap_move m;
        m.kind   = MOE_SWAP_MOVE_EVICT;
        m.block  = d.proposals[i].block;
        m.expert = d.proposals[i].evict_expert;
        d.batch.push_back(m);
    }
    // The guard below runs while a throw unwinds, where a throw of its own ends the process. This reserve is what makes
    // its pushes safe: room for one unsettled move per evict sent. The answers are recorded after the transaction has
    // moved what it moved, and a triplet cannot be taken back, so the pending list and the reason lists they go into
    // are reserved here too, one entry per evict sent: an allocation that failed in the middle of the recording would
    // leave the driver not knowing what moved.
    d.unsettled.reserve(d.unsettled.size() + keep);
    d.pending.reserve(d.pending.size() + keep);
    st.busy.reserve(st.busy.size() + keep);
    st.refused.reserve(st.refused.size() + keep);
    // If the transaction throws, each evict sent waits unsettled for the next view.
    moe_swap_detail::moe_swap_unwind_guard evict_throw([&]() {
        for (size_t i = 0; i < keep; ++i) {
            moe_swap_unsettled u;
            u.kind              = MOE_SWAP_MOVE_EVICT;
            u.pair.block        = d.proposals[i].block;
            u.pair.admit_expert = d.proposals[i].admit_expert;
            u.pair.evict_expert = d.proposals[i].evict_expert;
            u.pair.prompt       = prompt;
            d.unsettled.push_back(u);
        }
        st.unsettled_moves = d.unsettled.size();
    });
    // No admit is pending at a boundary that sends evicts, so what the transaction says of the room has no reader;
    // its cost is counted.
    const moe_swap_txn_room                answer = txn(d.batch);
    evict_throw.dismiss();
    moe_swap_detail::moe_swap_count_cost(st, answer.cost);
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
// each can make a context drop what it recorded), then what the driver could not do, and why. "held for device room"
// counts each boundary a pending admit was held at, so one admit held at ten boundaries counts ten.
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
    char head[704];
    std::snprintf(head, sizeof(head),
                  "[MOE-SWAP] device %d: accepted transactions %llu (evict %llu, admit %llu) over %llu decode graphs; "
                  "open slots %llu; deferred for host room %llu (%s); held for device room %llu; no admissible host "
                  "expert at %llu boundaries (%llu file-mapped triplets passed over); off: %s",
                  device, (ull) (st.transactions_evict + st.transactions_admit), (ull) st.transactions_evict,
                  (ull) st.transactions_admit, (ull) st.decode_graphs, (ull) st.open_slots, (ull) st.deferred_room,
                  room, (ull) st.admits_room_held, (ull) st.boundaries_no_admissible, (ull) st.file_mapped_passed, off);
    char tail[256];
    std::snprintf(tail, sizeof(tail),
                  "; settled from the plan after a throw: %llu batches (%llu moves moved); unsettled at report: %llu "
                  "moves",
                  (ull) st.settled_batches, (ull) st.settled_moved, (ull) st.unsettled_moves);
    char cost[320];
    std::snprintf(cost, sizeof(cost),
                  "; transaction cost over %llu: copy submit %llu us, id remap %llu us, meta lookup %llu us, total "
                  "%llu us, plan chunks cloned %llu, lent-view table walks %llu",
                  (ull) st.txn_sent, (ull) st.txn_copy_us, (ull) st.txn_remap_us, (ull) st.txn_meta_us,
                  (ull) st.txn_total_us, (ull) st.txn_plan_chunks_cloned, (ull) st.txn_view_scan_fallbacks);
    std::string out = std::string(head) + tail + cost +
                      "; refused: " + moe_swap_detail::moe_swap_reason_list(st.refused) +
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

// The line a boundary prints when its transaction refused admits because the device WEIGHT zone had no room: how many
// of the admits it sent, what the largest refused triplet asks of the zone in all (`triplet_bytes`: three slots when it
// has a slot, printed as 3 x `slot`; its members' own bytes summed when `slot` is 0), the largest free block the zone
// had when it refused, and what the zone had free after the batch, in bytes. zone_available is every free byte and
// zone_largest_free the largest free block, so a zone with the bytes and no block large enough for a member reads
// apart from one with no bytes.
inline std::string moe_swap_zone_refusal_line(int      device,
                                              uint64_t refused,
                                              uint64_t admits,
                                              size_t   slot,
                                              size_t   triplet_bytes,
                                              size_t   largest_at_refusal,
                                              size_t   zone_available,
                                              size_t   zone_largest_free) {
    using ull = unsigned long long;
    char need[128];
    if (slot != 0) {
        std::snprintf(need, sizeof(need), "3 x %zu = %zu bytes", slot, triplet_bytes);
    } else {
        std::snprintf(need, sizeof(need), "%zu bytes, its members at their own sizes (no slot)", triplet_bytes);
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "[MOE-SWAP] device %d: WEIGHT zone refused %llu of %llu admits for room: a triplet needs %s, largest "
                  "free block at refusal %zu bytes; after the batch zone_available %zu bytes, zone_largest_free %zu "
                  "bytes",
                  device, (ull) refused, (ull) admits, need, largest_at_refusal, zone_available, zone_largest_free);
    return line;
}

// The WEIGHT zone's free blocks, printed once per device at the first refusal for room: how many there are and their
// bytes, how many allocations of `member` bytes (the refused member's size) they hold, and the `top` largest sizes
// with their counts, largest first, the rest counted. Each block counts whole, as an allocation cannot span two, and
// each allocation as `member_block` bytes, the block the zone takes for `member` (its size rounded to the zone's
// grain, which the caller asks of the allocator), so a member off the grain is not counted into the bytes its
// rounding takes. Free bytes that hold no such allocation are holes, and the sizes say which grain of the zone they
// come from.
inline std::string moe_swap_zone_free_census_line(int                         device,
                                                  const std::vector<size_t> & free_sizes,
                                                  size_t                      member,
                                                  size_t                      member_block,
                                                  size_t                      top) {
    std::vector<size_t> sizes = free_sizes;
    std::sort(sizes.begin(), sizes.end(), [](size_t a, size_t b) { return a > b; });
    size_t bytes  = 0;
    size_t fitted = 0;
    for (const size_t s : sizes) {
        bytes += s;
        fitted += member_block != 0 ? s / member_block : 0;
    }
    char head[256];
    std::snprintf(head, sizeof(head),
                  "[MOE-SWAP] device %d: WEIGHT zone free blocks at the first refusal: %zu blocks, %zu bytes, room for "
                  "%zu allocations of the refused member's %zu bytes; largest:",
                  device, sizes.size(), bytes, fitted, member);
    std::string out   = head;
    size_t      shown = 0;
    size_t      i     = 0;
    char        item[64];
    for (; i < sizes.size() && shown < top; ++shown) {
        size_t j = i;
        while (j < sizes.size() && sizes[j] == sizes[i]) {
            ++j;
        }
        std::snprintf(item, sizeof(item), "%s %zu x %zu", shown == 0 ? "" : ",", sizes[i], j - i);
        out += item;
        i = j;
    }
    if (shown == 0) {
        out += " none";
    }
    if (i < sizes.size()) {
        size_t rest_sizes = 0;
        for (size_t j = i; j < sizes.size(); ++j) {
            rest_sizes += j == i || sizes[j] != sizes[j - 1];
        }
        std::snprintf(item, sizeof(item), ", and %zu blocks of %zu smaller size%s", sizes.size() - i, rest_sizes,
                      rest_sizes == 1 ? "" : "s");
        out += item;
    }
    return out;
}

}  // namespace ggml_sycl
