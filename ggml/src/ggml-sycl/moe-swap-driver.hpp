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
//   - A refused evict latches its block: the same refusal would come back.
//   - With loans outstanding nothing is sent: the loan's ring owns the room, and the policy is held in flight.
//   - The batch is trimmed to the host room, hottest first: an evict takes pinned host room for its three copies,
//     which it may not grow (the host zone was sized by the plan). What does not fit is deferred and counted apart
//     from refusals; a trimmed prompt batch keeps the prompt boundary for the rest.
//   - Off, with nothing read and nothing sent, on a plan that spans cards or holds the card only through experts.
//
// Cost: a boundary that reaches no boundary of the policy and has nothing pending reads nothing and allocates nothing.
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

// Settled boundaries in a row with one BUSY reason after which a pending admit is dropped.
constexpr int MOE_SWAP_PENDING_DROP_STREAK = 4;

// The published plan as the driver reads it, for one device over the route table's n_blocks x n_experts cells
// (block = blk.N). Built by moe_swap_plan_view_build whenever the plan snapshot changes.
struct moe_swap_plan_view {
    int                  n_blocks  = 0;
    int                  n_experts = 0;
    std::vector<uint8_t> resident;    // per cell: the whole triplet is on this card
    std::vector<uint8_t> admissible;  // per cell: the whole triplet is on the host, pinned, in a stageable layout
    std::vector<uint8_t> eligible;    // per block: every triplet whole on one side, the residents lendable
    std::vector<size_t>  pair_bytes;  // per block: the pinned host bytes one evict takes for its three copies
    uint64_t             file_mapped   = 0;      // host triplets of eligible blocks served from the file mapping
    bool                 device_held   = false;  // something other than an expert holds this card in the plan
    bool                 single_device = false;  // the plan does not span cards
};

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

// Everything the summary lines print. Cumulative over the driver's life.
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
    std::vector<uint8_t>              latched;     // per block: a final refusal of an evict or an admit
    std::vector<uint32_t>             open_slots;  // per block: slots a dropped or refused restore left open
    std::vector<moe_swap_pending>     pending;
    std::vector<moe_swap_proposal>    proposals;
    std::vector<moe_swap_move>        batch;
    bool                              last_boundary_moved = false;
    moe_swap_driver_stats             stats;
};

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

// A boundary the caller did not run the driver at, because another graph was in flight or the instance was claimed.
inline void moe_swap_driver_note_skipped(moe_swap_driver_state & d) {
    ++d.stats.boundaries;
    ++d.stats.boundaries_skipped;
}

// A boundary the caller turned the driver off at, with why (multithreaded graphs, say).
inline void moe_swap_driver_note_off(moe_swap_driver_state & d, const char * reason) {
    ++d.stats.boundaries;
    ++d.stats.boundaries_off;
    d.stats.off_reason = reason;
}

namespace moe_swap_detail {

inline bool moe_swap_view_fits(const moe_swap_driver_state & d, const moe_swap_plan_view & v) {
    const size_t nb    = static_cast<size_t>(d.policy.n_blocks);
    const size_t cells = nb * static_cast<size_t>(d.policy.n_experts);
    return d.policy.n_blocks > 0 && v.n_blocks == d.policy.n_blocks && v.n_experts == d.policy.n_experts &&
           v.resident.size() == cells && v.admissible.size() == cells && v.eligible.size() == nb &&
           v.pair_bytes.size() == nb;
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

}  // namespace moe_swap_detail

// One synchronize boundary. `counts(decode, prompt)` copies the route counters into two arrays of n_blocks x
// n_experts; it is called only for a policy call that absorbs. `room()` returns the pinned host bytes the evicts may
// take; it is called once, and only at a boundary with proposals to send. `txn(batch)` runs the named move on the
// batch and fills each move's outcome, reason and bytes.
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
    if (d.last_boundary_moved) {
        st.graphs_after_move += graphs;
    }
    d.last_boundary_moved = false;

    if (!moe_swap_detail::moe_swap_view_fits(d, view)) {
        moe_swap_detail::moe_swap_off(d, "plan view does not match the route table");
        return MOE_SWAP_DRIVER_OFF;
    }
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
    size_t       n_admitting = 0;
    for (size_t b = 0; b < nb; ++b) {
        moe_swap_block_input & in = d.blocks[b];
        in.decode_counts          = d.decode.data() + b * ne;
        in.prompt_counts          = d.prompt.data() + b * ne;
        in.resident               = view.resident.data() + b * ne;
        in.admissible             = view.admissible.data() + b * ne;
        in.eligible               = view.eligible[b] != 0 && d.latched[b] == 0 && d.open_slots[b] == 0;
        if (in.eligible) {
            for (size_t e = 0; e < ne; ++e) {
                n_admitting += view.admissible[b * ne + e] != 0 ? 1 : 0;
            }
        }
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
        txn(d.batch);
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
    if (keep < n) {
        st.deferred_room += n - keep;
        if (prompt) {
            moe_swap_prompt_boundary_defer(d.policy);
        }
    }
    if (keep == 0) {
        return MOE_SWAP_DRIVER_IDLE;
    }

    d.batch.clear();
    for (size_t i = 0; i < keep; ++i) {
        moe_swap_move m;
        m.kind   = MOE_SWAP_MOVE_EVICT;
        m.block  = d.proposals[i].block;
        m.expert = d.proposals[i].evict_expert;
        d.batch.push_back(m);
    }
    txn(d.batch);
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
            moe_swap_detail::moe_swap_count_reason(st.refused, m.reason);
            d.latched[static_cast<size_t>(q.block)] = 1;
        }
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

// The second summary line: what the driver could not do, and why.
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
    char head[512];
    std::snprintf(head, sizeof(head),
                  "[MOE-SWAP] device %d: open slots %llu; deferred for host room %llu (%s); no admissible host expert "
                  "at %llu boundaries (%llu file-mapped triplets passed over); off: %s",
                  device, (ull) st.open_slots, (ull) st.deferred_room, room, (ull) st.boundaries_no_admissible,
                  (ull) st.file_mapped_passed, off);
    return std::string(head) + "; refused: " + moe_swap_detail::moe_swap_reason_list(st.refused) +
           "; busy: " + moe_swap_detail::moe_swap_reason_list(st.busy) +
           "; dropped pending: " + moe_swap_detail::moe_swap_reason_list(st.dropped);
}

}  // namespace ggml_sycl
