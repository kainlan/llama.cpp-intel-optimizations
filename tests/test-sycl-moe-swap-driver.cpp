// The decode-boundary swap driver (ggml-sycl/moe-swap-driver.hpp) as a pure function: a plan view, the route counters,
// the policy state and a fake transaction. SYCL-free: no device, no model, no cache. The view stands for the published
// plan, and the rig updates it after each boundary from what the transaction moved, as a republished plan would.
#include "ggml-sycl/moe-swap-driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using ggml_sycl::moe_swap_driver_call;
using ggml_sycl::moe_swap_driver_state;
using ggml_sycl::moe_swap_driver_status;
using ggml_sycl::moe_swap_move;
using ggml_sycl::moe_swap_move_kind;
using ggml_sycl::moe_swap_move_outcome;
using ggml_sycl::moe_swap_params;
using ggml_sycl::moe_swap_plan_view;

using ggml_sycl::MOE_SWAP_DRIVER_HELD;
using ggml_sycl::MOE_SWAP_DRIVER_IDLE;
using ggml_sycl::MOE_SWAP_DRIVER_OFF;
using ggml_sycl::MOE_SWAP_DRIVER_SENT;
using ggml_sycl::MOE_SWAP_MOVE_ADMIT;
using ggml_sycl::MOE_SWAP_MOVE_BUSY;
using ggml_sycl::MOE_SWAP_MOVE_EVICT;
using ggml_sycl::MOE_SWAP_MOVE_MOVED;
using ggml_sycl::MOE_SWAP_MOVE_REFUSED;

static int g_failed = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                \
        }                                                              \
    } while (0)

// Every heap allocation of this binary is counted, so a test can say a boundary allocated nothing. One thread.
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

static bool same(const char * a, const char * b) {
    return a && b && std::strcmp(a, b) == 0;
}

static bool contains(const std::string & text, const char * part) {
    return text.find(part) != std::string::npos;
}

// The caller's side of the driver: the plan view, the cumulative counters, the host room, and a transaction whose
// answers the test scripts. A rule answers the moves it matches, first match first, `times` times (forever when < 0),
// and may mark a final evict refusal as one that latches the block; a move no rule matches moves, 3 bytes. Each pair
// costs 3 bytes of host room unless a test says otherwise. `throw_after` >= 0 makes the next transaction answer that
// many moves, publish what they moved, and throw, as the named move does when a submit throws part way.
struct rig {
    struct rule {
        moe_swap_move_kind    kind;
        int                   block;
        int                   expert;
        moe_swap_move_outcome outcome;
        const char *          reason;
        int                   times;
        bool                  latch;
    };

    int                                     nb;
    int                                     ne;
    moe_swap_plan_view                      view;
    std::vector<uint64_t>                   decode;
    std::vector<uint64_t>                   prompt;
    size_t                                  room = static_cast<size_t>(-1);
    std::vector<rule>                       rules;
    std::vector<std::vector<moe_swap_move>> calls;  // every batch the transaction got, with its answers
    int                                     counts_reads     = 0;
    int                                     room_reads       = 0;
    int                                     throw_after      = -1;
    int                                     throws           = 0;
    const uint64_t *                        counts_decode_at = nullptr;  // the arrays the counts callback was handed
    const uint64_t *                        counts_prompt_at = nullptr;
    moe_swap_driver_state                   d;

    rig(int n_blocks, int n_experts, int resident_per_block, const moe_swap_params & p = moe_swap_params{}) :
        nb(n_blocks),
        ne(n_experts),
        d(n_blocks, n_experts, p) {
        const size_t cells = static_cast<size_t>(nb) * static_cast<size_t>(ne);
        view.n_blocks      = nb;
        view.n_experts     = ne;
        view.resident.assign(cells, 0);
        view.admissible.assign(cells, 0);
        view.eligible.assign(static_cast<size_t>(nb), 1);
        view.pair_bytes.assign(static_cast<size_t>(nb), 3);
        view.device_held   = true;
        view.single_device = true;
        for (int b = 0; b < nb; ++b) {
            for (int e = 0; e < ne; ++e) {
                const bool res = e < resident_per_block;

                cell(view.resident, b, e)   = res ? 1 : 0;
                cell(view.admissible, b, e) = res ? 0 : 1;
            }
        }
        decode.assign(cells, 0);
        prompt.assign(cells, 0);
        recount();
    }

    // The view's per-block admissible counts, as moe_swap_plan_view_build leaves them; after a test edits cells.
    void recount() { ggml_sycl::moe_swap_plan_view_count_admissible(view); }

    size_t index(int b, int e) const {
        return static_cast<size_t>(b) * static_cast<size_t>(ne) + static_cast<size_t>(e);
    }

    uint8_t & cell(std::vector<uint8_t> & v, int b, int e) { return v[index(b, e)]; }

    void answer(moe_swap_move_kind    kind,
                int                   block,
                int                   expert,
                moe_swap_move_outcome outcome,
                const char *          reason,
                int                   times = -1,
                bool                  latch = false) {
        rules.push_back({ kind, block, expert, outcome, reason, times, latch });
    }

    void reply(moe_swap_move & m) {
        for (rule & r : rules) {
            if (r.times == 0 || r.kind != m.kind || r.block != m.block || r.expert != m.expert) {
                continue;
            }
            if (r.times > 0) {
                --r.times;
            }
            m.outcome       = r.outcome;
            m.reason        = r.reason;
            m.bytes         = 0;
            m.latches_block = r.latch;
            return;
        }
        m.outcome = MOE_SWAP_MOVE_MOVED;
        m.reason  = "OK";
        m.bytes   = 3;
    }

    // The plan the transaction publishes: a moved evict leaves the card for pinned host storage, a moved admit enters.
    void publish(const std::vector<moe_swap_move> & batch) {
        for (const moe_swap_move & m : batch) {
            if (m.outcome != MOE_SWAP_MOVE_MOVED) {
                continue;
            }
            const bool in = m.kind == MOE_SWAP_MOVE_ADMIT;

            cell(view.resident, m.block, m.expert)   = in ? 1 : 0;
            cell(view.admissible, m.block, m.expert) = in ? 0 : 1;
        }
        recount();
    }

    moe_swap_driver_status boundary(int    n_tokens     = 1,
                                    bool   after_prompt = false,
                                    double gate         = 0.0,
                                    bool   loans        = false,
                                    bool   settled      = true) {
        moe_swap_driver_call c;
        c.after_prompt      = after_prompt;
        c.n_tokens          = n_tokens;
        c.loans_outstanding = loans;
        c.cache_settled     = settled;
        c.prompt_gate       = gate;

        const size_t           before = calls.size();
        moe_swap_driver_status st     = MOE_SWAP_DRIVER_SENT;
        try {
            st = ggml_sycl::moe_swap_driver_step(
                d, view, c,
                [&](uint64_t * dc, uint64_t * pc) {
                    ++counts_reads;
                    counts_decode_at = dc;
                    counts_prompt_at = pc;
                    std::copy(decode.begin(), decode.end(), dc);
                    std::copy(prompt.begin(), prompt.end(), pc);
                },
                [&]() {
                    ++room_reads;
                    return room;
                },
                [&](std::vector<moe_swap_move> & batch) {
                    const size_t answered =
                        throw_after >= 0 ? std::min(batch.size(), static_cast<size_t>(throw_after)) : batch.size();
                    for (size_t i = 0; i < answered; ++i) {
                        reply(batch[i]);
                    }
                    calls.push_back(batch);
                    if (throw_after >= 0) {
                        throw_after = -1;
                        throw std::runtime_error("submit failed");
                    }
                });
        } catch (const std::runtime_error &) {
            ++throws;  // the step rethrew; what the moves answered before it is published below, as the move does
        }
        for (size_t i = before; i < calls.size(); ++i) {
            publish(calls[i]);
        }
        return st;
    }

    // The decode boundaries up to and including the next period boundary of the default period (4).
    moe_swap_driver_status to_period(bool settled = true) {
        moe_swap_driver_status st = MOE_SWAP_DRIVER_IDLE;
        do {
            st = boundary(1, false, 0.0, false, settled);
        } while (d.policy.tokens % 4 != 0);
        return st;
    }

    const std::vector<moe_swap_move> & last() const { return calls.back(); }

    size_t count(moe_swap_move_kind kind) const {
        size_t n = 0;
        for (const std::vector<moe_swap_move> & batch : calls) {
            for (const moe_swap_move & m : batch) {
                n += m.kind == kind ? 1 : 0;
            }
        }
        return n;
    }
};

static uint64_t reasons(const std::vector<ggml_sycl::moe_swap_reason_count> & v, const char * reason) {
    return ggml_sycl::moe_swap_reason_count_of(v, reason);
}

// ---------------------------------------------------------------------------------------------------------------------

// A period boundary with a pair to move sends its evicts and nothing else; each moved evict's admit is pending.
static void test_a_period_boundary_sends_its_evicts_alone() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    CHECK(r.boundary() == MOE_SWAP_DRIVER_IDLE);
    CHECK(r.boundary() == MOE_SWAP_DRIVER_IDLE);
    CHECK(r.boundary() == MOE_SWAP_DRIVER_IDLE);
    CHECK(r.calls.empty());
    CHECK(r.counts_reads == 0);  // no boundary of the policy yet, so the counters were not read
    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.calls.size() == 1);
    if (r.calls.size() == 1) {
        CHECK(r.last().size() == 1);
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_EVICT && r.last()[0].block == 0 && r.last()[0].expert == 0);
    }
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].admit_expert == 5 && r.d.pending[0].evict_expert == 0 && !r.d.pending[0].restore);
    }
    CHECK(r.d.stats.evicts == 1 && r.d.stats.evict_bytes == 3);
    CHECK(r.d.stats.pending_high_water == 1);
}

// The next boundary sends the pending admits and only them, before the policy is consulted, and while any is pending
// the policy is in flight: a period boundary then proposes nothing and absorbs nothing.
static void test_pending_admits_go_first_and_hold_the_policy_in_flight() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();
    CHECK(r.count(MOE_SWAP_MOVE_EVICT) == 1);

    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "WEIGHT zone full", 4);
    r.decode[r.index(0, 6)] = 50;  // a pair the policy would take, were it not in flight
    for (int i = 0; i < 4; ++i) {
        CHECK(r.boundary(1, false, 0.0, false, /*settled=*/false) == MOE_SWAP_DRIVER_SENT);
        CHECK(r.last().size() == 1);
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_ADMIT && r.last()[0].expert == 5);
    }
    CHECK(r.d.policy.tokens == 8);
    CHECK(r.count(MOE_SWAP_MOVE_EVICT) == 1);     // the period boundary at token 8 moved nothing
    CHECK(r.d.policy.usage_of(0, 6) == 0.0);      // and absorbed nothing
    CHECK(reasons(r.d.stats.busy, "WEIGHT zone full") == 4);
    CHECK(r.d.pending.size() == 1);               // unsettled boundaries never drop a pending admit

    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);  // token 9: the admit lands
    CHECK(r.d.pending.empty());
    CHECK(r.d.stats.admits == 1 && r.d.stats.admit_bytes == 3);
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);  // token 12: the policy proposes again
    CHECK(r.last().size() == 1 && r.last()[0].kind == MOE_SWAP_MOVE_EVICT);
    CHECK(r.d.pending.size() == 1 && r.d.pending[0].admit_expert == 6);
}

// A refused evict makes no pending admit. It latches its block only when the transaction marks the refusal as one
// every evict of the block would meet; any other final refusal drops that pair alone, counted, and the block is
// proposed again. A busy evict makes no pending admit either, and the policy proposes the pair again later.
static void test_a_refused_evict_makes_no_pending_admit_and_latches_its_block() {
    {
        rig r(1, 8, 4);
        r.decode[r.index(0, 5)] = 10;
        r.answer(MOE_SWAP_MOVE_EVICT, 0, 0, MOE_SWAP_MOVE_REFUSED, "shared storage", -1, /*latch=*/true);
        CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.d.pending.empty());
        CHECK(r.d.latched[0] != 0);
        CHECK(reasons(r.d.stats.refused, "shared storage") == 1);
        CHECK(r.d.stats.transactions_evict == 0);  // a transaction that moved nothing published nothing
        r.decode[r.index(0, 5)] = 40;
        r.to_period();
        CHECK(r.calls.size() == 1);  // the latched block is not proposed again
    }
    {
        // The same block refused finally for a reason of that triplet alone: no latch, and it is proposed again.
        rig r(1, 8, 4);
        r.decode[r.index(0, 5)] = 10;
        r.answer(MOE_SWAP_MOVE_EVICT, 0, 0, MOE_SWAP_MOVE_REFUSED, "host copy not pinned", 1);
        CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.d.pending.empty());
        CHECK(r.d.latched[0] == 0);
        CHECK(reasons(r.d.stats.refused, "host copy not pinned") == 1);
        r.decode[r.index(0, 5)] = 40;
        CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.calls.size() == 2 && r.last()[0].kind == MOE_SWAP_MOVE_EVICT && r.last()[0].block == 0);
        CHECK(r.d.pending.size() == 1);
    }
    {
        rig r(1, 8, 4);
        r.decode[r.index(0, 5)] = 10;
        r.answer(MOE_SWAP_MOVE_EVICT, 0, 0, MOE_SWAP_MOVE_BUSY, "host pinned pool exhausted", 1);
        CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.d.pending.empty());
        CHECK(r.d.latched[0] == 0);
        CHECK(reasons(r.d.stats.busy, "host pinned pool exhausted") == 1);
        CHECK(r.d.stats.transactions_evict == 0);
        r.decode[r.index(0, 5)] = 40;
        CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.calls.size() == 2 && r.last()[0].kind == MOE_SWAP_MOVE_EVICT && r.last()[0].expert == 0);
        CHECK(r.d.pending.size() == 1);
        CHECK(r.d.stats.transactions_evict == 1 && r.d.stats.transactions_admit == 0);
    }
}

// Residency is the view's at each boundary: a triplet that left the card by other means is no victim.
static void test_residency_comes_from_the_view_at_each_boundary() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();                        // evicts 0
    r.boundary();                         // admits 5
    CHECK(r.d.pending.empty());
    r.cell(r.view.resident, 0, 1)   = 0;  // expert 1 went to the host by another path
    r.cell(r.view.admissible, 0, 1) = 1;
    r.recount();
    r.decode[r.index(0, 6)] = 40;
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.last().size() == 1);
    if (!r.last().empty()) {
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_EVICT && r.last()[0].expert == 2);  // the coldest resident now
    }
}

// A block the view marks not eligible is never proposed.
static void test_nothing_is_proposed_for_an_ineligible_block() {
    rig r(2, 8, 4);
    r.view.eligible[1]      = 0;
    r.decode[r.index(0, 5)] = 10;
    r.decode[r.index(1, 5)] = 90;
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.last().size() == 1);
    if (!r.last().empty()) {
        CHECK(r.last()[0].block == 0);
    }
}

// A pending admit refused finally becomes a restore: the next boundary admits the triplet its pair evicted.
static void test_a_finally_refused_admit_sends_a_restore() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_REFUSED, "host copy not pinned");
    r.to_period();  // evicts 0
    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.last().size() == 1 && r.last()[0].expert == 5);
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].restore && r.d.pending[0].evict_expert == 0);
    }
    CHECK(r.d.latched[0] != 0);
    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.last().size() == 1);
    if (!r.last().empty()) {
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_ADMIT && r.last()[0].expert == 0);
    }
    CHECK(r.d.pending.empty());
    CHECK(r.d.stats.restores == 1);
    CHECK(r.d.stats.open_slots == 0);
    CHECK(r.cell(r.view.resident, 0, 0) == 1);  // the slot is filled again
}

// A restore dropped by the bounded-busy rule leaves the slot open: counted, and never filled by a new evict.
static void test_a_dropped_restore_is_one_open_slot() {
    rig r(2, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "WEIGHT zone full");
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 0, MOE_SWAP_MOVE_BUSY, "WEIGHT zone full");
    r.to_period();     // token 4: evicts 0
    for (int i = 0; i < ggml_sycl::MOE_SWAP_PENDING_DROP_STREAK; ++i) {
        r.boundary();  // tokens 5 to 8: the admit busy at four settled boundaries
    }
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].restore);
    }
    CHECK(reasons(r.d.stats.dropped, "WEIGHT zone full") == 1);
    for (int i = 0; i < ggml_sycl::MOE_SWAP_PENDING_DROP_STREAK; ++i) {
        r.boundary();  // tokens 9 to 12: the restore busy at four more
    }
    CHECK(r.d.pending.empty());
    CHECK(reasons(r.d.stats.dropped, "WEIGHT zone full") == 2);
    CHECK(r.d.stats.open_slots == 1);
    CHECK(r.d.open_slots[0] == 1);

    // Block 0 keeps a hot host expert, and block 1 gets one: only block 1 is proposed.
    r.decode[r.index(0, 6)] = 90;
    r.decode[r.index(1, 6)] = 30;
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.last().size() == 1);
    if (!r.last().empty()) {
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_EVICT && r.last()[0].block == 1);
    }
}

// The drop counts settled boundaries in a row with one reason: an unsettled boundary or another reason starts it over.
static void test_the_drop_streak_counts_settled_boundaries_with_one_reason() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "WEIGHT zone full", 3);
    r.to_period();
    for (int i = 0; i < 3; ++i) {
        r.boundary();  // three settled
    }
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "plan changed", 1);
    r.boundary();                                         // another reason: the streak starts over at 1
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "plan changed", 3);
    r.boundary(1, false, 0.0, false, /*settled=*/false);  // unsettled: starts over
    r.boundary();
    r.boundary();
    CHECK(r.d.pending.size() == 1);  // two settled in a row so far
    if (r.d.pending.size() == 1) {
        CHECK(!r.d.pending[0].restore && r.d.pending[0].busy_streak == 2);
    }
    CHECK(r.d.stats.dropped.empty());
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_BUSY, "plan changed", 2);
    r.boundary();
    r.boundary();  // the fourth settled boundary in a row with one reason
    CHECK(reasons(r.d.stats.dropped, "plan changed") == 1);
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].restore && r.d.pending[0].busy_streak == 0);
    }
}

// While loans are outstanding the driver sends nothing: a pending admit waits, and the policy is held in flight.
static void test_loans_outstanding_hold_every_transaction() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();  // evicts 0
    CHECK(r.boundary(1, false, 0.0, /*loans=*/true) == MOE_SWAP_DRIVER_HELD);
    CHECK(r.calls.size() == 1);
    CHECK(r.d.pending.size() == 1);
    r.boundary();  // admits 5
    r.decode[r.index(0, 6)] = 50;
    for (int i = 0; i < 4; ++i) {
        CHECK(r.boundary(1, false, 0.0, /*loans=*/true) == MOE_SWAP_DRIVER_HELD);
    }
    CHECK(r.calls.size() == 2);
    CHECK(r.d.stats.boundaries_held == 5);
}

// Off on a plan that does not hold the card through anything but experts, or that spans cards: nothing is read and
// nothing is sent, and the summary names why.
static void test_the_driver_is_off_without_a_held_card_or_on_a_multi_device_plan() {
    {
        rig r(1, 8, 4);
        r.decode[r.index(0, 5)] = 10;
        r.view.device_held      = false;
        for (int i = 0; i < 8; ++i) {
            CHECK(r.boundary() == MOE_SWAP_DRIVER_OFF);
        }
        CHECK(r.calls.empty() && r.counts_reads == 0 && r.room_reads == 0);
        CHECK(same(r.d.stats.off_reason, "card held only through experts"));
        CHECK(r.d.stats.boundaries_off == 8);
    }
    {
        rig r(1, 8, 4);
        r.view.single_device = false;
        CHECK(r.boundary() == MOE_SWAP_DRIVER_OFF);
        CHECK(same(r.d.stats.off_reason, "multi-device plan"));
    }
    {
        rig r(1, 8, 4);
        ggml_sycl::moe_swap_driver_note_off(r.d, "multithreaded graphs");
        ggml_sycl::moe_swap_driver_note_skipped(r.d);
        CHECK(same(r.d.stats.off_reason, "multithreaded graphs"));
        CHECK(r.d.stats.boundaries == 2 && r.d.stats.boundaries_off == 1 && r.d.stats.boundaries_skipped == 1);
        CHECK(contains(ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0), "off: multithreaded graphs"));
    }
}

// The gate reaches the policy: with K > 0 the first decode boundary after a prompt pass sends the whole gated batch
// at once, uncapped; with K = 0 it waits for the period.
static void test_the_gate_reaches_the_policy() {
    moe_swap_params p;
    p.max_swaps = 1;
    {
        rig r(1, 16, 8, p);
        r.prompt[r.index(0, 8)]  = 40;
        r.prompt[r.index(0, 9)]  = 30;
        r.prompt[r.index(0, 10)] = 20;
        CHECK(r.boundary(0, /*after_prompt=*/true) == MOE_SWAP_DRIVER_IDLE);
        CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_SENT);
        CHECK(r.last().size() == 3);
        CHECK(r.d.stats.prompt_boundaries == 1 && r.d.stats.prompt_proposals == 3);
        CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_SENT);  // the three admits
        CHECK(r.d.stats.prompt_admits == 3 && r.d.stats.admits == 3);
    }
    {
        rig r(1, 16, 8, p);
        r.prompt[r.index(0, 8)] = 40;
        r.boundary(0, true);
        CHECK(r.boundary(1, false, 0.0) == MOE_SWAP_DRIVER_IDLE);
        CHECK(r.calls.empty());
        CHECK(r.d.stats.prompt_boundaries == 0);
    }
}

// The host room bounds the batch: the hottest pairs that fit go, the rest are deferred and counted apart from
// refusals. The room is read once, and only at a boundary with a batch.
static void test_a_batch_is_trimmed_to_the_host_room_hottest_first() {
    rig r(1, 16, 8);
    r.decode[r.index(0, 8)]  = 40;
    r.decode[r.index(0, 9)]  = 30;
    r.decode[r.index(0, 10)] = 20;
    r.room                   = 7;  // two pairs of 3 bytes
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.room_reads == 1);
    CHECK(r.last().size() == 2);
    CHECK(r.d.pending.size() == 2);
    if (r.d.pending.size() == 2) {
        CHECK(r.d.pending[0].admit_expert == 8 && r.d.pending[1].admit_expert == 9);
    }
    CHECK(r.d.stats.deferred_room == 1);
    CHECK(r.d.stats.last_room == 7);
    CHECK(r.d.stats.refused.empty() && r.d.stats.busy.empty());
    r.boundary();                                  // the admits
    CHECK(r.room_reads == 1);                      // a boundary that sends admits reads no room
    r.room = 100;
    CHECK(r.to_period() == MOE_SWAP_DRIVER_SENT);  // the deferred pair is proposed again
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].admit_expert == 10);
    }
}

// A trimmed prompt batch keeps the prompt boundary: once the admits land, the next boundary takes the prompt rule
// again for the rest, with no period wait.
static void test_a_trimmed_prompt_batch_keeps_the_prompt_boundary() {
    rig r(1, 16, 8);
    r.prompt[r.index(0, 8)] = 40;
    r.prompt[r.index(0, 9)] = 30;
    r.room                  = 3;
    r.boundary(0, true);
    CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_SENT);  // token 1
    CHECK(r.last().size() == 1);
    CHECK(r.d.stats.deferred_room == 1);
    CHECK(r.d.policy.prompt_pending);
    CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_SENT);  // token 2: the admit of 8, the policy in flight
    CHECK(r.last().size() == 1 && r.last()[0].kind == MOE_SWAP_MOVE_ADMIT && r.last()[0].expert == 8);
    CHECK(r.d.policy.prompt_pending);
    CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_SENT);  // token 3: the prompt rule again
    CHECK(r.last().size() == 1 && r.last()[0].kind == MOE_SWAP_MOVE_EVICT);
    CHECK(r.d.pending.size() == 1);
    if (r.d.pending.size() == 1) {
        CHECK(r.d.pending[0].admit_expert == 9 && r.d.pending[0].prompt);
    }
    CHECK(r.d.stats.prompt_boundaries == 2);
}

// No room: nothing is sent and every pair is deferred.
static void test_zero_room_sends_nothing() {
    rig r(1, 16, 8);
    r.decode[r.index(0, 8)] = 40;
    r.decode[r.index(0, 9)] = 30;
    r.room                  = 0;
    CHECK(r.to_period() == MOE_SWAP_DRIVER_IDLE);
    CHECK(r.calls.empty());
    CHECK(r.d.pending.empty());
    CHECK(r.d.stats.deferred_room == 2);
    CHECK(r.d.stats.last_room == 0);
}

// Host experts that are file-mapped have no admissible host side: the driver proposes nothing, evicts nothing, and
// says so with the count it passed over.
static void test_no_admissible_host_expert_proposes_nothing_and_says_so() {
    rig r(1, 8, 4);
    std::fill(r.view.admissible.begin(), r.view.admissible.end(), static_cast<uint8_t>(0));
    r.recount();
    r.view.file_mapped      = 4;
    r.decode[r.index(0, 5)] = 90;
    r.to_period();
    CHECK(r.calls.empty());
    CHECK(r.d.stats.boundaries_no_admissible == 4);
    CHECK(r.d.stats.file_mapped_passed == 4);
    const std::string faults = ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0);
    CHECK(contains(faults, "no admissible host expert at 4 boundaries (4 file-mapped triplets passed over)"));
}

// The decode graphs that ran after a boundary that moved something are counted, so their tokens can be timed, and so
// are the accepted transactions, one per direction of the round: each one can make a context drop what it recorded.
static void test_decode_graphs_after_a_move_are_counted() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();  // moves an evict
    r.boundary(3);  // 3 graphs after it; moves the admit
    r.boundary(5);  // 5 graphs after that; moves nothing
    r.boundary(2);  // 2 graphs after a boundary that moved nothing
    CHECK(r.d.stats.graphs_after_move == 8);
    CHECK(r.d.stats.decode_graphs == 14);
    CHECK(r.d.stats.transactions_evict == 1 && r.d.stats.transactions_admit == 1);
    const std::string faults = ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0);
    CHECK(contains(faults, "accepted transactions 2 (evict 1, admit 1) over 14 decode graphs"));
}

// The two summary lines carry every figure: moves, prompt boundaries, K, open slots, deferrals, file-mapped triplets,
// the off reason and the refusal, busy and drop reasons.
static void test_the_summary_lines_name_every_figure() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 5, MOE_SWAP_MOVE_REFUSED, "host copy not pinned");
    r.answer(MOE_SWAP_MOVE_ADMIT, 0, 0, MOE_SWAP_MOVE_BUSY, "WEIGHT zone full");
    r.to_period();
    r.boundary();      // the admit refused finally: a restore
    for (int i = 0; i < ggml_sycl::MOE_SWAP_PENDING_DROP_STREAK; ++i) {
        r.boundary();  // the restore dropped: an open slot
    }
    const std::string moves = ggml_sycl::moe_swap_driver_summary_moves(r.d.stats, 1, "off (h2d unprobed, decode rule)");
    const std::string faults = ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 1);
    CHECK(contains(moves, "[MOE-SWAP] device 1: "));
    CHECK(contains(moves, "boundaries"));
    CHECK(contains(moves, "decode graphs"));
    CHECK(contains(moves, "after a move"));
    CHECK(contains(moves, "1 evicts"));
    CHECK(contains(moves, "0 admits"));
    CHECK(contains(moves, "0 restores"));
    CHECK(contains(moves, "pending high-water 1"));
    CHECK(contains(moves, "prompt boundaries 0"));
    CHECK(contains(moves, "K off (h2d unprobed, decode rule)"));
    CHECK(contains(faults, "[MOE-SWAP] device 1: "));
    // The evict moved; the admit refused finally and the busy restore moved nothing, so neither published.
    CHECK(contains(faults, "accepted transactions 1 (evict 1, admit 0) over 9 decode graphs"));
    CHECK(contains(faults, "open slots 1"));
    CHECK(contains(faults, "deferred for host room 0"));
    CHECK(contains(faults, "off: none"));
    CHECK(contains(faults, "refused: host copy not pinned x1"));
    CHECK(contains(faults, "busy: WEIGHT zone full x4"));
    CHECK(contains(faults, "dropped pending: WEIGHT zone full x1"));
    CHECK(contains(faults, "settled from the plan after a throw: 0 batches (0 moves moved)"));
    CHECK(!contains(faults, "no single-row decode graph"));
}

// A boundary that reaches no boundary of the policy and has nothing pending allocates nothing.
static void test_a_boundary_with_nothing_to_do_does_not_allocate() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();
    r.boundary();  // the admit
    const size_t before = g_allocs;
    r.boundary();
    r.boundary();
    CHECK(g_allocs == before);
}

// K is 2 x the CPU rate over the host-to-device rate, and 0, the decode rule, when either rate is not a positive
// number: an unprobed device (0), a rate below zero, or NaN.
static void test_the_prompt_gate_is_zero_unless_both_rates_are_positive() {
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(10.0, 5.0) == 4.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(3.0, 12.0) == 0.5);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(10.0, 0.0) == 0.0);  // the device's h2d rate is not probed
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(0.0, 5.0) == 0.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(0.0, 0.0) == 0.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(-1.0, 5.0) == 0.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(10.0, -1.0) == 0.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(std::nan(""), 5.0) == 0.0);
    CHECK(ggml_sycl::moe_swap_prompt_gate_of(10.0, std::nan("")) == 0.0);
}

// The route counters are copied into the driver's own snapshot arrays, the ones the policy's block inputs point at.
static void test_the_counts_go_into_the_drivers_own_arrays() {
    rig r(1, 8, 4);
    r.decode[r.index(0, 5)] = 10;
    r.to_period();
    CHECK(r.counts_reads == 1);
    CHECK(r.counts_decode_at == r.d.decode.data());
    CHECK(r.counts_prompt_at == r.d.prompt.data());
    CHECK(r.d.decode[r.index(0, 5)] == 10);
}

// An evict batch whose transaction threw after one move is held unsettled, and the step rethrows. The next boundary
// settles it from its view before anything else: the evict the plan shows moved leaves its admit pending, the two it
// shows resident are dropped and counted, and the pending admit goes in the same boundary.
static void test_an_evict_batch_that_threw_is_settled_from_the_next_view() {
    rig r(1, 16, 8);
    r.decode[r.index(0, 8)]  = 40;
    r.decode[r.index(0, 9)]  = 30;
    r.decode[r.index(0, 10)] = 20;
    r.throw_after            = 1;
    r.to_period();
    CHECK(r.throws == 1);
    CHECK(r.calls.size() == 1 && r.last().size() == 3);
    CHECK(r.d.unsettled.size() == 3);
    CHECK(r.d.pending.empty());
    CHECK(r.d.stats.evicts == 0 && r.d.stats.transactions_evict == 0);
    if (r.calls.size() != 1 || r.last().size() != 3) {
        return;
    }
    const int moved_evict = r.last()[0].expert;
    CHECK(r.last()[0].outcome == MOE_SWAP_MOVE_MOVED && r.last()[1].outcome == ggml_sycl::MOE_SWAP_MOVE_UNSENT);
    CHECK(r.cell(r.view.resident, 0, moved_evict) == 0);  // the transaction published what moved before it threw

    // Token 5.
    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);
    CHECK(r.d.unsettled.empty());
    CHECK(r.d.stats.settled_batches == 1 && r.d.stats.settled_moved == 1);
    CHECK(r.d.stats.evicts == 1 && r.d.stats.transactions_evict == 1);
    CHECK(reasons(r.d.stats.busy, ggml_sycl::MOE_SWAP_UNSETTLED_REASON) == 2);
    CHECK(r.d.stats.graphs_after_move == 1);  // the settled move counts as this boundary's
    CHECK(r.calls.size() == 2 && r.last().size() == 1);
    if (r.calls.size() == 2 && r.last().size() == 1) {
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_ADMIT && r.last()[0].expert == 8);
    }
    CHECK(r.d.pending.empty() && r.d.stats.admits == 1);
    const std::string faults = ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0);
    CHECK(contains(faults, "settled from the plan after a throw: 1 batches (1 moves moved)"));
    CHECK(contains(faults, "busy: transaction threw before it answered x2"));
}

// An admit batch whose transaction threw after one move: the next boundary counts the admit the plan shows resident
// as moved, and sends the one it does not show again, under the pending rule.
static void test_an_admit_batch_that_threw_is_settled_from_the_next_view() {
    rig r(1, 16, 8);
    r.decode[r.index(0, 8)] = 40;
    r.decode[r.index(0, 9)] = 30;
    r.to_period();  // token 4: two evicts
    CHECK(r.d.pending.size() == 2);
    r.throw_after = 1;
    r.boundary();  // token 5: the two admits, and the throw after the first
    CHECK(r.throws == 1);
    CHECK(r.d.pending.empty() && r.d.unsettled.size() == 2);
    CHECK(r.d.stats.admits == 0 && r.d.stats.transactions_admit == 0);
    CHECK(r.cell(r.view.resident, 0, 8) == 1 && r.cell(r.view.resident, 0, 9) == 0);

    CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);  // token 6: settles, then sends the admit of 9
    CHECK(r.d.stats.settled_batches == 1 && r.d.stats.settled_moved == 1);
    CHECK(r.calls.size() == 3 && r.last().size() == 1);
    if (r.calls.size() == 3 && r.last().size() == 1) {
        CHECK(r.last()[0].kind == MOE_SWAP_MOVE_ADMIT && r.last()[0].expert == 9);
    }
    CHECK(r.d.unsettled.empty() && r.d.pending.empty());
    CHECK(r.d.stats.admits == 2 && r.d.stats.transactions_admit == 2);
    CHECK(r.d.stats.busy.empty());
}

// A replaced route table under the same load and shape takes what the old driver has in flight, and the new driver
// sends it; another shape takes nothing. What cannot be carried is counted as open slots in the old driver's figures.
static void test_a_replaced_table_carries_or_drops_what_is_in_flight() {
    {
        rig r(1, 16, 8);
        r.decode[r.index(0, 8)] = 40;
        r.decode[r.index(0, 9)] = 30;
        r.to_period();
        CHECK(r.d.pending.size() == 2);
        moe_swap_driver_state fresh(1, 16);
        moe_swap_driver_state other(2, 16);
        CHECK(!ggml_sycl::moe_swap_driver_carry(r.d, other));
        CHECK(r.d.pending.size() == 2 && other.pending.empty());
        CHECK(ggml_sycl::moe_swap_driver_carry(r.d, fresh));
        CHECK(r.d.pending.empty() && fresh.pending.size() == 2);
        CHECK(fresh.stats.pending_high_water == 2);
        r.d = fresh;
        CHECK(r.boundary() == MOE_SWAP_DRIVER_SENT);
        CHECK(r.last().size() == 2 && r.last()[0].kind == MOE_SWAP_MOVE_ADMIT);
        CHECK(r.d.pending.empty() && r.d.stats.admits == 2);
    }
    {
        moe_swap_driver_state       d(2, 8);
        ggml_sycl::moe_swap_pending p;
        p.block = 0;
        d.pending.push_back(p);
        ggml_sycl::moe_swap_unsettled u;
        u.kind       = MOE_SWAP_MOVE_EVICT;
        u.pair.block = 1;
        d.unsettled.push_back(u);
        CHECK(ggml_sycl::moe_swap_driver_drop_in_flight(d) == 2);
        CHECK(d.pending.empty() && d.unsettled.empty());
        CHECK(d.open_slots[0] == 1 && d.open_slots[1] == 1 && d.stats.open_slots == 2);
        CHECK(contains(ggml_sycl::moe_swap_driver_summary_faults(d.stats, 0), "open slots 2"));
    }
}

// A prompt batch with room for no pair sends nothing and spends the prompt boundary: the next decode boundary follows
// the decode rule, and no second prompt boundary is counted.
static void test_a_prompt_batch_with_room_for_no_pair_spends_the_prompt_boundary() {
    rig r(1, 16, 8);
    r.prompt[r.index(0, 8)] = 40;
    r.prompt[r.index(0, 9)] = 30;
    // Below one pair of 3 bytes.
    r.room                  = 2;
    r.boundary(0, true);
    CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_IDLE);  // token 1
    CHECK(r.calls.empty());
    CHECK(r.d.stats.prompt_boundaries == 1 && r.d.stats.deferred_room == 2);
    CHECK(!r.d.policy.prompt_pending);
    CHECK(r.boundary(1, false, 5.0) == MOE_SWAP_DRIVER_IDLE);  // token 2: the decode rule, before its period
    CHECK(r.d.stats.prompt_boundaries == 1);
    CHECK(r.room_reads == 1);
}

// The view counts each block's admissible cells once, and a boundary reads that count, not the cells.
static void test_the_boundary_reads_each_blocks_admissible_count() {
    moe_swap_plan_view v;
    v.n_blocks   = 2;
    v.n_experts  = 3;
    v.admissible = { 1, 0, 1, 0, 0, 0 };
    ggml_sycl::moe_swap_plan_view_count_admissible(v);
    CHECK(v.admissible_count.size() == 2);
    if (v.admissible_count.size() == 2) {
        CHECK(v.admissible_count[0] == 2 && v.admissible_count[1] == 0);
    }
    v.admissible.pop_back();  // cells that do not match the shape count as none
    ggml_sycl::moe_swap_plan_view_count_admissible(v);
    CHECK(v.admissible_count.size() == 2 && v.admissible_count[0] == 0);

    {
        rig r(1, 8, 4);
        r.view.admissible_count[0] = 0;  // the cells still say four admissible: the count is what is read
        r.decode[r.index(0, 5)]    = 10;
        r.to_period();
        CHECK(r.d.stats.boundaries_no_admissible == 4);
    }
    {
        rig r(1, 8, 4);
        r.view.admissible_count.clear();
        CHECK(r.boundary() == MOE_SWAP_DRIVER_OFF);
        CHECK(same(r.d.stats.off_reason, "plan view does not match the route table"));
    }
}

// Boundaries that each followed a prompt graph and saw no single-row decode graph, as parallel sequences or
// speculative decoding produce, are named in the summary: the policy was seeded at each and never swapped.
static void test_the_summary_names_a_run_with_no_single_row_decode_graph() {
    rig r(1, 8, 4);
    for (int i = 0; i < 3; ++i) {
        r.boundary(0, /*after_prompt=*/true);
    }
    CHECK(r.d.stats.boundaries_after_prompt == 3 && r.d.stats.decode_graphs == 0);
    const char * note = "no single-row decode graph at 3 boundaries: every one read as a prompt, so nothing swapped";
    CHECK(contains(ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0), note));
    r.boundary(1);
    CHECK(!contains(ggml_sycl::moe_swap_driver_summary_faults(r.d.stats, 0), "no single-row decode graph"));
}

int main() {
    test_a_period_boundary_sends_its_evicts_alone();
    test_pending_admits_go_first_and_hold_the_policy_in_flight();
    test_a_refused_evict_makes_no_pending_admit_and_latches_its_block();
    test_residency_comes_from_the_view_at_each_boundary();
    test_nothing_is_proposed_for_an_ineligible_block();
    test_a_finally_refused_admit_sends_a_restore();
    test_a_dropped_restore_is_one_open_slot();
    test_the_drop_streak_counts_settled_boundaries_with_one_reason();
    test_loans_outstanding_hold_every_transaction();
    test_the_driver_is_off_without_a_held_card_or_on_a_multi_device_plan();
    test_the_gate_reaches_the_policy();
    test_a_batch_is_trimmed_to_the_host_room_hottest_first();
    test_a_trimmed_prompt_batch_keeps_the_prompt_boundary();
    test_zero_room_sends_nothing();
    test_no_admissible_host_expert_proposes_nothing_and_says_so();
    test_decode_graphs_after_a_move_are_counted();
    test_the_summary_lines_name_every_figure();
    test_a_boundary_with_nothing_to_do_does_not_allocate();
    test_the_prompt_gate_is_zero_unless_both_rates_are_positive();
    test_the_counts_go_into_the_drivers_own_arrays();
    test_an_evict_batch_that_threw_is_settled_from_the_next_view();
    test_an_admit_batch_that_threw_is_settled_from_the_next_view();
    test_a_replaced_table_carries_or_drops_what_is_in_flight();
    test_a_prompt_batch_with_room_for_no_pair_spends_the_prompt_boundary();
    test_the_boundary_reads_each_blocks_admissible_count();
    test_the_summary_names_a_run_with_no_single_row_decode_graph();

    if (g_failed != 0) {
        std::printf("test-sycl-moe-swap-driver: %d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test-sycl-moe-swap-driver: all checks passed\n");
    return 0;
}
