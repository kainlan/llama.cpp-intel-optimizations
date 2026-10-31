// llama.cpp-36j0 Task 7a: the layer stream list is a pure function of the plan, the regime and the ids.
// Host-only: a hand-built 4-layer, 8-expert plan with 3 host experts per layer (1, 5, 6), no queue, no device.
// A test-only reference list is also compared with the production moe_stream_plan_index_build at the end of main.

#include "ggml-sycl/moe-stream-plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::moe_exec_regime;
using ggml_sycl::moe_stream_item;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr int    kLayers  = 4;
constexpr int    kExperts = 8;
constexpr size_t kSrc     = 4096;
constexpr size_t kDst     = 4352;

bool is_host(int layer, int expert) {
    return layer != 3 && (expert == 1 || expert == 5 || expert == 6);  // layer 3 has none
}

placement_plan make_plan() {
    placement_plan           plan{};
    const expert_tensor_role roles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
    const char *             names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            for (int r = 0; r < 3; ++r) {
                const bool host = is_host(l, e);
                plan.entries.push_back({ "blk." + std::to_string(l) + "." + names[r] + ".weight", kSrc, kDst, 0,
                                         placement_priority::MOE_UP, l, e, roles[r], !host, host ? -1 : 0 });
            }
        }
    }
    // Entries that must never reach a list: a dense host tensor (expert_id -1) and a fused host GATE_UP for expert 1.
    plan.entries.push_back({ "blk.1.attn_q.weight", kSrc, kDst, 0, placement_priority::ATTENTION, 1, -1,
                             expert_tensor_role::UNKNOWN, false, -1 });
    plan.entries.push_back({ "blk.1.ffn_gate_up_exps.weight", kSrc, kDst, 0, placement_priority::MOE_UP, 1, 1,
                             expert_tensor_role::GATE_UP, false, -1 });
    plan.build_index();
    return plan;
}

// Test-only reference list (llama.cpp-8xdz M4): the runtime builds the same list inside moe_stream_plan_index_build,
// which also carries each entry's type, shape and staging. This one is the bare form Task 7a specified, kept here so
// the ordering and the host-only predicate stay testable on a hand-built plan with no device and no kernels.
//
// A gate/up/down entry of `layer` that is planned for the host. Fused GATE_UP and CHUNK_* host entries are deliberately
// NOT listed: the stream list is gate/up/down triplets, the only shape the ring stages. Host entries carry
// target_device -1 (the planner sets it at placement time), so they serve any device.
bool host_triplet_entry(const ggml_sycl::placement_entry & e, int layer) {
    if (e.expert_id < 0 || e.layer_id != layer || e.on_device) {
        return false;
    }
    return ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role);
}

bool id_listed(const int32_t * ids, size_t n_ids, int expert) {
    for (size_t i = 0; i < n_ids; ++i) {
        if (ids[i] == expert) {
            return true;
        }
    }
    return false;
}

// STREAM_ALL: every host-planned gate/up/down entry of `layer`, in plan.entries order, ids ignored.
// ROUTED_ONLY: the same entries restricted to the experts named in ids, still in plan.entries order.
// Any other regime lists nothing.
std::vector<moe_stream_item> plan_build(const placement_plan & plan,
                                        int                    layer,
                                        moe_exec_regime        regime,
                                        const int32_t *        ids,
                                        size_t                 n_ids) {
    std::vector<moe_stream_item> out;
    if (regime != moe_exec_regime::STREAM_ALL && regime != moe_exec_regime::ROUTED_ONLY) {
        return out;
    }
    for (const ggml_sycl::placement_entry & e : plan.entries) {
        if (!host_triplet_entry(e, layer)) {
            continue;
        }
        if (regime == moe_exec_regime::ROUTED_ONLY && !id_listed(ids, n_ids, e.expert_id)) {
            continue;
        }
        moe_stream_item item{};
        item.layer     = e.layer_id;
        item.expert_id = e.expert_id;
        item.role      = e.expert_role;
        item.src_bytes = e.src_size;
        out.push_back(item);
    }
    return out;
}

// Only gate/up/down triplets are considered (see host_triplet_entry): a layer whose host experts are fused GATE_UP or
// CHUNK_* entries has an empty want list and so answers true without anything staged.
// True iff every distinct id in ids that is host-planned in `layer` has gate, up and down in `staged`.
bool layer_fully_staged(const std::vector<moe_stream_item> & staged,
                        const placement_plan &               plan,
                        int                                  layer,
                        const int32_t *                      ids,
                        size_t                               n_ids) {
    const std::vector<moe_stream_item> want = plan_build(plan, layer, moe_exec_regime::ROUTED_ONLY, ids, n_ids);
    for (const moe_stream_item & w : want) {
        bool found = false;
        for (const moe_stream_item & s : staged) {
            if (s.layer == layer && s.expert_id == w.expert_id && s.role == w.role) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

// The make_plan() layout with each gate/up/down entry typed as one F16 matrix of kSrc bytes, so a claim can stage it
// (moe_stream_plan_index_build gives a matrix a slot only once its rows are whole blocks of src_bytes). F16 makes one
// expert's f16 bytes, ncols x nrows x 2 (one matrix), equal to its src bytes, so the window arithmetic below reads in
// kSrc units.
constexpr int64_t kTypedCols = 64;
constexpr int64_t kTypedRows = static_cast<int64_t>(kSrc) / (2 * kTypedCols);

// down_transposed gives the down entries the transposed shape, kTypedRows x kTypedCols, as a model's down matrix is
// [n_ff, n_embd] against gate/up's [n_embd, n_ff]: the same bytes, each dimension's maximum over the layer larger.
placement_plan make_typed_plan(bool down_transposed = false) {
    placement_plan plan = make_plan();
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.expert_id >= 0 && ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role)) {
            const bool t = down_transposed && e.expert_role == expert_tensor_role::DOWN;
            e.type       = GGML_TYPE_F16;
            e.ne[0]      = t ? kTypedRows : kTypedCols;
            e.ne[1]      = t ? kTypedCols : kTypedRows;
        }
    }
    return plan;
}

// The batched executor claims every weight: each host matrix stages AOS-only, one slot.
bool claims_all(void *, int, expert_tensor_role) {
    return true;
}

// The ring is sized to one executor window, min(role_items_max, G_plan), not to a whole layer's host matrices.
void test_window_sizing() {
    const placement_plan               plan = make_typed_plan();
    // fit = slot / (ncols x nrows x 2, one matrix), G_plan = fit >= 2 ? fit / 2 : fit (the executor's two-buffer
    // rule), capped by group_cap.
    ggml_sycl::moe_stream_window_hints hints;
    hints.weight_slot_bytes = 10 * kSrc;  // fit = 10 -> G_plan = 5
    hints.group_cap         = 0;
    auto idx                = ggml_sycl::moe_stream_plan_index_build(plan, /*device*/ 0, nullptr, claims_all, hints);
    CHECK(idx.host_matrices_per_layer_max == 9);  // a layer has 9 host matrices, one role of it 3
    CHECK(idx.role_items_max == 3);
    CHECK(idx.window_slots == 3);                 // min(3 host items per role, G_plan 5)
    hints.weight_slot_bytes = 4 * kSrc;           // fit = 4 -> G_plan = 2
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 2);
    hints.weight_slot_bytes = 1 * kSrc;  // fit = 1 -> G_plan = 1 (one buffer)
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 1);
    hints.weight_slot_bytes = kSrc - 1;  // fit = 0: the slot holds no expert
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 0);
    hints.weight_slot_bytes = 10 * kSrc;
    hints.group_cap         = 2;  // GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX caps G_plan
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 2);
    hints.group_cap         = 0;
    hints.weight_slot_bytes = 0;  // no planned slot (before the load publishes the inventory): no window, no ring
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 0);
    // Nothing staged (no claim, no prompt kernel): no streamable item, so no window whatever the slot holds.
    hints.weight_slot_bytes = 10 * kSrc;
    idx                     = ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, nullptr, hints);
    CHECK(idx.role_items_max == 0);
    CHECK(idx.window_slots == 0);
    // The geometry is sized from the window: 2 regions x 3 slots x 4096 (already 256-aligned) = 24576 bytes, against
    // 2 x 9 slots for the whole-layer ring.
    ggml_sycl::moe_stream_plan_index gidx;
    const auto geo = ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, hints, &gidx);
    CHECK(geo.region_slots == 3);
    CHECK(geo.n_slots == 6);
    CHECK(geo.bytes == 6 * kSrc);
    CHECK(gidx.window_slots == 3);
    // The per-layer matrix maximum the cost model now reads as its stage unit; layer 3 has no host expert.
    for (int l = 0; l < 3; ++l) {
        CHECK(gidx.layers.count(l) == 1 && gidx.layers.at(l).matrix_bytes_max == kSrc);
    }
    CHECK(gidx.layers.count(3) == 0);
    // The executor's unit is one matrix, ne0 x ne1 x 2 (moe_pp_batched_f16_plan_groups): a down matrix transposed
    // against gate/up is the same 4096 bytes, so fit stays 10 and the window 3. The product of the layer's
    // per-dimension maxima, 64 x 64 x 2, would make fit 5, G_plan 2 and the window 2.
    const placement_plan transposed = make_typed_plan(/*down_transposed*/ true);
    idx                             = ggml_sycl::moe_stream_plan_index_build(transposed, 0, nullptr, claims_all, hints);
    CHECK(idx.role_items_max == 3);
    CHECK(idx.window_slots == 3);
}

// The ring cap is loan-invariant (llama.cpp-7999, first half of llama.cpp-8p5x): a loan flips device triplets to host,
// so the index's resident bytes fall by the lent bytes L, and the caller hands L back through hints.lent_bytes. The
// geometry sized before the loan is then the geometry sized after it, even where the cap is tight.
void test_ring_cap_is_loan_invariant() {
    placement_plan                     plan = make_typed_plan();
    // A window of G_plan = 2 (fit = 4): every layer with host experts has 3 host items per role, so window_slots is
    // G_plan-capped at 2 and lending one more triplet into layer 0 (4 items per role) does not grow it. A loan that
    // grows window_slots is the second half of llama.cpp-8p5x and out of scope here.
    ggml_sycl::moe_stream_window_hints hints;
    hints.weight_slot_bytes     = 4 * kSrc;
    // The ring: 2 regions x 2 slots x 4096 = 16384 bytes. The cap is tight at R = 11 x 16384 = 180224 resident
    // bytes: the lendable triplet (layer 0, expert 0) carries 3 x kDst of them and one layer-3 entry the rest.
    constexpr size_t ring_bytes = 4 * kSrc;
    constexpr size_t resident   = 11 * ring_bytes;
    constexpr size_t lent       = 3 * kDst;
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.expert_id < 0 || !e.on_device) {
            continue;
        }
        const bool lendable = e.layer_id == 0 && e.expert_id == 0;
        const bool anchor   = e.layer_id == 3 && e.expert_id == 0 && e.expert_role == expert_tensor_role::GATE;
        e.dst_size          = lendable ? kDst : anchor ? resident - lent : 0;
    }
    ggml_sycl::moe_stream_plan_index idx;
    const auto before = ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, hints, &idx);
    CHECK(idx.resident_expert_bytes == resident);
    CHECK(idx.window_slots == 2);
    CHECK(before.region_slots == 2 && before.n_slots == 4 && before.bytes == ring_bytes);

    // Tight: one resident byte fewer refuses the ring.
    placement_plan short_plan = plan;
    for (ggml_sycl::placement_entry & e : short_plan.entries) {
        if (e.layer_id == 3 && e.expert_id == 0 && e.expert_role == expert_tensor_role::GATE) {
            e.dst_size -= 1;
        }
    }
    CHECK(ggml_sycl::moe_stream_plan_ring_geometry(short_plan, 0, nullptr, claims_all, hints).bytes == 0);

    // A sentinel lent_bytes saturates rather than wraps: R + SIZE_MAX would wrap to R - 1 = 180223, and
    // 180223 / 11 = 16383 refuses the ring; clamped to SIZE_MAX the cap admits it.
    ggml_sycl::moe_stream_window_hints sentinel_hints = hints;
    sentinel_hints.lent_bytes                         = SIZE_MAX;
    CHECK(ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, sentinel_hints).bytes == ring_bytes);

    // The loan: the lendable triplet moves to the host (target_device -1, as the transaction's
    // update_expert_placement sets it) and its device bytes come back through hints.lent_bytes.
    placement_plan lent_plan = plan;
    for (ggml_sycl::placement_entry & e : lent_plan.entries) {
        if (e.layer_id == 0 && e.expert_id == 0) {
            e.on_device     = false;
            e.target_device = -1;
        }
    }
    ggml_sycl::moe_stream_window_hints lent_hints = hints;
    lent_hints.lent_bytes                         = lent;
    ggml_sycl::moe_stream_plan_index lidx;
    const auto after = ggml_sycl::moe_stream_plan_ring_geometry(lent_plan, 0, nullptr, claims_all, lent_hints, &lidx);
    CHECK(lidx.resident_expert_bytes == resident - lent);
    CHECK(lidx.role_items_max == 4);
    CHECK(lidx.window_slots == 2);
    CHECK(after.slot_bytes == before.slot_bytes);
    CHECK(after.region_slots == before.region_slots);
    CHECK(after.n_slots == before.n_slots);
    CHECK(after.bytes == before.bytes);
}

// The loan request's stub hooks: what the phase's short reserve would post to the backend, and the backend's REFUSED
// count, which no case here moves.
struct loan_requests {
    int    n          = 0;
    size_t bytes      = 0;
    size_t slot_bytes = 0;
};

loan_requests g_requests;

void stub_request_loan(int, size_t bytes, size_t slot_bytes) {
    ++g_requests.n;
    g_requests.bytes      = bytes;
    g_requests.slot_bytes = slot_bytes;
}

uint64_t stub_loan_refusals(int) {
    return 0;
}

// The make_typed_plan() layout for the loan sizing. G_plan is 5 (a planned slot of 10 matrices), and
// each host layer has 3 host matrices per role, so the window is 3 and the ring 2 x 3 slots of kSrc = 24576 bytes.
// Every resident matrix charges kDst = 4352 bytes, one slot. The loan walk meets layer 3 first, which is whole on the
// device (bulk-staged, not lendable), then layer 2's resident experts 7, 4, 3, 2, 0: every triplet it lends lands in
// layer 2 and adds one host matrix to each of its roles, widening the window up to G_plan. `anchor_extra` adds device
// bytes to a layer-3 entry, so the cap admits the widest ring (2 x 5 slots = 40960 bytes, 11 x that resident) or, at 0,
// only the first one (300288 resident bytes: 24576 fits under an eleventh, 32768 does not).
constexpr size_t kAnchorExtra = 200000;

placement_plan make_loan_plan(size_t anchor_extra) {
    placement_plan plan = make_typed_plan();
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.layer_id == 3 && e.expert_id == 0 && e.expert_role == expert_tensor_role::GATE) {
            e.dst_size += anchor_extra;
        }
    }
    return plan;
}

ggml_sycl::moe_stream_window_hints loan_hints() {
    ggml_sycl::moe_stream_window_hints hints;
    hints.weight_slot_bytes = 10 * kSrc;  // fit = 10 -> G_plan = 5
    return hints;
}

ggml_sycl::moe_stream_loan_sizer loan_sizer(const placement_plan & plan) {
    ggml_sycl::moe_stream_loan_sizer sizer;
    sizer.plan           = &plan;
    sizer.device         = 0;
    sizer.library_claims = claims_all;
    sizer.hints          = loan_hints();
    sizer.geo            = ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, sizer.hints);
    return sizer;
}

// The geometry of `plan` with the first `n_triplets` triplets of layer 2's loan walk flipped to the host, as the
// transaction's update_expert_placement flips them, and their device bytes counted as lent: the ring prepare() would
// rebuild after that loan.
ggml_sycl::moe_stream_ring_geometry geometry_after_loan(
    const placement_plan &                     plan,
    int                                        n_triplets,
    const ggml_sycl::moe_stream_window_hints & base = loan_hints()) {
    const int      walk[5]   = { 7, 4, 3, 2, 0 };
    placement_plan lent_plan = plan;
    size_t         lent      = 0;
    for (ggml_sycl::placement_entry & e : lent_plan.entries) {
        for (int t = 0; t < n_triplets; ++t) {
            if (e.layer_id == 2 && e.expert_id == walk[t] &&
                ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role)) {
                e.on_device     = false;
                e.target_device = -1;
                lent += e.dst_size;
            }
        }
    }
    ggml_sycl::moe_stream_window_hints hints = base;
    hints.lent_bytes                         = lent;
    return ggml_sycl::moe_stream_plan_ring_geometry(lent_plan, 0, nullptr, claims_all, hints);
}

// The loan is sized from the plan it leaves: lending widens the window, so the
// request sized for the reserve's geometry would leave the next micro-batch's ring short again. The request iterates
// to the geometry its own loan leaves.
void test_loan_converges_on_the_post_loan_ring() {
    const placement_plan                   plan  = make_loan_plan(kAnchorExtra);
    const ggml_sycl::moe_stream_loan_sizer sizer = loan_sizer(plan);
    CHECK(sizer.geo.region_slots == 3 && sizer.geo.n_slots == 6 && sizer.geo.slot_bytes == kSrc);

    // No slot held: the reserve is the whole ring short. Its 6 slots take layer 2's experts 7 and 4 (3 slots each),
    // which give layer 2 five host matrices per role: the window becomes G_plan, and the ring 10 slots. The first
    // request no longer funds the ring it leaves.
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, 6 * kSrc, kSrc, nullptr).size() == 6);
    const auto first_post = geometry_after_loan(plan, 2);
    CHECK(first_post.region_slots == 5 && first_post.n_slots == 10 && first_post.bytes == 10 * kSrc);

    // Iterated: 10 slots take experts 7, 4, 3 and 2 (12 slots). Layer 2 then has 7 host matrices per role, and the
    // window stays at G_plan: the ring that loan leaves is the ring it was sized for.
    const ggml_sycl::moe_stream_loan_sizing whole = ggml_sycl::moe_stream_loan_size_post_loan(sizer, 6 * kSrc);
    CHECK(whole.converged);
    CHECK(whole.steps == 2);
    CHECK(whole.bytes == 10 * kSrc);
    CHECK(whole.n_triplets == 4);
    CHECK(whole.geo.region_slots == 5 && whole.geo.n_slots == 10 && whole.geo.slot_bytes == kSrc);
    // The overlay sizes what a really flipped plan sizes.
    const auto real_post = geometry_after_loan(plan, 4);
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(whole.geo, real_post));
    // The boundary's loan reselects the same four triplets from the same plan.
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, whole.bytes, whole.geo.slot_bytes, nullptr).size() == 12);

    // Five of six slots held: one slot short takes expert 7 alone, which widens the window to 4 and the ring to 8
    // slots. The 5 held slots still count (same slot size), so the second request is 3 slots, which expert 7 funds:
    // the ring stays at 8.
    const ggml_sycl::moe_stream_loan_sizing held = ggml_sycl::moe_stream_loan_size_post_loan(sizer, kSrc);
    CHECK(held.converged);
    CHECK(held.steps == 2);
    CHECK(held.bytes == 3 * kSrc);
    CHECK(held.n_triplets == 1);
    CHECK(held.geo.region_slots == 4 && held.geo.n_slots == 8);
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(held.geo, geometry_after_loan(plan, 1)));

    // Through the short reserve's loan step: the converged request, counted in that geometry's slots.
    g_requests = loan_requests();
    ggml_sycl::moe_stream_loan_latch latch;
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, 6 * kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::REQUEST);
    CHECK(g_requests.n == 1 && g_requests.bytes == 10 * kSrc && g_requests.slot_bytes == kSrc);
    CHECK(latch.asked && !latch.refused && !latch.unconverged);

    // A loan that leaves the window where it was converges at once, with the reserve's own shortfall: G_plan is capped
    // at 2 before and after the loan, so the window cannot move.
    ggml_sycl::moe_stream_loan_sizer capped = sizer;
    capped.hints.weight_slot_bytes          = 4 * kSrc;  // fit = 4 -> G_plan = 2
    capped.geo = ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, capped.hints);
    CHECK(capped.geo.n_slots == 4);
    const ggml_sycl::moe_stream_loan_sizing flat = ggml_sycl::moe_stream_loan_size_post_loan(capped, 2 * kSrc);
    CHECK(flat.converged && flat.steps == 1 && flat.bytes == 2 * kSrc && flat.n_triplets == 1);
}

// A loan can leave a SMALLER ring in the same slot size: a lent matrix whose f16 expert is larger than every host one
// lowers G_plan, and with it the window. That ring fits in the room the request frees, so the sizing converges at
// once and the request stays the one sized for the target; the geometry is the smaller ring prepare() reserves.
void test_loan_into_a_smaller_ring_is_covered() {
    placement_plan plan = make_loan_plan(kAnchorExtra);
    // Layer 2's expert 7, the walk's first lendable triplet, as Q4_0 matrices of 64 x 112: 2 blocks of 18 bytes a
    // row, 4032 source bytes, so one kSrc slot still holds it and the largest matrix stays kSrc; but 14336 f16 bytes,
    // so the planned slot of 10 x kSrc fits 2 of them and G_plan drops from 5 to 1 once it is a host matrix.
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.layer_id == 2 && e.expert_id == 7 && ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role)) {
            e.type     = GGML_TYPE_Q4_0;
            e.ne[0]    = 64;
            e.ne[1]    = 112;
            e.src_size = 112 * ggml_row_size(GGML_TYPE_Q4_0, 64);
        }
    }
    const ggml_sycl::moe_stream_loan_sizer sizer = loan_sizer(plan);
    CHECK(sizer.geo.region_slots == 3 && sizer.geo.n_slots == 6 && sizer.geo.slot_bytes == kSrc);
    CHECK(sizer.geo.bytes == 6 * kSrc);

    // The whole ring short: 6 slots take experts 7 and 4. Layer 2 then has 5 host matrices per role, but G_plan is 1,
    // so the ring the loan leaves is 2 slots of kSrc, smaller than the 6 it was sized for.
    const ggml_sycl::moe_stream_loan_sizing whole = ggml_sycl::moe_stream_loan_size_post_loan(sizer, 6 * kSrc);
    CHECK(whole.converged);
    CHECK(whole.steps == 1);
    CHECK(whole.bytes == sizer.geo.bytes);
    CHECK(whole.n_triplets == 2);
    CHECK(whole.geo.slot_bytes == kSrc && whole.geo.region_slots == 1 && whole.geo.n_slots == 2);
    CHECK(whole.geo.bytes == 2 * kSrc);
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(whole.geo, geometry_after_loan(plan, 2)));

    // Through the short reserve's loan step: the request is the target's, in its slot size.
    g_requests = loan_requests();
    ggml_sycl::moe_stream_loan_latch latch;
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, 6 * kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::REQUEST);
    CHECK(g_requests.n == 1 && g_requests.bytes == 6 * kSrc && g_requests.slot_bytes == kSrc);
    CHECK(latch.asked && !latch.refused && !latch.unconverged);
}

// A loan whose first triplet is larger than a slot moves the ring to a larger slot size. The reserve's held slots were
// cut in the old size, so they credit nothing in the new one: the next request is the whole new ring.
void test_loan_into_a_larger_slot_credits_no_held_slot() {
    placement_plan plan = make_loan_plan(1000000);  // resident room for the 8-slot ring of 2 x kSrc slots below
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.layer_id != 2 || !ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role) || !e.on_device) {
            continue;
        }
        // Each resident layer-2 matrix charges 2 x kDst = 8704 bytes: 2 slots of kSrc, 1 slot of 2 x kSrc.
        e.dst_size         = 2 * kDst;
        e.vram_charge_size = 2 * kDst;
        if (e.expert_id == 7) {
            // The walk's first lendable triplet as F16 matrices of 64 x 64: 2 x kSrc source and f16 bytes.
            e.ne[0]    = 64;
            e.ne[1]    = 64;
            e.src_size = 2 * kSrc;
        }
    }
    ggml_sycl::moe_stream_loan_sizer sizer = loan_sizer(plan);

    // A planned slot of 20 matrices of kSrc and a group cap of 4: G_plan is 4 before the loan (fit 20) and after it
    // (fit 10), so only the role counts and the slot size move the ring.
    sizer.hints.weight_slot_bytes = 20 * kSrc;
    sizer.hints.group_cap         = 4;
    sizer.geo                     = ggml_sycl::moe_stream_plan_ring_geometry(plan, 0, nullptr, claims_all, sizer.hints);
    CHECK(sizer.geo.region_slots == 3 && sizer.geo.n_slots == 6 && sizer.geo.slot_bytes == kSrc);

    // One slot short, five held. Step 1 asks for that one kSrc slot, which expert 7 funds alone (2 slots a matrix);
    // lent, it is a host matrix of 2 x kSrc, so the ring it leaves is 2 x 4 slots of 2 x kSrc. Step 2 asks for all 8
    // of them: the held slots are kSrc wide, not 2 x kSrc, and credit nothing. Experts 7, 4 and 3 (one slot a matrix)
    // fund that ring and leave the same one.
    const auto first_post = geometry_after_loan(plan, 1, sizer.hints);
    CHECK(first_post.slot_bytes == 2 * kSrc && first_post.region_slots == 4 && first_post.n_slots == 8);
    const ggml_sycl::moe_stream_loan_sizing held = ggml_sycl::moe_stream_loan_size_post_loan(sizer, kSrc);
    CHECK(held.converged);
    CHECK(held.steps == 2);
    CHECK(held.bytes == static_cast<size_t>(first_post.n_slots) * first_post.slot_bytes);
    CHECK(held.n_triplets == 3);
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(held.geo, first_post));
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(held.geo, geometry_after_loan(plan, 3, sizer.hints)));

    // Through the short reserve's loan step: the whole new ring, in its slot size.
    g_requests = loan_requests();
    ggml_sycl::moe_stream_loan_latch latch;
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::REQUEST);
    CHECK(g_requests.n == 1 && g_requests.bytes == 16 * kSrc && g_requests.slot_bytes == 2 * kSrc);
}

// A ring in another slot size is not comparable by bytes, so one that did not grow is refused, not covered: here the
// lent matrix is wider than a slot and, as Q8_0, its f16 expert is larger again, which drops G_plan to 2.
void test_loan_into_another_slot_size_that_does_not_grow_is_refused() {
    placement_plan plan = make_loan_plan(kAnchorExtra);
    for (ggml_sycl::placement_entry & e : plan.entries) {
        if (e.layer_id == 2 && e.expert_id == 7 && ggml_sycl::detail::moe_expert_role_is_triplet(e.expert_role)) {
            // Q8_0 matrices of 64 x 64: 2 blocks of 34 bytes a row, 4352 source bytes (one 4352-byte slot, 17 x the
            // 256-byte alignment), but 8192 f16 bytes, so the planned slot of 10 x kSrc fits 5 and G_plan is 2.
            e.type     = GGML_TYPE_Q8_0;
            e.ne[0]    = 64;
            e.ne[1]    = 64;
            e.src_size = 64 * ggml_row_size(GGML_TYPE_Q8_0, 64);
        }
    }
    const ggml_sycl::moe_stream_loan_sizer sizer = loan_sizer(plan);
    CHECK(sizer.geo.n_slots == 6 && sizer.geo.slot_bytes == kSrc && sizer.geo.bytes == 6 * kSrc);

    // The whole ring short takes experts 7 and 4: the ring they leave is 2 x 2 slots of 4352 bytes, smaller than the
    // 6 x kSrc target in bytes but in another slot size.
    const ggml_sycl::moe_stream_loan_sizing whole = ggml_sycl::moe_stream_loan_size_post_loan(sizer, 6 * kSrc);
    CHECK(!whole.converged);
    CHECK(whole.steps == 1);
    CHECK(whole.geo.slot_bytes == 4352 && whole.geo.n_slots == 4 && whole.geo.bytes == 4 * 4352);
    CHECK(ggml_sycl::moe_stream_ring_geometry_same(whole.geo, geometry_after_loan(plan, 2)));

    g_requests = loan_requests();
    ggml_sycl::moe_stream_loan_latch latch;
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, 6 * kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::REFUSED);
    CHECK(g_requests.n == 0);
    CHECK(latch.refused && latch.unconverged && !latch.asked);
    CHECK(latch.sizing_steps == 1 && latch.post_slots == 4 && latch.post_bytes == 4 * 4352);
}

// No loan converges when the ring it leaves exceeds the cap: the cap does not move with the loan (lent bytes count
// back as resident), but the ring widens. The sizing refuses, the short reserve's loan step answers REFUSED once and
// asks for nothing, and the latch holds for the prompt, as for a loan the boundary refused.
void test_loan_whose_post_loan_ring_exceeds_the_cap_is_refused() {
    const placement_plan                   plan  = make_loan_plan(0);
    const ggml_sycl::moe_stream_loan_sizer sizer = loan_sizer(plan);
    // 300288 resident bytes: the 24576-byte ring fits under an eleventh (27298), the 32768-byte one does not.
    CHECK(sizer.geo.n_slots == 6 && sizer.geo.bytes == 6 * kSrc);
    CHECK(geometry_after_loan(plan, 1).n_slots == 0);

    const ggml_sycl::moe_stream_loan_sizing whole = ggml_sycl::moe_stream_loan_size_post_loan(sizer, 6 * kSrc);
    CHECK(!whole.converged);
    CHECK(whole.steps == 1);
    CHECK(whole.geo.n_slots == 0);
    const ggml_sycl::moe_stream_loan_sizing held = ggml_sycl::moe_stream_loan_size_post_loan(sizer, kSrc);
    CHECK(!held.converged && held.steps == 1);

    g_requests = loan_requests();
    ggml_sycl::moe_stream_loan_latch latch;
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, 6 * kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::REFUSED);
    CHECK(g_requests.n == 0);
    CHECK(latch.refused && latch.unconverged && !latch.asked);
    // The reason the WARN prints: one step, and a post-loan ring of 0 slots, over the cap.
    CHECK(latch.sizing_steps == 1 && latch.post_slots == 0 && latch.post_bytes == 0);
    // Once per prompt: later short micro-batches ask nothing and answer NONE.
    CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(latch, 0, 6 * kSrc, kSrc, stub_request_loan, stub_loan_refusals,
                                                      &sizer) == ggml_sycl::moe_stream_loan_step::NONE);
    CHECK(g_requests.n == 0);

    // Nothing to size (no plan, no ring, or nothing short) converges to nothing, before any step.
    ggml_sycl::moe_stream_loan_sizer no_plan = sizer;
    no_plan.plan                             = nullptr;

    const ggml_sycl::moe_stream_loan_sizing planless = ggml_sycl::moe_stream_loan_size_post_loan(no_plan, 6 * kSrc);
    CHECK(!planless.converged && planless.steps == 0);
    ggml_sycl::moe_stream_loan_sizer no_ring = sizer;
    no_ring.geo                              = {};

    const ggml_sycl::moe_stream_loan_sizing ringless = ggml_sycl::moe_stream_loan_size_post_loan(no_ring, 6 * kSrc);
    CHECK(!ringless.converged && ringless.steps == 0);
    const ggml_sycl::moe_stream_loan_sizing nothing_short = ggml_sycl::moe_stream_loan_size_post_loan(sizer, 0);
    CHECK(!nothing_short.converged && nothing_short.steps == 0);
}

}  // namespace

int main() {
    const placement_plan plan      = make_plan();
    const int32_t        no_ids[1] = { 0 };

    // STREAM_ALL: 3 triplets in entries order, gate/up/down within each; ids ignored.
    const int32_t ignored[2] = { 0, 2 };
    const auto    all        = plan_build(plan, 1, moe_exec_regime::STREAM_ALL, nullptr, 0);
    CHECK(all.size() == 9);
    const int                exp_e[3] = { 1, 5, 6 };
    const expert_tensor_role exp_r[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
    for (size_t i = 0; i < all.size() && i < 9; ++i) {
        CHECK(all[i].layer == 1);
        CHECK(all[i].expert_id == exp_e[i / 3]);
        CHECK(all[i].role == exp_r[i % 3]);
        CHECK(all[i].src_bytes == kSrc);
        CHECK(all[i].type == 0 && all[i].ncols == 0 && all[i].nrows == 0);
    }

    // ROUTED_ONLY with ids {1, 1, 5, 7}: 1 and 5 are host, 7 is resident -> 2 triplets.
    const int32_t routed[4] = { 1, 1, 5, 7 };
    const auto    part      = plan_build(plan, 1, moe_exec_regime::ROUTED_ONLY, routed, 4);
    CHECK(part.size() == 6);
    for (size_t i = 0; i < part.size() && i < 6; ++i) {
        CHECK(part[i].expert_id == (i < 3 ? 1 : 5));
        CHECK(part[i].role == exp_r[i % 3]);
    }

    // Entries order, not ids order: {5, 1, 5, 7} still lists expert 1 before expert 5.
    const int32_t reordered[4] = { 5, 1, 5, 7 };
    const auto    ord          = plan_build(plan, 1, moe_exec_regime::ROUTED_ONLY, reordered, 4);
    CHECK(ord.size() == 6);
    for (size_t i = 0; i < ord.size() && i < 6; ++i) {
        CHECK(ord[i].expert_id == (i < 3 ? 1 : 5));
    }

    // A layer with no host experts lists nothing under either regime.
    CHECK(plan_build(plan, 3, moe_exec_regime::STREAM_ALL, no_ids, 1).empty());
    CHECK(plan_build(plan, 3, moe_exec_regime::ROUTED_ONLY, routed, 4).empty());
    CHECK(plan_build(plan, 1, moe_exec_regime::NONE, routed, 4).empty());

    // fully_staged: true for the STREAM_ALL list, false once one item is removed.
    CHECK(layer_fully_staged(all, plan, 1, routed, 4));
    CHECK(layer_fully_staged(all, plan, 1, ignored, 2));         // 0 and 2 are resident
    std::vector<moe_stream_item> short_list = all;
    short_list.erase(short_list.begin() + 4);                    // expert 5's up
    CHECK(!layer_fully_staged(short_list, plan, 1, routed, 4));
    CHECK(layer_fully_staged(short_list, plan, 1, ignored, 2));  // 5 is not asked for
    CHECK(!layer_fully_staged(std::vector<moe_stream_item>{}, plan, 1, routed, 4));
    CHECK(layer_fully_staged(std::vector<moe_stream_item>{}, plan, 3, routed, 4));

    // A staged item of another layer does not count.
    std::vector<moe_stream_item> wrong_layer = all;
    wrong_layer[4].layer                     = 2;
    CHECK(!layer_fully_staged(wrong_layer, plan, 1, routed, 4));

    // The reference above is checked against production: moe_stream_plan_index_build lists, for every layer with a
    // host expert, the same (expert_id, role) sequence as the STREAM_ALL reference, and omits layer 3 (no host
    // experts). The null library_claims makes the production staging NONE for every item, and the hand-built entries'
    // zeroed type and shape would force it too; only the list is compared. The comparison fails on a dropped role, a
    // reordered expert, or a fused GATE_UP host entry listed as a triplet (layer 1 would then have 10 items, not 9).
    const ggml_sycl::moe_stream_plan_index idx =
        ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, nullptr, ggml_sycl::moe_stream_window_hints());
    for (int l = 0; l < kLayers; ++l) {
        const auto ref = plan_build(plan, l, moe_exec_regime::STREAM_ALL, nullptr, 0);
        const auto it  = idx.layers.find(l);
        if (ref.empty()) {
            CHECK(it == idx.layers.end());
            continue;
        }
        CHECK(it != idx.layers.end());
        if (it == idx.layers.end()) {
            continue;
        }
        CHECK(it->second.items.size() == ref.size());
        for (size_t i = 0; i < ref.size() && i < it->second.items.size(); ++i) {
            CHECK(it->second.items[i].expert_id == ref[i].expert_id);
            CHECK(it->second.items[i].role == ref[i].role);
        }
    }

    test_window_sizing();
    test_ring_cap_is_loan_invariant();
    test_loan_converges_on_the_post_loan_ring();
    test_loan_whose_post_loan_ring_exceeds_the_cap_is_refused();
    test_loan_into_a_smaller_ring_is_covered();
    test_loan_into_a_larger_slot_credits_no_held_slot();
    test_loan_into_another_slot_size_that_does_not_grow_is_refused();

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-plan: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-plan: OK\n");
    return 0;
}
