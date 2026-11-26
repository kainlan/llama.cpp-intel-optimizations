// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it. The rule is the header
// comment of moe-ring-consumer.hpp; the figures measured on the cards are in docs/backend/sycl-memory-design.md.
//
// The first sections ask (a) to (c) (moe_pp_ring_check_resident_only, moe_pp_ring_resident_check::displaces_experts) on
// synthetic plans, one case per type class: the resident-only types with the MMID direct route (the drop set), the
// resident-only types without it, a staged type, the per-expert SOA staging and a type with no ring use. An expert the
// plan does not place on a device is on the host, by the plan's own convention (placement_plan::expert_on_device),
// whether it has a host entry, a name-keyed fallback entry on the host or no entry. The next sections ask (d) through
// the functions the load stage and the plan dry run call, in the order they call them: the RUNTIME room
// (moe_pp_ring_runtime_with_ring, moe_pp_ring_runtime_room), the WEIGHT room as the planner reads it
// (kv_weight_capacity, the pure part of unified_cache_kv_weight_capacity), whether a drop can return room
// (moe_pp_ring_drop_can_return_room), the decision (moe_pp_ring_decide) and its record
// (moe_pp_ring_drop_marks::record). The zone figures are the ones the lead's B50 runs printed, not a model of the
// arena's layout. The last sections pin the second plan after a kept give-back rebuild (moe_pp_ring_kept_plan_again)
// and the experts' worth a held ring keeps from its weights.
//
// Host-only: synthetic plans, no queue, no device, no model.

#include "ggml-sycl.h"
#include "ggml-sycl/kv-runtime-demotion.hpp"
#include "ggml-sycl/moe-ring-consumer.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::moe_pp_ring_decision;
using ggml_sycl::moe_pp_ring_drop_marks;
using ggml_sycl::moe_pp_ring_resident_check;
using ggml_sycl::moe_pp_ring_weight;
using ggml_sycl::moe_pp_ring_zone_answer;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_plan;

constexpr int    k_experts = 4;
constexpr size_t k_bytes   = 4096;
constexpr int    k_host    = -1;
constexpr size_t k_mib     = 1024 * 1024;

// A figure the runs printed in MB (MiB) to one decimal, in whole tenths of a MiB rounded up to a byte, so that the
// figures' differences are exact.
constexpr size_t k_tenth_mib = k_mib / 10 + 1;

constexpr size_t mib_tenths(size_t tenths) {
    return tenths * k_tenth_mib;
}

// The zone figures of the lead's B50 runs (docs/backend/sycl-memory-design.md, the measured tables of the ring drop,
// and docs/backend/data/inkling2/t8b-lead-runs-0f4060d0f.txt for the arena's total and its two layouts). The plan's
// budget is the arena's total, as a single-device plan packs the device to; each WEIGHT figure is the WEIGHT zone a
// drop line printed before and after the give-back.
constexpr size_t k_b50_budget                   = mib_tenths(142920);
constexpr size_t k_runtime_floor                = mib_tenths(5120);
constexpr size_t k_inkling_b50_runtime_with     = mib_tenths(5586);
constexpr size_t k_inkling_b50_runtime_without  = mib_tenths(5120);
constexpr size_t k_inkling_b50_weight_with      = mib_tenths(125799);
constexpr size_t k_inkling_b50_weight_without   = mib_tenths(126265);
constexpr size_t k_qwen_b50_runtime_early       = mib_tenths(7264);
constexpr size_t k_qwen_b50_runtime_late_with   = mib_tenths(11474);
constexpr size_t k_qwen_b50_runtime_without     = mib_tenths(7264);
constexpr size_t k_qwen_b50_weight_early        = mib_tenths(121646);
constexpr size_t k_qwen_b50_weight_late_with    = mib_tenths(121646);
constexpr size_t k_qwen_b50_weight_late_without = mib_tenths(125856);
// The same B50 reserve in one chunk and, right after a destroy, in two: the WEIGHT zone the tail chunk's KV share
// smaller.
constexpr size_t k_b50_weight_one_chunk         = mib_tenths(116046);
constexpr size_t k_b50_weight_two_chunks        = mib_tenths(114766);

int failures = 0;

const char * decision_name(moe_pp_ring_decision d) {
    switch (d) {
        case moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS:
            return "keep: no room returns";
        case moe_pp_ring_decision::KEEP_HELD:
            return "keep: the arena is held";
        case moe_pp_ring_decision::DROP:
            return "drop";
    }
    return "unknown";
}

// Where one expert is planned: a device id, or k_host; and its layout on the device.
struct expert_spot {
    int              device = k_host;
    ggml_layout_mode layout = GGML_LAYOUT_AOS;
};

struct weight_spec {
    std::string        name;
    ggml_type          type;
    expert_tensor_role role;
    int                layer;
};

// A plan of `weights`, each expert placed by `spot(weight index, expert)`, indexed as the stage indexes its plan. An
// unclassified plan gives its entries no layer, so the semantic index misses them and the plan answers for them only
// through its name-keyed fallback entries.
placement_plan make_plan(const std::vector<weight_spec> &                weights,
                         const std::function<expert_spot(size_t, int)> & spot,
                         bool                                            classified = true) {
    placement_plan plan{};
    plan.multi_device = true;
    for (size_t w = 0; w < weights.size(); ++w) {
        for (int e = 0; e < k_experts; ++e) {
            const expert_spot s = spot(w, e);
            placement_entry   entry;
            entry.name          = weights[w].name;
            entry.type          = weights[w].type;
            entry.src_size      = k_bytes;
            entry.dst_size      = k_bytes;
            entry.layer_id      = classified ? weights[w].layer : -1;
            entry.expert_id     = e;
            entry.expert_role   = weights[w].role;
            entry.layout        = s.layout;
            entry.on_device     = s.device != k_host;
            entry.target_device = s.device;
            plan.entries.push_back(entry);
        }
    }
    plan.build_index();
    return plan;
}

// The ring weights of `weights`, as the stage lists them from its inventory: each with its expert count, its bytes
// (k_bytes per expert), the uses its type reaches and whether the MMID direct route runs its type. `extra_experts` asks
// about experts the plan has no entry for.
std::vector<moe_pp_ring_weight> ring_weights(const std::vector<weight_spec> & weights, int extra_experts = 0) {
    std::vector<moe_pp_ring_weight> out;
    for (const weight_spec & w : weights) {
        moe_pp_ring_weight r;
        r.name                      = w.name.c_str();
        r.n_experts                 = k_experts + extra_experts;
        r.bytes                     = k_bytes * static_cast<size_t>(k_experts + extra_experts);
        r.uses                      = moe_pp_ring_uses_of_type(w.type);
        r.other_device_prompt_route = moe_mmid_direct_route_runs_type(w.type);
        out.push_back(r);
    }
    return out;
}

// What a case wants of the check: its four fields, and whether the plan displaced experts, so that (d) decides.
struct want_check {
    size_t  weights;
    bool    resident_only;
    bool    other_route;
    int64_t host;
    bool    displaces;
};

void expect(const std::string &                     what,
            const placement_plan &                  plan,
            const std::vector<moe_pp_ring_weight> & weights,
            const want_check &                      want) {
    const moe_pp_ring_resident_check c =
        ggml_sycl::moe_pp_ring_check_resident_only(plan, weights.data(), weights.size());
    std::printf("%s: weights=%zu resident_only=%d other_route=%d host_experts=%" PRId64
                " displaces=%d (want %zu %d %d %" PRId64 " %d)\n",
                what.c_str(), c.n_weights, c.resident_only ? 1 : 0, c.other_device_prompt_route ? 1 : 0, c.host_experts,
                c.displaces_experts() ? 1 : 0, want.weights, want.resident_only ? 1 : 0, want.other_route ? 1 : 0,
                want.host, want.displaces ? 1 : 0);
    if (c.n_weights != want.weights || c.resident_only != want.resident_only ||
        c.other_device_prompt_route != want.other_route || c.host_experts != want.host ||
        c.displaces_experts() != want.displaces) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::vector<weight_spec> inkling_weights(int n_layers) {
    std::vector<weight_spec> w;
    for (int l = 0; l < n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        w.push_back({ blk + "ffn_gate_up_exps.weight", GGML_TYPE_NVFP4, expert_tensor_role::GATE_UP, l });
        w.push_back({ blk + "ffn_down_exps.weight", GGML_TYPE_NVFP4, expert_tensor_role::DOWN, l });
    }
    return w;
}

std::vector<weight_spec> triplet_weights(ggml_type type, int n_layers) {
    std::vector<weight_spec> w;
    for (int l = 0; l < n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        w.push_back({ blk + "ffn_gate_exps.weight", type, expert_tensor_role::GATE, l });
        w.push_back({ blk + "ffn_up_exps.weight", type, expert_tensor_role::UP, l });
        w.push_back({ blk + "ffn_down_exps.weight", type, expert_tensor_role::DOWN, l });
    }
    return w;
}

expert_spot on_device(int device, ggml_layout_mode layout = GGML_LAYOUT_AOS) {
    return expert_spot{ device, layout };
}

expert_spot on_host(ggml_layout_mode layout = GGML_LAYOUT_AOS) {
    return expert_spot{ k_host, layout };
}

void expect_bytes(const char * what, size_t got, size_t want) {
    std::printf("%s: %zu (want %zu)\n", what, got, want);
    if (got != want) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void expect_flag(const char * what, bool got, bool want) {
    std::printf("%s: %d (want %d)\n", what, got ? 1 : 0, want ? 1 : 0);
    if (got != want) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void expect_decision(const std::string & what, moe_pp_ring_decision got, moe_pp_ring_decision want) {
    std::printf("%s: %s (want %s)\n", what.c_str(), decision_name(got), decision_name(want));
    if (got != want) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// A device whose plan's planner reads no WEIGHT zone, or that has no active arena: an active arena's WEIGHT zone is
// never empty, so an empty one stands for either here.
constexpr size_t k_no_weight_zone = 0;

// The WEIGHT room the backend reads for the ring decision (ggml_sycl_ring_plan_weight_room): none for
// k_no_weight_zone (the ring-drop source contract pins that gate), else the plan's planner's own figure, as
// unified_cache_kv_weight_capacity answers it: the WEIGHT zone, no more than the budget the plan packed the device to.
// kv_weight_capacity alone would answer the budget for an empty zone, as the planner does where it reads none.
size_t planner_weight_room(size_t budget, size_t weight_zone) {
    return weight_zone == k_no_weight_zone ? 0 : ggml_sycl::kv_weight_capacity(budget, weight_zone);
}

// What the arena answers a give-back in the load stage, as the stage reads it.
struct arena_reply {
    size_t weight_zone_after = 0;  // the WEIGHT zone after the give-back's ensure
    bool   held              = false;
};

// The load stage's (d) on a device whose plan displaced experts, through the functions it calls in its order: the
// WEIGHT room the plan was packed against; the arena asked only where a drop can return room, its reply read as the
// planner reads it; the decision; and its record in `marks` for load `txn`. `asked` says whether the arena was asked.
moe_pp_ring_decision stage_decide(moe_pp_ring_drop_marks & marks,
                                  int                      device,
                                  uint64_t                 txn,
                                  size_t                   budget,
                                  size_t                   weight_zone_with_ring,
                                  size_t                   runtime_room,
                                  const arena_reply &      reply,
                                  bool *                   asked = nullptr) {
    moe_pp_ring_zone_answer zone;
    zone.weight_with_ring    = planner_weight_room(budget, weight_zone_with_ring);
    zone.weight_without_ring = zone.weight_with_ring;
    const bool ask           = ggml_sycl::moe_pp_ring_drop_can_return_room(zone.weight_with_ring, runtime_room);
    if (ask) {
        zone.weight_without_ring = planner_weight_room(budget, reply.weight_zone_after);
        zone.held                = reply.held;
    }
    if (asked != nullptr) {
        *asked = ask;
    }
    const moe_pp_ring_decision decision = ggml_sycl::moe_pp_ring_decide(zone);
    marks.record(device, txn, decision);
    return decision;
}

// The plan dry run's (d): the same question on the WEIGHT room it read, then its own answer from the RUNTIME room.
moe_pp_ring_decision dry_run_decide(size_t weight_room, size_t runtime_room) {
    return ggml_sycl::moe_pp_ring_decide(ggml_sycl::moe_pp_ring_dry_run_answer(
        ggml_sycl::moe_pp_ring_drop_can_return_room(weight_room, runtime_room), runtime_room));
}

void expect_plan_again(const char * what, size_t packed_against, size_t weight_after, bool want) {
    const bool got = ggml_sycl::moe_pp_ring_kept_plan_again(packed_against, weight_after);
    std::printf("%s: %zu MiB packed against, %zu MiB after: %s (want %s)\n", what, packed_against / k_mib,
                weight_after / k_mib, got ? "plan again" : "stands", want ? "plan again" : "stands");
    if (got != want) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    // Inkling: NVFP4 reaches the resident claim alone and the MMID direct route runs it, so the plan displaced experts
    // only with an NVFP4 expert on the host.
    {
        const std::vector<weight_spec> w = inkling_weights(2);
        const auto                     r = ring_weights(w);
        const size_t                   n = w.size();

        // Every expert on device 0: none displaced.
        const placement_plan b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        expect("inkling all on device 0 displaces none", b70, r, { n, true, true, 0, false });

        // One expert of one weight on the host: the ring displaced it.
        const placement_plan one_host =
            make_plan(w, [](size_t i, int e) { return i == 3 && e == 2 ? on_host() : on_device(0); });
        expect("inkling one expert on the host displaces it", one_host, r, { n, true, true, 1, true });

        // Every expert on the device but in two layouts: none is on the host, although the claim takes one layout. The
        // check asks the host, not the claim.
        const placement_plan two_layouts =
            make_plan(w, [](size_t, int e) { return on_device(0, e == 1 ? GGML_LAYOUT_SOA : GGML_LAYOUT_AOS); });
        expect("inkling on device 0 in two layouts displaces none", two_layouts, r, { n, true, true, 0, false });

        // Experts split across the two cards, none on the host.
        const placement_plan split = make_plan(w, [](size_t, int e) { return on_device(e % 2); });
        expect("inkling split across devices displaces none", split, r, { n, true, true, 0, false });

        // An expert the plan has no entry for is on the host: the plan never places a missing expert on a device.
        expect("inkling asked for experts the plan lacks displaces them", b70, ring_weights(w, 1),
               { n, true, true, static_cast<int64_t>(n), true });

        // Experts the semantic index misses are read through their name-keyed fallback entries: on the host they count,
        // on the device they do not.
        const placement_plan fallback_host =
            make_plan(w, [](size_t i, int) { return i == 0 ? on_host() : on_device(0); }, /*classified=*/false);
        expect("inkling fallback entries on the host displace them", fallback_host, r,
               { n, true, true, k_experts, true });
        const placement_plan fallback_device = make_plan(w, [](size_t, int) { return on_device(0); }, false);
        expect("inkling fallback entries on device 0 displace none", fallback_device, r, { n, true, true, 0, false });

        // Every expert on the host: each is counted.
        const placement_plan all_host = make_plan(w, [](size_t, int) { return on_host(); });
        expect("inkling all on the host displaces them all", all_host, r,
               { n, true, true, static_cast<int64_t>(n) * k_experts, true });
    }

    // Inkling with its first layers resident and the rest on the host: the count is every expert of the other layers.
    {
        constexpr int                  k_inkling_layers  = 24;
        constexpr int                  k_resident_layers = 14;
        const std::vector<weight_spec> w                 = inkling_weights(k_inkling_layers);
        const placement_plan           tiered =
            make_plan(w, [&](size_t i, int) { return w[i].layer < k_resident_layers ? on_device(0) : on_host(); });
        const int64_t host = static_cast<int64_t>(k_inkling_layers - k_resident_layers) * 2 * k_experts;
        expect("inkling 14 layers resident and the rest on the host displaces the rest", tiered, ring_weights(w),
               { w.size(), true, true, host, true });
    }

    // The drop set by type: a resident-only type the MMID direct route runs (NVFP4 today) displaced experts when the
    // plan leaves one of its experts on the host, in the separate gate, up and down shape too (Qwen3.6 NVFP4).
    for (const ggml_type type : { GGML_TYPE_NVFP4 }) {
        const std::vector<weight_spec> w        = triplet_weights(type, 1);
        const std::string              name     = ggml_type_name(type);
        const placement_plan           resident = make_plan(w, [](size_t, int) { return on_device(0); });
        expect(name + " all on device 0 displaces none", resident, ring_weights(w), { w.size(), true, true, 0, false });
        const placement_plan one_host =
            make_plan(w, [](size_t i, int e) { return i == 2 && e == 0 ? on_host() : on_device(0); });
        expect(name + " one expert on the host displaces it: its resident nodes keep the MMID direct route", one_host,
               ring_weights(w), { w.size(), true, true, 1, true });
    }

    // The resident-only types the MMID direct route does not run (IQ1_S, IQ1_M and IQ4_XS today) keep the ring with
    // their experts on the host: the ring is their resident layers' only device prompt route (decode stays refused), so
    // without it a node the executor's claim admits would have no kernel. Their host experts are not counted.
    for (const ggml_type type : { GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_XS }) {
        const std::vector<weight_spec> w    = triplet_weights(type, 2);
        const std::string              name = ggml_type_name(type);
        const placement_plan           tiered =
            make_plan(w, [&](size_t i, int) { return w[i].layer == 0 ? on_device(0) : on_host(); });
        expect(name +
                   " with host experts keeps the ring: the ring is its resident layers' only device prompt route, "
                   "so dropping it would leave an admitted node with no kernel",
               tiered, ring_weights(w), { w.size(), true, false, 0, false });
    }

    // A ring with an NVFP4 weight beside an IQ4_XS one keeps the ring: the IQ4_XS layer has no device prompt route
    // without it, whatever the NVFP4 layer's host experts.
    {
        std::vector<weight_spec> w = inkling_weights(1);
        w.push_back({ "blk.1.ffn_down_exps.weight", GGML_TYPE_IQ4_XS, expert_tensor_role::DOWN, 1 });
        const placement_plan mixed =
            make_plan(w, [&](size_t i, int) { return w[i].type == GGML_TYPE_NVFP4 ? on_host() : on_device(0); });
        expect("an iq4_xs weight beside nvfp4 ones keeps the ring", mixed, ring_weights(w),
               { w.size(), true, false, 0, false });
    }

    // Qwen: IQ4_NL reaches the staged claim too, so its ring is not asked about however many experts are on the host.
    {
        const std::vector<weight_spec> w = triplet_weights(GGML_TYPE_IQ4_NL, 2);
        const auto                     r = ring_weights(w);

        const placement_plan tiered = make_plan(w, [](size_t, int e) { return e == 0 ? on_device(0) : on_host(); });
        expect("iq4_nl (qwen) host-tiered keeps the ring", tiered, r, { w.size(), false, false, 0, false });

        const placement_plan all_host = make_plan(w, [](size_t, int) { return on_host(); });
        expect("iq4_nl (qwen) all on the host keeps the ring", all_host, r, { w.size(), false, false, 0, false });
    }

    // GPT-OSS: MXFP4 reaches the per-expert SOA staging too, so its ring is not asked about either.
    {
        const std::vector<weight_spec> w = triplet_weights(GGML_TYPE_MXFP4, 2);
        const auto                     r = ring_weights(w);

        const placement_plan hybrid = make_plan(
            w, [](size_t, int e) { return e == 0 ? on_device(0, GGML_LAYOUT_SOA) : on_host(GGML_LAYOUT_SOA); });
        expect("mxfp4 (gpt-oss) hybrid keeps the ring", hybrid, r, { w.size(), false, false, 0, false });

        const placement_plan resident = make_plan(w, [](size_t, int) { return on_device(0, GGML_LAYOUT_XMX_TILED); });
        expect("mxfp4 (gpt-oss) all on device 0 keeps the ring", resident, r, { w.size(), false, false, 0, false });
    }

    // A ring with one staged weight beside resident-only ones keeps today's behaviour: the staged weight's host experts
    // are what the ring serves.
    {
        std::vector<weight_spec> w = inkling_weights(1);
        w.push_back({ "blk.1.ffn_down_exps.weight", GGML_TYPE_IQ4_NL, expert_tensor_role::DOWN, 1 });
        const placement_plan mixed = make_plan(w, [](size_t, int) { return on_host(); });
        expect("a staged weight beside nvfp4 ones keeps the ring", mixed, ring_weights(w),
               { w.size(), false, false, 0, false });
    }

    // A weight with no ring use is not a ring weight: it is skipped, and a list of only such weights asks nothing.
    {
        const std::vector<weight_spec> q1      = triplet_weights(GGML_TYPE_Q1_0, 1);
        const placement_plan           q1_host = make_plan(q1, [](size_t, int) { return on_host(); });
        expect("q1_0 weights alone ask nothing", q1_host, ring_weights(q1), { 0, false, false, 0, false });

        std::vector<weight_spec> w = inkling_weights(1);
        w.push_back({ "blk.1.ffn_down_exps.weight", GGML_TYPE_Q1_0, expert_tensor_role::DOWN, 1 });
        const placement_plan beside =
            make_plan(w, [&](size_t i, int) { return w[i].type == GGML_TYPE_Q1_0 ? on_host() : on_device(0); });
        expect("a q1_0 weight on the host beside resident nvfp4 keeps the ring", beside, ring_weights(w),
               { 2, true, true, 0, false });
    }

    // No ring weight, a device with no ring published, and an empty plan: the empty plan places no expert on a device,
    // so every one is on the host.
    {
        const std::vector<weight_spec>        w   = inkling_weights(1);
        const placement_plan                  b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        const std::vector<moe_pp_ring_weight> none;
        expect("no ring weight asks nothing", b70, none, { 0, false, false, 0, false });
        expect_flag("a device with no ring published displaces none", moe_pp_ring_resident_check{}.displaces_experts(),
                    false);
        expect("an empty plan has every expert on the host", placement_plan{}, ring_weights(w),
               { w.size(), true, true, static_cast<int64_t>(w.size()) * k_experts, true });
        const moe_pp_ring_resident_check null_list = ggml_sycl::moe_pp_ring_check_resident_only(b70, nullptr, 4);
        if (null_list.n_weights != 0 || null_list.resident_only || null_list.displaces_experts()) {
            std::printf("FAIL: a null weight list was asked about\n");
            ++failures;
        }
    }

    // The RUNTIME figures the load stage and the plan dry run read first (moe_pp_ring_runtime_with_ring,
    // moe_pp_ring_runtime_room): the zone with the ring is no more than the arena's RUNTIME zone where an arena is
    // active, and an unanswerable figure is never read as room.
    {
        using ggml_sycl::moe_pp_ring_runtime_room;
        using ggml_sycl::moe_pp_ring_runtime_with_ring;
        const size_t with  = k_inkling_b50_runtime_with;
        const size_t floor = k_runtime_floor;
        expect_bytes("with an arena, the planned zone below the arena's",
                     moe_pp_ring_runtime_with_ring(with, true, with + k_mib, floor), with);
        expect_bytes("with an arena, no more than the arena's RUNTIME zone",
                     moe_pp_ring_runtime_with_ring(with + k_mib, true, with, floor), with);
        expect_bytes("with an arena, an unanswerable planned zone reads as the arena's",
                     moe_pp_ring_runtime_with_ring(SIZE_MAX, true, with, floor), with);
        expect_bytes("with no arena, the planned zone", moe_pp_ring_runtime_with_ring(with, false, 0, floor), with);
        expect_bytes("with no arena, an unanswerable planned zone reads as the floor the ensure keeps",
                     moe_pp_ring_runtime_with_ring(SIZE_MAX, false, 0, floor), floor);

        expect_bytes("inkling b50: the room the ring takes",
                     moe_pp_ring_runtime_room(with, k_inkling_b50_runtime_without),
                     with - k_inkling_b50_runtime_without);
        expect_bytes("no less without the ring: no room", moe_pp_ring_runtime_room(floor, floor), 0);
        expect_bytes("more without the ring: no room", moe_pp_ring_runtime_room(floor, with), 0);
        expect_bytes("unanswerable without the ring: no room", moe_pp_ring_runtime_room(with, SIZE_MAX), 0);
    }

    // (d) in the load stage (stage_decide), on the B50 runs' figures: the WEIGHT room is the planner's, and the arena
    // is asked only where a drop can return room. A held arena is asked before the room: the ensure keeps it as built,
    // so its WEIGHT zone reads as no room, and the keep is the held one.
    {
        using D = moe_pp_ring_decision;
        const size_t inkling_room =
            ggml_sycl::moe_pp_ring_runtime_room(k_inkling_b50_runtime_with, k_inkling_b50_runtime_without);
        moe_pp_ring_drop_marks marks;
        bool                   asked = false;

        expect_decision("inkling b50: the give-back grows the WEIGHT zone by the RUNTIME room, so dropped",
                        stage_decide(marks, 1, 3, k_b50_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_without, false }, &asked),
                        D::DROP);
        expect_flag("inkling b50: the arena was asked", asked, true);
        expect_flag("inkling b50: the drop marks the device for its load", marks.dropped(1, 3), true);
        expect_bytes("inkling b50: the WEIGHT zone took the whole RUNTIME room",
                     k_inkling_b50_weight_without - k_inkling_b50_weight_with, inkling_room);

        expect_decision("the give-back's ensure kept the arena as built (another zone would grow): kept",
                        stage_decide(marks, 1, 4, k_b50_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_with, false }),
                        D::KEEP_NO_ROOM_RETURNS);
        expect_decision("the give-back's rebuild fell back to two chunks: the WEIGHT zone shrinks, so kept",
                        stage_decide(marks, 1, 5, k_b50_budget, k_b50_weight_one_chunk, inkling_room,
                                     { k_b50_weight_two_chunks + inkling_room, false }),
                        D::KEEP_NO_ROOM_RETURNS);
        expect_decision("held, although the WEIGHT zone would read room: held, asked first",
                        stage_decide(marks, 1, 6, k_b50_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_without, true }),
                        D::KEEP_HELD);
        expect_decision("held with no room: held",
                        stage_decide(marks, 1, 7, k_b50_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_with, true }),
                        D::KEEP_HELD);
        expect_decision("one arena grain more WEIGHT zone: dropped",
                        stage_decide(marks, 1, 8, k_b50_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_with + 256, false }),
                        D::DROP);

        // The planner's figure, not the zone: a plan packed to a budget below the WEIGHT zone reads the budget before
        // and after, so a growing zone returns it nothing.
        const size_t small_budget = k_inkling_b50_weight_with - k_mib;
        expect_decision("a budget below the WEIGHT zone: the zone grows, the planner's room does not, so kept",
                        stage_decide(marks, 1, 9, small_budget, k_inkling_b50_weight_with, inkling_room,
                                     { k_inkling_b50_weight_without, false }, &asked),
                        D::KEEP_NO_ROOM_RETURNS);
        expect_flag("a budget below the WEIGHT zone: the arena was asked", asked, true);

        // Where the planner reads no WEIGHT room (no active arena, or a multi-device plan in GLOBAL cache mode: the
        // backend reads 0, ggml_sycl_ring_plan_weight_room) or the ring takes no RUNTIME room, the arena is not asked,
        // whatever it would reply.
        expect_decision("no WEIGHT room read: the arena is not asked, so kept",
                        stage_decide(marks, 1, 10, k_b50_budget, k_no_weight_zone, inkling_room,
                                     { k_inkling_b50_weight_without, false }, &asked),
                        D::KEEP_NO_ROOM_RETURNS);
        expect_flag("no WEIGHT room read: the arena was not asked", asked, false);
        expect_decision("no RUNTIME room: the arena is not asked, so kept",
                        stage_decide(marks, 1, 11, k_b50_budget, k_inkling_b50_weight_with, 0,
                                     { k_inkling_b50_weight_without, false }, &asked),
                        D::KEEP_NO_ROOM_RETURNS);
        expect_flag("no RUNTIME room: the arena was not asked", asked, false);
        for (uint64_t txn = 4; txn <= 11; ++txn) {
            const bool dropped = txn == 8;
            if (marks.dropped(1, txn) != dropped) {
                std::printf("FAIL: load %" PRIu64 "'s mark is %d, its decision %s\n", txn,
                            marks.dropped(1, txn) ? 1 : 0, dropped ? "a drop" : "a keep");
                ++failures;
            }
        }
    }

    // Qwen3.6 NVFP4 on the B50, one load's two passes through the stage's functions. The early pass reads a RUNTIME
    // zone another consumer sizes, as large without the ring: no room, so the arena is not asked, the ring is kept and
    // the keep leaves no mark. The late pass's zone holds the ring's room: dropped, and the drop holds for the load.
    // The dry run's answer for the late pass, the WEIGHT zone taking the whole RUNTIME room, is what the stage
    // measured.
    {
        using ggml_sycl::moe_pp_ring_runtime_room;
        using ggml_sycl::moe_pp_ring_runtime_with_ring;
        constexpr int          k_device = 1;
        constexpr uint64_t     k_load   = 9;
        moe_pp_ring_drop_marks marks;
        bool                   asked = true;

        const size_t early_room = moe_pp_ring_runtime_room(
            moe_pp_ring_runtime_with_ring(k_qwen_b50_runtime_early, true, k_qwen_b50_runtime_early, k_runtime_floor),
            k_qwen_b50_runtime_without);
        expect_bytes("qwen3.6 b50, the early pass: no RUNTIME room", early_room, 0);
        expect_decision("qwen3.6 b50, the early pass: kept",
                        stage_decide(marks, k_device, k_load, k_b50_budget, k_qwen_b50_weight_early, early_room,
                                     { k_qwen_b50_weight_late_without, false }, &asked),
                        moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS);
        expect_flag("qwen3.6 b50, the early pass: the arena was not asked", asked, false);
        expect_flag("qwen3.6 b50, the early keep leaves no mark", marks.dropped(k_device, k_load), false);

        const size_t late_room =
            moe_pp_ring_runtime_room(moe_pp_ring_runtime_with_ring(k_qwen_b50_runtime_late_with, true,
                                                                   k_qwen_b50_runtime_late_with, k_runtime_floor),
                                     k_qwen_b50_runtime_without);
        expect_bytes("qwen3.6 b50, the late pass: the RUNTIME room", late_room,
                     k_qwen_b50_runtime_late_with - k_qwen_b50_runtime_without);
        expect_decision("qwen3.6 b50, the late pass: dropped",
                        stage_decide(marks, k_device, k_load, k_b50_budget, k_qwen_b50_weight_late_with, late_room,
                                     { k_qwen_b50_weight_late_without, false }),
                        moe_pp_ring_decision::DROP);
        expect_flag("qwen3.6 b50, the late drop holds for the load", marks.dropped(k_device, k_load), true);
        expect_flag("qwen3.6 b50, the late drop is the load's alone", marks.dropped(k_device, k_load + 1), false);
        expect_bytes("qwen3.6 b50, the late pass: the WEIGHT zone took the RUNTIME room once, as the dry run answers",
                     planner_weight_room(k_b50_budget, k_qwen_b50_weight_late_without) -
                         planner_weight_room(k_b50_budget, k_qwen_b50_weight_late_with),
                     ggml_sycl::moe_pp_ring_dry_run_answer(true, late_room).weight_without_ring);
    }

    // (d) in the plan dry run (dry_run_decide): the same question as the stage's, on the WEIGHT room it read, then the
    // RUNTIME room alone, as a give-back nothing points into on an arena whose layout it keeps.
    {
        using D = moe_pp_ring_decision;
        using ggml_sycl::moe_pp_ring_runtime_room;
        using ggml_sycl::moe_pp_ring_runtime_with_ring;
        const size_t weight_room = planner_weight_room(k_b50_budget, k_inkling_b50_weight_with);
        const size_t room        = moe_pp_ring_runtime_room(k_inkling_b50_runtime_with, k_inkling_b50_runtime_without);
        expect_decision("dry run, an active arena and the ring's RUNTIME room: dropped",
                        dry_run_decide(weight_room, room), D::DROP);
        expect_decision("dry run, no RUNTIME room: kept", dry_run_decide(weight_room, 0), D::KEEP_NO_ROOM_RETURNS);
        expect_decision(
            "dry run, no active arena (its plan reads no WEIGHT zone) with the ring's RUNTIME room: kept, as a load "
            "that reads no WEIGHT room keeps it",
            dry_run_decide(0, room), D::KEEP_NO_ROOM_RETURNS);

        // An unanswerable zone with the ring, read uncapped, would read as room: with no arena it is the floor, the
        // floor without it gives no room, and the ring is kept.
        const size_t unanswerable = moe_pp_ring_runtime_room(
            moe_pp_ring_runtime_with_ring(SIZE_MAX, false, 0, k_runtime_floor), k_runtime_floor);
        expect_bytes("no arena, unanswerable with the ring, the floor without it: no room", unanswerable, 0);
        expect_decision("no arena, unanswerable with the ring: kept", dry_run_decide(0, unanswerable),
                        D::KEEP_NO_ROOM_RETURNS);

        const moe_pp_ring_zone_answer dry = ggml_sycl::moe_pp_ring_dry_run_answer(true, k_mib);
        if (dry.held || dry.weight_with_ring != 0 || dry.weight_without_ring != k_mib) {
            std::printf("FAIL: the dry run's answer must be unheld, counted from 0, the room its WEIGHT growth\n");
            ++failures;
        }
        const moe_pp_ring_zone_answer none = ggml_sycl::moe_pp_ring_dry_run_answer(false, k_mib);
        if (none.held || none.weight_with_ring != 0 || none.weight_without_ring != 0) {
            std::printf("FAIL: where a drop can return no room the dry run's answer must read none\n");
            ++failures;
        }
    }

    // The plan made with a ring kept after a give-back rebuild (moe_pp_ring_kept_plan_again). The ring is put back and
    // the zones are asked for again with it: an ordinary rebuild, whose layout is the reserve's to choose. Where it
    // lands in two chunks the WEIGHT zone is the KV share smaller than the one chunk that plan packed against, and that
    // plan, standing, would place more than the zone holds: it is made once more. The same room or more, and it stands.
    {
        const size_t packed = planner_weight_room(k_b50_budget, k_b50_weight_one_chunk);
        expect_plan_again("the second ensure lands in two chunks: less room than the plan packed against", packed,
                          planner_weight_room(k_b50_budget, k_b50_weight_two_chunks), true);
        expect_plan_again("the second ensure lands in one chunk again: the same room", packed, packed, false);
        expect_plan_again("more room than the plan packed against", packed, packed + k_mib, false);
        expect_plan_again("one byte less", packed, packed - 1, true);
        expect_plan_again("no WEIGHT room read either time", 0, 0, false);
    }

    // The experts' worth of room a held ring keeps from its weights (moe_pp_ring_experts_worth): the room over the ring
    // weights' mean expert size, rounded down, a weight with no ring use skipped.
    {
        const std::vector<weight_spec>        w     = inkling_weights(2);
        const std::vector<moe_pp_ring_weight> r     = ring_weights(w);
        const auto                            worth = [](const std::vector<moe_pp_ring_weight> & weights, size_t room) {
            return ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), room);
        };
        const auto expect_worth = [](const char * what, int64_t got, int64_t want) {
            std::printf("%s: %" PRId64 " (want %" PRId64 ")\n", what, got, want);
            if (got != want) {
                std::printf("FAIL: %s\n", what);
                ++failures;
            }
        };
        expect_worth("three experts' room holds three experts", worth(r, 3 * k_bytes), 3);
        expect_worth("a room short of one expert holds none", worth(r, k_bytes - 1), 0);
        expect_worth("no room holds none", worth(r, 0), 0);
        std::vector<moe_pp_ring_weight> mixed = r;
        mixed[0].bytes                        = 3 * k_bytes * k_experts;  // one weight's experts three times the size
        expect_worth("one weight three times the size: a mean of 1.5 k_bytes, so 2 k_bytes hold one expert",
                     worth(mixed, 2 * k_bytes), 1);
        const std::vector<weight_spec> q1 = triplet_weights(GGML_TYPE_Q1_0, 1);
        expect_worth("weights with no ring use hold none", worth(ring_weights(q1), 3 * k_bytes), 0);
        expect_worth("a null list holds none", ggml_sycl::moe_pp_ring_experts_worth(nullptr, 4, 3 * k_bytes), 0);
    }

    // The per-device marks a load's later passes read (moe_pp_ring_drop_marks). A load's pass records its decision with
    // its transaction: a drop marks the device, a keep marks nothing. A direct planner entry (transaction 0) marks
    // nothing and never erases a load's mark. One load is open at a time (the lifecycle registry refuses a second
    // begin), so a later load's mark replaces a finished one's.
    {
        using D = moe_pp_ring_decision;
        moe_pp_ring_drop_marks marks;
        expect_flag("no mark before a drop", marks.dropped(0, 7), false);
        marks.record(0, 7, D::KEEP_NO_ROOM_RETURNS);
        expect_flag("a keep for want of room leaves no mark", marks.dropped(0, 7), false);
        marks.record(0, 7, D::KEEP_HELD);
        expect_flag("a held keep leaves no mark", marks.dropped(0, 7), false);
        marks.record(0, 7, D::DROP);
        expect_flag("load 7 sees its own drop on device 0", marks.dropped(0, 7), true);
        marks.record(0, 7, D::KEEP_NO_ROOM_RETURNS);
        expect_flag("a later keep of the same load leaves its drop", marks.dropped(0, 7), true);
        expect_flag("load 7's mark is on device 0 only", marks.dropped(1, 7), false);
        expect_flag("another load does not see load 7's mark", marks.dropped(0, 8), false);
        expect_flag("a direct entry never sees a mark", marks.dropped(0, 0), false);
        marks.record(0, 0, D::DROP);
        expect_flag("a direct entry's drop leaves load 7's mark", marks.dropped(0, 7), true);
        marks.record(-1, 7, D::DROP);
        marks.record(GGML_SYCL_MAX_DEVICES, 7, D::DROP);
        expect_flag("a device out of range is never marked", marks.dropped(GGML_SYCL_MAX_DEVICES, 7), false);
        expect_flag("a negative device is never marked", marks.dropped(-1, 7), false);
        marks.record(0, 8, D::DROP);
        expect_flag("a later load's drop replaces finished load 7's mark", marks.dropped(0, 7), false);
        expect_flag("the later load sees its own mark", marks.dropped(0, 8), true);
    }

    if (failures != 0) {
        std::printf("test-sycl-moe-ring-consumer: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("test-sycl-moe-ring-consumer: OK\n");
    return 0;
}
