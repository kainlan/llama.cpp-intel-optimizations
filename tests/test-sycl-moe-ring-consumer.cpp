// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it (moe-ring-consumer.hpp).
//
// The rule is by use. A ring is dropped only when every weight that uses it reaches the batched executor's resident
// claim alone, every one keeps a device route for a resident prompt node without the ring (the MMID direct route), the
// plan made with the ring leaves at least one expert of those weights on the host, and the drop returns room to the
// weights: the RUNTIME zone needs less without the ring, and the arena may be rebuilt. Each type class has a case that
// names its types: the resident-only types with the direct route (the drop set: NVFP4 today), the resident-only types
// without it (IQ1_S, IQ1_M and IQ4_XS today, whose only device prompt route is the ring, so it is kept), a staged type
// (Qwen's IQ4_NL), the per-expert SOA staging (GPT-OSS's MXFP4) and a type with no ring use (Q1_0). The model-shaped
// plans are Inkling 2 Mini (NVFP4, a fused gate_up and a down per layer), Qwen (IQ4_NL, separate gate, up and down) and
// GPT-OSS (MXFP4), each decided with a zone that returns room and an arena that may be rebuilt, so a case that says it
// drops the ring is decided by its plan. An expert the plan does not place on a device is on the host, by the plan's
// own convention (placement_plan::expert_on_device), whether it has a host entry, a name-keyed fallback entry on the
// host or no entry. A section then names every outcome of the decision (moe_pp_ring_decide), the two the zone decides
// with the figures measured on the B50; the last pins the per-device marks a load's later passes read. Each case names
// the answer it pins and the placement that decides it.
//
// Host-only: synthetic plans, no queue, no device, no model.

#include "ggml-sycl.h"
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

constexpr int    kExperts = 4;
constexpr size_t kBytes   = 4096;
constexpr int    kHost    = -1;
constexpr size_t kMiB     = 1024 * 1024;

// The RUNTIME zone figures of the two B50 runs, rounded to the MiB. Inkling 2 Mini: the ring took the zone to 558.6 MB
// and the zone was rebuilt at its 512 MB floor without it. Qwen3.6 NVFP4: another consumer sizes the zone, 726.4 MB with
// the ring and without it.
constexpr moe_pp_ring_zone_answer kInklingB50Zone = { 559 * kMiB, 512 * kMiB, true };
constexpr moe_pp_ring_zone_answer kQwenB50Zone    = { 726 * kMiB, 726 * kMiB, true };

int failures = 0;

const char * decision_name(moe_pp_ring_decision d) {
    switch (d) {
        case moe_pp_ring_decision::KEEP_NO_RING_WEIGHT:
            return "keep: no ring weight";
        case moe_pp_ring_decision::KEEP_STAGED_OR_SOA_USE:
            return "keep: a staged or SOA use";
        case moe_pp_ring_decision::KEEP_NO_OTHER_DEVICE_ROUTE:
            return "keep: a weight with no other device route";
        case moe_pp_ring_decision::KEEP_NO_HOST_EXPERT:
            return "keep: no expert on the host";
        case moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS:
            return "keep: no room returns";
        case moe_pp_ring_decision::KEEP_HELD:
            return "keep: the arena is held";
        case moe_pp_ring_decision::DROP:
            return "drop";
    }
    return "unknown";
}

// Where one expert is planned: a device id, or kHost; and its layout on the device.
struct expert_spot {
    int              device = kHost;
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
        for (int e = 0; e < kExperts; ++e) {
            const expert_spot s = spot(w, e);
            placement_entry   entry;
            entry.name          = weights[w].name;
            entry.type          = weights[w].type;
            entry.src_size      = kBytes;
            entry.dst_size      = kBytes;
            entry.layer_id      = classified ? weights[w].layer : -1;
            entry.expert_id     = e;
            entry.expert_role   = weights[w].role;
            entry.layout        = s.layout;
            entry.on_device     = s.device != kHost;
            entry.target_device = s.device;
            plan.entries.push_back(entry);
        }
    }
    plan.build_index();
    return plan;
}

// The ring weights of `weights`, as the stage lists them from its inventory: each with its expert count, its bytes
// (kBytes per expert), the uses its type reaches and whether the MMID direct route runs its type. `extra_experts` asks
// about experts the plan has no entry for.
std::vector<moe_pp_ring_weight> ring_weights(const std::vector<weight_spec> & weights, int extra_experts = 0) {
    std::vector<moe_pp_ring_weight> out;
    for (const weight_spec & w : weights) {
        moe_pp_ring_weight r;
        r.name                      = w.name.c_str();
        r.n_experts                 = kExperts + extra_experts;
        r.bytes                     = kBytes * static_cast<size_t>(kExperts + extra_experts);
        r.uses                      = moe_pp_ring_uses_of_type(w.type);
        r.other_device_prompt_route = moe_mmid_direct_route_runs_type(w.type);
        out.push_back(r);
    }
    return out;
}

// What a case wants of the check: its four fields and the decision with a zone that returns room and an arena that may
// be rebuilt (kInklingB50Zone), where the plan alone decides.
struct want_check {
    size_t  weights;
    bool    resident_only;
    bool    other_route;
    int64_t host;
    bool    drop;
};

void expect(const std::string &                     what,
            const placement_plan &                  plan,
            const std::vector<moe_pp_ring_weight> & weights,
            const want_check &                      want) {
    const moe_pp_ring_resident_check c =
        ggml_sycl::moe_pp_ring_check_resident_only(plan, weights.data(), weights.size());
    const moe_pp_ring_decision d = ggml_sycl::moe_pp_ring_decide(c, kInklingB50Zone);
    std::printf("%s: weights=%zu resident_only=%d other_route=%d host_experts=%" PRId64
                " displaces=%d decision=%s (want %zu %d %d %" PRId64 " %s)\n",
                what.c_str(), c.n_weights, c.resident_only ? 1 : 0, c.other_device_prompt_route ? 1 : 0, c.host_experts,
                c.displaces_experts() ? 1 : 0, decision_name(d), want.weights, want.resident_only ? 1 : 0,
                want.other_route ? 1 : 0, want.host, want.drop ? "drop" : "keep");
    if (c.n_weights != want.weights || c.resident_only != want.resident_only ||
        c.other_device_prompt_route != want.other_route || c.host_experts != want.host ||
        c.displaces_experts() != want.drop || (d == moe_pp_ring_decision::DROP) != want.drop) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The decision for `check` under `zone`, against the outcome it names.
void expect_decision(const std::string &                what,
                     const moe_pp_ring_resident_check & check,
                     const moe_pp_ring_zone_answer &    zone,
                     moe_pp_ring_decision               want) {
    const moe_pp_ring_decision got = ggml_sycl::moe_pp_ring_decide(check, zone);
    std::printf("%s: %s (want %s)\n", what.c_str(), decision_name(got), decision_name(want));
    if (got != want) {
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
    return expert_spot{ kHost, layout };
}

void expect_mark(const char * what, bool got, bool want) {
    std::printf("%s: %d (want %d)\n", what, got ? 1 : 0, want ? 1 : 0);
    if (got != want) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    // Inkling: NVFP4 reaches the resident claim alone and the MMID direct route runs it, so the ring is kept only with
    // no NVFP4 expert on the host.
    {
        const std::vector<weight_spec> w = inkling_weights(2);
        const auto                     r = ring_weights(w);
        const size_t                   n = w.size();

        // The B70: every expert on device 0. No expert on the host, so the ring is kept.
        const placement_plan b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        expect("inkling all on device 0 keeps the ring", b70, r, { n, true, true, 0, false });

        // One expert of one weight on the host: the ring displaced it, so it is dropped.
        const placement_plan one_host =
            make_plan(w, [](size_t i, int e) { return i == 3 && e == 2 ? on_host() : on_device(0); });
        expect("inkling one expert on the host drops the ring", one_host, r, { n, true, true, 1, true });

        // Every expert on the device but in two layouts: none is on the host, so the ring is kept although the claim
        // takes one layout. The decision asks the host, not the claim.
        const placement_plan two_layouts =
            make_plan(w, [](size_t, int e) { return on_device(0, e == 1 ? GGML_LAYOUT_SOA : GGML_LAYOUT_AOS); });
        expect("inkling on device 0 in two layouts keeps the ring", two_layouts, r, { n, true, true, 0, false });

        // Experts split across the two cards, none on the host: the ring is kept on each.
        const placement_plan split = make_plan(w, [](size_t, int e) { return on_device(e % 2); });
        expect("inkling split across devices keeps the ring", split, r, { n, true, true, 0, false });

        // An expert the plan has no entry for is on the host: the plan never places a missing expert on a device.
        expect("inkling asked for experts the plan lacks drops the ring", b70, ring_weights(w, 1),
               { n, true, true, static_cast<int64_t>(n), true });

        // Experts the semantic index misses are read through their name-keyed fallback entries: on the host they count,
        // on the device they do not.
        const placement_plan fallback_host =
            make_plan(w, [](size_t i, int) { return i == 0 ? on_host() : on_device(0); }, /*classified=*/false);
        expect("inkling fallback entries on the host drop the ring", fallback_host, r,
               { n, true, true, kExperts, true });
        const placement_plan fallback_device = make_plan(w, [](size_t, int) { return on_device(0); }, false);
        expect("inkling fallback entries on device 0 keep the ring", fallback_device, r, { n, true, true, 0, false });

        // Every expert on the host: each is counted.
        const placement_plan all_host = make_plan(w, [](size_t, int) { return on_host(); });
        expect("inkling all on the host drops the ring", all_host, r,
               { n, true, true, static_cast<int64_t>(n) * kExperts, true });
    }

    // Inkling on the B50: 14 layers resident and the rest on the host. The ring serves the 14 layers and still loses,
    // so it is dropped; the count is every expert of the other layers.
    {
        constexpr int                  kInklingLayers  = 24;
        constexpr int                  kResidentLayers = 14;
        const std::vector<weight_spec> w               = inkling_weights(kInklingLayers);
        const placement_plan           b50 =
            make_plan(w, [&](size_t i, int) { return w[i].layer < kResidentLayers ? on_device(0) : on_host(); });
        const int64_t host = static_cast<int64_t>(kInklingLayers - kResidentLayers) * 2 * kExperts;
        expect("inkling 14 layers resident and the rest on the host drops the ring", b50, ring_weights(w),
               { w.size(), true, true, host, true });
    }

    // The drop set by type: a resident-only type the MMID direct route runs (NVFP4 today) drops the ring when the plan
    // leaves one of its experts on the host, in the separate gate, up and down shape too (Qwen3.6 NVFP4).
    for (const ggml_type type : { GGML_TYPE_NVFP4 }) {
        const std::vector<weight_spec> w        = triplet_weights(type, 1);
        const std::string              name     = ggml_type_name(type);
        const placement_plan           resident = make_plan(w, [](size_t, int) { return on_device(0); });
        expect(name + " all on device 0 keeps the ring", resident, ring_weights(w), { w.size(), true, true, 0, false });
        const placement_plan one_host =
            make_plan(w, [](size_t i, int e) { return i == 2 && e == 0 ? on_host() : on_device(0); });
        expect(name + " one expert on the host drops the ring: its resident nodes keep the MMID direct route", one_host,
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

    // No ring weight, and an empty plan: the empty plan places no expert on a device, so every one is on the host.
    {
        const std::vector<weight_spec>        w   = inkling_weights(1);
        const placement_plan                  b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        const std::vector<moe_pp_ring_weight> none;
        expect("no ring weight asks nothing", b70, none, { 0, false, false, 0, false });
        expect("an empty plan has every expert on the host", placement_plan{}, ring_weights(w),
               { w.size(), true, true, static_cast<int64_t>(w.size()) * kExperts, true });
        const moe_pp_ring_resident_check null_list = ggml_sycl::moe_pp_ring_check_resident_only(b70, nullptr, 4);
        if (null_list.n_weights != 0 || null_list.resident_only || null_list.displaces_experts()) {
            std::printf("FAIL: a null weight list was asked about\n");
            ++failures;
        }
    }

    // Every outcome of the decision by name (moe_pp_ring_decide). The plan answers the first four and the zone the last
    // three: the drop is carried out only when it returns room to the weights, so a ring that does not enlarge the
    // RUNTIME zone is kept, and so is one whose arena may not be rebuilt in this load. Room is asked before the arena:
    // where none would return, nothing needs rebuilding.
    {
        using D             = moe_pp_ring_decision;
        const auto check_of = [](const placement_plan & plan, const std::vector<moe_pp_ring_weight> & weights) {
            return ggml_sycl::moe_pp_ring_check_resident_only(plan, weights.data(), weights.size());
        };
        const std::vector<weight_spec>   w = inkling_weights(2);
        const auto                       r = ring_weights(w);
        const moe_pp_ring_resident_check tiered =
            check_of(make_plan(w, [&](size_t i, int) { return w[i].layer == 0 ? on_device(0) : on_host(); }), r);

        expect_decision(
            "nvfp4 host-tiered with the inkling b50 zone: rebuilt at its floor without the ring, so dropped", tiered,
            kInklingB50Zone, D::DROP);
        moe_pp_ring_zone_answer one_grain = kInklingB50Zone;
        one_grain.runtime_without_ring    = one_grain.runtime_with_ring - 256;
        expect_decision("nvfp4 host-tiered with a zone one arena grain smaller without the ring: dropped", tiered,
                        one_grain, D::DROP);
        expect_decision("nvfp4 host-tiered with the qwen3.6 b50 zone, sized by another consumer: no room returns",
                        tiered, kQwenB50Zone, D::KEEP_NO_ROOM_RETURNS);
        moe_pp_ring_zone_answer larger = kInklingB50Zone;
        larger.runtime_without_ring    = larger.runtime_with_ring + kMiB;
        expect_decision("nvfp4 host-tiered with a zone that needs more without the ring: no room returns", tiered,
                        larger, D::KEEP_NO_ROOM_RETURNS);
        expect_decision("nvfp4 host-tiered on a device with no arena: no zone, so no room returns", tiered,
                        moe_pp_ring_zone_answer{}, D::KEEP_NO_ROOM_RETURNS);
        moe_pp_ring_zone_answer held = kInklingB50Zone;
        held.rebuild_allowed         = false;
        expect_decision("nvfp4 host-tiered with live allocations in the arena: held, the room cannot return", tiered,
                        held, D::KEEP_HELD);
        moe_pp_ring_zone_answer no_room_held = kQwenB50Zone;
        no_room_held.rebuild_allowed         = false;
        expect_decision("nvfp4 host-tiered with no room and a held arena: no room returns, asked first", tiered,
                        no_room_held, D::KEEP_NO_ROOM_RETURNS);

        expect_decision("nvfp4 all on device 0: no expert on the host, whatever the zone",
                        check_of(make_plan(w, [](size_t, int) { return on_device(0); }), r), kInklingB50Zone,
                        D::KEEP_NO_HOST_EXPERT);

        const std::vector<weight_spec> iq4_xs = triplet_weights(GGML_TYPE_IQ4_XS, 2);
        expect_decision(
            "iq4_xs host-tiered: the ring is its only device prompt route",
            check_of(make_plan(iq4_xs, [&](size_t i, int) { return iq4_xs[i].layer == 0 ? on_device(0) : on_host(); }),
                     ring_weights(iq4_xs)),
            kInklingB50Zone, D::KEEP_NO_OTHER_DEVICE_ROUTE);

        const std::vector<weight_spec> qwen = triplet_weights(GGML_TYPE_IQ4_NL, 2);
        expect_decision("iq4_nl (qwen) host-tiered: a staged use",
                        check_of(make_plan(qwen, [](size_t, int e) { return e == 0 ? on_device(0) : on_host(); }),
                                 ring_weights(qwen)),
                        kInklingB50Zone, D::KEEP_STAGED_OR_SOA_USE);
        const std::vector<weight_spec> gpt_oss = triplet_weights(GGML_TYPE_MXFP4, 2);
        expect_decision("mxfp4 (gpt-oss) hybrid: an SOA per-expert staging use",
                        check_of(make_plan(gpt_oss,
                                           [](size_t, int e) {
                                               return e == 0 ? on_device(0, GGML_LAYOUT_SOA) : on_host(GGML_LAYOUT_SOA);
                                           }),
                                 ring_weights(gpt_oss)),
                        kInklingB50Zone, D::KEEP_STAGED_OR_SOA_USE);

        const std::vector<weight_spec> q1 = triplet_weights(GGML_TYPE_Q1_0, 1);
        expect_decision("q1_0 weights alone: no ring weight",
                        check_of(make_plan(q1, [](size_t, int) { return on_host(); }), ring_weights(q1)),
                        kInklingB50Zone, D::KEEP_NO_RING_WEIGHT);
        expect_decision("a device with no ring published: no ring weight", moe_pp_ring_resident_check{},
                        kInklingB50Zone, D::KEEP_NO_RING_WEIGHT);
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
        expect_worth("three experts' room holds three experts", worth(r, 3 * kBytes), 3);
        expect_worth("a room short of one expert holds none", worth(r, kBytes - 1), 0);
        expect_worth("no room holds none", worth(r, 0), 0);
        std::vector<moe_pp_ring_weight> mixed = r;
        mixed[0].bytes                        = 3 * kBytes * kExperts;  // one weight's experts three times the size
        expect_worth("one weight three times the size: a mean of 1.5 kBytes, so 2 kBytes hold one expert",
                     worth(mixed, 2 * kBytes), 1);
        const std::vector<weight_spec> q1 = triplet_weights(GGML_TYPE_Q1_0, 1);
        expect_worth("weights with no ring use hold none", worth(ring_weights(q1), 3 * kBytes), 0);
        expect_worth("a null list holds none", ggml_sycl::moe_pp_ring_experts_worth(nullptr, 4, 3 * kBytes), 0);
    }

    // The per-device marks a load's later passes read (moe_pp_ring_drop_marks). A load's pass marks a device with its
    // transaction; a direct planner entry (transaction 0) marks nothing and never erases a load's mark.
    {
        moe_pp_ring_drop_marks marks;
        expect_mark("no mark before a drop", marks.dropped(0, 7), false);
        marks.mark(0, 7);
        expect_mark("load 7 sees its own mark on device 0", marks.dropped(0, 7), true);
        expect_mark("load 7's mark is on device 0 only", marks.dropped(1, 7), false);
        expect_mark("another load does not see load 7's mark", marks.dropped(0, 8), false);
        expect_mark("a direct entry never sees a mark", marks.dropped(0, 0), false);
        marks.mark(0, 0);
        expect_mark("a direct entry's drop leaves load 7's mark", marks.dropped(0, 7), true);
        marks.mark(-1, 7);
        marks.mark(GGML_SYCL_MAX_DEVICES, 7);
        expect_mark("a device out of range is never marked", marks.dropped(GGML_SYCL_MAX_DEVICES, 7), false);
        expect_mark("a negative device is never marked", marks.dropped(-1, 7), false);
        marks.mark(0, 8);
        expect_mark("an interleaved load's drop overwrites the slot", marks.dropped(0, 7), false);
        expect_mark("the interleaved load sees its own mark", marks.dropped(0, 8), true);
    }

    if (failures != 0) {
        std::printf("test-sycl-moe-ring-consumer: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("test-sycl-moe-ring-consumer: OK\n");
    return 0;
}
