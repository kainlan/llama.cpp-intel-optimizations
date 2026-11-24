// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it (moe-ring-consumer.hpp).
//
// A ring whose weights reach the batched executor's resident claim and nothing else (moe_pp_ring_use: NVFP4 today) is
// kept only while the plan made with it leaves none of their experts on the host. A ring with a weight that reaches
// the staged claim (Qwen's IQ4_NL) or the per-expert SOA staging (GPT-OSS's MXFP4) is not asked about: it serves host
// experts, and the load keeps it as before. The plans below are shaped as the three models that plan a ring: Inkling 2
// Mini (NVFP4, a fused gate_up and a down per layer), Qwen (IQ4_NL, separate gate, up and down) and GPT-OSS (MXFP4).
// Each case names the answer it pins and the placement that decides it.
//
// Host-only: synthetic plans, no queue, no device, no model.

#include "ggml-sycl.h"
#include "ggml-sycl/moe-ring-consumer.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::moe_pp_ring_resident_check;
using ggml_sycl::moe_pp_ring_weight;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_plan;

constexpr int    kExperts = 4;
constexpr size_t kBytes   = 4096;
constexpr int    kHost    = -1;

int failures = 0;

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

// A plan of `weights`, each expert placed by `spot(weight index, expert)`, indexed as the stage indexes its plan.
placement_plan make_plan(const std::vector<weight_spec> &                weights,
                         const std::function<expert_spot(size_t, int)> & spot) {
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
            entry.layer_id      = weights[w].layer;
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

// The ring weights of `weights`, as the stage lists them from its inventory: each with its expert count and the uses
// its type reaches. `extra_experts` asks about experts the plan has no entry for.
std::vector<moe_pp_ring_weight> ring_weights(const std::vector<weight_spec> & weights, int extra_experts = 0) {
    std::vector<moe_pp_ring_weight> out;
    for (const weight_spec & w : weights) {
        moe_pp_ring_weight r;
        r.name      = w.name.c_str();
        r.n_experts = kExperts + extra_experts;
        r.uses      = moe_pp_ring_uses_of_type(w.type);
        out.push_back(r);
    }
    return out;
}

// One case: the check's three fields and the decision it gives.
void expect(const char *                            what,
            const placement_plan &                  plan,
            const std::vector<moe_pp_ring_weight> & weights,
            size_t                                  want_weights,
            bool                                    want_resident_only,
            int64_t                                 want_host,
            bool                                    want_drop) {
    const moe_pp_ring_resident_check c =
        ggml_sycl::moe_pp_ring_check_resident_only(plan, weights.data(), weights.size());
    std::printf("%s: weights=%zu resident_only=%d host_experts=%" PRId64 " drop=%d (want %zu %d %" PRId64 " %d)\n",
                what, c.n_weights, c.resident_only ? 1 : 0, c.host_experts, c.displaces_experts() ? 1 : 0, want_weights,
                want_resident_only ? 1 : 0, want_host, want_drop ? 1 : 0);
    if (c.n_weights != want_weights || c.resident_only != want_resident_only || c.host_experts != want_host ||
        c.displaces_experts() != want_drop) {
        std::printf("FAIL: %s\n", what);
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

}  // namespace

int main() {
    // Inkling: NVFP4 reaches the resident claim only, so the ring is kept only with no NVFP4 expert on the host.
    {
        const std::vector<weight_spec> w = inkling_weights(2);
        const auto                     r = ring_weights(w);

        // The B70: every expert on device 0. No expert on the host, so the ring is kept.
        const placement_plan b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        expect("inkling all on device 0 keeps the ring", b70, r, w.size(), true, 0, false);

        // One expert of one weight on the host: the ring displaced it, so it is dropped.
        const placement_plan one_host =
            make_plan(w, [](size_t i, int e) { return i == 3 && e == 2 ? on_host() : on_device(0); });
        expect("inkling one expert on the host drops the ring", one_host, r, w.size(), true, 1, true);

        // Every expert on the device but in two layouts: none is on the host, so the ring is kept although the claim
        // takes one layout. The decision asks the host, not the claim.
        const placement_plan two_layouts =
            make_plan(w, [](size_t, int e) { return on_device(0, e == 1 ? GGML_LAYOUT_SOA : GGML_LAYOUT_AOS); });
        expect("inkling on device 0 in two layouts keeps the ring", two_layouts, r, w.size(), true, 0, false);

        // Experts split across the two cards, none on the host: the ring is kept on each.
        const placement_plan split = make_plan(w, [](size_t, int e) { return on_device(e % 2); });
        expect("inkling split across devices keeps the ring", split, r, w.size(), true, 0, false);

        // Experts the plan has no entry for are not on the host.
        expect("inkling asked for experts the plan lacks keeps the ring", b70, ring_weights(w, 1), w.size(), true, 0,
               false);

        // Every expert on the host: each is counted.
        const placement_plan all_host = make_plan(w, [](size_t, int) { return on_host(); });
        expect("inkling all on the host drops the ring", all_host, r, w.size(), true,
               static_cast<int64_t>(w.size()) * kExperts, true);
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
        expect("inkling 14 layers resident and the rest on the host drops the ring", b50, ring_weights(w), w.size(),
               true, host, true);
    }

    // Qwen: IQ4_NL reaches the staged claim too, so its ring is not asked about however many experts are on the host.
    {
        const std::vector<weight_spec> w = triplet_weights(GGML_TYPE_IQ4_NL, 2);
        const auto                     r = ring_weights(w);

        const placement_plan tiered = make_plan(w, [](size_t, int e) { return e == 0 ? on_device(0) : on_host(); });
        expect("qwen host-tiered keeps the ring", tiered, r, w.size(), false, 0, false);

        const placement_plan all_host = make_plan(w, [](size_t, int) { return on_host(); });
        expect("qwen all on the host keeps the ring", all_host, r, w.size(), false, 0, false);
    }

    // GPT-OSS: MXFP4 reaches the per-expert SOA staging too, so its ring is not asked about either.
    {
        const std::vector<weight_spec> w = triplet_weights(GGML_TYPE_MXFP4, 2);
        const auto                     r = ring_weights(w);

        const placement_plan hybrid = make_plan(
            w, [](size_t, int e) { return e == 0 ? on_device(0, GGML_LAYOUT_SOA) : on_host(GGML_LAYOUT_SOA); });
        expect("gpt-oss hybrid keeps the ring", hybrid, r, w.size(), false, 0, false);

        const placement_plan resident = make_plan(w, [](size_t, int) { return on_device(0, GGML_LAYOUT_XMX_TILED); });
        expect("gpt-oss all on device 0 keeps the ring", resident, r, w.size(), false, 0, false);
    }

    // A ring with one staged weight beside resident-only ones keeps today's behaviour: the staged weight's host experts
    // are what the ring serves.
    {
        std::vector<weight_spec> w = inkling_weights(1);
        w.push_back({ "blk.1.ffn_down_exps.weight", GGML_TYPE_IQ4_NL, expert_tensor_role::DOWN, 1 });
        const placement_plan mixed = make_plan(w, [](size_t, int) { return on_host(); });
        expect("a staged weight beside nvfp4 ones keeps the ring", mixed, ring_weights(w), w.size(), false, 0, false);
    }

    // A weight with no ring use is not a ring weight: it is skipped, and a list of only such weights asks nothing.
    {
        const std::vector<weight_spec> q1      = triplet_weights(GGML_TYPE_Q1_0, 1);
        const placement_plan           q1_host = make_plan(q1, [](size_t, int) { return on_host(); });
        expect("q1_0 weights alone ask nothing", q1_host, ring_weights(q1), 0, false, 0, false);

        std::vector<weight_spec> w = inkling_weights(1);
        w.push_back({ "blk.1.ffn_down_exps.weight", GGML_TYPE_Q1_0, expert_tensor_role::DOWN, 1 });
        const placement_plan beside =
            make_plan(w, [&](size_t i, int) { return w[i].type == GGML_TYPE_Q1_0 ? on_host() : on_device(0); });
        expect("a q1_0 weight on the host beside resident nvfp4 keeps the ring", beside, ring_weights(w), 2, true, 0,
               false);
    }

    // No ring weight, and an empty plan.
    {
        const std::vector<weight_spec>        w   = inkling_weights(1);
        const placement_plan                  b70 = make_plan(w, [](size_t, int) { return on_device(0); });
        const std::vector<moe_pp_ring_weight> none;
        expect("no ring weight asks nothing", b70, none, 0, false, 0, false);
        expect("an empty plan has no expert on the host", placement_plan{}, ring_weights(w), w.size(), true, 0, false);
        const moe_pp_ring_resident_check null_list = ggml_sycl::moe_pp_ring_check_resident_only(b70, nullptr, 4);
        if (null_list.n_weights != 0 || null_list.resident_only || null_list.displaces_experts()) {
            std::printf("FAIL: a null weight list was asked about\n");
            ++failures;
        }
    }

    if (failures != 0) {
        std::printf("test-sycl-moe-ring-consumer: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("test-sycl-moe-ring-consumer: OK\n");
    return 0;
}
