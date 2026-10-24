// llama.cpp-36j0 Task 7a: the layer stream list is a pure function of the plan, the regime and the ids.
// Host-only: a hand-built 4-layer, 8-expert plan with 3 host experts per layer (1, 5, 6), no queue, no device.

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
        item.dst_bytes = e.dst_size;
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
        CHECK(all[i].dst_bytes == kDst);
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
    // experts). The hand-built entries have a zeroed type and shape, so the production staging is NONE; only the list
    // is compared. The comparison fails on a dropped role, a reordered expert, or a fused GATE_UP host entry listed
    // as a triplet (layer 1 would then have 10 items, not 9).
    const ggml_sycl::moe_stream_plan_index idx =
        ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, nullptr, nullptr);
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

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-plan: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-plan: OK\n");
    return 0;
}
