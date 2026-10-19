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
    plan.build_index();
    return plan;
}

}  // namespace

int main() {
    const placement_plan plan      = make_plan();
    const int32_t        no_ids[1] = { 0 };

    // STREAM_ALL: 3 triplets in entries order, gate/up/down within each; ids ignored.
    const int32_t ignored[2] = { 0, 2 };
    const auto    all        = ggml_sycl::moe_stream_plan_build(plan, 0, 1, moe_exec_regime::STREAM_ALL, ignored, 2);
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
    const auto    part      = ggml_sycl::moe_stream_plan_build(plan, 0, 1, moe_exec_regime::ROUTED_ONLY, routed, 4);
    CHECK(part.size() == 6);
    for (size_t i = 0; i < part.size() && i < 6; ++i) {
        CHECK(part[i].expert_id == (i < 3 ? 1 : 5));
        CHECK(part[i].role == exp_r[i % 3]);
    }

    // A layer with no host experts lists nothing under either regime.
    CHECK(ggml_sycl::moe_stream_plan_build(plan, 0, 3, moe_exec_regime::STREAM_ALL, no_ids, 1).empty());
    CHECK(ggml_sycl::moe_stream_plan_build(plan, 0, 3, moe_exec_regime::ROUTED_ONLY, routed, 4).empty());
    CHECK(ggml_sycl::moe_stream_plan_build(plan, 0, 1, moe_exec_regime::NONE, routed, 4).empty());

    // fully_staged: true for the STREAM_ALL list, false once one item is removed.
    CHECK(ggml_sycl::moe_stream_layer_fully_staged(all, plan, 0, 1, routed, 4));
    CHECK(ggml_sycl::moe_stream_layer_fully_staged(all, plan, 0, 1, ignored, 2));         // 0 and 2 are resident
    std::vector<moe_stream_item> short_list = all;
    short_list.erase(short_list.begin() + 4);                                             // expert 5's up
    CHECK(!ggml_sycl::moe_stream_layer_fully_staged(short_list, plan, 0, 1, routed, 4));
    CHECK(ggml_sycl::moe_stream_layer_fully_staged(short_list, plan, 0, 1, ignored, 2));  // 5 is not asked for
    CHECK(!ggml_sycl::moe_stream_layer_fully_staged(std::vector<moe_stream_item>{}, plan, 0, 1, routed, 4));
    CHECK(ggml_sycl::moe_stream_layer_fully_staged(std::vector<moe_stream_item>{}, plan, 0, 3, routed, 4));

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-plan: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-plan: OK\n");
    return 0;
}
