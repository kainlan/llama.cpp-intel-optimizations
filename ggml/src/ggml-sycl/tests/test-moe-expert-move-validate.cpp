// The plan transaction's named moves are checked against the published plan before anything is minted or moved: a
// pure function of the plan. Host-only, one hand-built plan, no queue, no device, no model. Queried for device 0.
//   layer 0: experts 0-1 on device 0, experts 2-3 on the host, expert 4 on device 1, expert 5 on the host with a SOA
//            down entry, expert 6 with no down entry, expert 7 on device 0 with a SOA up entry (Q4_0: no inverse).
//   layer 1: experts 0-1 on device 0, experts 2-3 on the host, expert 4 split: its gate on device 0, up and down on
//            the host.

#include "ggml-sycl/moe-stream-plan.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::moe_expert_move_check;
using ggml_sycl::moe_expert_triplet_ref;
using ggml_sycl::placement_entry;
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

const expert_tensor_role kRoles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
const char *             kNames[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };

placement_entry make_entry(int layer, int expert, int role_idx, int target) {
    placement_entry pe("blk." + std::to_string(layer) + "." + kNames[role_idx] + ".weight", 4096, 4096, 0,
                       placement_priority::MOE_UP, layer, expert, kRoles[role_idx], target >= 0, target);
    pe.type = GGML_TYPE_Q4_0;
    return pe;
}

placement_plan make_plan() {
    placement_plan plan{};
    for (int e = 0; e < 8; ++e) {
        for (int r = 0; r < 3; ++r) {
            if (e == 6 && r == 2) {
                continue;
            }
            const int       target = e == 4 ? 1 : (e == 2 || e == 3 || e == 5 || e == 6) ? -1 : 0;
            placement_entry pe     = make_entry(0, e, r, target);
            if ((e == 5 && r == 2) || (e == 7 && r == 1)) {
                pe.layout = GGML_LAYOUT_SOA;
            }
            plan.entries.push_back(pe);
        }
    }
    for (int e = 0; e < 5; ++e) {
        for (int r = 0; r < 3; ++r) {
            const int target = e < 2 ? 0 : e < 4 ? -1 : (r == 0 ? 0 : -1);
            plan.entries.push_back(make_entry(1, e, r, target));
        }
    }
    plan.build_index();
    return plan;
}

moe_expert_triplet_ref tr(int layer, int expert) {
    moe_expert_triplet_ref t;
    t.layer_id  = layer;
    t.expert_id = expert;
    return t;
}

bool reason_is(const char * got, const char * want) {
    if (!got || !want) {
        return got == want;
    }
    return std::strcmp(got, want) == 0;
}

moe_expert_move_check validate(const placement_plan &                      plan,
                               const std::vector<moe_expert_triplet_ref> & evict,
                               const std::vector<moe_expert_triplet_ref> & admit) {
    return ggml_sycl::moe_expert_move_validate(plan, /*device=*/0, evict, admit);
}

// One answer per named triplet, in each list's order, and a movable triplet's answer is null.
void test_movable_triplets_pass() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(0, 0), tr(1, 1) }, { tr(0, 2), tr(0, 3), tr(1, 2) });
    CHECK(c.evict.size() == 2 && c.admit.size() == 3);
    for (const char * r : c.evict) {
        CHECK(r == nullptr);
    }
    for (const char * r : c.admit) {
        CHECK(r == nullptr);
    }
    const moe_expert_move_check none = validate(plan, {}, {});
    CHECK(none.evict.empty() && none.admit.empty() && none.net_by_layer.empty());
}

// A triplet with a member the plan does not have, or named with an id no entry carries, moves no member.
void test_incomplete_triplet_refused() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c =
        validate(plan, { tr(0, 6), tr(-1, 0) }, { tr(0, 6), tr(0, -1), tr(7, 0), tr(0, 99) });
    CHECK(c.evict.size() == 2 && c.admit.size() == 4);
    if (c.evict.size() == 2 && c.admit.size() == 4) {
        CHECK(reason_is(c.evict[0], "incomplete triplet"));
        CHECK(reason_is(c.evict[1], "incomplete triplet"));
        CHECK(reason_is(c.admit[0], "duplicate triplet"));  // named in the evict list first
        CHECK(reason_is(c.admit[1], "incomplete triplet"));
        CHECK(reason_is(c.admit[2], "incomplete triplet"));
        CHECK(reason_is(c.admit[3], "incomplete triplet"));
    }
    const moe_expert_move_check admit_only = validate(plan, {}, { tr(0, 6) });
    CHECK(admit_only.admit.size() == 1 && reason_is(admit_only.admit[0], "incomplete triplet"));
}

// An admit takes a triplet whose every member is on the host; an evict one whose every member is on the device. A
// split triplet is neither.
void test_wrong_side_refused() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(0, 2), tr(1, 4) }, { tr(0, 0), tr(1, 0) });
    CHECK(c.evict.size() == 2 && c.admit.size() == 2);
    if (c.evict.size() == 2 && c.admit.size() == 2) {
        CHECK(reason_is(c.evict[0], "not on device"));
        CHECK(reason_is(c.evict[1], "not on device"));
        CHECK(reason_is(c.admit[0], "not on host"));
        CHECK(reason_is(c.admit[1], "not on host"));
    }
    const moe_expert_move_check split = validate(plan, {}, { tr(1, 4) });
    CHECK(split.admit.size() == 1 && reason_is(split.admit[0], "not on host"));
}

// A member placed on another device belongs to that device's transaction, in either list.
void test_wrong_device_refused() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(0, 4) }, {});
    CHECK(c.evict.size() == 1 && reason_is(c.evict[0], "wrong device"));
    const moe_expert_move_check a = validate(plan, {}, { tr(0, 4) });
    CHECK(a.admit.size() == 1 && reason_is(a.admit[0], "wrong device"));
    // The same triplet is movable on its own device.
    const moe_expert_move_check own = ggml_sycl::moe_expert_move_validate(plan, 1, { tr(0, 4) }, {});
    CHECK(own.evict.size() == 1 && own.evict[0] == nullptr);
}

// A SOA member refuses the triplet: an admit stages AOS copies only, and a SOA evict needs an inverse of the load's
// reorder that does not exist for Q4_0.
void test_soa_member_refused() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(0, 7) }, { tr(0, 5) });
    CHECK(c.evict.size() == 1 && c.admit.size() == 1);
    if (c.evict.size() == 1 && c.admit.size() == 1) {
        CHECK(reason_is(c.evict[0], "layout"));
        CHECK(reason_is(c.admit[0], "layout"));
    }
}

// A triplet named twice moves once: the first naming is judged, every later one is refused, across both lists with
// the evicts first.
void test_duplicate_keys_refused() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(0, 0), tr(0, 0) }, { tr(0, 2), tr(0, 2), tr(0, 0) });
    CHECK(c.evict.size() == 2 && c.admit.size() == 3);
    if (c.evict.size() == 2 && c.admit.size() == 3) {
        CHECK(c.evict[0] == nullptr);
        CHECK(reason_is(c.evict[1], "duplicate triplet"));
        CHECK(c.admit[0] == nullptr);
        CHECK(reason_is(c.admit[1], "duplicate triplet"));
        CHECK(reason_is(c.admit[2], "duplicate triplet"));
    }
    // A refused first naming still claims the triplet.
    const moe_expert_move_check refused = validate(plan, {}, { tr(0, 0), tr(0, 0) });
    CHECK(refused.admit.size() == 2);
    if (refused.admit.size() == 2) {
        CHECK(reason_is(refused.admit[0], "not on host"));
        CHECK(reason_is(refused.admit[1], "duplicate triplet"));
    }
}

// Per block, the admits less the evicts the transaction may move, sorted by layer whatever the lists' order; a
// refused triplet counts in no block, and a block whose moves cancel reads 0.
void test_net_count_per_block() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c = validate(plan, { tr(1, 0), tr(1, 1), tr(0, 2) }, { tr(1, 2), tr(0, 3), tr(0, 2) });
    CHECK(c.evict.size() == 3 && c.admit.size() == 3);
    if (c.evict.size() == 3 && c.admit.size() == 3) {
        CHECK(reason_is(c.evict[2], "not on device"));
        CHECK(reason_is(c.admit[2], "duplicate triplet"));
    }
    const std::vector<std::pair<int, int>> want = { std::make_pair(0, 1), std::make_pair(1, -1) };
    CHECK(c.net_by_layer == want);
    const moe_expert_move_check            swap   = validate(plan, { tr(1, 0) }, { tr(1, 2) });
    const std::vector<std::pair<int, int>> cancel = { std::make_pair(1, 0) };
    CHECK(swap.net_by_layer == cancel);
    const moe_expert_move_check refused = validate(plan, { tr(0, 2) }, { tr(0, 0) });
    CHECK(refused.net_by_layer.empty());
}

// An evict-only call: the transaction reads each evict's answer at its naming's index, so an answer stays there
// whatever was refused before it, and only the movable triplet counts, its block reading -1 alone.
void test_evict_only_answers_in_order() {
    const placement_plan        plan = make_plan();
    const moe_expert_move_check c    = validate(plan, { tr(1, 2), tr(0, 4), tr(1, 0), tr(1, 0) }, {});
    CHECK(c.evict.size() == 4 && c.admit.empty());
    if (c.evict.size() == 4) {
        CHECK(reason_is(c.evict[0], "not on device"));
        CHECK(reason_is(c.evict[1], "wrong device"));
        CHECK(c.evict[2] == nullptr);
        CHECK(reason_is(c.evict[3], "duplicate triplet"));
    }
    const std::vector<std::pair<int, int>> want = { std::make_pair(1, -1) };
    CHECK(c.net_by_layer == want);
}

}  // namespace

int main() {
    test_movable_triplets_pass();
    test_incomplete_triplet_refused();
    test_wrong_side_refused();
    test_wrong_device_refused();
    test_soa_member_refused();
    test_duplicate_keys_refused();
    test_net_count_per_block();
    test_evict_only_answers_in_order();

    if (failures) {
        std::fprintf(stderr, "test-moe-expert-move-validate: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-expert-move-validate: OK\n");
    return 0;
}
