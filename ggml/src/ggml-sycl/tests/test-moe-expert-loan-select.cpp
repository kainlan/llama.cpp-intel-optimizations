// llama.cpp-36j0 Task 11a (llama.cpp-0zmf): loan selection is a pure function of the plan. Host-only, three hand-built
// plans, no queue, no device, no model:
//   make_plan:       3 layers, 6 experts, experts 0-3 on device 0 and 4-5 on the host, except layer 1 whose tensors sit
//                    entirely on device 0 (bulk-staged at load, so not lendable) and layer 0 expert 3 whose device layout
//                    is SOA of Q4_0 (no inverse).
//   make_split_plan: one layer where a single role differs (SOA up, host down, down on another device) and experts live
//                    on both devices, so the per-role loop, the same-device rule and the device filter are discriminated.
//   make_alt_plan:   alternate layouts: the alternate-only bulk branch, a same-device alternate, one expert, and two
//                    alternates resolving to one device.

#include "ggml-sycl/moe-stream-plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_placement_key;
using ggml_sycl::expert_tensor_role;
using ggml_sycl::placement_alternate_layout;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;
using ggml_sycl::placement_vram_charge_bytes;

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr int    kLayers  = 3;
constexpr int    kExperts = 6;
constexpr size_t kDst[3]  = { 65536, 16384,
                              32768 };  // the gate's charge is the largest: it alone reaches a small target

const expert_tensor_role kRoles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
const char *             kNames[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };

placement_entry make_entry(int layer, int expert, int role_idx, bool on_dev, int target) {
    placement_entry pe("blk." + std::to_string(layer) + "." + kNames[role_idx] + ".weight", 4096, kDst[role_idx], 0,
                       placement_priority::MOE_UP, layer, expert, kRoles[role_idx], on_dev, target);
    pe.type = GGML_TYPE_Q4_0;
    return pe;
}

placement_alternate_layout make_alt(int role_idx, int target) {
    placement_alternate_layout alt;
    alt.layout           = GGML_LAYOUT_AOS;
    alt.target_device    = target;
    alt.vram_charge_size = placement_vram_charge_bytes(kDst[role_idx]);
    return alt;
}

bool on_device(int layer, int expert) {
    return layer == 1 || expert < 4;
}

placement_plan make_plan() {
    placement_plan plan{};
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            for (int r = 0; r < 3; ++r) {
                const bool      dev = on_device(l, e);
                placement_entry pe  = make_entry(l, e, r, dev, dev ? 0 : -1);
                if (l == 0 && e == 3) {
                    pe.layout = GGML_LAYOUT_SOA;
                }
                plan.entries.push_back(pe);
            }
        }
    }
    plan.build_index();
    return plan;
}

// Layer 0. Expert 0: device 0. Expert 1: gate and up on device 0, down on device 1. Experts 2, 3: device 1.
// Expert 4: only UP is SOA. Expert 5: only DOWN is on the host. No tensor is bulk on either device.
placement_plan make_split_plan() {
    placement_plan plan{};
    for (int e = 0; e < 6; ++e) {
        for (int r = 0; r < 3; ++r) {
            int  target = e == 2 || e == 3 ? 1 : 0;
            bool dev    = true;
            if (e == 1 && r == 2) {
                target = 1;
            }
            if (e == 5 && r == 2) {
                dev    = false;
                target = -1;
            }
            placement_entry pe = make_entry(0, e, r, dev, target);
            if (e == 4 && r == 1) {
                pe.layout = GGML_LAYOUT_SOA;
            }
            plan.entries.push_back(pe);
        }
    }
    plan.build_index();
    return plan;
}

// Queried for device 0. Layer 0: experts 0-3 primary on device 0, experts 4-5 primary on device 1 with one alternate
// targeted at device 0, so load stages the tensor as ONE bulk allocation on device 0 (the alt-only branch). Layer 1: the
// same, plus expert 1 carries an alternate on its own device 0, which makes the tensor non-bulk. Layer 2: a single
// expert, so n_experts > 1 fails. Layer 3: the layer 0 shape, but expert 4 carries two alternates resolving to device
// 0, which is neither of the two staging branches, so the tensor is non-bulk.
placement_plan make_alt_plan() {
    placement_plan plan{};
    for (int l = 0; l < 4; ++l) {
        const int n = l == 2 ? 1 : kExperts;
        for (int e = 0; e < n; ++e) {
            for (int r = 0; r < 3; ++r) {
                const int       dev = e < 4 ? 0 : 1;
                placement_entry pe  = make_entry(l, e, r, true, dev);
                if (dev == 1 || (l == 1 && e == 1)) {
                    pe.alternate_layouts.push_back(make_alt(r, 0));
                }
                if (l == 3 && e == 4) {
                    pe.alternate_layouts.push_back(make_alt(r, 0));
                }
                plan.entries.push_back(pe);
            }
        }
    }
    plan.build_index();
    return plan;
}

size_t triplet_charge() {
    size_t s = 0;
    for (int r = 0; r < 3; ++r) {
        s += placement_vram_charge_bytes(kDst[r]);
    }
    return s;
}

bool key_is(const expert_placement_key & k, int layer, int expert, expert_tensor_role role) {
    return k.layer_id == layer && k.expert_id == expert && k.role == role;
}

bool has_expert(const std::vector<expert_placement_key> & keys, int expert) {
    for (const expert_placement_key & k : keys) {
        if (k.expert_id == expert) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    const placement_plan plan = make_plan();
    const size_t         tc   = triplet_charge();
    const size_t         huge = ~static_cast<size_t>(0);

    // Lendable predicate: a plain device triplet, then each skip.
    CHECK(ggml_sycl::moe_expert_triplet_lendable(plan, 2, 3));
    CHECK(ggml_sycl::moe_expert_triplet_lendable(plan, 0, 2));
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(plan, 1, 0));   // bulk: every expert of the tensor on device 0
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(plan, 1, 5));   // bulk, also for the high expert ids
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(plan, 0, 3));   // SOA layout, Q4_0 has no inverse
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(plan, 2, 4));   // host
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(plan, 2, 99));  // not in the plan

    // Everything lendable: layer 2 experts 3,2,1,0 then layer 0 experts 2,1,0, last entries first.
    size_t     sel       = 0;
    const auto all       = ggml_sycl::moe_expert_loan_select(plan, 0, huge, &sel);
    const int  want_l[7] = { 2, 2, 2, 2, 0, 0, 0 };
    const int  want_e[7] = { 3, 2, 1, 0, 2, 1, 0 };
    CHECK(all.size() == 21);
    for (size_t i = 0; i < all.size() && i < 21; ++i) {
        CHECK(key_is(all[i], want_l[i / 3], want_e[i / 3], kRoles[i % 3]));
    }
    CHECK(sel == 7 * tc);  // and below bytes_wanted: the lendable set was too small

    // Whole-triplet granularity: the gate alone reaches 1 byte and also its own charge, yet all three keys come back.
    for (size_t wanted : { size_t(1), placement_vram_charge_bytes(kDst[0]), tc }) {
        const auto one = ggml_sycl::moe_expert_loan_select(plan, 0, wanted, &sel);
        CHECK(one.size() == 3);
        CHECK(sel == tc);
        if (one.size() == 3) {
            CHECK(key_is(one[0], 2, 3, expert_tensor_role::GATE));
            CHECK(key_is(one[1], 2, 3, expert_tensor_role::UP));
            CHECK(key_is(one[2], 2, 3, expert_tensor_role::DOWN));
        }
    }

    // One byte past a triplet takes the next whole triplet.
    const auto two = ggml_sycl::moe_expert_loan_select(plan, 0, tc + 1, &sel);
    CHECK(two.size() == 6 && sel == 2 * tc);
    if (two.size() == 6) {
        CHECK(key_is(two[3], 2, 2, expert_tensor_role::GATE));
    }

    // Reaching the target stops the walk: 5 triplets cover 5 * tc exactly, 6 are not taken.
    const auto five = ggml_sycl::moe_expert_loan_select(plan, 0, 5 * tc, &sel);
    CHECK(five.size() == 15 && sel == 5 * tc);
    if (five.size() == 15) {
        CHECK(key_is(five[12], 0, 2, expert_tensor_role::GATE));  // fifth triplet: layer 0 expert 2; expert 3 skipped
    }

    // Nothing wanted, nothing on that device.
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, 0, &sel).empty() && sel == 0);
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 1, huge, &sel).empty() && sel == 0);
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, 1, nullptr).size() == 3);  // bytes_selected is optional

    // One role differs: every role is checked, not just the gate. Device 0 holds expert 0 whole; device 1 holds
    // experts 2 and 3 whole. Expert 1 is split across devices, expert 4 has a SOA up, expert 5 has a host down.
    const placement_plan split = make_split_plan();
    CHECK(ggml_sycl::moe_expert_triplet_lendable(split, 0, 0));
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(split, 0, 4));
    CHECK(!ggml_sycl::moe_expert_triplet_lendable(split, 0, 5));
    const auto split0 = ggml_sycl::moe_expert_loan_select(split, 0, huge, &sel);
    CHECK(split0.size() == 3 && sel == tc);
    CHECK(!split0.empty() && split0[0].expert_id == 0);
    CHECK(!has_expert(split0, 1) && !has_expert(split0, 4) && !has_expert(split0, 5));
    const auto split1 = ggml_sycl::moe_expert_loan_select(split, 1, huge, &sel);
    CHECK(split1.size() == 6 && sel == 2 * tc);
    if (split1.size() == 6) {
        CHECK(key_is(split1[0], 0, 3, expert_tensor_role::GATE));
        CHECK(key_is(split1[3], 0, 2, expert_tensor_role::GATE));
    }
    CHECK(!has_expert(split1, 1));  // its down entry is on device 1, but gate and up are not

    // Alternate-only bulk branch: experts 4-5 stage on device 0 through an alternate, so the whole tensor is one bulk
    // allocation there and the primary experts 0-3 are views into it.
    const placement_plan alt = make_alt_plan();
    for (int e = 0; e < 4; ++e) {
        CHECK(!ggml_sycl::moe_expert_triplet_lendable(alt, 0, e));
    }
    // A same-device alternate on one expert makes the tensor non-bulk: its experts are lendable.
    for (int e = 0; e < 4; ++e) {
        CHECK(ggml_sycl::moe_expert_triplet_lendable(alt, 1, e));
    }
    // One expert only: n_experts > 1 fails, so not bulk.
    CHECK(ggml_sycl::moe_expert_triplet_lendable(alt, 2, 0));
    // Two alternates resolving to device 0 on one non-primary expert: not bulk either.
    for (int e = 0; e < 4; ++e) {
        CHECK(ggml_sycl::moe_expert_triplet_lendable(alt, 3, e));
    }
    // Select on device 0: layer 3 experts 3..0, layer 2 expert 0, layer 1 experts 3..0; layer 0 is bulk and skipped.
    const auto alt_sel = ggml_sycl::moe_expert_loan_select(alt, 0, huge, &sel);
    CHECK(alt_sel.size() == 27 && sel == 9 * tc);
    if (alt_sel.size() == 27) {
        CHECK(key_is(alt_sel[0], 3, 3, expert_tensor_role::GATE));
        CHECK(key_is(alt_sel[12], 2, 0, expert_tensor_role::GATE));
        CHECK(key_is(alt_sel[15], 1, 3, expert_tensor_role::GATE));
        CHECK(key_is(alt_sel[24], 1, 0, expert_tensor_role::GATE));
    }

    if (failures) {
        std::fprintf(stderr, "test-moe-expert-loan-select: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-expert-loan-select: OK\n");
    return 0;
}
