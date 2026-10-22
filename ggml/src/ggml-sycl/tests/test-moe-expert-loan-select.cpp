// llama.cpp-36j0 Task 11a (llama.cpp-0zmf): loan selection is a pure function of the plan. Host-only: a hand-built
// 3-layer, 6-expert plan, experts 0-3 on device 0 and 4-5 on the host, except layer 1 whose tensors sit entirely on
// device 0 (bulk-staged at load, so not lendable) and layer 0 expert 3 whose device layout is SOA of Q4_0 (no inverse).

#include "ggml-sycl/moe-stream-plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ggml_sycl::expert_placement_key;
using ggml_sycl::expert_tensor_role;
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

constexpr int    kLayers  = 3;
constexpr int    kExperts = 6;
constexpr size_t kDst[3]  = { 65536, 16384, 32768 };  // gate is the largest: it alone can reach a small target

bool on_device(int layer, int expert) {
    return layer == 1 || expert < 4;
}

placement_plan make_plan() {
    placement_plan           plan{};
    const expert_tensor_role roles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
    const char *             names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            for (int r = 0; r < 3; ++r) {
                const bool      dev = on_device(l, e);
                placement_entry pe("blk." + std::to_string(l) + "." + names[r] + ".weight", 4096, kDst[r], 0,
                                   placement_priority::MOE_UP, l, e, roles[r], dev, dev ? 0 : -1);
                pe.type = GGML_TYPE_Q4_0;
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

// Alternate-layout plan, queried for device 0. Layer 0: experts 0-3 primary on device 0, experts 4-5 primary on device 1
// with one alternate targeted at device 0, so load stages the tensor as ONE bulk allocation on device 0 (the alt-only
// branch). Layer 1: the same, plus expert 1 carries an alternate on its own device 0, which makes the tensor non-bulk.
// Layer 2: a single expert, so n_experts > 1 fails and the tensor is non-bulk.
placement_plan make_alt_plan() {
    placement_plan           plan{};
    const expert_tensor_role roles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
    const char *             names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (int l = 0; l < 3; ++l) {
        const int n = l == 2 ? 1 : kExperts;
        for (int e = 0; e < n; ++e) {
            for (int r = 0; r < 3; ++r) {
                const int       dev = e < 4 ? 0 : 1;
                placement_entry pe("blk." + std::to_string(l) + "." + names[r] + ".weight", 4096, kDst[r], 0,
                                   placement_priority::MOE_UP, l, e, roles[r], true, dev);
                pe.type = GGML_TYPE_Q4_0;
                if (dev == 1 || (l == 1 && e == 1)) {
                    ggml_sycl::placement_alternate_layout alt;
                    alt.layout           = GGML_LAYOUT_AOS;
                    alt.target_device    = 0;
                    alt.vram_charge_size = ggml_sycl::placement_vram_charge_bytes(kDst[r]);
                    pe.alternate_layouts.push_back(alt);
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
        s += ggml_sycl::placement_vram_charge_bytes(kDst[r]);
    }
    return s;
}

bool key_is(const expert_placement_key & k, int layer, int expert, expert_tensor_role role) {
    return k.layer_id == layer && k.expert_id == expert && k.role == role;
}

}  // namespace

int main() {
    const placement_plan plan = make_plan();
    const size_t         tc   = triplet_charge();

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
    const auto all       = ggml_sycl::moe_expert_loan_select(plan, 0, ~static_cast<size_t>(0), &sel);
    const int  want_l[7] = { 2, 2, 2, 2, 0, 0, 0 };
    const int  want_e[7] = { 3, 2, 1, 0, 2, 1, 0 };
    CHECK(all.size() == 21);
    for (size_t i = 0; i < all.size() && i < 21; ++i) {
        const expert_tensor_role role = i % 3 == 0 ? expert_tensor_role::GATE :
                                        i % 3 == 1 ? expert_tensor_role::UP :
                                                     expert_tensor_role::DOWN;
        CHECK(key_is(all[i], want_l[i / 3], want_e[i / 3], role));
    }
    CHECK(sel == 7 * tc);
    CHECK(sel < ~static_cast<size_t>(0));  // too small a lendable set: bytes_selected < bytes_wanted

    // Whole-triplet granularity: the gate alone reaches 1 byte and also its own charge, yet all three keys come back.
    for (size_t wanted : { size_t(1), ggml_sycl::placement_vram_charge_bytes(kDst[0]), tc }) {
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

    // Bulk and layout skips never appear, whatever the target.
    for (const expert_placement_key & k : all) {
        CHECK(k.layer_id != 1);
        CHECK(!(k.layer_id == 0 && k.expert_id == 3));
        CHECK(k.expert_id < 4);
    }

    // Nothing wanted, nothing on that device.
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, 0, &sel).empty() && sel == 0);
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 1, ~static_cast<size_t>(0), &sel).empty() && sel == 0);
    CHECK(ggml_sycl::moe_expert_loan_select(plan, 0, 1, nullptr).size() == 3);  // bytes_selected is optional

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
    // Select on device 0: layer 2, then layer 1 experts 3..0; layer 0 is bulk and skipped.
    const auto alt_sel = ggml_sycl::moe_expert_loan_select(alt, 0, ~static_cast<size_t>(0), &sel);
    CHECK(alt_sel.size() == 15 && sel == 5 * tc);
    if (alt_sel.size() == 15) {
        CHECK(key_is(alt_sel[0], 2, 0, expert_tensor_role::GATE));
        CHECK(key_is(alt_sel[3], 1, 3, expert_tensor_role::GATE));
        CHECK(key_is(alt_sel[12], 1, 0, expert_tensor_role::GATE));
    }

    if (failures) {
        std::fprintf(stderr, "test-moe-expert-loan-select: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-expert-loan-select: OK\n");
    return 0;
}
