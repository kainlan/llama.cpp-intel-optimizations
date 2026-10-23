// llama.cpp-xod7 (plan Task 7b of llama.cpp-36j0): the prompt phase's pure parts, the ones moe_stream_phase consults
// before it touches a queue. Host-only: hand-built plans, no queue, no device, and no SYCL object is constructed.
//
// The plan: 5 layers x 8 experts, Q4_0 gate/up/down, resident on device 0 unless noted.
//   layers 0-2: experts 1, 5 and 6 on the host;
//   layer 1:    also a fused host GATE_UP entry for expert 1 (a shape the ring cannot stage);
//   layer 3:    no host expert (the phase skips it);
//   layer 4:    only a fused host GATE_UP for expert 2 (host experts present, no item).
//
// A second plan mixes in host experts of a type with no AOS->SOA reorder (IQ3_XXS). Whether the ring stages a matrix
// depends on the device's prompt kernels, which the tests pass as a predicate (the backend answers from
// ggml_sycl_moe_query_route_capability):
//   kernels_soa_q4: a stand-in no real table matches (no prompt kernel reads Q4_0 as SOA today; kernels_table has the
//                   tables' shape): Q4_0 is read as SOA and as AOS, IQ3_XXS in no layout (as IQ4_XS is on the real
//                   tables), so IQ3_XXS is listed (and counted untyped when staged) but given no slot;
//   kernels_aos:    both types are read as AOS only, as the batched MMVQ tables answer for Q4_0, the K types and
//                   IQ3_XXS, so each matrix takes one slot and is read in place (c-hmey);
//   kernels_table:  the device's prompt kernels as the tables answer today (moe-mmvq-tables.hpp): SOA for MXFP4 only,
//                   AOS for the rest, so a reorderable Q4_K stages AOS-only and MXFP4 stages SOA (lead ruling
//                   c-o9uw-2), whatever the capability table says.
//
// The ring regions' read gates (lead ruling c-o9uw-1) are kept by moe_stream_region_reads, which the phase holds over
// sycl::event; the test drives it over int, in the order stage(), layer_done() and reset_regions() call it.
//
// A matrix whose weight the batched prompt executor claims once its host experts are staged stages one AOS slot whatever
// the kernels answer, and its entry names the executor as its consumer (llama.cpp-3460): the executor dequantizes AOS
// itself, with a plain kernel on the compute queue, and the staged route refuses such an entry. The tests pass the
// claims as a second predicate; the backend answers with ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor.

#include "ggml-sycl/moe-stream-plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using ggml_sycl::alloc_metadata;
using ggml_sycl::alloc_role;
using ggml_sycl::alloc_tier;
using ggml_sycl::expert_tensor_role;
using ggml_sycl::host_zone_id;
using ggml_sycl::moe_exec_choice;
using ggml_sycl::moe_exec_decision;
using ggml_sycl::moe_exec_policy;
using ggml_sycl::moe_exec_rates;
using ggml_sycl::moe_exec_regime;
using ggml_sycl::moe_stream_consumer;
using ggml_sycl::moe_stream_item;
using ggml_sycl::moe_stream_layer_info;
using ggml_sycl::moe_stream_plan_index;
using ggml_sycl::moe_stream_staging;
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

constexpr int     kLayers  = 5;
constexpr int     kExperts = 8;
constexpr int64_t kCols    = 256;  // 8 Q4_0 blocks per row
constexpr int64_t kRows    = 16;
constexpr size_t  kFused   = 8192;

const expert_tensor_role kRoles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
const char *             kNames[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };

size_t src_bytes() {
    return ggml_row_size(GGML_TYPE_Q4_0, kCols) * static_cast<size_t>(kRows);
}

size_t dst_bytes() {
    return src_bytes() + 256;
}

bool triplet_on_host(int layer, int expert) {
    return layer <= 2 && (expert == 1 || expert == 5 || expert == 6);
}

placement_entry entry(int                layer,
                      int                expert,
                      expert_tensor_role role,
                      const char *       name,
                      bool               host,
                      int                device,
                      size_t             src,
                      size_t             dst) {
    placement_entry e("blk." + std::to_string(layer) + "." + name + ".weight", src, dst, 0, placement_priority::MOE_UP,
                      layer, expert, role, !host, host ? -1 : device);
    e.type  = GGML_TYPE_Q4_0;
    e.ne[0] = kCols;
    e.ne[1] = kRows;
    e.ne[2] = kExperts;
    return e;
}

placement_plan make_plan() {
    placement_plan plan{};
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            if (l == 4 && e == 2) {
                plan.entries.push_back(
                    entry(l, e, expert_tensor_role::GATE_UP, "ffn_gate_up_exps", true, 0, kFused, kFused));
                continue;
            }
            for (int r = 0; r < 3; ++r) {
                plan.entries.push_back(
                    entry(l, e, kRoles[r], kNames[r], triplet_on_host(l, e), 0, src_bytes(), dst_bytes()));
            }
        }
    }
    plan.entries.push_back(
        entry(1, 1, expert_tensor_role::GATE_UP, "ffn_gate_up_exps", true, 0, kFused, kFused));  // fused, layer 1
    // Never counted: a dense host tensor, and an expert resident on device 1.
    plan.entries.push_back(entry(1, -1, expert_tensor_role::UNKNOWN, "attn_q", true, 0, src_bytes(), dst_bytes()));
    plan.entries.push_back(entry(3, 0, expert_tensor_role::GATE, "ffn_gate_exps", false, 1, src_bytes(), 999));
    plan.build_index();
    return plan;
}

// Layer 0: experts 0 and 1 on the host as Q4_0 (streamable). Layer 1: experts 0-3 on the host as IQ3_XXS, which has no
// reorder, with a larger device size than any Q4_0 matrix. Experts 4-7 of both layers are resident Q4_0 on device 0.
// with_q4 = false drops layer 0's host experts, leaving only the IQ3_XXS ones.
placement_plan make_mixed_plan(bool with_q4) {
    const size_t   iq_src = ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * static_cast<size_t>(kRows);
    placement_plan plan{};
    for (int e = 0; e < kExperts; ++e) {
        for (int r = 0; r < 3; ++r) {
            const bool q4_host = with_q4 && e < 2;
            plan.entries.push_back(entry(0, e, kRoles[r], kNames[r], q4_host, 0, src_bytes(), dst_bytes()));
            if (e < 4) {
                placement_entry iq = entry(1, e, kRoles[r], kNames[r], true, 0, iq_src, 4 * dst_bytes());
                iq.type            = GGML_TYPE_IQ3_XXS;
                plan.entries.push_back(iq);
            } else {
                plan.entries.push_back(entry(1, e, kRoles[r], kNames[r], false, 0, src_bytes(), dst_bytes()));
            }
        }
    }
    plan.build_index();
    return plan;
}

// Layer 0 only: experts 0 and 1 on the host as Q4_0, experts 2 and 3 on the host as IQ3_XXS, the rest resident Q4_0.
placement_plan make_one_layer_mix_plan() {
    const size_t   iq_src = ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * static_cast<size_t>(kRows);
    placement_plan plan{};
    for (int e = 0; e < kExperts; ++e) {
        for (int r = 0; r < 3; ++r) {
            if (e == 2 || e == 3) {
                placement_entry iq = entry(0, e, kRoles[r], kNames[r], true, 0, iq_src, 4 * dst_bytes());
                iq.type            = GGML_TYPE_IQ3_XXS;
                plan.entries.push_back(iq);
            } else {
                plan.entries.push_back(entry(0, e, kRoles[r], kNames[r], e < 2, 0, src_bytes(), dst_bytes()));
            }
        }
    }
    plan.build_index();
    return plan;
}

size_t iq_src_bytes() {
    return ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * static_cast<size_t>(kRows);
}

bool kernels_soa_q4(void *, ggml_type type, ggml_layout_mode layout, int64_t, int64_t) {
    return type == GGML_TYPE_Q4_0 && (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_AOS);
}

bool kernels_aos(void *, ggml_type type, ggml_layout_mode layout, int64_t, int64_t) {
    return layout == GGML_LAYOUT_AOS && (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ3_XXS);
}

// kernels_soa_q4's answers, plus IQ3_XXS read as AOS: the one-layer mix stages Q4_0 as SOA and IQ3_XXS as AOS.
bool kernels_mixed(void *, ggml_type type, ggml_layout_mode layout, int64_t, int64_t) {
    return kernels_soa_q4(nullptr, type, layout, 0, 0) || (type == GGML_TYPE_IQ3_XXS && layout == GGML_LAYOUT_AOS);
}

// kernels_aos, counting its calls in *graph_ctx.
bool kernels_aos_counted(void * graph_ctx, ggml_type type, ggml_layout_mode layout, int64_t ne0, int64_t ne1) {
    ++*static_cast<int *>(graph_ctx);
    return kernels_aos(nullptr, type, layout, ne0, ne1);
}

// The tables' answers today: MXFP4 is the one type read as SOA, and every type here is read as AOS.
bool kernels_table(void *, ggml_type type, ggml_layout_mode layout, int64_t, int64_t) {
    if (layout == GGML_LAYOUT_SOA) {
        return type == GGML_TYPE_MXFP4;
    }
    return layout == GGML_LAYOUT_AOS &&
           (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_MXFP4 || type == GGML_TYPE_IQ3_XXS);
}

moe_stream_plan_index index_of(const placement_plan & plan, ggml_sycl::moe_stream_prompt_kernel_fn kernels) {
    return ggml_sycl::moe_stream_plan_index_build(plan, 0, kernels, nullptr, nullptr);
}

// The library executor's claims, handed to the index build through graph_ctx as the backend's call is.
struct claims_probe {
    int calls = 0;
};

// Claims layer 0's GATE weight and every weight of layer 2, counting its calls.
bool claims_l0_gate_l2(void * graph_ctx, int layer, expert_tensor_role role) {
    ++static_cast<claims_probe *>(graph_ctx)->calls;
    return (layer == 0 && role == expert_tensor_role::GATE) || layer == 2;
}

bool claims_all(void *, int, expert_tensor_role) {
    return true;
}

bool claims_none(void *, int, expert_tensor_role) {
    return false;
}

const moe_stream_layer_info * layer_info(const moe_stream_plan_index & idx, int layer) {
    const auto it = idx.layers.find(layer);
    return it == idx.layers.end() ? nullptr : &it->second;
}

std::unordered_set<uint64_t> keys_of(int layer, std::initializer_list<int> experts) {
    std::unordered_set<uint64_t> keys;
    for (int e : experts) {
        for (expert_tensor_role r : kRoles) {
            keys.insert(ggml_sycl::moe_stream_table_key(layer, e, r));
        }
    }
    return keys;
}

const char * fake_env_h2d(const char * name) {
    return std::strcmp(name, "GGML_SYCL_MOE_EXEC_H2D_GBPS") == 0 ? "31" : nullptr;
}

bool near(double a, double b) {
    return std::fabs(a - b) <= 1e-9 * std::fmax(1.0, std::fabs(b));
}

}  // namespace

int main() {
    // ---- the stage table's key: (layer << 32) | (expert << 8) | role ----
    const uint64_t k = ggml_sycl::moe_stream_table_key(1, 5, expert_tensor_role::UP);
    CHECK(k == ((1ull << 32) | (5ull << 8) | static_cast<uint64_t>(static_cast<uint8_t>(expert_tensor_role::UP))));
    CHECK(k != ggml_sycl::moe_stream_table_key(1, 5, expert_tensor_role::GATE));
    CHECK(k != ggml_sycl::moe_stream_table_key(1, 6, expert_tensor_role::UP));
    CHECK(k != ggml_sycl::moe_stream_table_key(2, 5, expert_tensor_role::UP));

    // ---- the ids marked once, sized n_expert; an id outside [0, n_expert) is ignored ----
    const int32_t              ids[6] = { 1, 1, 5, 7, -1, 99 };
    const std::vector<uint8_t> bits   = ggml_sycl::moe_stream_ids_bitmap(ids, 6, kExperts);
    CHECK(bits.size() == static_cast<size_t>(kExperts));
    for (int e = 0; e < kExperts && e < static_cast<int>(bits.size()); ++e) {
        CHECK(bits[e] == ((e == 1 || e == 5 || e == 7) ? 1 : 0));
    }
    CHECK(ggml_sycl::moe_stream_ids_bitmap(ids, 6, 0).empty());
    const std::vector<uint8_t> none = ggml_sycl::moe_stream_ids_bitmap(nullptr, 6, kExperts);
    CHECK(none.size() == static_cast<size_t>(kExperts) && none[1] == 0);

    // ---- the per-micro-batch index: one scan of the plan ----
    const placement_plan        plan = make_plan();
    const moe_stream_plan_index idx  = index_of(plan, kernels_soa_q4);
    CHECK(idx.layers.size() == 4);
    CHECK(layer_info(idx, 3) == nullptr);  // no host expert: skipped, never weighed

    const moe_stream_layer_info * l0 = layer_info(idx, 0);
    CHECK(l0 != nullptr);
    if (l0) {
        CHECK(l0->items.size() == 9);
        const int exp_e[3] = { 1, 5, 6 };
        for (size_t i = 0; i < l0->items.size() && i < 9; ++i) {
            const moe_stream_item & it = l0->items[i];
            CHECK(it.layer == 0 && it.expert_id == exp_e[i / 3] && it.role == kRoles[i % 3]);
            CHECK(it.type == GGML_TYPE_Q4_0 && it.ncols == kCols && it.nrows == kRows);
            CHECK(it.src_bytes == src_bytes() && it.dst_bytes == dst_bytes());
            CHECK(it.staging == moe_stream_staging::SOA);
        }
        CHECK(l0->n_host_experts == 3 && l0->n_untriplet == 0 && l0->n_streamable == 9);
        CHECK(l0->region_slots == 18);  // an AOS slot and a SOA slot per matrix
        CHECK(l0->host_bytes == 9 * src_bytes());
        CHECK(l0->expert_bytes_max == 3 * src_bytes());
        CHECK(l0->K == kCols && l0->N == kRows);
        CHECK(l0->host_mix.size() == 1 && l0->host_mix[0].first == GGML_TYPE_Q4_0 &&
              l0->host_mix[0].second == 9 * src_bytes());
    }
    const moe_stream_layer_info * l1 = layer_info(idx, 1);
    CHECK(l1 != nullptr);
    if (l1) {
        CHECK(l1->items.size() == 9 && l1->n_untriplet == 1);
        CHECK(l1->n_host_experts == 3);  // expert 1's fused entry is the same expert
        CHECK(l1->host_bytes == 9 * src_bytes() + kFused);
        // The fused entry takes no slot.
        CHECK(l1->expert_bytes_max == 3 * src_bytes());
    }
    const moe_stream_layer_info * l4 = layer_info(idx, 4);
    CHECK(l4 != nullptr);
    if (l4) {
        CHECK(l4->items.empty() && l4->n_host_experts == 1 && l4->n_untriplet == 1);
    }
    CHECK(idx.host_matrices_per_layer_max == 9);
    CHECK(idx.region_slots_max == 18);
    CHECK(idx.matrix_bytes_max == dst_bytes());
    // Resident on device 0: 24 triplet entries per layer for layers 3 and 4 minus the host ones, 15 for layers 0-2.
    const size_t resident_entries = 3 * (3 * 5) + 3 * 8 + 3 * 7;
    CHECK(idx.resident_expert_bytes == resident_entries * dst_bytes());

    // ---- the ring is sized over streamable matrices: a host expert of a type no prompt kernel reads is listed but
    // given no slot ----
    {
        const moe_stream_plan_index   mixed = index_of(make_mixed_plan(true), kernels_soa_q4);
        const moe_stream_layer_info * m0    = layer_info(mixed, 0);
        const moe_stream_layer_info * m1    = layer_info(mixed, 1);
        CHECK(m0 != nullptr && m1 != nullptr);
        if (m0 && m1) {
            // Both listed, so a stage counts the untyped ones; only layer 0's take slots.
            CHECK(m0->items.size() == 6 && m1->items.size() == 12);
            CHECK(m0->n_streamable == 6 && m1->n_streamable == 0);
            CHECK(m0->region_slots == 12 && m1->region_slots == 0);
            for (const moe_stream_item & it : m1->items) {
                CHECK(it.staging == moe_stream_staging::NONE);
            }
            CHECK(m0->expert_bytes_max == 3 * src_bytes());
            CHECK(m1->expert_bytes_max == 0);
            // The cost model still sees every host byte.
            CHECK(m1->n_host_experts == 4 && m1->host_bytes > 0);
        }
        // Layer 0's six, not layer 1's twelve, and a Q4_0 slot, not the IQ3_XXS device size.
        CHECK(mixed.host_matrices_per_layer_max == 6);
        CHECK(mixed.region_slots_max == 12);
        CHECK(mixed.matrix_bytes_max == dst_bytes());
        const moe_stream_plan_index iq_only = index_of(make_mixed_plan(false), kernels_soa_q4);
        CHECK(iq_only.host_matrices_per_layer_max == 0 && iq_only.matrix_bytes_max == 0);
        CHECK(iq_only.region_slots_max == 0);
        const moe_stream_layer_info * q1 = layer_info(iq_only, 1);
        CHECK(q1 != nullptr && q1->items.size() == 12);
    }

    // ---- AOS-only staging (c-hmey): a type the device reads only as AOS takes one slot and is read in place ----
    {
        // IQ3_XXS, no reorder, read as AOS: every matrix streamable, one slot each, and the slot sized from the AOS
        // bytes it holds, not from the device size the plan entry carries (4 x a Q4_0 matrix here).
        const moe_stream_plan_index   iq = index_of(make_mixed_plan(false), kernels_aos);
        const moe_stream_layer_info * q1 = layer_info(iq, 1);
        CHECK(q1 != nullptr);
        if (q1) {
            CHECK(q1->items.size() == 12 && q1->n_streamable == 12 && q1->region_slots == 12);
            for (const moe_stream_item & it : q1->items) {
                CHECK(it.staging == moe_stream_staging::AOS_ONLY);
            }
            CHECK(q1->expert_bytes_max == 3 * iq_src_bytes());
        }
        CHECK(iq.host_matrices_per_layer_max == 12 && iq.region_slots_max == 12);
        CHECK(iq.matrix_bytes_max == iq_src_bytes());

        // Q4_0 reorders, but no prompt kernel reads its SOA layout: AOS-only, one slot, sized from src_bytes.
        const moe_stream_plan_index   q4 = index_of(make_plan(), kernels_aos);
        const moe_stream_layer_info * a0 = layer_info(q4, 0);
        CHECK(a0 != nullptr);
        if (a0) {
            CHECK(a0->n_streamable == 9 && a0->region_slots == 9);
            for (const moe_stream_item & it : a0->items) {
                CHECK(it.staging == moe_stream_staging::AOS_ONLY);
            }
        }
        CHECK(q4.host_matrices_per_layer_max == 9 && q4.region_slots_max == 9);
        CHECK(q4.matrix_bytes_max == src_bytes());

        // One layer of both: six Q4_0 matrices at two slots, six IQ3_XXS at one. A slot holds the larger of the Q4_0
        // matrix's two sides and the IQ3_XXS matrix's AOS bytes.
        const moe_stream_plan_index   mix = index_of(make_one_layer_mix_plan(), kernels_mixed);
        const moe_stream_layer_info * x0  = layer_info(mix, 0);
        CHECK(x0 != nullptr);
        if (x0) {
            CHECK(x0->items.size() == 12 && x0->n_streamable == 12 && x0->region_slots == 18);
            int n_soa = 0;
            int n_aos = 0;
            for (const moe_stream_item & it : x0->items) {
                n_soa += it.staging == moe_stream_staging::SOA ? 1 : 0;
                n_aos += it.staging == moe_stream_staging::AOS_ONLY ? 1 : 0;
                CHECK(it.staging ==
                      (it.type == GGML_TYPE_Q4_0 ? moe_stream_staging::SOA : moe_stream_staging::AOS_ONLY));
            }
            CHECK(n_soa == 6 && n_aos == 6);
        }
        CHECK(mix.host_matrices_per_layer_max == 12 && mix.region_slots_max == 18);
        CHECK(mix.matrix_bytes_max == std::max(dst_bytes(), iq_src_bytes()));

        // No predicate: the ring cannot tell what the device reads, so nothing is streamable (fail closed).
        const moe_stream_plan_index blind = index_of(make_plan(), nullptr);
        CHECK(blind.host_matrices_per_layer_max == 0 && blind.region_slots_max == 0 && blind.matrix_bytes_max == 0);
        const moe_stream_layer_info * b0 = layer_info(blind, 0);
        CHECK(b0 != nullptr && b0->items.size() == 9 && b0->n_streamable == 0);

        // The predicate is asked once per distinct (type, shape, bytes) per build, not once per entry (c-7soh): the
        // plan has one host matrix shape, which takes a SOA question and then an AOS one.
        int calls = 0;
        (void) ggml_sycl::moe_stream_plan_index_build(make_plan(), 0, kernels_aos_counted, &calls, nullptr);
        CHECK(calls == 2);
    }

    // ---- a matrix whose weight the batched executor claims once staged takes one AOS slot, read by the executor's
    // dequant kernel: no reorder and no second slot, whatever the MMVQ kernels answer (llama.cpp-3460, resolving
    // llama.cpp-v7np) ----
    {
        claims_probe                probe;
        const moe_stream_plan_index claimed =
            ggml_sycl::moe_stream_plan_index_build(make_plan(), 0, kernels_soa_q4, &probe, claims_l0_gate_l2);
        const moe_stream_layer_info * c0 = layer_info(claimed, 0);
        const moe_stream_layer_info * c1 = layer_info(claimed, 1);
        const moe_stream_layer_info * c2 = layer_info(claimed, 2);
        CHECK(c0 != nullptr && c1 != nullptr && c2 != nullptr);
        if (c0 && c1 && c2) {
            CHECK(c0->items.size() == 9 && c2->items.size() == 9);
            for (const moe_stream_item & it : c0->items) {
                const bool gate = it.role == expert_tensor_role::GATE;
                CHECK(it.staging == (gate ? moe_stream_staging::AOS_ONLY : moe_stream_staging::SOA));
                CHECK(it.consumer == (gate ? moe_stream_consumer::EXECUTOR : moe_stream_consumer::ROUTE));
            }
            for (const moe_stream_item & it : c2->items) {
                CHECK(it.staging == moe_stream_staging::AOS_ONLY && it.consumer == moe_stream_consumer::EXECUTOR);
            }
            // Layer 0: three claimed gates of one slot, six unclaimed SOA matrices of two.
            CHECK(c0->n_streamable == 9 && c0->region_slots == 15 && c0->expert_bytes_max == 3 * src_bytes());
            CHECK(c1->n_streamable == 9 && c1->region_slots == 18);  // layer 1's weights are not claimed
            CHECK(c2->n_streamable == 9 && c2->region_slots == 9 && c2->expert_bytes_max == 3 * src_bytes());
        }
        CHECK(claimed.host_matrices_per_layer_max == 9 && claimed.region_slots_max == 18);
        // Asked once per (layer, role) with a matrix one AOS slot holds, not once per expert: layers 0-2, three roles.
        CHECK(probe.calls == 9);

        // Every weight claimed: every matrix takes one slot of its AOS bytes.
        const moe_stream_plan_index all =
            ggml_sycl::moe_stream_plan_index_build(make_plan(), 0, kernels_soa_q4, nullptr, claims_all);
        CHECK(all.host_matrices_per_layer_max == 9 && all.region_slots_max == 9 && all.matrix_bytes_max == src_bytes());

        // The executor's table is the authority, not the MMVQ kernels: a claimed IQ3_XXS matrix, which no kernel of
        // kernels_soa_q4 reads, takes one AOS slot. Unclaimed it takes none.
        const moe_stream_plan_index iq_free = index_of(make_mixed_plan(true), kernels_soa_q4);
        const moe_stream_plan_index iq_claimed =
            ggml_sycl::moe_stream_plan_index_build(make_mixed_plan(true), 0, kernels_soa_q4, nullptr, claims_all);
        const moe_stream_layer_info * i1 = layer_info(iq_claimed, 1);
        const moe_stream_layer_info * f1 = layer_info(iq_free, 1);
        CHECK(i1 != nullptr && f1 != nullptr);
        if (i1 && f1) {
            CHECK(f1->n_streamable == 0 && f1->region_slots == 0);
            CHECK(i1->items.size() == 12 && i1->n_streamable == 12 && i1->region_slots == 12);
            for (const moe_stream_item & it : i1->items) {
                CHECK(it.staging == moe_stream_staging::AOS_ONLY && it.consumer == moe_stream_consumer::EXECUTOR);
            }
        }
        CHECK(iq_claimed.matrix_bytes_max == std::max(src_bytes(), iq_src_bytes()));

        // A claimed matrix one AOS slot cannot hold (a zeroed inventory shape) stays NONE, and the claim is not asked
        // for its weight.
        placement_plan zeroed = make_plan();
        for (placement_entry & e : zeroed.entries) {
            if (e.layer_id == 2) {
                e.ne[0] = 0;
            }
        }
        zeroed.build_index();
        claims_probe                zprobe;
        const moe_stream_plan_index z =
            ggml_sycl::moe_stream_plan_index_build(zeroed, 0, kernels_soa_q4, &zprobe, claims_l0_gate_l2);
        const moe_stream_layer_info * z2 = layer_info(z, 2);
        CHECK(z2 != nullptr && z2->items.size() == 9 && z2->n_streamable == 0 && z2->region_slots == 0);
        if (z2) {
            for (const moe_stream_item & it : z2->items) {
                CHECK(it.staging == moe_stream_staging::NONE && it.consumer == moe_stream_consumer::ROUTE);
            }
        }
        CHECK(zprobe.calls == 6);

        // No weight claimed: the index is the one an unset predicate builds.
        const moe_stream_plan_index free_of =
            ggml_sycl::moe_stream_plan_index_build(make_plan(), 0, kernels_soa_q4, nullptr, claims_none);
        const moe_stream_layer_info * f0 = layer_info(free_of, 0);
        CHECK(free_of.host_matrices_per_layer_max == idx.host_matrices_per_layer_max &&
              free_of.region_slots_max == idx.region_slots_max && free_of.matrix_bytes_max == idx.matrix_bytes_max);
        CHECK(f0 != nullptr && f0->n_streamable == 9 && f0->region_slots == 18);
    }

    // ---- why a layer answered RING_UNAVAILABLE: the first rule that holds, in a fixed order ----
    {
        const moe_stream_plan_index iq_only = index_of(make_mixed_plan(false), kernels_soa_q4);
        const moe_stream_plan_index mixed   = index_of(make_mixed_plan(true), kernels_soa_q4);

        ggml_sycl::moe_stream_ring_geometry none;
        ggml_sycl::moe_stream_ring_geometry geo;
        geo.slot_bytes   = 4096;
        geo.region_slots = 12;
        geo.n_slots      = 24;
        geo.bytes        = 24 * 4096;
        const auto rule  = [](const moe_stream_plan_index & idx, const ggml_sycl::moe_stream_ring_geometry & g,
                             bool in_order, bool refused) {
            return std::string(ggml_sycl::moe_stream_ring_unavailable_rule(idx, g, in_order, refused));
        };
        // No streamable host matrix wins over every later rule, the queue included.
        CHECK(rule(iq_only, none, false, true) == "no host matrices of a streamable type");
        moe_stream_plan_index no_bytes = mixed;
        no_bytes.matrix_bytes_max      = 0;
        CHECK(rule(no_bytes, none, true, false) == "no matrix bytes");
        CHECK(rule(mixed, none, true, false) == "ring exceeds 10% of resident expert bytes");
        CHECK(rule(mixed, geo, false, false) == "queue not in order");
        CHECK(rule(mixed, geo, true, true) == "reserve refused");
        // None of the five: named anyway, so no ring-unavailable line goes unexplained.
        CHECK(rule(mixed, geo, true, false) == "region smaller than the largest streamable triplet");
    }

    // ---- what a decision stages ----
    if (l0) {
        CHECK(ggml_sycl::moe_stream_select(*l0, moe_exec_regime::STREAM_ALL, bits).size() == 9);
        const std::vector<uint8_t>         only5 = ggml_sycl::moe_stream_ids_bitmap(ids + 2, 2, kExperts);  // {5, 7}
        const std::vector<moe_stream_item> routed =
            ggml_sycl::moe_stream_select(*l0, moe_exec_regime::ROUTED_ONLY, only5);
        CHECK(routed.size() == 3);
        for (const moe_stream_item & it : routed) {
            CHECK(it.expert_id == 5);
        }
        CHECK(ggml_sycl::moe_stream_select(*l0, moe_exec_regime::NONE, bits).empty());
    }

    // ---- fully staged, answered from the table (c-7soh) ----
    const int32_t                      ids_15[2] = { 1, 5 };
    const int32_t                      ids_02[2] = { 0, 2 };
    const auto                         b15       = ggml_sycl::moe_stream_ids_bitmap(ids_15, 2, kExperts);
    const auto                         b02       = ggml_sycl::moe_stream_ids_bitmap(ids_02, 2, kExperts);
    const std::unordered_set<uint64_t> empty_table;
    std::unordered_set<uint64_t>       t15 = keys_of(0, { 1, 5 });
    CHECK(ggml_sycl::moe_stream_fully_staged(l0, b15, t15));
    CHECK(ggml_sycl::moe_stream_fully_staged(l0, b02, empty_table));  // the routed experts are resident
    t15.erase(ggml_sycl::moe_stream_table_key(0, 5, expert_tensor_role::UP));
    CHECK(!ggml_sycl::moe_stream_fully_staged(l0, b15, t15));
    CHECK(!ggml_sycl::moe_stream_fully_staged(l0, b15, keys_of(2, { 1, 5 })));  // another layer's entries
    CHECK(ggml_sycl::moe_stream_fully_staged(l0, bits, keys_of(0, { 1, 5, 6 })));
    // A host entry the ring cannot stage makes the layer not staged, whatever the table holds.
    CHECK(!ggml_sycl::moe_stream_fully_staged(l1, b15, keys_of(1, { 1, 5, 6 })));
    // Host experts present and no item: not staged (Task 10 relies on it).
    const int32_t ids_2[1] = { 2 };
    CHECK(!ggml_sycl::moe_stream_fully_staged(l4, ggml_sycl::moe_stream_ids_bitmap(ids_2, 1, kExperts), empty_table));
    CHECK(!ggml_sycl::moe_stream_fully_staged(l4, b02, empty_table));
    // No host expert: nothing to stage.
    CHECK(ggml_sycl::moe_stream_fully_staged(nullptr, b15, empty_table));

    // ---- an item the reorder can take into a slot ----
    moe_stream_item q{};
    q.type      = GGML_TYPE_Q4_0;
    q.ncols     = kCols;
    q.nrows     = kRows;
    q.src_bytes = src_bytes();
    q.dst_bytes = dst_bytes();
    CHECK(ggml_sycl::moe_stream_item_reorderable(q, 4096));
    CHECK(!ggml_sycl::moe_stream_item_reorderable(q, dst_bytes() - 1));  // does not fit the slot
    // The plan's default type: none known.
    moe_stream_item bad = q;
    bad.type            = GGML_TYPE_COUNT;
    CHECK(!ggml_sycl::moe_stream_item_reorderable(bad, 4096));
    bad           = q;
    bad.type      = GGML_TYPE_IQ3_XXS;  // no AOS->SOA reorder in the table
    bad.src_bytes = ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * kRows;
    CHECK(!ggml_sycl::moe_stream_item_reorderable(bad, 4096));
    bad       = q;
    bad.ncols = 100;  // not whole blocks
    CHECK(!ggml_sycl::moe_stream_item_reorderable(bad, 4096));
    bad           = q;
    bad.src_bytes = src_bytes() + 1;  // not the rows' bytes
    CHECK(!ggml_sycl::moe_stream_item_reorderable(bad, 4096));
    bad       = q;
    bad.nrows = 0;
    CHECK(!ggml_sycl::moe_stream_item_reorderable(bad, 4096));

    // ---- an item one slot holds as AOS, and how the ring stages an item (c-hmey) ----
    CHECK(ggml_sycl::moe_stream_item_aos_fits(q, src_bytes()));  // the AOS bytes alone: no SOA side to hold
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(q, src_bytes() - 1));
    moe_stream_item iq = q;
    iq.type            = GGML_TYPE_IQ3_XXS;
    iq.src_bytes       = iq_src_bytes();
    CHECK(ggml_sycl::moe_stream_item_aos_fits(iq, 4096));
    bad      = iq;
    bad.type = GGML_TYPE_COUNT;
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad       = iq;
    bad.ncols = 100;  // not whole blocks
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad           = iq;
    bad.src_bytes = iq_src_bytes() + 1;  // not the rows' bytes
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad       = iq;
    bad.nrows = 0;
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));

    CHECK(ggml_sycl::moe_stream_item_staging(q, 4096, kernels_soa_q4, nullptr) == moe_stream_staging::SOA);
    CHECK(ggml_sycl::moe_stream_item_staging(q, 4096, kernels_aos, nullptr) == moe_stream_staging::AOS_ONLY);
    CHECK(ggml_sycl::moe_stream_item_staging(q, src_bytes() - 1, kernels_aos, nullptr) == moe_stream_staging::NONE);
    CHECK(ggml_sycl::moe_stream_item_staging(iq, 4096, kernels_aos, nullptr) == moe_stream_staging::AOS_ONLY);
    CHECK(ggml_sycl::moe_stream_item_staging(iq, 4096, kernels_soa_q4, nullptr) == moe_stream_staging::NONE);
    CHECK(ggml_sycl::moe_stream_item_staging(q, 4096, nullptr, nullptr) == moe_stream_staging::NONE);

    // Lead ruling c-o9uw-2: SOA iff the type reorders and a prompt kernel reads it as SOA, else AOS-only iff one reads
    // it as AOS. Both types below reorder; under the tables' answers only MXFP4 is staged SOA.
    moe_stream_item q4k{};
    q4k.type           = GGML_TYPE_Q4_K;
    q4k.ncols          = kCols;
    q4k.nrows          = kRows;
    q4k.src_bytes      = ggml_row_size(GGML_TYPE_Q4_K, kCols) * static_cast<size_t>(kRows);
    q4k.dst_bytes      = q4k.src_bytes;
    moe_stream_item mx = q4k;
    mx.type            = GGML_TYPE_MXFP4;
    mx.src_bytes       = ggml_row_size(GGML_TYPE_MXFP4, kCols) * static_cast<size_t>(kRows);
    mx.dst_bytes       = mx.src_bytes;
    CHECK(ggml_sycl::moe_stream_item_reorderable(q4k, 4096) && ggml_sycl::moe_stream_item_reorderable(mx, 4096));
    CHECK(ggml_sycl::moe_stream_item_staging(q4k, 4096, kernels_table, nullptr) == moe_stream_staging::AOS_ONLY);
    CHECK(ggml_sycl::moe_stream_item_staging(mx, 4096, kernels_table, nullptr) == moe_stream_staging::SOA);
    CHECK(ggml_sycl::moe_stream_staging_slots(ggml_sycl::moe_stream_item_staging(q4k, 4096, kernels_table, nullptr)) ==
          1u);
    CHECK(ggml_sycl::moe_stream_staging_slots(ggml_sycl::moe_stream_item_staging(mx, 4096, kernels_table, nullptr)) ==
          2u);

    // ---- a host lease the ring may copy from directly (llama.cpp-9y0p) ----
    alloc_metadata owner{};
    owner.tier      = alloc_tier::HOST_PINNED;
    owner.role      = alloc_role::WEIGHT;
    owner.host_zone = host_zone_id::WEIGHT;
    owner.device    = 0;
    CHECK(ggml_sycl::moe_stream_owner_streamable(owner, 0));
    CHECK(!ggml_sycl::moe_stream_owner_streamable(owner, 1));  // another device's host zone
    alloc_metadata o = owner;
    o.tier           = alloc_tier::DEVICE_VRAM;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));
    o           = owner;
    o.host_zone = host_zone_id::COUNT;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));
    o      = owner;
    o.role = alloc_role::STAGING;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));

    // ---- the rates: seed, then the tuning-cache record where it measured, then the environment ----
    const std::vector<std::pair<ggml_type, size_t>> q4 = {
        { GGML_TYPE_Q4_0, 1000 }
    };
    const moe_exec_rates seed      = ggml_sycl::moe_exec_rates_seed(q4);
    const moe_exec_rates no_record = ggml_sycl::moe_stream_rates_assemble(q4, nullptr, nullptr);
    CHECK(near(no_record.h2d_gbps, seed.h2d_gbps) && near(no_record.reorder_gbps, seed.reorder_gbps));
    CHECK(near(no_record.dev_gflops, seed.dev_gflops) && near(no_record.cpu_gbps, seed.cpu_gbps));
    ggml_sycl_moe_exec_rates_record rec{};
    rec.h2d_gbps                  = 25.0;
    rec.reorder_gbps              = 300.0;
    rec.n_types                   = 1;
    rec.types[0]                  = GGML_TYPE_Q4_0;
    rec.dev_gflops[0]             = 1260.0;
    const moe_exec_rates measured = ggml_sycl::moe_stream_rates_assemble(q4, &rec, nullptr);
    CHECK(near(measured.h2d_gbps, 25.0) && near(measured.reorder_gbps, 300.0) && near(measured.dev_gflops, 1260.0));
    const std::vector<std::pair<ggml_type, size_t>> mix = {
        { GGML_TYPE_Q4_0,  1000 },
        { GGML_TYPE_MXFP4, 1000 }
    };
    const moe_exec_rates mixed = ggml_sycl::moe_stream_rates_assemble(mix, &rec, nullptr);
    CHECK(near(mixed.dev_gflops,
               2000.0 / (1000.0 / 1260.0 + 1000.0 / ggml_sycl::moe_exec_dev_gflops_seed(GGML_TYPE_MXFP4))));
    ggml_sycl_moe_exec_rates_record unmeasured = rec;
    unmeasured.h2d_gbps                        = 0.0;
    unmeasured.reorder_gbps                    = -1.0;
    const moe_exec_rates fallback              = ggml_sycl::moe_stream_rates_assemble(q4, &unmeasured, nullptr);
    CHECK(near(fallback.h2d_gbps, seed.h2d_gbps) && near(fallback.reorder_gbps, seed.reorder_gbps));
    CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, &rec, fake_env_h2d).h2d_gbps, 31.0));
    CHECK(ggml_sycl::moe_exec_rates_valid(measured) && ggml_sycl::moe_exec_rates_valid(fallback));

    // ---- the predicates begin_layer and resolve decide by ----
    // A micro-batch starts at the first op of a prompt, at a layer below the current one, or at the current layer in a
    // new graph. A new graph at a higher layer carries it on: a split micro-batch spans several graph_compute calls.
    CHECK(ggml_sycl::moe_stream_ubatch_starts(false, 3, 3, 7, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 2, 3, 7, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 0, 47, 8, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 3, 3, 8, 7));
    // the layer's later ops
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 3, 3, 7, 7));
    // the next layer in the same graph
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 4, 3, 7, 7));
    // the next layer in the next graph of a split micro-batch
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 16, 15, 8, 7));

    // Bytes move only when the cost model chose the ring and GGML_SYCL_MOE_EXEC=ring asked for it (lead ruling c-3skf).
    moe_exec_decision ring_all;
    ring_all.choice               = moe_exec_choice::RING;
    ring_all.regime               = moe_exec_regime::STREAM_ALL;
    moe_exec_decision ring_routed = ring_all;
    ring_routed.regime            = moe_exec_regime::ROUTED_ONLY;
    moe_exec_decision cpu_routed  = ring_routed;
    cpu_routed.choice             = moe_exec_choice::CPU;
    moe_exec_decision unconsulted;
    CHECK(ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::RING));
    CHECK(ggml_sycl::moe_stream_stage_now(ring_routed, moe_exec_policy::RING));
    CHECK(!ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::AUTO));
    CHECK(!ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::CPU));
    CHECK(!ggml_sycl::moe_stream_stage_now(cpu_routed, moe_exec_policy::RING));
    CHECK(!ggml_sycl::moe_stream_stage_now(unconsulted, moe_exec_policy::RING));

    // The next layer is resolved a layer ahead unless it is a ROUTED_ONLY ring, which needs that layer's own ids.
    CHECK(ggml_sycl::moe_stream_resolve_ahead(ring_all));
    CHECK(!ggml_sycl::moe_stream_resolve_ahead(ring_routed));
    CHECK(ggml_sycl::moe_stream_resolve_ahead(cpu_routed));
    CHECK(ggml_sycl::moe_stream_resolve_ahead(unconsulted));

    // ---- the trace's regime names ----
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::STREAM_ALL), "stream-all") == 0);
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::ROUTED_ONLY), "routed-only") == 0);
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::NONE), "none") == 0);

    // ---- the staged route's tally (llama.cpp-o9uw): served and refused lookups, and each refusal's cause and detail,
    // which end_prompt traces (plan Task 8: the trace says why) ----
    {
        using ggml_sycl::moe_staged_route_refusal;
        ggml_sycl::moe_staged_route_tally t;
        CHECK(t.served == 0 && t.refused == 0 && t.refusals_text().empty());
        t.count(moe_staged_route_refusal::NONE, nullptr);
        t.count(moe_staged_route_refusal::LAYOUT, nullptr);
        t.count(moe_staged_route_refusal::UNSUPPORTED, "local-layout-unsupported");
        t.count(moe_staged_route_refusal::LAYOUT, nullptr);
        // The same detail at another address is the same refusal: details compare by text.
        const char same_reason[] = "local-layout-unsupported";
        t.count(moe_staged_route_refusal::UNSUPPORTED, same_reason);
        t.count(moe_staged_route_refusal::RECIPE, "RECIPE_MISMATCH");
        t.count(moe_staged_route_refusal::RECIPE, "RECIPE_MISSING");
        CHECK(t.served == 1 && t.refused == 6 && t.refusals.size() == 4);
        CHECK(t.refusals_text() ==
              "layout=2 unsupported(local-layout-unsupported)=2 recipe(RECIPE_MISMATCH)=1 recipe(RECIPE_MISSING)=1");
        // Every cause has a name of its own.
        const moe_staged_route_refusal all[] = {
            moe_staged_route_refusal::NONE,          moe_staged_route_refusal::CLAIMED,
            moe_staged_route_refusal::UNSUPPORTED,   moe_staged_route_refusal::KERNEL,
            moe_staged_route_refusal::DIRECT_KERNEL, moe_staged_route_refusal::VIEW,
            moe_staged_route_refusal::LAYOUT,        moe_staged_route_refusal::ROUTE,
            moe_staged_route_refusal::RECIPE,
        };
        std::unordered_set<std::string> names;
        for (moe_staged_route_refusal r : all) {
            names.insert(ggml_sycl::moe_staged_route_refusal_name(r));
        }
        CHECK(names.size() == sizeof(all) / sizeof(all[0]) && names.count("unknown") == 0);
        CHECK(std::strcmp(ggml_sycl::moe_staged_route_refusal_name(moe_staged_route_refusal::DIRECT_KERNEL),
                          "direct-kernel") == 0);
    }

    // ---- the regions' last uses: what stage() refills, layer_done() gates and reset_regions() forgets ----
    {
        // The struct is inline and every input below is a constant, so at -O3 the compiler would prove each check true
        // and drop it: a check that cannot fail proves nothing. Every use goes through a fresh read of a volatile
        // pointer instead, so no store before one read is known at the next, and each check runs (its text is in the
        // binary).
        ggml_sycl::moe_stream_region_reads<int> reads;
        ggml_sycl::moe_stream_region_reads<int> * volatile reads_at = &reads;
        const auto rr = [&]() -> ggml_sycl::moe_stream_region_reads<int> & {
            return *reads_at;
        };
        CHECK(rr().layer[0] == -1 && rr().layer[1] == -1 && !rr().aos_only[0] && !rr().aos_only[1]);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        // stage(): region 0 now holds layer 3 and left an AOS-only slot; no read gate until layer 3's readers are done.
        rr().refill(0, 3, true);
        CHECK(rr().layer[0] == 3 && rr().aos_only[0] && !rr().read_gate[0]);
        CHECK(rr().layer[1] == -1 && !rr().aos_only[1] && !rr().read_gate[1]);
        // The down op of layer 4 gates no region: the region holding layer 3 waits for layer 3's own readers (the gate
        // that layer_done closes the open region on belongs to the region holding the next layer).
        rr().layer_done(4, 40);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        rr().layer_done(3, 30);
        CHECK(rr().read_gate[0] && *rr().read_gate[0] == 30 && !rr().read_gate[1]);
        // Region 1 holds layer 4, with SOA slots only, and takes layer 4's gate; region 0 keeps its own.
        rr().refill(1, 4, false);
        CHECK(rr().layer[1] == 4 && !rr().aos_only[1] && !rr().read_gate[1]);
        rr().layer_done(4, 41);
        CHECK(rr().read_gate[1] && *rr().read_gate[1] == 41 && rr().read_gate[0] && *rr().read_gate[0] == 30);
        // Refilled, region 0 holds layer 5 and forgets layer 3's gate and its AOS-only slot.
        rr().refill(0, 5, false);
        CHECK(rr().layer[0] == 5 && !rr().aos_only[0] && !rr().read_gate[0]);
        CHECK(rr().read_gate[1] && *rr().read_gate[1] == 41);
        // A late down op of a layer no region holds any more records nothing.
        rr().layer_done(3, 31);
        CHECK(!rr().read_gate[0] && rr().read_gate[1] && *rr().read_gate[1] == 41);
        // reset_regions(), for a new ring or a torn-down one: no layer, no AOS-only slot, no read gate in either
        // region, so a later layer_done of a layer the old ring held gates nothing.
        rr().refill(0, 6, true);
        rr().reset();
        CHECK(rr().layer[0] == -1 && rr().layer[1] == -1 && !rr().aos_only[0] && !rr().aos_only[1]);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        rr().layer_done(4, 42);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
    }

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-phase: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-phase: OK\n");
    return 0;
}
