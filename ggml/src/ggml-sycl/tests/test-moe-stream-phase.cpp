// llama.cpp-xod7 (plan Task 7b of llama.cpp-36j0): the prompt phase's pure parts, the ones moe_stream_phase consults
// before it touches a queue. Host-only: hand-built plans, no queue, no device, and no SYCL object is constructed.
//
// The plan: 5 layers x 8 experts, Q4_0 gate/up/down, resident on device 0 unless noted.
//   layers 0-2: experts 1, 5 and 6 on the host;
//   layer 1:    also a fused host GATE_UP entry for expert 1 (a shape the ring cannot stage);
//   layer 3:    no host expert (the phase skips it);
//   layer 4:    only a fused host GATE_UP for expert 2 (host experts present, no item).

#include "ggml-sycl/moe-stream-plan.hpp"

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
using ggml_sycl::moe_exec_rates;
using ggml_sycl::moe_exec_regime;
using ggml_sycl::moe_stream_item;
using ggml_sycl::moe_stream_layer_info;
using ggml_sycl::moe_stream_plan_index;
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
    const moe_stream_plan_index idx  = ggml_sycl::moe_stream_plan_index_build(plan, 0);
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
        }
        CHECK(l0->n_host_experts == 3 && l0->n_untriplet == 0);
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
        CHECK(l1->expert_bytes_max == 3 * src_bytes() + kFused);
    }
    const moe_stream_layer_info * l4 = layer_info(idx, 4);
    CHECK(l4 != nullptr);
    if (l4) {
        CHECK(l4->items.empty() && l4->n_host_experts == 1 && l4->n_untriplet == 1);
    }
    CHECK(idx.host_matrices_per_layer_max == 9);
    CHECK(idx.matrix_bytes_max == dst_bytes());
    // Resident on device 0: 24 triplet entries per layer for layers 3 and 4 minus the host ones, 15 for layers 0-2.
    const size_t resident_entries = 3 * (3 * 5) + 3 * 8 + 3 * 7;
    CHECK(idx.resident_expert_bytes == resident_entries * dst_bytes());

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

    // ---- a host lease the ring may copy from directly (9y0p M4) ----
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

    // ---- the trace's regime names ----
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::STREAM_ALL), "stream-all") == 0);
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::ROUTED_ONLY), "routed-only") == 0);
    CHECK(std::strcmp(ggml_sycl::moe_stream_regime_name(moe_exec_regime::NONE), "none") == 0);

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-phase: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-phase: OK\n");
    return 0;
}
