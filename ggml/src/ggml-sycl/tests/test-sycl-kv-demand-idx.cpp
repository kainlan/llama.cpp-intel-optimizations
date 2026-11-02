// llama.cpp-5efe C4 (part b): the indexer cache is host KV too, and the demotion counts it.
//
// A hybrid sparse-attention model (qwen4exp, llama_memory_hybrid_idx) carries a second per-token cache next to the
// attention KV: the "idx_" indexer keys, one `indexer_head_size`-wide K row per cell on every attention layer
// (src/llama-memory-hybrid-idx.cpp). A demoted layer's indexer keys go to the host KV buffer with its K and V, so the
// printed "X MB host KV" must count them: 64 MiB per layer, 768 MiB at Qwen's default context.
//
// The width arrives per layer (placement_plan::layer_idx_k_width, from the inventory's kv_idx_k_width_per_layer), and
// kv_size_for_layer() already charges it: it is the layer's K/V (kv_main_size_for_layer(), the buffer llama allocates
// for K and V) plus its indexer keys (kv_idx_size_for_layer(), the buffer of their own). kv_demand_for_layer(), the one
// accessor the two demotion sites read, is therefore kv_size_for_layer() itself; the K/V buffer alone is the figure
// that leaves the indexer out, so it is the control here.
//
// Pure: the real placement_plan and the real plan_runtime_kv_demotion(), no model, no device, no queue.

#include "../kv-runtime-demotion.hpp"
#include "../unified-cache.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using ggml_sycl::placement_plan;

static int g_checks = 0;
static int g_fail   = 0;

static void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

static void check_true(const char * what, bool cond) {
    g_checks++;
    if (!cond) {
        g_fail++;
        printf("  FAIL: %s\n", what);
    }
}

constexpr size_t MiB = 1024u * 1024u;

// Qwen's shape: 48 layers, every fourth a full-attention layer (3, 7, ..., 47), the rest recurrent and holding no
// attention KV. 262144 cells; K + V widths of 1024 elements, so a layer's K/V is 262144 * 1024 * 2 B = 512 MiB. The
// indexer key width is set on the full-attention layers only, as llama_kv_layer_owners_default() reports it.
static placement_plan make_qwen_plan(uint32_t idx_k_width) {
    placement_plan plan{};
    plan.layer_kind.assign(48, GGML_SYCL_KV_LAYER_SHARED);
    plan.layer_k_width.assign(48, 0);
    plan.layer_v_width.assign(48, 0);
    plan.layer_idx_k_width.assign(48, 0);
    for (uint32_t il = 3; il < 48; il += 4) {
        plan.layer_kind[il]        = GGML_SYCL_KV_LAYER_FULL;
        plan.layer_k_width[il]     = 512;
        plan.layer_v_width[il]     = 512;
        plan.layer_idx_k_width[il] = idx_k_width;
    }
    plan.planner_n_ctx     = 262144;
    plan.planner_n_ubatch  = 512;
    plan.planner_n_seq_max = 1;
    return plan;
}

static void test_demand_adds_the_indexer_to_every_kv_layer() {
    printf("(a) kv_demand_for_layer = the K/V buffer + the indexer keys, for a layer that carries KV\n");
    const placement_plan plan = make_qwen_plan(128);
    for (uint32_t il = 0; il < 48; ++il) {
        char label[64];
        snprintf(label, sizeof(label), "layer %u", il);
        if (il % 4 == 3) {
            // the K/V buffer, the indexer keys (262144 * 128 * 2 B), and the demand as one figure, not a second sum
            check_eq(label, plan.kv_main_size_for_layer(il), 512 * MiB);
            check_eq(label, plan.kv_idx_size_for_layer(il), 64 * MiB);
            check_eq(label, plan.kv_demand_for_layer(il), 576 * MiB);
            check_eq(label, plan.kv_demand_for_layer(il), plan.kv_size_for_layer(il));
        } else {
            check_eq(label, plan.kv_demand_for_layer(il), 0);  // a layer with no KV has no indexer either
        }
    }
}

static void test_zero_width_is_the_old_size() {
    printf("(b) no indexer (width 0): no layer has indexer keys, and the demand is the K/V buffer alone\n");
    const placement_plan plan = make_qwen_plan(0);
    for (uint32_t il = 0; il < 48; ++il) {
        char label[64];
        snprintf(label, sizeof(label), "layer %u", il);
        check_eq(label, plan.kv_idx_size_for_layer(il), 0);
        check_eq(label, plan.kv_demand_for_layer(il), plan.kv_main_size_for_layer(il));
    }
    check_eq("layer 3", plan.kv_demand_for_layer(3), 512 * MiB);
}

static void test_a_plan_without_per_layer_truth() {
    printf(
        "(c0) a legacy plan (no per-layer kind/width arrays): the indexer width has no layer to charge,\n"
        "     the demand is the uniform size\n");
    placement_plan plan{};
    plan.kv_per_layer  = 512 * MiB;
    plan.planner_n_ctx = 262144;
    plan.layer_idx_k_width.assign(8, 128);
    check_eq("indexer", plan.kv_idx_size_for_layer(5), 0);
    check_eq("demand", plan.kv_demand_for_layer(5), 512 * MiB);
}

// The demotion input is what the two sites in ggml-sycl.cpp build: layer_kv_bytes per layer from one accessor. The
// control reads the K/V buffer alone, the figure that leaves the indexer out.
static ggml_sycl::kv_demotion_result demote(const placement_plan & plan, bool use_demand, size_t room_bytes) {
    ggml_sycl::kv_demotion_input in;
    in.kv_device.assign(48, -1);
    in.swa_layer_mask.assign(48, 0);
    in.layer_kv_bytes.assign(48, 0);
    size_t kv_total = 0;
    for (uint32_t il = 0; il < 48; ++il) {
        in.layer_kv_bytes[il] = use_demand ? plan.kv_demand_for_layer(il) : plan.kv_main_size_for_layer(il);
        if (in.layer_kv_bytes[il] > 0) {
            in.kv_device[il] = 0;
            kv_total += in.layer_kv_bytes[il];
        }
    }
    const size_t no_kv  = 10240 * MiB;
    in.vram_bytes       = no_kv + kv_total;
    in.vram_budget      = no_kv + room_bytes;
    in.demote_swa       = true;
    return ggml_sycl::plan_runtime_kv_demotion(in);
}

static void test_demotion_prints_the_indexer_in_the_host_figure() {
    printf("(c) Qwen no -c: 12 layers demoted, 3..47, 6912.0 MB of host KV with the indexer\n");
    const placement_plan plan = make_qwen_plan(128);
    const size_t         room = static_cast<size_t>(287.8 * MiB);

    const ggml_sycl::kv_demotion_result with_demand = demote(plan, true, room);
    check_true("fits", with_demand.fits);
    check_eq("demoted layers", with_demand.demoted_layers.size(), 12);
    check_true("latest first: 47 .. 3", !with_demand.demoted_layers.empty() && with_demand.demoted_layers.front() == 47 &&
                                            with_demand.demoted_layers.back() == 3);
    check_eq("host KV bytes", with_demand.host_kv_bytes_added, 6912 * MiB);

    // The control, which is also the mutant: the demotion input reading the K/V buffer alone prints the figure without
    // the indexer.
    const ggml_sycl::kv_demotion_result with_size = demote(plan, false, room);
    check_eq("control: kv_main_size_for_layer prints", with_size.host_kv_bytes_added, 6144 * MiB);
    check_true("the two figures differ", with_size.host_kv_bytes_added != with_demand.host_kv_bytes_added);
}

int main() {
    printf("=== kv_demand_for_layer counts the indexer cache (llama.cpp-5efe C4 part b) ===\n");
    test_demand_adds_the_indexer_to_every_kv_layer();
    test_zero_width_is_the_old_size();
    test_a_plan_without_per_layer_truth();
    test_demotion_prints_the_indexer_in_the_host_figure();
    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
