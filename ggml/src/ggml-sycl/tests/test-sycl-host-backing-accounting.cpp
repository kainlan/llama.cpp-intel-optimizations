// llama.cpp-5efe C4 (part c): the planner's host-weight accounting is split by host backing.
//
// `placement_entry::host_backing { PINNED, FILE_MAPPED, HOST_CPU, CPU_OWNED }` says whose memory a host-planned weight
// lives in, and the plan carries four terms instead of one (design 5efe §4.4):
//
//   weight_host_bytes         host-executed PINNED / HOST_CPU entries; charged; sizes the WEIGHT zone
//   host_cpu_owned_bytes      CPU_OWNED entries; charged (anonymous memory); sizes nothing
//   host_mapped_bytes         host-executed FILE_MAPPED entries; not charged to host_bytes (the page cache is not ours)
//   host_mapped_source_bytes  device-planned FILE_MAPPED sources (the staging source); not charged
//
// "placed on host" (what the tiering verdict reads) is weight_host + cpu_owned + mapped; "charged" (what the ledger
// reads) is weight_host + cpu_owned. Every writer goes through placement_plan::charge_host_weight / uncharge_host_weight
// / charge_host_weight_entries, so no site can add to weight_host_bytes without testing the backing.
//
// Pure: the real placement_plan, no model, no device, no queue.

#include "../unified-cache.hpp"

#include <cstddef>
#include <cstdio>
#include <vector>

using ggml_sycl::expert_tensor_role;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_host_backing;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_size_basis;

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

static void test_defaults_are_pinned_and_zero() {
    printf("(a) a new entry is PINNED and a new plan has all four terms at 0\n");
    const placement_entry e;
    check_true("entry default backing", e.host_backing == placement_host_backing::PINNED);
    const placement_plan plan{};
    check_eq("weight_host_bytes", plan.weight_host_bytes, 0);
    check_eq("host_cpu_owned_bytes", plan.host_cpu_owned_bytes, 0);
    check_eq("host_mapped_bytes", plan.host_mapped_bytes, 0);
    check_eq("host_mapped_source_bytes", plan.host_mapped_source_bytes, 0);
}

static void test_each_backing_lands_in_its_own_term() {
    printf("(b) charge_host_weight puts each backing in its own term; only FILE_MAPPED stays out of host_bytes\n");
    placement_plan plan{};
    plan.charge_host_weight(placement_host_backing::PINNED, 100 * MiB);
    check_eq("PINNED: weight_host", plan.weight_host_bytes, 100 * MiB);
    check_eq("PINNED: host_bytes", plan.host_bytes, 100 * MiB);

    plan.charge_host_weight(placement_host_backing::HOST_CPU, 20 * MiB);
    check_eq("HOST_CPU: weight_host", plan.weight_host_bytes, 120 * MiB);
    check_eq("HOST_CPU: host_bytes", plan.host_bytes, 120 * MiB);

    plan.charge_host_weight(placement_host_backing::CPU_OWNED, 7 * MiB);
    check_eq("CPU_OWNED: cpu_owned", plan.host_cpu_owned_bytes, 7 * MiB);
    check_eq("CPU_OWNED: weight_host untouched", plan.weight_host_bytes, 120 * MiB);
    check_eq("CPU_OWNED: host_bytes", plan.host_bytes, 127 * MiB);

    plan.charge_host_weight(placement_host_backing::FILE_MAPPED, 1000 * MiB);
    check_eq("FILE_MAPPED: mapped", plan.host_mapped_bytes, 1000 * MiB);
    check_eq("FILE_MAPPED: weight_host untouched", plan.weight_host_bytes, 120 * MiB);
    check_eq("FILE_MAPPED: cpu_owned untouched", plan.host_cpu_owned_bytes, 7 * MiB);
    check_eq("FILE_MAPPED: host_bytes is not charged", plan.host_bytes, 127 * MiB);
}

static void test_placed_versus_charged() {
    printf("(c) placed = weight_host + cpu_owned + mapped; charged = weight_host + cpu_owned\n");
    placement_plan plan{};
    check_eq("empty placed", plan.placed_host_weight_bytes(), 0);
    check_eq("empty charged", plan.charged_host_weight_bytes(), 0);
    plan.charge_host_weight(placement_host_backing::PINNED, 5 * MiB);
    plan.charge_host_weight(placement_host_backing::CPU_OWNED, 3 * MiB);
    plan.charge_host_weight(placement_host_backing::FILE_MAPPED, 100 * MiB);
    check_eq("placed", plan.placed_host_weight_bytes(), 108 * MiB);
    check_eq("charged", plan.charged_host_weight_bytes(), 8 * MiB);

    // Qwen's shape: every host expert is a file-mapped, in-place AOS expert. Nothing is charged, yet the model still
    // places 100 GB on the CPU, and the tiering verdict reads "placed".
    placement_plan qwen{};
    qwen.charge_host_weight(placement_host_backing::FILE_MAPPED, 102383 * MiB);
    check_eq("qwen charged", qwen.charged_host_weight_bytes(), 0);
    check_eq("qwen weight_host", qwen.weight_host_bytes, 0);
    check_true("qwen still places weights on host", qwen.placed_host_weight_bytes() > 0);
}

static void test_uncharge_is_the_inverse_and_saturates() {
    printf("(d) uncharge_host_weight undoes charge_host_weight per backing, and saturates at 0\n");
    placement_plan plan{};
    const placement_host_backing all[] = { placement_host_backing::PINNED, placement_host_backing::FILE_MAPPED,
                                           placement_host_backing::HOST_CPU, placement_host_backing::CPU_OWNED };
    for (placement_host_backing b : all) {
        plan.charge_host_weight(b, 10 * MiB);
    }
    for (placement_host_backing b : all) {
        plan.uncharge_host_weight(b, 10 * MiB);
    }
    check_eq("weight_host", plan.weight_host_bytes, 0);
    check_eq("cpu_owned", plan.host_cpu_owned_bytes, 0);
    check_eq("mapped", plan.host_mapped_bytes, 0);
    check_eq("host_bytes", plan.host_bytes, 0);

    plan.charge_host_weight(placement_host_backing::PINNED, 4 * MiB);
    plan.uncharge_host_weight(placement_host_backing::PINNED, 9 * MiB);
    check_eq("saturating weight_host", plan.weight_host_bytes, 0);
    check_eq("saturating host_bytes", plan.host_bytes, 0);
    plan.uncharge_host_weight(placement_host_backing::FILE_MAPPED, 1 * MiB);
    check_eq("saturating mapped", plan.host_mapped_bytes, 0);
}

static placement_entry make_entry(const char *            name,
                                  size_t                  src,
                                  size_t                  dst,
                                  placement_host_backing  backing,
                                  bool                    on_device,
                                  int                     layer  = -1,
                                  int                     expert = -1) {
    placement_entry e;
    e.name         = name;
    e.src_size     = src;
    e.dst_size     = dst;
    e.host_backing = backing;
    e.on_device    = on_device;
    e.layer_id     = layer;
    e.expert_id    = expert;
    if (expert >= 0) {
        e.expert_role = expert_tensor_role::GATE;
    }
    return e;
}

static void test_a_group_is_charged_per_entry() {
    printf("(e) charge_host_weight_entries splits a mixed group by each entry's backing, on the chosen size basis\n");
    placement_plan plan{};
    plan.entries.push_back(make_entry("a", 10 * MiB, 12 * MiB, placement_host_backing::PINNED, false, 3));
    plan.entries.push_back(make_entry("b", 20 * MiB, 20 * MiB, placement_host_backing::FILE_MAPPED, false, 3));
    plan.entries.push_back(make_entry("c", 30 * MiB, 33 * MiB, placement_host_backing::CPU_OWNED, false, 3));
    plan.entries.push_back(make_entry("d", 40 * MiB, 44 * MiB, placement_host_backing::HOST_CPU, false, 3));
    const std::vector<size_t> all = { 0, 1, 2, 3 };

    placement_plan dst = plan;
    dst.charge_host_weight_entries(all, placement_size_basis::DST, 0);
    check_eq("dst: weight_host (a + d)", dst.weight_host_bytes, 56 * MiB);
    check_eq("dst: cpu_owned (c)", dst.host_cpu_owned_bytes, 33 * MiB);
    check_eq("dst: mapped (b)", dst.host_mapped_bytes, 20 * MiB);

    placement_plan src = plan;
    src.charge_host_weight_entries(all, placement_size_basis::SRC, 0);
    check_eq("src: weight_host (a + d)", src.weight_host_bytes, 50 * MiB);
    check_eq("src: cpu_owned (c)", src.host_cpu_owned_bytes, 30 * MiB);
    check_eq("src: mapped (b)", src.host_mapped_bytes, 20 * MiB);

    // host_bytes gets the caller's figure for the charged share only: the sites that charge a rounded figure keep it.
    placement_plan all_charged = plan;
    all_charged.entries[1].host_backing = placement_host_backing::PINNED;
    all_charged.charge_host_weight_entries(all, placement_size_basis::DST, 999);
    check_eq("all charged: host_bytes is the caller's figure", all_charged.host_bytes, 999);

    placement_plan none_charged = plan;
    for (auto & e : none_charged.entries) {
        e.host_backing = placement_host_backing::FILE_MAPPED;
    }
    none_charged.charge_host_weight_entries(all, placement_size_basis::DST, 999);
    check_eq("none charged: host_bytes stays 0", none_charged.host_bytes, 0);
    check_eq("none charged: mapped", none_charged.host_mapped_bytes, 109 * MiB);

    // an empty group charges no bytes, so it takes none of the figure either
    placement_plan empty = plan;
    empty.charge_host_weight_entries({}, placement_size_basis::DST, 999);
    check_eq("empty group: host_bytes stays 0", empty.host_bytes, 0);

    // a mixed group: the charged share of the caller's figure (20 of 40 bytes is charged -> 500 of 1000)
    placement_plan half{};
    half.entries.push_back(make_entry("p", 20, 20, placement_host_backing::PINNED, false));
    half.entries.push_back(make_entry("m", 20, 20, placement_host_backing::FILE_MAPPED, false));
    half.charge_host_weight_entries({ 0, 1 }, placement_size_basis::DST, 1000);
    check_eq("half charged: host_bytes is the charged share", half.host_bytes, 500);

    // an out-of-range index is skipped, as the planner's group walks do
    placement_plan stray{};
    stray.entries.push_back(make_entry("p", 8, 8, placement_host_backing::PINNED, false));
    stray.charge_host_weight_entries({ 0, 7 }, placement_size_basis::DST, 8);
    check_eq("stray index: weight_host", stray.weight_host_bytes, 8);
}

static void test_the_refresh_counts_cpu_owned_and_not_mapped() {
    printf("(f) refresh_kv_byte_totals: host_bytes = weight_host + cpu_owned + kv_host; mapped is not in it\n");
    placement_plan plan{};
    plan.layer_kind.assign(2, GGML_SYCL_KV_LAYER_FULL);
    plan.layer_k_width.assign(2, 64);
    plan.layer_v_width.assign(2, 64);
    plan.planner_n_ctx = 1024;
    // The planner records every layer's KV owner and writes -1 for a host layer; kv_layer_count() counts the
    // layers from those records, so a layer without one has no KV to total.
    plan.kv_device[0]  = -1;
    plan.kv_device[1]  = -1;
    plan.charge_host_weight(placement_host_backing::PINNED, 10 * MiB);
    plan.charge_host_weight(placement_host_backing::CPU_OWNED, 4 * MiB);
    plan.charge_host_weight(placement_host_backing::FILE_MAPPED, 500 * MiB);
    plan.refresh_kv_byte_totals();
    const size_t kv_host = plan.kv_host_bytes;
    check_true("the kv is on host (no device was assigned)", kv_host > 0);
    check_eq("host_bytes", plan.host_bytes, 14 * MiB + kv_host);
}

static void test_the_source_recount() {
    printf("(g) recount_host_mapped_source_bytes sums the device-planned FILE_MAPPED entries' file bytes\n");
    placement_plan plan{};
    plan.entries.push_back(make_entry("dev_mapped", 30 * MiB, 40 * MiB, placement_host_backing::FILE_MAPPED, true, 1));
    plan.entries.push_back(make_entry("dev_pinned", 50 * MiB, 50 * MiB, placement_host_backing::PINNED, true, 1));
    plan.entries.push_back(make_entry("host_mapped", 60 * MiB, 60 * MiB, placement_host_backing::FILE_MAPPED, false, 2));
    plan.entries.push_back(make_entry("dev_mapped2", 7 * MiB, 9 * MiB, placement_host_backing::FILE_MAPPED, true, 2));
    plan.host_mapped_source_bytes = 12345;  // stale: the recount replaces it
    plan.recount_host_mapped_source_bytes();
    check_eq("sources", plan.host_mapped_source_bytes, 37 * MiB);
    // a source is not charged and not placed on host
    check_eq("placed unchanged", plan.placed_host_weight_bytes(), 0);
    check_eq("charged unchanged", plan.charged_host_weight_bytes(), 0);
}

static void test_an_expert_moving_between_tiers_follows_its_backing() {
    printf("(h) update_expert_placement moves an expert's bytes between the VRAM terms and its backing's host term\n");
    const placement_host_backing all[] = { placement_host_backing::PINNED, placement_host_backing::HOST_CPU,
                                           placement_host_backing::CPU_OWNED, placement_host_backing::FILE_MAPPED };
    for (placement_host_backing b : all) {
        // src != dst pins the basis: the device side charges the materialized layout (dst, through
        // vram_charge_size), and the host side charges the file bytes the host executes (src), as the MoE host
        // group does.
        placement_plan plan{};
        plan.entries.push_back(make_entry("blk.0.ffn_gate_exps.weight", 64 * MiB, 80 * MiB, b, true, 0, 5));
        plan.entries[0].vram_charge_size = 80 * MiB;
        plan.entries[0].target_device    = 0;
        plan.weight_vram_bytes           = 80 * MiB;
        plan.vram_bytes                  = 80 * MiB;
        plan.build_index();
        plan.recount_host_mapped_source_bytes();
        const bool mapped = b == placement_host_backing::FILE_MAPPED;
        check_eq("on device: no host weight", plan.placed_host_weight_bytes(), 0);
        check_eq("on device: the mapped source", plan.host_mapped_source_bytes, mapped ? 64 * MiB : 0);

        check_true("demote", plan.update_expert_placement(0, 5, expert_tensor_role::GATE, false, -1));
        check_eq("demoted: weight_vram", plan.weight_vram_bytes, 0);
        const size_t host_expected = 64 * MiB;
        check_eq("demoted: placed", plan.placed_host_weight_bytes(), host_expected);
        check_eq("demoted: charged", plan.charged_host_weight_bytes(), mapped ? 0 : host_expected);
        check_eq("demoted: host_bytes", plan.host_bytes, mapped ? 0 : host_expected);
        check_eq("demoted: the source is the host weight itself", plan.host_mapped_source_bytes, 0);

        check_true("promote", plan.update_expert_placement(0, 5, expert_tensor_role::GATE, true, 0));
        check_eq("promoted: placed", plan.placed_host_weight_bytes(), 0);
        check_eq("promoted: host_bytes", plan.host_bytes, 0);
        check_eq("promoted: weight_vram", plan.weight_vram_bytes, 80 * MiB);
        check_eq("promoted: the mapped source again", plan.host_mapped_source_bytes, mapped ? 64 * MiB : 0);
    }
}

int main() {
    printf("=== the planner's host-weight accounting is split by backing (llama.cpp-5efe C4 part c) ===\n");
    test_defaults_are_pinned_and_zero();
    test_each_backing_lands_in_its_own_term();
    test_placed_versus_charged();
    test_uncharge_is_the_inverse_and_saturates();
    test_a_group_is_charged_per_entry();
    test_the_refresh_counts_cpu_owned_and_not_mapped();
    test_the_source_recount();
    test_an_expert_moving_between_tiers_follows_its_backing();
    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
