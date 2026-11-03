// Every planned weight's host backing comes from one published fact, load_maps_files (design 5efe §4.3-4.4).
//
// The inventory carries libllama's resolved use_mmap and use_mlock; the staging proc derives the fact once from them
// (use_mmap and a build with the file-mapping tier) into the planner's placement_kv_info, and the plan holds it. The
// planner gives every inventory entry its host_backing from the §4.4 table:
//
//   weight                                   load_maps_files true        load_maps_files false
//   device-planned dense or expert           FILE_MAPPED (staging src)   PINNED
//   host routed expert (AOS, in place)       FILE_MAPPED                 PINNED
//   host dense, GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu (repacked by ggml-cpu)   CPU_OWNED     PINNED
//   host dense, =plain (file layout)         FILE_MAPPED                 PINNED
//   token_embd (a row gather, file layout)   FILE_MAPPED                 PINNED
//
// The plan's terms follow from the backings: weight_host_bytes holds only PINNED / HOST_CPU host entries, so with the
// fact true it is 0 and the host WEIGHT zone is only its 64 MiB HOST_COMPUTE reserve; host_mapped_bytes holds the host
// FILE_MAPPED entries and is not charged unless -lm mlock locks them (§4.5); the DMA staging pool exists for streaming,
// which every FILE_MAPPED source refuses (§5.3), so it is 0 when no entry's host backing is PINNED (§4.4, "The DMA
// staging pool in mmap mode"). With the fact false every figure is today's.
//
// The inventory is Qwen3.8's (the 1223 SYCL weights of qwen38-q8, generated here from the GGUF's shape rules, the
// lazy per_layer_token_embd table left out as the loader leaves it out), run through the real planner under a mocked
// device table. run2 on the B70 left 4019 of the 24576 routed-expert groups on the device and 20557 on the host, which
// is where the pre-registered 102383.5 MB comes from. This host-only planner has no arena, so it cannot subtract the
// zones and KV room the B70's 30368.0 MB plan budget paid for, and a group's charge carries planner terms (the PP
// alternate reserve, the stream-ring and micro-batch holds) that no entry shows. So arm 0 finds the budget the same
// planner answers with those 4019 groups, by bisection on the device group count (first-fit over equal groups makes it
// monotone in the budget), and every later arm reads its figures at that budget.
//
// Not asserted here, because this tree has no such term yet: the SCRATCH zone (1732.0 MB with no RUNTIME-labelled or
// device-sizing term), the runtime pre-allocate (91.9 MB) and the base working set (3579.2 MB, 2 + 1 chunks) that the
// design pre-registers with this figure. Host zone selection, host scratch demand and the working set land with the
// host zone select task, which owns those figures.
//
// The §11 rule registers it gpu-serial (it runs the planner on a shaped inventory), and the lead runs it.

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-planner-test-fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ggml_sycl;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

constexpr size_t k_host_compute_reserve_bytes = 64ull * MiB;
constexpr size_t k_tlsf_grain                 = 256;

// Qwen3.8 (qwen38-q8): 48 layers, every fourth one attention, 512 routed experts per layer.
constexpr int     k_layers  = 48;
constexpr int     k_experts = 512;
constexpr int64_t k_embd    = 2560;
constexpr int64_t k_ffn     = 640;
constexpr int64_t k_vocab   = 248320;
constexpr int64_t k_hc      = 10240;
constexpr int64_t k_hc_rank = 320;

// The dump's figures (qwen38-q8-tensors.tsv less the lazy table): what the generator must reproduce.
constexpr size_t k_inventory_tensors = 1223;
constexpr size_t k_inventory_bytes   = 133813747200ull;
constexpr size_t k_expert_bytes      = 128345702400ull;

// run2's split on the B70, and the host figure it gives: 20557 groups x 3 matrices x 1740800 B = 102383.5 MB.
constexpr size_t k_device_groups        = 4019;
constexpr size_t k_host_groups          = 20557;
constexpr size_t k_expert_matrix_bytes  = 1740800;  // Q8_0, 2560 x 640: one expert of one stacked matrix
constexpr size_t k_host_mapped_bytes    = k_host_groups * 3 * k_expert_matrix_bytes;
// One stacked expert matrix (Q8_0, 2560 x 640 x 512): the largest tensor the host->device weight stream carries.
constexpr size_t k_stacked_expert_bytes = k_expert_matrix_bytes * k_experts;
constexpr size_t k_dma_pipeline_depth   = 2;

using host_planner_test::check;
using host_planner_test::check_eq;
using host_planner_test::g_checks;
using host_planner_test::g_fail;
using host_planner_test::make_mock_sycl_info;

// The mocked cards: B70-shaped discrete GPUs, the card run2 measured on, enough to plan, nothing touched.
constexpr size_t k_card_vram   = 32ull * 1024ull * MiB;
constexpr int    k_card_nsm    = 256;
constexpr char   k_card_name[] = "mock-host-backing-plan";

// --- the Qwen3.8 inventory ---

placement_tensor_info tensor(const std::string & name, ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    placement_tensor_info t(name, ggml_row_size(type, ne0) * static_cast<size_t>(ne1) * static_cast<size_t>(ne2));
    t.type  = type;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = 1;
    return t;
}

std::vector<placement_tensor_info> qwen_inventory() {
    std::vector<placement_tensor_info> inv;
    inv.push_back(tensor("output.weight", GGML_TYPE_Q8_0, k_embd, k_vocab));
    inv.push_back(tensor("output_hc_down.weight", GGML_TYPE_Q8_0, k_hc, k_hc_rank));
    inv.push_back(tensor("output_hc_norm.weight", GGML_TYPE_F32, k_hc));
    inv.push_back(tensor("output_hc_up.weight", GGML_TYPE_Q8_0, k_hc_rank, k_hc));
    // The file carries its own output head, so the loader marks token_embd a row gather only
    // (llama_model_sycl_mark_get_rows_only).
    placement_tensor_info embd = tensor("token_embd.weight", GGML_TYPE_Q8_0, k_embd, k_vocab);
    embd.get_rows_only         = true;
    inv.push_back(embd);

    for (int l = 0; l < k_layers; ++l) {
        const std::string blk       = "blk." + std::to_string(l) + ".";
        const bool        attention = (l + 1) % 4 == 0;
        if (attention) {
            inv.push_back(tensor(blk + "attn_k.weight", GGML_TYPE_Q8_0, k_embd, 512));
            inv.push_back(tensor(blk + "attn_k_norm.weight", GGML_TYPE_F32, 256));
            inv.push_back(tensor(blk + "attn_output.weight", GGML_TYPE_Q8_0, 6144, k_embd));
            inv.push_back(tensor(blk + "attn_q.weight", GGML_TYPE_Q8_0, k_embd, 12288));
            inv.push_back(tensor(blk + "attn_q_norm.weight", GGML_TYPE_F32, 256));
            inv.push_back(tensor(blk + "attn_v.weight", GGML_TYPE_Q8_0, k_embd, 512));
            inv.push_back(tensor(blk + "indexer.k_norm.weight", GGML_TYPE_F32, 128));
            inv.push_back(tensor(blk + "indexer.k_proj.weight", GGML_TYPE_BF16, k_embd, 128));
            inv.push_back(tensor(blk + "indexer.q_norm.weight", GGML_TYPE_F32, 128));
            inv.push_back(tensor(blk + "indexer.q_proj.weight", GGML_TYPE_BF16, k_embd, 512));
        } else {
            inv.push_back(tensor(blk + "attn_gate.weight", GGML_TYPE_Q8_0, k_embd, 6144));
            inv.push_back(tensor(blk + "attn_qkv.weight", GGML_TYPE_Q8_0, k_embd, 10240));
            inv.push_back(tensor(blk + "ssm_a", GGML_TYPE_F32, 48));
            inv.push_back(tensor(blk + "ssm_alpha.weight", GGML_TYPE_Q8_0, k_embd, 48));
            inv.push_back(tensor(blk + "ssm_beta.weight", GGML_TYPE_Q8_0, k_embd, 48));
            inv.push_back(tensor(blk + "ssm_conv1d.weight", GGML_TYPE_F32, 4, 10240));
            inv.push_back(tensor(blk + "ssm_dt.bias", GGML_TYPE_F32, 48));
            inv.push_back(tensor(blk + "ssm_norm.weight", GGML_TYPE_F32, 128));
            inv.push_back(tensor(blk + "ssm_out.weight", GGML_TYPE_Q8_0, 6144, k_embd));
        }
        inv.push_back(tensor(blk + "ffn_down_exps.weight", GGML_TYPE_Q8_0, k_ffn, k_embd, k_experts));
        inv.push_back(tensor(blk + "ffn_gate_exps.weight", GGML_TYPE_Q8_0, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_up_exps.weight", GGML_TYPE_Q8_0, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_down_shexp.weight", GGML_TYPE_Q8_0, k_ffn, k_embd));
        inv.push_back(tensor(blk + "ffn_gate_shexp.weight", GGML_TYPE_Q8_0, k_embd, k_ffn));
        inv.push_back(tensor(blk + "ffn_up_shexp.weight", GGML_TYPE_Q8_0, k_embd, k_ffn));
        inv.push_back(tensor(blk + "ffn_gate_inp.weight", GGML_TYPE_F32, k_embd, k_experts));
        inv.push_back(tensor(blk + "ffn_gate_inp_shexp.weight", GGML_TYPE_F32, k_embd));
        for (const char * hc : { "hc_attn_", "hc_ffn_" }) {
            inv.push_back(tensor(blk + hc + "down.weight", GGML_TYPE_Q8_0, k_hc, k_hc_rank));
            inv.push_back(tensor(blk + hc + "inject.weight", GGML_TYPE_Q8_0, k_hc, 4));
            inv.push_back(tensor(blk + hc + "norm.weight", GGML_TYPE_F32, k_hc));
            inv.push_back(tensor(blk + hc + "up.weight", GGML_TYPE_Q8_0, k_hc_rank, k_hc));
        }
        if (l == 1) {
            inv.push_back(tensor(blk + "ple_conv1d.weight", GGML_TYPE_F32, 4, 10240));
            inv.push_back(tensor(blk + "ple_key.weight", GGML_TYPE_Q8_0, k_embd, 10240));
            inv.push_back(tensor(blk + "ple_norm_conv.weight", GGML_TYPE_F32, 10240));
            inv.push_back(tensor(blk + "ple_norm_key.weight", GGML_TYPE_F32, 10240));
            inv.push_back(tensor(blk + "ple_norm_query.weight", GGML_TYPE_F32, 10240));
            inv.push_back(tensor(blk + "ple_value.weight", GGML_TYPE_Q8_0, k_embd, k_embd));
        }
    }
    return inv;
}

// --- the log lines the test reads ---

struct captured_log {
    std::string dma_line;  // "[SYCL-PLAN] DMA staging pool: ..."
};

captured_log g_log;  // reset to {} before each plan build

void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text != nullptr && std::strstr(text, "[SYCL-PLAN] DMA staging pool:")) {
        g_log.dma_line = text;
    }
}

// The fact as the planner reads it: placement_kv_info's three fields, which the staging proc derives.
// Each defaults as placement_kv_info does; an arm that models libllama's answer for GGML_SYCL_HOST_WEIGHT_LAYOUT's
// default, cpu, sets layout_cpu itself.
struct fact {
    bool maps_files  = false;
    bool locks_files = false;
    bool layout_cpu  = false;
};

placement_kv_info kv_for(const fact & f) {
    placement_kv_info kv{};
    kv.load_maps_files        = f.maps_files;
    kv.load_locks_files       = f.locks_files;
    kv.host_weight_layout_cpu = f.layout_cpu;
    return kv;
}

placement_plan plan_single(const std::vector<placement_tensor_info> & inv,
                           size_t                                     budget,
                           const fact &                               f,
                           captured_log *                             log = nullptr) {
    const ggml_sycl_device_info   info = make_mock_sycl_info(1, k_card_vram, k_card_nsm, k_card_name);
    test_sycl_info_override_guard info_guard(info);
    g_log               = {};
    placement_plan plan =
        compute_placement_plan(inv, budget, /*device_id=*/0, kv_for(f), nullptr, k_experts, host_ledger_budget());
    if (log) {
        *log = g_log;
    }
    return plan;
}

placement_plan plan_two_devices(const std::vector<placement_tensor_info> & inv, size_t budget, const fact & f) {
    const ggml_sycl_device_info      info = make_mock_sycl_info(2, k_card_vram, k_card_nsm, k_card_name);
    test_sycl_info_override_guard    info_guard(info);
    const std::vector<device_budget> budgets = {
        { 0, budget, 32ull * 1024ull * MiB, 1.0, true },
        { 1, budget, 32ull * 1024ull * MiB, 1.0, true },
    };
    return compute_multi_device_plan(budgets, inv, k_layers, multi_gpu_mode::HYBRID, kv_for(f), nullptr, k_experts,
                                     host_ledger_budget());
}

size_t device_groups(const placement_plan & plan) {
    size_t n = 0;
    for (const placement_entry & e : plan.entries) {
        n += e.on_device && e.expert_id >= 0;
    }
    return n / 3;
}

// What the plan says, counted from its entries by the test, independent of the terms under test.
struct census {
    size_t host_expert_entries      = 0;
    size_t host_expert_src          = 0;
    size_t device_expert_entries    = 0;
    size_t host_dense_entries       = 0;
    size_t device_src               = 0;  // device-planned entries' file bytes: the staging sources
    size_t host_not_file_mapped     = 0;  // host routed experts whose backing is not FILE_MAPPED
    size_t device_not_file_mapped   = 0;  // device-planned entries whose backing is not FILE_MAPPED
    size_t not_pinned               = 0;  // entries of any residency whose backing is not PINNED
    size_t host_pinned_or_host_cpu  = 0;  // host entries that hold a WEIGHT-zone block
    size_t host_cpu_owned_dense_src = 0;
    size_t host_file_mapped_dense   = 0;
    size_t host_cpu_owned_dense     = 0;
    size_t token_embd_host_backing  = SIZE_MAX;  // the backing as an integer, when token_embd is host-resident
};

census take_census(const placement_plan & plan) {
    census c;
    for (const placement_entry & e : plan.entries) {
        if (e.host_backing != placement_host_backing::PINNED) {
            c.not_pinned++;
        }
        if (e.on_device) {
            c.device_src += e.src_size;
            if (e.expert_id >= 0) {
                c.device_expert_entries++;
            }
            if (e.host_backing != placement_host_backing::FILE_MAPPED) {
                c.device_not_file_mapped++;
            }
            continue;
        }
        if (e.host_backing == placement_host_backing::PINNED || e.host_backing == placement_host_backing::HOST_CPU) {
            c.host_pinned_or_host_cpu++;
        }
        if (e.expert_id >= 0) {
            c.host_expert_entries++;
            c.host_expert_src += e.src_size;
            if (e.host_backing != placement_host_backing::FILE_MAPPED) {
                c.host_not_file_mapped++;
            }
            continue;
        }
        c.host_dense_entries++;
        if (e.name == "token_embd.weight") {
            c.token_embd_host_backing = static_cast<size_t>(e.host_backing);
            continue;
        }
        if (e.host_backing == placement_host_backing::CPU_OWNED) {
            c.host_cpu_owned_dense++;
            c.host_cpu_owned_dense_src += e.src_size;
        } else if (e.host_backing == placement_host_backing::FILE_MAPPED) {
            c.host_file_mapped_dense++;
        }
    }
    return c;
}

// Arm 0: the generator reproduces the dump, and the budget reproduces run2's split.
size_t arm0_inventory_and_budget(const std::vector<placement_tensor_info> & inv) {
    std::printf("(0) the Qwen3.8 inventory and the budget that leaves the device run2's %zu groups\n", k_device_groups);
    size_t total = 0;
    size_t exps  = 0;
    for (const placement_tensor_info & t : inv) {
        total += t.size;
        if (t.name.find("_exps") != std::string::npos) {
            exps += t.size;
        }
    }
    check_eq("inventory tensors (the dump less the lazy table)", inv.size(), k_inventory_tensors);
    check_eq("inventory bytes", total, k_inventory_bytes);
    check_eq("routed-expert bytes", exps, k_expert_bytes);

    // Bisect for the budget at which the planner leaves the device run2's groups: none fit at 1 MiB, and four times
    // the inventory holds every weight in any layout, alternates included.
    size_t lo     = 1 * MiB;
    size_t hi     = 4 * k_inventory_bytes;
    size_t budget = 0;
    int    plans  = 0;
    check_eq("device groups at the bisection's upper bound", device_groups(plan_single(inv, hi, fact{})),
             static_cast<size_t>(k_layers) * k_experts);
    while (budget == 0 && lo + 1 < hi && plans < 64) {
        const size_t mid = lo + (hi - lo) / 2;
        const size_t d   = device_groups(plan_single(inv, mid, fact{}));
        plans++;
        if (d == k_device_groups) {
            budget = mid;
        } else if (d < k_device_groups) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    check(budget != 0, "a budget leaves the device exactly run2's 4019 routed-expert groups");
    std::printf("  budget %.1f MB after %d plans (the B70 plan budget was 30368.0 MB)\n", budget / (double) MiB, plans);
    return budget;
}

// Arm 1: the fact true. Host experts are FILE_MAPPED, nothing is charged to the WEIGHT zone, and the DMA pool is 0.
void arm1_mapping_load(const std::vector<placement_tensor_info> & inv, size_t budget) {
    std::printf("(1) load_maps_files true: host experts FILE_MAPPED, weight_host_bytes 0, WEIGHT 64 MiB, DMA pool 0\n");
    fact f;
    f.maps_files = true;
    f.layout_cpu = true;
    captured_log         log;
    const placement_plan plan = plan_single(inv, budget, f, &log);
    const census         c    = take_census(plan);

    check_eq("device-planned routed-expert groups (run2's split)", c.device_expert_entries / 3, k_device_groups);
    check_eq("host routed-expert groups", c.host_expert_entries / 3, k_host_groups);
    check_eq("host routed experts whose backing is not FILE_MAPPED", c.host_not_file_mapped, 0);
    check_eq("device-planned entries whose backing is not FILE_MAPPED (their staging source)", c.device_not_file_mapped,
             0);
    check_eq("host dense entries (all fit on the device)", c.host_dense_entries, 0);
    check_eq("host entries holding a WEIGHT-zone block", c.host_pinned_or_host_cpu, 0);

    check_eq("weight_host_bytes", plan.weight_host_bytes, 0);
    check_eq("host_cpu_owned_bytes", plan.host_cpu_owned_bytes, 0);
    check_eq("host_mapped_bytes == the host experts' file bytes", plan.host_mapped_bytes, c.host_expert_src);
    check_eq("host_mapped_bytes == 102383.5 MB (20557 groups)", plan.host_mapped_bytes, k_host_mapped_bytes);
    std::printf("  host_mapped_bytes = %.1f MB\n", plan.host_mapped_bytes / (double) MiB);
    check_eq("host_mapped_source_bytes == the device-planned entries' file bytes", plan.host_mapped_source_bytes,
             c.device_src);
    check_eq("charged host weight bytes (unlocked mapped bytes are not charged)", plan.charged_host_weight_bytes(), 0);
    check_eq("placed host weight bytes", plan.placed_host_weight_bytes(), k_host_mapped_bytes);

    check_eq("host WEIGHT zone == the 64 MiB HOST_COMPUTE reserve", plan.host_zone_weight_bytes,
             k_host_compute_reserve_bytes);
    check_eq("DMA staging pool", plan.dma_staging_pool_bytes, 0);
    check(log.dma_line.find("(no pinned host weights)") != std::string::npos,
          "the DMA pool line says it is 0 because no host weight is pinned");
    if (!log.dma_line.empty()) {
        std::printf("  logged: %s", log.dma_line.c_str());
    }
}

// Arm 2: the fact false. Every entry PINNED and every figure today's: the WEIGHT zone is the host experts' exact bytes
// plus its named terms (122924.2 MB was the 1.2x figure before the zone became exact bytes plus terms), and the DMA
// pool is two stacked expert matrices, 1700 MB.
void arm2_copying_load(const std::vector<placement_tensor_info> & inv, size_t budget) {
    std::printf("(2) load_maps_files false: every entry PINNED, today's WEIGHT zone and 1700 MB DMA pool\n");
    captured_log         log;
    const placement_plan plan = plan_single(inv, budget, fact{}, &log);
    const census         c    = take_census(plan);

    check_eq("entries whose backing is not PINNED", c.not_pinned, 0);
    check_eq("host routed-expert groups", c.host_expert_entries / 3, k_host_groups);
    check_eq("weight_host_bytes == the host experts' file bytes", plan.weight_host_bytes, c.host_expert_src);
    check_eq("weight_host_bytes == 102383.5 MB", plan.weight_host_bytes, k_host_mapped_bytes);
    check_eq("host_mapped_bytes", plan.host_mapped_bytes, 0);
    check_eq("host_mapped_source_bytes", plan.host_mapped_source_bytes, 0);
    const size_t zone =
        plan.weight_host_bytes + k_host_compute_reserve_bytes + c.host_pinned_or_host_cpu * k_tlsf_grain;
    check_eq("host WEIGHT zone == host weights + 64 MiB + one TLSF grain per block", plan.host_zone_weight_bytes, zone);
    std::printf("  host WEIGHT zone = %.1f MB\n", plan.host_zone_weight_bytes / (double) MiB);
    check_eq("DMA staging pool == 2 x one stacked expert matrix (1700 MB)", plan.dma_staging_pool_bytes,
             k_dma_pipeline_depth * k_stacked_expert_bytes);
    check(log.dma_line.find("(no pinned host weights)") == std::string::npos,
          "the DMA pool line does not claim there are no pinned host weights");
}

// Arm 3: the fact true under -lm mlock. The tier locks the host-executed segments, so the mapped bytes are charged.
void arm3_locked_mapping(const std::vector<placement_tensor_info> & inv, size_t budget) {
    std::printf("(3) load_maps_files and load_locks_files true: the mapped bytes are charged\n");
    fact f;
    f.maps_files              = true;
    f.locks_files             = true;
    f.layout_cpu              = true;
    const placement_plan plan = plan_single(inv, budget, f);

    check_eq("host_mapped_bytes", plan.host_mapped_bytes, k_host_mapped_bytes);
    check_eq("charged host weight bytes == the locked mapped bytes", plan.charged_host_weight_bytes(),
             k_host_mapped_bytes);
    check(plan.host_bytes >= k_host_mapped_bytes, "host_bytes carries the locked mapped bytes");
    check_eq("weight_host_bytes", plan.weight_host_bytes, 0);
    check_eq("host WEIGHT zone (locking moves no bytes into it)", plan.host_zone_weight_bytes,
             k_host_compute_reserve_bytes);
    check_eq("DMA staging pool", plan.dma_staging_pool_bytes, 0);

    // The same plan without the lock, for the contrast: host_bytes differs by exactly the mapped bytes.
    fact unlocked;
    unlocked.maps_files            = true;
    unlocked.layout_cpu            = true;
    const placement_plan free_plan = plan_single(inv, budget, unlocked);
    check_eq("host_bytes(locked) - host_bytes(unlocked) == host_mapped_bytes", plan.host_bytes - free_plan.host_bytes,
             k_host_mapped_bytes);
}

// Arm 4: planner-host dense weights, under the two host weight layouts. A 1 MiB budget puts every weight on the host.
void arm4_host_dense(const std::vector<placement_tensor_info> & inv) {
    std::printf("(4) planner-host dense: CPU_OWNED when ggml-cpu repacks it, FILE_MAPPED in the file layout\n");
    fact repacked;
    repacked.maps_files     = true;
    repacked.layout_cpu     = true;
    const placement_plan rp = plan_single(inv, 1 * MiB, repacked);
    const census         rc = take_census(rp);
    check(rc.host_dense_entries > 1, "the starved budget puts dense weights on the host");
    check_eq("host dense entries (token_embd aside) that are CPU_OWNED under =cpu", rc.host_cpu_owned_dense,
             rc.host_dense_entries - 1);
    check_eq("token_embd's backing (a row gather in file layout) under =cpu", rc.token_embd_host_backing,
             static_cast<size_t>(placement_host_backing::FILE_MAPPED));
    check_eq("host_cpu_owned_bytes == the CPU_OWNED dense entries' bytes", rp.host_cpu_owned_bytes,
             rc.host_cpu_owned_dense_src);
    check_eq("weight_host_bytes under =cpu", rp.weight_host_bytes, 0);
    check_eq("host WEIGHT zone under =cpu", rp.host_zone_weight_bytes, k_host_compute_reserve_bytes);

    fact plain;
    plain.maps_files        = true;
    plain.layout_cpu        = false;
    const placement_plan pp = plan_single(inv, 1 * MiB, plain);
    const census         pc = take_census(pp);
    check_eq("host dense entries (token_embd aside) that are FILE_MAPPED under =plain", pc.host_file_mapped_dense,
             pc.host_dense_entries - 1);
    check_eq("token_embd's backing under =plain", pc.token_embd_host_backing,
             static_cast<size_t>(placement_host_backing::FILE_MAPPED));
    check_eq("host_cpu_owned_bytes under =plain", pp.host_cpu_owned_bytes, 0);
    check_eq("weight_host_bytes under =plain", pp.weight_host_bytes, 0);

    const placement_plan none = plan_single(inv, 1 * MiB, fact{});
    check_eq("entries whose backing is not PINNED with the fact false", take_census(none).not_pinned, 0);
}

// Arm 5: the multi-device builder derives the same backings from the same fact.
void arm5_multi_device(const std::vector<placement_tensor_info> & inv) {
    std::printf("(5) compute_multi_device_plan: the same derivation from the same fact\n");
    fact f;
    f.maps_files              = true;
    f.layout_cpu              = true;
    const placement_plan plan = plan_two_devices(inv, 1 * MiB, f);
    const census         c    = take_census(plan);
    check(c.host_expert_entries > 0, "the starved budget puts routed experts on the host");
    check_eq("host routed experts whose backing is not FILE_MAPPED", c.host_not_file_mapped, 0);
    check_eq("host entries holding a WEIGHT-zone block", c.host_pinned_or_host_cpu, 0);
    check_eq("weight_host_bytes", plan.weight_host_bytes, 0);
    check_eq("host WEIGHT zone", plan.host_zone_weight_bytes, k_host_compute_reserve_bytes);
    check_eq("DMA staging pool", plan.dma_staging_pool_bytes, 0);

    const placement_plan none = plan_two_devices(inv, 1 * MiB, fact{});
    check_eq("entries whose backing is not PINNED with the fact false", take_census(none).not_pinned, 0);
}

}  // namespace

int main() {
    std::printf("=== every planned weight's host backing comes from load_maps_files ===\n");
    ggml_log_callback prev_cb   = nullptr;
    void *            prev_data = nullptr;
    ggml_log_get(&prev_cb, &prev_data);
    ggml_log_set(capture_log, nullptr);

    // host_weight_layout_cpu has one default, the value a zeroed inventory carries (false), held by placement_kv_info
    // and this file's fact alike.
    const ggml_sycl_tensor_inventory zeroed_inventory{};
    check(placement_kv_info{}.host_weight_layout_cpu == zeroed_inventory.host_weight_layout_cpu &&
              fact{}.layout_cpu == zeroed_inventory.host_weight_layout_cpu && !zeroed_inventory.host_weight_layout_cpu,
          "host_weight_layout_cpu defaults to a zeroed inventory's false in placement_kv_info and the fact");

    const std::vector<placement_tensor_info> inv    = qwen_inventory();
    const size_t                             budget = arm0_inventory_and_budget(inv);
    if (budget != 0) {  // arms 1-3 read run2's split; without its budget they would only repeat arm 0's failure
        arm1_mapping_load(inv, budget);
        arm2_copying_load(inv, budget);
        arm3_locked_mapping(inv, budget);
    }
    arm4_host_dense(inv);
    arm5_multi_device(inv);

    ggml_log_set(prev_cb, prev_data);
    std::printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
