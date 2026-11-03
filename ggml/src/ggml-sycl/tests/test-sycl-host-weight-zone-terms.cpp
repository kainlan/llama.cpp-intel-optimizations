// The host WEIGHT zone is sized from exact bytes plus named terms, never from a ratio.
//
// The planner sizes host_zone_weight_bytes in populate_host_zone_sizing(), which both planner builders call. The
// size is a fact the plan already knows: every host-resident weight at the bytes the cache materializes for it, plus
// one named term for each real extra. Slack nobody can name is a second source for that fact, and for a DeepSeek-V4.1
// Q2_K load a 20 % ratio over ~149 GiB of host experts is ~30 GiB the host budget cannot hold.
//
// What the cache materializes for a host-resident weight (S1-PRELOAD, ggml-sycl.cpp): a host-planned expert is one
// pinned WEIGHT allocation of tensor->nb[2] bytes, a host-planned dense weight one of ggml_nbytes(tensor). The planner
// counts an entry at its AOS src_size, the inventory's ggml_nbytes and, for an expert, that over its expert count; arm
// 1 asserts entry by entry that the two agree, using the cache's expressions on tensors shaped from the inventory.
// They are two formulas for one size until one byte helper serves both the planner and S1-PRELOAD (llama.cpp-y5x9).
// The terms:
//
//   host_weights          the sum of those bytes over the host-planned entries the plan charges to
//                         weight_host_bytes (backed PINNED or HOST_CPU), one pinned block per entry; a FILE_MAPPED
//                         entry is served from the file's mapping and a CPU_OWNED one by ggml-cpu, so neither
//                         holds a block in this zone
//   host_compute_reserve  64 MiB for the runtime HOST_COMPUTE requests that also land in WEIGHT (the context's
//                         logits and embeddings host buffer, the server's prompt-checkpoint reads); the plan has no
//                         figure for those sizes, so the reserve is the term
//   layout_copies         a second, converted copy of a host weight the plan keeps beside the first: none today
//   tlsf_bound            the TLSF rounding of each block to its grain, so the block count times the grain
//
// One tenant of the zone is not a term: unplanned loader residency, the model loader's SYCL_Host buffer, which holds
// every weight the loader reads (role WEIGHT, category HOST_COMPUTE, from the buffer type the loader adds to a SYCL
// device's list when weights are evictable). Planning it would plan a duplicate copy of the weights, the defect the
// mapped host tier (SYCL_Mapped) removes. Which buffer the loader binds is the plan's fact, not the build's: on a load
// whose plan maps files the loader binds SYCL_Mapped for every FILE_MAPPED weight, whose bytes are the file's pages
// outside every host zone, so the class reads zero; on a load whose plan does not (-lm none, or a build without the
// tier) every weight keeps SYCL_Host and the class is the whole model. Arm 2 measures the second, arm 3 the first.
//
// Arm 1 (no model, no device: the real planner over a synthetic Q2_K-shaped inventory, under a mocked device table)
// asserts, for both planner builders, that the WEIGHT size is exactly the sum of those terms and is not the 1.2x
// ratio, and that the plan prints each term by name, and the unplanned class, on a [HOST-PLAN] line. It also asserts
// that the planned host weights equal the plan's weight_host_bytes for both builders, a fact about this fixture rather
// than about every plan. The single-device builder charges a host expert at its src_size. The multi-device builder
// does so in its cohesive host-layer branch, which it takes when the cards have no direct peer path, as under the
// mocked table; the arm asserts that branch's [PLACEMENT-BLOCK] line before relying on it. Its MoE group branch still
// charges a host-spilled group at the device layout's dst_size (llama.cpp-4qdx), so a plan that takes that branch can
// still trip the [HOST-PLAN-BUG] WARN.
//
// Arm 2 (the census) constructs device 0's cache, configures the host zones from that plan, and makes every planned
// WEIGHT-zone allocation through the request shapes the materialization paths use, then the loader's buffer through the
// loader's own buffer type. That plan is arm 1's, built with the fact false, so the loader's buffer is the one a load
// that does not map files takes, and the arm measures it on every build; the buffer must be allocated and its bytes
// must land in WEIGHT. Only GGML_SYCL_WEIGHTS_EVICTABLE=0 removes it: the loader then adds no SYCL_Host buffer type, so
// there is no loader buffer to measure, and the binary exits 77 naming that leg rather than passing. Each allocation is
// classed by the call that made it, and its owner metadata must agree with that class. Metadata separates host_weights
// from the HOST_COMPUTE tenants; the reserve and unplanned loader residency come from one buffer type with one request
// shape, so between those two the call alone attributes the bytes. Every byte the zone gains belongs to a named term or
// to unplanned loader residency, which is reported on its own line; any other byte fails. The terms stay within their
// bytes and the zone does not grow for them. It needs a device, so a host without one exits 77 after arms 1, 1b and 3a
// have passed.
//
// Arm 1b (no device) re-backs one host expert FILE_MAPPED and another CPU_OWNED on the single-device plan and asserts
// that neither one's bytes reach host_weights, blocks or tlsf_bound, while host_weights still equals the plan's
// weight_host_bytes. It then re-backs a third without moving its charge, and the two figures must differ: the positive
// control of the comparison the [HOST-PLAN-BUG] WARN makes.
//
// Arm 3 measures unplanned loader residency on a plan built with the fact true. 3a (no device) binds a load transaction
// with that plan staged, as the loader's begin and stage do, and asks ggml_backend_sycl_planned_host_backing for every
// weight, as select_weight_buft asks it: no weight may answer a backing that keeps SYCL_Host. Under a
// GGML_SYCL_HOST_WEIGHT_LAYOUT other than cpu every weight answers FILE_MAPPED; under cpu, the default, a host dense
// weight answers CPU_OWNED, which binds ggml-cpu's own buffer type and is reported on its own line. Its positive
// controls are the fact-false plan, whose answers keep the whole model on SYCL_Host, and a copy of the mapping plan
// whose dense entries the planner left PINNED, whose answers keep exactly those weights there. 3b (device) maps a
// stand-in model file through ggml_backend_sycl_mapped_buffer_from_file with that plan bound, addresses every weight in
// the buffer from its file offset as load_all_data does, reads one in place as the file's own bytes, and measures the
// WEIGHT zone around it: it must not grow. Its positive control is a pinned copy of the same bytes through S1-PRELOAD's
// request, which must grow the zone by at least those bytes, the effect a hook that copied into the pinned pool has.

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "../model-lifecycle.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-planner-test-fixture.hpp"
#include "sycl-test-skip.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(_WIN32)
#    include <unistd.h>
#endif

using namespace ggml_sycl;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

// The terms the test derives on its own, from the plan's entries, independent of the code under test.
constexpr size_t k_host_compute_reserve_bytes = 64ull * MiB;
constexpr size_t k_tlsf_grain                 = 256;

using host_planner_test::check;
using host_planner_test::check_eq;
using host_planner_test::g_checks;
using host_planner_test::g_fail;
using host_planner_test::make_mock_sycl_info;

// The mocked cards: B50-shaped discrete GPUs, enough to plan, nothing touched.
constexpr size_t k_card_vram   = 16ull * 1024ull * MiB;
constexpr int    k_card_nsm    = 128;
constexpr char   k_card_name[] = "mock-weight-zone-terms";

// --- a Q2_K-shaped inventory: DeepSeek's quant type, at a size a test can hold ---

constexpr int     k_layers  = 2;
constexpr int     k_experts = 8;
constexpr int64_t k_embd    = 1024;
constexpr int64_t k_ffn     = 512;
constexpr int64_t k_vocab   = 4096;

placement_tensor_info tensor(const std::string & name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    placement_tensor_info t(name, ggml_row_size(type, ne0) * static_cast<size_t>(ne1) * static_cast<size_t>(ne2));
    t.type  = type;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = 1;
    return t;
}

std::vector<placement_tensor_info> q2k_inventory() {
    std::vector<placement_tensor_info> inv;
    inv.push_back(tensor("token_embd.weight", GGML_TYPE_Q2_K, k_embd, k_vocab));
    for (int l = 0; l < k_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        inv.push_back(tensor(blk + "attn_norm.weight", GGML_TYPE_F32, k_embd, 1));
        inv.push_back(tensor(blk + "attn_q.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_k.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_v.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_output.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "ffn_norm.weight", GGML_TYPE_F32, k_embd, 1));
        inv.push_back(tensor(blk + "ffn_gate_inp.weight", GGML_TYPE_F32, k_embd, k_experts));
        inv.push_back(tensor(blk + "ffn_gate_exps.weight", GGML_TYPE_Q2_K, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_up_exps.weight", GGML_TYPE_Q2_K, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_down_exps.weight", GGML_TYPE_Q2_K, k_ffn, k_embd, k_experts));
    }
    inv.push_back(tensor("output_norm.weight", GGML_TYPE_F32, k_embd, 1));
    inv.push_back(tensor("output.weight", GGML_TYPE_Q2_K, k_embd, k_vocab));
    return inv;
}

// --- the cache's bytes: the tensors S1-PRELOAD copies, shaped from the inventory rows, never allocated ---

// The model's tensors as ggml describes them, from each inventory row's type and ne. S1-PRELOAD sizes a host
// expert's copy as tensor->nb[2] (register_host_aos_expert, through ggml_sycl_host_expert_alloc) and a host dense
// weight's as ggml_nbytes(tensor); bytes() answers with exactly those expressions, so a planner figure is compared
// with the cache's, not with a second derivation of the planner's.
struct cache_tensors {
    ggml_context *                       ctx = nullptr;
    std::map<std::string, ggml_tensor *> by_name;

    explicit cache_tensors(const std::vector<placement_tensor_info> & inventory) {
        ggml_init_params params{};
        params.mem_size   = ggml_tensor_overhead() * (inventory.size() + 1);
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        ctx               = ggml_init(params);
        for (const placement_tensor_info & row : inventory) {
            by_name[row.name] = ggml_new_tensor_3d(ctx, row.type, row.ne[0], row.ne[1], row.ne[2]);
        }
    }

    ~cache_tensors() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    cache_tensors(const cache_tensors &)             = delete;
    cache_tensors & operator=(const cache_tensors &) = delete;

    // What S1-PRELOAD allocates for `e`, or SIZE_MAX when no tensor has its name.
    size_t bytes(const placement_entry & e) const {
        const auto it = by_name.find(e.name);
        if (it == by_name.end()) {
            return SIZE_MAX;
        }
        return e.expert_id >= 0 ? static_cast<size_t>(it->second->nb[2]) : ggml_nbytes(it->second);
    }

    // What the loader's buffer holds: every tensor at ggml_nbytes.
    size_t total() const {
        size_t sum = 0;
        for (const auto & kv : by_name) {
            sum += ggml_nbytes(kv.second);
        }
        return sum;
    }
};

// Every entry's placement_entry_host_weight_bytes() is what the cache allocates for that entry.
void check_entry_bytes(const char * builder, const placement_plan & plan, const cache_tensors & tensors) {
    size_t n_expert   = 0;
    size_t n_dense    = 0;
    size_t unknown    = 0;
    size_t mismatched = 0;
    for (const placement_entry & e : plan.entries) {
        const size_t cache_bytes = tensors.bytes(e);
        if (cache_bytes == SIZE_MAX) {
            unknown++;
            continue;
        }
        (e.expert_id >= 0 ? n_expert : n_dense)++;
        const size_t planned = placement_entry_host_weight_bytes(e);
        if (planned != cache_bytes) {
            if (mismatched < 4) {
                std::printf("  FAIL detail: %s expert=%d planned=%zu cache=%zu\n", e.name.c_str(), e.expert_id, planned,
                            cache_bytes);
            }
            mismatched++;
        }
    }
    std::printf("  [%s] %zu expert entries against nb[2], %zu dense entries against ggml_nbytes\n", builder, n_expert,
                n_dense);
    check(n_expert > 0 && n_dense > 0, "the plan has both expert and dense entries to compare");
    check_eq("entries with no tensor of their name", unknown, 0);
    check_eq("entries whose planned host bytes are not the cache's", mismatched, 0);
}

// --- the log lines the test reads, captured from the log callback ---

// What one plan build logged, of the lines the test reads. capture_log is the one filter list: a line the test starts
// reading is a field here and a branch there, matched on the text the code prints.
struct captured_log {
    std::string host_plan_line;         // "[HOST-PLAN] device=%d WEIGHT zone: ..."
    bool        host_plan_bug = false;  // a "[HOST-PLAN-BUG] ..." line was logged
    std::string cohesive_host_line;     // "[PLACEMENT-BLOCK] cohesive host layers=%zu reason=..."
};

captured_log g_log;  // reset to {} before each plan build

void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text == nullptr) {
        return;
    }
    if (std::strstr(text, "[HOST-PLAN]") && std::strstr(text, "WEIGHT zone")) {
        g_log.host_plan_line = text;
    } else if (std::strstr(text, "[HOST-PLAN-BUG]")) {
        g_log.host_plan_bug = true;
    } else if (std::strstr(text, "[PLACEMENT-BLOCK] cohesive host layers=")) {
        g_log.cohesive_host_line = text;
    }
}

// The value printed after "<name>=" on the line, or SIZE_MAX when the name is absent.
size_t printed_term(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    if (at == std::string::npos) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(std::strtoull(line.c_str() + at + key.size(), nullptr, 10));
}

struct derived_terms {
    size_t host_weights = 0;
    size_t blocks       = 0;

    size_t total() const { return host_weights + k_host_compute_reserve_bytes + blocks * k_tlsf_grain; }
};

// The backings whose bytes placement_plan::charge_host_weight adds to weight_host_bytes, written out here rather than
// asked of the predicate under test, so the oracle stays independent of it.
bool backing_held_in_weight_zone(placement_host_backing backing) {
    return backing == placement_host_backing::PINNED || backing == placement_host_backing::HOST_CPU;
}

derived_terms derive_terms(const placement_plan & plan) {
    derived_terms t;
    for (const placement_entry & e : plan.entries) {
        if (e.on_device || !backing_held_in_weight_zone(e.host_backing)) {
            continue;
        }
        t.host_weights += e.src_size;
        t.blocks++;
    }
    return t;
}

// `kv` carries the load's host-backing fact; the default is a load that does not map files.
placement_plan plan_q2k(const std::vector<placement_tensor_info> & inventory, const placement_kv_info & kv = {}) {
    const ggml_sycl_device_info   info = make_mock_sycl_info(1, k_card_vram, k_card_nsm, k_card_name);
    test_sycl_info_override_guard info_guard(info);
    // A budget far below the experts: every expert, and with them the dense weights, is host-resident.
    return compute_placement_plan(inventory, /*vram_budget=*/1 * MiB, /*device_id=*/0, kv, nullptr, k_experts);
}

// The other planner builder, over two cards with the same starved budget.
placement_plan plan_q2k_two_devices(const std::vector<placement_tensor_info> & inventory) {
    const ggml_sycl_device_info      info = make_mock_sycl_info(2, k_card_vram, k_card_nsm, k_card_name);
    test_sycl_info_override_guard    info_guard(info);
    placement_kv_info                kv{};
    const std::vector<device_budget> budgets = {
        { 0, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true },
        { 1, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true },
    };
    return compute_multi_device_plan(budgets, inventory, k_layers, multi_gpu_mode::HYBRID, kv, nullptr, k_experts);
}

// The checks one plan's WEIGHT zone must pass, whichever builder made it. `log` is what that plan's build logged.
//
// host_weights == weight_host_bytes is asserted for both builders, as a fact about this fixture. The single-device
// builder charges a host expert at its AOS src_size, the bytes the cache materializes. The multi-device builder does
// so only in its cohesive host-layer branch; its MoE group branch charges a host-spilled group at the device layout's
// dst_size (llama.cpp-4qdx). For that builder, `needs_cohesive_branch`, the equality rests on the branch, so the
// branch's [PLACEMENT-BLOCK] line is asserted and printed first.
void check_weight_zone(const char *           builder,
                       const placement_plan & plan,
                       const captured_log &   log,
                       bool                   needs_cohesive_branch) {
    std::printf("  [%s]\n", builder);
    const std::string & line   = log.host_plan_line;
    const bool          warned = log.host_plan_bug;
    if (needs_cohesive_branch) {
        check(!log.cohesive_host_line.empty(),
              "the plan took the cohesive host-layer branch, which charges host experts at src_size");
        if (!log.cohesive_host_line.empty()) {
            std::printf("  logged: %s", log.cohesive_host_line.c_str());
        }
    }
    const derived_terms t                = derive_terms(plan);
    const size_t        n_expert_entries = static_cast<size_t>(k_layers) * 3 * k_experts;
    check(t.blocks >= n_expert_entries, "every routed expert is host-resident under the 1 MiB budget");
    check(t.host_weights > 0, "the plan placed weights on the host");
    // The [HOST-PLAN-BUG] WARN fires exactly when the entries and the plan's running total disagree. Both builders'
    // figures agree on this fixture, so each is a negative control of the WARN; arm 1b's mis-charged plan is the
    // comparison's positive control.
    const bool figures_differ = t.host_weights != plan.weight_host_bytes;
    std::printf("  [HOST-PLAN-BUG] control: figures %s, WARN %s\n", figures_differ ? "differ" : "agree",
                warned ? "logged" : "silent");
    check(warned == figures_differ, "[HOST-PLAN-BUG] is logged exactly when host_weights != weight_host_bytes");
    // The plan's own host-weight accounting is the same fact: the term sums what the cache materializes.
    check_eq("host_weights == the plan's weight_host_bytes", t.host_weights, plan.weight_host_bytes);

    const size_t ratio_size =
        static_cast<size_t>(static_cast<double>(t.host_weights) * 1.2) + k_host_compute_reserve_bytes;
    // Vacuity guard: the two answers must differ on this inventory, or "not 1.2x" would prove nothing.
    check(ratio_size != t.total(), "the fixture separates exact-plus-terms from the 1.2x ratio");
    check_eq("host_zone_weight_bytes == host_weights + host_compute_reserve + layout_copies + tlsf_bound",
             plan.host_zone_weight_bytes, t.total());
    check(plan.host_zone_weight_bytes != ratio_size, "host_zone_weight_bytes is not the 1.2x ratio");

    // The builder sized the zone through the one terms function.
    const host_weight_zone_terms terms = placement_plan_host_weight_zone_terms(plan);
    check_eq("placement_plan_host_weight_zone_terms().total() is the planned size", terms.total(),
             plan.host_zone_weight_bytes);
    check_eq("terms.host_weights", terms.host_weights, t.host_weights);
    check_eq("terms.blocks", terms.blocks, t.blocks);

    check(!line.empty(), "a [HOST-PLAN] WEIGHT zone line was logged");
    if (!line.empty()) {
        std::printf("  logged: %s", line.c_str());
        check_eq("[HOST-PLAN] host_weights", printed_term(line, "host_weights"), t.host_weights);
        check_eq("[HOST-PLAN] blocks", printed_term(line, "blocks"), t.blocks);
        check_eq("[HOST-PLAN] host_compute_reserve", printed_term(line, "host_compute_reserve"),
                 k_host_compute_reserve_bytes);
        check_eq("[HOST-PLAN] layout_copies", printed_term(line, "layout_copies"), 0);
        check_eq("[HOST-PLAN] tlsf_bound", printed_term(line, "tlsf_bound"), t.blocks * k_tlsf_grain);
        check_eq("[HOST-PLAN] total", printed_term(line, "total"), t.total());
        check(line.find("unplanned: loader_residency") != std::string::npos,
              "[HOST-PLAN] names unplanned loader residency as outside the terms");
    }
}

// Arm 1: the WEIGHT zone is the exact bytes plus the named terms, and not 1.2x, from both planner builders, and every
// entry's planned bytes are the bytes the cache allocates for it.
void arm1_weight_zone_is_exact_bytes_plus_named_terms(const std::vector<placement_tensor_info> & inventory,
                                                      const cache_tensors &                      tensors,
                                                      placement_plan &                           out_plan) {
    std::printf("(1) a Q2_K-shaped inventory's WEIGHT zone is exact bytes plus named terms, not 1.2x\n");

    ggml_log_callback prev_cb   = nullptr;
    void *            prev_data = nullptr;
    ggml_log_get(&prev_cb, &prev_data);
    ggml_log_set(capture_log, nullptr);
    g_log                                 = {};
    const placement_plan single_line_plan = plan_q2k(inventory);
    const captured_log   single_log       = g_log;
    g_log                                 = {};
    const placement_plan multi_plan       = plan_q2k_two_devices(inventory);
    const captured_log   multi_log        = g_log;

    // The fallback the planner takes for an expert row whose inventory shape was not populated: it splits the row by
    // the model-wide expert count. Here that count is the tensor's own ne[2], as it is for every architecture but
    // one, so the fallback still meets nb[2]; the check on each row below pins that equality.
    std::vector<placement_tensor_info> unshaped        = inventory;
    size_t                             n_unshaped_rows = 0;
    for (placement_tensor_info & row : unshaped) {
        if (row.name.find("_exps") != std::string::npos) {
            check_eq("the model-wide expert count plan_q2k passes equals the unshaped row's own ne[2]",
                     static_cast<size_t>(k_experts), static_cast<size_t>(row.ne[2]));
            n_unshaped_rows++;
            for (int64_t & n : row.ne) {
                n = 0;
            }
        }
    }
    check(n_unshaped_rows > 0, "the fallback case has expert rows to unshape");
    const placement_plan fallback_plan = plan_q2k(unshaped);
    ggml_log_set(prev_cb, prev_data);

    check_weight_zone("compute_placement_plan", single_line_plan, single_log, false);
    check_weight_zone("compute_multi_device_plan", multi_plan, multi_log, true);

    std::printf("  planned host bytes against the cache's expressions\n");
    check_entry_bytes("compute_placement_plan", single_line_plan, tensors);
    check_entry_bytes("compute_multi_device_plan", multi_plan, tensors);
    check_entry_bytes("compute_placement_plan, expert rows without a shape", fallback_plan, tensors);
    // Not asserted: an unshaped expert row whose own expert count is not the model-wide one (grovemoe's chexps trio,
    // ne[2] = n_expert / n_group_experts). The fallback splits it by the model-wide count, so its planned bytes are
    // nb[2] x ne[2] / n_expert, not nb[2]. No [HOST-PLAN-BUG] WARN fires for it either: the walked sum and the plan's
    // weight_host_bytes both carry that same src_size. It agrees once the inventory always carries ne[2]
    // (llama.cpp-y5x9).
    out_plan = single_line_plan;
}

// Arm 1b: the walk counts only the backings that hold a block in the WEIGHT zone.
//
// Every built plan backs its entries PINNED today, since nothing plans a FILE_MAPPED or CPU_OWNED weight yet, so the
// arm re-backs two of the plan's host experts itself. It moves each one's charge through the plan's own
// uncharge_host_weight and charge_host_weight, as a planner assigning the backing would. populate_host_zone_sizing()
// is file-static and cannot run again on the re-backed plan, so the arm asserts the comparison its [HOST-PLAN-BUG]
// WARN makes rather than the WARN itself.
void arm1b_walk_counts_only_weight_zone_backings(const placement_plan & built) {
    std::printf("(1b) a FILE_MAPPED and a CPU_OWNED host expert stay out of the WEIGHT terms\n");

    placement_plan    plan   = built;
    placement_entry * mapped = nullptr;
    placement_entry * owned  = nullptr;
    for (placement_entry & e : plan.entries) {
        if (e.on_device || e.expert_id < 0 || e.host_backing != placement_host_backing::PINNED) {
            continue;
        }
        if (mapped == nullptr) {
            mapped = &e;
        } else if (owned == nullptr) {
            owned = &e;
        }
    }
    check(mapped != nullptr && owned != nullptr, "the plan has two PINNED host experts to re-back");
    if (mapped == nullptr || owned == nullptr) {
        return;
    }
    const size_t mapped_bytes = mapped->src_size;
    const size_t owned_bytes  = owned->src_size;
    check(mapped_bytes > 0 && owned_bytes > 0, "both re-backed experts carry bytes");

    const derived_terms before = derive_terms(plan);
    check_eq("the built plan's host_weights == its weight_host_bytes", before.host_weights, plan.weight_host_bytes);

    const auto reback = [&plan](placement_entry & e, placement_host_backing backing) {
        plan.uncharge_host_weight(e.host_backing, e.src_size);
        e.host_backing = backing;
        plan.charge_host_weight(e.host_backing, e.src_size);
    };
    reback(*mapped, placement_host_backing::FILE_MAPPED);
    reback(*owned, placement_host_backing::CPU_OWNED);
    check_eq("the FILE_MAPPED bytes moved to host_mapped_bytes", plan.host_mapped_bytes,
             built.host_mapped_bytes + mapped_bytes);
    check_eq("the CPU_OWNED bytes moved to host_cpu_owned_bytes", plan.host_cpu_owned_bytes,
             built.host_cpu_owned_bytes + owned_bytes);

    const host_weight_zone_terms terms = placement_plan_host_weight_zone_terms(plan);
    check_eq("host_weights leaves out the FILE_MAPPED and CPU_OWNED bytes", terms.host_weights,
             before.host_weights - mapped_bytes - owned_bytes);
    check_eq("blocks leaves out the two re-backed experts", terms.blocks, before.blocks - 2);
    check_eq("tlsf_bound covers the remaining blocks only", terms.tlsf_bound, (before.blocks - 2) * k_tlsf_grain);
    check_eq("the terms match the test's own derivation", terms.total(), derive_terms(plan).total());

    const char * figures = terms.host_weights != plan.weight_host_bytes ? "differ" : "agree";
    std::printf("  [HOST-PLAN-BUG] comparison on the re-backed plan: figures %s\n", figures);
    check_eq("host_weights == weight_host_bytes on the re-backed plan, so the WARN stays silent", terms.host_weights,
             plan.weight_host_bytes);

    // The comparison's positive control, which needs no planner defect: re-back one more host expert without moving
    // its charge, the mistake the WARN exists to catch, and the two figures must differ. The expert is restored after.
    placement_entry * mis_charged = nullptr;
    for (placement_entry & e : plan.entries) {
        if (!e.on_device && e.expert_id >= 0 && e.host_backing == placement_host_backing::PINNED) {
            mis_charged = &e;
            break;
        }
    }
    bool mis_charged_differs = false;
    if (mis_charged != nullptr) {
        mis_charged->host_backing = placement_host_backing::FILE_MAPPED;
        const size_t walked       = placement_plan_host_weight_zone_terms(plan).host_weights;
        mis_charged->host_backing = placement_host_backing::PINNED;
        mis_charged_differs       = walked != plan.weight_host_bytes;
    }
    std::printf("  [HOST-PLAN-BUG] comparison on a mis-charged plan: figures %s\n",
                mis_charged_differs ? "differ" : "agree");
    check(mis_charged_differs, "a host expert re-backed without moving its charge makes the figures differ");
}

// --- arm 2: the census ---

// The classes a WEIGHT-zone allocation can belong to: one per named term it comes from, and the one tenant the plan
// deliberately does not size.
enum class weight_class { HOST_WEIGHTS, HOST_COMPUTE_RESERVE, LOADER_RESIDENCY, COUNT };

const char * weight_class_name(weight_class cls) {
    switch (cls) {
        case weight_class::HOST_WEIGHTS:
            return "host_weights";
        case weight_class::HOST_COMPUTE_RESERVE:
            return "host_compute_reserve";
        case weight_class::LOADER_RESIDENCY:
            return "loader_residency";
        case weight_class::COUNT:
            break;
    }
    return "?";
}

// Whether an allocation's owner metadata is what its origin's request makes. The class comes from the call the census
// made; metadata that disagrees with it means the allocation is not what that origin allocates, and it has no class.
// The metadata separates host_weights from the HOST_COMPUTE tenants only: with weights evictable the reserve and the
// loader's buffer both carry role WEIGHT, category HOST_COMPUTE and cohort "backend_host_buffer", because both come
// from the SYCL_Host buffer type, so between those two classes the call alone attributes the bytes.
bool metadata_fits(weight_class cls, const alloc_metadata & md) {
    switch (cls) {
        case weight_class::HOST_WEIGHTS:
            return md.role == alloc_role::WEIGHT && md.category != runtime_category::HOST_COMPUTE;
        case weight_class::HOST_COMPUTE_RESERVE:
            return md.category == runtime_category::HOST_COMPUTE;
        case weight_class::LOADER_RESIDENCY:
            return md.role == alloc_role::WEIGHT && md.category == runtime_category::HOST_COMPUTE;
        case weight_class::COUNT:
            break;
    }
    return false;
}

// The request S1-PRELOAD makes for a host-planned weight (ggml_sycl_host_expert_alloc for an expert, the dense copy's
// unified_alloc for a dense weight): a pinned WEIGHT allocation of the entry's bytes.
alloc_request host_weight_request(sycl::queue & q, size_t bytes) {
    alloc_request req{};
    req.queue                               = &q;
    req.device                              = 0;
    req.size                                = bytes;
    req.intent.role                         = alloc_role::WEIGHT;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    return req;
}

// The request a SYCL_Host buffer makes after the model load (the context's output buffer): category HOST_COMPUTE,
// which routes to WEIGHT whatever the role. Its role is the buffer type's: WEIGHT while weights are evictable (the
// default), STAGING otherwise, the same choice ggml_backend_sycl_host_buffer_type_alloc_buffer makes outside a load.
// The census uses it only when the loader adds no SYCL_Host buffer type; otherwise it allocates through that buffer
// type itself, so the request cannot drift from the real one.
alloc_request host_compute_request(sycl::queue & q, size_t bytes) {
    const alloc_role role = ggml_backend_sycl_weights_evictable() ? alloc_role::WEIGHT : alloc_role::STAGING;
    alloc_request    req{};
    req.queue                               = &q;
    req.device                              = 0;
    req.size                                = bytes;
    req.intent.role                         = role;
    req.intent.category                     = runtime_category::HOST_COMPUTE;
    req.intent.cohort_id                    = "backend_host_buffer";
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    return req;
}

struct weight_census {
    static constexpr int n_classes = static_cast<int>(weight_class::COUNT);

    unified_cache * cache                  = nullptr;
    size_t          class_bytes[n_classes] = {};
    size_t          unclassed_bytes = 0;  // bytes the zone gained from an allocation whose metadata fits no class
    size_t          not_in_weight   = 0;  // allocations of a term that did not land in WEIGHT
    size_t          failed          = 0;  // allocations of a term that were refused

    // Classes one allocation that has already been made: `landed` is what the WEIGHT zone gained for it.
    void record(weight_class cls, const alloc_metadata & md, size_t landed) {
        if (md.host_zone != host_zone_id::WEIGHT) {
            unclassed_bytes += landed;  // the zone grew for something that is not in it
            if (cls != weight_class::LOADER_RESIDENCY) {
                not_in_weight++;
            }
            return;
        }
        if (!metadata_fits(cls, md)) {
            unclassed_bytes += landed;
            std::printf("  FAIL detail: a %s allocation carries role=%d category=%d\n", weight_class_name(cls),
                        static_cast<int>(md.role), static_cast<int>(md.category));
            return;
        }
        class_bytes[static_cast<int>(cls)] += landed;
    }

    size_t bytes(weight_class cls) const { return class_bytes[static_cast<int>(cls)]; }

    size_t used() const { return cache->host_zone_used(host_zone_id::WEIGHT); }
};

// `measured_loader` is set once the loader's buffer has been taken and measured.
bool arm2_every_weight_zone_allocation_has_a_term(const placement_plan & plan,
                                                  const cache_tensors &  tensors,
                                                  bool &                 measured_loader) {
    std::printf("(2) census: every WEIGHT-zone byte belongs to a named term or to unplanned loader residency\n");
    measured_loader = false;
    std::optional<sycl::device> dev_opt =
        sycl_test_require_gpu("the WEIGHT-zone census (arms 1, 1b and 3a DID run and their results stand)");
    if (!dev_opt) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
    check(reg != nullptr && ggml_backend_reg_dev_count(reg) > 0, "the SYCL backend registers a device");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return true;
    }
    ggml_backend_dev_t dev0 = ggml_backend_reg_dev_get(reg, 0);

    // Device 0's cache, the one the loader's buffer type allocates from, so every allocation below shares one zone.
    weight_census census;
    census.cache = get_unified_cache_for_device(0);
    check(census.cache != nullptr, "device 0's unified cache is available");
    if (!census.cache) {
        return true;
    }
    unified_cache & cache = *census.cache;
    sycl::queue &   q     = cache.get_queue();
    if (!cache.host_zones_configured()) {
        cache.configure_host_zones(plan.host_zone_weight_bytes, 2 * MiB, 2 * MiB, 2 * MiB);
    }
    check(cache.host_zones_configured(), "host zones configured from the plan");
    if (!cache.host_zones_configured()) {
        return true;
    }

    const size_t capacity_before = cache.host_zone_capacity(host_zone_id::WEIGHT);
    const size_t used_before     = census.used();
    check(capacity_before >= plan.host_zone_weight_bytes, "the WEIGHT zone holds at least the plan's size");

    // The loader's buffer below is the one a load whose plan does not map files takes.
    check(!plan.host_backing_maps_files,
          "the census sizes from a plan whose fact is false, the load that keeps SYCL_Host");

    const derived_terms     t = derive_terms(plan);
    std::vector<mem_handle> held;

    // The SYCL_Host buffer type the loader adds to a SYCL device's list while weights are evictable: the context's
    // output buffer and the loader's own buffer both come from it. Only GGML_SYCL_WEIGHTS_EVICTABLE=0 leaves it out;
    // a current device with evictable weights always has one.
    const bool                 evictable = ggml_backend_sycl_weights_evictable();
    ggml_backend_buffer_type_t host_buft = evictable ? ggml_backend_sycl_host_buffer_type_for_device(dev0) : nullptr;
    check(!evictable || host_buft != nullptr, "device 0 has a SYCL_Host buffer type while weights are evictable");

    auto take = [&](weight_class cls, const alloc_request & req) {
        const size_t before = census.used();
        mem_handle   h      = unified_allocate(req);
        if (!h.valid()) {
            census.failed++;
            return;
        }
        alloc_metadata md{};
        if (!h.owner_metadata(&md)) {
            md.host_zone = host_zone_id::COUNT;
        }
        census.record(cls, md, census.used() - before);
        held.push_back(std::move(h));
    };

    // One buffer of `bytes` through `buft`, classed by its registered allocation's metadata; nullptr when refused.
    auto take_buffer = [&](weight_class cls, ggml_backend_buffer_type_t buft, size_t bytes) -> ggml_backend_buffer_t {
        const size_t          before = census.used();
        ggml_backend_buffer_t buf    = ggml_backend_buft_alloc_buffer(buft, bytes);
        if (buf == nullptr) {
            return nullptr;
        }
        alloc_metadata md{};
        if (!unified_lookup_runtime_allocation(ggml_backend_buffer_get_base(buf), &md)) {
            md.host_zone = host_zone_id::COUNT;
        }
        census.record(cls, md, census.used() - before);
        return buf;
    };

    // The terms: every planned host weight at the bytes the cache allocates for it, so the census compares cache bytes
    // with the planner's terms, then a logits-sized HOST_COMPUTE request the reserve covers.
    for (const placement_entry & e : plan.entries) {
        if (e.on_device) {
            continue;
        }
        const size_t cache_bytes = tensors.bytes(e);
        if (cache_bytes == SIZE_MAX) {
            census.failed++;
            continue;
        }
        take(weight_class::HOST_WEIGHTS, host_weight_request(q, cache_bytes));
    }
    ggml_backend_buffer_t reserve_buf = nullptr;
    if (host_buft != nullptr) {
        reserve_buf = take_buffer(weight_class::HOST_COMPUTE_RESERVE, host_buft, 8 * MiB);
        if (reserve_buf == nullptr) {
            census.failed++;
        }
    } else {
        take(weight_class::HOST_COMPUTE_RESERVE, host_compute_request(q, 8 * MiB));
    }

    check_eq("planned WEIGHT allocations that failed", census.failed, 0);
    check_eq("planned WEIGHT allocations that landed outside WEIGHT", census.not_in_weight, 0);
    const size_t host_weights_bytes = census.bytes(weight_class::HOST_WEIGHTS);
    const size_t reserve_bytes      = census.bytes(weight_class::HOST_COMPUTE_RESERVE);
    check(host_weights_bytes >= t.host_weights, "the host_weights term's allocations hold at least its exact bytes");
    check(host_weights_bytes <= t.host_weights + t.blocks * k_tlsf_grain,
          "the host_weights allocations stay within host_weights + tlsf_bound");
    check(reserve_bytes <= k_host_compute_reserve_bytes,
          "the HOST_COMPUTE allocations stay within host_compute_reserve");
    check_eq("the WEIGHT zone did not grow for the terms", cache.host_zone_capacity(host_zone_id::WEIGHT),
             capacity_before);
    check(census.used() - used_before <= plan.host_zone_weight_bytes, "the terms fit the planned WEIGHT zone");

    // The unplanned class on a load whose plan does not map files: the loader keeps SYCL_Host for every weight
    // (design 5efe §4.3), so its buffer comes through that same buffer type, sized as the loader sizes it, every
    // weight it reads. Arm 3 measures the class on a plan whose fact is true.
    const size_t          loader_requested = tensors.total();
    ggml_backend_buffer_t loader_buf       = nullptr;
    size_t                loader_grew_by   = 0;
    if (host_buft != nullptr) {
        const size_t capacity_pre = cache.host_zone_capacity(host_zone_id::WEIGHT);
        loader_buf                = take_buffer(weight_class::LOADER_RESIDENCY, host_buft, loader_requested);
        loader_grew_by            = cache.host_zone_capacity(host_zone_id::WEIGHT) - capacity_pre;
        check(loader_buf != nullptr, "the loader's SYCL_Host buffer was allocated");
    }
    const size_t loader_bytes = census.bytes(weight_class::LOADER_RESIDENCY);
    if (host_buft != nullptr) {
        check(loader_bytes > 0, "the loader's SYCL_Host buffer lands in WEIGHT as unplanned loader residency");
        measured_loader = true;
    }

    check_eq("WEIGHT bytes in neither a named term nor unplanned loader residency", census.unclassed_bytes, 0);
    check_eq("every WEIGHT byte gained is accounted to a class", census.used() - used_before,
             host_weights_bytes + reserve_bytes + loader_bytes);
    std::printf(
        "  census terms: host_weights=%zu (term %zu + tlsf %zu) host_compute_reserve=%zu (term %zu) blocks=%zu\n",
        host_weights_bytes, t.host_weights, t.blocks * k_tlsf_grain, reserve_bytes, k_host_compute_reserve_bytes,
        t.blocks);
    if (host_buft == nullptr) {
        std::printf("  loader leg not run: %s\n",
                    evictable ? "device 0 has no SYCL_Host buffer type" :
                                "GGML_SYCL_WEIGHTS_EVICTABLE=0, so the loader adds no SYCL_Host buffer type");
    } else if (loader_buf == nullptr) {
        std::printf("  unplanned loader_residency=0 B (the loader's %zu B buffer was refused)\n", loader_requested);
    } else {
        std::printf("  unplanned loader_residency=%zu B (loader buffer %zu B; WEIGHT grew %zu B for it)\n",
                    loader_bytes, loader_requested, loader_grew_by);
    }

    if (loader_buf != nullptr) {
        ggml_backend_buffer_free(loader_buf);
    }
    if (reserve_buf != nullptr) {
        ggml_backend_buffer_free(reserve_buf);
    }
    held.clear();
    check_eq("the census released every WEIGHT allocation", census.used(), used_before);
    return true;
}

// --- arm 3: unplanned loader residency on a load whose plan maps files ---

// A load transaction bound to this thread with `plan` staged as its candidate, as the loader's begin and stage make
// it, so ggml_backend_sycl_planned_host_backing and the from-file hook answer from that plan. It ends as a failed
// load, which publishes nothing.
struct bound_load {
    lifecycle::Registry & registry = lifecycle::global_registry();
    lifecycle::LoadTxnId  txn{};
    bool                  active = false;

    bound_load(placement_plan plan, const placement_kv_info & kv) {
        const lifecycle::begin_result begin = registry.begin_outer();
        if (begin.code != lifecycle::error::OK) {
            return;
        }
        txn = begin.txn;
        registry.bind_candidate(txn);
        lifecycle_stage_placement_plan(txn.value, std::move(plan), kv);
        active = true;
    }

    ~bound_load() {
        if (!active) {
            return;
        }
        registry.unbind_candidate(txn);
        lifecycle_abort_placement_plan(txn.value);
        (void) registry.end(txn, false);
    }

    bound_load(const bound_load &)             = delete;
    bound_load & operator=(const bound_load &) = delete;
};

const char * host_backing_name(ggml_sycl_host_backing backing) {
    switch (backing) {
        case GGML_SYCL_HOST_BACKING_PINNED:
            return "PINNED";
        case GGML_SYCL_HOST_BACKING_FILE_MAPPED:
            return "FILE_MAPPED";
        case GGML_SYCL_HOST_BACKING_HOST_CPU:
            return "HOST_CPU";
        case GGML_SYCL_HOST_BACKING_CPU_OWNED:
            return "CPU_OWNED";
    }
    return "?";
}

// The loader's answers over every weight it reads, summed by what each answer binds (design 5efe §4.3, the
// select_weight_buft row): FILE_MAPPED binds SYCL_Mapped, the file's pages; PINNED and HOST_CPU keep SYCL_Host, the
// loader's buffer in WEIGHT; CPU_OWNED binds ggml-cpu's own buffer type, and keeps SYCL_Host only where no such type
// holds the weight. unknown sums an answer outside the enum, and default_answer is the transaction default, the answer
// to a query with no name. bound is false when no load transaction could begin.
struct loader_answers {
    bool                   bound          = false;
    size_t                 sycl_host      = 0;
    size_t                 file_mapped    = 0;
    size_t                 cpu_owned      = 0;
    size_t                 unknown        = 0;
    ggml_sycl_host_backing default_answer = GGML_SYCL_HOST_BACKING_PINNED;
};

// Stages `plan` for a bound load and asks ggml_backend_sycl_planned_host_backing about every tensor, as
// select_weight_buft asks it.
loader_answers ask_loader(const placement_plan & plan, const placement_kv_info & kv, const cache_tensors & tensors) {
    loader_answers a;
    bound_load     load(plan, kv);
    a.bound = load.active;
    if (!load.active) {
        return a;
    }
    for (const auto & named : tensors.by_name) {
        const size_t bytes = ggml_nbytes(named.second);
        switch (ggml_backend_sycl_planned_host_backing(named.first.c_str())) {
            case GGML_SYCL_HOST_BACKING_PINNED:
            case GGML_SYCL_HOST_BACKING_HOST_CPU:
                a.sycl_host += bytes;
                break;
            case GGML_SYCL_HOST_BACKING_FILE_MAPPED:
                a.file_mapped += bytes;
                break;
            case GGML_SYCL_HOST_BACKING_CPU_OWNED:
                a.cpu_owned += bytes;
                break;
            default:
                a.unknown += bytes;
                break;
        }
    }
    a.default_answer = ggml_backend_sycl_planned_host_backing(nullptr);
    return a;
}

void print_answers(const char * what, const loader_answers & a) {
    std::printf("  [%s] SYCL_Host=%zu B SYCL_Mapped=%zu B cpu_owned=%zu B default=%s\n", what, a.sycl_host,
                a.file_mapped, a.cpu_owned, host_backing_name(a.default_answer));
}

// The load arm 3b maps: the fact true under a host weight layout other than cpu, so every weight is FILE_MAPPED.
struct mapping_load {
    placement_kv_info kv{};
    placement_plan    plan{};
};

// Arm 3a: on a load whose plan maps files, no weight answers a backing that keeps SYCL_Host.
mapping_load arm3a_mapping_load_keeps_no_weight_on_sycl_host(const std::vector<placement_tensor_info> & inventory,
                                                             const cache_tensors &                      tensors,
                                                             const placement_plan & fact_false_plan) {
    std::printf("(3a) the loader's answers on a load whose plan maps files keep no weight on SYCL_Host\n");
    const size_t total = tensors.total();

    // Positive control of the probe: arm 2's plan, the fact false, keeps every weight on SYCL_Host.
    const loader_answers off = ask_loader(fact_false_plan, placement_kv_info{}, tensors);
    check(off.bound, "a load transaction binds with the fact-false plan staged");
    print_answers("fact false", off);
    check_eq("fact false: the answers keep every weight on SYCL_Host", off.sycl_host, total);
    check(off.default_answer == GGML_SYCL_HOST_BACKING_PINNED, "fact false: the transaction default is PINNED");

    // The fact true under a layout other than cpu: every weight, a device-planned one included, is FILE_MAPPED.
    mapping_load plain;
    plain.kv.load_maps_files        = true;
    plain.kv.host_weight_layout_cpu = false;
    plain.plan                      = plan_q2k(inventory, plain.kv);
    check(plain.plan.host_backing_maps_files, "the builder copied the fact onto the plan");
    const loader_answers on = ask_loader(plain.plan, plain.kv, tensors);
    check(on.bound, "a load transaction binds with the mapping plan staged");
    print_answers("fact true, layout not cpu", on);
    check_eq("fact true: no weight answers a backing that keeps SYCL_Host", on.sycl_host, 0);
    check_eq("fact true, layout not cpu: every weight answers FILE_MAPPED", on.file_mapped, total);
    check_eq("fact true: no answer outside the enum", on.unknown, 0);
    check(on.default_answer == GGML_SYCL_HOST_BACKING_FILE_MAPPED, "fact true: the transaction default is FILE_MAPPED");

    // The fact true under the default layout, cpu: a host dense weight is CPU_OWNED. It binds ggml-cpu's own buffer
    // type, and the loader keeps SYCL_Host for one only where no such type holds it, which the host zone select
    // decides, so its bytes are reported apart from the class rather than counted as zero.
    placement_kv_info cpu_kv{};
    cpu_kv.load_maps_files        = true;
    cpu_kv.host_weight_layout_cpu = true;
    const loader_answers cpu      = ask_loader(plan_q2k(inventory, cpu_kv), cpu_kv, tensors);
    check(cpu.bound, "a load transaction binds with the default-layout mapping plan staged");
    print_answers("fact true, layout cpu", cpu);
    check_eq("fact true, layout cpu: no weight answers a backing that keeps SYCL_Host", cpu.sycl_host, 0);
    check(cpu.cpu_owned > 0, "fact true, layout cpu: a host dense weight answers CPU_OWNED");
    check_eq("fact true, layout cpu: every weight answers FILE_MAPPED or CPU_OWNED", cpu.file_mapped + cpu.cpu_owned,
             total);

    // Positive control of the zero, the planner defect it exists to catch: the mapping plan with its dense entries
    // left PINNED. A routed expert is not name-indexed and keeps the transaction default, so exactly the dense
    // weights must come back on SYCL_Host.
    placement_plan left_pinned = plain.plan;
    size_t         dense_bytes = 0;
    for (placement_entry & e : left_pinned.entries) {
        if (e.expert_id >= 0) {
            continue;
        }
        e.host_backing     = placement_host_backing::PINNED;
        const size_t bytes = tensors.bytes(e);
        dense_bytes += bytes == SIZE_MAX ? 0 : bytes;
    }
    const loader_answers pinned = ask_loader(left_pinned, plain.kv, tensors);
    check(pinned.bound, "a load transaction binds with the left-PINNED plan staged");
    print_answers("fact true, dense entries left PINNED", pinned);
    check(dense_bytes > 0, "the plan has dense weights to leave PINNED");
    check_eq("dense entries left PINNED come back on SYCL_Host", pinned.sycl_host, dense_bytes);
    return plain;
}

#if !defined(_WIN32)
// A stand-in model file: unlinked, `size` bytes, zero except where the arm writes. -1 when it cannot be made.
int make_model_file(size_t size) {
    const char * dir  = std::getenv("TMPDIR");
    std::string  path = std::string(dir != nullptr && dir[0] != '\0' ? dir : "/tmp") + "/zone-terms-model-XXXXXX";
    const int    fd   = mkstemp(path.data());
    if (fd < 0) {
        return -1;
    }
    unlink(path.c_str());
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}
#endif

// Arm 3b: the SYCL_Mapped buffer the loader builds on that load takes no WEIGHT-zone byte.
//
// The weights are addressed in the buffer from their file offsets, as load_all_data addresses them, but not bound
// into it: binding runs the buffer's init_tensor, which registers each weight with the bound load, and only a model's
// load end releases those rows. The bytes a weight reads are the buffer's bytes at that address either way.
void arm3b_mapped_buffer_takes_no_weight_zone_bytes(const mapping_load &                       load,
                                                    const std::vector<placement_tensor_info> & inventory) {
    std::printf("(3b) the loader's SYCL_Mapped buffer on that load takes no WEIGHT-zone byte\n");
#if defined(_WIN32)
    GGML_UNUSED(load);
    GGML_UNUSED(inventory);
    std::printf("  not run: this build has no file-mapping tier, so no load maps files\n");
#else
    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return;  // arm 2 already failed for it
    }
    ggml_backend_dev_t dev0  = ggml_backend_reg_dev_get(reg, 0);
    unified_cache *    cache = get_unified_cache_for_device(0);
    if (cache == nullptr || !cache->host_zones_configured()) {
        return;  // arm 2 already failed for it
    }
    unified_cache &            zones       = *cache;
    ggml_backend_buffer_type_t mapped_buft = ggml_backend_sycl_mapped_buffer_type_for_device(dev0);
    check(mapped_buft != nullptr, "device 0 has a SYCL_Mapped buffer type");
    if (mapped_buft == nullptr) {
        return;
    }

    // The weights' ranges in the stand-in file, in name order at GGUF's default 32-byte alignment: every weight,
    // since on this load every one answers FILE_MAPPED and binds SYCL_Mapped.
    std::vector<ggml_sycl_mapped_tensor_range> ranges;
    uint64_t                                   file_size = 0;
    for (const auto & row : inventory) {
        file_size = (file_size + 31) / 32 * 32;
        ranges.push_back({ row.name.c_str(), file_size, static_cast<uint64_t>(row.size) });
        file_size += row.size;
    }
    size_t mapped_bytes = 0;
    for (const ggml_sycl_mapped_tensor_range & r : ranges) {
        mapped_bytes += r.size;
    }

    // The probe the first weight must read in place: bytes written to the file, never to the buffer.
    const char   probe[]   = "zone-terms-probe";
    const size_t probe_len = sizeof(probe) - 1;
    const int    fd        = make_model_file(file_size);
    check(fd >= 0, "a stand-in model file was made");
    if (fd < 0) {
        return;
    }
    const bool probe_written =
        pwrite(fd, probe, probe_len, static_cast<off_t>(ranges[0].file_offset)) == static_cast<ssize_t>(probe_len);
    check(probe_written, "the probe bytes were written to the file");

    // Positive control of the measurement: a pinned copy of the same bytes through S1-PRELOAD's request, what a hook
    // that copied the weights into the pinned pool would allocate, grows the zone by at least those bytes.
    {
        const size_t before = zones.host_zone_used(host_zone_id::WEIGHT);
        mem_handle   copy   = unified_allocate(host_weight_request(zones.get_queue(), mapped_bytes));
        const size_t after  = zones.host_zone_used(host_zone_id::WEIGHT);
        const size_t grew   = after > before ? after - before : 0;
        check(copy.valid(), "the pinned copy of the mapped bytes was allocated");
        check(grew >= mapped_bytes, "a pinned copy of the mapped bytes grows the WEIGHT zone by at least those bytes");
        std::printf("  positive control: a pinned copy of %zu B grew WEIGHT by %zu B\n", mapped_bytes, grew);
    }

    {
        bound_load bound(load.plan, load.kv);
        check(bound.active, "a load transaction binds with the mapping plan staged");
        if (!bound.active) {
            close(fd);
            return;
        }

        constexpr int n_zones = static_cast<int>(host_zone_id::COUNT);
        size_t        used_before[n_zones];
        for (int z = 0; z < n_zones; ++z) {
            used_before[z] = zones.host_zone_used(static_cast<host_zone_id>(z));
        }
        const size_t weight_capacity_before = zones.host_zone_capacity(host_zone_id::WEIGHT);

        uint64_t              base_file_offset = 0;
        ggml_backend_buffer_t buf              = ggml_backend_sycl_mapped_buffer_from_file(
            dev0, /*file_idx=*/0, fd, file_size, ranges.data(), ranges.size(), nullptr, 0, &base_file_offset);
        check(buf != nullptr, "the from-file hook maps the file for the bound load");
        if (buf != nullptr) {
            check(ggml_backend_buffer_get_type(buf) == mapped_buft, "the buffer is device 0's SYCL_Mapped");
            check(ggml_backend_buffer_is_host(buf), "the CPU reads the buffer");
            alloc_metadata md{};
            const bool     registered = unified_lookup_runtime_allocation(ggml_backend_buffer_get_base(buf), &md);
            check(registered && md.host_zone != host_zone_id::WEIGHT,
                  "the buffer's registered allocation is in no WEIGHT zone");

            // Every weight's range lies in the buffer at base + (offset - base offset), load_all_data's address.
            const uint8_t * base     = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(buf));
            const size_t    size     = ggml_backend_buffer_get_size(buf);
            size_t          n_inside = 0;
            for (const ggml_sycl_mapped_tensor_range & r : ranges) {
                if (r.file_offset >= base_file_offset && r.file_offset - base_file_offset + r.size <= size) {
                    n_inside++;
                }
            }
            check_eq("weights whose range lies in the SYCL_Mapped buffer", n_inside, ranges.size());
            const bool probe_inside = ranges[0].file_offset >= base_file_offset &&
                                      ranges[0].file_offset - base_file_offset + probe_len <= size;
            check(probe_inside && std::memcmp(base + (ranges[0].file_offset - base_file_offset), probe, probe_len) == 0,
                  "the first weight reads the file's bytes in place");

            constexpr int weight_zone = static_cast<int>(host_zone_id::WEIGHT);
            size_t        weight_grew = 0;
            size_t        other_grew  = 0;
            for (int z = 0; z < n_zones; ++z) {
                const size_t used = zones.host_zone_used(static_cast<host_zone_id>(z));
                const size_t grew = used > used_before[z] ? used - used_before[z] : 0;
                (z == weight_zone ? weight_grew : other_grew) += grew;
            }
            check_eq("unplanned loader residency on a load whose plan maps files", weight_grew, 0);
            check_eq("the WEIGHT zone did not grow for the mapped buffer",
                     zones.host_zone_capacity(host_zone_id::WEIGHT), weight_capacity_before);
            check_eq("no other host zone gained a byte for the mapped buffer", other_grew, 0);
            std::printf(
                "  unplanned loader_residency=%zu B on a load whose plan maps files (SYCL_Mapped buffer %zu B "
                "of file pages for %zu B of weights)\n",
                weight_grew, size, mapped_bytes);
            ggml_backend_buffer_free(buf);
        }
    }
    close(fd);
#endif
}

}  // namespace

int main() {
    std::printf("=== the host WEIGHT zone is exact bytes plus named terms ===\n");
    // One chunk holds every zone, so the census measures TLSF rounding and not where a chunk ends.
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "512", /*overwrite=*/0);

    const std::vector<placement_tensor_info> inventory = q2k_inventory();
    const cache_tensors                      tensors(inventory);

    placement_plan plan{};
    arm1_weight_zone_is_exact_bytes_plus_named_terms(inventory, tensors, plan);
    arm1b_walk_counts_only_weight_zone_backings(plan);
    const mapping_load mapping = arm3a_mapping_load_keeps_no_weight_on_sycl_host(inventory, tensors, plan);

    bool       measured_loader = false;
    const bool ran_census      = arm2_every_weight_zone_allocation_has_a_term(plan, tensors, measured_loader);
    if (ran_census) {
        arm3b_mapped_buffer_takes_no_weight_zone_bytes(mapping, inventory);
    }

    std::printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        std::printf("FAILED\n");
        return 1;
    }
    if (!ran_census) {
        std::printf("SKIP: arms 1, 1b and 3a passed; arms 2 and 3b need a device and proved nothing\n");
        return SYCL_TEST_SKIP;
    }
    if (!measured_loader) {
        std::printf(
            "SKIP: every arm that ran passed, but arm 2's loader leg did not run "
            "(GGML_SYCL_WEIGHTS_EVICTABLE=0: the loader adds no SYCL_Host buffer type), so unplanned loader "
            "residency on a load that does not map files was not measured\n");
        return SYCL_TEST_SKIP;
    }
    std::printf("PASSED\n");
    return 0;
}
