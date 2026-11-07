//
// The host-only plan dry run (th32 §4.2 "One source for the number"): ggml_backend_sycl_plan_dry_run plans an
// inventory through the planner the stage calls, prints every host term by name from that planner's own plan, and
// stores nothing. T2's host_planned_bytes, the dry run's printout and 9g22's expected_host are one computation.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The planner's shaped path initialises SYCL (layout_policy::get_optimal calls ggml_sycl_info()), arm (a)
// stages a load through real SYCL backends, and the backends construct caches, so the binary is registered
// gpu-serial, serial, with the selector pinned to the discrete cards, and a host with no SYCL GPU exits 77 before any
// arm runs (th32 §11).
//
// Every arm names its engagement check, the observation that proves it reached the mechanism it scores, and the
// mutant it kills. An arm whose engagement check fails is VOID, and a VOID arm fails the binary like a FAIL. Every arm
// runs, so one failure does not hide the rest.

#include "../ggml-sycl-test.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-test-harness.hpp"
#include "host-plan-shaped-inventory.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using host_plan_shaped_inventory::loader_inventory;
using host_plan_shaped_inventory::qwen38_shape;

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;
constexpr uint64_t GiB = 1024ull * MiB;

const char * const k_prefix = "[PLAN-DRY-RUN]";

// The captured lines that contain `a`, and `b` too when it is given, in the order they were logged.
std::vector<std::string> lines_with(const char * a, const char * b = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::vector<std::string>    out;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            out.push_back(line);
        }
    }
    return out;
}

uint64_t stores() {
    uint64_t n = 0;
    (void) ggml_sycl_test_host_ledger_stores(&n);
    return n;
}

ggml_sycl_plan_dry_run_opts default_opts() {
    ggml_sycl_plan_dry_run_opts opts{};
    opts.struct_size = sizeof(opts);
    opts.version     = GGML_SYCL_PLAN_DRY_RUN_VERSION;
    return opts;
}

// The result is a few KiB, so it lives on the heap.
std::unique_ptr<ggml_sycl_plan_dry_run_result> fresh_result() {
    auto r         = std::make_unique<ggml_sycl_plan_dry_run_result>();
    *r             = ggml_sycl_plan_dry_run_result{};
    r->struct_size = sizeof(ggml_sycl_plan_dry_run_result);
    r->version     = GGML_SYCL_PLAN_DRY_RUN_VERSION;
    return r;
}

uint64_t residency_entries(const ggml_sycl_plan_dry_run_plan & p) {
    uint64_t n = 0;
    for (int b = 0; b < GGML_SYCL_PLAN_DRY_RUN_BACKING_COUNT; ++b) {
        n += p.residency_count[b];
    }
    return n;
}

// Zeroes the per-device planner inputs a stage publishes from an inventory (the PP pipeline scratch and the PP MoE
// oneDNN ring), so a dry run that follows plans from what it publishes itself, not from what the stage left.
void withdraw_published_inputs() {
    for (int d = 0; d < test_physical_device_count() && d < GGML_SYCL_MAX_DEVICES; ++d) {
        unified_cache_set_planned_pp_pipeline_scratch_bytes(d, 0);
        unified_cache_set_planned_pp_moe_onednn_scratch(d, 0, 0, 0, 0);
    }
}

// The per-device globals a dry run writes while it plans: the planner inputs it publishes from the inventory (the PP
// pipeline scratch and the PP MoE oneDNN ring) and the shapes the planner publishes from each plan (the oneDNN Graph
// scratch and non-FA attention shapes, the MoE CONTROL requirement, the MoE host scatter scratch). A later load reads
// them, so the dry run must leave each as it found it.
struct device_planner_state {
    size_t                             pp_pipeline = 0;
    size_t                             ring_weight = 0;
    size_t                             ring_act    = 0;
    size_t                             ring_out    = 0;
    uint32_t                           ring_depth  = 0;
    onednn_graph_scratch_planned_shape graph{};
    nonfa_attn_scratch_planned_shape   nonfa{};
    moe_control_requirement            control{};
    size_t                             scatter = 0;

    bool operator==(const device_planner_state & o) const {
        return pp_pipeline == o.pp_pipeline && ring_weight == o.ring_weight && ring_act == o.ring_act &&
               ring_out == o.ring_out && ring_depth == o.ring_depth && graph.n_head_ctx_max == o.graph.n_head_ctx_max &&
               graph.n_head_swa_max == o.graph.n_head_swa_max && graph.n_swa == o.graph.n_swa &&
               graph.n_ubatch == o.graph.n_ubatch && graph.n_ctx == o.graph.n_ctx && nonfa.n_head == o.nonfa.n_head &&
               nonfa.n_ubatch == o.nonfa.n_ubatch && nonfa.n_ctx == o.nonfa.n_ctx && control.bytes == o.control.bytes &&
               control.valid == o.control.valid && scatter == o.scatter;
    }
};

std::vector<device_planner_state> read_planner_state() {
    std::vector<device_planner_state> out;
    for (int d = 0; d < test_physical_device_count() && d < GGML_SYCL_MAX_DEVICES; ++d) {
        device_planner_state s;
        s.pp_pipeline = unified_cache_get_planned_pp_pipeline_scratch_bytes(d);
        s.ring_weight = unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(d);
        s.ring_act    = unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(d);
        s.ring_out    = unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(d);
        s.ring_depth  = unified_cache_get_planned_pp_moe_onednn_ring_depth(d);
        s.graph       = unified_cache_get_planned_onednn_graph_scratch_shape(d);
        s.nonfa       = unified_cache_get_planned_nonfa_attn_scratch_shape(d);
        s.control     = unified_cache_get_planned_moe_control_requirement(d);
        s.scatter     = unified_cache_get_planned_moe_host_scatter_scratch_bytes(d);
        out.push_back(s);
    }
    return out;
}

void write_planner_state(const std::vector<device_planner_state> & state) {
    for (size_t d = 0; d < state.size(); ++d) {
        const int                    dev = static_cast<int>(d);
        const device_planner_state & s   = state[d];
        unified_cache_set_planned_pp_pipeline_scratch_bytes(dev, s.pp_pipeline);
        unified_cache_set_planned_pp_moe_onednn_scratch(dev, s.ring_weight, s.ring_act, s.ring_out, s.ring_depth);
        unified_cache_set_planned_onednn_graph_scratch_shape(dev, s.graph.n_head_ctx_max, s.graph.n_head_swa_max,
                                                             s.graph.n_swa, s.graph.n_ubatch, s.graph.n_ctx);
        unified_cache_set_planned_nonfa_attn_scratch_shape(dev, s.nonfa.n_head, s.nonfa.n_ubatch, s.nonfa.n_ctx);
        unified_cache_set_planned_moe_control_requirement(dev, s.control);
        unified_cache_set_planned_moe_host_scatter_scratch_bytes(dev, s.scatter);
    }
}

// A value per device no plan of the shaped inventories publishes, so a write the dry run left behind shows. Each must
// be one its setter stores as given: the ring's setter stores the effective depth, one slot for any request, so the
// sentinel's depth is 1 and its slot sizes carry the difference.
std::vector<device_planner_state> sentinel_planner_state(size_t n) {
    std::vector<device_planner_state> out(n);
    for (size_t d = 0; d < n; ++d) {
        device_planner_state & s = out[d];
        const uint32_t         v = 7 + static_cast<uint32_t>(d);
        s.pp_pipeline            = 3 * MiB + d;
        s.ring_weight            = 5 * MiB + d;
        s.ring_act               = 256 * v;
        s.ring_out               = 512 * v;
        s.ring_depth             = 1;
        s.graph                  = { v, v, v, v, v };
        s.nonfa                  = { v, v, v };
        s.control                = { 4096 + d, true };
        s.scatter                = 8192 + d;
    }
    return out;
}

// (a) expected_host is the stage's host_planned_bytes. An early stage run in a load transaction plans the
// Qwen3.8-shaped inventory on each device the scheduler exposes, and its planner records host_planned_bytes on one
// [HOST-LEDGER] plan line per plan; the dry run over the same inventory, against the budgets the stage planned against
// (the same process, so the same caches and arena zones), reports the same figure for each plan, in the same order.
// Engagement: the stage run returned OK with at least one plan line, the dry run returned OK with as many plans, every
// dry-run plan put at least one weight on the host (the shape's experts overflow any card here), and its RUNTIME term
// carries the inventory's PP MoE ring, which the dry run published itself after the stage's publish was withdrawn.
// Mutants: a term dropped from or added to expected_host (it differs from the record by that term); the dry run not
// publishing the inventory's planner inputs (RUNTIME and expected_host lose the ring); a budget other than the stage's
// (the host weights, and with them expected_host, move).
arm_result arm_a_expected_host_is_the_stage_figure() {
    arm_result       r;
    loader_inventory inv(qwen38_shape);

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    log_clear();
    const ggml_sycl_lifecycle_result stage_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, nullptr, true);
    const std::vector<std::string>   stage    = lines_with("[HOST-LEDGER] device=", " plan host_budget=");
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (stage_rc != GGML_SYCL_LIFECYCLE_OK || stage.empty()) {
        void_arm(r, "the early stage run returned " + num(stage_rc) + " with " + num(stage.size()) + " plan lines");
        return r;
    }

    withdraw_published_inputs();
    ggml_sycl_plan_dry_run_opts      opts = default_opts();
    auto                             res  = fresh_result();
    const ggml_sycl_lifecycle_result rc   = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    if (rc != GGML_SYCL_LIFECYCLE_OK || res->n_plans != stage.size()) {
        void_arm(r, "the dry run returned " + num(rc) + " with " + num(res->n_plans) + " plans; the stage made " +
                        num(stage.size()));
        return r;
    }
    const size_t ring = inv.inv.pp_moe_onednn_scratch_bytes;
    for (uint32_t i = 0; i < res->n_plans; ++i) {
        const ggml_sycl_plan_dry_run_plan & p    = res->plans[i];
        const std::string                   name = "plan " + num(i) + " (device " + std::to_string(p.device) + ")";
        if (residency_entries(p) == 0) {
            void_arm(r, name + " put no weight on the host, so the WEIGHT terms were not exercised");
        }
        if (p.runtime < ring) {
            void_arm(r, name + ": RUNTIME " + num(p.runtime) + " does not carry the inventory's ring of " + num(ring));
        }
        // Each side samples MemAvailable itself, so the two budgets can differ by the host's drift. A budget above the
        // plan's whole host figure did not bind it, so the drift cannot move the plan; one at or below it may have,
        // and then the comparison is not like for like.
        const size_t stage_budget = printed(stage[i], "host_budget");
        const size_t low_budget   = std::min<size_t>(stage_budget, p.host_budget);
        if (stage_budget != p.host_budget && p.expected_host >= low_budget) {
            void_arm(r, name + ": the host budget binds (" + num(p.expected_host) + " host bytes) and moved between " +
                            "the stage (" + num(stage_budget) + ") and the dry run (" + num(p.host_budget) + ")");
        }
        expect(r, p.device == static_cast<int32_t>(printed(stage[i], "device")),
               name + ": the stage's plan line is for device " + num(printed(stage[i], "device")));
        expect(r, p.expected_host == printed(stage[i], "host_planned_bytes"),
               name + ": expected_host " + num(p.expected_host) + ", the stage's host_planned_bytes " +
                   num(printed(stage[i], "host_planned_bytes")));
    }
    return r;
}

// One printed figure: the dry run's own line for `term` on plan `plan`. Exactly one such line must exist.
size_t dry_run_figure(arm_result & r, uint32_t plan, const char * term) {
    const std::string              tag = " plan=" + num(plan) + " ";
    const std::string              key = std::string(" ") + term + "=";
    std::vector<std::string>       found;
    const std::vector<std::string> lines = lines_with(k_prefix, tag.c_str());
    for (const std::string & line : lines) {
        if (line.find(key) != std::string::npos) {
            found.push_back(line);
        }
    }
    if (found.size() != 1) {
        fail(r, "plan " + num(plan) + ": " + num(found.size()) + " printed lines carry " + term + ", want 1");
        return SIZE_MAX;
    }
    return printed(found[0], term);
}

placement_entry hand_entry(const char * name, size_t src, size_t dst, placement_host_backing backing, bool on_device) {
    placement_entry e;
    e.name         = name;
    e.src_size     = src;
    e.dst_size     = dst;
    e.host_backing = backing;
    e.on_device    = on_device;
    return e;
}

// The printed residency line of `backing` for hand-built plan `index`: exactly one must exist.
std::string residency_line(arm_result & r, uint32_t index, const char * backing) {
    const std::string              which = " plan=" + num(index) + " ";
    const std::string              tag   = std::string(" backing=") + backing + " ";
    const std::vector<std::string> lines = lines_with(k_prefix, tag.c_str());
    std::vector<std::string>       found;
    for (const std::string & line : lines) {
        if (line.find(which) != std::string::npos) {
            found.push_back(line);
        }
    }
    if (found.size() != 1) {
        fail(r,
             "hand-built plan " + num(index) + ": " + num(found.size()) + " " + backing + " residency lines, want 1");
        return std::string();
    }
    return found[0];
}

// Arm (b)'s second leg: the residency of a hand-built plan with one host entry of each backing and one device entry.
// The PINNED and HOST_CPU entries are charged on the SRC basis, so weight_host_bytes is their per-entry sum; the
// FILE_MAPPED and CPU_OWNED entries on the DST basis, and each field then grows by a further charge, as a planner's
// layout growth charges one, so neither field is the SRC or the DST sum of its entries.
void residency_from_the_plan_fields(arm_result & r) {
    constexpr uint32_t agree    = 90;
    constexpr uint32_t disagree = 91;
    placement_plan     plan{};
    plan.device_id = 0;
    plan.entries.push_back(hand_entry("pinned", 10 * MiB, 12 * MiB, placement_host_backing::PINNED, false));
    plan.entries.push_back(hand_entry("host_cpu", 4 * MiB, 4 * MiB, placement_host_backing::HOST_CPU, false));
    plan.entries.push_back(hand_entry("mapped", 20 * MiB, 24 * MiB, placement_host_backing::FILE_MAPPED, false));
    plan.entries.push_back(hand_entry("cpu_owned", 30 * MiB, 33 * MiB, placement_host_backing::CPU_OWNED, false));
    plan.entries.push_back(hand_entry("device", 8 * MiB, 8 * MiB, placement_host_backing::PINNED, true));
    plan.charge_host_weight_entries({ 2, 3 }, placement_size_basis::DST, 0);
    plan.charge_host_weight(placement_host_backing::FILE_MAPPED, 1 * MiB);
    plan.charge_host_weight(placement_host_backing::CPU_OWNED, 2 * MiB);
    placement_plan dst_charged = plan;
    plan.charge_host_weight_entries({ 0, 1 }, placement_size_basis::SRC, 0);
    dst_charged.charge_host_weight_entries({ 0, 1 }, placement_size_basis::DST, 0);
    if (plan.host_mapped_bytes == 20 * MiB || plan.host_mapped_bytes == 24 * MiB ||
        plan.host_cpu_owned_bytes == 30 * MiB || plan.host_cpu_owned_bytes == 33 * MiB) {
        void_arm(r,
                 "the hand-built plan's FILE_MAPPED or CPU_OWNED field is a per-entry sum, so a recomputed figure "
                 "would not show");
        return;
    }

    log_clear();
    const ggml_sycl_plan_dry_run_plan out   = test_plan_dry_run_record(agree, plan);
    const std::vector<std::string>    bug_a = lines_with("[PLAN-DRY-RUN-BUG]", " the PINNED and HOST_CPU entries ");

    struct want {
        int          backing;
        const char * name;
        uint64_t     bytes;
    };

    const want wants[] = {
        { GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED,      "PINNED",      10 * MiB                  },
        { GGML_SYCL_PLAN_DRY_RUN_BACKING_FILE_MAPPED, "FILE_MAPPED", plan.host_mapped_bytes    },
        { GGML_SYCL_PLAN_DRY_RUN_BACKING_HOST_CPU,    "HOST_CPU",    4 * MiB                   },
        { GGML_SYCL_PLAN_DRY_RUN_BACKING_CPU_OWNED,   "CPU_OWNED",   plan.host_cpu_owned_bytes },
    };
    for (const want & w : wants) {
        if (out.residency_count[w.backing] != 1) {
            void_arm(r, std::string("the hand-built plan's record counted ") + num(out.residency_count[w.backing]) +
                            " " + w.name + " entries, not its one");
            continue;
        }
        expect(r, out.residency_bytes[w.backing] == w.bytes,
               std::string("hand-built plan: the result's ") + w.name + " bytes are " +
                   num(out.residency_bytes[w.backing]) + ", the plan's " + num(w.bytes));
        const std::string line = residency_line(r, agree, w.name);
        expect(r, line.empty() || printed(line, "bytes") == w.bytes,
               std::string("hand-built plan: printed ") + w.name + " bytes=" + num(printed(line, "bytes")) +
                   ", the plan's " + num(w.bytes));
    }
    expect(r, bug_a.empty(),
           "the plan whose weight_host_bytes is its PINNED and HOST_CPU sum logged " + num(bug_a.size()) +
               " residency disagreement lines");
    expect(r, out.weight_host_bytes == plan.weight_host_bytes,
           "hand-built plan: the result's weight_host_bytes is " + num(out.weight_host_bytes) + ", the plan's " +
               num(plan.weight_host_bytes));

    log_clear();
    const ggml_sycl_plan_dry_run_plan out_d = test_plan_dry_run_record(disagree, dst_charged);
    const std::vector<std::string>    bug_d = lines_with("[PLAN-DRY-RUN-BUG]", " the PINNED and HOST_CPU entries ");
    expect(r, out_d.weight_host_bytes == dst_charged.weight_host_bytes,
           "the plan charged on the DST basis: the result's weight_host_bytes is " + num(out_d.weight_host_bytes) +
               ", the plan's " + num(dst_charged.weight_host_bytes));
    expect(r, bug_d.size() == 1 && bug_d[0].find(" plan=" + num(disagree) + " ") != std::string::npos,
           "the plan charged on the DST basis logged " + num(bug_d.size()) + " residency disagreement lines, want 1");
}

// (b) The printout names every term on its own line, and each figure is the plan's. The planner prints the plan's own
// record of each term: the WEIGHT zone's terms on its [HOST-PLAN] line, the commit terms, host_planned_bytes and the
// budget the plan recorded on its [HOST-LEDGER] plan line. The dry run prints each on a line of its own, under the
// term's name, and each must equal the planner's. The dry run logs no [PLAN-DRY-RUN-BUG] line for these plans. The
// PINNED and HOST_CPU residency is scored against the plan: those are the entries the WEIGHT walk counts, so their
// count is its blocks, and their bytes are the plan's own charge for them, weight_host_bytes, which the result
// carries. The shaped inventory maps no file (use_mmap_resolved is false), so its plan keeps every host entry PINNED,
// and a second leg scores the residency on a plan built by hand, through the record's private seam: its FILE_MAPPED
// and CPU_OWNED figures are the plan's host_mapped_bytes and host_cpu_owned_bytes, which the fixture charges apart from
// either per-entry sum, its weight_host_bytes is the plan's, and a plan whose weight_host_bytes is charged on the DST
// basis gets the one [PLAN-DRY-RUN-BUG] line the plan charged on the SRC basis does not.
// Engagement: the dry run returned OK, and each plan has exactly one planner line of each kind to score against; the
// hand-built plan's record counted one entry of each backing, and its two field figures differ from both the SRC and
// the DST sums of their entries.
// Mutants: a figure printed from a recomputed value (it differs from the planner's record: a rounding from another
// chunk, a WEIGHT total re-summed, a budget re-sampled, a residency byte figure re-summed per entry on either basis); a
// term not printed, or two terms on one line (the line count for that term is not 1); a residency line walking another
// set of entries; PINNED and HOST_CPU bytes that part from the plan's weight_host_bytes, or a result that does not
// carry that field; the disagreement WARN dropped, or raised on an agreeing plan.
arm_result arm_b_every_term_printed_from_the_plan() {
    arm_result       r;
    loader_inventory inv(qwen38_shape);

    ggml_sycl_plan_dry_run_opts opts = default_opts();
    auto                        res  = fresh_result();
    log_clear();
    const ggml_sycl_lifecycle_result rc     = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    const std::vector<std::string>   ledger = lines_with("[HOST-LEDGER] device=", " plan host_budget=");
    const std::vector<std::string>   weight = lines_with("[HOST-PLAN] device=", " WEIGHT zone:");
    if (rc != GGML_SYCL_LIFECYCLE_OK || res->n_plans == 0 || ledger.size() != res->n_plans ||
        weight.size() != res->n_plans) {
        void_arm(r, "the dry run returned " + num(rc) + " with " + num(res->n_plans) + " plans, " + num(ledger.size()) +
                        " [HOST-LEDGER] plan lines and " + num(weight.size()) + " [HOST-PLAN] WEIGHT lines");
        return r;
    }
    // The planner's own plans are one computation: a [PLAN-DRY-RUN-BUG] line names a term or a charge that parted from
    // the figure it must equal, so the shaped inventory's dry run logs none.
    const std::vector<std::string> bugs = lines_with("[PLAN-DRY-RUN-BUG]");
    expect(r, bugs.empty(),
           "the shaped inventory's dry run logged " + num(bugs.size()) + " [PLAN-DRY-RUN-BUG] lines" +
               (bugs.empty() ? std::string() : ", the first: " + bugs[0]));

    using dry_plan = ggml_sycl_plan_dry_run_plan;

    struct term {
        const char * printed_name;
        const char * planner_name;
        bool         from_ledger;
        uint64_t dry_plan::* field;
    };

    const term terms[] = {
        { "host_weights",         "host_weights",         false, &dry_plan::weight_host_weights         },
        { "blocks",               "blocks",               false, &dry_plan::weight_blocks               },
        { "host_compute_reserve", "host_compute_reserve", false, &dry_plan::weight_host_compute_reserve },
        { "layout_copies",        "layout_copies",        false, &dry_plan::weight_layout_copies        },
        { "tlsf_bound",           "tlsf_bound",           false, &dry_plan::weight_tlsf_bound           },
        { "weight",               "weight",               true,  &dry_plan::zone_weight                 },
        { "kv",                   "kv",                   true,  &dry_plan::zone_kv                     },
        { "staging",              "staging",              true,  &dry_plan::zone_staging                },
        { "scratch",              "scratch",              true,  &dry_plan::zone_scratch                },
        { "runtime",              "runtime",              true,  &dry_plan::runtime                     },
        { "rounding",             "rounding",             true,  &dry_plan::rounding                    },
        { "expected_host",        "host_planned_bytes",   true,  &dry_plan::expected_host               },
        { "host_budget",          "host_budget",          true,  &dry_plan::host_budget                 },
    };
    for (uint32_t i = 0; i < res->n_plans; ++i) {
        const ggml_sycl_plan_dry_run_plan & p = res->plans[i];
        for (const term & t : terms) {
            const size_t planner = printed(t.from_ledger ? ledger[i] : weight[i], t.planner_name);
            const size_t shown   = dry_run_figure(r, i, t.printed_name);
            expect(r, planner != SIZE_MAX, "plan " + num(i) + ": the planner's line carries no " + t.planner_name);
            expect(r, shown == planner,
                   "plan " + num(i) + ": printed " + t.printed_name + "=" + num(shown) + ", the plan's " +
                       t.planner_name + "=" + num(planner));
            expect(r, p.*(t.field) == planner,
                   "plan " + num(i) + ": the result's " + t.printed_name + " is " + num(p.*(t.field)) +
                       ", the plan's " + num(planner));
        }
        const uint64_t zone_count = p.residency_count[GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED] +
                                    p.residency_count[GGML_SYCL_PLAN_DRY_RUN_BACKING_HOST_CPU];
        const uint64_t zone_bytes = p.residency_bytes[GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED] +
                                    p.residency_bytes[GGML_SYCL_PLAN_DRY_RUN_BACKING_HOST_CPU];
        expect(r, zone_count == p.weight_blocks,
               "plan " + num(i) + ": PINNED + HOST_CPU residency counts " + num(zone_count) + " entries, the WEIGHT " +
                   "walk " + num(p.weight_blocks) + " blocks");
        expect(r, zone_bytes == p.weight_host_bytes,
               "plan " + num(i) + ": PINNED + HOST_CPU residency holds " + num(zone_bytes) + " bytes, the plan's " +
                   "weight_host_bytes " + num(p.weight_host_bytes));
        const char * const backings[] = { "PINNED", "FILE_MAPPED", "HOST_CPU", "CPU_OWNED" };
        for (int b = 0; b < GGML_SYCL_PLAN_DRY_RUN_BACKING_COUNT; ++b) {
            const std::string              tag   = " plan=" + num(i) + " ";
            const std::string              which = std::string(" backing=") + backings[b] + " ";
            const std::vector<std::string> lines = lines_with(k_prefix, which.c_str());
            size_t                         n     = 0;
            for (const std::string & line : lines) {
                if (line.find(tag) == std::string::npos) {
                    continue;
                }
                ++n;
                expect(r,
                       printed(line, "count") == p.residency_count[b] && printed(line, "bytes") == p.residency_bytes[b],
                       "plan " + num(i) + ": the " + backings[b] + " residency line is not the result's");
            }
            expect(r, n == 1, "plan " + num(i) + ": " + num(n) + " residency lines for " + backings[b] + ", want 1");
        }
    }
    residency_from_the_plan_fields(r);
    return r;
}

// (c) A dry run stores nothing and leaves no trace (T2 row arm (j)'s last leg; th32 §4.2 "The dry-run and the
// candidates do not store"). The arm first stages a load of the shaped inventory through the stage and ends it
// uncommitted, so the dry run runs over the caches and planner state a staged load leaves, whatever arms ran before.
// Three legs:
//   budget:  with B0 seeded as the stored budget, a dry run leaves the stored budget at B0 and the store count where it
//            was, and its plans record the budget it computed from its own sample, not B0;
//   trace:   with a sentinel in every per-device global the dry run writes while it plans, each reads its sentinel
//            again afterwards, so a later load reads what it would have read without the dry run;
//   refusal: while a load is open the dry run is refused with LOAD_BUSY before it samples or plans, and stores nothing.
// Engagement: a load was staged before the dry run (its stage returned OK with at least one plan line); the dry run
// returned OK, not LOAD_BUSY, so that load had ended, and planned at least one device; its budget is not B0 (so a store
// would have moved the value); it published the inventory's PP MoE ring while it planned (its RUNTIME term carries the
// ring, which the sentinel does not hold), so it wrote at least one of the globals; the refusal was observed.
// Mutants: a storing host_budget_pass_scope or a host_ledger_store_budget call in the dry run (the stored value leaves
// B0 and the store count moves); the planner recording the stored budget instead of the one passed in (the plans record
// B0); the restore dropped (a global reads what the dry run wrote, not its sentinel); the refusal dropped (the dry run
// plans inside the transaction and returns OK).
arm_result arm_c_a_dry_run_stores_nothing() {
    arm_result       r;
    loader_inventory inv(qwen38_shape);
    const uint64_t   B0 = 4096;

    ggml_sycl_load_txn staged{};
    if (ggml_backend_sycl_model_load_begin(&staged) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the staged load's transaction did not begin, so no load was staged before the dry run");
        return r;
    }
    log_clear();
    const ggml_sycl_lifecycle_result stage_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, nullptr, true);
    const size_t                     staged_plans = lines_with("[HOST-LEDGER] device=", " plan host_budget=").size();
    // An uncommitted end returns the transaction's finish reason, not OK; the dry run below returning OK, not
    // LOAD_BUSY, is what shows the load ended.
    (void) ggml_backend_sycl_model_load_end(staged, false, nullptr);
    if (stage_rc != GGML_SYCL_LIFECYCLE_OK || staged_plans == 0) {
        void_arm(r, "no load was staged before the dry run: the stage returned " + num(stage_rc) + " with " +
                        num(staged_plans) + " plan lines");
        return r;
    }

    const std::vector<device_planner_state> original = read_planner_state();
    const std::vector<device_planner_state> sentinel = sentinel_planner_state(original.size());
    write_planner_state(sentinel);
    if (!(read_planner_state() == sentinel)) {
        void_arm(r, "the sentinel planner state did not read back");
    }

    ggml_sycl_test_host_ledger_store_budget(B0);
    const uint64_t                          s0    = stores();
    ggml_sycl_plan_dry_run_opts             opts  = default_opts();
    auto                                    res   = fresh_result();
    const ggml_sycl_lifecycle_result        rc    = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    const std::vector<device_planner_state> after = read_planner_state();
    write_planner_state(original);
    if (rc != GGML_SYCL_LIFECYCLE_OK || res->n_plans == 0) {
        void_arm(r, "the dry run returned " + num(rc) + " with " + num(res->n_plans) + " plans");
    }
    if (res->host_budget == B0) {
        void_arm(r, "the dry run's own budget is B0, so a store would not show");
    }
    const size_t ring = inv.inv.pp_moe_onednn_scratch_bytes;
    for (uint32_t i = 0; i < res->n_plans; ++i) {
        if (res->plans[i].runtime < ring) {
            void_arm(r, "plan " + num(i) + " did not plan the inventory's ring, so the dry run wrote no global");
        }
        expect(r, res->plans[i].host_budget == res->host_budget,
               "plan " + num(i) + " recorded " + num(res->plans[i].host_budget) + ", not the dry run's budget " +
                   num(res->host_budget));
    }
    expect(r, host_ledger_budget() == B0, "the stored budget moved from B0 to " + num(host_ledger_budget()));
    expect(r, stores() == s0, "the dry run stored " + num(stores() - s0) + " times");
    for (size_t d = 0; d < after.size(); ++d) {
        expect(r, after[d] == sentinel[d],
               "device " + num(d) + ": a planner global reads what the dry run wrote, not its prior value (ring " +
                   num(after[d].ring_weight) + "/" + num(after[d].ring_act) + "/" + num(after[d].ring_out) +
                   ", non-FA n_head " + num(after[d].nonfa.n_head) + ", control " + num(after[d].control.bytes) + ")");
    }

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the refusal leg's load transaction did not begin");
        return r;
    }
    const uint64_t                   s1       = stores();
    const uint64_t                   b1       = host_ledger_budget();
    auto                             open_res = fresh_result();
    const ggml_sycl_lifecycle_result open_rc  = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, open_res.get());
    const uint64_t                   s2       = stores();
    const uint64_t                   b2       = host_ledger_budget();
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (open_rc != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        fail(r, "a dry run inside an open load returned " + num(open_rc) + ", not LOAD_BUSY");
    }
    expect(r, open_res->n_plans == 0, "the refused dry run reported " + num(open_res->n_plans) + " plans");
    expect(r, b2 == b1 && s2 == s1, "the refused dry run stored a budget");
    return r;
}

// (0) A caller built against another layout is refused before anything past the header is read. The struct_size gate
// is what lets the result's layout grow while GGML_SYCL_PLAN_DRY_RUN_VERSION stays 1. The arm runs first, before any
// other arm's dry run, so a refusal is the gate's and not a device's. Two refused legs, then the control:
//   result:  a result header of the size it had before weight_host_bytes joined each of its plans (each plan was then
//            176 bytes, the result 3008) is refused with NULL_OUTPUT and the named WARN, and the result is left
//            exactly as the caller wrote it;
//   opts:    an options header 8 bytes short is refused the same way, with its own WARN. The options' layout has not
//            changed, so there is no older size to send;
//   control: the current sizes return OK with a plan and no refused WARN.
// Engagement: the control returned OK with at least one plan. Neither refused leg sampled the host (no sample line),
// and the control logged exactly one sample line, so that absence is evidence.
// Mutants: the size gate dropped (the old size returns OK); the refusal silent (no WARN naming both sizes); a refused
// call that writes the result.
constexpr uint32_t k_plan_size_before_weight_host_bytes   = 176;
constexpr uint32_t k_result_size_before_weight_host_bytes = 3008;
static_assert(sizeof(ggml_sycl_plan_dry_run_result) - k_result_size_before_weight_host_bytes ==
                  GGML_SYCL_PLAN_DRY_RUN_MAX_DEVICES *
                      (sizeof(ggml_sycl_plan_dry_run_plan) - k_plan_size_before_weight_host_bytes),
              "the old result size is the old plan size times the result's plans");

arm_result arm_0_another_layout_is_refused() {
    arm_result        r;
    loader_inventory  inv(qwen38_shape);
    const std::string refused = "[PLAN-DRY-RUN] refused: ";
    const std::string sample  = "[PLAN-DRY-RUN] sample:";

    // result leg: the old size, with sentinels the refusal must not overwrite
    auto old_res         = fresh_result();
    old_res->struct_size = k_result_size_before_weight_host_bytes;
    old_res->n_plans     = 77;
    old_res->host_budget = 0xa5a5a5a5ull;

    const ggml_sycl_plan_dry_run_result before = *old_res;
    ggml_sycl_plan_dry_run_opts         opts   = default_opts();
    log_clear();
    const ggml_sycl_lifecycle_result rc_res = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, old_res.get());
    const std::string want_res = refused + "the result struct is " + num(k_result_size_before_weight_host_bytes) +
                                 " bytes of version " + num(GGML_SYCL_PLAN_DRY_RUN_VERSION) + "; this backend reads " +
                                 num(sizeof(ggml_sycl_plan_dry_run_result)) + " bytes";
    expect(r, rc_res == GGML_SYCL_LIFECYCLE_NULL_OUTPUT,
           "a result of the old size returned " + num(rc_res) + ", not NULL_OUTPUT");
    expect(r, lines_with(want_res.c_str()).size() == 1, "a result of the old size logged no '" + want_res + "' WARN");
    expect(r, std::memcmp(&before, old_res.get(), sizeof(before)) == 0,
           "the refused call wrote the result the caller sized for the old layout");
    expect(r, lines_with(sample.c_str()).empty(), "a result of the old size was refused only after the host sample");

    // opts leg: a short options header with a current result
    auto                        res        = fresh_result();
    ggml_sycl_plan_dry_run_opts short_opts = default_opts();
    short_opts.struct_size                 = static_cast<uint32_t>(sizeof(short_opts) - 8);
    log_clear();
    const ggml_sycl_lifecycle_result rc_opts = ggml_backend_sycl_plan_dry_run(&inv.inv, &short_opts, res.get());
    const std::string want_opts = refused + "the options struct is " + num(short_opts.struct_size) + " bytes";
    expect(r, rc_opts == GGML_SYCL_LIFECYCLE_NULL_OUTPUT,
           "an options header 8 bytes short returned " + num(rc_opts) + ", not NULL_OUTPUT");
    expect(r, lines_with(want_opts.c_str()).size() == 1,
           "an options header 8 bytes short logged no '" + want_opts + "' WARN");
    expect(r, res->n_plans == 0, "the refused call wrote " + num(res->n_plans) + " plans into the result");
    expect(r, lines_with(sample.c_str()).empty(), "a short options header was refused only after the host sample");

    // control: the current sizes are read
    log_clear();
    const ggml_sycl_lifecycle_result rc_ok = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    if (rc_ok != GGML_SYCL_LIFECYCLE_OK || res->n_plans == 0) {
        void_arm(r, "the current sizes returned " + num(rc_ok) + " with " + num(res->n_plans) +
                        " plans, so the refusals above may be another failure's");
        return r;
    }
    expect(r, lines_with(refused.c_str()).empty(), "the current sizes logged a refused WARN");
    // The refused legs' "no sample line" is evidence only if the OK path logs that line: a reworded, demoted or
    // uncaptured sample line would make their absence vacuous, so its absence here fails the arm.
    expect(r, lines_with(sample.c_str()).size() == 1,
           "the current sizes logged " + num(lines_with(sample.c_str()).size()) + " '" + sample +
               "' lines, want 1: the refused legs' missing sample line proves nothing");
    return r;
}

// (d) --mem-available is the sample's MemAvailable. With opts.mem_available_set the dry run takes the caller's bytes
// as the sample's MemAvailable, reports them as such, and its budget is host_ledger_compute_budget of that sample (the
// other inputs read live, as the result reports them) and the committed bytes it reports. Without the option it
// reports the live reading.
// Engagement: no accepted GGML_SYCL_HOST_RESERVE_MB (which caps the budget, so below the computed one it would be the
// budget), the available term binds (two readings give two budgets), and the reported pinned share is the live one.
// Mutants: the live reading used instead of the option (the reported MemAvailable is not the option, and the budget
// moves); the budget computed from another sample or another committed figure (it is not compute_budget of the
// reported inputs).
arm_result arm_d_mem_available_option_is_the_sample() {
    arm_result       r;
    loader_inventory inv;
    const uint64_t   M  = 64 * GiB;
    const uint64_t   M2 = 96 * GiB;

    ggml_sycl_plan_dry_run_opts opts     = default_opts();
    opts.mem_available_set               = 1;
    opts.mem_available_bytes             = M;
    auto                             res = fresh_result();
    const ggml_sycl_lifecycle_result rc  = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    const host_budget_sample         now = host_ledger_sample();
    if (rc != GGML_SYCL_LIFECYCLE_OK || res->n_plans == 0) {
        void_arm(r, "the dry run returned " + num(rc) + " with " + num(res->n_plans) + " plans");
        return r;
    }
    if (now.reserve.status == detail::host_reserve_parse_status::OVERRIDE || res->host_reserve_override != 0) {
        void_arm(r, "GGML_SYCL_HOST_RESERVE_MB is accepted, so it may cap the budget below the MemAvailable term");
        return r;
    }
    host_budget_sample reported = now;
    reported.total_bytes        = res->mem_total_bytes;
    reported.available_bytes    = M;
    reported.pinned_bytes       = res->pinned_bytes;
    reported.budget_pct         = static_cast<int>(res->budget_pct);
    host_budget_sample other    = reported;
    other.available_bytes       = M2;
    const size_t want           = host_ledger_compute_budget(reported, res->committed_bytes);
    if (want == host_ledger_compute_budget(other, res->committed_bytes)) {
        void_arm(r, "MemAvailable does not bind the budget here: " + num(M) + " and " + num(M2) + " give " + num(want));
    }
    if (res->pinned_bytes != now.pinned_bytes) {
        void_arm(r,
                 "the reported pinned share " + num(res->pinned_bytes) + " is not the live " + num(now.pinned_bytes));
    }
    expect(r, res->mem_available_from_opts == 1, "the result does not say MemAvailable came from the option");
    expect(r, res->mem_available_bytes == M,
           "the sample's MemAvailable is " + num(res->mem_available_bytes) + ", not the option's " + num(M));
    expect(r, res->host_budget == want,
           "the budget is " + num(res->host_budget) + ", compute_budget of the reported sample is " + num(want));
    const std::vector<std::string> budget_lines = lines_with(k_prefix, " host_budget=");
    expect(r, !budget_lines.empty() && printed(budget_lines.back(), "host_budget") == res->host_budget,
           "the printed budget is not the result's");

    ggml_sycl_plan_dry_run_opts      live     = default_opts();
    auto                             live_res = fresh_result();
    const ggml_sycl_lifecycle_result live_rc  = ggml_backend_sycl_plan_dry_run(&inv.inv, &live, live_res.get());
    expect(r, live_rc == GGML_SYCL_LIFECYCLE_OK && live_res->mem_available_from_opts == 0,
           "without the option the dry run does not report the live reading");
    expect(r, live_res->mem_available_bytes != 0, "the live MemAvailable reading is 0");
    return r;
}

const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

}  // namespace

int main() {
    if (!sycl_test_require_gpu("the plan dry run's device arms")) {
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);

    // The backends stay alive across the arms, so every arm plans against the same caches and arena zones.
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b0(ggml_backend_sycl_init(0), ggml_backend_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b1(
        ggml_backend_sycl_get_device_count() > 1 ? ggml_backend_sycl_init(1) : nullptr, ggml_backend_free);
    if (!b0) {
        std::printf("FAIL: backend 0 did not initialise\n");
        return 1;
    }

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "(0) a caller of another layout is refused",           arm_0_another_layout_is_refused          },
        { "(a) expected_host is the stage's host_planned_bytes", arm_a_expected_host_is_the_stage_figure  },
        { "(b) every term printed by name, from the plan",       arm_b_every_term_printed_from_the_plan   },
        { "(c) a dry run stores nothing and leaves no trace",    arm_c_a_dry_run_stores_nothing           },
        { "(d) --mem-available is the sample's MemAvailable",    arm_d_mem_available_option_is_the_sample },
    };
    int passed = 0, failed = 0, voided = 0;
    for (const named_arm & arm : arms) {
        const arm_result result = arm.run();
        std::printf("%s %s%s%s\n", outcome_name(result.result), arm.name, result.why.empty() ? "" : ": ",
                    result.why.c_str());
        std::fflush(stdout);
        passed += result.result == outcome::PASS;
        failed += result.result == outcome::FAIL;
        voided += result.result == outcome::VOID;
    }
    std::printf("plan dry run device arms: %d passed, %d failed, %d void\n", passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
