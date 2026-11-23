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
// arm runs (th32 §11). Arm (g) runs late passes, which configure every device's host zones from a small plan, so it
// runs last: no earlier arm may configure them. A load that ends with no model live releases every arena nothing
// holds, and the next load's start reserves the early arena again (arm (j)). So arms (a), (e) and (i) hold a page in
// every arena through the end of each load whose zones or terms they read after it, as a first model's allocations
// would. That end cannot tell such a page from a failed load's own buffers, so it marks the release deferred, and the
// next load's start runs it again: while the page is held the arena stays and that load runs on it, and once the page
// is gone the start empties the cache and puts its terms back at their start values; arm (k) scores that start, after
// (j), and its last load's end leaves every arena released, as (j)'s does. Arm (f) marks a model live from
// its load's begin to its end, as the live context its ring stands for would, and arms (f) and (h) first run any
// release an earlier arm's pages deferred (settle_deferred_releases), then reserve device 0's arena as a load's start
// does, before they put a ring in it. Arm (i)'s reverse leg leaves device 0's ONEDNN zone grown for a 1 GiB Graph
// floor after its hold goes, so (i) runs after (h), and the next load's start, arm (j)'s first, releases that arena.
// Before any arm, main stages the plain shaped inventory once in a load of its own and keeps device 0's zones from it:
// the zone decision of a first load, which arm (j) checks a later load against. That load's end releases every arena,
// and main reserves each again as a load's start would, so the arms find the arenas the caches were made with. Every
// device, arena and cache these comments name is one of the devices the scheduler sees, the ones the release of the
// last model walks; on a two-card host that shows the scheduler one device, device 1's cache is outside them.
// Every load that main and the arms stage of the Qwen3.8-shaped inventory maps its file (mapped_qwen38_inventory), so
// the stage's loader-pool refusal refuses none of them, whatever the host's budget, but arm (l)'s, which stages the
// unmapped inventory and scores that refusal. A load it refuses releases every arena nothing holds, as (k)'s last end
// does, so (l) runs after (k), and before (g). A mapped load pins its routed host experts for the window ring and
// moves them back to the mapping when the pinned plan does not fit the host budget, so arms (a) and (c) read which leg
// each plan took from the planner's own lines and score that leg; no verdict depends on the host's MemAvailable.
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

// The Qwen3.8-shaped inventory of a load that maps its file (use_mmap_resolved), the form of every staged load of that
// shape but arm (l)'s. Its host weights are the file's pages, so the loader's SYCL_Host buffer holds none of them and
// the plan is charged no loader term: the stage's loader-pool refusal, which (l) scores on the unmapped inventory,
// cannot refuse these loads on a host whose budget is below that pool. The stage only plans the inventory, so no file
// stands behind it. The dry-run-only arms (0) and (b) keep the unmapped inventory: a dry run refuses nothing, and (b)
// scores the loader's terms.
struct mapped_qwen38_inventory : loader_inventory {
    mapped_qwen38_inventory() : loader_inventory(qwen38_shape) { inv.use_mmap_resolved = true; }
};

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

// Every captured line, in the order they were logged.
std::vector<std::string> log_snapshot() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log_lines;
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

// The PP MoE oneDNN ring an inventory publishes, as the planner sizes it: while it sizes each record of a plan,
// populate_host_zone_sizing reads the ring back from the per-device globals the publish wrote and prints it on a
// "[SYCL-PLAN] PP MoE oneDNN scratch ring: " line before the record. The figures are the loader's weight slot and the
// activation and output slots pp_moe_onednn_slots_at gives at the load's micro-batch, at the ring's effective depth,
// in bytes. The ring is a VRAM allocation, so no host term carries it: the host RUNTIME term is the oneDNN scratchpad
// and the DMA staging pool (placement_plan_host_zone_commit_terms). Not sized when the slots cannot be sized.
struct planned_ring {
    bool     sized      = false;
    uint32_t depth      = 0;
    size_t   total      = 0;
    size_t   weight     = 0;
    size_t   activation = 0;
    size_t   output     = 0;
};

planned_ring planned_ring_of(const ggml_sycl_tensor_inventory & inv) {
    planned_ring ring;
    if (!pp_moe_onednn_slots_at(placement_load_n_ubatch(inv.n_ubatch), static_cast<uint32_t>(std::max(0, inv.n_expert)),
                                static_cast<uint32_t>(std::max(0, inv.n_expert_used)),
                                inv.pp_moe_onednn_activation_bytes_per_row, inv.pp_moe_onednn_output_bytes_per_row,
                                &ring.activation, &ring.output)) {
        return ring;
    }
    ring.sized  = true;
    ring.depth  = test_pp_moe_onednn_effective_ring_depth(inv.pp_moe_onednn_ring_depth);
    ring.weight = inv.pp_moe_onednn_weight_slot_bytes;
    ring.total  = test_pp_moe_onednn_planned_scratch_bytes(ring.weight, ring.activation, ring.output,
                                                           inv.pp_moe_onednn_ring_depth);
    return ring;
}

// The ring's figures, in MB, for a message.
std::string planned_ring_text(const planned_ring & ring) {
    const double mb = 1024.0 * 1024.0;
    char         text[192];
    std::snprintf(text, sizeof(text),
                  "depth=%u total=%.1f MB weight_slot=%.1f MB activation_slot=%.1f MB output_slot=%.1f MB", ring.depth,
                  ring.total / mb, ring.weight / mb, ring.activation / mb, ring.output / mb);
    return text;
}

// The decimal figure printed after " <name>=" on the line, or a negative value when the name is absent.
double printed_figure(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    return at == std::string::npos ? -1.0 : std::strtod(line.c_str() + at + key.size(), nullptr);
}

// Whether the ring line's text (what follows its marker) prints `ring`: the depth exactly, and each MB figure, read by
// name, within half a printed unit of the exact one, since the planner prints each to one decimal. Reading the fields
// by name rather than re-rendering the planner's format lets the line's wording change without the arms reading it as
// a ring the dry run did not publish.
bool ring_line_prints(const std::string & text, const planned_ring & ring) {
    const std::string line = " " + text;
    if (!ring.sized || printed(line, "depth") != ring.depth) {
        return false;
    }
    const double mb = 1024.0 * 1024.0;

    const struct {
        const char * name;
        size_t       bytes;
    } figures[] = {
        { "total",           ring.total      },
        { "weight_slot",     ring.weight     },
        { "activation_slot", ring.activation },
        { "output_slot",     ring.output     },
    };

    for (const auto & figure : figures) {
        const double shown = printed_figure(line, figure.name);
        if (shown < 0.0 || std::fabs(shown - figure.bytes / mb) > 0.05 + 1e-6) {
            return false;
        }
    }
    return true;
}

// What a run of the planner did with a mapped load's routed host experts for the window ring, as its own lines say:
// kept them pinned (the "pinned for the window ring" line, after the plan's record), moved them back to the mapping
// once the pinned plan failed its fit or a lazy table refused it (the "stay mapped" WARN, between the plan's two
// records), moved them back before the ledger because the plan holds no ring room (that WARN, before the plan's one
// record), or never pinned them (neither line). The host budget decides between the first two.
enum class pin_leg { UNPINNED, KEPT, WITHDRAWN, NO_ROOM };

const char * pin_leg_name(pin_leg leg) {
    switch (leg) {
        case pin_leg::KEPT:
            return "pin-kept";
        case pin_leg::WITHDRAWN:
            return "pin-withdrawn";
        case pin_leg::NO_ROOM:
            return "pin-no-ring-room";
        case pin_leg::UNPINNED:
            break;
    }
    return "unpinned";
}

// One device's plan as a run logged it. Its record is its last [HOST-LEDGER] plan line: a plan whose pin is withdrawn
// after the fit is recorded twice, the pinned plan first. `records` counts those lines, `max_planned` is the largest
// host_planned_bytes among them (the pinned plan's on that leg), `rings` holds for each record the ring line its sizing
// printed before it (empty when it printed none), and `leg` is what the run's [PLACEMENT] lines say of the pin.
struct logged_plan {
    size_t                   device = SIZE_MAX;
    std::string              record;
    size_t                   records     = 0;
    size_t                   max_planned = 0;
    std::vector<std::string> rings;
    pin_leg                  leg = pin_leg::UNPINNED;
};

// The text after `marker` on `line`, without its line end.
std::string text_after(const std::string & line, const std::string & marker) {
    std::string text = line.substr(line.find(marker) + marker.size());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

// The plans `lines` record, one per device, in the order the devices were first recorded. A pin line names no device,
// so it belongs to the plan the run is sizing when it is logged: the kept line and the fit's WARN follow that plan's
// first record, and the no-room WARN comes before its only record.
std::vector<logged_plan> logged_plans(const std::vector<std::string> & lines) {
    const std::string        ring_marker = "[SYCL-PLAN] PP MoE oneDNN scratch ring: ";
    std::vector<logged_plan> plans;
    std::string              ring;
    bool                     no_room = false;
    size_t                   last    = SIZE_MAX;  // the plan the last record was for
    for (const std::string & line : lines) {
        if (line.find(ring_marker) != std::string::npos) {
            ring = text_after(line, ring_marker);
            continue;
        }
        if (line.find("[PLACEMENT] routed host experts stay mapped") != std::string::npos) {
            if (line.find("the plan holds no window-ring room") != std::string::npos) {
                no_room = true;
            } else if (last != SIZE_MAX) {
                plans[last].leg = pin_leg::WITHDRAWN;
            }
            continue;
        }
        if (line.find("[PLACEMENT] routed host experts pinned for the window ring") != std::string::npos) {
            if (last != SIZE_MAX) {
                plans[last].leg = pin_leg::KEPT;
            }
            continue;
        }
        if (line.find("[HOST-LEDGER] device=") == std::string::npos ||
            line.find(" plan host_budget=") == std::string::npos) {
            continue;
        }
        const size_t device = printed(line, "device");
        last                = 0;
        while (last < plans.size() && plans[last].device != device) {
            ++last;
        }
        if (last == plans.size()) {
            plans.emplace_back();
            plans.back().device = device;
        }
        logged_plan & plan = plans[last];
        plan.record        = line;
        plan.records += 1;
        plan.max_planned = std::max(plan.max_planned, printed(line, "host_planned_bytes"));
        plan.rings.push_back(ring);
        ring.clear();
        if (no_room) {
            plan.leg = pin_leg::NO_ROOM;
            no_room  = false;
        }
    }
    return plans;
}

// Whether every record of `plan` printed `ring` on its ring line.
bool every_record_printed(const logged_plan & plan, const planned_ring & ring) {
    return plan.records > 0 && plan.rings.size() == plan.records &&
           std::all_of(plan.rings.begin(), plan.rings.end(),
                       [&](const std::string & text) { return ring_line_prints(text, ring); });
}

// The leg and the ring line of `plan`'s record, as a PASS line names them.
std::string leg_and_ring(const logged_plan & plan) {
    return "leg=" + std::string(pin_leg_name(plan.leg)) + " records=" + num(plan.records) + " ring=[" +
           (plan.rings.empty() ? std::string() : plan.rings.back()) + "]";
}

// What in `plan`'s lines contradicts the leg they were read as, or empty: a plan recorded twice took the withdrawn leg
// (the fit recorded the pinned plan, then the plan without the pin), and a plan whose pin was kept or never had room
// recorded once. A pin line the reading no longer recognizes, once reworded, then shows as a FAIL here rather than as
// another leg. The converse, a withdrawn leg recorded other than twice, is not checked.
std::string leg_record_mismatch(const logged_plan & plan) {
    if (plan.records == 2 && plan.leg != pin_leg::WITHDRAWN) {
        return std::string("recorded twice, so its pin was withdrawn, but its lines read as ") + pin_leg_name(plan.leg);
    }
    if ((plan.leg == pin_leg::KEPT || plan.leg == pin_leg::NO_ROOM) && plan.records != 1) {
        return std::string("read as ") + pin_leg_name(plan.leg) + ", which records once, and recorded " +
               num(plan.records) + " times";
    }
    return {};
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

// A 4 KiB allocation in `device`'s RUNTIME zone, which holds that device's arena as a live model's allocations would.
allocation_result allocate_runtime_page(unified_cache * cache, int device) {
    alloc_request req{};
    req.queue                               = &cache->get_queue();
    req.device                              = device;
    req.size                                = 4096;
    req.intent.role                         = alloc_role::GRAPH_TMP;
    req.intent.category                     = runtime_category::GRAPH;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;
    return unified_allocate_owner(req);
}

// Whether `holder` lies in a RUNTIME zone, where it holds its device's arena.
bool holds_the_runtime_zone(const mem_handle & holder) {
    const resolved_ptr p = holder.resolve();
    alloc_metadata     meta{};
    return p && p.on_device && unified_lookup(p.ptr, &meta) && meta.zone == vram_zone_id::RUNTIME;
}

// A RUNTIME-zone page in every device's arena, device 0's first, appended to the empty `pages`. With no model live a
// load's end releases every arena nothing holds, and on each device it emptied puts the published terms back at their
// start values (ggml_sycl_release_last_model_effects). An arm that reads a load's zones or terms after its end takes
// these pages inside the load, before the end, as the first model's allocations would hold the arenas. False when a
// device has no active arena or its page is not in that arena's RUNTIME zone.
bool hold_arenas(std::vector<mem_handle> & pages) {
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        unified_cache * cache = get_unified_cache_for_device(d);
        if (cache == nullptr || !cache->arena_active()) {
            return false;
        }
        allocation_result allocation = allocate_runtime_page(cache, d);
        if (!allocation) {
            return false;
        }
        // The handle's destructor is the page's one release point, when the arm clears `pages` or returns.
        pages.push_back(mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS));
        if (!holds_the_runtime_zone(pages.back())) {
            return false;
        }
    }
    return !pages.empty();
}

// A model live in the process, as the release of the last model reads it: every cache's live mask gains a slot no
// model holds. A load's begin publishes the registry's mask to every cache, so the mark goes on after the begin; it
// comes off with the mask device 0's cache held before it.
struct live_model_mark {
    uint32_t prior = 0;

    live_model_mark() {
        if (unified_cache * cache = get_unified_cache_for_device(0)) {
            prior = cache->live_model_mask();
        }
        unified_cache_set_live_model_mask(prior | (1u << 31));
    }

    ~live_model_mark() { unified_cache_set_live_model_mask(prior); }

    live_model_mark(const live_model_mark &)             = delete;
    live_model_mark & operator=(const live_model_mark &) = delete;
};

// Runs the release an earlier arm's held pages deferred at a load's end, as the next load's start runs it: when a cache
// carries the mark (unified_cache::last_release_deferred), an empty load begins and ends uncommitted. Its begin empties
// each marked cache nothing holds any longer and its end releases every arena nothing holds, so every cache is then as
// the last model's release leaves it.
void settle_deferred_releases() {
    bool deferred = false;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        unified_cache * cache = get_unified_cache_for_device(d);
        deferred              = deferred || (cache != nullptr && cache->last_release_deferred());
    }
    ggml_sycl_load_txn txn{};
    if (deferred && ggml_backend_sycl_model_load_begin(&txn) == GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    }
}

// Device 0's cache, its arena reserved as a load's start reserves it (unified_cache::reserve_released_arena): a load
// that ended with no model live released the arena nothing held. An arm that puts a ring in the arena before its load
// begins reserves it here, as a live model's state would have it, once any release an earlier arm deferred has run, so
// its own load's start does not run that release over the ring. Null when device 0 has no cache or no arena.
unified_cache * device0_arena_cache() {
    settle_deferred_releases();
    unified_cache *    cache = get_unified_cache_for_device(0);
    zone_arena_refusal refusal{};
    if (cache == nullptr || !cache->reserve_released_arena(&refusal) || !cache->arena_active()) {
        return nullptr;
    }
    return cache;
}

// (a) expected_host is the stage's host_planned_bytes. An early stage run in a load transaction plans the
// Qwen3.8-shaped inventory on each device the scheduler exposes, and its planner records host_planned_bytes on a
// [HOST-LEDGER] plan line. A plan's record is its last such line for its device: the load maps its file, so the planner
// pins the routed host experts for the window ring, and when the pinned plan does not fit the budget it moves them
// back to the mapping and records the plan again. The dry run over the same inventory, against the budgets the stage
// planned against (the same process, so the same caches and arena zones), plans the devices the stage planned and
// reports each record's figure, in the same order.
// The dry run's VRAM budget is the arena's whole reservation while the arena is active, so the stage's load holds a
// page in every arena through its end, which would otherwise release them, until the dry run has run.
// The dry run is handed the stage's own MemAvailable sample through --mem-available (arm (d) proves the option is the
// sample), read from the stage's one [HOST-LEDGER] stored-budget line, so both sides compute the budget from one
// sample: two samples differ by the host's drift, and when the budget binds that drift moves the plan.
// Legs: whether the pin is kept or withdrawn is the host budget's call (on a host whose budget is below the pinned
// plan, every run takes the withdrawn leg), so the arm reads each plan's leg from each run's own lines (pin_leg) and
// scores the leg taken. Both runs plan from one sample, so they take the same leg with the same number of records, and
// the dry run's PINNED residency says what its lines say: only a kept pin leaves a PINNED entry on a mapped load. Each
// run's leg must agree with its record count (leg_record_mismatch), so a reworded pin line FAILs the arm instead of
// reading as another leg.
// Engagement: the stage run returned OK with at least one plan line and stored one budget, every arena was held through
// its end, the dry run returned OK, every dry-run plan put at least one weight on the host (the shape's experts
// overflow any card here: PINNED on the kept leg, FILE_MAPPED on the others), and every record the dry run made printed
// the inventory's PP MoE ring on its ring line (planned_ring_of, read by ring_line_prints). That line is the planner
// reading back, while it sizes the record, the ring the dry run published after the stage's publish was withdrawn
// (zeroed), so it is printed on every leg and no host figure moves it.
// A budget, leg or record count that differs from the stage's VOIDs the arm only when an input each side reads live
// (the pinned share, the committed bytes, the RAM total) moved in between and the budget binds a record of either run;
// with none moved it FAILs, as one sample planned into two plans.
// Mutants, each killed on every leg unless it says otherwise:
//   a non-zero term dropped from or added to expected_host: it differs from the record by that term. Every commit term
//     but the loader's two is non-zero on every leg (the withdrawn leg's record charges a 64 MiB WEIGHT zone and a
//     15 MiB RUNTIME term). A mapped load charges no loader term, so a dropped loader_zone is arm (b)'s to kill, on
//     the unmapped inventory; loader_grown is 0 in every plan these arms make, so no arm kills dropping it;
//   the dry run not publishing the inventory's planner inputs: its sizing reads the zeroed ring and prints no ring line
//     (a VOID, which fails the binary);
//   a dry run planning against a budget other than the one it computed: its records' host_budget is not the stage's;
//   the dry run computing its budget from a sample other than the stage's: the budgets differ with nothing moved;
//   a dry run that does not pin the routed experts: its plans read unpinned against the stage's kept, withdrawn or
//     no-room leg (on the unpinned leg the stage did not pin either, and the mutant changes nothing);
//   the dry run reporting the plan its fit withdrew: expected_host is the pinned plan's figure, not the record's
//     (withdrawn leg only; on the other legs a plan has one record, and the mutant changes nothing).
arm_result arm_a_expected_host_is_the_stage_figure() {
    arm_result              r;
    mapped_qwen38_inventory inv;

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    log_clear();
    const ggml_sycl_lifecycle_result stage_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, nullptr, true);
    const std::vector<logged_plan>   stage    = logged_plans(log_snapshot());
    const std::vector<std::string>   stored   = lines_with("[HOST-LEDGER] stored budget=");
    std::vector<mem_handle>          held;  // the arenas the stage planned against, until the dry run has run
    const bool                       arenas_held = hold_arenas(held);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (stage_rc != GGML_SYCL_LIFECYCLE_OK || stage.empty()) {
        void_arm(r,
                 "the early stage run returned " + num(stage_rc) + " with " + num(stage.size()) + " devices recorded");
        return r;
    }
    if (!arenas_held) {
        void_arm(r, "an arena could not be held through the load's end, so the dry run would not plan against the " +
                        std::string("arena zones the stage planned against"));
        return r;
    }
    if (stored.size() != 1 || printed(stored[0], "available") == SIZE_MAX) {
        void_arm(r, "the stage logged " + num(stored.size()) + " stored-budget lines, want 1 with its sample, so " +
                        "the dry run cannot be handed the stage's MemAvailable");
        return r;
    }
    const size_t stage_available = printed(stored[0], "available");
    const size_t stage_budget    = printed(stored[0], "budget");

    withdraw_published_inputs();
    ggml_sycl_plan_dry_run_opts opts = default_opts();
    opts.mem_available_set           = 1;
    opts.mem_available_bytes         = stage_available;

    auto res = fresh_result();
    log_clear();
    const ggml_sycl_lifecycle_result rc  = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    const std::vector<logged_plan>   dry = logged_plans(log_snapshot());
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the dry run returned " + num(rc));
        return r;
    }
    if (res->n_plans != stage.size() || dry.size() != res->n_plans) {
        fail(r, "the dry run returned " + num(res->n_plans) + " plans and recorded " + num(dry.size()) +
                    " devices; the stage recorded " + num(stage.size()));
        return r;
    }
    const planned_ring ring = planned_ring_of(inv.inv);
    if (!ring.sized) {
        void_arm(r, "the inventory's ring cannot be sized, so no ring line can show the dry run published it");
    }
    // The budget's other inputs that each side reads live: the pinned share, the committed bytes and the RAM total.
    const bool moved = res->pinned_bytes != printed(stored[0], "pinned") ||
                       res->committed_bytes != printed(stored[0], "committed") ||
                       res->mem_total_bytes != printed(stored[0], "total");
    if (res->host_budget != stage_budget && !moved) {
        fail(r, "from the stage's own sample (MemAvailable " + num(stage_available) + ", pinned " +
                    num(res->pinned_bytes) + ", committed " + num(res->committed_bytes) + ") the dry run's host " +
                    "budget is " + num(res->host_budget) + ", the stage stored " + num(stage_budget));
    }
    std::string legs;
    for (uint32_t i = 0; i < res->n_plans; ++i) {
        const ggml_sycl_plan_dry_run_plan & p    = res->plans[i];
        const logged_plan &                 s    = stage[i];
        const logged_plan &                 d    = dry[i];
        const std::string                   name = "plan " + num(i) + " (device " + std::to_string(p.device) + ")";
        if (residency_entries(p) == 0) {
            void_arm(r, name + " put no weight on the host, so the WEIGHT terms were not exercised");
        }
        if (!every_record_printed(d, ring)) {
            void_arm(r, name + ": of the dry run's " + num(d.records) + " records, one printed the ring line [" +
                            (d.rings.empty() ? std::string() : d.rings.back()) + "], not the inventory's [" +
                            planned_ring_text(ring) + "], so the dry run did not publish its ring");
        }
        expect(r, p.device == static_cast<int32_t>(s.device) && d.device == s.device,
               name + ": the stage's record is for device " + num(s.device) + ", the dry run's for device " +
                   num(d.device));
        const std::string stage_mismatch = leg_record_mismatch(s);
        expect(r, stage_mismatch.empty(), name + ": the stage's plan was " + stage_mismatch);
        const std::string dry_mismatch = leg_record_mismatch(d);
        expect(r, dry_mismatch.empty(), name + ": the dry run's plan was " + dry_mismatch);
        // A budget above every record's host figure did not bind the plan, so a moved live input cannot move it; one
        // at or below it may have, and then neither the leg nor the figure is like for like. On the withdrawn leg the
        // record the budget bound is the pinned plan's, the larger.
        const size_t plan_budget = printed(s.record, "host_budget");
        const size_t low_budget  = std::min<size_t>(plan_budget, p.host_budget);
        if (moved && plan_budget != p.host_budget && std::max(s.max_planned, d.max_planned) >= low_budget) {
            void_arm(r, name + ": the host budget binds (" + num(std::max(s.max_planned, d.max_planned)) +
                            " host bytes) and the pinned share, committed bytes or RAM total moved between the stage " +
                            "(budget " + num(plan_budget) + ") and the dry run (" + num(p.host_budget) + ")");
            continue;
        }
        if (!moved) {
            expect(r, p.host_budget == plan_budget,
                   name + ": the dry run recorded host_budget " + num(p.host_budget) + ", the stage's record " +
                       num(plan_budget));
        }
        expect(r, d.leg == s.leg && d.records == s.records,
               name + ": the dry run took the " + pin_leg_name(d.leg) + " leg with " + num(d.records) +
                   " records, the stage the " + pin_leg_name(s.leg) + " leg with " + num(s.records));
        const uint64_t pinned = p.residency_count[GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED];
        expect(r, (d.leg == pin_leg::KEPT) == (pinned > 0),
               name + ": the dry run's lines say " + pin_leg_name(d.leg) + ", and its residency holds " + num(pinned) +
                   " PINNED entries");
        expect(r, p.expected_host == printed(s.record, "host_planned_bytes"),
               name + ": expected_host " + num(p.expected_host) + ", the stage's record's host_planned_bytes " +
                   num(printed(s.record, "host_planned_bytes")));
        legs += (legs.empty() ? "" : "; ") + name + " " + leg_and_ring(d);
    }
    if (r.result == outcome::PASS) {
        r.why = legs;
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
// term's name, and each must equal the planner's. The WEIGHT term swap_headroom is printed and not exported: it is
// held after the [HOST-PLAN] line, so its planner figure is the bytes on the spread plan's headroom line, and 0 for a
// plan that prints none. The WEIGHT terms loader_residency and loader_served are printed and not exported, so their
// printed figures alone are scored. The dry run logs no [PLAN-DRY-RUN-BUG] line for these plans. The
// PINNED and HOST_CPU residency is scored against the plan: those are the entries the WEIGHT walk counts, so their
// count is its blocks plus the host experts the loader serves in place (more than the blocks exactly when
// loader_served is not 0), their bytes are host_weights plus loader_served, and they are the plan's own charge for
// them, weight_host_bytes, which the result carries. The shaped inventory maps no file (use_mmap_resolved is false), so
// with evictable weights a single-device plan serves its host experts from the loader's buffer, and its plan keeps
// every host entry PINNED,
// and a second leg scores the residency on a plan built by hand, through the record's private seam: its FILE_MAPPED
// and CPU_OWNED figures are the plan's host_mapped_bytes and host_cpu_owned_bytes, which the fixture charges apart from
// either per-entry sum, its weight_host_bytes is the plan's, and a plan whose weight_host_bytes is charged on the DST
// basis gets the one [PLAN-DRY-RUN-BUG] line the plan charged on the SRC basis does not.
// Engagement: the dry run returned OK, and each plan has exactly one planner line of each kind to score against; the
// hand-built plan's record counted one entry of each backing, and its two field figures differ from both the SRC and
// the DST sums of their entries.
// Mutants: a figure printed from a recomputed value (it differs from the planner's record: a rounding from another
// chunk, a WEIGHT total re-summed, a budget re-sampled, a residency byte figure re-summed per entry on either basis); a
// term not printed, or two terms on one line (the line count for that term is not 1), the swap headroom among them; a
// swap headroom printed from another figure than the planner's line; a residency line walking another
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

    // A term with no field is printed and not exported (the result struct's layout stays): its printed figure alone
    // is scored against the planner's.
    struct term {
        const char * printed_name;
        const char * planner_name;
        bool         from_ledger;
        uint64_t dry_plan::* field;
    };

    const term terms[] = {
        { "host_weights",         "host_weights",         false, &dry_plan::weight_host_weights         },
        { "blocks",               "blocks",               false, &dry_plan::weight_blocks               },
        { "loader_residency",     "loader_residency",     false, nullptr                                },
        { "loader_served",        "loader_served",        false, nullptr                                },
        { "host_compute_reserve", "host_compute_reserve", false, &dry_plan::weight_host_compute_reserve },
        { "layout_copies",        "layout_copies",        false, &dry_plan::weight_layout_copies        },
        { "tlsf_bound",           "tlsf_bound",           false, &dry_plan::weight_tlsf_bound           },
        { "weight",               "weight",               true,  &dry_plan::zone_weight                 },
        { "loader_zone",          "loader_zone",          true,  nullptr                                },
        { "kv",                   "kv",                   true,  &dry_plan::zone_kv                     },
        { "staging",              "staging",              true,  &dry_plan::zone_staging                },
        { "scratch",              "scratch",              true,  &dry_plan::zone_scratch                },
        { "runtime",              "runtime",              true,  &dry_plan::runtime                     },
        { "loader_grown",         "loader_grown",         true,  nullptr                                },
        { "rounding",             "rounding",             true,  &dry_plan::rounding                    },
        { "expected_host",        "host_planned_bytes",   true,  &dry_plan::expected_host               },
        { "host_budget",          "host_budget",          true,  &dry_plan::host_budget                 },
    };
    // The swap headroom is printed and not exported. The planner's figure is its headroom line, one per spread plan;
    // a plan that does not spread prints none and holds none, so its printed term is 0.
    const std::vector<std::string> headroom = lines_with("[PLACEMENT-MOE] spread swap headroom: ");
    expect(r, headroom.empty() || headroom.size() == res->n_plans,
           "the planner printed " + num(headroom.size()) + " swap headroom lines for " + num(res->n_plans) +
               " plans, want none or one a plan");
    for (uint32_t i = 0; i < res->n_plans; ++i) {
        const ggml_sycl_plan_dry_run_plan & p = res->plans[i];
        for (const term & t : terms) {
            const size_t planner = printed(t.from_ledger ? ledger[i] : weight[i], t.planner_name);
            const size_t shown   = dry_run_figure(r, i, t.printed_name);
            expect(r, planner != SIZE_MAX, "plan " + num(i) + ": the planner's line carries no " + t.planner_name);
            expect(r, shown == planner,
                   "plan " + num(i) + ": printed " + t.printed_name + "=" + num(shown) + ", the plan's " +
                       t.planner_name + "=" + num(planner));
            if (t.field != nullptr) {
                expect(r, p.*(t.field) == planner,
                       "plan " + num(i) + ": the result's " + t.printed_name + " is " + num(p.*(t.field)) +
                           ", the plan's " + num(planner));
            }
        }
        const size_t swap_planned = i < headroom.size() ? printed(headroom[i], "bytes") : 0;
        const size_t swap_shown   = dry_run_figure(r, i, "swap_headroom");
        expect(r, swap_shown == swap_planned,
               "plan " + num(i) + ": printed swap_headroom=" + num(swap_shown) + ", the planner's headroom line " +
                   num(swap_planned));
        const uint64_t zone_count = p.residency_count[GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED] +
                                    p.residency_count[GGML_SYCL_PLAN_DRY_RUN_BACKING_HOST_CPU];
        const uint64_t zone_bytes = p.residency_bytes[GGML_SYCL_PLAN_DRY_RUN_BACKING_PINNED] +
                                    p.residency_bytes[GGML_SYCL_PLAN_DRY_RUN_BACKING_HOST_CPU];
        // The host experts the loader serves in place are PINNED entries that hold no block, so the walk's blocks are
        // the other entries, and the two byte terms split the entries' bytes between them.
        const size_t served = printed(weight[i], "loader_served");
        expect(r, served == 0 ? zone_count == p.weight_blocks : zone_count > p.weight_blocks,
               "plan " + num(i) + ": PINNED + HOST_CPU residency counts " + num(zone_count) + " entries, the WEIGHT " +
                   "walk " + num(p.weight_blocks) + " blocks, with " + num(served) + " bytes served in place");
        expect(r, zone_bytes == p.weight_host_bytes,
               "plan " + num(i) + ": PINNED + HOST_CPU residency holds " + num(zone_bytes) + " bytes, the plan's " +
                   "weight_host_bytes " + num(p.weight_host_bytes));
        expect(r, served != SIZE_MAX && p.weight_host_weights + served == zone_bytes,
               "plan " + num(i) + ": host_weights " + num(p.weight_host_weights) + " + loader_served " + num(served) +
                   " is not the PINNED + HOST_CPU residency's " + num(zone_bytes) + " bytes");
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
// would have moved the value); it published the inventory's PP MoE ring while it planned, so it wrote at least one of
// the globals: every record it made printed the inventory's ring on its ring line (planned_ring_of, read by
// ring_line_prints), which the planner reads back from the per-device globals while it sizes the record, and which
// shows the sentinel's ring when nothing was published; the refusal was observed. The ring line is printed for every
// record whether the routed host experts' pin is kept or withdrawn, which the host budget decides (the PASS line names
// the leg each plan took, read from the dry run's own lines, which must agree with its record count:
// leg_record_mismatch); no check here reads a host figure that moves with the leg.
// Mutants, each killed on every leg, since none of them is on the pin's path: a storing host_budget_pass_scope or a
// host_ledger_store_budget call in the dry run (the stored value leaves B0 and the store count moves); the planner
// recording the stored budget instead of the one passed in (every record, the last among them, records B0); the
// restore dropped (a global reads what the dry run wrote, not its sentinel: the ring line shows the publish wrote the
// inventory's ring over it); the refusal dropped (the dry run plans inside the transaction and returns OK).
arm_result arm_c_a_dry_run_stores_nothing() {
    arm_result              r;
    mapped_qwen38_inventory inv;
    const uint64_t          B0 = 4096;

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
    const uint64_t              s0   = stores();
    ggml_sycl_plan_dry_run_opts opts = default_opts();
    auto                        res  = fresh_result();
    log_clear();
    const ggml_sycl_lifecycle_result        rc    = ggml_backend_sycl_plan_dry_run(&inv.inv, &opts, res.get());
    const std::vector<device_planner_state> after = read_planner_state();
    const std::vector<logged_plan>          dry   = logged_plans(log_snapshot());
    write_planner_state(original);
    if (rc != GGML_SYCL_LIFECYCLE_OK || res->n_plans == 0) {
        void_arm(r, "the dry run returned " + num(rc) + " with " + num(res->n_plans) + " plans");
    }
    if (res->host_budget == B0) {
        void_arm(r, "the dry run's own budget is B0, so a store would not show");
    }
    const planned_ring ring = planned_ring_of(inv.inv);
    if (!ring.sized) {
        void_arm(r, "the inventory's ring cannot be sized, so no ring line can show the dry run published it");
    }
    if (rc == GGML_SYCL_LIFECYCLE_OK && dry.size() != res->n_plans) {
        void_arm(r, "the dry run returned " + num(res->n_plans) + " plans and its lines record " + num(dry.size()) +
                        " devices, so its records cannot be read");
    }
    std::string legs;
    for (uint32_t i = 0; i < res->n_plans && i < dry.size(); ++i) {
        if (!every_record_printed(dry[i], ring)) {
            void_arm(r, "plan " + num(i) + " did not plan the inventory's ring [" + planned_ring_text(ring) +
                            "] (a record printed [" + (dry[i].rings.empty() ? std::string() : dry[i].rings.back()) +
                            "]), so the dry run wrote no global");
        }
        const std::string mismatch = leg_record_mismatch(dry[i]);
        expect(r, mismatch.empty(), "plan " + num(i) + " was " + mismatch);
        expect(r, res->plans[i].host_budget == res->host_budget,
               "plan " + num(i) + " recorded " + num(res->plans[i].host_budget) + ", not the dry run's budget " +
                   num(res->host_budget));
        legs += (legs.empty() ? "" : "; ") + std::string("plan ") + num(i) + " (device " +
                std::to_string(res->plans[i].device) + ") " + leg_and_ring(dry[i]);
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
    if (r.result == outcome::PASS) {
        r.why = legs;
    }
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

// The device-wide terms a load publishes beside the planner state (device_planner_state), which a refused load puts
// back too: the ring's KV-zone slots, its row bytes and expert counts, its micro-batch, and the oneDNN pair with its
// stored sum. The dry run publishes none of them, so they are not in the sentinel the dry-run arms share, which holds
// what the dry run writes and must put back. Read through load_published_terms::snapshot, the pair's one reader;
// written through the setters.
struct device_load_extras {
    bool     act_in_kv          = false;
    bool     out_in_kv          = false;
    size_t   act_row            = 0;
    size_t   out_row            = 0;
    uint32_t n_expert           = 0;
    uint32_t n_expert_used      = 0;
    uint32_t n_ubatch           = 0;
    size_t   onednn_weights     = 0;
    size_t   onednn_activations = 0;
    size_t   onednn_stored      = 0;

    bool operator==(const device_load_extras & o) const {
        return act_in_kv == o.act_in_kv && out_in_kv == o.out_in_kv && act_row == o.act_row && out_row == o.out_row &&
               n_expert == o.n_expert && n_expert_used == o.n_expert_used && n_ubatch == o.n_ubatch &&
               onednn_weights == o.onednn_weights && onednn_activations == o.onednn_activations &&
               onednn_stored == o.onednn_stored;
    }
};

std::vector<device_load_extras> read_load_extras() {
    std::vector<device_load_extras> out;
    for (int d = 0; d < test_physical_device_count() && d < GGML_SYCL_MAX_DEVICES; ++d) {
        const load_published_terms t = load_published_terms::snapshot(d);
        device_load_extras         s;
        s.act_in_kv          = t.ring_activation_in_kv;
        s.out_in_kv          = t.ring_output_in_kv;
        s.act_row            = t.ring_activation_row;
        s.out_row            = t.ring_output_row;
        s.n_expert           = t.ring_n_expert;
        s.n_expert_used      = t.ring_n_expert_used;
        s.n_ubatch           = t.ring_n_ubatch;
        s.onednn_weights     = t.onednn_weights;
        s.onednn_activations = t.onednn_activations;
        s.onednn_stored      = t.onednn_stored;
        out.push_back(s);
    }
    return out;
}

void write_load_extras(const std::vector<device_load_extras> & extras) {
    for (size_t d = 0; d < extras.size(); ++d) {
        const int                  dev = static_cast<int>(d);
        const device_load_extras & s   = extras[d];
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(dev, s.act_in_kv, s.out_in_kv);
        unified_cache_set_planned_pp_moe_onednn_row_bytes(dev, s.act_row, s.out_row, s.n_expert, s.n_expert_used);
        unified_cache_set_planned_pp_moe_onednn_n_ubatch(dev, s.n_ubatch);
        // The pair's setter stores the halves' total as the sum; the sum goes in after it.
        unified_cache_set_planned_onednn_scratchpad_pair(dev, s.onednn_weights, s.onednn_activations);
        unified_cache_set_planned_onednn_scratchpad_bytes(dev, s.onednn_stored);
    }
}

// A value per device no load of the shaped inventories publishes: both ring slots in the KV zone, row bytes, expert
// counts and a micro-batch no inventory here carries, and a pair whose stored sum a reserve raised above its total.
std::vector<device_load_extras> sentinel_load_extras(size_t n) {
    std::vector<device_load_extras> out(n);
    for (size_t d = 0; d < n; ++d) {
        device_load_extras & s = out[d];
        const uint32_t       v = 7 + static_cast<uint32_t>(d);
        s.act_in_kv            = true;
        s.out_in_kv            = true;
        s.act_row              = 96 * v;
        s.out_row              = 192 * v;
        s.n_expert             = v;
        s.n_expert_used        = 2;
        s.n_ubatch             = 70 + v;
        s.onednn_weights       = 6 * MiB + d;
        s.onednn_activations   = 1 * MiB + d;
        s.onednn_stored        = 9 * MiB + d;
    }
    return out;
}

// One staged early pass of `inv` in a load transaction of its own, ended uncommitted. `during` and `during_extras` read
// the load's device-wide terms between the stage and the end, `published_weight` the ring weight slot the stage left on
// device 0. `before_end`, when given, runs last before the end: an arm takes its holds there (hold_arenas).
struct staged_early_load {
    ggml_sycl_lifecycle_result        begin = GGML_SYCL_LIFECYCLE_OK;
    ggml_sycl_lifecycle_result        stage = GGML_SYCL_LIFECYCLE_OK;
    std::vector<device_planner_state> during;
    std::vector<device_load_extras>   during_extras;
    size_t                            published_weight = 0;
};

staged_early_load stage_early_and_end(const ggml_sycl_tensor_inventory & inv,
                                      const std::function<void()> &      before_end = nullptr) {
    staged_early_load  out;
    ggml_sycl_load_txn txn{};
    out.begin = ggml_backend_sycl_model_load_begin(&txn);
    if (out.begin != GGML_SYCL_LIFECYCLE_OK) {
        return out;
    }
    out.stage            = ggml_backend_sycl_stage_load_inventory_plan(&inv, nullptr, true);
    out.during           = read_planner_state();
    out.during_extras    = read_load_extras();
    out.published_weight = unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(0);
    if (before_end) {
        before_end();
    }
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    return out;
}

// (e) A load whose plan needs a larger arena zone while a first allocation holds the arena is refused by name, and the
// allocation keeps its memory. The VRAM arena's zones grow only by a rebuild, which is refused while anything points
// into the arena, and the stage used to abort the process there. A sentinel goes first into every planner global and
// every other device-wide term a load publishes (device_load_extras: the ring's KV-zone slots, row bytes, expert counts
// and micro-batch, the oneDNN pair and its stored sum); one that does not read back VOIDs the arm before any check. A
// load's entry ensure sizes the zones before its own planner publishes, so it reads the MoE CONTROL requirement and the
// host scatter scratch as they stand, here the sentinels' 4 KiB and 8 KiB, which no arm above sized the arena for. So
// the shaped inventory is staged once with nothing holding the arena, which rebuilds the zones for the sentinels too.
// Before its end a 4 KiB RUNTIME-zone page in every device's arena stands for the first model, device 0's being the
// allocation followed below: with no model live the end would release each arena nothing holds, and put the published
// terms back at their start values over the sentinels. Two loads are staged beside the pages, each ended uncommitted:
//   control: the shaped inventory, whose zones the sizing load sized for the sentinels, stages OK, and every sentinel
//            reads back after its end;
//   refusal: the same inventory with a PP MoE ring weight slot 256 MiB larger than device 0's RUNTIME zone, at depth 1,
//            returns PLAN_REJECTED from the stage after one ERROR line that names device 0, the RUNTIME need and have
//            and the allocations that hold the arena; every sentinel reads back right after the stage and after the
//            end; the RUNTIME zone keeps its capacity; the allocation resolves to the same device pointer, and the
//            registry still holds it in the RUNTIME zone.
// Engagement: the sizing load staged OK with nothing holding the arena, so the zones hold the sentinels' terms, and
// every device's arena was active before its end and held through it; the allocation landed in device 0's RUNTIME
// zone; the control staged OK beside it, so a refusal is the
// oversized term's and not the allocation's alone; a refusal leg that returned OK had published the oversized ring (it
// owed a refusal), otherwise the ring was not planned and the arm is VOID. A control that did not stage leaves the
// refusal leg running: the arm is VOID for it only if no check failed, and a FAIL carries its reason after the FAIL's.
// Mutants: the abort kept (the binary dies); the refusal not propagated (the stage returns OK); the published terms not
// put back (a global reads the refused load's ring, not its sentinel); a restore that leaves the KV-zone slots, row
// bytes, expert counts, micro-batch, pair or sum (that term reads the refused load's value); a rebuild, eviction or
// zone reset under the allocation (the RUNTIME capacity moves, or the allocation no longer resolves where it was).
arm_result arm_e_held_arena_refuses_the_load() {
    arm_result      r;
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        void_arm(r, "device 0 has no cache, so nothing can hold its arena's zones");
        return r;
    }

    const std::vector<device_planner_state> original        = read_planner_state();
    const std::vector<device_planner_state> sentinel        = sentinel_planner_state(original.size());
    const std::vector<device_load_extras>   original_extras = read_load_extras();
    const std::vector<device_load_extras>   sentinel_extras = sentinel_load_extras(original_extras.size());
    const auto                              put_back        = [&]() {
        write_planner_state(original);
        write_load_extras(original_extras);
    };
    write_planner_state(sentinel);
    write_load_extras(sentinel_extras);
    if (!(read_planner_state() == sentinel) || !(read_load_extras() == sentinel_extras)) {
        put_back();
        void_arm(r, "the sentinel planner state or load terms did not read back, so no check below has a reference");
        return r;
    }
    mapped_qwen38_inventory fits;
    std::vector<mem_handle> held;  // released at the end of the arm
    bool                    arenas_held = false;
    const staged_early_load sizing      = stage_early_and_end(fits.inv, [&]() { arenas_held = hold_arenas(held); });
    if (sizing.begin != GGML_SYCL_LIFECYCLE_OK || sizing.stage != GGML_SYCL_LIFECYCLE_OK) {
        put_back();
        void_arm(r, "the sizing load's begin returned " + num(sizing.begin) + " and its stage " + num(sizing.stage) +
                        " with nothing holding the arena, so the zones may not hold the sentinels' terms");
        return r;
    }
    if (!arenas_held) {
        put_back();
        void_arm(r, "an arena was not active before the sizing load's end, or its 4 KiB RUNTIME-zone page was refused");
        return r;
    }

    const mem_handle & holder = held.front();
    const resolved_ptr first  = holder.resolve();
    alloc_metadata     meta{};
    if (!first || !first.on_device || !unified_lookup(first.ptr, &meta) || meta.zone != vram_zone_id::RUNTIME) {
        put_back();
        void_arm(r, "the allocation is not in device 0's RUNTIME zone, so it does not hold the arena");
        return r;
    }
    const size_t runtime_capacity = cache->zone_capacity(vram_zone_id::RUNTIME);

    // A control that did not stage does not stop the arm: the refusal leg still runs, and settle() reports the arm VOID
    // with the control's reason only when no check failed, and a FAIL with that reason after its own (the harness keeps
    // the first FAIL or VOID, so a VOID taken here would hide every later FAIL).
    std::string             control_void;
    const staged_early_load control = stage_early_and_end(fits.inv);
    if (control.begin != GGML_SYCL_LIFECYCLE_OK || control.stage != GGML_SYCL_LIFECYCLE_OK) {
        control_void = "the control load's begin returned " + num(control.begin) + " and its stage " +
                       num(control.stage) + " beside the allocation, so a refusal would not be the oversized term's";
    }
    const auto settle = [&]() -> arm_result {
        if (control_void.empty()) {
            return r;
        }
        arm_result out = r;
        if (out.result == outcome::PASS) {
            out.result = outcome::VOID;
            out.why    = control_void;
        } else {
            out.why += "; and " + control_void;
        }
        return out;
    };
    expect(r, read_planner_state() == sentinel,
           "after the control load's uncommitted end a planner global reads that load's value, not its sentinel");
    expect(r, read_load_extras() == sentinel_extras,
           "after the control load's uncommitted end a ring KV-zone slot, row byte, micro-batch or oneDNN pair term "
           "reads that load's value, not its sentinel");

    mapped_qwen38_inventory too_big;
    too_big.inv.pp_moe_onednn_weight_slot_bytes = runtime_capacity + 256 * MiB;
    too_big.inv.pp_moe_onednn_scratch_bytes     = too_big.inv.pp_moe_onednn_weight_slot_bytes;
    too_big.inv.pp_moe_onednn_ring_depth        = 1;
    log_clear();
    const staged_early_load                 refused = stage_early_and_end(too_big.inv);
    const std::vector<std::string>          named   = lines_with("[SYCL-PLAN] model load refused on device 0: ");
    const std::vector<device_planner_state> after   = read_planner_state();
    const std::vector<device_load_extras>   after_x = read_load_extras();
    put_back();

    if (refused.begin != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the refused load's transaction did not begin (" + num(refused.begin) + ")");
        return settle();
    }
    if (refused.stage == GGML_SYCL_LIFECYCLE_OK &&
        refused.published_weight != too_big.inv.pp_moe_onednn_weight_slot_bytes) {
        void_arm(r, "the stage returned OK and published a ring weight slot of " + num(refused.published_weight) +
                        ", not the oversized " + num(too_big.inv.pp_moe_onednn_weight_slot_bytes) +
                        ", so no zone had to grow");
        return settle();
    }
    expect(r, refused.stage == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
           "a load whose ring needs " + num(too_big.inv.pp_moe_onednn_weight_slot_bytes) + " bytes beside a " +
               num(runtime_capacity) + "-byte RUNTIME zone the allocation holds returned " + num(refused.stage) +
               ", not PLAN_REJECTED");
    expect(r, named.size() == 1,
           "the refused load logged " + num(named.size()) + " refusal lines naming device 0, want 1");
    expect(r,
           named.empty() ||
               (named[0].find("RUNTIME need ") != std::string::npos && named[0].find(" short)") != std::string::npos &&
                named[0].find("hold the arena") != std::string::npos),
           "the refusal line does not name the RUNTIME need and have, its shortfall and the holders: " +
               (named.empty() ? std::string("(none)") : named[0]));
    for (size_t d = 0; d < refused.during.size(); ++d) {
        expect(r, refused.during[d] == sentinel[d],
               "device " + num(d) + ": right after the refused stage a planner global reads the refused load's value " +
                   "(ring weight " + num(refused.during[d].ring_weight) + "), not its sentinel");
    }
    for (size_t d = 0; d < after.size(); ++d) {
        expect(r, after[d] == sentinel[d],
               "device " + num(d) + ": after the refused load's end a planner global reads " + "ring weight " +
                   num(after[d].ring_weight) + ", not its sentinel");
    }
    for (size_t d = 0; d < refused.during_extras.size() && d < sentinel_extras.size(); ++d) {
        expect(r, refused.during_extras[d] == sentinel_extras[d],
               "device " + num(d) + ": right after the refused stage the ring's KV-zone slots, row bytes, expert " +
                   "counts or micro-batch, or the oneDNN pair or its sum, read the refused load's (micro-batch " +
                   num(refused.during_extras[d].n_ubatch) + ", sum " + num(refused.during_extras[d].onednn_stored) +
                   "), not their sentinels");
    }
    for (size_t d = 0; d < after_x.size() && d < sentinel_extras.size(); ++d) {
        expect(r, after_x[d] == sentinel_extras[d],
               "device " + num(d) + ": after the refused load's end the ring's KV-zone slots, row bytes, expert " +
                   "counts or micro-batch, or the oneDNN pair or its sum, read micro-batch " +
                   num(after_x[d].n_ubatch) + ", sum " + num(after_x[d].onednn_stored) + ", not their sentinels");
    }
    expect(r, cache->arena_active() && cache->zone_capacity(vram_zone_id::RUNTIME) == runtime_capacity,
           "the RUNTIME zone moved from " + num(runtime_capacity) + " to " +
               num(cache->zone_capacity(vram_zone_id::RUNTIME)) + " bytes under the live allocation");
    const resolved_ptr again = holder.resolve();
    alloc_metadata     still{};
    expect(r, again.ptr == first.ptr && again.on_device,
           "the allocation no longer resolves to the device pointer it had before the refused load");
    expect(r, unified_lookup(first.ptr, &still) && still.zone == vram_zone_id::RUNTIME && still.id == meta.id,
           "the registry no longer holds the allocation in the RUNTIME zone");
    return settle();
}

// (f) A load refused in its early pass keeps the PP MoE ring a live context left in the shared KV zone, and the ring's
// KV-zone flags. The ring outlives the context that sized it, and a load releases a ring that still holds KV-zone bytes
// in the stage's effect step, after every refusal of the pass. The one release in a decide step is the entry ensure's
// last resort, taken only when the ring's slot buffers are the only holders of an arena whose zones must grow
// (arm (h)): a load refused in its early pass must leave any other ring, which a live context may still use.
// Device 0's arena is reserved as a load's start reserves it, and device 0 gets a one-slot ring of 1 MiB slots whose
// activation and output slots are planned in the KV zone, and a 4 KiB RUNTIME-zone allocation beside it, so the ring
// is never the arena's only holder and the entry ensure has no last resort to take. A load whose ring weight slot is
// 256 MiB larger than the RUNTIME zone is then staged early and ended uncommitted. The ring is a live context's, so a
// model is marked live from right after the begin, which publishes the registry's mask, through the end: with none
// live, the refused stage's abort and the end each release the ring with everything else sized for the last model
// (arm (j)).
// Engagement: the ring was reserved and holds KV-zone bytes, the allocation landed in the RUNTIME zone, and the stage
// returned PLAN_REJECTED.
// Mutants: the ring released where the load's planning state is reset, before the planner's refusals (the ring's
// KV-zone bytes read 0 after the refused stage, and the release line is logged); the release of the last model run
// while a model is live (the ring's KV-zone bytes read 0 after the stage or the end).
arm_result arm_f_early_refusal_keeps_the_kv_zone_ring() {
    arm_result      r;
    unified_cache * cache = device0_arena_cache();
    if (cache == nullptr) {
        void_arm(r, "device 0 has no active VRAM arena, so no ring can sit in its KV zone");
        return r;
    }
    // The ring's planned terms and KV-zone flags as found, written back on every exit.
    const size_t   found_weight = unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(0);
    const size_t   found_act    = unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(0);
    const size_t   found_out    = unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(0);
    const uint32_t found_depth  = unified_cache_get_planned_pp_moe_onednn_ring_depth(0);
    const bool     found_act_kv = unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone(0);
    const bool     found_out_kv = unified_cache_get_planned_pp_moe_onednn_output_in_kv_zone(0);
    const auto     put_back     = [&]() {
        unified_cache_set_planned_pp_moe_onednn_scratch(0, found_weight, found_act, found_out, found_depth);
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(0, found_act_kv, found_out_kv);
    };

    constexpr size_t slot = 1 * MiB;
    unified_cache_set_planned_pp_moe_onednn_scratch(0, slot, slot, slot, 1);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(0, true, true);
    if (!cache->reserve_pp_moe_onednn_scratch(slot, slot, slot, 1)) {
        put_back();
        void_arm(r, "a one-slot ring of 1 MiB slots was not reserved on device 0");
        return r;
    }
    const size_t held = unified_cache_get_pp_moe_onednn_kv_zone_bytes_held(0);
    if (held == 0) {
        (void) cache->release_pp_moe_onednn_scratch_ring();
        put_back();
        void_arm(r, "the ring holds no bytes in the KV zone, so a release could not show");
        return r;
    }

    alloc_request req{};
    req.queue                               = &cache->get_queue();
    req.device                              = 0;
    req.size                                = 4096;
    req.intent.role                         = alloc_role::GRAPH_TMP;
    req.intent.category                     = runtime_category::GRAPH;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;
    allocation_result allocation            = unified_allocate_owner(req);
    mem_handle        holder;
    alloc_metadata    meta{};
    if (allocation) {
        holder = mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
    }
    const resolved_ptr where = holder.resolve();
    if (!where || !where.on_device || !unified_lookup(where.ptr, &meta) || meta.zone != vram_zone_id::RUNTIME) {
        holder = {};
        (void) cache->release_pp_moe_onednn_scratch_ring();
        put_back();
        void_arm(r, "the 4 KiB allocation is not in device 0's RUNTIME zone, so the ring could be the only holder");
        return r;
    }

    mapped_qwen38_inventory too_big;
    too_big.inv.pp_moe_onednn_weight_slot_bytes = cache->zone_capacity(vram_zone_id::RUNTIME) + 256 * MiB;
    too_big.inv.pp_moe_onednn_scratch_bytes     = too_big.inv.pp_moe_onednn_weight_slot_bytes;
    too_big.inv.pp_moe_onednn_ring_depth        = 1;
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        holder = {};
        (void) cache->release_pp_moe_onednn_scratch_ring();
        put_back();
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    log_clear();
    ggml_sycl_lifecycle_result rc            = GGML_SYCL_LIFECYCLE_OK;
    size_t                     held_staged   = 0;
    bool                       act_kv_staged = false;
    bool                       out_kv_staged = false;
    {
        // A refused stage aborts the load's owner effects itself, so the mark covers the stage as well as the end.
        const live_model_mark live;
        rc            = ggml_backend_sycl_stage_load_inventory_plan(&too_big.inv, nullptr, true);
        held_staged   = unified_cache_get_pp_moe_onednn_kv_zone_bytes_held(0);
        act_kv_staged = unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone(0);
        out_kv_staged = unified_cache_get_planned_pp_moe_onednn_output_in_kv_zone(0);
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    }
    const size_t held_ended   = unified_cache_get_pp_moe_onednn_kv_zone_bytes_held(0);
    const bool   act_kv_ended = unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone(0);
    const bool   out_kv_ended = unified_cache_get_planned_pp_moe_onednn_output_in_kv_zone(0);
    const size_t released     = lines_with("[SYCL-PLAN] model load released the PP MoE oneDNN scratch ring").size();

    if (rc != GGML_SYCL_LIFECYCLE_PLAN_REJECTED) {
        void_arm(r, "the load whose ring cannot fit the RUNTIME zone returned " + num(rc) + ", not PLAN_REJECTED");
    }
    expect(r, held_staged == held,
           "right after the refused stage the ring holds " + num(held_staged) + " bytes in the KV zone, not its " +
               num(held) + ": the refused load released it");
    expect(r, held_ended == held,
           "after the refused load's end the ring holds " + num(held_ended) + " bytes in the KV zone, not its " +
               num(held));
    expect(r, released == 0,
           "the refused load logged " + num(released) + " ring release line(s), want none: it changed device 0");
    expect(r, act_kv_staged && out_kv_staged && act_kv_ended && out_kv_ended,
           "the ring's KV-zone flags read activation=" + num(act_kv_staged) + "/" + num(act_kv_ended) + " output=" +
               num(out_kv_staged) + "/" + num(out_kv_ended) + " (after the stage/after the end), not the planned 1/1");

    holder = {};
    expect(r, cache->release_pp_moe_onednn_scratch_ring(), "the ring kept after the refused load did not release");
    put_back();
    return r;
}

// (h) A load releases a ring that alone holds the arena, and asks for its zones again and gets them. The ring outlives
// the context that sized it, so a freed context can leave its slot buffers as the only holders of an arena whose zones
// a later load must grow; the entry ensure then releases the ring as its last resort and asks for the zones again,
// where any other holder refuses the load. Device 0's arena is reserved as a load's start reserves it, device 0 gets
// arm (f)'s one-slot ring of 1 MiB slots with nothing beside it, and the load arm (f) stages (a ring weight slot
// 256 MiB larger than the RUNTIME zone) is staged early. Between the last resort's release line and device 0's first
// plan line, which device 0 logs after its entry ensure and before any other device's planning starts, the stage must
// rebuild the arena once, to a RUNTIME zone at least the one its refused first ask needed. The arm's own ensure inside
// the open load must then get the zones too, and may rebuild once more: the plan publishes terms after the entry
// ensure sized the zones (the MoE control and host scatter terms), and the next ensure of a load takes them in, as in
// every load. The load is then ended uncommitted, and with no model live and nothing holding the rebuilt arena, its
// end releases the arena. The arm is written for the registered environment, where GGML_SYCL_MOE_MULTI_GPU is unset:
// with it set to 1 the plan line names device -1, and the window would run to the end of the log.
// Engagement: device 0's zones held no byte and its chunks no lease before the ring, and the ring added live bytes; a
// stage that refused another device, or refused device 0 before any release for a cause this arm does not set up, or
// for a reserve the device could not make, is VOID. A refusal that names only the ring as the holder of device 0's
// arena, and any other refusal of device 0 after the last resort's release, is a FAIL.
// Mutants: no last resort (the stage refuses device 0 with only the ring holding it); the release not followed by a
// second ask for the zones (the stage logs no rebuild after the release, and refuses the load if the ask's answer was
// the refusal); the ring released twice (two release lines).
arm_result arm_h_load_releases_a_lone_ring() {
    arm_result      r;
    unified_cache * cache = device0_arena_cache();
    if (cache == nullptr) {
        void_arm(r, "device 0 has no active VRAM arena, so no ring can hold it");
        return r;
    }
    // Every byte the zones count and every lease on the arena's chunks: the entry ensure's own holder readings.
    const auto zone_bytes = [&]() {
        size_t bytes = 0;
        for (size_t z = 0; z < static_cast<size_t>(vram_zone_id::COUNT); ++z) {
            bytes += cache->zone_used(static_cast<vram_zone_id>(z));
        }
        return bytes;
    };
    uint32_t leases = 0;
    for (int c = 0; c < cache->chunk_count(); ++c) {
        leases += cache->arena_chunk_lease_count(c);
    }
    const size_t before = zone_bytes();
    if (before != 0 || leases != 0) {
        void_arm(r, "device 0's zones hold " + num(before) + " bytes and its chunks " + num(leases) + " leases " +
                        "before the ring, so the ring would not be the arena's only holder");
        return r;
    }
    const size_t   found_weight = unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(0);
    const size_t   found_act    = unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(0);
    const size_t   found_out    = unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(0);
    const uint32_t found_depth  = unified_cache_get_planned_pp_moe_onednn_ring_depth(0);
    const bool     found_act_kv = unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone(0);
    const bool     found_out_kv = unified_cache_get_planned_pp_moe_onednn_output_in_kv_zone(0);
    const auto     put_back     = [&]() {
        (void) cache->release_pp_moe_onednn_scratch_ring();
        unified_cache_set_planned_pp_moe_onednn_scratch(0, found_weight, found_act, found_out, found_depth);
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(0, found_act_kv, found_out_kv);
    };

    constexpr size_t slot = 1 * MiB;
    unified_cache_set_planned_pp_moe_onednn_scratch(0, slot, slot, slot, 1);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(0, true, true);
    if (!cache->reserve_pp_moe_onednn_scratch(slot, slot, slot, 1) || zone_bytes() == 0) {
        put_back();
        void_arm(r, "a one-slot ring of 1 MiB slots was not reserved in device 0's arena");
        return r;
    }

    mapped_qwen38_inventory too_big;
    too_big.inv.pp_moe_onednn_weight_slot_bytes = cache->zone_capacity(vram_zone_id::RUNTIME) + 256 * MiB;
    too_big.inv.pp_moe_onednn_scratch_bytes     = too_big.inv.pp_moe_onednn_weight_slot_bytes;
    too_big.inv.pp_moe_onednn_ring_depth        = 1;
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        put_back();
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc  = ggml_backend_sycl_stage_load_inventory_plan(&too_big.inv, nullptr, true);
    const std::vector<std::string>   log = log_snapshot();
    const size_t                     held_after = unified_cache_get_pp_moe_onednn_kv_zone_bytes_held(0);
    log_clear();
    zone_arena_refusal again{};
    const bool         zones_got    = rc == GGML_SYCL_LIFECYCLE_OK && cache->ensure_planned_arena_zones(&again);
    // The arm's own ensure's rebuild lines, counted apart from the stage's: the log was cleared after the stage.
    const size_t       own_rebuilds = log_count_with("[VRAM-ARENA] Rebuilding unused early arena for planned zones");
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    put_back();

    const char * const release_line     = "[SYCL-PLAN] model load released the PP MoE oneDNN scratch ring";
    const char * const last_resort_text = "on device 0: it was the only holder of an arena";
    const char * const refusal_line     = "[SYCL-PLAN] model load refused on device ";
    const char * const held_line        = "[VRAM-ARENA] planned zones exceed active arena but live allocations prevent";
    const auto         has              = [](const std::string & line, const char * text) {
        return line.find(text) != std::string::npos;
    };
    size_t                   released = 0, last_resort = 0, kept = 0, ring_only = 0;
    size_t                   from = log.size();  // the last resort's release line
    std::vector<std::string> refused;
    std::string              device0_refusal;
    for (size_t i = 0; i < log.size(); ++i) {
        const std::string & line = log[i];
        if (has(line, release_line)) {
            ++released;
            if (has(line, last_resort_text)) {
                ++last_resort;
                from = std::min(from, i);
            }
        }
        kept += has(line, "[SYCL-PLAN] model load kept the PP MoE oneDNN scratch ring") ? 1 : 0;
        if (has(line, refusal_line)) {
            refused.push_back(line);
            if (has(line, "refused on device 0:") && device0_refusal.empty()) {
                device0_refusal = line;
            }
            ring_only +=
                has(line, "refused on device 0:") && has(line, "Only the cache's PP MoE ring holds it") ? 1 : 0;
        }
    }
    // The RUNTIME zone the refused first ask needed ("need ... runtime=<MB>"), from its last held line before the
    // release, and the stage's own rebuilds after the release, up to device 0's first plan line.
    double need_runtime = -1.0;
    for (size_t i = 0; i < from; ++i) {
        if (has(log[i], held_line)) {
            const size_t at = log[i].find(" runtime=");
            need_runtime    = at == std::string::npos ? -1.0 : std::strtod(log[i].c_str() + at + 9, nullptr);
        }
    }
    std::vector<std::string> rebuilds;
    for (size_t i = from; i < log.size() && !has(log[i], "[HOST-LEDGER] device=0 plan host_budget="); ++i) {
        if (has(log[i], "[VRAM-ARENA] Rebuilding unused early arena for planned zones")) {
            rebuilds.push_back(log[i]);
        }
    }
    double rebuilt_runtime = -1.0;  // the "runtime <from>-><to> MB" target of the one rebuild
    if (rebuilds.size() == 1) {
        const size_t at    = rebuilds[0].find("runtime ");
        const size_t arrow = at == std::string::npos ? std::string::npos : rebuilds[0].find("->", at);
        rebuilt_runtime    = arrow == std::string::npos ? -1.0 : std::strtod(rebuilds[0].c_str() + arrow + 2, nullptr);
    }

    // A refusal of device 0 after the last resort's release is the product's, not a missing precondition, unless the
    // device could not reserve the arena the zones need.
    const bool refused_after_release =
        last_resort != 0 && !device0_refusal.empty() && !has(device0_refusal, "could not reserve an arena");
    if (rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED && ring_only == 0 && kept == 0 && !refused_after_release) {
        if (!refused.empty()) {
            void_arm(r, "the stage refused for a cause this arm does not set up: " + refused[0]);
            return r;
        }
        if (last_resort == 0) {
            void_arm(r, "the stage returned PLAN_REJECTED with no zone refusal and no ring release");
            return r;
        }
    } else if (rc != GGML_SYCL_LIFECYCLE_OK && rc != GGML_SYCL_LIFECYCLE_PLAN_REJECTED) {
        void_arm(r, "the stage returned " + num(rc));
        return r;
    }
    expect(r, ring_only == 0,
           "the stage refused device 0 with only the ring holding its arena: the ring was not released");
    expect(r, kept == 0,
           "the stage kept the ring as the arena's only holder, though this arm holds no claim on a slot");
    expect(r, !refused_after_release, "after the last resort's release the stage refused device 0: " + device0_refusal);
    expect(r, rc == GGML_SYCL_LIFECYCLE_OK,
           "the stage returned " + num(rc) + " after " + num(last_resort) + " last-resort ring release line(s)");
    expect(r, released == 1 && last_resort == 1,
           "the stage logged " + num(released) + " ring release line(s), " + num(last_resort) + " of them the last " +
               "resort's, want exactly one, the last resort's");
    expect(r, held_after == 0, "after its release line the ring still holds " + num(held_after) + " KV-zone bytes");
    expect(r, last_resort == 0 || rebuilds.size() == 1,
           "between the last resort's release and device 0's plan line the stage rebuilt the arena " +
               num(rebuilds.size()) + " time(s), want 1: it did not ask for the zones again and get them");
    expect(r, rebuilds.size() != 1 || (need_runtime >= 0.0 && rebuilt_runtime >= need_runtime),
           "the stage rebuilt device 0's RUNTIME zone to " + std::to_string(rebuilt_runtime) + " MB, not at least " +
               "the " + std::to_string(need_runtime) + " MB its refused first ask needed");
    expect(r, rc != GGML_SYCL_LIFECYCLE_OK || zones_got,
           "inside the open load the arm's own ensure refused the zones the stage left (reason " +
               num(static_cast<int>(again.reason)) + ", after " + num(own_rebuilds) + " rebuild line(s) of its own)");
    return r;
}

// One staged early pass of `inv` in a load transaction of its own, ended uncommitted, with what device 0's ONEDNN zone
// was asked to hold read inside the open load: the planned requirement (the oneDNN pair and the Graph floor, as the
// zone ensure reads it), the pair alone, the ONEDNN zone the ensure sizes from them, and the Graph room a context check
// reads over the zone; and the Graph room once the load has ended.
struct graph_floor_load {
    ggml_sycl_lifecycle_result begin           = GGML_SYCL_LIFECYCLE_OK;
    ggml_sycl_lifecycle_result stage           = GGML_SYCL_LIFECYCLE_OK;
    size_t                     planned_onednn  = 0;
    size_t                     planned_pair    = 0;
    size_t                     zone_need       = 0;
    size_t                     room            = 0;
    bool                       room_read       = false;
    size_t                     room_after      = 0;
    bool                       room_after_read = false;
};

// The ONEDNN zone the ensure sizes from a planned requirement and pair: at least 256 MiB, clamped to a quarter of the
// available budget but never below the pair, which has no DIRECT path. What the clamp removes is served on the DIRECT
// path, so a requirement clamped to what the zone already holds leaves nothing short.
// Zero when device 0 has no cache.
size_t onednn_zone_need(unified_cache * cache, size_t planned_onednn, size_t planned_pair) {
    if (cache == nullptr) {
        return 0;
    }
    const size_t need = std::max<size_t>(256u * 1024 * 1024, planned_onednn);
    return std::min(need, std::max(cache->available_budget() / 4, planned_pair));
}

// `before_end`, when given, runs last before the end, as stage_early_and_end's does.
graph_floor_load stage_early_reading_the_graph_room(const ggml_sycl_tensor_inventory & inv,
                                                    const std::function<void()> &      before_end = nullptr) {
    graph_floor_load   out;
    ggml_sycl_load_txn txn{};
    out.begin = ggml_backend_sycl_model_load_begin(&txn);
    if (out.begin != GGML_SYCL_LIFECYCLE_OK) {
        return out;
    }
    out.stage          = ggml_backend_sycl_stage_load_inventory_plan(&inv, nullptr, true);
    out.planned_onednn = unified_cache_get_planned_onednn_scratchpad_bytes(0);
    out.planned_pair   = unified_cache_get_planned_onednn_scratchpad_bytes_stored(0);
    out.zone_need      = onednn_zone_need(get_unified_cache_for_device(0), out.planned_onednn, out.planned_pair);
    out.room_read      = unified_cache_get_onednn_graph_room(0, &out.room);
    if (before_end) {
        before_end();
    }
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    out.room_after_read = unified_cache_get_onednn_graph_room(0, &out.room_after);
    return out;
}

// Device 0's three arena zone capacities.
struct arena_zone_capacities {
    size_t scratch = 0;
    size_t onednn  = 0;
    size_t runtime = 0;

    bool operator==(const arena_zone_capacities & o) const {
        return scratch == o.scratch && onednn == o.onednn && runtime == o.runtime;
    }
};

arena_zone_capacities read_zone_capacities(unified_cache * cache) {
    return { cache->zone_capacity(vram_zone_id::SCRATCH), cache->zone_capacity(vram_zone_id::ONEDNN),
             cache->zone_capacity(vram_zone_id::RUNTIME) };
}

std::string zone_capacities_text(const arena_zone_capacities & z) {
    return "SCRATCH " + num(z.scratch) + ", ONEDNN " + num(z.onednn) + ", RUNTIME " + num(z.runtime);
}

// Device 0's zones as the process's first load finds them, read by main before any arm: the arena the cache reserved
// when it was made, from the start values of every term a load publishes. Arm (j) checks the arena a load's start
// reserves after a refused load against them.
arena_zone_capacities g_first_zones;
bool                  g_first_zones_read = false;

// Device 0's zones inside the process's first load: the plain shaped inventory staged early in a load of its own, read
// by main before any arm (read_first_load_zones). Arm (j) checks the zone decision of a load after a refused load
// against them. Unread, the reason is in g_first_load_zones_void.
arena_zone_capacities g_first_load_zones;
bool                  g_first_load_zones_read = false;
std::string           g_first_load_zones_void;

// Stages the first load's reference. Its end releases every arena nothing holds, so every cache reserves its arena
// again after it, as the next load's start would, and the arms find the arenas the caches were made with: device 0's
// zones must read g_first_zones again, or the reference is not read.
void read_first_load_zones() {
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr || !g_first_zones_read) {
        g_first_load_zones_void = "device 0's first arena was not read";
        return;
    }
    mapped_qwen38_inventory fits;
    bool                    active_inside = false;
    arena_zone_capacities   inside;
    const staged_early_load first = stage_early_and_end(fits.inv, [&]() {
        active_inside = cache->arena_active();
        inside        = active_inside ? read_zone_capacities(cache) : arena_zone_capacities{};
    });

    bool reserved = true;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        unified_cache *    c = get_unified_cache_for_device(d);
        zone_arena_refusal refusal{};
        reserved = (c == nullptr || c->reserve_released_arena(&refusal)) && reserved;
    }
    if (first.begin != GGML_SYCL_LIFECYCLE_OK || first.stage != GGML_SYCL_LIFECYCLE_OK || !active_inside) {
        g_first_load_zones_void = "the first load's begin returned " + num(first.begin) + " and its stage " +
                                  num(first.stage) + (active_inside ? "" : ", with no arena on device 0 inside it");
        return;
    }
    const bool                  active_after = cache->arena_active();
    const arena_zone_capacities after        = active_after ? read_zone_capacities(cache) : arena_zone_capacities{};
    if (!reserved || !active_after || !(after == g_first_zones)) {
        g_first_load_zones_void =
            "after the first load every arena was " + std::string(reserved ? "reserved again" : "not reserved again") +
            ", device 0's " + (active_after ? "with zones " + zone_capacities_text(after) : std::string("not active")) +
            ", not the first arena's " + zone_capacities_text(g_first_zones);
        return;
    }
    g_first_load_zones      = inside;
    g_first_load_zones_read = true;
}

// (i) A load whose only shortfall is the oneDNN Graph floor, in an arena a live allocation holds, keeps the zones as
// they are, and the reverse order takes the sufficient path. The holder is one 4 KiB allocation, not a second model:
// this arm is the narrow control of the keep, and the two co-resident model loads in both orders are
// test-sycl-coresident-graph-floor's. The ONEDNN zone holds the oneDNN primitive pair and the Graph scratch floor. The
// floor is a reservation: what the zone cannot hold of it is served on the DIRECT path under its cap, and a context
// whose Graph scratch does not fit is refused by name at the context. The zones grow only by a rebuild, which nothing
// live allows, so such a load keeps a short zone instead of being refused. Two shaped inventories differ only in
// n_head_ctx_max, which sizes the Graph floor and nothing else: 8 heads (a floor of a few MiB at the 512 x 512
// planning shape) and 2048 (3 GiB by the formula, held to the DIRECT cap of at most 1 GiB).
//   keep:    the small load stages with nothing holding the arena, so every zone holds what the large load needs but
//            the ONEDNN zone its floor; before its end, which would otherwise release every arena nothing holds, a
//            4 KiB RUNTIME-zone page in every device's arena holds it, device 0's being the allocation, and the large
//            load, ended uncommitted, stages OK with one "[VRAM-ARENA] device 0 keeps its ONEDNN zone" WARN and no
//            refusal line; all three zone capacities are unchanged; the Graph room it reads inside the open load is
//            below the planned floor (the zone could not hold it); once the load has ended uncommitted the Graph room
//            is what it was before the load (the failed load did not leave its zone plan stored); the allocation is
//            where it was;
//   reverse: with the pages released, the large load stages with nothing holding, so the zones hold its floor; before
//            its end new pages hold every arena, and the small load stages OK with no keep WARN and every zone
//            capacity unchanged.
// Engagement: each load beside nothing staged OK, and every device's arena was active before its end and held through
// it; inside the open large load beside the
// allocation the ONEDNN zone the ensure sizes (the planned requirement, at least 256 MiB, clamped to a quarter of the
// available budget but not below the pair) exceeds the zone, so the zones could not hold it as they stood (VOID
// otherwise: the Graph allocator is off, the plan spans both cards, the budget clamp leaves nothing short, or an
// earlier arm left a zone that holds it); after the reverse leg's large load the zone holds the planned requirement;
// each allocation lands in the RUNTIME zone; the Graph room inside the open load differs from the room before it, so
// the put-back check can see a plan left stored (VOID otherwise, with the two rooms named, once every other check of
// the arm has run).
// Mutants: the keep removed (the large load returns PLAN_REJECTED after a refusal line); the keep taken by a rebuild
// (a capacity moves, or the allocation no longer resolves where it was); the WARN dropped or repeated within the load;
// the failed load's zone plan left stored (the room after the load is the kept room, not the room before it); the
// reverse order kept short (a keep WARN, or a capacity moves).
// Not seen: the allocation holds no oneDNN pair, so the Graph room reads the zone less the plan's own pair whether the
// stored floor is clamped to that or larger, and the arm cannot tell a keep that stores the planned floor from the
// correct keep, nor a restore that stores the prior plan raw from the correct restore. Those are the keep source pin's
// (claim_one_rule_decides_the_keep and the put-back witnesses) and test-zone-sizing's (Cases 41 and 44).
arm_result arm_i_graph_floor_keeps_an_allocation_held_arena() {
    arm_result      r;
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        void_arm(r, "device 0 has no cache, so nothing can hold its arena's zones");
        return r;
    }
    const char * const      kept_warn = "[VRAM-ARENA] device 0 keeps its ONEDNN zone of ";
    const char * const      refused   = "[SYCL-PLAN] model load refused on device 0: ";
    mapped_qwen38_inventory small;
    mapped_qwen38_inventory large;
    small.inv.n_head_ctx_max = 8;
    large.inv.n_head_ctx_max = 2048;
    // Why the put-back check did not engage, if it did not: the arm is VOID for it at its end, after every FAIL.
    std::string put_back_unseen;

    // keep
    std::vector<mem_handle> keep_held;  // released at the end of this leg
    bool                    keep_arenas_held = false;
    const staged_early_load sizing =
        stage_early_and_end(small.inv, [&]() { keep_arenas_held = hold_arenas(keep_held); });
    if (sizing.begin != GGML_SYCL_LIFECYCLE_OK || sizing.stage != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the small load's begin returned " + num(sizing.begin) + " and its stage " + num(sizing.stage) +
                        " with nothing holding the arena, so the zones may not hold its terms");
        return r;
    }
    if (!keep_arenas_held) {
        void_arm(r, "an arena was not active before the small load's end, or its 4 KiB RUNTIME-zone page was refused");
        return r;
    }
    {
        const mem_handle &          holder           = keep_held.front();
        const resolved_ptr          first            = holder.resolve();
        const arena_zone_capacities before           = read_zone_capacities(cache);
        size_t                      room_before      = 0;
        const bool                  room_before_read = unified_cache_get_onednn_graph_room(0, &room_before);
        log_clear();
        const graph_floor_load         kept     = stage_early_reading_the_graph_room(large.inv);
        const std::vector<std::string> warned   = lines_with(kept_warn);
        const std::vector<std::string> refusals = lines_with(refused);
        const arena_zone_capacities    after    = read_zone_capacities(cache);
        if (kept.begin != GGML_SYCL_LIFECYCLE_OK) {
            void_arm(r, "the large load's transaction did not begin (" + num(kept.begin) + ")");
            return r;
        }
        if (kept.zone_need <= before.onednn) {
            void_arm(r, "inside the large load the ONEDNN zone the ensure sizes is " + num(kept.zone_need) +
                            " bytes (planned " + num(kept.planned_onednn) + "), which the " + num(before.onednn) +
                            "-byte zone holds, so nothing had to be kept (the Graph allocator is off, the plan spans " +
                            "both cards, the budget clamp leaves nothing short, or an earlier arm grew the zone)");
            return r;
        }
        expect(r, kept.stage == GGML_SYCL_LIFECYCLE_OK,
               "a load whose Graph floor alone outgrows the " + num(before.onednn) +
                   "-byte ONEDNN zone beside a live allocation returned " + num(kept.stage) + ", not OK" +
                   (refusals.empty() ? std::string() : ": " + refusals[0]));
        expect(r, warned.size() == 1, "the kept load logged " + num(warned.size()) + " keep WARN lines, want 1");
        expect(r, refusals.empty(), "the kept load logged a refusal line: " + (refusals.empty() ? "" : refusals[0]));
        expect(
            r, after == before,
            "a kept load moved the zones from " + zone_capacities_text(before) + " to " + zone_capacities_text(after));
        const size_t planned_floor =
            kept.planned_onednn > kept.planned_pair ? kept.planned_onednn - kept.planned_pair : 0;
        expect(r, kept.room_read && kept.room < planned_floor,
               "inside the kept load the Graph room reads " + num(kept.room) + " bytes (read " + num(kept.room_read) +
                   "), not below the " + num(planned_floor) + "-byte planned floor the zone could not hold");
        if (room_before_read && kept.room_read && kept.room != room_before) {
            expect(r, kept.room_after_read && kept.room_after == room_before,
                   "after the kept load ended uncommitted the Graph room reads " + num(kept.room_after) +
                       " bytes (read " + num(kept.room_after_read) + "), not the " + num(room_before) +
                       " bytes it read before the load: the failed load's zone plan stayed stored");
        } else {
            put_back_unseen = "the put-back check measured nothing: the Graph room read " + num(room_before) +
                              " bytes before the large load and " + num(kept.room) + " inside it (read " +
                              num(room_before_read) + " and " + num(kept.room_read) +
                              "), so a plan left stored would read the same";
        }
        const resolved_ptr again = holder.resolve();
        expect(r, again.ptr == first.ptr && again.on_device && holds_the_runtime_zone(holder),
               "the allocation no longer resolves to its device pointer in the RUNTIME zone after the kept load");
    }
    keep_held.clear();

    // reverse
    std::vector<mem_handle> reverse_held;  // released at the end of the arm
    bool                    reverse_arenas_held = false;
    log_clear();
    const graph_floor_load grown =
        stage_early_reading_the_graph_room(large.inv, [&]() { reverse_arenas_held = hold_arenas(reverse_held); });
    const arena_zone_capacities grown_zones = read_zone_capacities(cache);
    if (grown.begin != GGML_SYCL_LIFECYCLE_OK || grown.stage != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the large load's begin returned " + num(grown.begin) + " and its stage " + num(grown.stage) +
                        " with nothing holding the arena, so the reverse order has no zone sized for its floor");
        return r;
    }
    if (!reverse_arenas_held) {
        void_arm(r, "an arena was not active before the large load's end, or its 4 KiB RUNTIME-zone page was refused");
        return r;
    }
    if (grown_zones.onednn < grown.planned_onednn) {
        void_arm(r, "after the large load with nothing holding the arena the ONEDNN zone is " +
                        num(grown_zones.onednn) + " bytes, below its " + num(grown.planned_onednn) +
                        "-byte requirement, so the reverse order would not take the sufficient path");
        return r;
    }
    log_clear();
    const staged_early_load        second  = stage_early_and_end(small.inv);
    const std::vector<std::string> warned  = lines_with(kept_warn);
    const arena_zone_capacities    settled = read_zone_capacities(cache);
    expect(r, second.begin == GGML_SYCL_LIFECYCLE_OK && second.stage == GGML_SYCL_LIFECYCLE_OK,
           "the smaller load beside the allocation in zones sized for the larger floor returned begin " +
               num(second.begin) + " and stage " + num(second.stage) + ", not OK");
    expect(r, warned.empty(),
           "the smaller load took the keep, not the sufficient path: " + (warned.empty() ? std::string() : warned[0]));
    expect(r, settled == grown_zones,
           "the smaller load moved the zones from " + zone_capacities_text(grown_zones) + " to " +
               zone_capacities_text(settled));
    if (!put_back_unseen.empty()) {
        void_arm(r, put_back_unseen);
    }
    return r;
}

// The RUNTIME zone a "[VRAM-ARENA] Rebuilding unused early arena for planned zones" line rebuilds to, in bytes, or 0
// when the line names none.
size_t rebuilt_runtime_bytes(const std::string & line) {
    const size_t at    = line.find("runtime ");
    const size_t arrow = at == std::string::npos ? std::string::npos : line.find("->", at);
    if (arrow == std::string::npos) {
        return 0;
    }
    return static_cast<size_t>(std::strtod(line.c_str() + arrow + 2, nullptr) * static_cast<double>(MiB));
}

// (j) A load that ends refused while no model is live leaves nothing sized for it, and the next load plans and builds
// as the first load of the process did. The refused load's abort runs the release of the last model: every cache
// releases what it sized for a model and the arena nothing holds, and each device it emptied gets its published terms
// back at their start values, not at the values that stood before the load; the next load's start reserves the early
// arena again, from those values, and its zone decision is the first load's. Load A is the shaped inventory with a PP
// MoE ring weight slot 256 MiB larger than the RUNTIME zone of device 0's first arena, at depth 1. With nothing holding
// the arena its early pass rebuilds device 0's arena for that ring and then plans it, so A's plan alone is not refused:
// the stage's early failure hook ends the pass after device 0 planned (EFFECT_FAILED), the refusal whose abort runs
// the release, and A is ended uncommitted, the end a refused load takes. A sentinel goes into every planner global and
// every other device-wide term a load publishes after A's begin, whose start may put a cache's terms back, so terms put
// back to the values before the load read the sentinels. Load B stages the plain shaped inventory's early pass and is
// ended uncommitted.
//   A:         the armed stage returns EFFECT_FAILED; after A's end device 0's arena is not active, one line says
//              device 0's last release emptied it, and every device's published terms equal a load_published_terms{}
//              of start values;
//   B's begin: device 0's arena is active again, and its SCRATCH, ONEDNN and RUNTIME zones are those of the arena
//              device 0's cache reserved when it was made (g_first_zones);
//   B's stage: returns OK, and device 0's zones are those main read inside the process's first load of the same
//              inventory (g_first_load_zones).
// Engagement: the first arena's and the first load's zones were read before any arm; the hook is exported; the
// sentinels read back; A's stage logged a rebuild of the arena to a RUNTIME zone larger than the first arena's, so A
// sized the arena for itself. A stage that does not reach the hook leaves it armed, so a stage of the plain inventory
// in a load of its own takes it before the arm is VOID, and no later arm's stage meets it.
// Mutants: no release at a refused load's abort (the arena stays active, at A's zones, and B plans in them); the terms
// put back to the values before the load (a device reads its sentinels, and B plans from them); a reserve at the end
// (the arena is active after A's end); the start's reserve dropped (B's begin finds no arena); a reserve from terms
// other than the start values (B's begin zones differ from the first arena's); a release that leaves a term the zone
// decision reads (B's staged zones differ from the first load's).
arm_result arm_j_refused_load_leaves_nothing_sized_for_it() {
    arm_result      r;
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr || !g_first_zones_read) {
        void_arm(r, "device 0's first arena was not read before the arms, so a later reserve has nothing to equal");
        return r;
    }
    if (!g_first_load_zones_read) {
        void_arm(r, "main read no first load's zones before the arms, so B's zones have nothing to equal: " +
                        g_first_load_zones_void);
        return r;
    }
    using fail_next_fn            = void (*)();
    const fail_next_fn fail_early = reinterpret_cast<fail_next_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_sycl_reg(), "ggml_backend_sycl_test_fail_next_stage_inventory_plan_early_after_first_device"));
    if (fail_early == nullptr) {
        void_arm(r, "the stage's early failure hook is not exported");
        return r;
    }

    ggml_sycl_load_txn               a{};
    const ggml_sycl_lifecycle_result a_begin = ggml_backend_sycl_model_load_begin(&a);
    if (a_begin != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "load A's transaction did not begin (" + num(a_begin) + ")");
        return r;
    }
    const std::vector<device_planner_state> original        = read_planner_state();
    const std::vector<device_planner_state> sentinel        = sentinel_planner_state(original.size());
    const std::vector<device_load_extras>   original_extras = read_load_extras();
    const std::vector<device_load_extras>   sentinel_extras = sentinel_load_extras(original_extras.size());
    const auto                              put_back        = [&]() {
        write_planner_state(original);
        write_load_extras(original_extras);
    };
    write_planner_state(sentinel);
    write_load_extras(sentinel_extras);
    if (!(read_planner_state() == sentinel) || !(read_load_extras() == sentinel_extras)) {
        put_back();
        (void) ggml_backend_sycl_model_load_end(a, false, nullptr);
        void_arm(r, "the sentinel planner state or load terms did not read back, so a put-back to them would not show");
        return r;
    }

    mapped_qwen38_inventory grown;
    grown.inv.pp_moe_onednn_weight_slot_bytes = g_first_zones.runtime + 256 * MiB;
    grown.inv.pp_moe_onednn_scratch_bytes     = grown.inv.pp_moe_onednn_weight_slot_bytes;
    grown.inv.pp_moe_onednn_ring_depth        = 1;
    log_clear();
    fail_early();
    const ggml_sycl_lifecycle_result a_stage = ggml_backend_sycl_stage_load_inventory_plan(&grown.inv, nullptr, true);
    size_t                           rebuilt_runtime = 0;
    for (const std::string & line : lines_with("[VRAM-ARENA] Rebuilding unused early arena for planned zones: ")) {
        rebuilt_runtime = std::max(rebuilt_runtime, rebuilt_runtime_bytes(line));
    }
    (void) ggml_backend_sycl_model_load_end(a, false, nullptr);
    const bool                     active_after  = cache->arena_active();
    const size_t                   runtime_after = active_after ? cache->zone_capacity(vram_zone_id::RUNTIME) : 0;
    const char * const             last_release  = "[VRAM-ARENA] last release on device 0 (";
    const std::vector<std::string> released      = lines_with(last_release);
    const size_t                   emptied       = lines_with(last_release, "no model is live; the arena").size();
    std::vector<uint32_t>          differing;
    for (int d = 0; d < ggml_backend_sycl_get_device_count() && d < GGML_SYCL_MAX_DEVICES; ++d) {
        differing.push_back(load_published_terms::snapshot(d).differing(load_published_terms{}));
    }
    if (a_stage != GGML_SYCL_LIFECYCLE_EFFECT_FAILED) {
        // The hook fires only when a stage reaches it, so it may still be armed: the plain inventory's stage reaches
        // it, in a load of its own.
        put_back();
        mapped_qwen38_inventory plain;
        (void) stage_early_and_end(plain.inv);
        void_arm(r, "load A's armed early pass returned " + num(a_stage) + ", not EFFECT_FAILED");
        return r;
    }

    // Load B plans from what A's release left, so the sentinels stay until after its end.
    mapped_qwen38_inventory          fits;
    ggml_sycl_load_txn               b{};
    const ggml_sycl_lifecycle_result b_begin       = ggml_backend_sycl_model_load_begin(&b);
    const bool                       b_active      = cache->arena_active();
    const arena_zone_capacities      b_begin_zones = b_active ? read_zone_capacities(cache) : arena_zone_capacities{};
    ggml_sycl_lifecycle_result       b_stage       = b_begin;
    bool                             b_staged      = false;
    arena_zone_capacities            b_zones;
    if (b_begin == GGML_SYCL_LIFECYCLE_OK) {
        b_stage  = ggml_backend_sycl_stage_load_inventory_plan(&fits.inv, nullptr, true);
        b_staged = cache->arena_active();
        b_zones  = b_staged ? read_zone_capacities(cache) : arena_zone_capacities{};
        (void) ggml_backend_sycl_model_load_end(b, false, nullptr);
    }
    put_back();

    if (rebuilt_runtime <= g_first_zones.runtime) {
        void_arm(r, "load A's stage logged no rebuild of device 0's arena to a RUNTIME zone above the " +
                        num(g_first_zones.runtime) + " bytes of the first arena's (largest " + num(rebuilt_runtime) +
                        " bytes), so A did not size the arena for itself");
        return r;
    }
    const std::string first_line = released.empty() ? std::string("none") : released[0];
    expect(r, !active_after,
           "after load A's refused stage and uncommitted end with no model live device 0's arena is still active, its "
           "RUNTIME zone " +
               num(runtime_after) + " bytes; first last-release line: " + first_line);
    expect(r, emptied == 1,
           "load A logged " + num(emptied) + " lines saying device 0's last release emptied it, want 1; " +
               "first last-release line: " + first_line);
    for (size_t d = 0; d < differing.size(); ++d) {
        expect(r, differing[d] == 0,
               "device " + num(d) + ": after load A's end the published terms of load_published_group mask " +
                   num(differing[d]) + " are not at their start values");
    }
    expect(r, b_begin == GGML_SYCL_LIFECYCLE_OK, "load B's begin after load A returned " + num(b_begin) + ", not OK");
    expect(r, b_active && b_begin_zones == g_first_zones,
           "load B's begin left device 0's arena " +
               (b_active ? "with zones " + zone_capacities_text(b_begin_zones) : std::string("not active")) +
               ", not with the first arena's " + zone_capacities_text(g_first_zones));
    expect(r, b_stage == GGML_SYCL_LIFECYCLE_OK, "load B's early pass returned " + num(b_stage) + ", not OK");
    expect(r, b_staged && b_zones == g_first_load_zones,
           "load B's early pass left device 0's arena " +
               (b_staged ? "with zones " + zone_capacities_text(b_zones) : std::string("not active")) +
               ", not with the first load's " + zone_capacities_text(g_first_load_zones));
    return r;
}

// (k) A load's start runs the release a failed load's end deferred, for the caches that end marked and no other. Load
// L1 is arm (j)'s grown inventory, whose early pass rebuilds device 0's arena above the first arena's RUNTIME zone; a
// sentinel goes into every planner global and every other device-wide term after L1's begin, and a RUNTIME-zone page
// taken in device 0's arena before L1's uncommitted end holds that arena through the end. The end cannot tell the page
// from a failed load's buffers: device 0 keeps its arena and the terms the end put back (the sentinels), and its cache
// is marked; every other device the release walks empties.
//   L2's begin, the page held: one line names the start on device 0, the WARN that a load is starting on an arena a
//              failed load left held and that a holder still there after the failed model's free is a leak, and one
//              line names the holders; the mark stays, and device 0 keeps L1's zones, the page and its terms. L2 ends
//              uncommitted, which defers device 0's release again.
//   L3's begin, the page dropped: one line names the start on device 0, the INFO that the arena is released; the mark
//              is clear, the begin's reserve built the first arena's zones (g_first_zones), and device 0's published
//              terms are at their start values, but for the zone plan, which that reserve stores for the arena it
//              built. Device 1, which carried no mark, keeps a scatter sentinel written before L3's begin. The start's
//              release walks only the devices the scheduler sees, so that leg needs device 1 among them, with a cache
//              of its own. A two-card host that shows the scheduler only device 0 still makes device 1's cache, with
//              its early arena, at a load's begin, but no scope of the start's release reaches it. Without such a
//              device 1 the leg does not run and a WARN line says why. L3 ends uncommitted, which releases every arena
//              the release walks.
// Engagement: the first arena's zones were read before any arm; L1's begin returned OK and the sentinels read back;
// after L1's end the page lies in device 0's RUNTIME zone, that arena is active with a RUNTIME zone above the first
// arena's, device 0 is marked and reads terms other than the start values, and no other device the release walks is.
// Mutants: the start's release dropped (L3 logs no line and finds the mark, L1's zones and the sentinels); the start
// asking every cache (device 1's scatter goes back to its start value); the start's held arena logged as the abort's
// or as a model release's (no start WARN, or a second line naming the start), or without its holders; a start that
// reaps the page or tears the arena down (L2's zones or page move); a start that puts back the terms of a cache it did
// not empty (device 0's terms move at L2's begin); the mark cleared by a start that kept the arena, or kept by one that
// emptied it.
arm_result arm_k_load_start_runs_the_deferred_release() {
    arm_result      r;
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr || !g_first_zones_read) {
        void_arm(r, "device 0's first arena was not read before the arms, so the start's reserve has nothing to equal");
        return r;
    }
    ggml_sycl_load_txn               l1{};
    const ggml_sycl_lifecycle_result l1_begin = ggml_backend_sycl_model_load_begin(&l1);
    if (l1_begin != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "load L1's transaction did not begin (" + num(l1_begin) + ")");
        return r;
    }
    const std::vector<device_planner_state> original        = read_planner_state();
    const std::vector<device_planner_state> sentinel        = sentinel_planner_state(original.size());
    const std::vector<device_load_extras>   original_extras = read_load_extras();
    const std::vector<device_load_extras>   sentinel_extras = sentinel_load_extras(original_extras.size());
    // The page's handle is its one release point: dropped before L3, or by put_back on a VOID, whose settle then runs
    // the release the page deferred.
    std::vector<mem_handle>                 page;
    const auto                              put_back = [&]() {
        page.clear();
        settle_deferred_releases();
        write_planner_state(original);
        write_load_extras(original_extras);
    };
    write_planner_state(sentinel);
    write_load_extras(sentinel_extras);
    if (!(read_planner_state() == sentinel) || !(read_load_extras() == sentinel_extras)) {
        (void) ggml_backend_sycl_model_load_end(l1, false, nullptr);
        put_back();
        void_arm(r, "the sentinel planner state or load terms did not read back, so a put-back to them would not show");
        return r;
    }

    mapped_qwen38_inventory grown;
    grown.inv.pp_moe_onednn_weight_slot_bytes = g_first_zones.runtime + 256 * MiB;
    grown.inv.pp_moe_onednn_scratch_bytes     = grown.inv.pp_moe_onednn_weight_slot_bytes;
    grown.inv.pp_moe_onednn_ring_depth        = 1;
    const ggml_sycl_lifecycle_result l1_stage = ggml_backend_sycl_stage_load_inventory_plan(&grown.inv, nullptr, true);
    if (allocation_result allocation = allocate_runtime_page(cache, 0)) {
        page.push_back(mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS));
    }
    const bool held = !page.empty() && holds_the_runtime_zone(page.back());
    (void) ggml_backend_sycl_model_load_end(l1, false, nullptr);

    // The groups of device `d`'s published terms that are not at their start values, the zone plan left out: a start
    // reserve stores the plan of the arena it builds.
    const auto off_start = [](int d) {
        return load_published_terms::snapshot(d).differing(load_published_terms{}) & ~LOAD_PUBLISHED_GROUP_ZONE_PLAN;
    };
    const bool                  marked      = cache->last_release_deferred();
    const bool                  l1_active   = cache->arena_active();
    const arena_zone_capacities grown_zones = l1_active ? read_zone_capacities(cache) : arena_zone_capacities{};
    const load_published_terms  l1_left     = load_published_terms::snapshot(0);
    const uint32_t              l1_terms    = off_start(0);
    std::string                 other_marked;
    for (int d = 1; d < ggml_backend_sycl_get_device_count() && d < GGML_SYCL_MAX_DEVICES; ++d) {
        const unified_cache * c = get_unified_cache_for_device(d);
        if (c != nullptr && c != cache && c->last_release_deferred()) {
            other_marked += " " + num(d);
        }
    }
    if (!held || !l1_active || grown_zones.runtime <= g_first_zones.runtime || !marked || l1_terms == 0 ||
        !other_marked.empty()) {
        put_back();
        void_arm(r, "after load L1 (stage " + num(l1_stage) + ") the page is " + (held ? "" : "not ") +
                        "in device 0's RUNTIME zone, its arena " +
                        (l1_active ? "has zones " + zone_capacities_text(grown_zones) : std::string("is not active")) +
                        " against the first arena's RUNTIME " + num(g_first_zones.runtime) + ", its cache is " +
                        (marked ? "" : "not ") + "marked, its terms differ from the start values in groups " +
                        num(l1_terms) + (other_marked.empty() ? "" : ", and devices" + other_marked + " are marked"));
        return r;
    }

    // The start's site is in every line its release logs: the start's own lines, and the abort's or a model release's
    // when the start takes their path.
    const char * const named_start = "(load-start)";
    const char * const start_line  = "[VRAM-ARENA] last release on device 0 (load-start)";
    log_clear();
    ggml_sycl_load_txn               l2{};
    const ggml_sycl_lifecycle_result l2_begin = ggml_backend_sycl_model_load_begin(&l2);
    const size_t                     l2_named = lines_with(named_start).size();
    const size_t l2_warned  = lines_with(start_line, "a load is starting on an arena a failed load left held").size();
    const size_t l2_leak    = lines_with(start_line, "after that failed model was freed is a leak").size();
    const size_t l2_holders = lines_with("[VRAM-ARENA] live in the arena on device 0 (MB): ").size();
    const bool   l2_marked  = cache->last_release_deferred();
    const bool   l2_active  = cache->arena_active();
    const arena_zone_capacities l2_zones = l2_active ? read_zone_capacities(cache) : arena_zone_capacities{};
    const bool                  l2_held  = holds_the_runtime_zone(page.back());
    const uint32_t              l2_moved = load_published_terms::snapshot(0).differing(l1_left);
    if (l2_begin == GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(l2, false, nullptr);
    }

    // Device 1, when it has a cache of its own, carries no mark: a start that asked it would put its terms back. The
    // start's release walks the devices the scheduler sees (ggml_backend_sycl_get_device_count()), so a scope can reach
    // device 1 only when the scheduler sees it.
    const int  scheduled = ggml_backend_sycl_get_device_count();
    const bool other     = scheduled > 1 && sentinel.size() > 1 && get_unified_cache_for_device(1) != cache;
    if (!other) {
        // The other legs stand without device 1, so the arm is not VOID; the line says the start's scope went
        // unscored, so a start that asks every cache would still pass.
        std::string why = "device 1 shares device 0's cache";
        if (sentinel.size() < 2) {
            why = "there is no device 1";
        } else if (scheduled < 2) {
            why = "the start's release walks only the devices the scheduler sees (" + num(scheduled) + " of " +
                  num(sentinel.size()) + " GPUs), so no scope reaches device 1's cache";
        }
        std::printf("WARN (k) the leg that a load's start asks only the marked caches did not run: %s\n", why.c_str());
        std::fflush(stdout);
    }
    const size_t scatter = other ? sentinel[1].scatter : 0;
    page.clear();
    if (other) {
        unified_cache_set_planned_moe_host_scatter_scratch_bytes(1, scatter);
    }
    log_clear();
    ggml_sycl_load_txn               l3{};
    const ggml_sycl_lifecycle_result l3_begin   = ggml_backend_sycl_model_load_begin(&l3);
    const size_t                     l3_named   = lines_with(named_start).size();
    const size_t                     l3_emptied = lines_with(start_line, "no model is live; the arena").size();
    const bool                       l3_marked  = cache->last_release_deferred();
    const bool                       l3_active  = cache->arena_active();
    const arena_zone_capacities      l3_zones   = l3_active ? read_zone_capacities(cache) : arena_zone_capacities{};
    const uint32_t                   l3_terms   = off_start(0);
    const size_t l3_scatter = other ? unified_cache_get_planned_moe_host_scatter_scratch_bytes(1) : 0;
    if (l3_begin == GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(l3, false, nullptr);
    }
    put_back();

    expect(r, l2_begin == GGML_SYCL_LIFECYCLE_OK, "load L2's begin returned " + num(l2_begin) + ", not OK");
    expect(r, l2_warned == 1 && l2_leak == 1 && l2_named == 1,
           "load L2's begin, the page held, logged " + num(l2_warned) +
               " lines saying a load is starting on device 0's arena a failed load left held, " + num(l2_leak) +
               " saying a holder after the failed model's free is a leak, and " + num(l2_named) +
               " lines naming the start; want 1, 1 and 1");
    expect(r, l2_holders == 1, "load L2's begin named device 0's holders on " + num(l2_holders) + " lines, want 1");
    expect(r, l2_marked, "load L2's begin, which kept device 0's arena, cleared its mark");
    expect(r, l2_active && l2_zones == grown_zones && l2_held,
           "load L2's begin left device 0's arena " +
               (l2_active ? "with zones " + zone_capacities_text(l2_zones) : std::string("not active")) +
               (l2_held ? "" : " and the page out of its RUNTIME zone") + ", not as L1 left it, with zones " +
               zone_capacities_text(grown_zones));
    expect(r, l2_moved == 0,
           "load L2's begin moved device 0's published terms of load_published_group mask " + num(l2_moved) +
               ", which a cache the start did not empty keeps");
    expect(r, l3_begin == GGML_SYCL_LIFECYCLE_OK, "load L3's begin returned " + num(l3_begin) + ", not OK");
    expect(r, l3_emptied == 1 && l3_named == 1,
           "load L3's begin, the page dropped, logged " + num(l3_emptied) +
               " lines saying device 0's arena is released, and " + num(l3_named) +
               " lines naming the start; want 1 and 1");
    expect(r, !l3_marked, "load L3's begin, which emptied device 0's cache, left its mark");
    expect(r, l3_active && l3_zones == g_first_zones,
           "load L3's begin left device 0's arena " +
               (l3_active ? "with zones " + zone_capacities_text(l3_zones) : std::string("not active")) +
               ", not with the first arena's " + zone_capacities_text(g_first_zones));
    expect(r, l3_terms == 0,
           "after load L3's begin device 0's published terms of load_published_group mask " + num(l3_terms) +
               " are not at their start values");
    if (other) {
        expect(r, l3_scatter == scatter,
               "load L3's begin put device 1's terms back (scatter " + num(l3_scatter) + ", want " + num(scatter) +
                   "), though its cache carried no mark");
    }
    return r;
}

// (l) A plan whose loader pool cannot fit its host budget refuses the load by name, and one whose pool fits is
// admitted: the stage decides on the plan's own figures. The Qwen3.8-shaped inventory as the loader fills it for a load
// that maps no file, with evictable weights (the default): the loader holds the host weights in its SYCL_Host buffer,
// so the plan is charged the loader term, and the early pass refuses the load when the plan's commit total exceeds the
// budget (placement_plan_loader_pool_refusal). The arm takes its expectation from the plan line the planner prints for
// the plan the stage judges, the last device's ([HOST-LEDGER] device=N plan host_budget=...): the commit total is the
// sum of the line's weight, kv, staging, scratch, runtime, rounding and loader_grown, and the load is refused exactly
// when the line's loader_zone is not 0 (the plan was charged the term) and that total exceeds its host_budget. So it
// scores either outcome on any host, whatever its MemAvailable:
//   refused:  the stage returns PLAN_REJECTED after one ERROR naming loader-pool-over-budget from the early pass, whose
//             total, host_budget, loader_zone and loader_grown are the plan line's;
//   admitted: the stage returns OK and logs no such ERROR.
// A budget the arm fixed would make one leg certain, but neither the inventory nor the envelope carries one; the one
// input that caps it, GGML_SYCL_HOST_RESERVE_MB, is the process's environment, which arm (d) needs unset. The load ends
// uncommitted. With no model live a refused load's abort releases every arena nothing holds, as arm (k)'s last end
// leaves them, so the arm runs after (k) and before (g).
// Engagement: the load began, and its stage logged a plan line carrying every term.
// Mutants: the refusal dropped (a charged plan over its budget stages OK); the refusal taken for a plan not charged the
// term or within its budget (PLAN_REJECTED where the line admits); the ERROR printed without its figures, or from the
// other pass; the refusal judged on a figure that differs here from the commit total. Not killed here: one judged on
// host_planned_bytes or on a total without loader_grown, which equal it (no lazy table, no buffer above a chunk).
// Not seen: an early stage creates no tensor, so the admitted leg cannot show the loader's buffers then fitting the
// pool without a WEIGHT-zone grow; that is sycl-loader-in-place-terms' replay, and a full load's.
arm_result arm_l_loader_pool_over_budget_refuses_the_load() {
    arm_result       r;
    loader_inventory inv(qwen38_shape);  // maps no file, so the loader's buffer holds the host weights

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc      = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, nullptr, true);
    const std::vector<std::string>   plans   = lines_with("[HOST-LEDGER] device=", " plan host_budget=");
    const std::vector<std::string>   refused = lines_with("[HOST-PLAN] load refused: loader-pool-over-budget ");
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (plans.empty()) {
        void_arm(r, "the stage returned " + num(rc) + " and logged no plan line, so there is no plan to judge");
        return r;
    }
    // Each device's early plan replaces the one before it as the load's candidate, which the stage judges.
    const std::string & plan    = plans.back();
    const char * const  terms[] = { "weight", "kv", "staging", "scratch", "runtime", "rounding", "loader_grown" };
    size_t              total   = 0;
    for (const char * term : terms) {
        const size_t bytes = printed(plan, term);
        if (bytes == SIZE_MAX) {
            void_arm(r, std::string("the plan line carries no ") + term + ": " + plan);
            return r;
        }
        total += bytes;
    }
    const size_t budget       = printed(plan, "host_budget");
    const size_t loader_zone  = printed(plan, "loader_zone");
    const size_t loader_grown = printed(plan, "loader_grown");
    if (budget == SIZE_MAX || loader_zone == SIZE_MAX) {
        void_arm(r, "the plan line carries no host_budget or loader_zone: " + plan);
        return r;
    }
    const std::string figures = "commit total " + num(total) + ", host_budget " + num(budget) + ", loader_zone " +
                                num(loader_zone) + ", loader_grown " + num(loader_grown);
    if (loader_zone > 0 && total > budget) {
        expect(r, rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
               "a plan charged the loader term over its budget (" + figures + ") staged " + num(rc) +
                   ", not PLAN_REJECTED");
        expect(r, refused.size() == 1,
               "the over-budget plan (" + figures + ") logged " + num(refused.size()) +
                   " loader-pool-over-budget refusals, want 1");
        if (refused.size() == 1) {
            const std::string & line = refused[0];
            expect(r,
                   printed(line, "total") == total && printed(line, "host_budget") == budget &&
                       printed(line, "loader_zone") == loader_zone && printed(line, "loader_grown") == loader_grown,
                   "the refusal's figures are not the plan line's (" + figures + "): " + line);
            expect(r, line.find("(early=1)") != std::string::npos, "the refusal is not the early pass's: " + line);
        }
    } else {
        expect(r, rc == GGML_SYCL_LIFECYCLE_OK,
               "a plan whose loader pool fits its budget, or that is not charged the term (" + figures + "), staged " +
                   num(rc) + ", not OK");
        expect(r, refused.empty(),
               "a plan within its budget (" + figures +
                   ") logged a loader-pool-over-budget refusal: " + (refused.empty() ? std::string() : refused[0]));
    }
    return r;
}

// The q2k fixture's shape with 256 experts: each expert tensor is 32 times larger, so the plan's host staging and
// SCRATCH zones, which follow the largest copied tensor, are larger than the q2k shape's.
constexpr host_plan_shaped_inventory::moe_shape q2k_wide_shape = {
    GGML_TYPE_Q2_K, 2, 256, 2, 1024, 512, 4096, 16, 1, 0, 0, 0,
};

// (g) A load refused in its late pass configures no host zone, and the next load's host zones come from its own plan.
// The host zones are configured once per cache, by the first late pass that reaches the step, so a refused load that
// configured them would size them for every load after it. Load A (the q2k shape) stages its early pass, then its late
// pass with the PRIVATE_TESTING hook that fails the stage after its first device, which on the stage's decide step is
// before any device's effects. Load B (the same shape with 256 experts) stages both passes. Each ends uncommitted.
// A's refusal is checked on every device: the hook fires after the first device, so a loop that ran each device's
// effects before the next device's decision configures the first device, and on two cards no other. B's sizes are a
// last-device check: each device's late stage replaces the load's candidate with its own snapshot, the snapshots of the
// devices before it live only in the stage's own list, and a cache's plan is published only when its load commits, so
// before B ends the last device's plan is the one plan a test can read. The devices before it are checked configured,
// not sized; on one card the last device is device 0.
// Engagement: no device's cache had host zones before the arm; the hook is exported, and the armed late stage
// returned EFFECT_FAILED; B's late pass staged a plan with entries; A's and B's plans size at least one zone apart.
// Mutant: the host zones configured inside the device loop, before a later device's refusal (A's late pass configures
// the first device's zones; on one card B's zones are then A's sizes).
arm_result arm_g_refused_late_pass_configures_no_host_zones() {
    arm_result                   r;
    std::vector<unified_cache *> caches;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        unified_cache * c = get_unified_cache_for_device(d);
        if (c == nullptr || c->pinned_pool_chunk_size() == 0) {
            void_arm(r, "device " + num(d) + " has no cache with a pinned pool");
            return r;
        }
        if (c->host_zones_configured()) {
            void_arm(r, "device " + num(d) + "'s host zones were configured before the arm, so a refused load's " +
                            "zones cannot be told from an earlier one's");
            return r;
        }
        caches.push_back(c);
    }
    if (caches.empty()) {
        void_arm(r, "no SYCL device");
        return r;
    }
    const int       device = static_cast<int>(caches.size()) - 1;
    unified_cache * cache  = caches.back();

    // The devices whose host zones are configured (`configured`) or not, as a list for a message.
    const auto devices_where = [&](bool configured) {
        std::string list;
        for (size_t d = 0; d < caches.size(); ++d) {
            if (caches[d]->host_zones_configured() == configured) {
                list += " " + num(d);
            }
        }
        return list;
    };
    using fail_next_fn           = void (*)();
    const fail_next_fn fail_late = reinterpret_cast<fail_next_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_sycl_reg(), "ggml_backend_sycl_test_fail_next_stage_inventory_plan_late_after_first_device"));
    if (fail_late == nullptr) {
        void_arm(r, "the stage's late failure hook is not exported");
        return r;
    }
    const auto terms_of = [&](const placement_plan & plan) {
        return placement_plan_host_zone_commit_terms(
            plan, cache->onednn_weights_scratch_size() + cache->onednn_activations_scratch_size(),
            cache->pinned_pool_chunk_size());
    };

    loader_inventory   a_inv(host_plan_shaped_inventory::q2k_shape);
    ggml_sycl_load_txn a{};
    if (ggml_backend_sycl_model_load_begin(&a) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "load A's transaction did not begin");
        return r;
    }
    if (ggml_backend_sycl_stage_load_inventory_plan(&a_inv.inv, nullptr, true) != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(a, false, nullptr);
        void_arm(r, "load A's early pass did not stage");
        return r;
    }
    const auto a_early = lifecycle_find_candidate_placement_plan(a.id);
    if (!a_early || !a_early->plan || a_early->plan->entries.empty()) {
        (void) ggml_backend_sycl_model_load_end(a, false, nullptr);
        void_arm(r, "load A's early pass staged no plan with entries");
        return r;
    }
    const host_zone_commit_terms a_terms = terms_of(*a_early->plan);
    fail_late();
    const ggml_sycl_lifecycle_result a_rc = ggml_backend_sycl_stage_load_inventory_plan(&a_inv.inv, nullptr, false);
    const std::string                a_configured = devices_where(true);
    (void) ggml_backend_sycl_model_load_end(a, false, nullptr);
    if (a_rc != GGML_SYCL_LIFECYCLE_EFFECT_FAILED) {
        // The hook fires only when the stage reaches it, so it may still be armed: no further stage runs here.
        void_arm(r, "load A's armed late pass returned " + num(a_rc) + ", not EFFECT_FAILED");
        return r;
    }
    expect(r, a_configured.empty(),
           "load A's late pass, refused on its decide step, configured the host zones of device(s)" + a_configured);

    loader_inventory   b_inv(q2k_wide_shape);
    ggml_sycl_load_txn b{};
    if (ggml_backend_sycl_model_load_begin(&b) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "load B's transaction did not begin");
        return r;
    }
    const ggml_sycl_lifecycle_result b_early_rc =
        ggml_backend_sycl_stage_load_inventory_plan(&b_inv.inv, nullptr, true);
    const ggml_sycl_lifecycle_result b_late_rc =
        b_early_rc == GGML_SYCL_LIFECYCLE_OK ? ggml_backend_sycl_stage_load_inventory_plan(&b_inv.inv, nullptr, false) :
                                               b_early_rc;
    const auto                   b_late         = lifecycle_find_candidate_placement_plan(b.id);
    const std::string            b_unconfigured = devices_where(false);
    const size_t                 have_weight    = cache->host_zone_capacity(host_zone_id::WEIGHT);
    const size_t                 have_kv        = cache->host_zone_capacity(host_zone_id::KV);
    const size_t                 have_staging   = cache->host_zone_capacity(host_zone_id::STAGING);
    const size_t                 have_scratch   = cache->host_zone_capacity(host_zone_id::SCRATCH);
    const host_zone_commit_terms b_terms = b_late && b_late->plan ? terms_of(*b_late->plan) : host_zone_commit_terms{};
    (void) ggml_backend_sycl_model_load_end(b, false, nullptr);
    if (b_late_rc != GGML_SYCL_LIFECYCLE_OK || !b_late || !b_late->plan || b_late->plan->entries.empty()) {
        void_arm(r, "load B's passes returned " + num(b_early_rc) + " and " + num(b_late_rc) +
                        " without a staged plan with entries");
        return r;
    }
    if (a_terms.weight == b_terms.weight && a_terms.kv == b_terms.kv && a_terms.staging == b_terms.staging &&
        a_terms.scratch == b_terms.scratch) {
        void_arm(r, "load A's and load B's plans size every host zone alike, so the zones cannot say whose plan " +
                        std::string("configured them"));
        return r;
    }
    const size_t align    = pinned_chunk_pool::DEFAULT_ALIGNMENT;
    const auto   align_up = [](size_t bytes) {
        return (bytes + align - 1) & ~(align - 1);
    };
    expect(r, b_unconfigured.empty(), "load B's late pass configured no host zones on device(s)" + b_unconfigured);
    const auto zone = [&](const char * name, size_t have, size_t b_term, size_t a_term) {
        expect(r, have == align_up(b_term),
               std::string("the last device's (device ") + num(device) + ") " + name + " zone holds " + num(have) +
                   " bytes, not load B's " + num(b_term) + " (load A's plan sized it " + num(a_term) + ")");
    };
    zone("WEIGHT", have_weight, b_terms.weight, a_terms.weight);
    zone("KV", have_kv, b_terms.kv, a_terms.kv);
    zone("STAGING", have_staging, b_terms.staging, a_terms.staging);
    zone("SCRATCH", have_scratch, b_terms.scratch, a_terms.scratch);
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

    // The backends stay alive across the arms, so every arm plans against the same caches. An arena lives until a load
    // ends with nothing holding it, so an arm that reads one across a load's end holds it (hold_arenas).
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b0(ggml_backend_sycl_init(0), ggml_backend_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b1(
        ggml_backend_sycl_get_device_count() > 1 ? ggml_backend_sycl_init(1) : nullptr, ggml_backend_free);
    if (!b0) {
        std::printf("FAIL: backend 0 did not initialise\n");
        return 1;
    }
    if (unified_cache * cache = get_unified_cache_for_device(0); cache != nullptr && cache->arena_active()) {
        g_first_zones      = read_zone_capacities(cache);
        g_first_zones_read = true;
    }
    read_first_load_zones();

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "(0) a caller of another layout is refused",           arm_0_another_layout_is_refused                  },
        { "(a) expected_host is the stage's host_planned_bytes", arm_a_expected_host_is_the_stage_figure          },
        { "(b) every term printed by name, from the plan",       arm_b_every_term_printed_from_the_plan           },
        { "(c) a dry run stores nothing and leaves no trace",    arm_c_a_dry_run_stores_nothing                   },
        { "(d) --mem-available is the sample's MemAvailable",    arm_d_mem_available_option_is_the_sample         },
        { "(e) a load the held arena cannot hold is refused",    arm_e_held_arena_refuses_the_load                },
        { "(f) a load refused early keeps the KV-zone ring",     arm_f_early_refusal_keeps_the_kv_zone_ring       },
        { "(h) a load releases a ring alone holding the arena",  arm_h_load_releases_a_lone_ring                  },
        { "(i) a Graph floor keeps an allocation-held arena",    arm_i_graph_floor_keeps_an_allocation_held_arena },
        { "(j) a refused load leaves nothing sized for it",      arm_j_refused_load_leaves_nothing_sized_for_it   },
        { "(k) a load's start runs a deferred release",          arm_k_load_start_runs_the_deferred_release       },
        { "(l) a loader pool over its budget refuses the load",  arm_l_loader_pool_over_budget_refuses_the_load   },
        { "(g) a refused late pass configures no host zone",     arm_g_refused_late_pass_configures_no_host_zones },
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
