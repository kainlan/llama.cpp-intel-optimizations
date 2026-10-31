"""Source contract for explicit micro-batch placement: an explicit -ub whose PP MoE oneDNN scratch ring does not fit
the RUNTIME zone planned for the load's own micro-batch is placed, not refused.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8, `llama-bench -ub 2048 -p 2048 -n 0 -fa 1`): the load planned the ring for n_ubatch 512, the
context asked for 2048, the ring re-plan could not fit the four times larger activation and output slots, and the
context was refused with "[SYCL-PLAN] runtime context update rejected: PP MoE oneDNN scratch ring for n_ubatch=2048
needs ...", so the arm produced no row. KV that does not fit is re-placed to the host tier rather than refused; the ring
now follows the same rule.

What this gate pins:
- the transport: the micro-batch the caller names reaches the planner (llama_model_params::n_ubatch_hint, then
  ggml_sycl_tensor_inventory::n_ubatch_context, then placement_kv_info::n_ubatch_context), and both callers name none
  (0) when the backend picks the micro-batch itself, so the auto-ubatch trial never places;
- the planner holds the ring's RUNTIME room as a named term on the routed-expert budget, inside the MoE block after the
  dense weights, their KV and the KV context room, so only routed experts give it up; the hold is the whole room before
  and after the zone grows, the pack predicate is the budget less it, and what it moved is net, as for the KV context
  room: the device bytes a first-fit pack with the hold added back places beyond the real pack's, from the layers that
  keep fewer triplets on the device;
- the load grows the RUNTIME zone for that micro-batch before the plan it keeps, and falls back to the load's planning
  shape, planned without the room, when the arena cannot grow;
- before its first plan, a load whose zone already holds the room (grown by an earlier stage of the load or an earlier
  load in the process) re-publishes the micro-batch's ring and dense scratch, asking the planner's own decision with no
  expert budget, and publishes the planning shape again when that publish fails;
- a load announces the placement in ONE WARN line, with the bytes and layers moved and both packs' triplets, at the
  first non-probe context since the load whose room cost device bytes, whatever that context's micro-batch; the plan
  that context publishes carries a mark every later context's plan copies, so no later context repeats it;
- the fallback refusal names the largest -ub that fits, from the ring admission's own slot arithmetic;
- a load that placed its explicit micro-batch keeps the ring at that size for every context whose own micro-batch is no
  larger: the transaction re-plans the ring for placement_explicit_ubatch_ring_n_ubatch()'s answer, so the re-plan's
  idempotent return keeps the placed ring instead of releasing it for a smaller one the next context must re-admit;
- the hold-spill fit credits a compute buffer no larger than the planned compute term at the micro-batch the published
  single-device plan PLACED on the device, read once before the demand, whole or nothing; every other rung, and every
  load that placed none, is judged as before;
- a load whose plan was packed against a RUNTIME zone an earlier load in the process grew says so in ONE WARN after
  its late plan, only when the live zone is larger than the zone the arena reserves for that plan, so the first load
  of a process never prints it;
- the RUNTIME zone is one reservation: the arena's reserve, the zone the cache plans, the explicit micro-batch's room,
  the check for a grown zone and its WARN all ask zone_runtime_reserved_bytes() (the larger of the floor and the
  requirement, rounded up to the arena's 256-byte grain, checked), so none of them can round apart from the others;
  the cache asserts that grain is the zone allocator's block grain, so the two spellings cannot drift.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re

from sycl_gate import body, norm, ordered, read

SYCL_CPP = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
CACHE_CPP = read("ggml/src/ggml-sycl/unified-cache.cpp")
CACHE_HPP = read("ggml/src/ggml-sycl/unified-cache.hpp")
ADMISSION_CPP = read("ggml/src/ggml-sycl/moe-scratch-admission.cpp")
ZONE_CPP = read("ggml/src/ggml-sycl/zone-sizing.cpp")
ZONE_HPP = read("ggml/src/ggml-sycl/zone-sizing.hpp")
LLAMA_H = read("include/llama.h")
SYCL_H = read("ggml/include/ggml-sycl.h")
MODEL_CPP = read("src/llama-model.cpp")
COMMON_CPP = read("common/common.cpp")
BENCH_CPP = read("tools/llama-bench/llama-bench.cpp")


def struct_text(text: str, head: str) -> str:
    """The text of the first struct starting with `head`, up to its closing `};` (a struct of plain fields, which holds
    no nested braces). Empty when there is none."""
    at = text.find(head)
    if at < 0:
        return ""
    end = text.find("};", at)
    return text[at:end] if end >= 0 else ""


TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
STORE_SIG = "static void compute_and_store_plan_for_inventory("
PLANNER_SIG = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
DECIDE_SIG = "pp_moe_onednn_explicit_ubatch_decision pp_moe_onednn_explicit_ubatch_decide("
POPULATE_SIG = "static void llama_model_sycl_populate_inventory("
BENCH_HINT_SIG = "uint32_t n_ubatch_hint() const"

# The decision line: one WARN per load, at the first non-probe context since the load when the room cost device bytes.
WARN_LITERAL = ('GGML_LOG_WARN("[SYCL-PLAN] PP MoE oneDNN scratch ring for n_ubatch=%u needs %.1f MB: it re-placed '
                '%.1f MB of routed experts to the host tier, from %zu layers: %zu triplets on the device, against %zu '
                'without its room, on device %d\\n", xub.n_ubatch, xub.ring_bytes / mb, xub.moved_bytes() / mb, '
                'xub.moved_layers, xub.expert_groups, xub.expert_groups_without, ctx->device);')
WARN_GATE = "if (warn) {"
INFO_GATE = "} else if (xub.n_ubatch == next_kv_info.n_ubatch) {"
ANNOUNCE_DEF = "auto announce_explicit_ubatch_placement = [&](const ggml_sycl::placement_plan & final_plan, bool warn)"
ANNOUNCE_FOR = "if (xub.action != ggml_sycl::pp_moe_onednn_explicit_ubatch_action::PLACE) { return; }"
ANNOUNCE_PROBE = ("announce_kv_host_demotions(next_plan); "
                  "announce_explicit_ubatch_placement(next_plan, warn_explicit_ubatch);")
ANNOUNCE_COMMIT = ("announce_kv_host_demotions(*immutable->plan); "
                   "announce_explicit_ubatch_placement(*immutable->plan, warn_explicit_ubatch);")
# The mark: decided once on the final plan, set on next_plan before it is published, and carried by the two whole-plan
# copies a context makes (the load's or previous context's plan, and the KV demotion's working copy).
PLAN_COPY = "auto next_plan = ggml_sycl::placement_plan(*current->plan);"
DEMOTED_COPY = "ggml_sycl::placement_plan demoted_plan = next_plan;"
DEMOTED_BACK = "next_plan = std::move(demoted_plan);"
WARN_DECIDE = ("const bool warn_explicit_ubatch = "
               "ggml_sycl::placement_explicit_ubatch_announces(next_plan.explicit_ubatch, probe_mode);")
WARN_MARK = "if (warn_explicit_ubatch) { next_plan.explicit_ubatch.announced = true; }"
PROBE_BRANCH = "if (probe_mode) { const bool rollback_ok ="
PUBLISH = "next->plan = std::make_shared<const ggml_sycl::placement_plan>(std::move(next_plan));"
ANNOUNCES_SIG = "bool placement_explicit_ubatch_announces(const placement_explicit_ubatch & xub, bool probe_mode)"
ANNOUNCES_RULE = ("return !probe_mode && xub.action == pp_moe_onednn_explicit_ubatch_action::PLACE && "
                  "xub.moved_bytes() > 0 && !xub.announced;")
XUB_STRUCT = "struct placement_explicit_ubatch {"
XUB_MARK_FIELD = "bool announced = false;"

# The fallback refusal, by name, with the largest -ub that fits.
REFUSE_FOR = ("if (xub.action == ggml_sycl::pp_moe_onednn_explicit_ubatch_action::REFUSE && "
              "xub.n_ubatch == next_kv_info.n_ubatch) {")
FITS = 'std::snprintf(fits, sizeof(fits), "; the largest -ub that fits is %u", xub.largest_fitting_n_ubatch);'
REFUSE_LITERAL = ('"[SYCL-PLAN] runtime context update rejected: PP MoE oneDNN scratch ring for n_ubatch=%u needs '
                  '%.1f MB, more than the RUNTIME zone can hold with every device-resident expert re-placed to the '
                  'host tier (%.1f MB)%s (device=%d)\\n", xub.n_ubatch, xub.ring_bytes / mb, '
                  'xub.max_ring_runtime_bytes / mb, fits, ctx->device);')
RING_REPLAN = "ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, ring_n_ubatch, probe_mode, &ring_kv_zone);"
# The ring a context re-plans for: the placed micro-batch's when the plan PLACED it and the context's own is no larger.
RING_N_UBATCH = ("const uint32_t ring_n_ubatch = ggml_sycl::placement_explicit_ubatch_ring_n_ubatch("
                 "next_plan.explicit_ubatch, next_kv_info.n_ubatch);")
RING_HELPER_SIG = ("uint32_t placement_explicit_ubatch_ring_n_ubatch(const placement_explicit_ubatch & xub, "
                   "uint32_t n_ubatch)")
RING_HELPER_RULE = ("const bool kept = xub.action == pp_moe_onednn_explicit_ubatch_action::PLACE && n_ubatch != 0 && "
                    "n_ubatch <= xub.n_ubatch; return kept ? xub.n_ubatch : n_ubatch;")
LARGEST_FIT = "decision.largest_fitting_n_ubatch = pp_moe_onednn_admit_ring(fit).largest_fitting_n_ubatch;"
FIT_INPUTS = (
    "fit.runtime_available_bytes = decision.max_ring_runtime_bytes;",
    "fit.weight_slot_bytes = in.weight_slot_bytes;",
    "fit.activation_bytes_per_row = in.activation_bytes_per_row;",
    "fit.output_bytes_per_row = in.output_bytes_per_row;",
    "fit.ring_depth = in.ring_depth;",
    "fit.n_ubatch = in.n_ubatch;",
)

# The hold-spill fit credits the planned compute term at the explicit micro-batch the published plan placed on the
# device: the room comes from the placement's own helper (PLACE only), each field is assigned once, before the demand.
FIT_SIG = "static bool ggml_sycl_hold_spill_fit(const ggml_sycl_hold_fit_query & q, ggml_sycl_hold_fit_answer * answer)"
FIT_PLAN = "const auto room_plan = ggml_sycl_global_plan_owner();"
FIT_GATE = "if (!room_plan->multi_device && room_plan->device_id == q.device) {"
FIT_ROOM = ("room_bytes = ggml_sycl::placement_explicit_ubatch_compute_room(room_plan->explicit_ubatch, "
            "ggml_sycl::unified_cache_get_planned_compute_term_bytes(q.device), &room_n_ubatch);")
FIT_FIELDS = "in.compute_room_n_ubatch = room_n_ubatch; in.compute_room_bytes = room_bytes;"
FIT_DEMAND = "a.demand = ggml_sycl::zone_hold_fit_demand(in, q.n_ubatch);"
FIT_FITS = "const bool fits = ggml_sycl::zone_hold_fit(in, q.n_ubatch);"
ROOM_HELPER_SIG = "size_t placement_explicit_ubatch_compute_room(const placement_explicit_ubatch & xub,"
ROOM_HELPER_RULE = ("const bool placed = xub.action == pp_moe_onednn_explicit_ubatch_action::PLACE && xub.n_ubatch != 0; "
                    "if (n_ubatch != nullptr) { *n_ubatch = placed ? xub.n_ubatch : 0; } "
                    "return placed ? compute_term : 0;")
CREDIT_SIG = "bool zone_hold_fit_rung_has_compute_room(const zone_hold_fit_inputs & in, uint32_t n_ubatch, size_t request)"
CREDIT_RULE = "return n_ubatch != 0 && n_ubatch == in.compute_room_n_ubatch && request <= in.compute_room_bytes;"
DEMAND_SIG = "size_t zone_hold_fit_demand(const zone_hold_fit_inputs & in, uint32_t n_ubatch)"
DEMAND_CREDIT = ("size_t request = zone_hold_fit_request(in, n_ubatch); "
                 "if (zone_hold_fit_rung_has_compute_room(in, n_ubatch, request)) { request = 0; }")
DEMAND_BOUND = "zone_hold_spill_raw_demand(zone_hold_spill_bound(plan, request, n_ubatch, n_ubatch), in.kv_room);"

# A RUNTIME zone an earlier load grew: one WARN after the late plan, gated by the predicate on the zone the plan sizes.
SET_INV_SIG = ("static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl("
               "ggml_backend_t backend, const ggml_sycl_tensor_inventory * inventory)")
LATE_PLAN = ("compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct); "
             "ggml_sycl_warn_runtime_zone_kept_from_earlier_load(ctx->device);")
KEPT_SIG = "static void ggml_sycl_warn_runtime_zone_kept_from_earlier_load(int device)"
KEPT_READS = ("if (cache == nullptr || !cache->arena_active() || "
              "!ggml_sycl::unified_cache_get_planned_runtime_zone_requirement(device, &requirement)) { return; }")
KEPT_LIVE = "const size_t live = cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME);"
KEPT_FLOOR = "const size_t floor = ggml_sycl::unified_cache_runtime_zone_floor_bytes();"
KEPT_GATE = ("if (!ggml_sycl::zone_runtime_kept_above_plan(live, floor, requirement) || "
             "!ggml_sycl::zone_runtime_reserved_bytes(floor, requirement, &reserved)) { return; }")
KEPT_WARN = ('GGML_LOG_WARN("[SYCL-PLAN] RUNTIME zone on device %d holds %.1f MB where this plan requires %.1f MB: an '
             'earlier load in this process grew it, and the routed experts are planned against the grown zone\\n", '
             'device, live / mb, reserved / mb);')
KEPT_RULE_SIG = "bool zone_runtime_kept_above_plan(size_t live_zone_bytes, size_t floor_bytes, size_t requirement)"
KEPT_RULE = ("size_t reserved = 0; "
             "return zone_runtime_reserved_bytes(floor_bytes, requirement, &reserved) && live_zone_bytes > reserved;")

# The RUNTIME zone as the arena reserves it: one checked helper, asked by every site that sizes, plans against or
# compares with that zone.
ARENA_ALIGNMENT = "constexpr size_t k_zone_arena_alignment = 256;"
# The grain is a second spelling of the zone allocator's block grain, so the cache asserts the two are equal.
ARENA_GRAIN_ASSERT = "static_assert(k_zone_arena_alignment == tlsf_allocator::block_grain,"
RESERVED_SIG = "inline bool zone_runtime_reserved_bytes(size_t floor_bytes, size_t requirement, size_t * out)"
RESERVED_RULE = ("const size_t planned = requirement > floor_bytes ? requirement : floor_bytes; "
                 "if (planned > SIZE_MAX - (k_zone_arena_alignment - 1)) { return false; } "
                 "*out = (planned + k_zone_arena_alignment - 1) & ~(k_zone_arena_alignment - 1); return true;")
ARENA_RESERVE_SIG = "bool unified_cache::arena_reserve(sycl::queue & queue,"
ARENA_RESERVE_CALL = "if (!zone_runtime_reserved_bytes(0, runtime_bytes, &runtime_bytes)) {"
ENSURE_SIG = "bool unified_cache::ensure_planned_arena_zones()"
ENSURE_CALL = ("if (!unified_cache_get_planned_runtime_zone_requirement(dev_id, &planned_runtime_scratch) || "
               "!zone_runtime_reserved_bytes(runtime_floor, planned_runtime_scratch, &runtime_zone)) {")
ROOM_ZONE_SIG = "static bool pp_moe_onednn_runtime_zone_bytes(const pp_moe_onednn_explicit_ubatch_inputs & in,"
ROOM_ZONE_CALL = ("return zone_runtime_reserved_bytes(in.runtime_floor_bytes, "
                  "in.runtime_fixed_bytes + dense_bytes + ring_bytes, out);")

# The planner: a named hold on the expert budget, after the KV context room and before the pack's stats (where a
# second hold term composes); the pack predicate is the budget less the hold; what it moved is the difference between
# the pack with the hold added back and the real one, as the KV context room counts its cost.
KV_ROOM = "room = hold_kv_context_room(plan, kv_info, device_id, n_experts, room_expert_rate, remaining);"
HOLD_CALL = "const size_t ubatch_hold_bytes = hold_explicit_ubatch_room(plan, kv_info, remaining);"
STATS_DECL = "moe_triplet_pack_stats stats;"
WITHOUT_ROOM = "size_t without_ubatch_room = remaining + ubatch_hold_bytes;"
PACK_PREDICATE = "const bool on_device = group.charge_bytes <= remaining;"
MOVED = ("if (ubatch_hold_bytes > 0) { if (group.charge_bytes <= without_ubatch_room) { without_ubatch_room -= "
         "group.charge_bytes; plan.explicit_ubatch.expert_bytes_without += group.bytes; "
         "plan.explicit_ubatch.expert_groups_without++; ubatch_layer_drop[group.layer_id]++; } "
         "if (on_device) { plan.explicit_ubatch.expert_bytes += group.bytes; plan.explicit_ubatch.expert_groups++; "
         "ubatch_layer_drop[group.layer_id]--; } }")
MOVED_LAYERS = ("for (const auto & [layer_id, drop] : ubatch_layer_drop) { GGML_UNUSED(layer_id); "
                "plan.explicit_ubatch.moved_layers += drop > 0; }")
MOVED_NET = ("size_t moved_bytes() const { return expert_bytes_without > expert_bytes ? "
             "expert_bytes_without - expert_bytes : 0; }")
HOLD_SIG = ("static size_t hold_explicit_ubatch_room(placement_plan & plan, const placement_kv_info & kv_info, "
            "size_t & remaining)")
DECIDE = "pp_moe_onednn_explicit_ubatch_decide(room, remaining);"
HOLD_REASON = 'xub.hold_reason = "PP MoE oneDNN scratch ring RUNTIME room";'
HOLD_GATE = ("if (decision.action == pp_moe_onednn_explicit_ubatch_action::PLACE || "
             "decision.action == pp_moe_onednn_explicit_ubatch_action::GROW_ZONE) {")
HOLD_STEPS = ("remaining += room.room_reserved_bytes; xub.hold_bytes = room.room_bytes; "
              "remaining -= xub.hold_bytes;")

# The load: plan with the room, grow the zone for the micro-batch, plan again, and fall back to the planning shape
# planned without the room.
LAMBDA_ROOM = ("if (with_room) { ggml_sycl_fill_explicit_ubatch_room(ctx->device); } else { "
               "g_placement_kv_info.explicit_ubatch = {}; } return ggml_sycl::compute_placement_plan(")
FIRST_PLAN = "plan_candidate = plan_single_device(true);"
GROW_BRANCH = "if (plan_candidate.explicit_ubatch.action == ggml_sycl::pp_moe_onednn_explicit_ubatch_action::GROW_ZONE) {"
GROW_PUBLISH = ("bool placed = ggml_sycl_publish_load_ubatch_scratch(ctx->device, explicit_n_ubatch) && "
                "ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device);")
GROW_REPLAN = ("if (placed) { plan_candidate = plan_single_device(true); placed = "
               "plan_candidate.explicit_ubatch.action == ggml_sycl::pp_moe_onednn_explicit_ubatch_action::PLACE; }")
GROW_FALLBACK = ("if (!placed) { "
                 "(void) ggml_sycl_publish_load_ubatch_scratch(ctx->device, g_placement_kv_info.n_ubatch);")
FALLBACK_PLAN = "plan_candidate = plan_single_device(false);"
# The fallback says so at every verbosity: the user named a micro-batch and the load did not place it.
REVERT_WARN = ('GGML_LOG_WARN("[SYCL-PLAN] PP MoE oneDNN scratch ring for n_ubatch=%u needs %.1f MB more RUNTIME zone, '
               'which the arena could not reserve on device %d; the ring is re-planned when the context opens\\n", '
               'explicit_n_ubatch, room_short / (1024.0 * 1024.0), ctx->device);')

# Before the first plan: a zone that already holds the room gets the micro-batch's ring and dense scratch back, by the
# planner's decision with no expert budget, and a publish that fails part-way gets the planning shape back.
PRE_FILL = ("const uint32_t explicit_n_ubatch = g_placement_kv_info.n_ubatch_context; "
            "ggml_sycl_fill_explicit_ubatch_room(ctx->device);")
PRE_DECIDE = ("ggml_sycl::pp_moe_onednn_explicit_ubatch_decide(g_placement_kv_info.explicit_ubatch, 0).action == "
              "ggml_sycl::pp_moe_onednn_explicit_ubatch_action::PLACE")
PRE_GATE = ("if (" + PRE_DECIDE + " && !ggml_sycl_publish_load_ubatch_scratch(ctx->device, explicit_n_ubatch)) { "
            "(void) ggml_sycl_publish_load_ubatch_scratch(ctx->device, g_placement_kv_info.n_ubatch); }")

# The transport.
PARAM_FIELD = "uint32_t n_ctx_hint; uint32_t n_ubatch_hint; };"
# The default is the last positional initializer, labelled by a comment, so it is matched on the raw text.
MODEL_DEFAULT_RE = re.compile(r"/\*\.n_ctx_hint\s*=\*/\s*0,\s*/\*\.n_ubatch_hint\s*=\*/\s*0,\s*\};")
INVENTORY_SET = "inventory.n_ubatch_context = n_ubatch_hint;"
CALLER_PASSES = "hparams, model.get_n_ctx_hint(), model.get_n_ubatch_hint());"
BACKEND_COPY = "g_placement_kv_info.n_ubatch_context = inventory->n_ubatch_context;"
OFFSET_PIN = "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ubatch_context) == 188,"
# The header's own pins, compiled by both sides of the dlopen boundary (libllama fills the struct, libggml-sycl reads it).
HEADER_SIZE_PIN = ('static_assert(sizeof(ggml_sycl_tensor_inventory) == 192, '
                   '"ggml_sycl_tensor_inventory layout changed");')
HEADER_OFFSET_PIN = ('static_assert(offsetof(ggml_sycl_tensor_inventory, n_ubatch_context) == 188, '
                     '"n_ubatch_context moved");')
COMMON_NONE = "mparams.n_ubatch_hint = 0; if (!params.n_ubatch_auto && params.n_batch > 0) {"
BENCH_NONE = 'if (n_ubatch < 0 || n_batch <= 0 || ggml_backend_reg_by_name("SYCL") == nullptr) { return 0; }'
# Read from -b and -ub alone, so the prompt and generation tests of one pair name the same micro-batch.
BENCH_RETURN = "return n_ubatch > 0 ? std::min((uint32_t) n_batch, (uint32_t) n_ubatch) : (uint32_t) n_batch;"
BENCH_SET = "mparams.n_ubatch_hint = n_ubatch_hint();"
BENCH_EQUAL = "n_ubatch_hint() == other.n_ubatch_hint() &&"


def claim_placement_is_one_warn_line(sycl_cpp: str, cache_cpp: str, cache_hpp: str) -> bool:
    """A load announces the placement once, at WARN, with the experts, layers and bytes the room moved: at the first
    non-probe context since the load whose plan PLACED the micro-batch and moved experts, whatever that context's own
    micro-batch (nothing before the WARN reads it; only the INFO line is filtered by it). The decision is made once on
    the final plan, before the probe branch, and the context that announces marks next_plan before it is published.
    The mark rides the two whole-plan copies a context makes, and nothing else in the transaction writes it, so every
    later context's plan carries it. Both announce sites pass the decision, each right after the KV demotions' line."""
    txn = body(norm(sycl_cpp), TXN_SIG)
    announce = body(txn, ANNOUNCE_DEF)
    before_warn = announce.split(WARN_GATE, 1)[0] if WARN_GATE in announce else ""
    return (bool(announce) and ordered(announce, ANNOUNCE_FOR, WARN_GATE, WARN_LITERAL, INFO_GATE)
            and announce.count("GGML_LOG_WARN(") == 1
            and "next_kv_info" not in before_warn and "current->" not in before_warn
            and txn.count(PLAN_COPY) == 1 and txn.count(DEMOTED_COPY) == 1 and txn.count(DEMOTED_BACK) == 1
            and ordered(txn, PLAN_COPY, DEMOTED_COPY, DEMOTED_BACK, WARN_DECIDE, WARN_MARK, PROBE_BRANCH, PUBLISH)
            and txn.count(WARN_DECIDE) == 1 and txn.count(".announced") == 1
            and re.search(r"\.explicit_ubatch\s*=[^=]", txn) is None
            and txn.count(ANNOUNCE_PROBE) == 1 and txn.count(ANNOUNCE_COMMIT) == 1
            and txn.count("announce_explicit_ubatch_placement(") == 2
            and ANNOUNCES_RULE in body(norm(cache_cpp), ANNOUNCES_SIG)
            and XUB_MARK_FIELD in struct_text(norm(cache_hpp), XUB_STRUCT))


def claim_refusal_names_the_largest_fit(sycl_cpp: str, admission_cpp: str) -> bool:
    """The fallback refusal is for the micro-batch the load decided for, names the largest -ub that fits, and comes
    before the ring re-plan; the figure is the ring admission's binary search over the ring's own geometry."""
    txn = body(norm(sycl_cpp), TXN_SIG)
    decide = body(norm(admission_cpp), DECIDE_SIG)
    return (bool(txn) and ordered(txn, REFUSE_FOR, FITS, REFUSE_LITERAL, RING_REPLAN)
            and bool(decide) and LARGEST_FIT in decide and all(f in decide for f in FIT_INPUTS))


def claim_placed_ring_is_kept_for_smaller_contexts(sycl_cpp: str, cache_cpp: str, cache_hpp: str) -> bool:
    """The transaction re-plans the ring for the helper's answer, computed once after the fallback refusal, and never
    for the context's own micro-batch directly. The helper keeps the placed micro-batch only for a plan that PLACED it
    and a context naming a micro-batch no larger; anything else re-plans for the context's own."""
    txn = body(norm(sycl_cpp), TXN_SIG)
    return (bool(txn) and txn.count(RING_N_UBATCH) == 1 and txn.count(RING_REPLAN) == 1
            and ordered(txn, REFUSE_LITERAL, RING_N_UBATCH, RING_REPLAN)
            and "ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch" not in txn
            and RING_HELPER_RULE in body(norm(cache_cpp), RING_HELPER_SIG)
            and RING_HELPER_SIG + ";" in norm(cache_hpp))


def claim_fit_credits_the_placed_compute_room(sycl_cpp: str, zone_cpp: str, cache_cpp: str) -> bool:
    """The hold-spill fit credits the planned compute term at the explicit micro-batch the published single-device
    plan PLACED on the device, read once before the demand is asked: at that rung a request no larger than the term
    adds nothing to the bound (whole or nothing), the plan stays in it, and every other rung and load is judged as
    before."""
    fit = body(norm(sycl_cpp), FIT_SIG)
    zone = norm(zone_cpp)
    demand = body(zone, DEMAND_SIG)
    return (bool(fit) and ordered(fit, FIT_PLAN, FIT_GATE, FIT_ROOM, FIT_FIELDS, FIT_DEMAND, FIT_FITS)
            and fit.count("in.compute_room_n_ubatch =") == 1 and fit.count("in.compute_room_bytes =") == 1
            and ROOM_HELPER_RULE in body(norm(cache_cpp), ROOM_HELPER_SIG)
            and CREDIT_RULE in body(zone, CREDIT_SIG)
            and bool(demand) and ordered(demand, DEMAND_CREDIT, DEMAND_BOUND))


def claim_grown_zone_is_announced(sycl_cpp: str, zone_cpp: str, zone_hpp: str = ZONE_HPP) -> bool:
    """A load whose plan was packed against a RUNTIME zone an earlier load grew says so in ONE WARN after its late
    plan, and only when the live zone is larger than the zone the arena reserves for the plan (the larger of the floor
    and the requirement, rounded up to the arena's grain): the first load of a process, whose zone is its own plan's,
    never prints it."""
    sycl = norm(sycl_cpp)
    late = body(sycl, SET_INV_SIG)
    kept = body(sycl, KEPT_SIG)
    return (bool(late) and late.count(LATE_PLAN) == 1
            and sycl.count("ggml_sycl_warn_runtime_zone_kept_from_earlier_load(") == 2
            and bool(kept) and ordered(kept, KEPT_READS, KEPT_LIVE, KEPT_FLOOR, KEPT_GATE, KEPT_WARN)
            and kept.count("GGML_LOG_") == 1
            and KEPT_RULE in body(norm(zone_cpp), KEPT_RULE_SIG)
            and RESERVED_RULE in body(norm(zone_hpp), RESERVED_SIG))


def claim_runtime_zone_is_one_reservation(zone_hpp: str, zone_cpp: str, cache_cpp: str, admission_cpp: str,
                                          sycl_cpp: str) -> bool:
    """The RUNTIME zone is reserved, planned, planned against and compared through one checked helper: the arena's
    reserve, the zone ensure_planned_arena_zones() plans, the explicit micro-batch's room, the grown-zone rule and its
    WARN each call zone_runtime_reserved_bytes(), and none of them rounds the zone itself. The helper takes the larger
    of the floor and the requirement and rounds it up to the arena's named grain with the addition checked, and the
    cache asserts the grain is the zone allocator's block grain."""
    hpp = norm(zone_hpp)
    cache = norm(cache_cpp)
    reserve = body(cache, ARENA_RESERVE_SIG)
    ensure = body(cache, ENSURE_SIG)
    room = body(norm(admission_cpp), ROOM_ZONE_SIG)
    rule = body(norm(zone_cpp), KEPT_RULE_SIG)
    kept = body(norm(sycl_cpp), KEPT_SIG)
    sites = (reserve, ensure, room, rule, kept)
    return (ARENA_ALIGNMENT in hpp and cache.count(ARENA_GRAIN_ASSERT) == 1
            and RESERVED_RULE in body(hpp, RESERVED_SIG)
            and all(site.count("zone_runtime_reserved_bytes(") == 1 for site in sites)
            and ARENA_RESERVE_CALL in reserve and "runtime_bytes = (runtime_bytes +" not in reserve
            and ENSURE_CALL in ensure and "runtime_zone = planned_runtime_scratch;" not in ensure
            and ROOM_ZONE_CALL in room and "align" not in room
            and KEPT_RULE in rule and "std::max(" not in rule
            and KEPT_GATE in kept and KEPT_WARN in kept and "std::max(" not in kept)


def claim_planner_moves_only_experts(cache_cpp: str, cache_hpp: str = CACHE_HPP) -> bool:
    """The room is a named hold on the routed-expert budget, taken after the KV context room and before the pack's
    stats, against what the dense weights, their KV and that room left, so it never moves a dense layer or KV. The
    pack predicate is the budget less the hold. What it moved is net, as the KV context room counts its cost: both
    packs' device triplets are counted, the bytes moved are the difference, and the layers are the ones the real pack
    keeps fewer triplets in. Counting the triplets that fit the larger budget and not the real one over-states it,
    because that budget can take a larger triplet and skip smaller ones the real pack placed. The hold is the whole
    room on PLACE and GROW_ZONE alike, with the part the zone already holds returned first, so the plan the load grows
    the zone for is the plan it keeps."""
    cache = norm(cache_cpp)
    planner = body(cache, PLANNER_SIG)
    hold = body(cache, HOLD_SIG)
    return (bool(planner)
            and ordered(planner, KV_ROOM, HOLD_CALL, STATS_DECL, WITHOUT_ROOM, PACK_PREDICATE, MOVED, MOVED_LAYERS)
            and planner.count(PACK_PREDICATE) == 1 and planner.count("plan.explicit_ubatch.moved_layers") == 1
            and MOVED_NET in struct_text(norm(cache_hpp), XUB_STRUCT)
            and bool(hold) and ordered(hold, DECIDE, HOLD_REASON, HOLD_GATE, HOLD_STEPS))


def claim_load_grows_the_zone_before_the_plan(sycl_cpp: str) -> bool:
    """A room the experts can give grows the RUNTIME zone for the micro-batch, and the plan the load keeps is made
    again with the room held. A zone that cannot grow gets the load's planning shape back and is planned without the
    room, so no expert leaves the device for a zone that did not grow: the load's planning-shape plan."""
    store = body(norm(sycl_cpp), STORE_SIG)
    return (bool(store)
            and ordered(store, LAMBDA_ROOM, FIRST_PLAN, GROW_BRANCH, GROW_PUBLISH, GROW_REPLAN, GROW_FALLBACK,
                        FALLBACK_PLAN)
            and store.count("plan_single_device(false)") == 1)


def claim_held_room_is_republished_before_the_plan(sycl_cpp: str) -> bool:
    """Before the first plan, the room is filled from the plan as published and, when the planner's decision with no
    expert budget is PLACE (the zone already holds the room: an earlier stage of the load or an earlier load in the
    process grew it), the micro-batch's ring and dense scratch are published again so the plan reads them; a publish
    that fails part-way publishes the planning shape again, as the fallback does. The predicate is the decision, not a
    copy of its rule."""
    store = body(norm(sycl_cpp), STORE_SIG)
    return (bool(store) and ordered(store, PRE_FILL, PRE_GATE, FIRST_PLAN, GROW_BRANCH)
            and store.count(PRE_GATE) == 1 and "room_reserved_bytes >=" not in store)


def claim_failed_growth_warns(sycl_cpp: str) -> bool:
    """A zone that could not grow is said once, at WARN, after the fallback plan: the micro-batch the user named is not
    placed, and the line names the RUNTIME bytes the arena could not reserve."""
    store = body(norm(sycl_cpp), STORE_SIG)
    return bool(store) and ordered(store, GROW_FALLBACK, FALLBACK_PLAN, REVERT_WARN) and store.count(REVERT_WARN) == 1


def claim_transport_names_no_micro_batch_for_auto(llama_h: str, model_cpp: str, sycl_cpp: str, common_cpp: str,
                                                  bench_cpp: str, sycl_h: str = SYCL_H) -> bool:
    """The micro-batch the caller names reaches the planner, and neither caller names one when the backend picks the
    micro-batch itself, so the auto-ubatch trial's rungs never match a placed micro-batch. llama-bench reloads the
    model when the named micro-batch changes."""
    model = norm(model_cpp)
    sycl = norm(sycl_cpp)
    bench = norm(bench_cpp)
    bench_hint = body(bench, BENCH_HINT_SIG)
    return (PARAM_FIELD in norm(llama_h) and len(MODEL_DEFAULT_RE.findall(model_cpp)) == 1
            and INVENTORY_SET in body(model, POPULATE_SIG) and model.count(CALLER_PASSES) == 2
            and BACKEND_COPY in sycl and OFFSET_PIN in sycl
            and HEADER_SIZE_PIN in norm(sycl_h) and HEADER_OFFSET_PIN in norm(sycl_h)
            and COMMON_NONE in norm(common_cpp)
            and bool(bench_hint) and bench_hint.lstrip("{ ").startswith(BENCH_NONE)
            and BENCH_SET in bench and BENCH_EQUAL in bench)


def claim_bench_reloads_only_on_a_micro_batch_change(bench_cpp: str) -> bool:
    """llama-bench reloads the model when the named micro-batch changes and at no other test boundary: the hint reads
    -b and -ub and nothing sized by the context, so `-ub 512 -p 512 -n 128` loads once, and "-ub auto" and a build
    without the SYCL backend name none, so they never reload on it."""
    bench_hint = body(norm(bench_cpp), BENCH_HINT_SIG)
    return (bool(bench_hint) and bench_hint.lstrip("{ ").startswith(BENCH_NONE)
            and bench_hint.count("return ") == 2 and BENCH_RETURN in bench_hint
            and "n_ctx" not in bench_hint and "n_prompt" not in bench_hint and "n_gen" not in bench_hint)


def test_placement_is_one_warn_line():
    assert claim_placement_is_one_warn_line(SYCL_CPP, CACHE_CPP, CACHE_HPP)


def test_refusal_names_the_largest_fit():
    assert claim_refusal_names_the_largest_fit(SYCL_CPP, ADMISSION_CPP)


def test_placed_ring_is_kept_for_smaller_contexts():
    assert claim_placed_ring_is_kept_for_smaller_contexts(SYCL_CPP, CACHE_CPP, CACHE_HPP)


def test_fit_credits_the_placed_compute_room():
    assert claim_fit_credits_the_placed_compute_room(SYCL_CPP, ZONE_CPP, CACHE_CPP)


def test_grown_zone_is_announced():
    assert claim_grown_zone_is_announced(SYCL_CPP, ZONE_CPP)


def test_runtime_zone_is_one_reservation():
    assert claim_runtime_zone_is_one_reservation(ZONE_HPP, ZONE_CPP, CACHE_CPP, ADMISSION_CPP, SYCL_CPP)


def test_planner_moves_only_experts():
    assert claim_planner_moves_only_experts(CACHE_CPP)


def test_load_grows_the_zone_before_the_plan():
    assert claim_load_grows_the_zone_before_the_plan(SYCL_CPP)


def test_bench_reloads_only_on_a_micro_batch_change():
    assert claim_bench_reloads_only_on_a_micro_batch_change(BENCH_CPP)


def test_failed_growth_warns():
    assert claim_failed_growth_warns(SYCL_CPP)


def test_held_room_is_republished_before_the_plan():
    assert claim_held_room_is_republished_before_the_plan(SYCL_CPP)


def test_transport_names_no_micro_batch_for_auto():
    assert claim_transport_names_no_micro_batch_for_auto(LLAMA_H, MODEL_CPP, SYCL_CPP, COMMON_CPP, BENCH_CPP)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` normalized, with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def _one_warn(sycl_cpp: str = None, cache_cpp: str = None, cache_hpp: str = None) -> bool:
    return claim_placement_is_one_warn_line(sycl_cpp if sycl_cpp is not None else SYCL_CPP,
                                            cache_cpp if cache_cpp is not None else CACHE_CPP,
                                            cache_hpp if cache_hpp is not None else CACHE_HPP)


def test_mutant_decision_line_at_info_fails():
    """Below default verbosity the decision is invisible, which is how the refusal it replaces read as no row."""
    assert not _one_warn(_once(SYCL_CPP, WARN_LITERAL, WARN_LITERAL.replace("GGML_LOG_WARN(", "GGML_LOG_INFO(", 1)))


def test_mutant_decision_line_without_the_bytes_moved_fails():
    assert not _one_warn(
        _once(SYCL_CPP, WARN_LITERAL,
              WARN_LITERAL.replace("it re-placed %.1f MB of routed experts", "it re-placed routed experts", 1)
              .replace(" xub.moved_bytes() / mb,", "", 1)))


def test_mutant_decision_line_every_context_fails():
    """Without the mark in the rule every later context would print it again: llama-bench opens one context per
    instance on one model, so a -d or -fa list printed it twice with no expert moving."""
    assert not _one_warn(cache_cpp=_once(CACHE_CPP, ANNOUNCES_RULE, ANNOUNCES_RULE.replace(" && !xub.announced", "", 1)))


def test_mutant_decision_line_for_a_probe_fails():
    """A probe places nothing; the auto-ubatch trial runs in probe mode."""
    assert not _one_warn(cache_cpp=_once(CACHE_CPP, ANNOUNCES_RULE, ANNOUNCES_RULE.replace("!probe_mode && ", "", 1)))


def test_mutant_decision_line_for_a_room_that_moved_nothing_fails():
    """A room the weight zone's slack held moved no expert, so there is nothing to announce at WARN."""
    assert not _one_warn(
        cache_cpp=_once(CACHE_CPP, ANNOUNCES_RULE, ANNOUNCES_RULE.replace(" && xub.moved_bytes() > 0", "", 1)))


def test_mutant_decision_line_keyed_on_the_triplet_count_fails():
    """With triplets of unequal size the device can keep as many triplets and still hold fewer bytes, so a rule that
    reads the count leaves a room that cost device bytes unannounced."""
    assert not _one_warn(
        cache_cpp=_once(CACHE_CPP, ANNOUNCES_RULE,
                        ANNOUNCES_RULE.replace("xub.moved_bytes() > 0", "xub.expert_groups_without > xub.expert_groups",
                                               1)))


def test_mutant_commit_site_not_announced_fails():
    assert not _one_warn(_once(SYCL_CPP, ANNOUNCE_COMMIT, "announce_kv_host_demotions(*immutable->plan);"))


def test_mutant_mark_never_set_fails():
    """The announcing context publishes no mark, so every later context announces again."""
    assert not _one_warn(_once(SYCL_CPP, WARN_MARK, ""))


def test_mutant_mark_set_after_publish_fails():
    """A mark set once next_plan has been moved into the published snapshot never reaches it."""
    n = _once(SYCL_CPP, WARN_MARK, "")
    assert n.count(PUBLISH) == 1
    assert not _one_warn(n.replace(PUBLISH, PUBLISH + " " + WARN_MARK, 1))


def test_mutant_mark_reset_after_the_plan_copy_fails():
    """The previous context's mark is dropped from the copy, so this context announces again."""
    assert not _one_warn(_once(SYCL_CPP, PLAN_COPY, PLAN_COPY + " next_plan.explicit_ubatch.announced = false;"))


def test_mutant_demoted_plan_drops_the_mark_fails():
    """The KV demotion's working copy comes back without the decision, and with it the mark."""
    assert not _one_warn(_once(SYCL_CPP, DEMOTED_COPY, DEMOTED_COPY + " demoted_plan.explicit_ubatch = {};"))


def test_mutant_warn_gated_on_the_previous_micro_batch_fails():
    """The old gate: any context of another micro-batch reset it, so the named micro-batch's next context printed it
    again with no expert moving."""
    assert not _one_warn(
        _once(SYCL_CPP, WARN_GATE, "if (warn && current->kv_info.n_ubatch != next_kv_info.n_ubatch) {"))


def test_mutant_warn_filtered_by_the_context_micro_batch_fails():
    """With -p below -ub every context resolves a micro-batch other than the placed one, so a load that moved experts
    announced nothing at default verbosity."""
    assert not _one_warn(
        _once(SYCL_CPP, ANNOUNCE_FOR,
              "if (xub.action != ggml_sycl::pp_moe_onednn_explicit_ubatch_action::PLACE || "
              "xub.n_ubatch != next_kv_info.n_ubatch) { return; }"))


def test_mutant_mark_field_without_a_default_fails():
    """A load's fresh plan must start unmarked, or its first context never announces."""
    assert not _one_warn(cache_hpp=_once(CACHE_HPP, XUB_MARK_FIELD, "bool announced;"))


def test_mutant_refusal_without_the_largest_fit_fails():
    """The figure is what the user needs to retry with, and only a refusal has a retry to name."""
    assert not claim_refusal_names_the_largest_fit(_once(SYCL_CPP, FITS, ""), ADMISSION_CPP)


def test_mutant_refusal_literal_reworded_fails():
    assert not claim_refusal_names_the_largest_fit(
        _once(SYCL_CPP, REFUSE_LITERAL, REFUSE_LITERAL.replace("re-placed to the host tier", "evicted", 1)),
        ADMISSION_CPP)


def test_mutant_refusal_after_the_ring_replan_fails():
    """After the re-plan the legacy refusal (no largest fit) would answer first."""
    n = _once(SYCL_CPP, RING_REPLAN, "")
    at = n.find(REFUSE_FOR)
    assert at >= 0
    assert not claim_refusal_names_the_largest_fit(n[:at] + RING_REPLAN + " " + n[at:], ADMISSION_CPP)


def test_mutant_largest_fit_not_from_the_ring_arithmetic_fails():
    assert not claim_refusal_names_the_largest_fit(
        SYCL_CPP, _once(ADMISSION_CPP, LARGEST_FIT, "decision.largest_fitting_n_ubatch = in.n_ubatch_default;"))


def _ring_kept(sycl_cpp: str = None, cache_cpp: str = None) -> bool:
    return claim_placed_ring_is_kept_for_smaller_contexts(sycl_cpp if sycl_cpp is not None else SYCL_CPP,
                                                          cache_cpp if cache_cpp is not None else CACHE_CPP, CACHE_HPP)


def test_mutant_ring_replanned_for_the_context_micro_batch_fails():
    """The depth-sweep refusal: a -n 128 context shrank the placed ring, and the next -p 512 context could not regrow
    it in a zone with no slack beyond its terms."""
    assert not _ring_kept(sycl_cpp=_once(SYCL_CPP, RING_REPLAN,
                                         RING_REPLAN.replace("ring_n_ubatch", "next_kv_info.n_ubatch", 1)))


def test_mutant_ring_kept_for_a_larger_context_fails():
    """A context above the placed micro-batch would dispatch more rows than the kept ring's slots hold."""
    assert not _ring_kept(cache_cpp=_once(CACHE_CPP, RING_HELPER_RULE,
                                          RING_HELPER_RULE.replace(" && n_ubatch <= xub.n_ubatch", "", 1)))


def test_mutant_ring_kept_for_a_plan_that_placed_nothing_fails():
    """Only PLACE sized the zone for the micro-batch; a refused or ungrown plan has no placed ring to keep."""
    assert not _ring_kept(cache_cpp=_once(
        CACHE_CPP, RING_HELPER_RULE,
        RING_HELPER_RULE.replace("xub.action == pp_moe_onednn_explicit_ubatch_action::PLACE && ", "", 1)))


def test_mutant_ring_kept_for_a_context_naming_no_micro_batch_fails():
    """A context at 0 names no micro-batch: the re-plan reports it rather than answer for it."""
    assert not _ring_kept(cache_cpp=_once(CACHE_CPP, RING_HELPER_RULE,
                                          RING_HELPER_RULE.replace("n_ubatch != 0 && ", "", 1)))


def test_mutant_fit_credits_an_unplaced_room_fails():
    """A GROW_ZONE or REFUSE decision did not size the zone for the micro-batch, so there is no room to credit."""
    assert not claim_fit_credits_the_placed_compute_room(
        SYCL_CPP, ZONE_CPP,
        _once(CACHE_CPP, ROOM_HELPER_RULE,
              ROOM_HELPER_RULE.replace("xub.action == pp_moe_onednn_explicit_ubatch_action::PLACE && ", "", 1)))


def test_mutant_fit_room_without_the_helper_fails():
    """The term read straight from the getter credits a rung no PLACE decision sized the zone for."""
    assert not claim_fit_credits_the_placed_compute_room(
        _once(SYCL_CPP, FIT_ROOM,
              "room_bytes = ggml_sycl::unified_cache_get_planned_compute_term_bytes(q.device); "
              "room_n_ubatch = room_plan->explicit_ubatch.n_ubatch;"), ZONE_CPP, CACHE_CPP)


def test_mutant_fit_credits_another_device_fails():
    assert not claim_fit_credits_the_placed_compute_room(
        _once(SYCL_CPP, FIT_GATE, FIT_GATE.replace(" && room_plan->device_id == q.device", "", 1)), ZONE_CPP,
        CACHE_CPP)


def test_mutant_fit_credit_after_the_demand_fails():
    """A demand asked before the room is read is the capacity-blind bound that refused the placed context."""
    n = _once(SYCL_CPP, FIT_DEMAND, "")
    at = n.find(FIT_PLAN)
    assert at >= 0
    assert not claim_fit_credits_the_placed_compute_room(n[:at] + FIT_DEMAND + " " + n[at:], ZONE_CPP, CACHE_CPP)


def test_mutant_credit_takes_part_of_a_larger_request_fails():
    """Whole or nothing: a request above the room is held back whole and spills whole."""
    assert not claim_fit_credits_the_placed_compute_room(
        SYCL_CPP, _once(ZONE_CPP, CREDIT_RULE, "return n_ubatch != 0 && n_ubatch == in.compute_room_n_ubatch;"),
        CACHE_CPP)


def test_mutant_credit_ignores_the_rung_fails():
    """Every rung would be credited the room planned for one, the auto trial's included."""
    assert not claim_fit_credits_the_placed_compute_room(
        SYCL_CPP, _once(ZONE_CPP, CREDIT_RULE, "return n_ubatch != 0 && request <= in.compute_room_bytes;"), CACHE_CPP)


def test_mutant_grown_zone_rule_drops_the_comparison_fails():
    """Every load with a RUNTIME zone would print it, the first load of a process included."""
    assert not claim_grown_zone_is_announced(
        SYCL_CPP,
        _once(ZONE_CPP, KEPT_RULE, KEPT_RULE.replace("live_zone_bytes > reserved", "live_zone_bytes != 0", 1)))


def test_mutant_grown_zone_warn_ungated_fails():
    assert not claim_grown_zone_is_announced(_once(SYCL_CPP, KEPT_GATE, ""), ZONE_CPP)


def test_mutant_grown_zone_rule_without_the_slack_fails():
    """A first load whose requirement is not a multiple of 256 bytes would print it for the arena's own rounding."""
    rounding = "*out = (planned + k_zone_arena_alignment - 1) & ~(k_zone_arena_alignment - 1);"
    assert not claim_grown_zone_is_announced(SYCL_CPP, ZONE_CPP, _once(ZONE_HPP, rounding, "*out = planned;"))


def test_mutant_grown_zone_rule_ignores_the_floor_fails():
    """A first load whose zone the floor sized would print it."""
    assert not claim_grown_zone_is_announced(
        SYCL_CPP, _once(ZONE_CPP, KEPT_RULE, KEPT_RULE.replace("(floor_bytes, requirement,", "(0, requirement,", 1)))


def test_mutant_grown_zone_warn_prints_the_unrounded_plan_fails():
    """The WARN would name a zone the arena never reserved, up to 255 bytes short of the one it holds."""
    assert not claim_grown_zone_is_announced(
        _once(SYCL_CPP, KEPT_WARN, KEPT_WARN.replace("reserved / mb", "std::max(floor, requirement) / mb", 1)),
        ZONE_CPP)


# Each mutant re-inlines the RUNTIME zone's rounding at one of the five sites, the drift the one helper exists to stop.
def test_mutant_arena_reserve_rounds_the_runtime_zone_itself_fails():
    inlined = "runtime_bytes = (runtime_bytes + 255) & ~size_t(255); if (false) {"
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, ZONE_CPP, _once(CACHE_CPP, ARENA_RESERVE_CALL, inlined), ADMISSION_CPP, SYCL_CPP)


def test_mutant_ensure_plans_the_unrounded_requirement_fails():
    inlined = ("if (!unified_cache_get_planned_runtime_zone_requirement(dev_id, &planned_runtime_scratch)) {")
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, ZONE_CPP, _once(CACHE_CPP, ENSURE_CALL, inlined), ADMISSION_CPP, SYCL_CPP)


def test_mutant_room_rounds_the_zone_itself_fails():
    inlined = ("const size_t requirement = in.runtime_fixed_bytes + dense_bytes + ring_bytes; "
               "return pp_moe_onednn_checked_align_slot_bytes("
               "requirement > in.runtime_floor_bytes ? requirement : in.runtime_floor_bytes, out);")
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, ZONE_CPP, CACHE_CPP, _once(ADMISSION_CPP, ROOM_ZONE_CALL, inlined), SYCL_CPP)


def test_mutant_kept_rule_rounds_the_plan_itself_fails():
    inlined = ("const size_t planned = std::max(floor_bytes, requirement); "
               "return live_zone_bytes > planned && live_zone_bytes - planned > 255;")
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, _once(ZONE_CPP, KEPT_RULE, inlined), CACHE_CPP, ADMISSION_CPP, SYCL_CPP)


def test_mutant_kept_warn_prints_its_own_plan_fails():
    inlined = KEPT_WARN.replace("reserved / mb", "std::max(floor, requirement) / mb", 1)
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, ZONE_CPP, CACHE_CPP, ADMISSION_CPP, _once(SYCL_CPP, KEPT_WARN, inlined))


def test_mutant_reservation_ignores_the_floor_fails():
    assert not claim_runtime_zone_is_one_reservation(
        _once(ZONE_HPP, "const size_t planned = requirement > floor_bytes ? requirement : floor_bytes;",
              "const size_t planned = requirement;"),
        ZONE_CPP, CACHE_CPP, ADMISSION_CPP, SYCL_CPP)


def test_mutant_reservation_rounding_unchecked_fails():
    assert not claim_runtime_zone_is_one_reservation(
        _once(ZONE_HPP, "if (planned > SIZE_MAX - (k_zone_arena_alignment - 1)) { return false; }", ""),
        ZONE_CPP, CACHE_CPP, ADMISSION_CPP, SYCL_CPP)


def test_mutant_reservation_grain_unnamed_fails():
    assert not claim_runtime_zone_is_one_reservation(
        _once(ZONE_HPP, ARENA_ALIGNMENT, "constexpr size_t k_zone_arena_grain = 256;"),
        ZONE_CPP, CACHE_CPP, ADMISSION_CPP, SYCL_CPP)


def test_mutant_reservation_grain_not_the_allocator_grain_fails():
    """Without the assert, the arena's grain and the zone allocator's block grain are two sources free to drift."""
    assert not claim_runtime_zone_is_one_reservation(
        ZONE_HPP, ZONE_CPP,
        _once(CACHE_CPP, ARENA_GRAIN_ASSERT, "static_assert(k_zone_arena_alignment >= 1,"),
        ADMISSION_CPP, SYCL_CPP)


def test_mutant_grown_zone_warn_at_info_fails():
    assert not claim_grown_zone_is_announced(
        _once(SYCL_CPP, KEPT_WARN, KEPT_WARN.replace("GGML_LOG_WARN(", "GGML_LOG_INFO(", 1)), ZONE_CPP)


def test_mutant_grown_zone_warn_before_the_plan_fails():
    """Asked before the late plan, the requirement is the previous stage's, not this plan's."""
    assert not claim_grown_zone_is_announced(
        _once(SYCL_CPP, LATE_PLAN, "ggml_sycl_warn_runtime_zone_kept_from_earlier_load(ctx->device); "
                                   "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct);"), ZONE_CPP)


def test_mutant_grown_zone_warn_from_the_early_plan_too_fails():
    """Asked from the early plan as well, one load prints it twice."""
    early = ("compute_vram_budget_for_plan(ctx, vram_budget, budget_pct, base_mem, free_mem); "
             "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct); }")
    n = norm(SYCL_CPP)
    assert n.count(early) == 1
    n = n.replace(early, early[:-1] + "ggml_sycl_warn_runtime_zone_kept_from_earlier_load(ctx->device); }", 1)
    assert not claim_grown_zone_is_announced(n, ZONE_CPP)


def test_mutant_credit_drops_the_plan_fails():
    assert not claim_fit_credits_the_placed_compute_room(
        SYCL_CPP,
        _once(ZONE_CPP, DEMAND_BOUND,
              "zone_hold_spill_raw_demand(zone_hold_spill_bound(0, request, n_ubatch, n_ubatch), in.kv_room);"),
        CACHE_CPP)


def test_mutant_room_dips_into_the_kv_room_fails():
    """Held before the KV context room, the ring would take the device bytes the opening context's KV holds."""
    n = _once(CACHE_CPP, HOLD_CALL, "")
    at = n.find(KV_ROOM)
    assert at >= 0
    assert not claim_planner_moves_only_experts(n[:at] + HOLD_CALL + " " + n[at:])


def test_mutant_hold_only_on_place_fails():
    """Holding nothing on GROW_ZONE packs the experts into the bytes the zone is about to take: the plan the load grows
    the zone for would not be the plan it keeps."""
    assert not claim_planner_moves_only_experts(
        _once(CACHE_CPP, HOLD_GATE, "if (decision.action == pp_moe_onednn_explicit_ubatch_action::PLACE) {"))


def test_mutant_hold_without_returning_the_grown_zone_fails():
    """After the zone grew, its part of the room already left the weight capacity; holding it again double-counts."""
    assert not claim_planner_moves_only_experts(
        _once(CACHE_CPP, HOLD_STEPS, HOLD_STEPS.replace("remaining += room.room_reserved_bytes; ", "", 1)))


def test_mutant_pack_predicate_ignores_the_hold_fails():
    assert not claim_planner_moves_only_experts(
        _once(CACHE_CPP, PACK_PREDICATE,
              "const bool on_device = group.charge_bytes <= remaining + ubatch_hold_bytes;"))


def test_mutant_moved_counts_device_experts_fails():
    """The real pack's own triplets counted as the second pack's: the room would seem to cost nothing."""
    assert not claim_planner_moves_only_experts(
        _once(CACHE_CPP, MOVED, MOVED.replace("if (on_device) {", "if (group.charge_bytes <= without_ubatch_room) {", 1)))


def test_mutant_moved_is_the_gross_count_fails():
    """The gross count: a triplet the second pack places is counted only when the real pack does not, which credits
    the larger triplet it took and none of the smaller ones the real pack placed instead."""
    gross = MOVED.replace("plan.explicit_ubatch.expert_bytes_without += group.bytes; "
                          "plan.explicit_ubatch.expert_groups_without++; ",
                          "if (!on_device) { plan.explicit_ubatch.expert_bytes_without += group.bytes; "
                          "plan.explicit_ubatch.expert_groups_without++; } ", 1)
    assert gross != MOVED
    assert not claim_planner_moves_only_experts(_once(CACHE_CPP, MOVED, gross))


def test_mutant_moved_layers_count_any_change_fails():
    """A layer the room let keep more triplets is not one it moved experts out of."""
    assert not claim_planner_moves_only_experts(
        _once(CACHE_CPP, MOVED_LAYERS, MOVED_LAYERS.replace("drop > 0", "drop != 0", 1)))


def test_mutant_moved_bytes_gross_fails():
    """The second pack's whole device bytes, not the difference, is the room's cost only when the real pack placed
    nothing."""
    assert not claim_planner_moves_only_experts(
        CACHE_CPP,
        _once(CACHE_HPP, MOVED_NET,
              "size_t moved_bytes() const { return expert_bytes_without; }"))


def test_mutant_grown_zone_not_replanned_fails():
    """Growing the zone without planning again keeps a plan packed into weight capacity the zone now holds."""
    assert not claim_load_grows_the_zone_before_the_plan(_once(SYCL_CPP, GROW_REPLAN, "if (placed) { }"))


def test_mutant_failed_growth_keeps_the_micro_batch_fails():
    assert not claim_load_grows_the_zone_before_the_plan(_once(SYCL_CPP, GROW_FALLBACK, "if (!placed) {"))


def test_mutant_held_room_not_republished_fails():
    """Without it, a later stage of the load, or a later load, plans against the grown zone with the ring published at
    the planning shape, so the room it holds is never charged to the routed experts."""
    assert not claim_held_room_is_republished_before_the_plan(_once(SYCL_CPP, PRE_GATE, ""))


def test_mutant_held_room_predicate_rederived_fails():
    """A copy of the decision's rule is a second source for one fact, free to drift from the planner's."""
    rederived = ("g_placement_kv_info.explicit_ubatch.valid && g_placement_kv_info.explicit_ubatch.room_bytes > 0 && "
                 "g_placement_kv_info.explicit_ubatch.room_reserved_bytes >= "
                 "g_placement_kv_info.explicit_ubatch.room_bytes")
    assert not claim_held_room_is_republished_before_the_plan(
        _once(SYCL_CPP, PRE_GATE, PRE_GATE.replace(PRE_DECIDE, rederived, 1)))


def test_mutant_held_room_failed_publish_left_half_done_fails():
    assert not claim_held_room_is_republished_before_the_plan(
        _once(SYCL_CPP, PRE_GATE,
              "if (" + PRE_DECIDE + ") { "
              "(void) ggml_sycl_publish_load_ubatch_scratch(ctx->device, explicit_n_ubatch); }"))


def test_mutant_held_room_republished_after_the_plan_fails():
    """Published after the first plan, the plan has already read the planning-shape ring."""
    moved = _once(SYCL_CPP, PRE_GATE + " " + FIRST_PLAN, FIRST_PLAN + " " + PRE_GATE)
    assert not claim_held_room_is_republished_before_the_plan(moved)


def test_mutant_held_room_read_before_the_fill_fails():
    """Without the fill the decision reads the room an earlier plan left in placement_kv_info."""
    assert not claim_held_room_is_republished_before_the_plan(
        _once(SYCL_CPP, PRE_FILL, "const uint32_t explicit_n_ubatch = g_placement_kv_info.n_ubatch_context;"))


def test_mutant_failed_growth_plans_with_the_room_fails():
    """Planned with the room after the zone failed to grow, the fill reads GROW_ZONE again and the hold moves experts
    to the host tier for a ring the zone cannot hold: worse than the load's planning-shape plan."""
    assert not claim_load_grows_the_zone_before_the_plan(
        _once(SYCL_CPP, FALLBACK_PLAN, "plan_candidate = plan_single_device(true);"))


def test_mutant_failed_growth_at_info_fails():
    """At INFO the line is dropped at default verbosity, and the user is not told the micro-batch was not placed."""
    assert not claim_failed_growth_warns(
        _once(SYCL_CPP, REVERT_WARN, REVERT_WARN.replace("GGML_LOG_WARN(", "GGML_LOG_INFO(", 1)))


def test_mutant_failed_growth_reworded_fails():
    assert not claim_failed_growth_warns(
        _once(SYCL_CPP, REVERT_WARN, REVERT_WARN.replace("; the ring is re-planned when the context opens", "", 1)))


def test_mutant_failed_growth_silent_fails():
    assert not claim_failed_growth_warns(_once(SYCL_CPP, REVERT_WARN, ""))


def test_mutant_plan_without_room_keeps_the_room_fails():
    assert not claim_load_grows_the_zone_before_the_plan(
        _once(SYCL_CPP, LAMBDA_ROOM, LAMBDA_ROOM.replace(" g_placement_kv_info.explicit_ubatch = {};", "", 1)))


def test_mutant_common_names_the_auto_micro_batch_fails():
    """Naming the trial's micro-batch would let a rung place experts; the trial must stay unchanged."""
    assert not claim_transport_names_no_micro_batch_for_auto(
        LLAMA_H, MODEL_CPP, SYCL_CPP,
        _once(COMMON_CPP, COMMON_NONE, "mparams.n_ubatch_hint = 0; if (params.n_batch > 0) {"), BENCH_CPP)


def test_mutant_bench_names_the_auto_micro_batch_fails():
    mutant = _once(BENCH_CPP, BENCH_NONE, BENCH_NONE.replace("n_ubatch < 0 || ", "", 1))
    assert not claim_transport_names_no_micro_batch_for_auto(LLAMA_H, MODEL_CPP, SYCL_CPP, COMMON_CPP, mutant)
    assert not claim_bench_reloads_only_on_a_micro_batch_change(mutant)


def test_mutant_bench_names_a_micro_batch_without_sycl_fails():
    """Without the SYCL backend nothing reads the hint, so naming one there would only make a -ub sweep reload."""
    mutant = _once(BENCH_CPP, BENCH_NONE, BENCH_NONE.replace(' || ggml_backend_reg_by_name("SYCL") == nullptr', "", 1))
    assert not claim_bench_reloads_only_on_a_micro_batch_change(mutant)


def test_mutant_bench_hint_clamped_by_the_context_fails():
    """Clamped by the context, pp512 and tg128 of `-ub 512` name 512 and 128 and the model reloads between them."""
    clamped = "std::min(std::min((uint32_t) n_batch, n_ctx()), (uint32_t) n_ubatch)"
    mutant = _once(BENCH_CPP, BENCH_RETURN,
                   BENCH_RETURN.replace("std::min((uint32_t) n_batch, (uint32_t) n_ubatch)", clamped, 1))
    assert not claim_bench_reloads_only_on_a_micro_batch_change(mutant)


def test_mutant_bench_reuses_the_model_across_micro_batches_fails():
    assert not claim_transport_names_no_micro_batch_for_auto(
        LLAMA_H, MODEL_CPP, SYCL_CPP, COMMON_CPP, _once(BENCH_CPP, BENCH_EQUAL, ""))


def test_mutant_header_layout_unpinned_fails():
    """Without the header's pin, libllama could write the field where libggml-sycl does not read it."""
    assert not claim_transport_names_no_micro_batch_for_auto(
        LLAMA_H, MODEL_CPP, SYCL_CPP, COMMON_CPP, BENCH_CPP, _once(SYCL_H, HEADER_OFFSET_PIN, ""))


def test_mutant_backend_ignores_the_field_fails():
    assert not claim_transport_names_no_micro_batch_for_auto(
        LLAMA_H, MODEL_CPP, _once(SYCL_CPP, BACKEND_COPY, ""), COMMON_CPP, BENCH_CPP)


def test_mutant_default_hint_dropped_fails():
    """A positional initializer list one short leaves the field zero-initialized, but a reordered one would not be."""
    assert not claim_transport_names_no_micro_batch_for_auto(
        LLAMA_H, MODEL_CPP.replace("/*.n_ubatch_hint               =*/ 0,", "", 1), SYCL_CPP, COMMON_CPP, BENCH_CPP)


def test_mutant_one_loader_stage_drops_the_hint_fails():
    """Mutated on the raw text: the default's initializer labels are comments, which normalizing strips, so a mutant
    built from normalized text fails the claim whether or not the caller passes the hint."""
    anchor = "model.get_n_ctx_hint(), model.get_n_ubatch_hint());"
    assert MODEL_CPP.count(anchor) == 2
    assert claim_transport_names_no_micro_batch_for_auto(LLAMA_H, MODEL_CPP, SYCL_CPP, COMMON_CPP, BENCH_CPP)
    dropped = MODEL_CPP.replace(anchor, "model.get_n_ctx_hint());", 1)
    assert not claim_transport_names_no_micro_batch_for_auto(LLAMA_H, dropped, SYCL_CPP, COMMON_CPP, BENCH_CPP)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
