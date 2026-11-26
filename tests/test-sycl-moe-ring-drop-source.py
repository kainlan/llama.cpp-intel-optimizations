#!/usr/bin/env python3
"""Source contract: where the backend applies the decision on a PP MoE oneDNN scratch ring, and in what order.

Host-only: reads ggml-sycl.cpp and unified-cache.cpp. It builds nothing, loads no model and touches no device.

The rule is the header comment of ggml/src/ggml-sycl/moe-ring-consumer.hpp; what it measured is in
docs/backend/sycl-memory-design.md. The pure functions that answer it (the ring uses of a type, the direct route's
types, the resident check, the decision, the zone rules, the drop marks) are checked on values by
test-sycl-moe-ring-consumer, test-sycl-moe-mmvq-tables and test-zone-sizing. This contract pins what those tests cannot
see: which backend functions call them, in what order, through which getter, and where the stage writes, puts a ring
back, refuses and records a drop. These checks pin it:

  U2  the backend defines no direct-route type set of its own, and the route's admission asks the tables' set;
  W1  the backend's ring weight list takes the ring sizing's filter, ne[2] experts, their bytes and the direct route's
      type set;
  D1  one decision: the backend's resident check answers nothing for a device with no ring published and is the only
      caller of the rule; the stage and the dry run are the only callers of the resident check, of the decision (each
      deciding once) and of moe_pp_ring_drop_can_return_room, and with the capped helper the only readers of the
      RUNTIME zone figures; only the dry run answers from the RUNTIME room alone; the planner rule is asked by the
      WEIGHT-room helper and the no-room reason, the WEIGHT room and the reason by the stage and the dry run, the
      reason once in each; the zone figure, the cap, the planner rule, the WEIGHT room and the reason are pinned whole;
  S1  the stage keeps before it writes where (a) to (c) keep the ring, then snapshots, reads the RUNTIME zone with the
      ring, unpublishes the ring, reads the room and the WEIGHT room, asks the zones with the give-back only where
      moe_pp_ring_drop_can_return_room says a drop can return room, refusing on a failed ensure, reads the WEIGHT room
      and the held flag again, decides and records the decision; it writes through the unpublish alone, plans nothing,
      reads the WEIGHT room through the planner's figure alone, names no cache getter and reads the arena's RUNTIME
      zone after a drop through a helper that creates no cache and answers 0 with no arena;
  K1  every kept exit after the write is inside the keep block, which puts the ring back first and holds no drop
      exit; the kept rebuild asks the zones again and plans again where the WEIGHT room shrank, and each keep logs
      which it is, the no-room keep naming the reason where the plan read no WEIGHT room with the ring and printing
      the figures otherwise; every ensure in the stage is a refusal's condition, and the stage refuses nowhere else; the
      put-back restores the dense scratch, the ring and planner terms and the inventory's ring, and the ring terms'
      put-back is called by the stage's put-back and the dry run alone; the held WARN names no guessed holder;
  S2  the stage is the backend's one writer of the give-back option;
  S3  the planner is the stage's one caller, once per branch, with the replan each answer asks for;
  S4  the drop's record is read by the reader and written by the stage alone, through the decision's record (no mark
      of its own), and the planner publishes no ring before its entry ensure where an earlier pass of the load dropped
      it;
  S5  the early and the late pass each plan through the planner, the stage reads no mark, and the planner asks the
      reader once, before its entry ensure;
  Y1  the dry run decides once per device plan, publishes no ring and puts the planner terms back before it reads the
      room and the WEIGHT room, asks the stage's predicate of both, keeps with the ring terms put back and a line that
      says why (the no-room reason, with whether a load reads no WEIGHT room either or may differ on the arena its
      entry ensure builds, or no RUNTIME room), drops with a line that says what it cannot see, sizes no zone, records
      no load, and is called by the dry run's plans alone, which plan again on a drop;
  Z1  the cache's ensure gives back only when asked: the record is filled before the arena is read, a surplus is asked
      only with a record, an arena with none to give is kept as built, a held arena is kept and marked held, the
      rebuild takes the give-back's sizes and marks rebuilt; nothing else in the cache touches the record.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. The decision's answers are the C++ tests' above, and what a card does with
them is the lead's runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "backend": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "cache": ROOT / "ggml/src/ggml-sycl/unified-cache.cpp",
}

RING_DECIDE_CALL = "moe_pp_ring_decide("
ZONE_FN = "static size_t ggml_sycl_planned_runtime_zone_bytes(int device) {"
ZONE_BODY = ("{ size_t requirement = 0; size_t reserved = 0; if "
             "(!ggml_sycl::unified_cache_get_planned_runtime_zone_requirement(device, &requirement) || "
             "!ggml_sycl::zone_runtime_reserved_bytes(ggml_sycl::unified_cache_runtime_zone_floor_bytes(), "
             "requirement, &reserved)) { return SIZE_MAX; } return reserved; }")
ZONE_CALL = "ggml_sycl_planned_runtime_zone_bytes("
WITH_RING_FN = "static size_t ggml_sycl_ring_runtime_with_ring_bytes(int device) {"
WITH_RING_BODY = ("{ ggml_sycl::unified_cache * const cache = "
                  "ggml_sycl::get_existing_unified_cache_for_device(device); "
                  "const bool arena = cache != nullptr && cache->arena_active(); return "
                  "ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), arena, arena "
                  "? "
                  "cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME) : 0, "
                  "ggml_sycl::unified_cache_runtime_zone_floor_bytes()); }")
WITH_RING_CALL = "ggml_sycl_ring_runtime_with_ring_bytes("
ROOM_CALL = ("const size_t runtime_room = ggml_sycl::moe_pp_ring_runtime_room(runtime_with_ring, "
             "ggml_sycl_planned_runtime_zone_bytes(device));")
PLANNER_ZONE_FN = "static bool ggml_sycl_ring_planner_reads_zone(const ggml_sycl::placement_plan & plan) {"
PLANNER_ZONE_BODY = ("{ return ggml_sycl::kv_reads_device_arena(plan.multi_device, "
                     "ggml_sycl::unified_cache_mode_is_global()); }")
PLANNER_ZONE_CALL = "ggml_sycl_ring_planner_reads_zone("
WEIGHT_ROOM_FN = "static size_t ggml_sycl_ring_plan_weight_room(int device, const ggml_sycl::placement_plan & plan) {"
WEIGHT_ROOM_BODY = ("{ ggml_sycl::unified_cache * const cache = "
                    "ggml_sycl::get_existing_unified_cache_for_device(device); "
                    "if (!ggml_sycl_ring_planner_reads_zone(plan) || cache == nullptr || !cache->arena_active()) { "
                    "return 0; } size_t budget = plan.vram_budget; if (plan.multi_device) { budget = 0; for (size_t i "
                    "= "
                    "0; i < plan.devices.size() && i < plan.per_device_vram_budgets.size(); ++i) { if (plan.devices[i] "
                    "== device) { budget = plan.per_device_vram_budgets[i]; } } } return "
                    "ggml_sycl::unified_cache_kv_weight_capacity(device, budget, plan.multi_device); }")
WEIGHT_ROOM_CALL = "ggml_sycl_ring_plan_weight_room("
NO_ROOM_STRUCT = "struct ggml_sycl_ring_no_weight_room {"
NO_ROOM_STRUCT_BODY = "{ const char * reason = nullptr; bool load_reads_none = false; }"
NO_ROOM_FN = ("static ggml_sycl_ring_no_weight_room ggml_sycl_ring_no_weight_room_reason("
              "const ggml_sycl::placement_plan & plan, size_t weight_room) {")
NO_ROOM_BODY = ('{ if (weight_room > 0) { return {}; } if (!ggml_sycl_ring_planner_reads_zone(plan)) { return { "its '
                'multi-device plan in GLOBAL cache mode packs every device to its budget and reads no WEIGHT zone", '
                'true }; } return { "its plan reads no WEIGHT room on the device, which has no active arena or no '
                'budget in the plan", false }; }')
NO_ROOM_CALL = "ggml_sycl_ring_no_weight_room_reason("
DRY_ANSWER_CALL = "moe_pp_ring_dry_run_answer("
CAN_RETURN_CALL = "moe_pp_ring_drop_can_return_room("
ARENA_RUNTIME_FN = "static size_t ggml_sycl_ring_arena_runtime_zone_bytes(int device) {"
ARENA_RUNTIME_BODY = ("{ ggml_sycl::unified_cache * const cache = "
                      "ggml_sycl::get_existing_unified_cache_for_device(device); "
                      "return cache != nullptr && cache->arena_active() ? "
                      "cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME) : 0; }")
ARENA_RUNTIME_CALL = "ggml_sycl_ring_arena_runtime_zone_bytes("
CACHE_WORDS = ("unified_cache_for_device(", "cache->")
DIRECT_ADMISSION = "moe_mmid_direct_route_runs_type(src0->type) && direct_layout == GGML_LAYOUT_AOS"
WEIGHTS_FN = ("static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights( "
              "const std::vector<ggml_sycl::placement_tensor_info> & detail) {")
WEIGHTS_BODY = ("{ std::vector<ggml_sycl::moe_pp_ring_weight> weights; for (const ggml_sycl::placement_tensor_info & "
                "info : detail) { if (moe_classify_tensor(info.name.c_str()) == MOE_TENSOR_UNKNOWN || "
                "!info.has_shape()) { continue; } const uint32_t uses = moe_pp_ring_uses_of_type(info.type); if (uses "
                "== MOE_PP_RING_USE_NONE) { continue; } ggml_sycl::moe_pp_ring_weight weight; weight.name = "
                "info.name.c_str(); weight.n_experts = info.ne[2] > 0 ? info.ne[2] : 1; weight.bytes = info.size; "
                "weight.uses = uses; "
                "weight.other_device_prompt_route = moe_mmid_direct_route_runs_type(info.type); "
                "weights.push_back(weight); } return weights; }")
DECIDE_FN = ("static ggml_sycl::moe_pp_ring_resident_check ggml_sycl_plan_ring_resident_check( int device, "
             "const std::vector<ggml_sycl::placement_tensor_info> & detail, const ggml_sycl::placement_plan & plan) {")
DECIDE_BODY = ("{ if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(device) == 0) { return {}; } "
               "const std::vector<ggml_sycl::moe_pp_ring_weight> weights = ggml_sycl_inventory_ring_weights(detail); "
               "return ggml_sycl::moe_pp_ring_check_resident_only(plan, weights.data(), weights.size()); }")
DECIDE_CALL = "ggml_sycl_plan_ring_resident_check("
RULE_CALL = "moe_pp_ring_check_resident_only("
STAGE_FN = ("static ggml_sycl_stage_ring_drop ggml_sycl_stage_drop_displacing_ring( int device, "
            "const ggml_sycl::placement_plan & plan, const ggml_sycl::planner_published_terms & published_before) {")
STAGE_ORDER = (
    "const ggml_sycl::moe_pp_ring_resident_check check = ggml_sycl_plan_ring_resident_check(device, "
    "g_tensor_inventory_detail, plan);",
    "if (!check.displaces_experts()) { return ggml_sycl_stage_ring_drop::KEPT; }",
    "const size_t ring_bytes = ggml_sycl::unified_cache_get_planned_pp_moe_onednn_scratch_bytes(device);",
    "const ggml_sycl_stage_ring_published with_ring = ggml_sycl_stage_ring_published::snapshot(device);",
    "const size_t runtime_with_ring = ggml_sycl_ring_runtime_with_ring_bytes(device);",
    "ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);",
    ROOM_CALL,
    "zone.weight_with_ring = ggml_sycl_ring_plan_weight_room(device, plan); "
    "zone.weight_without_ring = zone.weight_with_ring;",
    "if (ggml_sycl::moe_pp_ring_drop_can_return_room(zone.weight_with_ring, runtime_room)) {",
    "ggml_sycl::zone_arena_ensure_options options = ggml_sycl_load_arena_options();",
    "options.give_back = &give_back;",
    "if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)) {",
    "return ggml_sycl_stage_ring_drop::REFUSED; }",
    "zone.weight_without_ring = ggml_sycl_ring_plan_weight_room(device, plan); zone.held = give_back.held; }",
    "const ggml_sycl::moe_pp_ring_decision decision = ggml_sycl::moe_pp_ring_decide(zone);",
    "g_sycl_ring_drop_marks.record(device, g_sycl_plan_load_effect_txn, decision);",
)
UNPUBLISH_FN = ("static void ggml_sycl_stage_unpublish_ring(int device, uint32_t planning_n_ubatch, "
                "const ggml_sycl::planner_published_terms & published_before) {")
UNPUBLISH_BODY = ("{ if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch(device) != planning_n_ubatch) { "
                  "(void) ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch); } "
                  "ggml_sycl_stage_publish_no_ring(device); published_before.restore(device); }")
KEEP_BLOCK = "if (decision != ggml_sycl::moe_pp_ring_decision::DROP) {"
PUT_BACK_FIRST = "with_ring.put_back(device, planning_n_ubatch);"
RE_ENSURE = ("ggml_sycl::zone_arena_refusal refusal; if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, "
             "&refusal, ggml_sycl_load_arena_options())) {")
KEEP_ORDER = (
    KEEP_BLOCK + " " + PUT_BACK_FIRST + " bool plan_again = false; if (give_back.rebuilt) {",
    RE_ENSURE,
    "return ggml_sycl_stage_ring_drop::REFUSED; }",
    "const size_t weight_kept = ggml_sycl_ring_plan_weight_room(device, plan);",
    "plan_again = ggml_sycl::moe_pp_ring_kept_plan_again(zone.weight_with_ring, weight_kept);",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): its arena was rebuilt with '
    'a "',
    "device, ring_bytes / mb, (give_back.have_runtime - give_back.need_runtime) / mb, zone.weight_with_ring / mb, "
    "zone.weight_without_ring / mb, weight_kept / mb, plan_again ?",
    "} else if (decision == ggml_sycl::moe_pp_ring_decision::KEEP_HELD) {",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB) although the plan made with '
    'it "',
    "the ring displaces %\" PRId64 \" experts' worth of room from the weights\\n\",",
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb, "
    "ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), runtime_room)); } else {",
    "const ggml_sycl_ring_no_weight_room no_room = ggml_sycl_ring_no_weight_room_reason(plan, zone.weight_with_ring); "
    "if (no_room.reason != nullptr) {",
    'GGML_LOG_INFO("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop would give the "',
    '"weights no room, as %s\\n", device, ring_bytes / mb, no_room.reason); } else {',
    'GGML_LOG_INFO("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop would give the "',
    '"weights no room (its RUNTIME zone needs %.1f MB less without it, and the WEIGHT room its plan "',
    "device, ring_bytes / mb, runtime_room / mb, zone.weight_with_ring / mb, zone.weight_without_ring / mb); } }",
    "if (plan_again) { return ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN; }",
    "return ggml_sycl_stage_ring_drop::KEPT; }",
    "const size_t runtime_rebuilt = ggml_sycl_ring_arena_runtime_zone_bytes(device);",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d drops its PP MoE oneDNN scratch ring (%.1f MB): the plan made with it '
    'left %"',
    '"trades prompt room for a decode loss on every token; its RUNTIME zone was rebuilt at %.1f MB, %.1f MB less, "',
    '"and its WEIGHT zone grew from %.1f MB to %.1f MB, and the device is planned once more without it. The "',
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_rebuilt / mb, "
    "(give_back.have_runtime - runtime_rebuilt) / mb, zone.weight_with_ring / mb, zone.weight_without_ring / mb);",
    "return ggml_sycl_stage_ring_drop::DROPPED; }",
)
KEPT_EXIT = "return ggml_sycl_stage_ring_drop::KEPT;"
KEPT_AGAIN_EXIT = "return ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN;"
DROPPED_EXIT = "return ggml_sycl_stage_ring_drop::DROPPED;"
REFUSED_EXIT = "return ggml_sycl_stage_ring_drop::REFUSED;"
ENSURE_CALL = "unified_cache_ensure_planned_arena_zones("
ENSURE_IF = "if (!ggml_sycl::unified_cache_ensure_planned_arena_zones("
PUBLISHED_STRUCT = "struct ggml_sycl_stage_ring_published {"
PUBLISHED_SNAPSHOT = ("{ ggml_sycl_stage_ring_published published; published.terms = "
                      "ggml_sycl::load_published_terms::snapshot(device); published.inventory_weight_slot_bytes = "
                      "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes; published.inventory_activation_slot_bytes = "
                      "g_tensor_inventory_pp_moe_onednn_activation_slot_bytes; published.inventory_output_slot_bytes = "
                      "g_tensor_inventory_pp_moe_onednn_output_slot_bytes; published.inventory_ring_depth = "
                      "g_tensor_inventory_pp_moe_onednn_ring_depth; return published; }")
PUT_BACK_FN = "void put_back(int device, uint32_t planning_n_ubatch) const {"
PUT_BACK_BODY = ("{ if (terms.ring_n_ubatch != planning_n_ubatch) { (void) "
                 "ggml_sycl_publish_load_ubatch_scratch(device, terms.ring_n_ubatch); } "
                 "ggml_sycl_put_back_ring_terms(device, terms); g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = "
                 "inventory_weight_slot_bytes; g_tensor_inventory_pp_moe_onednn_activation_slot_bytes = "
                 "inventory_activation_slot_bytes; g_tensor_inventory_pp_moe_onednn_output_slot_bytes = "
                 "inventory_output_slot_bytes; g_tensor_inventory_pp_moe_onednn_ring_depth = inventory_ring_depth; }")
TERMS_FN = ("static void ggml_sycl_put_back_ring_terms(int device, const ggml_sycl::load_published_terms & with_ring) "
            "{")
TERMS_BODY = ("{ with_ring.restore(device, ggml_sycl::LOAD_PUBLISHED_GROUP_RING | "
              "ggml_sycl::LOAD_PUBLISHED_GROUP_GRAPH | ggml_sycl::LOAD_PUBLISHED_GROUP_NONFA | "
              "ggml_sycl::LOAD_PUBLISHED_GROUP_CONTROL | ggml_sycl::LOAD_PUBLISHED_GROUP_SCATTER | "
              "ggml_sycl::LOAD_PUBLISHED_GROUP_MOE_MMVQ); }")
TERMS_CALL = "ggml_sycl_put_back_ring_terms("
PLANNING = "const uint32_t planning_n_ubatch = ggml_sycl::placement_load_n_ubatch(g_placement_kv_info.n_ubatch);"
NO_RING_FN = "static void ggml_sycl_publish_no_pp_moe_onednn_ring(int device) {"
NO_RING_BODY = "{ ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(device, 0, 0, 0, 0); }"
GIVE_BACK_WRITE = ".give_back ="
PLAN_FN = ("static bool compute_and_store_plan_for_inventory(ggml_backend_sycl_context * ctx, size_t vram_budget, "
           "int budget_pct) {")
MULTI_LAMBDA = ("auto plan_multi_device = [&]() { const std::vector<ggml_sycl::device_budget> budgets = "
                "ggml_sycl_plan_device_budgets(budget_pct, cache_device_ready); return "
                "ggml_sycl::compute_multi_device_plan(budgets,")
MULTI_SITE = ("const ggml_sycl::planner_published_terms terms_before_plan = "
              "ggml_sycl::planner_published_terms::snapshot(ctx->device); plan_candidate = plan_multi_device(); "
              "switch (ggml_sycl_stage_drop_displacing_ring(ctx->device, plan_candidate, terms_before_plan)) { case "
              "ggml_sycl_stage_ring_drop::KEPT: break; case ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN: case "
              "ggml_sycl_stage_ring_drop::DROPPED: plan_candidate = plan_multi_device(); break; case "
              "ggml_sycl_stage_ring_drop::REFUSED: return false; }")
GROW = "if (plan_candidate.explicit_ubatch.action == ggml_sycl::pp_moe_onednn_explicit_ubatch_action::GROW_ZONE) {"
SINGLE_SITE = ("switch (ggml_sycl_stage_drop_displacing_ring(ctx->device, plan_candidate, published_before)) { case "
               "ggml_sycl_stage_ring_drop::KEPT: break; case ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN: "
               "plan_candidate = plan_single_device(candidate_with_room); break; case "
               "ggml_sycl_stage_ring_drop::DROPPED: plan_candidate = plan_single_device(false); break; case "
               "ggml_sycl_stage_ring_drop::REFUSED: return false; } have_plan = true;")
WITH_ROOM_FIRST = "plan_candidate = plan_single_device(true); bool candidate_with_room = true;"
WITH_ROOM_FALLBACK = "candidate_with_room = false; plan_candidate = plan_single_device(false); GGML_LOG_WARN("
EARLY_SIG = "static int ggml_sycl_compute_placement_plan_early_impl("
LATE_SIG = "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl("
PLAN_CALL = "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)"
READER_CALL = "ggml_sycl_stage_ring_dropped_by_this_load("
STAGE_CALL = "ggml_sycl_stage_drop_displacing_ring("
STICKY = ("if (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) && "
          "ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx->device) != 0) { "
          "ggml_sycl_stage_publish_no_ring(ctx->device);")
ENTRY_ENSURE = "!ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)) {"
READER_FN = "static bool ggml_sycl_stage_ring_dropped_by_this_load(int device) {"
READER_BODY = "{ return g_sycl_ring_drop_marks.dropped(device, g_sycl_plan_load_effect_txn); }"
TXN = "g_sycl_ring_drop_marks"
TXN_DECL = "static ggml_sycl::moe_pp_ring_drop_marks g_sycl_ring_drop_marks;"
TXN_WRITE = "g_sycl_ring_drop_marks.record(device, g_sycl_plan_load_effect_txn, decision);"
TXN_MARK = "g_sycl_ring_drop_marks.mark("
HELD_GUESSES = ("a live model", "freed context")
DRY_FN = ("static bool ggml_sycl_plan_dry_run_drop_displacing_ring(int device, "
          "const std::vector<ggml_sycl::placement_tensor_info> & detail, const ggml_sycl::placement_plan & plan, "
          "const ggml_sycl::planner_published_terms & terms_before_plan) {")
DRY_ORDER = (
    "const ggml_sycl::moe_pp_ring_resident_check check = ggml_sycl_plan_ring_resident_check(device, detail, plan);",
    "if (!check.displaces_experts()) { return false; }",
    "const ggml_sycl::load_published_terms with_ring = ggml_sycl::load_published_terms::snapshot(device);",
    "const size_t runtime_with_ring = ggml_sycl_ring_runtime_with_ring_bytes(device);",
    "ggml_sycl_publish_no_pp_moe_onednn_ring(device);",
    "terms_before_plan.restore(device);",
    ROOM_CALL,
    "const size_t weight_room = ggml_sycl_ring_plan_weight_room(device, plan);",
    "const bool can_return = ggml_sycl::moe_pp_ring_drop_can_return_room(weight_room, runtime_room);",
    "if (ggml_sycl::moe_pp_ring_decide(ggml_sycl::moe_pp_ring_dry_run_answer(can_return, runtime_room)) != "
    "ggml_sycl::moe_pp_ring_decision::DROP) { ggml_sycl_put_back_ring_terms(device, with_ring);",
    "const ggml_sycl_ring_no_weight_room no_room = ggml_sycl_ring_no_weight_room_reason(plan, weight_room); "
    "if (no_room.reason != nullptr) {",
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop returns no room '
    'it "',
    '"sees, as %s%s\\n", device, ring_bytes / mb, no_room.reason, no_room.load_reads_none ? "; a load reads none '
    'there either" : ". A load\'s entry ensure builds the arena first, and its own drop " "decision, on that arena, '
    'may differ"); } else {',
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d keeps its PP MoE oneDNN scratch ring (%.1f MB), as a load would: its '
    'RUNTIME "',
    "device, ring_bytes / mb, runtime_with_ring / mb); }",
    "return false; }",
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d drops its PP MoE oneDNN scratch ring (%.1f MB), as a load would whose '
    'arena nothing "',
    '"would grow another zone or the give-back\'s rebuild would leave the WEIGHT zone no larger, which the dry run "',
    '"cannot see\\n",',
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb);",
    "return true; }",
)
DRY_FORBIDDEN = ("ensure_planned_arena_zones", TXN, "give_back", "ggml_sycl_stage_")
DRY_PLANS_SIG = "static std::vector<ggml_sycl::placement_plan> ggml_sycl_plan_dry_run_plans("
DRY_SITES = (
    "const ggml_sycl::planner_published_terms terms_before_plan = ggml_sycl::planner_published_terms::snapshot(i); "
    "plans.push_back(plan_multi()); if (ggml_sycl_plan_dry_run_drop_displacing_ring(i, detail, plans.back(), "
    "terms_before_plan)) { plans.back() = plan_multi(); }",
    "const ggml_sycl::planner_published_terms terms_before_plan = ggml_sycl::planner_published_terms::snapshot(i); "
    "plans.push_back(plan_single()); if (ggml_sycl_plan_dry_run_drop_displacing_ring(i, detail, plans.back(), "
    "terms_before_plan)) { plans.back() = plan_single(); }",
)
DRY_CALL = "ggml_sycl_plan_dry_run_drop_displacing_ring("
ENSURE_FN = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
             "const zone_arena_ensure_options & options) {")
GIVE_BACK_KEEP = "if (!surplus_asked && (zones_sufficient || options.give_back != nullptr)) {"
ENSURE_ORDER = (
    "bool giving_back = false; if (options.give_back != nullptr) { *options.give_back = {}; "
    "options.give_back->need_runtime = runtime_zone; options.give_back->have_runtime = arena_active() ? "
    "zone_capacity(vram_zone_id::RUNTIME) : 0; } if (arena_active()) {",
    "const bool surplus_asked = options.give_back != nullptr && zone_arena_runtime_surplus(arena_zones, "
    "planned_zones);",
    GIVE_BACK_KEEP,
    "zone_arena_holders holders = arena_read_holders(live_zone_used);",
    "if (surplus_asked && !zone_arena_rebuild_allowed(holders)) { options.give_back->held = true; "
    "keep_arena_as_built(); onednn_zone_plan_keep_and_store(live_plan); return true; }",
    "if (surplus_asked) { const zone_arena_sizes sizes = zone_arena_give_back_sizes(arena_zones, planned_zones); "
    "scratch_zone = sizes.scratch; onednn_zone = sizes.onednn; runtime_zone = sizes.runtime; giving_back = true; }",
    "if (scratch_kept < scratch_zone && !zone_arena_rebuild_allowed(holders)) {",
    '"[VRAM-ARENA] Rebuilding unused early arena for planned zones: "',
    "if (!arena_reserve(queue_, budget_, max_alloc, safe_max_alloc, scratch_zone, onednn_zone, runtime_zone,",
    "compute_arena_bind(); if (giving_back) { keep_arena_as_built(); onednn_zone_plan_keep_and_store(live_plan); "
    "options.give_back->rebuilt = true; return true; }",
    "onednn_graph_scratch_direct_cap_plan_snapshot_bytes_.store(available_budget(), std::memory_order_release);",
)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def blank_comments(text: str) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def brace_text(text: str, open_at: int, pin: str) -> str:
    """The balanced block opening at `open_at`, skipping string and character literals."""
    depth = 0
    k = open_at
    n = len(text)
    while k < n:
        c = text[k]
        if c in "\"'":
            k += 1
            while k < n and text[k] != c:
                k += 2 if text[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : k + 1]
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: unbalanced block")


def need_once(hay: str, needle: str, pin: str, what: str) -> int:
    n = squash(needle)
    if hay.count(n) != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` appears {hay.count(n)} times, not once")
    return hay.find(n)


def need_in(hay: str, needle: str, pin: str, what: str) -> None:
    if squash(needle) not in hay:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` is missing")


def need_ordered(hay: str, parts: tuple, pin: str, what: str) -> None:
    pos = 0
    for p in parts:
        k = hay.find(squash(p), pos)
        if k == -1:
            raise ContractError(f"FAIL [pin {pin}]: {what}: `{p}` is missing or out of order")
        pos = k + len(squash(p))


def span(hay: str, head: str, pin: str, what: str) -> tuple[int, int, int]:
    """Where the one function or struct whose head is `head` starts, opens its body and ends. A head that stops before
    its opening brace (a long parameter list) runs to the first ") {" after it."""
    start = need_once(hay, head, pin, what)
    h = squash(head)
    open_at = start + len(h) - 1 if h.endswith("{") else hay.find(") {", start) + 2
    return start, open_at, open_at + len(brace_text(hay, open_at, pin))


def fn(hay: str, head: str, pin: str, what: str) -> str:
    """The body of the one function or struct whose head is `head`, from its opening brace."""
    _, open_at, end = span(hay, head, pin, what)
    return hay[open_at:end]


def only_in(hay: str, needle: str, holders: dict, pin: str, what: str) -> None:
    """`needle` appears in each holder (a name for the message -> the head of a function or struct, which counts as
    part of it) and nowhere else in `hay`."""
    n = squash(needle)
    spans = []
    for name, head in holders.items():
        start, _, end = span(hay, head, pin, what)
        if n not in hay[start:end]:
            raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` is missing from {name}")
        spans.append((start, end))
    pos = 0
    for start, end in sorted(spans):
        if n in hay[pos:start]:
            break
        pos = max(pos, end)
    else:
        if n not in hay[pos:]:
            return
    raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` appears outside {', '.join(holders)}")


def blocks(hay: str, head: str, pin: str) -> list[tuple[int, int]]:
    """Each block whose `if` condition starts with `head`: where the head starts and where its block ends."""
    out = []
    h = squash(head)
    at = hay.find(h)
    while at != -1:
        open_at = hay.find(") {", at) + 2
        out.append((at, open_at + len(brace_text(hay, open_at, pin))))
        at = hay.find(h, at + len(h))
    return out


class Sources:
    def __init__(self, raws: dict[str, str]):
        self.sq = {name: squash(blank_comments(text)) for name, text in raws.items()}


def check_decision(s: Sources) -> None:
    be = s.sq["backend"]
    # U2
    if "mmid_direct_route_runs_type(ggml_type" in be or "mmid_direct_route_runs_type(enum ggml_type" in be:
        raise ContractError("FAIL [pin U2]: the backend must not define its own direct-route type set")
    need_in(be, DIRECT_ADMISSION, "U2", "the direct route admits by the tables' type set")
    # W1
    if fn(be, WEIGHTS_FN, "W1", "the backend's ring weight list") != squash(WEIGHTS_BODY):
        raise ContractError("FAIL [pin W1]: the ring weight list must take the ring sizing's filter, ne[2] experts, "
                            "their bytes and the direct route's type set")
    # D1
    stage_and_dry = {"the stage": STAGE_FN, "the dry run": DRY_FN}
    if fn(be, DECIDE_FN, "D1", "the resident check") != squash(DECIDE_BODY):
        raise ContractError("FAIL [pin D1]: the resident check must answer nothing without a ring and ask the rule "
                            "once")
    only_in(be, RULE_CALL, {"the resident check": DECIDE_FN}, "D1", "the backend asks the rule through its check")
    only_in(be, DECIDE_CALL, {"the resident check": DECIDE_FN, **stage_and_dry}, "D1",
            "the stage and the dry run are the resident check's callers")
    only_in(be, RING_DECIDE_CALL, stage_and_dry, "D1", "the stage and the dry run are the decision's callers")
    for name, head in stage_and_dry.items():
        need_once(fn(be, head, "D1", name), RING_DECIDE_CALL, "D1", f"{name} decides once")
    if fn(be, ZONE_FN, "D1", "the zone figure") != squash(ZONE_BODY):
        raise ContractError("FAIL [pin D1]: the zone figure must be the reserve's rounding over the floor and the "
                            "planned requirement, SIZE_MAX when unanswerable")
    if fn(be, WITH_RING_FN, "D1", "the RUNTIME zone with the ring") != squash(WITH_RING_BODY):
        raise ContractError("FAIL [pin D1]: the RUNTIME zone with the ring must be capped by the arena's RUNTIME zone, "
                            "and read as the floor when unanswerable with no arena, through the consumer's rule")
    only_in(be, WITH_RING_CALL, {"its definition": WITH_RING_FN, **stage_and_dry}, "D1",
            "the stage and the dry run read the RUNTIME zone with the ring through the one cap")
    only_in(be, ZONE_CALL, {"its definition": ZONE_FN, "the cap": WITH_RING_FN, **stage_and_dry}, "D1",
            "the zone figure is read by the cap, the stage and the dry run")
    only_in(be, DRY_ANSWER_CALL, {"the dry run": DRY_FN}, "D1", "only the dry run answers from the RUNTIME room alone")
    if fn(be, PLANNER_ZONE_FN, "D1", "whether the plan's planner reads a WEIGHT zone") != squash(PLANNER_ZONE_BODY):
        raise ContractError("FAIL [pin D1]: whether the plan's planner reads a WEIGHT zone is the KV rule on the "
                            "plan's "
                            "mode, so a multi-device plan in GLOBAL cache mode reads none")
    only_in(be, PLANNER_ZONE_CALL,
            {"its definition": PLANNER_ZONE_FN, "the WEIGHT room": WEIGHT_ROOM_FN, "the no-room reason": NO_ROOM_FN},
            "D1", "the planner rule is asked by the WEIGHT-room helper and the no-room reason")
    if fn(be, WEIGHT_ROOM_FN, "D1", "the WEIGHT room") != squash(WEIGHT_ROOM_BODY):
        raise ContractError("FAIL [pin D1]: the WEIGHT room is the planner's own figure at the budget the plan packed "
                            "the device to, 0 where it reads no zone or the device has no active arena")
    only_in(be, WEIGHT_ROOM_CALL, {"its definition": WEIGHT_ROOM_FN, **stage_and_dry}, "D1",
            "the WEIGHT room is read by the stage and the dry run")
    if fn(be, NO_ROOM_STRUCT, "D1", "the no-room reason's answer") != squash(NO_ROOM_STRUCT_BODY):
        raise ContractError("FAIL [pin D1]: the no-room reason answers a reason, null by default, and whether a load "
                            "reads no WEIGHT room either")
    if fn(be, NO_ROOM_FN, "D1", "the no-room reason") != squash(NO_ROOM_BODY):
        raise ContractError("FAIL [pin D1]: the no-room reason names none where the WEIGHT room is the planner's own "
                            "figure, the GLOBAL multi-device plan (which a load reads alike) where the planner "
                            "reads no zone, and otherwise no active arena or no budget")
    only_in(be, NO_ROOM_CALL, {"its definition": NO_ROOM_FN, **stage_and_dry}, "D1",
            "the stage and the dry run name why they read no WEIGHT room through the one reason")
    for name, head in stage_and_dry.items():
        need_once(fn(be, head, "D1", name), NO_ROOM_CALL, "D1", f"{name} asks the no-room reason once")
    only_in(be, CAN_RETURN_CALL, stage_and_dry, "D1",
            "the stage and the dry run ask whether a drop can return room through the one predicate")


def check_stage(s: Sources) -> None:
    be = s.sq["backend"]
    stage = fn(be, STAGE_FN, "S5", "the stage helper")
    plan = fn(be, PLAN_FN, "S5", "the planner")
    # S5
    for head, which in ((EARLY_SIG, "the early pass"), (LATE_SIG, "the late pass")):
        need_in(fn(be, head, "S5", which), PLAN_CALL, "S5", f"{which} plans through the planner that calls the stage")
    for word in (READER_CALL, ".dropped("):
        if word in stage:
            raise ContractError(f"FAIL [pin S5]: the stage decides afresh on every pass, but reads a mark: `{word}`")
    # S1
    need_ordered(stage, STAGE_ORDER, "S1", "the stage helper's order")
    need_once(stage, PLANNING, "S1", "the planning shape the dense scratch goes back to")
    if fn(be, UNPUBLISH_FN, "S1", "the unpublish") != squash(UNPUBLISH_BODY):
        raise ContractError("FAIL [pin S1]: the unpublish must put the dense scratch at the planning shape, publish no "
                            "ring and restore the planner terms from before the plan")
    if fn(be, NO_RING_FN, "S1", "the no-ring publish") != squash(NO_RING_BODY):
        raise ContractError("FAIL [pin S1]: publishing no ring must zero every slot and the depth")
    for word in ("ggml_sycl_stage_publish_no_ring(", "published_before.restore(", "plan_single_device(",
                 "plan_multi_device("):
        if word in stage:
            raise ContractError(f"FAIL [pin S1]: the stage helper writes through the unpublish alone and plans "
                                f"nothing, but names `{word}`")
    for word in ("vram_zone_id::WEIGHT", "arena_active("):
        if word in stage:
            raise ContractError(f"FAIL [pin S1]: the stage reads the WEIGHT room through the planner's figure alone, "
                                f"but names `{word}`")
    for word in CACHE_WORDS:
        if word in stage:
            raise ContractError(f"FAIL [pin S1]: the stage reads the cache only through helpers that create none, but "
                                f"names `{word}`")
    if fn(be, ARENA_RUNTIME_FN, "S1", "the arena's RUNTIME zone") != squash(ARENA_RUNTIME_BODY):
        raise ContractError("FAIL [pin S1]: the arena's RUNTIME zone is read from a cache that exists, 0 with no "
                            "active arena")
    only_in(be, ARENA_RUNTIME_CALL, {"its definition": ARENA_RUNTIME_FN, "the stage": STAGE_FN}, "S1",
            "the stage reads the arena's RUNTIME zone after a drop")
    # K1
    need_ordered(stage, KEEP_ORDER, "K1", "a kept ring is put back before either kept line")
    unpublish = stage.find("ggml_sycl_stage_unpublish_ring(")
    if stage.find(squash(KEPT_EXIT)) > unpublish:
        raise ContractError("FAIL [pin K1]: the stage's first KEPT exit must come before anything is written")
    keep_at = need_once(stage, KEEP_BLOCK, "K1", "the keep block")
    keep_open = keep_at + len(squash(KEEP_BLOCK)) - 1
    keep = brace_text(stage, keep_open, "K1")
    if not keep.startswith("{ " + squash(PUT_BACK_FIRST)):
        raise ContractError("FAIL [pin K1]: the keep block puts the ring back before anything else")
    if squash(DROPPED_EXIT) in keep:
        raise ContractError("FAIL [pin K1]: the keep block holds no drop exit")
    outside_keep = stage[:keep_at] + "\0" + stage[keep_open + len(keep):]
    if "put_back(" in outside_keep:
        raise ContractError("FAIL [pin K1]: the stage puts a ring back only in the keep block")
    for word in (KEPT_EXIT, KEPT_AGAIN_EXIT):
        if squash(word) in stage[unpublish:keep_at] + "\0" + stage[keep_open + len(keep):]:
            raise ContractError(f"FAIL [pin K1]: after the write, `{word}` is an exit of the keep block alone")
    refusing = blocks(stage, ENSURE_IF, "K1")
    if stage.count(ENSURE_CALL) != len(refusing):
        raise ContractError("FAIL [pin K1]: every ensure of the zones in the stage is a refusal's condition")
    rest = stage
    for start, end in reversed(refusing):
        if squash(REFUSED_EXIT) not in stage[start:end]:
            raise ContractError("FAIL [pin K1]: a failed ensure of the zones refuses the load")
        rest = rest[:start] + rest[end:]
    if squash(REFUSED_EXIT) in rest:
        raise ContractError("FAIL [pin K1]: the stage refuses only where an ensure of the zones fails")
    for guess in HELD_GUESSES:
        if guess in stage:
            raise ContractError(f"FAIL [pin K1]: the held WARN names what is known, not a guessed holder: `{guess}`")
    published = fn(be, PUBLISHED_STRUCT, "K1", "the snapshot of what a drop writes")
    need_once(published, PUBLISHED_SNAPSHOT, "K1", "the snapshot takes the load's terms and the inventory's ring")
    if fn(published, PUT_BACK_FN, "K1", "the put-back") != squash(PUT_BACK_BODY):
        raise ContractError("FAIL [pin K1]: the put-back must plan the dense scratch at the ring's micro-batch "
                            "where it moved, restore the ring terms and the inventory's ring")
    if fn(be, TERMS_FN, "K1", "the put-back of the ring terms") != squash(TERMS_BODY):
        raise ContractError("FAIL [pin K1]: the put-back of the ring terms must restore the ring's group and every "
                            "planner group the unpublish writes")
    only_in(be, TERMS_CALL,
            {"its definition": TERMS_FN, "the stage's put-back": PUBLISHED_STRUCT, "the dry run": DRY_FN}, "K1",
            "the ring terms' put-back is called by the stage's put-back and the dry run's keep")
    # S2
    only_in(be, GIVE_BACK_WRITE, {"the stage": STAGE_FN}, "S2", "the stage is the one writer of the give-back option")
    need_once(stage, "options.give_back = &give_back;", "S2", "the stage asks for the give-back with its own record")
    # S3
    only_in(be, STAGE_CALL, {"its definition": STAGE_FN, "the planner": PLAN_FN}, "S3",
            "the planner is the stage's one caller")
    need_once(plan, MULTI_LAMBDA, "S3", "the multi-device plan reads its budgets at each plan")
    need_once(plan, MULTI_SITE, "S3", "the multi-device branch drops, plans once more, or refuses")
    need_once(plan, SINGLE_SITE, "S3", "the single-device branch drops, plans once more without the room, or refuses")
    need_ordered(plan, (WITH_ROOM_FIRST, GROW, WITH_ROOM_FALLBACK, SINGLE_SITE), "S3",
                 "the single-device drop follows the micro-batch grow leg, and a kept ring planned once more takes the "
                 "room the candidate was made with")
    # S4
    if fn(be, READER_FN, "S4", "the drop's reader") != squash(READER_BODY):
        raise ContractError("FAIL [pin S4]: the reader must answer only for this thread's staged load")
    need_once(be, TXN_DECL, "S4", "the drop's record")
    only_in(be.replace(squash(TXN_DECL), "", 1), TXN, {"the reader": READER_FN, "the stage": STAGE_FN}, "S4",
            "the drop's record is read by the reader and written by the stage alone")
    need_once(stage, TXN_WRITE, "S4", "the stage records its decision for the load")
    if TXN_MARK in be:
        raise ContractError("FAIL [pin S4]: the backend records the decision and marks nothing of its own")
    need_ordered(plan, (STICKY, ENTRY_ENSURE), "S4", "a dropped ring is published as none before the entry ensure")
    # S5, after S4 has pinned the reader's one use
    only_in(be, READER_CALL, {"its definition": READER_FN, "the planner": PLAN_FN}, "S5",
            "the drop's reader is asked by the planner alone")
    need_once(plan, READER_CALL, "S5", "the planner asks the reader only to publish no ring before its entry ensure")


def check_dry_run(s: Sources) -> None:
    be = s.sq["backend"]
    # Y1
    dry = fn(be, DRY_FN, "Y1", "the dry run's helper")
    need_ordered(dry, DRY_ORDER, "Y1", "the dry run's helper decides, publishes no ring and says so")
    for word in DRY_FORBIDDEN:
        if word in dry:
            raise ContractError(f"FAIL [pin Y1]: the dry run sizes no zone and records no load, but names `{word}`")
    plans = fn(be, DRY_PLANS_SIG, "Y1", "the dry run's plans")
    for site in DRY_SITES:
        need_once(plans, site, "Y1", "each dry-run plan is made again without a dropped ring")
    only_in(be, DRY_CALL, {"its definition": DRY_FN, "the dry run's plans": DRY_PLANS_SIG}, "Y1",
            "the dry run's plans are its helper's one caller")


def check_zones(s: Sources) -> None:
    uc = s.sq["cache"]
    # Z1
    ensure = fn(uc, ENSURE_FN, "Z1", "the zone ensure")
    need_ordered(ensure, ENSURE_ORDER, "Z1", "the ensure's give-back")
    only_in(uc, "give_back", {"the ensure": ENSURE_FN}, "Z1", "the cache touches the give-back record in its ensure "
            "alone")


def check(s: Sources) -> None:
    check_decision(s)
    check_stage(s)
    check_dry_run(s)
    check_zones(s)


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def mutate(file: str, edits: list) -> dict[str, str]:
        out = dict(raws)
        text = out[file]
        for old, new in edits:
            if text.count(old) != 1:
                raise MutantNotApplied(f"anchor matches {text.count(old)} times in {file}: {old[:70]!r}")
            text = text.replace(old, new, 1)
        out[file] = text
        return out

    def expect_fail(name: str, pin: str, file: str, edits: list) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, edits)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(Sources(mutated))
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught by the wrong pin ({e})")
                print(f"  mutant {name}: caught by the wrong pin, not {pin}: {e}")
            else:
                print(f"  mutant {name}: caught by {pin}")
            return
        failures.append(f"{name}: SURVIVED")
        print(f"  mutant {name}: SURVIVED (pin {pin})")

    B, U = "backend", "cache"
    # U2, W1, D1: the direct route's types, the weight list and the one decision.
    expect_fail("route-admission-own-type-test", "U2", B,
                [("moe_mmid_direct_route_runs_type(src0->type) && direct_layout == GGML_LAYOUT_AOS",
                  "(src0->type == GGML_TYPE_NVFP4) && direct_layout == GGML_LAYOUT_AOS")])
    expect_fail("backend-own-direct-set", "U2", B,
                [("static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights(",
                  "static bool ggml_sycl_moe_mmid_direct_route_runs_type(ggml_type type) {\n"
                  "    return type == GGML_TYPE_NVFP4;\n}\n\n"
                  "static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights(")])
    expect_fail("weights-without-shape", "W1", B,
                [("MOE_TENSOR_UNKNOWN || !info.has_shape()) {", "MOE_TENSOR_UNKNOWN) {")])
    expect_fail("weights-one-expert", "W1", B,
                [("weight.n_experts                 = info.ne[2] > 0 ? info.ne[2] : 1;",
                  "weight.n_experts                 = 1;")])
    expect_fail("weights-without-bytes", "W1", B, [("        weight.bytes                     = info.size;\n", "")])
    expect_fail("weights-route-for-any-type", "W1", B,
                [("weight.other_device_prompt_route = moe_mmid_direct_route_runs_type(info.type);",
                  "weight.other_device_prompt_route = moe_mmid_direct_route_runs_type(info.type) || true;")])
    expect_fail("decision-ignores-depth", "D1", B,
                [("    if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(device) == 0) {\n"
                  "        return {};\n    }\n    const std::vector<ggml_sycl::moe_pp_ring_weight> weights",
                  "    const std::vector<ggml_sycl::moe_pp_ring_weight> weights")])
    expect_fail("dry-run-own-rule", "D1", B,
                [("check = ggml_sycl_plan_ring_resident_check(device, detail, plan);",
                  "check = ggml_sycl::moe_pp_ring_check_resident_only(plan, nullptr, 0);")])
    expect_fail("planner-zone-ignores-the-mode", "D1", B,
                [("return ggml_sycl::kv_reads_device_arena(plan.multi_device, "
                  "ggml_sycl::unified_cache_mode_is_global());",
                  "return true;")])
    expect_fail("weight-room-ignores-the-planner", "D1", B,
                [("    if (!ggml_sycl_ring_planner_reads_zone(plan) || cache == nullptr || !cache->arena_active()) {\n",
                  "    if (cache == nullptr || !cache->arena_active()) {\n")])
    expect_fail("weight-room-reads-a-zone-with-no-arena", "D1", B,
                [("    if (!ggml_sycl_ring_planner_reads_zone(plan) || cache == nullptr || !cache->arena_active()) {\n",
                  "    if (!ggml_sycl_ring_planner_reads_zone(plan) || cache == nullptr) {\n")])
    expect_fail("weight-room-one-budget-for-every-device", "D1", B,
                [("                budget = plan.per_device_vram_budgets[i];\n",
                  "                budget = plan.vram_budget;\n")])
    expect_fail("weight-room-uncapped-by-the-budget", "D1", B,
                [("    return ggml_sycl::unified_cache_kv_weight_capacity(device, budget, plan.multi_device);\n",
                  "    return cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT);\n")])
    expect_fail("dry-run-own-predicate", "D1", B,
                [("ggml_sycl::moe_pp_ring_drop_can_return_room(weight_room, runtime_room);",
                  "runtime_room > 0;")])
    expect_fail("dry-run-answers-past-the-predicate", "Y1", B,
                [("ggml_sycl::moe_pp_ring_dry_run_answer(can_return, runtime_room)",
                  "ggml_sycl::moe_pp_ring_dry_run_answer(true, runtime_room)")])
    # D1: the no-room reason, one helper for the stage's and the dry run's keep lines.
    expect_fail("no-room-reasons-swapped", "D1", B,
                [("    if (!ggml_sycl_ring_planner_reads_zone(plan)) {\n",
                  "    if (ggml_sycl_ring_planner_reads_zone(plan)) {\n")])
    expect_fail("no-room-reason-for-the-planners-figure", "D1", B,
                [("    if (weight_room > 0) {\n        return {};\n    }\n", "")])
    expect_fail("no-room-global-reads-as-no-arena", "D1", B,
                [("budget and reads no WEIGHT zone\",\n                 true };\n",
                  "budget and reads no WEIGHT zone\",\n                 false };\n")])
    expect_fail("no-room-reason-defaults-to-a-load-reading-none", "D1", B,
                [("    bool         load_reads_none = false;\n", "    bool         load_reads_none = true;\n")])
    expect_fail("dry-run-asks-the-planner-itself", "D1", B,
                [("                no_room.load_reads_none ? \"; a load reads none there either\" :\n",
                  "                !ggml_sycl_ring_planner_reads_zone(plan) ? \"; a load reads none there "
                  "either\" :\n")])
    expect_fail("dry-run-own-no-room-branch", "D1", B,
                [("        const ggml_sycl_ring_no_weight_room no_room = ggml_sycl_ring_no_weight_room_reason(plan, "
                  "weight_room);\n",
                  "        const ggml_sycl_ring_no_weight_room no_room = { weight_room == 0 ? \"no WEIGHT room\" : "
                  "nullptr, false };\n")])
    # K1: the stage's no-room keep line.
    expect_fail("stage-no-room-prints-figures", "K1", B,
                [("ggml_sycl_ring_no_weight_room_reason(plan, zone.weight_with_ring);\n"
                  "            if (no_room.reason != nullptr) {\n",
                  "ggml_sycl_ring_no_weight_room_reason(plan, zone.weight_with_ring);\n"
                  "            if (false) {\n")])
    expect_fail("stage-no-room-reads-the-room-without-the-ring", "K1", B,
                [("ggml_sycl_ring_no_weight_room_reason(plan, zone.weight_with_ring);",
                  "ggml_sycl_ring_no_weight_room_reason(plan, zone.weight_without_ring);")])
    expect_fail("stage-no-room-line-drops-the-reason", "K1", B,
                [("\"weights no room, as %s\\n\",\n                    device, ring_bytes / mb, no_room.reason);\n",
                  "\"weights no room\\n\",\n                    device, ring_bytes / mb);\n")])
    # Y1: the dry run's no-room keep line.
    expect_fail("dry-run-no-room-prints-the-runtime-line", "Y1", B,
                [("ggml_sycl_ring_no_weight_room_reason(plan, weight_room);\n"
                  "        if (no_room.reason != nullptr) {\n",
                  "ggml_sycl_ring_no_weight_room_reason(plan, weight_room);\n"
                  "        if (false) {\n")])
    expect_fail("dry-run-no-arena-says-a-load-reads-none", "Y1", B,
                [("                no_room.load_reads_none ? \"; a load reads none there either\" :\n",
                  "                true ? \"; a load reads none there either\" :\n")])
    expect_fail("dry-run-global-drops-the-load-answer", "Y1", B,
                [("no_room.load_reads_none ? \"; a load reads none there either\" :",
                  "no_room.load_reads_none ? \"\" :")])
    expect_fail("zone-figure-without-floor", "D1", B,
                [("!ggml_sycl::zone_runtime_reserved_bytes(ggml_sycl::unified_cache_runtime_zone_floor_bytes(), "
                  "requirement,",
                  "!ggml_sycl::zone_runtime_reserved_bytes(0, requirement,")])
    expect_fail("dry-run-decides-on-the-plan-alone", "D1", B,
                [("    if (ggml_sycl::moe_pp_ring_decide(ggml_sycl::moe_pp_ring_dry_run_answer(can_return, "
                  "runtime_room)) "
                  "!=\n"
                  "        ggml_sycl::moe_pp_ring_decision::DROP) {\n",
                  "    if (!check.displaces_experts()) {\n")])
    expect_fail("dry-run-uncapped", "D1", B,
                [("    const size_t                          runtime_with_ring = "
                  "ggml_sycl_ring_runtime_with_ring_bytes(device);\n",
                  "    const size_t                          runtime_with_ring = "
                  "ggml_sycl_planned_runtime_zone_bytes(device);\n")])
    expect_fail("stage-uncapped", "D1", B,
                [("    const size_t                         runtime_with_ring = "
                  "ggml_sycl_ring_runtime_with_ring_bytes(device);\n",
                  "    const size_t                         runtime_with_ring = "
                  "ggml_sycl_planned_runtime_zone_bytes(device);\n")])
    expect_fail("helper-ignores-the-arena", "D1", B,
                [("    return ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), "
                  "arena,",
                  "    return ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), "
                  "false,")])
    # S1-S5, K1: the stage.
    expect_fail("terms-not-restored", "S1", B, [("    published_before.restore(device);\n", "")])
    expect_fail("dense-left-at-micro-batch", "S1", B,
                [("        (void) ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch);\n", "")])
    expect_fail("ensure-without-give-back", "S1", B,
                [("        options.give_back                            = &give_back;\n", "")])
    expect_fail("refused-ensure-drops", "S1", B,
                [("            return ggml_sycl_stage_ring_drop::REFUSED;\n        }\n"
                  "        zone.weight_without_ring",
                  "        }\n        zone.weight_without_ring")])
    expect_fail("stage-own-predicate", "D1", B,
                [("    if (ggml_sycl::moe_pp_ring_drop_can_return_room(zone.weight_with_ring, runtime_room)) {\n",
                  "    if (runtime_room > 0) {\n")])
    expect_fail("arena-asked-where-no-weight-room-is-read", "S1", B,
                [("    if (ggml_sycl::moe_pp_ring_drop_can_return_room(zone.weight_with_ring, runtime_room)) {\n",
                  "    if (ggml_sycl::moe_pp_ring_drop_can_return_room(SIZE_MAX, runtime_room)) {\n")])
    expect_fail("creating-getter-in-the-stage", "S1", B,
                [("    const ggml_sycl_stage_ring_published with_ring         = "
                  "ggml_sycl_stage_ring_published::snapshot(device);\n",
                  "    ggml_sycl::unified_cache * const     cache             = "
                  "ggml_sycl::get_unified_cache_for_device(device);\n"
                  "    const ggml_sycl_stage_ring_published with_ring         = "
                  "ggml_sycl_stage_ring_published::snapshot(device);\n")])
    expect_fail("drop-reads-the-arena-unchecked", "S1", B,
                [("    const size_t runtime_rebuilt = ggml_sycl_ring_arena_runtime_zone_bytes(device);\n",
                  "    const size_t runtime_rebuilt =\n"
                  "        ggml_sycl::get_existing_unified_cache_for_device(device)->zone_capacity("
                  "ggml_sycl::vram_zone_id::RUNTIME);\n")])
    expect_fail("arena-runtime-without-an-arena", "S1", B,
                [("    return cache != nullptr && cache->arena_active() ? "
                  "cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME) : 0;\n",
                  "    return cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME);\n")])
    expect_fail("no-decision-record", "S1", B,
                [("    g_sycl_ring_drop_marks.record(device, g_sycl_plan_load_effect_txn, decision);\n", "")])
    expect_fail("record-replaced-by-a-mark", "S1", B,
                [("    g_sycl_ring_drop_marks.record(device, g_sycl_plan_load_effect_txn, decision);\n",
                  "    if (decision == ggml_sycl::moe_pp_ring_decision::DROP) {\n"
                  "        g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);\n    }\n")])
    expect_fail("stage-reads-cache-zeros-zone", "S1", B,
                [("    zone.weight_with_ring    = ggml_sycl_ring_plan_weight_room(device, plan);\n",
                  "    zone.weight_with_ring    = cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT);\n")])
    expect_fail("held-not-read", "S1", B, [("        zone.held                = give_back.held;\n", "")])
    expect_fail("drop-decided-on-runtime", "S1", B,
                [("        zone.weight_without_ring = ggml_sycl_ring_plan_weight_room(device, plan);\n",
                  "        zone.weight_without_ring = zone.weight_with_ring + give_back.have_runtime -\n"
                  "                                   cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME);\n")])
    expect_fail("weight-read-after-the-ensure", "S1", B,
                [("    zone.weight_with_ring    = ggml_sycl_ring_plan_weight_room(device, plan);\n"
                  "    zone.weight_without_ring = zone.weight_with_ring;\n", ""),
                 ("        zone.held                = give_back.held;\n",
                  "        zone.held                = give_back.held;\n"
                  "        zone.weight_with_ring    = ggml_sycl_ring_plan_weight_room(device, plan);\n")])
    expect_fail("room-read-after-unpublish", "S1", B,
                [("    const size_t                         runtime_with_ring = "
                  "ggml_sycl_ring_runtime_with_ring_bytes(device);\n"
                  "    ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);\n",
                  "    ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);\n"
                  "    const size_t runtime_with_ring = ggml_sycl_ring_runtime_with_ring_bytes(device);\n")])
    expect_fail("stage-writes-the-ring-itself", "S1", B,
                [("    ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);\n",
                  "    ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);\n"
                  "    ggml_sycl_stage_publish_no_ring(device);\n")])
    expect_fail("keep-leaves-ring-unpublished", "K1", B,
                [("        with_ring.put_back(device, planning_n_ubatch);\n", "")])
    expect_fail("held-keeps-without-put-back", "K1", B,
                [("        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        bool plan_again = false;\n",
                  "        bool plan_again = false;\n"),
                 ("        } else {\n            const ggml_sycl_ring_no_weight_room no_room =\n",
                  "        } else {\n            with_ring.put_back(device, planning_n_ubatch);\n"
                  "            const ggml_sycl_ring_no_weight_room no_room =\n")])
    expect_fail("held-keep-replans", "K1", B,
                [("                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), "
                  "runtime_room));\n",
                  "                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), "
                  "runtime_room));\n"
                  "            return ggml_sycl_stage_ring_drop::DROPPED;\n")])
    expect_fail("put-back-without-re-ensure", "K1", B,
                [("            ggml_sycl::zone_arena_refusal refusal;\n"
                  "            if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal,\n"
                  "                                                                     "
                  "ggml_sycl_load_arena_options()))"
                  " {\n"
                  "                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "                return ggml_sycl_stage_ring_drop::REFUSED;\n"
                  "            }\n", "")])
    expect_fail("re-ensure-refusal-kept", "K1", B,
                [("                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "                return ggml_sycl_stage_ring_drop::REFUSED;\n"
                  "            }\n            const size_t weight_kept",
                  "                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "            }\n            const size_t weight_kept")])
    expect_fail("re-ensure-before-put-back", "K1", B,
                [("        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        bool plan_again = false;\n"
                  "        if (give_back.rebuilt) {\n",
                  "        if (give_back.rebuilt) {\n"
                  "            (void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, nullptr, {});\n"
                  "        }\n        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        bool plan_again = false;\n"
                  "        if (give_back.rebuilt) {\n")])
    expect_fail("kept-rebuild-never-plans-again", "K1", B,
                [("        if (plan_again) {\n            return ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN;\n        "
                  "}\n",
                  "")])
    expect_fail("kept-rebuild-judged-on-the-give-back-room", "K1", B,
                [("ggml_sycl::moe_pp_ring_kept_plan_again(zone.weight_with_ring, weight_kept);",
                  "ggml_sycl::moe_pp_ring_kept_plan_again(zone.weight_with_ring, zone.weight_without_ring);")])
    expect_fail("kept-rebuild-reads-the-room-before-the-re-ensure", "K1", B,
                [("            ggml_sycl::zone_arena_refusal refusal;\n"
                  "            if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal,\n",
                  "            const size_t early_room = ggml_sycl_ring_plan_weight_room(device, plan);\n"
                  "            (void) early_room;\n"
                  "            ggml_sycl::zone_arena_refusal refusal;\n"
                  "            if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal,\n"),
                 ("            const size_t weight_kept = ggml_sycl_ring_plan_weight_room(device, plan);\n",
                  "            const size_t weight_kept = early_room;\n")])
    expect_fail("dry-run-keep-skips-terms-put-back", "K1", B,
                [("        ggml_sycl_put_back_ring_terms(device, with_ring);\n", "")])
    expect_fail("keep-marks-the-load", "K1", B,
                [("        with_ring.put_back(device, planning_n_ubatch);\n",
                  "        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);\n")])
    expect_fail("put-back-skips-dense", "K1", B,
                [("        if (terms.ring_n_ubatch != planning_n_ubatch) {\n"
                  "            (void) ggml_sycl_publish_load_ubatch_scratch(device, terms.ring_n_ubatch);\n"
                  "        }\n", "")])
    expect_fail("put-back-skips-inventory", "K1", B,
                [("        g_tensor_inventory_pp_moe_onednn_ring_depth            = inventory_ring_depth;\n", "")])
    expect_fail("put-back-skips-planner-terms", "K1", B,
                [("ggml_sycl::LOAD_PUBLISHED_GROUP_RING | ggml_sycl::LOAD_PUBLISHED_GROUP_GRAPH |",
                  "ggml_sycl::LOAD_PUBLISHED_GROUP_RING |")])
    expect_fail("snapshot-misses-inventory", "K1", B,
                [("        published.inventory_ring_depth            = g_tensor_inventory_pp_moe_onednn_ring_depth;\n",
                  "")])
    expect_fail("held-guesses-holder", "K1", B,
                [("less without it, but something points into the arena, so it could not be rebuilt,",
                  "less without it, but a live model holds the arena, so it could not be rebuilt,")])
    expect_fail("held-warn-without-worth", "K1", B,
                [("                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), "
                  "runtime_room));\n",
                  "                int64_t(0));\n")])
    expect_fail("warn-without-host-count", "K1", B,
                [("device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_rebuilt / mb,",
                  "device, ring_bytes / mb, int64_t(0), check.n_weights, runtime_rebuilt / mb,")])
    expect_fail("second-give-back-writer", "S2", B,
                [("    options.load_txn = g_sycl_plan_load_effect_txn;\n",
                  "    options.load_txn = g_sycl_plan_load_effect_txn;\n    options.give_back = nullptr;\n")])
    expect_fail("single-replans-with-room", "S3", B,
                [("            case ggml_sycl_stage_ring_drop::DROPPED:\n"
                  "                plan_candidate = plan_single_device(/*with_room=*/false);",
                  "            case ggml_sycl_stage_ring_drop::DROPPED:\n"
                  "                plan_candidate = plan_single_device(/*with_room=*/true);")])
    expect_fail("single-plans-again-without-room", "S3", B,
                [("                plan_candidate = plan_single_device(candidate_with_room);\n",
                  "                plan_candidate = plan_single_device(/*with_room=*/false);\n")])
    expect_fail("single-plans-again-with-room-after-the-fallback", "S3", B,
                [("                candidate_with_room = false;\n", "")])
    expect_fail("multi-plan-again-stands", "S3", B,
                [("            case ggml_sycl_stage_ring_drop::KEPT:\n                break;\n"
                  "            case ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN:\n"
                  "            case ggml_sycl_stage_ring_drop::DROPPED:\n"
                  "                plan_candidate = plan_multi_device();\n",
                  "            case ggml_sycl_stage_ring_drop::KEPT:\n"
                  "            case ggml_sycl_stage_ring_drop::KEPT_PLAN_AGAIN:\n                break;\n"
                  "            case ggml_sycl_stage_ring_drop::DROPPED:\n"
                  "                plan_candidate = plan_multi_device();\n")])
    expect_fail("stage-defers-to-an-earlier-pass", "S5", B,
                [("    if (!check.displaces_experts()) {\n        return ggml_sycl_stage_ring_drop::KEPT;\n",
                  "    if (!check.displaces_experts() || ggml_sycl_stage_ring_dropped_by_this_load(device)) {\n"
                  "        return ggml_sycl_stage_ring_drop::KEPT;\n")])
    expect_fail("planner-skips-the-stage-after-an-earlier-pass", "S3", B,
                [("        switch (ggml_sycl_stage_drop_displacing_ring(ctx->device, plan_candidate, "
                  "published_before)) {\n",
                  "        switch (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) ? "
                  "ggml_sycl_stage_ring_drop::KEPT :\n"
                  "                ggml_sycl_stage_drop_displacing_ring(ctx->device, plan_candidate, "
                  "published_before)) {\n")])
    expect_fail("multi-refusal-plans-on", "S3", B,
                [("            case ggml_sycl_stage_ring_drop::REFUSED:\n                return false;\n        }\n"
                  "        if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {",
                  "            case ggml_sycl_stage_ring_drop::REFUSED:\n                break;\n        }\n"
                  "        if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {")])
    expect_fail("multi-budgets-read-once", "S3", B,
                [("        auto plan_multi_device = [&]() {\n"
                  "            const std::vector<ggml_sycl::device_budget> budgets =\n"
                  "                ggml_sycl_plan_device_budgets(budget_pct, cache_device_ready);\n",
                  "        const std::vector<ggml_sycl::device_budget> budgets =\n"
                  "            ggml_sycl_plan_device_budgets(budget_pct, cache_device_ready);\n"
                  "        auto plan_multi_device = [&]() {\n")])
    expect_fail("drop-marks-the-load-again", "S4", B,
                [("    const size_t runtime_rebuilt = ggml_sycl_ring_arena_runtime_zone_bytes(device);\n",
                  "    g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);\n"
                  "    const size_t runtime_rebuilt = ggml_sycl_ring_arena_runtime_zone_bytes(device);\n")])
    expect_fail("late-pass-asks-for-ring", "S4", B,
                [("    if (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) &&\n"
                  "        ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx->device) != 0) {\n"
                  "        ggml_sycl_stage_publish_no_ring(ctx->device);\n",
                  "    if (false) {\n")])
    # Structure: who calls what, and where the stage exits.
    stage_head = "static ggml_sycl_stage_ring_drop ggml_sycl_stage_drop_displacing_ring(\n"
    expect_fail("a-third-decision-caller", "D1", B,
                [(stage_head,
                  "static int ggml_sycl_ring_decide_elsewhere() {\n"
                  "    return static_cast<int>(ggml_sycl::moe_pp_ring_decide({}));\n}\n\n" + stage_head)])
    expect_fail("kept-exit-outside-the-keep-block", "K1", B,
                [("    return ggml_sycl_stage_ring_drop::DROPPED;\n",
                  "    if (ring_bytes == 0) {\n        return ggml_sycl_stage_ring_drop::KEPT;\n    }\n"
                  "    return ggml_sycl_stage_ring_drop::DROPPED;\n")])
    expect_fail("refusal-outside-an-ensure", "K1", B,
                [("    const uint32_t planning_n_ubatch = "
                  "ggml_sycl::placement_load_n_ubatch(g_placement_kv_info.n_ubatch);\n",
                  "    const uint32_t planning_n_ubatch = "
                  "ggml_sycl::placement_load_n_ubatch(g_placement_kv_info.n_ubatch);\n"
                  "    if (ring_bytes == 0) {\n        return ggml_sycl_stage_ring_drop::REFUSED;\n    }\n")])
    expect_fail("ensure-outside-a-refusal", "K1", B,
                [("    const uint32_t planning_n_ubatch = "
                  "ggml_sycl::placement_load_n_ubatch(g_placement_kv_info.n_ubatch);\n",
                  "    const uint32_t planning_n_ubatch = "
                  "ggml_sycl::placement_load_n_ubatch(g_placement_kv_info.n_ubatch);\n"
                  "    (void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, nullptr,\n"
                  "                                                             ggml_sycl_load_arena_options());\n")])
    expect_fail("a-second-stage-caller", "S3", B,
                [("static bool compute_and_store_plan_for_inventory(ggml_backend_sycl_context * ctx, size_t "
                  "vram_budget, "
                  "int budget_pct) {\n",
                  "static void ggml_sycl_ring_stage_elsewhere(const ggml_sycl::placement_plan & plan) {\n"
                  "    (void) ggml_sycl_stage_drop_displacing_ring(0, plan, {});\n}\n\n"
                  "static bool compute_and_store_plan_for_inventory(ggml_backend_sycl_context * ctx, size_t "
                  "vram_budget, "
                  "int budget_pct) {\n")])
    expect_fail("a-second-reader-caller", "S5", B,
                [(stage_head,
                  "static bool ggml_sycl_ring_skip_the_stage(int device) {\n"
                  "    return ggml_sycl_stage_ring_dropped_by_this_load(device);\n}\n\n" + stage_head)])
    expect_fail("the-record-read-elsewhere", "S4", B,
                [(stage_head,
                  "static bool ggml_sycl_ring_dropped_anywhere(int device) {\n"
                  "    return g_sycl_ring_drop_marks.dropped(device, 1);\n}\n\n" + stage_head)])
    expect_fail("give-back-read-elsewhere-in-the-cache", "Z1", U,
                [("bool unified_cache::ensure_planned_arena_zones(",
                  "static bool give_back_asked(const zone_arena_ensure_options & options) {\n"
                  "    return options.give_back != nullptr;\n}\n\n"
                  "bool unified_cache::ensure_planned_arena_zones(")])
    # Y1: the dry run.
    expect_fail("dry-run-keeps-ring-plan", "Y1", B, [("                plans.back() = plan_single();\n", "")])
    expect_fail("dry-run-gives-back", "Y1", B,
                [("    ggml_sycl_publish_no_pp_moe_onednn_ring(device);\n    terms_before_plan.restore(device);\n",
                  "    ggml_sycl_publish_no_pp_moe_onednn_ring(device);\n    terms_before_plan.restore(device);\n"
                  "    (void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, nullptr, {});\n")])
    expect_fail("dry-run-keep-leaves-ring-unpublished", "Y1", B,
                [("        ggml_sycl_put_back_ring_terms(device, with_ring);\n", ""),
                 ("        device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb);\n"
                  "    return true;\n",
                  "        device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb);\n"
                  "    ggml_sycl_put_back_ring_terms(device, with_ring);\n    return true;\n")])
    expect_fail("dry-run-room-with-plan-terms", "Y1", B,
                [("    terms_before_plan.restore(device);\n", "")])
    # Z1: the cache's ensure.
    expect_fail("surplus-without-record", "Z1", U,
                [("options.give_back != nullptr && zone_arena_runtime_surplus(arena_zones, planned_zones);",
                  "zone_arena_runtime_surplus(arena_zones, planned_zones);")])
    expect_fail("held-unmarked", "Z1", U, [("            options.give_back->held = true;\n", "")])
    expect_fail("held-rebuilds", "Z1", U,
                [("        if (surplus_asked && !zone_arena_rebuild_allowed(holders)) {\n",
                  "        if (false) {\n")])
    expect_fail("rebuilt-unmarked", "Z1", U, [("        options.give_back->rebuilt = true;\n", "")])
    expect_fail("give-back-takes-the-ordinary-rebuild", "Z1", U,
                [("if (!surplus_asked && (zones_sufficient || options.give_back != nullptr)) {",
                  "if (zones_sufficient && !surplus_asked) {")])
    # The NOT APPLIED report must fire on an anchor the sources do not hold, or a stale mutant would pass unseen.

    # The NOT APPLIED report must fire on an anchor the sources do not hold, or a stale mutant would pass unseen.
    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(Sources(raws))
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    raws = {name: path.read_text() for name, path in FILES.items()}
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(Sources(raws))
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the stage and the dry run are the decision's callers and ask the one predicate; the stage names no "
          "cache getter, writes through its unpublish, puts a kept ring back first, refuses only where an ensure fails "
          "and records its decision; the dry run sizes no zone")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
