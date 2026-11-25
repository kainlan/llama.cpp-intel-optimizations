#!/usr/bin/env python3
"""Source contract: a load drops a resident-only PP MoE oneDNN scratch ring that displaced experts and returns room.

Host-only: reads ggml-sycl.cpp, moe-mmvq-tables.hpp, moe-ring-consumer.hpp, unified-cache.cpp and zone-sizing.cpp/.hpp.
It builds nothing, loads no model and touches no device.

The loader sizes the ring for a model with an expert weight of a type a ring consumer runs, and the ring's room comes
out of the device's weight capacity. The rule is by use: a ring is dropped only when every weight that uses it reaches
the batched executor's resident claim alone, every one has another device route for a resident prompt node (the MMID
direct route), the plan made with the ring leaves at least one expert of those weights on the host, and the drop
returns room to the weights: the arena's WEIGHT zone, the one both planners pack the weights against, grows once the
zones are asked for without the ring, with the RUNTIME zone's surplus given back in an arena nothing points into. Today
the drop set is NVFP4 alone. Such a ring serves a layer only when every expert of it is on the device, and stages no
host expert. On the B50 with Inkling 2 Mini the ring moved one more layer's experts to the host, and decode and prompt
both ran slower than with no ring. So the load then publishes no ring, has the arena rebuilt with a smaller RUNTIME zone
and a larger WEIGHT zone, and plans once more, and that plan is final. Where the ring takes no RUNTIME room (another
consumer sizes the zone), where the plan without the ring would grow another zone (the give-back then keeps the arena as
built), where the room goes to no zone (an N-chunk tail chunk at its cap), or where the arena is held, a drop would give
the weights nothing: the ring is published again as it was and the plan made with it stands, with no second plan, and
an arena the give-back rebuilt is asked for again with the ring. A ring with any STAGED or SOA_EXPERT use, or with a
weight whose only device prompt route is the ring (IQ1_S, IQ1_M and IQ4_XS today), is sized, published and kept exactly
as before. These checks pin it:

  U1  moe_pp_ring_uses_of_type names the uses from the executors' own type tables: RESIDENT from
      moe_pp_onednn_ring_serves_type, STAGED from moe_pp_batched_takes_staged_type, SOA_EXPERT for MXFP4 at SOA;
  U2  one definition of the types the MMID direct route runs, in moe-mmvq-tables.hpp: the backend defines none of its
      own and asks it at the route's two admissions, the hybrid decode record build and the ring weight list;
  R1  moe_pp_ring_check_resident_only skips a weight with no use, is resident-only only when every weight's use is
      RESIDENT alone, has the other route only when every weight has it, counts host experts only then, by the plan's
      own convention on any device (every expert it does not place on a device, a missing one included), and
      displaces_experts() is all three with at least one host expert;
  R2  moe_pp_ring_decide answers (a) to (d) in order, each keep by name: no ring weight, a staged or SOA use, a weight
      with no other device route, no host expert, held (the arena kept as built because something points into it),
      no room returns (the WEIGHT zone does not grow without the ring), and DROP only past all of them; the answer is
      the WEIGHT zone with the ring and without it and the held flag; the RUNTIME zone with the ring is no more than
      the arena's, an unanswerable one with no arena the floor; the RUNTIME room is 0 where the zone needs no less;
      the dry run's answer is an unheld give-back whose WEIGHT zone takes the room; moe_pp_ring_experts_worth is the
      room over the ring weights' mean expert size;
  W1  the backend builds the weight list with the filter the ring's sizing reads: a gate, up, down or gate_up expert
      with a shape, of a type with a use, with ne[2] experts and its bytes, and the other route from the direct route's
      type set;
  D1  one decision: ggml_sycl_plan_ring_resident_check answers nothing for a device with no ring published, is the
      backend's only caller of moe_pp_ring_check_resident_only, and is called by the load stage's helper and the dry
      run's helper alone; both decide once through moe_pp_ring_decide, the backend's only callers of it, read the
      RUNTIME zone with the ring through one capped helper (the ruling-3 cap, so neither reads an unanswerable figure
      as room) and the zone without it through the reserve's own rounding over the floor and the planned requirement,
      and take the room through the consumer's rule; only the dry run answers from the room alone;
  S1  the stage helper returns KEPT when (a) to (c) keep the ring, before it writes anything; then it snapshots what it
      will write, reads the RUNTIME zone with the ring, publishes the device as a model with no ring consumer (the
      dense scratch at the planning shape, no ring, the planner terms from before the plan), reads the room, reads the
      WEIGHT zone, asks the zones with the give-back only on an arena where the ring takes RUNTIME room, refusing on a
      failed ensure, and reads the WEIGHT zone after it and the held flag; then it decides once;
  K1  a ring the decision keeps is put back: the stage has exactly two KEPT exits, the one before any write and the
      one after the put-back, and the put-back comes first on every kept outcome; where the give-back rebuilt the
      arena, the zones are asked for again with the ring (refusing on a failed ensure) and a WARN says so; otherwise a
      WARN with the experts' worth of room that names no holder where held, else an INFO that a drop would give the
      weights no room; the stage refuses at those two ensures alone; the put-back plans the dense scratch at the
      ring's micro-batch where the drop moved it, restores the ring's group and the planners' groups from the
      snapshot, and restores the inventory's copy of the ring, and the ring terms' put-back has the stage's put-back
      and the dry run's keep as its only callers; only DROP marks the load, WARNs that the RUNTIME zone was rebuilt
      smaller and the WEIGHT zone grew, and returns DROPPED;
  S2  only that helper asks for a give-back: one writer of options.give_back in the backend;
  S3  compute_and_store_plan_for_inventory calls the helper once per branch: the multi-device one on a plan made with
      budgets read at each plan, after a snapshot of the planner terms, and the single-device one after the micro-batch
      grow leg and before have_plan; KEPT changes nothing, DROPPED plans once more (single-device without the
      micro-batch room), REFUSED refuses the load;
  S4  the drop holds for the load: the helper marks the device with the load transaction, a mark is written and seen
      only for a staged load (txn != 0, so a direct planner entry never erases one), the reader asks the marks, and
      the planner publishes no ring before its entry ensure where an earlier pass dropped it;
  Y1  the dry run calls the same decision after each device's plan, on the RUNTIME room read with the ring and without
      it (the planner terms put back from a snapshot taken before that plan), as a give-back on an arena nothing points
      into, and says it cannot see a plan that would grow another zone; it sizes no zone and records no load; a ring
      it keeps is put back through the stage's put-back of the ring terms, and only a dropped one makes the caller
      replace that device's plan with one made without the ring;
  Z1  the cache's ensure gives back only when asked: the record is filled before the arena is read, a surplus is asked
      only with a record, an arena with none to give is kept as built when a give-back is asked, even where another
      zone would grow (a give-back never takes the ordinary rebuild), an arena something holds is kept as built and
      marked held, the rebuild takes zone_arena_give_back_sizes, and its tail keeps the snapshot as a kept arena does
      and marks rebuilt; nothing else in the cache touches the record;
  Z2  the zone rules: a surplus is every zone held and RUNTIME larger, and the give-back keeps SCRATCH and ONEDNN at
      least as built and sizes RUNTIME at the plan's.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. The decision's answers on plans and zone figures are
test-sycl-moe-ring-consumer's, the zone rules' are test-zone-sizing Case 51's, and that the B50 gets its base plan back,
the B70 keeps its ring, Qwen3.6 NVFP4 keeps its ring on the B50, and an arena whose SCRATCH zone would grow keeps it,
is the lead's run on the cards.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "backend": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "tables": ROOT / "ggml/src/ggml-sycl/moe-mmvq-tables.hpp",
    "consumer": ROOT / "ggml/src/ggml-sycl/moe-ring-consumer.hpp",
    "cache": ROOT / "ggml/src/ggml-sycl/unified-cache.cpp",
    "zs_hpp": ROOT / "ggml/src/ggml-sycl/zone-sizing.hpp",
    "zs_cpp": ROOT / "ggml/src/ggml-sycl/zone-sizing.cpp",
}

USES_FN = "inline uint32_t moe_pp_ring_uses_of_type(enum ggml_type type) {"
USES_BODY = ("{ uint32_t uses = MOE_PP_RING_USE_NONE; if (moe_pp_onednn_ring_serves_type(type)) { uses |= "
             "MOE_PP_RING_USE_RESIDENT; } if (moe_pp_batched_takes_staged_type(type)) { uses |= "
             "MOE_PP_RING_USE_STAGED; } if (type == GGML_TYPE_MXFP4 && moe_pp_batched_f16_supports_layout(type, "
             "GGML_LAYOUT_SOA)) { uses |= MOE_PP_RING_USE_SOA_EXPERT; } return uses; }")
RULE_FN = ("inline moe_pp_ring_resident_check moe_pp_ring_check_resident_only(const placement_plan & plan, "
           "const moe_pp_ring_weight * weights, size_t n_weights) {")
RULE_SKIP = "if (weights[i].uses == MOE_PP_RING_USE_NONE) { continue; } ++check.n_weights;"
RULE_MIXED = "if (weights[i].uses != MOE_PP_RING_USE_RESIDENT) { resident_only = false; }"
RULE_ROUTE = "if (!weights[i].other_device_prompt_route) { other_route = false; }"
RULE_ONLY = "check.resident_only = resident_only && check.n_weights > 0;"
RULE_BOTH = "check.other_device_prompt_route = check.resident_only && other_route;"
RULE_RETURN = "if (!check.other_device_prompt_route) { return check; }"
RULE_COUNT = ("check.host_experts += weights[i].n_experts - plan.count_experts_on_device(weights[i].name, "
              "weights[i].n_experts, -1);")
DISPLACES = ("bool displaces_experts() const { return resident_only && other_device_prompt_route && host_experts > 0; "
             "}")
DECIDE_RULE_FN = ("inline moe_pp_ring_decision moe_pp_ring_decide(const moe_pp_ring_resident_check & check, "
                  "const moe_pp_ring_zone_answer & zone) {")
DECIDE_RULE_BODY = ("{ if (check.n_weights == 0) { return moe_pp_ring_decision::KEEP_NO_RING_WEIGHT; } if "
                    "(!check.resident_only) { return moe_pp_ring_decision::KEEP_STAGED_OR_SOA_USE; } if "
                    "(!check.other_device_prompt_route) { return moe_pp_ring_decision::KEEP_NO_OTHER_DEVICE_ROUTE; } "
                    "if (check.host_experts <= 0) { return moe_pp_ring_decision::KEEP_NO_HOST_EXPERT; } if "
                    "(zone.held) { return moe_pp_ring_decision::KEEP_HELD; } if (zone.weight_without_ring <= "
                    "zone.weight_with_ring) { return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS; } return "
                    "moe_pp_ring_decision::DROP; }")
ZONE_ANSWER = ("struct moe_pp_ring_zone_answer { size_t weight_with_ring = 0; size_t weight_without_ring = 0; "
               "bool held = false; };")
WITH_RING_RULE_FN = ("inline size_t moe_pp_ring_runtime_with_ring(size_t planned, bool arena_active, size_t held, "
                     "size_t floor) {")
WITH_RING_RULE_BODY = ("{ if (arena_active) { return planned < held ? planned : held; } return planned == SIZE_MAX ? "
                       "floor : planned; }")
ROOM_FN = "inline size_t moe_pp_ring_runtime_room(size_t with_ring, size_t without_ring) {"
ROOM_BODY = "{ return without_ring < with_ring ? with_ring - without_ring : 0; }"
DRY_ANSWER_FN = "inline moe_pp_ring_zone_answer moe_pp_ring_dry_run_answer(size_t runtime_room) {"
DRY_ANSWER_BODY = "{ moe_pp_ring_zone_answer zone; zone.weight_without_ring = runtime_room; return zone; }"
WORTH_FN = ("inline int64_t moe_pp_ring_experts_worth(const moe_pp_ring_weight * weights, size_t n_weights, "
            "size_t room) {")
WORTH_ORDER = (
    "if (weights[i].uses == MOE_PP_RING_USE_NONE || weights[i].n_experts <= 0) { continue; }",
    "bytes += weights[i].bytes; experts += static_cast<uint64_t>(weights[i].n_experts);",
    "const uint64_t expert_bytes = experts == 0 ? 0 : bytes / experts;",
    "return expert_bytes == 0 ? 0 : static_cast<int64_t>(room / expert_bytes);",
)
RING_DECIDE_CALL = "moe_pp_ring_decide("
ZONE_FN = "static size_t ggml_sycl_planned_runtime_zone_bytes(int device) {"
ZONE_BODY = ("{ size_t requirement = 0; size_t reserved = 0; if "
             "(!ggml_sycl::unified_cache_get_planned_runtime_zone_requirement(device, &requirement) || "
             "!ggml_sycl::zone_runtime_reserved_bytes(ggml_sycl::unified_cache_runtime_zone_floor_bytes(), "
             "requirement, &reserved)) { return SIZE_MAX; } return reserved; }")
ZONE_CALL = "ggml_sycl_planned_runtime_zone_bytes("
WITH_RING_FN = "static size_t ggml_sycl_ring_runtime_with_ring_bytes(int device) {"
WITH_RING_BODY = ("{ ggml_sycl::unified_cache * const cache = ggml_sycl::get_existing_unified_cache_for_device(device); "
                  "const bool arena = cache != nullptr && cache->arena_active(); return "
                  "ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), arena, arena ? "
                  "cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME) : 0, "
                  "ggml_sycl::unified_cache_runtime_zone_floor_bytes()); }")
WITH_RING_CALL = "ggml_sycl_ring_runtime_with_ring_bytes("
ROOM_CALL = ("const size_t runtime_room = ggml_sycl::moe_pp_ring_runtime_room(runtime_with_ring, "
             "ggml_sycl_planned_runtime_zone_bytes(device));")
DRY_ANSWER_CALL = "moe_pp_ring_dry_run_answer("
DIRECT_FN = "inline bool moe_mmid_direct_route_runs_type(enum ggml_type type) {"
DIRECT_BODY = "{ return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4; }"
DIRECT_CALL = "moe_mmid_direct_route_runs_type("
DIRECT_CALLS = 4
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
    "ggml_sycl::unified_cache * const cache = ggml_sycl::get_unified_cache_for_device(device);",
    "const bool arena = cache != nullptr && cache->arena_active();",
    "const ggml_sycl_stage_ring_published with_ring = ggml_sycl_stage_ring_published::snapshot(device);",
    "const size_t runtime_with_ring = ggml_sycl_ring_runtime_with_ring_bytes(device);",
    "ggml_sycl_stage_unpublish_ring(device, planning_n_ubatch, published_before);",
    ROOM_CALL,
    "if (arena) { zone.weight_with_ring = cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT); "
    "zone.weight_without_ring = zone.weight_with_ring; }",
    "if (arena && runtime_room > 0) {",
    "ggml_sycl::zone_arena_ensure_options options = ggml_sycl_load_arena_options();",
    "options.give_back = &give_back;",
    "if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)) {",
    "return ggml_sycl_stage_ring_drop::REFUSED; }",
    "zone.weight_without_ring = cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT); zone.held = give_back.held; }",
    "const ggml_sycl::moe_pp_ring_decision decision = ggml_sycl::moe_pp_ring_decide(check, zone);",
)
UNPUBLISH_FN = ("static void ggml_sycl_stage_unpublish_ring(int device, uint32_t planning_n_ubatch, "
                "const ggml_sycl::planner_published_terms & published_before) {")
UNPUBLISH_BODY = ("{ if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch(device) != planning_n_ubatch) { "
                  "(void) ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch); } "
                  "ggml_sycl_stage_publish_no_ring(device); published_before.restore(device); }")
KEEP_EXIT = ("if (decision != ggml_sycl::moe_pp_ring_decision::DROP) { with_ring.put_back(device, planning_n_ubatch); "
             "if (give_back.rebuilt) {")
RE_ENSURE = ("ggml_sycl::zone_arena_refusal refusal; if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, "
             "&refusal, ggml_sycl_load_arena_options())) {")
KEEP_ORDER = (
    KEEP_EXIT,
    RE_ENSURE,
    "return ggml_sycl_stage_ring_drop::REFUSED; }",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): its arena was rebuilt with '
    'a "',
    "device, ring_bytes / mb, (give_back.have_runtime - give_back.need_runtime) / mb, zone.weight_with_ring / mb, "
    "zone.weight_without_ring / mb);",
    "} else if (decision == ggml_sycl::moe_pp_ring_decision::KEEP_HELD) {",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB) although the plan made with '
    'it "',
    "the ring displaces %\" PRId64 \" experts' worth of room from the weights\\n\",",
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb, "
    "ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), runtime_room)); } else {",
    'GGML_LOG_INFO("[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop would give the '
    'weights "',
    "device, ring_bytes / mb, runtime_room / mb, zone.weight_with_ring / mb, zone.weight_without_ring / mb); }",
    "return ggml_sycl_stage_ring_drop::KEPT; }",
    "g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);",
    "const size_t runtime_rebuilt = cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME);",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d drops its PP MoE oneDNN scratch ring (%.1f MB): the plan made with it '
    'left %"',
    '"trades prompt room for a decode loss on every token; its RUNTIME zone was rebuilt at %.1f MB, %.1f MB less, "',
    '"and its WEIGHT zone grew from %.1f MB to %.1f MB, and the device is planned once more without it. The "',
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_rebuilt / mb, "
    "(give_back.have_runtime - runtime_rebuilt) / mb, zone.weight_with_ring / mb, zone.weight_without_ring / mb);",
    "return ggml_sycl_stage_ring_drop::DROPPED; }",
)
KEPT_EXIT = "return ggml_sycl_stage_ring_drop::KEPT;"
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
              "ggml_sycl_stage_ring_drop::KEPT: break; case ggml_sycl_stage_ring_drop::DROPPED: plan_candidate = "
              "plan_multi_device(); break; case ggml_sycl_stage_ring_drop::REFUSED: return false; }")
GROW = "if (plan_candidate.explicit_ubatch.action == ggml_sycl::pp_moe_onednn_explicit_ubatch_action::GROW_ZONE) {"
SINGLE_SITE = ("switch (ggml_sycl_stage_drop_displacing_ring(ctx->device, plan_candidate, published_before)) { case "
               "ggml_sycl_stage_ring_drop::KEPT: break; case ggml_sycl_stage_ring_drop::DROPPED: plan_candidate = "
               "plan_single_device(false); break; case ggml_sycl_stage_ring_drop::REFUSED: return false; } "
               "have_plan = true;")
STAGE_CALL = "ggml_sycl_stage_drop_displacing_ring("
STICKY = ("if (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) && "
          "ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx->device) != 0) { "
          "ggml_sycl_stage_publish_no_ring(ctx->device);")
ENTRY_ENSURE = "!ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)) {"
READER_FN = "static bool ggml_sycl_stage_ring_dropped_by_this_load(int device) {"
READER_BODY = "{ return g_sycl_ring_drop_marks.dropped(device, g_sycl_plan_load_effect_txn); }"
TXN = "g_sycl_ring_drop_marks"
TXN_WRITE = "g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);"
MARK_BODY = ("void mark(int device, uint64_t txn) { if (txn != 0 && device_in_range(device)) { load_txn[device] = "
             "txn; } }")
DROPPED_BODY = ("bool dropped(int device, uint64_t txn) const { return txn != 0 && device_in_range(device) && "
                "load_txn[device] == txn; }")
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
    "if (ggml_sycl::moe_pp_ring_decide(check, ggml_sycl::moe_pp_ring_dry_run_answer(runtime_room)) != "
    "ggml_sycl::moe_pp_ring_decision::DROP) { ggml_sycl_put_back_ring_terms(device, with_ring);",
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d keeps its PP MoE oneDNN scratch ring (%.1f MB), as a load would: its '
    'RUNTIME "',
    "device, ring_bytes / mb, runtime_with_ring / mb);",
    "return false; }",
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d drops its PP MoE oneDNN scratch ring (%.1f MB), as a load would whose '
    'arena nothing "',
    '"would grow another zone, which the dry run cannot see\\n",',
    "device, ring_bytes / mb, check.host_experts, check.n_weights, runtime_room / mb);",
    "return true; }",
)
DRY_FORBIDDEN = ("ensure_planned_arena_zones", TXN, "give_back", "ggml_sycl_stage_")
DRY_PLANS_FN = "static std::vector<ggml_sycl::placement_plan> ggml_sycl_plan_dry_run_plans("
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
ENSURE_GIVE_BACK_USES = 9
SURPLUS = ("bool zone_arena_runtime_surplus(const zone_arena_sizes & have, const zone_arena_sizes & need) { return "
           "have.scratch >= need.scratch && have.onednn >= need.onednn && have.runtime > need.runtime; }")
GIVE_BACK_SIZES = ("zone_arena_sizes zone_arena_give_back_sizes(const zone_arena_sizes & have, const "
                   "zone_arena_sizes & need) { return { std::max(have.scratch, need.scratch), "
                   "std::max(have.onednn, need.onednn), need.runtime }; }")
OPTION_MEMBER = "zone_arena_give_back * give_back = nullptr;"


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


def need_count(hay: str, needle: str, count: int, pin: str, what: str) -> None:
    n = squash(needle)
    if hay.count(n) != count:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` appears {hay.count(n)} times, not {count}")


def need_ordered(hay: str, parts: tuple, pin: str, what: str) -> None:
    pos = 0
    for p in parts:
        k = hay.find(squash(p), pos)
        if k == -1:
            raise ContractError(f"FAIL [pin {pin}]: {what}: `{p}` is missing or out of order")
        pos = k + len(squash(p))


def fn(hay: str, sig: str, pin: str, what: str) -> str:
    """The body of the one function whose signature, ending at its opening brace, is `sig`."""
    at = need_once(hay, sig, pin, what) + len(squash(sig)) - 1
    return brace_text(hay, at, pin)


class Sources:
    def __init__(self, raws: dict[str, str]):
        self.sq = {name: squash(blank_comments(text)) for name, text in raws.items()}


def check_decision(s: Sources) -> None:
    be, tb, rc = s.sq["backend"], s.sq["tables"], s.sq["consumer"]
    # U1
    if fn(tb, USES_FN, "U1", "the ring uses of a type") != squash(USES_BODY):
        raise ContractError("FAIL [pin U1]: moe_pp_ring_uses_of_type must name each use from its executor's table")
    # R1
    rule = fn(rc, RULE_FN, "R1", "the resident-only check")
    need_ordered(rule, (RULE_SKIP, RULE_MIXED, RULE_ROUTE, RULE_ONLY, RULE_BOTH, RULE_RETURN, RULE_COUNT), "R1",
                 "the check skips no-use weights, is resident-only only for RESIDENT alone, has the other route only "
                 "when every weight has it, then counts every expert the plan does not place on a device")
    need_once(rc, DISPLACES, "R1", "a ring displaces experts only when resident-only, routed and with a host expert")
    # R2
    if fn(rc, DECIDE_RULE_FN, "R2", "the decision") != squash(DECIDE_RULE_BODY):
        raise ContractError("FAIL [pin R2]: the decision must ask (a) to (d) in order, the held arena before the WEIGHT "
                            "zone's growth, and keep by name")
    need_once(rc, ZONE_ANSWER, "R2", "the arena's answer is the WEIGHT zone with the ring and without it, and held")
    if fn(rc, WITH_RING_RULE_FN, "R2", "the RUNTIME zone with the ring") != squash(WITH_RING_RULE_BODY):
        raise ContractError("FAIL [pin R2]: the RUNTIME zone with the ring is no more than the arena's, and an "
                            "unanswerable one with no arena is the floor")
    if fn(rc, ROOM_FN, "R2", "the RUNTIME room") != squash(ROOM_BODY):
        raise ContractError("FAIL [pin R2]: the RUNTIME room is the zone with the ring less the zone without it, 0 "
                            "where it needs no less")
    if fn(rc, DRY_ANSWER_FN, "R2", "the dry run's answer") != squash(DRY_ANSWER_BODY):
        raise ContractError("FAIL [pin R2]: the dry run answers as an unheld give-back whose WEIGHT zone takes the room")
    need_ordered(fn(rc, WORTH_FN, "R2", "the experts' worth"), WORTH_ORDER, "R2",
                 "the experts' worth is the room over the ring weights' mean expert size")
    # U2
    if fn(tb, DIRECT_FN, "U2", "the direct route's types") != squash(DIRECT_BODY):
        raise ContractError("FAIL [pin U2]: the MMID direct route's type set must be Q1_0 and NVFP4, defined once")
    if "mmid_direct_route_runs_type(ggml_type" in be or "mmid_direct_route_runs_type(enum ggml_type" in be:
        raise ContractError("FAIL [pin U2]: the backend must not define its own direct-route type set")
    need_count(be, DIRECT_CALL, DIRECT_CALLS, "U2",
               "the backend asks the tables' direct-route set at its route admissions, the decode record build and "
               "the ring weight list")
    # W1
    if fn(be, WEIGHTS_FN, "W1", "the backend's ring weight list") != squash(WEIGHTS_BODY):
        raise ContractError("FAIL [pin W1]: the ring weight list must take the ring sizing's filter and ne[2] experts")
    # D1
    if fn(be, DECIDE_FN, "D1", "the one decision") != squash(DECIDE_BODY):
        raise ContractError("FAIL [pin D1]: the decision must answer nothing without a ring and ask the rule once")
    need_count(be, RULE_CALL, 1, "D1", "the backend asks the rule only through the decision")
    need_count(be, DECIDE_CALL, 3, "D1", "the decision is defined once and called by the stage and the dry run")
    for sig in (STAGE_FN, DRY_FN):
        need_count(fn(be, sig, "D1", "a caller of the decision"), DECIDE_CALL, 1, "D1",
                   "each of the stage's and the dry run's helpers asks the decision once")
    need_count(be, RING_DECIDE_CALL, 2, "D1", "the stage and the dry run each ask the decision, and nothing else does")
    if fn(be, ZONE_FN, "D1", "the zone figure") != squash(ZONE_BODY):
        raise ContractError("FAIL [pin D1]: the zone figure must be the reserve's rounding over the floor and the "
                            "planned requirement, SIZE_MAX when unanswerable")
    if fn(be, WITH_RING_FN, "D1", "the RUNTIME zone with the ring") != squash(WITH_RING_BODY):
        raise ContractError("FAIL [pin D1]: the RUNTIME zone with the ring must be capped by the arena's RUNTIME zone, "
                            "and read as the floor when unanswerable with no arena, through the consumer's rule")
    need_count(be, WITH_RING_CALL, 3, "D1", "the RUNTIME zone with the ring is defined once and read by the stage and "
               "the dry run")
    need_count(be, ZONE_CALL, 4, "D1", "the zone figure is defined once and read by the capped helper, the stage and "
               "the dry run")
    need_count(be, DRY_ANSWER_CALL, 1, "D1", "only the dry run answers from the RUNTIME room alone")
    for sig in (STAGE_FN, DRY_FN):
        body = fn(be, sig, "D1", "a reader of the RUNTIME zones")
        need_count(body, RING_DECIDE_CALL, 1, "D1", "each of the stage and the dry run decides once")
        need_count(body, WITH_RING_CALL, 1, "D1", "each reads the RUNTIME zone with the ring through the one cap")
        need_count(body, ZONE_CALL, 1, "D1", "each reads the RUNTIME zone without the ring once")
        need_once(body, ROOM_CALL, "D1", "each reads the RUNTIME room through the consumer's rule")


def check_stage(s: Sources) -> None:
    be = s.sq["backend"]
    # S1
    stage = fn(be, STAGE_FN, "S1", "the stage helper")
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
    # K1
    need_ordered(stage, KEEP_ORDER, "K1", "a kept ring is put back before either kept line, and only a drop marks")
    need_count(stage, KEPT_EXIT, 2, "K1", "the stage keeps at the (a)-(c) exit and after the put-back, nowhere else")
    need_count(stage, "return ggml_sycl_stage_ring_drop::REFUSED;", 2, "K1",
               "the stage refuses where the give-back's ensure or the put-back's ensure fails, nowhere else")
    need_count(stage, "unified_cache_ensure_planned_arena_zones(", 2, "K1",
               "the stage asks the zones for the give-back and, after a give-back rebuild it keeps, with the ring")
    first_kept = stage.find(squash(KEPT_EXIT))
    if stage.find("ggml_sycl_stage_unpublish_ring(") < first_kept:
        raise ContractError("FAIL [pin K1]: the stage's first KEPT exit must come before anything is written")
    need_count(stage, "put_back(", 1, "K1", "one put-back, on the one kept exit after the write")
    for guess in HELD_GUESSES:
        if guess in stage:
            raise ContractError(f"FAIL [pin K1]: the held WARN names what is known, not a guessed holder: `{guess}`")
    at = need_once(be, PUBLISHED_STRUCT, "K1", "the snapshot of what a drop writes")
    published = brace_text(be, at + len(squash(PUBLISHED_STRUCT)) - 1, "K1")
    need_once(published, PUBLISHED_SNAPSHOT, "K1", "the snapshot takes the load's terms and the inventory's ring")
    if fn(published, PUT_BACK_FN, "K1", "the put-back") != squash(PUT_BACK_BODY):
        raise ContractError("FAIL [pin K1]: the put-back must plan the dense scratch at the ring's micro-batch "
                            "where it moved, restore the ring terms and the inventory's ring")
    if fn(be, TERMS_FN, "K1", "the put-back of the ring terms") != squash(TERMS_BODY):
        raise ContractError("FAIL [pin K1]: the put-back of the ring terms must restore the ring's group and every "
                            "planner group the unpublish writes")
    need_count(be, TERMS_CALL, 3, "K1", "the ring terms' put-back is defined once, called by the stage's put-back "
               "and the dry run's keep")
    # S2
    need_count(be, GIVE_BACK_WRITE, 1, "S2", "one writer of the give-back option in the backend")
    need_once(stage, "options.give_back = &give_back;", "S2", "the stage helper is the give-back's writer")
    # S3
    plan = fn(be, PLAN_FN, "S3", "the planner")
    need_count(be, STAGE_CALL, 3, "S3", "the stage helper is defined once and called once per branch")
    need_once(plan, MULTI_LAMBDA, "S3", "the multi-device plan reads its budgets at each plan")
    need_once(plan, MULTI_SITE, "S3", "the multi-device branch drops, plans once more, or refuses")
    need_once(plan, SINGLE_SITE, "S3", "the single-device branch drops, plans once more without the room, or refuses")
    need_ordered(plan, (GROW, SINGLE_SITE), "S3", "the single-device drop follows the micro-batch grow leg")
    # S4
    need_once(be, READER_FN, "S4", "the drop's reader")
    if fn(be, READER_FN, "S4", "the drop's reader") != squash(READER_BODY):
        raise ContractError("FAIL [pin S4]: the reader must answer only for this thread's staged load")
    need_count(be, TXN, 3, "S4", "the drop's record is declared, read and written once each")
    need_once(stage, TXN_WRITE, "S4", "the stage helper records the load that dropped the ring")
    rc = s.sq["consumer"]
    need_once(rc, MARK_BODY, "S4", "a mark is written only for a staged load on a device in range")
    need_once(rc, DROPPED_BODY, "S4", "a mark is seen only by the staged load that wrote it")
    need_ordered(plan, (STICKY, ENTRY_ENSURE), "S4", "a dropped ring is published as none before the entry ensure")


def check_dry_run(s: Sources) -> None:
    be = s.sq["backend"]
    # Y1
    dry = fn(be, DRY_FN, "Y1", "the dry run's helper")
    need_ordered(dry, DRY_ORDER, "Y1", "the dry run's helper decides, publishes no ring and says so")
    for word in DRY_FORBIDDEN:
        if word in dry:
            raise ContractError(f"FAIL [pin Y1]: the dry run sizes no zone and records no load, but names `{word}`")
    at = need_once(be, DRY_PLANS_FN, "Y1", "the dry run's plans")
    plans = brace_text(be, be.find(") {", at) + 2, "Y1")
    for site in DRY_SITES:
        need_once(plans, site, "Y1", "each dry-run plan is made again without a dropped ring")
    need_count(be, DRY_CALL, 3, "Y1", "the dry run's helper is defined once and called once per branch")


def check_zones(s: Sources) -> None:
    uc, zc, zh = s.sq["cache"], s.sq["zs_cpp"], s.sq["zs_hpp"]
    # Z1
    ensure = fn(uc, ENSURE_FN, "Z1", "the zone ensure")
    need_ordered(ensure, ENSURE_ORDER, "Z1", "the ensure's give-back")
    need_count(ensure, "give_back", ENSURE_GIVE_BACK_USES, "Z1", "the ensure's give-back record and its reads")
    if uc.count("give_back") != ensure.count("give_back"):
        raise ContractError("FAIL [pin Z1]: nothing in the cache but the ensure touches the give-back record")
    # Z2
    need_once(zc, SURPLUS, "Z2", "a surplus is every zone held and RUNTIME larger")
    need_once(zc, GIVE_BACK_SIZES, "Z2", "the give-back keeps SCRATCH and ONEDNN and sizes RUNTIME at the plan's")
    need_once(zh, OPTION_MEMBER, "Z2", "the ensure option that asks for the give-back")


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

    B, T, C, U, ZC, ZH = "backend", "tables", "consumer", "cache", "zs_cpp", "zs_hpp"
    # U1, R1: the uses and the rule.
    expect_fail("staged-use-dropped", "U1", T,
                [("    if (moe_pp_batched_takes_staged_type(type)) {\n        uses |= MOE_PP_RING_USE_STAGED;",
                  "    if (false) {\n        uses |= MOE_PP_RING_USE_STAGED;")])
    expect_fail("soa-use-any-type", "U1", T,
                [("if (type == GGML_TYPE_MXFP4 && moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_SOA)) {",
                  "if (moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_SOA)) {")])
    expect_fail("mixed-ring-resident-only", "R1", C, [("            resident_only = false;\n", "")])
    expect_fail("host-count-device-zero", "R1", C,
                [("plan.count_experts_on_device(weights[i].name, weights[i].n_experts, -1);",
                  "plan.count_experts_on_device(weights[i].name, weights[i].n_experts, 0);")])
    expect_fail("host-count-semantic-only", "R1", C,
                [("weights[i].n_experts - plan.count_experts_on_device(weights[i].name, weights[i].n_experts, -1);",
                  "plan.count_planned_experts(weights[i].name, weights[i].n_experts, -1).host;")])
    expect_fail("displaces-any-host", "R1", C,
                [("return resident_only && other_device_prompt_route && host_experts > 0;",
                  "return host_experts > 0;")])
    expect_fail("displaces-without-route", "R1", C,
                [("return resident_only && other_device_prompt_route && host_experts > 0;",
                  "return resident_only && host_experts > 0;")])
    expect_fail("route-ignored", "R1", C, [("            other_route = false;\n", "")])
    expect_fail("route-without-resident-only", "R1", C,
                [("check.other_device_prompt_route = check.resident_only && other_route;",
                  "check.other_device_prompt_route = other_route;")])
    expect_fail("direct-set-widened", "U2", T,
                [("inline bool moe_mmid_direct_route_runs_type(enum ggml_type type) {\n"
                  "    return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4;",
                  "inline bool moe_mmid_direct_route_runs_type(enum ggml_type type) {\n"
                  "    return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4 || type == GGML_TYPE_IQ4_XS;")])
    expect_fail("route-admission-own-type-test", "U2", B,
                [("moe_mmid_direct_route_runs_type(src0->type) && direct_layout == GGML_LAYOUT_AOS",
                  "(src0->type == GGML_TYPE_NVFP4) && direct_layout == GGML_LAYOUT_AOS")])
    expect_fail("backend-own-direct-set", "U2", B,
                [("static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights(",
                  "static bool ggml_sycl_moe_mmid_direct_route_runs_type(ggml_type type) {\n"
                  "    return type == GGML_TYPE_NVFP4;\n}\n\n"
                  "static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights(")])
    # W1, D1: the weight list and the one decision.
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
    expect_fail("decide-drops-without-room", "R2", C,
                [("    if (zone.weight_without_ring <= zone.weight_with_ring) {\n"
                  "        return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS;\n    }\n", "")])
    expect_fail("decide-drops-held", "R2", C,
                [("    if (zone.held) {\n        return moe_pp_ring_decision::KEEP_HELD;\n    }\n", "")])
    expect_fail("decide-room-at-equal", "R2", C,
                [("if (zone.weight_without_ring <= zone.weight_with_ring) {",
                  "if (zone.weight_without_ring < zone.weight_with_ring) {")])
    expect_fail("decide-room-before-held", "R2", C,
                [("    if (zone.held) {\n        return moe_pp_ring_decision::KEEP_HELD;\n    }\n"
                  "    if (zone.weight_without_ring <= zone.weight_with_ring) {\n"
                  "        return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS;\n    }\n",
                  "    if (zone.weight_without_ring <= zone.weight_with_ring) {\n"
                  "        return moe_pp_ring_decision::KEEP_NO_ROOM_RETURNS;\n    }\n"
                  "    if (zone.held) {\n        return moe_pp_ring_decision::KEEP_HELD;\n    }\n")])
    expect_fail("with-ring-beyond-the-arena", "R2", C,
                [("        return planned < held ? planned : held;\n", "        return planned;\n")])
    expect_fail("with-ring-unanswerable-as-room", "R2", C,
                [("return planned == SIZE_MAX ? floor : planned;", "return planned;")])
    expect_fail("room-of-an-unanswerable-zone", "R2", C,
                [("return without_ring < with_ring ? with_ring - without_ring : 0;",
                  "return with_ring - without_ring;")])
    expect_fail("dry-answer-held", "R2", C,
                [("    zone.weight_without_ring = runtime_room;\n    return zone;\n",
                  "    zone.weight_without_ring = runtime_room;\n    zone.held = true;\n    return zone;\n")])
    expect_fail("worth-counts-no-use", "R2", C,
                [("        if (weights[i].uses == MOE_PP_RING_USE_NONE || weights[i].n_experts <= 0) {\n"
                  "            continue;\n        }\n        bytes += weights[i].bytes;",
                  "        if (weights[i].n_experts <= 0) {\n"
                  "            continue;\n        }\n        bytes += weights[i].bytes;")])
    expect_fail("zone-figure-without-floor", "D1", B,
                [("!ggml_sycl::zone_runtime_reserved_bytes(ggml_sycl::unified_cache_runtime_zone_floor_bytes(), "
                  "requirement,",
                  "!ggml_sycl::zone_runtime_reserved_bytes(0, requirement,")])
    expect_fail("dry-run-decides-on-the-plan-alone", "D1", B,
                [("    if (ggml_sycl::moe_pp_ring_decide(check, ggml_sycl::moe_pp_ring_dry_run_answer(runtime_room)) !=\n"
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
                [("    return ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), arena,",
                  "    return ggml_sycl::moe_pp_ring_runtime_with_ring(ggml_sycl_planned_runtime_zone_bytes(device), false,")])
    # S1-S4, K1: the stage.
    expect_fail("terms-not-restored", "S1", B, [("    published_before.restore(device);\n", "")])
    expect_fail("dense-left-at-micro-batch", "S1", B,
                [("        (void) ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch);\n", "")])
    expect_fail("ensure-without-give-back", "S1", B,
                [("        options.give_back                            = &give_back;\n", "")])
    expect_fail("refused-ensure-drops", "S1", B,
                [("            return ggml_sycl_stage_ring_drop::REFUSED;\n        }\n"
                  "        zone.weight_without_ring",
                  "        }\n        zone.weight_without_ring")])
    expect_fail("arena-asked-without-room", "S1", B,
                [("    if (arena && runtime_room > 0) {\n", "    if (arena) {\n")])
    expect_fail("held-not-read", "S1", B, [("        zone.held                = give_back.held;\n", "")])
    expect_fail("drop-decided-on-runtime", "S1", B,
                [("        zone.weight_without_ring = cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT);\n",
                  "        zone.weight_without_ring = zone.weight_with_ring + give_back.have_runtime -\n"
                  "                                   cache->zone_capacity(ggml_sycl::vram_zone_id::RUNTIME);\n")])
    expect_fail("weight-read-after-the-ensure", "S1", B,
                [("    if (arena) {\n        zone.weight_with_ring    = "
                  "cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT);\n"
                  "        zone.weight_without_ring = zone.weight_with_ring;\n    }\n", ""),
                 ("        zone.held                = give_back.held;\n",
                  "        zone.held                = give_back.held;\n"
                  "        zone.weight_with_ring    = cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT);\n")])
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
                  "        if (give_back.rebuilt) {\n",
                  "        if (give_back.rebuilt) {\n"),
                 ("        } else {\n            GGML_LOG_INFO(\n"
                  "                \"[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop would "
                  "give the weights \"\n",
                  "        } else {\n            with_ring.put_back(device, planning_n_ubatch);\n"
                  "            GGML_LOG_INFO(\n"
                  "                \"[SYCL-PLAN] device %d keeps its PP MoE oneDNN scratch ring (%.1f MB): a drop would "
                  "give the weights \"\n")])
    expect_fail("held-keep-replans", "K1", B,
                [("                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), runtime_room));\n",
                  "                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), runtime_room));\n"
                  "            return ggml_sycl_stage_ring_drop::DROPPED;\n")])
    expect_fail("put-back-without-re-ensure", "K1", B,
                [("            ggml_sycl::zone_arena_refusal refusal;\n"
                  "            if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal,\n"
                  "                                                                     ggml_sycl_load_arena_options()))"
                  " {\n"
                  "                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "                return ggml_sycl_stage_ring_drop::REFUSED;\n"
                  "            }\n", "")])
    expect_fail("re-ensure-refusal-kept", "K1", B,
                [("                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "                return ggml_sycl_stage_ring_drop::REFUSED;\n"
                  "            }\n            GGML_LOG_WARN(\n",
                  "                GGML_LOG_ERROR(\"%s\\n\", ggml_sycl::zone_arena_refusal_text(refusal).c_str());\n"
                  "            }\n            GGML_LOG_WARN(\n")])
    expect_fail("re-ensure-before-put-back", "K1", B,
                [("        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        if (give_back.rebuilt) {\n",
                  "        if (give_back.rebuilt) {\n"
                  "            (void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, nullptr, {});\n"
                  "        }\n        with_ring.put_back(device, planning_n_ubatch);\n"
                  "        if (give_back.rebuilt) {\n")])
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
                [("                ggml_sycl::moe_pp_ring_experts_worth(weights.data(), weights.size(), runtime_room));\n",
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
    expect_fail("no-drop-record", "K1", B,
                [("    g_sycl_ring_drop_marks.mark(device, g_sycl_plan_load_effect_txn);\n", "")])
    expect_fail("direct-entry-sticky", "S4", C,
                [("return txn != 0 && device_in_range(device) && load_txn[device] == txn;",
                  "return device_in_range(device) && load_txn[device] == txn;")])
    expect_fail("direct-entry-erases-mark", "S4", C,
                [("if (txn != 0 && device_in_range(device)) {", "if (device_in_range(device)) {")])
    expect_fail("late-pass-asks-for-ring", "S4", B,
                [("    if (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) &&\n"
                  "        ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx->device) != 0) {\n"
                  "        ggml_sycl_stage_publish_no_ring(ctx->device);\n",
                  "    if (false) {\n")])
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
    # Z1, Z2: the cache and the zone rules.
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
    expect_fail("surplus-at-equal", "Z2", ZC,
                [("have.runtime > need.runtime;", "have.runtime >= need.runtime;")])
    expect_fail("give-back-shrinks-scratch", "Z2", ZC,
                [("return { std::max(have.scratch, need.scratch),", "return { need.scratch,")])
    expect_fail("option-removed", "Z2", ZH,
                [("    zone_arena_give_back * give_back          = nullptr;\n", "    void * give_back = nullptr;\n")])

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
    print("PASS: a resident-only PP MoE ring whose weights keep another device prompt route and that displaced experts "
          "is dropped only where the WEIGHT zone grows once its RUNTIME surplus is given back, and the device planned "
          "once more; a ring that returns no room or whose arena is held is put back and its plan stands; a ring with "
          "a staged or SOA use, or a weight with no other route, is kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
