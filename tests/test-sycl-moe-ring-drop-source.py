#!/usr/bin/env python3
"""Source contract: a load drops a PP MoE oneDNN scratch ring that serves resident layers only and displaced experts.

Host-only: reads ggml-sycl.cpp, moe-mmvq-tables.hpp, moe-ring-consumer.hpp, unified-cache.cpp and zone-sizing.cpp/.hpp.
It builds nothing, loads no model and touches no device.

The loader sizes the ring for a model with an expert weight of a type a ring consumer runs, and the ring's room comes
out of the device's weight capacity. NVFP4 experts reach the batched executor's resident claim and nothing else: such a
ring serves a layer only when every expert of it is on the device, and stages no host expert. On the B50 with Inkling 2
Mini the ring moved one more layer's experts to the host, and decode and prompt both ran slower than with no ring. So a
load keeps such a ring only when the plan made with it leaves no expert of those weights on the host; otherwise it
publishes no ring, gives the RUNTIME zone's surplus back when nothing holds the arena, and plans once more, and that plan
is final. A ring with a weight that reaches the staged claim or the MXFP4 SOA staging (Qwen, GPT-OSS) is kept as
before, so those plans do not change. These checks pin it:

  U1  moe_pp_ring_uses_of_type names the uses from the executors' own type tables: RESIDENT from
      moe_pp_onednn_ring_serves_type, STAGED from moe_pp_batched_takes_staged_type, SOA_EXPERT for MXFP4 at SOA;
  R1  moe_pp_ring_check_resident_only skips a weight with no use, is resident-only only when every weight's use is
      RESIDENT alone, counts host experts only then, through the plan's semantic index on any device, and
      displaces_experts() is resident-only with at least one host expert;
  W1  the backend builds the weight list with the filter the ring's sizing reads: a gate, up, down or gate_up expert
      with a shape, of a type with a use, with ne[2] experts;
  D1  one decision: ggml_sycl_plan_ring_resident_check answers nothing for a device with no ring published, is the
      backend's only caller of moe_pp_ring_check_resident_only, and is called by the load stage's helper and the dry
      run's helper alone;
  S1  the stage helper decides, returns KEPT when the ring displaces nothing, then puts the dense scratch back at the
      planning shape, publishes no ring, restores the planner terms the plans made with the ring published, asks for
      the zones with the give-back, refuses on a failed ensure, WARNs (the device, the ring's MB, the host experts and
      the zone's fate) and returns DROPPED;
  S2  only that helper asks for a give-back: one writer of options.give_back in the backend;
  S3  compute_and_store_plan_for_inventory calls the helper once per branch: the multi-device one on a plan made with
      budgets read at each plan, after a snapshot of the planner terms, and the single-device one after the micro-batch
      grow leg and before have_plan; KEPT changes nothing, DROPPED plans once more (single-device without the
      micro-batch room), REFUSED refuses the load;
  S4  the drop holds for the load: the helper records the load transaction, the reader answers only for a staged load
      (txn != 0), and the planner publishes no ring before its entry ensure where an earlier pass dropped it;
  Y1  the dry run calls the same decision after each device's plan, publishes no ring, sizes no zone, records no load,
      and replaces that device's plan with one made without the ring;
  Z1  the cache's ensure gives back only when asked: the record is filled before the arena is read, a surplus is asked
      only with a record, an arena something holds is kept as built and marked held, the rebuild takes
      zone_arena_give_back_sizes, and its tail keeps the snapshot as a kept arena does and marks rebuilt; nothing else in
      the cache touches the record;
  Z2  the zone rules: a surplus is every zone held and RUNTIME larger, and the give-back keeps SCRATCH and ONEDNN at
      least as built and sizes RUNTIME at the plan's.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. The decision's answers on plans are test-sycl-moe-ring-consumer's, the zone
rules' are test-zone-sizing Case 51's, and that the B50 gets its base plan back and the B70 keeps its ring is the lead's
run on the cards.
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
             "MOE_PP_RING_USE_RESIDENT; } if (moe_pp_batched_takes_staged_type(type)) { uses |= MOE_PP_RING_USE_STAGED; "
             "} if (type == GGML_TYPE_MXFP4 && moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_SOA)) { uses |= "
             "MOE_PP_RING_USE_SOA_EXPERT; } return uses; }")
RULE_FN = ("inline moe_pp_ring_resident_check moe_pp_ring_check_resident_only(const placement_plan & plan, "
           "const moe_pp_ring_weight * weights, size_t n_weights) {")
RULE_SKIP = "if (weights[i].uses == MOE_PP_RING_USE_NONE) { continue; } ++check.n_weights;"
RULE_MIXED = "if (weights[i].uses != MOE_PP_RING_USE_RESIDENT) { resident_only = false; }"
RULE_ONLY = "check.resident_only = resident_only && check.n_weights > 0;"
RULE_RETURN = "if (!check.resident_only) { return check; }"
RULE_COUNT = "check.host_experts += plan.count_planned_experts(weights[i].name, weights[i].n_experts, -1).host;"
DISPLACES = "bool displaces_experts() const { return resident_only && host_experts > 0; }"
WEIGHTS_FN = ("static std::vector<ggml_sycl::moe_pp_ring_weight> ggml_sycl_inventory_ring_weights( "
              "const std::vector<ggml_sycl::placement_tensor_info> & detail) {")
WEIGHTS_BODY = ("{ std::vector<ggml_sycl::moe_pp_ring_weight> weights; for (const ggml_sycl::placement_tensor_info & "
                "info : detail) { if (moe_classify_tensor(info.name.c_str()) == MOE_TENSOR_UNKNOWN || "
                "!info.has_shape()) { continue; } const uint32_t uses = moe_pp_ring_uses_of_type(info.type); if (uses "
                "== MOE_PP_RING_USE_NONE) { continue; } ggml_sycl::moe_pp_ring_weight weight; weight.name = "
                "info.name.c_str(); weight.n_experts = info.ne[2] > 0 ? info.ne[2] : 1; weight.uses = uses; "
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
    "if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch(device) != planning_n_ubatch) { (void) "
    "ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch); }",
    "ggml_sycl_stage_publish_no_ring(device);",
    "published_before.restore(device);",
    "ggml_sycl::zone_arena_ensure_options options = ggml_sycl_load_arena_options();",
    "options.give_back = &give_back;",
    "if (ggml_sycl::get_unified_cache_for_device(device) && "
    "!ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)) {",
    "return ggml_sycl_stage_ring_drop::REFUSED; }",
    "if (give_back.rebuilt) {",
    "} else if (give_back.held) {",
    'GGML_LOG_WARN("[SYCL-PLAN] device %d drops its PP MoE oneDNN scratch ring (%.1f MB): the plan made with it left %"',
    "device, ring_bytes / mb, check.host_experts, check.n_weights, zone);",
    "return ggml_sycl_stage_ring_drop::DROPPED; }",
)
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
READER_BODY = ("{ const uint64_t txn = g_sycl_plan_load_effect_txn; return txn != 0 && device >= 0 && device < "
               "GGML_SYCL_MAX_DEVICES && g_sycl_ring_dropped_load_txn[device] == txn; }")
TXN = "g_sycl_ring_dropped_load_txn"
TXN_WRITE = "g_sycl_ring_dropped_load_txn[device] = g_sycl_plan_load_effect_txn;"
DRY_FN = ("static bool ggml_sycl_plan_dry_run_drop_displacing_ring(int device, "
          "const std::vector<ggml_sycl::placement_tensor_info> & detail, const ggml_sycl::placement_plan & plan) {")
DRY_ORDER = (
    "const ggml_sycl::moe_pp_ring_resident_check check = ggml_sycl_plan_ring_resident_check(device, detail, plan);",
    "if (!check.displaces_experts()) { return false; }",
    "ggml_sycl_publish_no_pp_moe_onednn_ring(device);",
    'GGML_LOG_INFO("[PLAN-DRY-RUN] device=%d drops its PP MoE oneDNN scratch ring (%.1f MB), as a load would:',
    "return true; }",
)
DRY_FORBIDDEN = ("ensure_planned_arena_zones", TXN, "give_back", "ggml_sycl_stage_")
DRY_PLANS_FN = "static std::vector<ggml_sycl::placement_plan> ggml_sycl_plan_dry_run_plans("
DRY_SITES = (
    "plans.push_back(plan_multi()); if (ggml_sycl_plan_dry_run_drop_displacing_ring(i, detail, plans.back())) { "
    "plans.back() = plan_multi(); }",
    "plans.push_back(plan_single()); if (ggml_sycl_plan_dry_run_drop_displacing_ring(i, detail, plans.back())) { "
    "plans.back() = plan_single(); }",
)
DRY_CALL = "ggml_sycl_plan_dry_run_drop_displacing_ring("
ENSURE_FN = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
             "const zone_arena_ensure_options & options) {")
ENSURE_ORDER = (
    "bool giving_back = false; if (options.give_back != nullptr) { *options.give_back = {}; "
    "options.give_back->need_runtime = runtime_zone; options.give_back->have_runtime = arena_active() ? "
    "zone_capacity(vram_zone_id::RUNTIME) : 0; } if (arena_active()) {",
    "const bool surplus_asked = options.give_back != nullptr && zone_arena_runtime_surplus(arena_zones, "
    "planned_zones);",
    "if (zones_sufficient && !surplus_asked) {",
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
ENSURE_GIVE_BACK_USES = 8
SURPLUS = ("bool zone_arena_runtime_surplus(const zone_arena_sizes & have, const zone_arena_sizes & need) { return "
           "have.scratch >= need.scratch && have.onednn >= need.onednn && have.runtime > need.runtime; }")
GIVE_BACK_SIZES = ("zone_arena_sizes zone_arena_give_back_sizes(const zone_arena_sizes & have, const zone_arena_sizes & "
                   "need) { return { std::max(have.scratch, need.scratch), std::max(have.onednn, need.onednn), "
                   "need.runtime }; }")
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
    need_ordered(rule, (RULE_SKIP, RULE_MIXED, RULE_ONLY, RULE_RETURN, RULE_COUNT), "R1",
                 "the check skips no-use weights, is resident-only only for RESIDENT alone, then counts host experts")
    need_once(rc, DISPLACES, "R1", "a ring displaces experts only when resident-only with a host expert")
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


def check_stage(s: Sources) -> None:
    be = s.sq["backend"]
    # S1
    stage = fn(be, STAGE_FN, "S1", "the stage helper")
    need_ordered(stage, STAGE_ORDER, "S1", "the stage helper's order")
    need_once(stage, PLANNING, "S1", "the planning shape the dense scratch goes back to")
    if fn(be, NO_RING_FN, "S1", "the no-ring publish") != squash(NO_RING_BODY):
        raise ContractError("FAIL [pin S1]: publishing no ring must zero every slot and the depth")
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
                [("weights[i].n_experts, -1).host;", "weights[i].n_experts, 0).host;")])
    expect_fail("displaces-any-host", "R1", C,
                [("return resident_only && host_experts > 0;", "return host_experts > 0;")])
    # W1, D1: the weight list and the one decision.
    expect_fail("weights-without-shape", "W1", B,
                [("MOE_TENSOR_UNKNOWN || !info.has_shape()) {", "MOE_TENSOR_UNKNOWN) {")])
    expect_fail("weights-one-expert", "W1", B,
                [("weight.n_experts = info.ne[2] > 0 ? info.ne[2] : 1;", "weight.n_experts = 1;")])
    expect_fail("decision-ignores-depth", "D1", B,
                [("    if (ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(device) == 0) {\n"
                  "        return {};\n    }\n    const std::vector<ggml_sycl::moe_pp_ring_weight> weights",
                  "    const std::vector<ggml_sycl::moe_pp_ring_weight> weights")])
    expect_fail("dry-run-own-rule", "D1", B,
                [("check = ggml_sycl_plan_ring_resident_check(device, detail, plan);",
                  "check = ggml_sycl::moe_pp_ring_check_resident_only(plan, nullptr, 0);")])
    # S1-S4: the stage.
    expect_fail("terms-not-restored", "S1", B, [("    published_before.restore(device);\n", "")])
    expect_fail("dense-left-at-micro-batch", "S1", B,
                [("        (void) ggml_sycl_publish_load_ubatch_scratch(device, planning_n_ubatch);\n", "")])
    expect_fail("ensure-without-give-back", "S1", B,
                [("    options.give_back                            = &give_back;\n", "")])
    expect_fail("refused-ensure-drops", "S1", B,
                [("        return ggml_sycl_stage_ring_drop::REFUSED;\n    }\n    constexpr double mb",
                  "    }\n    constexpr double mb")])
    expect_fail("warn-without-host-count", "S1", B,
                [("device, ring_bytes / mb, check.host_experts, check.n_weights, zone);",
                  "device, ring_bytes / mb, int64_t(0), check.n_weights, zone);")])
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
    expect_fail("no-drop-record", "S4", B,
                [("        g_sycl_ring_dropped_load_txn[device] = g_sycl_plan_load_effect_txn;\n", "")])
    expect_fail("direct-entry-sticky", "S4", B,
                [("    return txn != 0 && device >= 0 && device < GGML_SYCL_MAX_DEVICES &&",
                  "    return device >= 0 && device < GGML_SYCL_MAX_DEVICES &&")])
    expect_fail("late-pass-asks-for-ring", "S4", B,
                [("    if (ggml_sycl_stage_ring_dropped_by_this_load(ctx->device) &&\n"
                  "        ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx->device) != 0) {\n"
                  "        ggml_sycl_stage_publish_no_ring(ctx->device);\n",
                  "    if (false) {\n")])
    # Y1: the dry run.
    expect_fail("dry-run-keeps-ring-plan", "Y1", B, [("                plans.back() = plan_single();\n", "")])
    expect_fail("dry-run-gives-back", "Y1", B,
                [("    ggml_sycl_publish_no_pp_moe_onednn_ring(device);\n    GGML_LOG_INFO(\n",
                  "    ggml_sycl_publish_no_pp_moe_onednn_ring(device);\n"
                  "    (void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, nullptr, {});\n"
                  "    GGML_LOG_INFO(\n")])
    # Z1, Z2: the cache and the zone rules.
    expect_fail("surplus-without-record", "Z1", U,
                [("options.give_back != nullptr && zone_arena_runtime_surplus(arena_zones, planned_zones);",
                  "zone_arena_runtime_surplus(arena_zones, planned_zones);")])
    expect_fail("held-unmarked", "Z1", U, [("            options.give_back->held = true;\n", "")])
    expect_fail("held-rebuilds", "Z1", U,
                [("        if (surplus_asked && !zone_arena_rebuild_allowed(holders)) {\n",
                  "        if (false) {\n")])
    expect_fail("rebuilt-unmarked", "Z1", U, [("        options.give_back->rebuilt = true;\n", "")])
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
    print("PASS: a resident-only PP MoE ring that displaced experts is dropped, its RUNTIME surplus given back when "
          "nothing holds the arena, and the device planned once more; a ring with a staged use is kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
