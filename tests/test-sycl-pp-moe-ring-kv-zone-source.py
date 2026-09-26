#!/usr/bin/env python3
"""Pin how the runtime transaction grows the PP MoE oneDNN ring past the RUNTIME
zone (llama.cpp-u1bb).

The arithmetic -- where each slot goes and whether it is admitted -- is
pp_moe_onednn_admit_ring(), host-tested by
ggml/src/ggml-sycl/tests/test-pp-moe-ring-admission.cpp. What that test cannot
see is whether the transaction feeds it the right numbers at the right moment
and acts on the answer, which is what this file pins:

  * the ring is re-planned after KV is admitted, against the final plan's KV
    and the same live KV capacity (ggml_sycl_kv_capacity_live) the KV re-fit
    and the "-c" hints read, and a rollback is not re-admitted;
  * KV wins: that capacity counts the ring's KV-zone slots on this device as
    free, and a re-fit forces the ring to be re-admitted after KV;
  * the arena refusal is the admission's answer, the placement is published
    before the ceiling and restored with it on a refusal;
  * the KV-zone bytes are charged to the plan's vram_bytes;
  * the MMID pools, placed in the RUNTIME zone after the ring, are materialized
    only for a route that can run them, and otherwise the ring leaves them room;
  * the cache allocates a flagged slot from the KV zone, never spilling past
    the arena, and does not size the RUNTIME zone for it;
  * what the ring holds in the KV zone is read from its slots, not from the
    planned placement, and a model load releases a ring that holds any;
  * the KV-zone part must fit the zone's largest free block, and a refusal
    never names the -ub it refuses.

A source gate authored after the fix goes green on every tree that has it, so
every check below is witnessed by a mutation that must turn it red.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_CPP = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
UNIFIED_CACHE_CPP = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"

TRANSACTION_SIGNATURE = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction"
REPLAN_SIGNATURE = "static ggml_sycl_ring_replan_result ggml_sycl_replan_pp_moe_onednn_ring("
LIVE_HINT_SIGNATURE = "static uint32_t ggml_sycl_largest_fitting_n_ctx_live"
KV_CAPACITY_SIGNATURE = "static size_t ggml_sycl_kv_capacity_live("
KV_WITH_SLACK_SIGNATURE = "static size_t ggml_sycl_device_kv_bytes_with_slack("
RESERVE_SIGNATURE = "bool unified_cache::reserve_pp_moe_onednn_scratch("
RUNTIME_REQUIREMENT_SIGNATURE = "bool unified_cache_get_planned_runtime_zone_requirement("
KV_ZONE_BYTES_SIGNATURE = "static size_t unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes(int device_id) {"
HELD_SIGNATURE = "size_t unified_cache::pp_moe_onednn_kv_zone_bytes_held() {"
HELD_FREE_SIGNATURE = "size_t unified_cache_get_pp_moe_onednn_kv_zone_bytes_held(int device_id) {"


def function(text: str, signature: str) -> str:
    """One brace-delimited body from `signature`, skipping braces in comments
    and literals (same helper as test-sycl-kv-layer-sizing-source.py)."""
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments, keeping string and char literals intact."""
    out: list[str] = []
    state = "code"
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch in "\"'":
                state = "string" if ch == '"' else "char"
            out.append(ch)
        elif state == "line":
            if ch == "\n":
                state = "code"
                out.append(ch)
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        else:
            out.append(ch)
            if ch == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                state = "code"
        i += 1
    return "".join(out)


def code_of(text: str, signature: str) -> str | None:
    if signature not in text:
        return None
    return strip_comments(function(text, signature))


def transaction_violations(sycl_cpp: str) -> list[str]:
    txn = code_of(sycl_cpp, TRANSACTION_SIGNATURE)
    if txn is None:
        return ["the transaction is missing"]
    found: list[str] = []

    calls = re.findall(r"ggml_sycl_replan_pp_moe_onednn_ring\(([^;]*?)\)\s*(?:;|==)", txn)
    rollbacks = [c for c in calls if re.match(r"\s*ctx->device\s*,\s*pre_replan_pp_moe_ring_n_ubatch\b", c)]
    admitting = [c for c in calls if c not in rollbacks]
    if len(admitting) != 1 or not re.fullmatch(r"\s*ctx->device\s*,\s*next_kv_info\.n_ubatch\s*,\s*probe_mode\s*,"
                                               r"\s*&\s*ring_kv_zone\s*", admitting[0]):
        found.append("the ring is not re-planned with the transaction's KV-zone inputs")
    if len(rollbacks) != 4 or any("ring_kv_zone" in c for c in rollbacks):
        found.append("a ring rollback is re-admitted against the KV zone")

    call_at = txn.find("&ring_kv_zone)")
    final_residency = txn.find("if (!replan_ok) {")
    capacity = re.search(r"ring_kv_zone\.kv_capacity_bytes\s*=\s*ggml_sycl_kv_capacity_live\(\s*next_plan\s*,"
                         r"\s*ctx->device\s*,\s*admitted_kv\s*,\s*ctx->device\s*\)\s*;", txn)
    if capacity is None or not 0 <= final_residency < capacity.start() < call_at:
        found.append("the ring is not admitted against the live KV capacity the final plan's KV was fitted to")
    kv_bytes = re.search(r"ring_kv_zone\.kv_bytes\s*=\s*ggml_sycl_device_kv_bytes_with_slack\(\s*next_plan\s*,"
                         r"\s*ctx->device\s*\)\s*;", txn)
    if kv_bytes is None or not 0 <= final_residency < kv_bytes.start() < call_at:
        found.append("the ring is not admitted against the final plan's KV")

    refit = re.search(r"plan_runtime_kv_residency\(in\)", txn)
    push = re.search(r"in\.available\.push_back\(\s*ggml_sycl_kv_capacity_live\(\s*next_plan\s*,\s*device\s*,"
                     r"\s*nullptr\s*,\s*ctx->device\s*\)\s*\)\s*;", txn)
    if push is None or refit is None or push.start() > refit.start():
        found.append("the KV re-fit does not count this device's ring KV-zone slots as free")
    if not re.search(r"const\s+size_t\s+ring_kv_zone_bytes\s*=\s*ggml_sycl::"
                     r"unified_cache_get_pp_moe_onednn_kv_zone_bytes_held\(\s*ctx->device\s*\)\s*;", txn):
        found.append("the KV re-fit does not read this device's ring KV-zone slots")
    readmit = re.search(r"ring_readmit\s*=\s*ring_kv_zone_bytes\s*>\s*0\s*;", txn)
    if readmit is None or refit is None or not push or not push.start() < readmit.start() < refit.start() or \
            not re.search(r"ring_kv_zone\.readmit\s*=\s*ring_readmit\s*;", txn):
        found.append("a re-fit that freed the ring's KV-zone slots does not force the ring to be re-admitted")

    probe_at = txn.find("if (probe_mode) {", call_at)
    charge = re.search(r"const\s+size_t\s+charge\s*=\s*ggml_sycl::unified_cache_get_pp_moe_onednn_kv_zone_bytes_held"
                       r"\(\s*ring_devices\[i\]\s*\)\s*;\s*next_plan\.vram_bytes\s*\+=\s*charge\s*;", txn)
    refused = txn.find('return refuse("PP MoE oneDNN scratch ring does not fit");')
    if charge is None or not 0 <= refused < charge.start() < probe_at:
        found.append("the ring's KV-zone bytes are not charged to the plan's vram_bytes")

    if not re.search(r"const\s+bool\s+mmid_route_reachable\s*=\s*ggml_sycl_moe_mmid_route_reachable\(\s*\*ctx\s*\)\s*;",
                     txn):
        found.append("the transaction does not ask whether the MMID route is reachable")
    if not re.search(r"if\s*\(\s*!stable_mmid\s*&&\s*mmid_route_reachable\s*&&\s*"
                     r"!ggml_sycl_materialize_published_mmid_workspaces\(", txn):
        found.append("the MMID pools are materialized for a route that cannot run them")
    pending = re.search(r"if\s*\(\s*mmid_route_reachable\s*&&\s*!ggml_sycl_same_mmid_workspace_plan\(\s*\*current->plan"
                        r"\s*,\s*next_plan\s*\)\s*\)\s*\{[^{}]*\{\s*if\s*\(\s*workspace\.owner_device\s*==\s*ctx->device"
                        r"\s*\)\s*\{\s*ring_kv_zone\.runtime_pending_bytes\s*\+=\s*workspace\.device_pool_bytes\s*;", txn)
    if pending is None or pending.start() > call_at:
        found.append("the ring does not leave the RUNTIME zone room for the MMID pools placed after it")
    return found


def replan_violations(sycl_cpp: str) -> list[str]:
    body = code_of(sycl_cpp, REPLAN_SIGNATURE)
    if body is None:
        return ["the ring re-plan is missing"]
    found: list[str] = []
    if not re.search(r"n_ubatch\s*==\s*ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch\(\s*device\s*\)\s*&&"
                     r"\s*!\s*\(\s*kv_zone\s*&&\s*kv_zone->readmit\s*\)\s*\)\s*\{", body):
        found.append("the idempotent early return ignores a forced re-admission")

    admit = re.search(r"if\s*\(\s*arena\s*\)\s*\{(.*?)admission\s*=\s*ggml_sycl::pp_moe_onednn_admit_ring\(\s*admit_in"
                      r"\s*\)\s*;", body, re.S)
    if admit is None:
        found.append("the arena route does not ask pp_moe_onednn_admit_ring")
    else:
        inputs = admit.group(1)
        for pattern, what in (
                (r"kv_zone_available_bytes\s*=\s*kv_zone\s*\?\s*kv_zone->kv_capacity_bytes\s*:\s*"
                 r"std::numeric_limits<size_t>::max\(\)\s*;", "the live KV capacity"),
                (r"kv_admitted_bytes\s*=\s*kv_zone\s*\?\s*kv_zone->kv_bytes\s*:\s*0\s*;", "the plan's KV"),
                (r"kv_zone_largest_block_bytes\s*=\s*kv_zone\s*\?\s*cache->zone_largest_free\(\s*ggml_sycl::"
                 r"vram_zone_id::KV\s*\)\s*:\s*std::numeric_limits<size_t>::max\(\)\s*;",
                 "the KV zone's largest free block"),
                (r"compute_reserve_bytes_per_row\s*=\s*kv_zone\s*\?\s*k_pp_moe_ring_compute_reserve_bytes_per_row\s*:"
                 r"\s*0\s*;", "the compute-buffer reserve"),
                (r"runtime_available_bytes\s*=\s*runtime_net_bytes\s*;", "the RUNTIME zone less what follows the ring")):
            if not re.search(pattern, inputs, re.S):
                found.append(f"the admission is not given {what}")
        if not re.search(r"runtime_pending\s*=\s*arena\s*&&\s*kv_zone\s*\?\s*kv_zone->runtime_pending_bytes\s*:\s*0\s*;"
                         r"\s*const\s+size_t\s+runtime_net_bytes\s*=\s*capacity_bytes\s*>\s*runtime_pending\s*\?\s*"
                         r"capacity_bytes\s*-\s*runtime_pending\s*:\s*0\s*;", body):
            found.append("the admission is not given the RUNTIME zone less what follows the ring")

    guard = re.search(r"if\s*\(\s*arena\s*&&\s*!\s*admission\.admit\s*\)\s*\{\s*return\s+refuse_and_restore\(\)\s*;",
                      body)
    publish = re.search(r"unified_cache_set_planned_pp_moe_onednn_kv_zone_slots\(\s*device\s*,\s*"
                        r"admission\.activation_in_kv_zone\s*,\s*admission\.output_in_kv_zone\s*\)\s*;", body)
    ceiling = re.search(r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*device\s*,\s*weight_slot_bytes\s*,\s*"
                        r"new_activation_slot_bytes", body)
    if guard is None:
        found.append("the arena refusal is not the admission's answer")
    if publish is None or ceiling is None or guard is None or not guard.end() < publish.start() < ceiling.start():
        found.append("the slots' zones are not published after admission and before the ceiling")

    release = body.find("cache->release_pp_moe_onednn_scratch_ring()")
    if admit is not None and not 0 <= release < admit.start():
        found.append("the admission reads the KV zone before the old ring is released")
    if not re.search(r"if\s*\(\s*fits\s*>=\s*n_ubatch\s*&&[^{]*\{[^}]*\}\s*else\s+if\s*\(\s*fits\s*>=\s*32\s*&&[^{]*\{"
                     r"[^}]*largest -ub that fits", body):
        found.append("a refusal can name the -ub it refuses")
    if not re.search(r"zone_name\s*,\s*runtime_net_bytes\s*/\s*mb\s*,\s*fits_clause\s*\)", body) or \
            not re.search(r"runtime_net_bytes\s*=\s*capacity_bytes\s*>\s*runtime_pending\s*\?", body):
        found.append("the refusal does not print the RUNTIME bytes the admission used")

    restore = re.search(r"auto\s+refuse_and_restore\s*=\s*\[&\]\(\)\s*->\s*ggml_sycl_ring_replan_result\s*\{\s*"
                        r"ggml_sycl::unified_cache_set_planned_pp_moe_onednn_kv_zone_slots\(\s*device\s*,\s*"
                        r"old_activation_in_kv_zone\s*,\s*old_output_in_kv_zone\s*\)\s*;", body)
    if restore is None or not re.search(r"old_activation_in_kv_zone\s*=\s*ggml_sycl::"
                                        r"unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone\(\s*device\s*\)",
                                        body):
        found.append("a refused re-plan does not restore the old slots' zones")
    return found


def hint_violations(sycl_cpp: str) -> list[str]:
    live = code_of(sycl_cpp, LIVE_HINT_SIGNATURE)
    if live is None:
        return ["the live -c hint is missing"]
    if not re.search(r"ggml_sycl_kv_capacity_live\(\s*plan\s*,\s*d\s*,\s*admitted\s*,\s*ring_device\s*\)", live):
        return ["the -c hint does not count the ring's KV-zone slots as free"]
    return []


def capacity_violations(sycl_cpp: str) -> list[str]:
    found: list[str] = []
    capacity = code_of(sycl_cpp, KV_CAPACITY_SIGNATURE)
    if capacity is None:
        return ["ggml_sycl_kv_capacity_live is missing"]
    if not re.search(r"size_t\s+capacity\s*=\s*ggml_sycl::unified_cache_kv_vram_available\(", capacity):
        found.append("the KV capacity is not the live KV headroom")
    if not re.search(r"if\s*\(\s*admitted\s*\)\s*\{\s*capacity\s*\+=\s*admitted->device_kv_vram_bytes\(\s*device"
                     r"\s*\)\s*;", capacity):
        found.append("the KV capacity counts an admitted context's allocated KV as used")
    if not re.search(r"if\s*\(\s*device\s*==\s*ring_device\s*\)\s*\{\s*capacity\s*\+=\s*ggml_sycl::"
                     r"unified_cache_get_pp_moe_onednn_kv_zone_bytes_held\(\s*device\s*\)\s*;", capacity):
        found.append("the KV capacity does not count the ring's KV-zone slots as free")

    with_slack = code_of(sycl_cpp, KV_WITH_SLACK_SIGNATURE)
    if with_slack is None or not re.search(r"return\s+plan\.device_kv_vram_bytes\(\s*device\s*\)\s*\+\s*n_resident"
                                           r"\s*\*\s*ggml_sycl::kv_alloc_slack_per_layer\s*;", with_slack):
        found.append("the plan's KV does not count the allocator's per-layer slack")
    return found


def cache_violations(cache_cpp: str) -> list[str]:
    found: list[str] = []
    reserve = code_of(cache_cpp, RESERVE_SIGNATURE)
    if reserve is None:
        return ["reserve_pp_moe_onednn_scratch is missing"]
    if not re.search(r"req\.intent\.constraints\.prefer_vram_zone\s*=\s*zone\s*;", reserve) or \
            not re.search(r"req\.intent\.constraints\.forbid_vram_zone_spill\s*=\s*true\s*;", reserve):
        found.append("a ring slot is not allocated from its planned zone without spilling")
    for kind in ("activation", "output"):
        if not re.search(rf"{kind}_zone\s*=\s*unified_cache_get_planned_pp_moe_onednn_{kind}_in_kv_zone\(\s*device_id"
                         rf"\s*\)\s*\?\s*vram_zone_id::KV\s*:\s*vram_zone_id::RUNTIME\s*;", reserve) or \
                not re.search(rf"allocate_buffer\(\s*{kind}_slot_bytes\s*,\s*\"pp_moe_onednn_{kind}\"\s*,\s*{kind}_zone"
                              rf"\s*,", reserve):
            found.append(f"the {kind} slot does not follow its planned zone")
    if not re.search(r"allocate_buffer\(\s*weight_slot_bytes\s*,\s*\"pp_moe_onednn_weight\"\s*,\s*"
                     r"vram_zone_id::RUNTIME\s*,", reserve):
        found.append("the weight slot can leave the RUNTIME zone")

    for kind in ("activation", "output"):
        if not re.search(rf"slot\.{kind}_in_kv_zone\s*=\s*arena_active\(\)\s*&&\s*{kind}_zone\s*==\s*vram_zone_id::KV\s*;",
                         reserve):
            found.append(f"the {kind} slot does not record the zone it was allocated from")
    held = code_of(cache_cpp, HELD_SIGNATURE)
    if held is None or \
            not re.search(r"\{\s*&pp_moe_onednn_scratch_slots_\s*,\s*&pp_moe_onednn_retired_slots_\s*\}", held) or \
            not re.search(r"slot\.activation_in_kv_zone\s*&&\s*slot\.activation\s*\?\s*slot\.activation_size", held) or \
            not re.search(r"slot\.output_in_kv_zone\s*&&\s*slot\.output\s*\?\s*slot\.output_size", held):
        found.append("the ring's KV-zone bytes are not read from all of its slots")
    held_free = code_of(cache_cpp, HELD_FREE_SIGNATURE)
    if held_free is None or "get_existing_unified_cache_for_device(device_id)" not in held_free:
        found.append("reading the ring's KV-zone bytes can create a cache")

    requirement = code_of(cache_cpp, RUNTIME_REQUIREMENT_SIGNATURE)
    kv_bytes = code_of(cache_cpp, KV_ZONE_BYTES_SIGNATURE)
    if requirement is None or kv_bytes is None or \
            "unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes(device_id)" not in requirement:
        found.append("the RUNTIME zone is sized for ring slots that live in the KV zone")
    return found


def load_plan_violations(sycl_cpp: str) -> list[str]:
    code = strip_comments(sycl_cpp)
    load = re.search(r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*ctx->device\s*,[^;]*;(.*?)"
                     r"ggml_sycl::unified_cache_set_planned_pp_moe_onednn_kv_zone_slots\(\s*ctx->device\s*,\s*false"
                     r"\s*,\s*false\s*\)\s*;", code, re.S)
    if load is None:
        return ["a model load keeps a previous context's KV-zone placement"]
    if not re.search(r"const\s+size_t\s+(\w+)\s*=\s*ggml_sycl::unified_cache_get_pp_moe_onednn_kv_zone_bytes_held\("
                     r"\s*ctx->device\s*\)\s*;\s*if\s*\(\s*\1\s*>\s*0\s*\)\s*\{[^{}]*"
                     r"->release_pp_moe_onednn_scratch_ring\(\)", load.group(1)):
        return ["a model load keeps a previous context's ring in the KV zone"]
    return []


def all_violations(sycl_cpp: str, cache_cpp: str) -> list[str]:
    return (transaction_violations(sycl_cpp) + replan_violations(sycl_cpp) + hint_violations(sycl_cpp) +
            capacity_violations(sycl_cpp) + load_plan_violations(sycl_cpp) + cache_violations(cache_cpp))


def test_ring_kv_zone_wiring_holds() -> None:
    assert all_violations(GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()) == []


# (label, file, old, new, expected violation). Each mutation undoes one part of
# the wiring the way a plausible edit would, and must turn its check red.
MUTATIONS = [
    ("rollback-free call", "sycl",
     "next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)", "next_kv_info.n_ubatch, probe_mode)",
     "the ring is not re-planned with the transaction's KV-zone inputs"),
    ("rollback re-admitted", "sycl",
     "(void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch);",
     "(void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch, false, &ring_kv_zone);",
     "a ring rollback is re-admitted against the KV zone"),
    ("capacity ignores the admitted context", "sycl",
     "ggml_sycl_kv_capacity_live(next_plan, ctx->device, admitted_kv, ctx->device);",
     "ggml_sycl_kv_capacity_live(next_plan, ctx->device, nullptr, ctx->device);",
     "the ring is not admitted against the live KV capacity the final plan's KV was fitted to"),
    ("capacity read before the final plan", "sycl",
     "    ring_kv_zone.kv_capacity_bytes = ggml_sycl_kv_capacity_live(next_plan, ctx->device, admitted_kv, ctx->device);\n",
     "",
     "the ring is not admitted against the live KV capacity the final plan's KV was fitted to"),
    ("ring admitted against no KV", "sycl",
     "ring_kv_zone.kv_bytes          = ggml_sycl_device_kv_bytes_with_slack(next_plan, ctx->device);",
     "ring_kv_zone.kv_bytes          = 0;",
     "the ring is not admitted against the final plan's KV"),
    ("re-fit without the add-back", "sycl",
     "ggml_sycl_kv_capacity_live(next_plan, device, nullptr, ctx->device));",
     "ggml_sycl_kv_capacity_live(next_plan, device, nullptr, -1));",
     "the KV re-fit does not count this device's ring KV-zone slots as free"),
    ("no forced re-admission", "sycl",
     "        ring_readmit = ring_kv_zone_bytes > 0;\n", "",
     "a re-fit that freed the ring's KV-zone slots does not force the ring to be re-admitted"),
    ("no vram_bytes charge", "sycl",
     "            next_plan.vram_bytes += charge;\n", "",
     "the ring's KV-zone bytes are not charged to the plan's vram_bytes"),
    ("MMID ungated", "sycl",
     "if (!stable_mmid && mmid_route_reachable &&", "if (!stable_mmid &&",
     "the MMID pools are materialized for a route that cannot run them"),
    ("no MMID room", "sycl",
     "                ring_kv_zone.runtime_pending_bytes += workspace.device_pool_bytes;\n", "",
     "the ring does not leave the RUNTIME zone room for the MMID pools placed after it"),
    ("idempotent despite a re-fit", "sycl",
     " &&\n        !(kv_zone && kv_zone->readmit)) {", ") {",
     "the idempotent early return ignores a forced re-admission"),
    ("no reserve", "sycl",
     "kv_zone ? k_pp_moe_ring_compute_reserve_bytes_per_row : 0;", "0;",
     "the admission is not given the compute-buffer reserve"),
    ("KV zone admitted against nothing", "sycl",
     "admit_in.kv_admitted_bytes       = kv_zone ? kv_zone->kv_bytes : 0;",
     "admit_in.kv_admitted_bytes       = 0;",
     "the admission is not given the plan's KV"),
    ("KV zone read raw", "sycl",
     "kv_zone ? kv_zone->kv_capacity_bytes : std::numeric_limits<size_t>::max();",
     "kv_zone ? cache->zone_available(ggml_sycl::vram_zone_id::KV) : std::numeric_limits<size_t>::max();",
     "the admission is not given the live KV capacity"),
    ("no largest free block", "sycl",
     "kv_zone ? cache->zone_largest_free(ggml_sycl::vram_zone_id::KV) : std::numeric_limits<size_t>::max();",
     "std::numeric_limits<size_t>::max();",
     "the admission is not given the KV zone's largest free block"),
    ("names the refused -ub", "sycl",
     "if (fits >= n_ubatch && clause_len >= 0", "if (false && clause_len >= 0",
     "a refusal can name the -ub it refuses"),
    ("gross RUNTIME in the refusal", "sycl",
     "zone_name, runtime_net_bytes / mb, fits_clause", "zone_name, capacity_bytes / mb, fits_clause",
     "the refusal does not print the RUNTIME bytes the admission used"),
    ("old RUNTIME-only guard", "sycl",
     "if (arena && !admission.admit) {", "if (arena && needed_total > capacity_bytes) {",
     "the arena refusal is not the admission's answer"),
    ("zones not restored", "sycl",
     "        ggml_sycl::unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, old_activation_in_kv_zone,\n"
     "                                                                         old_output_in_kv_zone);\n", "",
     "a refused re-plan does not restore the old slots' zones"),
    ("hint without the add-back", "sycl",
     "ggml_sycl_kv_capacity_live(plan, d, admitted, ring_device)", "ggml_sycl_kv_capacity_live(plan, d, admitted, -1)",
     "the -c hint does not count the ring's KV-zone slots as free"),
    ("capacity without the ring add-back", "sycl",
     "    if (device == ring_device) {\n", "    if (false) {\n",
     "the KV capacity does not count the ring's KV-zone slots as free"),
    ("capacity without the admitted add-back", "sycl",
     "        capacity += admitted->device_kv_vram_bytes(device);\n", "",
     "the KV capacity counts an admitted context's allocated KV as used"),
    ("capacity from a snapshot", "sycl",
     "size_t capacity = ggml_sycl::unified_cache_kv_vram_available(device, plan.multi_device);",
     "size_t capacity = plan.per_device_vram_budgets[device];",
     "the KV capacity is not the live KV headroom"),
    ("plan's KV without slack", "sycl",
     " + n_resident * ggml_sycl::kv_alloc_slack_per_layer;\n}", ";\n}",
     "the plan's KV does not count the allocator's per-layer slack"),
    ("load keeps a KV-zone ring", "sycl",
     "if (ring_cache && ring_cache->release_pp_moe_onednn_scratch_ring()) {", "if (false) {",
     "a model load keeps a previous context's ring in the KV zone"),
    ("load keeps the placement", "sycl",
     "    ggml_sycl::unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(ctx->device, false, false);\n", "",
     "a model load keeps a previous context's KV-zone placement"),
    ("ring slots always RUNTIME", "cache",
     "req.intent.constraints.prefer_vram_zone       = zone;",
     "req.intent.constraints.prefer_vram_zone       = vram_zone_id::RUNTIME;",
     "a ring slot is not allocated from its planned zone without spilling"),
    ("activation slot ignores its zone", "cache",
     '"pp_moe_onednn_activation",\n                                                        activation_zone,',
     '"pp_moe_onednn_activation",\n                                                        vram_zone_id::RUNTIME,',
     "the activation slot does not follow its planned zone"),
    ("output slot ignores its zone", "cache",
     "? vram_zone_id::KV : vram_zone_id::RUNTIME;\n", "? vram_zone_id::RUNTIME : vram_zone_id::RUNTIME;\n",
     "the output slot does not follow its planned zone"),
    ("weight slot follows the activation zone", "cache",
     '"pp_moe_onednn_weight", vram_zone_id::RUNTIME,', '"pp_moe_onednn_weight", activation_zone,',
     "the weight slot can leave the RUNTIME zone"),
    ("activation zone not recorded", "cache",
     "slot.activation_in_kv_zone = arena_active() && activation_zone == vram_zone_id::KV;",
     "slot.activation_in_kv_zone = false;",
     "the activation slot does not record the zone it was allocated from"),
    ("held ignores retired slots", "cache",
     "{ &pp_moe_onednn_scratch_slots_, &pp_moe_onednn_retired_slots_ }", "{ &pp_moe_onednn_scratch_slots_ }",
     "the ring's KV-zone bytes are not read from all of its slots"),
    ("held creates a cache", "cache",
     "unified_cache * cache = get_existing_unified_cache_for_device(device_id);\n    return cache ? cache->pp_moe",
     "unified_cache * cache = get_unified_cache_for_device(device_id);\n    return cache ? cache->pp_moe",
     "reading the ring's KV-zone bytes can create a cache"),
    ("RUNTIME sized for KV-zone slots", "cache",
     "unified_cache_get_planned_pp_moe_onednn_kv_zone_bytes(device_id)", "0",
     "the RUNTIME zone is sized for ring slots that live in the KV zone"),
]


@pytest.mark.parametrize("label,which,old,new,expected", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_mutation_is_witnessed(label: str, which: str, old: str, new: str, expected: str) -> None:
    sycl_cpp = GGML_SYCL_CPP.read_text()
    cache_cpp = UNIFIED_CACHE_CPP.read_text()
    source = sycl_cpp if which == "sycl" else cache_cpp
    assert old in source, f"{label}: mutation target not found -- update this witness to match the source"
    mutated = source.replace(old, new, 1)
    violations = all_violations(mutated, cache_cpp) if which == "sycl" else all_violations(sycl_cpp, mutated)
    assert any(expected in v for v in violations), f"{label}: expected {expected!r}, got {violations!r}"
