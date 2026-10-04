#!/usr/bin/env python3
"""oneDNN PP scratch admission source contract (llama.cpp-8ony).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: GPT-OSS 20B on the B50 (perplexity -c 512 -ub 512 --chunks 2) aborted on chunk 2 with a
DEQUANT-F16 breach on output.weight. The planner puts the LM head's f16 dequant in the RUNTIME-zone dense
dequant buffers (llama.cpp-479i) and keeps it out of the ONEDNN zone's sizing, but the dispatch asked only
"is this op a oneDNN PP candidate?", sent it to the oneDNN scratch (1104.6 MiB against a 256 MiB zone), and the
graph-entry walk skipped it on the same assumption. Once weights were resident the arena refused to grow the
zone, the scratch thrashed through unified-cache direct growth, and the planned dequant buffers it fell back to
had never been sized.

One fact, one source: whether an op's f16 copies are planned into the ONEDNN zone is decided by the zone the arena
was built with (zone_onednn_pp_scratch_planned over unified_cache_get_onednn_pp_pair_bound), and the op arm and
the graph-entry walk must ask that same question. The companion unit test (test-zone-sizing, Cases 14a-14f) proves
the arithmetic; this gate proves both consumers use it.

Two follow-on facts, pinned here too:
  * The direct unified_alloc that reserve_onednn_scratch made when the arena replan was refused (the
    "growing through the unified cache" path) is itself an unplanned allocation. An op the plan routes elsewhere
    is turned away at the one choke point every caller shares (acquire_onednn_pp_scratch), BEFORE reserve is
    asked, and reserve refuses a request that still exceeds the zone after its replan attempt instead of
    allocating around the plan.
  * A smaller request must never shrink a held scratch: reserve sizes the new pair with
    zone_onednn_scratch_reserve_target (per-component maximum with what is held), so a layer-0 request cannot
    replace the block a later op needed and force the regrowth that failed.

Three more facts, all gated below:
  * "the scratch supplies this op" is admission AND the type/env enablement (acquire_onednn_pp_scratch refuses
    every type but Q4_0 / Q8_0 / MXFP4 unless GGML_SYCL_ONEDNN_PP_UNIFIED_SCRATCH forces it, and every type when it
    is 0) AND the plan. The walk asked only admission + plan, so a K-quant op was skipped by the walk and refused by
    acquire, then drew a planned dequant buffer nothing had sized. One helper (ggml_sycl_onednn_pp_scratch_supplies)
    answers it for the op arm and the walk, from one function of (src0, src1, columns).
  * The unified kernel's oneDNN f16 route (Route A) fell back to a per-op ctx.pool() copy of the whole weight when
    acquire refused an over-zone op. It draws the planned dequant buffers instead, and the walk counts such a node
    (ggml_sycl_mul_mat_unified_pp_dequant_route), so the pair is sized before anything is submitted.
  * The op arm and the walk pass the pair's arguments to that one helper: the op its column tile (src1_ncols), the
    walk src1->ne[1]; both derive the element counts inside it.

A tensor consumed only by a row gather (token_embd.weight, GET_ROWS) is no MUL_MAT operand and is planned into
neither the dequant buffers nor the Q8_1 src1 buffer; a Q4_0 Mistral paid a 254 MB RUNTIME zone for it. The model
loader's own op table says which tensors are gathered, and its tied-embedding rule (an output head the file does not
carry is token_embd, treated as the output) says when a gathered tensor is also the head. That role travels in
ggml_sycl_tensor_info::get_rows_only, is copied into the inventory the planner sees, and is honoured by the pure
classifier; test-zone-sizing Case 14j proves the classifier, this gate proves the producer, the carrier and the
plumbing.

Run with --self-test to prove every check fires against a mutant of the thing it forbids.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--cache", default=str(sycl / "unified-cache.cpp"))
parser.add_argument("--cache-hpp", default=str(sycl / "unified-cache.hpp"))
parser.add_argument("--zone-sizing", default=str(sycl / "zone-sizing.cpp"))
parser.add_argument("--model", default=str(root / "src/llama-model.cpp"))
parser.add_argument("--placement", default=str(sycl / "onednn-pp-placement.hpp"))
parser.add_argument("--unified-types", default=str(sycl / "unified-types.hpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
parser.add_argument("--dispatch", default=str(sycl / "dispatch.hpp"))
parser.add_argument("--sycl-header", default=str(root / "ggml/include/ggml-sycl.h"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    """Remove C/C++ comments, keeping string literals and the line count."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def function_body(source, signature_regex):
    """Return the brace-balanced body of the first function matching the regex."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def statements_calling(body, callee):
    """Every ';'-terminated statement (or '{'-terminated condition) of `body` that contains `callee`."""
    found = []
    for m in re.finditer(re.escape(callee), body):
        lo = max(body.rfind(";", 0, m.start()), body.rfind("{", 0, m.start()), body.rfind("}", 0, m.start()))
        hi_candidates = [x for x in (body.find(";", m.end()), body.find("{", m.end())) if x >= 0]
        hi = min(hi_candidates) if hi_candidates else len(body)
        found.append(body[lo + 1:hi])
    return found


HELPER = "ggml_sycl_onednn_pp_scratch_planned_bytes("
# The already-reserved reuse test in reserve_onednn_scratch (its first code line).
REUSE_TEST = "if (onednn_weights_scratch_ && onednn_activations_scratch_ && onednn_weights_scratch_size_ >= weights_size"


def evaluate(backend, cache, cache_hpp, zone_sizing, model, header, common, dispatch, placement, utypes):
    results = {}
    bytes_helper = function_body(
        backend, r"static bool ggml_sycl_onednn_pp_scratch_planned_bytes\([^)]*\)\s*\{")
    op_sycl = function_body(backend, r"inline void ggml_sycl_op_mul_mat_sycl\([^)]*\)\s*try\s*\{")
    dq_walk = function_body(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\([^)]*\)\s*\{")
    results["anchor: the by-bytes admission core exists"] = bytes_helper is not None
    results["anchor: op_mul_mat_sycl exists"] = op_sycl is not None
    results["anchor: the dequant graph walk exists"] = dq_walk is not None
    if None in (bytes_helper, op_sycl, dq_walk):
        return results

    # One source: the helper answers from the zone the arena was built with, through the pure predicate.
    results["the helper asks the pure predicate"] = "zone_onednn_pp_scratch_planned(" in bytes_helper
    # ABSENCE: it must not be answered from the stored planned bytes, which the runtime growth signal rewrites.
    results["the helper does not read the (mutable) planned scratchpad figure"] = \
        "unified_cache_get_planned_onednn_scratchpad_bytes" not in bytes_helper

    # The choke point: every caller reaches reserve_onednn_scratch through acquire_onednn_pp_scratch, and an op the
    # plan routes elsewhere must be refused there, before the reserve (whose replan attempt also bumps the stored
    # plan upward) is ever asked.
    acquire = function_body(backend, r"static bool acquire_onednn_pp_scratch\([^)]*\)\s*\{")
    results["anchor: acquire_onednn_pp_scratch exists"] = acquire is not None
    if acquire is not None:
        gate_at = acquire.find("ggml_sycl_onednn_pp_scratch_planned_bytes(")
        reserve_at = acquire.find("unified_cache_reserve_onednn_scratch(")
        results["acquire asks the plan before it asks for a reserve"] = 0 <= gate_at < reserve_at
        results["acquire turns an unplanned op away"] = \
            gate_at >= 0 and "return false" in acquire[gate_at:reserve_at if reserve_at > 0 else len(acquire)]

    reserve = function_body(cache, r"bool unified_cache::reserve_onednn_scratch\([^)]*\)\s*\{")
    results["anchor: reserve_onednn_scratch exists"] = reserve is not None
    if reserve is not None:
        starts = [m.start() for m in re.finditer(re.escape("if (total_needed > zone_cap) {"), reserve)]
        results["anchor: reserve compares the request with the zone"] = len(starts) >= 2
        if len(starts) >= 2:
            # The second comparison is the one AFTER the replan attempt; the first is the replan trigger itself.
            after_replan = function_body(reserve[starts[-1]:], r"if \(total_needed > zone_cap\) \{") or ""
            results["reserve refuses an over-zone request after its replan attempt"] = \
                "return finish(false)" in after_replan
            results["reserve does not allocate around the plan for an over-zone request"] = \
                "arena_zone_exhausted" not in after_replan
        results["reserve no longer labels an over-zone request a direct-growth cause"] = \
            "zone_undersized" not in reserve
        results["reserve sizes the pair with the never-shrink target"] = \
            "zone_onednn_scratch_reserve_target(" in reserve
        results["the never-shrink target runs before the already-reserved test"] = \
            0 <= reserve.find("zone_onednn_scratch_reserve_target(") < reserve.find(REUSE_TEST)

    # Both consumers ask it, every time they ask whether oneDNN PP supplies the f16 copies.
    op_candidate = re.search(r"const bool legacy_pp_scratch_candidate\s*=([^;]*);", op_sycl)
    results["anchor: the op arm's scratch candidate statement exists"] = op_candidate is not None
    walk_skips = statements_calling(dq_walk, "ggml_sycl_onednn_pp_candidate(")

    # ---- ONE helper answers "the scratch supplies this op", for both consumers ----
    supplies_helper = function_body(
        backend, r"static bool ggml_sycl_onednn_pp_scratch_supplies\([^)]*\)\s*\{")
    enabled_fn = function_body(cache, r"bool onednn_pp_unified_scratch_enabled\(ggml_type type\)\s*\{")
    results["anchor: the shared supplies helper exists"] = supplies_helper is not None
    results["anchor: the type/env enablement function exists"] = enabled_fn is not None
    if supplies_helper is not None:
        results["the supplies helper asks the PP admission"] = "ggml_sycl_onednn_pp_candidate(" in supplies_helper
        results["the supplies helper asks the type/env enablement"] = \
            "onednn_pp_unified_scratch_enabled(" in supplies_helper
        results["the supplies helper asks the pure verdict"] = "zone_onednn_pp_scratch_supplies(" in supplies_helper
        results["the supplies helper derives the pair from src0 and the column count"] = \
            "ne[1]" in supplies_helper and "ne[0]" in supplies_helper
    if enabled_fn is not None:
        results["the enablement function asks the pure predicate"] = \
            "zone_onednn_pp_scratch_type_enabled(" in enabled_fn
    results["the enablement function is declared once, for the cache and the backend to share"] = \
        re.search(r"bool onednn_pp_unified_scratch_enabled\(ggml_type type\);", cache_hpp) is not None and \
        "static bool onednn_pp_unified_scratch_enabled(" not in backend

    # ---- the dequant plan covers the ops the scratch will not supply (the adapter marks them; the pure classifier decides)
    adapter = function_body(
        cache, r"std::vector<zone_tensor_desc> unified_cache_adapt_zone_inventory\([^)]*\)\s*\{")
    results["anchor: the zone-inventory adapter exists"] = adapter is not None
    if adapter is not None:
        results["the adapter marks the unified-kernel types for the conditional dequant plan"] = \
            "unified_kernel_serves_type(" in adapter and "dequant_f16_if_unsupplied_weight_bytes" in adapter and \
            "dequant_f16_if_unsupplied_src1_bytes_per_token" in adapter
        results["the adapter hands the classifier the type/env enablement, not its own copy"] = \
            "pp_scratch_type_enabled" in adapter and "onednn_pp_unified_scratch_enabled(" in adapter
        results["the adapter excludes expert stacks from the conditional mark too"] = \
            adapter.count("expert_tensor_role_from_tensor_name(") >= 3
    SUPPLIES = "ggml_sycl_onednn_pp_scratch_supplies("
    results["the op arm's candidate asks the shared supplies helper with its column tile"] = \
        bool(op_candidate) and SUPPLIES in op_candidate.group(1) and "src1_ncols" in op_candidate.group(1) and \
        "ctx.device" in op_candidate.group(1)
    results["the op arm no longer asks admission and plan separately"] = \
        bool(op_candidate) and "ggml_sycl_onednn_pp_candidate(" not in op_candidate.group(1) and \
        HELPER not in op_candidate.group(1)
    walk_supplies = statements_calling(dq_walk, SUPPLIES)
    results["the walk skips an op only through the shared supplies helper, with src1->ne[1]"] = \
        len(walk_supplies) >= 1 and all("src1->ne[1]" in x and "ctx.device" in x for x in walk_supplies)
    results["the walk no longer skips on admission and plan separately"] = \
        len(walk_skips) == 0 and HELPER not in dq_walk

    # ---- Route A draws the planned dequant buffers, and the walk counts such a node ----
    route_a = function_body(
        backend, r"static bool ggml_sycl_mul_mat_unified_pp_dequant_route\([^)]*\)\s*\{")
    results["anchor: the walk's Route A predicate exists"] = route_a is not None
    if route_a is not None:
        route_a_norm = re.sub(r"\s+", " ", route_a)
        results["the Route A predicate asks the pure verdict"] = "zone_unified_pp_draws_dequant(" in route_a
        results["the Route A predicate reads the router's unified answer"] = \
            "MatmulBackend::UnifiedKernel" in route_a
        results["the Route A predicate names the unified types"] = "should_use_unified(" in route_a
        results["the Route A predicate keeps the unified-dispatch gate"] = \
            "ggml_sycl_unified_dispatch_enabled()" in route_a
        results["the Route A predicate keeps every plain-src1 term"] = \
            all(t in route_a_norm for t in ("ggml_is_contiguous(src1)", "!ggml_is_transposed(src1)",
                                            "!ggml_is_permuted(src1)")) and "src1_plain = true" not in route_a_norm
        results["the Route A predicate passes the pure verdict its own inputs"] = \
            "(primary_unified, unified_type, src1_plain, pp_candidate, scratch_supplies)" in route_a_norm
    dq_walk_norm = re.sub(r"\s+", " ", dq_walk)
    results["the walk counts a Route A node with the answers it already has"] = \
        "ggml_sycl_mul_mat_unified_pp_dequant_route(src0, src1, primary, pp_candidate, supplied)" in dq_walk_norm
    results["the walk takes the PP admission from its one supplies call"] = "&pp_candidate)" in dq_walk_norm
    acq_at = backend.find("using_scratch = acquire_onednn_pp_scratch(")
    pool_at = backend.find("src0_f16_alloc.alloc(src0_elems)", acq_at) if acq_at >= 0 else -1
    results["anchor: Route A's acquire and pool fallback exist"] = 0 <= acq_at < pool_at
    if 0 <= acq_at < pool_at:
        seg = backend[acq_at:pool_at]
        seg_norm = re.sub(r"\s+", " ", seg)
        results["Route A draws the planned src0 dequant buffer before any pool copy"] = \
            re.search(r"ggml_sycl_dequant_f16_scratch\(\s*ctx\.dequant_f16_src0_scratch", seg) is not None
        results["Route A draws the planned src1 dequant buffer before any pool copy"] = \
            re.search(r"ggml_sycl_dequant_f16_scratch\(\s*ctx\.dequant_f16_src1_scratch", seg) is not None
        # The consumers run on the context's own in-order queue; a planned buffer is race-free on that queue only.
        results["Route A draws both planned buffers on the context's own queue"] = \
            "ctx.dequant_f16_src0_scratch, ctx.device, *ctx.stream(), src0," in seg_norm and \
            "ctx.dequant_f16_src1_scratch, ctx.device, *ctx.stream(), src0," in seg_norm
        arm_at = seg.find("GGML_SYCL_DEQUANT_F16_ARM")
        else_at = seg.find("#    else")
        results["Route A's planned draws sit inside the dequant-arm scope, the pool copy in its #else"] = \
            0 <= arm_at < seg.find("ggml_sycl_dequant_f16_scratch(") < else_at < pool_at - acq_at
        # Route A's acquire asks for exactly the helper's pair: N*K and M*K f16 elements.
        acq_end = backend.find(";", acq_at)
        acq_norm = re.sub(r"\s+", " ", backend[acq_at:acq_end])
        results["Route A's acquire asks for the pair the supplies helper derives"] = \
            "acquire_onednn_pp_scratch(ctx.device, src0->type, weights_bytes, activations_bytes, &weights_scratch, " \
            "&activations_scratch, pp_scratch_guard)" in acq_norm
        before = re.sub(r"\s+", " ", backend[max(0, acq_at - 3500):acq_at])
        results["Route A's pair is N*K and M*K f16 elements"] = \
            "src0_elems = static_cast<size_t>(N) * static_cast<size_t>(K);" in before and \
            "src1_elems = static_cast<size_t>(M) * static_cast<size_t>(K);" in before and \
            "weights_bytes = src0_elems * sizeof(sycl::half);" in before and \
            "activations_bytes = src1_elems * sizeof(sycl::half);" in before

    # ---- Route A has no precision check, so the walk must not drop a node on precision before asking it ----
    pre = dq_walk.find("const ggml_tensor * src1 = node->src[1];")
    post = dq_walk.find("const bool need_src0_f16")
    results["anchor: the walk's node pre-filter exists"] = 0 <= pre < post
    if 0 <= pre < post:
        # The only precision read in the whole walk is the one that decides the legacy arm's own draw; a drop
        # anywhere else in the loop body (pre-filter or after the arms are combined) would skip Route A's nodes.
        results["the walk's pre-filter does not drop a node on precision"] = \
            dq_walk.count("op_params[0]") == 1 and \
            re.search(r"const bool\s+prec_default\s*=\s*node->op_params\[0\]\s*==\s*GGML_PREC_DEFAULT;", dq_walk) is not None
    results["the walk combines the two arms through the pure verdict"] = "zone_walk_f16_node_draws(" in dq_walk

    # ---- the walk asks the router, the admission and the supplies question once per node ----
    results["the walk asks the PP admission only through the supplies helper"] = \
        "ggml_sycl_onednn_pp_candidate(" not in dq_walk
    results["the walk asks the router once per node"] = dq_walk.count(".select(") == 1
    results["the walk asks the supplies helper once per node"] = dq_walk.count("ggml_sycl_onednn_pp_scratch_supplies(") == 1
    if route_a is not None:
        results["the Route A predicate reuses the walk's answers instead of asking again"] = \
            ".select(" not in route_a and "ggml_sycl_onednn_pp_candidate(" not in route_a and \
            "ggml_sycl_onednn_pp_scratch_supplies(" not in route_a

    # ---- the bound: max(stored plan, capacity - stored Graph floor), capped at the capacity; one stored snapshot ----
    pure = function_body(
        zone_sizing, r"size_t zone_onednn_pp_pair_bound\([^)]*\)\s*\{")
    pair_bound = function_body(
        cache, r"bool unified_cache_get_onednn_pp_pair_bound\(int device_id, size_t \* bound\)\s*\{")
    bound_plan = function_body(
        cache, r"static size_t onednn_pp_pair_bound_for\(const zone_onednn_plan & plan, size_t capacity_bytes\)\s*\{")
    bound_dev = function_body(
        cache, r"static size_t onednn_pp_pair_bound_for\(int device_id, size_t capacity_bytes\)\s*\{")
    bound_for = bound_plan
    results["anchor: the pure pair bound exists"] = pure is not None
    results["anchor: the pair-bound accessor is defined"] = pair_bound is not None
    results["anchor: the shared bound helper is defined"] = bound_plan is not None and bound_dev is not None
    results["the pair-bound accessor is declared"] = \
        re.search(r"bool unified_cache_get_onednn_pp_pair_bound\(int device_id, size_t \* bound\);", cache_hpp) is not None
    if pure is not None:
        pure_norm = re.sub(r"\s+", " ", pure)
        results["the pure bound subtracts the floor from the capacity"] = \
            "capacity_bytes - graph_floor_bytes" in pure_norm and "capacity_bytes > graph_floor_bytes" in pure_norm
        results["the pure bound never drops below the plan (max with the plan)"] = \
            "std::max(bare_plan_bytes," in pure_norm
        results["the pure bound never exceeds the capacity (min with the capacity)"] = \
            "std::min(capacity_bytes," in pure_norm
    if pair_bound is not None:
        results["the accessor reads the arena's real ONEDNN zone capacity"] = \
            "vram_zone_id::ONEDNN" in pair_bound and "arena_active()" in pair_bound
        results["the accessor answers through the shared bound helper"] = "onednn_pp_pair_bound_for(" in pair_bound
    if bound_for is not None:
        results["the helper reads the stored zone-plan snapshot, bare plan and floor together"] = \
            bound_dev is not None and "onednn_zone_plan_load(" in bound_dev and \
            "plan.bare_bytes" in bound_for and "plan.graph_floor_bytes" in bound_for
        results["the helper answers through the pure bound"] = "zone_onednn_pp_pair_bound(" in bound_for
        results["the helper does not read a live planner figure or recompute the floor"] = \
            "onednn_graph_scratch_zone_floor_bytes" not in bound_for and \
            "unified_cache_get_planned_onednn_scratchpad_bytes" not in bound_for and \
            "get_planned_onednn_graph_scratch_shape" not in bound_for
    ensure_zones = function_body(cache, r"bool unified_cache::ensure_planned_arena_zones\([^)]*\)\s*\{")
    results["anchor: ensure_planned_arena_zones exists"] = ensure_zones is not None
    if ensure_zones is not None:
        results["the kept-zone exit stores the larger of the held and live plan, in one critical section"] = \
            re.search(r"onednn_zone_plan_keep_and_store\(dev_id,\s*live_plan\);\s*return true;", ensure_zones) is not None
        keep_store = function_body(
            cache, r"static void onednn_zone_plan_keep_and_store\(int device_id,\s*const zone_onednn_plan & plan\)\s*\{")
        results["anchor: the locked keep-and-store helper exists"] = keep_store is not None
        if keep_store is not None:
            # ABSENCE: calling the public load or store from inside would take the same mutex twice.
            results["the keep-and-store helper keeps under one lock and calls neither accessor"] = \
                keep_store.count("lock_guard") == 1 and "zone_onednn_plan_keep(" in keep_store and \
                "onednn_zone_plan_load(" not in keep_store and "onednn_zone_plan_store(" not in keep_store
        results["the rebuilt-zone exit stores the live plan the zone was built from"] = \
            re.search(r"onednn_zone_plan_store\(dev_id,\s*live_plan\);\s*return true;", ensure_zones) is not None
        results["zone sizing reads the planned pair and floor once, as one pair"] = \
            ensure_zones.count("onednn_planned_pair_and_floor(") == 1 and \
            "unified_cache_get_planned_onednn_scratchpad_bytes" not in ensure_zones
    reserve = function_body(cache, r"bool unified_cache::reserve_onednn_scratch\([^)]*\)\s*\{")
    results["anchor: reserve_onednn_scratch exists"] = reserve is not None
    if reserve is not None:
        results["reserve bounds the never-shrink merge by the pair bound, not the raw capacity"] = \
            "onednn_pp_pair_bound_for(" in reserve and \
            re.search(r"zone_onednn_scratch_reserve_target\(arena_on,\s*pair_bound,", reserve) is not None
        # The refusal is only a refusal if it runs before the zone allocation it guards (a refusal after it is
        # unreachable for the case it exists for).
        results["reserve refuses a pair above the bound, before it allocates from the zone"] = \
            0 <= reserve.find("total_needed > pair_bound_now") < \
            reserve.find("zone_alloc(vram_zone_id::ONEDNN, weights_size)")
        # The first reservation is the planned pair, from the SAME snapshot the bound reads, only with an arena.
        results["reserve sizes the first reservation to the snapshot's planned pair"] = \
            re.search(r"zone_onednn_scratch_reserve_target\(arena_on,\s*pair_bound,\s*held_w,\s*held_a,\s*"
                      r"zone_plan\.weights_bytes,\s*zone_plan\.activations_bytes,", reserve) is not None
        results["reserve reads the planned pair from the stored snapshot, only with an arena"] = \
            re.search(r"zone_plan\s*=\s*arena_on\s*\?\s*onednn_zone_plan_load\(", reserve) is not None
    pair_reader = function_body(cache, r"static zone_onednn_plan onednn_planned_pair_and_floor\(int device_id\)\s*\{")
    results["anchor: the planned pair-and-floor reader exists"] = pair_reader is not None
    if pair_reader is not None:
        results["the planned pair reader carries both pair halves into the plan"] = \
            "g_planned_onednn_pair_weights_bytes" in pair_reader and \
            "g_planned_onednn_pair_activations_bytes" in pair_reader
    results["the inventory planner stores the pair's two halves, not only their sum"] = \
        re.search(r"unified_cache_set_planned_onednn_scratchpad_pair\(\s*ctx->device,\s*"
                  r"inventory_maxima\.onednn_reorder,\s*inventory_maxima\.onednn_eligible\)", backend) is not None
    for name, body in (("by-bytes core", bytes_helper),):
        results["the %s reads the pair bound, not the raw capacity" % name] = \
            "unified_cache_get_onednn_pp_pair_bound(" in body and "unified_cache_get_onednn_zone_capacity(" not in body
    sup = function_body(backend, r"static bool ggml_sycl_onednn_pp_scratch_supplies\([^)]*\)\s*\{")
    if sup is not None:
        results["the supplies helper reads the pair bound, not the raw capacity"] = \
            "unified_cache_get_onednn_pp_pair_bound(" in sup and "unified_cache_get_onednn_zone_capacity(" not in sup
    # ---- a row-gather tensor is no MUL_MAT operand (llama.cpp-8ony): producer, carrier, plumbing, classifier ----
    results["the adapter hands the classifier the loader's gather-only role"] = \
        adapter is not None and re.search(r"desc\.get_rows_only\s*=\s*item\.get_rows_only\s*;", adapter) is not None
    detail_loop = function_body(backend, r"for \(size_t i = 0; i < inventory->count; i\+\+\)\s*\{")
    results["anchor: the inventory copy loop exists"] = detail_loop is not None
    if detail_loop is not None:
        results["the backend copies the role into the inventory the planner sees"] = \
            re.search(r"info\.get_rows_only\s*=\s*inventory->tensors\[i\]\.get_rows_only\s*;", detail_loop) is not None
    results["the planner's tensor description carries the role"] = \
        re.search(r"struct placement_tensor_info \{.*?\bbool\s+get_rows_only\b.*?placement_tensor_info\(\) = default", cache_hpp,
                  re.S) is not None
    classifier = function_body(
        zone_sizing, r"path_scoped_maxima zone_scoped_maxima\([^)]*\)\s*\{")
    results["anchor: the pure classifier exists"] = classifier is not None
    if classifier is not None:
        results["the pure classifier excludes a gather-only tensor from the MUL_MAT-side marks"] = \
            "get_rows_only" in classifier
    info_struct = re.search(r"struct ggml_sycl_tensor_info \{(.*?)\};", header, re.S)
    results["anchor: ggml_sycl_tensor_info exists"] = info_struct is not None
    if info_struct is not None:
        body = info_struct.group(1)
        flag_at = body.find("get_rows_only")
        results["the role sits in the padding after type, so the array stride does not move"] = \
            0 <= body.find("type;") < flag_at < body.find("ne[")
    mark = function_body(model, r"static void llama_model_sycl_mark_get_rows_only\([^)]*\)\s*\{")
    results["anchor: the loader-side role function exists"] = mark is not None
    if mark is not None:
        mark_norm = re.sub(r"\s+", " ", mark)
        results["the role comes from the loader's op table, not a name list"] = \
            "llm_tensor_info_for(" in mark and "GGML_OP_GET_ROWS" in mark and "LLM_TENSOR_LAYER_INPUT" in mark
        results["a tied token embedding stays a MUL_MAT operand when the file carries no head"] = \
            "tied_head = input == LLM_TENSOR_TOKEN_EMBD && !file_carries_head" in mark_norm and \
            "tensor.get_rows_only = !tied_head" in mark_norm
        results["the file's head is the loader's own output name over its own tensor set"] = \
            re.search(r"ml\.weights_map\.find\(\s*tn\(LLM_TENSOR_OUTPUT", mark) is not None
    for fname, sig in (("early plan", r"static void llama_model_sycl_compute_early_plan\([^)]*\)\s*\{"),
                       ("late inventory", r"static void llama_model_sycl_set_late_inventory\([^)]*\)\s*\{")):
        fbody = function_body(model, sig)
        results["anchor: the %s exists" % fname] = fbody is not None
        if fbody is not None:
            at = fbody.find("llama_model_sycl_mark_get_rows_only(tensors, ml)")
            results["the %s marks the role before it builds the inventory" % fname] = \
                0 <= at < fbody.find("llama_model_sycl_populate_inventory(")
    # ---- ONE source for "which weight types the unified kernel serves" (llama.cpp-8ony) ----
    # A shared header holds the one switch; the router and the planner's adapter both call it, and no second list of
    # the types exists to drift (the planner's hand-kept mirror in common.hpp is gone).
    serves = function_body(utypes, r"inline bool unified_kernel_serves_type\(ggml_type type\)\s*\{")
    results["anchor: the shared unified type predicate exists"] = serves is not None
    if serves is not None:
        results["the shared predicate lists the types"] = "GGML_TYPE_Q4_0" in serves and "GGML_TYPE_MXFP4" in serves
    router = function_body(dispatch, r"inline bool should_use_unified\(ggml_type type\)\s*\{")
    results["anchor: the router's unified type predicate exists"] = router is not None
    if router is not None:
        results["the router asks the shared predicate and keeps no list of its own"] = \
            "unified_kernel_serves_type(type)" in router and "GGML_TYPE_" not in router
    results["the router includes the shared header"] = '#include "unified-types.hpp"' in dispatch
    results["the planner includes the shared header"] = \
        '#include "unified-types.hpp"' in cache or '#include "unified-types.hpp"' in cache_hpp
    results["the planner's hand-kept mirror of the type set is gone"] = \
        "ggml_sycl_should_use_unified_type" not in common and "ggml_sycl_should_use_unified_type" not in cache

    # ---- ONE statement of the type-level PP refusal, shared by the pure admission and the planner (llama.cpp-8ony) ----
    term = function_body(placement, r"inline bool onednn_pp_type_term_refused\(bool enabled, bool skip_type\)\s*\{")
    results["anchor: the shared type-level refusal term exists"] = term is not None
    decide = function_body(placement, r"inline onednn_pp_refusal onednn_pp_admission_decide\([^)]*\)\s*\{")
    results["anchor: the pure admission exists"] = decide is not None
    if decide is not None:
        results["the pure admission takes the type-level answer as an input, not the two environment terms"] = \
            "in.type_admitted" in decide and "in.enabled" not in decide and "in.skip_type" not in decide and \
            "onednn_pp_type_term_refused" not in decide
    admitted_fn = function_body(backend, r"bool ggml_sycl_onednn_pp_type_admitted\(ggml_type type\)\s*\{")
    if admitted_fn is not None:
        results["the planner's admission asks the same shared term"] = \
            "onednn_pp_type_term_refused(" in admitted_fn
        # Every SYCL source, not this TU alone: a caller in the planner, the zone sizing or a header would be a
        # second reader of the term the admission exists to hold once. The fixed files are the ones this gate is
        # handed (so a mutant of any of them is seen); rest_sycl is everything else under ggml/src/ggml-sycl.
        every_sycl = "\n".join((backend, cache, cache_hpp, zone_sizing, common, dispatch, placement, utypes, rest_sycl))
        term_uses = len(re.findall(r"\bonednn_pp_type_term_refused\s*\(", every_sycl))
        term_defs = len(re.findall(r"\binline\s+bool\s+onednn_pp_type_term_refused\s*\(", every_sycl))
        results["the type admission is the only caller of the shared term"] = \
            term_uses - term_defs == 1 and len(re.findall(r"\bonednn_pp_type_term_refused\s*\(", backend)) == 1

    # ---- the reserve reads the stored snapshot once, and the bound comes from that read (llama.cpp-8ony) ----
    reserve_fn = function_body(cache, r"bool unified_cache::reserve_onednn_scratch\([^)]*\)\s*\{")
    if reserve_fn is not None:
        first_block = reserve_fn[:reserve_fn.find("zone_onednn_scratch_reserve_target(")]
        results["the reserve's first bound comes from the snapshot it already loaded"] = \
            first_block.count("onednn_zone_plan_load(") == 1 and \
            re.search(r"onednn_pp_pair_bound_for\(\s*zone_plan,", first_block) is not None
    # ---- the 8-argument reserve-target overload was a test-only shim; one definition remains ----
    results["one definition of the reserve target"] = zone_sizing.count("void zone_onednn_scratch_reserve_target(") == 1

    # ---- the conditional plan follows the same PP admission gates the op takes (llama.cpp-8ony) ----
    admitted = function_body(backend, r"bool ggml_sycl_onednn_pp_type_admitted\(ggml_type type\)\s*\{")
    results["anchor: the shared PP type admission exists"] = admitted is not None
    if admitted is not None:
        results["the PP type admission asks the environment gate and the type skip"] = \
            "ggml_sycl_onednn_pp_enabled()" in admitted and "ggml_sycl_onednn_pp_skip_type(type)" in admitted
    results["the PP type admission is declared for the planner to call"] = \
        re.search(r"bool\s+ggml_sycl_onednn_pp_type_admitted\(ggml_type type\);", common) is not None
    cand = function_body(backend, r"static bool ggml_sycl_onednn_pp_candidate\([^)]*route = [^)]*\)\s*\{")
    results["anchor: the PP candidate exists"] = cand is not None
    if cand is not None:
        results["the PP candidate takes its type-level answer from the one type admission"] = \
            "admission.type_admitted = ggml_sycl_onednn_pp_type_admitted(src0->type)" in " ".join(cand.split()) and \
            "ggml_sycl_onednn_pp_enabled()" not in cand and "ggml_sycl_onednn_pp_skip_type(" not in cand
    if adapter is not None:
        results["the adapter plans the conditional mark only for a type the PP admission serves"] = \
            "unified_kernel_serves_type(item.type) && ggml_sycl_onednn_pp_type_admitted(item.type)" in \
            " ".join(adapter.split())
    return results


def run(label, sources, expect_fail=None):
    srcs = tuple(sources)
    if len(srcs) == 3:
        srcs += (zone_sizing,)
    if len(srcs) == 4:
        srcs += (model, header)
    if len(srcs) == 6:
        srcs += (common, dispatch)
    if len(srcs) == 8:
        srcs += (placement, utypes)
    results = evaluate(*srcs)
    bad = [k for k, v in results.items() if not v]
    if expect_fail is None:
        for k, v in results.items():
            print(("PASS: " if v else "FAIL: ") + k)
        return bad
    if expect_fail not in results:
        print("SELF-TEST BROKEN: %s names no check %r" % (label, expect_fail))
        return [label]
    if results[expect_fail]:
        print("SELF-TEST FAIL: mutant %r did not trip %r" % (label, expect_fail))
        return [label]
    print("self-test ok: %r trips %r" % (label, expect_fail))
    return []


def mutate(text, old, new, count=1):
    if old not in text:
        raise SystemExit("self-test mutation anchor missing: %r" % old)
    return text.replace(old, new, count)


def mutate_arm_scope(text):
    """Turn Route A's `#if GGML_SYCL_DEQUANT_F16_ARM` into `#if 1` (comments are already stripped)."""
    at = text.find("size_t src0_region_bytes = 0;")
    start = text.rfind("#    if GGML_SYCL_DEQUANT_F16_ARM", 0, at)
    if at < 0 or start < 0:
        raise SystemExit("self-test mutation anchor missing: Route A's dequant-arm scope")
    return text[:start] + "#    if 1" + text[start + len("#    if GGML_SYCL_DEQUANT_F16_ARM"):]


def mutate_in_func(text, signature_regex, old, new):
    body = function_body(text, signature_regex)
    if body is None or old not in body:
        raise SystemExit("self-test mutation anchor missing in %s: %r" % (signature_regex, old))
    return text.replace(body, body.replace(old, new, 1), 1)


zone_sizing = strip_comments(Path(args.zone_sizing).read_text())
backend = strip_comments(Path(args.backend).read_text())
cache = strip_comments(Path(args.cache).read_text())
cache_hpp = strip_comments(Path(args.cache_hpp).read_text())
common = strip_comments(Path(args.common).read_text())
dispatch = strip_comments(Path(args.dispatch).read_text())
placement = strip_comments(Path(args.placement).read_text())
# The shared header may not exist yet; its absence is a failed check, not a crash.
utypes_path = Path(args.unified_types)
utypes = strip_comments(utypes_path.read_text()) if utypes_path.exists() else ""
model = strip_comments(Path(args.model).read_text())
header = strip_comments(Path(args.sycl_header).read_text())
# Every other SYCL source (tests excluded: they call the pure term on purpose).
_named = {Path(p).resolve() for p in (args.backend, args.cache, args.cache_hpp, args.zone_sizing, args.common,
                                      args.dispatch, args.placement, args.unified_types)}
rest_sycl = "\n".join(
    strip_comments(p.read_text())
    for p in sorted(Path(args.backend).resolve().parent.rglob("*"))
    if p.suffix in (".cpp", ".hpp", ".h", ".inc", ".cuh") and "tests" not in p.relative_to(
        Path(args.backend).resolve().parent).parts and p.resolve() not in _named)

failed = run("tree", (backend, cache, cache_hpp, zone_sizing))

if args.self_test and not failed:
    print("\n--- mutants ---")
    op_sig = r"inline void ggml_sycl_op_mul_mat_sycl\("
    walk_sig = r"static bool ggml_sycl_dequant_f16_ensure_for_graph\("
    helper_sig = r"static bool ggml_sycl_onednn_pp_scratch_planned_bytes\("
    supplies_sig = r"static bool ggml_sycl_onednn_pp_scratch_supplies\("
    enabled_sig = r"bool onednn_pp_unified_scratch_enabled\(ggml_type type\)"
    adapter_sig = r"std::vector<zone_tensor_desc> unified_cache_adapt_zone_inventory\("
    bound_sig = r"static size_t onednn_pp_pair_bound_for\(const zone_onednn_plan"
    ensure_sig = r"bool unified_cache::ensure_planned_arena_zones\("
    reserve_sig = r"bool unified_cache::reserve_onednn_scratch\("
    route_a_sig = r"static bool ggml_sycl_mul_mat_unified_pp_dequant_route\("
    SUPPLIES_NAME = "ggml_sycl_onednn_pp_scratch_supplies("
    mutants = [
        ("op arm ignores the shared question", "the op arm's candidate asks the shared supplies helper with its column tile",
         (mutate_in_func(backend, op_sig, SUPPLIES_NAME, "ggml_sycl_XXXX("), cache, cache_hpp)),
        ("op arm asks the admission alone", "the op arm no longer asks admission and plan separately",
         (mutate_in_func(backend, op_sig, SUPPLIES_NAME, "ggml_sycl_onednn_pp_candidate("), cache, cache_hpp)),
        ("op arm passes another column count", "the op arm's candidate asks the shared supplies helper with its column tile",
         (mutate_in_func(backend, op_sig, "ctx.device, src0, src1, dst, src1_ncols)",
                         "ctx.device, src0, src1, dst, ne11)"), cache, cache_hpp)),
        ("walk skips on admission alone", "the walk skips an op only through the shared supplies helper, with src1->ne[1]",
         (mutate_in_func(backend, walk_sig, SUPPLIES_NAME, "ggml_sycl_onednn_pp_candidate("), cache, cache_hpp)),
        ("supplies drops the type enablement", "the supplies helper asks the type/env enablement",
         (mutate_in_func(backend, supplies_sig, "onednn_pp_unified_scratch_enabled(", "XXXX("), cache, cache_hpp)),
        ("supplies drops the admission", "the supplies helper asks the PP admission",
         (mutate_in_func(backend, supplies_sig, "ggml_sycl_onednn_pp_candidate(", "XXXX("), cache, cache_hpp)),
        ("supplies drops the zone", "the supplies helper asks the pure verdict",
         (mutate_in_func(backend, supplies_sig, "zone_onednn_pp_scratch_supplies(", "zone_XXXX("), cache, cache_hpp)),
        ("enablement bypasses the pure predicate", "the enablement function asks the pure predicate",
         (backend, mutate_in_func(cache, enabled_sig, "zone_onednn_pp_scratch_type_enabled(", "zone_XXXX("), cache_hpp)),
        ("adapter forgets the unified types", "the adapter marks the unified-kernel types for the conditional dequant plan",
         (backend, mutate_in_func(cache, adapter_sig, "unified_kernel_serves_type(", "XXXX("), cache_hpp)),
        ("adapter keeps its own enablement", "the adapter hands the classifier the type/env enablement, not its own copy",
         (backend, mutate_in_func(cache, adapter_sig, "onednn_pp_unified_scratch_enabled(", "XXXX("), cache_hpp)),
        ("adapter marks expert stacks", "the adapter excludes expert stacks from the conditional mark too",
         (backend, mutate_in_func(cache, adapter_sig, "expert_tensor_role_from_tensor_name(", "XXXX("), cache_hpp)),
        ("backend keeps a private enablement copy", "the enablement function is declared once, for the cache and the backend to share",
         (backend + "\nstatic bool onednn_pp_unified_scratch_enabled(ggml_type t) { return true; }\n", cache,
          cache_hpp)),
        ("walk forgets Route A", "the walk counts a Route A node with the answers it already has",
         (mutate_in_func(backend, walk_sig, "ggml_sycl_mul_mat_unified_pp_dequant_route(", "ggml_sycl_XXXX("),
          cache, cache_hpp)),
        ("Route A predicate ignores the router", "the Route A predicate reads the router's unified answer",
         (mutate_in_func(backend, route_a_sig, "MatmulBackend::UnifiedKernel", "MatmulBackend::LegacyKernel"),
          cache, cache_hpp)),
        ("Route A predicate drops the unified-dispatch gate", "the Route A predicate keeps the unified-dispatch gate",
         (mutate_in_func(backend, route_a_sig, "ggml_sycl_unified_dispatch_enabled() && ", ""), cache, cache_hpp)),
        ("Route A predicate forces src1 plain", "the Route A predicate keeps every plain-src1 term",
         (mutate_in_func(backend, route_a_sig,
                         "ggml_is_contiguous(src1) && !ggml_is_transposed(src1) && !ggml_is_permuted(src1)",
                         "true"), cache, cache_hpp)),
        ("Route A predicate drops a plain-src1 term", "the Route A predicate keeps every plain-src1 term",
         (mutate_in_func(backend, route_a_sig, "&& !ggml_is_permuted(src1)", ""), cache, cache_hpp)),
        ("Route A predicate swaps the verdict's inputs", "the Route A predicate passes the pure verdict its own inputs",
         (mutate_in_func(backend, route_a_sig, "pp_candidate,\n                                                    scratch_supplies);",
                         "scratch_supplies,\n                                                    pp_candidate);"),
          cache, cache_hpp)),
        ("walk passes the whole batch to the supplies call", "the walk skips an op only through the shared supplies helper, with src1->ne[1]",
         (mutate_in_func(backend, walk_sig, "node, src1->ne[1], &pp_candidate)", "node, ggml_nrows(src1), &pp_candidate)"),
          cache, cache_hpp)),
        ("walk hands the predicate a constant", "the walk counts a Route A node with the answers it already has",
         (mutate_in_func(backend, walk_sig, "primary, pp_candidate, supplied)", "primary, true, supplied)"),
          cache, cache_hpp)),
        ("walk forgets the admission out-param", "the walk takes the PP admission from its one supplies call",
         (mutate_in_func(backend, walk_sig, "&pp_candidate)", "nullptr)"), cache, cache_hpp)),
        ("walk filters on precision again", "the walk's pre-filter does not drop a node on precision",
         (mutate_in_func(backend, walk_sig, "!ggml_is_contiguous(src0) || ggml_nrows(src1) <= 1",
                         "!ggml_is_contiguous(src0) || node->op_params[0] != GGML_PREC_DEFAULT || ggml_nrows(src1) <= 1"),
          cache, cache_hpp)),
        ("walk filters on precision late", "the walk's pre-filter does not drop a node on precision",
         (mutate_in_func(backend, walk_sig, "if (!ggml_sycl::zone_walk_f16_node_draws(prec_default, legacy_draws, unified_draws)) {",
                         "if (node->op_params[0] != GGML_PREC_DEFAULT || !ggml_sycl::zone_walk_f16_node_draws(prec_default, legacy_draws, unified_draws)) {"),
          cache, cache_hpp)),
        ("walk ignores the pure arm combination", "the walk combines the two arms through the pure verdict",
         (mutate_in_func(backend, walk_sig, "zone_walk_f16_node_draws(", "zone_XXXX("), cache, cache_hpp)),
        ("walk asks the router twice", "the walk asks the router once per node",
         (mutate_in_func(backend, walk_sig, "const ggml_sycl::MatmulDecision primary      = ctx.matmul_orchestrator.select(src0, src1, node);",
                         "const ggml_sycl::MatmulDecision primary      = ctx.matmul_orchestrator.select(src0, src1, node);\n        (void) ctx.matmul_orchestrator.select(src0, src1, node);"),
          cache, cache_hpp)),
        ("walk asks the admission itself", "the walk asks the PP admission only through the supplies helper",
         (mutate_in_func(backend, walk_sig, "const bool                      prec_default",
                         "const bool pp_again = ggml_sycl_onednn_pp_candidate(src0, src1, node, ctx.device); (void) pp_again;\n        const bool                      prec_default"),
          cache, cache_hpp)),
        ("Route A predicate asks the router again", "the Route A predicate reuses the walk's answers instead of asking again",
         (mutate_in_func(backend, route_a_sig, "const bool primary_unified",
                         "(void) ctx_XXXX.matmul_orchestrator.select(src0, src1, nullptr); const bool primary_unified"),
          cache, cache_hpp)),
        ("Route A draws on another queue", "Route A draws both planned buffers on the context's own queue",
         (mutate(backend, "ctx.dequant_f16_src0_scratch, ctx.device, *ctx.stream(), src0,",
                 "ctx.dequant_f16_src0_scratch, ctx.device, *ctx.stream(ctx.device, 1), src0,"), cache, cache_hpp)),
        ("Route A src1 draws on another queue", "Route A draws both planned buffers on the context's own queue",
         (mutate(backend, "ctx.dequant_f16_src1_scratch, ctx.device, *ctx.stream(), src0,",
                 "ctx.dequant_f16_src1_scratch, ctx.device, *ctx.stream(ctx.device, 1), src0,"), cache, cache_hpp)),
        ("Route A draws outside the arm scope", "Route A's planned draws sit inside the dequant-arm scope, the pool copy in its #else",
         (mutate_arm_scope(backend), cache, cache_hpp)),
        ("Route A asks acquire for one byte more", "Route A's acquire asks for the pair the supplies helper derives",
         (mutate(backend, "using_scratch = acquire_onednn_pp_scratch(ctx.device, src0->type, weights_bytes,\n                                                                              activations_bytes,",
                 "using_scratch = acquire_onednn_pp_scratch(ctx.device, src0->type, weights_bytes,\n                                                                              activations_bytes + 1,"),
          cache, cache_hpp)),
        ("Route A's activations lose a factor", "Route A's pair is N*K and M*K f16 elements",
         (mutate(backend, "const size_t activations_bytes = src1_elems * sizeof(sycl::half);",
                 "const size_t activations_bytes = src1_elems;"), cache, cache_hpp)),
        ("Route A predicate loses the pure verdict", "the Route A predicate asks the pure verdict",
         (mutate_in_func(backend, route_a_sig, "zone_unified_pp_draws_dequant(", "zone_XXXX("), cache, cache_hpp)),
        ("Route A falls back to the pool again", "Route A draws the planned src0 dequant buffer before any pool copy",
         (mutate(backend, "ggml_sycl_dequant_f16_scratch(\n                                            ctx.dequant_f16_src0_scratch",
                 "ggml_sycl_XXXX(\n                                            ctx.dequant_f16_src0_scratch"), cache, cache_hpp)),
        ("Route A src1 left to the pool", "Route A draws the planned src1 dequant buffer before any pool copy",
         (mutate(backend, "ggml_sycl_dequant_f16_scratch(\n                                            ctx.dequant_f16_src1_scratch",
                 "ggml_sycl_XXXX(\n                                            ctx.dequant_f16_src1_scratch"), cache, cache_hpp)),
        ("helper bypasses the pure predicate", "the helper asks the pure predicate",
         (mutate_in_func(backend, helper_sig, "zone_onednn_pp_scratch_planned(", "zone_XXXX("), cache, cache_hpp)),
        ("helper reads the mutable plan", "the helper does not read the (mutable) planned scratchpad figure",
         (mutate_in_func(backend, helper_sig, "unified_cache_get_onednn_pp_pair_bound(",
                         "unified_cache_get_planned_onednn_scratchpad_bytes_stored("), cache, cache_hpp)),
        ("helper reads the raw capacity again", "the by-bytes core reads the pair bound, not the raw capacity",
         (mutate_in_func(backend, helper_sig, "unified_cache_get_onednn_pp_pair_bound(",
                         "unified_cache_get_onednn_zone_capacity("), cache, cache_hpp)),
        ("helper ignores the zone", "the by-bytes core reads the pair bound, not the raw capacity",
         (mutate_in_func(backend, helper_sig, "unified_cache_get_onednn_pp_pair_bound(", "XXXX("),
          cache, cache_hpp)),
        ("supplies helper reads the raw capacity again", "the supplies helper reads the pair bound, not the raw capacity",
         (mutate_in_func(backend, r"static bool ggml_sycl_onednn_pp_scratch_supplies\(",
                         "unified_cache_get_onednn_pp_pair_bound(", "unified_cache_get_onednn_zone_capacity("),
          cache, cache_hpp)),
        ("accessor reports another zone", "the accessor reads the arena's real ONEDNN zone capacity",
         (backend, mutate_in_func(cache, r"bool unified_cache_get_onednn_pp_pair_bound\(",
                                  "vram_zone_id::ONEDNN", "vram_zone_id::RUNTIME"), cache_hpp)),
        ("bound drops the floor term", "the pure bound subtracts the floor from the capacity",
         (backend, cache, cache_hpp,
          mutate_in_func(zone_sizing, r"size_t zone_onednn_pp_pair_bound\(",
                         "capacity_bytes - graph_floor_bytes", "capacity_bytes"))),
        ("bound swaps max for min", "the pure bound never drops below the plan (max with the plan)",
         (backend, cache, cache_hpp,
          mutate_in_func(zone_sizing, r"size_t zone_onednn_pp_pair_bound\(", "std::max(bare_plan_bytes,",
                         "std::min(bare_plan_bytes,"))),
        ("bound loses the capacity cap", "the pure bound never exceeds the capacity (min with the capacity)",
         (backend, cache, cache_hpp,
          mutate_in_func(zone_sizing, r"size_t zone_onednn_pp_pair_bound\(", "std::min(capacity_bytes,",
                         "std::max(capacity_bytes,"))),
        ("accessor skips the shared helper", "the accessor answers through the shared bound helper",
         (backend, mutate_in_func(cache, r"bool unified_cache_get_onednn_pp_pair_bound\(",
                                  "onednn_pp_pair_bound_for(", "XXXX("), cache_hpp)),
        ("helper recomputes the floor at the site", "the helper does not read a live planner figure or recompute the floor",
         (backend, mutate_in_func(cache, bound_sig, "plan.graph_floor_bytes",
                                  "onednn_graph_scratch_zone_floor_bytes(0, 0, 0, 0, 0)"), cache_hpp)),
        ("helper reads the live plan as the floor", "the helper does not read a live planner figure or recompute the floor",
         (backend, mutate_in_func(cache, bound_sig, "plan.graph_floor_bytes",
                                  "unified_cache_get_planned_onednn_scratchpad_bytes(device_id)"), cache_hpp)),
        ("helper reads the live bare plan", "the helper does not read a live planner figure or recompute the floor",
         (backend, mutate_in_func(cache, bound_sig, "plan.bare_bytes",
                                  "unified_cache_get_planned_onednn_scratchpad_bytes_stored(device_id)"), cache_hpp)),
        ("helper ignores the snapshot's bare plan", "the helper reads the stored zone-plan snapshot, bare plan and floor together",
         (backend, mutate_in_func(cache, bound_sig, "plan.bare_bytes", "0"), cache_hpp)),
        ("kept-zone exit describes the zone by the live plan", "the kept-zone exit stores the larger of the held and live plan, in one critical section",
         (backend, mutate_in_func(cache, ensure_sig, "onednn_zone_plan_keep_and_store(dev_id, live_plan)",
                                  "onednn_zone_plan_store(dev_id, live_plan)"), cache_hpp)),
        ("rebuilt-zone exit stops storing", "the rebuilt-zone exit stores the live plan the zone was built from",
         (backend, mutate_in_func(cache, ensure_sig, "onednn_zone_plan_store(dev_id, live_plan);\n    return true;\n}",
                                  "return true;\n}"), cache_hpp)),
        ("zone sizing reads the live bare plan again", "zone sizing reads the planned pair and floor once, as one pair",
         (backend, mutate_in_func(cache, ensure_sig, "planned_onednn_bare)",
                                  "unified_cache_get_planned_onednn_scratchpad_bytes_stored(dev_id))"), cache_hpp)),
        ("reserve merges against the raw capacity", "reserve bounds the never-shrink merge by the pair bound, not the raw capacity",
         (backend, mutate_in_func(cache, reserve_sig, "zone_onednn_scratch_reserve_target(arena_on, pair_bound,",
                                  "zone_onednn_scratch_reserve_target(arena_on, arena_on ? zone_capacity(vram_zone_id::ONEDNN) : 0,"),
          cache_hpp)),
        ("reserve forgets the planned pair", "reserve sizes the first reservation to the snapshot's planned pair",
         (backend, mutate_in_func(cache, reserve_sig, "zone_plan.weights_bytes", "0"), cache_hpp)),
        ("reserve reads the planned pair without an arena", "reserve reads the planned pair from the stored snapshot, only with an arena",
         (backend, mutate_in_func(cache, reserve_sig, "arena_on ? onednn_zone_plan_load(",
                                  "true ? onednn_zone_plan_load("), cache_hpp)),
        ("pair reader drops the weights half", "the planned pair reader carries both pair halves into the plan",
         (backend, mutate_in_func(cache, r"static zone_onednn_plan onednn_planned_pair_and_floor\(",
                                  "g_planned_onednn_pair_weights_bytes", "0"), cache_hpp)),
        ("planner stores only the sum", "the inventory planner stores the pair's two halves, not only their sum",
         (mutate(backend, "unified_cache_set_planned_onednn_scratchpad_pair(", "unified_cache_set_XXXX("), cache,
          cache_hpp)),
        ("reserve serves a pair above the bound", "reserve refuses a pair above the bound, before it allocates from the zone",
         (backend, mutate_in_func(cache, reserve_sig, "total_needed > pair_bound_now", "false"), cache_hpp)),
        ("acquire without the plan", "acquire asks the plan before it asks for a reserve",
         (mutate_in_func(backend, r"static bool acquire_onednn_pp_scratch\(",
                         "ggml_sycl_onednn_pp_scratch_planned_bytes(", "ggml_sycl_XXXX("), cache, cache_hpp)),
        ("acquire gate after the reserve", "acquire asks the plan before it asks for a reserve",
         (mutate_in_func(
             mutate_in_func(backend, r"static bool acquire_onednn_pp_scratch\(",
                            "ggml_sycl_onednn_pp_scratch_planned_bytes(", "ggml_sycl_XXXX("),
             r"static bool acquire_onednn_pp_scratch\(", "scratch = ggml_sycl::unified_cache_get_onednn_scratch(",
             "ggml_sycl_onednn_pp_scratch_planned_bytes(0, 0, 0); scratch = ggml_sycl::unified_cache_get_onednn_scratch("),
          cache, cache_hpp)),
        ("reserve grows around the plan", "reserve refuses an over-zone request after its replan attempt",
         (backend, mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                                  "zone_cap / (1024.0f * 1024.0f));\n            return finish(false);",
                                  "zone_cap / (1024.0f * 1024.0f));"), cache_hpp)),
        ("reserve shrinks a held scratch", "reserve sizes the pair with the never-shrink target",
         (backend, mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                                  "zone_onednn_scratch_reserve_target(", "zone_XXXX("), cache_hpp)),
        ("target after the reuse test", "the never-shrink target runs before the already-reserved test",
         (backend, mutate_in_func(
             mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                            "zone_onednn_scratch_reserve_target(", "zone_XXXX("),
             r"bool unified_cache::reserve_onednn_scratch\(", "direct_attempt = true;",
             "zone_onednn_scratch_reserve_target(); direct_attempt = true;"), cache_hpp)),
        ("adapter drops the gather-only role", "the adapter hands the classifier the loader's gather-only role",
         (backend, mutate_in_func(cache, adapter_sig, "desc.get_rows_only = item.get_rows_only", "desc.get_rows_only = false"),
          cache_hpp)),
        ("backend drops the role copy", "the backend copies the role into the inventory the planner sees",
         (mutate(backend, "info.get_rows_only = inventory->tensors[i].get_rows_only", "info.get_rows_only = false"), cache,
          cache_hpp)),
        ("planner description loses the role", "the planner's tensor description carries the role",
         (backend, cache, mutate(cache_hpp, "bool        get_rows_only", "bool        XXXX"))),
        ("classifier ignores the role", "the pure classifier excludes a gather-only tensor from the MUL_MAT-side marks",
         (backend, cache, cache_hpp, mutate(zone_sizing, "get_rows_only", "XXXX"), model, header)),
        ("role moved past ne", "the role sits in the padding after type, so the array stride does not move",
         (backend, cache, cache_hpp, zone_sizing, model,
          mutate(mutate(header, "bool           get_rows_only;", ""), "int64_t        ne[GGML_MAX_DIMS];",
                 "int64_t        ne[GGML_MAX_DIMS];\n    bool get_rows_only;"))),
        ("role drops the tied rule", "a tied token embedding stays a MUL_MAT operand when the file carries no head",
         (backend, cache, cache_hpp, zone_sizing, mutate(model, "&& !file_carries_head", "&& false"), header)),
        ("role from a name list", "the role comes from the loader's op table, not a name list",
         (backend, cache, cache_hpp, zone_sizing, mutate(model, "info.op != GGML_OP_GET_ROWS", "false"), header)),
        ("role reads another tensor set", "the file's head is the loader's own output name over its own tensor set",
         (backend, cache, cache_hpp, zone_sizing, mutate(model, "weights_map.find(", "XXXX.find("), header)),
        ("early plan forgets the role", "the early plan marks the role before it builds the inventory",
         (backend, cache, cache_hpp, zone_sizing,
          mutate_in_func(model, r"static void llama_model_sycl_compute_early_plan\(",
                         "llama_model_sycl_mark_get_rows_only(tensors, ml)", "(void) 0"), header)),
        ("late inventory forgets the role", "the late inventory marks the role before it builds the inventory",
         (backend, cache, cache_hpp, zone_sizing,
          mutate_in_func(model, r"static void llama_model_sycl_set_late_inventory\(",
                         "llama_model_sycl_mark_get_rows_only(tensors, ml)", "(void) 0"), header)),
        ("adapter ignores the PP gates", "the adapter plans the conditional mark only for a type the PP admission serves",
         (backend, mutate_in_func(cache, adapter_sig, "ggml_sycl_onednn_pp_type_admitted(", "XXXX("), cache_hpp)),
        ("PP admission drops the env gate", "the PP type admission asks the environment gate and the type skip",
         (mutate_in_func(backend, r"bool ggml_sycl_onednn_pp_type_admitted\(", "ggml_sycl_onednn_pp_enabled()", "true"),
          cache, cache_hpp)),
        ("PP admission drops the type skip", "the PP type admission asks the environment gate and the type skip",
         (mutate_in_func(backend, r"bool ggml_sycl_onednn_pp_type_admitted\(", "ggml_sycl_onednn_pp_skip_type(type)", "false"),
          cache, cache_hpp)),
        ("PP admission undeclared", "the PP type admission is declared for the planner to call",
         (backend, cache, cache_hpp, zone_sizing, model, header,
          mutate(common, "ggml_sycl_onednn_pp_type_admitted(", "ggml_sycl_XXXX("), dispatch)),
        ("router keeps its own list", "the router asks the shared predicate and keeps no list of its own",
         (backend, cache, cache_hpp, zone_sizing, model, header, common,
          mutate_in_func(dispatch, r"inline bool should_use_unified\(", "unified_kernel_serves_type(type)",
                         "type == GGML_TYPE_Q4_0"))),
        ("router drops the include", "the router includes the shared header",
         (backend, cache, cache_hpp, zone_sizing, model, header, common,
          mutate(dispatch, '#include "unified-types.hpp"', "")),),
        ("planner brings the mirror back", "the planner's hand-kept mirror of the type set is gone",
         (backend, cache, cache_hpp, zone_sizing, model, header,
          common + "\ninline bool ggml_sycl_should_use_unified_type(ggml_type t) { return true; }\n", dispatch)),
        ("shared predicate loses a type", "the shared predicate lists the types",
         (backend, cache, cache_hpp, zone_sizing, model, header, common, dispatch, placement,
          mutate(utypes, "GGML_TYPE_MXFP4", "GGML_TYPE_XXXX"))),
        ("decide restates the term", "the pure admission takes the type-level answer as an input, not the two environment terms",
         (backend, cache, cache_hpp, zone_sizing, model, header, common, dispatch,
          mutate_in_func(placement, r"inline onednn_pp_refusal onednn_pp_admission_decide\(",
                         "!in.type_admitted", "onednn_pp_type_term_refused(true, false)"),
          utypes)),
        ("decide ignores the type answer", "the pure admission takes the type-level answer as an input, not the two environment terms",
         (backend, cache, cache_hpp, zone_sizing, model, header, common, dispatch,
          mutate_in_func(placement, r"inline onednn_pp_refusal onednn_pp_admission_decide\(",
                         "!in.type_admitted", "false"),
          utypes)),
        ("candidate reads the environment itself", "the PP candidate takes its type-level answer from the one type admission",
         (mutate_in_func(backend, r"static bool ggml_sycl_onednn_pp_candidate\(",
                         "ggml_sycl_onednn_pp_type_admitted(src0->type)", "ggml_sycl_onednn_pp_enabled()"),
          cache, cache_hpp)),
        ("second caller of the shared term", "the type admission is the only caller of the shared term",
         (backend + "\nstatic bool x() { return !ggml_sycl::onednn_pp_type_term_refused(true, false); }\n",
          cache, cache_hpp)),
        ("caller of the shared term in the zone sizing TU", "the type admission is the only caller of the shared term",
         (backend, cache, cache_hpp,
          zone_sizing + "\nstatic bool y() { return !ggml_sycl::onednn_pp_type_term_refused(true, false); }\n")),
        ("caller of the shared term in the planner TU", "the type admission is the only caller of the shared term",
         (backend, cache + "\nstatic bool z() { return !ggml_sycl::onednn_pp_type_term_refused(true, false); }\n",
          cache_hpp)),
        ("planner restates the term", "the planner's admission asks the same shared term",
         (mutate_in_func(backend, r"bool ggml_sycl_onednn_pp_type_admitted\(", "onednn_pp_type_term_refused(",
                         "XXXX("), cache, cache_hpp)),
        ("reserve reloads the snapshot for its bound", "the reserve's first bound comes from the snapshot it already loaded",
         (backend, mutate_in_func(cache, reserve_sig, "onednn_pp_pair_bound_for(zone_plan,",
                                  "onednn_pp_pair_bound_for(bound_dev,"), cache_hpp)),
        ("test shim comes back", "one definition of the reserve target",
         (backend, cache, cache_hpp,
          zone_sizing + "\nvoid zone_onednn_scratch_reserve_target(bool a, size_t b, size_t c, size_t d, size_t e, size_t f, "
                        "size_t * g, size_t * h) {}\n")),
        ("refusal after the zone allocation", "reserve refuses a pair above the bound, before it allocates from the zone",
         (backend, mutate_in_func(cache, reserve_sig, "const size_t pair_bound_now",
                                  "void * early_probe = zone_alloc(vram_zone_id::ONEDNN, weights_size); (void) early_probe; "
                                  "const size_t pair_bound_now"), cache_hpp)),
        ("kept exit loads then stores", "the kept-zone exit stores the larger of the held and live plan, in one critical section",
         (backend, mutate_in_func(cache, ensure_sig, "onednn_zone_plan_keep_and_store(dev_id, live_plan)",
                                  "onednn_zone_plan_store(dev_id, zone_onednn_plan_keep(onednn_zone_plan_load(dev_id), "
                                  "live_plan))"), cache_hpp)),
        ("keep-and-store takes two locks", "the keep-and-store helper keeps under one lock and calls neither accessor",
         (backend, mutate_in_func(cache, r"static void onednn_zone_plan_keep_and_store\(",
                                  "zone_onednn_plan_keep(", "zone_onednn_plan_keep(onednn_zone_plan_load(device_id), "),
          cache_hpp)),
        ("accessor undeclared", "the pair-bound accessor is declared",
         (backend, cache, mutate(cache_hpp, "unified_cache_get_onednn_pp_pair_bound(", "unified_cache_get_XXXX("))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
