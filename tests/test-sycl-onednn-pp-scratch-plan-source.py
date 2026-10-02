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
was built with (zone_onednn_pp_scratch_planned over unified_cache_get_onednn_zone_capacity), and the op arm and
the graph-entry walk must ask that same question. The companion unit test (test-zone-sizing, Cases 14a-14d) proves
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


def evaluate(backend, cache, cache_hpp):
    results = {}
    bytes_helper = function_body(
        backend, r"static bool ggml_sycl_onednn_pp_scratch_planned_bytes\([^)]*\)\s*\{")
    op_sycl = function_body(backend, r"inline void ggml_sycl_op_mul_mat_sycl\([^)]*\)\s*try\s*\{")
    dq_walk = function_body(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\([^)]*\)\s*\{")
    accessor = function_body(
        cache, r"bool unified_cache_get_onednn_zone_capacity\(int device_id, size_t \* capacity\)\s*\{")
    results["anchor: the by-bytes admission core exists"] = bytes_helper is not None
    results["anchor: op_mul_mat_sycl exists"] = op_sycl is not None
    results["anchor: the dequant graph walk exists"] = dq_walk is not None
    results["anchor: the zone-capacity accessor is defined"] = accessor is not None
    results["the zone-capacity accessor is declared"] = \
        re.search(r"bool unified_cache_get_onednn_zone_capacity\(int device_id, size_t \* capacity\);",
                  cache_hpp) is not None
    if None in (bytes_helper, op_sycl, dq_walk, accessor):
        return results

    # One source: the helper answers from the zone the arena was built with, through the pure predicate.
    results["the helper asks the pure predicate"] = "zone_onednn_pp_scratch_planned(" in bytes_helper
    results["the helper reads the arena's real ONEDNN zone capacity"] = \
        "unified_cache_get_onednn_zone_capacity(" in bytes_helper
    # ABSENCE: it must not be answered from the stored planned bytes, which the runtime growth signal rewrites.
    results["the helper does not read the (mutable) planned scratchpad figure"] = \
        "unified_cache_get_planned_onednn_scratchpad_bytes" not in bytes_helper
    results["the accessor reports the ONEDNN zone"] = \
        "vram_zone_id::ONEDNN" in accessor and "arena_active()" in accessor

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

    # ---- review r1 I1/M4: ONE helper answers "the scratch supplies this op", for both consumers ----
    supplies_helper = function_body(
        backend, r"static bool ggml_sycl_onednn_pp_scratch_supplies\([^)]*\)\s*\{")
    enabled_fn = function_body(backend, r"static bool onednn_pp_unified_scratch_enabled\([^)]*\)\s*\{")
    results["anchor: the shared supplies helper exists"] = supplies_helper is not None
    results["anchor: the type/env enablement function exists"] = enabled_fn is not None
    if supplies_helper is not None:
        results["the supplies helper asks the PP admission"] = "ggml_sycl_onednn_pp_candidate(" in supplies_helper
        results["the supplies helper asks the type/env enablement"] = \
            "onednn_pp_unified_scratch_enabled(" in supplies_helper
        results["the supplies helper asks the pure verdict"] = "zone_onednn_pp_scratch_supplies(" in supplies_helper
        results["the supplies helper reads the arena's real ONEDNN zone capacity"] = \
            "unified_cache_get_onednn_zone_capacity(" in supplies_helper
        results["the supplies helper derives the pair from src0 and the column count"] = \
            "ne[1]" in supplies_helper and "ne[0]" in supplies_helper
    if enabled_fn is not None:
        results["the enablement function asks the pure predicate"] = \
            "zone_onednn_pp_scratch_type_enabled(" in enabled_fn
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

    # ---- review r1 I2: Route A draws the planned dequant buffers, and the walk counts such a node ----
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

    # ---- m1: Route A has no precision check, so the walk asks it before filtering on precision ----
    pre = dq_walk.find("const ggml_tensor * src1 = node->src[1];")
    post = dq_walk.find("const bool need_src0_f16")
    results["anchor: the walk's node pre-filter exists"] = 0 <= pre < post
    if 0 <= pre < post:
        results["the walk's pre-filter does not drop a node on precision"] = \
            "op_params[0]" not in dq_walk[pre:post]
    results["the walk combines the two arms through the pure verdict"] = "zone_walk_f16_node_draws(" in dq_walk

    # ---- m6: the walk asks the router, the admission and the supplies question once per node ----
    results["the walk asks the PP admission only through the supplies helper"] = \
        "ggml_sycl_onednn_pp_candidate(" not in dq_walk
    results["the walk asks the router once per node"] = dq_walk.count(".select(") == 1
    results["the walk asks the supplies helper once per node"] = dq_walk.count("ggml_sycl_onednn_pp_scratch_supplies(") == 1
    if route_a is not None:
        results["the Route A predicate reuses the walk's answers instead of asking again"] = \
            ".select(" not in route_a and "ggml_sycl_onednn_pp_candidate(" not in route_a and \
            "ggml_sycl_onednn_pp_scratch_supplies(" not in route_a
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
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


backend = strip_comments(Path(args.backend).read_text())
cache = strip_comments(Path(args.cache).read_text())
cache_hpp = strip_comments(Path(args.cache_hpp).read_text())

failed = run("tree", (backend, cache, cache_hpp))

if args.self_test and not failed:
    print("\n--- mutants ---")
    op_sig = r"inline void ggml_sycl_op_mul_mat_sycl\("
    walk_sig = r"static bool ggml_sycl_dequant_f16_ensure_for_graph\("
    helper_sig = r"static bool ggml_sycl_onednn_pp_scratch_planned_bytes\("
    supplies_sig = r"static bool ggml_sycl_onednn_pp_scratch_supplies\("
    enabled_sig = r"static bool onednn_pp_unified_scratch_enabled\("
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
         (mutate_in_func(backend, enabled_sig, "zone_onednn_pp_scratch_type_enabled(", "zone_XXXX("), cache, cache_hpp)),
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
         (mutate_in_func(backend, helper_sig, "unified_cache_get_onednn_zone_capacity(",
                         "unified_cache_get_planned_onednn_scratchpad_bytes_stored("), cache, cache_hpp)),
        ("helper ignores the zone", "the helper reads the arena's real ONEDNN zone capacity",
         (mutate_in_func(backend, helper_sig, "unified_cache_get_onednn_zone_capacity(", "XXXX("),
          cache, cache_hpp)),
        ("accessor reports another zone", "the accessor reports the ONEDNN zone",
         (backend, mutate_in_func(cache, r"bool unified_cache_get_onednn_zone_capacity\(",
                                  "vram_zone_id::ONEDNN", "vram_zone_id::RUNTIME"), cache_hpp)),
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
        ("accessor undeclared", "the zone-capacity accessor is declared",
         (backend, cache, mutate(cache_hpp, "unified_cache_get_onednn_zone_capacity(", "unified_cache_get_XXXX("))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
