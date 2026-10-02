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
the graph-entry walk must ask that same question. The companion unit test (test-zone-sizing, Case 14) proves the
arithmetic; this gate proves both consumers use it.

Two follow-on facts, pinned here too:
  * The direct unified_alloc that reserve_onednn_scratch made when the arena replan was refused (the
    "growing through the unified cache" path) is itself an unplanned allocation. An op the plan routes elsewhere
    is turned away at the one choke point every caller shares (acquire_onednn_pp_scratch), BEFORE reserve is
    asked, and reserve refuses a request that still exceeds the zone after its replan attempt instead of
    allocating around the plan.
  * A smaller request must never shrink a held scratch: reserve sizes the new pair with
    zone_onednn_scratch_reserve_target (per-component maximum with what is held), so a layer-0 request cannot
    replace the block a later op needed and force the regrowth that failed.

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


HELPER = "ggml_sycl_onednn_pp_scratch_planned("
# The already-reserved reuse test in reserve_onednn_scratch (its first code line).
REUSE_TEST = "if (onednn_weights_scratch_ && onednn_activations_scratch_ && onednn_weights_scratch_size_ >= weights_size"


def evaluate(backend, cache, cache_hpp):
    results = {}
    helper = function_body(
        backend, r"static bool ggml_sycl_onednn_pp_scratch_planned\([^)]*\)\s*\{")
    op_sycl = function_body(backend, r"inline void ggml_sycl_op_mul_mat_sycl\([^)]*\)\s*try\s*\{")
    dq_walk = function_body(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\([^)]*\)\s*\{")
    accessor = function_body(
        cache, r"bool unified_cache_get_onednn_zone_capacity\(int device_id, size_t \* capacity\)\s*\{")
    results["anchor: the shared admission helper exists"] = helper is not None
    results["anchor: op_mul_mat_sycl exists"] = op_sycl is not None
    results["anchor: the dequant graph walk exists"] = dq_walk is not None
    results["anchor: the zone-capacity accessor is defined"] = accessor is not None
    results["the zone-capacity accessor is declared"] = \
        re.search(r"bool unified_cache_get_onednn_zone_capacity\(int device_id, size_t \* capacity\);",
                  cache_hpp) is not None
    if None in (helper, op_sycl, dq_walk, accessor):
        return results

    # One source: the helper answers from the zone the arena was built with, through the pure predicate.
    results["the helper asks the pure predicate"] = "zone_onednn_pp_scratch_planned(" in helper
    results["the helper reads the arena's real ONEDNN zone capacity"] = \
        "unified_cache_get_onednn_zone_capacity(" in helper
    # ABSENCE: it must not be answered from the stored planned bytes, which the runtime growth signal rewrites.
    results["the helper does not read the (mutable) planned scratchpad figure"] = \
        "unified_cache_get_planned_onednn_scratchpad_bytes" not in helper
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
    results["the op arm's candidate asks the helper"] = bool(op_candidate) and HELPER in op_candidate.group(1)
    results["the op arm still asks the PP admission"] = \
        bool(op_candidate) and "ggml_sycl_onednn_pp_candidate(" in op_candidate.group(1)
    walk_skips = statements_calling(dq_walk, "ggml_sycl_onednn_pp_candidate(")
    results["anchor: the walk asks the PP admission"] = len(walk_skips) >= 1
    results["every PP skip in the walk also asks the helper"] = \
        len(walk_skips) >= 1 and all(HELPER in s for s in walk_skips)
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
    helper_sig = r"static bool ggml_sycl_onednn_pp_scratch_planned\("
    mutants = [
        ("op arm ignores the plan", "the op arm's candidate asks the helper",
         (mutate_in_func(backend, op_sig, HELPER, "ggml_sycl_XXXX("), cache, cache_hpp)),
        ("op arm drops the admission", "the op arm still asks the PP admission",
         (mutate_in_func(backend, op_sig, "ggml_sycl_onednn_pp_candidate(", "ggml_sycl_XXXX("), cache, cache_hpp)),
        ("walk skips on admission alone", "every PP skip in the walk also asks the helper",
         (mutate_in_func(backend, walk_sig, HELPER, "ggml_sycl_XXXX("), cache, cache_hpp)),
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
         (mutate_in_func(backend, r"static bool acquire_onednn_pp_scratch\(",
                         "unified_cache_reserve_onednn_scratch(",
                         "ggml_sycl_onednn_pp_scratch_planned_bytes(0,0,0); ggml_sycl::unified_cache_reserve_onednn_scratch("),
          cache, cache_hpp)),
        ("reserve grows around the plan", "reserve refuses an over-zone request after its replan attempt",
         (backend, mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                                  "return finish(false);\n        }\n        if (total_needed <= zone_cap)",
                                  "}\n        if (total_needed <= zone_cap)"), cache_hpp)),
        ("reserve shrinks a held scratch", "reserve sizes the pair with the never-shrink target",
         (backend, mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                                  "zone_onednn_scratch_reserve_target(", "zone_XXXX("), cache_hpp)),
        ("target after the reuse test", "the never-shrink target runs before the already-reserved test",
         (backend, mutate_in_func(cache, r"bool unified_cache::reserve_onednn_scratch\(",
                                  REUSE_TEST, "zone_onednn_scratch_reserve_target(" + REUSE_TEST),
          cache_hpp)),
        ("accessor undeclared", "the zone-capacity accessor is declared",
         (backend, cache, mutate(cache_hpp, "unified_cache_get_onednn_zone_capacity(", "unified_cache_get_XXXX("))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
