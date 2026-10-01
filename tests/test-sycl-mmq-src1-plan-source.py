#!/usr/bin/env python3
"""Planned dense MMQ/MMVQ Q8_1 src1 scratch source contract (llama.cpp-479i).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: every dense quantized MUL_MAT at batch > 1 quantized its activations
into a scratch minted PER OP (ggml_sycl_op_mul_mat's scoped_mmvq_scratch_handle),
preferring the weight zone and, once that was full, spilling to a raw
sycl::malloc_device outside the arena. The planner accounted for none of it. With
nothing draining the queue between ops, the retained buffers piled up until the
driver headroom was gone: Qwen3.6-27B on the B50 with GGML_SYCL_ONEDNN_PP=0 minted
122 raw allocations (508.5 MB) and then failed a CONCAT kernel submission.

The fix this gate pins: ONE persistent per-backend-context, per-device buffer in
the RUNTIME zone with spill forbidden, sized from the planner and checked against
the exact graph demand BEFORE any submission. The companion unit test
(test-zone-sizing, Case 12) proves the arithmetic; this gate proves the
production code actually uses it and that the old per-op path is gone.

A5 (same task): the oneDNN f16 dequant arm of ggml_sycl_op_mul_mat_sycl minted an f16 copy of
the WHOLE weight (and of the activations) from the SCRATCH pool per op -- 11 raw 60 MiB spills on
the same run once the Q8 buffer was planned. It gets the same treatment: one persistent
RUNTIME-zone, spill-forbidden buffer (common.hpp dequant_f16_scratch_t), planned from the
inventory and checked against the graph's own demand before anything is submitted.

Run with --self-test to prove every check fires against a mutant of the thing it
forbids; a check that cannot fail is decoration.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
parser.add_argument("--cache", default=str(sycl / "unified-cache.cpp"))
parser.add_argument("--zone", default=str(sycl / "zone-sizing.hpp"))
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


def read(path):
    return strip_comments(Path(path).read_text())


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


def evaluate(backend, common, cache, zone):
    results = {}

    # --- anchors: every symbol the checks key off must exist ----------------
    ensure_body = function_body(common, r"void \* ensure_buffer\([^)]*\)\s*\{")
    graph_entry = function_body(backend, r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{")
    req_fn = function_body(cache, r"bool unified_cache_get_planned_runtime_zone_requirement\([^)]*\)\s*\{")
    adapter = function_body(cache, r"std::vector<zone_tensor_desc> unified_cache_adapt_zone_inventory\([^)]*\)\s*\{")
    results["anchor: ensure_buffer exists"] = ensure_body is not None
    results["anchor: graph_compute_unchecked exists"] = graph_entry is not None
    results["anchor: runtime zone requirement exists"] = req_fn is not None
    results["anchor: zone adapter exists"] = adapter is not None
    dq_struct = function_body(common, r"struct dequant_f16_scratch_t\s*\{")
    dq_ensure = function_body(dq_struct, r"void \* ensure_buffer\([^)]*\)\s*\{") if dq_struct else None
    dq_acquire = function_body(backend, r"static void \* ggml_sycl_dequant_f16_scratch\([^)]*\)\s*\{")
    dq_walk = function_body(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\([^)]*\)\s*\{")
    op_sycl = function_body(backend, r"inline void ggml_sycl_op_mul_mat_sycl\([^)]*\)\s*try\s*\{")
    results["anchor: dequant_f16_scratch_t exists"] = dq_struct is not None
    results["anchor: dequant ensure_buffer exists"] = dq_ensure is not None
    results["anchor: dequant acquire helper exists"] = dq_acquire is not None
    results["anchor: dequant graph walk exists"] = dq_walk is not None
    results["anchor: op_mul_mat_sycl exists"] = op_sycl is not None
    if None in (ensure_body, graph_entry, req_fn, adapter, dq_struct, dq_ensure, dq_acquire, dq_walk, op_sycl):
        return results

    # --- the buffer is a RUNTIME-zone, spill-forbidden, planned allocation ---
    results["ensure_buffer routes to the RUNTIME zone"] = "vram_zone_id::RUNTIME" in ensure_body
    results["ensure_buffer forbids the raw-malloc spill"] = "forbid_vram_zone_spill = true" in ensure_body
    # ABSENCE: the weight-zone routing that let the buffer spill outside the arena.
    results["ensure_buffer no longer prefers the WEIGHT zone"] = "vram_zone_id::WEIGHT" not in ensure_body

    # --- the per-op scratch is gone from the non-split dispatch --------------
    # The only remaining src1_ddq_scratch allocation is the row-split (multi-stream) path,
    # whose buffer cannot be shared across ops without cross-stream chaining.
    allocs = [m.start() for m in re.finditer(r"src1_ddq_scratch\[i\]\.allocate\(", backend)]
    results["exactly one per-op scratch allocation remains"] = len(allocs) == 1
    if len(allocs) == 1:
        window = backend[max(0, allocs[0] - 700):allocs[0]]
        results["the remaining per-op scratch is split-only"] = re.search(r"\bsplit\b", window) is not None
    else:
        results["the remaining per-op scratch is split-only"] = False
    # ABSENCE: the old resource-failure laundering. A failed Q8 buffer used to warn and
    # `return false`, which fell through to the generic BLAS dequant path (another unplanned
    # buffer, ~178 MB for ffn_down) and then aborted far from the cause.
    results["no 'failed to allocate Q8 scratch' decline remains"] = "failed to allocate Q8 scratch" not in backend
    results["the dispatch names a plan breach, not a missing kernel"] = "ggml_sycl_mmq_src1_plan_breach(" in backend

    # --- the planner accounts for it -----------------------------------------
    results["the runtime zone requirement folds in the planned src1 bytes"] = \
        "unified_cache_get_planned_mmq_src1_scratch_bytes(" in req_fn
    results["the inventory site publishes the planned src1 bytes"] = \
        "unified_cache_set_planned_mmq_src1_scratch(" in backend
    results["the adapter derives bytes-per-token from the pure helper"] = \
        "zone_mmq_src1_bytes_per_token(" in adapter
    # ABSENCE: the expert predicate must not decide MMQ operand-ness (llama.cpp-8xbt).
    results["the adapter does not key operand-ness on ne[2] > 1"] = \
        re.search(r"ne\[2\]\s*[>!]=?\s*1", adapter) is None
    results["the zone header exposes the pure demand helpers"] = \
        all(s in zone for s in ("zone_mmq_src1_row_bytes", "zone_mmq_src1_required_bytes",
                                "zone_mmq_src1_bytes_per_token", "zone_mmq_src1_scratch_bytes"))

    # --- the graph-entry check runs before any submission --------------------
    entry_at = graph_entry.find("ggml_sycl_mmq_src1_ensure_for_graph(")
    first_compute_at = graph_entry.find("compute_impl")
    results["graph entry ensures the buffer from the graph's nodes"] = entry_at >= 0
    results["the ensure precedes every compute_impl use"] = entry_at >= 0 and 0 <= entry_at < first_compute_at
    results["a failed ensure refuses the graph cleanly"] = \
        entry_at >= 0 and "GGML_STATUS_ALLOC_FAILED" in graph_entry[entry_at:entry_at + 900]

    # --- A5: the oneDNN f16 dequant scratch ----------------------------------
    results["dequant scratch routes to the RUNTIME zone"] = "vram_zone_id::RUNTIME" in dq_ensure
    results["dequant scratch forbids the raw-malloc spill"] = "forbid_vram_zone_spill = true" in dq_ensure
    results["dequant scratch never prefers the WEIGHT or SCRATCH zone"] = \
        "vram_zone_id::WEIGHT" not in dq_ensure and "vram_zone_id::SCRATCH" not in dq_ensure
    # The f16 arm takes its operands from the planned buffer; the pool is the named residual only.
    results["the f16 arm acquires the planned dequant scratch"] = "ggml_sycl_dequant_f16_scratch(" in op_sycl
    results["src0 pool alloc is skipped when the planned scratch holds it"] = "if (!src0_dq_scratch)" in op_sycl
    results["src1 pool alloc is skipped when the planned scratch holds it"] = "if (!src1_dq_scratch)" in op_sycl
    results["the planned scratch is only used on the context's main stream"] = \
        re.search(r"stream\s*==\s*ctx\.stream\(", op_sycl) is not None
    results["the dequant breach is a plan breach, not a decline"] = "ggml_sycl_dequant_f16_plan_breach(" in backend and \
        "ggml_sycl_dequant_f16_plan_breach(" in dq_acquire
    results["no growth is attempted while a graph is being recorded"] = "ggml_sycl_graph_recording_active()" in dq_acquire
    # The walk predicts the route with the dispatch's own router (one fact, one source).
    results["the dequant walk asks the dispatch's router"] = "matmul_orchestrator.select(" in dq_walk
    results["the dequant walk fails closed on an overflowing demand"] = "return false" in dq_walk
    results["the runtime zone requirement folds in the planned dequant bytes"] = \
        "unified_cache_get_planned_dequant_f16_scratch_bytes(" in req_fn
    results["the inventory site publishes the planned dequant bytes"] = \
        "unified_cache_set_planned_dequant_f16_scratch(" in backend
    results["the adapter derives the dequant fields from the pure helpers"] = \
        "zone_dequant_f16_weight_bytes(" in adapter and "zone_dequant_f16_src1_bytes_per_token(" in adapter
    results["the zone header exposes the dequant helpers"] = \
        all(s in zone for s in ("zone_dequant_f16_weight_bytes", "zone_dequant_f16_src1_bytes_per_token",
                                "zone_dequant_f16_required_bytes", "zone_dequant_f16_scratch_bytes"))
    dq_entry_at = graph_entry.find("ggml_sycl_dequant_f16_ensure_for_graph(")
    results["graph entry ensures the dequant scratch before any submission"] = \
        dq_entry_at >= 0 and dq_entry_at < graph_entry.find("compute_impl")
    results["a failed dequant ensure refuses the graph cleanly"] = \
        dq_entry_at >= 0 and "GGML_STATUS_ALLOC_FAILED" in graph_entry[dq_entry_at:dq_entry_at + 900]
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


backend, common, cache, zone = (read(args.backend), read(args.common), read(args.cache), read(args.zone))
failed = run("tree", (backend, common, cache, zone))

if args.self_test:
    def mutate(src, old, new, count=1):
        if old not in src:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return src
        return src.replace(old, new, count)

    def mutate_after(src, anchor, old, new):
        """Replace the first `old` that follows `anchor` (the struct under test, not its sibling)."""
        at = src.find(anchor)
        if at < 0 or src.find(old, at) < 0:
            print(f"FAIL: self-test anchor missing: {anchor!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, at)
        return src[:k] + new + src[k + len(old):]

    def mutate_in_func(src, sig_regex, old, new):
        m = re.search(sig_regex, src)
        if not m or src.find(old, m.end()) < 0:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, m.end())
        return src[:k] + new + src[k + len(old):]

    mutants = [
        ("weight zone", "ensure_buffer routes to the RUNTIME zone",
         (backend, mutate(common, "vram_zone_id::RUNTIME", "vram_zone_id::XXXX"), cache, zone)),
        ("spill allowed", "ensure_buffer forbids the raw-malloc spill",
         (backend, mutate(common, "forbid_vram_zone_spill = true", "forbid_vram_zone_spill = false"), cache, zone)),
        ("blas launder", "no 'failed to allocate Q8 scratch' decline remains",
         (backend + '\nGGML_LOG_WARN("[MMVQ-SOA] failed to allocate Q8 scratch");', common, cache, zone)),
        ("planner blind", "the runtime zone requirement folds in the planned src1 bytes",
         (backend, common, mutate(cache, "const size_t mmq_src1 = unified_cache_get_planned_mmq_src1_scratch_bytes(", "const size_t mmq_src1 = unified_cache_get_planned_XXXX("), zone)),
        ("expert predicate", "the adapter does not key operand-ness on ne[2] > 1",
         (backend, common, mutate(cache, "zone_mmq_src1_bytes_per_token(", "zone_mmq_src1_bytes_per_token(item.ne[2] > 1 ? 0 : 1 + "), zone)),
        ("dequant spill allowed", "dequant scratch forbids the raw-malloc spill",
         (backend, mutate_after(common, "struct dequant_f16_scratch_t", "forbid_vram_zone_spill = true",
                                "forbid_vram_zone_spill = false"), cache, zone)),
        ("dequant weight zone", "dequant scratch never prefers the WEIGHT or SCRATCH zone",
         (backend, mutate_after(common, "struct dequant_f16_scratch_t", "vram_zone_id::RUNTIME",
                                "vram_zone_id::WEIGHT"), cache, zone)),
        ("dequant planner blind", "the runtime zone requirement folds in the planned dequant bytes",
         (backend, common, mutate(cache, "unified_cache_get_planned_dequant_f16_scratch_bytes(",
                                  "unified_cache_get_planned_XXXX("), zone)),
        ("dequant pool restored", "src0 pool alloc is skipped when the planned scratch holds it",
         (mutate(backend, "if (!src0_dq_scratch)", "if (true)"), common, cache, zone)),
        ("dequant growth while recording", "no growth is attempted while a graph is being recorded",
         (mutate_in_func(backend, r"static void \* ggml_sycl_dequant_f16_scratch\(",
                         "ggml_sycl_graph_recording_active()", "false"), common, cache, zone)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
