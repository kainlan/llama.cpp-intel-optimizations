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

The same defect, second arm: the dense f16 dequant arm of ggml_sycl_op_mul_mat_sycl minted an f16 copy of
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
    q8_walk = function_body(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\([^)]*\)\s*\{")
    acquire = function_body(backend, r"static void \* ggml_sycl_planned_scratch_acquire\([^)]*\)\s*\{")
    runtime_ensure = function_body(common, r"inline void \* ggml_sycl_runtime_scratch_ensure\([^)]*\)\s*\{")
    pin = function_body(backend, r"static void ggml_sycl_planned_scratch_pin\([^)]*\)\s*\{")
    stats_fn = function_body(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)\s*\{") or ""
    results["anchor: stats function exists"] = bool(stats_fn)
    q8_lambda_at = backend.find("auto acquire_planned_q8")
    q8_lambda = backend[q8_lambda_at:q8_lambda_at + 900] if q8_lambda_at >= 0 else None
    results["anchor: dequant_f16_scratch_t exists"] = dq_struct is not None
    results["anchor: dequant ensure_buffer exists"] = dq_ensure is not None
    results["anchor: dequant acquire helper exists"] = dq_acquire is not None
    results["anchor: dequant graph walk exists"] = dq_walk is not None
    results["anchor: op_mul_mat_sycl exists"] = op_sycl is not None
    results["anchor: Q8 graph walk exists"] = q8_walk is not None
    results["anchor: shared planned-scratch acquire exists"] = acquire is not None
    results["anchor: shared runtime-scratch ensure exists"] = runtime_ensure is not None
    results["anchor: planned-scratch pin helper exists"] = pin is not None
    results["anchor: Q8 acquire lambda exists"] = q8_lambda is not None
    if None in (ensure_body, graph_entry, req_fn, adapter, dq_struct, dq_ensure, dq_acquire, dq_walk, op_sycl,
                q8_walk, acquire, runtime_ensure, q8_lambda, pin):
        return results

    # --- the buffer is a RUNTIME-zone, spill-forbidden, planned allocation ---
    # Both buffers allocate through ggml_sycl_runtime_scratch_ensure, so the zone and spill facts are
    # properties of that one helper; ensure_buffer itself must only delegate to it.
    results["ensure_buffer routes to the RUNTIME zone"] = "vram_zone_id::RUNTIME" in runtime_ensure
    results["ensure_buffer forbids the raw-malloc spill"] = "forbid_vram_zone_spill = true" in runtime_ensure
    # ABSENCE: the weight-zone routing that let the buffer spill outside the arena.
    results["ensure_buffer no longer prefers the WEIGHT zone"] = \
        "vram_zone_id::WEIGHT" not in ensure_body and "vram_zone_id::WEIGHT" not in dq_ensure

    # --- the per-op scratch is gone from the non-split dispatch --------------
    # No per-op Q8 scratch remains anywhere: ctx.stream(device, idx) returns the SAME in-order
    # queue for every idx, so the row-split path orders its producer and consumers on the queue the planned
    # buffer already lives on. The premise that retained a per-op allocation there was false.
    results["no per-op Q8 scratch allocation remains"] = "src1_ddq_scratch" not in backend
    results["the per-op scratch type is gone"] = "scoped_mmvq_scratch_handle" not in backend
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

    # --- the dense f16 dequant scratch ---------------------------------------
    results["dequant scratch routes to the RUNTIME zone"] = \
        "vram_zone_id::RUNTIME" in runtime_ensure and "ggml_sycl_runtime_scratch_ensure<" in dq_ensure
    results["dequant scratch forbids the raw-malloc spill"] = \
        "forbid_vram_zone_spill = true" in runtime_ensure and "ggml_sycl_runtime_scratch_ensure<" in dq_ensure
    results["dequant scratch never prefers the WEIGHT or SCRATCH zone"] = \
        "vram_zone_id::SCRATCH" not in runtime_ensure and "vram_zone_id::SCRATCH" not in dq_ensure and \
        "ggml_sycl_runtime_scratch_ensure<" in dq_ensure
    # The f16 arm takes its operands from the planned buffer; the pool is the named residual only.
    results["the f16 arm acquires the planned dequant scratch"] = "ggml_sycl_dequant_f16_scratch(" in op_sycl
    results["src0 pool alloc is skipped when the planned scratch holds it"] = "if (!src0_dq_scratch)" in op_sycl
    results["src1 pool alloc is skipped when the planned scratch holds it"] = "if (!src1_dq_scratch)" in op_sycl
    results["src0 and src1 copies are separate planned buffers"] = \
        "ctx.dequant_f16_src0_scratch" in op_sycl and "ctx.dequant_f16_src1_scratch" in op_sycl
    results["the src0 copy is acquired after the WoQ arm, not up front"] = \
        op_sycl.find("used_woq") >= 0 and 0 <= op_sycl.find("used_woq") < op_sycl.find("ctx.dequant_f16_src0_scratch")
    results["the planned scratch is only used on the context's main stream"] = \
        re.search(r"stream\s*==\s*ctx\.stream\(", op_sycl) is not None
    results["the dequant breach is a plan breach, not a decline"] = "ggml_sycl_dequant_f16_plan_breach(" in backend and \
        "ggml_sycl_dequant_f16_plan_breach(" in dq_acquire
    results["no growth is attempted while this thread is recording a graph"] = \
        "ggml_sycl_graph_recording_this_thread()" in acquire and "growth refused while recording" in acquire
    # The refusal and the pin answer "will MY allocation be captured into a graph?", which is a per-thread
    # question; the process-wide predicate is true on threads that have no graph (common.hpp: llama.cpp-f9tg).
    results["the planned scratch never asks the process-wide recording predicate"] = \
        "ggml_sycl_graph_recording_active()" not in acquire and \
        "ggml_sycl_graph_recording_active()" not in (pin or "")
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
                                "zone_dequant_f16_region_bytes", "zone_dequant_f16_plan_bytes"))
    dq_entry_at = graph_entry.find("ggml_sycl_dequant_f16_ensure_for_graph(")
    results["graph entry ensures the dequant scratch before any submission"] = \
        dq_entry_at >= 0 and dq_entry_at < graph_entry.find("compute_impl")
    results["a failed dequant ensure refuses the graph cleanly"] = \
        dq_entry_at >= 0 and "GGML_STATUS_ALLOC_FAILED" in graph_entry[dq_entry_at:dq_entry_at + 900]

    # --- a recorded graph bakes the buffer's raw pointer: three independent defences ------------
    results["the Q8 graph walk ensures at least the planned bytes"] = \
        "unified_cache_get_planned_mmq_src1_scratch_bytes(" in q8_walk
    results["the dequant graph walk ensures at least the planned bytes"] = \
        "unified_cache_get_planned_dequant_f16_buffer_bytes(" in dq_walk
    results["acquiring while recording pins the handle into the graph's retention"] = \
        "ggml_sycl_planned_scratch_pin(" in acquire and "retain_handles_until_event(" in pin and \
        "buffer.handle(" in pin
    # The pin is conditional on THIS thread recording, and made once per backing per recording rather than once
    # per op (every pin is a retention-sink entry that lives as long as the graph).
    results["the pin is conditional on this thread recording"] = \
        re.match(r"\s*\{\s*if\s*\(\s*!\s*ggml_sycl_graph_recording_this_thread\(\)\s*\)\s*\{?\s*return", pin) is not None
    results["the pin is made once per backing per recording"] = \
        "graph_retention_token(" in pin and "owner_control_id(" in pin
    # A cache hit hands out the same pointer: the recorders that use local sinks need the pin there too.
    hit_at = backend.find("ctx.mmvq_q8_activation_cache.cached_q8_1(i)")
    results["a Q8 cache hit pins the handle too"] = \
        hit_at >= 0 and "ggml_sycl_planned_scratch_pin(ctx.mmvq_q8_activation_cache" in backend[hit_at:hit_at + 900]
    results["growth is refused while recording, by name"] = \
        "growth refused while recording" in acquire
    results["both consumers acquire through the shared helper"] = \
        "ggml_sycl_planned_scratch_acquire(" in q8_lambda and "ggml_sycl_planned_scratch_acquire(" in dq_acquire
    results["the Q8 breach is still a plan breach"] = "ggml_sycl_mmq_src1_plan_breach(" in q8_lambda
    # in-op growth is visible and counted.
    results["in-op growth warns once, naming the cohort"] = \
        "GGML_LOG_WARN" in acquire and "cohort" in acquire
    results["in-op growth feeds the mispredict accounting"] = "zone_sizing_record_underestimate(" in acquire
    # The observation count is fed once, at teardown, from the per-slot use count: the per-op call took a global
    # mutex and a string-keyed map lookup on every op and duplicated stats.uses.
    results["an op does not take the zone-sizing mutex per use"] = "zone_sizing_record_observation" not in acquire
    results["uses reach the observation count once, at teardown"] = \
        "zone_sizing_record_observations(" in stats_fn
    # the walk does not repeat dispatch's logging.
    results["the dequant walk asks the router quietly"] = "ggml_sycl_select_quiet_scope" in dq_walk
    results["the router's forced-kernel WARNs honour the quiet scope"] = \
        re.search(r"ggml_sycl_select_quiet\(\)[^;{]*\)\s*\{?\s*GGML_LOG_WARN\(\"\[SYCL\] %s kernel %s not eligible for batch", backend) is not None
    # the generic BLAS fallback names itself when it breaches.
    fallback_at = backend.find('trace_decision("dispatch-generic-blas-fallback"')
    results["the generic BLAS fallback sets the caller scope"] = \
        fallback_at >= 0 and "ggml_sycl_scratch_caller_scope" in backend[fallback_at:fallback_at + 2500]
    results["both breaches print the caller"] = backend.count("ggml_sycl_scratch_caller()") >= 2
    # the shared ensure asserts the one fact the shared buffer rests on.
    results["the shared ensure asserts an in-order queue"] = \
        "has_property<sycl::property::queue::in_order>" in runtime_ensure and "GGML_ASSERT" in runtime_ensure
    # one allocator for both buffers.
    results["the Q8 buffer ensures through the shared allocator"] = \
        "ggml_sycl_runtime_scratch_ensure<" in ensure_body
    results["the dequant buffer ensures through the shared allocator"] = \
        "ggml_sycl_runtime_scratch_ensure<" in dq_ensure
    results["the shared allocator is RUNTIME-zone and spill-forbidden"] = \
        "vram_zone_id::RUNTIME" in runtime_ensure and "forbid_vram_zone_spill = true" in runtime_ensure
    # Verifiability: a WARN-level line a normal run prints.
    stats_at = backend.find("[SCRATCH-STATS]")
    results["a WARN-level per-cohort stats line exists"] = \
        stats_at >= 0 and "GGML_LOG_WARN" in backend[max(0, stats_at - 300):stats_at]
    results["the stats are printed at context teardown"] = "log_planned_scratch_stats()" in backend
    # Sub-MB demands must not print as 0.0: the line is the evidence that a buffer was exercised, so it prints KB.
    results["the stats line prints KB, not MB"] = \
        stats_at >= 0 and "peak_demand=%.1f KB" in backend[stats_at:stats_at + 400] and \
        "MB" not in backend[stats_at:stats_at + 400]
    # growths counted the initial allocation, which is not a growth: the field is allocs.
    results["the stats line counts allocs, not growths"] = \
        stats_at >= 0 and "allocs=%u" in backend[stats_at:stats_at + 400] and \
        re.search(r"(?<!op_)growths=", backend[stats_at:stats_at + 400]) is None
    # Nit: the Q8 walk is cheap on the decode hot path.
    results["the Q8 walk does not ask the PP predicate for single-row ops"] = \
        re.search(r"ggml_nrows\(src1\)\s*(<=|==)\s*1", q8_walk) is not None
    # A single-row node still contributes its (cheap, arithmetic-only) demand: the plan floor is zero when
    # nothing is quantized on the weight side, and decode must still size the buffer it will draw from.
    single_at = re.search(r"ggml_nrows\(src1\)\s*(<=|==)\s*1", q8_walk)
    results["a single-row node still adds its demand"] = \
        single_at is not None and "zone_mmq_src1_required_bytes(" in q8_walk[single_at.end():single_at.end() + 700]
    # One fact, one source: which nodes quantize src1 is the router's decision, not a second predicate. A node the
    # router sends to the unified or oneDNN kernel never touches this buffer, so counting it is an idle RUNTIME
    # reservation and can refuse a graph whose real route needs nothing.
    results["the Q8 walk asks the dispatch's router"] = "matmul_orchestrator.select(" in q8_walk
    results["the Q8 walk asks the router quietly"] = "ggml_sycl_select_quiet_scope" in q8_walk
    results["the Q8 walk counts only kernels that quantize src1"] = "ggml_sycl_mul_mat_kernel_quantizes_src1(" in q8_walk
    results["a buffer with no counted node gets no graph-entry ensure"] = \
        re.search(r"if\s*\(\s*!\s*saw_dense_node\s*\)\s*\{?\s*return true", q8_walk) is not None
    # A row-split weight runs on several devices; each one draws from its own buffer.
    # The row ranges come from the one helper the op itself uses, so the walk and the op cannot disagree.
    results["the walks prime every device a row-split weight touches"] = \
        "ggml_sycl_mul_mat_device_rows(" in q8_walk and "ggml_sycl_mul_mat_device_rows(" in dq_walk and \
        backend.count("ggml_sycl_mul_mat_device_rows(") >= 4
    # The walks mutate slots the context owns, so they run under the graph lock (still before any submission).
    lock_at = graph_entry.find("graph_mutex")
    results["the walks run under the graph lock"] = \
        0 <= lock_at < entry_at and lock_at < dq_entry_at
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
# A comment is not code, so the stripped sources cannot see it. This one was a false claim a reader acted on
# : the whole-graph recording path does NOT keep the Q8 buffer's handle alive.
raw_backend = re.sub(r"\s*\n\s*//\s*", " ", Path(args.backend).read_text())  # un-wrap line comments
if "recorded graph or pointer table that baked the old pointer keeps its handle" in raw_backend:
    print("FAIL: the false 'recorded graph keeps its handle' comment is still in ggml-sycl.cpp")
    failed.append("false recorded-graph comment")
else:
    print("PASS: the false 'recorded graph keeps its handle' comment is gone")

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
        ("process-wide recording predicate", "no growth is attempted while this thread is recording a graph",
         (mutate_in_func(backend, r"static void \* ggml_sycl_planned_scratch_acquire\(",
                         "ggml_sycl_graph_recording_this_thread()", "ggml_sycl_graph_recording_active()"),
          common, cache, zone)),
        ("pin asks the wide predicate", "the planned scratch never asks the process-wide recording predicate",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_pin\(",
                         "ggml_sycl_graph_recording_this_thread()", "ggml_sycl_graph_recording_active()"),
          common, cache, zone)),
        ("unconditional pin", "the pin is conditional on this thread recording",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_pin\(",
                         "if (!ggml_sycl_graph_recording_this_thread())", "if (false)"), common, cache, zone)),
        ("pin per op", "the pin is made once per backing per recording",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_pin\(",
                         "graph_retention_token(", "graph_XXXX("), common, cache, zone)),
        ("cache hit unpinned", "a Q8 cache hit pins the handle too",
         (mutate_after(backend, "ctx.mmvq_q8_activation_cache.cached_q8_1(i)", "ggml_sycl_planned_scratch_pin(",
                       "ggml_sycl_XXXX("), common, cache, zone)),
        ("Q8 walk without the router", "the Q8 walk asks the dispatch's router",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "matmul_orchestrator.select(", "matmul_orchestrator.XXXX("), common, cache, zone)),
        ("Q8 walk counts every kernel", "the Q8 walk counts only kernels that quantize src1",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "ggml_sycl_mul_mat_kernel_quantizes_src1(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("Q8 walk ensures an unused buffer", "a buffer with no counted node gets no graph-entry ensure",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "if (!saw_dense_node)", "if (false)"), common, cache, zone)),
        ("decode demand dropped", "a single-row node still adds its demand",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "zone_mmq_src1_required_bytes(", "zone_mmq_src1_XXXX("), common, cache, zone)),
        ("per-use observation", "an op does not take the zone-sizing mutex per use",
         (mutate_in_func(backend, r"static void \* ggml_sycl_planned_scratch_acquire\(",
                         "stats.note_use(required_bytes);",
                         "stats.note_use(required_bytes); ggml_sycl::zone_sizing_record_observation(cohort);"),
          common, cache, zone)),
        ("weight zone", "ensure_buffer routes to the RUNTIME zone",
         (backend, mutate_after(common, "inline void * ggml_sycl_runtime_scratch_ensure(", "vram_zone_id::RUNTIME",
                                "vram_zone_id::XXXX"), cache, zone)),
        ("spill allowed", "ensure_buffer forbids the raw-malloc spill",
         (backend, mutate_after(common, "inline void * ggml_sycl_runtime_scratch_ensure(",
                                "forbid_vram_zone_spill = true", "forbid_vram_zone_spill = false"), cache, zone)),
        ("blas launder", "no 'failed to allocate Q8 scratch' decline remains",
         (backend + '\nGGML_LOG_WARN("[MMVQ-SOA] failed to allocate Q8 scratch");', common, cache, zone)),
        ("planner blind", "the runtime zone requirement folds in the planned src1 bytes",
         (backend, common, mutate(cache, "const size_t mmq_src1 = unified_cache_get_planned_mmq_src1_scratch_bytes(", "const size_t mmq_src1 = unified_cache_get_planned_XXXX("), zone)),
        ("expert predicate", "the adapter does not key operand-ness on ne[2] > 1",
         (backend, common, mutate(cache, "zone_mmq_src1_bytes_per_token(", "zone_mmq_src1_bytes_per_token(item.ne[2] > 1 ? 0 : 1 + "), zone)),
        ("dequant spill allowed", "dequant scratch forbids the raw-malloc spill",
         (backend, mutate_after(common, "inline void * ggml_sycl_runtime_scratch_ensure(",
                                "forbid_vram_zone_spill = true", "forbid_vram_zone_spill = false"), cache, zone)),
        ("dequant weight zone", "dequant scratch never prefers the WEIGHT or SCRATCH zone",
         (backend, mutate_after(common, "struct dequant_f16_scratch_t", "ggml_sycl_runtime_scratch_ensure<",
                                "ggml_sycl_runtime_scratch_ensure_scratch_zone<"), cache, zone)),
        ("dequant planner blind", "the runtime zone requirement folds in the planned dequant bytes",
         (backend, common, mutate(cache, "const size_t dequant_f16 = unified_cache_get_planned_dequant_f16_scratch_bytes(",
                                  "const size_t dequant_f16 = unified_cache_get_planned_XXXX("), zone)),
        ("dequant pool restored", "src0 pool alloc is skipped when the planned scratch holds it",
         (mutate(backend, "if (!src0_dq_scratch)", "if (true)"), common, cache, zone)),
        ("growth while recording", "no growth is attempted while this thread is recording a graph",
         (mutate_in_func(backend, r"static void \* ggml_sycl_planned_scratch_acquire\(",
                         "growth refused while recording", "growth allowed"), common, cache, zone)),
        ("Q8 walk ignores the plan", "the Q8 graph walk ensures at least the planned bytes",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "unified_cache_get_planned_mmq_src1_scratch_bytes(", "unified_cache_get_planned_XXXX("),
          common, cache, zone)),
        ("dequant walk ignores the plan", "the dequant graph walk ensures at least the planned bytes",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "unified_cache_get_planned_dequant_f16_buffer_bytes(", "unified_cache_get_planned_XXXX("),
          common, cache, zone)),
        ("no pin", "acquiring while recording pins the handle into the graph's retention",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_pin\(",
                         "retain_handles_until_event(", "retain_XXXX("), common, cache, zone)),
        ("silent growth", "in-op growth feeds the mispredict accounting",
         (mutate_in_func(backend, r"static void \* ggml_sycl_planned_scratch_acquire\(",
                         "zone_sizing_record_underestimate(", "zone_sizing_record_XXXX("), common, cache, zone)),
        ("loud walk", "the dequant walk asks the router quietly",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "ggml_sycl_select_quiet_scope", "ggml_sycl_select_XXXX"), common, cache, zone)),
        ("out-of-order queue allowed", "the shared ensure asserts an in-order queue",
         (backend, mutate_in_func(common, r"inline void \* ggml_sycl_runtime_scratch_ensure\(",
                                  "has_property<sycl::property::queue::in_order>", "has_property<XXXX>"), cache, zone)),
        ("private allocator", "the dequant buffer ensures through the shared allocator",
         (backend, mutate_after(common, "struct dequant_f16_scratch_t", "ggml_sycl_runtime_scratch_ensure<",
                                "ggml_sycl_XXXX<"), cache, zone)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
