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
parser.add_argument("--context", default=str(root / "src/llama-context.cpp"))
parser.add_argument("--header", default=str(root / "ggml/include/ggml-sycl.h"))
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
    route_pred = function_body(backend, r"static bool ggml_sycl_mul_mat_src1_quantizing_route\([^)]*\)\s*\{") or ""
    route_core = function_body(backend, r"static bool ggml_sycl_mul_mat_scratch_route\([^)]*\)\s*\{") or ""
    results["the Q8 walk asks the dispatch's router"] = \
        "ggml_sycl_mul_mat_src1_quantizing_route(" in q8_walk and "matmul_orchestrator.select(" in route_core
    results["the Q8 walk asks the router quietly"] = "ggml_sycl_select_quiet_scope" in q8_walk
    results["the Q8 walk counts only kernels that quantize src1"] = \
        "ggml_sycl_mul_mat_kernel_quantizes_src1" in route_pred and route_core.count("draws(") >= 2  # the router's answer and the decline's
    results["a buffer with no counted node gets no graph-entry ensure"] = \
        re.search(r"if\s*\(\s*!\s*saw_dense_node\s*\)\s*\{[^{}]*return true", q8_walk) is not None
    # A row-split weight runs on several devices; each one draws from its own buffer.
    # The row ranges come from the one helper the op itself uses, so the walk and the op cannot disagree.
    results["the walks prime every device a row-split weight touches"] = \
        "ggml_sycl_mul_mat_device_rows(" in q8_walk and "ggml_sycl_mul_mat_device_rows(" in dq_walk and \
        backend.count("ggml_sycl_mul_mat_device_rows(") >= 4
    # A src1 with no rows has no demand; it must not reach the overflow refusal, whose message would lie.
    results["the Q8 walk skips a src1 with no rows"] = \
        re.search(r"ggml_nrows\(src1\)\s*<=\s*0", q8_walk) is not None
    # A ubatch that produces no outputs has its last layer trimmed to ZERO rows by the output gather, so a zero-row
    # MUL_MAT is on every multi-ubatch prompt. The dispatch treats it as a no-op (ggml_sycl_is_noop); the walks must
    # ask the same predicate rather than re-deriving "empty", or they refuse a graph the dispatch would have run.
    results["both walks skip the nodes the dispatch treats as no-ops"] = \
        "ggml_sycl_is_noop(" in q8_walk and "ggml_sycl_is_noop(" in dq_walk
    # The walks run once per graph, decode included: no per-node zero-initialised array, and no scan of the
    # compile-time device maximum when the real device count is known.
    walk_loop = "for (int i = 0; i < cgraph->n_nodes; i++)"
    results["the walks declare the tensor split once, outside the node loop"] = \
        0 <= q8_walk.find("tensor_split") < q8_walk.find(walk_loop) and \
        0 <= dq_walk.find("tensor_split") < dq_walk.find(walk_loop)
    results["the walks scan the real device count, not the compile-time maximum"] = \
        re.search(r"for\s*\(int d = 0;\s*d < GGML_SYCL_MAX_DEVICES", q8_walk + dq_walk) is None
    # Two unrelated functions were named alike: the cooperative split and the row-split weight lookup.
    results["the row-split weight lookup does not overload the cooperative split's name"] = \
        re.search(r"ggml_sycl_mul_mat_tensor_split\(\s*const ggml_tensor", backend) is None and \
        backend.count("ggml_sycl_mul_mat_src0_tensor_split(") >= 4
    # The walks mutate slots the context owns, so they run under the graph lock (still before any submission).
    lock_at = graph_entry.find("graph_mutex")
    results["the walks run under the graph lock"] = \
        0 <= lock_at < entry_at and lock_at < dq_entry_at

    # --- llama.cpp-kpjw: the plan is a function of the RUNTIME n_ubatch, and it is RESERVED ----------------
    # B70, full card, Qwen3.6-27B perplexity: the plan was sized at the load-time n_ubatch (512), auto-ubatch
    # chose 2048, and the compute buffers of that rung had filled the RUNTIME zone before the first graph
    # materialized the planned buffer ("zone has 0.3 MB free"). Three defects, one per group below.
    txn = function_body(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\([^)]*\)\s*\{")
    hold_refresh = function_body(backend, r"static void ggml_sycl_planned_scratch_hold_refresh\([^)]*\)\s*\{")
    txn_guard = function_body(backend, r"struct ggml_sycl_dense_scratch_txn_guard\s*\{")
    replan_fn = function_body(cache, r"bool unified_cache_replan_planned_dense_scratch\([^)]*\)\s*\{")
    fit_fn = function_body(cache, r"bool unified_cache_dense_scratch_runtime_fit\([^)]*\)\s*\{")
    quantizing_route = function_body(backend, r"static bool ggml_sycl_mul_mat_src1_quantizing_route\([^)]*\)\s*\{")
    fallback_decision = function_body(
        backend, r"static ggml_sycl::MatmulDecision ggml_sycl_mul_mat_legacy_fallback_decision\([^)]*\)\s*\{")
    unified_alloc_fn = function_body(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)\s*\{")
    scratch_route = function_body(backend, r"static bool ggml_sycl_mul_mat_scratch_route\([^)]*\)\s*\{")
    f16_route = function_body(backend, r"static bool ggml_sycl_mul_mat_f16_dequant_route\([^)]*\)\s*\{")
    nonfa_check = function_body(backend, r"static bool ggml_sycl_check_nonfa_attn_scratch\([^)]*\)\s*\{")
    note_spill_fn = function_body(cache, r"void unified_cache_note_planned_hold_spill\([^)]*\)\s*\{")
    take_spill_fn = function_body(cache, r"void unified_cache_take_planned_hold_spills\([^)]*\)\s*\{")
    stats_fn = function_body(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)\s*\{")
    set_q8_fn = function_body(cache, r"bool unified_cache_set_planned_mmq_src1_scratch\([^)]*\)\s*\{")
    set_f16_fn = function_body(cache, r"bool unified_cache_set_planned_dequant_f16_scratch\([^)]*\)\s*\{")
    results["anchor: unified_alloc exists"] = unified_alloc_fn is not None
    results["anchor: shared scratch route exists"] = scratch_route is not None
    results["anchor: f16 dequant route predicate exists"] = f16_route is not None
    results["anchor: non-FA outside-arena check exists"] = nonfa_check is not None
    results["anchor: hold-spill note exists"] = note_spill_fn is not None
    results["anchor: hold-spill take exists"] = take_spill_fn is not None
    results["anchor: scratch stats logger exists"] = stats_fn is not None
    results["anchor: Q8 plan setter exists"] = set_q8_fn is not None
    results["anchor: f16 plan setter exists"] = set_f16_fn is not None
    (unified_alloc_fn, scratch_route, f16_route, nonfa_check, note_spill_fn, take_spill_fn, stats_fn, set_q8_fn,
     set_f16_fn) = (b or "" for b in (unified_alloc_fn, scratch_route, f16_route, nonfa_check, note_spill_fn,
                                      take_spill_fn, stats_fn, set_q8_fn, set_f16_fn))
    results["anchor: runtime-context transaction exists"] = txn is not None
    results["anchor: hold refresh exists"] = hold_refresh is not None
    results["anchor: transaction guard exists"] = txn_guard is not None
    results["anchor: dense replan exists"] = replan_fn is not None
    results["anchor: dense runtime fit exists"] = fit_fn is not None
    results["anchor: src1-quantizing route exists"] = quantizing_route is not None
    results["anchor: legacy fallback decision exists"] = fallback_decision is not None
    # No early return: a missing anchor must not hide which contract checks fail (a RED run lists every one).
    txn, hold_refresh, txn_guard, replan_fn, fit_fn, quantizing_route, fallback_decision = (
        b or "" for b in (txn, hold_refresh, txn_guard, replan_fn, fit_fn, quantizing_route, fallback_decision))

    # D1: planned but not RESERVED. A spill-capable RUNTIME request (a compute buffer) must leave the planned
    # bytes alone; the planned consumers themselves forbid the spill and are the claimants.
    results["a RUNTIME zone request asks the pure held-back predicate"] = \
        "zone_runtime_alloc_held_back(" in unified_alloc_fn and "unified_cache_note_runtime_request(" in unified_alloc_fn
    # Argument order is the contract (runtime zone, forbid-spill, zone free bytes, hold, request size) and the hold is
    # read for THIS request's device. A swapped pair compiles and answers a different question.
    results["the held-back predicate gets (runtime zone, forbid-spill, free, hold, size) for this device"] = (
        re.search(r"zone_runtime_alloc_held_back\(\s*zid\s*==\s*vram_zone_id::RUNTIME\s*,\s*"
                  r"req\.intent\.constraints\.forbid_vram_zone_spill\s*,\s*[\w>.-]*zone_available\(\s*zid\s*\)\s*,\s*"
                  r"hold_at_decision\s*,\s*alloc_size\s*\)",
                  unified_alloc_fn) is not None and
        re.search(r"hold_at_decision\s*=\s*unified_cache_note_runtime_request\(\s*req\.device\s*,\s*alloc_size",
                  unified_alloc_fn) is not None)
    results["the hold binds only spill-capable requests, never a forbid-spill claimant"] = \
        re.search(r"zone_runtime_alloc_held_back\([^;]*forbid_vram_zone_spill", unified_alloc_fn) is not None
    results["the hold is derived from the pure helper over the planned buffers"] = \
        "zone_planned_scratch_hold_bytes(" in hold_refresh and "unified_cache_set_planned_scratch_hold(" in hold_refresh
    results["the hold covers all three planned buffers"] = \
        all(s_ in hold_refresh for s_ in ("mmvq_q8_activation_cache", "dequant_f16_src0_scratch",
                                          "dequant_f16_src1_scratch"))
    results["the Q8 walk releases the hold it satisfied"] = "ggml_sycl_planned_scratch_hold_refresh(" in q8_walk
    results["the dequant walk releases the hold it satisfied"] = "ggml_sycl_planned_scratch_hold_refresh(" in dq_walk
    teardown_at = backend.find("mmvq_q8_activation_cache.release()")
    results["context teardown drops only the hold this context published"] = \
        teardown_at >= 0 and \
        "unified_cache_release_planned_scratch_hold(device, planned_scratch_owner)" in backend[max(0, teardown_at - 900):teardown_at + 900] and \
        re.search(r"unified_cache_set_planned_scratch_hold\(\s*device\s*,\s*0\s*\)",
                  backend[max(0, teardown_at - 900):teardown_at + 900]) is None
    release_fn = function_body(cache, r"bool unified_cache_release_planned_scratch_hold\([^)]*\)\s*\{") or ""
    results["the release clears the hold only for its owner"] = \
        re.search(r"if\s*\(\s*state\.owner\s*!=\s*owner\s*\)\s*\{\s*return false;", release_fn) is not None and \
        release_fn.find("state.owner != owner") < release_fn.find("state.hold")

    # D2: the plan follows the runtime n_ubatch, decided in the runtime-context transaction (before any graph
    # records, so it is not mid-recording growth), and a rung the RUNTIME zone cannot hold is refused there --
    # the auto-ubatch probe is that same transaction, so rung selection consults the plan without a second source.
    results["the load-time setters keep their per-token inputs for a re-plan"] = \
        all(n in cache for n in ("g_planned_mmq_src1_bytes_per_token", "g_planned_dequant_f16_weight_bytes_max",
                                 "g_planned_dequant_f16_src1_bytes_per_token", "g_planned_dense_scratch_n_ubatch"))
    results["the re-plan goes through the same two setters as the load-time plan"] = \
        "unified_cache_set_planned_mmq_src1_scratch(" in replan_fn and \
        "unified_cache_set_planned_dequant_f16_scratch(" in replan_fn
    results["the fit check uses the pure largest-ubatch helper"] = "zone_dense_scratch_largest_ubatch(" in fit_fn
    results["the transaction re-plans at the runtime n_ubatch, publish only"] = \
        re.search(r"!\s*probe_mode[^{;]*\)\s*\{[^{}]*unified_cache_replan_planned_dense_scratch\(", txn) is not None
    results["the transaction refuses a rung the RUNTIME zone cannot hold"] = \
        "unified_cache_dense_scratch_runtime_fit(" in txn and "does not fit the RUNTIME zone" in txn
    results["the refusal names the largest -ub that fits"] = \
        re.search(r"largest -ub[^\"]*", txn) is not None
    results["the ring admission counts the dense scratch as pending RUNTIME demand"] = \
        re.search(r"runtime_pending_bytes\s*\+=\s*dense_scratch_runtime_bytes\s*;", txn) is not None
    results["a refused or rolled-back transaction restores the plan and the hold"] = \
        "ggml_sycl_dense_scratch_txn_guard" in txn and \
        "unified_cache_replan_planned_dense_scratch(" in txn_guard and \
        "unified_cache_set_planned_scratch_hold(" in txn_guard
    # Only the success tail commits: exactly one commit(), after the success flag, with no early exit between them.
    succ_at = txn.find("g_runtime_update_succeeded = true;")
    commit_at = txn.find("dense_guard.commit()")
    results["only the success tail commits the guard, once, with no exit between"] = \
        txn.count("dense_guard.commit()") == 1 and 0 <= succ_at < commit_at and \
        re.search(r"\breturn\b", txn[succ_at:commit_at]) is None
    results["the success tail refreshes the hold after the commit"] = \
        0 <= commit_at < txn.find("ggml_sycl_planned_scratch_hold_refresh(", commit_at)
    # The guard zeroes the hold before the transaction places its MoE pools in the zone, and does it for the
    # context that publishes it. Both anchors must exist: an absent one used to read as a pass.
    mat_at = txn.find("ggml_sycl_materialize_published_mmid_workspaces(")
    guard_at = txn.find("ggml_sycl_dense_scratch_txn_guard ")
    results["the guard is constructed before the transaction materializes its pools"] = \
        0 <= guard_at < mat_at
    results["the guard zeroes the hold for its owner"] = \
        re.search(r"unified_cache_set_planned_scratch_hold\(\s*device\s*,\s*0\s*,", txn_guard) is not None

    # D3: the walk and the dispatch ask ONE question about the route. select() cannot see the unified kernel's
    # runtime decline (unresolved weight pointer, GGML_SYCL_UNIFIED_FORCE_LEGACY, ...); the dispatch answers that
    # decline with select(allow_unified=false), so the walk asks the same fallback through the same helper.
    results["the Q8 walk asks the shared route predicate"] = "ggml_sycl_mul_mat_src1_quantizing_route(" in q8_walk
    results["the Q8 walk no longer asks select() on its own"] = "matmul_orchestrator.select(" not in q8_walk and \
        "ggml_sycl_mul_mat_src1_quantizing_route(" in q8_walk
    results["the route predicate asks the shared fallback decision"] = \
        "ggml_sycl_mul_mat_legacy_fallback_decision(" in scratch_route and "UnifiedKernel" in scratch_route
    results["the route predicate still asks the router first, quietly"] = \
        "matmul_orchestrator.select(" in scratch_route
    results["the dispatch's decline re-select goes through the shared fallback decision"] = \
        backend.count("ggml_sycl_mul_mat_legacy_fallback_decision(") >= 3
    results["the allow_unified=false re-select is written once"] = \
        len(re.findall(r"select\([^;]*std::nullopt,\s*false\)", backend)) == 1 and \
        re.search(r"select\([^;]*std::nullopt,\s*false\)", fallback_decision) is not None

    # --- llama.cpp-kpjw review r1 -------------------------------------------------------------------------
    # I1: the f16 walk asks the same route question the Q8 walk does, and the hold counts the f16 plan the way it
    # counts the Q8 plan, from the first plan, because the decline-served node it used to miss draws from it.
    results["the f16 walk asks the shared f16 route predicate"] = \
        "ggml_sycl_mul_mat_f16_dequant_route(" in dq_walk and "matmul_orchestrator.select(" not in dq_walk
    results["both route predicates share one decision core"] = \
        "ggml_sycl_mul_mat_scratch_route(" in quantizing_route and "ggml_sycl_mul_mat_scratch_route(" in f16_route and \
        "zone_route_draws_scratch(" in scratch_route and \
        "ggml_sycl_mul_mat_legacy_fallback_decision(" in scratch_route and "matmul_orchestrator.select(" in scratch_route
    f16_draws = function_body(backend, r"static bool ggml_sycl_mul_mat_kernel_draws_dequant_f16\([^)]*\)\s*\{") or ""
    results["the f16 route predicate selects the oneDNN legacy kernels"] = \
        "ggml_sycl_mul_mat_kernel_draws_dequant_f16" in f16_route and \
        all(k in f16_draws for k in ("ONEDNN_AOS", "ONEDNN_COALESCED", "ONEDNN_SOA"))
    results["the hold counts the f16 buffers from the first plan, not once backed"] = \
        re.search(r"capacity\([^)]*\)\s*!=\s*0\s*\?", hold_refresh) is None and \
        "unified_cache_get_planned_dequant_f16_buffer_bytes(d, false)" in hold_refresh and \
        "unified_cache_get_planned_dequant_f16_buffer_bytes(d, true)" in hold_refresh
    results["the hold is published for this context's device and owner"] = \
        re.search(r"const\s+int\s+d\s*=\s*ctx\.device\s*;", hold_refresh) is not None and \
        re.search(r"unified_cache_set_planned_scratch_hold\(\s*d\s*,\s*hold\s*,\s*ctx\.planned_scratch_owner\s*\)", hold_refresh) is not None
    plan_f16_at = backend.find("unified_cache_set_planned_dequant_f16_scratch(")
    results["an f16 plan the build cannot draw is not planned (it would reserve bytes nothing draws)"] = \
        "ggml_sycl_dequant_f16_scratch_drawable()" in backend[max(0, plan_f16_at - 700):plan_f16_at + 400] if plan_f16_at >= 0 else False

    # I2: a hold-induced spill is not silent, and it can never cost cached weights.
    results["a held-back request is counted and warned about"] = \
        "unified_cache_note_planned_hold_spill(" in unified_alloc_fn and "GGML_LOG_WARN" in note_spill_fn and \
        "state.spill_count++" in note_spill_fn
    results["the warning is once per device per context (the take resets the count)"] = \
        re.search(r"first\s*=\s*state\.spill_count\s*==\s*0", note_spill_fn) is not None and \
        re.search(r"state\.spill_count\s*=\s*0\s*;", take_spill_fn) is not None
    results["the warning names the requester and the bytes"] = \
        "tag" in note_spill_fn and re.search(r"GGML_LOG_WARN\([^;]*%\.1f MB", note_spill_fn) is not None
    results["teardown reports the hold spills with the scratch stats"] = \
        "unified_cache_take_planned_hold_spills(" in stats_fn and "hold_spills=" in stats_fn
    evict_at = unified_alloc_fn.find("evict_and_flush(")
    results["the overcommit guard refuses a hold-induced spill instead of evicting weights"] = \
        0 <= unified_alloc_fn.find("hold_spill") < evict_at and \
        re.search(r"if\s*\(\s*hold_spill\s*\)\s*\{[^}]*return false;", unified_alloc_fn[:evict_at]) is not None
    results["the held-back request still takes the ordinary spill path"] = \
        re.search(r"if\s*\(\s*!hold_spill\s*\)\s*\{[^}]*zone_alloc\(", unified_alloc_fn) is not None
    results["the outside-arena headroom check is told the worst-case hold spill"] = \
        "hold_spill_bytes" in nonfa_check and \
        re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^;]*hold_spill_bytes", txn) is not None and \
        re.search(r"hold_spill_bytes\s*=\s*ggml_sycl_planned_scratch_hold_spill_bound\(", txn) is not None

    # I3: the Q8 walk refreshes the hold on every successful exit, including the one that saw no counted node.
    returns_true = [m.start() for m in re.finditer(r"return true;", q8_walk)]
    results["every successful exit of the Q8 walk refreshes the hold"] = \
        len(returns_true) >= 2 and all(
            "ggml_sycl_planned_scratch_hold_refresh(ctx);" in q8_walk[max(0, at - 90):at] for at in returns_true)

    # M1: the refusal text names the runtime n_ubatch (the plan follows it), and the stale doc comment is gone.
    joined_walks = re.sub(r'"\s*"', "", q8_walk), re.sub(r'"\s*"', "", dq_walk)  # adjacent literals are one string
    results["both refusals name the runtime n_ubatch, not the load-time one"] = \
        all("load-time n_ubatch" not in w and "runtime n_ubatch" in w for w in joined_walks)

    # M4: the plan inputs are device-global; a second live model must not shrink the first one's plan.
    results["the load-time plan keeps another live model's inputs"] = \
        "ggml_sycl_other_backend_context_live(ctx->device, ctx)" in backend[max(0, plan_f16_at - 2400):plan_f16_at + 400] and \
        "zone_dense_scratch_merge_input(" in set_q8_fn and "zone_dense_scratch_merge_input(" in set_f16_fn

    # M8: validate first, then store: a rejected figure leaves the stored inputs alone.
    results["the Q8 setter validates before it stores"] = \
        0 <= set_q8_fn.find("zone_mmq_src1_scratch_bytes(") < set_q8_fn.find("g_planned_mmq_src1_bytes_per_token[device_id].store(")
    results["the f16 setter validates before it stores"] = \
        0 <= set_f16_fn.find("zone_dequant_f16_plan_bytes(") < \
        set_f16_fn.find("g_planned_dequant_f16_weight_bytes_max[device_id].store(")
    bytes_at_fn = function_body(cache, r"bool unified_cache_planned_dense_scratch_bytes_at\([^)]*\)\s*\{") or ""
    results["a rejected figure is flagged, so no plan is derived from the stale inputs"] = \
        "g_planned_dense_scratch_invalid" in set_q8_fn and "g_planned_dense_scratch_invalid" in set_f16_fn and \
        "g_planned_dense_scratch_invalid" in bytes_at_fn and "g_planned_dense_scratch_invalid" in replan_fn

    # --- llama.cpp-kpjw review r2 ---------------------------------------------------------------------------
    # F1: held back means the zone ALONE would have served the request and the hold is what keeps it out. A
    # zone-full spill (request larger than the free bytes) is an ordinary spill and keeps its eviction.
    held_back_fn = function_body(zone, r"bool zone_runtime_alloc_held_back\([^)]*\)\s*\{") or ""
    results["anchor: held-back predicate defined"] = held_back_fn != ""
    results["held back requires a hold, a request the zone could serve, and the hold's refusal"] = \
        re.search(r"\bhold\s*>\s*0\b", held_back_fn) is not None and \
        re.search(r"\balloc_size\s*<=\s*zone_available\b", held_back_fn) is not None and \
        "zone_runtime_alloc_respects_hold(" in held_back_fn

    # F2: ONE source for the worst-case outside-arena spill, and it counts the right quantity (the whole request
    # that can be held back, so the bound is the hold plus the largest spill-capable request seen).
    recheck_fn = function_body(
        backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\([^)]*\)\s*\{") or ""
    bound_fn = function_body(backend, r"static size_t ggml_sycl_planned_scratch_hold_spill_bound\([^)]*\)\s*\{") or ""
    results["anchor: hold-spill bound exists"] = bound_fn != ""
    results["anchor: recheck exists"] = recheck_fn != ""
    results["the spill bound is the plan plus the largest spill-capable request"] = \
        "unified_cache_planned_dense_scratch_bytes_at(" in bound_fn and "unified_cache_get_runtime_request_hwm(" in bound_fn
    results["the allocator records the largest spill-capable RUNTIME request"] = \
        "unified_cache_note_runtime_request(" in unified_alloc_fn
    results["the transaction and the recheck ask the one bound"] = \
        re.search(r"ggml_sycl_planned_scratch_hold_spill_bound\([^;]*n_ubatch", txn) is not None and \
        "ggml_sycl_planned_scratch_hold_spill_bound(" in recheck_fn and \
        "unified_cache_get_planned_scratch_hold(" not in recheck_fn

    # F3: flash attention is the default, so the by-name refusal must cover it too: the same live free-memory
    # comparison, with the spill figure as the whole demand.
    hold_headroom_fn = function_body(backend, r"static bool ggml_sycl_check_hold_spill_headroom\([^)]*\)\s*\{") or ""
    results["anchor: FA-on hold-spill headroom check exists"] = hold_headroom_fn != ""
    results["the FA-on early return runs the headroom check instead of skipping it"] = \
        re.search(r"if\s*\(\s*flash_attn_enabled\s*\)\s*\{\s*return ggml_sycl_check_hold_spill_headroom\(", nonfa_check) is not None
    results["the FA-on check compares the spill with the device's live free memory"] = \
        "ggml_backend_sycl_get_device_memory(" in hold_headroom_fn and "GGML_SYCL_RUNTIME_TXN_REFUSAL" in hold_headroom_fn

    # M-a / M-b: the plan inputs are merged per device, at re-plan too, and one model's overflow does not
    # invalidate another live model's plan.
    results["the dense replan merges across live contexts too"] = \
        re.search(r"bool unified_cache_replan_planned_dense_scratch\([^)]*other_model_live", cache) is not None and \
        "other_model_live" in replan_fn
    results["the transaction asks whether another context is live on THIS device"] = \
        "ggml_sycl_other_backend_context_live(" in txn and "ggml_sycl_other_backend_context_live(" in backend[
            max(0, plan_f16_at - 2400):plan_f16_at + 400]
    results["a rejected figure does not invalidate another live model's plan"] = all(
        re.search(r"if\s*\(\s*!\s*other_model_live\s*\)\s*\{[^{}]*g_planned_dense_scratch_invalid", f) is not None
        for f in (set_q8_fn, set_f16_fn))

    # M-d / M-e: the owner is a monotonic id, and the hold state moves as one unit.
    results["the owner token is a monotonic context id, not an address"] = \
        "unified_cache_mint_planned_scratch_owner(" in common + backend and \
        "unified_cache_release_planned_scratch_hold(device, planned_scratch_owner)" in backend
    results["the hold state is one mutex-guarded unit"] = \
        re.search(r"struct planned_scratch_hold_state\s*\{[^}]*std::mutex", cache) is not None
    results["taking the hold spills is scoped to the context that owns them"] = \
        re.search(r"void unified_cache_take_planned_hold_spills\([^)]*owner", cache) is not None and \
        "planned_scratch_owner" in stats_fn

    # M-f: the f16 arm's compile condition is written once.
    results["the f16 arm's compile condition has one source"] = \
        "GGML_SYCL_DEQUANT_F16_ARM" in dq_walk and "defined(GGML_SYCL_F16)" not in dq_walk and \
        len(re.findall(r"GGML_SYCL_DNNL\s*&&\s*defined\(GGML_SYCL_F16\)", backend)) == 1 and \
        "GGML_SYCL_DEQUANT_F16_ARM != 0" in (function_body(backend, r"static constexpr bool ggml_sycl_dequant_f16_scratch_drawable\(\)\s*\{") or "")

    # M-c: the f16 walk has one success exit and it refreshes the hold, so a walk that satisfied the plan releases
    # what was held for it (and a walk that counted nothing proves nothing, so the hold stands -- by design).
    results["the f16 walk's success exit refreshes the hold"] = \
        re.search(r"ggml_sycl_planned_scratch_hold_refresh\(ctx\);\s*#\s*else", dq_walk) is not None
    # F1 once more, on the allocator side: a request the hold keeps out of a zone that COULD serve it never reaches
    # zone_alloc; a request larger than the zone's free bytes is an ordinary spill, so the overcommit guard still
    # evicts for it (the guard refuses only on hold_spill, which F1's predicate no longer sets for it).
    results["the overcommit guard refuses only a hold-induced spill"] = \
        re.search(r"if\s*\(\s*hold_spill\s*\)\s*\{[^{}]*return false;", unified_alloc_fn) is not None

    # r2/r3 (hardware, B50 auto-ub1024): the ladder's "does this rung fit" must see a hold-induced spill that
    # pushed the card under the driver headroom. The rung's compute buffers exist only after sched_reserve()
    # returns (the recheck inside it runs after a 1-token probe reserve, before the worst-case pp/tg reserves, and
    # only for the first rung under auto_fa), so the check runs from try_candidate, after sched_reserve(); see
    # evaluate_context. This half pins the backend entry and the one rule it and the transaction share.
    epoch_fn = function_body(cache, r"void unified_cache_begin_planned_hold_epoch\([^)]*\)\s*\{") or ""
    realized_fn = function_body(backend, r"static bool ggml_sycl_check_hold_spill_realized\([^)]*\)\s*\{") or ""
    graph_headroom_fn = function_body(backend, r"static void ggml_sycl_check_graph_scratch_headroom\([^)]*\)\s*\{") or ""
    entry_fn = function_body(backend, r"bool ggml_backend_sycl_planned_hold_spill_fits\([^)]*\)\s*\{") or ""
    hold_headroom_fn = function_body(backend, r"static bool ggml_sycl_check_hold_spill_headroom\([^)]*\)\s*\{") or ""
    results["anchor: realized hold-spill check exists"] = realized_fn != ""
    results["the realized check asks the pure rule, the live free memory and the spills since this publish"] = \
        "zone_hold_spill_realized_fits(" in realized_fn and "ggml_backend_sycl_get_device_memory(" in realized_fn and \
        "unified_cache_get_recent_planned_hold_spills(" in realized_fn and "GGML_SYCL_RUNTIME_TXN_REFUSAL" in realized_fn
    results["the exported entry asks the realized check for this backend's context and owner"] = \
        entry_fn != "" and re.search(r"ggml_sycl_check_hold_spill_realized\([^;]*planned_scratch_owner", entry_fn) is not None
    results["the exported entry is registered for a backend-DL build"] = \
        re.search(r'strcmp\(name,\s*"ggml_backend_sycl_planned_hold_spill_fits"\)\s*==\s*0\)\s*\{\s*return \(void \*\)\s*ggml_backend_sycl_planned_hold_spill_fits;', backend) is not None
    results["the recheck no longer claims the realized spill (it runs before the rung's worst-case reserves)"] = \
        "ggml_sycl_check_hold_spill_realized(" not in recheck_fn
    results["a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)"] = \
        "unified_cache_begin_planned_hold_epoch(" in txn and \
        re.search(r"request_hwm\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"recent_bytes\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"recent_count\s*=\s*0\s*;", epoch_fn) is not None and "state.owner != owner" in epoch_fn
    results["the driver headroom the realized check uses is the graph-entry check's constant (one source)"] = \
        "kSyclArenaMinExternalHeadroomBytes" in realized_fn and "kSyclArenaMinExternalHeadroomBytes" in graph_headroom_fn and \
        re.search(r"arena_min_external_headroom\s*=\s*256", graph_headroom_fn) is None
    # I1: the bound follows the candidate rung, and the FA-on check and the realized check ask ONE question.
    results["the spill bound scales the largest request to the candidate rung"] = \
        "zone_hold_spill_bound(" in bound_fn and "unified_cache_get_runtime_request_hwm(" in bound_fn
    results["the FA-on check asks the realized rule (predicted free after the spill), not a second one"] = \
        "zone_hold_spill_realized_fits(" in hold_headroom_fn and "spill_bytes <= free_mem" not in hold_headroom_fn
    note_fn = function_body(cache, r"size_t unified_cache_note_runtime_request\([^)]*\)\s*\{") or ""
    results["the largest request is recorded with the n_ubatch it was seen at, from the first publish on"] = \
        "epoch_n_ubatch" in note_fn and "request_hwm_n_ubatch" in note_fn
    # M3: one take of the leaf hold mutex per RUNTIME request on the allocation path.
    results["unified_alloc takes the hold state once per request"] = \
        len(re.findall(r"unified_cache_note_runtime_request\(", unified_alloc_fn)) == 1 and \
        "unified_cache_get_planned_scratch_hold(" not in unified_alloc_fn
    # M4: spill attribution follows the current owner.
    set_hold_fn = function_body(cache, r"void unified_cache_set_planned_scratch_hold\([^)]*\)\s*\{") or ""
    results["a new owner starts with its own spill counts"] = \
        "state.owner != owner" in set_hold_fn and re.search(r"spill_count\s*=\s*0\s*;", set_hold_fn) is not None
    # M1: the dispatch arm that draws the planned f16 buffers is compiled under the one macro too.
    op_body = op_sycl or ""
    dq_calls = [m.start() for m in re.finditer(r"ggml_sycl_dequant_f16_scratch\(", op_body)]
    results["both f16 buffer acquisitions in the dispatch arm sit under the one macro"] = \
        len(dq_calls) == 2 and all(
            re.findall(r"#\s*if[^\n]*", op_body[:at])[-1:] == ["#if GGML_SYCL_DEQUANT_F16_ARM"] or
            re.findall(r"#\s*if[^\n]*", op_body[:at])[-1:] == ["#    if GGML_SYCL_DEQUANT_F16_ARM"]
            for at in dq_calls)
    return results


def evaluate_context(context, header):
    """r3 C1: WHERE the hold-spill fit check runs in the auto-ubatch trial. A rung's compute buffers exist only once
    sched_reserve() has returned, so the check belongs in try_candidate, after it, for every rung, every
    flash-attention mode and the cached rung -- not in the recheck inside the reserve."""
    results = {}
    try_fn = function_body(context, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{") or ""
    results["anchor: try_candidate exists"] = try_fn != ""
    reserve_at = try_fn.find("sched_reserve();")
    hook_at = try_fn.find("hold_spill_fn(")
    last_ok = try_fn.rfind("return nullptr;")
    results["the hold-spill check runs in try_candidate, after sched_reserve() and before the rung is accepted"] = \
        0 <= reserve_at < hook_at < last_ok
    results["a rung whose hold spill pushed the card under its headroom loses with its own stop reason"] = \
        re.search(r"if\s*\(\s*hold_spill_fn\s*&&\s*!\s*hold_spill_fn\([^)]*\)\s*\)\s*\{[^{}]*return \"[^\"]+\";", try_fn) is not None and \
        "sched_matches_last_good" in try_fn[hook_at:last_ok]
    results["the hook is resolved for a direct build and for a backend-DL build"] = \
        "&ggml_backend_sycl_planned_hold_spill_fits" in context and \
        re.search(r'llama_context_sycl_proc_addr\([^;]*"ggml_backend_sycl_planned_hold_spill_fits"', context) is not None
    results["the header declares the exported entry"] = \
        re.search(r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_planned_hold_spill_fits\(\s*ggml_backend_t", header) is not None
    return results


def run_context(label, sources, expect_fail=None):
    results = evaluate_context(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


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


# The zone-sizing declarations and definitions are one source for the checks (the shape of a pure predicate lives
# in the .cpp; the contract comments live in the .hpp, which the stripped text drops anyway).
zone_impl = str(Path(args.zone).with_suffix(".cpp"))
backend, common, cache, zone = (read(args.backend), read(args.common), read(args.cache),
                                read(args.zone) + "\n" + (read(zone_impl) if Path(zone_impl).exists() else ""))
context_src = read(args.context)
header_src = read(args.header)
failed = run("tree", (backend, common, cache, zone))
failed += run_context("tree", (context_src, header_src))
# A comment is not code, so the stripped sources cannot see it. This one was a false claim a reader acted on
# : the whole-graph recording path does NOT keep the Q8 buffer's handle alive.
raw_backend = re.sub(r"\s*\n\s*//\s*", " ", Path(args.backend).read_text())  # un-wrap line comments
raw_common = Path(args.common).read_text()
if "uses, growths, capacity" in raw_common:
    print("FAIL: a common.hpp comment still names the stats field 'growths' (it is 'allocs')")
    failed.append("stale growths comment")
else:
    print("PASS: the stats comment names allocs")
if "this walk did not count" in raw_backend or "the dispatch's re-select with allow_unified=false can land" in raw_backend:
    print("FAIL: the Q8 walk's doc comment still describes the decline as unmodelled (the walk models it now)")
    failed.append("stale Q8 walk comment")
else:
    print("PASS: the Q8 walk's doc comment no longer calls the decline unmodelled")
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

    def mutate_all_in_func(src, sig_regex, old, new, times):
        for _ in range(times):
            src = mutate_in_func(src, sig_regex, old, new)
        return src

    mutants = [
        ("Q8 walk counts no-op nodes", "both walks skip the nodes the dispatch treats as no-ops",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "ggml_sycl_is_noop(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("dequant walk counts no-op nodes", "both walks skip the nodes the dispatch treats as no-ops",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "ggml_sycl_is_noop(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("zero-row src1 reaches the overflow refusal", "the Q8 walk skips a src1 with no rows",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "ggml_nrows(src1) <= 0", "ggml_nrows(src1) < -1"), common, cache, zone)),
        ("per-node tensor split", "the walks declare the tensor split once, outside the node loop",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split{};",
                         "int unrelated = 0;"), common, cache, zone)),
        ("device scan to the maximum", "the walks scan the real device count, not the compile-time maximum",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "for (int d = 0; d < device_count;", "for (int d = 0; d < GGML_SYCL_MAX_DEVICES;"),
          common, cache, zone)),
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
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_scratch_route\(",
                         "matmul_orchestrator.select(", "matmul_orchestrator.XXXX("), common, cache, zone)),
        ("Q8 walk counts every kernel", "the Q8 walk counts only kernels that quantize src1",
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_src1_quantizing_route\(",
                         "ggml_sycl_mul_mat_kernel_quantizes_src1", "ggml_sycl_XXXX"), common, cache, zone)),
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
        # llama.cpp-kpjw
        ("zone request ignores the hold", "a RUNTIME zone request asks the pure held-back predicate",
         (backend, common, mutate(cache, "zone_runtime_alloc_held_back(", "zone_runtime_XXXX("), zone)),
        ("hold binds the claimants too", "the hold binds only spill-capable requests, never a forbid-spill claimant",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "req.intent.constraints.forbid_vram_zone_spill,", "false,"), zone)),
        ("hold arguments swapped", "the held-back predicate gets (runtime zone, forbid-spill, free, hold, size) for this device",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "hold_at_decision, alloc_size)",
                                          "alloc_size, hold_at_decision)"), zone)),
        ("hold read for the wrong device", "the held-back predicate gets (runtime zone, forbid-spill, free, hold, size) for this device",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "unified_cache_note_runtime_request(req.device, alloc_size",
                                          "unified_cache_note_runtime_request(0, alloc_size"), zone)),
        ("hold from the shortfall", "the hold is derived from the pure helper over the planned buffers",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_hold_refresh\(",
                         "zone_planned_scratch_hold_bytes(", "zone_planned_XXXX("), common, cache, zone)),
        ("hold forgets the f16 src1 buffer", "the hold covers all three planned buffers",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_hold_refresh\(",
                         "dequant_f16_src1_scratch", "dequant_f16_XXXX"), common, cache, zone)),
        ("Q8 walk never drops the hold", "the Q8 walk releases the hold it satisfied",
         (mutate_all_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                             "ggml_sycl_planned_scratch_hold_refresh(", "ggml_sycl_XXXX(", 2), common, cache, zone)),
        ("teardown keeps the hold", "context teardown drops only the hold this context published",
         (mutate_after(backend, "mmvq_q8_activation_cache.release();",
                       "ggml_sycl::unified_cache_release_planned_scratch_hold(device, planned_scratch_owner);",
                       "(void) device;"),
          common, cache, zone)),
        ("teardown drops any context's hold", "context teardown drops only the hold this context published",
         (mutate_after(backend, "mmvq_q8_activation_cache.release();",
                       "ggml_sycl::unified_cache_release_planned_scratch_hold(device, planned_scratch_owner);",
                       "ggml_sycl::unified_cache_set_planned_scratch_hold(device, 0);"), common, cache, zone)),
        ("release ignores the owner", "the release clears the hold only for its owner",
         (backend, common, mutate_in_func(cache, r"bool unified_cache_release_planned_scratch_hold\(",
                                          "if (state.owner != owner) {", "if (false) {"), zone)),
        ("plan frozen at load time", "the transaction re-plans at the runtime n_ubatch, publish only",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "unified_cache_replan_planned_dense_scratch(", "unified_cache_XXXX("), common, cache, zone)),
        ("probe mutates the plan", "the transaction re-plans at the runtime n_ubatch, publish only",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "if (!probe_mode && next_kv_info.n_ubatch != 0) {", "if (next_kv_info.n_ubatch != 0) {"),
          common, cache, zone)),
        ("transaction never checks the fit", "the transaction refuses a rung the RUNTIME zone cannot hold",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "unified_cache_dense_scratch_runtime_fit(", "unified_cache_XXXX("), common, cache, zone)),
        ("ring blind to the dense scratch", "the ring admission counts the dense scratch as pending RUNTIME demand",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "runtime_pending_bytes += dense_scratch_runtime_bytes;", "runtime_pending_bytes += 0;"),
          common, cache, zone)),
        ("no rollback guard", "a refused or rolled-back transaction restores the plan and the hold",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "ggml_sycl_dense_scratch_txn_guard", "ggml_sycl_XXXX_guard"), common, cache, zone)),
        ("load-time inputs dropped", "the load-time setters keep their per-token inputs for a re-plan",
         (backend, common, mutate(cache, "g_planned_dense_scratch_n_ubatch", "g_planned_XXXX_n_ubatch", 99), zone)),
        ("Q8 walk back to select()", "the Q8 walk no longer asks select() on its own",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "ggml_sycl_mul_mat_src1_quantizing_route(",
                         "ctx.matmul_orchestrator.select("), common, cache, zone)),
        ("route predicate blind to the decline", "the route predicate asks the shared fallback decision",
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_scratch_route\(",
                         "ggml_sycl_mul_mat_legacy_fallback_decision(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("second decline re-select", "the allow_unified=false re-select is written once",
         (backend + "\nstatic void kpjw_second_source(ggml_backend_sycl_context & c) { c.matmul_orchestrator."
                    "select(nullptr, nullptr, nullptr, nullptr, std::nullopt, false); }", common, cache, zone)),
        # llama.cpp-kpjw review r1
        ("f16 walk back to select()", "the f16 walk asks the shared f16 route predicate",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "ggml_sycl_mul_mat_f16_dequant_route(", "ctx.matmul_orchestrator.select("),
          common, cache, zone)),
        ("f16 route off the shared core", "both route predicates share one decision core",
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_f16_dequant_route\(",
                         "ggml_sycl_mul_mat_scratch_route(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("f16 route kernels dropped", "the f16 route predicate selects the oneDNN legacy kernels",
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_kernel_draws_dequant_f16\(",
                         "ONEDNN_COALESCED", "XXXX"), common, cache, zone)),
        ("hold waits for the f16 backing", "the hold counts the f16 buffers from the first plan, not once backed",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_hold_refresh\(",
                         "{ ggml_sycl::unified_cache_get_planned_dequant_f16_buffer_bytes(d, false),",
                         "{ ctx.dequant_f16_src0_scratch.capacity(d) != 0 ? ggml_sycl::"
                         "unified_cache_get_planned_dequant_f16_buffer_bytes(d, false) : 0,"),
          common, cache, zone)),
        ("undrawable f16 still planned", "an f16 plan the build cannot draw is not planned (it would reserve bytes nothing draws)",
         (mutate(backend, "ggml_sycl_dequant_f16_scratch_drawable();", "true;"), common, cache, zone)),
        ("hold published without an owner", "the hold is published for this context's device and owner",
         (mutate_in_func(backend, r"static void ggml_sycl_planned_scratch_hold_refresh\(",
                         "hold, ctx.planned_scratch_owner)", "hold, 0)"), common, cache, zone)),
        ("hold spill not counted", "a held-back request is counted and warned about",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "unified_cache_note_planned_hold_spill(", "unified_cache_XXXX("), zone)),
        ("hold spill warned every time", "the warning is once per device per context (the take resets the count)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_note_planned_hold_spill\(",
                                          "spill_count == 0;", "spill_count >= 0;"), zone)),
        ("take does not reset", "the warning is once per device per context (the take resets the count)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_take_planned_hold_spills\(",
                                          "state.spill_count = 0;", "state.spill_count += 0;"), zone)),
        ("stats omit the hold spills", "teardown reports the hold spills with the scratch stats",
         (mutate_in_func(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)",
                         "hold_spills=", "hold_XXXX="), common, cache, zone)),
        ("guard evicts weights for a hold spill", "the overcommit guard refuses a hold-induced spill instead of evicting weights",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "if (hold_spill) {", "if (false) {"), zone)),
        ("held-back request never spills", "the held-back request still takes the ordinary spill path",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "if (!hold_spill) {", "if (true) {"), zone)),
        ("headroom blind to the hold", "the outside-arena headroom check is told the worst-case hold spill",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "probe_mode, hold_spill_bytes)", "probe_mode, 0)"), common, cache, zone)),
        ("idle Q8 exit skips the refresh", "every successful exit of the Q8 walk refreshes the hold",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "if (!saw_dense_node) {", "if (!saw_dense_node) { return true; }\n    if (false) {"),
          common, cache, zone)),
        ("refusal back to the load-time n_ubatch", "both refusals name the runtime n_ubatch, not the load-time one",
         (mutate_in_func(backend, r"static bool ggml_sycl_mmq_src1_ensure_for_graph\(",
                         "at the runtime \"", "at the load-time \""), common, cache, zone)),
        ("second commit", "only the success tail commits the guard, once, with no exit between",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "dense_guard.commit();", "dense_guard.commit(); dense_guard.commit();"), common, cache, zone)),
        ("early exit before the commit", "only the success tail commits the guard, once, with no exit between",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "dense_guard.commit();", "if (probe_mode) return refuse(\"x\"); dense_guard.commit();"),
          common, cache, zone)),
        ("guard built after the pools", "the guard is constructed before the transaction materializes its pools",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "ggml_sycl_dense_scratch_txn_guard dense_guard(*ctx, !probe_mode);",
                         "ggml_sycl_materialize_published_mmid_workspaces(); "
                         "ggml_sycl_dense_scratch_txn_guard dense_guard(*ctx, !probe_mode);"), common, cache, zone)),
        ("guard keeps the hold", "the guard zeroes the hold for its owner",
         (mutate_after(backend, "struct ggml_sycl_dense_scratch_txn_guard",
                       "unified_cache_set_planned_scratch_hold(device, 0, owner);", "(void) owner;"),
          common, cache, zone)),
        ("inputs not merged across models", "the load-time plan keeps another live model's inputs",
         (backend, common, mutate_all_in_func(cache, r"bool unified_cache_set_planned_mmq_src1_scratch\(",
                                              "zone_dense_scratch_merge_input(", "zone_XXXX(", 2), zone)),
        ("planning ignores live models", "the load-time plan keeps another live model's inputs",
         (mutate_after(backend, "const bool other_model_live", "ggml_sycl_other_backend_context_live(ctx->device, ctx)",
                       "false"), common, cache, zone)),
        ("Q8 setter stores before validating", "the Q8 setter validates before it stores",
         (backend, common, mutate_in_func(cache, r"bool unified_cache_set_planned_mmq_src1_scratch\(",
                                          "if (!zone_mmq_src1_scratch_bytes(",
                                          "g_planned_mmq_src1_bytes_per_token[device_id].store(bytes_per_token, "
                                          "std::memory_order_release); if (!zone_mmq_src1_scratch_bytes("), zone)),
        ("f16 setter stores before validating", "the f16 setter validates before it stores",
         (backend, common, mutate_in_func(cache, r"bool unified_cache_set_planned_dequant_f16_scratch\(",
                                          "if (!zone_dequant_f16_plan_bytes(",
                                          "g_planned_dequant_f16_weight_bytes_max[device_id].store(max_weight_bytes, "
                                          "std::memory_order_release); if (!zone_dequant_f16_plan_bytes("), zone)),
        ("held back without requiring a hold", "held back requires a hold, a request the zone could serve, and the hold's refusal",
         (backend, common, cache, mutate_in_func(zone, r"bool zone_runtime_alloc_held_back\(", "hold > 0 &&", "hold >= 0 &&"))),
        ("zone-full spill counts as held back", "held back requires a hold, a request the zone could serve, and the hold's refusal",
         (backend, common, cache, mutate_in_func(zone, r"bool zone_runtime_alloc_held_back\(",
                                                 "alloc_size <= zone_available", "alloc_size <= SIZE_MAX"))),
        ("bound ignores the largest request", "the spill bound is the plan plus the largest spill-capable request",
         (mutate_in_func(backend, r"static size_t ggml_sycl_planned_scratch_hold_spill_bound\(",
                         "unified_cache_get_runtime_request_hwm(", "unified_cache_XXXX("), common, cache, zone)),
        ("allocator does not record requests", "the allocator records the largest spill-capable RUNTIME request",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "unified_cache_note_runtime_request(", "unified_cache_XXXX("), zone)),
        ("recheck passes the current hold", "the transaction and the recheck ask the one bound",
         (mutate_in_func(backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\(",
                         "ggml_sycl_planned_scratch_hold_spill_bound(",
                         "ggml_sycl::unified_cache_get_planned_scratch_hold(ctx->device); (void) ggml_sycl_XXXX("),
          common, cache, zone)),
        ("FA-on skips the headroom check", "the FA-on early return runs the headroom check instead of skipping it",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_nonfa_attn_scratch\(",
                         "return ggml_sycl_check_hold_spill_headroom(device, hold_spill_bytes, probe_mode);",
                         "return true;"), common, cache, zone)),
        ("FA-on check ignores live memory", "the FA-on check compares the spill with the device's live free memory",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_headroom\(",
                         "ggml_backend_sycl_get_device_memory(", "ggml_backend_sycl_XXXX("), common, cache, zone)),
        ("replan ignores other live contexts", "the dense replan merges across live contexts too",
         (backend, common, mutate(cache, "uint32_t n_ubatch, bool other_model_live) {",
                                  "uint32_t n_ubatch) {"), zone)),
        ("transaction ignores other live contexts", "the transaction asks whether another context is live on THIS device",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "ggml_sycl_other_backend_context_live(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("Q8 overflow invalidates every model", "a rejected figure does not invalidate another live model's plan",
         (backend, common, mutate_in_func(cache, r"bool unified_cache_set_planned_mmq_src1_scratch\(",
                                          "if (!other_model_live) {", "if (true) {"), zone)),
        ("f16 overflow invalidates every model", "a rejected figure does not invalidate another live model's plan",
         (backend, common, mutate_in_func(cache, r"bool unified_cache_set_planned_dequant_f16_scratch\(",
                                          "if (!other_model_live) {", "if (true) {"), zone)),
        ("owner is not minted", "the owner token is a monotonic context id, not an address",
         (backend, mutate(common, "unified_cache_mint_planned_scratch_owner()", "0"), cache, zone)),
        ("owner is an address again", "the owner token is a monotonic context id, not an address",
         (mutate(backend, "unified_cache_release_planned_scratch_hold(device, planned_scratch_owner)",
                 "unified_cache_release_planned_scratch_hold(device, (uint64_t) (uintptr_t) this)"),
          common, cache, zone)),
        ("hold state without a mutex", "the hold state is one mutex-guarded unit",
         (backend, common, mutate(cache, "    std::mutex mutex;\n    size_t     hold ", "    int        mutex;\n    size_t     hold "), zone)),
        ("any context resets the spill count", "taking the hold spills is scoped to the context that owns them",
         (mutate_in_func(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)",
                         "planned_scratch_owner", "0"), common, cache, zone)),
        ("f16 arm condition written twice", "the f16 arm's compile condition has one source",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "#    if GGML_SYCL_DEQUANT_F16_ARM", "#    if GGML_SYCL_DNNL && defined(GGML_SYCL_F16)"),
          common, cache, zone)),
        ("f16 walk exit skips the refresh", "the f16 walk's success exit refreshes the hold",
         (mutate_in_func(backend, r"static bool ggml_sycl_dequant_f16_ensure_for_graph\(",
                         "ggml_sycl_planned_scratch_hold_refresh(ctx);\n#    else", "(void) ctx;\n#    else"),
          common, cache, zone)),
        ("guard refuses every spill", "the overcommit guard refuses only a hold-induced spill",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "if (hold_spill) {", "if (true) {"), zone)),
        ("realized check moved back into the recheck",
         "the recheck no longer claims the realized spill (it runs before the rung's worst-case reserves)",
         (mutate_in_func(backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\(",
                         "const auto current = ggml_sycl_global_plan_snapshot();",
                         "ggml_sycl_check_hold_spill_realized(0, 0, false); const auto current = ggml_sycl_global_plan_snapshot();"),
          common, cache, zone)),
        ("realized check ignores the pure rule", "the realized check asks the pure rule, the live free memory and the spills since this publish",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_realized\(",
                         "zone_hold_spill_realized_fits(", "zone_XXXX("), common, cache, zone)),
        ("realized check reads lifetime spills", "the realized check asks the pure rule, the live free memory and the spills since this publish",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_realized\(",
                         "unified_cache_get_recent_planned_hold_spills(", "unified_cache_take_planned_hold_spills("),
          common, cache, zone)),
        ("publish keeps the old epoch", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "unified_cache_begin_planned_hold_epoch(", "unified_cache_XXXX("), common, cache, zone)),
        ("epoch keeps the request mark", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(",
                                          "state.request_hwm          = 0;", "state.request_hwm          += 0;"), zone)),
        ("epoch keeps the spilled bytes", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(",
                                          "state.recent_bytes         = 0;", "state.recent_bytes         += 0;"), zone)),
        ("graph-entry check keeps its own 256", "the driver headroom the realized check uses is the graph-entry check's constant (one source)",
         (mutate_in_func(backend, r"static void ggml_sycl_check_graph_scratch_headroom\(",
                         "= kSyclArenaMinExternalHeadroomBytes;", "= 256ull * 1024ull * 1024ull;"), common, cache, zone)),
        ("replan trusts stale inputs", "a rejected figure is flagged, so no plan is derived from the stale inputs",
         (backend, common, mutate_in_func(mutate_in_func(cache, r"bool unified_cache_replan_planned_dense_scratch\(",
                                                         "g_planned_dense_scratch_invalid", "g_planned_XXXX"),
                                          r"bool unified_cache_replan_planned_dense_scratch\(",
                                          "g_planned_dense_scratch_invalid", "g_planned_XXXX"), zone)),

        # r3 backend-side mutants
        ("entry not registered for backend-DL", "the exported entry is registered for a backend-DL build",
         (mutate(backend, 'strcmp(name, "ggml_backend_sycl_planned_hold_spill_fits")',
                 'strcmp(name, "ggml_backend_sycl_XXXX")'), common, cache, zone)),
        ("entry checks another owner", "the exported entry asks the realized check for this backend's context and owner",
         (mutate_in_func(backend, r"bool ggml_backend_sycl_planned_hold_spill_fits\(", "planned_scratch_owner", "planned_XXXX"),
          common, cache, zone)),
        ("bound unscaled", "the spill bound scales the largest request to the candidate rung",
         (mutate_in_func(backend, r"static size_t ggml_sycl_planned_scratch_hold_spill_bound\(", "zone_hold_spill_bound(",
                         "zone_XXXX("), common, cache, zone)),
        ("FA-on check keeps a second criterion", "the FA-on check asks the realized rule (predicted free after the spill), not a second one",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_headroom\(", "zone_hold_spill_realized_fits(",
                         "zone_XXXX("), common, cache, zone)),
        ("request mark has no n_ubatch", "the largest request is recorded with the n_ubatch it was seen at, from the first publish on",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_note_runtime_request\(", "request_hwm_n_ubatch",
                                          "request_hwm_XXXX"), zone)),
        ("alloc takes the hold twice", "unified_alloc takes the hold state once per request",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "hold_at_decision  = unified_cache_note_runtime_request(",
                                          "hold_at_decision  = unified_cache_get_planned_scratch_hold(req.device) + "
                                          "unified_cache_note_runtime_request("), zone)),
        ("new owner inherits spill counts", "a new owner starts with its own spill counts",
         (backend, common, mutate_in_func(cache, r"void unified_cache_set_planned_scratch_hold\(", "spill_count", "spill_XXXX"),
          zone)),
        ("f16 dispatch arm outside the macro", "both f16 buffer acquisitions in the dispatch arm sit under the one macro",
         (mutate(backend, "#if GGML_SYCL_DEQUANT_F16_ARM", "#if 1", 1), common, cache, zone)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)
    ctx_mutants = [
        ("check before the reserve", "the hold-spill check runs in try_candidate, after sched_reserve() and before the rung is accepted",
         (mutate_after(context_src, "auto try_candidate = [&](uint32_t c) -> const char * {", "sched_reserve();",
                       "(void) hold_spill_fn(nullptr); sched_reserve();"), header_src)),
        ("hook dropped", "the hold-spill check runs in try_candidate, after sched_reserve() and before the rung is accepted",
         (context_src.replace("hold_spill_fn(", "hold_XXXX(", 1), header_src)),
        ("refusal keeps the rung", "a rung whose hold spill pushed the card under its headroom loses with its own stop reason",
         (context_src.replace("return \"hold spill left no headroom\";", "(void) 0;", 1), header_src)),
        ("no DL resolution", "the hook is resolved for a direct build and for a backend-DL build",
         (context_src.replace("\"ggml_backend_sycl_planned_hold_spill_fits\"", "\"ggml_backend_sycl_XXXX\"", 1), header_src)),
        ("header lacks the entry", "the header declares the exported entry",
         (context_src, header_src.replace("ggml_backend_sycl_planned_hold_spill_fits", "ggml_backend_sycl_XXXX"))),
    ]
    for label, expect, sources in ctx_mutants:
        failed += run_context(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
