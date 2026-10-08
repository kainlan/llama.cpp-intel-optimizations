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
parser.add_argument("--context-header", default=str(root / "src/llama-context.h"))
parser.add_argument("--auto-header", default=str(root / "src/llama-auto-ubatch.h"))
parser.add_argument("--cmake", default=str(root / "tests/CMakeLists.txt"))
parser.add_argument("--gpu-test", default=str(root / "tests/test-sycl-compute-buffer-kv-zone.cpp"))
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


def balanced_block(source, open_at):
    """The brace-balanced block whose '{' is at `open_at` (None when it never closes)."""
    depth = 0
    for i in range(open_at, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[open_at:i + 1]
    return None


def join_literals(source):
    """Adjacent C string literals are one string: join them so a message split over lines can be searched."""
    return re.sub(r'"\s*"', "", source)


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
    # The per-cohort line specifically: other [SCRATCH-STATS] lines (the hold spills, a compute buffer's landing zone)
    # print MB and are not cohort stats.
    stats_at = backend.find("[SCRATCH-STATS] device=%d cohort=")
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
    # The route is the wrapper that asks the router (scratch_route) plus the decided core both walks share
    # (scratch_route_decided); the checks read them together.
    route_core = (function_body(backend, r"static bool ggml_sycl_mul_mat_scratch_route\([^)]*\)\s*\{") or "") + \
        (function_body(backend, r"static bool ggml_sycl_mul_mat_scratch_route_decided\([^)]*\)\s*\{") or "")
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
    scratch_route_wrapper = function_body(backend, r"static bool ggml_sycl_mul_mat_scratch_route\([^)]*\)\s*\{")
    scratch_route_decided = function_body(
        backend, r"static bool ggml_sycl_mul_mat_scratch_route_decided\([^)]*\)\s*\{")
    scratch_route = None if None in (scratch_route_wrapper, scratch_route_decided) else \
        scratch_route_wrapper + scratch_route_decided
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
        "ggml_sycl_mul_mat_f16_dequant_route(" in dq_walk and dq_walk.count("matmul_orchestrator.select(") == 1
    results["both route predicates share one decision core"] = \
        "ggml_sycl_mul_mat_scratch_route(" in quantizing_route and \
        "ggml_sycl_mul_mat_scratch_route_decided(" in f16_route and \
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
        re.search(r"first\s*=\s*!\s*state\.warned_raw", note_spill_fn) is not None and \
        re.search(r"first\s*=\s*!\s*state\.warned_arena", note_spill_fn) is not None and \
        re.search(r"state\.spill_count\s*=\s*0\s*;", take_spill_fn) is not None and \
        re.search(r"state\.warned_raw\s*=\s*false\s*;", take_spill_fn) is not None
    results["the warning names the requester and the bytes"] = \
        "tag" in note_spill_fn and re.search(r"GGML_LOG_WARN\([^;]*%\.1f MB", note_spill_fn) is not None
    results["teardown reports the hold spills with the scratch stats"] = \
        "unified_cache_take_planned_hold_spills(" in stats_fn and "hold_spills_raw=" in stats_fn
    evict_at = unified_alloc_fn.find("evict_and_flush(")
    results["the overcommit guard refuses a hold-induced spill instead of evicting weights"] = \
        0 <= unified_alloc_fn.find("hold_spill") < evict_at and \
        re.search(r"if\s*\(\s*hold_spill\s*\)\s*\{[^}]*return false;", unified_alloc_fn[:evict_at]) is not None
    results["the held-back request still takes the ordinary spill path"] = \
        re.search(r"if\s*\(\s*!hold_spill\s*\)\s*\{[^}]*zone_alloc\(", unified_alloc_fn) is not None
    results["the outside-arena headroom check is told the worst-case hold spill"] = \
        "hold_query" in nonfa_check and \
        re.search(r"hold_query\s*\?\s*ggml_sycl_planned_scratch_hold_spill_bound\(\*hold_query\)\s*:\s*0", nonfa_check) is not None and \
        re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^;]*&hold_query", txn) is not None and \
        re.search(r"hold_query\s*=\s*\{[^}]*kv_pending", txn) is not None

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
    results["the allocator records the largest spill-capable RUNTIME request"] = \
        "unified_cache_note_runtime_request(" in unified_alloc_fn
    results["the transaction and the recheck ask the one bound"] = \
        re.search(r"hold_query\s*=\s*\{\s*ctx->device\s*,\s*ctx->planned_scratch_owner\s*,\s*next_kv_info\.n_ubatch", txn) is not None and \
        re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^;]*&recheck_hold_query", recheck_fn) is not None and "unified_cache_get_planned_scratch_hold(" not in recheck_fn

    # F3: flash attention is the default, so the by-name refusal must cover it too: the same live free-memory
    # comparison, with the spill figure as the whole demand.
    hold_headroom_fn = function_body(backend, r"static bool ggml_sycl_check_hold_spill_headroom\([^)]*\)\s*\{") or ""
    results["anchor: FA-on hold-spill headroom check exists"] = hold_headroom_fn != ""
    results["the FA-on early return runs the headroom check instead of skipping it"] = \
        re.search(r"if\s*\(\s*flash_attn_enabled\s*\)\s*\{\s*return\s*!hold_query\s*\|\|\s*ggml_sycl_check_hold_spill_headroom\(", nonfa_check) is not None
    results["the FA-on check compares the spill with the device's live free memory"] = \
        "ggml_sycl_hold_spill_fit(" in hold_headroom_fn and "GGML_SYCL_RUNTIME_TXN_REFUSAL" in hold_headroom_fn

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
        re.search(r"hold_fit_query\s+q\s*=\s*\{\s*device\s*,\s*owner\s*,\s*n_ubatch\s*,\s*0\s*,\s*true\s*\}", realized_fn) is not None and \
        "ggml_sycl_hold_spill_fit(" in realized_fn and "unified_cache_get_recent_planned_hold_spills(" in realized_fn and \
        "GGML_SYCL_RUNTIME_TXN_REFUSAL" in realized_fn
    results["the exported entry asks the realized check for this backend's context and owner"] = \
        entry_fn != "" and re.search(r"ggml_sycl_check_hold_spill_realized\([^;]*planned_scratch_owner", entry_fn) is not None
    results["the exported entry is registered for a backend-DL build"] = \
        re.search(r'strcmp\(name,\s*"ggml_backend_sycl_planned_hold_spill_fits"\)\s*==\s*0\)\s*\{\s*return \(void \*\)\s*ggml_backend_sycl_planned_hold_spill_fits;', backend) is not None
    results["the recheck no longer claims the realized spill (it runs before the rung's worst-case reserves)"] = \
        "ggml_sycl_check_hold_spill_realized(" not in recheck_fn
    results["a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)"] = \
        "unified_cache_begin_planned_hold_epoch(" in txn and \
        re.search(r"spill_bytes\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"spill_count\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"spill_arena_bytes\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"spill_arena_count\s*=\s*0\s*;", epoch_fn) is not None and "state.owner != owner" in epoch_fn and \
        re.search(r"baseline_owner\s*=\s*0\s*;", epoch_fn) is not None and "unified_cache_raw_device" not in epoch_fn
    results["the driver headroom the realized check uses is the graph-entry check's constant (one source)"] = \
        "kSyclArenaMinExternalHeadroomBytes" in realized_fn and "kSyclArenaMinExternalHeadroomBytes" in graph_headroom_fn and \
        re.search(r"arena_min_external_headroom\s*=\s*256", graph_headroom_fn) is None
    # I1: the bound follows the candidate rung, and the FA-on check and the realized check ask ONE question.
    # kpjw-g7: F3 and the -ub a refusal names must be one computation. F3 asks the shared predicate (the realized rule
    # applied to the predicted free memory); it keeps no inline copy of it.
    note_fn = function_body(cache, r"size_t unified_cache_note_runtime_request\([^)]*\)\s*\{") or ""
    results["the largest request is recorded with the n_ubatch it was seen at, from the first publish on"] = \
        re.search(r"r\.n_ubatch\s*==\s*state\.epoch_n_ubatch", note_fn) is not None and \
        re.search(r"r\.n_ubatch\s*=\s*state\.epoch_n_ubatch\s*;", note_fn) is not None and "rung_requests" in note_fn and "record" in note_fn
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

    # r3 design change (hardware: a 460-512 MB compute buffer spilled to RAW device memory, outside the arena, ate the
    # driver headroom): a compute buffer the RUNTIME zone will not serve (held back by the hold, or the zone is full)
    # is placed in the arena's KV zone first; raw device memory is the last resort. The decision is made in the
    # allocator, ahead of the overcommit guard (an in-arena placement cannot overcommit the device), so the hold's
    # spill is attributed to where it actually landed.
    ua = unified_alloc_fn
    pred_at = ua.find("zone_runtime_spill_prefers_kv_zone(")
    kv_at = ua.find("zone_alloc(vram_zone_id::KV")
    guard_at = ua.find("unified_alloc_total_vram(")
    raw_at = ua.find("unified_cache_malloc_device_tracked(")
    results["a compute buffer the RUNTIME zone will not serve tries the KV zone before the overcommit guard and raw memory"] = \
        0 <= pred_at < kv_at < guard_at < raw_at and "spill_to_kv_zone_before_raw" in ua and \
        re.search(r"zone_runtime_spill_prefers_kv_zone\([^;]*zone_available\(\s*vram_zone_id::KV\s*\)", ua) is not None
    results["a placement in the KV zone is skipped by the guard and the zone routing, and counts as an in-arena hold spill"] = \
        re.search(r"unified_cache_note_planned_hold_spill\([^;]*,\s*true\s*\)\s*;", ua) is not None and \
        re.search(r"unified_cache_note_planned_hold_spill\([^;]*,\s*false\s*\)\s*;", ua) is not None and \
        len(re.findall(r"!\s*kv_placed", ua)) >= 2 and "kv_placed" in ua
    kv_flag_site = backend.find('"backend-buffer-runtime-zone"')
    results["the runtime buffer allocator asks for the KV-zone-first placement"] = \
        kv_flag_site > 0 and "spill_to_kv_zone_before_raw = kv_zone_first" in backend[kv_flag_site:kv_flag_site + 1400]
    results["the hold-spill counters are split by where the buffer landed (raw outside the arena, KV zone)"] = \
        "in_arena" in note_spill_fn and "spill_arena_bytes" in cache and \
        "hold_spills_raw=" in stats_fn and "hold_spills_kv_zone=" in stats_fn
    results["the realized check counts only the raw outside-arena portion"] = \
        "raw_bytes" in realized_fn and "arena_bytes" not in realized_fn and "arena_count" not in realized_fn and \
        "zone_full" not in realized_fn
    hold_fit_fn = function_body(backend, r"static bool ggml_sycl_hold_spill_fit\([^)]*\)\s*\{") or ""
    kv_room_fn = function_body(backend, r"static size_t ggml_sycl_hold_kv_room\([^)]*\)\s*\{") or ""
    results["the transaction-time bound is the part of the worst-case spill the KV zone cannot take"] = \
        "zone_kv_room_for_compute(" in kv_room_fn and "zone_largest_free(ggml_sycl::vram_zone_id::KV)" in kv_room_fn and \
        "zone_available(ggml_sycl::vram_zone_id::KV)" not in kv_room_fn and "cache->arena_active()" in kv_room_fn and \
        "ggml_sycl_hold_kv_room(" in hold_fit_fn and \
        re.search(r"hold_kv_room\s*=\s*ggml_sycl_hold_kv_room\(\s*ctx->device\s*,\s*kv_pending\s*\)", txn) is not None and \
        re.search(r"unified_cache_begin_planned_hold_epoch\([^;]*hold_kv_room\s*\)", txn) is not None

    # ---- review r4 ---------------------------------------------------------------------------------------------
    # I3: the KV pre-placement block, window by window. A block-wide "contains" check is how mutants that change what
    # the block PUBLISHES (zone id, publication record, from_arena) or COUNTS survived r3.
    kv_start = ua.find("bool kv_placed")
    kv_end = ua.find("if (tier == alloc_tier::DEVICE_VRAM) {", kv_start) if kv_start >= 0 else -1
    kvb = ua[kv_start:kv_end] if 0 <= kv_start < kv_end else ""
    results["anchor: the KV pre-placement block exists"] = kvb != ""
    results["the KV pre-placement runs only for the flagged RUNTIME class, in an active arena"] = \
        "spill_to_kv_zone_before_raw" in kvb and \
        re.search(r"prefer_vram_zone\s*==\s*vram_zone_id::RUNTIME", kvb) is not None and \
        re.search(r"kv_cache\s*&&\s*kv_cache->arena_active\(\)", kvb) is not None
    results["the KV pre-placement is told the request's forbid-spill flag"] = \
        re.search(r"zone_runtime_spill_prefers_kv_zone\(\s*true\s*,\s*true\s*,\s*req\.intent\.constraints\.forbid_vram_zone_spill\s*,", kvb) is not None
    results["the KV-zone-first flag is set by exactly one request, the scheduler compute buffer's"] = \
        len(re.findall(r"spill_to_kv_zone_before_raw", backend)) == 1
    results["the KV pre-placement reads the zone-full arm as well as the hold"] = \
        re.search(r"hold_spill\s*\|\|\s*kv_cache->zone_available\(\s*vram_zone_id::RUNTIME\s*\)\s*<\s*alloc_size", kvb) is not None
    results["the KV pre-placement publishes through the KV zone (record, allocation, metadata)"] = \
        re.search(r"prepare_arena_publication\(\s*vram_zone_id::KV\s*\)", kvb) is not None and \
        re.search(r"zone_alloc\(\s*vram_zone_id::KV\s*,\s*alloc_size\s*,\s*req\.alignment\s*!=\s*0\s*\?\s*req\.alignment\s*:\s*64", kvb) is not None and \
        re.search(r"output_metadata\.vram_zone\s*=\s*vram_zone_id::KV\s*;", kvb) is not None and \
        re.search(r"output_metadata\.zone_managed\s*=\s*true\s*;", kvb) is not None and \
        re.search(r"\bfrom_arena\s*=\s*true\s*;", kvb) is not None
    results["a publication that was attempted and failed is not retried raw"] = \
        re.search(r"if\s*\(\s*!ptr\s*&&\s*arena_publication\.attempted\s*\)\s*\{\s*return false;\s*\}", kvb) is not None and \
        re.search(r"if\s*\(\s*!prepare_arena_publication\(\s*vram_zone_id::KV\s*\)\s*\)\s*\{\s*return false;\s*\}", kvb) is not None
    results["a hold-induced KV placement counts as an in-arena hold spill, a zone-full one is counted and warned separately"] = \
        re.search(r"if\s*\(\s*hold_spill\s*\)\s*\{\s*unified_cache_note_planned_hold_spill\([^;]*true\s*\)\s*;\s*\}\s*else\s*\{\s*"
                  r"unified_cache_note_zone_full_kv_placement\(", kvb) is not None
    take_fn = function_body(cache, r"void unified_cache_take_planned_hold_spills\([^)]*\)\s*\{") or ""
    zone_full_fn = function_body(cache, r"void unified_cache_note_zone_full_kv_placement\([^)]*\)\s*\{") or ""
    results["anchor: the zone-full KV placement note exists"] = zone_full_fn != ""
    results["the zone-full KV placement has its own counter and a once-only WARN, separate from the hold spills"] = \
        "zone_full_count++" in zone_full_fn and "GGML_LOG_WARN" in zone_full_fn and "warned_zone_full" in zone_full_fn and \
        "spill_count++" not in zone_full_fn and "spill_arena_count++" not in zone_full_fn
    results["the landing-site counters stay separate (the KV-zone counter is not the raw one)"] = \
        re.search(r"if\s*\(\s*in_arena\s*\)\s*\{[^{}]*spill_arena_count\+\+\s*;[^{}]*\}\s*else\s*\{[^{}]*spill_count\+\+\s*;", note_spill_fn) is not None
    results["a take hands every counter to the owner and clears every one"] = all(
        re.search(rf"state\.{f}\s*=\s*(0|false)\s*;", take_fn) is not None
        for f in ("spill_count", "spill_bytes", "spill_arena_count", "spill_arena_bytes", "zone_full_count", "zone_full_bytes",
                  "spill_owner", "warned_raw", "warned_arena", "warned_zone_full"))
    results["the stats line is printed whatever landed, not only for raw spills"] = \
        re.search(r"if\s*\(\s*hold_spills\.raw_count\s*!=\s*0\s*\|\|\s*hold_spills\.arena_count\s*!=\s*0\s*\|\|\s*"
                  r"hold_spills\.zone_full_count\s*!=\s*0\s*\)", stats_fn) is not None and \
        "hold_spills_kv_zone_full=" in stats_fn
    # m5: the counters the final context reports are its own: a publish restarts them, so a losing rung's spills do
    # not appear in what the finished context prints, and the once-only WARN latches survive the restart.
    results["a publish restarts the counters the teardown stats report, and not the WARN latches"] = \
        re.search(r"zone_full_count\s*=\s*0\s*;", epoch_fn) is not None and \
        re.search(r"spill_owner\s*=\s*0\s*;", epoch_fn) is not None and "warned_" not in epoch_fn and \
        "recent_count" not in cache and "recent_arena_count" not in cache

    # m4: the flag reaches ONLY scheduler compute buffers. Model-weight backing buffers come from the same buffer type
    # during a model load, before the KV cache exists; one sent to the KV zone would take this context's KV room.
    flag_site = backend.find('"backend-buffer-runtime-zone"')
    flag_win = backend[flag_site:flag_site + 1400] if flag_site > 0 else ""
    # m3: for the flagged class unified_alloc already tried the KV zone, so the caller's own KV step is a repeat and its
    # SCRATCH step is a placement the graph-entry headroom check aborts on; both are skipped for it.
    guard_re = re.search(r"if\s*\(\s*!kv_zone_first\s*&&\s*\(\s*alloc_role\s*==\s*ggml_sycl::alloc_role::COMPUTE\s*\|\|\s*should_use_runtime\s*\)\s*\)\s*\{",
                         backend[flag_site:flag_site + 6000]) if flag_site > 0 else None
    scratch_at = backend.find("scratch_req.queue", flag_site) if flag_site > 0 else -1
    kvreq_at = backend.find("kv_req.queue", flag_site) if flag_site > 0 else -1
    results["the caller's KV and SCRATCH fallbacks are unreachable for the flagged class"] = \
        guard_re is not None and kvreq_at > 0 and scratch_at > kvreq_at and \
        flag_site + guard_re.start() < kvreq_at < scratch_at

    # Which zone each flagged compute buffer landed in is printed per buffer, by name and size (a throughput
    # difference between zones would otherwise be invisible).
    landing_fn = function_body(backend, r"static void ggml_sycl_log_compute_buffer_landing\([^)]*\)\s*\{") or ""
    results["each flagged compute buffer's landing zone is printed by name and size"] = \
        "[SCRATCH-STATS]" in landing_fn and "compute_buffer=%s" in landing_fn and "size=%.1f MB" in landing_fn and \
        "zone=%s" in landing_fn and "GGML_LOG_WARN" in landing_fn and \
        re.search(r"if\s*\(\s*ggml_backend_buffer_t\s+published\s*=\s*ggml_backend_sycl_buffer_publish\(\s*buft\s*,\s*ctx\s*,\s*size\s*,\s*\"arena RUNTIME zone\"\s*\)\s*\)\s*\{\s*"
                  r"if\s*\(\s*kv_zone_first\s*\)\s*\{\s*ggml_sycl_log_compute_buffer_landing\(\s*buft_ctx->device\s*,\s*buft_ctx->name\s*,\s*size\s*,\s*landed_zone\s*\)",
                  backend[flag_site:flag_site + 3200]) is not None and \
        re.search(r"const char \*\s+landed_zone\s*=\s*ggml_sycl_compute_buffer_zone\(\s*runtime_h\s*\)\s*;\s*ggml_backend_sycl_buffer_context \*\s*ctx\s*=", backend[flag_site:flag_site + 3200]) is not None
    # I1: the transaction's bound is net of the KV the same transaction is about to place, and works with the
    # largest free block (a buffer is indivisible), not the sum of the zone's free bytes.
    results["the transaction passes the KV bytes its plan adds; the recheck, which runs with KV live, passes none"] = \
        re.search(r"const size_t\s+kv_with_slack\s*=\s*ggml_sycl_device_kv_bytes_with_slack\(\s*next_plan", txn) is not None and \
        re.search(r"const size_t\s+kv_pending\s*=[^;]*kv_with_slack[^;]*kv_admitted", txn) is not None and \
        re.search(r"next_kv_info\.n_ubatch\s*,\s*kv_pending\s*,\s*false\s*\}", txn) is not None and \
        re.search(r"planner_n_ubatch\s*,\s*0\s*,\s*true\s*\}", recheck_fn) is not None
    results["the realized check names the largest -ub that fits, through the exported entry"] = \
        re.search(r"\*largest_ub\s*=\s*a\.largest_ub\s*;", realized_fn) is not None and "largest_ub" in entry_fn and \
        re.search(r"bool ggml_backend_sycl_planned_hold_spill_fits\(\s*ggml_backend_t\s+backend\s*,\s*uint32_t\s+n_ubatch\s*,\s*uint32_t\s*\*\s*largest_ub", backend) is not None
    results["the F3 refusal says its figure is net of the KV-zone room"] = "net of the KV" in hold_headroom_fn
    # r5 T1: the KV this context already admitted is not pending. Forcing it to 0 makes every rung above the first
    # look like it is about to place all of its KV again, which shrinks the room and over-refuses it.
    results["the transaction nets out the KV already admitted, so a rung above the first is not over-refused"] = \
        re.search(r"admitted_kv\s*=\s*ctx->runtime_kv_admitted\s*\?\s*current->plan\.get\(\)\s*:\s*nullptr", txn) is not None and \
        re.search(r"const size_t\s+kv_admitted\s*=\s*admitted_kv\s*\?\s*ggml_sycl_device_kv_bytes_with_slack\(\s*\*admitted_kv\s*,\s*ctx->device\s*\)\s*:\s*0\s*;", txn) is not None
    # r5 T6/T7/T8: the -ub a refusal names is computed from what THIS rung measured, with the rung's own n_ubatch, and
    # the out-parameter is zeroed first so a refusal-free call (or a backend that is not SYCL) never reports a stale one.
    # kpjw-g7: the -ub the refusal prints must be one F3 accepts. The spill's linear share alone named 512 for a pinned
    # -ub 1024 on the B50 while F3 refused 512 (470 MB worst case, 132.7 MB left), so the advice died with result=19.
    # The name is the smaller of that share and the largest rung F3's own predicate accepts, asked with F3's own bound
    # for each rung and the card as it was before this plan's raw buffers (free now + the spill).
    results["the -ub a refusal names is one the F3 publish accepts: the smaller of the spill's share and F3's own answer"] = \
        re.search(r"a\.largest_ub\s*=\s*ggml_sycl::zone_hold_fit_largest_ub\(\s*in\s*,\s*q\.n_ubatch\s*\)", hold_fit_fn) is not None and \
        re.search(r"\*largest_ub\s*=\s*a\.largest_ub\s*;", realized_fn) is not None and \
        "a.largest_ub" in hold_headroom_fn and \
        "zone_hold_fit_largest_ub(" not in realized_fn + hold_headroom_fn
    zero_at = re.search(r"if\s*\(\s*largest_ub\s*\)\s*\{\s*\*largest_ub\s*=\s*0\s*;\s*\}", entry_fn)
    results["the exported entry zeroes the out-parameter first and passes the caller's n_ubatch to the realized check"] = \
        zero_at is not None and zero_at.start() < entry_fn.find("return ggml_sycl_check_hold_spill_realized(") and \
        re.search(r"return ggml_sycl_check_hold_spill_realized\(\s*ctx->device\s*,\s*ctx->planned_scratch_owner\s*,\s*true\s*,\s*"
                  r"n_ubatch\s*,\s*largest_ub\s*\)", entry_fn) is not None
    # r5 minor 1: the F3 message printed the same number twice and called the CURRENT free memory what the spill
    # "would leave". It names the free memory after the worst-case spill and, separately, the free memory now.
    results["the F3 refusal prints the free memory it would leave and the free memory now, each once"] = \
        re.search(r"a\.demand\s*/\s*mb\s*,\s*q\.device\s*,\s*free_after\s*/\s*mb\s*,\s*a\.free_before\s*/\s*mb\s*,\s*"
                  r"kSyclArenaMinExternalHeadroomBytes\s*/\s*mb\s*,", hold_headroom_fn) is not None and \
        "would leave device %d %.1f MB free (of %.1f MB free before it)" in join_literals(hold_headroom_fn)
    # r5 F3/F4: the landing line names the zone the bytes physically are in. A swapped name sends a reader to the wrong
    # allocator (KV printed as "runtime", a buffer outside the arena printed as "kv").
    zone_names = {"KV": "kv", "WEIGHT": "weight", "ONEDNN": "onednn", "RUNTIME": "runtime", "SCRATCH": "scratch"}
    zone_fn = function_body(backend, r"static const char \* ggml_sycl_compute_buffer_zone\([^)]*\)\s*\{") or ""
    results["the landing line names each zone as itself, a buffer outside the arena as raw, host memory as host-pinned"] = \
        zone_fn != "" and 'const char * zone = "none";' in zone_fn and \
        all(re.search(rf'case ggml_sycl::vram_zone_id::{k}:\s*zone = "{v}";\s*break;', zone_fn) is not None
            for k, v in zone_names.items()) and \
        re.search(r'default:\s*zone = "raw";\s*break;', zone_fn) is not None and \
        re.search(r'if\s*\(\s*handle\.tier\s*==\s*ggml_sycl::alloc_tier::HOST_PINNED\s*\)\s*\{\s*zone = "host-pinned";', zone_fn) is not None
    # r5 minor 7: a flagged buffer nothing in the arena placed is placed by the legacy path below, and ITS landing (raw
    # device memory or host-pinned) is the one that matters; the first line must not say "none" and stop there.
    alloc_ok_at = backend.find("alloc_succeeded:")
    legacy_log_at = backend.find("ggml_sycl_log_compute_buffer_landing(buft_ctx->device, buft_ctx->name, size, main_alloc)", alloc_ok_at) \
        if alloc_ok_at > 0 else -1
    legacy_move_at = backend.find("set_managed_owner(std::move(main_alloc))", alloc_ok_at) if alloc_ok_at > 0 else -1
    legacy_run = backend[backend.find('"backend-buffer-runtime-zone"'):]
    results["a flagged buffer the legacy path places gets its final landing line, from the handle that path made"] = \
        alloc_ok_at > 0 and \
        re.search(r'const char \*\s+legacy_zone\s*=\s*legacy_landing_pending\s*\?\s*ggml_sycl_compute_buffer_zone\(\s*main_alloc\s*\)\s*:\s*"none"\s*;',
                  backend[alloc_ok_at:alloc_ok_at + 600]) is not None and \
        re.search(r"legacy_published\s*=\s*ggml_backend_sycl_buffer_publish\([^;]*\)\s*;\s*if\s*\(\s*legacy_published\s*&&\s*legacy_landing_pending\s*\)\s*\{\s*"
                  r"ggml_sycl_log_compute_buffer_landing\(\s*buft_ctx->device\s*,\s*buft_ctx->name\s*,\s*size\s*,\s*legacy_zone\s*\)", backend[alloc_ok_at:]) is not None and \
        re.search(r"bool\s+legacy_landing_pending\s*=\s*false\s*;", backend[:backend.find('"backend-buffer-runtime-zone"')]) is not None and \
        re.search(r"if\s*\(\s*kv_zone_first\s*&&\s*!runtime_h\.ptr\s*\)\s*\{\s*legacy_landing_pending\s*=\s*true\s*;", legacy_run[:2600]) is not None and \
        re.search(r"\"arena RUNTIME zone\"\s*\)\s*\)\s*\{[^{}]*\{[^{}]*\}[^{}]*return published;\s*\}\s*if\s*\(\s*kv_zone_first\s*\)\s*\{\s*legacy_landing_pending\s*=\s*true\s*;", legacy_run[:3200]) is not None

    # kpjw-g7 unification (P4: one fact, one source). F3 (the transaction-time bound), the ladder's realized check
    # and the pinned-ub check were three computations over one fact (two predicates, a history-carrying high-water
    # mark, and a driver free-memory read taken after a release, whose credit lags: 602.7 MB read where 1097 MB was
    # true). They are ONE function now, over the plan, the rung's own recorded request, the KV room net of what is
    # pending, and the cache's own ledger of free memory.
    hold_fit_fn = function_body(backend, r"static bool ggml_sycl_hold_spill_fit\([^)]*\)\s*\{") or ""
    results["anchor: the one hold-spill fit function exists"] = hold_fit_fn != ""
    unified_consumers = {"the F3 headroom check": hold_headroom_fn, "the realized check": realized_fn,
                         "the spill-demand bound": bound_fn}
    results["F3, the realized check and the demand bound all ask the one fit function"] = \
        all("ggml_sycl_hold_spill_fit(" in body for body in unified_consumers.values())
    own_formula = ("zone_hold_spill_bound_fits(", "zone_hold_spill_realized_fits(", "zone_hold_spill_bound(",
                   "zone_hold_spill_raw_demand(", "zone_hold_spill_largest_ub", "ggml_backend_sycl_get_device_memory(",
                   "unified_cache_get_runtime_request_hwm(")
    results["none of the three keeps a formula, a free-memory read or a high-water mark of its own"] = \
        all(not any(f in body for f in own_formula) for body in unified_consumers.values())
    results["the fit function composes the zone predicate over the plan, the rung's record, the KV room and the ledger"] = \
        hold_fit_fn != "" and all(f in hold_fit_fn for f in (
            "zone_hold_fit(", "zone_hold_fit_largest_ub(", "unified_cache_hold_free_before(",
            "unified_cache_get_hold_rung_requests(", "kSyclArenaMinExternalHeadroomBytes", "ggml_sycl_hold_kv_room(",
            "ggml_sycl_hold_plan_at")) and \
        "unified_cache_planned_dense_scratch_bytes_at(" in (function_body(backend, r"static size_t ggml_sycl_hold_plan_at\([^)]*\)\s*\{") or "") and \
        "unified_cache_get_runtime_request_hwm(" not in hold_fit_fn
    free_before_fn = function_body(cache, r"size_t unified_cache_hold_free_before\([^)]*\)\s*\{") or ""
    results["the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read"] = \
        free_before_fn != "" and all(f in free_before_fn for f in (
            "unified_cache_raw_device_held_bytes(", "zone_hold_free_cold(", "zone_hold_cold_update(",
            "zone_hold_persistent_raw(", "zone_hold_free_before(")) and \
        re.search(r"state\.baseline_cold\s*=\s*zone_hold_cold_update\(\s*state\.baseline_owner\s*==\s*owner\s*,\s*state\.baseline_cold\s*,\s*cand\s*\)", free_before_fn) is not None and \
        re.search(r"zone_hold_persistent_raw\(\s*raw_held\s*,\s*compute_live\s*,\s*rung_live\s*\)", free_before_fn) is not None and \
        "get_device_memory" not in free_before_fn
    results["the request mark is per rung and survives a publish (no history-carrying high-water mark)"] = \
        "request_hwm" not in cache and "rung_requests" in cache and \
        "rung_requests" not in epoch_fn and "epoch_kv_room" in epoch_fn
    results["the zone header declares the one fit and the ledger arithmetic, and no second predicate walk"] = \
        all(f in zone for f in ("zone_hold_fit_inputs", "zone_hold_fit(", "zone_hold_fit_largest_ub(",
                                "zone_hold_free_cold(", "zone_hold_free_before(")) and \
        "zone_hold_spill_largest_ub" not in zone
    # kpjw-g7 A: a scheduler compute buffer is identified positively, by allocation origin. The recurrent-state
    # buffer (461.3 MB, cache_r_l*/cache_s_l*) was allocated through the same buffer type outside a model load and
    # was flagged compute by the `!in_model_load` timing discriminator, so it fed the request mark and the spill
    # counters.
    kv_first_m = re.search(r"const bool\s+kv_zone_first\s*=\s*([^;]*);", backend)
    results["a compute buffer is flagged by an explicit scheduler scope, not by the absence of a model load"] = \
        kv_first_m is not None and "compute_alloc_scope" in kv_first_m.group(1) and \
        "g_sycl_in_model_load" not in kv_first_m.group(1) and \
        re.search(r"void ggml_backend_sycl_compute_alloc_scope\(\s*bool\s+\w+\s*\)", backend) is not None and \
        re.search(r'strcmp\(name,\s*"ggml_backend_sycl_compute_alloc_scope"\)\s*==\s*0\)\s*\{\s*return \(void \*\)\s*ggml_backend_sycl_compute_alloc_scope;', backend) is not None
    results["only a flagged compute request feeds the request mark and the hold-spill counters"] = \
        re.search(r"unified_cache_note_runtime_request\(\s*req\.device\s*,\s*alloc_size\s*,[^;]*spill_to_kv_zone_before_raw", unified_alloc_fn) is not None and \
        re.search(r"if\s*\(\s*req\.intent\.constraints\.spill_to_kv_zone_before_raw\s*\)\s*\{\s*unified_cache_note_planned_hold_spill\([^;]*false\s*\)", unified_alloc_fn) is not None
    return results


def evaluate_context(context, header, ctx_header=None, auto_header=None):
    ctx_header = ctx_header_src if ctx_header is None else ctx_header
    auto_header = auto_header_src if auto_header is None else auto_header
    """r3 C1: WHERE the hold-spill fit check runs in the auto-ubatch trial. A rung's compute buffers exist only once
    sched_reserve() has returned, so the check belongs in try_candidate, after it, for every rung, every
    flash-attention mode and the cached rung -- not in the recheck inside the reserve."""
    results = {}
    try_fn = function_body(context, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{") or ""
    select_fn = function_body(context, r"void llama_context::sycl_select_auto_ubatch\([^)]*\)\s*\{") or ""
    results["anchor: sycl_select_auto_ubatch exists"] = select_fn != ""
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
        re.search(r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_planned_hold_spill_fits\(\s*ggml_backend_t\s+backend\s*,\s*uint32_t\s+n_ubatch\s*,\s*uint32_t\s*\*\s*largest_ub", header) is not None
    results["the trial asks the entry with the rung's n_ubatch"] = \
        re.search(r"hold_spill_fn\(\s*sb\.backend\s*,\s*c\s*,", try_fn) is not None

    # r4 I2: the previous rung's compute buffers are released BEFORE this rung's transaction, so the KV headroom the
    # transaction (and the probe) measures is not depressed by buffers a rung that already lost left in the KV zone.
    probe_at = try_fn.find("probe_fn(")
    rel_m = re.search(r"auto release_rung_buffers = \[&\]\(\) \{", context)
    rel_block = balanced_block(context, rel_m.end() - 1) if rel_m else ""
    results["the previous rung's compute buffers are released before the next rung's transaction"] = \
        rel_block != "" and "synchronize();" in rel_block and "sched.reset();" in rel_block and \
        re.search(r"gf_res_reserve\.reset\(\)", rel_block) is not None and \
        re.search(r"sched_need_reserve\s*=\s*true\s*;", rel_block) is not None and \
        re.search(r"sched_matches_last_good\s*=\s*false\s*;", rel_block) is not None and \
        re.search(r"^\{\s*rung_fit_refused\s*=\s*false\s*;\s*release_rung_buffers\(\)\s*;", try_fn) is not None and \
        0 <= try_fn.find("release_rung_buffers();") < probe_at
    # r4 I1: the realized check also runs after a reserve the ladder did not make (a pinned -ub, a ladder that never
    # ran), by name, with the largest -ub that fits; and it runs after the WHOLE trial/else block.
    trial_at = context.find("sycl_select_auto_ubatch();")
    else_reserve_at = context.find("sched_reserve();", trial_at)
    call_at = context.find("llama_context_sycl_hold_spill_fits(backends", trial_at)
    guard_at = context.find("quantized V cache was requested", trial_at)
    results["the realized hold-spill check also runs after a reserve the ladder did not make"] = \
        0 <= trial_at < else_reserve_at < call_at < guard_at
    results["a refusal there is a context-init refusal naming the largest -ub that fits"] = \
        call_at > 0 and "largest -ub that fits is about" in context[call_at:call_at + 1600] and \
        "throw std::runtime_error(" in context[call_at:call_at + 1600]

    # r5 I-A: the context-init refusal is the only thing standing between a pinned -ub and the first graph's hang, and
    # a position check cannot see it neutered. Pin the helper (any refusing backend refuses; the -ub named is the
    # smallest the refusers name; the rung's own n_ubatch goes down), the call (taken for any SYCL backend, with
    # cparams.n_ubatch), and the throw (INSIDE the refusing branch, not borrowed from the unrelated quantized-V throw
    # a character window further on).
    helper = function_body(context, r"static bool llama_context_sycl_hold_spill_fits\([^)]*\)\s*\{") or ""
    results["anchor: the hold-spill helper exists"] = helper != ""
    refuse_m = re.search(r"if\s*\(\s*!hold_spill_fn\(\s*backend\.get\(\)\s*,\s*n_ubatch\s*,\s*&backend_largest\s*\)\s*\)\s*\{", helper)
    refuse_block = balanced_block(helper, refuse_m.end() - 1) if refuse_m else ""
    merge_m = re.search(r"\*largest_ub\s*=\s*fits\s*\?\s*backend_largest\s*:\s*std::min\(\s*\*largest_ub\s*,\s*backend_largest\s*\)\s*;", refuse_block or "")
    fits_false_m = re.search(r"\bfits\s*=\s*false\s*;", refuse_block or "")
    results["the helper refuses when ANY backend's check refuses, and passes the rung's own n_ubatch down"] = \
        refuse_m is not None and fits_false_m is not None and re.search(r"bool\s+fits\s*=\s*true\s*;", helper) is not None and \
        re.search(r"return\s+fits\s*;\s*\}\s*$", helper) is not None
    results["the helper keeps the smallest -ub the refusing backends name (the first refuser's, then the minimum)"] = \
        merge_m is not None and fits_false_m is not None and merge_m.start() < fits_false_m.start()
    results["the helper skips a backend that is not SYCL and zeroes the out-parameter first"] = \
        re.search(r"if\s*\(\s*!llama_context_dev_is_sycl\(dev\)\s*\)\s*\{\s*continue\s*;", helper) is not None and \
        re.search(r"^\{\s*if\s*\(\s*largest_ub\s*\)\s*\{\s*\*largest_ub\s*=\s*0\s*;\s*\}", helper) is not None
    ctor_m = re.compile(r"if\s*\(\s*llama_context_has_sycl_backend\(backends\)\s*&&\s*sycl_hold_spill_validated_ub\s*!=\s*"
                        r"cparams\.n_ubatch\s*\)\s*\{").search(context, max(trial_at, 0))
    ctor_block = balanced_block(context, ctor_m.end() - 1) if ctor_m else ""
    inner_m = re.search(r"if\s*\(\s*!llama_context_sycl_hold_spill_fits\(\s*backends\s*,\s*cparams\.n_ubatch\s*,\s*&largest_ub\s*\)\s*\)\s*\{",
                        ctor_block or "")
    inner_block = balanced_block(ctor_block, inner_m.end() - 1) if inner_m else ""
    results["the context-init check runs for any SYCL backend unless the trial already validated this -ub"] = \
        ctor_m is not None and else_reserve_at < ctor_m.start() < guard_at and ctor_block != "" and \
        "quantized V cache" not in ctor_block
    results["the context-init check asks with cparams.n_ubatch and its refusal is a throw inside the refusing branch"] = \
        inner_m is not None and "throw std::runtime_error(" in inner_block and \
        "largest -ub that fits is about" in inner_block and "LLAMA_LOG" not in inner_block

    # r5 minor 4: the ctor check re-reads live free memory after the ladder's own per-rung check passed, and the two
    # readings can disagree at the margin and refuse a winner the ladder just validated. The trial records the -ub it
    # validated for the sched that is final; anything else (a pinned -ub, an early exit, a settle that re-reserved,
    # a missing hook) leaves 0 and the ctor check runs.
    results["the context records the -ub its trial validated, for the sched that is final, and nothing else"] = \
        re.search(r"uint32_t\s+sycl_hold_spill_validated_ub\s*=\s*0\s*;", ctx_header) is not None and \
        re.search(r"^\{\s*sycl_hold_spill_validated_ub\s*=\s*0\s*;", select_fn) is not None and \
        re.search(r"sycl_hold_spill_validated_ub\s*=\s*\(\s*sched_matches_last_good\s*&&\s*cparams\.n_ubatch\s*==\s*last_good\s*\)\s*\?\s*"
                  r"hold_spill_validated_ub\s*:\s*0\s*;", select_fn) is not None and \
        select_fn.find("sycl_hold_spill_validated_ub = (") < select_fn.find("if (!sched_matches_last_good || cparams.n_ubatch != last_good)") and \
        re.search(r"hold_spill_validated_ub\s*=\s*hold_spill_fn\s*\?\s*c\s*:\s*0\s*;\s*return nullptr;", try_fn) is not None and \
        re.search(r"sched_matches_last_good\s*=\s*false\s*;\s*hold_spill_validated_ub\s*=\s*0\s*;", rel_block) is not None

    # kpjw-g6 (item 0) and g7: the default rung refused must not turn a loadable model into an init failure. B50,
    # Qwen3.6-27B, auto n_ubatch: 512 spills 495 MB outside the arena and leaves 107.7 MB against the 256 MB headroom.
    # A smaller -ub is not a smaller context: when nothing at or above the default wins AND the loss was a real fit
    # refusal, the trial continues DOWNWARD (256, 128, 64) and settles on the first rung that fits. A pinned -ub never
    # reaches the trial and still refuses by name. A refused settle is a named error whose -ub comes from the one
    # hold-spill fit function (there is no settle descent: two predicates over one fact).
    results["the pure downward walk exists: halves to a multiple of 64, floor 64, rungs above the cap skipped"] = \
        re.search(r"llama_auto_ubatch_descent_floor\s*=\s*64\s*;", auto_header) is not None and \
        re.search(r"uint32_t\s+llama_auto_ubatch_next_lower\(\s*uint32_t\s+from\s*\)", auto_header) is not None and \
        re.search(r"const uint32_t\s+half\s*=\s*from\s*/\s*2\s*;\s*return\s+half\s*>=\s*llama_auto_ubatch_descent_floor\s*\?\s*half\s*-\s*half\s*%\s*64\s*:\s*0\s*;", auto_header) is not None and \
        re.search(r"uint32_t\s+llama_auto_ubatch_descend\(\s*uint32_t\s+fallback\s*,\s*uint32_t\s+cap\s*,\s*F\s+try_rung\s*\)", auto_header) is not None and \
        re.search(r"if\s*\(\s*c\s*>\s*cap\s*\)\s*\{\s*continue\s*;", auto_header) is not None and \
        re.search(r"if\s*\(\s*try_rung\(\s*c\s*\)\s*\)\s*\{\s*return c\s*;", auto_header) is not None
    advice_norm = re.sub(r"\s+", " ", auto_header)
    results["the advice caps the fit function's answer under the lowest refused rung and names nothing under the floor"] = \
        re.search(r"inline uint32_t llama_auto_ubatch_advice\(uint32_t largest_fit, uint32_t lowest_refused\) \{", advice_norm) is not None and \
        "if (largest_fit == 0) { return 0; }" in advice_norm and \
        "if (lowest_refused == 0 || largest_fit < lowest_refused) { return largest_fit; }" in advice_norm and \
        re.search(r"while \(p <= lowest_refused / 2 && p \* 2 < lowest_refused\) \{ p \*= 2; \}", advice_norm) is not None and \
        "return lowest_refused > 1 && p >= llama_auto_ubatch_descent_floor ? p : 0;" in advice_norm
    results["there is no settle refusal descent"] = \
        "settle_refusal_descend" not in auto_header and "settle_refusal_descend" not in context and \
        "refusal_largest_ub" not in context
    loop_at = select_fn.find("for (uint32_t c : rung_ladder) {")
    fallback_assign_at = select_fn.find("if (last_good == 0) {")
    descent_m = re.search(r"if\s*\(\s*last_good\s*==\s*0\s*&&\s*ladder_needed\s*&&\s*fallback_tried\s*&&\s*rung_fit_refused\s*\)\s*\{", select_fn)
    descent_block = balanced_block(select_fn, descent_m.end() - 1) if descent_m else ""
    results["the trial continues downward when the default rung and everything above it lost to a real fit refusal, and only then"] = \
        descent_m is not None and 0 < loop_at < descent_m.start() < fallback_assign_at and \
        re.search(r"llama_auto_ubatch_descend\(\s*fallback_ubatch\s*,\s*cap\s*,", descent_block) is not None and \
        re.search(r"const char \*\s*rung_reason\s*=\s*try_candidate\(c\)\s*;\s*if\s*\(\s*rung_reason\s*==\s*nullptr\s*\)\s*\{\s*return true\s*;\s*\}", descent_block) is not None and \
        re.search(r"note_loss\(\s*c\s*,\s*rung_reason\s*\)\s*;\s*descent_aborted\s*=\s*!rung_fit_refused\s*;\s*return descent_aborted\s*;", descent_block) is not None and \
        re.search(r"if\s*\(\s*ended\s*!=\s*0\s*&&\s*!descent_aborted\s*\)\s*\{", descent_block) is not None and \
        re.search(r"last_good\s*=\s*ended\s*;", descent_block) is not None and \
        re.search(r"lowered_from\s*=\s*fallback_ubatch\s*;", descent_block) is not None and \
        re.search(r"descent_ran\s*=\s*true\s*;", descent_block) is not None and \
        re.search(r"publish_dirty\s*=\s*true\s*;", descent_block) is not None
    stop_race_at = select_fn.find("const bool stop_is_pure_race")
    results["a rung that lost is only lowered from when the default itself was tried (a non-rung default is not skipped)"] = \
        0 < stop_race_at < (descent_m.start() if descent_m else 0) and \
        re.search(r"if\s*\(\s*c\s*==\s*fallback_ubatch\s*\)\s*\{\s*fallback_tried\s*=\s*true\s*;\s*\}\s*tried\s*\+=", select_fn) is not None and \
        re.search(r"fallback_tried\s*=\s*fallback_tried\s*\|\|\s*cached_ubatch\s*==\s*fallback_ubatch\s*;", select_fn) is not None
    results["rung_fit_refused is cleared at the start of every rung and set by exactly the three real fit refusals"] = \
        re.search(r"^\{\s*rung_fit_refused\s*=\s*false\s*;", try_fn) is not None and \
        len(re.findall(r"rung_fit_refused\s*=\s*true\s*;", try_fn)) == 2 and \
        re.search(r"rung_fit_refused\s*=\s*dynamic_cast<const\s+llama_auto_ubatch_fit_refusal\s*\*>\(\s*&e\s*\)\s*!=\s*nullptr\s*;\s*return \"compute buffers did not fit\"\s*;", try_fn) is not None and \
        re.search(r"if\s*\(\s*!probe\.accepted\s*\)\s*\{\s*rung_fit_refused\s*=\s*true\s*;\s*return \"transaction refused\"\s*;", try_fn) is not None and \
        re.search(r"rung_fit_refused\s*=\s*true\s*;\s*return \"hold spill left no headroom\"\s*;", try_fn) is not None
    results["the trial state starts at nothing-happened and every rung asked is named in tried"] = \
        all(re.search(pat, select_fn) is not None for pat in (
            r"bool\s+descent_ran\s*=\s*false\s*;", r"bool\s+fallback_tried\s*=\s*false\s*;", r"uint32_t\s+lowered_from\s*=\s*0\s*;",
            r"bool\s+rung_fit_refused\s*=\s*false\s*;", r"uint32_t\s+lowest_refused\s*=\s*0\s*;",
            r"uint32_t\s+cache_refused_ub\s*=\s*0\s*;")) and \
        re.search(r"tried\s*\+=\s*\(tried\.empty\(\)\s*\?\s*\"\"\s*:\s*\",\"\)\s*\+\s*std::to_string\(cached_ubatch\)\s*;", select_fn) is not None and \
        re.search(r"tried\s*\+=\s*\(tried\.empty\(\)\s*\?\s*\"\"\s*:\s*\",\"\)\s*\+\s*std::to_string\(c\)\s*;", select_fn) is not None and \
        re.search(r"tried\.append\(tried\.empty\(\)\s*\?\s*\"\"\s*:\s*\",\"\)\.append\(std::to_string\(c\)\)\s*;", descent_block) is not None
    results["the trial never writes the context size: a smaller -ub is not a smaller context"] = \
        re.search(r"cparams\.n_ctx\s*(?:[-+*/]?=(?!=)|\+\+|--)", select_fn) is None
    results["a refused cached rung is known to this start and its entry is overwritten, not re-paid"] = \
        re.search(r"if\s*\(\s*rung_fit_refused\s*\)\s*\{\s*cache_refused_ub\s*=\s*cached_ubatch\s*;\s*cache_refused_reason\s*=\s*cache_reason\s*;", select_fn) is not None and \
        re.search(r"if\s*\(\s*cache_refused_ub\s*!=\s*0\s*&&\s*c\s*>=\s*cache_refused_ub\s*\)\s*\{\s*stop\s*=\s*cache_refused_reason\s*;\s*last_stop\s*=\s*cache_refused_reason\s*;\s*break\s*;\s*\}", select_fn) is not None and \
        re.search(r"const bool\s+store_outcome\s*=\s*lowered_from\s*==\s*0\s*\?\s*!descent_ran\s*:\s*cache_refused_ub\s*!=\s*0\s*;", select_fn) is not None
    results["a lowered result is announced once, after the settle, and is stored only to overwrite a refused cached rung"] = \
        re.search(r"lowered_cause\s*=\s*format\(\s*\"the default did not fit \(%s\)\"\s*,\s*stop\s*\)", descent_block) is not None and \
        re.search(r"if\s*\(\s*lowered_from\s*!=\s*0\s*\)\s*\{\s*LLAMA_LOG_WARN\(\s*\"\[SYCL-PLAN\] auto n_ubatch lowered from %u to %u: %s;", select_fn) is not None and \
        select_fn.count("auto n_ubatch lowered from %u to %u") == 1 and \
        re.search(r"if\s*\(\s*ladder_needed\s*&&\s*!stop_is_pure_race\s*&&\s*store_outcome\s*&&\s*!resumed_outcome_unchanged\s*&&", select_fn) is not None
    # kpjw-g7: the descent above only runs when NOTHING at or above the default won. The B50 run never got there: the
    # default 512 won the ladder, 1024 was refused at its probe, and the SETTLE then republished the winner and was
    # refused. The settle's refusal is handed to the one fit function: it accepts the rung -> the refusal was some
    # other reason and leaves as it came; it refuses -> a named error carrying the -ub that function accepts.
    settle_gate = "if (!sched_matches_last_good || cparams.n_ubatch != last_good) {"
    settle_at = select_fn.find(settle_gate)
    settle_blk = balanced_block(select_fn, settle_at + len(settle_gate) - 1) if settle_at >= 0 else ""
    refused_m = re.search(r"if\s*\(\s*settle_error\s*\)\s*\{", settle_blk)
    refused_blk = balanced_block(settle_blk, refused_m.end() - 1) if refused_m else ""
    store_at = select_fn.find("cache_store_fn(&cache_key")
    fits_at = refused_blk.find("llama_context_sycl_hold_spill_fits(backends, last_good, &largest_ub)")
    rethrow_at = refused_blk.find("std::rethrow_exception(settle_error);")
    advice_at = refused_blk.find("llama_auto_ubatch_advice(largest_ub, lowest_refused)")
    throw_at = refused_blk.find("throw std::runtime_error(")
    results["a refused settle publish is recorded and is a named error from the one fit function, with no second descent"] = \
        settle_at > 0 and refused_m is not None and \
        re.search(r"std::exception_ptr\s+settle_error\s*;", settle_blk) is not None and \
        re.search(r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\)\s*;\s*\}\s*catch\s*\(\s*const std::exception\s*&\s*e\s*\)\s*\{\s*"
                  r"settle_error\s*=\s*std::current_exception\(\)\s*;\s*settle_refusal\s*=\s*e\.what\(\)\s*;\s*\}", settle_blk) is not None and \
        -1 not in (fits_at, rethrow_at, advice_at, throw_at) and fits_at < rethrow_at < advice_at < throw_at and \
        "try_candidate(" not in refused_blk and "return" not in refused_blk and refused_blk.count("throw std::runtime_error(") == 1 and \
        re.search(r"\}\s*else\s*\{\s*sched_need_reserve\s*=\s*true\s*;\s*sched_reserve\(\)\s*;\s*\}", settle_blk) is not None
    results["the tuning-cache store runs after the settle, so a rung the settle refused is never persisted"] = \
        settle_at > 0 and store_at > settle_at
    results["the named settle error names the scope, the rungs tried, the LAST rung's stop reason and the fit function's -ub"] = \
        refused_m is not None and \
        "no -ub from %u down to %u fits this context" in refused_blk and "%u does not fit this context" in refused_blk and \
        "tried %s; last stop: %s" in refused_blk and "largest -ub that fits is about" in refused_blk and \
        re.search(r"last_stop\s*!=\s*nullptr\s*\?\s*last_stop\s*:\s*stop", refused_blk) is not None and \
        "llama_auto_ubatch_descent_floor" in refused_blk and \
        re.search(r"auto note_loss\s*=\s*\[&\]\(uint32_t c, const char \* reason\)\s*\{\s*last_stop\s*=\s*reason\s*;\s*lowest_refused\s*=\s*lowest_refused\s*==\s*0\s*\?\s*c\s*:\s*std::min\(lowest_refused, c\)\s*;\s*\}", select_fn) is not None and \
        select_fn.count("note_loss(") == 3
    # kpjw-g7 A: a buffer is a scheduler compute buffer by an explicit scope opened around the reserve and the graph
    # allocation (llama-context), resolved once per context, never by the absence of a model load.
    ctx_flat = re.sub(r"\s+", " ", context)
    results["the scheduler compute scope is opened around the reserve and the graph allocation, resolved once per context"] = \
        re.search(r"struct sycl_compute_scope_guard \{ void \(\*fn\)\(bool\); explicit sycl_compute_scope_guard\(void \(\*f\)\(bool\)\) : fn\(f\) \{ if \(fn\) \{ fn\(true\); \} \} ~sycl_compute_scope_guard\(\) \{ if \(fn\) \{ fn\(false\); \} \}", ctx_flat) is not None and \
        re.search(r"bool llama_context::sched_alloc_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); return ggml_backend_sched_alloc_graph\(sched\.get\(\), gf\); \}", ctx_flat) is not None and \
        re.search(r"bool llama_context::sched_reserve_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); return ggml_backend_sched_reserve\(sched\.get\(\), gf\); \}", ctx_flat) is not None and \
        "if (!sched_alloc_graph(gf)) {" in ctx_flat and "if (!reserved) {" in ctx_flat and \
        ctx_flat.count("ggml_backend_sched_alloc_graph(") == 1 and len(re.findall(r"ggml_backend_sched_reserve\(", ctx_flat)) == 2 and \
        "const bool reserved = state.measure ? ggml_backend_sched_reserve(state.sched.get(), gf) : sched_reserve_graph(gf);" in ctx_flat and \
        "if (!sycl_compute_scope_resolved) { sycl_compute_scope_resolved = true;" in ctx_flat and \
        "sycl_compute_scope_cached = &ggml_backend_sycl_compute_alloc_scope;" in ctx_flat and \
        '"ggml_backend_sycl_compute_alloc_scope"' in ctx_flat and \
        re.search(r"sycl_compute_scope_fn_t\s+sycl_compute_scope_fn\(\)\s*;", ctx_header) is not None and \
        re.search(r"bool\s+sycl_compute_scope_resolved\s*=\s*false\s*;", ctx_header) is not None and \
        re.search(r"bool\s+sched_alloc_graph\(\s*ggml_cgraph\s*\*\s*gf\s*\)\s*;", ctx_header) is not None and \
        re.search(r"bool\s+sched_reserve_graph\(\s*ggml_cgraph\s*\*\s*gf\s*\)\s*;", ctx_header) is not None
    # r5 R7/R8: releasing the previous rung's buffers means the cached graph results too, not only the sched.
    results["the release drops every cached graph result and the active pointer, not only the sched"] = \
        re.search(r"for\s*\(\s*auto\s*&\s*res\s*:\s*gf_res_prev\s*\)\s*\{\s*res\.reset\(\)\s*;\s*\}", rel_block) is not None and \
        re.search(r"gf_res_prev_active\s*=\s*nullptr\s*;", rel_block) is not None
    return results


def evaluate_raw(backend_raw, zone_hpp_raw, cmake_text, gpu_test_text, cache_raw="", context_raw=""):
    """Claims that live in comments, strings the stripper keeps out, and the build registration."""
    results = {}
    flat = re.sub(r"\s*\n\s*//\s*", " ", backend_raw)
    at = flat.find("static size_t ggml_sycl_planned_scratch_hold_spill_bound(")
    at = flat.find("struct ggml_sycl_hold_fit_query {")
    results["the spill bound's comment says the KV-room netting is an estimate"] = \
        at > 0 and "the KV-room netting is an estimate" in flat[max(0, at - 3200):at]
    zflat = re.sub(r"\s*\n\s*//\s*", " ", zone_hpp_raw)
    zat = zflat.find("zone_hold_spill_raw_demand(")
    results["the zone-sizing contract says the raw demand is an estimate"] = \
        zat > 0 and "ESTIMATE" in zflat[max(0, zat - 1800):zat]
    results["the zone-sizing header declares the two r4 helpers"] = \
        "zone_kv_room_for_compute(" in zone_hpp_raw and "zone_hold_fit_largest_ub(" in zone_hpp_raw
    results["the GPU test is registered under the mem-handle label"] = \
        re.search(r"llama_build\(test-sycl-compute-buffer-kv-zone\.cpp\)", cmake_text) is not None and \
        re.search(r"set_tests_properties\(test-sycl-compute-buffer-kv-zone PROPERTIES\s+LABELS\s+\"[^\"]*mem-handle[^\"]*\"", cmake_text) is not None
    results["the GPU test asserts the landing zone, the zone-used delta and the return to the baseline"] = \
        "handle.vram_zone == vram_zone_id::KV" in gpu_test_text and "zone_used(vram_zone_id::KV) >= kv_before + big" in gpu_test_text and \
        "zone_used(vram_zone_id::KV) == kv_before" in gpu_test_text and "kv_first=*/false" in gpu_test_text
    # r5 minor 10: a free the allocator refused would leave every zone number below comparing against a leak, and the
    # unflagged held-back control is the raw spill the flag exists to avoid: it must be counted and warned about.
    results["the GPU test checks every unified_free, and asserts the raw spill and its WARN on the unflagged control"] = \
        len(re.findall(r"check\(unified_free\(handle\)", gpu_test_text)) >= 4 and \
        not re.search(r"^\s*unified_free\(handle\);", gpu_test_text, re.M) and \
        "totals.raw_count == 0 && totals.raw_bytes == 0" in gpu_test_text and "g_warn_raw == 0" in gpu_test_text and \
        "unified_cache_get_hold_rung_requests(" in gpu_test_text and \
        "g_warn_arena == 1" in gpu_test_text and "g_warn_zone_full == 1" in gpu_test_text and \
        "ggml_log_set(count_warnings" in gpu_test_text
    # Comments the stripped sources cannot see. Each was a statement a reader acted on or a claim the code does not make.
    cflat = re.sub(r"\s*\n\s*//\s*", " ", cache_raw)
    results["the release comment says the teardown take ran BEFORE it (it does: log_planned_scratch_stats, then the release)"] = \
        "(after this release)" not in cflat and "BEFORE this release" in cflat
    bflat = re.sub(r"\s*\n\s*//\s*", " ", backend_raw)
    kv_at = bflat.find("const bool kv_zone_first")
    results["the m4 flag's comment says it is a process-global timing discriminator, not an identity one"] = \
        kv_at > 0 and "POSITIVELY" in bflat[max(0, kv_at - 2400):kv_at] and \
        "recurrent state" in bflat[max(0, kv_at - 2400):kv_at] and \
        "timing discriminator" in bflat[max(0, kv_at - 2400):kv_at]
    entry_at = bflat.find("bool ggml_backend_sycl_planned_hold_spill_fits(")
    results["the exported entry's comment states the arity change and what an old DSO does"] = \
        entry_at > 0 and "arity" in bflat[max(0, entry_at - 2200):entry_at] and "largest_ub stays 0" in bflat[max(0, entry_at - 2200):entry_at] and \
        "no version gate" in bflat[max(0, entry_at - 2200):entry_at]
    xflat = re.sub(r"\s*\n\s*//\s*", " ", context_raw)
    ctor_at = xflat.find("if (llama_context_has_sycl_backend(backends) && sycl_hold_spill_validated_ub")
    results["the context-init check's comment names the gap: a later lazy re-reserve is not covered"] = \
        ctor_at > 0 and "lazy" in xflat[max(0, ctor_at - 2400):ctor_at] and "not covered" in xflat[max(0, ctor_at - 2400):ctor_at] and \
        "sched_need_reserve" in xflat[max(0, ctor_at - 2400):ctor_at]
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
ctx_header_src = read(args.context_header)
auto_header_src = read(args.auto_header)
failed = run("tree", (backend, common, cache, zone))
failed += run_context("tree", (context_src, header_src))
raw_inputs = (Path(args.backend).read_text(), Path(args.zone).read_text(), Path(args.cmake).read_text(),
              Path(args.gpu_test).read_text() if Path(args.gpu_test).exists() else "", Path(args.cache).read_text(),
              Path(args.context).read_text())
for k, v in sorted(evaluate_raw(*raw_inputs).items()):
    print(("PASS: " if v else "FAIL: ") + k)
    if not v:
        failed.append(k)
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

    def mutate_re_in_func(src, sig_regex, pattern, repl):
        """Regex replace of the first `pattern` after the function's signature (formatting-independent)."""
        m = re.search(sig_regex, src)
        r = re.compile(pattern)
        mm = r.search(src, m.end()) if m else None
        if not mm:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {pattern!r}")
            failed.append("self-test anchor " + pattern)
            return src
        return src[:mm.start()] + repl + src[mm.end():]

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
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                             r"unified_cache_note_runtime_request\(\s*req\.device\s*,",
                                             "unified_cache_note_runtime_request(0,"), zone)),
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
         (mutate_in_func(backend, r"static bool ggml_sycl_mul_mat_scratch_route_decided\(",
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
                         "ggml_sycl_mul_mat_scratch_route_decided(", "ggml_sycl_XXXX("), common, cache, zone)),
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
         (backend, common, mutate_all_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                              "unified_cache_note_planned_hold_spill(", "unified_cache_XXXX(", 2), zone)),
        ("hold spill warned every time", "the warning is once per device per context (the take resets the count)",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_note_planned_hold_spill\(",
                                             r"first\s*= !state\.warned_raw;", "first = true;"), zone)),
        ("take does not reset", "the warning is once per device per context (the take resets the count)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_take_planned_hold_spills\(",
                                          "state.spill_count       = 0;", "state.spill_count       += 0;"), zone)),
        ("stats omit the hold spills", "teardown reports the hold spills with the scratch stats",
         (mutate_in_func(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)",
                         "hold_spills_raw=", "hold_XXXX="), common, cache, zone)),
        ("guard evicts weights for a hold spill", "the overcommit guard refuses a hold-induced spill instead of evicting weights",
         (backend, common, mutate_after(cache, "used_vram + alloc_size > total_vram) {",
                                        "if (hold_spill) {", "if (false) {"), zone)),
        ("held-back request never spills", "the held-back request still takes the ordinary spill path",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "if (!hold_spill) {", "if (true) {"), zone)),
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
        ("allocator does not record requests", "the allocator records the largest spill-capable RUNTIME request",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                          "unified_cache_note_runtime_request(", "unified_cache_XXXX("), zone)),
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
         (backend, common, mutate_after(cache, "used_vram + alloc_size > total_vram) {",
                                        "if (hold_spill) {", "if (true) {"), zone)),
        ("realized check moved back into the recheck",
         "the recheck no longer claims the realized spill (it runs before the rung's worst-case reserves)",
         (mutate_in_func(backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\(",
                         "const auto current = ggml_sycl_global_plan_snapshot();",
                         "ggml_sycl_check_hold_spill_realized(0, 0, false); const auto current = ggml_sycl_global_plan_snapshot();"),
          common, cache, zone)),
        ("realized check reads lifetime spills", "the realized check asks the pure rule, the live free memory and the spills since this publish",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_realized\(",
                         "unified_cache_get_recent_planned_hold_spills(", "unified_cache_take_planned_hold_spills("),
          common, cache, zone)),
        ("publish keeps the old epoch", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "unified_cache_begin_planned_hold_epoch(", "unified_cache_XXXX("), common, cache, zone)),
        ("epoch keeps the spilled bytes", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(",
                                             r"state\.spill_bytes\s*= 0;", "state.spill_bytes += 0;"), zone)),
        ("graph-entry check keeps its own 256", "the driver headroom the realized check uses is the graph-entry check's constant (one source)",
         (mutate_in_func(backend, r"static void ggml_sycl_check_graph_scratch_headroom\(",
                         "= kSyclArenaMinExternalHeadroomBytes;", "= 256ull * 1024ull * 1024ull;"), common, cache, zone)),
        # r3 design change: the KV zone before raw memory
        ("compute buffer goes raw before the KV zone", "a compute buffer the RUNTIME zone will not serve tries the KV zone before the overcommit guard and raw memory",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "kv_cache->zone_alloc(vram_zone_id::KV,",
                                          "kv_cache->zone_alloc(vram_zone_id::SCRATCH,"), zone)),
        ("KV placement not asked for", "a compute buffer the RUNTIME zone will not serve tries the KV zone before the overcommit guard and raw memory",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "zone_runtime_spill_prefers_kv_zone(",
                                          "zone_runtime_XXXX("), zone)),
        ("KV placement ignores the KV room", "a compute buffer the RUNTIME zone will not serve tries the KV zone before the overcommit guard and raw memory",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "kv_cache->zone_available(vram_zone_id::KV), alloc_size)",
                                          "alloc_size, alloc_size)"), zone)),
        ("guard runs for a KV placement", "a placement in the KV zone is skipped by the guard and the zone routing, and counts as an in-arena hold spill",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "if (!kv_placed && req.device >= 0", "if (req.device >= 0"), zone)),
        ("zone routing runs for a KV placement", "a placement in the KV zone is skipped by the guard and the zone routing, and counts as an in-arena hold spill",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "if (!kv_placed && req.intent.constraints.prefer_vram_zone",
                                          "if (req.intent.constraints.prefer_vram_zone"), zone)),
        ("KV placement counted as raw", "a placement in the KV zone is skipped by the guard and the zone routing, and counts as an in-arena hold spill",
         (backend, common, mutate_after(cache, "bool kv_placed = false;", "true);", "false);"), zone)),
        ("buffer allocator does not ask for the KV zone", "the runtime buffer allocator asks for the KV-zone-first placement",
         (mutate(backend, "spill_to_kv_zone_before_raw = kv_zone_first", "spill_to_kv_zone_before_raw = false"), common, cache, zone)),
        ("stats do not split the landing site", "the hold-spill counters are split by where the buffer landed (raw outside the arena, KV zone)",
         (mutate_in_func(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)",
                         "hold_spills_kv_zone=", "hold_XXXX="), common, cache, zone)),
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
        ("alloc takes the hold twice", "unified_alloc takes the hold state once per request",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                             r"hold_at_decision\s*=\s*unified_cache_note_runtime_request\(",
                                             "hold_at_decision = unified_cache_get_planned_scratch_hold(req.device) + "
                                             "unified_cache_note_runtime_request("), zone)),
        ("new owner inherits spill counts", "a new owner starts with its own spill counts",
         (backend, common, mutate_in_func(cache, r"void unified_cache_set_planned_scratch_hold\(", "spill_count", "spill_XXXX"),
          zone)),
        ("f16 dispatch arm outside the macro", "both f16 buffer acquisitions in the dispatch arm sit under the one macro",
         (mutate(backend, "#if GGML_SYCL_DEQUANT_F16_ARM", "#if 1", 1), common, cache, zone)),        # review r4
        ("KV placement published as RUNTIME", "the KV pre-placement publishes through the KV zone (record, allocation, metadata)",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "output_metadata.vram_zone    = vram_zone_id::KV;",
                                          "output_metadata.vram_zone    = vram_zone_id::RUNTIME;"), zone)),
        ("publication prepared for RUNTIME", "the KV pre-placement publishes through the KV zone (record, allocation, metadata)",
         (backend, common, mutate_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", "prepare_arena_publication(vram_zone_id::KV)",
                                          "prepare_arena_publication(vram_zone_id::RUNTIME)"), zone)),
        ("failed publication retried raw", "a publication that was attempted and failed is not retried raw",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", r"if \(!ptr && arena_publication\.attempted\) \{\s*return false;\s*\}(?=\s*if \(ptr\) \{\s*kv_placed)", ""), zone)),
        ("KV placement not from_arena", "the KV pre-placement publishes through the KV zone (record, allocation, metadata)",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", r"(?<=kv_placed)(\s*)= true;\s*from_arena\s*= true;", " = true;"), zone)),
        ("KV placement counted for every request", "a hold-induced KV placement counts as an in-arena hold spill, a zone-full one is counted and warned separately",
         (backend, common, mutate_after(cache, "bool kv_placed = false;", "if (hold_spill) {", "if (true) {"), zone)),
        ("zone-full arm dropped", "the KV pre-placement reads the zone-full arm as well as the hold",
         (backend, common, mutate_after(cache, "bool kv_placed = false;",
                                        "hold_spill || kv_cache->zone_available(vram_zone_id::RUNTIME) < alloc_size", "hold_spill"), zone)),
        ("KV-zone counter folded into raw", "the landing-site counters stay separate (the KV-zone counter is not the raw one)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_note_planned_hold_spill\(", "spill_arena_count++;", "spill_count++;"), zone)),
        ("take leaves the arena counters", "a take hands every counter to the owner and clears every one",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_take_planned_hold_spills\(", r"state\.spill_arena_count\s*= 0;", ""), zone)),
        ("take keeps the WARN latch", "a take hands every counter to the owner and clears every one",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_take_planned_hold_spills\(", r"state\.warned_raw\s*= false;", ""), zone)),
        ("zone-full note shares the hold counters", "the zone-full KV placement has its own counter and a once-only WARN, separate from the hold spills",
         (backend, common, mutate_in_func(cache, r"void unified_cache_note_zone_full_kv_placement\(", "zone_full_count++", "spill_count++"), zone)),
        ("epoch keeps the KV-zone counters", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(", r"state\.spill_arena_count\s*= 0;", ""), zone)),
        ("epoch resets the WARN latches", "a publish restarts the counters the teardown stats report, and not the WARN latches",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(", r"state\.zone_full_count\s*= 0;",
                                             "state.zone_full_count = 0; state.warned_raw = false;"), zone)),
        ("stats gate fires on raw only", "the stats line is printed whatever landed, not only for raw spills",
         (mutate_re_in_func(backend, r"void ggml_backend_sycl_context::log_planned_scratch_stats\(\)",
                            r"hold_spills\.raw_count != 0 \|\| hold_spills\.arena_count != 0 \|\| hold_spills\.zone_full_count != 0",
                            "hold_spills.raw_count != 0"), common, cache, zone)),
        ("caller SCRATCH step reachable", "the caller's KV and SCRATCH fallbacks are unreachable for the flagged class",
         (mutate(backend, "if (!kv_zone_first && (alloc_role ==", "if ((alloc_role =="), common, cache, zone)),
        ("KV pre-placement ignores forbid-spill", "the KV pre-placement is told the request's forbid-spill flag",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)", r"true, true, req\.intent\.constraints\.forbid_vram_zone_spill,", "true, true, false,"), zone)),
        ("a second request carries the flag", "the KV-zone-first flag is set by exactly one request, the scheduler compute buffer's",
         (mutate(backend, "kv_req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::KV;",
                 "kv_req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::KV;\n"
                 "                    kv_req.intent.constraints.spill_to_kv_zone_before_raw = true;"), common, cache, zone)),
        ("landing line dropped", "each flagged compute buffer's landing zone is printed by name and size",
         (mutate(backend, "ggml_sycl_log_compute_buffer_landing(buft_ctx->device,", "ggml_sycl_XXXX(buft_ctx->device,"), common, cache, zone)),
        ("landing line without the zone", "each flagged compute buffer's landing zone is printed by name and size",
         (mutate_in_func(backend, r"static void ggml_sycl_log_compute_buffer_landing\(", "zone=%s", "zone=?"), common, cache, zone)),
        ("F3 text not net of KV", "the F3 refusal says its figure is net of the KV-zone room",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_hold_spill_headroom\(", "net of the KV", "net of the XX"), common, cache, zone)),
        # review r5: each of these survived every gate and every unit test.
        ("kv_admitted forced to 0 (T1)", "the transaction nets out the KV already admitted, so a rung above the first is not over-refused",
         (mutate_re_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                            r"kv_admitted\s*=\s*admitted_kv\s*\?[^;]*;", "kv_admitted = 0;"), common, cache, zone)),
        ("admitted_kv always the live plan (T1b)", "the transaction nets out the KV already admitted, so a rung above the first is not over-refused",
         (mutate_re_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                            r"admitted_kv\s*=\s*ctx->runtime_kv_admitted\s*\?\s*current->plan\.get\(\)\s*:\s*nullptr",
                            "admitted_kv = current->plan.get()"), common, cache, zone)),
        ("entry drops the caller's n_ubatch (T7)", "the exported entry zeroes the out-parameter first and passes the caller's n_ubatch to the realized check",
         (mutate_re_in_func(backend, r"bool ggml_backend_sycl_planned_hold_spill_fits\(",
                            r"planned_scratch_owner\s*,\s*true\s*,\s*n_ubatch", "planned_scratch_owner, true, 0"), common, cache, zone)),
        ("entry leaves the out-parameter unset (T8)", "the exported entry zeroes the out-parameter first and passes the caller's n_ubatch to the realized check",
         (mutate_re_in_func(backend, r"bool ggml_backend_sycl_planned_hold_spill_fits\(",
                            r"if\s*\(\s*largest_ub\s*\)\s*\{\s*\*largest_ub\s*=\s*0\s*;\s*\}", ""), common, cache, zone)),
        ("legacy placement never pending", "a flagged buffer the legacy path places gets its final landing line, from the handle that path made",
         (mutate_re_in_func(backend, r"static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer\(",
                            r"legacy_landing_pending\s*=\s*true\s*;", "(void) 0;"), common, cache, zone)),
    ]
    # kpjw-g7 unification: each mutant gives one consumer its own formula, free-memory read or history, or removes one
    # property of the one function, and must die.
    F1 = "F3, the realized check and the demand bound all ask the one fit function"
    F2 = "none of the three keeps a formula, a free-memory read or a high-water mark of its own"
    F3S = r"static bool ggml_sycl_check_hold_spill_headroom\("
    F4S = r"static bool ggml_sycl_check_hold_spill_realized\("
    F5S = r"static size_t ggml_sycl_planned_scratch_hold_spill_bound\("
    F6S = r"static bool ggml_sycl_hold_spill_fit\("
    mutants += [
        ("F3 asks its own predicate instead of the fit", F1,
         (mutate_in_func(backend, F3S, "ggml_sycl_hold_spill_fit(q, &a)", "zone_hold_spill_bound_fits(0, 0, 0)"), common, cache, zone)),
        ("realized check asks its own predicate instead of the fit", F1,
         (mutate_in_func(backend, F4S, "ggml_sycl_hold_spill_fit(q, &a)", "zone_hold_spill_bound_fits(0, 0, 0)"), common, cache, zone)),
        ("demand bound asks its own predicate instead of the fit", F1,
         (mutate_in_func(backend, F5S, "ggml_sycl_hold_spill_fit(q, &a)", "zone_hold_spill_bound_fits(0, 0, 0)"), common, cache, zone)),
        ("F3 carries its own formula beside the fit", F2,
         (mutate_in_func(backend, F3S, "ggml_sycl_hold_spill_fit(q, &a)",
                         "zone_hold_spill_bound_fits(0, 0, 0) || ggml_sycl_hold_spill_fit(q, &a)"), common, cache, zone)),
        ("realized check carries its own formula beside the fit", F2,
         (mutate_in_func(backend, F4S, "ggml_sycl_hold_spill_fit(q, &a)",
                         "zone_hold_spill_realized_fits(0, 0, 0) || ggml_sycl_hold_spill_fit(q, &a)"), common, cache, zone)),
        ("realized check reads the driver after the release", F2,
         (mutate_in_func(backend, F4S, "ggml_sycl_hold_spill_fit(q, &a)",
                         "(ggml_backend_sycl_get_device_memory(device, nullptr, nullptr), true) && ggml_sycl_hold_spill_fit(q, &a)"), common, cache, zone)),
        ("demand bound keeps its own request mark", F2,
         (mutate_in_func(backend, F5S, "ggml_sycl_hold_spill_fit(q, &a)",
                         "(ggml_sycl::unified_cache_get_runtime_request_hwm(0, nullptr, nullptr), true) && ggml_sycl_hold_spill_fit(q, &a)"), common, cache, zone)),
        ("fit reads no ledger", "the fit function composes the zone predicate over the plan, the rung's record, the KV room and the ledger",
         (mutate_in_func(backend, F6S, "unified_cache_hold_free_before(", "unified_cache_XXXX("), common, cache, zone)),
        ("fit ignores the rung's record", "the fit function composes the zone predicate over the plan, the rung's record, the KV room and the ledger",
         (mutate_in_func(backend, F6S, "unified_cache_get_hold_rung_requests(", "unified_cache_XXXX("), common, cache, zone)),
        ("fit has its own KV room", "the fit function composes the zone predicate over the plan, the rung's record, the KV room and the ledger",
         (mutate_in_func(backend, F6S, "ggml_sycl_hold_kv_room(", "ggml_sycl_XXXX("), common, cache, zone)),
        ("fit takes no plan", "the fit function composes the zone predicate over the plan, the rung's record, the KV room and the ledger",
         (mutate_re_in_func(backend, F6S, r"in\.plan_of\s*=\s*ggml_sycl_hold_plan_at;", "in.plan_of = ggml_sycl_XXXX;"), common, cache, zone)),
        ("fit names no -ub", "the -ub a refusal names is one the F3 publish accepts: the smaller of the spill's share and F3's own answer",
         (mutate_in_func(backend, F6S, "a.largest_ub = ggml_sycl::zone_hold_fit_largest_ub(in, q.n_ubatch);", "a.largest_ub = q.n_ubatch / 2;"), common, cache, zone)),
        ("ledger re-reads the driver", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_hold_free_before\(", "zone_hold_free_before(",
                                          "ggml_backend_sycl_get_device_memory(0, 0, 0); zone_hold_free_before("), zone)),
        ("ledger ignores the live outside-arena bytes", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_hold_free_before\(", "unified_cache_raw_device_held_bytes(", "unified_cache_XXXX("), zone)),
        ("ledger re-baselines on every call", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_re_in_func(cache, r"size_t unified_cache_hold_free_before\(", r"state\.baseline_owner\s*==\s*owner", "false"), zone)),
        ("ledger credits every raw byte as the rung's own", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_re_in_func(cache, r"size_t unified_cache_hold_free_before\(", r"zone_hold_persistent_raw\(\s*raw_held\s*,\s*compute_live\s*,\s*rung_live\s*\)",
                                             "zone_hold_persistent_raw(raw_held, compute_live, false)"), zone)),
        ("ledger takes the maximum of nothing", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_hold_free_before\(", "zone_hold_cold_update(", "zone_XXXX("), zone)),
        ("ledger has no cold baseline", "the free memory is the cache's ledger: a cold reading plus live outside-arena bytes, never a re-read",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_hold_free_before\(", "zone_hold_free_cold(", "zone_XXXX("), zone)),
        ("request mark is a running maximum again", "the request mark is per rung and survives a publish (no history-carrying high-water mark)",
         (backend, common, mutate(cache, "std::vector<zone_hold_rung_request> rung_requests;",
                                  "std::vector<zone_hold_rung_request> rung_requests;\n    size_t request_hwm = 0;"), zone)),
        ("a publish forgets the per-rung records", "the request mark is per rung and survives a publish (no history-carrying high-water mark)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(", "state.epoch_n_ubatch       = n_ubatch;",
                                          "state.rung_requests.clear(); state.epoch_n_ubatch       = n_ubatch;"), zone)),
        ("a publish keeps no KV-room snapshot", "the request mark is per rung and survives a publish (no history-carrying high-water mark)",
         (backend, common, mutate_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(", "state.epoch_kv_room", "state.epoch_XXXX"), zone)),
        ("a publish keeps the previous window's baseline", "a publish starts a new spill epoch (a losing rung's spills and mark do not carry to the next rung)",
         (backend, common, mutate_re_in_func(cache, r"void unified_cache_begin_planned_hold_epoch\(", r"state\.baseline_owner\s*=\s*0\s*;",
                                             "(void) 0;"), zone)),
        ("zone header keeps the second -ub walk", "the zone header declares the one fit and the ledger arithmetic, and no second predicate walk",
         (backend, common, cache, zone + "\nuint32_t zone_hold_spill_largest_ub(uint32_t n_ubatch, size_t spill_bytes, size_t free_after, size_t headroom_target);")),
        ("zone header lacks the ledger arithmetic", "the zone header declares the one fit and the ledger arithmetic, and no second predicate walk",
         (backend, common, cache, zone.replace("zone_hold_free_before(", "zone_hold_XXXX("))),
        ("flag is the model-load timing again", "a compute buffer is flagged by an explicit scheduler scope, not by the absence of a model load",
         (mutate_re_in_func(backend, r"static bool ggml_sycl_compute_alloc_scope_active\(\)\s*\{", r"return g_sycl_compute_alloc_scope_depth > 0;",
                            "return !g_sycl_in_model_load.load(std::memory_order_acquire);") if False else
          re.sub(r"(const bool\s+kv_zone_first\s*=\s*)ggml_sycl_compute_alloc_scope_active\(\)", r"\1!g_sycl_in_model_load.load(std::memory_order_acquire)", backend, count=1),
          common, cache, zone)),
        ("scope export not registered", "a compute buffer is flagged by an explicit scheduler scope, not by the absence of a model load",
         (backend.replace('"ggml_backend_sycl_compute_alloc_scope"', '"ggml_backend_sycl_XXXX"', 1), common, cache, zone)),
        ("request mark fed by an unflagged request", "only a flagged compute request feeds the request mark and the hold-spill counters",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                             r"req\.intent\.constraints\.spill_to_kv_zone_before_raw\s*&&\s*!req\.intent\.constraints\.forbid_vram_zone_spill",
                                             "!req.intent.constraints.forbid_vram_zone_spill"), zone)),
        ("raw spill counted for an unflagged request", "only a flagged compute request feeds the request mark and the hold-spill counters",
         (backend, common, mutate_re_in_func(cache, r"bool unified_alloc\(const alloc_request & req_in, alloc_handle \* out\)",
                                             r"\}\s*else if\s*\(\s*req\.intent\.constraints\.spill_to_kv_zone_before_raw\s*\)\s*\{", "} else {"), zone)),
        # rewritten claims
        ("transaction passes no hold query", "the outside-arena headroom check is told the worst-case hold spill",
         (mutate_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                         "probe_mode, &hold_query)", "probe_mode, nullptr)"), common, cache, zone)),
        ("the non-FA check ignores the query", "the outside-arena headroom check is told the worst-case hold spill",
         (mutate_re_in_func(mutate_re_in_func(backend, r"static bool ggml_sycl_check_nonfa_attn_scratch\(",
                                              r"return\s*!hold_query\s*\|\|\s*ggml_sycl_check_hold_spill_headroom\(\*hold_query,\s*probe_mode\);", "return true;"),
                            r"static bool ggml_sycl_check_nonfa_attn_scratch\(",
                            r"hold_query\s*\?\s*ggml_sycl_planned_scratch_hold_spill_bound\(\*hold_query\)\s*:\s*0", "0"), common, cache, zone)),
        ("transaction asks for another owner", "the transaction and the recheck ask the one bound",
         (mutate_re_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                            r"hold_query\s*=\s*\{\s*ctx->device\s*,\s*ctx->planned_scratch_owner", "hold_query = { ctx->device, 0"), common, cache, zone)),
        ("recheck asks no query", "the transaction and the recheck ask the one bound",
         (mutate_re_in_func(backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\(", r"&recheck_hold_query", "nullptr"), common, cache, zone)),
        ("FA-on early return skips the check", "the FA-on early return runs the headroom check instead of skipping it",
         (mutate_in_func(backend, r"static bool ggml_sycl_check_nonfa_attn_scratch\(", "return !hold_query || ggml_sycl_check_hold_spill_headroom(*hold_query, probe_mode);",
                         "return true;"), common, cache, zone)),
        ("FA-on check never asks the fit", "the FA-on check compares the spill with the device's live free memory",
         (mutate_in_func(backend, F3S, "ggml_sycl_hold_spill_fit(q, &a)", "true"), common, cache, zone)),
        ("realized check asks as a transaction", "the realized check asks the pure rule, the live free memory and the spills since this publish",
         (mutate_re_in_func(backend, F4S, r"\{\s*device\s*,\s*owner\s*,\s*n_ubatch\s*,\s*0\s*,\s*true\s*\}", "{ device, owner, n_ubatch, 0, false }"), common, cache, zone)),
        ("recorded n_ubatch is not the epoch's", "the largest request is recorded with the n_ubatch it was seen at, from the first publish on",
         (backend, common, mutate_in_func(cache, r"size_t unified_cache_note_runtime_request\(", "r.n_ubatch = state.epoch_n_ubatch;", "r.n_ubatch = 0;"), zone)),
        ("KV room sums the zone", "the transaction-time bound is the part of the worst-case spill the KV zone cannot take",
         (mutate_in_func(backend, r"static size_t ggml_sycl_hold_kv_room\(", "zone_largest_free(", "zone_available("), common, cache, zone)),
        ("KV room ignores the pending KV", "the transaction-time bound is the part of the worst-case spill the KV zone cannot take",
         (mutate_in_func(backend, r"static size_t ggml_sycl_hold_kv_room\(", "zone_kv_room_for_compute(", "zone_XXXX("), common, cache, zone)),
        ("KV room read without the arena", "the transaction-time bound is the part of the worst-case spill the KV zone cannot take",
         (mutate_in_func(backend, r"static size_t ggml_sycl_hold_kv_room\(", "cache->arena_active()", "true"), common, cache, zone)),
        ("the epoch keeps no KV room", "the transaction-time bound is the part of the worst-case spill the KV zone cannot take",
         (mutate_re_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                            r"unified_cache_begin_planned_hold_epoch\(([^;]*?)hold_kv_room\s*\)", r"unified_cache_begin_planned_hold_epoch(\1 0)"), common, cache, zone)),
        ("transaction passes no pending KV", "the transaction passes the KV bytes its plan adds; the recheck, which runs with KV live, passes none",
         (mutate_re_in_func(backend, r"static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction\(",
                            r"next_kv_info\.n_ubatch\s*,\s*kv_pending\s*,\s*false", "next_kv_info.n_ubatch, 0, false"), common, cache, zone)),
        ("recheck passes pending KV", "the transaction passes the KV bytes its plan adds; the recheck, which runs with KV live, passes none",
         (mutate_re_in_func(backend, r"ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn\(",
                            r"current->plan->planner_n_ubatch\s*,\s*0\s*,\s*true", "current->plan->planner_n_ubatch, 1, true"), common, cache, zone)),
        ("realized check names no -ub", "the realized check names the largest -ub that fits, through the exported entry",
         (mutate_in_func(backend, F4S, "*largest_ub = a.largest_ub;", "(void) 0;"), common, cache, zone)),
        ("F3 refusal omits the advice", "the -ub a refusal names is one the F3 publish accepts: the smaller of the spill's share and F3's own answer",
         (mutate_in_func(backend, F3S, "a.largest_ub", "0"), common, cache, zone)),
        ("F3 prints the ledger twice", "the F3 refusal prints the free memory it would leave and the free memory now, each once",
         (mutate_re_in_func(backend, F3S, r"free_after\s*/\s*mb,\s*a\.free_before\s*/\s*mb", "a.free_before / mb, a.free_before / mb"), common, cache, zone)),
        ("landing logged before the publish", "each flagged compute buffer's landing zone is printed by name and size",
         (mutate_re_in_func(backend, r"static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer\(",
                            r"if\s*\(\s*kv_zone_first\s*\)\s*\{\s*ggml_sycl_log_compute_buffer_landing\(\s*buft_ctx->device,\s*buft_ctx->name,\s*size,\s*landed_zone\s*\);\s*\}\s*return published;",
                            "return published;"), common, cache, zone)),
        ("landing zone read after the handle moved", "each flagged compute buffer's landing zone is printed by name and size",
         (mutate_re_in_func(backend, r"static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer\(",
                            r"const char \*\s*landed_zone\s*=\s*ggml_sycl_compute_buffer_zone\(runtime_h\);", "const char * landed_zone = \"none\";"), common, cache, zone)),
        ("landing: KV named runtime", "the landing line names each zone as itself, a buffer outside the arena as raw, host memory as host-pinned",
         (mutate_in_func(backend, r"static const char \* ggml_sycl_compute_buffer_zone\(", 'zone = "kv";', 'zone = "runtime";'), common, cache, zone)),
        ("landing: outside the arena named kv", "the landing line names each zone as itself, a buffer outside the arena as raw, host memory as host-pinned",
         (mutate_in_func(backend, r"static const char \* ggml_sycl_compute_buffer_zone\(", 'zone = "raw";', 'zone = "kv";'), common, cache, zone)),
        ("landing: host memory named raw", "the landing line names each zone as itself, a buffer outside the arena as raw, host memory as host-pinned",
         (mutate_in_func(backend, r"static const char \* ggml_sycl_compute_buffer_zone\(", 'zone = "host-pinned";', 'zone = "raw";'), common, cache, zone)),
        ("landing: nothing placed named raw", "the landing line names each zone as itself, a buffer outside the arena as raw, host memory as host-pinned",
         (mutate_in_func(backend, r"static const char \* ggml_sycl_compute_buffer_zone\(", 'const char * zone = "none";', 'const char * zone = "raw";'), common, cache, zone)),
        ("legacy placement logged before the publish", "a flagged buffer the legacy path places gets its final landing line, from the handle that path made",
         (mutate(backend, "legacy_published && legacy_landing_pending", "legacy_landing_pending"), common, cache, zone)),
        ("legacy placement never logged", "a flagged buffer the legacy path places gets its final landing line, from the handle that path made",
         (mutate(backend, "ggml_sycl_log_compute_buffer_landing(buft_ctx->device, buft_ctx->name, size, legacy_zone)", "(void) legacy_zone"), common, cache, zone)),
        ("a refused publish leaves no legacy line pending", "a flagged buffer the legacy path places gets its final landing line, from the handle that path made",
         (mutate_re_in_func(backend, r"static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer\(",
                            r"\}\s*if\s*\(\s*kv_zone_first\s*\)\s*\{\s*legacy_landing_pending\s*=\s*true\s*;\s*\}\s*\}", "}\n                }"), common, cache, zone)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)
    ctx_mutants = [
        ("check before the reserve", "the hold-spill check runs in try_candidate, after sched_reserve() and before the rung is accepted",
         (mutate_after(context_src, "auto try_candidate = [&](uint32_t c) -> const char * {", "sched_reserve();",
                       "(void) hold_spill_fn(nullptr); sched_reserve();"), header_src)),
        ("hook dropped", "the hold-spill check runs in try_candidate, after sched_reserve() and before the rung is accepted",
         (mutate_after(context_src, "auto try_candidate = [&](uint32_t c) -> const char * {", "hold_spill_fn(", "hold_XXXX("), header_src)),
        ("refusal keeps the rung", "a rung whose hold spill pushed the card under its headroom loses with its own stop reason",
         (context_src.replace("return \"hold spill left no headroom\";", "(void) 0;", 1), header_src)),
        ("no DL resolution", "the hook is resolved for a direct build and for a backend-DL build",
         (context_src.replace("\"ggml_backend_sycl_planned_hold_spill_fits\"", "\"ggml_backend_sycl_XXXX\"", 1), header_src)),
        ("header lacks the entry", "the header declares the exported entry",
         (context_src, header_src.replace("ggml_backend_sycl_planned_hold_spill_fits", "ggml_backend_sycl_XXXX"))),
        ("rung buffers kept across rungs", "the previous rung's compute buffers are released before the next rung's transaction",
         (mutate_after(context_src, "auto release_rung_buffers = [&]() {", "sched.reset();", "(void) 0;"), header_src)),
        ("rung buffers not released by try_candidate", "the previous rung's compute buffers are released before the next rung's transaction",
         (re.sub(r"(auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{\s*rung_fit_refused\s*=\s*false\s*;\s*)release_rung_buffers\(\)\s*;", r"\1", context_src, count=1), header_src)),
        ("pinned path unchecked", "the realized hold-spill check also runs after a reserve the ladder did not make",
         (context_src.replace("llama_context_sycl_hold_spill_fits(backends", "llama_context_sycl_XXXX(backends", 1), header_src)),
        ("pinned refusal unnamed", "a refusal there is a context-init refusal naming the largest -ub that fits",
         (context_src.replace("largest -ub that fits is about", "largest -ub XXXX", 1), header_src)),
        ("hook asked without the rung", "the trial asks the entry with the rung's n_ubatch",
         (re.sub(r"hold_spill_fn\(\s*sb\.backend\s*,\s*c\s*,", "hold_spill_fn(sb.backend, 0,", context_src, count=1), header_src)),
        # review r5 I-A / R7 / R8 / minor 4: every one survived all gates.
        ("ctor passes n_ubatch 0 (R1)", "the context-init check asks with cparams.n_ubatch and its refusal is a throw inside the refusing branch",
         (re.sub(r"llama_context_sycl_hold_spill_fits\(\s*backends\s*,\s*cparams\.n_ubatch\s*,\s*&largest_ub\s*\)",
                 "llama_context_sycl_hold_spill_fits(backends, 0, &largest_ub)", context_src, count=1), header_src)),
        ("merge is last-wins (R2)", "the helper keeps the smallest -ub the refusing backends name (the first refuser's, then the minimum)",
         (re.sub(r"std::min\(\s*\*largest_ub\s*,\s*backend_largest\s*\)", "backend_largest", context_src, count=1), header_src)),
        ("merge is max (R2b)", "the helper keeps the smallest -ub the refusing backends name (the first refuser's, then the minimum)",
         (re.sub(r"std::min\(\s*\*largest_ub\s*,\s*backend_largest\s*\)", "std::max(*largest_ub, backend_largest)", context_src, count=1), header_src)),
        ("helper never refuses (R3)", "the helper refuses when ANY backend's check refuses, and passes the rung's own n_ubatch down",
         (mutate_re_in_func(context_src, r"static bool llama_context_sycl_hold_spill_fits\(", r"\bfits\s*=\s*false\s*;", "(void) 0;"), header_src)),
        ("helper hands the backend 0 (R3b)", "the helper refuses when ANY backend's check refuses, and passes the rung's own n_ubatch down",
         (mutate_re_in_func(context_src, r"static bool llama_context_sycl_hold_spill_fits\(", r"hold_spill_fn\(\s*backend\.get\(\)\s*,\s*n_ubatch",
                            "hold_spill_fn(backend.get(), 0"), header_src)),
        ("helper does not zero the out-parameter", "the helper skips a backend that is not SYCL and zeroes the out-parameter first",
         (mutate_re_in_func(context_src, r"static bool llama_context_sycl_hold_spill_fits\(", r"\*largest_ub\s*=\s*0\s*;", "(void) 0;"), header_src)),
        ("ctor throw becomes a log (R4)", "the context-init check asks with cparams.n_ubatch and its refusal is a throw inside the refusing branch",
         (re.sub(r"throw std::runtime_error\(format\(\s*\"compute buffers held out", "LLAMA_LOG_ERROR(\"%s\", format(\"compute buffers held out",
                 context_src, count=1), header_src)),
        ("ctor check only for the pinned path (R6)", "the context-init check runs for any SYCL backend unless the trial already validated this -ub",
         (re.sub(r"llama_context_has_sycl_backend\(backends\)\s*&&\s*sycl_hold_spill_validated_ub\s*!=\s*cparams\.n_ubatch",
                 "llama_context_has_sycl_backend(backends) && !sycl_auto_ubatch_trial", context_src, count=1), header_src)),
        ("ctor check never skipped (minor 4)", "the context-init check runs for any SYCL backend unless the trial already validated this -ub",
         (re.sub(r"llama_context_has_sycl_backend\(backends\)\s*&&\s*sycl_hold_spill_validated_ub\s*!=\s*cparams\.n_ubatch",
                 "llama_context_has_sycl_backend(backends)", context_src, count=1), header_src)),
        ("ctor check skips on any recorded -ub", "the context-init check runs for any SYCL backend unless the trial already validated this -ub",
         (re.sub(r"sycl_hold_spill_validated_ub\s*!=\s*cparams\.n_ubatch", "sycl_hold_spill_validated_ub == 0", context_src, count=1), header_src)),
        ("trial never records the validated -ub", "the context records the -ub its trial validated, for the sched that is final, and nothing else",
         (mutate_re_in_func(context_src, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{",
                            r"hold_spill_validated_ub\s*=\s*hold_spill_fn\s*\?\s*c\s*:\s*0\s*;", "hold_spill_validated_ub = 0;"), header_src)),
        ("trial keeps a stale validated -ub across rungs", "the context records the -ub its trial validated, for the sched that is final, and nothing else",
         (mutate_re_in_func(context_src, r"auto release_rung_buffers = \[&\]\(\) \{",
                            r"sched_matches_last_good\s*=\s*false\s*;\s*hold_spill_validated_ub\s*=\s*0\s*;", "sched_matches_last_good = false;"), header_src)),
        ("settle publishes the validated -ub unconditionally", "the context records the -ub its trial validated, for the sched that is final, and nothing else",
         (re.sub(r"\(\s*sched_matches_last_good\s*&&\s*cparams\.n_ubatch\s*==\s*last_good\s*\)\s*\?\s*hold_spill_validated_ub\s*:\s*0",
                 "hold_spill_validated_ub", context_src, count=1), header_src)),
        ("validated -ub not reset at the start", "the context records the -ub its trial validated, for the sched that is final, and nothing else",
         (mutate_re_in_func(context_src, r"void llama_context::sycl_select_auto_ubatch\(", r"sycl_hold_spill_validated_ub\s*=\s*0\s*;", "(void) 0;"), header_src)),
        ("header lacks the validated member", "the context records the -ub its trial validated, for the sched that is final, and nothing else",
         (context_src, header_src, ctx_header_src.replace("sycl_hold_spill_validated_ub", "sycl_hold_spill_XXXX"))),
        ("previous rung's graph results kept (R7)", "the release drops every cached graph result and the active pointer, not only the sched",
         (mutate_re_in_func(context_src, r"auto release_rung_buffers = \[&\]\(\) \{",
                            r"for\s*\(\s*auto\s*&\s*res\s*:\s*gf_res_prev\s*\)\s*\{\s*res\.reset\(\)\s*;\s*\}", ""), header_src)),
        ("active graph result pointer kept (R8)", "the release drops every cached graph result and the active pointer, not only the sched",
         (mutate_re_in_func(context_src, r"auto release_rung_buffers = \[&\]\(\) \{",
                            r"gf_res_prev_active\s*=\s*nullptr\s*;", "(void) 0;"), header_src)),
    ]
    # kpjw item 0: the downward continuation. Each mutant removes one property of it and must be caught.
    def ctx_mut(label, expect, new_ctx=None, new_auto=None):
        return (label, expect, (new_ctx if new_ctx is not None else context_src, header_src, ctx_header_src,
                                new_auto if new_auto is not None else auto_header_src))

    descent_cond = r"if\s*\(\s*last_good\s*==\s*0\s*&&\s*ladder_needed\s*&&\s*fallback_tried\s*&&\s*rung_fit_refused\s*\)"
    WALK = "the pure downward walk exists: halves to a multiple of 64, floor 64, rungs above the cap skipped"
    DESC = "the trial continues downward when the default rung and everything above it lost to a real fit refusal, and only then"
    TRIED = "a rung that lost is only lowered from when the default itself was tried (a non-rung default is not skipped)"
    FLAG = "rung_fit_refused is cleared at the start of every rung and set by exactly the three real fit refusals"
    INIT = "the trial state starts at nothing-happened and every rung asked is named in tried"
    NCTX = "the trial never writes the context size: a smaller -ub is not a smaller context"
    CREF = "a refused cached rung is known to this start and its entry is overwritten, not re-paid"
    LOW = "a lowered result is announced once, after the settle, and is stored only to overwrite a refused cached rung"
    SETTLE = "a refused settle publish is recorded and is a named error from the one fit function, with no second descent"
    NAMED = "the named settle error names the scope, the rungs tried, the LAST rung's stop reason and the fit function's -ub"
    SCOPE = "the scheduler compute scope is opened around the reserve and the graph allocation, resolved once per context"
    ADV = "the advice caps the fit function's answer under the lowest refused rung and names nothing under the floor"
    GONE = "there is no settle refusal descent"
    def ctx_mut_h(label, expect, new_ctx=None, new_hdr=None):
        return (label, expect, (new_ctx if new_ctx is not None else context_src, header_src,
                                new_hdr if new_hdr is not None else ctx_header_src, auto_header_src))
    ctx_mutants += [
        ctx_mut("descent floor lowered below 64", WALK, new_auto=re.sub(r"descent_floor\s*=\s*64", "descent_floor = 16", auto_header_src, count=1)),
        ctx_mut("descent rungs not snapped to 64", WALK, new_auto=re.sub(r"half\s*-\s*half\s*%\s*64", "half", auto_header_src, count=1)),
        ctx_mut("descent rungs snapped to 32 again", WALK, new_auto=re.sub(r"half\s*%\s*64", "half % 32", auto_header_src, count=1)),
        ctx_mut("descent ignores the cap", WALK, new_auto=re.sub(r"if\s*\(\s*c\s*>\s*cap\s*\)\s*\{\s*continue\s*;\s*\}", "", auto_header_src, count=1)),
        ctx_mut("descent accepts a refused rung", WALK, new_auto=re.sub(r"if\s*\(\s*try_rung\(\s*c\s*\)\s*\)", "if (!try_rung(c))", auto_header_src, count=1)),
        ctx_mut("advice returns the answer uncapped", ADV, new_auto=re.sub(r"if\s*\(\s*lowest_refused\s*==\s*0\s*\|\|\s*largest_fit\s*<\s*lowest_refused\s*\)\s*\{\s*return largest_fit;\s*\}", "return largest_fit;", auto_header_src, count=1)),
        ctx_mut("advice names a rung under the floor", ADV, new_auto=auto_header_src.replace("p >= llama_auto_ubatch_descent_floor ? p : 0", "p", 1)),
        ctx_mut("advice names the refused rung itself", ADV, new_auto=auto_header_src.replace("p * 2 < lowest_refused", "p * 2 <= lowest_refused", 1)),
        ctx_mut("advice invents an answer when none is known", ADV, new_auto=re.sub(r"if\s*\(\s*largest_fit\s*==\s*0\s*\)\s*\{\s*return 0;\s*\}", "", auto_header_src, count=1)),
        ctx_mut("settle refusal descent comes back (header)", GONE, new_auto=auto_header_src + "\ntemplate <typename F> inline uint32_t llama_auto_ubatch_settle_refusal_descend(uint32_t r, bool d, uint32_t c, F f) { return d ? 0 : llama_auto_ubatch_descend(r, c, f); }\n"),
        ctx_mut("settle refusal descent comes back (trial)", GONE, new_ctx=context_src.replace("const uint32_t advice =", "(void) llama_auto_ubatch_settle_refusal_descend; const uint32_t advice =", 1)),
        ctx_mut("refusal_largest_ub is back", GONE, new_ctx=context_src.replace("uint32_t       lowest_refused", "uint32_t refusal_largest_ub = 0; uint32_t       lowest_refused", 1)),
        ctx_mut("descent never runs", DESC, new_ctx=re.sub(descent_cond, "if (false)", context_src, count=1)),
        ctx_mut("descent runs after any loss, race included", DESC,
                new_ctx=re.sub(descent_cond, "if (last_good == 0 && ladder_needed && fallback_tried)", context_src, count=1)),
        ctx_mut("descent runs without a ladder", DESC,
                new_ctx=re.sub(descent_cond, "if (last_good == 0 && fallback_tried && rung_fit_refused)", context_src, count=1)),
        ctx_mut("descent gated on the old race test", DESC,
                new_ctx=re.sub(descent_cond, "if (last_good == 0 && ladder_needed && fallback_tried && !stop_is_pure_race)", context_src, count=1)),
        ctx_mut("descent ignores whether the default was tried", TRIED,
                new_ctx=re.sub(r"fallback_tried\s*&&\s*rung_fit_refused", "rung_fit_refused", context_src, count=1)),
        ctx_mut("fallback_tried never set by the ladder", TRIED,
                new_ctx=re.sub(r"if\s*\(\s*c\s*==\s*fallback_ubatch\s*\)\s*\{\s*fallback_tried\s*=\s*true\s*;\s*\}", "", context_src, count=1)),
        ctx_mut("fallback_tried never set by a refused cached default", TRIED,
                new_ctx=re.sub(r"fallback_tried\s*=\s*fallback_tried\s*\|\|\s*cached_ubatch\s*==\s*fallback_ubatch\s*;", "(void) 0;", context_src, count=1)),
        ctx_mut("descent starts from the wrong rung", DESC,
                new_ctx=re.sub(r"llama_auto_ubatch_descend\(\s*fallback_ubatch\s*,", "llama_auto_ubatch_descend(cparams.n_ubatch,", context_src, count=1)),
        ctx_mut("descent winner not adopted", DESC,
                new_ctx=re.sub(r"last_good\s*=\s*ended\s*;", "(void) ended;", context_src, count=1)),
        ctx_mut("descent loser counted as a winner", DESC,
                new_ctx=mutate_re_in_func(context_src, r"void llama_context::sycl_select_auto_ubatch\(", r"if\s*\(\s*rung_reason\s*==\s*nullptr\s*\)\s*\{\s*return true\s*;\s*\}", "if (rung_reason != nullptr) { return true; }")),
        ctx_mut("descent walks on past a non-fit loss", DESC,
                new_ctx=re.sub(r"descent_aborted\s*=\s*!rung_fit_refused\s*;", "descent_aborted = false;", context_src, count=1)),
        ctx_mut("descent adopts the rung it aborted at", DESC,
                new_ctx=re.sub(r"if\s*\(\s*ended\s*!=\s*0\s*&&\s*!descent_aborted\s*\)", "if (ended != 0)", context_src, count=1)),
        ctx_mut("descent does not mark the ring dirty", DESC,
                new_ctx=mutate_re_in_func(context_src, r"if \(last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused\)", r"publish_dirty\s*=\s*true\s*;", "(void) 0;")),
        ctx_mut("descent forgets that it ran", DESC,
                new_ctx=mutate_re_in_func(context_src, r"if \(last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused\)", r"descent_ran\s*=\s*true\s*;", "(void) 0;")),
        ctx_mut("descent loses never noted", DESC,
                new_ctx=mutate_re_in_func(context_src, r"if \(last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused\)", r"note_loss\(\s*c\s*,\s*rung_reason\s*\)\s*;", "(void) 0;")),
        ctx_mut("fit flag never cleared per rung", FLAG, new_ctx=re.sub(r"(auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{\s*)rung_fit_refused\s*=\s*false\s*;", r"\1", context_src, count=1)),
        ctx_mut("a lifecycle failure is a fit refusal", FLAG,
                new_ctx=mutate_re_in_func(context_src, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{", r"if\s*\(\s*rc\s*!=\s*GGML_SYCL_LIFECYCLE_OK\s*\)\s*\{", "if (rc != GGML_SYCL_LIFECYCLE_OK) { rung_fit_refused = true;")),
        ctx_mut("a demoted KV is a fit refusal", FLAG,
                new_ctx=context_src.replace('return "KV would be demoted";', 'rung_fit_refused = true; return "KV would be demoted";', 1)),
        ctx_mut("a host fallback is a fit refusal", FLAG,
                new_ctx=context_src.replace('return "compute buffer fell back to host";', 'rung_fit_refused = true; return "compute buffer fell back to host";', 1)),
        ctx_mut("a publish that threw is a fit refusal", FLAG,
                new_ctx=mutate_re_in_func(context_src, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{", r"publish_dirty\s*=\s*true\s*;\s*return \"transaction refused\"\s*;", 'publish_dirty = true; rung_fit_refused = true; return "transaction refused";')),
        ctx_mut("a probe refusal is not a fit refusal", FLAG,
                new_ctx=mutate_re_in_func(context_src, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{", r"if\s*\(\s*!probe\.accepted\s*\)\s*\{\s*rung_fit_refused\s*=\s*true\s*;", "if (!probe.accepted) {")),
        ctx_mut("a compute buffer that did not fit is not a fit refusal", FLAG,
                new_ctx=re.sub(r"rung_fit_refused\s*=\s*dynamic_cast<const\s+llama_auto_ubatch_fit_refusal\s*\*>\(\s*&e\s*\)\s*!=\s*nullptr\s*;\s*return \"compute buffers did not fit\"", 'return "compute buffers did not fit"', context_src, count=1)),
        ctx_mut("every reserve exception is a fit refusal", FLAG,
                new_ctx=re.sub(r"rung_fit_refused\s*=\s*dynamic_cast<const\s+llama_auto_ubatch_fit_refusal\s*\*>\(\s*&e\s*\)\s*!=\s*nullptr\s*;", "rung_fit_refused = true;", context_src, count=1)),
        ctx_mut("a hold spill is not a fit refusal", FLAG,
                new_ctx=re.sub(r"rung_fit_refused\s*=\s*true\s*;\s*return \"hold spill left no headroom\"", 'return "hold spill left no headroom"', context_src, count=1)),
        ctx_mut("descent_ran starts true (B1)", INIT, new_ctx=re.sub(r"bool(\s+)descent_ran(\s+)=\s*false\s*;", r"bool\1descent_ran\2= true;", context_src, count=1)),
        ctx_mut("fallback_tried starts true (B2)", INIT, new_ctx=re.sub(r"bool(\s+)fallback_tried(\s+)=\s*false\s*;", r"bool\1fallback_tried\2= true;", context_src, count=1)),
        ctx_mut("lowered_from starts set (B3)", INIT, new_ctx=re.sub(r"uint32_t(\s+)lowered_from(\s+)=\s*0\s*;", r"uint32_t\1lowered_from\2= 1;", context_src, count=1)),
        ctx_mut("rung_fit_refused starts true", INIT, new_ctx=re.sub(r"bool(\s+)rung_fit_refused(\s+)=\s*false\s*;", r"bool\1rung_fit_refused\2= true;", context_src, count=1)),
        ctx_mut("lowest_refused starts set", INIT, new_ctx=re.sub(r"uint32_t(\s+)lowest_refused(\s+)=\s*0\s*;", r"uint32_t\1lowest_refused\2= 1;", context_src, count=1)),
        ctx_mut("cache_refused_ub starts set", INIT, new_ctx=re.sub(r"uint32_t(\s+)cache_refused_ub(\s+)=\s*0\s*;", r"uint32_t\1cache_refused_ub\2= 1;", context_src, count=1)),
        ctx_mut("the cached rung is not named in tried", INIT, new_ctx=context_src.replace("std::to_string(cached_ubatch);", "std::string();", 1)),
        ctx_mut("the ladder rung is not named in tried", INIT, new_ctx=re.sub(r"tried\s*\+=\s*\(tried\.empty\(\)\s*\?\s*\"\"\s*:\s*\",\"\)\s*\+\s*std::to_string\(c\)\s*;", "", context_src, count=1)),
        ctx_mut("the descent rung is not named in tried", INIT, new_ctx=re.sub(r"tried\.append\(tried\.empty\(\)\s*\?\s*\"\"\s*:\s*\",\"\)\.append\(std::to_string\(c\)\)\s*;", "", context_src, count=1)),
        ctx_mut("the settle shrinks the context", NCTX, new_ctx=re.sub(r"cparams\.n_ubatch\s*=\s*last_good\s*;", "cparams.n_ubatch = last_good; cparams.n_ctx = last_good;", context_src, count=1)),
        ctx_mut("the descent shrinks the context", NCTX, new_ctx=mutate_re_in_func(context_src, r"if \(last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused\)", r"descent_ran\s*=\s*true\s*;", "descent_ran = true; cparams.n_ctx /= 2;")),
        ctx_mut("the trial settles on a smaller context after a refusal", NCTX, new_ctx=mutate_re_in_func(context_src, r"auto try_candidate = \[&\]\(uint32_t c\) -> const char \* \{", r"cparams\.n_ubatch\s*=\s*c\s*;", "cparams.n_ubatch = c; cparams.n_ctx -= 0;")),
        ctx_mut("a refused cached rung is forgotten", CREF, new_ctx=re.sub(r"cache_refused_ub\s*=\s*cached_ubatch\s*;", "(void) 0;", context_src, count=1)),
        ctx_mut("a refused cached rung is remembered after a race too", CREF,
                new_ctx=re.sub(r"if\s*\(\s*rung_fit_refused\s*\)\s*\{\s*cache_refused_ub", "if (true) { cache_refused_ub", context_src, count=1)),
        ctx_mut("the ladder asks the refused cached rung again", CREF,
                new_ctx=re.sub(r"if\s*\(\s*cache_refused_ub\s*!=\s*0\s*&&\s*c\s*>=\s*cache_refused_ub\s*\)", "if (false)", context_src, count=1)),
        ctx_mut("the ladder skips the refused rung and climbs on", CREF,
                new_ctx=re.sub(r"(c\s*>=\s*cache_refused_ub\s*\)\s*\{\s*stop\s*=\s*cache_refused_reason\s*;\s*last_stop\s*=\s*cache_refused_reason\s*;\s*)break\s*;", r"\1continue;", context_src, count=1)),
        ctx_mut("a refused cached rung's entry is never overwritten", CREF,
                new_ctx=re.sub(r"lowered_from\s*==\s*0\s*\?\s*!descent_ran\s*:\s*cache_refused_ub\s*!=\s*0", "lowered_from == 0 ? !descent_ran : false", context_src, count=1)),
        ctx_mut("a walk that found nothing stores the default", CREF,
                new_ctx=re.sub(r"lowered_from\s*==\s*0\s*\?\s*!descent_ran\s*:", "lowered_from == 0 ? true :", context_src, count=1)),
        ctx_mut("lowered result stored without a refused cache", CREF,
                new_ctx=re.sub(r"store_outcome\s*=\s*lowered_from\s*==\s*0\s*\?\s*!descent_ran\s*:\s*cache_refused_ub\s*!=\s*0", "store_outcome = true", context_src, count=1)),
        ctx_mut("lowered result not announced", LOW,
                new_ctx=context_src.replace("auto n_ubatch lowered from %u to %u", "auto n_ubatch XXXX", 1)),
        ctx_mut("store gate lost its store_outcome term", LOW,
                new_ctx=re.sub(r"!stop_is_pure_race\s*&&\s*store_outcome\s*&&", "!stop_is_pure_race &&", context_src, count=1)),
        ctx_mut("settle refusal thrown instead of recorded", SETTLE,
                new_ctx=re.sub(r"settle_error\s*=\s*std::current_exception\(\)\s*;", "throw;", context_src, count=1)),
        ctx_mut("settle refusal tries rungs again", SETTLE,
                new_ctx=mutate_re_in_func(context_src, r"if \(settle_error\) \{", r"uint32_t\s+largest_ub\s*=\s*0\s*;", "uint32_t largest_ub = 0; (void) try_candidate(last_good / 2);")),
        ctx_mut("settle refusal never asks the fit function", SETTLE,
                new_ctx=re.sub(r"if\s*\(\s*llama_context_sycl_hold_spill_fits\(\s*backends\s*,\s*last_good\s*,\s*&largest_ub\s*\)\s*\)", "if (false)", context_src, count=1)),
        ctx_mut("settle refusal asks the fit function about another rung", SETTLE,
                new_ctx=re.sub(r"llama_context_sycl_hold_spill_fits\(\s*backends\s*,\s*last_good\s*,\s*&largest_ub\s*\)", "llama_context_sycl_hold_spill_fits(backends, fallback_ubatch, &largest_ub)", context_src, count=1)),
        ctx_mut("settle race is swallowed into the named error", SETTLE,
                new_ctx=re.sub(r"std::rethrow_exception\(\s*settle_error\s*\)", "(void) 0", context_src, count=1)),
        ctx_mut("settle named error never thrown", SETTLE,
                new_ctx=mutate_re_in_func(context_src, r"if \(settle_error\) \{", r"throw std::runtime_error\(", "(void) std::runtime_error(")),
        ctx_mut("settle refusal returns", SETTLE,
                new_ctx=mutate_re_in_func(context_src, r"if \(settle_error\) \{", r"const uint32_t\s+advice\s*=", "return; const uint32_t advice =")),
        ctx_mut("settle reserve skipped when the publish was fine", SETTLE,
                new_ctx=re.sub(r"\}\s*else\s*\{\s*sched_need_reserve\s*=\s*true\s*;\s*sched_reserve\(\)\s*;\s*\}", "} else { }", context_src, count=1)),
        ctx_mut("settle advice not from the advice helper", SETTLE,
                new_ctx=re.sub(r"llama_auto_ubatch_advice\(\s*largest_ub\s*,\s*lowest_refused\s*\)", "largest_ub", context_src, count=1)),
        ctx_mut("cache store ahead of the settle", "the tuning-cache store runs after the settle, so a rung the settle refused is never persisted",
                new_ctx=context_src.replace("if (!sched_matches_last_good || cparams.n_ubatch != last_good) {", "cache_store_fn(&cache_key, 0, stop); if (!sched_matches_last_good || cparams.n_ubatch != last_good) {", 1)),
        ctx_mut("named error unnamed", NAMED, new_ctx=context_src.replace("no -ub from %u down to %u fits this context", "XXXX", 1)),
        ctx_mut("named error forgets a non-descent scope", NAMED, new_ctx=context_src.replace("%u does not fit this context", "XXXX", 1)),
        ctx_mut("named error names no -ub", NAMED, new_ctx=mutate_in_func(context_src, r"if \(settle_error\) \{", "the largest -ub that fits is about", "XXXX")),
        ctx_mut("named error omits the last stop", NAMED, new_ctx=context_src.replace("tried %s; last stop: %s", "tried %s; stop: %s", 1)),
        ctx_mut("named error reports the first stop, not the last", NAMED,
                new_ctx=re.sub(r"last_stop\s*!=\s*nullptr\s*\?\s*last_stop\s*:\s*stop", "stop", context_src, count=1)),
        ctx_mut("a loss records no stop reason", NAMED, new_ctx=re.sub(r"last_stop\s*=\s*reason\s*;", "(void) reason;", context_src, count=1)),
        ctx_mut("a loss lowers no bound", NAMED, new_ctx=re.sub(r"lowest_refused\s*=\s*lowest_refused\s*==\s*0\s*\?\s*c\s*:\s*std::min\(lowest_refused, c\)\s*;", "(void) c;", context_src, count=1)),
        ctx_mut("the cached loss is not noted", NAMED, new_ctx=re.sub(r"note_loss\(\s*cached_ubatch\s*,\s*cache_reason\s*\)\s*;", "(void) 0;", context_src, count=1)),
        ctx_mut("the ladder loss is not noted", NAMED, new_ctx=re.sub(r"note_loss\(\s*c\s*,\s*reason\s*\)\s*;", "(void) 0;", context_src, count=1)),
        ctx_mut_h("the guard never opens the scope", SCOPE, new_ctx=context_src.replace("fn(true);", "(void) 0;", 1)),
        ctx_mut_h("the guard never closes the scope", SCOPE, new_ctx=context_src.replace("fn(false);", "(void) 0;", 1)),
        ctx_mut_h("the graph allocation runs outside the scope", SCOPE,
                  new_ctx=re.sub(r"(bool llama_context::sched_alloc_graph\(ggml_cgraph \* gf\) \{\s*)sycl_compute_scope_guard\s+sycl_scope\(sycl_compute_scope_fn\(\)\)\s*;\s*", r"\1", context_src, count=1)),
        ctx_mut_h("the reserve runs outside the scope", SCOPE,
                  new_ctx=re.sub(r"(bool llama_context::sched_reserve_graph\(ggml_cgraph \* gf\) \{\s*)sycl_compute_scope_guard\s+sycl_scope\(sycl_compute_scope_fn\(\)\)\s*;\s*", r"\1", context_src, count=1)),
        ctx_mut_h("a bare graph allocation comes back", SCOPE,
                  new_ctx=context_src.replace("if (!sched_alloc_graph(gf)) {", "if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {", 1)),
        ctx_mut_h("a bare reserve comes back", SCOPE,
                  new_ctx=context_src.replace(": sched_reserve_graph(gf);", ": ggml_backend_sched_reserve(sched.get(), gf);", 1)),
        ctx_mut_h("the scope is resolved on every call", SCOPE, new_ctx=context_src.replace("sycl_compute_scope_resolved = true;", "(void) 0;", 1)),
        ctx_mut_h("the direct-build scope function is not bound", SCOPE, new_ctx=context_src.replace("sycl_compute_scope_cached = &ggml_backend_sycl_compute_alloc_scope;", "(void) 0;", 1)),
        ctx_mut_h("the backend-DL scope function is not looked up", SCOPE, new_ctx=context_src.replace('"ggml_backend_sycl_compute_alloc_scope"', '"ggml_backend_sycl_XXXX"', 1)),
        ctx_mut_h("the context header lacks the scope resolver", SCOPE, new_hdr=ctx_header_src.replace("sycl_compute_scope_fn_t sycl_compute_scope_fn();", "")),
        ctx_mut_h("the context header lacks the resolved flag", SCOPE, new_hdr=ctx_header_src.replace("bool                    sycl_compute_scope_resolved = false;", "")),
    ]
    for label, expect, sources in ctx_mutants:
        failed += run_context(label, sources, expect)

    raw_mutants = [
        ("GPU test not registered with mem-handle", "the GPU test is registered under the mem-handle label",
         (raw_inputs[0], raw_inputs[1], raw_inputs[2].replace("sycl;mem-handle;bugfix", "sycl;bugfix"), raw_inputs[3])),
        ("GPU test does not return to the baseline", "the GPU test asserts the landing zone, the zone-used delta and the return to the baseline",
         (raw_inputs[0], raw_inputs[1], raw_inputs[2], raw_inputs[3].replace("zone_used(vram_zone_id::KV) == kv_before", "true"))),
    ]
    def raw_with(index, old, new):
        out = list(raw_inputs)
        if old not in out[index]:
            print(f"FAIL: self-test anchor missing: raw[{index}] {old!r}")
            failed.append("self-test anchor " + old)
            return tuple(out)
        out[index] = out[index].replace(old, new, 1)
        return tuple(out)

    raw_mutants += [
        ("GPU test ignores unified_free", "the GPU test checks every unified_free, and asserts the raw spill and its WARN on the unflagged control",
         raw_with(3, 'check(unified_free(handle), "unified_free accepted the handle");', "unified_free(handle);")),
        ("release comment says after", "the release comment says the teardown take ran BEFORE it (it does: log_planned_scratch_stats, then the release)",
         raw_with(4, "BEFORE this release", "after this release")),
        ("entry arity comment lost", "the exported entry's comment states the arity change and what an old DSO does",
         raw_with(0, "largest_ub stays 0", "nothing happens")),
        ("lazy re-reserve gap unnamed", "the context-init check's comment names the gap: a later lazy re-reserve is not covered",
         raw_with(5, "not covered", "covered")),
    ]
    for label, expect, inputs in raw_mutants:
        results_raw = evaluate_raw(*inputs)
        fired = not results_raw[expect]
        print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect}'")
        if not fired:
            failed.append(label)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
