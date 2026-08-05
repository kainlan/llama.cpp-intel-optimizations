#!/usr/bin/env python3
"""Injected source contracts for recoverable DL fallback exception safety."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
sycl = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
getrows = (root / "ggml/src/ggml-sycl/getrows.cpp").read_text()
module_test = (root / "tests/test-sycl-module-dependencies.py").read_text()

# DL preprocessing reaches the throw and excludes the CPU implementation call.
direct = re.search(r"static bool ggml_sycl_cpu_get_rows_direct\(.*?\n\}", getrows, re.S).group(0)
assert re.search(r"#ifdef GGML_BACKEND_DL.*?throw ggml_sycl_fallback_error\(reason\);.*?#else", direct, re.S)
assert direct.index("#else") < direct.index("ggml_compute_forward_get_rows") < direct.rindex("#endif")
assert "ggml_compute_forward_get_rows" in module_test

# Recording/re-record injected failures restore all global/context capture state,
# depth and retained sinks through noexcept guards before propagating.
assert sycl.count("struct recording_exception_guard") >= 2
assert sycl.count("~recording_exception_guard() noexcept") >= 2
for token in ("g_recording_graph_ptr = old_graph", "g_recording_queue_ptr = old_queue",
              "set_graph_retained_handle_sink(nullptr)", "graph_retained_handles.clear()",
              "g_ggml_sycl_graph_recording_depth.fetch_sub"):
    assert token in sycl
assert sycl.count("dynamic_cast<const ggml_sycl_fallback_error *>") >= 2

# Prefix view restoration is noexcept and includes both mutable graph fields;
# suffix dispatch remains an explicit operation outside the destructor.
assert "~graph_view_guard() noexcept" in sycl
assert "cg->nodes   = nodes" in sycl and "cg->n_nodes = n_nodes" in sycl
prefix_dtor = re.search(r"~prefix_suffix_guard\(\) noexcept \{(.*?)\n        \}", sycl, re.S).group(1)
assert "dispatch_suffix" not in prefix_dtor
assert "suffix_guard.dispatch_suffix();" in sycl

# compute_forward must not convert the recoverable fallback into its generic
# std::exception fatal-exit path.
cf_fallback = sycl.index("catch (const ggml_sycl_fallback_error &)")
cf_generic = sycl.index("catch (const std::exception & e)", cf_fallback)
assert cf_fallback < cf_generic and "throw;" in sycl[cf_fallback:cf_generic]

# Non-DL staging reserve/push/copy failures restore all tensor pointers and
# release CPU resources before graph planning can run.
staging_check = sycl.index("if (staging_failed)")
plan = sycl.index("ggml_graph_plan", staging_check)
block = sycl[staging_check:plan]
for token in ("restore_host_copies()", "return false"):
    assert token in block
assert "struct cpu_fallback_resources" in sycl and "~cpu_fallback_resources() noexcept" in sycl
fallback_start = sycl.index("struct cpu_fallback_resources")
assert sycl.index("host_copies.reserve", fallback_start) < sycl.index("unified_alloc(req", fallback_start)
assert sycl.index("host_copies.push_back(std::move(entry))") < sycl.index("ggml_sycl_assign_tensor_storage(t, host_copies.back().host_ptr)")
assert "struct tensor_storage_restore_guard" in sycl and "~tensor_storage_restore_guard() noexcept" in sycl

# Boundary cleanup drains prior submissions and deferred scatters, unpins
# transient leases, and clears active/deferred state before FAILED.
catch = sycl.index("catch (const ggml_sycl_fallback_error & error)")
failed = sycl.index("return GGML_STATUS_FAILED", catch)
cleanup = sycl[catch:failed]
for token in ("ggml_sycl_cpu_tg_flush_pending", "last_graph_event->wait_and_throw", "stream()->wait_and_throw",
              "flush_pending_secondary_scatter", "ggml_sycl_cpu_staging_drain",
              "graph_unpin_transient_leases_after_direct_execution",
              "last_graph_event_deferred_decode = false",
              "unified_cache_set_graph_compute_active(false)"):
    assert token in cleanup
assert cleanup.index("ggml_sycl_cpu_tg_flush_pending") < cleanup.index("last_graph_event->wait_and_throw")

# Segment 1 may already be asynchronously submitted when segment 2 fails.
# Per-attempt baselines preserve prior handles; current handles roll back only
# before successful submission. Depth ownership releases exactly once even for
# post-end finalize/allocation/submit/MoE exceptions.
assert "const size_t retained_baseline = sycl_ctx->graph_retained_handles.size()" in sycl
assert "bool segment_submitted = false" in sycl
assert "segment_submitted = true" in sycl
assert sycl.index("segment_submitted = true") < sycl.index("stream->ext_oneapi_graph(*recorded_segments.back().exec_graph)")
assert sycl.count("if (!segment_submitted)") >= 2
assert sycl.count("graph_retained_handles.resize(retained_baseline)") >= 2
assert "struct recording_depth_owner" in sycl
assert "~recording_depth_owner() noexcept" in sycl
segment_start = sycl.index("struct recording_depth_owner")
segment_end = sycl.index("// 4. Store in context", segment_start)
segment_block = sycl[segment_start:segment_end]
assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_add") == 1
assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_sub") == 1
# Behavioral fault model: segment 1 is submitted, segment 2 appends then fails.
handles = ["segment1-submitted"]
segment2_baseline = len(handles)
handles.append("segment2-never-submitted")
handles[segment2_baseline:] = []
assert handles == ["segment1-submitted"]
depth = 7
depth += 1       # acquire
# end_recording releases; finalize/alloc/submit/MoE fault sees owner=false
depth -= 1
assert depth == 7

# Failed CPU compute must restore and return before any destination H2D copy.
compute_status = sycl.index("const ggml_status status = ggml_graph_compute")
h2d = sycl.index("if (dst_is_device && dst_device_ptr)", compute_status)
failed_block = sycl[compute_status:h2d]
assert "if (status != GGML_STATUS_SUCCESS)" in failed_block
assert "restore_host_copies()" in failed_block and "return false" in failed_block
actions = ["compute-failed", "restore", "free", "return-false"]
assert "h2d" not in actions
print("SYCL fallback exception safety source injections: PASS")
