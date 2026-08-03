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

# Non-DL staging OOM/copy exceptions restore all tensor pointers and free CPU
# resources before graph planning can run.
staging_check = sycl.index("if (staging_failed)")
plan = sycl.index("ggml_graph_plan", staging_check)
block = sycl[staging_check:plan]
for token in ("restore_host_copies()", "ggml_threadpool_free(tp)", "ggml_free(gctx)", "return false"):
    assert token in block

# Boundary cleanup drains prior submissions and deferred scatters, unpins
# transient leases, and clears active/deferred state before FAILED.
catch = sycl.index("catch (const ggml_sycl_fallback_error & error)")
failed = sycl.index("return GGML_STATUS_FAILED", catch)
cleanup = sycl[catch:failed]
for token in ("last_graph_event->wait_and_throw", "stream()->wait_and_throw",
              "flush_pending_secondary_scatter", "ggml_sycl_cpu_staging_drain",
              "graph_unpin_transient_leases_after_direct_execution",
              "last_graph_event_deferred_decode = false",
              "unified_cache_set_graph_compute_active(false)"):
    assert token in cleanup
print("SYCL fallback exception safety source injections: PASS")
