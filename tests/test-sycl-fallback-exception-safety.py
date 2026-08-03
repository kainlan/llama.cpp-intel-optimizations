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

# Each segmented attempt snapshots retained handles and rolls additions back on
# either SYCL or generic failure before publishing context graph state.
assert "const size_t segmented_retained_baseline = sycl_ctx->graph_retained_handles.size()" in sycl
assert sycl.count("graph_retained_handles.resize(segmented_retained_baseline)") >= 2
assert sycl.index("graph_retained_handles.resize(segmented_retained_baseline)") < sycl.index("sycl_ctx->moe_segments           = std::move(recorded_segments)")
print("SYCL fallback exception safety source injections: PASS")
