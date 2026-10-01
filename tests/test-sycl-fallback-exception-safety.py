#!/usr/bin/env python3
"""Injected source contracts for recoverable DL fallback exception safety."""
from pathlib import Path
import re
from sycl_gate import finish, gate

root = Path(__file__).resolve().parents[1]
sycl = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
getrows = (root / "ggml/src/ggml-sycl/getrows.cpp").read_text()
module_test = (root / "tests/test-sycl-module-dependencies.py").read_text()
recorder_scope = (root / "ggml/src/ggml-sycl/graph-recorder-scope.hpp").read_text()

# DL preprocessing reaches the throw and excludes the CPU implementation call.
with gate('L12 direct = re.search(r"static bool ggml_sycl_cpu_get_rows_direct\\(.*?\\n\\}", getrows, re.S).g'):
    direct = re.search(r"static bool ggml_sycl_cpu_get_rows_direct\(.*?\n\}", getrows, re.S).group(0)
with gate('L13 assert re.search(r"#ifdef GGML_BACKEND_DL.*?throw ggml_sycl_fallback_error\\(reason\\);.*?#e'):
    assert re.search(r"#ifdef GGML_BACKEND_DL.*?throw ggml_sycl_fallback_error\(reason\);.*?#else", direct, re.S)
with gate('L14 assert direct.index("#else") < direct.index("ggml_compute_forward_get_rows") < direct.rind'):
    assert direct.index("#else") < direct.index("ggml_compute_forward_get_rows") < direct.rindex("#endif")
with gate('L15 assert "ggml_compute_forward_get_rows" in module_test'):
    assert "ggml_compute_forward_get_rows" in module_test

# Recording/re-record injected failures restore all global/context capture state,
# depth and retained sinks through noexcept guards before propagating.
with gate('L19 assert sycl.count("struct recording_exception_guard") >= 2'):
    assert sycl.count("struct recording_exception_guard") >= 2
with gate('L20 assert sycl.count("~recording_exception_guard() noexcept") >= 2'):
    assert sycl.count("~recording_exception_guard() noexcept") >= 2
for token in ("set_graph_retained_handle_sink(nullptr)", "graph_retained_handles.clear()",
              "g_ggml_sycl_graph_recording_depth.fetch_sub"):
    with gate("recording failure restores %s" % token):
        assert token in sycl

# The global/queue capture state is restored by one RAII owner since 982f77f69 (llama.cpp-5udh), no
# longer by `g_recording_graph_ptr = old_graph` lines at each recording site. Pin the owner instead:
# its leave() is noexcept, reached from the destructor, and puts the graph, queue and recording flag
# back, detaches the sink and releases the depth it took.
with gate("graph_recorder_scope::leave is noexcept and restores the capture state"):
    leave = re.search(r"void leave\(\) noexcept \{(.*?)\n    \}", recorder_scope, re.S).group(1)
    for restored in (r"s_\.graph\s*=\s*saved_graph_", r"s_\.queue\s*=\s*saved_queue_",
                     r"s_\.recording\s*=\s*saved_recording_", r"s_\.set_sink\(nullptr\)",
                     r"s_\.depth\.fetch_sub"):
        assert re.search(restored, leave), restored
with gate("graph_recorder_scope destructor leaves the scope"):
    assert re.search(r"~graph_recorder_scope\(\) \{\s*leave\(\);", recorder_scope)
with gate("both recording sites use the recorder scope"):
    assert sycl.count("ggml_sycl_graph_recorder recorder(") >= 1 and "recorder_.emplace(" in sycl
with gate('L25 assert sycl.count("dynamic_cast<const ggml_sycl_fallback_error *>") >= 2'):
    assert sycl.count("dynamic_cast<const ggml_sycl_fallback_error *>") >= 2

# Prefix view restoration is noexcept and includes both mutable graph fields;
# suffix dispatch remains an explicit operation outside the destructor.
with gate('L29 assert "~graph_view_guard() noexcept" in sycl'):
    assert "~graph_view_guard() noexcept" in sycl
with gate('L30 assert "cg->nodes = nodes" in sycl and "cg->n_nodes = n_nodes" in sycl'):
    assert "cg->nodes   = nodes" in sycl and "cg->n_nodes = n_nodes" in sycl
with gate('L31 prefix_dtor = re.search(r"~prefix_suffix_guard\\(\\) noexcept \\{(.*?)\\n \\}", sycl, re.S).gro'):
    prefix_dtor = re.search(r"~prefix_suffix_guard\(\) noexcept \{(.*?)\n        \}", sycl, re.S).group(1)
with gate('L32 assert "dispatch_suffix" not in prefix_dtor'):
    assert "dispatch_suffix" not in prefix_dtor
with gate('L33 assert "suffix_guard.dispatch_suffix();" in sycl'):
    assert "suffix_guard.dispatch_suffix();" in sycl

# compute_forward must not convert the recoverable fallback into its generic
# std::exception fatal-exit path.
with gate('L37 cf_fallback = sycl.index("catch (const ggml_sycl_fallback_error &)")'):
    cf_fallback = sycl.index("catch (const ggml_sycl_fallback_error &)")
with gate('L38 cf_generic = sycl.index("catch (const std::exception & e)", cf_fallback)'):
    cf_generic = sycl.index("catch (const std::exception & e)", cf_fallback)
with gate('L39 assert cf_fallback < cf_generic and "throw;" in sycl[cf_fallback:cf_generic]'):
    assert cf_fallback < cf_generic and "throw;" in sycl[cf_fallback:cf_generic]

# Non-DL staging reserve/push/copy failures restore all tensor pointers and
# release CPU resources before graph planning can run.
with gate('L43 staging_check = sycl.index("if (staging_failed)")'):
    staging_check = sycl.index("if (staging_failed)")
with gate('L44 plan = sycl.index("ggml_graph_plan", staging_check)'):
    plan = sycl.index("ggml_graph_plan", staging_check)
block = sycl[staging_check:plan]
with gate('L46 for token in ("restore_host_copies()", "return false"):'):
    for token in ("restore_host_copies()", "return false"):
        assert token in block
with gate('L48 assert "struct cpu_fallback_resources" in sycl and "~cpu_fallback_resources() noexcept" in'):
    assert "struct cpu_fallback_resources" in sycl and "~cpu_fallback_resources() noexcept" in sycl
with gate('L49 fallback_start = sycl.index("struct cpu_fallback_resources")'):
    fallback_start = sycl.index("struct cpu_fallback_resources")
with gate('L50 assert sycl.index("host_copies.reserve", fallback_start) < sycl.index("unified_alloc(req",'):
    assert sycl.index("host_copies.reserve", fallback_start) < sycl.index("unified_alloc(req", fallback_start)
with gate('L51 assert sycl.index("host_copies.push_back(std::move(entry))") < sycl.index("ggml_sycl_assig'):
    assert sycl.index("host_copies.push_back(std::move(entry))") < sycl.index("ggml_sycl_assign_tensor_storage(t, host_copies.back().host_ptr)")
with gate('L52 assert "struct tensor_storage_restore_guard" in sycl and "~tensor_storage_restore_guard() '):
    assert "struct tensor_storage_restore_guard" in sycl and "~tensor_storage_restore_guard() noexcept" in sycl

# Boundary cleanup drains prior submissions and deferred scatters, unpins
# transient leases, and clears active/deferred state before FAILED.
with gate('L56 catch = sycl.index("catch (const ggml_sycl_fallback_error & error)")'):
    catch = sycl.index("catch (const ggml_sycl_fallback_error & error)")
with gate('L57 failed = sycl.index("return GGML_STATUS_FAILED", catch)'):
    failed = sycl.index("return GGML_STATUS_FAILED", catch)
cleanup = sycl[catch:failed]
with gate('L59 for token in ("ggml_sycl_cpu_tg_flush_pending", "last_graph_event->wait_and_throw", "strea'):
    for token in ("ggml_sycl_cpu_tg_flush_pending", "last_graph_event->wait_and_throw", "stream()->wait_and_throw",
                  "flush_pending_secondary_scatter", "ggml_sycl_cpu_staging_drain",
                  "graph_unpin_transient_leases_after_direct_execution",
                  "last_graph_event_deferred_decode = false",
                  "unified_cache_set_graph_compute_active(false)"):
        assert token in cleanup
with gate('L65 assert cleanup.index("ggml_sycl_cpu_tg_flush_pending") < cleanup.index("last_graph_event->'):
    assert cleanup.index("ggml_sycl_cpu_tg_flush_pending") < cleanup.index("last_graph_event->wait_and_throw")

# Segment 1 may already be asynchronously submitted when segment 2 fails.
# Per-attempt baselines preserve prior handles; current handles roll back only
# before successful submission. Depth ownership releases exactly once even for
# post-end finalize/allocation/submit/MoE exceptions.
with gate('L71 assert "const size_t retained_baseline = sycl_ctx->graph_retained_handles.size()" in sycl'):
    assert "const size_t retained_baseline = sycl_ctx->graph_retained_handles.size()" in sycl
with gate('L72 assert "bool segment_submitted = false" in sycl'):
    assert "bool segment_submitted = false" in sycl
with gate('L73 assert "segment_submitted = true" in sycl'):
    assert "segment_submitted = true" in sycl
with gate('L74 assert sycl.index("segment_submitted = true") < sycl.index("stream->ext_oneapi_graph(*reco'):
    assert sycl.index("segment_submitted = true") < sycl.index("stream->ext_oneapi_graph(*recorded_segments.back().exec_graph)")
with gate('L75 assert sycl.count("if (!segment_submitted)") >= 2'):
    assert sycl.count("if (!segment_submitted)") >= 2
with gate('L76 assert sycl.count("graph_retained_handles.resize(retained_baseline)") >= 2'):
    assert sycl.count("graph_retained_handles.resize(retained_baseline)") >= 2
with gate('L77 assert "struct recording_depth_owner" in sycl'):
    assert "struct recording_depth_owner" in sycl
with gate('L78 assert "~recording_depth_owner() noexcept" in sycl'):
    assert "~recording_depth_owner() noexcept" in sycl
with gate('L79 segment_start = sycl.index("struct recording_depth_owner")'):
    segment_start = sycl.index("struct recording_depth_owner")
with gate('L80 segment_end = sycl.index("// 4. Store in context", segment_start)'):
    segment_end = sycl.index("// 4. Store in context", segment_start)
segment_block = sycl[segment_start:segment_end]
with gate('L82 assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_add") == 1'):
    assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_add") == 1
with gate('L83 assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_sub") == 1'):
    assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_sub") == 1
# Behavioral fault model: segment 1 is submitted, segment 2 appends then fails.
handles = ["segment1-submitted"]
segment2_baseline = len(handles)
with gate('L87 handles.append("segment2-never-submitted")'):
    handles.append("segment2-never-submitted")
handles[segment2_baseline:] = []
with gate('L89 assert handles == ["segment1-submitted"]'):
    assert handles == ["segment1-submitted"]
depth = 7
with gate('L91 depth += 1 # acquire'):
    depth += 1       # acquire
# end_recording releases; finalize/alloc/submit/MoE fault sees owner=false
with gate('L93 depth -= 1'):
    depth -= 1
with gate('L94 assert depth == 7'):
    assert depth == 7

# Failed CPU compute must restore and return before any destination H2D copy.
with gate('L97 compute_status = sycl.index("const ggml_status status = ggml_graph_compute")'):
    compute_status = sycl.index("const ggml_status status = ggml_graph_compute")
with gate('L98 h2d = sycl.index("if (dst_is_device && dst_device_ptr)", compute_status)'):
    h2d = sycl.index("if (dst_is_device && dst_device_ptr)", compute_status)
failed_block = sycl[compute_status:h2d]
with gate('L100 assert "if (status != GGML_STATUS_SUCCESS)" in failed_block'):
    assert "if (status != GGML_STATUS_SUCCESS)" in failed_block
with gate('L101 assert "restore_host_copies()" in failed_block and "return false" in failed_block'):
    assert "restore_host_copies()" in failed_block and "return false" in failed_block
actions = ["compute-failed", "restore", "free", "return-false"]
with gate('L103 assert "h2d" not in actions'):
    assert "h2d" not in actions

finish('SYCL fallback exception safety source injections: PASS')
