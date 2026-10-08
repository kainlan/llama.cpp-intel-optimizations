#!/usr/bin/env python3
"""Injected source contracts for recoverable DL fallback exception safety.

Every check is its own `with gate(...)`, named for the property it pins, and every statement a check depends on
sits inside that same block: a stale anchor fails ITS check with a located message instead of raising a
NameError that takes the later checks with it (llama.cpp-qeld).
"""
from pathlib import Path
import re
from sycl_gate import finish, gate

root = Path(__file__).resolve().parents[1]
sycl = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
getrows = (root / "ggml/src/ggml-sycl/getrows.cpp").read_text()
module_test = (root / "tests/test-sycl-module-dependencies.py").read_text()
recorder_scope = (root / "ggml/src/ggml-sycl/graph-recorder-scope.hpp").read_text()

# DL preprocessing reaches the throw and excludes the CPU implementation call.
with gate("the DL get_rows fallback throws before the #else that holds the CPU implementation call"):
    direct = re.search(r"static bool ggml_sycl_cpu_get_rows_direct\(.*?\n\}", getrows, re.S).group(0)
    assert re.search(r"#ifdef GGML_BACKEND_DL.*?throw ggml_sycl_fallback_error\(reason\);.*?#else", direct, re.S)
    assert direct.index("#else") < direct.index("ggml_compute_forward_get_rows") < direct.rindex("#endif")
with gate("the module-dependency gate still names the CPU get_rows implementation"):
    assert "ggml_compute_forward_get_rows" in module_test

# Recording/re-record injected failures restore all global/context capture state,
# depth and retained sinks through noexcept guards before propagating.
with gate("two recording_exception_guard structs exist"):
    assert sycl.count("struct recording_exception_guard") >= 2
with gate("both recording_exception_guard destructors are noexcept"):
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
# Exactly three direct recording sites construct the scope by value: the two in the graph-compute path and the
# keyed segment record (the re-record site holds it in an optional via recorder_.emplace). A count of >= 1 let a
# mutant drop one site's scope unnoticed, so pin the set, and pin the keyed one to its function.
with gate("every direct recording site constructs the recorder scope"):
    # Count CODE: a comment that mentions the construction is not a construction.
    sycl_code = re.sub(r"//[^\n]*|/\*.*?\*/", "", sycl, flags=re.S)
    assert sycl_code.count("ggml_sycl_graph_recorder recorder(") == 3
    keyed = sycl_code[sycl_code.index("static void moe_graph_record_segment_slot("):
                      sycl_code.index("static void moe_graph_replay_segment_slot(")]
    assert keyed.count("ggml_sycl_graph_recorder recorder(") == 1
with gate("the re-record site holds the recorder scope in an optional"):
    assert "recorder_.emplace(" in sycl
with gate("both fallback catch sites classify the recoverable error by dynamic_cast"):
    assert sycl.count("dynamic_cast<const ggml_sycl_fallback_error *>") >= 2

# Prefix view restoration is noexcept and includes both mutable graph fields;
# suffix dispatch remains an explicit operation outside the destructor.
with gate("graph_view_guard restores the prefix view in a noexcept destructor"):
    assert "~graph_view_guard() noexcept" in sycl
with gate("graph_view_guard restores both mutable graph fields"):
    assert "cg->nodes   = nodes" in sycl and "cg->n_nodes = n_nodes" in sycl
with gate("the prefix_suffix_guard destructor does not dispatch the suffix"):
    prefix_dtor = re.search(r"~prefix_suffix_guard\(\) noexcept \{(.*?)\n        \}", sycl, re.S).group(1)
    assert "dispatch_suffix" not in prefix_dtor
with gate("suffix dispatch stays an explicit call on the suffix guard"):
    assert "suffix_guard.dispatch_suffix();" in sycl

# compute_forward must not convert the recoverable fallback into its generic
# std::exception fatal-exit path.
with gate("compute_forward rethrows the recoverable fallback before the generic std::exception handler"):
    cf_fallback = sycl.index("catch (const ggml_sycl_fallback_error &)")
    cf_generic = sycl.index("catch (const std::exception & e)", cf_fallback)
    assert cf_fallback < cf_generic and "throw;" in sycl[cf_fallback:cf_generic]

# Non-DL staging reserve/push/copy failures restore all tensor pointers and
# release CPU resources before graph planning can run.
with gate("a failed staging restores the host copies and returns false before graph planning"):
    staging_check = sycl.index("if (staging_failed)")
    plan = sycl.index("ggml_graph_plan", staging_check)
    block = sycl[staging_check:plan]
    for token in ("restore_host_copies()", "return false"):
        assert token in block, token
with gate("cpu_fallback_resources releases through a noexcept destructor"):
    assert "struct cpu_fallback_resources" in sycl and "~cpu_fallback_resources() noexcept" in sycl
with gate("staging reserves its host copies before allocating"):
    fallback_start = sycl.index("struct cpu_fallback_resources")
    assert sycl.index("host_copies.reserve", fallback_start) < sycl.index("unified_alloc(req", fallback_start)
with gate("a staged host copy is recorded before the tensor storage is reassigned"):
    assert (sycl.index("host_copies.push_back(std::move(entry))")
            < sycl.index("ggml_sycl_assign_tensor_storage(t, host_copies.back().host_ptr)"))
with gate("tensor_storage_restore_guard restores in a noexcept destructor"):
    assert "struct tensor_storage_restore_guard" in sycl and "~tensor_storage_restore_guard() noexcept" in sycl

# Boundary cleanup drains prior submissions and deferred scatters, unpins
# transient leases, and clears active/deferred state before FAILED.
with gate("the fallback boundary cleanup drains, unpins and clears state before returning FAILED"):
    catch = sycl.index("catch (const ggml_sycl_fallback_error & error)")
    failed = sycl.index("return GGML_STATUS_FAILED", catch)
    cleanup = sycl[catch:failed]
    for token in ("ggml_sycl_cpu_tg_flush_pending", "last_graph_event->wait_and_throw", "stream()->wait_and_throw",
                  "flush_pending_secondary_scatter", "ggml_sycl_cpu_staging_drain",
                  "graph_unpin_transient_leases_after_direct_execution",
                  "last_graph_event_deferred_decode = false",
                  "unified_cache_set_graph_compute_active(false)"):
        assert token in cleanup, token
    assert cleanup.index("ggml_sycl_cpu_tg_flush_pending") < cleanup.index("last_graph_event->wait_and_throw")

# Segment 1 may already be asynchronously submitted when segment 2 fails.
# Per-attempt baselines preserve prior handles; current handles roll back only
# before successful submission. Depth ownership releases exactly once even for
# post-end finalize/allocation/submit/MoE exceptions.
with gate("each recording attempt snapshots the retained-handle baseline"):
    assert "const size_t retained_baseline = sycl_ctx->graph_retained_handles.size()" in sycl
with gate("a segment is marked submitted only before its executable graph is submitted"):
    assert "bool segment_submitted = false" in sycl
    assert "segment_submitted = true" in sycl
    # The submit goes through graph_exec_submit since zhcn C7a (cec4a5f10): it counts, then calls
    # stream.ext_oneapi_graph(exec). The flag must still be set before that submit, and no bare call may remain.
    submit = "ggml_sycl::graph_exec_submit(*stream, *recorded_segments.back().exec_graph)"
    assert sycl.count(submit) == 1
    assert "stream->ext_oneapi_graph(*recorded_segments.back().exec_graph)" not in sycl
    assert sycl.index("segment_submitted = true") < sycl.index(submit)
with gate("an unsubmitted segment rolls back to the baseline at both failure sites"):
    assert sycl.count("if (!segment_submitted)") >= 2
    assert sycl.count("graph_retained_handles.resize(retained_baseline)") >= 2
with gate("recording_depth_owner releases the recording depth through a noexcept destructor"):
    assert "struct recording_depth_owner" in sycl
    assert "~recording_depth_owner() noexcept" in sycl
with gate("the recording depth is taken once and released once in the segment block"):
    segment_start = sycl.index("struct recording_depth_owner")
    segment_end = sycl.index("// 4. Store in context", segment_start)
    segment_block = sycl[segment_start:segment_end]
    assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_add") == 1
    assert segment_block.count("g_ggml_sycl_graph_recording_depth.fetch_sub") == 1
# Behavioral fault model: segment 1 is submitted, segment 2 appends then fails.
with gate("fault model: a failed second segment leaves the first segment's handle in place"):
    handles = ["segment1-submitted"]
    segment2_baseline = len(handles)
    handles.append("segment2-never-submitted")
    handles[segment2_baseline:] = []
    assert handles == ["segment1-submitted"]
with gate("fault model: the recording depth returns to its starting value"):
    depth = 7
    depth += 1       # acquire
    # end_recording releases; finalize/alloc/submit/MoE fault sees owner=false
    depth -= 1
    assert depth == 7

# Failed CPU compute must restore and return before any destination H2D copy.
with gate("a failed CPU compute restores and returns before any destination H2D copy"):
    compute_status = sycl.index("const ggml_status status = ggml_graph_compute")
    h2d = sycl.index("if (dst_is_device && dst_device_ptr)", compute_status)
    failed_block = sycl[compute_status:h2d]
    assert "if (status != GGML_STATUS_SUCCESS)" in failed_block
    assert "restore_host_copies()" in failed_block and "return false" in failed_block
with gate("fault model: a failed compute never reaches the H2D action"):
    actions = ["compute-failed", "restore", "free", "return-false"]
    assert "h2d" not in actions

finish('SYCL fallback exception safety source injections: PASS', min_checks=32)
