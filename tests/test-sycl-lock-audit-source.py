#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
sycl = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
mem = (root / "ggml/src/ggml-sycl/mem-handle.hpp").read_text()


def slice_between(src: str, start: str, end: str) -> str:
    i = src.find(start)
    if i < 0:
        raise SystemExit(f"missing start marker: {start}")
    j = src.find(end, i)
    if j < 0:
        raise SystemExit(f"missing end marker after {start}: {end}")
    return src[i:j]


graph_compute = slice_between(sycl, "static ggml_status ggml_backend_sycl_graph_compute", "template <typename F> static void ggml_backend_sycl_event_boundary")
sync_body = slice_between(sycl, "static void ggml_backend_sycl_synchronize", "// =============================================================================\n// Helper functions for RMS_NORM + MUL + MUL_MAT fusion")
event_boundary = slice_between(sycl, "template <typename F> static void ggml_backend_sycl_event_boundary", "static void ggml_backend_sycl_event_record")
device_event_sync = slice_between(sycl, "static void ggml_backend_sycl_device_event_synchronize", "static const ggml_backend_device_i ggml_backend_sycl_device_interface")
for_each = slice_between(sycl, "template<typename F>\nstatic void ggml_sycl_execution_for_each_bound_backend", "static void ggml_sycl_execution_unbind_backend")
extract_body = slice_between(sycl, "ggml_sycl_execution_result ggml_backend_sycl_execution_context_extract_control_host_allocs", "ggml_sycl_execution_result ggml_backend_sycl_execution_context_finish_drain")
claim_body = slice_between(sycl, "static bool pp_moe_onednn_claim_scratch_slot", "static void pp_moe_onednn_bind_scratch_slot_generation")
destructor_body = slice_between(sycl, "ggml_backend_sycl_context::~ggml_backend_sycl_context()", "// Release cached MoE ids buffers")
shutdown_body = slice_between(sycl, "void ggml_backend_sycl_shutdown(void)", "// backend reg")
cleanup_helper = slice_between(sycl, "static void ggml_backend_sycl_graph_boundary_exception_cleanup(ggml_backend_sycl_context * cleanup_ctx,\n                                                               const char *                stage,\n                                                               const char *                what) noexcept {", "static ggml_status ggml_backend_sycl_graph_compute")

checks = {
    "graph_compute catches std and unknown": "catch (const std::exception & exc)" in graph_compute and "catch (...)" in graph_compute,
    "graph_compute returns failed not escape": graph_compute.count("return GGML_STATUS_FAILED;") >= 3 and "std::exit(1)" not in graph_compute,
    "graph_compute cleanup quarantines and releases exact invocation": "ggml_sycl_execution_quarantine_graph(cleanup_ctx)" in sycl and "ggml_sycl_execution_release_graph(cleanup_ctx)" in sycl,
    "synchronize catches std and unknown": "catch (const std::exception & exc)" in sync_body and "catch (...)" in sync_body,
    "synchronize no c++ escape": "std::exit(1)" not in sync_body and "ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, \"synchronize\"" in sync_body,
    "event boundary logs not exits": "GGML_LOG_ERROR" in event_boundary and "std::exit(1)" not in event_boundary,
    "device event synchronize logs not exits": "GGML_LOG_ERROR" in device_event_sync and "std::exit(1)" not in device_event_sync,
    "for_each snapshots binding before callbacks": "ggml_sycl_execution_pin_bound_backends_locked(context_id)" in for_each and "ggml_sycl_execution_release_backend_pin(pin);" in for_each,
    "extract snapshots bound backends before swaps": "std::vector<ggml_sycl_execution_bound_backend_pin> bound_backends;" in extract_body and "binding_lock(g_execution_backend_binding_mutex)" in extract_body and "ggml_sycl_execution_pin_bound_backends_locked(ticket->context_id.value)" in extract_body,
    "extract swaps control owners outside binding lock": "for (size_t index = 0; index < bound_backends.size(); ++index)" in extract_body and "control_host_allocs.swap" in extract_body,
    "pp-moe claim snapshots event under lock": "wait_event      = state.done_events[wait_slot];" in claim_body and "wait_generation = state.generations[wait_slot];" in claim_body,
    "pp-moe claim waits outside lock and revalidates generation": "wait_event.wait_and_throw();" in claim_body and "state.generations[wait_slot] != wait_generation" in claim_body,
    "binding pins and draining exist": "uint32_t                pin_count = 0;" in sycl and "bool                    draining = false;" in sycl and "std::condition_variable cv;" in sycl,
    "for_each releases callback pins": "ggml_sycl_execution_release_backend_pin(pin);" in for_each,
    "abort invocation registry api used": "abort_invocation(" in sycl and "registry.abort_invocation" in sycl,
    "exception cleanup releases after waits": cleanup_helper.index("ggml_sycl_execution_quarantine_graph(cleanup_ctx)") < cleanup_helper.index("last_graph_event->wait_and_throw") < cleanup_helper.index("ggml_sycl_execution_release_graph(cleanup_ctx)"),
    "no detached pp-moe waiter": "std::thread([device, ring_depth, slot, generation" not in sycl and "pp_moe_onednn_drain_scratch_slots" in sycl,
    "pp-moe slot state retains owners": "std::vector<std::vector<ggml_sycl::mem_handle>> retained_owners;" in sycl and "pp_moe_onednn_scratch_claim_state.bind_owners" in sycl,
    "destructor drains pp-moe and aborts before unbind": "pp_moe_onednn_drain_scratch_slots(device);" in destructor_body and destructor_body.index("ggml_sycl_execution_abort_and_release_graph(this);") < destructor_body.index("ggml_sycl_execution_unbind_backend(this);"),
    "shutdown drains outstanding pp-moe slots": "pp_moe_onednn_drain_scratch_slots(device);" in shutdown_body,
    "retained handles swap out before destruction": "retired.swap(graph_retained_handles);" in sycl,
    "control_host_allocs swap out before destruction": "retired_control_host_allocs.swap(control_host_allocs);" in sycl,
    "H8 lock inventory comment exists": "H8 lock classes:" in sycl and "H8 lock class: MEM_HANDLE" in mem,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("lock audit source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("lock audit source contract: PASS")
