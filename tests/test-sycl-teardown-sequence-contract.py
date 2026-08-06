#!/usr/bin/env python3
"""Owner-targeted teardown contract (llama.cpp-o6jx, canonical §12.4/§12.8).

Host-only: reads sources, runs no build, loads no model and touches no device.

Each check below fails against the pre-o6jx tree, so a green result is evidence
that the migration is present rather than that the check could not fire. The
`positive controls` section proves every anchor the checks key off actually
exists, so a renamed symbol reports a missing anchor instead of passing
vacuously.
"""
from pathlib import Path
import re
import sys

root    = Path(__file__).resolve().parents[1]
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
public  = (root / "ggml/include/ggml-sycl.h").read_text()
context = (root / "src/llama-context.cpp").read_text()


def body_of(source, signature_prefix):
    """Text from a definition's signature to its first column-0 closing brace.

    `[^;]*?` before the opening brace is what skips a forward declaration: the
    declaration's terminating `;` cannot be crossed, so the match lands on the
    definition even when the prototype appears first.
    """
    match = re.search(re.escape(signature_prefix) + r"[^;]*?\)\s*(?:noexcept\s*)?(?:try\s*)?\{.*?^\}\n",
                      source, re.S | re.M)
    return match.group(0) if match else ""


drain_terminal_impl = body_of(backend, "static void ggml_sycl_execution_drain_context_terminal_events(uint64_t")
drain_terminal_abi  = body_of(backend, "ggml_sycl_execution_result ggml_backend_sycl_execution_context_drain_terminal_events(")
release_allocs_abi  = body_of(backend, "ggml_sycl_execution_result ggml_backend_sycl_execution_context_release_control_host_allocs(")
owned_by_impl       = body_of(backend, "static void ggml_sycl_execution_for_each_backend_owned_by(")
slot_release_impl   = body_of(backend, "static void ggml_sycl_release_model_slot_resources(")
owner_leases_impl   = body_of(backend, "static void ggml_sycl_release_graph_leases_for_owner(")
teardown_caller     = body_of(context, "static void llama_context_sycl_exec_drain_and_close(")
destructor          = body_of(context, "llama_context::~llama_context(")

# Positive controls: an empty body means the anchor moved or was renamed, which
# would make every check over it pass on absence.
anchors = {
    "ggml_sycl_execution_drain_context_terminal_events body": drain_terminal_impl,
    "drain_terminal_events ABI wrapper body": drain_terminal_abi,
    "release_control_host_allocs ABI wrapper body": release_allocs_abi,
    "ggml_sycl_execution_for_each_backend_owned_by body": owned_by_impl,
    "ggml_sycl_release_model_slot_resources body": slot_release_impl,
    "ggml_sycl_release_graph_leases_for_owner body": owner_leases_impl,
    "llama_context_sycl_exec_drain_and_close body": teardown_caller,
    "llama_context::~llama_context body": destructor,
}
missing_anchors = sorted(name for name, text in anchors.items() if not text)


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


checks = {
    # The exported sequence exists and is reachable through the DL proc table,
    # otherwise a GGML_BACKEND_DL build silently keeps the legacy close.
    "drain_terminal_events is declared in the public header":
        "ggml_backend_sycl_execution_context_drain_terminal_events(" in public,
    "release_control_host_allocs is declared in the public header":
        "ggml_backend_sycl_execution_context_release_control_host_allocs(" in public,
    "drain_terminal_events is resolvable via get_proc_address":
        '"ggml_backend_sycl_execution_context_drain_terminal_events"' in backend,
    "release_control_host_allocs is resolvable via get_proc_address":
        '"ggml_backend_sycl_execution_context_release_control_host_allocs"' in backend,

    # The terminal wait is owner-targeted and unlocked.
    "terminal wait is scoped to one context's bound backends":
        "ggml_sycl_execution_for_each_bound_backend(" in drain_terminal_impl,
    "terminal wait drains the queue before releasing the invocation":
        ordered(drain_terminal_impl, "wait_and_throw()", "ggml_sycl_execution_release_graph("),
    "an unprovable terminal quarantines instead of releasing":
        "ggml_sycl_execution_abort_and_release_graph(" in drain_terminal_impl,
    "the recorded exec graph is invalidated before control allocs are retired":
        "sycl_exec_graph_clear_active(" in drain_terminal_impl,
    "an unknown or closed context is a typed no-op, not a sweep":
        ordered(drain_terminal_abi, "global_registry().extract(", "return ggml_sycl_execution_c_result(rc);",
                "ggml_sycl_execution_drain_context_terminal_events("),

    # Final handle destruction happens outside every lock, before finish_drain.
    "release_control_host_allocs validates the exact drain ticket identity":
        all(field in release_allocs_abi for field in
            ("storage->context_id != ticket.context_id.value", "storage->serial != ticket.serial",
             "storage->session_id != ticket.session_id.value", "storage->reset_epoch != ticket.reset_epoch.value")),
    "release_control_host_allocs takes no lock around handle destruction":
        "lock_guard" not in release_allocs_abi and "unique_lock" not in release_allocs_abi,
    "release_control_host_allocs destroys the owning mem_handles":
        "alloc.owner = {};" in release_allocs_abi,

    # Model teardown is owner-targeted, never an all-device sweep.
    "model teardown releases graph leases for the dying owner":
        "ggml_sycl_release_graph_leases_for_owner(owner)" in slot_release_impl,
    "the retired all-device sweep TODO is gone":
        "deliberately not swept across all devices" not in backend,
    "owner matching compares the full ModelToken, not a bare slot":
        all(field in owned_by_impl for field in
            ("state.root_model_id == owner.model.value", "state.root_load_txn_id == owner.load.value",
             "state.root_slot == owner.owner.slot", "state.root_slot_generation == owner.owner.generation")),
    "owner iteration releases the binding lock before its callback":
        ordered(owned_by_impl, "std::lock_guard<std::mutex> lock(g_execution_backend_binding_mutex);",
                "}", "for (auto & pin : snapshot)", "fn(pin.backend);"),

    # The caller performs the mandated order.
    "teardown caller runs quiesce -> begin_drain -> wait -> extract -> release -> finish":
        ordered(teardown_caller, "hooks.drain_terminal(context)", "hooks.begin_drain(context, &ticket)",
                "hooks.drain_terminal(context)", "hooks.extract_allocs(&ticket, &batch)",
                "hooks.release_allocs(ticket, &batch)", "hooks.finish_drain(ticket, &batch)"),
    "teardown caller treats a repeat of an already-closed context as success":
        "GGML_SYCL_EXECUTION_STALE" in teardown_caller,
    "close_if_idle survives only as the older-DSO fallback":
        "has_drain_sequence()" in teardown_caller,
    "~llama_context uses the drain sequence, not a bare close_if_idle":
        "llama_context_sycl_exec_drain_and_close(" in destructor
        and "close_if_idle" not in destructor,
}

failed = sorted(name for name, ok in checks.items() if not ok)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

print("sycl teardown sequence contract: PASS ({} checks)".format(len(checks)))
