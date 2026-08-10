#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
registry_h = (root / "ggml/src/ggml-sycl/execution-lifecycle.hpp").read_text()
registry_cpp = (root / "ggml/src/ggml-sycl/execution-lifecycle.cpp").read_text()


def function_body(source, signature_regex):
    """Body of the one definition whose signature matches signature_regex.

    Ordering clauses below must not be scored against the whole file: a
    `wait_and_throw()` anywhere in a 100k-line translation unit would satisfy
    "waits before releasing" no matter what the function under test does. Returns
    "" when the definition is absent, which fails every clause that reads it.

    The anchor is a regex, not a literal, so the vertical-alignment padding this
    file uses (`ggml_backend_sycl_context *   ctx`) is not load-bearing. Pinning
    the exact run of spaces would make a routine reformat that collapses it read
    as a FAIL of the invariants below rather than as the formatting change it is.
    """
    match = re.search(signature_regex + r".*?^}\n", source, re.S | re.M)
    return match.group(0) if match else ""


def ordered(body, *needles):
    """True when every needle is present and appears in the order given."""
    at = -1
    for needle in needles:
        found = body.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def catch_block(body):
    """Body of the first `catch (...) {` block, up to its closing brace.

    Scoped deliberately: a clause that merely asserts `waited = false;` appears
    somewhere after `catch (...)` also passes when the assignment sits AFTER the
    handler, where it would run unconditionally. Only the statements inside the
    handler prove the unproven-wait path is the one that clears the flag.
    """
    start = body.find("catch (...) {")
    if start < 0:
        return ""
    end = body.find("\n    }", start)
    return body[start:end] if end > start else ""


# llama.cpp-2dgc (RCA c-5ozb): the u7vj tail guard submits on every early exit
# and deliberately does not release, so a caller that computes twice with no
# ggml_backend_sycl_synchronize() between reaches begin_graph with its OWN
# invocation COMPLETE-but-unreleased -- a state no existing path can clear, which
# refused the owner's next begin DEVICE_BUSY. The begin path is now the drain
# point for it.
own_drain_body = function_body(
    backend,
    r"static void ggml_sycl_execution_drain_own_terminal\s*\(\s*ggml_backend_sycl_context\s*\*\s*ctx\s*\)\s*noexcept\s*\{",
)
begin_graph_body = function_body(
    backend,
    r"static bool ggml_sycl_execution_begin_graph\s*\(\s*ggml_backend_sycl_context\s*\*\s*ctx\s*,",
)

checks = {
    "begin drains the owner's own terminal invocation": bool(own_drain_body)
        and "ggml_sycl_execution_drain_own_terminal(ctx);" in begin_graph_body,
    # The drain waits on a queue; both of begin_graph's mutexes are taken after
    # it, because a queue wait under either is a stall hazard.
    "own-terminal drain runs before begin takes any lock": ordered(
        begin_graph_body,
        "ggml_sycl_execution_drain_own_terminal(ctx);",
        "std::lock_guard<std::mutex> binding_lock(g_execution_backend_binding_mutex);",
        "std::lock_guard<std::mutex> state_lock(ctx->execution_state_mutex);",
    ),
    # Wait, then release, then retire -- the order
    # ggml_sycl_execution_drain_context_terminal_events() establishes. The wait is
    # what makes the release legal (M7_SUBMIT_RELEASES_DEVICES_EARLY models the
    # fault of releasing with kernels still in flight).
    "own-terminal drain waits before it releases": ordered(
        own_drain_body,
        "wait_and_throw();",
        "ggml_sycl_execution_release_graph(ctx)",
        "ggml_sycl_execution_try_retire_terminal(ctx);",
    ),
    # An unproven terminal must not be released: a throwing wait routes to the
    # quarantine path instead. The handler's assignment is the load-bearing part
    # and is checked inside the handler -- deleting just `waited = false;` leaves
    # the function releasing on an unproven terminal, which is the exact fault it
    # exists to prevent, and every other clause here still passes.
    "own-terminal drain marks an unproven wait inside the catch":
        "waited = false;" in catch_block(own_drain_body),
    "own-terminal drain quarantines on an unproven wait": ordered(
        own_drain_body,
        "catch (...)",
        "if (!waited || !ggml_sycl_execution_release_graph(ctx)) {",
        "ggml_sycl_execution_abort_and_release_graph(ctx);",
    ),
    # Only a terminal invocation is drainable, and only this root's own.
    "own-terminal drain admits only this root's terminal invocation": all(
        clause in own_drain_body
        for clause in (
            "graph_phase::COMPLETE",
            "graph_phase::QUARANTINED",
            "snapshot.token_root == owner",
        )
    ),
    # begin is per-backend; the whole-context sweep belongs to teardown only.
    "own-terminal drain is scoped to this context": bool(own_drain_body)
        and "ggml_sycl_execution_for_each_bound_backend" not in own_drain_body,
    "registry abort api declared": "error abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch," in registry_h,
    "registry abort api defined": "error Registry::abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch," in registry_cpp,
    "registry abort marks quarantined terminal without releasing": "graph.state = graph_phase::QUARANTINED;" in registry_cpp and "graph.pending_participant_count = 0;" in registry_cpp and "device_owners_[claimed_device] = {};" not in registry_cpp.split("error Registry::abort_invocation",1)[1].split("error Registry::retire_graph",1)[0],
    "binding drain pins exist": "pin_count = 0;" in backend and "draining = false;" in backend and "ggml_sycl_execution_release_backend_pin" in backend,
    "callback snapshot pins before deref": "ggml_sycl_execution_pin_bound_backends_locked(context_id)" in backend and "fn(pin.backend, *pin.binding);" in backend,
    "destructor drains callbacks before unbind": "binding->cv.wait(lock, [&] { return binding->pin_count == 0; });" in backend,
    "pp-moe detached waiter removed": "std::thread([device, ring_depth, slot, generation" not in backend,
    "pp-moe shutdown drain exists": "pp_moe_onednn_drain_scratch_slots(device);" in backend,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("execution abort source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("execution abort source contract: PASS")
