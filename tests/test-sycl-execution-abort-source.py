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


def slice_between(source, start_needle, end_needle):
    """Text strictly between the first occurrence of start_needle and the
    first occurrence of end_needle that follows it.

    Used instead of function_body() when the region under test is a handful
    of statements inside ggml_backend_sycl_graph_compute_impl() -- by far the
    largest function in this TU (CLAUDE.md), so extracting its WHOLE body via
    function_body()'s "next line that is a bare `}`" regex is both slow and
    fragile (any column-0 `}` anywhere in that function's thousands of lines
    would truncate the match early). Empty when either anchor is missing,
    which fails every clause that reads it -- same fail-closed contract as
    function_body().
    """
    start = source.find(start_needle)
    if start < 0:
        return ""
    start += len(start_needle)
    end = source.find(end_needle, start)
    if end < 0:
        return ""
    return source[start:end]


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

# llama.cpp-8q35 (TKV-15): a terminal-but-unreleased SIBLING context's device
# claim (own_drain_body above only ever drains this SAME ctx's own terminal --
# see "own-terminal drain admits only this root's terminal invocation") must
# also be drained, via the SAME primitive teardown already uses
# (ggml_sycl_execution_drain_context_terminal_events). This happens in
# ggml_backend_sycl_graph_compute_impl() -- the CALLER of begin_graph, not
# inside begin_graph itself: begin_graph's fresh branch holds
# g_execution_backend_binding_mutex (BINDING, see the H8 lock-class comment)
# for the rest of its body, and ggml_sycl_execution_drain_context_terminal_
# events() also takes that same mutex and performs a blocking queue wait --
# retrying from inside begin_graph's locked body would self-deadlock. Scoped
# to the handful of statements between begin_graph's first call and the
# state_lock block that turns a still-failed attempt into the thrown error,
# not begin_graph_body -- see slice_between()'s docstring for why.
device_owner_context_body = function_body(
    registry_cpp,
    r"error Registry::device_owner_context\s*\(\s*int\s+device\s*,\s*ContextId\s*\*\s*out\s*\)\s*const\s*noexcept\s*\{",
)
graph_compute_impl_retry_body = slice_between(
    backend,
    "bool execution_graph_active = ggml_sycl_execution_begin_graph(sycl_ctx, &execution_graph_error);",
    "std::lock_guard<std::mutex> state_lock(sycl_ctx->execution_state_mutex);",
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
    # --- llama.cpp-8q35 (TKV-15): foreign-terminal drain on DEVICE_BUSY ---
    # Lives in graph_compute_impl's retry slice, NOT begin_graph_body -- see
    # the comment above graph_compute_impl_retry_body's definition for why.
    "begin drains a terminal FOREIGN owner before refusing DEVICE_BUSY":
        "ggml_sycl_execution_drain_context_terminal_events(" in graph_compute_impl_retry_body,
    # The foreign-terminal drain must fire only on the contract-compliant
    # DEVICE_BUSY refusal, never on a registry defect (MISMATCH/STALE/
    # OVERFLOW) -- those must keep failing immediately, unmodified.
    "foreign drain only fires on DEVICE_BUSY, never on MISMATCH/STALE/OVERFLOW": ordered(
        graph_compute_impl_retry_body,
        "error::DEVICE_BUSY",
        "ggml_sycl_execution_drain_context_terminal_events(",
    ),
    # Only a TERMINAL foreign owner (COMPLETE/QUARANTINED) may be drained -- an
    # OPEN/SEALED owner is genuine concurrent use and must keep refusing hard,
    # per canonical contract §12.3 ("unsupported ... no optimistic overlap").
    # Ordered after the DEVICE_BUSY gate.
    "foreign drain admits only COMPLETE/QUARANTINED owners": ordered(
        graph_compute_impl_retry_body,
        "error::DEVICE_BUSY",
        "graph_phase::COMPLETE",
        "graph_phase::QUARANTINED",
        "ggml_sycl_execution_drain_context_terminal_events(",
    ),
    # Exactly one retry within the slice -- the slice starts AFTER the
    # original begin_graph call (consumed as the start_needle), so the retry
    # call is the only occurrence expected here. No loop.
    "foreign drain retries begin_graph exactly once":
        graph_compute_impl_retry_body.count(
            "ggml_sycl_execution_begin_graph(sycl_ctx, &execution_graph_error)"
        ) == 1,
    # The deadlock this design avoids (see the comment above
    # graph_compute_impl_retry_body): the retry slice must never ACQUIRE
    # begin_graph's own BINDING lock itself, since
    # ggml_sycl_execution_drain_context_terminal_events() already takes it
    # internally and performs a blocking wait under it. A lock ACQUISITION
    # pattern, not a bare mention -- the surrounding rationale comment names
    # this same mutex legitimately, so testing for the identifier's mere
    # presence would fail on the comment that explains why it's absent.
    "foreign drain never re-enters begin_graph's own binding lock":
        re.search(r"std::lock_guard<std::mutex>\s*\w*\s*\(\s*g_execution_backend_binding_mutex\s*\)",
                  graph_compute_impl_retry_body) is None,
    "registry device-owner accessor is declared":
        "error device_owner_context(int device, ContextId * out) const noexcept;" in registry_h,
    # Read-only: must not mint a new id (next_id) or write into device_owners_
    # -- it may only report the ContextId already recorded there.
    "registry device-owner accessor is read-only": bool(device_owner_context_body)
        and "next_id(" not in device_owner_context_body
        and re.search(r"device_owners_\[[^\]]*\]\s*=", device_owner_context_body) is None,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("execution abort source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("execution abort source contract: PASS")
