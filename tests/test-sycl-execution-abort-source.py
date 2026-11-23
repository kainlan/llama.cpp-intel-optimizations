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
    # The whole statement is pinned. A refused release gets one recovery attempt
    # before the abort, but an unproven wait still goes straight to the abort:
    # `!waited ||` comes first, so neither the release nor the recovery runs on it.
    "own-terminal drain quarantines on an unproven wait": ordered(
        own_drain_body,
        "catch (...)",
        "if (!waited || (!ggml_sycl_execution_release_graph(ctx) && !ggml_sycl_execution_recover_and_release_graph(ctx))) {",
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
    # Named "graph_compute_impl drains ..." (not "begin drains ...") on
    # purpose: this is graph_compute_impl's own drain-and-retry, and the
    # subject of every clause name below says so explicitly -- begin_graph is
    # only ever the function BEING retried, never the actor doing the
    # draining. Spec review flagged an earlier draft's ambiguous "begin ..."
    # phrasing as readable the other way around.
    #
    # The needle is the call WITH its argument
    # ("...(owner_ctx.value)"), not the bare function name. The bare name
    # alone also appears in the rationale comment above graph_compute_impl_
    # retry_body's own definition ("...drain_context_terminal_events() also
    # takes [the lock]"), so a needle without the argument self-matches that
    # comment and would still report PASS with the real call deleted. Spec
    # review caught this; verified by mutation (delete the call, keep the
    # comment -- the old needle stayed green, this one goes red).
    "graph_compute_impl drains a terminal FOREIGN owner before refusing DEVICE_BUSY":
        "ggml_sycl_execution_drain_context_terminal_events(owner_ctx.value)" in graph_compute_impl_retry_body,
    # The foreign-terminal drain must fire only on the contract-compliant
    # DEVICE_BUSY refusal, never on a registry defect (MISMATCH/STALE/
    # OVERFLOW) -- those must keep failing immediately, unmodified.
    "foreign drain only fires on DEVICE_BUSY, never on MISMATCH/STALE/OVERFLOW": ordered(
        graph_compute_impl_retry_body,
        "error::DEVICE_BUSY",
        "ggml_sycl_execution_drain_context_terminal_events(owner_ctx.value)",
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
        "ggml_sycl_execution_drain_context_terminal_events(owner_ctx.value)",
    ),
    # PASS 1 (the validation walk) must never drain anything -- that is the
    # whole point of the two-pass restructure: nothing is touched until every
    # contended device is proven drainable. Delta review found this had NO
    # regression gate: reverting to the old single-pass shape (drain inside
    # the validation loop, before checking the next device) left every
    # existing clause green, because they only assert the drain call exists
    # SOMEWHERE in the retry slice, not that it is absent from pass 1.
    # `bool(pass1_body) and ...`, not just `... not in pass1_body`, on
    # purpose: slice_between() returns "" when an anchor is missing (e.g. a
    # revert that also deletes the PASS 1/2 comments), and "x not in ''' is
    # vacuously True -- exactly the empty-probe trap this repo has hit
    # before. An empty slice must fail this clause, not pass it.
    "PASS 1 (validation) never drains before PASS 2 confirms all_drainable": bool(
        (lambda pass1_body: pass1_body and
            "ggml_sycl_execution_drain_context_terminal_events(owner_ctx.value)" not in pass1_body)(
            slice_between(
                graph_compute_impl_retry_body,
                "// PASS 1: validate every contended device's current owner before",
                "// PASS 2: only now, with every contended device proven drainable,",
            )
        )
    ),
    # Exactly one retry within the slice -- the slice starts AFTER the
    # original begin_graph call (consumed as the start_needle), so the retry
    # call is the only occurrence expected here.
    "graph_compute_impl retries begin_graph exactly once":
        graph_compute_impl_retry_body.count(
            "ggml_sycl_execution_begin_graph(sycl_ctx, &execution_graph_error)"
        ) == 1,
    # count==1 alone passes on a retry LOOP that wraps the same call text --
    # e.g. `for (...) { ...begin_graph(...); }` still contains the literal
    # exactly once even though it can execute any number of times. Spec
    # review caught this; verified by mutation (wrap the retry call in a
    # `for` loop -- the count==1 check alone stayed green, this one goes red
    # because "for (" now appears between the drain and the retry). Slicing
    # from the drain call to the retry call and requiring no loop keyword in
    # between catches it: in the real fix the retry sits after the drain
    # loop has already closed, so nothing between them opens a new one.
    "graph_compute_impl's retry is not wrapped in a loop": bool(
        (lambda between: between and "for (" not in between and "while (" not in between)(
            slice_between(
                graph_compute_impl_retry_body,
                "ggml_sycl_execution_drain_context_terminal_events(owner_ctx.value);",
                "ggml_sycl_execution_begin_graph(sycl_ctx, &execution_graph_error)",
            )
        )
    ),
    # The deadlock this design avoids (see the comment above
    # graph_compute_impl_retry_body): graph_compute_impl's retry must never
    # ACQUIRE begin_graph's own BINDING lock itself, since
    # ggml_sycl_execution_drain_context_terminal_events() already takes it
    # internally and performs a blocking wait under it. A lock ACQUISITION
    # pattern, not a bare mention -- the surrounding rationale comment names
    # this same mutex legitimately, so testing for the identifier's mere
    # presence would fail on the comment that explains why it's absent.
    "graph_compute_impl's retry never re-enters begin_graph's own binding lock":
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


# --- Every node dispatch of graph_compute runs inside an open tracked invocation ---
#
# compute_impl begins the context's tracked graph. The keyed and whole-split segment record and replay and the block
# graphlets dispatch nodes without compute_impl, so before they opened the same scope an op there found the graph a
# synchronize had already released: the Q1_0/NVFP4 direct route declined with failed=graph_epoch,invocation_id on
# the first decode split that took a keyed slot. The census below starts from graph_compute_unchecked's body, follows
# every top-level function it calls down to a node dispatch, and requires that function to declare the dispatch scope
# in its outermost block, so the scope lives to the function's end, with every dispatch after it and only the
# statements listed for that function in PRE_SCOPE_CALLS before it.
#
# A node dispatch is a call of compute_forward, compute_forward_impl or the direct-node helper, or of an op entry point
# compute_forward_impl's op switch hands the context to, so a path that calls an op directly is counted too.
DISPATCH_ANCHORS = ("ggml_sycl_compute_forward", "ggml_sycl_compute_forward_impl", "moe_graph_dispatch_direct_node")
DISPATCH_OP_SWITCH = re.compile(r"\bswitch\s*\(\s*dst->op\s*\)\s*\{")
DISPATCH_OP_CALL = re.compile(r"\b(ggml_sycl_\w+)\s*\(\s*ctx\b")
CALL = re.compile(r"\b(\w+)\s*\(")
DISPATCH_SCOPE_OPEN = re.compile(r"\bggml_sycl_execution_dispatch_scope\s+\w+\s*\(\s*\w+\s*\)\s*;")
DISPATCH_ROOT = "ggml_backend_sycl_graph_compute_unchecked"
DISPATCH_DEPTH = 3
COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)


def blank_comments_and_literals(source):
    """The source with comments and literals turned to spaces (newlines kept), so a name in prose or in a format
    string is not a call and every offset still points at the same place."""
    return COMMENT_OR_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), source)


def matching_close(text, open_at, opening="{", closing="}"):
    depth = 0
    for at in range(open_at, len(text)):
        if text[at] == opening:
            depth += 1
        elif text[at] == closing:
            depth -= 1
            if depth == 0:
                return at
    return -1


def function_bodies(blank):
    """name -> body of every column-0 `static` function definition in the blanked source, function-try-blocks
    included (compute_forward_impl is one)."""
    bodies = {}
    for m in re.finditer(r"^static\b[^;{}()]*?\b(\w+)\s*\(", blank, re.M):
        params_close = matching_close(blank, m.end() - 1, "(", ")")
        if params_close < 0:
            continue
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?(?:try\s*)?\{", blank[params_close + 1:])
        if not tail:
            continue
        open_at = params_close + 1 + tail.end() - 1
        close_at = matching_close(blank, open_at)
        if close_at > open_at:
            bodies.setdefault(m.group(1), blank[open_at:close_at + 1])
    return bodies


def dispatch_names(bodies):
    """The calls that dispatch a node: the anchors and every op entry point the op switch of compute_forward_impl
    hands the context to. None when the switch is gone, so the census cannot shrink to the anchors unnoticed."""
    impl = bodies.get("ggml_sycl_compute_forward_impl", "")
    switch = DISPATCH_OP_SWITCH.search(impl)
    if not switch:
        return None
    cases = impl[switch.end() - 1:matching_close(impl, switch.end() - 1) + 1]
    return set(DISPATCH_ANCHORS) | {call.group(1) for call in DISPATCH_OP_CALL.finditer(cases)}


def dispatch_scope_violations(source):
    """The functions graph_compute_unchecked calls that reach a node dispatch outside a dispatch scope, each with the
    reason: no scope, a scope in a nested block, a dispatch before the scope, or an unlisted call before it."""
    bodies = function_bodies(blank_comments_and_literals(source))
    root = bodies.get(DISPATCH_ROOT)
    if not root:
        return [DISPATCH_ROOT + " not found"]
    dispatch = dispatch_names(bodies)
    if dispatch is None or not {"ggml_sycl_mul_mat", "ggml_sycl_mul_mat_id"} <= dispatch:
        return ["the op switch of ggml_sycl_compute_forward_impl yields no matmul entry point"]
    memo = {}

    def reaches(name, depth):
        """Whether `name`'s body calls a node dispatch within `depth` further levels."""
        key = (name, depth)
        if key not in memo:
            memo[key] = False  # a recursive cycle reaches nothing new
            memo[key] = any(call.group(1) in dispatch or (depth > 0 and call.group(1) != name
                                                          and call.group(1) in bodies
                                                          and reaches(call.group(1), depth - 1))
                            for call in CALL.finditer(bodies[name]))
        return memo[key]

    violations = set()
    callees = {call.group(1) for call in CALL.finditer(root)}
    for callee in sorted(callees):
        if callee == DISPATCH_ROOT or callee not in bodies:
            continue
        if callee in DISPATCH_EXEMPT:
            if not reaches(callee, DISPATCH_DEPTH):
                violations.add(callee + ": exempt but reaches no node dispatch")
            continue
        body = bodies[callee]
        sites = [call for call in CALL.finditer(body)
                 if call.group(1) in dispatch or (call.group(1) != callee and call.group(1) in bodies
                                                  and reaches(call.group(1), DISPATCH_DEPTH - 1))]
        if not sites:
            continue
        scope = DISPATCH_SCOPE_OPEN.search(body)
        if not scope:
            violations.add(callee + ": no dispatch scope")
            continue
        if body.count("{", 0, scope.start()) - body.count("}", 0, scope.start()) != 1:
            violations.add(callee + ": the dispatch scope is declared in a nested block")
        if sites[0].start() < scope.start():
            violations.add(callee + ": " + sites[0].group(1) + " dispatches before the scope")
        listed = PRE_SCOPE_CALLS.get(callee, frozenset())
        before = {call.group(1) for call in CALL.finditer(body, 0, scope.start()) if call.group(1) in bodies}
        for name in sorted(before - listed):
            violations.add(callee + ": " + name + " runs before the scope and is not listed")
        for name in sorted(listed - before):
            violations.add(callee + ": listed pre-scope call " + name + " no longer precedes the scope")
    return sorted(violations)


def dispatchers_found(source):
    """The functions the census reaches, so a census that reaches nothing cannot pass vacuously."""
    bodies = function_bodies(blank_comments_and_literals(source))
    return {name for name in bodies if DISPATCH_SCOPE_OPEN.search(bodies[name])}


# The paths that dispatch without compute_impl, each of which must open the scope itself.
DISPATCH_SCOPE_OWNERS = {
    "ggml_backend_sycl_graph_compute_impl",
    "moe_graph_record_segment_slot",
    "moe_graph_replay_segment_slot",
    "moe_graph_record_segments",
    "moe_graph_replay_segments",
    "moe_graph_try_block_graphlets",
}

# The calls into this file's functions that may run before a path's scope. None of them reaches a node; a new one
# must be added here on purpose. A path not named here may run none.
PRE_SCOPE_CALLS = {
    # The per-graph precomputed-skip reset, before the compute guard is taken.
    "ggml_backend_sycl_graph_compute_impl": frozenset({"ggml_sycl_moe_precomputed_skip_new_graph"}),
    "moe_graph_try_block_graphlets": frozenset({
        # The aggregation decision and its diagnostics, which may still decline the graphlets.
        "ggml_sycl_graph_diag_enabled",
        "ggml_sycl_moe_aggregation_diag",
        "moe_aggregation_selected_decision",
        "moe_block_graphlet_requested_size",
        "moe_default_fast_path_policy_for_decode",
        # The dispatch identities the recorded graphlets are matched against.
        "moe_graph_collect_dispatch_indices",
        "moe_graph_descriptor_moe_dispatch_candidate_count",
        "moe_graph_sequence_dispatch_identity_signature",
        "moe_sequence_graphlet_mode_hash",
        # The per-graph cache resets compute_impl's flush would do.
        "ggml_sycl_moe_ids_cache_new_graph",
        "moe_shared_act_new_graph",
        # The input prestage, which may decline the graph, and the input refresh.
        "graph_prestage_or_decline",
        "graph_refresh_input_tensors",
    }),
}

# graph_compute_unchecked's callees that call an op entry point and open no scope, with the reason.
DISPATCH_EXEMPT = {
    name: "the opt-in persistent TG path (GGML_SYCL_PERSISTENT_TG, off by default) runs untracked as a whole, outside "
          "any dispatch scope: the plan builder's GET_ROWS pre-execution on every persistent token, the fused kernel, "
          "and the debug references under GGML_SYCL_PERSISTENT_TG_USE_BASE_* and the SET_ROWS validator. "
          "should_use_persistent_tg declines every MoE graph, so the MoE direct route never runs there."
    for name in ("extract_persistent_plan", "extract_persistent_plan_split", "debug_validate_persistent_set_rows")
}

# A scope must live from before the first node to the function's end: one declared in a block that closes first, or a
# temporary, leaves the nodes after it outside any invocation.
REPLAY_SLOT_SCOPE = (
    "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n"
    "    g_graph_diag_counters.seg_replay.fetch_add(1, std::memory_order_relaxed);\n"
    "    moe_graph_segmented_call_begin(sycl_ctx, cgraph);\n")
RECORD_SLOT_SCOPE = (
    "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n"
    "    g_graph_diag_counters.seg_record_attempts.fetch_add(1, std::memory_order_relaxed);\n"
    "    moe_graph_segmented_call_begin(sycl_ctx, cgraph);\n")
SCOPE_DECL = "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n"
GRAPHLETS_REFRESH = (
    "    graph_refresh_input_tensors(sycl_ctx, cgraph);\n"
    "    // Replay and record both run nodes here, outside compute_impl.\n")
UNCHECKED_OPEN = (
    "static ggml_status ggml_backend_sycl_graph_compute_unchecked(ggml_backend_t backend, ggml_cgraph * cgraph) {")
REPLAY_SEGMENTS_OPEN = (
    "static void moe_graph_replay_segments(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {\n"
    + SCOPE_DECL)
GRAPHLETS_SCOPE = SCOPE_DECL + "\n    try {\n        if (moe_graph_block_graphs_match"
RECORD_SLOT_ATTEMPTS = "    g_graph_diag_counters.seg_record_attempts.fetch_add(1, std::memory_order_relaxed);\n"
KEYED_PLAN_LOOP = "    for (const moe_graph_keyed_item & item : plan) {\n"

dispatch_violations = dispatch_scope_violations(backend)
dispatch_mutants = {
    # A bypass path that does not open the scope.
    "replay-slot-unscoped": backend.replace(REPLAY_SLOT_SCOPE, REPLAY_SLOT_SCOPE.replace(SCOPE_DECL, ""), 1),
    # The block graphlets reach a dispatch two calls down; their scope is gone.
    "block-graphlets-unscoped": backend.replace(GRAPHLETS_SCOPE, GRAPHLETS_SCOPE.replace(SCOPE_DECL, ""), 1),
    # The scope opens after the first node already ran.
    "record-slot-scoped-late": backend.replace(
        SCOPE_DECL + RECORD_SLOT_ATTEMPTS, RECORD_SLOT_ATTEMPTS, 1).replace(
        KEYED_PLAN_LOOP + "        if (item.boundary) {",
        KEYED_PLAN_LOOP + "    " + SCOPE_DECL + "        if (item.boundary) {", 1),
    # A new dispatcher graph_compute calls, with no scope.
    "new-unscoped-dispatcher": backend.replace(
        UNCHECKED_OPEN,
        "static void moe_graph_unscoped_probe(ggml_backend_sycl_context * c, ggml_tensor * t) {\n"
        "    ggml_sycl_compute_forward(*c, t);\n"
        "}\n" + UNCHECKED_OPEN + "\n    moe_graph_unscoped_probe(nullptr, nullptr);", 1),
}

dispatch_mutants.update({
    # The scope closes with the block it was declared in, before the slot's first node.
    "scope-in-dead-block": backend.replace(
        REPLAY_SLOT_SCOPE,
        REPLAY_SLOT_SCOPE.replace(SCOPE_DECL,
                                  "    { ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx); }\n"),
        1),
    # A temporary scope ends at its own semicolon.
    "scope-temporary": backend.replace(
        REPLAY_SLOT_SCOPE,
        REPLAY_SLOT_SCOPE.replace(SCOPE_DECL, "    (void) ggml_sycl_execution_dispatch_scope(sycl_ctx);\n"), 1),
    # The first node runs inside a scoped block; every node after the block runs outside it.
    "dispatch-after-scope-block": backend.replace(
        RECORD_SLOT_SCOPE,
        RECORD_SLOT_SCOPE.replace(
            SCOPE_DECL,
            "    {\n"
            "        ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n"
            "        ggml_sycl_compute_forward(*sycl_ctx, cgraph->nodes[0]);\n"
            "    }\n"), 1),
    # A new path that calls an op entry point directly, not through compute_forward.
    "direct-op-call": backend.replace(
        UNCHECKED_OPEN,
        "static void moe_graph_unscoped_probe(ggml_backend_sycl_context * c, ggml_tensor * t) {\n"
        "    ggml_sycl_mul_mat_id(*c, t);\n"
        "}\n" + UNCHECKED_OPEN + "\n    moe_graph_unscoped_probe(nullptr, nullptr);", 1),
    # A call before the graphlets' scope that is not on their list of pre-scope statements.
    "unlisted-call-before-scope": backend.replace(
        GRAPHLETS_REFRESH,
        "    graph_refresh_input_tensors(sycl_ctx, cgraph);\n"
        "    ggml_sycl_moe_precomputed_skip_new_graph();\n"
        "    // Replay and record both run nodes here, outside compute_impl.\n", 1),
    # A node runs before a top-level scope.
    "dispatch-before-scope": backend.replace(
        REPLAY_SEGMENTS_OPEN,
        REPLAY_SEGMENTS_OPEN.replace(SCOPE_DECL,
                                     "    ggml_sycl_compute_forward(*sycl_ctx, cgraph->nodes[0]);\n" + SCOPE_DECL),
        1),
    # A listed pre-scope call that moved after the scope leaves the list wider than the code.
    "listed-call-moved-after-scope": backend.replace(
        GRAPHLETS_REFRESH + SCOPE_DECL,
        "    // Replay and record both run nodes here, outside compute_impl.\n" + SCOPE_DECL
        + "    graph_refresh_input_tensors(sycl_ctx, cgraph);\n", 1),
    # An exempt function that no longer calls an op entry point keeps an exemption it does not need.
    "exemption-outlives-its-op-call": backend.replace(
        "    ggml_sycl_op_set_rows(ctx, ggml_sycl::sycl_tensor(&dst, ctx.device));\n", "", 1),
})

checks.update({
    "every graph_compute path that dispatches nodes opens the tracked invocation first": not dispatch_violations,
    "the dispatch scope is opened by compute_impl and by each path that bypasses it":
        DISPATCH_SCOPE_OWNERS <= dispatchers_found(backend),
    **{
        f"dispatch-scope census witnesses mutation {name}": mutated != backend
        and bool(dispatch_scope_violations(mutated))
        for name, mutated in dispatch_mutants.items()
    },
})
if dispatch_violations:
    print("dispatch without an open tracked invocation in: " + ", ".join(dispatch_violations), file=sys.stderr)


# --- This backend's own terminal is drained before graph_compute takes its locks ---
#
# A compute with no synchronize after it leaves its invocation submitted and unreleased, and the next begin drains it:
# a wait on the device. Every path that begins a graph runs under g_sycl_graph_compute_mutex (L2) and the context's
# graph_mutex (L3), and no wait may sit under a listed lock (canonical contract section 12.5). So
# graph_compute_unchecked drains its own terminal before it takes L2, and the begin under the locks finds nothing.
OWN_DRAIN_STMT = ("    ggml_sycl_execution_drain_own_terminal("
                  "static_cast<ggml_backend_sycl_context *>(backend->context));\n")
L2_TAKE_STMT = "    std::unique_lock<std::mutex> global_graph_lock(g_sycl_graph_compute_mutex);\n"
L3_TAKE_STMT = "    std::lock_guard<std::mutex> graph_lock(sycl_ctx->graph_mutex);\n"


def own_drain_violations(source):
    root = function_bodies(blank_comments_and_literals(source)).get(DISPATCH_ROOT, "")
    l2, l3 = L2_TAKE_STMT.strip(), L3_TAKE_STMT.strip()
    if root.count(l2) != 1 or root.count(l3) != 1:
        return ["graph_compute_unchecked takes L2 and L3 once each"]
    if root.count("ggml_sycl_execution_drain_own_terminal(") != 1 or \
            not ordered(root, OWN_DRAIN_STMT.strip(), l2, l3):
        return ["graph_compute_unchecked drains this backend's own terminal before it takes L2"]
    return []


own_drain_found = own_drain_violations(backend)
own_drain_mutants = {
    # The drain moved under L2, where the wait it may make is forbidden.
    "own-drain-under-l2": backend.replace(OWN_DRAIN_STMT + L2_TAKE_STMT, L2_TAKE_STMT + OWN_DRAIN_STMT, 1),
}
checks.update({
    "graph_compute drains its own terminal before it takes g_sycl_graph_compute_mutex": not own_drain_found,
    **{
        f"own-drain contract witnesses mutation {name}": mutated != backend and bool(own_drain_violations(mutated))
        for name, mutated in own_drain_mutants.items()
    },
})
if own_drain_found:
    print("own-drain contract: " + "; ".join(own_drain_found), file=sys.stderr)


# --- The tripwire: ending or clearing a graph a dispatch scope holds open is reported, by site ---
#
# Each lifecycle verb that ends the invocation (submit, release, quarantine) or zeroes the backend's copy of it (the
# clear, the binding reset) asks the tripwire first, with its caller's site; the scope counts itself out before its
# own seal or quarantine. The tripwire decides under the state lock and the line is logged after every lock of the
# function is released: a log call runs the sink a user installed (the H8 rule). The binding reset hands its report
# back, because one caller resets under g_execution_backend_binding_mutex.
LOCK_DECL = re.compile(r"\bstd::(?:lock_guard|unique_lock|scoped_lock)\s*<[^>]*>\s*\w+\s*\(")
TRIPWIRE_LOG_CALL = re.compile(r"\b(?:ggml_sycl_execution_tripwire_log|GGML_LOG_\w+)\s*\(")


def enclosing_open(text, at):
    """Offset of the `{` that opens the innermost block containing `at`."""
    depth = 0
    for i in range(at - 1, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            if depth == 0:
                return i
            depth -= 1
    return -1


def tripwire_log_lock_violations(source):
    found = []
    blank = blank_comments_and_literals(source)
    bodies = function_bodies(blank)
    for name, body in sorted(bodies.items()):
        if "ggml_sycl_execution_tripwire_log(" not in body:
            continue
        held = []
        for lock in LOCK_DECL.finditer(body):
            block = enclosing_open(body, lock.start())
            held.append((lock.start(), matching_close(body, block)))
        for call in TRIPWIRE_LOG_CALL.finditer(body):
            if any(start < call.start() < end for start, end in held):
                found.append(name + " logs the tripwire under a lock")
                break
    for call in re.finditer(r"\bggml_sycl_execution_reset_backend_binding_state\s*\(", blank):
        before = blank[:call.start()].rstrip()
        if not (before.endswith("ggml_sycl_execution_tripwire_log(") or before.endswith("=")
                or re.search(r"static ggml_sycl_execution_tripwire_report$", before)):
            found.append("a binding reset drops the tripwire's report")
    return found


def tripwire_violations(source):
    found = []

    def body(name):
        return function_body(source, r"static \w+ " + name + r"\s*\(")

    check = body("ggml_sycl_execution_tripwire_check_locked")
    if not (re.search(r"execution_dispatch_depth == 0 \|\|", check)
            and "execution_tripwire_invocation = ctx->execution_invocation_id;" in check and "GGML_LOG" not in check
            and "site=%s:%d" in body("ggml_sycl_execution_tripwire_log")):
        found.append("tripwire reports once per invocation, only while a scope is open, with the site")
    for name, action, before in (
        ("ggml_sycl_execution_clear_graph_tracking", '"clearing"', "ctx->execution_graph_epoch "),
        ("ggml_sycl_execution_reset_backend_binding_state", '"unbinding"', "ctx->execution_context_id "),
    ):
        asked = "ggml_sycl_execution_tripwire_check_locked(ctx, " + action + ", site, line);"
        if not ordered(body(name), asked, before):
            found.append(name + " asks the tripwire before it zeroes")
    found.extend(tripwire_log_lock_violations(source))
    for name, action, before in (
        ("ggml_sycl_execution_submit_graph", '"submitting"', "registry.submit_invocation("),
        ("ggml_sycl_execution_release_graph", '"releasing"', "registry.release_invocation("),
        ("ggml_sycl_execution_quarantine_graph", '"quarantining"', "registry.abort_invocation("),
    ):
        if not ordered(body(name), "ggml_sycl_execution_tripwire(ctx, " + action + ", site, line);", before):
            found.append(name + " asks the tripwire before it ends the invocation")
    if not ordered(body("ggml_sycl_execution_abort_and_release_graph"),
                   "ggml_sycl_execution_quarantine_graph(ctx, site, line);",
                   "ggml_sycl_execution_release_graph(ctx, site, line);",
                   "ggml_sycl_execution_clear_graph_tracking(ctx, site, line);"):
        found.append("abort_and_release passes its caller's site on")
    if not ordered(body("ggml_sycl_execution_release_graph"),
                   "ggml_sycl_execution_clear_graph_tracking(backend_ctx, site, line);",
                   "ggml_sycl_execution_try_retire_terminal(ctx, site, line);"):
        found.append("release passes its caller's site to the clear and the retire")
    scope = slice_between(source, "struct ggml_sycl_execution_dispatch_scope {", "\n};\n")
    if not ordered(scope, "if (active) {", "++ctx->execution_dispatch_depth;", "~ggml_sycl_execution_dispatch_scope()",
                   "--ctx->execution_dispatch_depth", "ggml_sycl_execution_quarantine_graph(ctx);",
                   "ggml_sycl_execution_seal_graph(ctx);"):
        found.append("the scope counts itself in after a begin and out before its own seal or quarantine")
    return found


# The destructor's two pieces, as the source writes them: counting itself out (and returning when an outer scope still
# holds the graph), then ending the graph. The mutants below are built from them.
SCOPE_COUNT_OUT = ("        bool outermost = false;\n"
                   "        {\n"
                   "            std::lock_guard<std::mutex> state_lock(ctx->execution_state_mutex);\n"
                   "            outermost = --ctx->execution_dispatch_depth == 0;\n"
                   "        }\n"
                   "        if (!outermost) {\n"
                   "            return;\n"
                   "        }\n")
SCOPE_END = ("        if (std::uncaught_exceptions() > uncaught) {\n"
             "            ggml_sycl_execution_quarantine_graph(ctx);\n"
             "        } else {\n"
             "            ggml_sycl_execution_seal_graph(ctx);\n"
             "        }\n")
GRAPH_ENDERS = re.compile(r"\bggml_sycl_execution_(?:seal_graph|quarantine_graph|release_graph|submit_graph|"
                          r"abort_and_release_graph|clear_graph_tracking|try_retire_terminal|"
                          r"reset_backend_binding_state)\s*\(")


# --- The outermost dispatch scope owns the end of the tracked graph ---
#
# Scopes can nest: a scoped path that falls back into compute_impl opens a second one on the same backend. Only the
# scope that lowers the depth to zero seals or quarantines. An inner scope leaves the graph to the scope that opened
# it, on a normal exit and on an unwind alike; the exception reaches that scope's destructor by unwinding. An inner
# scope that ended the graph would do it while the outer one still counts, and the tripwire would report a clean
# unwind as something ending the graph under a dispatch.
def scope_end_violations(source):
    blank = blank_comments_and_literals(source)
    at = blank.find("~ggml_sycl_execution_dispatch_scope() {")
    if at < 0:
        return ["the dispatch scope's destructor was not found"]
    open_at = blank.find("{", at)
    dtor = " ".join(blank[open_at:matching_close(blank, open_at) + 1].split())
    found = []
    lowered = dtor.find("outermost = --ctx->execution_dispatch_depth == 0;")
    if lowered < 0 or dtor.count("execution_dispatch_depth") != 1 or not ordered(
            dtor, "std::lock_guard<std::mutex> state_lock(ctx->execution_state_mutex);",
            "outermost = --ctx->execution_dispatch_depth == 0;"):
        found.append("the destructor lowers the depth and tests it for zero in one expression, under the state lock")
    guard = dtor.find("if (!outermost) { return; }")
    enders = [m.start() for m in GRAPH_ENDERS.finditer(dtor)]
    if guard < 0 or guard < lowered or any(ender < guard for ender in enders):
        found.append("an inner scope returns before anything in the destructor seals, quarantines or ends the graph")
    if not enders:
        found.append("the outermost scope ends the graph")
    return found


tripwire_found = tripwire_violations(backend)
tripwire_mutants = {
    "clear-unwired": backend.replace(
        '        report = ggml_sycl_execution_tripwire_check_locked(ctx, "clearing", site, line);\n', "", 1),
    # The line logged with the state lock still held.
    "tripwire-logs-under-state-lock": backend.replace(
        "        report = ggml_sycl_execution_tripwire_check_locked(ctx, action, site, line);\n"
        "    }\n    ggml_sycl_execution_tripwire_log(report);\n",
        "        report = ggml_sycl_execution_tripwire_check_locked(ctx, action, site, line);\n"
        "        ggml_sycl_execution_tripwire_log(report);\n    }\n", 1),
    # A binding reset's line logged inside the binding lock's loop.
    "binding-reset-logs-under-binding-lock": backend.replace(
        "                if (report.action && n_tripped < GGML_SYCL_MAX_DEVICES) {\n",
        "                ggml_sycl_execution_tripwire_log(report);\n"
        "                if (report.action && n_tripped < GGML_SYCL_MAX_DEVICES) {\n", 1),
    # A binding reset whose report nobody logs.
    "binding-reset-report-dropped": backend.replace(
        "    ggml_sycl_execution_tripwire_log(ggml_sycl_execution_reset_backend_binding_state(ctx));\n",
        "    (void) ggml_sycl_execution_reset_backend_binding_state(ctx);\n", 1),
    "release-unwired": backend.replace('    ggml_sycl_execution_tripwire(ctx, "releasing", site, line);\n', "", 1),
    "release-site-dropped": backend.replace(
        "            ggml_sycl_execution_clear_graph_tracking(backend_ctx, site, line);\n        });\n"
        "        ggml_sycl_execution_try_retire_terminal(ctx, site, line);",
        "            ggml_sycl_execution_clear_graph_tracking(backend_ctx);\n        });\n"
        "        ggml_sycl_execution_try_retire_terminal(ctx, site, line);", 1),
    "scope-counts-out-late": backend.replace(SCOPE_COUNT_OUT + SCOPE_END, SCOPE_END + SCOPE_COUNT_OUT, 1),
    "tripwire-without-site": backend.replace("while a dispatch scope holds it open: site=%s:%d ",
                                             "while a dispatch scope holds it open: ", 1),
}
checks.update({
    "the tripwire guards every verb that ends or clears an open graph": not tripwire_found,
    **{
        f"tripwire contract witnesses mutation {name}": mutated != backend and bool(tripwire_violations(mutated))
        for name, mutated in tripwire_mutants.items()
    },
})
if tripwire_found:
    print("tripwire contract: " + "; ".join(tripwire_found), file=sys.stderr)

scope_end_found = scope_end_violations(backend)
scope_end_mutants = {
    # The shape before the outermost rule: every scope counts out and then seals or quarantines.
    "inner-scope-ends-graph": backend.replace(
        SCOPE_COUNT_OUT,
        "        {\n"
        "            std::lock_guard<std::mutex> state_lock(ctx->execution_state_mutex);\n"
        "            --ctx->execution_dispatch_depth;\n"
        "        }\n", 1),
    # An inner scope that steps aside on a normal exit but still quarantines on an unwind.
    "inner-scope-quarantines-on-unwind": backend.replace(
        "        if (!outermost) {\n", "        if (!outermost && std::uncaught_exceptions() <= uncaught) {\n", 1),
    # The depth tested after the lock is dropped, where another scope can move it between the two.
    "depth-tested-outside-the-lock": backend.replace(
        "            outermost = --ctx->execution_dispatch_depth == 0;\n        }\n",
        "            --ctx->execution_dispatch_depth;\n        }\n"
        "        outermost = ctx->execution_dispatch_depth == 0;\n", 1),
}
checks.update({
    "only the outermost dispatch scope seals or quarantines the tracked graph": not scope_end_found,
    **{
        f"outermost-scope contract witnesses mutation {name}": mutated != backend
        and bool(scope_end_violations(mutated))
        for name, mutated in scope_end_mutants.items()
    },
})
if scope_end_found:
    print("outermost-scope contract: " + "; ".join(scope_end_found), file=sys.stderr)

failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("execution abort source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("execution abort source contract: PASS")
