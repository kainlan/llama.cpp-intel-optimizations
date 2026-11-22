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
# every top-level function it calls down to a node dispatch, and requires that function to open the dispatch scope
# before its first dispatch.
DISPATCH_CALL = re.compile(r"\b(?:ggml_sycl_compute_forward|moe_graph_dispatch_direct_node)\s*\(")
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
    """name -> body of every column-0 `static` function definition in the blanked source."""
    bodies = {}
    for m in re.finditer(r"^static\b[^;{}()]*?\b(\w+)\s*\(", blank, re.M):
        params_close = matching_close(blank, m.end() - 1, "(", ")")
        if params_close < 0:
            continue
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", blank[params_close + 1:])
        if not tail:
            continue
        open_at = params_close + 1 + tail.end() - 1
        close_at = matching_close(blank, open_at)
        if close_at > open_at:
            bodies.setdefault(m.group(1), blank[open_at:close_at + 1])
    return bodies


def dispatch_scope_violations(source):
    """Functions graph_compute_unchecked calls that reach a node dispatch with no dispatch scope opened before it."""
    bodies = function_bodies(blank_comments_and_literals(source))
    root = bodies.get(DISPATCH_ROOT)
    if not root:
        return [DISPATCH_ROOT + " not found"]
    memo = {}

    def first_dispatch(name, depth):
        """Offset in `name`'s body of the first call that reaches a node dispatch within `depth` levels, or None."""
        key = (name, depth)
        if key in memo:
            return memo[key]
        memo[key] = None  # a recursive cycle reaches nothing new
        body = bodies[name]
        found = None
        for call in re.finditer(r"\b(\w+)\s*\(", body):
            if DISPATCH_CALL.match(body, call.start()):
                found = call.start()
                break
            callee = call.group(1)
            if depth > 0 and callee != name and callee in bodies and first_dispatch(callee, depth - 1) is not None:
                found = call.start()
                break
        memo[key] = found
        return found

    violations = set()
    callees = {call.group(1) for call in re.finditer(r"\b(\w+)\s*\(", root)}
    for callee in sorted(callees):
        if callee == DISPATCH_ROOT or callee not in bodies:
            continue
        at = first_dispatch(callee, DISPATCH_DEPTH)
        if at is None:
            continue
        scope = DISPATCH_SCOPE_OPEN.search(bodies[callee])
        if not scope or scope.start() > at:
            violations.add(callee)
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

dispatch_violations = dispatch_scope_violations(backend)
dispatch_mutants = {
    # A bypass path that does not open the scope.
    "replay-slot-unscoped": backend.replace(
        "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n    g_graph_diag_counters.seg_replay", "    g_graph_diag_counters.seg_replay", 1),
    # The block graphlets reach a dispatch two calls down; their scope is gone.
    "block-graphlets-unscoped": backend.replace(
        "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n\n    try {\n        if (moe_graph_block_graphs_match",
        "\n    try {\n        if (moe_graph_block_graphs_match", 1),
    # The scope opens after the first node already ran.
    "record-slot-scoped-late": backend.replace(
        "    ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n    g_graph_diag_counters.seg_record_attempts.fetch_add(1, std::memory_order_relaxed);",
        "    g_graph_diag_counters.seg_record_attempts.fetch_add(1, std::memory_order_relaxed);", 1).replace(
        "    for (const moe_graph_keyed_item & item : plan) {\n        if (item.boundary) {",
        "    for (const moe_graph_keyed_item & item : plan) {\n        ggml_sycl_execution_dispatch_scope execution_scope(sycl_ctx);\n        if (item.boundary) {", 1),
    # A new dispatcher graph_compute calls, with no scope.
    "new-unscoped-dispatcher": backend.replace(
        "static ggml_status ggml_backend_sycl_graph_compute_unchecked(ggml_backend_t backend, ggml_cgraph * cgraph) {",
        "static void moe_graph_unscoped_probe(ggml_backend_sycl_context * c, ggml_tensor * t) { ggml_sycl_compute_forward(*c, t); }\n"
        "static ggml_status ggml_backend_sycl_graph_compute_unchecked(ggml_backend_t backend, ggml_cgraph * cgraph) {\n"
        "    moe_graph_unscoped_probe(nullptr, nullptr);", 1),
}

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


# --- The tripwire: ending or clearing a graph a dispatch scope holds open is reported, by site ---
#
# Each lifecycle verb that ends the invocation (submit, release, quarantine) or zeroes the backend's copy of it (the
# clear, the binding reset) asks the tripwire first, with its caller's site; the scope counts itself out before its
# own seal or quarantine.
def tripwire_violations(source):
    found = []

    def body(name):
        return function_body(source, r"static \w+ " + name + r"\s*\(")

    tripwire = body("ggml_sycl_execution_tripwire_locked")
    if not (re.search(r"execution_dispatch_depth == 0 \|\|", tripwire) and "site=%s:%d" in tripwire
            and "execution_tripwire_invocation = ctx->execution_invocation_id;" in tripwire):
        found.append("tripwire reports once per invocation, only while a scope is open, with the site")
    for name, action, before in (
        ("ggml_sycl_execution_clear_graph_tracking", '"clearing"', "ctx->execution_graph_epoch = 0;"),
        ("ggml_sycl_execution_reset_backend_binding_state", '"unbinding"', "ctx->execution_context_id = 0;"),
    ):
        if not ordered(body(name), "ggml_sycl_execution_tripwire_locked(ctx, " + action + ", site, line);", before):
            found.append(name + " asks the tripwire before it zeroes")
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
        '    ggml_sycl_execution_tripwire_locked(ctx, "clearing", site, line);\n', "", 1),
    "release-unwired": backend.replace('    ggml_sycl_execution_tripwire(ctx, "releasing", site, line);\n', "", 1),
    "release-site-dropped": backend.replace(
        "                ggml_sycl_execution_clear_graph_tracking(backend_ctx, site, line);\n            });\n"
        "        ggml_sycl_execution_try_retire_terminal(ctx, site, line);",
        "                ggml_sycl_execution_clear_graph_tracking(backend_ctx);\n            });\n"
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
