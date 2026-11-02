"""Source contract for llama.cpp-5efe C4 (part a): a refused segmented node fails the graph, it is never skipped.

The MoE segmented record and replay paths used to discard `ggml_sycl_compute_forward`'s result at six sites, while the
ordinary node loop asserts on the same `false`: a host attention dispatch refused for want of a planned staging slot
(design 5efe §4.4 "A refusal fails the graph on every path", §M69 (b); tracked as llama.cpp-573t) was a WARN and a
skipped op, which is wrong tokens, not a refusal. This gate pins the fix:

1. the six sites go through ONE helper, `ggml_sycl_segment_forward_or_throw`, and no other `ggml_sycl_compute_forward`
   call remains in `moe_graph_record_segments` or `moe_graph_replay_segments`; each call names its role, in order;
2. the helper prints the ordinary loop's unsupported-node line and throws `ggml_sycl_fallback_error` with a string
   literal (the type stores the pointer, and the catch prints `what()` after the unwind). It never asserts, aborts or
   returns on a refusal: a refusal is a capability refusal and the throw is what reaches `graph_compute`'s cleanup;
3. `moe_graph_record_segments` refuses to record over a context that still holds segments after a failed retire (its
   entry guard), and scopes its record loop under a handover whose destructor, only while an exception leaves the
   block, moves the executables it has recorded into the context rather than letting the unwind destroy them while
   their submission may be outstanding. A destructor, not a rethrowing `catch (...)`: the alloc-zone contract's
   no-swallow rule counts every catch-all, and its rethrow-first clause would let the refusal skip the move;
4. `graph_compute`'s `ggml_sycl_fallback_error` catch has a cleanup step immediately after the MMID recovery and
   before the transient-lease unpin, a call of the noexcept `ggml_sycl_graph_compute_fallback_clear_graphs`, whose
   own `try { ... } catch (...) {}` clears the cached whole-graph state (or just the segments) after the queue waits
   and the staging drain.

The runtime counterpart is the lead-run device test `test-sycl-seg-refusal-cleanup`. Plain file I/O, comment-stripped
text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces")


def block_end(text: str, open_idx: int) -> int:
    """Index just past the brace that closes the one at `open_idx`."""
    return open_idx + len(balanced(text, open_idx)) + 2


def function_body(text: str, signature_re: str) -> str:
    hits = list(re.finditer(signature_re, text))
    assert len(hits) == 1, f"{signature_re!r} matched {len(hits)} times"
    return balanced(text, text.index("{", hits[0].end()))


def code() -> str:
    return strip_comments((SYCL / "ggml-sycl.cpp").read_text())


RECORD = r"static\s+bool\s+moe_graph_record_segments\s*\("
REPLAY = r"static\s+void\s+moe_graph_replay_segments\s*\("
HELPER = r"static\s+void\s+ggml_sycl_segment_forward_or_throw\s*\("
CLEAR_STEP = (
    r"static\s+void\s+ggml_sycl_graph_compute_fallback_clear_graphs\s*\(\s*ggml_backend_sycl_context\s*\*\s*ctx\s*\)"
    r"\s*noexcept\b"
)
CLEAR_CALL = re.compile(r"\bggml_sycl_graph_compute_fallback_clear_graphs\s*\(\s*cleanup_ctx\s*\)\s*;")

# the six sites, by function and role, in source order
RECORD_ROLES = [
    "direct segment",
    "moe_after of a direct segment",
    "recording",
    "moe_after of a recorded segment",
]
REPLAY_ROLES = ["replay direct segment", "replay moe node"]

HELPER_CALL = re.compile(r"\bggml_sycl_segment_forward_or_throw\s*\(([^;]*?)\)\s*;", flags=re.DOTALL)
RAW_CALL = re.compile(r"\bggml_sycl_compute_forward\s*\(")


def discards(body: str) -> list:
    """Every direct `ggml_sycl_compute_forward(` call in a function body, as the text that starts at it."""
    return [body[m.start() : m.start() + 70].split("\n")[0] for m in RAW_CALL.finditer(body)]


def roles(body: str) -> list:
    out = []
    for m in HELPER_CALL.finditer(body):
        lit = re.search(r'"([^"]*)"\s*$', m.group(1).strip())
        out.append(lit.group(1) if lit else None)
    return out


# ---- 1. the six sites ----


def test_no_segmented_record_site_discards_the_forward_result():
    body = function_body(code(), RECORD)
    left = discards(body)
    assert (
        not left
    ), "moe_graph_record_segments calls ggml_sycl_compute_forward directly (a refusal is discarded): " + str(left)
    assert roles(body) == RECORD_ROLES, f"record sites, by role and in order: {roles(body)}, expected {RECORD_ROLES}"


def test_no_segmented_replay_site_discards_the_forward_result():
    body = function_body(code(), REPLAY)
    left = discards(body)
    assert (
        not left
    ), "moe_graph_replay_segments calls ggml_sycl_compute_forward directly (a refusal is discarded): " + str(left)
    assert roles(body) == REPLAY_ROLES, f"replay sites, by role and in order: {roles(body)}, expected {REPLAY_ROLES}"


def test_the_two_functions_hold_exactly_six_helper_calls():
    text = code()
    n = len(HELPER_CALL.findall(function_body(text, RECORD))) + len(HELPER_CALL.findall(function_body(text, REPLAY)))
    assert n == 6, f"the segmented functions hold {n} helper calls, not six: a seventh call routed around cannot hide"


def test_the_discard_reader_flags_a_planted_discard_and_a_planted_wrapper():
    planted = "void f() { ggml_sycl_compute_forward(*ctx, node); }"
    assert len(discards(planted)) == 1
    wrapped = 'void f() { ggml_sycl_segment_forward_or_throw(*ctx, node, "recording"); }'
    assert not discards(wrapped) and roles(wrapped) == ["recording"]


# ---- 2. the helper ----


def helper_body() -> str:
    return function_body(code(), HELPER)


def test_the_helper_prints_the_unsupported_node_line_and_throws_a_literal():
    body = helper_body()
    assert len(RAW_CALL.findall(body)) == 1, "the helper does not make the one forward call"
    assert re.search(
        r'fprintf\s*\(\s*stderr\s*,\s*"\[SYCL\] graph compute unsupported node', body
    ), "the helper does not print the ordinary loop's unsupported-node line"
    throws = re.findall(r"\bthrow\s+ggml_sycl_fallback_error\s*\(\s*(.*?)\s*\)\s*;", body, flags=re.DOTALL)
    assert len(throws) == 1, f"the helper throws ggml_sycl_fallback_error {len(throws)} times, expected 1"
    assert re.fullmatch(r'"[^"\\]*"', throws[0]), f"the throw argument is not a string literal: {throws[0]!r}"


def test_the_helper_never_asserts_aborts_or_returns_on_a_refusal():
    body = helper_body()
    for bad in (
        r"\bGGML_ASSERT\b",
        r"\bGGML_ABORT\b",
        r"\babort\s*\(",
        r"\bstd::abort\b",
        r"\bexit\s*\(",
        r"\breturn\s+false\b",
    ):
        assert not re.search(bad, body), f"the helper has {bad}: a refusal must throw, not end the process or return"
    assert re.search(
        r"if\s*\(\s*ggml_sycl_compute_forward\s*\([^)]*\)\s*\)\s*\{?\s*return\s*;", body
    ), "the helper does not return only when the forward succeeded"
    assert body.index("fprintf") < body.index("throw"), "the throw comes before the unsupported-node line"


# ---- 3. the record function ----


def test_the_record_function_refuses_a_context_that_still_holds_segments():
    body = function_body(code(), RECORD)
    guard = re.search(
        r"if\s*\(\s*!\s*sycl_ctx\s*->\s*moe_segments\s*\.\s*empty\s*\(\s*\)\s*\|\|\s*"
        r"!\s*sycl_ctx\s*->\s*moe_dispatch_graphs\s*\.\s*empty\s*\(\s*\)\s*\)\s*\{(.*?)return\s+false\s*;",
        body,
        flags=re.DOTALL,
    )
    assert guard, "moe_graph_record_segments has no entry guard against a context that still holds segments"
    assert "record refused" in guard.group(1), "the entry guard does not print the `record refused` WARN"
    assert re.search(r"GGML_LOG_WARN\s*\(", guard.group(1)), "the entry guard is not a WARN"
    first_local = body.index("recorded_segments;")
    assert guard.start() < body.index("seg_record_attempts"), "the guard comes after the function has begun recording"
    assert guard.start() < first_local, "the guard comes after the locals that would hold the new executables"


def test_the_record_loop_moves_its_executables_into_the_context_before_the_throw_leaves():
    body = function_body(code(), RECORD)
    locals_end = body.index("recorded_moe_graphs.reserve")
    rest = body[locals_end:]
    # the instance opens the block that holds the record loop, bound to this context and these two vectors
    scoped = re.search(
        r"\{\s*(\w+)\s+\w+\s*\{\s*sycl_ctx\s*,\s*recorded_segments\s*,\s*recorded_moe_graphs\s*\}\s*;\s*"
        r"for\s*\(\s*size_t\s+seg_idx\s*=\s*0\s*;",
        rest,
    )
    assert scoped, "the record loop is not the first statement of a block opened by a handover instance"
    name = scoped.group(1)
    # its type: a destructor that moves both vectors only while an exception is leaving the block
    decl = re.search(r"\bstruct\s+" + name + r"\s*\{", rest[: scoped.start()])
    assert decl, f"the handover type {name} is not declared between the locals and the record loop"
    struct_text = balanced(rest, rest.index("{", decl.start()))
    assert re.search(
        r"int\s+uncaught\s*=\s*std::uncaught_exceptions\s*\(\s*\)\s*;", struct_text
    ), "the handover does not record the exceptions in flight when it is constructed"
    dtor = re.search(r"~\s*" + name + r"\s*\(\s*\)\s*\{", struct_text)
    assert dtor, "the handover type has no destructor"
    dtor_text = balanced(struct_text, struct_text.index("{", dtor.start()))
    cond = re.match(r"\s*if\s*\(\s*std::uncaught_exceptions\s*\(\s*\)\s*>\s*uncaught\s*\)\s*\{", dtor_text)
    assert cond, "the handover's destructor does not act only while an exception leaves the block"
    moved = balanced(dtor_text, cond.end() - 1)
    assert re.search(
        r"ctx\s*->\s*moe_segments\s*=\s*std::move\s*\(\s*segments\s*\)", moved
    ), "the handover does not move the recorded segments into the context"
    assert re.search(
        r"ctx\s*->\s*moe_dispatch_graphs\s*=\s*std::move\s*\(\s*moe_graphs\s*\)", moved
    ), "the handover does not move the recorded MoE graphs into the context"
    assert re.search(
        r"ctx\s*->\s*moe_segments_valid\s*=\s*false\s*;", moved
    ), "the handover does not mark the segments invalid (the flag may be stale-true after a refused retire)"
    assert not re.search(r"\bwait\s*\(|wait_and_throw", struct_text), "the handover waits: that would precede the cleanup step"
    block_open = locals_end + scoped.start()
    after = body[block_end(body, block_open) :]
    assert not re.match(r"\s*catch\b", after), "the record loop's block is a try again: a catch-all there is a G-CATCH finding"


# ---- 4. graph_compute's fallback catch ----


def fallback_catch() -> str:
    text = code()
    marker = text.index("[SYCL] recoverable runtime fallback failed")
    start = text.rindex("catch (const ggml_sycl_fallback_error & error)", 0, marker)
    return balanced(text, text.index("{", start))


def test_the_catch_cleanup_follows_the_mmid_recovery_and_precedes_the_unpin():
    body = fallback_catch()
    recover = body.index("ggml_sycl_recover_exact_mmid_after_drain")
    unpin = body.index("graph_unpin_transient_leases_after_direct_execution")
    clear = CLEAR_CALL.search(body)
    assert clear, "the catch has no cleanup step that clears the cached graph state"
    assert len(CLEAR_CALL.findall(body)) == 1, "the catch clears the cached graph state more than once"
    assert (
        recover < clear.start() < unpin
    ), "the cleanup step is not between the MMID recovery and the transient-lease unpin"


def test_the_clear_step_clears_the_live_graph_or_the_handed_over_segments():
    step = function_body(code(), CLEAR_STEP)
    clear = re.search(r'sycl_exec_graph_clear_active\s*\(\s*ctx\s*,\s*"fallback-error"\s*\)', step)
    assert clear, "the clear step does not clear the active execution graph"
    assert re.search(
        r"exec_graph\b[^;{]*\|\|[^;{]*active_exec_graph\s*\.\s*valid", step[: clear.start()]
    ), "the step does not clear the active graph when an execution graph is live or marked valid"
    assert re.search(
        r"ctx\s*->\s*invalidate_moe_segments\s*\(\s*\)", step
    ), "the clear step does not invalidate the MoE segments when no execution graph is live"


def test_the_catch_cleanup_runs_after_every_wait_and_in_its_own_guard():
    body = fallback_catch()
    clear = CLEAR_CALL.search(body).start()
    for wait in ("wait_and_throw", "wait_pending_secondary_scatter_events", "ggml_sycl_cpu_staging_drain"):
        assert body.index(wait) < clear, f"the cleanup step runs before {wait}: an executable could still be in flight"
    # its own try { ... } catch (...) {} in a noexcept function: an exception out of the handler would cross the boundary
    step = function_body(code(), CLEAR_STEP)
    opened = re.match(r"\s*try\s*\{", step)
    assert opened, "the clear step's body is not one try block"
    try_open = step.index("{", opened.start())
    end = block_end(step, try_open)
    assert end > step.index("sycl_exec_graph_clear_active"), "the clear is not inside the step's try block"
    assert re.fullmatch(
        r"\s*catch\s*\(\s*\.\.\.\s*\)\s*\{\s*\}\s*", step[end:]
    ), "the clear step's try has no catch (...) {}, or the step does more after it"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
