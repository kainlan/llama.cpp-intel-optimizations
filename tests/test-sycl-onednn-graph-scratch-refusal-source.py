"""Source contract for the runtime-context refusal of a oneDNN Graph attention scratch the load did not plan.

The ONEDNN zone holds the Graph scratch room the load sized for the context it was told. A later context can ask
for more (a larger -c than the hint, or an API caller that gave none). The part above the room is served outside the
arena, by the DIRECT path, which draws raw device memory and aborts the process when the card has none left. The
runtime-context transaction therefore asks, before it commits anything, whether that part fits outside the arena
above the driver headroom beside the hold's worst-case spill, and refuses the context by name when it does not. Only
the part above the room is refused: a context whose scratch outgrows the zone but fits outside the arena runs.

The arithmetic is host-tested (test-zone-sizing, Case 24h). This file pins the wiring no host test can reach:
- the transaction calls the check after the non-FA check and before its first state that a refusal must undo;
- the check returns early when flash attention is off, and when the Graph allocator is disabled;
- the check runs in probe mode too (the micro-batch ladder must see the verdict a real context gets);
- the refusal names the largest -ub the same fit accepts.
Each claim has a mutant that fails its own check and no other; the order claim has one for each half (the call
moved above the non-FA check, and moved past the first guarded state).

Host-only text assertions on comment-stripped source. llama_test_pytest hands this file to pytest.main(), so every
check lives in a test_*() function.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

# One left-to-right alternation, so a `/*` inside a `//` comment's prose cannot open a block comment.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def normalize(src: str) -> str:
    return re.sub(r"\s+", " ", strip_comments(src))


SYCL_NORM = normalize(GGML_SYCL_CPP)

TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
CHECK_NAME = "ggml_sycl_check_onednn_graph_scratch"
CHECK_SIG = f"static bool {CHECK_NAME}("
CHECK_CALL = f"if (!{CHECK_NAME}(ctx->device, next_plan, flash_attn_enabled, probe_mode, hold_query)) {{"
NONFA_CALL = "if (!ggml_sycl_check_nonfa_attn_scratch("
NONFA_REFUSE ='return refuse("non-FA attention scratch exceeds the device\'s outside-arena headroom");'
# The first state the transaction restores on a refusal: the dense scratch's transaction guard.
FIRST_GUARDED_STATE = "ggml_sycl_dense_scratch_txn_guard dense_guard("
EARLY_RETURN_RE = re.compile(r"^ #if GGML_SYCL_DNNL if \(([^{]*)\) \{ return true; \}")
FA_OFF = "!flash_attn_enabled"
ALLOCATOR_OFF = "!ggml_sycl::onednn_graph_allocator_enabled()"
REFUSAL_TEXT = "runtime context update rejected: the oneDNN Graph attention scratch"
LARGEST_ASSIGN = "const uint32_t largest = ggml_sycl::zone_onednn_graph_fit_largest_ub("
ADVICE_ARG = "ggml_sycl_hold_fit_advice(largest).c_str()"


def _body(src: str, sig: str) -> tuple:
    """The [start, end) span of the brace-balanced body of the one function whose signature starts with `sig`."""
    assert src.count(sig) == 1, f"expected exactly one {sig!r}, found {src.count(sig)}"
    open_at = src.index("{", src.index(sig))
    depth = 0
    for i in range(open_at, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return open_at, i + 1
    raise AssertionError(f"unbalanced body for {sig!r}")


def _body_text(src: str, sig: str) -> str:
    start, end = _body(src, sig)
    return src[start:end]


def _depth_at(body: str, pos: int) -> int:
    return body[:pos].count("{") - body[:pos].count("}")


def _called_in_order(src: str) -> bool:
    """One call, after the non-FA refusal and before the first state a refusal must undo, refusing on false."""
    body = _body_text(src, TXN_SIG)
    if body.count(CHECK_CALL) != 1 or body.count(NONFA_REFUSE) != 1 or body.count(FIRST_GUARDED_STATE) != 1:
        return False
    call = body.index(CHECK_CALL)
    after = body[call + len(CHECK_CALL) :].lstrip()
    return body.index(NONFA_REFUSE) < call < body.index(FIRST_GUARDED_STATE) and after.startswith('return refuse("')


def _runs_in_probe_mode(src: str) -> bool:
    """The call sits at the transaction's top level, under no condition: a probe asks it as a real context does."""
    body = _body_text(src, TXN_SIG)
    if body.count(CHECK_CALL) != 1:
        return False
    return _depth_at(body, body.index(CHECK_CALL)) == 1


def _early_return_condition(src: str):
    m = EARLY_RETURN_RE.match(_body_text(src, CHECK_SIG)[1:])
    return [term.strip() for term in m.group(1).split("||")] if m else None


def _returns_early_when_fa_off(src: str) -> bool:
    terms = _early_return_condition(src)
    return terms is not None and FA_OFF in terms


def _returns_early_when_allocator_off(src: str) -> bool:
    terms = _early_return_condition(src)
    return terms is not None and ALLOCATOR_OFF in terms


def _call_text(text: str, open_paren: int) -> str:
    """The parenthesised argument text that opens at `open_paren`, string literals skipped (a format string may hold
    a parenthesis or a semicolon)."""
    depth, i, in_str = 0, open_paren, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren : i + 1]
        i += 1
    raise AssertionError("unbalanced call")


def _refusal_names_the_largest_ub(src: str) -> bool:
    """The refusal's last argument is the advice for the -ub the fit returned."""
    body = _body_text(src, CHECK_SIG)
    at = body.find("GGML_SYCL_RUNTIME_TXN_REFUSAL(")
    if at < 0 or body.count(LARGEST_ASSIGN) != 1 or body.index(LARGEST_ASSIGN) > at:
        return False
    args = _call_text(body, body.index("(", at))
    return REFUSAL_TEXT in args and args.endswith(f"{ADVICE_ARG})")


def _checks(src: str) -> dict:
    return {
        "order": _called_in_order(src),
        "probe": _runs_in_probe_mode(src),
        "fa_off": _returns_early_when_fa_off(src),
        "allocator_off": _returns_early_when_allocator_off(src),
        "names_ub": _refusal_names_the_largest_ub(src),
    }


def test_transaction_calls_the_check_after_the_nonfa_check_and_before_any_commit():
    assert _called_in_order(SYCL_NORM), (
        f"the transaction must call {CHECK_NAME}() once, after the non-FA refusal and before {FIRST_GUARDED_STATE}...), "
        "and refuse when it returns false"
    )


def test_the_check_runs_in_probe_mode():
    assert _runs_in_probe_mode(SYCL_NORM), (
        f"{CHECK_NAME}() must be called at the transaction's top level, under no condition, so a probe sees the "
        "verdict a real context gets"
    )


def test_the_check_returns_early_without_flash_attention_or_the_graph_allocator():
    assert _returns_early_when_fa_off(SYCL_NORM), f"{CHECK_NAME}() must return true first when {FA_OFF}"
    assert _returns_early_when_allocator_off(SYCL_NORM), f"{CHECK_NAME}() must return true first when {ALLOCATOR_OFF}"


def test_the_refusal_names_the_largest_ub_that_fits():
    assert _refusal_names_the_largest_ub(SYCL_NORM), (
        f"the refusal must end with {ADVICE_ARG}, the -ub zone_onednn_graph_fit_largest_ub() accepts"
    )


def _mutate_in(src: str, sig: str, old: str, new: str) -> str:
    start, end = _body(src, sig)
    body = src[start:end]
    assert body.count(old) == 1, f"mutation anchor must occur once in {sig!r}: {old!r}"
    return src[:start] + body.replace(old, new, 1) + src[end:]


def _assert_only(mutant: str, failing: str, why: str) -> None:
    results = _checks(mutant)
    expected = {name: name != failing for name in results}
    assert results == expected, f"mutation witness for {failing!r} ({why}) is broken: {results}"


def test_every_check_has_a_mutation_witness_that_fails_only_it():
    src = SYCL_NORM
    assert all(_checks(src).values()), "sanity: every check passes on the source itself"

    # The call moved past the first guarded state: everything else unchanged.
    start, end = _body(src, TXN_SIG)
    body = src[start:end]
    call_at = body.index(CHECK_CALL)
    call_end = body.index("}", body.index('return refuse("', call_at)) + 1
    block = body[call_at:call_end]
    moved = body[:call_at] + body[call_end:]
    guard_end = moved.index(";", moved.index(FIRST_GUARDED_STATE)) + 1
    moved = moved[:guard_end] + " " + block + moved[guard_end:]
    _assert_only(src[:start] + moved + src[end:], "order", "the check after the dense scratch guard")

    # The call moved above the non-FA check, still before the guard: the other half of the order claim.
    moved = body[:call_at] + body[call_end:]
    assert moved.count(NONFA_CALL) == 1, f"mutation anchor must occur once in the transaction: {NONFA_CALL!r}"
    nonfa_at = moved.index(NONFA_CALL)
    moved = moved[:nonfa_at] + block + " " + moved[nonfa_at:]
    _assert_only(src[:start] + moved + src[end:], "order", "the check before the non-FA check")

    _assert_only(_mutate_in(src, TXN_SIG, block, f"if (!probe_mode) {{ {block} }}"), "probe",
                 "the check skipped for a probe")

    _assert_only(_mutate_in(src, CHECK_SIG, f"if ({FA_OFF} || {ALLOCATOR_OFF})", f"if ({ALLOCATOR_OFF})"), "fa_off",
                 "a context without flash attention judged by the Graph check")

    _assert_only(_mutate_in(src, CHECK_SIG, f"if ({FA_OFF} || {ALLOCATOR_OFF})", f"if ({FA_OFF})"), "allocator_off",
                 "a disabled Graph allocator judged by the Graph check")

    _assert_only(_mutate_in(src, CHECK_SIG, ADVICE_ARG, "ggml_sycl_hold_fit_advice(0).c_str()"), "names_ub",
                 "the refusal names no -ub")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
