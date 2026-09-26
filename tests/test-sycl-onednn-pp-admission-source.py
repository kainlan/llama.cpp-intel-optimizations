#!/usr/bin/env python3
"""Source gate for llama.cpp-je3b: "may oneDNN PP run here" has ONE source.

ggml_sycl_onednn_pp_candidate() answers that question for the unified
dispatcher: enabled (GGML_SYCL_ONEDNN_PP), skip_type
(GGML_SYCL_SKIP_ONEDNN_Q4_0), the batch floor, operand types, a contiguous
quantized weight, and finally executability on the device (placement). The
MXFP4 per-expert direct dispatch in ggml_sycl_mul_mat carried two oneDNN
arms (SOA and AOS) that answered it themselves with
`M >= 2 && ggml_sycl_onednn_pp_executable_on_device(...)` -- no enabled, no
skip_type, no batch floor. GGML_SYCL_ONEDNN_PP=0 therefore did not turn
those arms off. That is the CLAUDE.md "one fact, two sources" shape.

The arms differ from the dispatcher in exactly one respect, the batch
floor, and that difference now lives in the candidate (its `route`
argument, resolved by the pure onednn_pp_min_batch_for() in
onednn-pp-placement.hpp), not in the arm.

What this file pins:
  1. ggml_sycl_onednn_pp_executable_on_device() is only called from inside
     the candidate. It is one ingredient of the answer, not the answer.
  2. The batch-floor, enabled and skip_type accessors are only read where
     the question is answered (the candidate; skip_type/enabled also by the
     dense WOQ second-copy predicate, which answers a different question).
     This covers the accessors only: GGML_SYCL_ONEDNN_PP_MIN_BATCH is also
     parsed by raw getenv in mmvq.cpp (MMVQ-MoE PP threshold) and in the
     GGML_SYCL_PP_PIPELINE pre-scan, a partial admission copy tracked on
     llama.cpp-gkus.
  3. Every oneDNN GEMM in the MXFP4 direct block is guarded by a boolean
     taken from the candidate called with the MXFP4_DIRECT route. Neither
     that boolean nor any guard around a GEMM compares the batch (M or
     src1->ne[1]) itself.
  4. The candidate delegates its checks to the pure
     onednn_pp_admission_decide() and asks for executability once.
  5. The pure predicate and the per-route floor are defined once, in the
     header, and nowhere in the backend.

Runs under pytest (llama_test_pytest registration) and as a plain script.
Point it at alternate copies (the mutation controls use a git-archive
extract) via GGML_SYCL_ONEDNN_PP_BACKEND_SOURCE /
GGML_SYCL_ONEDNN_PP_PLACEMENT_SOURCE. No SYCL device or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(
    os.environ.get("GGML_SYCL_ONEDNN_PP_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
)
PLACEMENT = Path(
    os.environ.get("GGML_SYCL_ONEDNN_PP_PLACEMENT_SOURCE", str(ROOT / "ggml/src/ggml-sycl/onednn-pp-placement.hpp"))
)


def in_number(text, i):
    """True when the ' at text[i] is a C++14 digit separator (1'000,
    0xFF'FF): it sits inside a token that starts with a digit. A char
    literal's prefix (u8'a', L'a') starts with a letter, so it is not one.
    A number may also start with a dot (.5'0)."""
    j = i
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_.'"):
        j -= 1
    starts_number = text[j].isdigit() or (text[j] == "." and j + 1 < i and text[j + 1].isdigit())
    return j < i and starts_number and i + 1 < len(text) and text[i + 1].isalnum()


def strip_comments(text):
    """Blank out // and /* */ comments and string/char literal contents,
    preserving every offset (newlines are kept), so a comment that quotes
    code can neither satisfy nor trip a check."""
    out = list(text)
    i = 0
    n = len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'" and not in_number(text, i):
                state = "chr"
        elif state == "line":
            if ch == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                state = "code"
                i += 2
                continue
            if ch != "\n":
                out[i] = " "
        elif state in ("str", "chr"):
            quote = '"' if state == "str" else "'"
            if ch == "\\":
                out[i] = " "
                if i + 1 < n and text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            if ch == quote:
                state = "code"
            elif ch != "\n":
                out[i] = " "
        i += 1
    return "".join(out)


backend_raw = BACKEND.read_text()
placement_raw = PLACEMENT.read_text()
backend = strip_comments(backend_raw)
placement = strip_comments(placement_raw)


def ws_pattern(needle):
    """`needle` with every whitespace run flexible, so a clang-format rewrap
    does not break a match. Offsets are into the original text."""
    tokens = needle.split()
    assert tokens, "empty needle"
    return re.compile(r"\s+".join(re.escape(t) for t in tokens))


def matching(text, open_idx, open_ch, close_ch):
    """Match a bracket in comment-stripped text."""
    assert text[open_idx] == open_ch, f"expected {open_ch!r} at {open_idx}"
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError(f"unbalanced {open_ch}{close_ch}")


def function_span(text, name):
    """(start, end) of the body of the single DEFINITION of `name`: a
    `static <ret> name(` whose parameter list is followed by `{`."""
    spans = []
    for m in re.finditer(r"\bstatic\s+[\w:<>]+\s+" + re.escape(name) + r"\s*\(", text):
        close = matching(text, m.end() - 1, "(", ")")
        rest = text[close + 1 :].lstrip()
        if rest.startswith("{"):
            open_idx = text.index("{", close)
            spans.append((open_idx, matching(text, open_idx, "{", "}")))
    assert len(spans) == 1, f"expected exactly one definition of {name}, found {len(spans)}"
    return spans[0]


def call_sites(text, name):
    """Offsets of calls to `name`, excluding its own definition's name."""
    sites = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
        before = text[max(0, m.start() - 80) : m.start()]
        if re.search(r"\bstatic\s+[\w:<>]+\s*$", before):
            continue  # the definition (or a forward declaration)
        sites.append(m.start())
    return sites


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def enclosing_if_conditions(text, lo, hi, pos):
    """Conditions of every `if (...) {` block in text[lo:hi] whose BODY
    contains `pos`, innermost last."""
    conds = []
    for m in re.finditer(r"\bif\s*\(", text[lo:hi]):
        paren = lo + m.end() - 1
        close = matching(text, paren, "(", ")")
        after = text[close + 1 :]
        stripped = after.lstrip()
        if not stripped.startswith("{"):
            continue
        brace = close + 1 + (len(after) - len(stripped))
        end = matching(text, brace, "{", "}")
        if brace < pos < end:
            conds.append(text[paren + 1 : close])
    return conds


MXFP4_GUARD = "src0->type == GGML_TYPE_MXFP4 && !src0_planned_host"

# Any comparison on the batch, from either side: M >= 2, M != 1, M > min_m,
# src1->ne[1] >= 2, 2 <= M, static_cast<int>(M) >= 2. Closing parens may sit
# between the batch and the operator; on the reversed side they may not, or
# the `>` closing `static_cast<int>` in front of `(M)` would read as `> M`.
_BATCH = r"(?:\bM\b|\bsrc1\s*->\s*ne\s*\[\s*1\s*\])"
_CMP = r"(?:>=|<=|!=|==|>|<)"
BATCH_COMPARISON = re.compile(_BATCH + r"(?:\s*\))*\s*" + _CMP + r"|" + _CMP + r"\s*" + _BATCH)


def top_level_conjuncts(expr):
    """Split `expr` on && outside any parentheses, after peeling parens
    that wrap the whole expression. `!(a && b)` stays one conjunct."""
    expr = expr.strip()
    while expr.startswith("(") and matching(expr, 0, "(", ")") == len(expr) - 1:
        expr = expr[1:-1].strip()
    parts, depth, start, i = [], 0, 0, 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr.startswith("&&", i):
            parts.append(expr[start:i].strip())
            start = i + 2
            i += 2
            continue
        i += 1
    parts.append(expr[start:].strip())
    return parts


def mxfp4_direct_span():
    """Body of the MXFP4 per-expert direct dispatch block in ggml_sycl_mul_mat."""
    matches = list(ws_pattern(MXFP4_GUARD).finditer(backend))
    assert len(matches) == 1, f"expected one MXFP4 direct guard, found {len(matches)}"
    if_idx = backend.rfind("if (", 0, matches[0].start())
    close = matching(backend, backend.index("(", if_idx), "(", ")")
    brace = backend.index("{", close)
    return brace, matching(backend, brace, "{", "}")


def test_executable_on_device_is_only_an_ingredient_of_the_candidate():
    lo, hi = function_span(backend, "ggml_sycl_onednn_pp_candidate")
    stray = [s for s in call_sites(backend, "ggml_sycl_onednn_pp_executable_on_device") if not (lo < s < hi)]
    assert not stray, (
        "ggml_sycl_onednn_pp_executable_on_device() is called outside ggml_sycl_onednn_pp_candidate() at "
        f"line(s) {[line_of(backend, s) for s in stray]}: that is a second, partial answer to 'may oneDNN PP "
        "run here' (no enabled / skip_type / batch floor). Ask the candidate instead."
    )


def test_admission_inputs_are_read_only_where_the_question_is_answered():
    cand = function_span(backend, "ggml_sycl_onednn_pp_candidate")
    woq = function_span(backend, "ggml_sycl_dense_woq_alternate_eligible_impl")

    def inside(s, spans):
        return any(lo < s < hi for lo, hi in spans)

    for name, allowed in (
        ("ggml_sycl_onednn_pp_min_batch", [cand]),
        ("ggml_sycl_onednn_pp_enabled", [cand, woq]),
        ("ggml_sycl_onednn_pp_skip_type", [cand, woq]),
    ):
        sites = call_sites(backend, name)
        assert any(inside(s, [cand]) for s in sites), f"the candidate no longer reads {name}()"
        stray = [line_of(backend, s) for s in sites if not inside(s, allowed)]
        assert not stray, f"{name}() is read outside the admission answer at line(s) {stray}"


def test_every_mxfp4_direct_onednn_gemm_is_admitted_by_the_candidate():
    lo, hi = mxfp4_direct_span()
    region = backend[lo:hi]
    # The answer may be narrowed by conjuncts that are not about the batch
    # (e.g. skipping the question for a layout with no oneDNN arm), but the
    # candidate's call must be one whole top-level && conjunct: no ||, no ?:,
    # nothing negating or comparing it.
    cands = list(re.finditer(r"const\s+bool\s+(\w+)\s*=([^;]*?)\bggml_sycl_onednn_pp_candidate\s*\(", region))
    assert len(cands) == 1, (
        f"expected the MXFP4 direct block to ask ggml_sycl_onednn_pp_candidate() exactly once into a const bool, "
        f"found {len(cands)}"
    )
    var = cands[0].group(1)
    call_open = lo + cands[0].end() - 1
    init = backend[lo + cands[0].start(2) : backend.index(";", call_open)]
    init_flat = " ".join(init.split())
    call_start = backend.rindex("ggml_sycl_onednn_pp_candidate", 0, call_open + 1)
    call_flat = " ".join(backend[call_start : matching(backend, call_open, "(", ")") + 1].split())
    conjuncts = top_level_conjuncts(init_flat)
    assert "?" not in init_flat and "||" not in init_flat and call_flat in conjuncts, (
        f"`{var}` is initialised as `{init_flat}`: the candidate's call must be one whole && conjunct of it, "
        "with no ||, no ?:, and nothing negating or comparing it"
    )
    assert not BATCH_COMPARISON.search(init), (
        f"`{var}` is initialised as `{init_flat}`, which compares the batch itself; the floor is the candidate's"
    )
    call_args = backend[call_open : matching(backend, call_open, "(", ")") + 1]
    assert re.search(r"onednn_pp_route\s*::\s*MXFP4_DIRECT\b", call_args), (
        "the MXFP4 direct block must ask the candidate with onednn_pp_route::MXFP4_DIRECT: its batch floor is "
        "the candidate's to express, not the arm's"
    )
    assert len(re.findall(r"ggml_sycl_onednn_pp_candidate\s*\(", region)) == 1, (
        "the MXFP4 direct block asks the candidate more than once; one answer per dispatch"
    )

    gemms = [lo + m.start() for m in re.finditer(r"DnnlGemmWrapper\s*::\s*row_gemm\s*\(", region)]
    # Positive control: the block has an SOA arm and an AOS arm. If this
    # count moves, re-read the block before trusting the checks below.
    assert len(gemms) == 2, f"expected 2 oneDNN GEMMs (SOA and AOS arms) in the MXFP4 direct block, found {len(gemms)}"
    for g in gemms:
        conds = enclosing_if_conditions(backend, lo, hi, g)
        guards = [c for c in conds if re.search(r"\b" + re.escape(var) + r"\b", c)]
        assert guards, f"oneDNN GEMM at line {line_of(backend, g)} is not guarded by the candidate's answer `{var}`"
        # `var` must be a plain conjunct of its guard: `var || x` or `!var`
        # names the answer without being bound by it.
        for c in guards:
            flat = " ".join(c.split())
            conjuncts = [t.strip() for t in flat.split("&&")]
            assert "||" not in flat and var in conjuncts, (
                f"oneDNN GEMM at line {line_of(backend, g)} is guarded by `{flat}`; `{var}` must be a bare "
                "conjunct of that guard, with no `||`"
            )
        adhoc = [c for c in conds if BATCH_COMPARISON.search(c)]
        assert not adhoc, (
            f"oneDNN GEMM at line {line_of(backend, g)} carries its own batch admission "
            f"({' '.join(adhoc[0].split())}); the batch floor belongs to the candidate"
        )


def test_mxfp4_direct_route_is_named_only_by_the_mxfp4_direct_block():
    # The asker names the route; the candidate, which interprets it, may too.
    spans = [mxfp4_direct_span(), function_span(backend, "ggml_sycl_onednn_pp_candidate")]
    uses = [m.start() for m in re.finditer(r"onednn_pp_route\s*::\s*MXFP4_DIRECT\b", backend)]
    assert uses, "onednn_pp_route::MXFP4_DIRECT is never used: the MXFP4 direct block does not ask the candidate"
    stray = [line_of(backend, s) for s in uses if not any(lo < s < hi for lo, hi in spans)]
    assert not stray, (
        f"onednn_pp_route::MXFP4_DIRECT used outside the MXFP4 direct block at line(s) {stray}: dense sites keep "
        "the dispatcher's floor"
    )


def test_candidate_delegates_to_the_pure_predicate_and_asks_placement_once():
    lo, hi = function_span(backend, "ggml_sycl_onednn_pp_candidate")
    body = backend[lo:hi]
    assert len(re.findall(r"\bonednn_pp_admission_decide\s*\(", body)) == 1, (
        "the candidate must decide admission through ggml_sycl::onednn_pp_admission_decide() exactly once"
    )
    assert len(re.findall(r"\bonednn_pp_min_batch_for\s*\(", body)) == 1, (
        "the candidate must resolve its batch floor through ggml_sycl::onednn_pp_min_batch_for() exactly once"
    )
    assert len(re.findall(r"\bggml_sycl_onednn_pp_executable_on_device\s*\(", body)) == 1, (
        "the candidate must ask executability exactly once"
    )
    # The inline checks the predicate replaced must not survive beside it.
    assert not re.search(r"src1\s*->\s*ne\s*\[\s*1\s*\]\s*<", body), (
        "the candidate compares the batch against a floor inline; that is the pure predicate's job"
    )
    admit = re.search(r"\bonednn_pp_admission_decide\s*\(", body).start()
    execu = re.search(r"\bggml_sycl_onednn_pp_executable_on_device\s*\(", body).start()
    assert admit < execu, "admission must be decided before executability (which may take the cache lock)"


def test_pure_predicate_lives_once_in_the_header():
    for name in ("onednn_pp_admission_decide", "onednn_pp_min_batch_for"):
        defs = re.findall(r"\binline\s+[\w:<>]+\s+" + name + r"\s*\(", placement)
        assert len(defs) == 1, f"expected one inline definition of {name} in onednn-pp-placement.hpp, found {len(defs)}"
        assert not re.search(r"\b(?:static|inline)\s+[\w:<>]+\s+" + name + r"\s*\(", backend), (
            f"{name} is (re)defined in ggml-sycl.cpp"
        )
    assert re.search(r"enum\s+class\s+onednn_pp_route\b", placement), "onednn_pp_route missing from the header"


def test_strip_comments_handles_digit_separators():
    # One separator (an odd count) used to open a char literal that ran to
    # the next quote, blanking the code after it.
    src = (
        "int x = 1'000; foo(); // gone\nchar c = u8'a'; long y = 0xFF'FF'FF; bar(); /* gone */\n"
        "double d = .5'0; baz(); // gone\n"
    )
    out = strip_comments(src)
    assert len(out) == len(src)
    assert "foo();" in out and "bar();" in out and "baz();" in out, f"code after a digit separator was blanked: {out!r}"
    assert "gone" not in out, f"a comment survived: {out!r}"
    assert "u8' '" in out, f"a char literal's content survived: {out!r}"
    assert "1'000" in out and "0xFF'FF'FF" in out and ".5'0" in out, f"a numeric literal was altered: {out!r}"


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
