"""Source contract for llama.cpp-9goq: ggml/src/ggml-sycl/CMakeLists.txt must
not regress the way three test targets did (test-attn-host-pool,
test-attn-host-flash-attn-identity, test-sycl-expert-predictor-guard, all
introduced between 2026-08-27 and 2026-09-02): each called
`target_link_libraries(<test> ... ggml-sycl ...)` unconditionally. Under
-DGGML_BACKEND_DL=ON, ggml-sycl is a MODULE library and CMake hard-errors --
"Target \"ggml-sycl\" of type MODULE_LIBRARY may not be linked into another
target" -- which broke .devops/intel.Dockerfile's configure step (that stage
passes exactly GGML_BACKEND_DL=ON + GGML_CPU_ALL_VARIANTS=ON +
LLAMA_BUILD_TESTS=OFF; see scripts/sycl-dockerfile-configure-check.sh, the
configure-only smoke test that reproduces this directly against the
Dockerfile's own flags rather than a hardcoded copy of them).

Two independent checks, both host-only, pure text assertions against the
CMakeLists.txt source (Python is test-only; its absence must not break a
clean SYCL configuration -- llama_test_pytest skips at run time, not
configure time, matching test-sycl-compute-buffer-fallback-source.py's own
convention):

(A) Every target_link_libraries(...) call in that file that names the bare
`ggml-sycl` target as one of the libraries to link -- as opposed to being the
target the call itself configures, which is ggml-sycl's own production
setup at the top of the file and already unconditionally correct -- must sit
under a condition that guarantees GGML_BACKEND_DL is off: either directly
inside an `if (...)` whose condition contains `NOT GGML_BACKEND_DL`
(ANDed with anything else), or inside the `else()` branch of an `if (...)`
whose ENTIRE condition is the single term `GGML_BACKEND_DL` (both shapes are
used in the file today -- the former is the overwhelming majority, the
latter is test-sycl-device-uuid-api's `if (GGML_BACKEND_DL) ... else()`).
A target_link_libraries call that does not resolve to either shape is
reported as a violation.

(B) Every add_executable(...) test target registered in that file must sit
under an `if (...)` whose condition mentions BUILD_TESTING (or
LLAMA_BUILD_TESTS) somewhere in its currently-open if-stack, so that
LLAMA_BUILD_TESTS=OFF (which never calls include(CTest), leaving
BUILD_TESTING unset/false -- see the top-level CMakeLists.txt) stops the
target from being added at all instead of silently ignoring
LLAMA_BUILD_TESTS. This check REPORTS A COUNT rather than a boolean: on the
source before the llama.cpp-9goq fix, 119 test targets fail it, the first at
~:950 (test-xmx-hardware-detect), which is exactly the "SYCL test targets
ignore LLAMA_BUILD_TESTS" half of the ticket. The fix wraps that whole
region in one outer `if (BUILD_TESTING)` (opened right after the last
statement that configures the ordinary ggml-sycl target, closed at end of
file), so the count must be exactly 0 on a tree carrying that fix.

Checks run against COMMENT-STRIPPED text so a positive structural check
cannot be fooled by prose that quotes a call the code does not actually
make -- same convention as test-sycl-compute-buffer-fallback-source.py.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKELISTS_PATH = ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"
CMAKELISTS_RAW = CMAKELISTS_PATH.read_text()

# ---------------------------------------------------------------------------
# Comment-stripping helper, copied verbatim (not imported) from
# test-sycl-compute-buffer-fallback-source.py, which documents this as the
# established convention in this file family: each source-gate test file is
# collected standalone by pytest with no shared conftest.py / importable
# helper module in tests/, so copying with attribution is the working,
# already-proven pattern here. CMake has no block comments, only `#` line
# comments, but the same lexeme-alternation shape (keep string literals,
# drop comments) is reused for the identical reason: an occurrence of
# "target_link_libraries" or "ggml-sycl" inside a `#` comment must not count
# as a real call. Parser limit: the string-literal branch
# (`(?:\\.|[^"\\\n])*`) excludes newlines, so a MULTI-LINE quoted CMake
# string containing a `#` would have that `#` onward mis-stripped as a line
# comment instead of kept as part of the string -- none exists in this file
# today.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|#[^\n]*",  # line comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove CMake `#` line comments; keep string literals verbatim."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] == '"':
            return tok
        return ""

    return _LEXEME_RE.sub(repl, src)


CMAKELISTS_CODE = strip_comments(CMAKELISTS_RAW)


# ---------------------------------------------------------------------------
# A minimal if/elseif/else/endif tracker plus a paren-balanced statement
# extractor. This is not a CMake evaluator -- it only tracks the handful of
# shapes this file actually uses for GGML_BACKEND_DL/BUILD_TESTING guards
# (verified against every `if (...GGML_BACKEND_DL...)` in the file at the
# time this test was written; see the module docstring). No guard in the file
# today uses `elseif`, and this tracker does not attempt to reason about one:
# an `elseif` REPLACES the enclosing frame's condition outright (the frame
# forgets the original `if`'s condition, exactly matching CMake's own
# semantics that only one of if/elseif/.../else is active at a time), so a
# target_link_libraries or add_executable call recorded while an elseif
# branch is active is judged purely against ITS condition text. If neither
# `NOT GGML_BACKEND_DL` nor `BUILD_TESTING`/`LLAMA_BUILD_TESTS` appears in
# that elseif's own condition, the call reads as unguarded -- fails closed,
# not silently guarded. Rather than trust that reasoning to stay correct as
# this file grows, a call this checker forms a verdict from (a bare
# ggml-sycl link candidate, or any add_executable) that was recorded with
# an elseif-derived frame anywhere in its stack raises immediately (see
# `_assert_not_via_elseif`; other tracked calls, e.g. ggml-sycl's own oneCCL
# link, are recorded with via_elseif=True and never judged), so a future elseif
# guard gets a human to check this file instead of a silent misread.
# ---------------------------------------------------------------------------

_STATEMENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _iter_statements(code: str):
    """Yield (name, args_text, start_line) for every CMake command
    invocation in `code`, in document order. `start_line` is 1-based and
    refers to the line the command keyword starts on. Paren depth is tracked
    character-by-character, SKIPPING parens inside quoted string literals
    (e.g. the "not built: ... (needs ...)" placeholder messages this file's
    own DL-guard else() branches print) so a stray paren in prose cannot
    desynchronize the statement boundary. This is not merely defensive: the
    parser must correctly bound a call whose message string contains
    parens, not merely happen to work because today's strings are
    paren-balanced by coincidence."""
    pos = 0
    line_no = 1
    for m in _STATEMENT_RE.finditer(code):
        if m.start() < pos:
            continue
        line_no += code.count("\n", pos, m.start())
        pos = m.start()
        name = m.group(1)
        depth = 1
        i = m.end()
        in_string = False
        while depth > 0:
            if i >= len(code):
                raise AssertionError(f"unbalanced parens starting at line {line_no} ({name})")
            c = code[i]
            if in_string:
                if c == "\\":
                    i += 1  # skip the escaped character too
                elif c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        args_text = code[m.end() : i - 1]
        yield name, args_text, line_no
        line_no += code.count("\n", pos, i)
        pos = i


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _Frame:
    __slots__ = ("cond", "in_else", "via_elseif")

    def __init__(self, cond: str, via_elseif: bool = False):
        self.cond = cond
        self.in_else = False
        self.via_elseif = via_elseif


def _dl_off_active(stack) -> bool:
    """True if some currently-open frame guarantees GGML_BACKEND_DL is
    false along this path -- either an `if` branch whose condition contains
    `NOT GGML_BACKEND_DL`, or an `else` branch of an `if` whose ENTIRE
    condition is exactly `GGML_BACKEND_DL` (no other ANDed terms).
    `stack` is a snapshot of (cond, in_else, via_elseif) tuples, not live
    _Frame objects -- see the note on `_walk`'s snapshot for why that
    distinction is load-bearing."""
    for cond, in_else, _via_elseif in stack:
        if not in_else and re.search(r"\bNOT\s+GGML_BACKEND_DL\b", cond):
            return True
        if in_else and cond == "GGML_BACKEND_DL":
            return True
    return False


def _build_testing_active(stack) -> bool:
    """True if some currently-open `if` (not `else`) frame's condition
    mentions BUILD_TESTING or LLAMA_BUILD_TESTS WITHOUT negating it -- i.e.
    that frame guarantees BUILD_TESTING is required for its branch to run at
    all. Mirrors `_dl_off_active`'s polarity handling rather than treating
    mere presence-in-condition as sufficient: an `else` branch is not a
    guarantee (a target reached only via `else()` of `if (BUILD_TESTING)` is
    reachable when BUILD_TESTING is false, the exact vacuous-pass shape this
    check exists to catch), and a condition containing
    `NOT BUILD_TESTING`/`NOT LLAMA_BUILD_TESTS` is rejected even in the `if`
    branch (that branch runs when BUILD_TESTING is off, the opposite of what
    this check needs). `stack` is the same (cond, in_else, via_elseif)
    snapshot as `_dl_off_active` above."""
    for cond, in_else, _via_elseif in stack:
        if in_else:
            continue
        if re.search(r"\bNOT\s+(BUILD_TESTING|LLAMA_BUILD_TESTS)\b", cond):
            continue
        if re.search(r"\b(BUILD_TESTING|LLAMA_BUILD_TESTS)\b", cond):
            return True
    return False


def _assert_not_via_elseif(stack, what: str, line: int) -> None:
    """Raise when `stack` contains a frame this checker's guard reasoning
    does not cover -- an elseif REPLACES the enclosing frame's condition
    (see the comment above `_STATEMENT_RE`), so a call recorded under one is
    judged purely on that elseif's own condition text. Call this only for a
    call the checker is about to actually rely on a verdict for (a genuine
    bare-ggml-sycl-link candidate, or any add_executable), not for every
    target_link_libraries call recorded by `_walk` -- most of those are
    ordinary production configuration (e.g. ggml-sycl's own
    `elseif (TARGET ccl)` oneCCL link) that no check here ever consults."""
    if any(via_elseif for _cond, _in_else, via_elseif in stack):
        raise AssertionError(
            f"{what} at line {line} sits under an elseif-derived frame this checker "
            "does not model past its own condition text -- update _dl_off_active/"
            "_build_testing_active (and this assertion) before trusting this "
            "checker's verdict on it"
        )


def _walk(code: str):
    """Replay the file's if/elseif/else/endif structure, yielding
    (kind, args_text, line, stack_snapshot) for every target_link_libraries
    and add_executable call, where stack_snapshot is a list of (cond,
    in_else) tuples for every frame open at that point (top of stack last).

    Snapshotting as plain tuples -- not a `list(stack)` of the live _Frame
    objects -- is load-bearing, not merely tidy: a _Frame is mutated in
    place by a LATER `else()` on the same block (`frame.in_else = True`), so
    a shallow copy of the stack list still aliases that same object. A
    target_link_libraries call recorded inside the `if` branch would then
    read back, once parsing reaches that block's `else()`, as if it had
    been inside the `else` branch all along -- silently misclassifying
    every guarded call in a block that also has an else (which is every
    __one of the three llama.cpp-9goq targets fixed by this ticket, since
    each of them gained a disabled-placeholder else()). Caught by running
    this checker against the real, already-fixed source and finding
    test-attn-host-pool reported as a violation despite being correctly
    guarded -- i.e. the checker failed against a known-GREEN case, the
    same shape as a positive-control failure elsewhere in this file family.

    Records a call found under an elseif-derived frame with via_elseif=True
    and returns it; the consumers that form a verdict from such a call raise
    (`_assert_not_via_elseif`) -- see the comment above `_STATEMENT_RE`."""
    stack = []
    results = []
    for name, args_text, line in _iter_statements(code):
        lname = name.lower()
        if lname == "if":
            stack.append(_Frame(_normalize_ws(args_text)))
        elif lname == "elseif":
            if not stack:
                raise AssertionError(f"elseif with no open if at line {line}")
            # New condition at the same nesting depth; reset the else flag.
            # via_elseif=True marks this frame so a call recorded under it
            # trips the assertion below rather than being silently judged
            # against only the elseif's own condition text.
            stack[-1] = _Frame(_normalize_ws(args_text), via_elseif=True)
        elif lname == "else":
            if not stack:
                raise AssertionError(f"else with no open if at line {line}")
            stack[-1].in_else = True
        elif lname == "endif":
            if not stack:
                raise AssertionError(f"endif with no open if at line {line}")
            stack.pop()
        elif lname in ("target_link_libraries", "add_executable"):
            # Snapshot via_elseif alongside (cond, in_else) so a consumer can
            # raise when it is about to judge a call this checker's guard
            # reasoning does not cover -- see the comment above
            # `_STATEMENT_RE`. Not raised HERE: most target_link_libraries
            # calls recorded under an elseif frame in this file are ordinary
            # production configuration (e.g. ggml-sycl's own oneCCL link,
            # `elseif (TARGET ccl)`) that neither check ever consults, so a
            # blanket raise here is a false positive against real, harmless
            # source -- caught the same way the aliasing bug on `_Frame` was:
            # this checker run against the real file raised on a call check A
            # was always going to `continue` past anyway (target ==
            # "ggml-sycl", i.e. ggml-sycl configuring itself, not a candidate
            # offender).
            results.append((lname, args_text, line, [(f.cond, f.in_else, f.via_elseif) for f in stack]))
    if stack:
        raise AssertionError(f"unbalanced if/endif: {len(stack)} still open at EOF")
    return results


_STATEMENTS = _walk(CMAKELISTS_CODE)


def _iter_bare_ggml_sycl_link_candidates(statements):
    """Yield (target_being_configured, line, stack) for every
    target_link_libraries(...) call that names the bare `ggml-sycl` target
    as a LINKED library -- not as the target-being-configured, i.e. not its
    own first argument, which is ggml-sycl's production setup and never a
    candidate offender. Shared by _find_bare_ggml_sycl_link_violations and
    _count_all_bare_ggml_sycl_links so the candidate-detection logic (and
    the elseif guard below) lives in exactly one place.

    Asserts (via `_assert_not_via_elseif`) rather than silently judging a
    candidate recorded under an elseif-derived frame -- scoped to genuine
    candidates only, not every target_link_libraries call `_walk` recorded
    (most of those are ordinary production configuration this checker never
    consults)."""
    for kind, args_text, line, stack in statements:
        if kind != "target_link_libraries":
            continue
        tokens = args_text.split()
        if not tokens:
            continue
        target_being_configured, *rest = tokens
        if target_being_configured == "ggml-sycl":
            continue  # ggml-sycl configuring itself; not a candidate offender.
        for tok in rest:
            if tok in ("PRIVATE", "PUBLIC", "INTERFACE"):
                continue
            if "$<" in tok:
                continue  # generator expression, e.g. $<TARGET_FILE:ggml-sycl>
            if tok == "ggml-sycl":
                _assert_not_via_elseif(stack, f"target_link_libraries({target_being_configured} ...)", line)
                yield target_being_configured, line, stack
                break


def _find_bare_ggml_sycl_link_violations(statements):
    """Every bare-ggml-sycl-link candidate (see
    _iter_bare_ggml_sycl_link_candidates) outside a GGML_BACKEND_DL-is-off
    guard."""
    return [
        (target, line)
        for target, line, stack in _iter_bare_ggml_sycl_link_candidates(statements)
        if not _dl_off_active(stack)
    ]


def _count_all_bare_ggml_sycl_links(statements):
    """Total bare-ggml-sycl-link candidates -- guarded AND unguarded
    together. A lower bound on this count is the sanity check that keeps
    test_no_bare_ggml_sycl_link_outside_a_dl_off_guard from passing having
    silently examined zero calls (a token-matching regression, a quoted
    "ggml-sycl" string that should not count, a variable-expanded library
    list, or a quote-parity desync inside one of the `[=[ ... ]=]`
    bracket-argument calls elsewhere in the file that `_iter_statements`
    does not model) -- `violations == []` is also true of an empty list."""
    return sum(1 for _ in _iter_bare_ggml_sycl_link_candidates(statements))


def _find_unguarded_test_targets(statements):
    """Every add_executable(...) whose currently-open if-stack does not
    mention BUILD_TESTING/LLAMA_BUILD_TESTS anywhere. Every add_executable is
    a candidate for this check (unlike check A, there is no "configuring
    itself" case to skip), so the elseif guard applies to all of them."""
    unguarded = []
    for kind, args_text, line, stack in statements:
        if kind != "add_executable":
            continue
        target = args_text.split()[0] if args_text.split() else "<unknown>"
        _assert_not_via_elseif(stack, f"add_executable({target} ...)", line)
        if not _build_testing_active(stack):
            unguarded.append((target, line))
    return unguarded


# ---------------------------------------------------------------------------
# Check A: bare `ggml-sycl` links must be DL-guarded.
# ---------------------------------------------------------------------------

# Shared by test_no_bare_ggml_sycl_link_outside_a_dl_off_guard's positive
# control and test_bare_ggml_sycl_link_check_has_a_mutation_witness below --
# one guarded call, moved outside its guard, so it is exactly one violation.
_UNGUARDED_LINK_SNIPPET = (
    "if (NOT GGML_BACKEND_DL)\n"
    "    add_executable(test-example test-example.cpp)\n"
    "endif()\n"
    "target_link_libraries(test-example PRIVATE ggml-base ggml ggml-sycl Threads::Threads)\n"
)


def test_no_bare_ggml_sycl_link_outside_a_dl_off_guard():
    # Lower-bound sanity count first: a checker that silently stopped seeing
    # any bare ggml-sycl links would also report violations == [] below, for
    # the wrong reason. The file carries 61 such calls today (guarded and
    # unguarded together); 55 gives headroom for future additions/removals
    # without making this brittle.
    total_links = _count_all_bare_ggml_sycl_links(_STATEMENTS)
    assert total_links >= 55, (
        f"only found {total_links} target_link_libraries(...) call(s) naming the bare "
        "ggml-sycl target (guarded or not) -- the file has 61 today. A count this low "
        "means the token match likely broke silently (a quoted \"ggml-sycl\" string "
        "wrongly counted or excluded, a variable-expanded library list, or a "
        "quote-parity desync inside one of the `[=[ ... ]=]` bracket-argument calls "
        "this parser does not model), and the assertion below may be passing having "
        "examined nothing."
    )

    violations = _find_bare_ggml_sycl_link_violations(_STATEMENTS)
    assert violations == [], (
        "target_link_libraries(...) linking the bare `ggml-sycl` MODULE target "
        "outside a `NOT GGML_BACKEND_DL` (or equivalent else-of-GGML_BACKEND_DL) "
        f"guard: {violations}. Under -DGGML_BACKEND_DL=ON this is a hard CMake "
        'configure error ("Target \\"ggml-sycl\\" of type MODULE_LIBRARY may not '
        'be linked into another target"), exactly the llama.cpp-9goq regression. '
        "Wrap the target's real link line in `if (NOT GGML_BACKEND_DL) ... else() "
        "<disabled placeholder add_test> endif()`, matching the other ~85 targets "
        "in this file that already do."
    )

    # Positive control, in the same test: the identical checker, given a copy
    # with exactly one such link moved outside its guard, must report exactly
    # one violation -- not zero (which would mean it stopped seeing links
    # entirely, the same failure mode the count above guards against) and not
    # more than one (double-counting).
    control_violations = _find_bare_ggml_sycl_link_violations(_walk(_UNGUARDED_LINK_SNIPPET))
    assert len(control_violations) == 1, (
        "positive control is broken: one un-guarded bare ggml-sycl link should "
        f"produce exactly 1 violation, got {control_violations}"
    )


def test_bare_ggml_sycl_link_check_has_a_mutation_witness():
    """Positive control, run through the IDENTICAL `_walk` /
    `_find_bare_ggml_sycl_link_violations` pipeline used against the real
    file: a bare ggml-sycl link inside `if (NOT GGML_BACKEND_DL)` must read
    clean, and the same call moved outside that guard must be reported.

    Deliberately synthetic rather than a string-mutation of the real
    5700-line file: an exact-substring mutation is brittle against
    reflowing (this file's own add_executable calls span multiple lines,
    which an earlier version of this witness got wrong), and it is the
    _checker_ this test needs to exercise, not this file's current
    formatting."""
    guarded_snippet = (
        "if (NOT GGML_BACKEND_DL)\n"
        "    add_executable(test-example test-example.cpp)\n"
        "    target_link_libraries(test-example PRIVATE ggml-base ggml ggml-sycl Threads::Threads)\n"
        "endif()\n"
    )
    assert _find_bare_ggml_sycl_link_violations(_walk(guarded_snippet)) == [], (
        "checker false-positives on a correctly DL-guarded link"
    )
    # Reuses _UNGUARDED_LINK_SNIPPET (module-level, above check A) -- the
    # same one test_no_bare_ggml_sycl_link_outside_a_dl_off_guard uses as its
    # own positive control, which already asserts len(violations) == 1; this
    # test additionally names the target to confirm WHICH call was flagged.
    violations = _find_bare_ggml_sycl_link_violations(_walk(_UNGUARDED_LINK_SNIPPET))
    assert any(target == "test-example" for target, _line in violations), (
        "mutation witness is broken: an un-guarded bare ggml-sycl link should have been reported"
    )

    # The else-of-GGML_BACKEND_DL shape (test-sycl-device-uuid-api's own
    # pattern, bare ggml-sycl linked only in the else() branch of a plain
    # `if (GGML_BACKEND_DL)`) must also read as guarded.
    else_guarded_snippet = (
        "if (GGML_BACKEND_DL)\n"
        "    target_link_libraries(test-example PRIVATE ggml ${CMAKE_DL_LIBS})\n"
        "else()\n"
        "    target_link_libraries(test-example PRIVATE ggml-base ggml-sycl ggml-cpu)\n"
        "endif()\n"
    )
    assert _find_bare_ggml_sycl_link_violations(_walk(else_guarded_snippet)) == []


# ---------------------------------------------------------------------------
# Check B: every test target must be reachable only under BUILD_TESTING.
# ---------------------------------------------------------------------------


def test_build_testing_gate_check_has_a_mutation_witness():
    """Positive control for check B, run through the identical `_walk` /
    `_find_unguarded_test_targets` pipeline: a target inside
    `if (BUILD_TESTING)` must read as guarded, and the same target with the
    wrap deleted must be reported -- proving the check is not vacuously
    green (e.g. because the stack-tracker silently treats an unrecognized
    structure as always-guarded). Synthetic rather than a string-mutation
    of the real file for the same reflow-fragility reason given in
    test_bare_ggml_sycl_link_check_has_a_mutation_witness above (this
    check's real wrap boundary sits next to comment lines that
    strip_comments removes from CMAKELISTS_CODE entirely, so a literal
    substring match against the stripped text is not even the right shape
    of witness)."""
    guarded_snippet = "if (BUILD_TESTING)\n    add_executable(test-example test-example.cpp)\nendif()\n"
    unguarded_snippet = "add_executable(test-example test-example.cpp)\n"

    assert _find_unguarded_test_targets(_walk(guarded_snippet)) == [], (
        "checker false-positives on a target correctly gated by BUILD_TESTING"
    )
    unguarded = _find_unguarded_test_targets(_walk(unguarded_snippet))
    assert any(target == "test-example" for target, _line in unguarded), (
        "mutation witness is broken: a test target outside BUILD_TESTING should have been reported"
    )

    # Two fail-open shapes the pre-fix _build_testing_active read as guarded
    # (mere presence of BUILD_TESTING in an open condition, ignoring negation
    # and in_else): a target under `if (NOT BUILD_TESTING)`, and a target
    # reached only through the `else()` of `if (BUILD_TESTING)`. Both are
    # reachable with BUILD_TESTING unset/false and must be reported unguarded.
    not_build_testing_snippet = (
        "if (NOT BUILD_TESTING)\n    add_executable(test-example test-example.cpp)\nendif()\n"
    )
    not_unguarded = _find_unguarded_test_targets(_walk(not_build_testing_snippet))
    assert any(target == "test-example" for target, _line in not_unguarded), (
        "mutation witness is broken: a target under `if (NOT BUILD_TESTING)` should have been reported unguarded"
    )

    else_branch_snippet = (
        "if (BUILD_TESTING)\nelse()\n    add_executable(test-example test-example.cpp)\nendif()\n"
    )
    else_unguarded = _find_unguarded_test_targets(_walk(else_branch_snippet))
    assert any(target == "test-example" for target, _line in else_unguarded), (
        "mutation witness is broken: a target under the else() of `if (BUILD_TESTING)` "
        "should have been reported unguarded"
    )


def test_build_testing_gate_is_clean_on_the_real_file():
    """The real file's own count must be exactly 0 once llama.cpp-9goq's
    file-spanning `if (BUILD_TESTING)` wrap is in place -- before the
    llama.cpp-9goq fix, 119 of the add_executable() calls after line ~472
    failed this same check (see the module docstring); this is the GREEN
    side of that RED baseline. Also sanity-checks that the parser is
    actually walking the whole file (a silently-empty statement list would
    make the `== []` assertion pass for the wrong reason)."""
    total_add_executable = sum(1 for kind, *_ in _STATEMENTS if kind == "add_executable")
    assert total_add_executable > 50, (
        f"only found {total_add_executable} add_executable() calls -- the parser "
        "likely stopped walking the file early rather than the file having " "few test targets"
    )
    unguarded = _find_unguarded_test_targets(_STATEMENTS)
    assert unguarded == [], (
        f"{len(unguarded)} add_executable() test target(s) in "
        "ggml/src/ggml-sycl/CMakeLists.txt are reachable with LLAMA_BUILD_TESTS=OFF "
        f"(BUILD_TESTING unset): {unguarded[:10]}"
        + (" ... (truncated)" if len(unguarded) > 10 else "")
    )


def test_the_three_llama_cpp_9goq_offenders_are_now_dl_guarded():
    """Named regression check for the exact three targets the ticket
    reported (test-attn-host-pool, test-attn-host-flash-attn-identity,
    test-sycl-expert-predictor-guard): each must have NO bare-ggml-sycl-link
    violation and must sit under a BUILD_TESTING-mentioning guard."""
    violation_targets = {t for t, _line in _find_bare_ggml_sycl_link_violations(_STATEMENTS)}
    for target in (
        "test-attn-host-pool",
        "test-attn-host-flash-attn-identity",
        "test-sycl-expert-predictor-guard",
    ):
        assert target not in violation_targets, f"{target} still links bare ggml-sycl outside a DL-off guard"

    unguarded_targets = {t for t, _line in _find_unguarded_test_targets(_STATEMENTS)}
    for target in (
        "test-attn-host-pool",
        "test-attn-host-flash-attn-identity",
        "test-sycl-expert-predictor-guard",
    ):
        assert target not in unguarded_targets, f"{target} is not reachable only under BUILD_TESTING"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
