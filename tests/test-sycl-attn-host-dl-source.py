"""Source contract for llama.cpp-n4ee: DL-independence for the attention
host-dispatch path, checked WITHOUT a GGML_BACKEND_DL=ON build.

Background: the SYCL module's DL contract ("SYCL module must be CPU-backend
independent", 806bd5dda/cd02ffda8) is enforced today by
tests/test-sycl-module-dependencies.py -- a POST_BUILD custom command that
only runs under `-DGGML_BACKEND_DL=ON` (ggml/src/ggml-sycl/CMakeLists.txt
~:108-116), independent of BUILD_TESTING. The workstation's ordinary DL=OFF
build NEVER runs it, which is exactly how the TKV-13 (B2) attention
host-dispatch work (2026-08-27/28) landed two regressions invisible for two
weeks: three ggml_backend_graph_compute() call sites
(ggml_sycl_dispatch_host_flash_attn_async, ggml_sycl_dispatch_host_flash_attn_sync,
ggml_sycl_dispatch_host_set_rows_sync) and the ggml_backend_cpu_init() call in
ggml_sycl_attn_host_cpu_backend() -- plus a THIRD occurrence this ticket's own
investigation turned up that the ticket text does not name:
ggml_get_type_traits_cpu() in ggml_sycl_dispatch_host_set_rows_sync's
inline-scatter hot path. All are genuinely unreachable at runtime under
GGML_BACKEND_DL (ggml_sycl_attn_host_cpu_backend() always returns nullptr
there, and every caller checks that before reaching these lines), but a call
INSTRUCTION referencing a forbidden symbol still appears in the compiled
object regardless of runtime reachability -- the RTLD_NOW-load audit sees the
symbol table, not a reachability proof -- so each one must be compiled out
under `#ifdef GGML_BACKEND_DL`/`#ifndef GGML_BACKEND_DL`, not merely left
behind a runtime check.

This gate closes the two-week blind spot by running the SAME forbidden-symbol
list as a pure TEXT scan over ggml-sycl.cpp, gated only on Python being
available -- no SYCL device, no DL build, no BUILD_TESTING. It tracks
#ifdef/#ifndef/#else/#endif nesting for GGML_BACKEND_DL and fails, naming
file:line, on any forbidden identifier found outside a region the nesting
guarantees is excluded from a GGML_BACKEND_DL build -- i.e. inside
`#ifndef GGML_BACKEND_DL` (not its `#else`, if any), or inside the `#else` of
`#ifdef GGML_BACKEND_DL`. It cannot replace test-sycl-module-dependencies.py
(that check also verifies genuine undefined-symbol linkage, which this text
scan does not model), but it converges on the same list without needing a
second GGML_BACKEND_DL=ON build to discover the next masked failure.

The forbidden-symbol list is extracted BY TEXT from
tests/test-sycl-module-dependencies.py's own `for forbidden in (...)` tuple --
never a second, independently-maintained copy that could silently drift from
it (that file is a standalone pytest/ctypes script with real subprocess/nm
calls at import time, so this gate parses its source rather than importing
it).

Checks run against COMMENT-AND-STRING-STRIPPED text (comments and string/char
literals removed, line count preserved) so a positive structural check cannot
be fooled by prose that quotes a call the code does not actually make -- the
existing comment block at ggml-sycl.cpp ~:80121 does exactly this, naming
both ggml_backend_cpu_init() and ggml_backend_graph_compute() in prose to
explain the design, and must not itself be read as a violation.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Plain file I/O, not the codescout index/search_text -- ggml-sycl.cpp is
# ~100k+ lines and CLAUDE.md documents that index (and search_text's live
# scan) as blind/oversized for this specific file, so a tool-assisted search
# here would silently miss real occurrences. Same convention as
# test-sycl-compute-buffer-fallback-source.py.
GGML_SYCL_CPP_PATH = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
GGML_SYCL_CPP_RAW = GGML_SYCL_CPP_PATH.read_text()

MODULE_DEPS_PATH = ROOT / "tests/test-sycl-module-dependencies.py"
MODULE_DEPS_SRC = MODULE_DEPS_PATH.read_text()

# ---------------------------------------------------------------------------
# Comment/string stripping, copied verbatim (not imported) from
# test-sycl-compute-buffer-fallback-source.py, itself copied from
# test-sycl-ubatch-ring-replan-source.py / test-sycl-nonfa-attn-scratch-guard-
# source.py -- the established convention in this file family (no shared
# conftest.py / importable helper module in tests/, so each source-gate file
# is collected standalone by pytest). Block comments are replaced by their
# own newline count so LINE NUMBERS survive stripping -- this gate reports
# file:line, so that property is load-bearing here, not merely tidy.
# ---------------------------------------------------------------------------
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments; keep string/char literals verbatim;
    preserve line numbering."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP_RAW)

# ---------------------------------------------------------------------------
# A SECOND, deliberately narrower pass, applied on top of strip_comments and
# used ONLY for the forbidden-symbol occurrence search below: blank out the
# CONTENTS of string/char literals (quotes and length kept, so line/column
# positions are unaffected) rather than keeping them verbatim. This is a
# real file property, not a hypothetical -- the WARN messages this ticket
# touches used to read `"[ATTN-HOST] ggml_backend_cpu_init() failed; ..."`,
# i.e. a forbidden symbol's name spelled out as prose inside a string
# literal that makes no call at all (llama.cpp-n4ee reworded those three
# specifically so they no longer say this, but a future WARN/ERROR message
# quoting a forbidden symbol by name is exactly the kind of prose this
# family's shared strip_comments deliberately does NOT strip, since most
# gates in this file family need string content preserved for unrelated
# reasons). Symbol-reference detection is the one thing this file does that
# genuinely needs string bodies gone too, so it gets its own pass rather
# than changing the shared convention for every other check.
# ---------------------------------------------------------------------------
_STRING_BODY_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # double-quoted string literal
    r"|'(?:\\.|[^'\\\n])*'",  # char/single-quoted literal
)


def blank_string_bodies(src: str) -> str:
    """Replace string/char literal CONTENTS with spaces, keeping the
    delimiting quotes and overall length (so this must run on
    already comment-stripped text, else a `//` or `/* */` inside a string
    would already have been mishandled upstream -- it is not)."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return tok[0] + (" " * (len(tok) - 2)) + tok[-1]

    return _STRING_BODY_RE.sub(repl, src)


GGML_SYCL_CPP_SEARCH_TEXT = blank_string_bodies(GGML_SYCL_CPP_CODE)


# ---------------------------------------------------------------------------
# Forbidden-symbol list: extracted BY TEXT from
# test-sycl-module-dependencies.py, never re-typed here.
# ---------------------------------------------------------------------------
def _extract_forbidden_symbols(module_deps_src: str):
    """Pull the string literals out of test-sycl-module-dependencies.py's
    `for forbidden in (...):` tuple. Reads the SOURCE TEXT rather than
    importing that module -- it is a standalone pytest/ctypes script that
    does real subprocess/nm/readelf work at import time (see its own
    top-level code), not an importable library, matching every other file in
    this family's "no shared helper module" convention.

    A CHARACTER SCANNER, not a `\\).*?\\)\\s*:` regex -- the tuple spans
    several lines and carries an explanatory `#` comment INSIDE it (the
    ggml_backend_cpu_init entry added by llama.cpp-n4ee), and that comment's
    own prose can legitimately contain a `):` substring (e.g. "(llama.cpp-
    n4ee): a DL module ..."). A non-greedy regex anchored on the first `):`
    it finds stops at THAT one, silently truncating the tuple before ever
    reaching the entries quoted after the comment -- caught empirically: an
    earlier regex-based version of this function returned a list missing
    ggml_backend_cpu_init with no error, and this scanner exists because
    that failure mode is silent, not because the regex was merely
    inelegant. Tracks quote state so a `#` inside a string is not mistaken
    for a comment start, and paren depth so the scan ends at the tuple's
    OWN closing paren rather than any other."""
    start = module_deps_src.find("for forbidden in (")
    if start == -1:
        raise AssertionError(
            "could not find `for forbidden in (` in tests/test-sycl-module-"
            "dependencies.py -- its shape changed; update _extract_forbidden_symbols"
        )
    i = start + len("for forbidden in (")
    n = len(module_deps_src)
    depth = 1
    names = []
    in_string = False
    string_char = ""
    buf = []
    while depth > 0:
        if i >= n:
            raise AssertionError(
                "ran off the end of test-sycl-module-dependencies.py while scanning "
                "the forbidden-symbol tuple -- unbalanced parens or an unterminated string"
            )
        c = module_deps_src[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == string_char:
                names.append("".join(buf))
                buf = []
                in_string = False
            else:
                buf.append(c)
        elif c in "\"'":
            in_string = True
            string_char = c
        elif c == "#":
            nl = module_deps_src.find("\n", i)
            i = n if nl == -1 else nl
            continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if not names:
        raise AssertionError(
            "found the `for forbidden in (...):` tuple in "
            "test-sycl-module-dependencies.py but extracted zero symbol names "
            "from it -- the scanner broke"
        )
    return names


FORBIDDEN_SYMBOLS = _extract_forbidden_symbols(MODULE_DEPS_SRC)

_FORBIDDEN_RE = {name: re.compile(r"\b" + re.escape(name) + r"\b") for name in FORBIDDEN_SYMBOLS}


# ---------------------------------------------------------------------------
# #ifdef/#ifndef/#else/#endif nesting tracker for GGML_BACKEND_DL. Deliberately
# narrow -- like test-sycl-cmake-dl-link-source.py's if/elseif/else/endif
# tracker for CMake, this is not a preprocessor evaluator. It recognizes
# exactly the shapes this file uses today (verified by the module-level grep
# in the llama.cpp-n4ee investigation: only `#ifdef GGML_BACKEND_DL` and
# `#ifndef GGML_BACKEND_DL`, both with a plain `#else`, no `#elif`) plus their
# `#if defined(...)` / `#if !defined(...)` equivalents in case a future edit
# uses that spelling. A frame this tracker cannot classify (an unrelated
# macro, or a compound `#if defined(GGML_BACKEND_DL) && X`) is recorded as
# "other": it neither counts as excluding DL nor raises by itself, but a
# violation judged while ONLY an "other"/"elif_derived" frame is open still
# gets a positive verdict (unexcluded == violation), which is the fail-closed
# direction. An "elif_derived" frame specifically raises via
# `_assert_not_via_elif` before a violation verdict is trusted from under it
# (mirrors test-sycl-cmake-dl-link-source.py's `_assert_not_via_elseif`) --
# not needed by the real file today, but kept so a future `#elif` gets a
# human to check this tracker instead of a silent misread.
# ---------------------------------------------------------------------------

_IFDEF_RE = re.compile(r"^\s*#\s*ifdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_IF_RE = re.compile(r"^\s*#\s*if\s+(.*)$")
_ELIF_RE = re.compile(r"^\s*#\s*elif\b")
_ELSE_RE = re.compile(r"^\s*#\s*else\b")
_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")

_DEFINED_DL_RE = re.compile(r"^\(?\s*defined\s*\(\s*GGML_BACKEND_DL\s*\)\s*\)?$")
_NOT_DEFINED_DL_RE = re.compile(r"^\(?\s*!\s*defined\s*\(\s*GGML_BACKEND_DL\s*\)\s*\)?$")


def _classify_if_condition(cond: str) -> str:
    cond = cond.strip()
    if _DEFINED_DL_RE.match(cond):
        return "ifdef_dl"
    if _NOT_DEFINED_DL_RE.match(cond):
        return "ifndef_dl"
    return "other"


class _Frame:
    __slots__ = ("kind", "in_else")

    def __init__(self, kind: str):
        self.kind = kind  # "ifdef_dl" | "ifndef_dl" | "other" | "elif_derived"
        self.in_else = False


def _dl_excluded_active(stack) -> bool:
    """True if some currently-open frame guarantees this position is NOT
    compiled into a GGML_BACKEND_DL=ON build -- either the `if`-branch (not
    `else`) of an `#ifndef GGML_BACKEND_DL`, or the `#else`-branch of an
    `#ifdef GGML_BACKEND_DL`. `stack` is a snapshot of (kind, in_else) tuples,
    not live _Frame objects -- see `_walk_lines`'s note on why a shallow copy
    of live frames would alias a later mutation, the same aliasing hazard
    test-sycl-cmake-dl-link-source.py's own `_walk` documents and was caught
    by."""
    for kind, in_else in stack:
        if kind == "ifndef_dl" and not in_else:
            return True
        if kind == "ifdef_dl" and in_else:
            return True
    return False


def _assert_not_via_elif(stack, what: str, line: int) -> None:
    if any(kind == "elif_derived" for kind, _in_else in stack):
        raise AssertionError(
            f"{what} at ggml-sycl.cpp:{line} sits under an #elif-derived frame "
            "this tracker does not model past its own condition text -- update "
            "_classify_if_condition/_dl_excluded_active (and this assertion) "
            "before trusting this gate's verdict on it"
        )


def _walk_lines(stripped_code: str):
    """Yield (line_no, line_text, stack_snapshot) for every physical line of
    `stripped_code`, where stack_snapshot is the list of (kind, in_else)
    tuples open AFTER applying any directive on that line (so content on a
    directive line itself is judged by the state that directive establishes,
    which is harmless since a directive line carries no C++ call -- and
    content on every line AFTER a directive sees its effect immediately, one
    line late is not on offer since #ifdef/#endif are always the sole token
    on their line in this file).

    Snapshots are plain (kind, in_else) tuples, not references to the live
    _Frame objects, for the identical reason
    test-sycl-cmake-dl-link-source.py's `_walk` snapshots defensively: a
    _Frame is mutated in place by a LATER `#else` on the same block
    (`frame.in_else = True`), so a bare `list(stack)` of live objects would
    let an already-yielded snapshot's frame retroactively read as
    else-branched once parsing reaches that block's `#else` -- silently
    misclassifying every guarded line in a block that also has an else."""
    stack = []
    for i, line in enumerate(stripped_code.split("\n"), start=1):
        m = _IFDEF_RE.match(line)
        if m:
            stack.append(_Frame("ifdef_dl" if m.group(1) == "GGML_BACKEND_DL" else "other"))
        else:
            m = _IFNDEF_RE.match(line)
            if m:
                stack.append(_Frame("ifndef_dl" if m.group(1) == "GGML_BACKEND_DL" else "other"))
            else:
                m = _IF_RE.match(line)
                if m:
                    stack.append(_Frame(_classify_if_condition(m.group(1))))
                elif _ELIF_RE.match(line):
                    if not stack:
                        raise AssertionError(f"#elif with no open #if at ggml-sycl.cpp:{i}")
                    stack[-1] = _Frame("elif_derived")
                elif _ELSE_RE.match(line):
                    if not stack:
                        raise AssertionError(f"#else with no open #if/#ifdef/#ifndef at ggml-sycl.cpp:{i}")
                    stack[-1].in_else = True
                elif _ENDIF_RE.match(line):
                    if not stack:
                        raise AssertionError(f"#endif with no open #if/#ifdef/#ifndef at ggml-sycl.cpp:{i}")
                    stack.pop()
        yield i, line, [(f.kind, f.in_else) for f in stack]
    if stack:
        raise AssertionError(f"unbalanced #if*/#endif in ggml-sycl.cpp: {len(stack)} still open at EOF")


def find_dl_unsafe_forbidden_occurrences(stripped_code: str, forbidden_names):
    """Every (line_no, name) for a forbidden identifier occurrence NOT inside
    a region `_dl_excluded_active` guarantees is excluded from a
    GGML_BACKEND_DL build."""
    violations = []
    for line_no, line_text, stack in _walk_lines(stripped_code):
        if _dl_excluded_active(stack):
            continue
        for name in forbidden_names:
            if _FORBIDDEN_RE[name].search(line_text):
                _assert_not_via_elif(stack, f"forbidden symbol {name}", line_no)
                violations.append((line_no, name))
    return violations


# ---------------------------------------------------------------------------
# The real check.
# ---------------------------------------------------------------------------


def test_forbidden_symbol_list_extraction_is_sane():
    """Sanity/mutation-witness for `_extract_forbidden_symbols`: a checker
    that silently stopped extracting anything would make the real-file
    assertion below pass having examined zero symbols. The list carries 11
    names today (10 original + ggml_backend_cpu_init added by llama.cpp-n4ee);
    also pins that both of THIS ticket's symbols are actually present in the
    extracted set, since a regex that quietly dropped one would make this
    gate blind to exactly the regression it exists to catch."""
    assert len(FORBIDDEN_SYMBOLS) >= 10, (
        f"only extracted {len(FORBIDDEN_SYMBOLS)} forbidden symbol name(s) from "
        "test-sycl-module-dependencies.py -- the extraction regex likely broke "
        "silently and the real-file check below may be passing having examined "
        "nothing"
    )
    for name in ("ggml_backend_graph_compute", "ggml_backend_cpu_init", "ggml_get_type_traits_cpu"):
        assert name in FORBIDDEN_SYMBOLS, f"{name} missing from the extracted forbidden-symbol list"


def test_dl_exclusion_tracker_has_a_mutation_witness():
    """Positive control (unguarded call -> 1 violation) and negative control
    (guarded call -> 0 violations), run through the IDENTICAL
    find_dl_unsafe_forbidden_occurrences pipeline used against the real file.
    Also proves BOTH bounds of an excluded region are honored -- a call
    BEFORE the region's opening directive, and a call AFTER its closing
    #endif, must each be reported, not just the call inside it left alone."""
    unguarded_snippet = "    ggml_backend_graph_compute(cpu_backend, graph);\n"
    unguarded = find_dl_unsafe_forbidden_occurrences(unguarded_snippet, FORBIDDEN_SYMBOLS)
    assert unguarded == [(1, "ggml_backend_graph_compute")], (
        f"mutation witness is broken: an unguarded forbidden call should be exactly "
        f"one violation, got {unguarded}"
    )

    ifndef_guarded_snippet = "#ifndef GGML_BACKEND_DL\n    ggml_backend_graph_compute(cpu_backend, graph);\n#endif\n"
    assert find_dl_unsafe_forbidden_occurrences(ifndef_guarded_snippet, FORBIDDEN_SYMBOLS) == [], (
        "checker false-positives on a call correctly guarded by #ifndef GGML_BACKEND_DL"
    )

    ifdef_else_guarded_snippet = (
        "#ifdef GGML_BACKEND_DL\n"
        "    GGML_UNUSED(cpu_backend);\n"
        "#else\n"
        "    ggml_backend_graph_compute(cpu_backend, graph);\n"
        "#endif\n"
    )
    assert find_dl_unsafe_forbidden_occurrences(ifdef_else_guarded_snippet, FORBIDDEN_SYMBOLS) == [], (
        "checker false-positives on a call correctly guarded by the #else of #ifdef GGML_BACKEND_DL"
    )

    # A call inside the #ifdef-GGML_BACKEND_DL branch itself (not its #else)
    # is DL-only code, not DL-excluded code -- must still be reported.
    ifdef_branch_snippet = "#ifdef GGML_BACKEND_DL\n    ggml_backend_graph_compute(cpu_backend, graph);\n#endif\n"
    ifdef_branch_violations = find_dl_unsafe_forbidden_occurrences(ifdef_branch_snippet, FORBIDDEN_SYMBOLS)
    assert ifdef_branch_violations == [(2, "ggml_backend_graph_compute")], (
        f"a call inside the #ifdef GGML_BACKEND_DL branch itself (compiled INTO a "
        f"DL build) must be reported, got {ifdef_branch_violations}"
    )

    # Both bounds of an excluded region: a call BEFORE the #ifndef opens, and
    # a call AFTER its #endif closes, must both be reported even though a
    # third call in between (correctly inside the region) is clean.
    both_bounds_snippet = (
        "    ggml_backend_graph_compute(before, graph);\n"
        "#ifndef GGML_BACKEND_DL\n"
        "    ggml_backend_graph_compute(inside, graph);\n"
        "#endif\n"
        "    ggml_backend_graph_compute(after, graph);\n"
    )
    both_bounds_violations = find_dl_unsafe_forbidden_occurrences(both_bounds_snippet, FORBIDDEN_SYMBOLS)
    assert sorted(both_bounds_violations) == [(1, "ggml_backend_graph_compute"), (5, "ggml_backend_graph_compute")], (
        f"expected exactly the before-region (line 1) and after-region (line 5) calls "
        f"to be reported, and the inside-region call (line 3) to read clean, got "
        f"{both_bounds_violations}"
    )

    # A comment or string literal quoting a forbidden name must not count --
    # the real file's ~:80121 block does exactly this in prose.
    comment_snippet = (
        "// Uses ggml_backend_cpu_init()/ggml_backend_graph_compute() -- the standard\n"
        '    const char * msg = "ggml_backend_graph_compute is forbidden here";\n'
    )
    assert find_dl_unsafe_forbidden_occurrences(blank_string_bodies(strip_comments(comment_snippet)), FORBIDDEN_SYMBOLS) == [], (
        "checker false-positives on a comment/string literal merely naming a forbidden symbol"
    )


def test_unbalanced_directives_raise():
    """A dangling #ifndef or a stray #endif must raise rather than be
    silently absorbed -- an absorbed imbalance would desync every
    exclusion-region verdict for the rest of the file."""
    with pytest.raises(AssertionError):
        list(_walk_lines("#ifndef GGML_BACKEND_DL\n    ggml_backend_graph_compute(a, b);\n"))
    with pytest.raises(AssertionError):
        list(_walk_lines("#endif\n"))


def test_no_forbidden_symbol_outside_a_dl_excluded_region():
    """The real check: ggml-sycl.cpp must carry zero forbidden-symbol
    occurrences outside a region the #ifdef/#ifndef nesting guarantees is
    excluded from a GGML_BACKEND_DL=ON build. This is the DL=OFF-reachable
    equivalent of tests/test-sycl-module-dependencies.py's POST_BUILD audit,
    and is what would have caught the llama.cpp-n4ee regression (three
    ggml_backend_graph_compute call sites, one ggml_backend_cpu_init call,
    and one ggml_get_type_traits_cpu call) without needing a
    GGML_BACKEND_DL=ON build to discover it."""
    violations = find_dl_unsafe_forbidden_occurrences(GGML_SYCL_CPP_SEARCH_TEXT, FORBIDDEN_SYMBOLS)
    assert violations == [], (
        f"{len(violations)} forbidden CPU-backend symbol reference(s) in "
        "ggml/src/ggml-sycl/ggml-sycl.cpp are not excluded from a "
        f"GGML_BACKEND_DL=ON build: {violations[:10]}"
        + (" ... (truncated)" if len(violations) > 10 else "")
        + ". Wrap each call site in `#ifndef GGML_BACKEND_DL ... #else ... #endif` "
        "(or the whole function in `#ifdef GGML_BACKEND_DL ... #else ... #endif` if "
        "the entire body is CPU-backend-only), matching "
        "ggml_sycl_attn_host_cpu_backend()'s own pattern. Do not allowlist the "
        "symbol in tests/test-sycl-module-dependencies.py instead."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
