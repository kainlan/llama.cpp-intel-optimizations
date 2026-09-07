#!/usr/bin/env python3
"""Source gate for llama.cpp-5q1r (plan task S6, following on plan task S2 /
llama.cpp-2x3m): every SYCL GPU test whose own main() pins
ONEAPI_DEVICE_SELECTOR via setenv() must do so either through the shared
re-exec fallback (tests/sycl-selector-fallback.hpp,
sycl_test_selector_fallback()) or through the canonical inline re-exec block
S2 landed first (759b5647d) -- `if (!getenv("ONEAPI_DEVICE_SELECTOR")) {
setenv(...); execv("/proc/self/exe", argv); ...}` -- and that call (or block)
must be the FIRST real statement inside main(), per the helper's own
contract ("Call this as the FIRST statement of main ... anything run before
this call runs twice"). Never a bespoke `setenv()`-only copy with no execv;
never BOTH the helper call and a leftover bespoke setenv (a half-finished
port reads as a real fix); never an inline block sitting beside a SEPARATE,
stray bare setenv elsewhere in the same file (the same half-finished shape,
one level over); never buried after other code.

WHY THIS GATE EXISTS: a plain `setenv("ONEAPI_DEVICE_SELECTOR", ...)` inside
main() is a NO-OP in this fork. Every SYCL test here links ggml-sycl, which
links oneCCL for its tensor-parallelism ALL_REDUCE path
(ggml/src/ggml-sycl/CMakeLists.txt, GGML_SYCL_ONECCL, default ON), and
oneCCL's libccl.so.1 carries a static initializer that constructs a
sycl::event at process LOAD time -- before main() runs -- which makes
libsycl read and memoize ONEAPI_DEVICE_SELECTOR immediately. A bare
(non-ctest) invocation of an affected test therefore silently enumerates
every device, including the integrated GPU whose global_mem_size reports
231.7 GB of system RAM as phantom "VRAM" (llama.cpp-403s), with nothing in
the test's own output to say so. The fix is to re-exec the process
(setenv() then execv("/proc/self/exe", argv)) so the child's libccl
initializer observes the variable already set -- which only works if
nothing runs (and re-runs, on the second pass) before it; see
tests/sycl-selector-fallback.hpp for the full rationale and the guards
(re-exec only when setenv() itself succeeded, warn-and-continue rather than
exit(77) on failure).

HISTORY (each round found a gap the previous round's evidence did not
cover):
  RED (before the llama.cpp-5q1r port): ~20 sibling tests called the no-op
  `setenv()`-only form directly in main(), and two more
  (test-sycl-mmvq-q8-0-soa-numerics.cpp, test-sycl-fattn-tile-d512-decode.cpp)
  had already grown their OWN correct-but-independent execv() copy (landed by
  plan task S2) -- both counted as offenders in the first version of this
  gate, since the point was exactly one implementation, not several correct
  ones.
  Spec round 1 (rev-5q1r-spec-1, c-k45u) found two more offenders this gate
  could not even see (ggml/src/ggml-sycl/tests/test-cross-model-weight-
  usage.cpp, ggml/src/ggml-sycl/tests/test-canonical-checksum-owner-scope.cpp
  -- live, ctest-registered targets in ggml/src/ggml-sycl/CMakeLists.txt),
  one exclusion that was wrong (tests/test-sycl-compute-buffer-extra-reuse.cpp
  IS registered on master, in ggml/src/ggml-sycl/CMakeLists.txt, not
  tests/CMakeLists.txt), and a fail-open on a half-finished helper port
  (helper call plus a leftover bespoke setenv passed as clean).
  Spec round 2 (rev-5q1r-spec-2, c-xicg) found three more gaps, all fixed
  here: (a) nothing checked the helper/inline call was actually the FIRST
  statement of main, only that it existed somewhere in the file; (b) the
  inline-block acceptance check accepted a file with one correct block AND
  a separate stray bare setenv, the same half-finished shape as (a) in
  round 1 but for the inline form; (c) the scan was non-recursive, so *.cpp
  files in subdirectories of either scan root (tests/peg-parser,
  tests/e1-rca, tests/sycl-canary, tests/sycl-alloc-policy-fixtures/*, ...)
  were invisible to it.
  Quality round 1 (rev-5q1r-quality-1, c-h331) found the (a) fix from round
  2 itself failed open (any first statement merely starting with "if"
  passed) and a backslash-continuation gap in the preprocessor-line
  skipper that a later round's own fix attempt initially missed reproducing
  correctly.
  Quality round 3 (rev-5q1r-quality-3, c-autq) found the (a)/found-relevant-
  main check was FILE-scoped rather than per-offending-call: a stray brace
  inside a string literal could mis-bound one main() body early enough to
  ORPHAN its real offending call (landing in neither that truncated body
  nor any other), while an unrelated, even textually-dead (e.g. an `#if
  0`-guarded) second "int main(...)" elsewhere in the same file happened to
  satisfy the per-body check -- reading the whole file compliant despite the
  orphaned offender. See _find_main_bodies' docstring for the enforced fix.
  Quality round 4 (rev-5q1r-quality-4, c-q3hv) found the mirror case round
  3's own fix introduced: a stray `{` (too-LATE, not too-early) can push
  the brace-scan depth high enough that main()'s real closing brace only
  returns it to 1, so the scan runs to EOF and yields a body spanning the
  WHOLE REST OF THE FILE -- a later, genuinely unrelated free function's
  helper call then falls "inside" that corrupted span and satisfies round
  3's own outside-span check on bad data, reading compliant. Closed by
  reporting whether each main()'s scan actually balanced and treating an
  unbalanced one as its own offender rather than trusting a to-EOF span;
  see _find_main_bodies' docstring.
GREEN (after every round above): every in-scope file calls the shared
helper as the first statement of main, or -- for a sibling mid-review on
another branch -- may still legitimately carry the canonical inline block
as its ENTIRE handling of the variable, also as the first statement.

SCOPE: this gate scans TWO directories, RECURSIVELY -- tests/ and
ggml/src/ggml-sycl/tests/, including every subdirectory of each -- for
every *.cpp file containing a literal `setenv("ONEAPI_DEVICE_SELECTOR"`
call or a call to the shared helper in its own source, minus the files in
EXCLUDED_FILES below (keyed by path relative to the repo root, since the
two directories could in principle share a basename). This is deliberately
a superset of the `test-sycl-*.cpp` glob, of a single directory, and of a
top-level-only listing: the census this gate encodes (see llama.cpp-5q1r)
was done with a plain `grep -l` across all of tests/*.cpp, several
offenders there do not carry the `test-sycl-` prefix (e.g.
tests/test-layout-bytes.cpp), spec round 1 found live offenders outside
tests/ entirely, and spec round 2 found the scan missed nested
subdirectories (none of which currently carry the pattern, but a future
file there should not be invisible by construction).

ACCEPTED FORMS (a file that setenv()s ONEAPI_DEVICE_SELECTOR or calls the
helper is NOT an offender only if ALL of the following hold):
  1. Its ENTIRE handling of the variable is exactly one of:
       (a) a call to the shared helper, sycl_test_selector_fallback(...), or
       (b) the canonical inline block -- EVERY setenv("ONEAPI_DEVICE_SELECTOR"
           call in the file must be followed, within a short window, by
           execv("/proc/self/exe" (this is what plan task S2 landed first,
           and what a sibling test on another, not-yet-merged branch may
           still legitimately carry; consolidating it onto the shared
           helper is a fine follow-up, not a hard gate condition).
     Combining (a) and (b) in the same file, or having a stray bare setenv
     alongside a correct instance of either, is a half-finished port and an
     offender either way.
  2. That call (or the inline block's guarding `if`) is the FIRST real
     statement inside main() -- the first thing found after main()'s
     opening brace once blank lines, `//` comments, `/* */` blocks, and
     preprocessor lines (`#if`, `#endif`, `#else`, ...) are skipped. A file
     where the pattern exists somewhere in the source but not inside any
     main() body at all also fails this: it cannot be main()'s first
     statement if it is not in main() at all.

This gate reads SOURCE TEXT only -- no compiler, no SYCL device.
"""

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
GGML_SYCL_TESTS_DIR = REPO_ROOT / "ggml" / "src" / "ggml-sycl" / "tests"
SCAN_DIRS = (TESTS_DIR, GGML_SYCL_TESTS_DIR)

TESTS_CMAKE = TESTS_DIR / "CMakeLists.txt"
GGML_SYCL_CMAKE = REPO_ROOT / "ggml" / "src" / "ggml-sycl" / "CMakeLists.txt"
CMAKE_FILES = (TESTS_CMAKE, GGML_SYCL_CMAKE)

SETENV_PATTERN = re.compile(r'setenv\(\s*"ONEAPI_DEVICE_SELECTOR"')
EXECV_PATTERN = re.compile(r'execv\(\s*"/proc/self/exe"')
HELPER_CALL_PATTERN = re.compile(r"sycl_test_selector_fallback\s*\(")
# The canonical inline block's guard, both accepted spellings: `!getenv(...)`
# and `!std::getenv(...)`. Quality round 1 (rev-5q1r-quality-1, c-h331
# should-fix 1) found the earlier `remainder.lstrip(...).startswith("if")`
# check accepted ANY first statement beginning with the two letters "if" --
# `if (want_debug()) { enable_debug(); }` or a bare `iface_init();` (which
# merely starts with the same two characters) both read as compliant as long
# as the canonical block appeared somewhere later in the 500-char window.
INLINE_GUARD_PATTERN = re.compile(r'if\s*\(\s*!\s*(?:std::)?getenv\s*\(\s*"ONEAPI_DEVICE_SELECTOR"')
MAIN_WITH_ARGV_PATTERN = re.compile(r"int\s+main\s*\([^)]*char\s*\*\*\s*argv[^)]*\)")
MAIN_SIGNATURE_PATTERN = re.compile(r"int\s+main\s*\([^)]*\)\s*\{")

_WHITESPACE = re.compile(r"\s+")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# A preprocessor directive line, following backslash-newline continuations
# (a multi-line `#define X(a) \` in main()'s preamble must be skipped whole,
# not just its first physical line -- quality round 1, c-h331 nit 4).
#
# Quality round 2 (rev-5q1r-quality-2, c-dhg6) found the first attempt at
# this, `#[^\n]*(?:\\\n[^\n]*)*`, was a no-op for its own stated purpose: the
# greedy `[^\n]*` consumes the trailing backslash on the first line (a
# backslash is not `\n`), so the match already succeeds before the
# continuation alternative is ever tried, and nothing forces a backtrack.
# `_PP_LINE.match("#define X(a) \\\n    do_stuff(a)\n")` returned only
# `"#define X(a) \\"`.  The fix processes character-by-character with the
# continuation as its own alternative, tried FIRST, so a literal `\` right
# before a newline is consumed together with that newline instead of ending
# the match:
_PP_LINE = re.compile(r"#(?:\\\n|[^\n])*")

# Files that setenv("ONEAPI_DEVICE_SELECTOR" in their own main() but are
# deliberately NOT part of this port (llama.cpp-5q1r, lead guidance c-kjzk),
# keyed by path relative to the repo root (POSIX separators). Each entry is
# the reason, not a rubber stamp -- re-check it before adding another name,
# and see test_excluded_files_are_still_accounted_for, which makes an
# exclusion self-expire the moment its stated reason stops being true.
EXCLUDED_FILES = {
    # THE DISCRIMINATING REASON (quality round 1, c-h331 nit 6: an earlier
    # version of this comment led with "not registered in either
    # CMakeLists.txt", which does NOT discriminate -- tests/test-sycl-
    # compute-buffers.cpp and tests/test-sycl-pointer-types.cpp are equally
    # unregistered and are both ported and gated. What actually justifies
    # excluding THIS file is its mechanism): most of this file's
    # ONEAPI_DEVICE_SELECTOR occurrences are inside subprocess COMMAND
    # STRINGS (`std::string(...) + binary + ...`) that set the variable for
    # a CHILD llama-cli process it shells out to via run_command() -- a
    # fresh process with its own libccl load, immune to this process's
    # memoization. Only its own top-level main() setenv() shares the bug
    # shape this gate is about. It is ALSO not registered in either
    # CMakeLists.txt (`grep -rn "test-tiled-weight-loading"
    # tests/CMakeLists.txt ggml/src/ggml-sycl/CMakeLists.txt` returns
    # nothing) -- dead source, nothing to build or gate -- but that fact
    # alone would not be enough to justify excluding it, since plenty of
    # ported, gated files share it.
    "tests/test-tiled-weight-loading.cpp",
}


def _iter_source_files():
    files = []
    for d in SCAN_DIRS:
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.cpp")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in EXCLUDED_FILES:
                continue
            files.append(path)
    return files


def _has_canonical_inline_block(text):
    """True if EVERY setenv("ONEAPI_DEVICE_SELECTOR" call in the text is
    followed, within a short window, by execv("/proc/self/exe" -- i.e. the
    file's entire bespoke handling of this variable is the canonical inline
    re-exec block, with no stray bare setenv left anywhere. A single
    correctly-followed match used to be enough, which meant a file could
    carry one correct inline block AND a separate stray bare setenv and
    still read as compliant (spec round 2, c-xicg finding 3 -- round 1's
    half-finished-port fail-open, one level over, for the inline form
    instead of the helper form)."""
    matches = list(SETENV_PATTERN.finditer(text))
    if not matches:
        return False
    for m in matches:
        window = text[m.end() : m.end() + 400]
        if not EXECV_PATTERN.search(window):
            return False
    return True


def _mask_comments_and_literals(text):
    """Returns a copy of `text` the same length, with every `//` line
    comment, `/* */` block comment, string literal ("...") and char literal
    ('...') replaced character-for-character by spaces (newlines inside a
    masked span are kept as newlines, everything else becomes a space).
    Every offset in the returned copy therefore still lines up 1:1 with
    `text` -- a position or span found by scanning the MASKED copy can be
    used directly to slice or search the ORIGINAL text.

    Escapes inside a literal (`\\"`, `\\\\`, `\\'`) are honoured so an
    escaped quote does not end the literal early.

    THREE THINGS THIS DOES NOT HANDLE, LEFT AS KNOWN RESIDUAL GAPS rather
    than papered over (quality round 5, rev-5q1r-quality-5, c-i7ep
    should-fix 2: masking must not be sold as closing every direction):
      - Raw string literals (`R"delim(...)delim"`) are not recognized as a
        distinct construct. `R"` is treated as an ordinary string start, so
        a raw string's own embedded unescaped quotes or parentheses are not
        specially honoured -- masking can end early or run long across one.
        No file in tests/ or ggml/src/ggml-sycl/tests/ uses a raw string
        literal at the time of writing.
      - Digraphs (`<%`/`%>` for `{`/`}`, `<:`/`:>` for `[`/`]`, ...) are
        invisible to both the mask and the brace walk -- a digraph brace is
        simply never counted, in either direction. No file here uses them.
      - A `main` spelled through a macro (`#define ENTRY int main` /
        `ENTRY(argc, argv) { ... }`) is invisible to MAIN_SIGNATURE_PATTERN
        regardless of masking; this was already true before masking existed
        and is unrelated to it."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            end = j if j != -1 else n
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:end]))
            i = end
        elif two == "/*":
            j = text.find("*/", i + 2)
            end = (j + 2) if j != -1 else n
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:end]))
            i = end
        elif text[i] in "\"'":
            quote = text[i]
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == quote:
                    j += 1
                    break
                j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _find_main_bodies(text):
    """Yields (body_start, body_end, balanced) for each main(...) function
    found. body_start/body_end are character offsets into `text` (valid in
    both `text` and its masked form, since masking preserves length and
    position), strictly between the function's opening brace and the
    position the depth-counted brace scan reached; `balanced` is True only
    if that scan actually returned to depth 0 (found a real matching
    closing brace) before EOF.

    Both the main() signature search and the brace-depth walk run on
    `_mask_comments_and_literals(text)`, NOT on `text` itself -- see that
    function's docstring for exactly what it masks and its known residual
    gaps. This defeats three real, reproduced exploit shapes, all of which
    previously fooled the raw (unmasked) scanner:

      - TOO-EARLY (quality round 3, c-autq): a stray `}` inside a string or
        comment ending a main() body before its real closing brace, ORPHANING
        a real offending call outside every yielded body.
      - TOO-LATE (quality round 4, c-q3hv): the mirror -- a stray `{` inside
        a string or comment pushing the depth count so main()'s real closing
        brace never returns it to 0, yielding a body that runs to EOF and
        swallows a later, unrelated function's call.
      - SPURIOUS SIGNATURE (quality round 5, c-i7ep): a `main(...) {`-shaped
        SUBSTRING inside a `//` or `/* */` comment (or a string literal)
        elsewhere in the file being matched by MAIN_SIGNATURE_PATTERN as if
        it were a second, real main() -- e.g. `// historical: int
        main(int, char ** argv) {` inside a later free function. That
        spurious "body" balances at the enclosing function's own real `}`
        and satisfies BOTH the round-3 and round-4 fixes on a span that
        corresponds to no function at all, reading the file compliant.

    Masking removes the substring these three exploits depend on before
    either the signature search or the brace walk ever sees it, rather than
    fencing off each failure direction with a separate check on raw text --
    a durable fix at the source of the ambiguity instead of one patch per
    discovered shape. `#if 0` and `#if !defined(GGML_USE_SYCL)` guarded
    main() definitions are NOT masked: they are real code with real braces
    (dead at compile time, not textually fake), and must keep scanning
    exactly as before -- an unbalanced one still routes to
    offender_unbalanced_main_body (kept as defence in depth alongside the
    outside-span check in `_first_statement_verdict`, even though masking
    closes the specific shapes that used to need them)."""
    masked = _mask_comments_and_literals(text)
    for m in MAIN_SIGNATURE_PATTERN.finditer(masked):
        start = m.end()
        depth = 1
        i = start
        n = len(masked)
        while i < n and depth > 0:
            c = masked[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        yield start, i - 1, depth == 0


def _first_real_offset(text, start, end):
    """Advances past whitespace, `//` line comments, `/* */` block
    comments, and preprocessor directive lines (#if, #endif, #else, ...) to
    find where the first real (non-comment, non-preprocessor) statement
    begins inside a main() body."""
    pos = start
    while pos < end:
        for pattern in (_WHITESPACE, _LINE_COMMENT, _BLOCK_COMMENT, _PP_LINE):
            m = pattern.match(text, pos)
            if m:
                pos = m.end()
                break
        else:
            break
    return pos


def _first_statement_is_valid(text, offset):
    """True if the real code starting at `offset` is either a direct call
    to the shared helper, or specifically the canonical inline block's
    guarding `if (!getenv("ONEAPI_DEVICE_SELECTOR"` / `if
    (!std::getenv("ONEAPI_DEVICE_SELECTOR"` (INLINE_GUARD_PATTERN), whose
    setenv() is itself followed by execv() within a short window.

    Quality round 1 (rev-5q1r-quality-1, c-h331 should-fix 1) found the
    predecessor of this check -- `remainder.lstrip(" \\t").startswith("if")`
    -- accepted ANY first statement merely beginning with the two letters
    "if", not specifically the ONEAPI_DEVICE_SELECTOR guard: both
    `if (want_debug()) { enable_debug(); }` and a bare `iface_init();`
    (which does not even start with "if (", just the substring "if") read as
    compliant as long as the canonical block appeared later in the 500-char
    window. INLINE_GUARD_PATTERN anchors the match to the guard's actual
    condition instead of its first two characters."""
    remainder = text[offset : offset + 500]
    if HELPER_CALL_PATTERN.match(remainder):
        return True
    guard_m = INLINE_GUARD_PATTERN.match(remainder)
    if guard_m:
        setenv_m = SETENV_PATTERN.search(remainder, guard_m.end())
        if setenv_m and setenv_m.start() < 200:
            window = remainder[setenv_m.end() : setenv_m.end() + 400]
            if EXECV_PATTERN.search(window):
                return True
    return False


def _first_statement_verdict(text):
    """Returns "ok", "not_first", "outside_main", or "unbalanced" for a
    single file's handling of main()-body position.

    "unbalanced": some main(...) function's brace scan never returned to
    depth 0 before EOF (see _find_main_bodies' TOO-LATE case). Checked
    FIRST and unconditionally for any in-scope file: an unbalanced body
    cannot be trusted to bound anything, so no other verdict is computed
    from it.

    "not_first": some computed (balanced) main() body mentions the helper
    or a bespoke setenv("ONEAPI_DEVICE_SELECTOR" but not as its first real
    statement, OR no computed body mentions it at all (it cannot be main()'s
    first statement if it is not in main() at all -- spec round 2, c-xicg
    finding 2).

    "outside_main": every computed body's OWN first-statement check passed
    (or found nothing to check), but some occurrence of the helper call or
    a bespoke setenv("ONEAPI_DEVICE_SELECTOR" in the file falls OUTSIDE
    every computed body's span -- see _find_main_bodies' TOO-EARLY case."""
    bodies = []
    for start, end, balanced in _find_main_bodies(text):
        if not balanced:
            return "unbalanced"
        bodies.append((start, end))
    found_relevant_main = False
    for start, end in bodies:
        body = text[start:end]
        if not (HELPER_CALL_PATTERN.search(body) or SETENV_PATTERN.search(body)):
            continue
        found_relevant_main = True
        offset = _first_real_offset(text, start, end)
        if not _first_statement_is_valid(text, offset):
            return "not_first"
    if not found_relevant_main:
        return "not_first"
    for pattern in (HELPER_CALL_PATTERN, SETENV_PATTERN):
        for m in pattern.finditer(text):
            if not any(start <= m.start() < end for start, end in bodies):
                return "outside_main"
    return "ok"


def _classify(text):
    """Returns one of "not_in_scope", "offender_half_finished",
    "offender_bare_setenv", "offender_not_first_statement",
    "offender_call_outside_scanned_main", "offender_unbalanced_main_body",
    "compliant" for a single file's text."""
    has_helper = bool(HELPER_CALL_PATTERN.search(text))
    has_setenv = bool(SETENV_PATTERN.search(text))
    if not has_setenv and not has_helper:
        return "not_in_scope"
    if has_helper and has_setenv:
        return "offender_half_finished"
    if has_setenv and not _has_canonical_inline_block(text):
        return "offender_bare_setenv"
    verdict = _first_statement_verdict(text)
    if verdict == "not_first":
        return "offender_not_first_statement"
    if verdict == "outside_main":
        return "offender_call_outside_scanned_main"
    if verdict == "unbalanced":
        return "offender_unbalanced_main_body"
    return "compliant"


def _offenders():
    offenders = []
    for path in _iter_source_files():
        verdict = _classify(path.read_text(encoding="utf-8"))
        if verdict.startswith("offender"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel} ({verdict})")
    return offenders


def test_no_offenders_outside_excluded_files():
    offenders = _offenders()
    assert not offenders, (
        "the following files mishandle ONEAPI_DEVICE_SELECTOR -- a bare setenv() with no execv(), "
        "a helper call or inline block that is not main()'s first real statement, or a half-finished "
        "combination of forms -- a plain setenv() in main() is a no-op in this fork, see "
        "llama.cpp-2x3m / llama.cpp-5q1r: " + ", ".join(offenders)
    )


def test_half_finished_port_is_caught():
    """Fixture proving the half-finished-port combination (helper call PLUS
    a leftover bespoke setenv) is actually detected, not just described --
    spec round 1 (c-k45u finding 2) found the first version of this gate
    failed open on exactly this combination."""
    half_finished = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(half_finished) == "offender_half_finished"

    clean_helper_only = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(clean_helper_only) == "compliant"

    clean_inline_only = (
        'int main(int, char ** argv) {\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(clean_inline_only) == "compliant"

    bare_setenv_only = (
        'int main() {\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(bare_setenv_only) == "offender_bare_setenv"

    unrelated = 'int main() { return 0; }\n'
    assert _classify(unrelated) == "not_in_scope"


def test_stray_setenv_beside_inline_block_is_caught():
    """Fixture proving a file with one CORRECT canonical inline block AND a
    SEPARATE stray bare setenv is caught -- spec round 2 (c-xicg finding 3)
    found the single-match version of _has_canonical_inline_block() missed
    exactly this: it stopped looking the moment it found one setenv()
    followed by an execv(), so a second, bare setenv() elsewhere in the same
    file went unnoticed."""
    inline_block_plus_stray = (
        'int main(int, char ** argv) {\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    some_other_setup();\n'
        '    // a leftover from a half-finished edit -- never followed by execv\n'
        '    setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);\n'
        '    return 0;\n'
        '}\n'
    )
    assert not _has_canonical_inline_block(inline_block_plus_stray)
    assert _classify(inline_block_plus_stray) == "offender_bare_setenv"


def test_first_statement_position_is_checked():
    """Fixture proving the position check actually runs, not just exists --
    spec round 2 (c-xicg finding 2) found the gate never verified the
    helper/inline call was main()'s FIRST statement, only that it appeared
    somewhere in the file."""
    helper_after_other_code = (
        'int main(int, char ** argv) {\n'
        '    do_some_setup_first();\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(helper_after_other_code) == "offender_not_first_statement"

    inline_after_other_code = (
        'int main(int, char ** argv) {\n'
        '    do_some_setup_first();\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(inline_after_other_code) == "offender_not_first_statement"

    # Blank lines, a line comment, a block comment, and a preprocessor guard
    # line between the opening brace and the call must all be tolerated --
    # none of them is a real statement.
    helper_after_only_noise = (
        'int main(int, char ** argv) {\n'
        '\n'
        '    // a comment explaining why this matters\n'
        '    /* a block comment\n'
        '       spanning two lines */\n'
        '#if defined(SOME_GUARD)\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '#endif\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(helper_after_only_noise) == "compliant"

    # A multi-line preprocessor directive (backslash-newline continuation)
    # before the call must be skipped WHOLE, not just its first physical
    # line -- quality round 2 (c-dhg6) found the predecessor _PP_LINE regex
    # was a no-op for exactly this shape: it silently stopped at the
    # trailing backslash and treated the continuation line as the first
    # real statement, misclassifying this fixture as an offender.
    helper_after_multiline_define = (
        'int main(int, char ** argv) {\n'
        '#define LOCAL_HELPER(a) \\\n'
        '    do_stuff(a)\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(helper_after_multiline_define) == "compliant"

    # The pattern exists in the file but not inside ANY main() body at all
    # -- it cannot be main()'s first statement if it is not in main().
    setenv_outside_main = (
        'static void setup() {\n'
        '    setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);\n'
        '    execv("/proc/self/exe", nullptr);\n'
        '}\n'
        '\n'
        'int main() {\n'
        '    setup();\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(setenv_outside_main) == "offender_not_first_statement"


def test_inline_guard_pattern_is_specific():
    """Fixture proving the inline-form first-statement check is anchored to
    the actual ONEAPI_DEVICE_SELECTOR guard, not to any statement that
    merely starts with the two letters "if" -- quality round 1
    (rev-5q1r-quality-1, c-h331 should-fix 1) constructed exactly these two
    counter-examples against the predecessor check
    (`remainder.lstrip(" \\t").startswith("if")`), both of which it wrongly
    accepted as compliant because the canonical block happened to appear
    later in the same 500-char lookahead window."""
    unrelated_if_before_block = (
        'int main(int, char ** argv) {\n'
        '    if (want_debug()) { enable_debug(); }\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(unrelated_if_before_block) == "offender_not_first_statement"

    if_prefixed_identifier_before_block = (
        'int main(int, char ** argv) {\n'
        '    iface_init();\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(if_prefixed_identifier_before_block) == "offender_not_first_statement"

    # Both canonical spellings of the guard must still be accepted when they
    # genuinely are the first statement.
    bare_getenv_first = (
        'int main(int, char ** argv) {\n'
        '    if (!getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(bare_getenv_first) == "compliant"

    std_getenv_first = (
        'int main(int, char ** argv) {\n'
        '    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {\n'
        '        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);\n'
        '        execv("/proc/self/exe", argv);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(std_getenv_first) == "compliant"


def test_orphaned_call_outside_scanned_main_is_caught():
    """Fixture proving the FILE-scoped found_relevant_main fail-open is
    closed -- quality round 3 (rev-5q1r-quality-3, c-autq) reproduced a real
    shape where it was not: a stray `}` inside a string literal mis-bounds
    the first main()'s computed body early enough to orphan its own real
    offending call, while a textually unrelated SECOND `int main(...)` later
    in the file (guarded by `#if 0`) happens to call the helper correctly
    and satisfies the per-body loop on its own.

    HISTORY OF THIS FIXTURE'S EXPECTED VERDICT (each fix made it more
    precise, never less correct): before round 3's fix (c349bd259's
    predecessor, 5d2ddf675^) it classified "compliant" -- the bug. Round 3
    (c349bd259) fixed the outside-span gap and it became
    offender_call_outside_scanned_main. Quality round 5's masking fix
    (rev-5q1r-quality-5, c-i7ep) additionally makes the stray `}` inside the
    string invisible to the brace walk in the first place, so main()'s real
    closing brace is found directly and the SAME defect (do_something_first()
    genuinely precedes the call) is now caught earlier and more specifically,
    as offender_not_first_statement -- correct either way, but this is the
    verdict a proper string-aware scanner gives without ever needing the
    outside-span check for THIS particular file."""
    arm_stray_brace_plus_dead_main = (
        'int main(int, char ** argv) {\n'
        '    printf("stray: }\\n");\n'
        '    do_something_first();\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        '#if 0\n'
        'int main() {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
        '#endif\n'
    )
    assert _classify(arm_stray_brace_plus_dead_main) == "offender_not_first_statement"

    # Control: the same file WITHOUT the stray brace and WITHOUT the dead
    # second main -- this must still be a plain offender_not_first_statement
    # (do_something_first() genuinely runs before the call), proving the fix
    # did not just make everything with two "main"-shaped regions an
    # offender for the wrong reason.
    control_without_stray_brace = (
        'int main(int, char ** argv) {\n'
        '    do_something_first();\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(control_without_stray_brace) == "offender_not_first_statement"


def test_unbalanced_main_body_is_its_own_offender():
    """Fixture proving the too-LATE mirror of the round-3 gap is closed --
    quality round 4 (rev-5q1r-quality-4, c-q3hv) reproduced a real shape
    where round 3's own fix was itself corrupted by bad data: a stray `{`
    inside a string literal pushes the brace-scan depth high enough that
    main()'s real closing brace only returns it to 1, so the scan ran to EOF
    and yielded a body spanning the whole rest of the file, swallowing a
    LATER, genuinely unrelated free function's helper call and satisfying
    round 3's outside-span check on corrupted data.

    HISTORY OF THIS FIXTURE'S EXPECTED VERDICT: before round 4's fix
    (c349bd259) it classified "compliant" -- the bug. Round 4
    (c349bd259) added the offender_unbalanced_main_body bucket and this arm
    classified that way, because without masking the stray `{` was a real
    brace to the walk and main() genuinely never re-balanced. Quality round
    5's masking fix (rev-5q1r-quality-5, c-i7ep) makes that stray `{`
    (inside a string literal) invisible to the brace walk in the first
    place, so main() now balances correctly at its own real closing brace
    and the late free function's call is directly caught as
    offender_call_outside_scanned_main -- the SAME underlying defect, now
    identified without ever needing the unbalanced-body bucket for THIS
    particular file. That bucket is kept as defence in depth and exercised
    below by a main() whose imbalance is real code, not a masked-away
    literal."""
    arm_too_late_brace = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    printf("open brace: {\\n");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static void late_setup(char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:1");\n'
        '}\n'
    )
    assert _classify(arm_too_late_brace) == "offender_call_outside_scanned_main"

    # Control: identical WITHOUT the stray `{` -- main()'s body already
    # balanced correctly at its own real closing brace even before masking,
    # so late_setup()'s call genuinely falls outside any computed main body
    # either way, proving the fix did not change this file's verdict.
    control_no_stray_brace = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    printf("no brace here\\n");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static void late_setup(char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:1");\n'
        '}\n'
    )
    assert _classify(control_no_stray_brace) == "offender_call_outside_scanned_main"

    # The bucket itself is kept as defence in depth even though masking
    # closes the string/comment-hidden-brace shapes that used to need it
    # (see _find_main_bodies' docstring) -- it must still fire for a main()
    # whose brace imbalance is REAL CODE, not a brace masking now makes
    # invisible.
    arm_real_unclosed_brace = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    if (some_real_condition()) {\n'
        '        do_stuff();\n'
        '    return 0;\n'
        '}\n'
    )
    assert _classify(arm_real_unclosed_brace) == "offender_unbalanced_main_body"


def test_comment_embedded_main_signature_does_not_mint_a_body():
    """Fixture proving a `main(...) {`-shaped SUBSTRING inside a comment (or
    a string literal) does not mint a spurious second main() body -- quality
    round 5 (rev-5q1r-quality-5, c-i7ep) found a `// historical: int
    main(int, char ** argv) {` comment inside a later free function created
    a second, textually-matched "main" that balanced at the enclosing
    function's own `}` and covered that function's real (offending) call,
    satisfying both the round-3 outside-span check and the round-4
    unbalanced-body check on a span that corresponds to no function at all.

    Before this fix, arm_comment_embedded_signature below classified
    "compliant" -- the bug. The control (identical, without the comment
    trick) already correctly classified offender_call_outside_scanned_main
    both before and after this fix; masking makes the two arms agree,
    neutralizing the exploit without changing the control's verdict."""
    arm_comment_embedded_signature = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static void late_setup(char ** argv) {\n'
        '    // historical: int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:1");\n'
        '}\n'
    )
    assert _classify(arm_comment_embedded_signature) == "offender_call_outside_scanned_main"

    arm_string_embedded_signature = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static void late_setup(char ** argv) {\n'
        '    log("int main(int, char ** argv) {");\n'
        '    sycl_test_selector_fallback(argv, "level_zero:1");\n'
        '}\n'
    )
    assert _classify(arm_string_embedded_signature) == "offender_call_outside_scanned_main"

    control_no_comment_trick = (
        'int main(int, char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:0");\n'
        '    return 0;\n'
        '}\n'
        '\n'
        'static void late_setup(char ** argv) {\n'
        '    sycl_test_selector_fallback(argv, "level_zero:1");\n'
        '}\n'
    )
    assert _classify(control_no_comment_trick) == "offender_call_outside_scanned_main"


def test_recursive_scan_reaches_known_subdirectories():
    """Confirms rglob (not glob) is actually used and actually descends --
    spec round 2 (c-xicg finding 4) found *.cpp files in subdirectories of
    both scan roots (tests/peg-parser, tests/e1-rca, tests/sycl-canary,
    tests/sycl-alloc-policy-fixtures/*, ...) were invisible to a
    non-recursive glob. None of those currently carry the
    ONEAPI_DEVICE_SELECTOR pattern, so this asserts reach, not verdicts."""
    anchor = TESTS_DIR / "peg-parser" / "test-basic.cpp"
    assert anchor.is_file(), (
        "fixture assumption broken: tests/peg-parser/test-basic.cpp no longer exists -- "
        "pick another known nested .cpp file to anchor this test"
    )
    scanned = {p.relative_to(REPO_ROOT).as_posix() for p in _iter_source_files()}
    assert "tests/peg-parser/test-basic.cpp" in scanned, (
        "the scan does not reach tests/peg-parser/test-basic.cpp -- recursion into "
        "subdirectories of the scan roots is broken"
    )


def _registration_form(stem, cmake_text):
    """Returns the matched CMake registration call for `stem` (a .cpp
    file's basename without extension) in `cmake_text`, or None -- an
    add_executable(...), llama_build(...)/llama_build_and_test(...) (which
    take the .cpp filename), or add_test(NAME ...) call, never a bare
    substring match.

    Quality round 2 (rev-5q1r-quality-2, c-dhg6 nit A) found the
    predecessor of this check, `stem not in cmake_text`, is satisfiable by
    PROSE ALONE: tests/CMakeLists.txt:2680 mentions "test-sycl-model-
    repro.cpp" inside a comment (a catalogued "restorable source" note),
    with no registration anywhere nearby -- a bare substring check would
    have reported that file as registered the moment its name was merely
    discussed, false-alarming a human into believing an exclusion needs
    dropping when nothing was actually registered."""
    escaped = re.escape(stem)
    for pattern in (
        rf"add_executable\s*\(\s*{escaped}\b",
        rf"llama_build\s*\(\s*{escaped}\.cpp\b",
        rf"llama_build_and_test\s*\(\s*{escaped}\.cpp\b",
        rf"add_test\s*\(\s*NAME\s+{escaped}\b",
    ):
        m = re.search(pattern, cmake_text)
        if m:
            return m.group(0)
    return None


def test_excluded_files_are_still_accounted_for():
    """Guards the exclusion list itself against silent rot: each excluded
    file must still exist, still contain a bespoke ONEAPI_DEVICE_SELECTOR
    setenv() (the pattern that justified excluding it), and must still lack
    an actual CMake registration (add_executable / llama_build* / add_test)
    in either CMakeLists.txt -- checked via _registration_form(), not a
    bare substring, precisely because a bare substring is satisfiable by
    prose alone (see that function's docstring). "Not registered" is a
    secondary, non-load-bearing fact for the current exclusion (its
    PRIMARY reason -- the subprocess-command-string mechanism, see
    EXCLUDED_FILES -- is what actually discriminates it from ported, gated
    files that are equally unregistered), but it is still checked here: the
    moment a currently-excluded file gains a real registration, a human
    should re-examine whether the exclusion (for whichever reason) still
    holds, rather than the exclusion silently continuing to skip a newly
    built, ctest-registered test (spec round 1, c-k45u finding 3)."""
    for rel in EXCLUDED_FILES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"excluded file {rel} no longer exists -- update this gate's EXCLUDED_FILES"
        text = path.read_text(encoding="utf-8")
        assert SETENV_PATTERN.search(text), (
            f"excluded file {rel} no longer contains a bespoke ONEAPI_DEVICE_SELECTOR setenv() -- "
            "the exclusion may no longer be needed; re-check and drop it from EXCLUDED_FILES if so"
        )
        stem = path.stem  # basename without .cpp -- the CMake target/source name
        for cmake_path in CMAKE_FILES:
            cmake_text = cmake_path.read_text(encoding="utf-8")
            found = _registration_form(stem, cmake_text)
            assert found is None, (
                f"excluded file {rel} (target name '{stem}') now has a real registration "
                f"({found!r}) in {cmake_path.relative_to(REPO_ROOT)} -- re-examine whether the "
                "exclusion still holds for its stated (primary) reason; if not, port it to the "
                "shared helper and drop it from EXCLUDED_FILES."
            )


def test_helper_header_exists_and_defines_the_function():
    header = TESTS_DIR / "sycl-selector-fallback.hpp"
    assert header.is_file(), "tests/sycl-selector-fallback.hpp is missing"
    text = header.read_text(encoding="utf-8")
    assert "sycl_test_selector_fallback" in text
    assert "execv(" in text, "the helper must re-exec, not just setenv()"
    assert 'setenv("ONEAPI_DEVICE_SELECTOR"' in text, "the helper must set ONEAPI_DEVICE_SELECTOR itself"


def test_every_helper_caller_declares_argv_in_main():
    """A file calling sycl_test_selector_fallback(argv, ...) needs an argv to
    pass, so its main() must declare a char ** parameter -- otherwise the
    file would not compile, which would silently make this gate's GREEN
    state meaningless (nothing was actually built to run)."""
    missing = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        if not HELPER_CALL_PATTERN.search(text):
            continue
        if not MAIN_WITH_ARGV_PATTERN.search(text):
            missing.append(path.relative_to(REPO_ROOT).as_posix())
    assert not missing, (
        "the following files call sycl_test_selector_fallback(argv, ...) but their main() "
        "does not appear to declare a char ** argv parameter: " + ", ".join(missing)
    )


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
