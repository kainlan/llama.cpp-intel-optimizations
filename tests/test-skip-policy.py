#!/usr/bin/env python3
"""A test that skips must exit 77, and its registration must say so.

`tests/get-model.cpp:16` exited EXIT_SUCCESS when no model file was available.
`LLAMACPP_TEST_MODELFILE` is set only inside `ci/run.sh`, and the three
consumers are registered with no ARGS, so on every local `ctest` run they took
the skip path and reported green:

    1/3 test-model-load-cancel ...  Passed  0.18 sec
    2/3 test-autorelease .........  Passed  0.18 sec
    3/3 test-backend-sampler .....  Passed  0.18 sec

0.18 s was process startup plus the skip. Nothing was exercised, and this had
presumably read as green for as long as the tests had existed (llama.cpp-nwip).
`test-thread-safety` carried a duplicate of the same logic; a bare invocation of
it produced a zero-line capture and status 0, and only an implausible 2.2 GB
memory peak revealed it had done nothing.

Skipping stays allowed -- a model-less or CPU-only runner legitimately skips --
but it may not claim to have passed. Two things must hold for that, and losing
either one silently restores the old behaviour:

  * the binary exits 77 rather than 0 (tests/test-skip.h), and
  * its registration carries SKIP_RETURN_CODE 77, or ctest scores that 77 as a
    FAILURE instead of a skip.

Both are one edit away from gone: `tests/get-model.cpp` is upstream code that a
rebase can revert, a new test can copy-paste `exit(EXIT_SUCCESS)`, and the
property lives in a single line of `llama_build_and_test()`. Nothing else
asserts any of it -- the same gap that let `llama.cpp-4hvq` sit undetected for
months and that `test-src-cmake-coverage.py` (this file's model) exists to
close.

This gate fails loudly rather than skipping when it cannot find what it checks.
A gate against vacuous passes that passes vacuously is worth less than no gate,
because it also reads as coverage.

Revision history: llama.cpp-g290, commits 848bbd3bc, 4915e340b, 4d1420116,
f2c6ae1ec.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tests" / "CMakeLists.txt"

# ctest's conventional "skipped" status, and the value of SKIP_RETURN_CODE on
# every test llama_build_and_test() registers. Kept in sync with
# LLAMA_TEST_EXIT_SKIP in tests/test-skip.h by test_skip_header_defines_77.
EXIT_SKIP = 77

# The two directories test sources live under, one per CMake registration
# mechanism: tests/CMakeLists.txt's llama_build_and_test(), and the individual
# add_executable()/add_test() pairs in ggml/src/ggml-sycl/CMakeLists.txt.
# See llama.cpp-g290.
TEST_SOURCE_DIRS = (ROOT / "tests", ROOT / "ggml" / "src" / "ggml-sycl" / "tests")
TEST_SOURCE_EXTS = frozenset({".c", ".cc", ".cpp", ".h", ".hpp"})
SKIP_HEADER = ROOT / "tests" / "test-skip.h"

# Matches 77 used as an executable exit/return value -- `return 77;`,
# `exit(77)`, `_Exit(77)`, `std::exit(77)` (any inner spacing) -- and nothing
# else. Anchored on the return/exit keyword, not on the digits alone, so it
# does not fire on a float literal (test-llama-archs.cpp's
# `3.5565588200778455f`), an array size, or a line/ticket number mentioned in
# a comment. Kept as a second, stricter check alongside BROAD_LITERAL_77_RE
# below: a change that breaks the broad scan's allowlist should not also
# silently lose this narrower, allowlist-free one.
LITERAL_SKIP_EXIT_RE = re.compile(r"\breturn\s+77\s*;" r"|\b(?:std::)?_?[Ee]xit\s*\(\s*77\s*\)")

# The broad scan: literal 77 (with an optional C++ integer-literal suffix --
# `77u`, `77U`, `77L`, `77UL`, `77ull`, up to 3 chars of u/U/l/L -- so a skip
# constant spelled `77u` is not invisible to this scan the way it was until
# this widening) as a standalone integer token, wherever it appears in code
# (not inside a comment or a string/char/raw-string literal -- see
# _strip_comments). This is what catches 77 bound to a named constant that is
# only later `return`ed, which LITERAL_SKIP_EXIT_RE above cannot see by
# construction.
#
# One structural exclusion is baked in rather than allowlisted per-occurrence:
# 77 immediately followed by `.<digit>` is the start of a float/double literal
# (e.g. `77.0f`), never an exit code, and this pattern recurs anywhere a fill
# or comparison value happens to be chosen as 77.0 -- allowlisting every such
# constant by file+line would make the allowlist grow with unrelated test
# data instead of staying small.
#
# Array-size literals (`int arr[77];`) are deliberately NOT given the same
# structural exclusion: unlike a float suffix, `[77]` is not syntactically
# distinguishable from an argument or a sentinel value, and this scan already
# has zero real array-size-shaped occurrences to generalize from. A real one
# is allowlisted by entry (file + line-content regex), the same as every
# other non-exit literal below -- see the array-size-shape assertion inside
# test_broad_literal_77_regex_has_controls.
BROAD_LITERAL_77_RE = re.compile(r"\b77[uUlL]{0,3}\b(?!\.\d)")

# Small, explicit, re-verified allowlist of genuine non-exit uses of the
# literal 77 found by BROAD_LITERAL_77_RE across the tree. Keyed on (relative
# file path, a regex matching the offending LINE's content) rather than on
# line numbers, so it does not rot as the surrounding file is edited -- a
# line-number-keyed allowlist silently stops covering its target the moment
# an unrelated edit shifts it. test_allowed_non_exit_77_entries_are_all_live
# asserts every entry below still matches something, so a stale one (the
# named line moved or was deleted) fails loudly instead of quietly rotting
# into dead weight.
#
# Every entry was read in context and confirmed to be tensor-shape data, a
# fixture id/argument, or a sentinel value that happens to equal 77 -- never
# a skip-exit path. Do not add an entry without reading the line it excludes.
ALLOWED_NON_EXIT_77 = (
    # Tensor-shape test-case constructors: 64x77 / {77,...} dimensions, not
    # exit codes.
    ("tests/test-backend-ops.cpp", re.compile(r"test_mul_mat\(.*\b64,\s*77,")),
    ("tests/test-backend-ops.cpp", re.compile(r"test_soft_max\(GGML_TYPE_F32,\s*\{77,")),
    # A per-element marker value added to a cross-device buffer, checked back
    # out on the other device -- arithmetic data, not a return code.
    ("tests/test-crossdev-sync.cpp", re.compile(r"static_cast<int>\(i\[0\]\)\s*\+\s*77")),
    # Prose inside a --exclude diagnostic string ("...then 77 is the honest
    # answer..."), not code that returns or exits.
    ("tests/test-llama-archs.cpp", re.compile(r"then 77 is the honest")),
    # A fill_random() seed/value argument.
    ("tests/test-sycl-cpu-dispatch.cpp", re.compile(r"fill_random\(.*,\s*77\)")),
    # A fixture lifecycle-plan-snapshot model_id, compared and assigned.
    ("tests/test-sycl-tensor-placement.cpp", re.compile(r"model_id\s*(!=|=)\s*77\b")),
    # A forged pointer marker value, and a device-ordinal argument to a
    # weight-lease-snapshot constructor -- neither is this process's exit code.
    ("ggml/src/ggml-sycl/tests/test-mem-handle-wrong-device.cpp", re.compile(r"\bmarker\s*=\s*77;")),
    ("ggml/src/ggml-sycl/tests/test-mem-handle-wrong-device.cpp", re.compile(r"key,\s*77,\s*&first_storage")),
    # A generation/table/owner id argument in retention and MoE-discovery
    # fixtures.
    ("ggml/src/ggml-sycl/tests/test-moe-discovery-owner-state.cpp", re.compile(r"owner_of\(77,")),
    ("ggml/src/ggml-sycl/tests/test-moe-graph-retention.cpp", re.compile(r"create\(first_key,\s*501,\s*77,")),
    ("ggml/src/ggml-sycl/tests/test-moe-graph-retention.cpp", re.compile(r"add_table\(\{\s*501,\s*77,")),
    # Lease-acquire/release fixture arguments, and a deliberately-poisoned
    # sentinel that a correct classifier must overwrite before the assertion
    # below it can pass.
    ("ggml/src/ggml-sycl/tests/test-moe-mmid-workspace-plan.cpp", re.compile(r"\bacquire\(77,\s*88\)")),
    ("ggml/src/ggml-sycl/tests/test-moe-mmid-workspace-plan.cpp", re.compile(r"terminal_release\(later_slot\.lease,\s*77,\s*88\)")),
    ("ggml/src/ggml-sycl/tests/test-moe-mmid-workspace-plan.cpp", re.compile(r"\badmitted\s*=\s*77;")),
)


# A raw string literal's introducer: an optional encoding prefix (u8/u/U/L),
# then `R"`, then a delimiter of up to 16 chars drawn from the C++ standard's
# d-char set (anything but space, `(`, `)`, backslash, or a control char),
# then the `(` that starts the literal's body. Matched at a fixed position
# (via .match(text, pos), not .search) so it only fires exactly where a raw
# string literal actually starts.
_RAW_STRING_START_RE = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\v\f\n]{0,16})\(')


def _strip_comments(text: str, _disable_raw_strings: bool = False) -> str:
    """Blank `//` and `/* */` comments to same-length whitespace, in place.

    Contract: every character keeps its original position -- including
    every newline, so a line number computed on the result matches the
    original text even across a multi-line block comment -- no character
    is ever moved, and string, char, and raw-string literals are scanned
    only to find their end and are otherwise left byte-for-byte untouched,
    so a `//` or `/*` inside one of them is never mistaken for a real
    comment marker.

    `_disable_raw_strings=True` disables the raw-string branch only (no
    other behaviour changes), so a test can construct a known-broken
    scanner without duplicating this function -- see
    test_the_cross_check_actually_fires. Never pass it outside a test.

    History: llama.cpp-g290, commits 848bbd3bc (introduced),
    4d1420116 (raw string literals), f2c6ae1ec (C++14 digit separators).
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        raw_match = None if _disable_raw_strings else _RAW_STRING_START_RE.match(text, i)
        if raw_match:
            delim = raw_match.group(1)
            terminator = ")" + delim + '"'
            body_start = raw_match.end()
            end = text.find(terminator, body_start)
            end = end + len(terminator) if end != -1 else n
            i = end
            continue
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                if text[j] == "\n":
                    break  # unterminated literal on this line; stop here
                j += 1
            i = j
            continue
        if c == "'":
            # A C++ character literal holds exactly one character or one
            # backslash-escape unit -- unlike a string literal's arbitrary
            # length. A C++14 digit separator (`2'000'000'000`) is a bare
            # `'` between digits, not a character literal; only the closing
            # `'` IMMEDIATELY after one plain character or one `\<char>`
            # escape counts as a literal, so a digit separator is correctly
            # left as ordinary code instead of desyncing the scanner.
            if i + 3 < n and text[i + 1] == "\\" and text[i + 3] == "'":
                i += 4
                continue
            if i + 2 < n and text[i + 1] != "\\" and text[i + 2] == "'":
                i += 3
                continue
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = i
            while j < n and text[j] != "\n":
                out[j] = " "
                j += 1
            i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            end = end + 2 if end != -1 else n
            for k in range(i, end):
                if text[k] != "\n":
                    out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


# A second, structurally different stripper with the same contract as
# _strip_comments: one alternation regex (raw string via a `(?P=delim)`
# backreference; plain string, escape-aware; plain char, exactly one
# character or escape unit, matching a real C++ character literal; `//`;
# `/* */`, DOTALL and non-greedy), rather than a hand-written char-by-char
# scanner. It transcribes the SAME raw-string delimiter specification as
# _RAW_STRING_START_RE (both independently encode the same d-char-set rule),
# so agreement between the two rules out a MECHANISM bug in either
# implementation -- not a bug both would share because they encode the same
# (possibly wrong) specification. See
# test_strip_comments_agrees_with_an_independent_implementation.
_INDEPENDENT_TOKEN_RE = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delim>[^ ()\\\t\v\f\n]{0,16})\(.*?\)(?P=delim)")'
    r'|(?P<str>"(?:\\.|[^"\\\n])*")'
    r'|(?P<chr>\'(?:\\.|[^\'\\\n]){1}\')'
    r'|(?P<line>//[^\n]*)'
    r'|(?P<block>/\*.*?\*/)',
    re.DOTALL,
)


def _independent_strip_comments(text: str) -> str:
    """Same contract as _strip_comments, built as a single regex
    tokenization instead of a hand-rolled scanner. Used only to cross-check
    _strip_comments -- see
    test_strip_comments_agrees_with_an_independent_implementation.
    """
    out = list(text)
    for m in _INDEPENDENT_TOKEN_RE.finditer(text):
        if m.group("line") is not None or m.group("block") is not None:
            for k in range(m.start(), m.end()):
                if text[k] != "\n":
                    out[k] = " "
        # raw / str / chr: a literal, left untouched -- already correct in `out`.
    return "".join(out)


def _iter_test_sources():
    for src_dir in TEST_SOURCE_DIRS:
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*")):
            if path.is_file() and path.suffix in TEST_SOURCE_EXTS:
                yield path


def _is_allowlisted(rel_path: str, line: str) -> bool:
    return any(entry_path == rel_path and pattern.search(line) for entry_path, pattern in ALLOWED_NON_EXIT_77)


def _assert_scan_is_not_vacuous(scanned: int, what: str) -> None:
    """Fail loudly if a whole-tree scan looked at zero files. A scan that
    finds no offenders because it looked at nothing is indistinguishable,
    by its result alone, from a scan that looked at everything and found
    nothing real -- this makes them distinguishable.
    """
    assert scanned > 0, (
        f"{what}: scanned 0 files under {TEST_SOURCE_DIRS} -- this "
        "assertion would pass vacuously (nothing to check), which is "
        "worse than no assertion at all."
    )


def _scan_tree_for_offenders(regex: "re.Pattern[str]", allowlist_check=None):
    """Shared engine for the two tree-wide "stray literal 77" scans below.
    Walks _iter_test_sources() exactly once, strips comments once per file,
    and collects every `regex` match not vetoed by
    `allowlist_check(rel_path, line_text)` (when given). tests/test-skip.h
    is always excluded -- the one place the literal is allowed to live.
    Returns (offenders, scanned_count) so a caller's vacuity guard never
    needs a second walk of _iter_test_sources().
    """
    offenders = []
    scanned = 0
    for path in _iter_test_sources():
        if path.resolve() == SKIP_HEADER.resolve():
            continue
        scanned += 1
        rel = str(path.relative_to(ROOT))
        code = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        lines = code.splitlines()
        for m in regex.finditer(code):
            line_no = code.count("\n", 0, m.start()) + 1
            line_text = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else m.group(0)
            if allowlist_check is not None and allowlist_check(rel, line_text):
                continue
            offenders.append(f"{rel}:{line_no}: {line_text.strip()!r}")
    return offenders, scanned


def build_and_test_body() -> str:
    """The text of llama_build_and_test()'s body, or raise."""
    cmake = CMAKE.read_text(encoding="utf-8")
    start = cmake.find("function(llama_build_and_test")
    assert start != -1, (
        f"{CMAKE} no longer defines llama_build_and_test() -- this gate cannot "
        "check a registration helper that does not exist. Point it at the "
        "helper that replaced it."
    )
    end = cmake.find("endfunction()", start)
    assert end != -1, f"unterminated function(llama_build_and_test in {CMAKE}"
    return cmake[start:end]


def test_registrations_carry_skip_return_code() -> None:
    body = build_and_test_body()
    assert re.search(r"SKIP_RETURN_CODE\s+77", body), (
        "llama_build_and_test() no longer sets SKIP_RETURN_CODE 77. Every test "
        "it registers that exits 77 to mean 'skipped' is now scored by ctest "
        "as FAILED (test-llama-archs) -- or, for the model-requiring tests, "
        "would report Passed again if the exit code regressed too."
    )


def test_skip_header_defines_77() -> None:
    header = (ROOT / "tests" / "test-skip.h").read_text(encoding="utf-8")
    assert f"define LLAMA_TEST_EXIT_SKIP {EXIT_SKIP}" in header, (
        f"tests/test-skip.h no longer defines LLAMA_TEST_EXIT_SKIP as "
        f"{EXIT_SKIP}; it must match SKIP_RETURN_CODE in tests/CMakeLists.txt."
    )


def test_model_requiring_tests_use_the_shared_skip() -> None:
    # The no-model skip moved from tests/get-model.cpp into
    # common_get_model_or_exit() in common/common.cpp when upstream b10630
    # dissolved get-model.cpp (Phase C merge). The helper is upstream-shaped,
    # so a future rebase can revert it to exit(EXIT_SUCCESS) without a
    # conflict. test-thread-safety.cpp held a duplicate of the same logic with
    # `return 0`; the point of the shared helper is that there is one skip
    # policy, so a second copy reappearing is itself the regression.
    for name in ("../common/common.cpp", "test-thread-safety.cpp"):
        src = (ROOT / "tests" / name).read_text(encoding="utf-8")
        assert "test_skip_no_model()" in src, (
            f"tests/{name} no longer routes its no-model skip through "
            "test_skip_no_model() in tests/test-skip.h. If it was reverted by "
            "an upstream rebase, restore the call; if it grew its own copy, "
            "delete the copy."
        )


def test_a_model_requiring_binary_exits_77_when_it_skips() -> None:
    # The behavioural half. The three assertions above are text; this one runs
    # the thing. test-autorelease is the cheapest consumer of
    # get_model_or_exit(): with no model available it returns from the helper's
    # skip path before touching llama at all, so this loads no model and needs
    # no GPU.
    binary = os.environ.get("LLAMA_TEST_SKIP_BINARY") or str(ROOT / "build" / "bin" / "test-autorelease")
    assert pathlib.Path(binary).is_file(), (
        f"{binary} is not built, so the skip exit code cannot be verified. "
        "Build test-autorelease (or set LLAMA_TEST_SKIP_BINARY). This gate "
        "fails rather than skips here on purpose: a skipped check of skip "
        "semantics is exactly the vacuous pass it exists to catch."
    )

    env = dict(os.environ)
    env.pop("LLAMACPP_TEST_MODELFILE", None)  # force the skip path
    proc = subprocess.run([binary], capture_output=True, text=True, env=env, timeout=120)

    assert proc.returncode == EXIT_SKIP, (
        f"{pathlib.Path(binary).name} exited {proc.returncode} with no model "
        f"available; expected {EXIT_SKIP} (skipped). 0 means the skip is "
        "reported to ctest as a PASS again -- the whole defect. A loader or "
        "startup failure looks like this too, so read the output before "
        f"blaming the skip path.\nstderr:\n{proc.stderr[-2000:]}"
    )


def test_literal_skip_exit_regex_has_a_positive_control() -> None:
    # llama.cpp-g290: "77 means skip" used to have three independent homes --
    # tests/test-skip.h (canonical), tests/test-llama-archs.cpp, and
    # ggml/src/ggml-sycl/tests/test-mmq-xmx-dispatch.cpp, each hardcoding the
    # literal with its own explanatory comment. The tree-wide scan below can
    # only prove something if it can be shown to fire on a fixture known to
    # contain the thing it looks for -- otherwise "found nothing" could mean
    # either "the tree is clean" or "the regex is dead". This is that fixture,
    # evaluated in-memory, never written to disk.
    positive_fixture = """
    int main(int argc, char ** argv) {
        if (devices.empty()) {
            fprintf(stderr, "SKIP: no SYCL GPU devices available\\n");
            return 77;
        }
        return 0;
    }
    """
    assert LITERAL_SKIP_EXIT_RE.search(positive_fixture), (
        "LITERAL_SKIP_EXIT_RE did not match a fixture containing 'return 77;' "
        "-- the regex is broken, which would make "
        "test_no_stray_literal_skip_exit_outside_header() pass vacuously "
        "(finding nothing because it looks for nothing, not because the tree "
        "is clean)."
    )
    for call_form in ("exit(77)", "exit( 77 )", "_Exit(77)", "std::exit(77)"):
        assert LITERAL_SKIP_EXIT_RE.search(call_form), (
            f"LITERAL_SKIP_EXIT_RE did not match {call_form!r} -- one of the "
            "exit-call spellings the tree-wide scan is supposed to catch."
        )
    # Negative control alongside the positive one: digits that are not an
    # exit/return value must NOT fire, or the tree-wide scan would flag
    # test-llama-archs.cpp's `3.5565588200778455f` and every comment that
    # mentions "77" while explaining the convention (there are many, on
    # purpose -- this ticket does not ask for those to be deleted).
    for benign in (
        "ms.add_kv(LLM_KV_RESIDUAL_SCALE, 3.5565588200778455f);",
        "// (llama.cpp-k208 covers the 77 that follows from them)",
        "// without it ctest renders 77 as FAILED.",
        "int arr[77];",
    ):
        assert not LITERAL_SKIP_EXIT_RE.search(_strip_comments(benign)), (
            f"LITERAL_SKIP_EXIT_RE false-positived on benign text {benign!r} "
            "-- it must anchor on the return/exit keyword, not the digits alone."
        )


def test_no_stray_literal_skip_exit_outside_header() -> None:
    # The consolidation this ticket is for: tests/test-skip.h is the one
    # documented home for "77 means skip" (LLAMA_TEST_EXIT_SKIP). Every other
    # C/C++ test source under tests/ and ggml/src/ggml-sycl/tests/ must spell
    # a skip exit as LLAMA_TEST_EXIT_SKIP, never as the bare literal, or the
    # convention drifts back to having multiple homes that can disagree.
    offenders, scanned = _scan_tree_for_offenders(LITERAL_SKIP_EXIT_RE)
    _assert_scan_is_not_vacuous(scanned, "test_no_stray_literal_skip_exit_outside_header")

    assert not offenders, (
        "found the literal skip code 77 used as an exit/return value outside "
        "tests/test-skip.h -- use LLAMA_TEST_EXIT_SKIP from tests/test-skip.h "
        "instead (see llama.cpp-g290), so '77 means skip' has exactly one "
        "home. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_strip_comments_preserves_line_numbers() -> None:
    # The old stripper used re.sub(..., flags=re.DOTALL) to DELETE block
    # comments, which removes their embedded newlines too and shifts every
    # line number after them. A 3-line block comment made a real offender
    # on line 4 get reported as line 2. _strip_comments must blank comment
    # bodies in place instead.
    fixture = "int a;\n/* line2\n   line3\n   line4 */\nreturn 77;\n"
    stripped = _strip_comments(fixture)
    assert "return 77;" in stripped
    line_no = stripped.count("\n", 0, stripped.index("return 77;")) + 1
    assert line_no == 5, (
        f"expected the offender to stay on line 5 of the original text after "
        f"stripping, computed line {line_no} -- the stripper is shifting line "
        "numbers, which is exactly the bug this control exists to catch."
    )


def test_strip_comments_does_not_hide_code_after_a_slash_in_a_string_literal() -> None:
    # A naive `//` -> end-of-line stripper treats the `//` inside "// x" as
    # a real comment marker and blanks everything after it on the line,
    # including a `return 77;` that follows the string on the SAME line.
    # _strip_comments must recognise the string literal as a unit and never
    # look for comment markers inside it.
    fixture = 'const char * s = "// x"; int main(){ return 77; }'
    stripped = _strip_comments(fixture)
    assert LITERAL_SKIP_EXIT_RE.search(stripped), (
        f"the // inside the string literal hid `return 77;`: stripped output "
        f"was {stripped!r}. The comment stripper must skip whole string "
        "literals before looking for // or /* inside them."
    )


def test_strip_comments_is_raw_string_aware() -> None:
    # A raw string literal (`R"delim(...)delim"`) can embed an unescaped `"`
    # -- the plain quote-tracker is not enough, because IT is what desyncs
    # on that embedded `"`. Observed for real at
    # tests/peg-parser/test-json-parser.cpp:85-86 (two genuine // comments
    # left unstripped, harmlessly there); the dangerous direction is a false
    # negative, reproduced by the fixture below (blanks a real `return 77;`).
    dangerous = 'const char * s = R"(x " // y)"; return 77;'
    stripped = _strip_comments(dangerous)
    assert LITERAL_SKIP_EXIT_RE.search(stripped) and BROAD_LITERAL_77_RE.search(stripped), (
        f"a `\"` embedded in a raw string literal desynchronised the "
        f"scanner and hid `return 77;`: stripped output was {stripped!r}. "
        "_RAW_STRING_START_RE / the raw-string branch in _strip_comments "
        "must treat the whole R\"(...)\" as one opaque literal."
    )

    # A raw string body containing what LOOK like a block comment and a line
    # comment must not desync either -- both must stay inert (part of the
    # literal), and the real code after the literal must still be seen.
    fake_comments = 'const char * s = R"(/* not a comment */ // also not)"; return 77;'
    stripped = _strip_comments(fake_comments)
    assert LITERAL_SKIP_EXIT_RE.search(stripped), (
        f"a `/*` or `//` inside a raw string body was treated as a real "
        f"comment marker: stripped output was {stripped!r}."
    )

    # A custom delimiter (R"xy(...)xy") whose body contains a bare `)"` must
    # not terminate the literal early -- only the literal's OWN `)xy"`
    # terminator may end it.
    custom_delim = 'const char * s = R"xy(some )" text)xy"; return 77;'
    stripped = _strip_comments(custom_delim)
    assert LITERAL_SKIP_EXIT_RE.search(stripped), (
        f"a bare `)\"` inside a custom-delimiter raw string body terminated "
        f"the literal early: stripped output was {stripped!r}."
    )


def test_strip_comments_agrees_with_an_independent_implementation() -> None:
    # Cross-checks _strip_comments against _independent_strip_comments (a
    # structurally different implementation, per _INDEPENDENT_TOKEN_RE's
    # comment) across every real file in scope, INCLUDING tests/test-skip.h
    # itself -- byte-for-byte agreement on the stripped output catches both
    # over-blanking (real code wrongly treated as a comment) and
    # under-blanking (a real comment left unstripped) alike. This replaces
    # a circular self-check (commit 4d1420116) that validated a scanner's
    # survivors against spans that SAME scanner had recorded, so a desynced
    # scanner's phantom span could cover its own survivors and pass
    # trivially. See test_the_cross_check_actually_fires for the RED control
    # proving this comparison can actually fail.
    mismatches = []
    scanned = 0
    for path in _iter_test_sources():
        scanned += 1
        original = path.read_text(encoding="utf-8", errors="replace")
        by_hand_written = _strip_comments(original)
        by_independent = _independent_strip_comments(original)
        if by_hand_written != by_independent:
            for offset, (a, b) in enumerate(zip(by_hand_written, by_independent)):
                if a != b:
                    line_no = by_hand_written.count("\n", 0, offset) + 1
                    mismatches.append(
                        f"{path.relative_to(ROOT)}:{line_no}: hand-written scanner produced "
                        f"{a!r}, independent regex tokenizer produced {b!r}"
                    )
                    break

    _assert_scan_is_not_vacuous(scanned, "test_strip_comments_agrees_with_an_independent_implementation")

    assert not mismatches, (
        "_strip_comments and _independent_strip_comments -- two separately "
        "implemented comment/literal scanners -- disagree on what is code "
        "vs. comment/literal for at least one real file. One of them has a "
        "bug. Mismatches:\n  " + "\n  ".join(mismatches)
    )


def test_the_cross_check_actually_fires() -> None:
    # The RED control for test_strip_comments_agrees_with_an_independent_
    # implementation: a cross-check that never disagrees is exactly as
    # suspect as the circular self-check it replaced, unless it is shown to
    # disagree on a KNOWN-broken scanner.
    # `_strip_comments(text, _disable_raw_strings=True)` is that known-broken
    # scanner -- the pre-raw-string-aware behaviour, which desyncs on an
    # embedded `"` inside a raw string -- reached without duplicating
    # _strip_comments' body.
    def broken(text: str) -> str:
        return _strip_comments(text, _disable_raw_strings=True)

    dangerous = 'const char * s = R"(x " // y)"; return 77;'
    assert broken(dangerous) != _independent_strip_comments(dangerous), (
        "the cross-check did not fire on the dangerous fixture: the "
        "raw-string-disabled scanner agreed with the independent parser, "
        "which means test_strip_comments_agrees_with_an_independent_"
        "implementation would not have caught the regression this control "
        "exists to prove it can catch."
    )

    peg_parser_path = ROOT / "tests" / "peg-parser" / "test-json-parser.cpp"
    assert peg_parser_path.is_file(), f"{peg_parser_path} is missing -- the real-file half of this control cannot run."
    peg_parser_text = peg_parser_path.read_text(encoding="utf-8", errors="replace")
    assert broken(peg_parser_text) != _independent_strip_comments(peg_parser_text), (
        "the cross-check did not fire on the real peg-parser file: the "
        "raw-string-disabled scanner agreed with the independent parser "
        "there too, which would leave that real-world desync undetected."
    )

    # And the un-broken scanner must NOT disagree with the independent
    # parser on these same two fixtures -- otherwise this control would be
    # unable to distinguish "the check works" from "the independent parser
    # itself is unreliable on these inputs".
    assert _strip_comments(dangerous) == _independent_strip_comments(dangerous), (
        "the un-broken (raw-string-aware) scanner disagrees with the "
        "independent parser on the dangerous fixture -- this control cannot "
        "tell 'the check works' apart from 'the independent parser is wrong "
        "here' unless this holds."
    )
    assert _strip_comments(peg_parser_text) == _independent_strip_comments(peg_parser_text), (
        "the un-broken (raw-string-aware) scanner disagrees with the "
        "independent parser on the real peg-parser file -- same concern as "
        "the dangerous-fixture assertion above, for the real-file half."
    )


def test_broad_literal_77_regex_has_controls() -> None:
    # The narrow LITERAL_SKIP_EXIT_RE cannot see 77 bound to a named
    # constant that is only later `return`ed (`static const int X = 77;`
    # ... `return X;`), which is exactly the shape every one of the ten
    # round-1 offenders used. BROAD_LITERAL_77_RE exists to catch that
    # shape; prove it fires on the three spellings the offenders actually
    # used, and on 77 with an integer-literal suffix.
    for positive in (
        "static const int X = 77;",
        "#define X 77",
        "constexpr int X = 77;",
        "return 77u;",
        "return 77U;",
        "return 77L;",
    ):
        assert BROAD_LITERAL_77_RE.search(positive), (
            f"BROAD_LITERAL_77_RE did not match {positive!r} -- the widened "
            "scan would miss exactly the shape it was added for."
        )

    # Negative controls: shapes that must NOT match on purely syntactic
    # grounds, so they never need an allowlist entry at all.
    for benign in (
        "ms.add_kv(LLM_KV_RESIDUAL_SCALE, 3.5565588200778455f);",  # 77 embedded in a longer digit run
        "std::fill(output.ptr, output.ptr + output_values, 77.0f);",  # 77. -- a float literal, not a bare 77
    ):
        assert not BROAD_LITERAL_77_RE.search(benign), (
            f"BROAD_LITERAL_77_RE false-positived on benign text {benign!r}."
        )

    # Array-size shapes are the one case this file DELIBERATELY does not
    # exclude structurally (see the ALLOWED_NON_EXIT_77 comment above): a
    # `[77]` is not syntactically distinguishable from any other bracketed
    # non-exit use, so BROAD_LITERAL_77_RE matches it like any other bare 77,
    # and a real occurrence needs a specific ALLOWED_NON_EXIT_77 entry (by
    # entry, not by shape) rather than a general bracket exclusion that could
    # just as easily hide a real offender written as `return arr[77];`.
    assert BROAD_LITERAL_77_RE.search("int arr[77];"), (
        "BROAD_LITERAL_77_RE unexpectedly ignored an array-size literal -- "
        "if a structural bracket exclusion was added, this test (and the "
        "'by entry, not by shape' decision it documents) must be updated "
        "together with ALLOWED_NON_EXIT_77."
    )


def test_allowed_non_exit_77_entries_are_all_live() -> None:
    # A stale ALLOWED_NON_EXIT_77 entry fails SILENTLY today: if the line it
    # named moves or is deleted, the entry simply never matches anything,
    # so it neither hides a real offender (there is nothing left to hide)
    # nor is itself flagged as unnecessary -- it just accretes as dead
    # weight nobody notices. Assert every entry matches at least one line of
    # its target file's stripped text.
    dead = []
    stripped_lines_by_file: dict[str, list[str]] = {}
    for rel_path, pattern in ALLOWED_NON_EXIT_77:
        if rel_path not in stripped_lines_by_file:
            full_path = ROOT / rel_path
            assert full_path.is_file(), f"ALLOWED_NON_EXIT_77 names {rel_path}, which does not exist"
            text = full_path.read_text(encoding="utf-8", errors="replace")
            stripped_lines_by_file[rel_path] = _strip_comments(text).splitlines()
        if not any(pattern.search(line) for line in stripped_lines_by_file[rel_path]):
            dead.append(f"{rel_path}: {pattern.pattern!r} matches no line")

    assert not dead, (
        "ALLOWED_NON_EXIT_77 has an entry that matches nothing in its target "
        "file -- delete it, or the line it was meant to exempt moved or was "
        "deleted without the allowlist being updated. Dead entries:\n  " + "\n  ".join(dead)
    )


def test_allowlist_matcher_has_controls() -> None:
    # The allowlist itself needs a control, the same way the regexes do: an
    # allowlist entry that matches everything would make
    # test_no_unallowlisted_bare_77_outside_header pass vacuously no matter
    # what the tree contains.
    assert ALLOWED_NON_EXIT_77, "ALLOWED_NON_EXIT_77 is empty -- nothing to control against"

    # Positive: a real entry's file + a line matching its regex is
    # allowlisted. Selected by searching for the file rather than indexing
    # ALLOWED_NON_EXIT_77[0], so this control survives the allowlist being
    # reordered or extended.
    sample_path = "tests/test-backend-ops.cpp"
    sample_entries = [entry for entry in ALLOWED_NON_EXIT_77 if entry[0] == sample_path]
    assert sample_entries, f"expected at least one ALLOWED_NON_EXIT_77 entry for {sample_path}"
    sample_line = "test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, 77, {12,1}, {1,1}));"
    assert _is_allowlisted(sample_path, sample_line), (
        f"{sample_path}'s tensor-dims ALLOWED_NON_EXIT_77 entry did not match "
        f"a known tensor-dims line ({sample_line!r}) -- the allowlist matcher "
        "regressed."
    )

    # Negative: the same file, but a line that merely contains a bare 77 in an
    # unrelated shape, is NOT allowlisted -- proves the entry regex is scoped
    # to its specific shape rather than "any line in this file".
    assert not _is_allowlisted(sample_path, "    return 77;  // not a tensor-dims line"), (
        f"{sample_path} allowlisted an unrelated 'return 77;' line -- the "
        "matcher is not scoped to the entry's specific shape."
    )

    # Negative: a line that WOULD match an entry's regex, but under a
    # different file, is not allowlisted -- proves matching is keyed on the
    # file too, not just the line content.
    assert not _is_allowlisted("tests/some-other-file.cpp", sample_line), (
        "a line matching a real entry's regex was allowlisted under a "
        "DIFFERENT file -- the matcher is not keyed on file path."
    )


def test_no_unallowlisted_bare_77_outside_header() -> None:
    # The widened scan: same shape as test_no_stray_literal_skip_exit_
    # outside_header above, but over BROAD_LITERAL_77_RE with
    # ALLOWED_NON_EXIT_77 subtracted, so it also catches 77 bound to a named
    # constant (`static const int X = 77;` ... `return X;`) rather than
    # only a bare `return 77;`.
    offenders, scanned = _scan_tree_for_offenders(BROAD_LITERAL_77_RE, allowlist_check=_is_allowlisted)
    _assert_scan_is_not_vacuous(scanned, "test_no_unallowlisted_bare_77_outside_header")

    assert not offenders, (
        "found the literal 77 bound to what looks like a skip-exit constant "
        "outside tests/test-skip.h -- either route it through "
        "LLAMA_TEST_EXIT_SKIP (see llama.cpp-g290), or, if it is genuinely "
        "not an exit code (a tensor dimension, a fixture id, a sentinel "
        "value), add a re-verified entry to ALLOWED_NON_EXIT_77 keyed on the "
        "file and the line's content. Offenders:\n  " + "\n  ".join(offenders)
    )
