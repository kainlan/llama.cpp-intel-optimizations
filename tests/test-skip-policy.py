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

llama.cpp-g290 round 2 widened the tree-wide scan below from a narrow
return/exit-keyword regex to a broad `\b77\b` literal scan, because the narrow
regex missed 77 bound to a named constant first (`static const int X = 77;`
... `return X;`) rather than written directly as `return 77;`. The narrow scan
stays as a second, stricter check -- see LITERAL_SKIP_EXIT_RE below -- and the
broad one adds a small, explicit, re-verified allowlist for the tree's genuine
non-exit numeric literals (tensor dimensions, fixture ids, sentinel values)
instead of trying to infer intent from context.
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
# a comment. Kept as the second, stricter check alongside BROAD_LITERAL_77_RE
# below (llama.cpp-g290 round 2): a change that breaks the broad scan's
# allowlist should not also silently lose this narrower, allowlist-free one.
LITERAL_SKIP_EXIT_RE = re.compile(r"\breturn\s+77\s*;" r"|\b(?:std::)?_?[Ee]xit\s*\(\s*77\s*\)")

# The broad scan: literal 77 as a standalone integer token, wherever it
# appears in code (not inside a comment or a string/char literal -- see
# _strip_comments). This is what catches 77 bound to a named constant that is
# only later `return`ed, which LITERAL_SKIP_EXIT_RE above cannot see by
# construction (llama.cpp-g290 spec review, finding 2).
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
# other non-exit literal below -- see test_int_array_size_shape_needs_an_allowlist_entry.
BROAD_LITERAL_77_RE = re.compile(r"\b77\b(?!\.\d)")

# Small, explicit, re-verified allowlist of genuine non-exit uses of the
# literal 77 found by BROAD_LITERAL_77_RE across the tree (llama.cpp-g290
# round 2). Keyed on (relative file path, a regex matching the offending
# LINE's content) rather than on line numbers, so it does not rot as the
# surrounding file is edited -- a line-number-keyed allowlist silently stops
# covering its target the moment an unrelated edit shifts it.
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


def _strip_comments(text: str) -> str:
    """Blank out // and /* */ comments, character-for-character in place.

    Every character that is not part of a comment keeps its original
    position, including every newline -- so a line number computed by
    counting "\\n" in the RESULT is the same line number in the ORIGINAL
    text (llama.cpp-g290 spec review, finding 3: the previous DOTALL-based
    stripper deleted block comments instead of blanking them, so offender
    line numbers reported after a multi-line comment were wrong).

    A small hand-written scanner, not a full tokenizer, but string- and
    char-literal aware (finding 4): it skips over "..." and '...' literals
    whole, honouring backslash escapes, BEFORE it ever considers whether `/`
    starts a comment. So a `//` or a `/*` inside a string literal cannot
    swallow the rest of the line (or file) the way a plain
    `re.sub(r"//.*", ...)` would -- see
    test_strip_comments_does_not_hide_code_after_a_slash_in_a_string_literal.
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == quote:
                    j += 1
                    break
                if text[j] == "\n":
                    break  # unterminated literal on this line; stop here
                j += 1
            i = j
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


def _iter_test_sources():
    for src_dir in TEST_SOURCE_DIRS:
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*")):
            if path.is_file() and path.suffix in TEST_SOURCE_EXTS:
                yield path


def _is_allowlisted(rel_path: str, line: str) -> bool:
    return any(entry_path == rel_path and pattern.search(line) for entry_path, pattern in ALLOWED_NON_EXIT_77)


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
    offenders = []
    for path in _iter_test_sources():
        if path.resolve() == SKIP_HEADER.resolve():
            continue  # the one place the literal 77 is allowed to live
        code = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for m in LITERAL_SKIP_EXIT_RE.finditer(code):
            line_no = code.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {m.group(0)!r}")

    non_header_sources = [p for p in _iter_test_sources() if p.resolve() != SKIP_HEADER.resolve()]
    assert non_header_sources, (
        f"found no C/C++ test sources under {TEST_SOURCE_DIRS} other than "
        "test-skip.h itself -- this assertion would pass vacuously (nothing "
        "to scan), which is worse than no assertion at all."
    )

    assert not offenders, (
        "found the literal skip code 77 used as an exit/return value outside "
        "tests/test-skip.h -- use LLAMA_TEST_EXIT_SKIP from tests/test-skip.h "
        "instead (see llama.cpp-g290), so '77 means skip' has exactly one "
        "home. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_strip_comments_preserves_line_numbers() -> None:
    # llama.cpp-g290 spec review, finding 3: the old stripper used
    # re.sub(..., flags=re.DOTALL) to DELETE block comments, which removes
    # their embedded newlines too and shifts every line number after them. A
    # 3-line block comment made a real offender on line 4 get reported as
    # line 2. _strip_comments must blank comment bodies in place instead.
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
    # llama.cpp-g290 spec review, finding 4: a naive `//` -> end-of-line
    # stripper treats the `//` inside "// x" as a real comment marker and
    # blanks everything after it on the line, including a `return 77;` that
    # follows the string on the SAME line. _strip_comments must recognise the
    # string literal as a unit and never look for comment markers inside it.
    fixture = 'const char * s = "// x"; int main(){ return 77; }'
    stripped = _strip_comments(fixture)
    assert LITERAL_SKIP_EXIT_RE.search(stripped), (
        f"the // inside the string literal hid `return 77;`: stripped output "
        f"was {stripped!r}. The comment stripper must skip whole string "
        "literals before looking for // or /* inside them."
    )


def test_broad_literal_77_regex_has_controls() -> None:
    # llama.cpp-g290 spec review, finding 1/2: the narrow LITERAL_SKIP_EXIT_RE
    # cannot see 77 bound to a named constant that is only later `return`ed
    # (`static const int X = 77; ... return X;`), which is exactly the shape
    # every one of the ten round-1 offenders used. BROAD_LITERAL_77_RE exists
    # to catch that shape; prove it fires on the three spellings the offenders
    # actually used.
    for positive in (
        "static const int X = 77;",
        "#define X 77",
        "constexpr int X = 77;",
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


def test_allowlist_matcher_has_controls() -> None:
    # The allowlist itself needs a control, the same way the regexes do: an
    # allowlist entry that matches everything would make
    # test_no_stray_bound_skip_constant_outside_header pass vacuously no
    # matter what the tree contains.
    assert ALLOWED_NON_EXIT_77, "ALLOWED_NON_EXIT_77 is empty -- nothing to control against"

    # Positive: a real entry's file + a line matching its regex is allowlisted.
    sample_path, sample_re = ALLOWED_NON_EXIT_77[0]
    assert sample_path == "tests/test-backend-ops.cpp"
    assert _is_allowlisted(sample_path, "test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, 77, {12,1}, {1,1}));")

    # Negative: the same file, but a line that merely contains a bare 77 in an
    # unrelated shape, is NOT allowlisted -- proves the entry regex is scoped
    # to its specific shape rather than "any line in this file".
    assert not _is_allowlisted(sample_path, "    return 77;  // not a tensor-dims line")

    # Negative: a line that WOULD match an entry's regex, but under a
    # different file, is not allowlisted -- proves matching is keyed on the
    # file too, not just the line content.
    assert not _is_allowlisted("tests/some-other-file.cpp", "test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, 77, {12,1}, {1,1}));")


def test_no_stray_bound_skip_constant_outside_header() -> None:
    # llama.cpp-g290 round 2: the widened scan. Same shape as
    # test_no_stray_literal_skip_exit_outside_header above, but over
    # BROAD_LITERAL_77_RE with ALLOWED_NON_EXIT_77 subtracted, so it also
    # catches 77 bound to a named constant (`static const int X = 77;`
    # ... `return X;`) rather than only a bare `return 77;`.
    offenders = []
    for path in _iter_test_sources():
        if path.resolve() == SKIP_HEADER.resolve():
            continue  # the one place the literal 77 is allowed to live
        rel = str(path.relative_to(ROOT))
        original = path.read_text(encoding="utf-8", errors="replace")
        code = _strip_comments(original)
        lines = code.splitlines()
        for m in BROAD_LITERAL_77_RE.finditer(code):
            line_no = code.count("\n", 0, m.start()) + 1
            line_text = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else ""
            if _is_allowlisted(rel, line_text):
                continue
            offenders.append(f"{rel}:{line_no}: {line_text.strip()!r}")

    non_header_sources = [p for p in _iter_test_sources() if p.resolve() != SKIP_HEADER.resolve()]
    assert non_header_sources, (
        f"found no C/C++ test sources under {TEST_SOURCE_DIRS} other than "
        "test-skip.h itself -- this assertion would pass vacuously (nothing "
        "to scan), which is worse than no assertion at all."
    )

    assert not offenders, (
        "found the literal 77 bound to what looks like a skip-exit constant "
        "outside tests/test-skip.h -- either route it through "
        "LLAMA_TEST_EXIT_SKIP (see llama.cpp-g290), or, if it is genuinely "
        "not an exit code (a tensor dimension, a fixture id, a sentinel "
        "value), add a re-verified entry to ALLOWED_NON_EXIT_77 keyed on the "
        "file and the line's content. Offenders:\n  " + "\n  ".join(offenders)
    )
