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
# a comment.
LITERAL_SKIP_EXIT_RE = re.compile(r"\breturn\s+77\s*;" r"|\b(?:std::)?_?[Ee]xit\s*\(\s*77\s*\)")


def _strip_comments(text: str) -> str:
    """Best-effort removal of // and /* */ comments before scanning for code.

    Not a real tokenizer -- a string literal containing "//" would confuse it
    -- but nothing in these files does that around a skip exit, and the point
    is only to keep prose that discusses "77" (there is plenty, and all of it
    legitimate) from being mistaken for a `return 77;` / `exit(77)` site.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _iter_test_sources():
    for src_dir in TEST_SOURCE_DIRS:
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*")):
            if path.is_file() and path.suffix in TEST_SOURCE_EXTS:
                yield path


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
