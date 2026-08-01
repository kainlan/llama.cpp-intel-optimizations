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
    # get-model.cpp is upstream, so a rebase can revert it to exit(EXIT_SUCCESS)
    # without a conflict. test-thread-safety.cpp held a duplicate of the same
    # logic with `return 0`; the point of the shared helper is that there is one
    # skip policy, so a second copy reappearing is itself the regression.
    for name in ("get-model.cpp", "test-thread-safety.cpp"):
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
