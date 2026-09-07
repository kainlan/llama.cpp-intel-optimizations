#!/usr/bin/env python3
"""Source gate for llama.cpp-5q1r (plan task S6, following on plan task S2 /
llama.cpp-2x3m): every SYCL GPU test whose own main() pins
ONEAPI_DEVICE_SELECTOR via setenv() must do so either through the shared
re-exec fallback (tests/sycl-selector-fallback.hpp,
sycl_test_selector_fallback()) or through the canonical inline re-exec block
S2 landed first (759b5647d) -- `if (!getenv("ONEAPI_DEVICE_SELECTOR")) {
setenv(...); execv("/proc/self/exe", argv); ...}` -- never a bespoke
`setenv()`-only copy with no execv, and never BOTH the helper call and a
leftover bespoke setenv (a half-finished port reads as a real fix, so that
combination is an offender too).

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
initializer observes the variable already set; see
tests/sycl-selector-fallback.hpp for the full rationale and the guards
(re-exec only when setenv() itself succeeded, warn-and-continue rather than
exit(77) on failure).

RED (before the llama.cpp-5q1r port): ~20 sibling tests called the no-op
`setenv()`-only form directly in main(), and two more
(test-sycl-mmvq-q8-0-soa-numerics.cpp, test-sycl-fattn-tile-d512-decode.cpp)
had already grown their OWN correct-but-independent execv() copy (landed by
plan task S2) -- both counted as offenders in the first version of this
gate, since the point was exactly one implementation, not several correct
ones. Spec round 1 (rev-5q1r-spec-1, c-k45u) found two more offenders this
gate could not even see (ggml/src/ggml-sycl/tests/test-cross-model-weight-
usage.cpp, ggml/src/ggml-sycl/tests/test-canonical-checksum-owner-scope.cpp
-- live, ctest-registered targets in ggml/src/ggml-sycl/CMakeLists.txt) and
one exclusion that was wrong (tests/test-sycl-compute-buffer-extra-reuse.cpp
IS registered on master, in ggml/src/ggml-sycl/CMakeLists.txt, not
tests/CMakeLists.txt -- the earlier "not registered here" rationale checked
only tests/CMakeLists.txt).
GREEN (after both the original port and the spec round 1 fixes): every one
of those calls the shared helper (or, for a sibling mid-review on another
branch, may legitimately still carry the canonical inline block -- see
SCOPE below).

SCOPE: this gate scans TWO directories -- tests/ and
ggml/src/ggml-sycl/tests/ -- for every *.cpp file containing a literal
`setenv("ONEAPI_DEVICE_SELECTOR"` call in its own source, minus the files in
EXCLUDED_FILES below (keyed by path relative to the repo root, since the two
directories could in principle share a basename). This is deliberately a
superset of the `test-sycl-*.cpp` glob and of a single directory: the
census this gate encodes (see llama.cpp-5q1r) was done with a plain
`grep -l` across all of tests/*.cpp, several offenders there do not carry
the `test-sycl-` prefix (e.g. tests/test-layout-bytes.cpp), and spec round 1
found live offenders outside tests/ entirely.

ACCEPTED FORMS (a file that setenv()s ONEAPI_DEVICE_SELECTOR is NOT an
offender if it uses either):
  1. The shared helper: a call to sycl_test_selector_fallback(...).
  2. The canonical inline block: a setenv("ONEAPI_DEVICE_SELECTOR" call
     followed, within a short window, by execv("/proc/self/exe" -- this is
     what plan task S2 landed first, and what a sibling test on another,
     not-yet-merged branch (e.g. plan task L2b's
     tests/test-sycl-kv-view-extra-reuse.cpp) may still legitimately carry;
     consolidating it onto the shared helper is a fine follow-up, but must
     not be a hard gate condition that breaks a merge in flight.
A file combining BOTH forms (calls the helper AND still has a bespoke
setenv("ONEAPI_DEVICE_SELECTOR" left in) is an offender: there is no
legitimate reason a helper-calling file needs its own literal setenv() for
this variable, and the combination is exactly what a half-finished
consolidation would leave behind.

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
MAIN_WITH_ARGV_PATTERN = re.compile(r"int\s+main\s*\([^)]*char\s*\*\*\s*argv[^)]*\)")

# Files that setenv("ONEAPI_DEVICE_SELECTOR" in their own main() but are
# deliberately NOT part of this port (llama.cpp-5q1r, lead guidance c-kjzk),
# keyed by path relative to the repo root (POSIX separators). Each entry is
# the reason, not a rubber stamp -- re-check it before adding another name,
# and see test_excluded_files_are_still_accounted_for, which makes an
# exclusion self-expire the moment its stated reason stops being true.
EXCLUDED_FILES = {
    # Not registered in EITHER CMakeLists.txt (`grep -rn
    # "test-tiled-weight-loading" tests/CMakeLists.txt
    # ggml/src/ggml-sycl/CMakeLists.txt` returns nothing) -- dead source,
    # nothing to build or gate. Its ONEAPI_DEVICE_SELECTOR usage is also a
    # different mechanism from every other file here: most occurrences are
    # inside subprocess COMMAND STRINGS (`std::string(...) + binary + ...`)
    # that set the variable for a CHILD llama-cli process it shells out to
    # via run_command() -- a fresh process with its own libccl load, immune
    # to this process's memoization. Only its own top-level main() setenv()
    # shares the bug shape, and porting a file nothing builds would not fix
    # anything a test run could observe.
    "tests/test-tiled-weight-loading.cpp",
}


def _iter_source_files():
    files = []
    for d in SCAN_DIRS:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.cpp")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in EXCLUDED_FILES:
                continue
            files.append(path)
    return files


def _has_canonical_inline_block(text):
    """True if a setenv("ONEAPI_DEVICE_SELECTOR" call is followed, within a
    short window, by execv("/proc/self/exe" -- the canonical inline re-exec
    block plan task S2 landed first, still valid for a file that has not yet
    been consolidated onto the shared helper."""
    for m in SETENV_PATTERN.finditer(text):
        window = text[m.end() : m.end() + 400]
        if EXECV_PATTERN.search(window):
            return True
    return False


def _classify(text):
    """Returns one of "not_in_scope", "offender_half_finished",
    "offender_bare_setenv", "compliant" for a single file's text."""
    has_helper = bool(HELPER_CALL_PATTERN.search(text))
    has_setenv = bool(SETENV_PATTERN.search(text))
    if not has_setenv and not has_helper:
        return "not_in_scope"
    if has_helper and has_setenv:
        return "offender_half_finished"
    if has_setenv and not _has_canonical_inline_block(text):
        return "offender_bare_setenv"
    return "compliant"


def _offenders():
    offenders = []
    for path in _iter_source_files():
        verdict = _classify(path.read_text())
        if verdict.startswith("offender"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel} ({verdict})")
    return offenders


def test_no_offenders_outside_excluded_files():
    offenders = _offenders()
    assert not offenders, (
        "the following files call setenv(\"ONEAPI_DEVICE_SELECTOR\") without the shared "
        "sycl_test_selector_fallback() helper or the canonical inline execv() block, or call the "
        "helper while still retaining a bespoke setenv() (a half-finished port) -- a plain "
        "setenv() in main() is a no-op in this fork, see llama.cpp-2x3m / llama.cpp-5q1r: "
        + ", ".join(offenders)
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


def test_excluded_files_are_still_accounted_for():
    """Guards the exclusion list itself against silent rot: each excluded
    file must still exist, still contain a bespoke ONEAPI_DEVICE_SELECTOR
    setenv() (the pattern that justified excluding it), and -- since every
    current exclusion's stated reason is "not registered anywhere" -- its
    bare name must still be ABSENT from both CMakeLists.txt files. The
    moment a file is registered, this assertion fails and forces a human to
    either port it or write a new, still-true reason instead of the
    exclusion silently protecting a now-built, now-ctest-registered test
    (spec round 1, c-k45u finding 3)."""
    for rel in EXCLUDED_FILES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"excluded file {rel} no longer exists -- update this gate's EXCLUDED_FILES"
        text = path.read_text()
        assert SETENV_PATTERN.search(text), (
            f"excluded file {rel} no longer contains a bespoke ONEAPI_DEVICE_SELECTOR setenv() -- "
            "the exclusion may no longer be needed; re-check and drop it from EXCLUDED_FILES if so"
        )
        stem = path.stem  # basename without .cpp -- the CMake target/source name
        for cmake_path in CMAKE_FILES:
            cmake_text = cmake_path.read_text()
            assert stem not in cmake_text, (
                f"excluded file {rel} (target name '{stem}') now appears in "
                f"{cmake_path.relative_to(REPO_ROOT)} -- it has been registered, so the "
                "\"not registered anywhere\" exclusion reason no longer holds. Port it to the "
                "shared helper and drop it from EXCLUDED_FILES instead of leaving the exclusion in place."
            )


def test_helper_header_exists_and_defines_the_function():
    header = TESTS_DIR / "sycl-selector-fallback.hpp"
    assert header.is_file(), "tests/sycl-selector-fallback.hpp is missing"
    text = header.read_text()
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
        text = path.read_text()
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
