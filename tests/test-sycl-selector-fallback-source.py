#!/usr/bin/env python3
"""Source gate for llama.cpp-5q1r (plan task S6, following on plan task S2 /
llama.cpp-2x3m): every tests/*.cpp whose own main() pins
ONEAPI_DEVICE_SELECTOR via setenv() must do so through the shared re-exec
fallback (tests/sycl-selector-fallback.hpp, sycl_test_selector_fallback()),
never a bespoke `if (!getenv(...)) setenv(...);` copy and never a bespoke
execv()-of-its-own implementing the same idea a second time.

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
plan task S2) -- both count as offenders here too, since the point of this
gate is exactly one implementation, not several correct ones.
GREEN (after): every one of those calls the shared helper instead.

SCOPE: every tests/*.cpp containing a literal `setenv("ONEAPI_DEVICE_SELECTOR"`
call in its own source, minus the two files in EXCLUDED_FILES below, which
are deliberately not part of this port. This is a superset of the
`test-sycl-*.cpp` glob on purpose -- the census this gate encodes (see
llama.cpp-5q1r) was done with a plain `grep -l` across all of tests/*.cpp,
and several offenders (e.g. tests/test-layout-bytes.cpp,
tests/test-unified-cache-bugs.cpp) do not carry the `test-sycl-` prefix.

This gate reads SOURCE TEXT only -- no compiler, no SYCL device.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SETENV_PATTERN = re.compile(r'setenv\(\s*"ONEAPI_DEVICE_SELECTOR"')
HELPER_CALL_PATTERN = re.compile(r"sycl_test_selector_fallback\s*\(")
MAIN_WITH_ARGV_PATTERN = re.compile(r"int\s+main\s*\([^)]*char\s*\*\*\s*argv[^)]*\)")

# Files that setenv("ONEAPI_DEVICE_SELECTOR" in their own main() but are
# deliberately NOT part of this port (llama.cpp-5q1r, lead guidance c-kjzk):
# both belong to work landed on OTHER, not-yet-integrated branches, and this
# task does not own them. Each entry is the reason, not a rubber stamp --
# re-check it before adding a third name here.
EXCLUDED_FILES = {
    # llama.cpp-dfo0 (plan task L2): the .cpp landed on master, but its
    # tests/CMakeLists.txt registration did not -- `grep -n
    # "compute-buffer-extra-reuse" tests/CMakeLists.txt` returns nothing, so
    # this file is not built or run by ctest in this tree at all today. Its
    # sibling tests/test-sycl-kv-view-extra-reuse.cpp (plan task L2b) does not
    # exist in this tree yet either. Follow-up for L2/L2b's owner once both
    # are registered: port this file to the shared helper and drop this
    # exclusion in the same change.
    "test-sycl-compute-buffer-extra-reuse.cpp",
    # Not registered in tests/CMakeLists.txt at all (`grep -rn
    # "test-tiled-weight-loading" tests/CMakeLists.txt` returns nothing) --
    # dead source, nothing to build or gate. Its ONEAPI_DEVICE_SELECTOR usage
    # is also a different mechanism from every other file here: most
    # occurrences are inside subprocess COMMAND STRINGS
    # (`std::string(...) + binary + ...`) that set the variable for a CHILD
    # llama-cli process it shells out to via run_command() -- a fresh process
    # with its own libccl load, immune to this process's memoization. Only
    # its own top-level main() setenv() shares the bug shape, and porting a
    # file nothing builds would not fix anything a test run could observe.
    "test-tiled-weight-loading.cpp",
}


def _iter_source_files():
    return sorted(p for p in ROOT.glob("*.cpp") if p.name not in EXCLUDED_FILES)


def _files_with_bespoke_setenv():
    """tests/*.cpp files that setenv("ONEAPI_DEVICE_SELECTOR" directly and do
    NOT also call the shared helper -- offenders under either RED shape (the
    original no-op setenv()-only form, or S2's independent execv() copy)."""
    offenders = []
    for path in _iter_source_files():
        text = path.read_text()
        if not SETENV_PATTERN.search(text):
            continue
        if not HELPER_CALL_PATTERN.search(text):
            offenders.append(path.name)
    return offenders


def test_no_bespoke_selector_setenv_outside_excluded_files():
    offenders = _files_with_bespoke_setenv()
    assert not offenders, (
        "the following tests/*.cpp files call setenv(\"ONEAPI_DEVICE_SELECTOR\") "
        "directly instead of the shared sycl_test_selector_fallback() helper "
        "(tests/sycl-selector-fallback.hpp) -- a plain setenv() in main() is a "
        "no-op in this fork, see llama.cpp-2x3m / llama.cpp-5q1r: " + ", ".join(offenders)
    )


def test_excluded_files_are_still_accounted_for():
    """Guards the exclusion list itself against silent rot: each excluded
    name must still exist and still contain the pattern that justified
    excluding it, so a future rename or registration change is caught here
    instead of the exclusion quietly protecting nothing (or the wrong file)."""
    for name in EXCLUDED_FILES:
        path = ROOT / name
        assert path.is_file(), f"excluded file {name} no longer exists -- update this gate's EXCLUDED_FILES"
        text = path.read_text()
        assert SETENV_PATTERN.search(text), (
            f"excluded file {name} no longer contains a bespoke ONEAPI_DEVICE_SELECTOR setenv() -- "
            "the exclusion may no longer be needed; re-check and drop it from EXCLUDED_FILES if so"
        )


def test_helper_header_exists_and_defines_the_function():
    header = ROOT / "sycl-selector-fallback.hpp"
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
            missing.append(path.name)
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
