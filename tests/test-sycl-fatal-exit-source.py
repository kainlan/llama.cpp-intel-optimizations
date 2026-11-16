#!/usr/bin/env python3
"""No backend source ends the process with std::exit: a fatal path leaves through ggml_sycl_fatal_exit.

std::exit runs static destructors, and after a device loss the Level Zero library's destructor does not return, so a
process that left a catch through std::exit held its memory and the GPU until it was killed. ggml_sycl_fatal_exit
leaves through std::_Exit instead.

llama_test_pytest hands this file to pytest, so every check lives inside a test_*() function: a file with only
module-level checks collects nothing and pytest's exit 5 fails the ctest. A missing source file or marker fails a test,
never skips it. Checks read comment-stripped text, so prose that mentions std::exit does not count.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

EXIT = re.compile(r"\b(?:std::)?exit\s*\(")


def code_of(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)}: missing"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@pytest.mark.parametrize("name", ["ggml-sycl.cpp", "cpy.cpp", "mmq.cpp"])
def test_no_exit_call_in_backend_source(name):
    hits = len(EXIT.findall(code_of(SYCL / name)))
    assert hits == 0, f"{name}: {hits} exit( call(s); a fatal path must call ggml_sycl_fatal_exit"


def test_common_hpp_keeps_only_the_bad_arch_exit():
    hpp = code_of(SYCL / "common.hpp")
    hpp_hits = [m.start() for m in EXIT.finditer(hpp)]
    bad_arch = hpp.find("static inline void bad_arch(")
    assert len(hpp_hits) == 1 and bad_arch >= 0 and bad_arch < hpp_hits[0] < bad_arch + 400, (
        f"common.hpp: expected exactly one exit( call, inside bad_arch; found {len(hpp_hits)}"
    )


def test_fatal_exit_is_defined_after_the_watchdog():
    cpp = code_of(SYCL / "common.cpp")
    watchdog = cpp.find("static void watchdog_thread_fn(")
    helper = cpp.find("void ggml_sycl_fatal_exit(int code, const char * message) noexcept {")
    assert watchdog >= 0 and helper >= 0 and helper > watchdog, (
        "common.cpp: ggml_sycl_fatal_exit must be defined after watchdog_thread_fn"
    )


def test_fatal_exit_ends_in_exit_without_host_cleanup():
    cpp = code_of(SYCL / "common.cpp")
    helper = cpp.find("void ggml_sycl_fatal_exit(int code, const char * message) noexcept {")
    body = cpp[helper:helper + 900] if helper >= 0 else ""
    assert "std::_Exit(code);" in body and "ggml_sycl_cleanup_host_allocations" not in body, (
        "common.cpp: ggml_sycl_fatal_exit must end in std::_Exit(code) and not call the host cleanup"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
