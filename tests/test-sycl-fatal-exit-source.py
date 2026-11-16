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


def test_check_try_error_records_a_device_loss():
    hpp = code_of(SYCL / "common.hpp")
    start = hpp.find("#define CHECK_TRY_ERROR(expr)")
    assert start >= 0, "common.hpp: CHECK_TRY_ERROR is missing"
    macro = hpp[start:]
    macro = macro[:macro.find("}()")]
    note = macro.find("ggml_sycl_note_exception(e.what());")
    ret = macro.find("return dpct::default_error;")
    assert 0 <= note < ret, "common.hpp: CHECK_TRY_ERROR must record a device loss before it returns the error"


def test_sycl_error_takes_the_fast_exit_when_a_loss_is_latched():
    hpp = code_of(SYCL / "common.hpp")
    start = hpp.find("static void ggml_sycl_error(")
    assert start >= 0, "common.hpp: ggml_sycl_error is missing"
    body = hpp[start:]
    body = body[:body.find("#define SYCL_CHECK(err)")]
    assert re.search(r"ggml_sycl_device_lost_latched\(\).*ggml_sycl_fatal_exit\(.*GGML_ABORT", body, flags=re.S), (
        "common.hpp: ggml_sycl_error must take the fast exit when a device loss is latched, before it aborts"
    )


def test_latch_registers_its_exit_handler_once():
    cpp = code_of(SYCL / "common.cpp")
    start = cpp.find("void ggml_sycl_note_exception(const char * what) noexcept {")
    assert start >= 0, "common.cpp: ggml_sycl_note_exception is missing"
    body = cpp[start:cpp.find("\n}\n", start)]
    guard = body.find("if (!g_sycl_device_lost_latched.exchange(true")
    handler = body.find("std::atexit(ggml_sycl_device_lost_at_exit);")
    assert 0 <= guard < handler, "common.cpp: the exit handler must be registered only by the first recorded loss"
    assert len(re.findall(r"\batexit\s*\(", cpp)) == 1, "common.cpp: the latch must be the only exit handler it registers"


def test_graph_boundary_cleanup_records_a_device_loss():
    main = code_of(SYCL / "ggml-sycl.cpp")
    m = re.search(r"static void ggml_backend_sycl_graph_boundary_exception_cleanup\([^;{]*\)\s*noexcept\s*\{\s*", main)
    assert m, "ggml-sycl.cpp: the definition of ggml_backend_sycl_graph_boundary_exception_cleanup is missing"
    assert main.startswith("ggml_sycl_note_exception(what);", m.end()), (
        "ggml-sycl.cpp: the graph boundary's cleanup must record a device loss as its first statement"
    )


@pytest.mark.parametrize("log", [
    "SYCL graph completion barrier failed: %s\\n\", exc.what()",
    "[SYCL] graph_compute deferred exit barrier failed: %s; draining queue\\n\", exc.what()",
    "[SYCL] graph_compute exit wait failed: %s\\n\", wait_exc.what()",
    "[SYCL] graph_compute exit wait failed: %s\\n\", exc.what()",
])
def test_graph_exit_catches_record_a_device_loss(log):
    main = code_of(SYCL / "ggml-sycl.cpp")
    var = log.rsplit(", ", 1)[1].split(".")[0]
    want = re.compile(r"catch \(const sycl::exception & " + var + r"\) \{\s*ggml_sycl_note_exception\(" + var +
                      r"\.what\(\)\);\s*GGML_LOG_ERROR\(\"" + re.escape(log))
    hits = len(want.findall(main))
    assert hits == 1, f"ggml-sycl.cpp: the catch that logs {log!r} must record a device loss first ({hits} found)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
