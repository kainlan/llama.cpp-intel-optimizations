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


def braced_block(text: str, start: int) -> str:
    """The text from `start` to the closing brace that matches the first opening brace after it."""
    open_at = text.find("{", start)
    assert open_at >= 0, "no opening brace"
    depth = 0
    for i in range(open_at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
    raise AssertionError("no matching closing brace")


REGISTRATION = "std::atexit(ggml_sycl_device_lost_at_exit)"


def guarded_registration_problems(cpp: str) -> list:
    """Why common.cpp does not register the exit handler exactly once, inside the first-loss block."""
    guard = cpp.find("if (!g_sycl_device_lost_latched.exchange(true")
    if guard < 0:
        return ["common.cpp: the first-loss guard on g_sycl_device_lost_latched is missing"]
    block = braced_block(cpp, guard)
    problems = []
    if block.count(REGISTRATION) != 1 or cpp.count(REGISTRATION) != 1:
        problems.append("common.cpp: the exit handler must be registered once, inside the first-loss block")
    if len(re.findall(r"\batexit\s*\(", cpp)) != 1:
        problems.append("common.cpp: the latch must be the only exit handler it registers")
    return problems


def test_latch_registers_its_exit_handler_once():
    problems = guarded_registration_problems(code_of(SYCL / "common.cpp"))
    assert not problems, "\n".join(problems)


def test_the_registration_reader_refuses_a_handler_registered_outside_the_block():
    cpp = code_of(SYCL / "common.cpp")
    guard = cpp.find("if (!g_sycl_device_lost_latched.exchange(true")
    block = braced_block(cpp, guard)
    moved = block.replace(REGISTRATION, "false", 1)
    mutant = cpp.replace(block, moved + "\n    (void) " + REGISTRATION + ";", 1)
    assert mutant != cpp and guarded_registration_problems(mutant), "the reader admits a registration past the block"


def test_a_failed_registration_is_reported():
    cpp = code_of(SYCL / "common.cpp")
    assert re.search(r"if \(" + re.escape(REGISTRATION) + r" != 0\) \{\s*std::fprintf\(\s*stderr,\s*"
                     r"\"\[SYCL-FATAL\] could not register the exit handler; a normal exit runs teardown\\n\"\);"
                     r"\s*std::fflush\(stderr\);", cpp), (
        "common.cpp: a failed std::atexit must print that a normal exit runs teardown")


def test_the_exit_handler_flushes_every_stream_before_the_fast_exit():
    cpp = code_of(SYCL / "common.cpp")
    start = cpp.find("static void ggml_sycl_device_lost_at_exit() {")
    assert start >= 0, "common.cpp: ggml_sycl_device_lost_at_exit is missing"
    body = braced_block(cpp, start)
    flush = body.find("std::fflush(nullptr);")
    fatal = body.find("ggml_sycl_fatal_exit(GGML_SYCL_EXIT_DEVICE_LOST,")
    assert 0 <= flush < fatal, "common.cpp: the exit handler must flush every open stream before the fast exit"


def test_the_sync_diagnostic_is_printed_without_the_log_queue():
    main = code_of(SYCL / "ggml-sycl.cpp")
    start = main.find("static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {")
    assert start >= 0, "ggml-sycl.cpp: ggml_backend_sycl_synchronize is missing"
    body = braced_block(main, start)
    calls = re.findall(r"(\w+(?:::\w+)?)\s*\(\s*(?:stderr,\s*)?\"\[SYCL-SYNC-DIAG\]", body)
    assert calls == ["std::fprintf"], (
        f"ggml-sycl.cpp: the [SYCL-SYNC-DIAG] line must be printed with std::fprintf(stderr, ...), found {calls}")
    diag = body.find("\"[SYCL-SYNC-DIAG]")
    flush = body.find("std::fflush(stderr);", diag)
    assert 0 <= flush < body.find("SYCL_CHECK(err);", diag), (
        "ggml-sycl.cpp: the [SYCL-SYNC-DIAG] line must be flushed before SYCL_CHECK(err) can end the process")


def test_graph_boundary_cleanup_records_a_device_loss():
    main = code_of(SYCL / "ggml-sycl.cpp")
    m = re.search(r"static void ggml_backend_sycl_graph_boundary_exception_cleanup\([^;{]*\)\s*noexcept\s*\{\s*", main)
    assert m, "ggml-sycl.cpp: the definition of ggml_backend_sycl_graph_boundary_exception_cleanup is missing"
    assert main.startswith("ggml_sycl_note_exception(what);", m.end()), (
        "ggml-sycl.cpp: the graph boundary's cleanup must record a device loss as its first statement"
    )


@pytest.mark.parametrize("log", [
    "[SYCL] graph warmup wait failed: %s\\n\", exc.what()",
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
