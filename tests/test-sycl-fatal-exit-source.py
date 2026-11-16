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
    body = braced_block(cpp, helper) if helper >= 0 else ""
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


SETTLE = "settle_after_device_child();"
DEVICE_BRANCHES = [
    'if (std::strcmp(mode, "mmid") == 0 || std::strcmp(mode, "op") == 0) {',
    'if (std::strcmp(mode, "sync") == 0 || std::strcmp(mode, "graph") == 0) {',
]


def unsettled_device_branches(test_cpp: str) -> list:
    """Why a device branch of parent_main can return after its child ran without holding the card for the next one.

    The settle call must be a statement of the branch itself, after the skip check and before every return that
    follows it, so no path out of the branch passes it by. A statement of its own means nothing but whitespace stands
    between it and the `;`, `{` or `}` before it, so a braceless `if (...) settle_after_device_child();` does not count.
    The skip path, where no device process ran, must not wait.
    """
    parent = test_cpp.find("int parent_main(")
    if parent < 0:
        return ["test-sycl-fatal-exit.cpp: parent_main is missing"]
    body = braced_block(test_cpp, parent)
    problems = []
    for head in DEVICE_BRANCHES:
        at = body.find(head)
        if at < 0:
            problems.append(f"parent_main: the branch {head!r} is missing")
            continue
        branch = braced_block(body, at)
        inner = branch[branch.find("{") + 1:-1]
        ran = inner.find("run_child(self, mode)")
        skip_at = inner.find("if (r.exited && r.status == LLAMA_TEST_EXIT_SKIP)")
        if ran < 0 or skip_at < ran:
            problems.append(f"{head!r}: the branch must run its child and then check for a skip")
            continue
        skip = braced_block(inner, skip_at)
        if SETTLE in skip:
            problems.append(f"{head!r}: the skip path waits, but no device process ran")
        rest = inner[skip_at + len(skip):]
        depth, settled_at = 0, -1
        for i, ch in enumerate(rest):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth == 0 and settled_at < 0 and rest.startswith(SETTLE, i):
                before = rest[:i]
                if before[max(before.rfind(c) for c in ";{}") + 1:].strip():
                    problems.append(f"{head!r}: {SETTLE} is not a statement of its own")
                else:
                    settled_at = i
        returns = [m.start() for m in re.finditer(r"\breturn\b", rest)]
        if not returns or settled_at < 0 or any(pos < settled_at for pos in returns):
            problems.append(f"{head!r}: a return after the child ran comes before the branch's own {SETTLE}")
    return problems


def test_a_device_branch_settles_the_card_before_every_return_after_its_child_ran():
    problems = unsettled_device_branches(code_of(SYCL / "tests/test-sycl-fatal-exit.cpp"))
    assert not problems, "\n".join(problems)


SETTLE_WAIT = "std::this_thread::sleep_for(k_device_settle)"
SETTLE_SPACING = re.compile(r"constexpr\s+std::chrono::seconds\s+k_device_settle\s*\{\s*(\d+)\s*\}\s*;")


def settle_wait_problems(test_cpp: str) -> list:
    """Why the settle call does not hold the next process off the card for long enough."""
    problems = []
    at = test_cpp.find("void settle_after_device_child()")
    if at < 0:
        problems.append("test-sycl-fatal-exit.cpp: settle_after_device_child is missing")
    elif SETTLE_WAIT not in braced_block(test_cpp, at):
        problems.append(f"settle_after_device_child must wait: {SETTLE_WAIT} is missing from its body")
    spacing = SETTLE_SPACING.findall(test_cpp)
    if len(spacing) != 1:
        problems.append("test-sycl-fatal-exit.cpp: k_device_settle must be one constexpr std::chrono::seconds")
    elif int(spacing[0]) < 10:
        problems.append(f"k_device_settle is {spacing[0]} s: 12 s is the measured spacing, and runs one second apart "
                        "reset the card")
    return problems


def test_the_settle_call_waits_for_the_measured_spacing():
    problems = settle_wait_problems(code_of(SYCL / "tests/test-sycl-fatal-exit.cpp"))
    assert not problems, "\n".join(problems)


CAPTURE_CALL = "ggml_sycl_fatal_capture();"
CAPTURE_DEF = "static void ggml_sycl_fatal_capture() noexcept"
FATAL_EXIT_DEF = "void ggml_sycl_fatal_exit(int code, const char * message) noexcept {"


def capture_body(cpp: str) -> str:
    at = cpp.find(CAPTURE_DEF)
    assert at >= 0, "common.cpp: ggml_sycl_fatal_capture is missing"
    return braced_block(cpp, at)


def test_a_recorded_loss_captures_before_it_registers_the_exit_handler():
    cpp = code_of(SYCL / "common.cpp")
    guard = cpp.find("if (!g_sycl_device_lost_latched.exchange(true")
    assert guard >= 0, "common.cpp: the first-loss guard on g_sycl_device_lost_latched is missing"
    block = braced_block(cpp, guard)
    call, registration = block.find(CAPTURE_CALL), block.find(REGISTRATION)
    assert 0 <= call < registration, (
        "common.cpp: the first-loss block must take the capture before it registers the exit handler, so the map is "
        "the one taken at the loss"
    )


def test_the_fatal_exit_captures_before_it_ends():
    cpp = code_of(SYCL / "common.cpp")
    at = cpp.find(FATAL_EXIT_DEF)
    assert at >= 0, "common.cpp: the definition of ggml_sycl_fatal_exit is missing"
    body = braced_block(cpp, at)
    call, end = body.find(CAPTURE_CALL), body.find("std::_Exit(code);")
    assert 0 <= call < end, (
        "common.cpp: ggml_sycl_fatal_exit must take the capture before std::_Exit, when no loss took it earlier"
    )


# The calls through which the capture writes a file or prints its line, directly or through its copy helper.
CAPTURE_OUTPUTS = ("open(", "ggml_sycl_fatal_capture_copy(", "fprintf(")


def test_a_second_capture_call_returns_before_it_writes():
    body = capture_body(code_of(SYCL / "common.cpp"))
    guard = body.find("done.exchange(true")
    outputs = [at for at in (body.find(call) for call in CAPTURE_OUTPUTS) if at >= 0]
    after = body[body.find("{", guard) + 1:] if guard >= 0 else ""
    returns = after.lstrip().startswith("return;")
    assert "static std::atomic<bool> done" in body and outputs and 0 <= guard < min(outputs) and returns, (
        "common.cpp: ggml_sycl_fatal_capture must return on a second call, through done.exchange(true, ...), before "
        "it writes a file or prints a line: the loss and the fatal exit both call it, and the files are written once"
    )


def test_the_capture_is_linux_only():
    body = capture_body(code_of(SYCL / "common.cpp"))
    inner = body[body.find("{") + 1:body.rfind("}")].strip()
    assert inner.startswith("#if defined(__linux__)") and inner.endswith("#endif"), (
        "common.cpp: the body of ggml_sycl_fatal_capture must sit inside #if defined(__linux__): it reads /proc/self"
    )


SETTLE_CALL = "ggml_sycl::exit_settle();"


def fatal_exit_settle_problems(cpp: str) -> list:
    """Why ggml_sycl_fatal_exit does not settle as its last step before the exit."""
    at = cpp.find(FATAL_EXIT_DEF)
    if at < 0:
        return ["common.cpp: ggml_sycl_fatal_exit is missing"]
    body = braced_block(cpp, at)
    signal_at = body.find("ggml_sycl::cpu_job_signal_signal_all();")
    settle_at = body.find(SETTLE_CALL)
    exit_at = body.find("std::_Exit(code);")
    problems = []
    if body.count(SETTLE_CALL) != 1 or not 0 <= signal_at < settle_at < exit_at:
        problems.append("ggml_sycl_fatal_exit must call ggml_sycl::exit_settle() once, after it released the CPU "
                        "expert jobs and before std::_Exit(code): a ring parked on such a job is not idle")
    elif body[settle_at + len(SETTLE_CALL):exit_at].strip():
        problems.append("ggml_sycl_fatal_exit: nothing may stand between the settle and std::_Exit(code)")
    return problems


def test_the_fatal_exit_settles_last():
    problems = fatal_exit_settle_problems(code_of(SYCL / "common.cpp"))
    assert not problems, "\n".join(problems)


def test_the_settle_reader_refuses_a_settle_before_the_job_release():
    cpp = code_of(SYCL / "common.cpp")
    body = braced_block(cpp, cpp.find(FATAL_EXIT_DEF))
    moved = body.replace(SETTLE_CALL, "", 1).replace("ggml_sycl::cpu_job_signal_signal_all();",
                                                     SETTLE_CALL + "\n    ggml_sycl::cpu_job_signal_signal_all();", 1)
    assert fatal_exit_settle_problems(cpp.replace(body, moved, 1)), "the reader accepted a settle before the release"


SETTLE_INCLUDES = ("<cerrno>", "<chrono>", "<climits>", "<cstdio>", "<cstdlib>", "<thread>")
INCLUDE = re.compile(r"^\s*#\s*include\s*([<\"][^>\"]+[>\"])", flags=re.M)


def settle_header_problems(hpp: str) -> list:
    """Why exit-settle.hpp could reach a device: it may include only the standard headers a sleep and a print need,
    so no queue, event or wait of the backend or of SYCL is in reach, through any include."""
    found = INCLUDE.findall(hpp)
    problems = [f"exit-settle.hpp: #include {name} is not allowed: the settle may only sleep and print"
                for name in found if name not in SETTLE_INCLUDES]
    problems += [f"exit-settle.hpp: #include {name} is missing" for name in SETTLE_INCLUDES if name not in found]
    return problems


def test_the_settle_touches_no_device():
    problems = settle_header_problems(code_of(SYCL / "exit-settle.hpp"))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("extra", ['#include "ggml-backend.h"', "#include <sycl/sycl.hpp>"])
def test_the_settle_header_reader_refuses_a_backend_include(extra):
    hpp = code_of(SYCL / "exit-settle.hpp")
    problems = settle_header_problems(hpp.replace("#include <thread>", "#include <thread>\n" + extra, 1))
    name = extra.split(None, 1)[1]
    assert any(name in p for p in problems), f"the reader accepted {extra}"


PROCESS_END = re.compile(r"\b(?:std::)?(?:abort|_Exit)\s*\(")
# One alternation, so whichever token starts first wins: a // inside a string is not a comment, and a quote inside a
# comment or a character literal does not open a string.
COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'", flags=re.S)
# Files whose process ends are deliberate as they are: the number of abort( and _Exit( calls the file keeps, and why.
UNSETTLED_ALLOWED = {
    "a7l5w-probe.hpp": (2, "debug-build instrumentation: the dump must hold the state at the failing check, so other "
                           "threads must not run on first"),
}


def blank_comments_and_literals(text: str) -> str:
    """The text with comments, string literals and character literals blanked, keeping every newline, so an offset
    or a reported line is the file's own."""
    return COMMENT_OR_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def code_only(path: Path) -> str:
    """The source with comments and literals blanked, line for line, so a reported line is the file's line."""
    assert path.is_file(), f"{path.relative_to(ROOT)}: missing"
    return blank_comments_and_literals(path.read_text(encoding="utf-8"))


def unsettled_process_ends(name: str, code: str) -> list:
    """The abort() and _Exit( calls of one source that the settle does not directly precede.

    Directly means the settle is the statement before the call. Inside a multi-line macro the line continuation
    between the two does not count.
    """
    problems = []
    for m in PROCESS_END.finditer(code):
        before = re.sub(r"\\\s*$", "", code[:m.start()].rstrip()).rstrip()
        if not before.endswith(SETTLE_CALL):
            line = code.count("\n", 0, m.start()) + 1
            problems.append(f"{name}:{line}: {m.group(0)} is not directly preceded by {SETTLE_CALL}")
    return problems


def test_every_abort_and_exit_call_of_the_backend_settles_first():
    """Every literal abort( and _Exit( call in the top-level *.cpp and *.hpp files of the backend settles first.

    It reaches only those literal calls. It does not reach an abort through ggml's ggml_abort (GGML_ABORT,
    GGML_ASSERT), std::terminate from a throw out of a noexcept function, std::exit, a re-raised crash signal, or a
    kill from outside the process.
    """
    problems = []
    for path in sorted(list(SYCL.glob("*.cpp")) + list(SYCL.glob("*.hpp"))):
        if path.name in UNSETTLED_ALLOWED or path.name == "exit-settle.hpp":
            continue
        problems += unsettled_process_ends(path.name, code_only(path))
    assert not problems, "\n".join(problems)


def test_the_process_end_reader_refuses_a_bare_abort():
    assert unsettled_process_ends("x.cpp", "void f() {\n    std::fflush(stderr);\n    std::abort();\n}\n")
    assert not unsettled_process_ends("x.cpp", "void f() {\n    ggml_sycl::exit_settle();\n    std::abort();\n}\n")
    assert not unsettled_process_ends("x.hpp", "#define A do { \\\n    ggml_sycl::exit_settle(); \\\n    abort(); \\\n} while (0)\n")


@pytest.mark.parametrize("source", [
    'void f() {\n    std::fprintf(stderr, "see http://x\\n"); std::abort();\n}\n',
    "void f() {\n    char c='\"'; std::abort();\n}\n",
])
def test_the_process_end_reader_sees_a_call_after_a_literal(source):
    problems = unsettled_process_ends("x.cpp", blank_comments_and_literals(source))
    assert len(problems) == 1 and problems[0].startswith("x.cpp:2:"), problems


def test_the_process_end_reader_reads_the_real_files():
    hpp = (SYCL / "tlsf-allocator.hpp").read_text(encoding="utf-8")
    assert hpp.count(SETTLE_CALL) == 1, "tlsf-allocator.hpp: expected one settle call in TLSF_ASSERT"
    settle_at = hpp.find(SETTLE_CALL)
    abort_line = hpp.count("\n", 0, hpp.find("abort();", settle_at)) + 1
    unsettled = blank_comments_and_literals(hpp.replace(SETTLE_CALL, "", 1))
    problems = unsettled_process_ends("tlsf-allocator.hpp", unsettled)
    assert problems == [f"tlsf-allocator.hpp:{abort_line}: abort( is not directly preceded by {SETTLE_CALL}"], problems


def test_the_allowlist_holds_the_exact_number_of_unsettled_calls():
    for name, (count, _reason) in UNSETTLED_ALLOWED.items():
        found = len(PROCESS_END.findall(code_only(SYCL / name)))
        assert found == count, f"{name}: {found} abort( or _Exit( call(s), the allowlist names {count}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
