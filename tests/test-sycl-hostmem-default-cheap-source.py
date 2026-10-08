"""Source contract for llama.cpp-z5fn: the default [HOSTMEM] path must not reach
the expensive readers.

test-host-mem-plan.cpp pins the decision (host_mem_plan_for). This gate pins the
wiring: in ggml_sycl_log_host_mem, the calls to mallinfo2() and
ggml_sycl_host_mem_top_anon_mappings() (a /proc/self/smaps walk) may appear only
inside the `if (plan.full) { ... }` block, and the graph_compute ladder may run
only under `if (ggml_sycl_host_mem_full())`. Without it a later edit could move a
reader above the guard and the plan test would still pass.

Host-only text assertions on COMMENT-STRIPPED source, with a planted mutation
as the positive control.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)

EXPENSIVE = ("mallinfo2(", "ggml_sycl_host_mem_top_anon_mappings(")


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return tok if tok[0] in "\"'" else "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def matching_brace(code: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def function_body(code: str, signature: str) -> str:
    start = code.index(signature)
    brace = code.index("{", start)
    return code[brace : matching_brace(code, brace) + 1]


def unguarded_expensive_calls(code: str) -> list:
    body = function_body(code, "bool ggml_sycl_log_host_mem(ggml_sycl::host_mem_phase phase")
    guard = re.search(r"if\s*\(\s*plan\.full\s*\)\s*\{", body)
    assert guard, "the plan.full guard block is missing from ggml_sycl_log_host_mem"
    g_open = guard.end() - 1
    g_close = matching_brace(body, g_open)
    bad = []
    for name in EXPENSIVE:
        for m in re.finditer(re.escape(name), body):
            if not (g_open < m.start() < g_close):
                bad.append(name)
    return bad


def test_expensive_readers_only_under_plan_full():
    code = strip_comments(SRC)
    body = function_body(code, "bool ggml_sycl_log_host_mem(ggml_sycl::host_mem_phase phase")
    for name in EXPENSIVE:
        assert name in body, f"{name} no longer appears in the log function; the gate would pass vacuously"
    assert unguarded_expensive_calls(code) == []


def test_mutation_witness_a_reader_above_the_guard_is_caught():
    code = strip_comments(SRC)
    mutated = code.replace(
        "const ggml_sycl::host_mem_plan plan =",
        "size_t planted = mallinfo2().uordblks; (void) planted;\n    const ggml_sycl::host_mem_plan plan =",
        1,
    )
    assert mutated != code, "mutation did not apply"
    assert unguarded_expensive_calls(mutated) == ["mallinfo2("]


LOG_SIG = "bool ggml_sycl_log_host_mem(ggml_sycl::host_mem_phase phase"
PLAN_CALL = re.compile(
    r"host_mem_plan\s+plan\s*=\s*ggml_sycl::host_mem_plan_for\(\s*phase\s*,\s*full_mode\s*,\s*first_pp_to_tg\s*\)"
)
FULL_MODE_DECL = re.compile(r"const\s+bool\s+full_mode\s*=\s*ggml_sycl_host_mem_full\(\)\s*;")
EARLY_OUT = re.compile(r"if\s*\(\s*!\s*plan\.emit\s*\)\s*\{\s*return\s+false\s*;\s*\}")


def plan_wiring_problems(code: str) -> list:
    """The plan must come from host_mem_plan_for(phase, <the env-derived full_mode>, ...), full_mode must
    be ggml_sycl_host_mem_full(), and the emit early-out must precede every reader."""
    body = function_body(code, LOG_SIG)
    bad = []
    if not FULL_MODE_DECL.search(body):
        bad.append("full_mode is not initialised from ggml_sycl_host_mem_full()")
    plan = PLAN_CALL.search(body)
    if not plan:
        bad.append("plan is not host_mem_plan_for(phase, full_mode, first_pp_to_tg)")
    out = EARLY_OUT.search(body)
    if not out:
        bad.append("the `if (!plan.emit) return false;` early-out is missing")
    if plan and out and out.start() < plan.start():
        bad.append("the emit early-out precedes the plan")
    if out:
        for name in EXPENSIVE:
            i = body.find(name)
            if i != -1 and i < out.end():
                bad.append(f"{name} runs before the emit early-out")
    return bad


def test_plan_comes_from_the_env_and_the_early_out_stays():
    code = strip_comments(SRC)
    assert plan_wiring_problems(code) == []


def test_mutation_witness_a_hardwired_full_mode_is_caught():
    code = strip_comments(SRC)
    mutated = code.replace("host_mem_plan_for(phase, full_mode, first_pp_to_tg)",
                           "host_mem_plan_for(phase, true, first_pp_to_tg)", 1)
    assert mutated != code, "mutation did not apply"
    assert plan_wiring_problems(mutated) == ["plan is not host_mem_plan_for(phase, full_mode, first_pp_to_tg)"]


def test_mutation_witness_a_dropped_early_out_is_caught():
    code = strip_comments(SRC)
    mutated = EARLY_OUT.sub("", code, count=1)
    assert mutated != code, "mutation did not apply"
    assert plan_wiring_problems(mutated) == ["the `if (!plan.emit) return false;` early-out is missing"]


def test_mutation_witness_an_env_independent_full_mode_is_caught():
    code = strip_comments(SRC)
    mutated = FULL_MODE_DECL.sub("const bool full_mode = true;", code, count=1)
    assert mutated != code, "mutation did not apply"
    problems = plan_wiring_problems(mutated)
    assert problems == ["full_mode is not initialised from ggml_sycl_host_mem_full()"]


def test_ladder_runs_only_in_full_mode():
    code = strip_comments(SRC)
    m = re.search(r"if\s*\(\s*ggml_sycl_host_mem_full\(\)\s*\)\s*\{", code)
    assert m, "the ladder guard is missing"
    block = code[m.end() - 1 : matching_brace(code, m.end() - 1) + 1]
    assert "host_mem_phase::LADDER" in block
    # and the ladder is not logged anywhere else
    assert code.count("host_mem_phase::LADDER") == 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
