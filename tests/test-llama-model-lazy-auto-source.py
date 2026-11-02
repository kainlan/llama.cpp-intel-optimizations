"""Source contract for llama.cpp-z5fn: load_tensors must hand the lazy-AUTO decision the
planner-owns-placement fact for SYCL devices.

tests/test-lazy-mode-resolve.cpp pins llama_lazy_auto_enabled() in isolation; it cannot see
whether src/llama-model.cpp feeds it the right input. If the call site drops
planner_owns_placement (or hardwires it false), the SYCL device props -- which omit
caps.mmap_support -- veto AUTO again and every lazy read silently turns OFF, and the pure
test still passes. This gate pins the wiring: planner_owns_placement is derived from
llama_model_dev_is_sycl(dev.dev) under the SYCL / DL build guard, is stored in the caps vector
passed to llama_lazy_auto_enabled, and a mismatch resolves to LLAMA_LAZY_MODE_OFF.

Host-only text assertions on COMMENT-STRIPPED source, with planted-mutation witnesses.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src/llama-model.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return tok if tok[0] in "\"'" else "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


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


def auto_block(code: str) -> str:
    m = re.search(r"if\s*\(\s*ml\.lazy\.mode\s*==\s*LLAMA_LAZY_MODE_AUTO\s*\)\s*\{", code)
    assert m, "the lazy AUTO resolution block is missing from load_tensors"
    return norm(code[m.end() - 1 : matching_brace(code, m.end() - 1) + 1])


SYCL_DERIVATION = "const bool planner_owns_placement = llama_model_dev_is_sycl(dev.dev);"
CAPS_PUSH = "lazy_caps.push_back({ props.caps.mmap_support, planner_owns_placement });"
VERDICT = "if (!llama_lazy_auto_enabled(lazy_caps.data(), lazy_caps.size())) { ml.lazy.mode = LLAMA_LAZY_MODE_OFF; }"


def wiring_problems(code: str) -> list:
    block = auto_block(code)
    bad = []
    if SYCL_DERIVATION not in block:
        bad.append("planner_owns_placement is not derived from llama_model_dev_is_sycl(dev.dev)")
    # the derivation must sit under the SYCL / backend-DL build guard, with a false fallback
    guarded = re.search(
        r"#\s*if defined\(GGML_USE_SYCL\) \|\| defined\(GGML_BACKEND_DL\)\s*"
        + re.escape(SYCL_DERIVATION)
        + r"\s*#\s*else\s*const bool planner_owns_placement = false;\s*#\s*endif",
        block,
    )
    if SYCL_DERIVATION in block and not guarded:
        bad.append("the SYCL derivation is not under the GGML_USE_SYCL || GGML_BACKEND_DL guard with a false fallback")
    if CAPS_PUSH not in block:
        bad.append("planner_owns_placement is not passed into the caps given to llama_lazy_auto_enabled")
    if VERDICT not in block:
        bad.append("a rejected AUTO no longer resolves to LLAMA_LAZY_MODE_OFF")
    return bad


def test_load_tensors_passes_planner_owns_placement_into_the_lazy_auto_decision():
    assert wiring_problems(strip_comments(SRC)) == []


def test_mutation_witness_reverting_the_argument_is_caught():
    code = strip_comments(SRC)
    mutated = code.replace(CAPS_PUSH, "lazy_caps.push_back({ props.caps.mmap_support, false });", 1)
    assert mutated != code, "mutation did not apply"
    assert wiring_problems(mutated) == ["planner_owns_placement is not passed into the caps given to llama_lazy_auto_enabled"]


def test_mutation_witness_a_hardwired_false_derivation_is_caught():
    code = strip_comments(SRC)
    mutated = code.replace(SYCL_DERIVATION, "const bool planner_owns_placement = false;", 1)
    assert mutated != code, "mutation did not apply"
    problems = wiring_problems(mutated)
    assert "planner_owns_placement is not derived from llama_model_dev_is_sycl(dev.dev)" in problems


def test_mutation_witness_dropping_the_off_fallback_is_caught():
    code = strip_comments(SRC)
    mutated = code.replace("ml.lazy.mode = LLAMA_LAZY_MODE_OFF;", "", 1)
    assert mutated != code, "mutation did not apply"
    assert wiring_problems(mutated) == ["a rejected AUTO no longer resolves to LLAMA_LAZY_MODE_OFF"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
