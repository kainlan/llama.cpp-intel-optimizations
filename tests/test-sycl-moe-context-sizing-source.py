#!/usr/bin/env python3
"""Source pins: the runtime context transaction sizes the per-context MoE structures from the context's micro-batch.

- The immutable Q1_0/NVFP4 host recipes' CPU workspace is re-sized at the context's micro-batch, in tokens, by the
  function the load sized it with. It sits after the transaction sets next_plan.planner_n_ubatch and before the ring's
  pending terms are built, so the probe and the real call both run it, and an overflow refuses the context.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

# One left-to-right pass, so a `/*` inside a `//` comment cannot open a block comment.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def squash(text: str) -> str:
    """One space for every whitespace run, and none after `(` or before `)` and `,`, so a call clang-format wraps
    still matches its one-line spelling."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"\s+,", ",", text)


GGML_SYCL_CPP_CODE = squash(strip_comments(GGML_SYCL_CPP))

_TXN_START = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
_TXN_END = "#undef GGML_SYCL_RUNTIME_TXN_REFUSAL"
_S1_ANCHOR = "next_plan.planner_n_ubatch = next_kv_info.n_ubatch;"
_S8_ANCHOR = "ggml_sycl_ring_kv_zone_inputs ring_kv_zone;"

_RECIPE_RESIZE = (
    "ggml_sycl::placement_moe_host_recipe_workspace_bytes_at(next_plan.moe_host_recipe_shapes, "
    "ggml_sycl::placement_load_n_ubatch(next_plan.planner_n_ubatch), &next_plan.moe_host_recipe_workspace_bytes)"
)
_RECIPE_REFUSE = 'return refuse("MoE host recipe workspace overflows");'


def transaction(code: str) -> str:
    start = code.find(_TXN_START)
    assert start != -1, "the runtime context transaction is not found"
    end = code.find(_TXN_END, start)
    assert end != -1, "the transaction's end is not found"
    return code[start:end]


def between_s1_and_s8(txn: str, needle: str, what: str) -> list:
    s1 = txn.find(_S1_ANCHOR)
    s8 = txn.find(_S8_ANCHOR)
    at = txn.find(needle)
    if s1 == -1 or s8 == -1:
        return [f"{what}: an anchor is missing (S1 {s1}, S8 {s8})"]
    if at == -1:
        return [f"{what}: not found in the transaction"]
    if not s1 < at < s8:
        return [f"{what}: must sit after planner_n_ubatch is set and before the ring's pending terms"]
    return []


def recipe_resize_violations(code: str) -> list:
    txn = transaction(code)
    violations = between_s1_and_s8(txn, _RECIPE_RESIZE, "the host recipe workspace re-size")
    violations += between_s1_and_s8(txn, _RECIPE_REFUSE, "the host recipe workspace refusal")
    return violations


def test_transaction_resizes_the_host_recipe_workspace():
    assert recipe_resize_violations(GGML_SYCL_CPP_CODE) == []


def test_recipe_resize_check_has_a_mutation_witness():
    assert GGML_SYCL_CPP_CODE.count(_RECIPE_RESIZE) == 1, "the re-size must appear exactly once"
    mutated = GGML_SYCL_CPP_CODE.replace(_RECIPE_RESIZE, "true", 1)
    assert recipe_resize_violations(mutated) != [], "the check must catch the re-size being removed"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
