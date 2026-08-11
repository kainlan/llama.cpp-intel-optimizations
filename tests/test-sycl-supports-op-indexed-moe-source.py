#!/usr/bin/env python3
"""Source contract for indexed-MoE supports_op residency architecture."""

import re
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
FUNCTION_START = "static bool ggml_backend_sycl_device_supports_op("
FUNCTION_END = "static bool ggml_backend_sycl_device_supports_buft("
EARLY_GUARD = "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) {"
PLANNER_GUARD = "if (ggml_sycl_op_is_planned_on_host(op, device)) {"
OP_SWITCH = "switch (op->op) {"
NEXT_EARLY_BRANCH = "if (g_moe_multi_gpu_active.load"


def braced_body(text: str, header: str) -> Tuple[int, int, str]:
    header_start = text.index(header)
    opening = text.index("{", header_start + len(header) - 1)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return header_start, position, text[opening + 1 : position]
    raise ValueError(f"unclosed guard: {header}")


def supports_function(text: str) -> str:
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END, start)
    return text[start:end]


def contract(text: str) -> bool:
    try:
        function = supports_function(text)
        early, early_close, early_body = braced_body(function, EARLY_GUARD)
        next_branch = function.index(NEXT_EARLY_BRANCH, early_close)
        planner = function.index(PLANNER_GUARD)
        switch = function.index(OP_SWITCH)
    except ValueError:
        return False

    switch_body = function[switch:]
    mul_mat_start = switch_body.index("case GGML_OP_MUL_MAT:")
    mul_mat_end = switch_body.index("case GGML_OP_OUT_PROD:", mul_mat_start)
    mul_mat_case = switch_body[mul_mat_start:mul_mat_end]
    later_indexed_case = re.search(r"\bcase\s+GGML_OP_MUL_MAT_ID\s*:", switch_body)
    return (
        early < early_close < next_branch < planner < switch
        and early_body.count("return true;") == 1
        and "GGML_OP_ADD_ID" in function[early : early_close + 1]
        and "GGML_OP_MUL_MAT_ID" in function[early : early_close + 1]
        and later_indexed_case is None
        and len(re.findall(r"\bcase\s+GGML_OP_MUL_MAT\s*:", switch_body)) == 1
        and "op->op == GGML_OP_MUL_MAT" not in mul_mat_case
    )


def replace_in_supports_function(text: str, old: str, new: str) -> str:
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END, start)
    function = text[start:end]
    assert function.count(old) == 1
    return text[:start] + function.replace(old, new, 1) + text[end:]


def test_indexed_moe_early_return_is_the_only_capability_decision() -> None:
    assert contract(SOURCE)


def test_removing_only_early_return_is_rejected() -> None:
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert early_body.count("return true;") == 1
    mutated_body = early_body.replace("return true;", "", 1)
    assert not contract(replace_in_supports_function(SOURCE, early_body, mutated_body))


def test_moving_return_immediately_outside_guard_is_rejected() -> None:
    function = supports_function(SOURCE)
    early, early_close, early_body = braced_body(function, EARLY_GUARD)
    guard = function[early : early_close + 1]
    assert early_body.count("return true;") == 1
    moved = guard.replace("return true;", "", 1) + "\n    return true;"
    assert not contract(replace_in_supports_function(SOURCE, guard, moved))


def test_reinserting_later_mul_mat_id_case_is_rejected() -> None:
    mutated = replace_in_supports_function(
        SOURCE,
        "        case GGML_OP_MUL_MAT:\n",
        "        case GGML_OP_MUL_MAT:\n        case GGML_OP_MUL_MAT_ID:\n",
    )
    assert not contract(mutated)
