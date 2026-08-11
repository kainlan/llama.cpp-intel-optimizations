#!/usr/bin/env python3
"""Source contract for indexed-MoE supports_op residency architecture."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
FUNCTION_START = "static bool ggml_backend_sycl_device_supports_op("
FUNCTION_END = "static bool ggml_backend_sycl_device_supports_buft("
EARLY_GUARD = "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) {"
PLANNER_GUARD = "if (ggml_sycl_op_is_planned_on_host(op, device)) {"
OP_SWITCH = "switch (op->op) {"
NEXT_EARLY_BRANCH = "if (g_moe_multi_gpu_active.load"


def supports_function(text: str) -> str:
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END, start)
    return text[start:end]


def contract(text: str) -> bool:
    try:
        function = supports_function(text)
        early = function.index(EARLY_GUARD)
        early_end = function.index(NEXT_EARLY_BRANCH, early)
        planner = function.index(PLANNER_GUARD)
        switch = function.index(OP_SWITCH)
    except ValueError:
        return False

    early_block = function[early:early_end]
    switch_body = function[switch:]
    later_indexed_case = re.search(r"\bcase\s+GGML_OP_MUL_MAT_ID\s*:", switch_body)
    return (
        early < planner < switch
        and early_block.count("return true;") == 1
        and "GGML_OP_ADD_ID" in early_block
        and "GGML_OP_MUL_MAT_ID" in early_block
        and later_indexed_case is None
        and len(re.findall(r"\bcase\s+GGML_OP_MUL_MAT\s*:", switch_body)) == 1
    )


def replace_in_supports_function(text: str, old: str, new: str) -> str:
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END, start)
    function = text[start:end]
    assert function.count(old) == 1
    return text[:start] + function.replace(old, new, 1) + text[end:]


def test_indexed_moe_early_return_is_the_only_capability_decision() -> None:
    assert contract(SOURCE)


def test_removing_early_return_is_rejected() -> None:
    function = supports_function(SOURCE)
    early = function.index(EARLY_GUARD)
    early_end = function.index(NEXT_EARLY_BRANCH, early)
    early_block = function[early:early_end]
    assert not contract(replace_in_supports_function(SOURCE, early_block, ""))


def test_reinserting_later_mul_mat_id_case_is_rejected() -> None:
    mutated = replace_in_supports_function(
        SOURCE,
        "        case GGML_OP_MUL_MAT:\n",
        "        case GGML_OP_MUL_MAT:\n        case GGML_OP_MUL_MAT_ID:\n",
    )
    assert not contract(mutated)
