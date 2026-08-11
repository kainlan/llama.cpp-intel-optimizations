#!/usr/bin/env python3
"""Source contract for indexed-MoE supports_op residency architecture."""

import re
from collections import Counter
from pathlib import Path
from typing import Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
FUNCTION_START = "static bool ggml_backend_sycl_device_supports_op("
FUNCTION_END = "static bool ggml_backend_sycl_device_supports_buft("
EARLY_GUARD = "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) {"
PLANNER_GUARD = "if (ggml_sycl_op_is_planned_on_host(op, device)) {"
OP_SWITCH = "switch (op->op) {"
NEXT_EARLY_BRANCH = "if (g_moe_multi_gpu_active.load"
TYPE_HELPER_START = "static bool ggml_sycl_mul_mat_type_supported(ggml_type type) {"
TYPE_HELPER_END = FUNCTION_START
MUL_MAT_TYPE_ORDER = (
    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "MXFP4",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
)
MUL_MAT_TYPES = set(MUL_MAT_TYPE_ORDER)


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


def executable_body(body: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    without_comments = re.sub(r"//[^\n]*", "", without_block_comments)
    return re.sub(r"\s+", "", without_comments)


def matching_delimiter(text: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == open_char:
            depth += 1
        elif text[position] == close_char:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError(f"unclosed delimiter: {open_char}")


def decision_clauses(code: str):
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", "", code)
    decisions = []
    position = 0
    while position < len(code):
        while position < len(code) and code[position].isspace():
            position += 1
        if position >= len(code):
            break
        if re.match(r"if\b", code[position:]):
            condition_open = code.index("(", position + 2)
            condition_close = matching_delimiter(code, condition_open, "(", ")")
            condition = re.sub(r"\s+", "", code[condition_open + 1 : condition_close])
            body_start = condition_close + 1
            while body_start < len(code) and code[body_start].isspace():
                body_start += 1
            if code[body_start] == "{":
                body_close = matching_delimiter(code, body_start, "{", "}")
                body = code[body_start + 1 : body_close]
                position = body_close + 1
            else:
                body_close = code.index(";", body_start)
                body = code[body_start : body_close + 1]
                position = body_close + 1
            decisions.append(("if", condition, tuple(decision_clauses(body))))
            continue
        statement_end = code.index(";", position)
        statement = re.sub(r"\s+", "", code[position:statement_end])
        if statement.startswith("return"):
            decisions.append(("return", statement[len("return") :]))
        position = statement_end + 1
    return decisions


def expected_pre_guard_decisions():
    reject = (("return", "false"),)
    return [
        ("if", "a->ne[3]!=b->ne[3]", reject),
        ("if", "ggml_is_quantized(a_type)&&(ggml_is_permuted(a)||!ggml_is_contiguous(a))", reject),
        ("if", "a_type==GGML_TYPE_Q4_0", (("if", "a->ne[2]!=b->ne[2]||a->ne[2]>2", reject),)),
        ("if", "a_type==GGML_TYPE_Q4_1&&b->ne[1]==1", reject),
        ("if", "a_type==GGML_TYPE_IQ4_NL||a_type==GGML_TYPE_IQ4_XS||a_type==GGML_TYPE_IQ3_XXS||a_type==GGML_TYPE_IQ3_S||a_type==GGML_TYPE_IQ2_XXS||a_type==GGML_TYPE_IQ2_XS||a_type==GGML_TYPE_IQ2_S||a_type==GGML_TYPE_IQ1_S||a_type==GGML_TYPE_IQ1_M", (("if", "b->ne[1]==1&&ggml_nrows(b)>1", reject),)),
        ("if", "src0_type==GGML_TYPE_BF16", reject),
        ("if", "ggml_is_permuted(a)&&!ggml_is_contiguous(a)&&a->ne[2]>1&&a->ne[3]>1&&src0_type==GGML_TYPE_F16", reject),
        ("if", "!ggml_is_permuted(a)&&ggml_is_permuted(b)&&b->ne[2]>1&&b->ne[3]>1&&a->ne[0]>128&&a->ne[2]==1&&src0_type==GGML_TYPE_F16", reject),
    ]


def mul_mat_type_policy(text: str) -> Tuple[Set[str], bool]:
    start = text.index(TYPE_HELPER_START)
    end = text.index(TYPE_HELPER_END, start)
    helper = text[start:end]
    cases = set(re.findall(r"\bcase\s+GGML_TYPE_([A-Z0-9_]+)\s*:", helper))
    default = re.search(r"\bdefault\s*:\s*return\s+(true|false)\s*;", helper)
    if default is None:
        raise ValueError("MUL_MAT type helper has no Boolean default")
    return cases, default.group(1) == "true"


def helper_labels_flow_to_shared_true(body: str) -> bool:
    normalized = executable_body(body)
    match = re.fullmatch(
        r"switch\(type\)\{((?:caseGGML_TYPE_[A-Z0-9_]+:)+)returntrue;default:returnfalse;\}",
        normalized,
    )
    if match is None:
        return False
    labels = re.findall(r"caseGGML_TYPE_([A-Z0-9_]+):", match.group(1))
    return len(labels) == len(set(labels)) and set(labels) == MUL_MAT_TYPES


def mul_mat_type_supported(text: str, type_name: str) -> bool:
    cases, default = mul_mat_type_policy(text)
    return True if type_name in cases else default


def contract(text: str) -> bool:
    try:
        function = supports_function(text)
        early, early_close, early_body = braced_body(function, EARLY_GUARD)
        next_branch = function.index(NEXT_EARLY_BRANCH, early_close)
        planner = function.index(PLANNER_GUARD)
        switch = function.index(OP_SWITCH)
        allowed_types, default_type_support = mul_mat_type_policy(text)
        _, _, type_helper_body = braced_body(text, TYPE_HELPER_START)
    except ValueError:
        return False

    switch_body = function[switch:]
    mul_mat_start = switch_body.index("case GGML_OP_MUL_MAT:")
    mul_mat_end = switch_body.index("case GGML_OP_OUT_PROD:", mul_mat_start)
    mul_mat_case = switch_body[mul_mat_start:mul_mat_end]
    case_body_open = mul_mat_case.index("{")
    type_guard = "if (!ggml_sycl_mul_mat_type_supported(a_type)) {"
    try:
        type_guard_start, _, type_guard_body = braced_body(mul_mat_case, type_guard)
    except ValueError:
        return False
    later_indexed_case = re.search(r"\bcase\s+GGML_OP_MUL_MAT_ID\s*:", switch_body)
    return (
        early < early_close < next_branch < planner < switch
        and executable_body(early_body) == "returntrue;"
        and "GGML_OP_ADD_ID" in function[early : early_close + 1]
        and "GGML_OP_MUL_MAT_ID" in function[early : early_close + 1]
        and later_indexed_case is None
        and len(re.findall(r"\bcase\s+GGML_OP_MUL_MAT\s*:", switch_body)) == 1
        and "op->op == GGML_OP_MUL_MAT" not in mul_mat_case
        and allowed_types == MUL_MAT_TYPES
        and not default_type_support
        and helper_labels_flow_to_shared_true(type_helper_body)
        and mul_mat_case.count(type_guard) == 1
        and Counter(decision_clauses(mul_mat_case[case_body_open + 1 : type_guard_start])) ==
            Counter(expected_pre_guard_decisions())
        and executable_body(type_guard_body) == "returnfalse;"
        and executable_body(mul_mat_case[type_guard_start:]) ==
            "if(!ggml_sycl_mul_mat_type_supported(a_type)){returnfalse;}returntrue;}"
    )


def replace_in_supports_function(text: str, old: str, new: str) -> str:
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END, start)
    function = text[start:end]
    assert function.count(old) == 1
    return text[:start] + function.replace(old, new, 1) + text[end:]


def replace_in_type_helper(text: str, old: str, new: str) -> str:
    start = text.index(TYPE_HELPER_START)
    end = text.index(TYPE_HELPER_END, start)
    helper = text[start:end]
    assert helper.count(old) == 1
    return text[:start] + helper.replace(old, new, 1) + text[end:]


def test_indexed_moe_early_return_is_the_only_capability_decision() -> None:
    assert contract(SOURCE)


def test_dense_mul_mat_type_policy_fails_closed() -> None:
    assert mul_mat_type_policy(SOURCE) == (MUL_MAT_TYPES, False)
    assert not mul_mat_type_supported(SOURCE, "Q1_0")
    assert not mul_mat_type_supported(SOURCE, "NVFP4")
    assert not mul_mat_type_supported(SOURCE, "FUTURE_UNKNOWN_TYPE")
    assert mul_mat_type_supported(SOURCE, "Q8_0")
    assert mul_mat_type_supported(SOURCE, "MXFP4")


def test_dense_mul_mat_type_policy_mutations_are_rejected() -> None:
    default_open = replace_in_type_helper(SOURCE, "            return false;", "            return true;")
    assert mul_mat_type_supported(default_open, "Q1_0")
    assert mul_mat_type_supported(default_open, "NVFP4")
    assert not contract(default_open)

    allowed_false = replace_in_type_helper(SOURCE, "            return true;", "            return false;")
    assert not contract(allowed_false)

    reordered = replace_in_type_helper(SOURCE, "        case GGML_TYPE_F32:\n", "")
    reordered = replace_in_type_helper(
        reordered,
        "        case GGML_TYPE_IQ4_XS:\n",
        "        case GGML_TYPE_IQ4_XS:\n        case GGML_TYPE_F32:\n",
    )
    assert contract(reordered)

    for positive in ("Q8_0", "MXFP4"):
        removed = replace_in_type_helper(SOURCE, f"        case GGML_TYPE_{positive}:\n", "")
        assert not mul_mat_type_supported(removed, positive)
        assert not contract(removed)

    unknown_added = replace_in_type_helper(
        SOURCE,
        "        case GGML_TYPE_IQ4_XS:\n",
        "        case GGML_TYPE_IQ4_XS:\n        case GGML_TYPE_COUNT:\n",
    )
    assert mul_mat_type_supported(unknown_added, "COUNT")
    assert not contract(unknown_added)

    type_guard = "                if (!ggml_sycl_mul_mat_type_supported(a_type)) {\n"
    pre_guard_mutations = (
        "                return true;\n",
        "                if (a_type == GGML_TYPE_F32) { return false; }\n",
        "                if (a_type == GGML_TYPE_COUNT) { return 1; }\n",
        "                return static_cast<bool>(1);\n",
    )
    for injected in pre_guard_mutations:
        mutated = replace_in_supports_function(SOURCE, type_guard, injected + type_guard)
        assert not contract(mutated)


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


def test_conditioning_return_on_add_id_is_rejected() -> None:
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert early_body.count("return true;") == 1
    conditional = early_body.replace(
        "return true;",
        "if (op->op == GGML_OP_ADD_ID) { return true; }",
        1,
    )
    assert not contract(replace_in_supports_function(SOURCE, early_body, conditional))


def test_altering_mul_mat_id_guard_is_rejected() -> None:
    altered = replace_in_supports_function(
        SOURCE,
        EARLY_GUARD,
        "if (op->op == GGML_OP_ADD_ID) {",
    )
    assert not contract(altered)


def test_reinserting_later_mul_mat_id_case_is_rejected() -> None:
    mutated = replace_in_supports_function(
        SOURCE,
        "        case GGML_OP_MUL_MAT:\n",
        "        case GGML_OP_MUL_MAT:\n        case GGML_OP_MUL_MAT_ID:\n",
    )
    assert not contract(mutated)
