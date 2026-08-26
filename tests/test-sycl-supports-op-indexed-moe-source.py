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
ROUTER_FLAG = "const bool is_multi_gpu_router_logits ="
PLANNER_GUARD = "if (!is_multi_gpu_router_logits && ggml_sycl_op_is_planned_on_host(op, device)) {"
OP_SWITCH = "switch (op->op) {"
EXPECTED_PRE_INDEXED_GUARD_PREFIX = (
    "ggml_backend_sycl_device_context*sycl_ctx="
    "(ggml_backend_sycl_device_context*)dev->context;"
    "intdevice=sycl_ctx->device;"
)
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
        if re.match(r"else\b", code[position:]):
            raise ValueError("else clauses are not valid decision-contract syntax")
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
        if not statement:
            raise ValueError("empty statement is not valid decision-contract syntax")
        if statement.startswith("return"):
            decisions.append(("return", statement[len("return") :]))
        else:
            decisions.append(("statement", statement))
        position = statement_end + 1
    return decisions


def expected_pre_guard_decisions():
    reject = (("return", "false"),)
    return [
        ("statement", "structggml_tensor*a=op->src[0]"),
        ("statement", "structggml_tensor*b=op->src[1]"),
        ("if", "a->ne[3]!=b->ne[3]", reject),
        ("statement", "ggml_typea_type=a->type"),
        ("if", "ggml_is_quantized(a_type)&&(ggml_is_permuted(a)||!ggml_is_contiguous(a))", reject),
        ("if", "a_type==GGML_TYPE_Q4_0", (("if", "a->ne[2]!=b->ne[2]||a->ne[2]>2", reject),)),
        ("if", "a_type==GGML_TYPE_Q4_1&&b->ne[1]==1", reject),
        ("if", "a_type==GGML_TYPE_IQ4_NL||a_type==GGML_TYPE_IQ4_XS||a_type==GGML_TYPE_IQ3_XXS||a_type==GGML_TYPE_IQ3_S||a_type==GGML_TYPE_IQ2_XXS||a_type==GGML_TYPE_IQ2_XS||a_type==GGML_TYPE_IQ2_S||a_type==GGML_TYPE_IQ1_S||a_type==GGML_TYPE_IQ1_M", (("if", "b->ne[1]==1&&ggml_nrows(b)>1", reject),)),
        ("statement", "ggml_typesrc0_type=op->src[0]->type"),
        ("if", "src0_type==GGML_TYPE_BF16", reject),
        ("if", "ggml_is_permuted(a)&&!ggml_is_contiguous(a)&&a->ne[2]>1&&a->ne[3]>1&&src0_type==GGML_TYPE_F16", reject),
        ("if", "!ggml_is_permuted(a)&&ggml_is_permuted(b)&&b->ne[2]>1&&b->ne[3]>1&&a->ne[0]>128&&a->ne[2]==1&&src0_type==GGML_TYPE_F16", reject),
        ("if", "!ggml_is_transposed(a)&&!ggml_is_transposed(b)&&a_type==GGML_TYPE_F16&&b->type==GGML_TYPE_F32&&b->ne[1]==1&&b->ne[3]>1&&b->ne[3]==a->ne[3]&&b->ne[2]>a->ne[2]&&a->nb[1]>ggml_row_size(a_type,a->ne[0])&&b->nb[1]>ggml_row_size(b->type,b->ne[0])&&(((a->nb[1]/ggml_type_size(a_type))&1)!=0||((b->nb[1]/ggml_type_size(b->type))&1)!=0)", reject),
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
        function_body_open = function.index("{")
        early, early_close, early_body = braced_body(function, EARLY_GUARD)
        router_flag = function.index(ROUTER_FLAG, early_close)
        planner, planner_close, planner_body = braced_body(function, PLANNER_GUARD)
        switch = function.index(OP_SWITCH)
        allowed_types, default_type_support = mul_mat_type_policy(text)
        _, _, type_helper_body = braced_body(text, TYPE_HELPER_START)

        switch_body = function[switch:]
        mul_mat_start = switch_body.index("case GGML_OP_MUL_MAT:")
        mul_mat_end = switch_body.index("case GGML_OP_OUT_PROD:", mul_mat_start)
        mul_mat_case = switch_body[mul_mat_start:mul_mat_end]
        case_body_open = mul_mat_case.index("{")
        type_guard = "if (!ggml_sycl_mul_mat_type_supported(a_type)) {"
        type_guard_start, _, type_guard_body = braced_body(mul_mat_case, type_guard)
        planner_decisions = decision_clauses(planner_body)
        planner_control_decisions = tuple(decision for decision in planner_decisions if decision[0] != "statement")
        pre_guard_decisions = decision_clauses(mul_mat_case[case_body_open + 1 : type_guard_start])
    except (IndexError, ValueError):
        return False

    router_residency_exception = executable_body(function[early_close + 1 : planner])
    later_indexed_case = re.search(r"\bcase\s+GGML_OP_MUL_MAT_ID\s*:", switch_body)
    return (
        function_body_open < early < early_close < router_flag < planner < planner_close < switch
        and executable_body(function[function_body_open + 1 : early]) == EXPECTED_PRE_INDEXED_GUARD_PREFIX
        and executable_body(early_body) == "constggml_typeindexed_a_type=op->src[0]->type;if(indexed_a_type!=GGML_TYPE_Q1_0&&indexed_a_type!=GGML_TYPE_NVFP4&&!ggml_sycl_mul_mat_type_supported(indexed_a_type)){returnfalse;}returntrue;"
        and "GGML_OP_ADD_ID" in function[early : early_close + 1]
        and "GGML_OP_MUL_MAT_ID" in function[early : early_close + 1]
        and router_residency_exception ==
            "constboolis_multi_gpu_router_logits="
            "g_moe_multi_gpu_active.load(std::memory_order_acquire)&&"
            "ggml_sycl_op_is_moe_router_logits_matmul(op);"
        and planner_control_decisions == (("return", "false"),)
        and executable_body(function[planner_close + 1 : switch]) == ""
        and later_indexed_case is None
        and len(re.findall(r"\bcase\s+GGML_OP_MUL_MAT\s*:", switch_body)) == 1
        and "op->op == GGML_OP_MUL_MAT" not in mul_mat_case
        and allowed_types == MUL_MAT_TYPES
        and not default_type_support
        and helper_labels_flow_to_shared_true(type_helper_body)
        and mul_mat_case.count(type_guard) == 1
        and Counter(pre_guard_decisions) == Counter(expected_pre_guard_decisions())
        and executable_body(type_guard_body) == "returnfalse;"
        and executable_body(mul_mat_case[type_guard_start:]) ==
            "if(!ggml_sycl_mul_mat_type_supported(a_type)){returnfalse;}returntrue;}"
    )


def modeled_router_mul_mat_support(text: str, type_name: str, planned_on_host: bool) -> bool:
    """Host-only model of the validated router path after the source contract is locked."""
    if not contract(text):
        return False
    # Router logits bypass only this planner-residency rejection, then enter MUL_MAT.
    is_multi_gpu_router_logits = True
    if planned_on_host and not is_multi_gpu_router_logits:
        return False
    return mul_mat_type_supported(text, type_name)


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


def test_router_logits_bypass_only_planner_then_obey_dense_type_policy() -> None:
    assert not modeled_router_mul_mat_support(SOURCE, "FUTURE_UNKNOWN_TYPE", planned_on_host=True)
    assert not modeled_router_mul_mat_support(SOURCE, "NVFP4", planned_on_host=True)
    for supported in ("F32", "Q8_0", "MXFP4"):
        assert modeled_router_mul_mat_support(SOURCE, supported, planned_on_host=True)
        assert modeled_router_mul_mat_support(SOURCE, supported, planned_on_host=False)


def test_no_early_success_can_bypass_indexed_or_switch_validation() -> None:
    before_indexed_guard = replace_in_supports_function(
        SOURCE,
        EARLY_GUARD,
        "return true;\n    " + EARLY_GUARD,
    )
    assert not contract(before_indexed_guard)

    function = supports_function(SOURCE)
    planner, planner_close, _ = braced_body(function, PLANNER_GUARD)
    planner_guard = function[planner : planner_close + 1]
    after_planner_guard = replace_in_supports_function(
        SOURCE,
        planner_guard,
        planner_guard + "\n    return true;",
    )
    assert not contract(after_planner_guard)


def test_planner_guard_rejects_non_boolean_spelled_early_successes() -> None:
    function = supports_function(SOURCE)
    _, _, planner_body = braced_body(function, PLANNER_GUARD)
    assert planner_body.count("return false;") == 1

    for early_success in ("return 1;", "return static_cast<bool>(1);"):
        mutated_body = planner_body.replace(
            "return false;",
            early_success + "\n        return false;",
            1,
        )
        mutated = replace_in_supports_function(SOURCE, planner_body, mutated_body)
        assert not contract(mutated)


def test_router_logits_control_flow_mutations_are_rejected() -> None:
    declaration = (
        "    const bool is_multi_gpu_router_logits =\n"
        "        g_moe_multi_gpu_active.load(std::memory_order_acquire) && "
        "ggml_sycl_op_is_moe_router_logits_matmul(op);\n\n"
    )
    old_early_success = (
        "    if (g_moe_multi_gpu_active.load(std::memory_order_acquire) && "
        "ggml_sycl_op_is_moe_router_logits_matmul(op)) {\n"
        "        return true;\n"
        "    }\n\n"
    )
    assert not contract(replace_in_supports_function(SOURCE, declaration, old_early_success))

    no_exception = replace_in_supports_function(
        SOURCE,
        PLANNER_GUARD,
        "if (ggml_sycl_op_is_planned_on_host(op, device)) {",
    )
    assert not contract(no_exception)

    all_ops_bypass_planner = replace_in_supports_function(
        SOURCE,
        "const bool is_multi_gpu_router_logits =\n        g_moe_multi_gpu_active.load(std::memory_order_acquire) && "
        "ggml_sycl_op_is_moe_router_logits_matmul(op);",
        "const bool is_multi_gpu_router_logits = true;",
    )
    assert not contract(all_ops_bypass_planner)


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


def test_decision_parser_rejects_else_and_unknown_statements() -> None:
    first_rejection = (
        "                if (a->ne[3] != b->ne[3]) {\n"
        "                    return false;\n"
        "                }"
    )
    else_success = first_rejection + " else { return true; }"
    assert not contract(replace_in_supports_function(SOURCE, first_rejection, else_success))

    unknown_statement = first_rejection + "\n                audit_router_shape();"
    assert not contract(replace_in_supports_function(SOURCE, first_rejection, unknown_statement))


# The tail of the early guard: q1_0/nvfp4 flow on to the runtime route
# oracle (the sanctioned c-wps7 fail-closed class); every other type outside
# the MUL_MAT allowlist is refused here before it can compute wrong answers.
EARLY_RETURN = "return true;"
EARLY_TYPE_GUARD = (
    "if (indexed_a_type != GGML_TYPE_Q1_0 && indexed_a_type != GGML_TYPE_NVFP4 &&\n"
    "            !ggml_sycl_mul_mat_type_supported(indexed_a_type)) {"
)


def test_removing_only_early_return_is_rejected() -> None:
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert early_body.count(EARLY_RETURN) == 1
    mutated_body = early_body.replace(EARLY_RETURN, "", 1)
    assert not contract(replace_in_supports_function(SOURCE, early_body, mutated_body))


def test_reopening_early_guard_to_unconditional_admission_is_rejected() -> None:
    # The pre-b10630 form admitted every expert type into the MoE executor;
    # q2_0 MMID then computed ERR ~90 wrong answers. Stripping the type guard
    # back to a bare `return true;` body must fail the contract.
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert EARLY_TYPE_GUARD in early_body
    mutated_body = "\n        return true;\n    "
    assert not contract(replace_in_supports_function(SOURCE, early_body, mutated_body))


def test_widening_type_guard_exemptions_is_rejected() -> None:
    # Adding another exempt type (here q2_0) reopens the wrong-answer path.
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert EARLY_TYPE_GUARD in early_body
    widened = early_body.replace(
        "indexed_a_type != GGML_TYPE_NVFP4",
        "indexed_a_type != GGML_TYPE_NVFP4 && indexed_a_type != GGML_TYPE_Q2_0",
        1,
    )
    assert not contract(replace_in_supports_function(SOURCE, early_body, widened))


def test_moving_return_immediately_outside_guard_is_rejected() -> None:
    function = supports_function(SOURCE)
    early, early_close, early_body = braced_body(function, EARLY_GUARD)
    guard = function[early : early_close + 1]
    assert early_body.count(EARLY_RETURN) == 1
    moved = guard.replace(EARLY_RETURN, "", 1) + "\n    " + EARLY_RETURN
    assert not contract(replace_in_supports_function(SOURCE, guard, moved))


def test_conditioning_return_on_add_id_is_rejected() -> None:
    function = supports_function(SOURCE)
    _, _, early_body = braced_body(function, EARLY_GUARD)
    assert early_body.count(EARLY_RETURN) == 1
    conditional = early_body.replace(
        EARLY_RETURN,
        "if (op->op == GGML_OP_ADD_ID) { " + EARLY_RETURN + " }",
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
