#!/usr/bin/env python3
"""Host-only regression checks for SYCL supports_op/kernel contracts."""

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
CONCAT_SOURCE = ROOT / "ggml/src/ggml-sycl/concat.cpp"
BACKEND_OPS_SOURCE = ROOT / "tests/test-backend-ops.cpp"


def _supports_source(text: Optional[str] = None) -> str:
    source = text if text is not None else SUPPORT_SOURCE.read_text(encoding="utf-8")
    return source[source.index("static bool ggml_backend_sycl_device_supports_op") :]


def _case(source: str, op: str, next_op: str) -> str:
    start = source.index(f"case GGML_OP_{op}:")
    end = source.index(f"case GGML_OP_{next_op}:", start)
    return source[start:end]


def _normalized_return(case: str) -> str:
    match = re.search(r"\breturn\s+(.+?);", case, re.DOTALL)
    assert match is not None, "operation case has no return expression"
    return re.sub(r"\s+", "", match.group(1))


def _assert_expression(case: str, expected: str) -> None:
    assert _normalized_return(case) == re.sub(r"\s+", "", expected)


def _concat_case_sets(text: str) -> Tuple[Set[str], Set[str]]:
    helper_start = text.index("bool ggml_sycl_concat_type_supported")
    dispatch_start = text.index("void ggml_sycl_op_concat", helper_start)
    helper = text[helper_start:dispatch_start]
    dispatch = text[dispatch_start:]
    case_pattern = r"\bcase\s+GGML_TYPE_([A-Z0-9_]+)\s*:"
    return set(re.findall(case_pattern, helper)), set(re.findall(case_pattern, dispatch))


def _evaluate_rope_predicate(expression: str, out_of_place: bool, contiguous: bool) -> bool:
    python_expression = expression
    python_expression = python_expression.replace("op->view_src!=nullptr", str(not out_of_place))
    python_expression = python_expression.replace("op->view_src==nullptr", str(out_of_place))
    python_expression = python_expression.replace("ggml_is_contiguous(op)", str(contiguous))
    python_expression = python_expression.replace("||", " or ").replace("&&", " and ")
    assert re.fullmatch(r"[() TrueFalsenotandor]+", python_expression), python_expression
    return bool(eval(python_expression, {"__builtins__": {}}, {}))


def _assert_rope_contract(full_source: str) -> None:
    source = _supports_source(full_source)
    rope_expression = _normalized_return(_case(source, "ROPE", "IM2COL"))
    for out_of_place in (False, True):
        for contiguous in (False, True):
            actual = _evaluate_rope_predicate(rope_expression, out_of_place, contiguous)
            expected = out_of_place or contiguous
            assert actual == expected, (out_of_place, contiguous, rope_expression)
    assert rope_expression == "op->view_src==nullptr||ggml_is_contiguous(op)"
    _assert_expression(_case(source, "IM2COL", "UPSCALE"), "true")


def _assert_rope_inventory_contract(inventory: str) -> None:
    start = inventory.index("// single inplace test per type/mode/ff")
    end = inventory.index("for (int v :", start)
    block = inventory[start:end]

    def loop_values(declaration: str) -> List[str]:
        match = re.search(rf"for \({declaration} : \{{([^}}]+)\}}\)", block)
        assert match is not None
        return [value.strip() for value in match.group(1).split(",")]

    types = loop_values("ggml_type type")
    modes = loop_values("int mode")
    factors = loop_values("bool ff")
    discovered_rows = re.findall(r"test_cases\.emplace_back\(new test_rope\((.*?)\)\);", block, re.DOTALL)
    parsed_views = []
    for row in discovered_rows:
        match = re.search(r"ff,\s*([-+]?\d+)\s*,\s*true\s*,\s*true\s*$", row)
        assert match is not None, f"unparsed inplace ROPE inventory row: {row!r}"
        parsed_views.append(int(match.group(1)))
    assert len(parsed_views) == len(discovered_rows)

    assert types == ["GGML_TYPE_F32", "GGML_TYPE_F16"]
    assert modes == [
        "GGML_ROPE_TYPE_NORMAL",
        "GGML_ROPE_TYPE_NEOX",
        "GGML_ROPE_TYPE_MROPE",
        "GGML_ROPE_TYPE_IMROPE",
        "GGML_ROPE_TYPE_VISION",
    ]
    assert factors == ["false", "true"]
    assert parsed_views == [0, 1, 1]

    multiplicity = len(types) * len(modes) * len(factors)
    decisions = [view == 0 for view in parsed_views for _ in range(multiplicity)]
    assert decisions.count(False) == 40
    assert decisions.count(True) == 20


def _contract_rejects(oracle, source: str) -> bool:
    try:
        oracle(source)
    except AssertionError:
        return True
    return False


_ODD_STRIDE_COMMENT = "// oneMKL/oneDNN currently miscomputes the padded, odd-row-stride F16 x F32 batched path"
_ODD_STRIDE_NEXT_GUARD = "if (!ggml_sycl_mul_mat_type_supported(a_type))"
_ODD_STRIDE_EXPRESSION = """
    !ggml_is_transposed(a) && !ggml_is_transposed(b) && a_type == GGML_TYPE_F16 &&
    b->type == GGML_TYPE_F32 && b->ne[1] == 1 && b->ne[3] > 1 && b->ne[3] == a->ne[3] &&
    b->ne[2] > a->ne[2] && a->nb[1] > ggml_row_size(a_type, a->ne[0]) &&
    b->nb[1] > ggml_row_size(b->type, b->ne[0]) &&
    (((a->nb[1] / ggml_type_size(a_type)) & 1) != 0 ||
     ((b->nb[1] / ggml_type_size(b->type)) & 1) != 0)
"""


def _odd_stride_guard(full_source: str) -> str:
    mul_mat = _case(_supports_source(full_source), "MUL_MAT", "OUT_PROD")
    comment = mul_mat.index(_ODD_STRIDE_COMMENT)
    guard = mul_mat.index("if (", comment)
    guard_end = mul_mat.index("return false;", guard)
    next_guard = mul_mat.index(_ODD_STRIDE_NEXT_GUARD, guard_end)
    assert guard < guard_end < next_guard, "odd-stride rejection must precede the final dense type guard"
    return mul_mat[guard:guard_end]


def _assert_odd_stride_source_contract(full_source: str) -> None:
    guard = _odd_stride_guard(full_source)
    match = re.fullmatch(r"if\s*\((.*)\)\s*\{\s*", guard, re.DOTALL)
    assert match is not None, "odd-stride rejection is not a single fail-closed if"
    assert re.sub(r"\s+", "", match.group(1)) == re.sub(r"\s+", "", _ODD_STRIDE_EXPRESSION)


@dataclass(frozen=True)
class _MulMatShape:
    op: str = "MUL_MAT"
    a_type: str = "F16"
    b_type: str = "F32"
    a_transposed: bool = False
    b_transposed: bool = False
    m: int = 128
    n: int = 1
    k: int = 1057
    k_v: int = 2113
    a_ne2: int = 1
    b_ne2: int = 4
    a_ne3: int = 3
    b_ne3: int = 3
    a_nb0: int = 2
    b_nb0: int = 4
    a_nb1: int = 2 * 2113
    b_nb1: int = 4 * 2113
    bs0: int = 1


def _type_size(type_name: str) -> int:
    return {"F16": 2, "F32": 4}[type_name]


def _contiguous_rows_helper(shape: _MulMatShape, source: str) -> bool:
    # This mirrors ggml_is_contiguous_rows: padded nb[1] is deliberately irrelevant.
    nb0 = shape.a_nb0 if source == "a" else shape.b_nb0
    type_name = shape.a_type if source == "a" else shape.b_type
    return nb0 == _type_size(type_name)


_MODEL_CLAUSES: Dict[str, Callable[[_MulMatShape], bool]] = {
    "regular_mul_mat": lambda s: s.op == "MUL_MAT",
    "src0_f16": lambda s: s.a_type == "F16",
    "src1_f32": lambda s: s.b_type == "F32",
    "src0_not_transposed": lambda s: not s.a_transposed,
    "src1_not_transposed": lambda s: not s.b_transposed,
    "src1_single_row": lambda s: s.n == 1,
    "multi_dim3": lambda s: s.b_ne3 > 1,
    "matching_dim3": lambda s: s.b_ne3 == s.a_ne3,
    "dim2_broadcast": lambda s: s.b_ne2 > s.a_ne2,
    "src0_padded_rows": lambda s: s.a_nb1 > _type_size(s.a_type) * s.k,
    "src1_padded_rows": lambda s: s.b_nb1 > _type_size(s.b_type) * s.k,
    "odd_element_row_stride": lambda s: (
        ((s.a_nb1 // _type_size(s.a_type)) & 1) != 0 or ((s.b_nb1 // _type_size(s.b_type)) & 1) != 0
    ),
}


def _decline_odd_stride(shape: _MulMatShape, omitted: Optional[Set[str]] = None) -> bool:
    omitted = omitted or set()
    return all(predicate(shape) for name, predicate in _MODEL_CLAUSES.items() if name not in omitted)


def _failure_shapes() -> List[_MulMatShape]:
    return [_MulMatShape(m=m, bs0=bs0, a_ne2=bs0, b_ne2=bs0 * 4) for m in (128, 129) for bs0 in (1, 2, 4, 8)]


def _adjacent_controls() -> List[Tuple[str, _MulMatShape]]:
    base = _MulMatShape()
    return [
        ("even element row stride", replace(base, k_v=2114, a_nb1=2 * 2114, b_nb1=4 * 2114)),
        ("dim3=1", replace(base, a_ne3=1, b_ne3=1)),
        ("no dim2 broadcast", replace(base, b_ne2=base.a_ne2)),
        ("F32 equivalent", replace(base, a_type="F32", a_nb0=4, a_nb1=4 * base.k_v)),
        ("MUL_MAT_ID", replace(base, op="MUL_MAT_ID")),
        ("src1 ne1>1", replace(base, n=2)),
        ("unpadded rows", replace(base, a_nb1=2 * base.k, b_nb1=4 * base.k)),
        ("transposed src0", replace(base, a_transposed=True)),
    ]


def _clause_witnesses() -> Dict[str, _MulMatShape]:
    base = _MulMatShape()
    return {
        "regular_mul_mat": replace(base, op="MUL_MAT_ID"),
        "src0_f16": replace(base, a_type="F32", a_nb0=4, a_nb1=4 * 2113),
        "src1_f32": replace(base, b_type="F16", b_nb0=2, b_nb1=2 * 2113),
        "src0_not_transposed": replace(base, a_transposed=True),
        "src1_not_transposed": replace(base, b_transposed=True),
        "src1_single_row": replace(base, n=2),
        "multi_dim3": replace(base, a_ne3=1, b_ne3=1),
        "matching_dim3": replace(base, b_ne3=4),
        "dim2_broadcast": replace(base, b_ne2=base.a_ne2),
        "src0_padded_rows": replace(base, a_nb1=2 * base.k),
        "src1_padded_rows": replace(base, b_nb1=4 * base.k),
        "odd_element_row_stride": replace(base, a_nb1=2 * 2114, b_nb1=4 * 2114),
    }


def test_f16_odd_stride_mul_mat_source_contract_and_decision_census() -> None:
    _assert_odd_stride_source_contract(SUPPORT_SOURCE.read_text(encoding="utf-8"))
    failures = _failure_shapes()
    controls = _adjacent_controls()
    assert len(failures) == 8 and all(_decline_odd_stride(shape) for shape in failures)
    assert all(shape.a_nb0 == _type_size(shape.a_type) for shape in failures)
    assert all(shape.b_nb0 == _type_size(shape.b_type) for shape in failures)
    assert all(shape.a_nb1 == _type_size(shape.a_type) * shape.k_v for shape in failures)
    assert all(shape.b_nb1 == _type_size(shape.b_type) * shape.k_v for shape in failures)
    assert all(shape.a_nb1 > _type_size(shape.a_type) * shape.k for shape in failures)
    assert all(shape.b_nb1 > _type_size(shape.b_type) * shape.k for shape in failures)
    assert all(_contiguous_rows_helper(shape, "a") for shape in failures)
    assert all(_contiguous_rows_helper(shape, "b") for shape in failures)
    assert {
        (
            shape.a_type,
            shape.b_type,
            shape.n,
            shape.k,
            shape.k_v,
            shape.a_ne3,
            shape.b_ne2 // shape.a_ne2,
            shape.m,
            shape.bs0,
        )
        for shape in failures
    } == {
        ("F16", "F32", 1, 1057, 2113, 3, 4, m, bs0)
        for m in (128, 129)
        for bs0 in (1, 2, 4, 8)
    }
    assert len(controls) == 8 and all(not _decline_odd_stride(shape) for _, shape in controls)


def test_f16_odd_stride_mul_mat_model_rejects_removed_conjunctions() -> None:
    witnesses = _clause_witnesses()
    assert witnesses.keys() == _MODEL_CLAUSES.keys()
    for clause, witness in witnesses.items():
        assert not _decline_odd_stride(witness), clause
        assert _decline_odd_stride(witness, {clause}), f"removing {clause} was not observable"


def test_pool_2d_does_not_inherit_acc_restrictions() -> None:
    _assert_expression(_case(_supports_source(), "POOL_2D", "ACC"), "true")


def test_rope_predicate_and_inplace_inventory_counts() -> None:
    _assert_rope_contract(SUPPORT_SOURCE.read_text(encoding="utf-8"))
    _assert_rope_inventory_contract(BACKEND_OPS_SOURCE.read_text(encoding="utf-8"))


def test_norm_family_predicates_match_complete_kernel_contracts() -> None:
    source = _supports_source()
    expected = {
        ("NORM", "RMS_NORM"): """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            (op->src[0]->ne[0] % WARP_SIZE) == 0 && ggml_sycl_norm_rows_supported(op->src[0]) &&
            ggml_is_contiguous(op)
        """,
        ("RMS_NORM", "L2_NORM"): """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            (op->src[0]->ne[0] % WARP_SIZE) == 0 && ggml_sycl_norm_rows_supported(op->src[0]) &&
            ggml_is_contiguous(op)
        """,
        ("L2_NORM", "GROUP_NORM"): """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
            (op->src[0]->ne[0] % WARP_SIZE) == 0
        """,
        ("GROUP_NORM", "RMS_NORM_BACK"): """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[3] == 1 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op)
        """,
        ("RMS_NORM_BACK", "SCALE"): """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            op->src[1]->type == GGML_TYPE_F32 && (op->src[0]->ne[0] % WARP_SIZE) == 0 &&
            ggml_sycl_norm_rows_supported(op->src[0]) && ggml_sycl_norm_rows_supported(op->src[1]) &&
            ggml_sycl_norm_rows_supported(op)
        """,
    }
    for (op, next_op), expression in expected.items():
        _assert_expression(_case(source, op, next_op), expression)

    full_source = SUPPORT_SOURCE.read_text(encoding="utf-8")
    helper_start = full_source.index("static bool ggml_sycl_norm_rows_supported")
    helper_end = full_source.index("static bool ggml_backend_sycl_device_supports_op", helper_start)
    helper = full_source[helper_start:helper_end]
    assert "if (!t)" in helper and "return false;" in helper
    _assert_expression(
        helper[helper.index("const size_t ts") :],
        "t->nb[0] == ts && t->nb[1] % ts == 0 && t->nb[2] % ts == 0 && t->nb[3] % ts == 0",
    )


def test_acc_and_concat_predicates_match_complete_contracts() -> None:
    source = _supports_source()
    _assert_expression(
        _case(source, "ACC", "PAD"),
        """
            op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            op->src[1]->type == GGML_TYPE_F32 && op->ne[3] == 1 && op->src[1]->ne[3] == 1 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op)
        """,
    )
    _assert_expression(
        _case(source, "CONCAT", "DUP"),
        """
            ggml_sycl_concat_type_supported(op->type) && op->src[0]->type == op->type &&
            op->src[1]->type == op->type
        """,
    )


def test_concat_support_helper_and_dispatch_case_sets_are_identical() -> None:
    text = CONCAT_SOURCE.read_text(encoding="utf-8")
    helper_cases, dispatch_cases = _concat_case_sets(text)
    expected = {"F32", "F16", "BF16", "I32", "I16", "I64", "I8"}
    assert helper_cases == expected
    assert dispatch_cases == expected
    assert helper_cases == dispatch_cases


def test_review_mutations_are_detected() -> None:
    full_support = SUPPORT_SOURCE.read_text(encoding="utf-8")
    odd_stride_guard = _odd_stride_guard(full_support)
    odd_stride_start = full_support.index(odd_stride_guard)
    broadening_mutations = (
        ("!ggml_is_transposed(a)", "true"),
        ("!ggml_is_transposed(b)", "true"),
        ("a_type == GGML_TYPE_F16", "a_type != GGML_TYPE_BF16"),
        ("b->type == GGML_TYPE_F32", "b->type != GGML_TYPE_F16"),
        ("b->ne[1] == 1", "b->ne[1] >= 1"),
        ("b->ne[3] > 1", "b->ne[3] >= 1"),
        ("b->ne[3] == a->ne[3]", "b->ne[3] >= a->ne[3]"),
        ("b->ne[2] > a->ne[2]", "b->ne[2] >= a->ne[2]"),
        ("a->nb[1] > ggml_row_size(a_type, a->ne[0])", "true"),
        ("b->nb[1] > ggml_row_size(b->type, b->ne[0])", "true"),
        ("a->nb[1] > ggml_row_size(a_type, a->ne[0])", "!ggml_is_contiguous_rows(a)"),
        ("b->nb[1] > ggml_row_size(b->type, b->ne[0])", "!ggml_is_contiguous_rows(b)"),
        ("(((a->nb[1] / ggml_type_size(a_type)) & 1) != 0 ||", "(true ||"),
    )
    for old, new in broadening_mutations:
        assert odd_stride_guard.count(old) == 1, old
        mutated_guard = odd_stride_guard.replace(old, new, 1)
        mutated_source = (
            full_support[:odd_stride_start]
            + mutated_guard
            + full_support[odd_stride_start + len(odd_stride_guard) :]
        )
        assert _contract_rejects(_assert_odd_stride_source_contract, mutated_source), old

    support_start = full_support.index("static bool ggml_backend_sycl_device_supports_op")
    rope_start = full_support.index("case GGML_OP_ROPE:", support_start)
    rope_end = full_support.index("case GGML_OP_IM2COL:", rope_start)
    rope_case = full_support[rope_start:rope_end]
    for old, new in ((" == nullptr", " != nullptr"), (" || ", " && ")):
        assert old in rope_case
        mutated_case = rope_case.replace(old, new, 1)
        mutated_source = full_support[:rope_start] + mutated_case + full_support[rope_end:]
        assert _contract_rejects(_assert_rope_contract, mutated_source)

    inventory = BACKEND_OPS_SOURCE.read_text(encoding="utf-8")
    inventory_start = inventory.index("// single inplace test per type/mode/ff")
    row_start = inventory.index("test_cases.emplace_back(new test_rope", inventory_start)
    row_end = inventory.index("\n", row_start)
    row = inventory[row_start:row_end]
    injected_row = row.replace("ff, 0, true, true", "ff, 2, true, true")
    assert injected_row != row
    mutated_inventory = inventory[:row_end] + "\n" + injected_row + inventory[row_end:]
    assert _contract_rejects(_assert_rope_inventory_contract, mutated_inventory)

    support = _supports_source(full_support)
    group = _case(support, "GROUP_NORM", "RMS_NORM_BACK")
    expected_group = _normalized_return(group)
    mutated_group = group.replace(" && ", " || ", 1)
    assert _normalized_return(mutated_group) != expected_group

    concat = CONCAT_SOURCE.read_text(encoding="utf-8")
    dispatch_start = concat.index("void ggml_sycl_op_concat")
    mutated_concat = concat[:dispatch_start] + concat[dispatch_start:].replace("case GGML_TYPE_I8:", "", 1)
    helper_cases, dispatch_cases = _concat_case_sets(mutated_concat)
    assert helper_cases != dispatch_cases
