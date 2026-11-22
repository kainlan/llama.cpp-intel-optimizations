#!/usr/bin/env python3
"""Host-only regression checks for SYCL supports_op/kernel contracts."""

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
CONCAT_SOURCE = ROOT / "ggml/src/ggml-sycl/concat.cpp"
BINBCAST_SOURCE = ROOT / "ggml/src/ggml-sycl/binbcast.cpp"
BACKEND_OPS_SOURCE = ROOT / "tests/test-backend-ops.cpp"


# The decisions live in ggml_sycl_device_supports_op_impl; ggml_backend_sycl_device_supports_op is a one-line
# wrapper over it. Anchor on the impl DEFINITION (the forward declaration before it ends in `;`, so it does not
# match).
SUPPORTS_IMPL_DEFINITION = re.compile(r"static\s+bool\s+ggml_sycl_device_supports_op_impl\s*\([^)]*\)\s*\{")


def _supports_impl_start(source: str) -> int:
    found = list(SUPPORTS_IMPL_DEFINITION.finditer(source))
    assert len(found) == 1, f"supports_op impl definition occurs {len(found)} times, expected exactly 1"
    return found[0].start()


def _supports_source(text: Optional[str] = None) -> str:
    source = text if text is not None else SUPPORT_SOURCE.read_text(encoding="utf-8")
    return source[_supports_impl_start(source) :]


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


# b10630 ggml_rope_set_offset: the kernels predate n_offs (op_params[15]), so the ROPE case opens with
# a fail-closed refusal. It is an early `return false;`, so reading the case's FIRST return (as this gate
# used to) yields the literal `false` rather than the view_src/contiguity predicate that follows it
# (llama.cpp-6e08: ROPE support had not become false; the extractor latched onto the wrong statement).
ROPE_N_OFFS_GUARD = re.compile(
    r"if\s*\(\s*\(\(const\s+int32_t\s*\*\)\s*op->op_params\)\[15\]\s*!=\s*0\s*\)\s*\{\s*return\s+false\s*;\s*\}"
)


def _rope_case_without_n_offs_guard(case: str) -> str:
    code = re.sub(r"//[^\n]*", "", case)
    assert ROPE_N_OFFS_GUARD.search(code), "ROPE case lost its n_offs fail-closed guard"
    # The remainder is a single `return <predicate>;`, which is what the rest of this contract evaluates.
    return ROPE_N_OFFS_GUARD.sub("", code, count=1)


def _assert_rope_contract(full_source: str) -> None:
    source = _supports_source(full_source)
    rope_expression = _normalized_return(_rope_case_without_n_offs_guard(_case(source, "ROPE", "IM2COL")))
    for out_of_place in (False, True):
        for contiguous in (False, True):
            actual = _evaluate_rope_predicate(rope_expression, out_of_place, contiguous)
            expected = out_of_place or contiguous
            assert actual == expected, (out_of_place, contiguous, rope_expression)
    assert rope_expression == "op->view_src==nullptr||ggml_is_contiguous(op)"
    _assert_expression(_case(source, "IM2COL", "UPSCALE"), "true")
    # The kernels are instantiated for f32/f16 sources only; the inventory model (_sycl_admits_forward_rope)
    # relies on this type check, which runs before the impl.
    _assert_expression(
        _case(_core_semantics_source(full_source), "ROPE", "MUL_MAT"),
        "op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16",
    )


def _core_semantics_source(full_source: str) -> str:
    start = full_source.index("static bool ggml_sycl_op_core_semantics_supported(const ggml_tensor * op) {")
    return full_source[start : full_source.index("\n}\n", start)]


# test_rope's constructor parameters in order, and the defaults of the ones the SYCL decision reads.
_ROPE_CTOR_PARAMS = (
    "type", "ne_a", "n_dims", "mode", "n_ctx", "fs", "ef", "af", "ff", "v", "forward", "inplace", "n_offs",
    "n_ctx_orig", "freq_base", "beta_fast", "truncate",
)
_ROPE_CTOR_DEFAULTS = {"type": "GGML_TYPE_F32", "ff": False, "v": 0, "forward": True, "inplace": False, "n_offs": 0}
_ROPE_LOOP_HEADER = re.compile(r"for\s*\(\s*[\w:]+\s+(\w+)\s*:\s*\{([^}]*)\}\s*\)\s*\{")
_ROPE_ROW_OPEN = "test_cases.emplace_back(new test_rope("


def _matching_close(text: str, open_index: int) -> int:
    pairs = {"(": ")", "{": "}"}
    closer = pairs[text[open_index]]
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == text[open_index]:
            depth += 1
        elif text[index] == closer:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unbalanced {text[open_index]!r} at {open_index}")


def _split_top_level_args(args: str) -> List[str]:
    parts, depth, current = [], 0, ""
    for char in args:
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return parts


def _skip_blank_and_comments(text: str, index: int) -> int:
    while True:
        match = re.compile(r"\s+|//[^\n]*").match(text, index)
        if match is None or match.end() == index:
            return index
        index = match.end()


def _next_statement(text: str, index: int) -> Tuple[str, int]:
    """One statement of a loop body: a brace-balanced for-loop or a single test_rope row."""
    if text.startswith("for", index):
        header = _ROPE_LOOP_HEADER.match(text, index)
        assert header is not None, f"unparsed loop header: {text[index:index + 80]!r}"
        end = _matching_close(text, header.end() - 1) + 1
        return text[index:end], end
    if text.startswith(_ROPE_ROW_OPEN, index):
        close = _matching_close(text, index + len(_ROPE_ROW_OPEN) - 1)
        end = text.index(";", close) + 1
        assert text[close + 1 : end] == ");", f"unparsed row ending: {text[close:end]!r}"
        return text[index:end], end
    raise AssertionError(f"unparsed ROPE inventory statement: {text[index:index + 80]!r}")


def _rope_literal(token: str, env: Dict[str, object]) -> object:
    if token in env:
        return env[token]
    if token in ("true", "false"):
        return token == "true"
    if re.fullmatch(r"[-+]?\d+", token):
        return int(token)
    return token


def _expand_rope_statements(text: str, env: Dict[str, object]) -> List[Dict[str, object]]:
    """Every test_rope instance the statements in `text` construct, with the loop variables bound."""
    instances: List[Dict[str, object]] = []
    index = _skip_blank_and_comments(text, 0)
    while index < len(text):
        statement, index = _next_statement(text, index)
        if statement.startswith("for"):
            header = _ROPE_LOOP_HEADER.match(statement)
            body = statement[header.end() : -1]
            for value in _split_top_level_args(header.group(2)):
                instances += _expand_rope_statements(body, {**env, header.group(1): _rope_literal(value, env)})
        else:
            args = _split_top_level_args(statement[len(_ROPE_ROW_OPEN) : statement.rindex("));")])
            assert len(args) <= len(_ROPE_CTOR_PARAMS), statement
            instance = dict(_ROPE_CTOR_DEFAULTS)
            instance.update({name: _rope_literal(arg, env) for name, arg in zip(_ROPE_CTOR_PARAMS, args)})
            instances.append(instance)
        index = _skip_blank_and_comments(text, index)
    return instances


def _rope_inventory_blocks(inventory: str) -> List[str]:
    """The ROPE inventory from the single-inplace nest on: each top-level statement that constructs a test_rope,
    up to the first top-level statement that constructs none. Bounded by statement structure, so a block added
    in the middle is read, and a block that follows (CONCAT's loops) is not."""
    index = _skip_blank_and_comments(inventory, inventory.index("// single inplace test per type/mode/ff"))
    blocks = []
    while True:
        try:
            statement, end = _next_statement(inventory, index)
        except AssertionError:
            break
        if _ROPE_ROW_OPEN not in statement:
            break
        blocks.append(statement)
        index = _skip_blank_and_comments(inventory, end)
    return blocks


# What the SYCL backend answers for one forward ROPE node, as ggml_backend_sycl_device_supports_op computes it:
# ggml_sycl_op_core_semantics_supported (f32/f16 sources), then the impl's n_offs guard, then its in-place
# predicate. An in-place result is a view of `a`, contiguous exactly when `a` is (v == 0). A backward row builds
# ROPE_BACK, which this contract does not model.
def _sycl_admits_forward_rope(instance: Dict[str, object]) -> bool:
    if instance["type"] not in ("GGML_TYPE_F32", "GGML_TYPE_F16"):
        return False
    if instance["n_offs"] != 0:
        return False
    return (not instance["inplace"]) or instance["v"] == 0


def _assert_rope_inventory_contract(inventory: str) -> None:
    blocks = _rope_inventory_blocks(inventory)
    assert len(blocks) == 8, f"ROPE inventory has {len(blocks)} top-level statements, expected 8"

    nest = blocks[0]
    headers = _ROPE_LOOP_HEADER.findall(nest)
    assert [(name, [v.strip() for v in values.split(",")]) for name, values in headers] == [
        ("type", ["GGML_TYPE_F32", "GGML_TYPE_F16"]),
        ("mode", [
            "GGML_ROPE_TYPE_NORMAL",
            "GGML_ROPE_TYPE_NEOX",
            "GGML_ROPE_TYPE_MROPE",
            "GGML_ROPE_TYPE_IMROPE",
            "GGML_ROPE_TYPE_VISION",
        ]),
        ("ff", ["false", "true"]),
    ]
    nest_rows = _expand_rope_statements(nest, {})
    assert all(row["inplace"] and row["forward"] and row["n_offs"] == 0 for row in nest_rows)
    assert [row["v"] for row in nest_rows[:3]] == [0, 1, 1]
    nest_decisions = [_sycl_admits_forward_rope(row) for row in nest_rows]
    assert len(nest_decisions) == 60
    assert nest_decisions.count(False) == 40
    assert nest_decisions.count(True) == 20

    # Per top-level statement: (forward instances, admitted, backward instances). In order: the single-inplace
    # nest; the BF16 block (declined by type, its in-place v=0 rows too); the offset block over F32/F16/BF16
    # (declined by the n_offs guard); the truncate block (its in-place v=1 rows declined); four real-model rows.
    expected = [(60, 20, 0), (15, 0, 10), (27, 0, 24), (40, 20, 20), (1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0)]
    actual = []
    for block in blocks:
        rows = _expand_rope_statements(block, {})
        forward = [row for row in rows if row["forward"]]
        actual.append((len(forward), sum(_sycl_admits_forward_rope(row) for row in forward), len(rows) - len(forward)))
    assert actual == expected, actual
    forward_total = sum(f for f, _, _ in actual)
    admitted_total = sum(a for _, a, _ in actual)
    assert (forward_total, admitted_total, forward_total - admitted_total) == (146, 44, 102)


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
    helper_end = _supports_impl_start(full_source)
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


_BIN_BCAST_ARM = re.compile(
    r"src0->type\s*==\s*GGML_TYPE_(\w+)\s*&&\s*src1->type\s*==\s*GGML_TYPE_(\w+)\s*&&\s*"
    r"dst->type\s*==\s*GGML_TYPE_(\w+)"
)
_BIN_BCAST_TRIPLE = re.compile(r"\{\s*GGML_TYPE_(\w+)\s*,\s*GGML_TYPE_(\w+)\s*,\s*GGML_TYPE_(\w+)\s*\}")
_BF16_GUARD = "#ifdef GGML_SYCL_HAS_BF16"


def _guarded_matches(region: str, pattern: "re.Pattern[str]") -> List[Tuple[str, str, str, bool]]:
    """Every (src0, src1, dst) the pattern finds, and whether it sits under the bf16 guard. Any other
    preprocessor conditional in the region fails the parse: the guard state would no longer be known."""
    found: List[Tuple[str, str, str, bool]] = []
    guarded = False
    for line in region.splitlines():
        stripped = line.strip()
        if stripped == _BF16_GUARD:
            assert not guarded, "nested bf16 guard"
            guarded = True
            continue
        if stripped.startswith("#endif") and guarded:
            guarded = False
            continue
        assert not re.match(r"#\s*(if|ifdef|ifndef|else|elif|endif)\b", stripped), f"unmodelled conditional: {stripped}"
        found += [(*match.groups(), guarded) for match in pattern.finditer(line)]
    assert not guarded, "unterminated bf16 guard"
    return found


def _bin_bcast_chain_arms(text: str) -> List[Tuple[str, str, str, bool]]:
    start = text.index("inline void ggml_sycl_op_bin_bcast(")
    end = text.index("kernel_event_valid = true;", start)
    chain = text[start:end]
    arms = _guarded_matches(chain, _BIN_BCAST_ARM)
    # One kernel launch per arm, so an arm the parse misses (or a launch outside an arm) shows here.
    launches = len(re.findall(r"kernel_event\s*=\s*op\(\)\(", chain))
    assert launches == len(arms), "chain arms and kernel launches disagree"
    return arms


def _bin_bcast_predicate_triples(text: str) -> List[Tuple[str, str, str, bool]]:
    start = text.index("bool ggml_sycl_bin_bcast_types_supported(ggml_type src0, ggml_type src1, ggml_type dst) {")
    end = text.index("\n}\n", start)
    return _guarded_matches(text[start:end], _BIN_BCAST_TRIPLE)


def _assert_bin_bcast_type_contract(binbcast: str) -> None:
    arms = _bin_bcast_chain_arms(binbcast)
    triples = _bin_bcast_predicate_triples(binbcast)
    assert len(set(arms)) == len(arms) and len(set(triples)) == len(triples), "duplicate type triple"
    assert set(triples) == set(arms), (sorted(set(arms) - set(triples)), sorted(set(triples) - set(arms)))
    # The upstream bf16 arms are the chain's, under the guard, and nothing else is guarded.
    assert {arm for arm in arms if arm[3]} == {("BF16", "BF16", "BF16", True), ("BF16", "F32", "BF16", True)}


def _assert_bin_bcast_support_contract(full_source: str) -> None:
    source = _supports_source(full_source)
    binary = _case(source, "ADD", "CONCAT")
    labels = re.sub(r"//[^\n]*", "", binary[: binary.index("return")])
    assert re.sub(r"\s+", "", labels) == "caseGGML_OP_ADD:caseGGML_OP_SUB:caseGGML_OP_MUL:caseGGML_OP_DIV:"
    _assert_expression(
        binary,
        """
            ggml_sycl_bin_bcast_types_supported(op->src[0]->type, op->src[1]->type, op->type) &&
            ggml_sycl_binbcast_layout_supported(op->src[0], op->src[1], op)
        """,
    )
    # REPEAT hands (dst, src, dst) to the kernel, so the triple it asks about is (op, src0, op).
    _assert_expression(
        _case(source, "REPEAT", "ADD"),
        """
            ggml_sycl_bin_bcast_types_supported(op->type, op->src[0]->type, op->type) &&
            ggml_sycl_binbcast_layout_supported(op, op->src[0], op)
        """,
    )


def test_bin_bcast_admits_exactly_the_type_triples_its_chain_has() -> None:
    _assert_bin_bcast_type_contract(BINBCAST_SOURCE.read_text(encoding="utf-8"))
    _assert_bin_bcast_support_contract(SUPPORT_SOURCE.read_text(encoding="utf-8"))


def test_bin_bcast_mutations_are_detected() -> None:
    binbcast = BINBCAST_SOURCE.read_text(encoding="utf-8")
    # A chain arm added without a predicate triple.
    last_arm = binbcast.index("    } else {\n        fprintf(stderr, \"%s: unsupported types")
    added_arm = (
        "    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {\n"
        "        kernel_event = op()((const float *) src0_d, (const sycl::half *) src1_d, (float *) dst_d);\n"
    )
    assert _contract_rejects(_assert_bin_bcast_type_contract, binbcast[:last_arm] + added_arm + binbcast[last_arm:])
    # A predicate triple without a chain arm.
    triple = "        { GGML_TYPE_I16,  GGML_TYPE_I16,  GGML_TYPE_I16  },\n"
    assert binbcast.count(triple) == 1
    widened = binbcast.replace(triple, triple + "        { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32 },\n", 1)
    assert _contract_rejects(_assert_bin_bcast_type_contract, widened)
    # A bf16 triple moved out from under the guard.
    bf16_triple = "        { GGML_TYPE_BF16, GGML_TYPE_F32,  GGML_TYPE_BF16 },\n"
    assert binbcast.count(bf16_triple) == 1
    unguarded = binbcast.replace(bf16_triple, "", 1).replace(triple, triple + bf16_triple, 1)
    assert _contract_rejects(_assert_bin_bcast_type_contract, unguarded)

    full_support = SUPPORT_SOURCE.read_text(encoding="utf-8")
    support_start = _supports_impl_start(full_support)
    # The predicate call removed from the ADD/SUB/MUL/DIV case, and from REPEAT.
    for start_label, end_label, call in (
        ("case GGML_OP_ADD:", "case GGML_OP_CONCAT:",
         "ggml_sycl_bin_bcast_types_supported(op->src[0]->type, op->src[1]->type, op->type) &&"),
        ("case GGML_OP_REPEAT:", "case GGML_OP_ADD:",
         "ggml_sycl_bin_bcast_types_supported(op->type, op->src[0]->type, op->type) &&"),
    ):
        case_start = full_support.index(start_label, support_start)
        case_end = full_support.index(end_label, case_start)
        case = full_support[case_start:case_end]
        assert case.count(call) == 1, call
        mutated = full_support[:case_start] + case.replace(call, "", 1) + full_support[case_end:]
        assert _contract_rejects(_assert_bin_bcast_support_contract, mutated), start_label


ROUTE_SOURCE = ROOT / "ggml/src/ggml-sycl/nonfa-stage.hpp"
_SRC1_READER_SIGNATURE = re.compile(r"static\s+bool\s+ggml_sycl_mul_mat_src1_has_reader\s*\([^)]*\)\s*\{")
_SRC1_READER_GUARD = "if(b->type!=GGML_TYPE_F32&&!ggml_sycl_mul_mat_src1_has_reader(a,b,op,device)){returnfalse;}"
_UNIFIED_ARM_SIGNATURE = re.compile(r"static\s+bool\s+ggml_sycl_mul_mat_takes_unified_arm\s*\([^)]*\)\s*\{")
_UNIFIED_ARM_BODY = (
    "returnggml_sycl_unified_dispatch_enabled()&&ggml_sycl::is_unified_kernel_enabled_for_device(device)&&"
    "ggml_sycl::should_use_unified(src0->type)&&ggml_is_contiguous(src1)&&!ggml_is_transposed(src1)&&"
    "!ggml_is_permuted(src1);"
)
_SELECT_SIGNATURE = re.compile(r"MatmulDecision\s+UnifiedMatmulOrchestrator::select\s*\([^)]*\)\s*const\s*\{")
_SELECT_UNIFIED_ARM = "if(allow_unified&&ggml_sycl_mul_mat_takes_unified_arm(src0,src1,ctx_.device)){"
_BATCHED_SIGNATURES = (
    re.compile(r"\[\[nodiscard\]\]\s+static\s+bool\s+ggml_sycl_mul_mat_batched_sycl\s*\([^)]*\)\s*try\s*\{"),
    re.compile(r"static\s+bool\s+ggml_sycl_mul_mat_batched_f16_fallback\s*\([^)]*\)\s*(?:try\s*)?\{"),
)
_UNIFIED_DISPATCH_BLOCK = "if (decision.valid && decision.backend == ggml_sycl::MatmulBackend::UnifiedKernel) {"
_UNIFIED_FLOAT_READ = "static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(src1, ctx.device))"
_BATCHED_NC_CONVERT = "const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);"
_INDEXED_CONTIGUITY_GUARD = "if(op->op==GGML_OP_MUL_MAT_ID){if(!ggml_is_contiguous(op->src[0])){returnfalse;}"
_PROMPT_STAGING_SITE = "if (!pp_mxfp4_soa_f16_ready && !ensure_src1_contiguous_f32()) {"
_PROMPT_STAGING_ABORT = "Failed to allocate PP F32 activation staging"
# What each kernel family reads, from its own source. The p021 and nc kernels assert an F32 src1. The batched
# routes read F16 as it is and convert any other type through the strided f16 converter (their oneMath and
# scalar-fallback arms), so a type that converter lacks has no reader. Outside the f16 chain, the orchestrator's
# unified-kernel arm reads src1 as F32 with no type check, and every other arm ends in ggml_sycl_op_mul_mat, which
# reads another type as one 2-D plane only.
_BATCHED_ANSWER = "src1->type==GGML_TYPE_F16||ggml_get_to_fp16_nc_sycl(src1->type)!=nullptr"
_SRC1_READER_ANSWERS = {
    "NONE": "!ggml_sycl_mul_mat_takes_unified_arm(src0,src1,device)&&src1->ne[2]==1&&src1->ne[3]==1",
    "KQ_P021": "false",
    "KQ_BATCHED": _BATCHED_ANSWER,
    "VEC_NC": "false",
    "KQKV_BATCHED": _BATCHED_ANSWER,
    "KQKV_SCALAR": _BATCHED_ANSWER,
}


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", _strip_comments(text))


def _function_body(text: str, signature: "re.Pattern[str]") -> str:
    match = signature.search(text)
    assert match is not None, signature.pattern
    close = _matching_close(text, match.end() - 1)
    return text[match.end() : close]


def _route_enumerators(route_header: str) -> List[str]:
    match = re.search(r"enum\s+\w+\s*\{([^}]*GGML_SYCL_MUL_MAT_F16_ROUTE_NONE[^}]*)\}", route_header)
    assert match is not None, "f16 route enum"
    return re.findall(r"GGML_SYCL_MUL_MAT_F16_ROUTE_(\w+)\s*,", _strip_comments(match.group(1)))


def _src1_reader_answers(full_source: str) -> Dict[str, str]:
    body = _compact(_function_body(full_source, _SRC1_READER_SIGNATURE))
    prefix = (
        "if(src1->type==GGML_TYPE_F32){returntrue;}"
        "switch(ggml_sycl_mul_mat_f16_route_of(src0,src1,dst,ggml_sycl_mul_mat_route_env_of(src0,src1))){"
    )
    suffix = "}returnfalse;"
    assert body.startswith(prefix) and body.endswith(suffix), body
    switch = body[len(prefix) : -len(suffix)]
    groups = re.findall(r"((?:caseGGML_SYCL_MUL_MAT_F16_ROUTE_\w+:)+)return([^;]+);", switch)
    assert "".join(f"{labels}return{answer};" for labels, answer in groups) == switch, switch
    answers: Dict[str, str] = {}
    for labels, answer in groups:
        for route in re.findall(r"caseGGML_SYCL_MUL_MAT_F16_ROUTE_(\w+):", labels):
            assert route not in answers, route
            answers[route] = answer
    return answers


def _assert_mul_mat_src1_contract(full_source: str, route_header: str) -> None:
    routes = _route_enumerators(route_header)
    answers = _src1_reader_answers(full_source)
    # Every route the dispatch's classifier can return has an answer, and the answer is its kernel's.
    assert sorted(answers) == sorted(routes) == sorted(_SRC1_READER_ANSWERS), (routes, answers)
    assert answers == _SRC1_READER_ANSWERS, answers
    for kernel in ("ggml_sycl_mul_mat_vec_p021", "ggml_sycl_mul_mat_vec_nc"):
        body = _function_body(full_source, re.compile(r"static\s+void\s+" + kernel + r"\s*\([^)]*\)\s*try\s*\{"))
        assert "GGML_ASSERT(src1->type == GGML_TYPE_F32);" in body, kernel
    op_mul_mat_signature = re.compile(r"static\s+bool\s+ggml_sycl_op_mul_mat\s*\([^)]*\)\s*(?:try\s*)?\{")
    op_mul_mat = _function_body(full_source, op_mul_mat_signature)
    assert "GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));" in op_mul_mat
    # The unified-kernel arm is one function: the orchestrator's select takes its arm from it and supports_op asks
    # it, so the two cannot disagree.
    assert _compact(_function_body(full_source, _UNIFIED_ARM_SIGNATURE)) == _UNIFIED_ARM_BODY
    select = _compact(_function_body(full_source, _SELECT_SIGNATURE))
    assert select.count(_SELECT_UNIFIED_ARM) == 1 and "should_use_unified(" not in select, "select's unified arm"
    # The block the arm's decision leads to reads src1 through a float pointer and never tests its type.
    assert full_source.count(_UNIFIED_DISPATCH_BLOCK) == 1, "unified dispatch block"
    block_open = full_source.index(_UNIFIED_DISPATCH_BLOCK) + len(_UNIFIED_DISPATCH_BLOCK) - 1
    unified_dispatch = full_source[block_open : _matching_close(full_source, block_open)]
    assert unified_dispatch.count(_UNIFIED_FLOAT_READ) == 1, "the unified dispatch reads src1 as float"
    assert "f32_to_fp16(src1_data," in unified_dispatch, "its oneDNN prompt path converts src1 from float"
    assert "src1->type" not in unified_dispatch, "the unified dispatch tests src1's type"
    # Both batched executors convert a non-F16 src1 through the strided converter the helper asks about.
    for signature in _BATCHED_SIGNATURES:
        batched = _function_body(full_source, signature)
        assert "if (src1->type != GGML_TYPE_F16) {" in batched and _BATCHED_NC_CONVERT in batched, signature.pattern

    support = _compact(_supports_source(full_source))
    mul_mat = support[support.index("caseGGML_OP_MUL_MAT:") : support.index("caseGGML_OP_OUT_PROD:")]
    assert mul_mat.count(_SRC1_READER_GUARD) == 1, "MUL_MAT asks whether a route reads its src1"
    # A strided view of an expert table has no MUL_MAT_ID route; the indexed prologue declines it first.
    assert support.count(_INDEXED_CONTIGUITY_GUARD) == 1, "MUL_MAT_ID declines a non-contiguous src0"


def _assert_prompt_staging_contract(full_source: str) -> None:
    site = full_source.find(_PROMPT_STAGING_SITE)
    assert site >= 0 and full_source.count(_PROMPT_STAGING_SITE) == 1, "prompt activation staging site"
    open_brace = site + len(_PROMPT_STAGING_SITE) - 1
    body = _compact(full_source[open_brace + 1 : _matching_close(full_source, open_brace)])
    # Before any expert kernel is submitted, a staging failure refuses the graph through the recoverable path.
    assert "GGML_ABORT" not in body, body
    assert re.search(r"throwggml_sycl_fallback_error\(\"[^\"]+\"\);$", body), body
    # The two per-expert sites stay aborts: an unwind there would release expert-weight leases and scratch
    # owners that kernels already submitted still read.
    later = [m.start() for m in re.finditer(re.escape(_PROMPT_STAGING_ABORT), full_source)]
    assert len(later) == 2 and all(position > site for position in later), later
    abort_call = re.compile(r"GGML_ABORT\(\s*\"\[MoE\] " + re.escape(_PROMPT_STAGING_ABORT))
    aborts = [m.start() for m in abort_call.finditer(full_source)]
    assert len(aborts) == 2 and all(position > site for position in aborts), aborts


def test_mul_mat_and_mul_mat_id_decline_what_no_route_reads() -> None:
    full_source = SUPPORT_SOURCE.read_text(encoding="utf-8")
    _assert_mul_mat_src1_contract(full_source, ROUTE_SOURCE.read_text(encoding="utf-8"))
    _assert_prompt_staging_contract(full_source)


def test_mul_mat_route_and_staging_mutations_are_detected() -> None:
    full_source = SUPPORT_SOURCE.read_text(encoding="utf-8")
    route_header = ROUTE_SOURCE.read_text(encoding="utf-8")

    def rejects_src1(source: str, header: str = route_header) -> bool:
        return _contract_rejects(lambda text: _assert_mul_mat_src1_contract(text, header), source)

    def mutate(source: str, old: str, new: str) -> str:
        assert source.count(old) == 1, old
        return source.replace(old, new, 1)

    # The src1 question removed from MUL_MAT.
    guard = "                if (b->type != GGML_TYPE_F32 && !ggml_sycl_mul_mat_src1_has_reader(a, b, op, device)) {\n"
    assert rejects_src1(mutate(full_source, guard, "                if (false) {\n"))
    # A route that asserts an F32 src1 answered as a reader.
    assert rejects_src1(mutate(
        full_source,
        "        case GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_P021:\n        case GGML_SYCL_MUL_MAT_F16_ROUTE_VEC_NC:\n",
        "        case GGML_SYCL_MUL_MAT_F16_ROUTE_VEC_NC:\n",
    ).replace(
        "        case GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED:\n",
        "        case GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED:\n        case GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_P021:\n", 1,
    ))
    none_answer = (
        "            return !ggml_sycl_mul_mat_takes_unified_arm(src0, src1, device) && src1->ne[2] == 1 && "
        "src1->ne[3] == 1;\n"
    )
    # The orchestrator answered as reading any src1 shape.
    assert rejects_src1(mutate(full_source, none_answer, "            return true;\n"))
    # The unified-kernel arm, which reads src1 as F32, answered as reading a one-plane src1 of any type.
    assert rejects_src1(mutate(full_source, none_answer, "            return src1->ne[2] == 1 && src1->ne[3] == 1;\n"))
    # select deciding its unified arm inline again, a second copy supports_op does not ask.
    select_arm = "    if (allow_unified && ggml_sycl_mul_mat_takes_unified_arm(src0, src1, ctx_.device)) {\n"
    assert rejects_src1(mutate(
        full_source, select_arm,
        "    if (allow_unified && ggml_sycl_unified_dispatch_enabled() && "
        "ggml_sycl::should_use_unified(src0->type)) {\n",
    ))
    # The batched routes answered as reading every type, including those the strided converter lacks.
    batched_answer = (
        "            return src1->type == GGML_TYPE_F16 || ggml_get_to_fp16_nc_sycl(src1->type) != nullptr;\n"
    )
    assert rejects_src1(mutate(full_source, batched_answer, "            return true;\n"))
    # A route added to the classifier that the helper does not answer.
    assert rejects_src1(full_source, mutate(
        route_header,
        "    GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_SCALAR,",
        "    GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_SCALAR,\n    GGML_SYCL_MUL_MAT_F16_ROUTE_EXTRA,",
    ))
    # The contiguity decline removed from MUL_MAT_ID.
    contiguity = "            if (!ggml_is_contiguous(op->src[0])) {\n                return false;\n            }\n"
    assert rejects_src1(mutate(full_source, contiguity, ""))

    # The first staging site aborting again.
    site = full_source.index(_PROMPT_STAGING_SITE)
    throw = full_source.index("throw ggml_sycl_fallback_error(", site)
    end = full_source.index(";", throw) + 1
    aborting = full_source[:throw] + 'GGML_ABORT("staging");' + full_source[end:]
    assert _contract_rejects(_assert_prompt_staging_contract, aborting)
    # A per-expert site turned into log-and-throw with its message kept (its unwind would release live leases).
    loop_message = full_source.index(_PROMPT_STAGING_ABORT, site)
    loop_abort = full_source.rindex("GGML_ABORT(", site, loop_message)
    loop_end = full_source.index(");", loop_message) + 2
    thrown = (full_source[:loop_abort] + "GGML_LOG_ERROR(" + full_source[loop_abort + len("GGML_ABORT(") : loop_end]
              + ' throw ggml_sycl_fallback_error("staging");' + full_source[loop_end:])
    assert full_source.count(_PROMPT_STAGING_ABORT) == thrown.count(_PROMPT_STAGING_ABORT)
    assert _contract_rejects(_assert_prompt_staging_contract, thrown)

    # The unified dispatch block gaining a src1 type check, or reading src1 some other way.
    unified_block = full_source.index(_UNIFIED_DISPATCH_BLOCK)
    checked = (full_source[: unified_block + len(_UNIFIED_DISPATCH_BLOCK)]
               + "\n            if (src1->type != GGML_TYPE_F32) { return; }"
               + full_source[unified_block + len(_UNIFIED_DISPATCH_BLOCK) :])
    assert rejects_src1(checked)
    float_read = full_source.index(_UNIFIED_FLOAT_READ, unified_block)
    float_end = float_read + len(_UNIFIED_FLOAT_READ)
    reread = full_source[:float_read] + "static_cast<const float *>(src1->data)" + full_source[float_end:]
    assert rejects_src1(reread)


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

    support_start = _supports_impl_start(full_support)
    rope_start = full_support.index("case GGML_OP_ROPE:", support_start)
    rope_end = full_support.index("case GGML_OP_IM2COL:", rope_start)
    rope_case = full_support[rope_start:rope_end]
    for old, new in ((" == nullptr", " != nullptr"), (" || ", " && "), ("[15] != 0", "[15] == 0")):
        assert old in rope_case
        mutated_case = rope_case.replace(old, new, 1)
        mutated_source = full_support[:rope_start] + mutated_case + full_support[rope_end:]
        assert _contract_rejects(_assert_rope_contract, mutated_source), old

    inventory = BACKEND_OPS_SOURCE.read_text(encoding="utf-8")
    inventory_start = inventory.index("// single inplace test per type/mode/ff")
    row_start = inventory.index("test_cases.emplace_back(new test_rope", inventory_start)
    row_end = inventory.index("\n", row_start)
    row = inventory[row_start:row_end]
    injected_row = row.replace("ff, 0, true, true", "ff, 2, true, true")
    assert injected_row != row
    mutated_inventory = inventory[:row_end] + "\n" + injected_row + inventory[row_end:]
    assert _contract_rejects(_assert_rope_inventory_contract, mutated_inventory)
    # A BF16 in-place row turned F16 is admitted, so the BF16 block's admitted count moves.
    bf16_inplace = (
        "test_rope(GGML_TYPE_BF16, {128, 32, 2, 1}, 128, mode, 512, 1.4245f, 0.7465f, 1.4245f, false, 0, true, true)"
    )
    assert inventory.count(bf16_inplace) == 1
    mutated_inventory = inventory.replace(bf16_inplace, bf16_inplace.replace("GGML_TYPE_BF16", "GGML_TYPE_F16"), 1)
    assert _contract_rejects(_assert_rope_inventory_contract, mutated_inventory)
    # A new ROPE block between the offset block and the real-model rows is read, not skipped.
    real_model = inventory.index("// Real-model RoPE")
    added_block = "for (bool ff : {false, true}) {\n" + row.strip() + "\n}\n    "
    assert _contract_rejects(
        _assert_rope_inventory_contract, inventory[:real_model] + added_block + inventory[real_model:]
    )
    # Widening ROPE's type check to BF16 breaks the model the inventory counts rely on.
    core = _core_semantics_source(full_support)
    core_start = full_support.index(core)
    widened = core.replace(
        "op->src[0]->type == GGML_TYPE_F16;",
        "op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16;",
        1,
    )
    assert widened != core
    assert _contract_rejects(
        _assert_rope_contract, full_support[:core_start] + widened + full_support[core_start + len(core) :]
    )

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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
