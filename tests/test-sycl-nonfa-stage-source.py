"""Source gate for the non-FA staging cohort's two shared functions and its measure visitor
(zhcn design 2.8).

The batched f16 mul_mat's src1 staging becomes the `context-nonfa-stage` cohort. Its slot is
sized by the measure pass and claimed by the op, so the visitor must count exactly what the
site stages. This gate pins, on comment-stripped text, that there is one derivation of each
fact and every user calls it:

- the route classifier and the staged element count are defined once, in `nonfa-stage.hpp`,
  and `ggml_sycl_mul_mat` takes its branch from `ggml_sycl_mul_mat_f16_route_of` -- the
  predicates of the old chain (`ggml_is_permuted(src0)`, `ggml_is_transposed(src0)`, the
  inline `kqv` lambda) are gone from its body;
- `ggml_sycl_mul_mat_batched_sycl` sizes its src1 staging only through
  `ggml_sycl_batched_f16_src1_stage_elems`, once per path, with the path the env names;
- the environment (row-split, weight, debug override, staging path) is one function that the
  dispatch and the measure view's callback both call;
- the visitor reads the same two functions, demands slot 0 of the NONFA_STAGE cohort in
  element bytes, takes the maximum, and names an f16 mul_mat it cannot classify;
- the visitor's row is in the table; the non-FA guard and its two call sites are unchanged
  (the guard is retired only once a claim exists, which is not this change).

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
SYCL_CPP = (SYCL / "ggml-sycl.cpp").read_text()
STAGE_HPP = (SYCL / "nonfa-stage.hpp").read_text()
MEASURE_HPP = (SYCL / "context-tenant-measure.hpp").read_text()
MEASURE_CPP = (SYCL / "context-tenant-measure.cpp").read_text()
SYCL_CMAKE = (SYCL / "CMakeLists.txt").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_MUL_MAT = (
    "static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, "
    "const ggml_tensor * src1, ggml_tensor * dst, const layout_mode * forced_layout = nullptr)"
)
_BATCHED = (
    "static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, "
    "const ggml_tensor * src1, ggml_tensor * dst) try"
)
_ENV_OF = (
    "static ggml_sycl_mul_mat_route_env ggml_sycl_mul_mat_route_env_of(const ggml_tensor * src0, "
    "const ggml_tensor * src1)"
)
_ADAPTER = (
    "bool context_measure_mul_mat_route_env(void *, const ggml_tensor * node, ggml_sycl_mul_mat_route_env * env)"
)
_VISIT = (
    "void context_nonfa_stage_visit(const ggml_tensor * node, const context_measure_view & view, "
    "context_demand_accum & acc)"
)
_ROUTE_OF = (
    "inline ggml_sycl_mul_mat_f16_route ggml_sycl_mul_mat_f16_route_of(const ggml_tensor * src0, "
    "const ggml_tensor * src1, const ggml_tensor * dst, const ggml_sycl_mul_mat_route_env & env)"
)
_STAGE_ELEMS = "inline size_t ggml_sycl_batched_f16_src1_stage_elems(const ggml_tensor * src1, bool strided)"

_R = "GGML_SYCL_MUL_MAT_F16_ROUTE_"


# ---- the dispatch takes its branch from the classifier ----
def dispatch_ok(code: str) -> bool:
    b = function_body(code, _MUL_MAT)
    pins = [
        "const ggml_sycl_mul_mat_route_env route_env = ggml_sycl_mul_mat_route_env_of(src0, src1);",
        "const bool force_simple_kqv = route_env.kqv_force_simple && ggml_sycl_mul_mat_is_kqv(src0, src1, dst);",
        "const bool split = route_env.split;",
        "const bool batched_has_weight = route_env.has_weight;",
        "const ggml_sycl_mul_mat_f16_route f16_route = ggml_sycl_mul_mat_f16_route_of(src0, src1, dst, route_env);",
        f"if (f16_route == {_R}KQ_P021 || f16_route == {_R}KQ_BATCHED) {{",
        f"if (f16_route == {_R}KQ_P021) {{",
        f"}} else if (f16_route == {_R}VEC_NC) {{",
        f"}} else if (f16_route == {_R}KQKV_BATCHED || f16_route == {_R}KQKV_SCALAR) {{",
        f"if (f16_route == {_R}KQKV_BATCHED) {{ try {{ ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);",
    ]
    for p in pins:
        if b.count(z(p)) != 1:
            return False
    # the batched op is reached from exactly the two batched branches
    if b.count(z("ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);")) != 2:
        return False
    # the old chain's predicates on src0 and the inline kqv lambda are gone
    for gone in ("ggml_is_permuted(src0)", "ggml_is_transposed(src0)", "is_kqv_matmul", "kqv_matmul"):
        if z(gone) in b:
            return False
    # one environment per dispatch, one classification
    return b.count(z("ggml_sycl_mul_mat_route_env_of(")) == 1 and b.count(z("ggml_sycl_mul_mat_f16_route_of(")) == 1


def test_the_dispatch_takes_its_branch_from_the_classifier():
    assert dispatch_ok(code_of(SYCL_CPP))


def test_dispatch_mutants():
    code = code_of(SYCL_CPP)
    b = function_body(code, _MUL_MAT)
    for name, old, new in [
        ("the env read from a second derivation", "ggml_sycl_mul_mat_route_env_of(src0, src1);", "ggml_sycl_mul_mat_route_env();"),
        ("split derived inline", "const bool split = route_env.split;", "const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);"),
        ("has_weight derived inline", "const bool batched_has_weight = route_env.has_weight;",
         "const bool batched_has_weight = ggml_sycl_tensor_is_weight(src0) || ggml_sycl_tensor_is_weight(src1);"),
        ("the kqv branch not the classifier's", f"}} else if (f16_route == {_R}VEC_NC) {{",
         "} else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0)) {"),
        ("the scalar branch lost", f"f16_route == {_R}KQKV_BATCHED || f16_route == {_R}KQKV_SCALAR", f"f16_route == {_R}KQKV_BATCHED"),
        ("the batched branch not the classifier's", f"if (f16_route == {_R}KQKV_BATCHED) {{ try {{", "if (!force_simple_kqv) { try {"),
    ]:
        assert not dispatch_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    # an inline chain predicate returns
    assert not dispatch_ok(code.replace(b, b[:-1] + z("bool x = ggml_is_permuted(src0);") + "}", 1)), "mutant 'src0 predicate back' slipped through"


# ---- the batched op sizes its staging through the shared function ----
def batched_ok(code: str) -> bool:
    b = function_body(code, _BATCHED)
    if b.count(z("ggml_sycl_batched_f16_src1_stage_elems(src1, true)")) != 1:
        return False
    if b.count(z("ggml_sycl_batched_f16_src1_stage_elems(src1, false)")) != 1:
        return False
    # the strided path is the oneDNN one, and the two paths split on the same use_onemath the env reads
    if b.count(z("const bool use_onemath_batched = ggml_sycl_batched_f16_use_onemath(src0, src1);")) != 1:
        return False
    if b.count(z("if (!use_onemath_batched) {")) != 1:
        return False
    # the old inline derivations are gone
    return not any(z(s) in b for s in ("src1->nb[last_str]", "ggml_nelements(src1)", "largest_str"))


def test_the_batched_op_sizes_its_staging_through_the_shared_function():
    assert batched_ok(code_of(SYCL_CPP))


def test_batched_mutants():
    code = code_of(SYCL_CPP)
    b = function_body(code, _BATCHED)
    for name, old, new in [
        ("strided path counts elements", "ggml_sycl_batched_f16_src1_stage_elems(src1, true)", "ggml_sycl_batched_f16_src1_stage_elems(src1, false)"),
        ("element path counts the strided span", "ggml_sycl_batched_f16_src1_stage_elems(src1, false)", "ggml_sycl_batched_f16_src1_stage_elems(src1, true)"),
        ("the strided path derived inline", "static_cast<int64_t>(ggml_sycl_batched_f16_src1_stage_elems(src1, true))", "src1->nb[last_str] * src1->ne[last_dim] / type_size_src1"),
        ("the element path derived inline", "static_cast<int64_t>(ggml_sycl_batched_f16_src1_stage_elems(src1, false))", "ggml_nelements(src1)"),
        ("the path split on another condition", "if (!use_onemath_batched) {", "if (src1->ne[2] > 1) {"),
    ]:
        assert not batched_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


# ---- the environment is one function ----
_ENV_FROM_CALL = (
    "return ggml_sycl_mul_mat_route_env_from(src0, src1, ggml_backend_buffer_is_sycl_split, ggml_sycl_tensor_is_weight, "
    "g_ggml_sycl_kqv_force_simple || g_ggml_sycl_kqv_disable_fp16, stage_strided);"
)
_ENV_FROM = (
    "inline ggml_sycl_mul_mat_route_env ggml_sycl_mul_mat_route_env_from(const ggml_tensor * src0, const ggml_tensor * src1, "
    "bool (*is_split_buffer)(ggml_backend_buffer_t), bool (*is_weight_tensor)(const ggml_tensor *), "
    "bool kqv_force_simple, bool stage_strided)"
)


def env_ok(code: str) -> bool:
    e = function_body(code, _ENV_OF)
    pins = [
        _ENV_FROM_CALL,
        "const bool stage_strided = !ggml_sycl_batched_f16_use_onemath(src0, src1);",
        "const bool stage_strided = false;",
    ]
    if any(e.count(z(p)) != 1 for p in pins):
        return False
    # the strided path exists only where the oneDNN staging is compiled: the #if / #else on the raw text
    region = strip_ws(SYCL_CPP_ENV_REGION)
    if not re.search(r"#\s*if GGML_SYCL_DNNL [^#]*stage_strided = !ggml_sycl_batched_f16_use_onemath\(src0, src1\); #\s*else [^#]*stage_strided = false; #\s*endif", region):
        return False
    a = function_body(code, "namespace ggml_sycl { " + _ADAPTER)
    return a.count(z("*env = ggml_sycl_mul_mat_route_env_of(node->src[0], node->src[1]);")) == 1 and \
        z("node->op != GGML_OP_MUL_MAT") in a


def strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# The #if around stage_strided is a preprocessor line the comment-stripped text keeps, so it is checked
# on the raw region of the function instead.
_env_start = SYCL_CPP.index("static ggml_sycl_mul_mat_route_env ggml_sycl_mul_mat_route_env_of")
SYCL_CPP_ENV_REGION = SYCL_CPP[_env_start : SYCL_CPP.index("stage_strided);", _env_start) + len("stage_strided);")]


def test_the_environment_is_one_function_both_sites_call():
    code = code_of(SYCL_CPP)
    assert env_ok(code)
    a = "namespace ggml_sycl { " + _ADAPTER
    assert z(a) in code
    # the two environment readers are the only callers
    assert code.count(z("ggml_sycl_mul_mat_route_env_of(")) == 3  # definition, dispatch, adapter
    assert code.count(z("ggml_sycl_mul_mat_route_env_from(")) == 1  # only the one function builds it


def test_environment_mutants():
    code = code_of(SYCL_CPP)
    e = function_body(code, _ENV_OF)
    for name, old, new in [
        ("split from the wrong predicate", "ggml_backend_buffer_is_sycl_split, ggml_sycl_tensor_is_weight", "ggml_backend_buffer_is_sycl_split, nullptr"),
        ("only one override", "g_ggml_sycl_kqv_force_simple || g_ggml_sycl_kqv_disable_fp16", "g_ggml_sycl_kqv_force_simple"),
        ("the staging path inverted", "const bool stage_strided = !ggml_sycl_batched_f16_use_onemath(src0, src1);", "const bool stage_strided = ggml_sycl_batched_f16_use_onemath(src0, src1);"),
        ("the staging path fixed", "const bool stage_strided = !ggml_sycl_batched_f16_use_onemath(src0, src1);", "const bool stage_strided = true;"),
    ]:
        assert not env_ok(code.replace(e, mutate(e, old, new), 1)), f"mutant {name!r} slipped through"


def env_from_ok(hpp: str) -> bool:
    h = code_of(hpp)
    if z(_ENV_FROM) not in h:
        return False
    b = function_body(h, _ENV_FROM)
    pins = [
        "env.split = src0->buffer != nullptr && is_split_buffer(src0->buffer);",
        "env.has_weight = (src0->buffer != nullptr && is_weight_tensor(src0)) || (src1->buffer != nullptr && is_weight_tensor(src1));",
        "env.kqv_force_simple = kqv_force_simple;",
        "env.stage_strided = stage_strided;",
    ]
    return all(b.count(z(p)) == 1 for p in pins)


def test_the_environment_builder_treats_an_unplaced_operand_as_neither_split_nor_weight():
    assert env_from_ok(STAGE_HPP)


def test_environment_builder_mutants():
    h = code_of(STAGE_HPP)
    b = function_body(h, _ENV_FROM)
    for name, old, new in [
        ("split asked of a null buffer", "src0->buffer != nullptr && is_split_buffer(src0->buffer)", "is_split_buffer(src0->buffer)"),
        ("src0's weight asked of a null buffer", "(src0->buffer != nullptr && is_weight_tensor(src0))", "is_weight_tensor(src0)"),
        ("src1's weight asked of a null buffer", "(src1->buffer != nullptr && is_weight_tensor(src1))", "is_weight_tensor(src1)"),
        ("only src0's weight", " || (src1->buffer != nullptr && is_weight_tensor(src1))", ""),
        ("the override dropped", "env.kqv_force_simple = kqv_force_simple;", "env.kqv_force_simple = false;"),
    ]:
        mutated = h.replace(b, mutate(b, old, new), 1)
        # env_from_ok takes raw header text; the mutated text is already comment-stripped, which code_of leaves unchanged
        assert not env_from_ok(mutated), f"mutant {name!r} slipped through"


# ---- the visitor ----
def visit_ok(code: str) -> bool:
    v = function_body(code, _VISIT)
    pins = [
        "if (node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || node->src[1] == nullptr || node->src[0]->type != GGML_TYPE_F16) { return; }",
        "if (view.mul_mat_route_env == nullptr || !view.mul_mat_route_env(view.sched_ctx, node, &env)) {",
        "acc.fail(std::string(\"context-nonfa-stage: no route environment for the f16 mul_mat \") + node->name);",
        "if (!ggml_sycl_mul_mat_routes_batched_f16(node->src[0], node->src[1], node, env)) { return; }",
        "const size_t elems = ggml_sycl_batched_f16_src1_stage_elems(node->src[1], env.stage_strided);",
        "acc.demand(view, GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE, 0, (uint64_t) elems * GGML_SYCL_NONFA_STAGE_ELEM_BYTES);",
    ]
    if any(v.count(z(p)) != 1 for p in pins):
        return False
    # the route is tested before the size is read, and the visitor sizes nothing itself
    if not v.index(z("ggml_sycl_mul_mat_routes_batched_f16(")) < v.index(z("ggml_sycl_batched_f16_src1_stage_elems(")):
        return False
    if any(s in v for s in (z("ggml_nelements("), z("->nb["), z("ggml_is_permuted"), z("ggml_is_transposed"))):
        return False
    # the row is in the table, before the terminator
    t = code.find(z("const ggml_sycl::context_measure_visitor k_visitors[] = {"))
    if t == -1:
        return False
    table = code[t : code.index("};", t)]
    row = z('{ "nonfa-stage", ggml_sycl::context_nonfa_stage_visit },')
    term = z("{ nullptr, nullptr },")
    return table.count(row) == 1 and table.count(term) == 1 and table.index(row) < table.index(term)


def test_the_visitor_sizes_the_slot_from_the_shared_functions():
    assert visit_ok(code_of(MEASURE_CPP))


def test_visitor_mutants():
    code = code_of(MEASURE_CPP)
    v = function_body(code, _VISIT)
    for name, old, new in [
        ("a quantized src0 reaches the environment", " || node->src[0]->type != GGML_TYPE_F16", ""),
        ("a missing environment reads as zero", "acc.fail(std::string(\"context-nonfa-stage: no route environment for the f16 mul_mat \") + node->name);", ""),
        ("every f16 mul_mat counted", "if (!ggml_sycl_mul_mat_routes_batched_f16(node->src[0], node->src[1], node, env)) { return; }", ""),
        ("the staging path fixed", "env.stage_strided", "true"),
        ("the wrong cohort", "GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE", "GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE"),
        ("the wrong slot", "GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE, 0,", "GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE, 1,"),
        ("the size in elements", " * GGML_SYCL_NONFA_STAGE_ELEM_BYTES", ""),
        ("the size derived a second way", "ggml_sycl_batched_f16_src1_stage_elems(node->src[1], env.stage_strided)", "ggml_nelements(node->src[1])"),
    ]:
        assert not visit_ok(code.replace(v, mutate(v, old, new), 1)), f"mutant {name!r} slipped through"
    row = '{ "nonfa-stage", ggml_sycl::context_nonfa_stage_visit },'
    assert not visit_ok(code.replace(z(row), "", 1)), "mutant 'the row left out of the table' slipped through"
    tail = z("{ nullptr, nullptr },")
    assert not visit_ok(code.replace(z(row), "", 1).replace(tail, tail + z(row), 1)), "mutant 'the row after the terminator' slipped through"


# ---- one definition of each shared function ----
def test_the_shared_functions_are_defined_once_in_the_header():
    stage = code_of(STAGE_HPP)
    for sig in (_ROUTE_OF, _STAGE_ELEMS):
        assert z(sig) in stage
    for name in ("ggml_sycl_mul_mat_routes_batched_f16", "ggml_sycl_mul_mat_f16_route_of", "ggml_sycl_batched_f16_src1_stage_elems",
                 "ggml_sycl_mul_mat_is_kqv"):
        defs = 0
        for path in list(SYCL.glob("*.cpp")) + list(SYCL.glob("*.hpp")) + list(SYCL.glob("*.h")):
            text = code_of(path.read_text())
            defs += len(re.findall(r"(?:inline|static) (?:bool|size_t|ggml_sycl_mul_mat_f16_route) " + re.escape(name) + r"\(", text))
        assert defs == 1, f"{name} is defined {defs} times"
    # the header names ggml types only, so a host test builds it
    assert "sycl" not in code_of(STAGE_HPP).replace("ggml_sycl", "").replace("GGML_SYCL", "").lower()


# ---- the guard is unchanged ----
def test_the_nonfa_guard_and_its_two_calls_are_unchanged():
    code = code_of(SYCL_CPP)
    assert code.count(z("ggml_sycl_check_nonfa_attn_scratch(")) == 3  # the definition and the two call sites
    # no tenants_planned yet: the guard is retired only once a claim exists
    assert "tenants_planned" not in code


# ---- the host test is registered ----
def test_the_host_test_is_registered_and_links_ggml_base_only():
    c = z(SYCL_CMAKE)
    assert z("add_executable(test-context-nonfa-stage tests/test-context-nonfa-stage.cpp context-tenant-measure.cpp)") in c
    assert z("target_link_libraries(test-context-nonfa-stage PRIVATE ggml-base)") in c
    assert z("add_test(NAME sycl-context-nonfa-stage COMMAND test-context-nonfa-stage)") in c
