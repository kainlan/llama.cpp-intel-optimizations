"""Source gate for the batched F32 MUL_MAT route: one oneDNN matmul for a node whose batch the per-plane loop of
ggml_sycl_op_mul_mat would run as one GEMM per plane.

Which nodes the route takes is the host test's (test-sycl-mul-mat-f32-batched-route); what it computes is the device
test's (test-sycl-mul-mat-f32-batched-device). This gate pins the form those two cannot see, on comment-stripped,
whitespace-canonical text:

- the dispatch consults the route before the orchestrator: in ggml_sycl_mul_mat the predicate
  ggml_sycl_mul_mat_f32_batched_route_of is asked once, and the route op is called once, in the condition of the
  `else if` that directly precedes the `else` holding the orchestrator's select(), and only for a route that is not
  NONE;
- a decline falls through and does not throw: the route op returns the has_value() of the one gemm() result it
  holds, and its body has no throw and no abort, so a declined scratchpad (nullopt) is false and the dispatch goes
  on to the orchestrator;
- the environment the predicate reads is filled from the backend's own facts: the recording flag from
  ggml_sycl_graph_recording_active() and the queue's state, tensor parallelism from g_sycl_tp_config, split and
  weight from the route environment the f16 chain reads;
- the route body makes one submission and keeps nothing: exactly one DnnlGemmWrapper::gemm call, no loop, no
  allocation of any kind, no wait, no static and no store through a member or a pointer;
- gemm() chooses its profiler label by batch count: both of its labels name one variable that is the batched row
  when a batch count is above 1 and the unbatched row otherwise.

Every clause has mutants that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
SYCL_CPP = (SYCL / "ggml-sycl.cpp").read_text()
GEMM_HPP = (SYCL / "gemm.hpp").read_text()
STAGE_HPP = (SYCL / "nonfa-stage.hpp").read_text()

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
_ROUTE_OP = (
    "[[nodiscard]] static bool ggml_sycl_mul_mat_f32_batched(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, "
    "const ggml_tensor * src1, ggml_tensor * dst, ggml_sycl_mul_mat_f32_batched_route route)"
)
_ENV_OF = (
    "static ggml_sycl_mul_mat_f32_batched_env ggml_sycl_mul_mat_f32_batched_env_of(ggml_backend_sycl_context & ctx, "
    "const ggml_sycl_mul_mat_route_env & route_env, bool on_device)"
)
_ROUTE_OF = (
    "inline ggml_sycl_mul_mat_f32_batched_route ggml_sycl_mul_mat_f32_batched_route_of(const ggml_tensor * src0, "
    "const ggml_tensor * src1, const ggml_tensor * dst, const ggml_sycl_mul_mat_f32_batched_env & env)"
)
_NONE = "GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_NONE"


def balanced_end(code: str, open_at: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket that closes the one at open_at."""
    depth = 0
    for j in range(open_at, len(code)):
        if code[j] == open_ch:
            depth += 1
        elif code[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    raise AssertionError("unbalanced brackets")


def call_args(code: str, open_at: int) -> list:
    """The top-level arguments of the call whose `(` is at open_at."""
    end = balanced_end(code, open_at, "(", ")")
    args, depth, start = [], 0, open_at + 1
    for j in range(open_at + 1, end):
        ch = code[j]
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(code[start:j])
            start = j + 1
    args.append(code[start:end])
    return args


# ---- the dispatch consults the route before the orchestrator, and a decline falls through to it ----
def dispatch_errors(code: str) -> list:
    errs = []
    b = function_body(code, _MUL_MAT)
    if b.count("ggml_sycl_mul_mat_f32_batched_route_of(") != 1:
        errs.append("the predicate is not asked exactly once")
    calls = [m.start() for m in re.finditer(r"\bggml_sycl_mul_mat_f32_batched\(", b)]
    if len(calls) != 1:
        return errs + ["the route op is not called exactly once (%d)" % len(calls)]
    call = calls[0]
    asked = b.find("ggml_sycl_mul_mat_f32_batched_route_of(")
    if asked == -1 or asked > call:
        errs.append("the predicate is not asked before the route op runs")
    # the call is in the condition of an `else if`, after a route != NONE guard joined by &&
    head = b.rfind("}else if(", 0, call)
    if head == -1:
        return errs + ["the route op is not in an else-if condition"]
    cond_open = head + len("}else if")
    cond_close = balanced_end(b, cond_open, "(", ")")
    if not (cond_open < call < cond_close):
        return errs + ["the route op is not in the else-if's condition"]
    cond = b[cond_open + 1 : cond_close]
    m = re.fullmatch(r"(\w+)!=" + _NONE + r"&&ggml_sycl_mul_mat_f32_batched\(ctx,src0,src1,dst,(\w+)\)", cond)
    if not m or m.group(1) != m.group(2):
        errs.append("the condition is not `route != NONE && op(ctx, src0, src1, dst, route)`: %r" % cond)
    elif not re.search(r"\b" + m.group(1) + r"=ggml_sycl_mul_mat_f32_batched_route_of\(", b):
        errs.append("the route the op is called with is not the predicate's answer")
    # the branch body is empty: a taken route did the op; its `else` is the orchestrator's
    body_open = cond_close + 1
    if b[body_open] != "{":
        return errs + ["the else-if has no block"]
    body_close = balanced_end(b, body_open, "{", "}")
    if b[body_open + 1 : body_close].strip():
        errs.append("the taken branch does more than end the chain: %r" % b[body_open + 1 : body_close][:120])
    tail = b[body_close:]
    if not tail.startswith("}else{ggml_sycl::test_record_orchestrator_call();"):
        errs.append("the branch after the route is not the orchestrator's else")
    else:
        else_close = balanced_end(b, body_close + len("}else"), "{", "}")
        if "ctx.matmul_orchestrator.select(" not in b[body_close:else_close]:
            errs.append("the orchestrator's select() is not in the else after the route")
    return errs


def test_the_dispatch_consults_the_route_before_the_orchestrator():
    assert dispatch_errors(code_of(SYCL_CPP)) == []


def _route_branch(b: str) -> str:
    """The z-spelled `else if (...) {...}` text of the route's branch."""
    call = b.index("ggml_sycl_mul_mat_f32_batched(ctx,")
    head = b.rfind("}else if(", 0, call)
    cond_close = balanced_end(b, head + len("}else if"), "(", ")")
    return b[head : balanced_end(b, cond_close + 1, "{", "}") + 1]


def test_dispatch_mutants():
    code = code_of(SYCL_CPP)
    b = function_body(code, _MUL_MAT)
    branch = _route_branch(b)
    cond = branch[len("}else if(") : branch.index("){")]
    route_var = cond.split("!=")[0]
    for name, new_b in [
        ("the route branch is gone", b.replace(branch, "}", 1)),
        ("the route is asked after the orchestrator",
         b.replace(branch, "}", 1).replace("ggml_sycl::test_record_orchestrator_call();",
                                           "ggml_sycl::test_record_orchestrator_call();if(" + cond + "){return;}", 1)),
        ("a decline returns instead of falling through", mutate(b, branch, branch + "else if(true){return;}")),
        ("the op runs whatever the predicate said", mutate(b, cond, cond.replace(route_var + "!=" + _NONE + "&&", ""))),
        ("the op runs only for a NONE route", mutate(b, cond, cond.replace("!=" + _NONE, "==" + _NONE))),
        ("the op runs with another route", mutate(b, cond, cond.replace(",dst," + route_var + ")",
                                                                         ",dst,GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_BATCHED)"))),
        ("a taken route also runs the orchestrator", mutate(b, branch, branch[:-1] + "ctx.matmul_orchestrator.select(src0,src1,dst,forced_layout);}")),
        ("the route is called a second time",
         mutate(b, branch, branch + "else if(ggml_sycl_mul_mat_f32_batched(ctx,src0,src1,dst," + route_var + ")){}")),
        ("the predicate is asked twice", b[:-1] + z("(void) ggml_sycl_mul_mat_f32_batched_route_of(src0, src1, dst, {});") + "}"),
    ]:
        assert dispatch_errors(code.replace(b, new_b, 1)), f"mutant {name!r} slipped through"


# ---- the route op: one submission, a decline is false, nothing kept ----
_ALLOC = re.compile(r"\bunified_alloc\w*\(|\bunified_allocate\w*\(|\bggml_sycl_pool_alloc\b|\bmalloc\w*\(|\.alloc\(|"
                    r"\bsycl::malloc\w*|\bmem_handle\b|\balloc_owner\b|\bfree\(")
_WAIT = re.compile(r"\bwait(?:_and_throw)?\(|\bext_oneapi_submit_barrier\(|\bsynchronize\(")
_STORE = re.compile(r"(?:->|\.)\w+(?:\[[^\]]*\])*=(?!=)")


def route_op_errors(code: str) -> list:
    errs = []
    full = function_body(code, _ROUTE_OP)
    b = full[full.index("{") :]
    gemms = [m.start() for m in re.finditer(r"\bDnnlGemmWrapper::gemm\(", b)]
    if len(gemms) != 1:
        errs.append("the route op does not call gemm() exactly once (%d)" % len(gemms))
    if re.search(r"\bDnnlGemmWrapper::\w+\(", b.replace("DnnlGemmWrapper::gemm(", "").replace("DnnlGemmWrapper::to_dt<", "")):
        errs.append("the route op calls another DnnlGemmWrapper entry")
    if re.search(r"\bfor\(|\bwhile\(|\bdo\{", b):
        errs.append("the route op loops")
    if re.search(r"\bthrow\b|\bGGML_ABORT\b|\babort\(", b):
        errs.append("the route op throws or aborts")
    if _ALLOC.search(b):
        errs.append("the route op allocates: %r" % _ALLOC.search(b).group(0))
    if _WAIT.search(b):
        errs.append("the route op waits: %r" % _WAIT.search(b).group(0))
    if re.search(r"\bstatic\b", b):
        errs.append("the route op keeps a static")
    if _STORE.search(b):
        errs.append("the route op stores through a member or pointer: %r" % _STORE.search(b).group(0))
    if gemms:
        # the one result is held, and its has_value() is what the op returns: nullopt (a declined scratchpad) is false
        m = re.search(r"(?:const)?std::optional<sycl::event>(\w+)=DnnlGemmWrapper::gemm\(", b)
        if not m:
            errs.append("the gemm() result is not held in a std::optional<sycl::event>")
        else:
            returns = re.findall(r"\breturn\b[^;]*;", b)
            dnnl_returns = [r for r in returns if r != "return false;"]
            if dnnl_returns != ["return " + m.group(1) + ".has_value();"]:
                errs.append("the op does not return exactly the gemm() result's has_value(): %r" % returns)
            # its batch counts are one value on both sides
            args = call_args(b, gemms[0] + len("DnnlGemmWrapper::gemm"))
            if len(args) < 19 or args[17] != args[18]:
                errs.append("gemm()'s two batch counts differ: %r" % args[17:19])
    return errs


def test_the_route_op_makes_one_submission_and_keeps_nothing():
    assert route_op_errors(code_of(SYCL_CPP)) == []


def test_route_op_mutants():
    code = code_of(SYCL_CPP)
    full = function_body(code, _ROUTE_OP)
    m = re.search(r"(?:const)?std::optional<sycl::event>(\w+)=DnnlGemmWrapper::gemm\(", full)
    var = m.group(1)
    ret = "return " + var + ".has_value();"
    held = full[m.start() : full.index(";", m.start()) + 1]
    for name, new in [
        ("a decline throws", mutate(full, ret, "if(!" + var + "){throw std::runtime_error(\"declined\");}" + ret)),
        ("a decline aborts", mutate(full, ret, "if(!" + var + "){GGML_ABORT(\"declined\");}" + ret)),
        ("a decline reads as done", mutate(full, ret, "return true;")),
        ("the result is negated", mutate(full, ret, "return !" + var + ".has_value();")),
        ("one gemm per plane", mutate(full, held, "for(int64_t p=0;p<ne12;++p){" + held + "}")),
        ("a second gemm", mutate(full, held, held + held.replace(var, var + "_2", 1))),
        ("a unified allocation", mutate(full, ret, "ggml_sycl::unified_alloc(ctx.device,1);" + ret)),
        ("a pool allocation", mutate(full, ret, "ggml_sycl_pool_alloc<float>tmp(ctx.pool(),1);" + ret)),
        ("a handle kept", mutate(full, ret, "ggml_sycl::mem_handle keep;" + ret)),
        ("a host wait", mutate(full, ret, "stream->wait();" + ret)),
        ("a wait on the event", mutate(full, ret, var + "->wait_and_throw();" + ret)),
        ("a stored pointer", mutate(full, ret, "dst->extra=nullptr;" + ret)),
        ("a stored pointer in the context", mutate(full, ret, "ctx.last_ptr=nullptr;" + ret)),
        ("a static", mutate(full, ret, "static const void*last=nullptr;" + ret)),
    ]:
        assert route_op_errors(code.replace(full, new, 1)), f"mutant {name!r} slipped through"


# ---- the environment is filled from the backend's own facts ----
def env_errors(code: str) -> list:
    errs = []
    b = function_body(code, _ENV_OF)
    for field, source in (
        ("split", r"route_env\.split"),
        ("has_weight", r"route_env\.has_weight"),
        ("tensor_parallel", r"g_sycl_tp_config\.enabled"),
        ("recording", r"ggml_sycl_graph_recording_active\(\)\|\|[^;]*ext_oneapi_get_state\(\)==sycl::ext::oneapi::experimental::queue_state::recording"),
        ("on_device", r"on_device"),
    ):
        found = re.findall(r"\benv\." + field + r"=([^;]*);", b)
        if len(found) != 1 or not re.fullmatch(source, found[0]):
            errs.append("env.%s is not filled once from %s: %r" % (field, source, found))
    if code.count("ggml_sycl_mul_mat_f32_batched_env_of(") != 2:  # the definition and the dispatch
        errs.append("the environment is not built in exactly one place")
    return errs


def test_the_environment_is_filled_from_the_backends_facts():
    assert env_errors(code_of(SYCL_CPP)) == []


def test_env_mutants():
    code = code_of(SYCL_CPP)
    b = function_body(code, _ENV_OF)
    rec = re.search(r"\benv\.recording=[^;]*;", b).group(0)
    tp = re.search(r"\benv\.tensor_parallel=[^;]*;", b).group(0)
    for name, new in [
        ("recording never set", mutate(b, rec, "env.recording=false;")),
        ("recording reads only the backend flag", mutate(b, rec, "env.recording=ggml_sycl_graph_recording_active();")),
        ("tensor parallelism never set", mutate(b, tp, "env.tensor_parallel=false;")),
        ("split from elsewhere", mutate(b, "env.split=route_env.split;", "env.split=false;")),
        ("weight from elsewhere", mutate(b, "env.has_weight=route_env.has_weight;", "env.has_weight=false;")),
        ("on_device forced", mutate(b, "env.on_device=on_device;", "env.on_device=true;")),
    ]:
        assert env_errors(code.replace(b, new, 1)), f"mutant {name!r} slipped through"


# ---- the predicate lives beside the f16 one ----
def test_the_predicate_is_defined_once_in_nonfa_stage():
    hdr = code_of(STAGE_HPP)
    assert hdr.count(z(_ROUTE_OF) + "{") == 1
    assert z(_ROUTE_OF) not in code_of(SYCL_CPP)


# ---- gemm() labels by batch count ----
def gemm_body(gemm_code: str) -> str:
    start = gemm_code.index("static std::optional<sycl::event>gemm(")
    return gemm_code[start : gemm_code.index("static std::optional<sycl::event>row_gemm(", start)]


_LABEL = re.compile(r"const char\*(\w+)=([^;?]*)\?\"mulmat\.onednn_gemm\.batched\":\"mulmat\.onednn_gemm\.unbatched\";")


def label_errors(gemm_code: str) -> list:
    errs = []
    b = gemm_body(gemm_code)
    m = _LABEL.search(b)
    if not m or len(_LABEL.findall(b)) != 1:
        return ["gemm() does not choose one label name by a condition, batched first"]
    var, cond = m.group(1), m.group(2)
    if not re.fullmatch(r"\(?std::max\(batches_a,batches_b\)>1\)?|\(?batches_a>1\|\|batches_b>1\)?", cond):
        errs.append("the label is not chosen by a batch count above 1: %r" % cond)
    names = re.findall(r"\bgemm_label\.name=([^;]*);", b)
    if names != [var, var]:
        errs.append("gemm()'s two labels do not both take the chosen name: %r" % names)
    return errs


def test_gemm_labels_by_batch_count():
    assert label_errors(code_of(GEMM_HPP)) == []


def test_label_mutants():
    code = code_of(GEMM_HPP)
    m = _LABEL.search(gemm_body(code))
    decl, var, cond = m.group(0), m.group(1), m.group(2)
    swapped = decl.replace('"mulmat.onednn_gemm.batched":"mulmat.onednn_gemm.unbatched"',
                           '"mulmat.onednn_gemm.unbatched":"mulmat.onednn_gemm.batched"')
    for name, new in [
        ("the rows are swapped", code.replace(decl, swapped, 1)),
        ("only batches_a is read", code.replace(decl, decl.replace(cond, "batches_a>1"), 1)),
        ("a batch of 1 reads as batched", code.replace(decl, decl.replace(cond, cond.replace(">1", ">0")), 1)),
        ("one label keeps the old literal", code.replace("gemm_label.name=" + var + ";", 'gemm_label.name="mulmat.onednn_gemm.unbatched";', 1)),
        ("the cached label keeps the old literal",
         code[: code.rindex("gemm_label.name=" + var + ";")] + 'gemm_label.name="mulmat.onednn_gemm.unbatched";'
         + code[code.rindex("gemm_label.name=" + var + ";") + len("gemm_label.name=" + var + ";") :]),
    ]:
        assert label_errors(new), f"mutant {name!r} slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
