"""Source gate for the batched F32 MUL_MAT route: one oneDNN matmul for a node whose batch the per-plane loop of
ggml_sycl_op_mul_mat would run as one GEMM per plane.

Which nodes the route takes is the host test's (test-sycl-mul-mat-f32-batched-route); what it computes is the device
test's (test-sycl-mul-mat-f32-batched-device). This gate pins the form those two cannot see, on comment-stripped,
whitespace-canonical text:

- the dispatch consults the route after the f16 chain and before the orchestrator, and reads no environment for a
  node the tensors decline: in ggml_sycl_mul_mat the tensor predicate ggml_sycl_mul_mat_f32_batched_route_of, the
  environment predicate ggml_sycl_mul_mat_f32_batched_env_admits and the environment's builder are each evaluated
  once, all inside the condition of one `else if` of the chain that tests the f16 route, which declares the tensor
  route and then reads `route != NONE && environment admits && op(..., route, ...)` in that order (so the queue
  state is read only for a node the tensors admit, and nothing of the route runs for a node an f16 branch takes);
  its `else` is the orchestrator's, reached with select() when the route declines; the environment is built from
  the function's own route environment and the conjunction of exactly the three operands' on-device results; the op
  receives the three resolved operands whose on_device made that gate; a taken node only returns (the block after the
  chain is TP debug and TP post-processing, neither of which applies to a node the route takes);
- a decline falls through and does not throw: the route op returns the has_value() of the one gemm() result it
  holds, and its body has no throw and no abort, so a declined scratchpad (nullopt) is false and the dispatch goes
  on to the orchestrator; a pointer that is null is a decline before gemm();
- the route op uses the dispatch's resolution and nothing else: its three pointers are the .ptr of the resolved
  operands it receives, A from src0's, B from src1's and C from dst's; it calls no resolver of its own; and before
  gemm() it chains each operand's ready event onto the queue gemm() runs on, as the per-plane loop chains src0's;
- the environment the predicate reads is filled from the backend's own facts: the recording flag from
  ggml_sycl_graph_recording_active() and the queue's state, tensor parallelism from g_sycl_tp_config, split and
  weight from the route environment the f16 chain reads;
- the route body makes one submission and keeps nothing: exactly one DnnlGemmWrapper::gemm call and no other
  submission or USM call (memset, memcpy, fill, copy, submit, parallel_for, single_task, std::mem*), no loop, no
  allocation of any kind, no wait, no static, and no store through a member, a dereference or an index;
- gemm() chooses its profiler label by batch count: both of its labels name one variable that is the batched row
  when a batch count is above 1 and the unbatched row otherwise;
- the route computes in F32: gemm() takes its float math as a ggml_sycl_dnnl_fpmath parameter whose default,
  GGML_SYCL_DNNL_FPMATH_BUILD, is the build's choice (fpmath f16 under GGML_SYCL_F16, otherwise nothing set); only
  GGML_SYCL_DNNL_FPMATH_STRICT sets fpmath strict; the primitive cache key carries the mode in a field its equality and its
  hash both read, so a strict and an f16 primitive of one shape are two entries; the route op passes STRICT, and no
  other caller does.

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
    "const ggml_tensor * src1, ggml_tensor * dst, const ggml_sycl::resolved_ptr & src0_r, "
    "const ggml_sycl::resolved_ptr & src1_r, const ggml_sycl::resolved_ptr & dst_r, "
    "ggml_sycl_mul_mat_f32_batched_route route)"
)
_ENV_OF = (
    "static ggml_sycl_mul_mat_f32_batched_env ggml_sycl_mul_mat_f32_batched_env_of(ggml_backend_sycl_context & ctx, "
    "const ggml_sycl_mul_mat_route_env & route_env, bool on_device)"
)
_ROUTE_OF = (
    "inline ggml_sycl_mul_mat_f32_batched_route ggml_sycl_mul_mat_f32_batched_route_of(const ggml_tensor * src0, "
    "const ggml_tensor * src1, const ggml_tensor * dst)"
)
_ENV_ADMITS_DEF = (
    "inline bool ggml_sycl_mul_mat_f32_batched_env_admits(const ggml_sycl_mul_mat_f32_batched_env & env)"
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
        if ch == ">" and code[j - 1] == "-":
            continue  # `->` is member access, not a closing template bracket
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(code[start:j])
            start = j + 1
    args.append(code[start:end])
    return args


# ---- the dispatch consults the route after the f16 chain and before the orchestrator; a decline falls through ----
_ENV_ADMITS = "ggml_sycl_mul_mat_f32_batched_env_admits("
_ENV_OF_CALL = "ggml_sycl_mul_mat_f32_batched_env_of("
_ROUTE_OF_CALL = "ggml_sycl_mul_mat_f32_batched_route_of("


def split_top(text: str, sep: str) -> list:
    """`text` split at each `sep` outside brackets."""
    parts, depth, start, j = [], 0, 0, 0
    while j < len(text):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(sep, j):
            parts.append(text[start:j])
            j += len(sep)
            start = j
            continue
        j += 1
    parts.append(text[start:])
    return parts


def _chain_reaches(b: str, chain_if: int, head: int) -> bool:
    """Whether the if-else chain whose `if(` is at chain_if has the `}else if(` at head as one of its links."""
    pos = chain_if
    while True:
        cond_close = balanced_end(b, pos + len("if"), "(", ")")
        if b[cond_close + 1] != "{":
            return False
        block_close = balanced_end(b, cond_close + 1, "{", "}")
        if block_close == head:
            return True
        if not b.startswith("else if(", block_close + 1):
            return False
        pos = block_close + 1 + len("else ")


def gate_resolutions(b: str) -> dict:
    """{tensor: (resolved name, on-device name)} for the dispatch's gate: each `const bool X = R && R.on_device;` whose
    R is the dispatch's `ggml_sycl_resolve(tensor, ctx.device)` of src0, src1 or dst."""
    out = {}
    for o in re.finditer(r"\bconst bool (\w+)=(\w+)&&\2\.on_device;", b):
        r = re.search(r"\bconst auto " + o.group(2) + r"=ggml_sycl_resolve\((src0|src1|dst),ctx\.device\);", b)
        if r:
            out.setdefault(r.group(1), []).append((o.group(2), o.group(1)))
    return {t: v[0] for t, v in out.items() if len(v) == 1}


def gate_input_errors(b: str, env_term: str, op: str) -> list:
    """The environment's inputs and the op's resolved operands are the dispatch's own."""
    errs = []
    gate = gate_resolutions(b)
    if sorted(gate) != ["dst", "src0", "src1"]:
        return ["the dispatch does not resolve src0, src1 and dst once each with an on-device result"]
    route_env = re.search(r"\bconst ggml_sycl_mul_mat_route_env (\w+)=ggml_sycl_mul_mat_route_env_of\(src0,src1\);", b)
    at = env_term.find(_ENV_OF_CALL)
    args = call_args(env_term, at + len(_ENV_OF_CALL) - 1) if at != -1 else []
    if len(args) != 3 or args[0] != "ctx" or not route_env or args[1] != route_env.group(1):
        errs.append("the environment is not built from ctx and the function's own route environment: %r" % args)
    if len(args) == 3 and sorted(split_top(args[2], "&&")) != sorted(o for _, o in gate.values()):
        errs.append("the environment's on_device is not the conjunction of the three operands' on-device results: %r"
                    % args[2])
    # the op's resolved operands are the ones whose on_device made the gate, in src0, src1, dst order
    op_args = call_args(op, len("ggml_sycl_mul_mat_f32_batched"))
    want = [gate[t][0] for t in ("src0", "src1", "dst")]
    if [a for a in op_args if a in want] != want:
        errs.append("the op does not receive the gate's resolved src0, src1 and dst, in that order: %r" % op_args)
    return errs


def dispatch_errors(code: str) -> list:
    errs = []
    b = function_body(code, _MUL_MAT)
    for call, what in ((_ROUTE_OF_CALL, "the tensor predicate"), (_ENV_ADMITS, "the environment predicate"),
                       (_ENV_OF_CALL, "the environment")):
        if b.count(call) != 1:
            errs.append("%s is not asked exactly once (%d)" % (what, b.count(call)))
    if "ext_oneapi_get_state(" in b:
        errs.append("the dispatch reads the queue state itself")
    calls = [m.start() for m in re.finditer(r"\bggml_sycl_mul_mat_f32_batched\(", b)]
    if len(calls) != 1:
        return errs + ["the route op is not called exactly once (%d)" % len(calls)]
    call = calls[0]
    # the route is a link of the f16 chain: the chain's first if tests the f16 route, and every f16 branch comes first
    f16 = re.search(r"\bggml_sycl_mul_mat_f16_route (\w+)=ggml_sycl_mul_mat_f16_route_of\([^;]*\);", b)
    head = b.rfind("}else if(", 0, call)
    if not f16 or head == -1:
        return errs + ["the route op is not in an else-if of the f16 route's chain"]
    chain_if = b.find("if(" + f16.group(1) + "==", f16.end())
    if chain_if == -1 or not _chain_reaches(b, chain_if, head):
        errs.append("the route's else-if is not a link of the chain that tests the f16 route")
    cond_open = head + len("}else if")
    cond_close = balanced_end(b, cond_open, "(", ")")
    if not (cond_open < call < cond_close):
        return errs + ["the route op is not in the else-if's condition"]
    cond = b[cond_open + 1 : cond_close]
    # nothing of the route is evaluated outside that condition: not ahead of the f16 chain, not before its own turn
    for c in (_ROUTE_OF_CALL, _ENV_ADMITS, _ENV_OF_CALL):
        if b.count(c) != cond.count(c):
            errs.append("%s) is evaluated outside the route's else-if condition" % c[:-1])
    init_test = split_top(cond, ";")
    if len(init_test) != 2:
        return errs + ["the else-if does not declare the tensor route in its condition: %r" % cond[:160]]
    init = re.fullmatch(r"(?:const )?ggml_sycl_mul_mat_f32_batched_route (\w+)=" + re.escape(_ROUTE_OF_CALL) + r".*\)",
                        init_test[0])
    if not init:
        return errs + ["the else-if's declaration is not the tensor predicate's answer: %r" % init_test[0]]
    var = init.group(1)
    terms = split_top(init_test[1], "&&")
    # the tensors first, then the environment, then the op: no environment read before the tensors admit the node
    route_test = [i for i, t in enumerate(terms) if t in (var + "!=" + _NONE, _NONE + "!=" + var)]
    env_test = [i for i, t in enumerate(terms) if t.startswith(_ENV_ADMITS) and _ENV_OF_CALL in t]
    op = terms[-1]
    if not (op.startswith("ggml_sycl_mul_mat_f32_batched(") and op.endswith(")")
            and var in [a for a in call_args(op, len("ggml_sycl_mul_mat_f32_batched"))]):
        errs.append("the condition does not end in the route op called with the route: %r" % op)
    if len(terms) != 3 or route_test != [0] or env_test != [1]:
        errs.append("the condition is not `route != NONE && environment admits && op(route)`, in that order: %r"
                    % init_test[1])
    errs += gate_input_errors(b, terms[1], op)
    # a taken route did the op and only returns: the block after the chain is TP debug and TP post-processing, which a
    # node the route takes (no weight, no tensor parallelism) never needs; its `else` is the orchestrator's
    body_open = cond_close + 1
    if b[body_open] != "{":
        return errs + ["the else-if has no block"]
    body_close = balanced_end(b, body_open, "{", "}")
    if b[body_open + 1 : body_close] != "return;":
        errs.append("the taken branch does more or less than return: %r" % b[body_open + 1 : body_close][:120])
    if not b.startswith("}else{", body_close):
        errs.append("the branch after the route is not a plain else")
    else:
        else_close = balanced_end(b, body_close + len("}else"), "{", "}")
        if "ctx.matmul_orchestrator.select(" not in b[body_close:else_close]:
            errs.append("the declined route does not reach the orchestrator's select() in the else")
    return errs


def test_the_dispatch_consults_the_route_after_the_f16_chain_and_before_the_orchestrator():
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
    init, test = split_top(cond, ";")
    route_var = re.search(r"route (\w+)=", init).group(1)
    route_test, env_term, op = split_top(test, "&&")
    chain_if = b.index("if(f16_route==")
    env_of = env_term[len(_ENV_ADMITS) : -1]

    # the order of the route test's operands and of the three on-device results is spelling: they pass
    flipped = code.replace(b, mutate(b, route_test + "&&", _NONE + "!=" + route_var + "&&"), 1)
    assert dispatch_errors(flipped) == [], dispatch_errors(flipped)
    reordered = code.replace(b, mutate(b, env_term, env_term.replace(
        "src0_on_device_r&&src1_on_device_r&&dst_on_device_r", "dst_on_device_r&&src0_on_device_r&&src1_on_device_r")), 1)
    assert dispatch_errors(reordered) == [], dispatch_errors(reordered)

    for name, new_b in [
        ("the route branch is gone", b.replace(branch, "}", 1)),
        ("the route is asked after the orchestrator",
         b.replace(branch, "}", 1).replace("ctx.matmul_orchestrator.select(",
                                           "if(" + cond + "){return;}ctx.matmul_orchestrator.select(", 1)),
        ("a decline returns instead of falling through", mutate(b, branch, branch + "else if(true){return;}")),
        ("the op runs whatever the tensors said", mutate(b, test, env_term + "&&" + op)),
        ("the op runs only for a NONE route", mutate(b, route_test + "&&", route_var + "==" + _NONE + "&&")),
        ("the op runs with another route",
         mutate(b, op, op.replace("," + route_var + ")", ",GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_BATCHED)"))),
        ("a taken route also runs the orchestrator",
         mutate(b, branch, branch[:-1] + "ctx.matmul_orchestrator.select(src0,src1,dst,forced_layout);}")),
        ("the route is called a second time",
         mutate(b, branch, branch + "else if(ggml_sycl_mul_mat_f32_batched(ctx,src0,src1,dst,"
                + "GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_BATCHED)){}")),
        ("the predicate is asked twice", b[:-1] + z("(void) ggml_sycl_mul_mat_f32_batched_route_of(src0, src1, dst);") + "}"),
        ("the tensor route is hoisted ahead of the f16 chain",
         b[:chain_if] + init + ";" + b[chain_if:].replace(cond, test, 1)),
        ("the environment is read before the tensors admit the node",
         mutate(b, test, env_term + "&&" + route_test + "&&" + op)),
        ("the environment is built ahead of the f16 chain",
         b[:chain_if] + "const auto f32_env=" + env_of + ";" + b[chain_if:].replace(env_of, "f32_env", 1)),
        ("the environment is not asked", mutate(b, test, route_test + "&&" + op)),
        ("the dispatch reads the queue state",
         b[:chain_if] + "(void)ctx.stream()->ext_oneapi_get_state();" + b[chain_if:]),
        ("a taken route falls through to the post-chain block", mutate(b, branch, branch.replace("{return;}", "{}"))),
        ("the environment is told every operand is on the device",
         mutate(b, env_term, env_term.replace("src0_on_device_r&&src1_on_device_r&&dst_on_device_r", "true"))),
        ("the environment reads one operand's on-device result",
         mutate(b, env_term, env_term.replace("src0_on_device_r&&src1_on_device_r&&dst_on_device_r", "src1_on_device_r"))),
        ("the environment gets a default route environment",
         mutate(b, env_term, env_term.replace(",route_env,", ",ggml_sycl_mul_mat_route_env{},"))),
        ("the op gets a fresh resolution of src1",
         mutate(b, op, op.replace(",src1_resolved,", ",ggml_sycl_resolve(src1,ctx.device),"))),
        ("the op gets src0's and src1's resolutions swapped",
         mutate(b, op, op.replace("src0_resolved,src1_resolved", "src1_resolved,src0_resolved"))),
        ("a declined route does not reach the orchestrator's select",
         b.replace("ctx.matmul_orchestrator.select(src0,src1,dst,forced_layout)",
                   "ctx.matmul_orchestrator.pick(src0,src1,dst,forced_layout)", 1)),
    ]:
        assert new_b != b, f"mutant {name!r} changed nothing"
        assert dispatch_errors(code.replace(b, new_b, 1)), f"mutant {name!r} slipped through"


# ---- the route op: one submission, a decline is false, nothing kept ----
_ALLOC = re.compile(r"\bunified_alloc\w*\(|\bunified_allocate\w*\(|\bggml_sycl_pool_alloc\b|\bmalloc\w*\(|\.alloc\(|"
                    r"\bsycl::malloc\w*|\bmem_handle\b|\balloc_owner\b|\bfree\(")
_WAIT = re.compile(r"\bwait(?:_and_throw)?\(|\bext_oneapi_submit_barrier\(|\bsynchronize\(")
# a store through a member (`p->x =`, `s.x =`), a dereference (`*p =`) or an index (`p[i] =`); declarations such as
# `float * c = ...` are not stores, since a statement's first character is never `*` in them
_STORE = re.compile(r"(?:->|\.)\w+(?:\[[^\]]*\])*=(?!=)|[;{}]\*[^;=]*=(?!=)|[;{}][\w.>-]+\[[^\]]*\]=(?!=)")
# a submission or USM call other than the one gemm()
_SUBMIT = re.compile(r"(?:->|\.)(?:memset|memcpy|fill|copy|submit|parallel_for|single_task|prefetch|mem_advise)\(|"
                     r"\bstd::mem\w+\(|(?<![\w:])(?:memset|memcpy|memmove)\(")
_RESOLVERS = re.compile(r"\bggml_sycl_resolve\w*\(|\bggml_sycl_get_data_ptr\w*\(|\bggml_sycl_get_weight\w*ptr\w*\(")
_RESOLVED_PARAM = re.compile(r"const ggml_sycl::resolved_ptr&(\w+)")


_NULL_TERM = re.compile(r"!(\w+)|(\w+)==nullptr|nullptr==(\w+)")


def _pointers(b: str) -> list:
    """The pointer variables the route op takes from a resolved operand's .ptr, as (name, operand) matches."""
    return [m for m in re.finditer(r"\b(\w+)=(?:static_cast<[^>]*>\()?(\w+)\.ptr\)?;", b)]


def null_check_errors(b: str, gemm_at: int) -> list:
    """A null pointer is a decline: after the three pointers are taken and before gemm(), one `if` whose condition
    tests each of them for null returns false (the orchestrator resolves and reports on its own path)."""
    resolved = _pointers(b)
    if len(resolved) != 3:
        return ["the route op does not take exactly three pointers from resolved operands (%d)" % len(resolved)]
    names = {m.group(1) for m in resolved}
    after = resolved[-1].end()  # the match ends with the last pointer's `;`
    for m in re.finditer(r"\bif\(", b):
        cond_close = balanced_end(b, m.end() - 1, "(", ")")
        tested = [_NULL_TERM.fullmatch(t) for t in split_top(b[m.end() : cond_close], "||")]
        if not all(tested) or {next(g for g in t.groups() if g) for t in tested} != names:
            continue
        if not (after <= m.start() < gemm_at):
            return ["the null check is not between the pointers' resolution and gemm()"]
        if b[cond_close + 1] != "{":
            return ["the null check has no block"]
        block = b[cond_close + 2 : balanced_end(b, cond_close + 1, "{", "}")]
        if not re.fullmatch(r"(?:GGML_SYCL_DEBUG\([^;]*\);)?return false;", block):
            return ["a null pointer is not a plain decline: %r" % block]
        return []
    return ["no check declines when one of %s resolves to null" % sorted(names)]


def resolution_errors(full: str, b: str, gemm_at: int) -> list:
    """The pointers are the dispatch's: A, B and C are the .ptr of the src0, src1 and dst operands the op receives, no
    resolver runs here, and each operand's ready event is chained onto gemm()'s queue before gemm()."""
    errs = []
    params = _RESOLVED_PARAM.findall(full[: full.index("{")])
    if len(params) != 3:
        return ["the route op does not receive three resolved operands: %r" % params]
    if _RESOLVERS.search(b):
        errs.append("the route op resolves a tensor itself: %r" % _RESOLVERS.search(b).group(0))
    args = call_args(b, gemm_at + len("DnnlGemmWrapper::gemm"))
    taken = {m.group(1): m.group(2) for m in _pointers(b)}
    if len(args) < 17 or [taken.get(args[i]) for i in (4, 9, 14)] != params:
        errs.append("gemm()'s A, B and C are not the .ptr of the src0, src1 and dst operands: %r from %r"
                    % ([args[i] for i in (4, 9, 14)] if len(args) > 14 else args, taken))
        return errs
    chained = []
    for m in re.finditer(r"\bggml_sycl_chain_ready_event_if_needed\(", b):
        c = call_args(b, m.end() - 1)
        if m.start() < gemm_at and len(c) >= 2 and c[0] == "*" + args[16]:
            chained.append(c[1])
    if sorted(chained) != sorted(params):
        errs.append("before gemm() the ready events chained onto its queue are %r, not each operand's once" % chained)
    return errs


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
        errs.append("the route op stores through a member, a dereference or an index: %r" % _STORE.search(b).group(0))
    if _SUBMIT.search(b):
        errs.append("the route op submits or calls USM besides gemm(): %r" % _SUBMIT.search(b).group(0))
    if gemms:
        errs += null_check_errors(b, gemms[0])
        errs += resolution_errors(full, b, gemms[0])
        # has_value() is the semantic, not a spelling: true only when gemm() produced an event, nullopt (a declined
        # scratchpad) being false
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
        ("a store through a dereference", mutate(full, ret, "*c=0.0f;" + ret)),
        ("a store through an index", mutate(full, ret, "c[0]=0.0f;" + ret)),
        ("a memset", mutate(full, ret, "stream->memset(c,0,4);" + ret)),
        ("a memcpy", mutate(full, ret, "stream->memcpy(c,a,4);" + ret)),
        ("a submit", mutate(full, ret, "stream->submit([&](sycl::handler&h){});" + ret)),
        ("a parallel_for", mutate(full, ret, "stream->parallel_for(sycl::range<1>(1),[=](sycl::id<1>){});" + ret)),
        ("a std::memset", mutate(full, ret, "std::memset(nullptr,0,0);" + ret)),
    ]:
        assert route_op_errors(code.replace(full, new, 1)), f"mutant {name!r} slipped through"


def test_resolution_mutants():
    code = code_of(SYCL_CPP)
    full = function_body(code, _ROUTE_OP)
    b = full[full.index("{") :]
    ptrs = {m.group(2): m.group(0) for m in _pointers(b)}
    chains = re.findall(r"ggml_sycl_chain_ready_event_if_needed\([^;]*\);", b)
    held = re.search(r"(?:const)?std::optional<sycl::event>\w+=DnnlGemmWrapper::gemm\(", full)
    gemm_stmt = full[held.start() : full.index(";", held.start()) + 1]
    src1_chain = [c for c in chains if ",src1_r" in c][0]
    for name, new in [
        ("B from a second resolver",
         mutate(full, ptrs["src1_r"], ptrs["src1_r"].replace("src1_r.ptr", "ggml_sycl_resolve_tensor_ptr(src1,ctx.device)"))),
        ("B from get_data_ptr", mutate(full, ptrs["src1_r"], ptrs["src1_r"].replace("src1_r.ptr", "ggml_sycl_get_data_ptr(src1,ctx.device)"))),
        ("A from src1's operand", mutate(full, ptrs["src0_r"], ptrs["src0_r"].replace("src0_r.ptr", "src1_r.ptr"))),
        ("src1's ready event not chained", mutate(full, src1_chain, "")),
        ("src1's ready event chained twice instead of dst's",
         mutate(full, [c for c in chains if ",dst_r" in c][0], src1_chain)),
        ("a ready event chained onto another queue",
         mutate(full, src1_chain, src1_chain.replace("(*stream,", "(*ctx.stream(ctx.device,1),"))),
        ("the ready events chained after gemm()",
         full.replace("".join(chains), "", 1).replace(gemm_stmt, gemm_stmt + "".join(chains), 1)),
    ]:
        assert new != full, f"mutant {name!r} changed nothing"
        assert route_op_errors(code.replace(full, new, 1)), f"mutant {name!r} slipped through"


def test_null_pointer_mutants():
    code = code_of(SYCL_CPP)
    full = function_body(code, _ROUTE_OP)
    check = re.search(r"if\(!\w+\|\|!\w+\|\|!\w+\)\{[^}]*\}", full).group(0)
    names = re.findall(r"!(\w+)", check)
    held = re.search(r"(?:const)?std::optional<sycl::event>\w+=DnnlGemmWrapper::gemm\(", full)
    gemm_stmt = full[held.start() : full.index(";", held.start()) + 1]
    # the spelling of a null test and the order of the three are not the form: they pass
    respelled = check.replace("!" + names[0], names[0] + "==nullptr", 1).replace(
        "!" + names[1] + "||!" + names[2], "!" + names[2] + "||!" + names[1], 1)
    assert route_op_errors(code.replace(full, mutate(full, check, respelled), 1)) == []
    for name, new in [
        ("no null check", mutate(full, check, "")),
        ("one pointer unchecked", mutate(full, check, check.replace("||!" + names[2], "", 1))),
        ("a null pointer reads as done", mutate(full, check, check.replace("return false;", "return true;"))),
        ("the check comes after gemm()", full.replace(check, "", 1).replace(gemm_stmt, gemm_stmt + check, 1)),
    ]:
        assert new != full, f"mutant {name!r} changed nothing"
        assert route_op_errors(code.replace(full, new, 1)), f"mutant {name!r} slipped through"


# ---- the environment is filled from the backend's own facts ----
_BACKEND_REC = r"ggml_sycl_graph_recording_active\(\)"
_QUEUE_REC = r"[^;|]*ext_oneapi_get_state\(\)==sycl::ext::oneapi::experimental::queue_state::recording"
_RECORDING = "(?:" + _BACKEND_REC + r"\|\|" + _QUEUE_REC + "|" + _QUEUE_REC + r"\|\|" + _BACKEND_REC + ")"
def env_errors(code: str) -> list:
    errs = []
    b = function_body(code, _ENV_OF)
    for field, source in (
        ("split", r"route_env\.split"),
        ("has_weight", r"route_env\.has_weight"),
        ("tensor_parallel", r"g_sycl_tp_config\.enabled"),
        ("recording", _RECORDING),
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
    # the order of the two recording terms is spelling: it passes
    terms = rec[len("env.recording=") : -1].split("||")
    swapped = mutate(b, rec, "env.recording=" + terms[1] + "||" + terms[0] + ";")
    assert env_errors(code.replace(b, swapped, 1)) == []
    for name, new in [
        ("recording never set", mutate(b, rec, "env.recording=false;")),
        ("recording reads only the backend flag", mutate(b, rec, "env.recording=ggml_sycl_graph_recording_active();")),
        ("recording reads only the queue state", mutate(b, rec, rec.replace("ggml_sycl_graph_recording_active()||", ""))),
        ("tensor parallelism never set", mutate(b, tp, "env.tensor_parallel=false;")),
        ("split from elsewhere", mutate(b, "env.split=route_env.split;", "env.split=false;")),
        ("weight from elsewhere", mutate(b, "env.has_weight=route_env.has_weight;", "env.has_weight=false;")),
        ("on_device forced", mutate(b, "env.on_device=on_device;", "env.on_device=true;")),
    ]:
        assert env_errors(code.replace(b, new, 1)), f"mutant {name!r} slipped through"


# ---- the predicates live beside the f16 one; the tensor part reads no environment ----
def test_the_predicates_are_defined_once_in_nonfa_stage():
    hdr = code_of(STAGE_HPP)
    for sig in (_ROUTE_OF, _ENV_ADMITS_DEF):
        assert hdr.count(z(sig) + "{") == 1, sig
        assert z(sig) not in code_of(SYCL_CPP), sig
    assert "env" not in function_body(hdr, _ROUTE_OF), "the tensor predicate reads an environment"


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


# ---- the route computes in F32; every other caller keeps the build's float math ----
_STRICT = "GGML_SYCL_DNNL_FPMATH_STRICT"
_BUILD = "GGML_SYCL_DNNL_FPMATH_BUILD"
_FPMATH_PARAM = re.compile(r"ggml_sycl_dnnl_fpmath (\w+)=(\w+)")


def _between(code: str, start: str, end: str) -> str:
    i = code.index(start)
    return code[i : code.index(end, i)]


def fpmath_errors(gemm_code: str, sycl_code: str, others: dict) -> list:
    """`others` maps each other ggml-sycl source that names the strict mode to its z-spelled code."""
    errs = []
    gb = gemm_body(gemm_code)
    params = call_args(gb, gb.index("gemm(") + len("gemm"))
    found = [(i, _FPMATH_PARAM.fullmatch(p)) for i, p in enumerate(params) if _FPMATH_PARAM.fullmatch(p)]
    if len(found) != 1:
        return ["gemm() does not take one ggml_sycl_dnnl_fpmath parameter with a default (%d)" % len(found)]
    index, m = found[0]
    param = m.group(1)
    if m.group(2) != _BUILD:
        errs.append("gemm()'s float-math default is not the build's: %r" % m.group(2))

    # the primitive's attribute: strict for STRICT, and for anything else exactly the build's choice
    sets = re.findall(r"(\w+)\.set_fpmath_mode\(dnnl::fpmath_mode::(\w+)\);", gb)
    if sorted(mode for _, mode in sets) != ["f16", "strict"] or len({attr for attr, _ in sets}) != 1:
        errs.append("gemm() does not set exactly one strict and one f16 float math on one attribute: %r" % sets)
    else:
        attr = sets[0][0]
        if not re.search(r"\bget_or_create\([^;]*," + attr + r"\);", gb):
            errs.append("the attribute holding the float math is not the one the cache creates the primitive with")
        cond = re.search(r"\bif\((?:" + param + "==" + _STRICT + "|" + _STRICT + "==" + param + r")\)\{", gb)
        if not cond:
            errs.append("the strict float math is not chosen by the parameter being " + _STRICT)
        else:
            then_close = balanced_end(gb, cond.end() - 1, "{", "}")
            then = gb[cond.end() : then_close]
            if then != attr + ".set_fpmath_mode(dnnl::fpmath_mode::strict);":
                errs.append("the STRICT branch does more or less than set strict: %r" % then)
            if not gb.startswith("else{", then_close + 1):
                errs.append("the build's float math is not the STRICT branch's else")
            else:
                other = gb[then_close + 1 + len("else") : balanced_end(gb, then_close + 1 + len("else"), "{", "}") + 1]
                if not re.fullmatch(r"\{#(?:ifdef GGML_SYCL_F16|if defined\(GGML_SYCL_F16\)) " + attr
                                    + r"\.set_fpmath_mode\(dnnl::fpmath_mode::f16\);#endif\}", other):
                    errs.append("the else is not exactly f16 under GGML_SYCL_F16: %r" % other)

    # the cache key carries the mode: a strict and an f16 primitive of one shape are two entries
    key = _between(gemm_code, "struct DnnlPrimitiveKey{", "struct DnnlPrimitiveKeyHash{")
    fields = re.findall(r"\bggml_sycl_dnnl_fpmath (\w+);", key)
    if len(fields) != 1:
        errs.append("the primitive key has no one ggml_sycl_dnnl_fpmath field: %r" % fields)
    else:
        field = fields[0]
        eq = key[key.index("operator==(") :]
        if not re.search(r"(?:&&|return )" + field + "==other\\." + field + r"\b", eq):
            errs.append("the key's equality does not compare its float math")
        hashed = _between(gemm_code, "struct DnnlPrimitiveKeyHash{", "struct DnnlCachedPrimitive{")
        if not re.search(r"\bh\^=std::hash<\w+>\{\}\((?:static_cast<\w+>\()?k\." + field + r"\)?\)", hashed):
            errs.append("the key's hash does not read its float math")
        if re.findall(r"\bkey\." + field + r"=([^;]*);", gb) != [param]:
            errs.append("gemm() does not put its float-math parameter in its key")

    # the route op passes STRICT at that parameter's place, and nothing else names STRICT
    route = function_body(sycl_code, _ROUTE_OP)
    at = route.find("DnnlGemmWrapper::gemm(")
    args = call_args(route, at + len("DnnlGemmWrapper::gemm")) if at != -1 else []
    if len(args) <= index or args[index] != _STRICT:
        errs.append("the route op does not pass %s as gemm()'s float math: %r" % (_STRICT, args[index:]))
    if sycl_code.count(_STRICT) != route.count(_STRICT) or route.count(_STRICT) != 1:
        errs.append("ggml-sycl.cpp names %s outside the one route call" % _STRICT)
    if gemm_code.count(_STRICT) != 2:  # the enumerator and gemm()'s one test of it
        errs.append("gemm.hpp names %s %d times, not as the enumerator and its one test"
                    % (_STRICT, gemm_code.count(_STRICT)))
    for path, text in others.items():
        if _STRICT in text:
            errs.append("%s passes %s" % (path, _STRICT))
    return errs


def _others() -> dict:
    out = {}
    for p in sorted(SYCL.rglob("*")):
        if p.suffix in (".cpp", ".hpp", ".h") and p.name not in ("gemm.hpp", "ggml-sycl.cpp"):
            raw = p.read_text(errors="replace")
            if _STRICT in raw:
                out[str(p.relative_to(ROOT))] = code_of(raw)
    return out


def test_the_route_computes_in_f32_and_every_other_caller_keeps_the_builds_math():
    assert fpmath_errors(code_of(GEMM_HPP), code_of(SYCL_CPP), _others()) == []


def test_fpmath_mutants():
    g = code_of(GEMM_HPP)
    s = code_of(SYCL_CPP)
    gb = gemm_body(g)
    m = re.search(r"ggml_sycl_dnnl_fpmath (\w+)=" + _BUILD, gb)
    param = m.group(1)
    strict_set = re.search(r"(\w+)\.set_fpmath_mode\(dnnl::fpmath_mode::strict\);", gb).group(0)
    f16_set = re.search(r"(\w+)\.set_fpmath_mode\(dnnl::fpmath_mode::f16\);", gb).group(0)
    key = _between(g, "struct DnnlPrimitiveKey{", "struct DnnlPrimitiveKeyHash{")
    field = re.search(r"\bggml_sycl_dnnl_fpmath (\w+);", key).group(1)
    hashed = _between(g, "struct DnnlPrimitiveKeyHash{", "struct DnnlCachedPrimitive{")
    hash_line = re.search(r"h\^=std::hash<\w+>\{\}\((?:static_cast<\w+>\()?k\." + field + r"\)?\)<<\d+;", hashed).group(0)
    route = function_body(s, _ROUTE_OP)
    f16_route = "return DnnlGemmWrapper::gemm(ctx,a1,b1,a0,src0,"
    f16_args_end = balanced_end(s, s.index(f16_route) + len("return DnnlGemmWrapper::gemm"), "(", ")")
    row = g.index("static std::optional<sycl::event>row_gemm(")
    row_call_end = balanced_end(g, g.index("return gemm(", row) + len("return gemm"), "(", ")")

    def in_gemm(old: str, new: str) -> str:
        return g.replace(gb, mutate(gb, old, new), 1)

    gemm_mutants = [
        ("the default is strict", in_gemm(param + "=" + _BUILD, param + "=" + _STRICT)),
        ("the parameter is gone", g.replace(gb, gb.replace(",ggml_sycl_dnnl_fpmath " + param + "=" + _BUILD, ""), 1)),
        ("the STRICT branch sets f16", in_gemm(strict_set, strict_set.replace("::strict", "::f16"))),
        ("f16 also overrides strict", in_gemm(strict_set, strict_set + f16_set)),
        ("f16 without GGML_SYCL_F16", g.replace(gb, gb.replace("#ifdef GGML_SYCL_F16 ", "", 1).replace("#endif}", "}", 1), 1)),
        ("strict for the build default", g.replace(gb, gb.replace(param + "==" + _STRICT, param + "!=" + _STRICT, 1), 1)),
        ("the key ignores the parameter", in_gemm("key." + field + "=" + param + ";", "key." + field + "=" + _BUILD + ";")),
        ("equality forgets the mode", g.replace("&&" + field + "==other." + field, "", 1)),
        ("the hash forgets the mode", g.replace(hash_line, "", 1)),
        ("the key has no mode field", g.replace(key, key.replace("ggml_sycl_dnnl_fpmath " + field + ";", ""), 1)),
        ("row_gemm passes strict", g[:row_call_end] + "," + _STRICT + g[row_call_end:]),
    ]
    for name, new_g in gemm_mutants:
        assert new_g != g, f"mutant {name!r} changed nothing"
        assert fpmath_errors(new_g, s, {}), f"mutant {name!r} slipped through"

    route_call = route[route.index("DnnlGemmWrapper::gemm(") :]
    route_call = route_call[: balanced_end(route_call, len("DnnlGemmWrapper::gemm"), "(", ")") + 1]
    sycl_mutants = [
        ("the route passes the build's math", s.replace(route, mutate(route, route_call, route_call.replace(_STRICT, _BUILD)), 1)),
        ("the route omits the math", s.replace(route, mutate(route, route_call, route_call.replace("," + _STRICT, "")), 1)),
        ("the F16 batched route passes strict", s[:f16_args_end] + ",false," + _STRICT + s[f16_args_end:]),
    ]
    for name, new_s in sycl_mutants:
        assert new_s != s, f"mutant {name!r} changed nothing"
        assert fpmath_errors(g, new_s, {}), f"mutant {name!r} slipped through"
    assert fpmath_errors(g, s, {"ggml/src/ggml-sycl/outprod.cpp": "gemm(" + _STRICT + ")"}), \
        "mutant 'outprod passes strict' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
