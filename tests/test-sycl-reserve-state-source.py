"""Source gate for the sched_reserve split (zhcn design gate 6 and the type and
compute-print clauses of gate 37, as far as ALLOC-only code can satisfy them).

`sched_reserve()` is a thin wrapper over
`sched_reserve_result sched_reserve_impl(sched_reserve_mode, sched_reserve_state &)`.
The reserve state carries everything a reserve reads and writes about its
scheduler (the scheduler, both arenas of previous graph results, the
reserve-time n_outputs and n_input_tensors, and the cparams the graphs are
built with), so a later MEASURE can run on storage of its own without a single
write to the context's members. This gate pins, on comment-stripped text:

- the status type has no BUSY, the result type and the mode enum exist;
- `sched_reserve_impl`, the state overload of `graph_reserve` and
  `resolve_fused_ops` mention the scheduler, the graph-result arenas, cparams,
  n_input_tensors and `this->n_outputs` only through the state, and the impl
  throws nothing: its three refusals are FAILED returns;
- the member-backed state names exactly the context's own members;
- `sched_reserve()` runs the reserve transaction (ALLOC for an unplanned
  context) and throws the reason of any non-OK result, so its callers see
  today's exceptions;
- the compute line is printed in exactly one place, inside the impl, with the
  literal prefix "sched_reserve";
- decode and encode reserve through `sched_reserve_nothrow()`, never the
  throwing `sched_reserve()`: nothing above them catches, so the nothrow form
  catches std::exception, logs, returns false and leaves sched_need_reserve
  set so the next call starts over (gate 26).

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
CONTEXT_H = (ROOT / "src/llama-context.h").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def z(text: str) -> str:
    """Canonical spelling: whitespace runs collapse, then every space that is
    not between two identifier characters goes."""
    return re.sub(r"(?<!\w) | (?!\w)", "", re.sub(r"\s+", " ", text))


def code_of(raw: str) -> str:
    return z(strip_comments(raw))


def mutate(body: str, old: str, new: str) -> str:
    zo = z(old)
    assert body.count(zo) == 1, f"mutant anchor must match exactly once: {old!r} x{body.count(zo)}"
    return body.replace(zo, z(new), 1)


def function_body(code: str, signature: str) -> str:
    """The text of the function whose z-spelled definition starts with
    `signature`, through its matching closing brace."""
    start = code.find(z(signature) + "{")
    assert start != -1, f"{signature!r} not defined"
    i = code.index("{", start)
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[start : j + 1]
    raise AssertionError(f"unbalanced braces in {signature!r}")


_IMPL = "sched_reserve_result llama_context::sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state)"
_GRAPH_RESERVE = "ggml_cgraph * llama_context::graph_reserve(sched_reserve_state & state, uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes)"
_RESOLVE = "void llama_context::resolve_fused_ops(sched_reserve_state & state, const llama_memory_context_i * mctx, uint32_t n_seqs)"
_WRAPPER = "void llama_context::sched_reserve()"
_MEMBER_STATE = "sched_reserve_state llama_context::member_reserve_state()"

# Context members a MEASURE must not write, so none may be named bare.
_MEMBER_RE = re.compile(
    r"(?<![\w.>])(?<!::)(?:sched|gf_res_prev|gf_res_reserve|gf_res_prev_active|cparams|n_input_tensors)\b(?!\s*\()"
    r"|this->n_outputs|this->sched|this->cparams|this->gf_res"
)


def bodies(raw: str = CONTEXT_CPP):
    code = code_of(raw)
    return {
        "impl": function_body(code, _IMPL),
        "graph_reserve": function_body(code, _GRAPH_RESERVE),
        "resolve_fused_ops": function_body(code, _RESOLVE),
    }


_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


def member_hits(body: str):
    """Member names in the body proper, not in the parameter list or in a
    string literal (a log line may say "sched copies")."""
    return _MEMBER_RE.findall(_STRING_RE.sub('""', body[body.index("{") :]))


def pure_ok(bs: dict) -> bool:
    for name, b in bs.items():
        if member_hits(b):
            return False
    return "throw" not in bs["impl"]


def test_reserve_bodies_touch_only_the_state():
    bs = bodies()
    for name, b in bs.items():
        hits = member_hits(b)
        assert not hits, f"{name} names context members instead of the reserve state: {hits}"
    assert "throw" not in bs["impl"], "sched_reserve_impl must return its refusals, not throw them"
    assert pure_ok(bs)


def test_purity_mutants():
    bs = bodies()
    mutants = [
        ("impl writes the member scheduler", "impl", "state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, state.cparams.pipeline_parallel, state.cparams.op_offload));", "sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, state.cparams.pipeline_parallel, state.cparams.op_offload));"),
        ("impl writes member cparams on the PP retry", "impl", "state.cparams.pipeline_parallel = false;", "cparams.pipeline_parallel = false;"),
        ("impl reads member cparams", "impl", "const uint32_t n_seqs = state.cparams.n_seq_max;", "const uint32_t n_seqs = cparams.n_seq_max;"),
        ("impl clears the member arena", "impl", "for (auto & res : state.gf_res_prev) {", "for (auto & res : gf_res_prev) {"),
        ("impl throws a refusal", "impl", 'return { sched_reserve_status::FAILED, "failed to initialize memory module" };', 'throw std::runtime_error("failed to initialize memory module");'),
        ("graph_reserve resets the member scheduler", "graph_reserve", "ggml_backend_sched_reset(state.sched.get());", "ggml_backend_sched_reset(sched.get());"),
        ("graph_reserve writes this->n_outputs", "graph_reserve", "state.n_outputs = n_outputs;", "this->n_outputs = n_outputs;"),
        ("graph_reserve builds into the member result", "graph_reserve", "auto * res = state.gf_res_reserve.get();", "auto * res = gf_res_reserve.get();"),
        ("graph_reserve writes the member n_input_tensors", "graph_reserve", "state.n_input_tensors = llama_graph_n_input_tensors(gf, !measure_only);", "n_input_tensors = llama_graph_n_input_tensors(gf, !measure_only);"),
        ("resolve_fused_ops probes on the member scheduler", "resolve_fused_ops", "ggml_backend_sched_get_tensor_backend(state.sched.get(), node.tensor)", "ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor)"),
        ("resolve_fused_ops writes member cparams", "resolve_fused_ops", "state.cparams.auto_fgdn = false;", "cparams.auto_fgdn = false;"),
    ]
    for name, which, old, new in mutants:
        mutated = dict(bs)
        mutated[which] = mutate(bs[which], old, new)
        assert not pure_ok(mutated), f"mutant {name!r} slipped through the purity gate"


# --- the member-backed state and the wrapper ----------------------------------


def wrapper_ok(code: str) -> bool:
    ms = function_body(code, _MEMBER_STATE)
    if z("return { sched, gf_res_prev, gf_res_reserve, gf_res_prev_active, n_outputs, n_input_tensors, cparams };") not in ms:
        return False
    w = function_body(code, _WRAPPER)
    return (
        z("const sched_reserve_result result = sched_reserve_transaction();") in w
        and z("if (result.status != sched_reserve_status::OK) { if (result.fit_refusal) { throw llama_auto_ubatch_fit_refusal(result.reason); } throw std::runtime_error(result.reason); }") in w
        and z("if (!sched_need_reserve) { return; }") in w
    )


def test_wrapper_runs_the_transaction():
    assert wrapper_ok(code_of(CONTEXT_CPP)), (
        "sched_reserve() must run the reserve transaction and throw the reason of any non-OK result; "
        "member_reserve_state() must name the context's own members in the state's order"
    )


def test_wrapper_mutants():
    code = code_of(CONTEXT_CPP)
    mutants = [
        ("a member dropped from the state", "n_input_tensors, cparams };", "cparams };"),
        ("members in the wrong order", "return { sched, gf_res_prev, gf_res_reserve,", "return { sched, gf_res_reserve, gf_res_prev,"),
        ("status ignored", "sched_reserve_transaction(); if (result.status != sched_reserve_status::OK) {", "sched_reserve_transaction(); if (false) {"),
        ("the transaction bypassed", "const sched_reserve_result result = sched_reserve_transaction();", "const sched_reserve_result result = {};"),
        ("early return dropped", "if (!sched_need_reserve) { return; }", ""),
    ]
    for name, old, new in mutants:
        # the impl call recurs in the nothrow wrapper: mutate inside whichever of the two bodies carries the anchor
        scope = next(b for b in (function_body(code, _WRAPPER), function_body(code, _MEMBER_STATE)) if z(old) in b)
        mutated = code.replace(scope, mutate(scope, old, new), 1)
        assert not wrapper_ok(mutated), f"mutant {name!r} slipped through the wrapper gate"


# --- the types ----------------------------------------------------------------


def types_ok(header: str) -> bool:
    h = code_of(header)
    if z("enum class sched_reserve_status { OK, REFUSED, FAILED };") not in h:
        return False
    if z("enum class sched_reserve_mode { MEASURE, ALLOC };") not in h:
        return False
    if z("struct sched_reserve_result { sched_reserve_status status = sched_reserve_status::OK; std::string reason; bool fit_refusal = false; };") not in h:
        return False
    # the impl's declaration names the result type
    return z("sched_reserve_result sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state);") in h


def test_types():
    assert types_ok(CONTEXT_H), "sched_reserve_status {OK, REFUSED, FAILED}, sched_reserve_result, sched_reserve_mode and the impl's declaration must exist"
    assert "BUSY" not in strip_comments(CONTEXT_H).split("struct llama_context {")[0], "the status type has no BUSY"


def test_type_mutants():
    h = code_of(CONTEXT_H)
    # work on the canonical text, through a shim that accepts it
    def ok(text: str) -> bool:
        return types_ok_z(text)

    def types_ok_z(text: str) -> bool:
        return (
            z("enum class sched_reserve_status { OK, REFUSED, FAILED };") in text
            and z("enum class sched_reserve_mode { MEASURE, ALLOC };") in text
            and z("struct sched_reserve_result { sched_reserve_status status = sched_reserve_status::OK; std::string reason; bool fit_refusal = false; };") in text
            and z("sched_reserve_result sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state);") in text
        )

    assert ok(h)
    mutants = [
        ("a BUSY status", "enum class sched_reserve_status { OK, REFUSED, FAILED };", "enum class sched_reserve_status { OK, BUSY, REFUSED, FAILED };"),
        ("a status dropped", "enum class sched_reserve_status { OK, REFUSED, FAILED };", "enum class sched_reserve_status { OK, FAILED };"),
        ("the impl declared void", "sched_reserve_result sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state);", "void sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state);"),
        ("the reason dropped", "std::string reason;", "int reason;"),
    ]
    for name, old, new in mutants:
        assert not ok(mutate(h, old, new)), f"mutant {name!r} slipped through the type gate"


# --- the one compute print ----------------------------------------------------

_COMPUTE_PRINT_RE = re.compile(r"LLAMA_LOG_\w+\([^;]*" + re.escape(z("compute buffer size =")) + r"[^;]*;")


def compute_print_ok(code: str) -> bool:
    prints = _COMPUTE_PRINT_RE.findall(code)
    if len(prints) != 1:
        return False
    p = prints[0]
    # the literal prefix, never __func__
    if '"sched_reserve"' not in p or "__func__" in p:
        return False
    impl = function_body(code, _IMPL)
    if p not in impl:
        return False
    # it sits after the allocation and reads the expected-size member
    return impl.index(p) > impl.index("graph_reserve(state") and "backend_buf_exp_size[i]" in p


def test_one_compute_print_in_the_impl():
    assert compute_print_ok(code_of(CONTEXT_CPP)), (
        "exactly one statement prints `compute buffer size =`, inside sched_reserve_impl, with the literal "
        '"sched_reserve" prefix, after the reserves, reading backend_buf_exp_size[i]'
    )


def test_compute_print_mutants():
    code = code_of(CONTEXT_CPP)
    prints = _COMPUTE_PRINT_RE.findall(code)
    assert len(prints) == 1
    p = prints[0]
    mutants = {
        "prefix restored to __func__": code.replace(p, p.replace('"sched_reserve"', "__func__", 1), 1),
        "a second print in the wrapper": code.replace(
            z("sched_reserve_state state = member_reserve_state();"),
            z("sched_reserve_state state = member_reserve_state();") + p,
            1,
        ),
        "the print moved out of the impl": code.replace(p, "", 1).replace(
            z("if (!sched_need_reserve) { return; }"), z("if (!sched_need_reserve) { return; }") + p, 1
        ),
        "the print removed": code.replace(p, "", 1),
    }
    for name, mutated in mutants.items():
        assert mutated != code, name
        assert not compute_print_ok(mutated), f"mutant {name!r} slipped through the compute-print gate"


# --- graph_params -------------------------------------------------------------


_GP_INIT = (
    "return { model.arch, model.hparams, @CP@, ubatch, gtype, @SC@, backend_cpu, cvec.get(), loras.get(), mctx, "
    "&cross, &model.prec_policy, sampling.samplers, @NO@, graph_get_cb(@SC@), res, };"
)


def gp(cp: str, sc: str, no: str) -> str:
    return _GP_INIT.replace("@CP@", cp).replace("@SC@", sc).replace("@NO@", no)


def graph_params_ok(code: str) -> bool:
    old = function_body(
        code,
        "llm_graph_params llama_context::graph_params(llm_graph_result * res, const llama_ubatch & ubatch, "
        "const llama_memory_context_i * mctx, llm_graph_type gtype) const",
    )
    new = function_body(
        code,
        "llm_graph_params llama_context::graph_params(llm_graph_result * res, const llama_ubatch & ubatch, "
        "const llama_memory_context_i * mctx, llm_graph_type gtype, ggml_backend_sched_t sched_arg, "
        "const llama_cparams & cparams_arg, uint32_t n_outputs_arg) const",
    )
    return (
        z("return graph_params(res, ubatch, mctx, gtype, sched.get(), cparams, n_outputs);") in old
        and z(gp("cparams_arg", "sched_arg", "n_outputs_arg")) in new
    )


def test_graph_params_takes_the_state_values():
    assert graph_params_ok(code_of(CONTEXT_CPP))


def test_graph_params_mutants():
    code = code_of(CONTEXT_CPP)
    good = gp("cparams_arg", "sched_arg", "n_outputs_arg")
    pairs = [
        ("n_outputs from the member", good, gp("cparams_arg", "sched_arg", "n_outputs")),
        ("sched from the member", good, gp("cparams_arg", "sched.get()", "n_outputs_arg")),
        ("cparams from the member", good, gp("cparams", "sched_arg", "n_outputs_arg")),
        ("the old entry stops delegating", "return graph_params(res, ubatch, mctx, gtype, sched.get(), cparams, n_outputs);", "return graph_params(res, ubatch, mctx, gtype, nullptr, cparams, n_outputs);"),
    ]
    for name, old, new in pairs:
        assert not graph_params_ok(mutate(code, old, new)), f"mutant {name!r} slipped through the graph_params gate"


# --- gate 26: decode and encode reserve without throwing ----------------------

_NOTHROW = "bool llama_context::sched_reserve_nothrow()"
_ENCODE = "int llama_context::encode(const llama_batch_ext & batch_inp)"
_DECODE = "int llama_context::decode(const llama_batch_ext & batch_inp)"
_CALL_SITE = z("if (!sched_reserve_nothrow()) { LLAMA_LOG_ERROR(\"%s: failed to reserve the compute buffers\\n\", __func__); return -2; }")


def nothrow_ok(code: str) -> bool:
    b = function_body(code, _NOTHROW)
    impl_call = z("const sched_reserve_result result = sched_reserve_transaction();")
    if impl_call not in b or "catch(conststd::exception&" not in b.replace(" ", ""):
        return False
    # success returns true; every other path logs, re-arms the reserve and returns false
    tail = z("sched_need_reserve = true; return false; }")
    if not b.endswith(tail):
        return False
    if b.count("return true;") != 2 or b.count("return false;") != 1:
        return False
    # `sched_reserve_nothrow` itself contains "throw": only a bare throw statement counts
    return z("if (!sched_need_reserve) { return true; }") in b and not re.search(r"(?<![A-Za-z_])throw(?![A-Za-z_])", b)


def call_sites_ok(code: str) -> bool:
    for sig in (_ENCODE, _DECODE):
        b = function_body(code, sig)
        if _CALL_SITE not in b or z("sched_reserve();") in b:
            return False
        if b.count("sched_reserve_nothrow()") != 1:
            return False
    return True


def test_nothrow_form_catches_and_rearms():
    assert nothrow_ok(code_of(CONTEXT_CPP)), (
        "sched_reserve_nothrow must run the impl as ALLOC, catch std::exception, log, set sched_need_reserve "
        "again and return false"
    )


def test_decode_and_encode_reserve_without_throwing():
    assert call_sites_ok(code_of(CONTEXT_CPP)), (
        "encode and decode must call sched_reserve_nothrow() once, return -2 on false, and not call "
        "the throwing sched_reserve()"
    )


def test_nothrow_mutants():
    code = code_of(CONTEXT_CPP)
    mutants = [
        ("the catch dropped", "catch (const std::exception & err) {", "catch (const int & err) {"),
        ("the re-arm dropped", "sched_need_reserve = true; return false; }", "return false; }"),
        ("a failure reported as success", "sched_need_reserve = true; return false; }", "sched_need_reserve = true; return true; }"),
        ("the transaction bypassed", "const sched_reserve_result result = sched_reserve_transaction(); if (result.status == sched_reserve_status::OK) {", "const sched_reserve_result result = {}; if (result.status == sched_reserve_status::OK) {"),
        ("the early return removed", "if (!sched_need_reserve) { return true; }", ""),
    ]
    body = function_body(code, _NOTHROW)
    for name, old, new in mutants:
        # mutate the wrapper body only: the same text recurs in other functions
        mutated = code.replace(body, mutate(body, old, new), 1)
        assert mutated != code
        assert not nothrow_ok(mutated), f"mutant {name!r} slipped through the nothrow gate"


def test_call_site_mutants():
    code = code_of(CONTEXT_CPP)
    # the throwing form back in the encode path, then in the decode path
    for anchor_sig in (_ENCODE, _DECODE):
        b = function_body(code, anchor_sig)
        mutated_body = b.replace(_CALL_SITE, z("sched_reserve();"), 1)
        assert mutated_body != b
        assert not call_sites_ok(code.replace(b, mutated_body, 1)), anchor_sig
        # the nothrow call kept but its failure ignored
        ignored = b.replace(_CALL_SITE, z("sched_reserve_nothrow();"), 1)
        assert not call_sites_ok(code.replace(b, ignored, 1)), anchor_sig
        # a second reserve call
        doubled = b.replace(_CALL_SITE, _CALL_SITE + z("sched_reserve();"), 1)
        assert not call_sites_ok(code.replace(b, doubled, 1)), anchor_sig
