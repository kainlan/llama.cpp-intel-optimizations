"""Source gate for the MEASURE reserve and the plan scopes (zhcn design gates 6,
19 and 25, as far as the llama side can satisfy them before the fixpoint
acquires the chunk-cap copy).

MEASURE is `sched_reserve_impl(sched_reserve_mode::MEASURE, ...)`, which hands
over to `sched_measure_impl`. It reserves every graph of
`llama_measure_graph_set` on a scheduler of its own, inside a MEASURE plan
scope opened with the context's `plan_caps`, and plans each compute buft's
chunks from the layouts they left. This gate pins, on comment-stripped text:

- the types: the measure plan, the state's trailing `measure` pointer (null
  for ALLOC), the measure storage that owns what a MEASURE writes, and
  `plan_caps` as a `unique_ptr` whose deleter holds the backend's `_free`;
- the dispatch: MEASURE is the impl's first branch;
- purity: `sched_measure_impl` names no context member a MEASURE must not
  write, and calls nothing that waits, publishes or reserves the context;
- the scopes: a MEASURE scope in the measure, an ALLOC scope in the ALLOC
  reserve, each constructed from `plan_caps.get()` before the scheduler it
  covers is created, and the scope RAII opens only for a copy and closes
  when it leaves scope;
- pipeline parallelism cannot reach a planned scheduler: the measure asserts
  it off and builds its scheduler with it off, and the ALLOC retry asserts
  the context has no copy;
- the graph set and the chunk plan: the measure builds the graph set from
  `llama_measure_graph_set`, reserves a stream graph on `init_reserve`,
  reserves size-only, reads the layout through the sched query, and plans
  through `llama_measure_chunk_plan` with `ggml_gallocr_max_chunks()`;
- a failed scope read refuses before the chunk plan and before ALLOC's OK;
- the reserve transaction: a planned context runs MEASURE, then publishes what
  the measure resolved, then ALLOC, returns the first non-OK status, and the
  resolved fused-op flags reach the context's cparams only after the publish
  succeeded;
- the training abort and the re-check guard.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
CONTEXT_H = (ROOT / "src/llama-context.h").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

strip_comments = _gate.strip_comments
z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_MEASURE = "sched_reserve_result llama_context::sched_measure_impl(sched_reserve_state & state)"
_IMPL = _gate._IMPL
_RESOLVE = _gate._RESOLVE
_OPT_INIT = "void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params)"

_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


def keep(raw: str) -> str:
    """Comment-stripped text with the string literals kept (code_of keeps them
    too, but z() would squeeze the spaces inside them: here we only want to
    find a literal, so the canonical spelling of the whole text is enough)."""
    return code_of(raw)


# --- the types ----------------------------------------------------------------


def types_ok(header: str) -> bool:
    h = code_of(header)
    needed = [
        # the state's trailing field is the measure pointer, null for ALLOC
        "llama_cparams & cparams; sched_measure_plan * measure = nullptr; };",
        # the measure's plan
        "struct sched_measure_buft { ggml_backend_buffer_type_t buft = nullptr; size_t max_chunk_size = 0; std::vector<std::vector<size_t>> peaks; std::vector<size_t> cap; };",
        "struct sched_measure_plan { std::vector<llama_measure_graph> graphs; std::vector<sched_measure_buft> bufts; uint32_t n_measured = 0; double measure_ms = 0.0; };",
        # the storage a MEASURE writes: the scheduler is declared first, so it dies last
        "struct sched_measure_storage { ggml_backend_sched_ptr sched;",
        "return { sched, gf_res_prev, gf_res_reserve, gf_res_prev_active, n_outputs, n_input_tensors, cparams, &plan };",
        # the copy and its deleter
        "struct llama_plan_caps_deleter { decltype(&ggml_backend_sycl_plan_caps_free) free_fn = nullptr;",
        "using llama_plan_caps_ptr = std::unique_ptr<ggml_backend_sycl_plan_caps, llama_plan_caps_deleter>;",
        "llama_plan_caps_ptr plan_caps;",
        "sched_reserve_result sched_measure_impl(sched_reserve_state & state);",
    ]
    return all(z(n) in h for n in needed)


def test_types():
    assert types_ok(CONTEXT_H)


def test_type_mutants():
    h = code_of(CONTEXT_H)
    mutants = [
        ("measure pointer is not null by default", "sched_measure_plan * measure = nullptr;", "sched_measure_plan * measure;"),
        ("measure pointer is not trailing", "llama_cparams & cparams; sched_measure_plan * measure = nullptr; };", "sched_measure_plan * measure = nullptr; llama_cparams & cparams; };"),
        ("a buft keeps no peaks", "std::vector<std::vector<size_t>> peaks;", "std::vector<size_t> peaks;"),
        ("storage state drops the plan", "n_input_tensors, cparams, &plan };", "n_input_tensors, cparams, nullptr };"),
        ("the copy is a raw pointer", "llama_plan_caps_ptr plan_caps;", "ggml_backend_sycl_plan_caps * plan_caps = nullptr;"),
        ("the copy has no deleter", "std::unique_ptr<ggml_backend_sycl_plan_caps, llama_plan_caps_deleter>", "std::unique_ptr<ggml_backend_sycl_plan_caps>"),
        ("the deleter loses the proc", "decltype(&ggml_backend_sycl_plan_caps_free) free_fn = nullptr;", "int free_fn = 0;"),
    ]
    for name, old, new in mutants:
        assert not types_ok_z(mutate(h, old, new)), f"mutant {name!r} slipped through the type gate"


def types_ok_z(text: str) -> bool:
    # types_ok works on raw header text; the mutants arrive already canonical
    needed_mutated = [
        "llama_cparams & cparams; sched_measure_plan * measure = nullptr; };",
        "std::vector<std::vector<size_t>> peaks;",
        "return { sched, gf_res_prev, gf_res_reserve, gf_res_prev_active, n_outputs, n_input_tensors, cparams, &plan };",
        "llama_plan_caps_ptr plan_caps;",
        "std::unique_ptr<ggml_backend_sycl_plan_caps, llama_plan_caps_deleter>",
        "decltype(&ggml_backend_sycl_plan_caps_free) free_fn = nullptr;",
    ]
    return all(z(n) in text for n in needed_mutated)


# --- the dispatch and the measure's purity ------------------------------------

# Context members a MEASURE must not write (the reserve-state gate's list, plus what
# only a measure could reach for): none may be named bare in the body proper.
_MEMBER_RE = re.compile(
    _gate._MEMBER_RE.pattern
    + r"|(?<![\w.>])(?<!::)(?:backend_buf_exp_size|sched_need_reserve|opt_ctx|mem_storage)\b"
)
# Calls a MEASURE must not make: they wait, publish, or reserve the context itself.
_FORBIDDEN_CALLS = [
    "synchronize(",
    "sched_reserve(",
    "sched_reserve_nothrow(",
    "sycl_publish_runtime_context(",
    "sycl_resync_runtime_context_flash_attn(",
    "sycl_recheck_runtime_context_flash_attn(",
    "memory_update(",
    "ggml_backend_sched_reset(",
]


def measure_body(code: str) -> str:
    return function_body(code, _MEASURE)


def purity_ok(body: str) -> bool:
    proper = _STRING_RE.sub('""', body[body.index("{") :])
    if _MEMBER_RE.findall(proper):
        return False
    for call in _FORBIDDEN_CALLS:
        if re.search(r"(?<![\w:])" + re.escape(call), proper):
            return False
    return not re.search(r"(?<![A-Za-z_])throw(?![A-Za-z_])", proper)


def test_measure_is_pure():
    assert purity_ok(measure_body(code_of(CONTEXT_CPP))), (
        "sched_measure_impl must touch only its state: no context scheduler, arena, cparams, expected-size or "
        "need-reserve member, no synchronize, publish, reserve or reset, and no throw"
    )


def test_measure_purity_mutants():
    body = measure_body(code_of(CONTEXT_CPP))
    mutants = [
        ("a member scheduler write", "state.sched.reset(", "sched.reset("),
        ("a member cparams read", "const uint32_t n_seqs = state.cparams.n_seq_max;", "const uint32_t n_seqs = cparams.n_seq_max;"),
        ("the expected sizes written", "plan.graphs = graphs;", "backend_buf_exp_size[0] = 0; plan.graphs = graphs;"),
        ("the context synchronized", "const int64_t t_start_us = ggml_time_us();", "synchronize(); const int64_t t_start_us = ggml_time_us();"),
        ("the context published", "const int64_t t_start_us = ggml_time_us();", "sycl_publish_runtime_context(false); const int64_t t_start_us = ggml_time_us();"),
        ("the context reserved", "const int64_t t_start_us = ggml_time_us();", "sched_reserve(); const int64_t t_start_us = ggml_time_us();"),
        ("the re-check called", "const int64_t t_start_us = ggml_time_us();", "sycl_recheck_runtime_context_flash_attn(); const int64_t t_start_us = ggml_time_us();"),
        ("need-reserve cleared", "plan.graphs = graphs;", "sched_need_reserve = false; plan.graphs = graphs;"),
        ("a refusal thrown", 'return { sched_reserve_status::FAILED, "failed to initialize memory module" };', 'throw std::runtime_error("failed to initialize memory module");'),
    ]
    for name, old, new in mutants:
        assert not purity_ok(mutate(body, old, new)), f"mutant {name!r} slipped through the purity gate"


def dispatch_ok(code: str) -> bool:
    b = function_body(code, _IMPL)
    first = z("if (mode == sched_reserve_mode::MEASURE) { return sched_measure_impl(state); }")
    return b[b.index("{") + 1 :].startswith(first)


def test_measure_is_the_impls_first_branch():
    assert dispatch_ok(code_of(CONTEXT_CPP))


def test_dispatch_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _IMPL)
    for name, old, new in [
        ("the branch dropped", "if (mode == sched_reserve_mode::MEASURE) { return sched_measure_impl(state); }", ""),
        ("the branch inverted", "if (mode == sched_reserve_mode::MEASURE)", "if (mode == sched_reserve_mode::ALLOC)"),
        ("the measure result discarded", "return sched_measure_impl(state);", "sched_measure_impl(state);"),
    ]:
        mutated = code.replace(b, mutate(b, old, new), 1)
        assert not dispatch_ok(mutated), f"mutant {name!r} slipped through the dispatch gate"


# --- the scopes ---------------------------------------------------------------


def scope_ctor(mode: str) -> str:
    return (
        "llama_plan_scope plan_scope(plan_procs, (uint32_t) sycl_exec_context.value, GGML_SYCL_PLAN_SCOPE_"
        + mode
        + ", plan_caps.get());"
    )


def scopes_ok(code: str) -> bool:
    for body, mode in ((measure_body(code), "MEASURE"), (function_body(code, _IMPL), "ALLOC")):
        c = z(scope_ctor(mode))
        if body.count(c) != 1:
            return False
        # the scheduler the scope covers is created after it
        sched_new = body.find("ggml_backend_sched_new(")
        if sched_new == -1 or body.index(c) > sched_new:
            return False
        if z("const llama_context_sycl_plan_procs plan_procs = llama_context_sycl_plan_procs_for(backends);") not in body:
            return False
    # a planned context whose scope does not open refuses, in both
    if z('if (!plan_scope.is_open()) { return { sched_reserve_status::FAILED, "the MEASURE plan scope did not open" }; }') not in measure_body(code):
        return False
    if z('if (plan_caps && !plan_scope.is_open()) { return { sched_reserve_status::FAILED, "the ALLOC plan scope did not open" }; }') not in function_body(code, _IMPL):
        return False
    return True


def test_scopes_cover_their_schedulers():
    assert scopes_ok(code_of(CONTEXT_CPP)), (
        "the MEASURE and ALLOC reserves must each construct a plan scope from plan_caps.get() before they create "
        "their scheduler, and refuse when it did not open"
    )


def test_scope_mutants():
    code = code_of(CONTEXT_CPP)
    m = measure_body(code)
    a = function_body(code, _IMPL)
    mutants = [
        ("the measure scope is an ALLOC scope", m, scope_ctor("MEASURE"), scope_ctor("ALLOC")),
        ("the alloc scope is a MEASURE scope", a, scope_ctor("ALLOC"), scope_ctor("MEASURE")),
        ("the measure scope takes no copy", m, "GGML_SYCL_PLAN_SCOPE_MEASURE, plan_caps.get());", "GGML_SYCL_PLAN_SCOPE_MEASURE, nullptr);"),
        ("the alloc scope takes no copy", a, "GGML_SYCL_PLAN_SCOPE_ALLOC, plan_caps.get());", "GGML_SYCL_PLAN_SCOPE_ALLOC, nullptr);"),
        ("a closed measure scope ignored", m, 'if (!plan_scope.is_open()) { return { sched_reserve_status::FAILED, "the MEASURE plan scope did not open" }; }', ""),
        ("a closed alloc scope ignored", a, 'if (plan_caps && !plan_scope.is_open()) { return { sched_reserve_status::FAILED, "the ALLOC plan scope did not open" }; }', ""),
    ]
    for name, body, old, new in mutants:
        assert not scopes_ok(code.replace(body, mutate(body, old, new), 1)), f"mutant {name!r} slipped through the scope gate"

    # moved, not dropped: the same scope constructed after the scheduler it covers
    for name, body, mode in (("measure", m, "MEASURE"), ("alloc", a, "ALLOC")):
        ctor = scope_ctor(mode)
        moved = mutate(mutate(body, ctor, ""), "llama_memory_context_ptr mctx;", ctor + " llama_memory_context_ptr mctx;")
        assert not scopes_ok(code.replace(body, moved, 1)), f"mutant 'the {name} scope opens after its scheduler' slipped through"


def raii_ok(code: str) -> bool:
    start = code.find("structllama_plan_scope{")
    if start == -1:
        start = code.find("struct llama_plan_scope{")
    if start == -1:
        return False
    i = code.index("{", start)
    depth = 0
    end = -1
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    b = code[start : end + 1]
    return (
        # it opens only for a copy and only with both procs
        z("if (caps != nullptr && procs.scope_open != nullptr && close_fn != nullptr) { scope = procs.scope_open(context_id, mode, caps); }") in b
        # it closes what it opened, once, from the destructor
        and z("~llama_plan_scope() { if (scope != nullptr) { close_fn(scope); } }") in b
        # it is neither copyable nor assignable
        and z("llama_plan_scope(const llama_plan_scope &) = delete;") in b
        and z("llama_plan_scope & operator=(const llama_plan_scope &) = delete;") in b
        # the failure read is behind the open scope
        and z("const char * failure() const { return scope != nullptr && failure_fn != nullptr ? failure_fn(scope) : nullptr; }") in b
    )


def test_scope_raii():
    assert raii_ok(code_of(CONTEXT_CPP)), (
        "llama_plan_scope must open only for a copy, close in its destructor, be non-copyable, and read a failure "
        "only through an open scope"
    )


def test_scope_raii_mutants():
    code = code_of(CONTEXT_CPP)
    mutants = [
        ("opens without a copy", "if (caps != nullptr && procs.scope_open != nullptr && close_fn != nullptr) {", "if (procs.scope_open != nullptr && close_fn != nullptr) {"),
        ("never closes", "~llama_plan_scope() { if (scope != nullptr) { close_fn(scope); } }", "~llama_plan_scope() {}"),
        ("closes unconditionally", "~llama_plan_scope() { if (scope != nullptr) { close_fn(scope); } }", "~llama_plan_scope() { close_fn(scope); }"),
        ("copyable", "llama_plan_scope(const llama_plan_scope &) = delete;", ""),
        ("failure read without an open scope", "return scope != nullptr && failure_fn != nullptr ? failure_fn(scope) : nullptr;", "return failure_fn != nullptr ? failure_fn(scope) : nullptr;"),
    ]
    for name, old, new in mutants:
        assert not raii_ok(mutate(code, old, new)), f"mutant {name!r} slipped through the RAII gate"


# --- pipeline parallelism -----------------------------------------------------


def pipeline_ok(code: str) -> bool:
    m = measure_body(code)
    a = function_body(code, _IMPL)
    retry = z("if (state.cparams.pipeline_parallel) { GGML_ASSERT(!plan_caps);")
    return (
        z("GGML_ASSERT(!state.cparams.pipeline_parallel);") in m
        and z("state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, state.cparams.op_offload));") in m
        and retry in a
    )


def test_pipeline_parallel_cannot_reach_a_planned_scheduler():
    assert pipeline_ok(code_of(CONTEXT_CPP))


def test_pipeline_mutants():
    code = code_of(CONTEXT_CPP)
    m = measure_body(code)
    a = function_body(code, _IMPL)
    for name, body, old, new in [
        ("the measure no longer asserts", m, "GGML_ASSERT(!state.cparams.pipeline_parallel);", ""),
        ("the measure scheduler pipelined", m, "max_nodes, false, state.cparams.op_offload", "max_nodes, state.cparams.pipeline_parallel, state.cparams.op_offload"),
        ("the retry may rebuild a planned scheduler", a, "if (state.cparams.pipeline_parallel) { GGML_ASSERT(!plan_caps);", "if (state.cparams.pipeline_parallel) {"),
    ]:
        assert not pipeline_ok(code.replace(body, mutate(body, old, new), 1)), f"mutant {name!r} slipped through"


# --- the graph set and the chunk plan -----------------------------------------


def plan_ok(code: str) -> bool:
    m = measure_body(code)
    needed = [
        "const std::vector<llama_measure_graph> graphs = llama_measure_graph_set(params);",
        "const int max_chunks = ggml_gallocr_max_chunks();",
        # a stream graph is built on a memory context spanning exactly its streams
        "stream_mctx = memory->init_reserve(g.n_streams);",
        "if (g.n_streams != 0 && memory) {",
        # size-only: the measure allocates nothing
        "auto * gf = graph_reserve(state, g.n_tokens, g.n_seqs, g.n_outputs, graph_mctx, true, sizes.data());",
        # the layout each graph left, per buft once
        "ggml_backend_sched_get_reserved_chunk_peaks(state.sched.get(), backend_ptrs[i], peak.data(), max_chunks, &max_chunk_size);",
        "if (entry->peaks.size() > gi) { continue; }",
        # the plan, through the pure helper
        "llama_measure_chunk_plan(entry.peaks, entry.max_chunk_size, (size_t) max_chunks, entry.cap, reason)",
        # nextn variants only where the model has layers for them
        "if (model.hparams.n_layer_nextn > 0) {",
        "params.pp_again_single_seq = model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01;",
        "params.n_layer_nextn = model.hparams.n_layer_nextn;",
    ]
    if not all(z(n) in m for n in needed):
        return False
    # a failed scope read refuses before the chunk plan is made
    return m.index("plan_scope.failure()") < m.index("llama_measure_chunk_plan(")


def test_measure_builds_the_graph_set_and_plans_the_chunks():
    assert plan_ok(code_of(CONTEXT_CPP))


def test_plan_mutants():
    code = code_of(CONTEXT_CPP)
    m = measure_body(code)
    mutants = [
        ("a hand-written graph set", "llama_measure_graph_set(params);", "std::vector<llama_measure_graph>{};"),
        ("the gallocr chunk limit hard-coded", "const int max_chunks = ggml_gallocr_max_chunks();", "const int max_chunks = 8;"),
        ("streams measured on the full memory", "stream_mctx = memory->init_reserve(g.n_streams);", "stream_mctx = memory->init_full();"),
        ("every graph treated as a stream graph", "if (g.n_streams != 0 && memory) {", "if (memory) {"),
        ("the measure allocates", "graph_mctx, true, sizes.data());", "graph_mctx, false, nullptr);"),
        ("a buft counted once per backend", "if (entry->peaks.size() > gi) { continue; }", ""),
        ("the chunk plan reimplemented", "llama_measure_chunk_plan(entry.peaks, entry.max_chunk_size, (size_t) max_chunks, entry.cap, reason)", "true"),
        ("nextn varied for every model", "if (model.hparams.n_layer_nextn > 0) {", "if (true) {"),
        ("the closing pp graph never single-sequence", "model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01;", "false;"),
    ]
    for name, old, new in mutants:
        assert not plan_ok(code.replace(m, mutate(m, old, new), 1)), f"mutant {name!r} slipped through the plan gate"


def failure_order_ok(code: str) -> bool:
    m = measure_body(code)
    a = function_body(code, _IMPL)
    tail = z("if (const char * failure = plan_scope.failure()) { return { sched_reserve_status::REFUSED, failure }; }")
    # the ALLOC reserve refuses after its reserves and before it reports OK
    ok_return = z('return { sched_reserve_status::OK, "" };')
    return (
        m.count(tail) == 1
        and a.count(tail) == 1
        and a.index(tail) < a.rindex(ok_return)
        and a.index(tail) > a.index("ggml_backend_sched_get_n_copies(")
    )


def test_a_failed_scope_read_refuses():
    assert failure_order_ok(code_of(CONTEXT_CPP))


def test_failure_mutants():
    code = code_of(CONTEXT_CPP)
    m = measure_body(code)
    a = function_body(code, _IMPL)
    tail = "if (const char * failure = plan_scope.failure()) { return { sched_reserve_status::REFUSED, failure }; }"
    for name, body in (("measure", m), ("alloc", a)):
        assert not failure_order_ok(code.replace(body, mutate(body, tail, ""), 1)), f"{name}: failure check dropped"
        assert not failure_order_ok(
            code.replace(body, mutate(body, tail, tail.replace("REFUSED", "OK")), 1)
        ), f"{name}: failure reported as OK"


# --- training, and the re-check ------------------------------------------------


def training_ok(code: str) -> bool:
    b = function_body(code, _OPT_INIT)
    return b[b.index("{") + 1 :].startswith(
        z(
            'if (plan_caps) { GGML_ABORT("training is not supported while a SYCL placement plan is active: ggml-opt would reallocate this "'
            " \"context's planned scheduler outside its claim scope; llama.cpp-q64b\"); }"
        )
    )


def test_training_aborts_under_a_plan():
    assert training_ok(code_of(CONTEXT_CPP)), "opt_init must abort first thing when the context owns a chunk-cap copy"


def test_training_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _OPT_INIT)
    for name, old, new in [
        ("the abort dropped", "if (plan_caps) {", "if (false) {"),
        ("the abort unnamed", "llama.cpp-q64b", "llama.cpp"),
    ]:
        assert not training_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    # the abort moved behind the first statement
    first = z("if (plan_caps) {")
    moved = b.replace(first, z("GGML_ASSERT(!opt_ctx);") + first, 1)
    assert not training_ok(code.replace(b, moved, 1)), "mutant 'abort after other work' slipped through"


def recheck_ok(code: str) -> bool:
    b = function_body(code, _RESOLVE)
    return b.count("sycl_recheck_runtime_context_flash_attn()") == 1 and z(
        "if (state.measure == nullptr) { sycl_recheck_runtime_context_flash_attn(); }"
    ) in b


def test_a_measure_does_not_recheck():
    assert recheck_ok(code_of(CONTEXT_CPP))


def test_recheck_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _RESOLVE)
    for name, old, new in [
        ("the guard dropped", "if (state.measure == nullptr) { sycl_recheck_runtime_context_flash_attn(); }", "sycl_recheck_runtime_context_flash_attn();"),
        ("the guard inverted", "if (state.measure == nullptr) {", "if (state.measure != nullptr) {"),
    ]:
        assert not recheck_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


# --- the reserve transaction ---------------------------------------------------

_TXN = "sched_reserve_result llama_context::sched_reserve_transaction()"
_FLAGS = [
    "flash_attn", "auto_fa", "fused_gdn_ar", "fused_gdn_ch", "auto_fgdn", "fused_lid", "auto_flid",
    "fused_dsv4_hc_pre", "fused_dsv4_hc_comb", "fused_dsv4_hc_post", "auto_fhc",
]


def txn_ok(code: str) -> bool:
    b = function_body(code, _TXN)
    steps = [
        "if (plan_caps) {",
        "sched_measure_storage storage(cparams);",
        "sched_reserve_state measure_state = storage.state();",
        "const sched_reserve_result measured = sched_reserve_impl(sched_reserve_mode::MEASURE, measure_state);",
        "if (measured.status != sched_reserve_status::OK) { return measured; }",
        "const sched_reserve_result published = sycl_publish_runtime_context(storage.cparams.flash_attn);",
        "if (published.status != sched_reserve_status::OK) { return published; }",
    ]
    at = 0
    for step in steps:
        i = b.find(z(step), at)
        if i == -1:
            return False
        at = i + len(z(step))
    # the resolved flags are applied after the publish check, and all of them are
    for flag in _FLAGS:
        w = z(f"cparams.{flag} = storage.cparams.{flag};")
        if b.count(w) != 1 or b.index(w) < b.index(z("if (published.status != sched_reserve_status::OK) { return published; }")):
            return False
    # ALLOC runs last, on the member state, for planned and unplanned contexts alike
    alloc = z("sched_reserve_state state = member_reserve_state(); return sched_reserve_impl(sched_reserve_mode::ALLOC, state); }")
    return b.endswith(alloc) and b.index(alloc) > b.index(z("storage.cparams.auto_fhc;"))


def test_transaction_measures_publishes_then_allocates():
    assert txn_ok(code_of(CONTEXT_CPP))


def test_transaction_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _TXN)
    mutants = [
        ("measured status ignored", "if (measured.status != sched_reserve_status::OK) { return measured; }", ""),
        ("published status ignored", "if (published.status != sched_reserve_status::OK) { return published; }", ""),
        ("the publish carries the context's own flash_attn", "sycl_publish_runtime_context(storage.cparams.flash_attn)", "sycl_publish_runtime_context(cparams.flash_attn)"),
        ("the measure runs as ALLOC", "sched_reserve_impl(sched_reserve_mode::MEASURE, measure_state)", "sched_reserve_impl(sched_reserve_mode::ALLOC, measure_state)"),
        ("an unplanned context measures", "if (plan_caps) {", "if (true) {"),
        ("the measure runs on the context's own cparams", "sched_measure_storage storage(cparams);", "sched_measure_storage storage(llama_cparams{});"),
        ("a resolved flag dropped", "cparams.fused_lid = storage.cparams.fused_lid;", ""),
        ("a resolved flag applied twice", "cparams.auto_fa = storage.cparams.auto_fa;", "cparams.auto_fa = storage.cparams.auto_fa; cparams.auto_fa = storage.cparams.auto_fa;"),
    ]
    for name, old, new in mutants:
        assert not txn_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through the transaction gate"
    # the flags applied before the publish is checked
    first = z("cparams.flash_attn = storage.cparams.flash_attn;")
    pub = z("const sched_reserve_result published = sycl_publish_runtime_context(storage.cparams.flash_attn);")
    early = b.replace(first, "", 1).replace(pub, first + pub, 1)
    assert not txn_ok(code.replace(b, early, 1)), "mutant 'flags applied before the publish' slipped through"
    # ALLOC dropped from the transaction
    no_alloc = b.replace(z("sched_reserve_state state = member_reserve_state(); return sched_reserve_impl(sched_reserve_mode::ALLOC, state); }"), "return {}; }", 1)
    assert not txn_ok(code.replace(b, no_alloc, 1)), "mutant 'ALLOC dropped' slipped through"
