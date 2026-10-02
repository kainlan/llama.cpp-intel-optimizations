"""Source gate for the load-time measure and its late-check call site (zhcn design 2.10, C7h-3).

`llama_load_measure` builds one measure-only context over a load's placement. This gate pins, on
comment-stripped text:

- the unwind order of its locals: the measure backends (in `args`), then the context holder, then the
  plan-override guard, with the guard installed (and checked) before the context is constructed, so on
  every exit the override clears first, then the context goes, then the backends;
- the backends are the model's SYCL devices' measure backends with the CPU backend last;
- every refusal goes through the one `[LOAD-PLAN] compute-slot measure failed at ... (refused)` text;
- the shape is n_ubatch 512 and the caller's n_ctx (the training context for 0);
- `llama_load_late_check` measures only when the backend exports the L4 entry points, puts the weight
  stand-ins up before the measure, measures at stage (c), and turns a REFUSED answer into the load's
  refusal;
- the loader calls the late check exactly once, after the dev_layer sync and before the mappings are
  initialised, and a non-empty answer is thrown.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEXT_CPP = (SRC / "llama-context.cpp").read_text()
MODEL_CPP = (SRC / "llama-model.cpp").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_MEASURE = (
    "llama_load_measure_result llama_load_measure(const llama_model & model, uint32_t n_ctx, uint64_t load_txn, "
    "enum ggml_sycl_measure_stage stage)"
)
_LATE = (
    "std::string llama_load_late_check(const llama_model & model, uint32_t n_ctx, struct ggml_sycl_load_txn txn, "
    "const std::vector<llama_measure_dummy_entry> & weights)"
)
_REFUSAL = "static std::string llama_load_measure_refusal(enum ggml_sycl_measure_stage stage, int device, const std::string & reason)"


def measure_ok(code: str) -> bool:
    b = function_body(code, _MEASURE)
    order = [
        "llama_measure_context_args args;",
        "llama_context_sycl_measure_backend_init(",
        "ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)",
        "std::unique_ptr<llama_context> holder;",
        "llama_measure_plan_override guard(",
        "if (!guard.installed())",
        "holder.reset(new llama_context(model, params, &args));",
    ]
    pos = [b.find(z(t)) for t in order]
    if -1 in pos or pos != sorted(pos):
        return False
    # the CPU backend is the last one in the list
    if b.find(z("args.backends.emplace_back(cpu);")) < b.find(z("ggml_backend_init_by_type(")):
        return False
    if b.find(z("args.backends.emplace_back(cpu);")) > b.find(z("std::unique_ptr<llama_context> holder;")):
        return False
    needs = [
        "args.stage = stage;",
        "params.n_ubatch = 512;",
        "params.n_ctx = n_ctx != 0 ? n_ctx : model.hparams.n_ctx_train;",
        "status.status != sched_reserve_status::OK",
        "out.n_splits = holder->get_measure_plan().n_splits_max;",
    ]
    if not all(z(n) in b for n in needs):
        return False
    # a refusal is never built outside the one text
    return b.count(z("out.refusal = ")) == b.count(z("llama_load_measure_refusal(")) and "[LOAD-PLAN]" not in b


def refusal_ok(code: str) -> bool:
    b = function_body(code, _REFUSAL)
    return z('"[LOAD-PLAN] compute-slot measure failed at %s on device %d: %s (refused)"') in b and all(
        z(f'return "{s}";') in code for s in ("probe", "admitted", "late")
    )


def late_ok(code: str) -> bool:
    b = function_body(code, _LATE)
    order = [
        "if (!procs.available())",
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        "llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)",
        "llama_sycl_l4_late_check(procs, txn, d.device, d.total) == GGML_SYCL_LATE_CHECK_REFUSED",
    ]
    pos = [b.find(z(t)) for t in order]
    return -1 not in pos and pos == sorted(pos) and z("if (!measured.ok) { return measured.refusal; }") in b


def site_ok(code: str) -> bool:
    c = code
    sync = c.find(z("dev_layer sync: corrected"))
    call = c.find(z("llama_load_late_check("))
    maps = c.find(z("ml.init_mappings(true,"))
    prec = c.find(z("prec_policy.load(ml, *this);"))
    if min(sync, call, maps, prec) == -1 or not (sync < call < prec < maps):
        return False
    if c.count(z("llama_load_late_check(")) != 1:
        return False
    seg = c[call : call + 400]
    return z("throw std::runtime_error(late_refusal);") in c[call : prec] and "sycl_model_loading_guard.txn" in seg


def test_the_measure_orders_its_locals_for_unwind():
    assert measure_ok(code_of(CONTEXT_CPP))


def test_measure_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _MEASURE)
    for name, old, new in [
        ("the guard declared before the holder", "std::unique_ptr<llama_context> holder;", ""),
        ("the guard unchecked", "if (!guard.installed())", "if (false)"),
        ("cpu not last", "args.backends.emplace_back(cpu);", "args.backends.insert(args.backends.begin(), ggml_backend_ptr(cpu));"),
        ("ubatch not pinned", "params.n_ubatch = 512;", ""),
        ("stage not carried", "args.stage = stage;", ""),
        ("status unread", "status.status != sched_reserve_status::OK", "false"),
        ("a refusal outside the text", "out.refusal = llama_load_measure_refusal(stage, first_device, e.what());", "out.refusal = e.what();"),
    ]:
        assert not measure_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_refusal_text_is_one():
    assert refusal_ok(code_of(CONTEXT_CPP))
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _REFUSAL)
    assert not refusal_ok(code.replace(b, mutate(b, "(refused)", "(failed)"), 1))
    assert not refusal_ok(code.replace(z('return "late";'), z('return "later";'), 1))


def test_the_late_check_measures_only_with_l4_and_at_stage_c():
    assert late_ok(code_of(CONTEXT_CPP))


def test_late_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _LATE)
    for name, old, new in [
        ("no L4 gate", "if (!procs.available())", "if (false)"),
        ("measured at stage b", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)"),
        ("a refused answer ignored", "== GGML_SYCL_LATE_CHECK_REFUSED", "== GGML_SYCL_LATE_CHECK_EQUAL"),
        ("a failed measure ignored", "if (!measured.ok) { return measured.refusal; }", ""),
        ("stand-ins after the measure", "llama_measure_dummy_scope dummies(weights);", ""),
    ]:
        assert not late_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_loader_calls_the_late_check_once_before_the_mappings():
    assert site_ok(code_of(MODEL_CPP))


def test_site_mutants():
    code = code_of(MODEL_CPP)
    assert not site_ok(code.replace(z("throw std::runtime_error(late_refusal);"), "", 1))
    assert not site_ok(code + z("llama_load_late_check(*this, 0, sycl_model_loading_guard.txn, {});"))
    moved = code.replace(z("ml.init_mappings(true,"), z("llama_load_late_check(") + z("ml.init_mappings(true,"), 1)
    assert not site_ok(moved)
