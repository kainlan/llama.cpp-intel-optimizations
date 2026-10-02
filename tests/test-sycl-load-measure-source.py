"""Source gate for the load-time measure and its late-check call site (zhcn design 2.10, C7h-3, and the
C7 review folds I2, I3, I4, I5, I6, M1, M3).

`llama_load_measure` builds the measure backends and hands them to `llama_load_measure_run`, which builds
one measure-only context over a load's placement. This gate pins, on comment-stripped text:

- the unwind order of the run's locals: the caller's `args` (with the measure backends, the CPU backend last)
  outlive the context holder, then the plan-override guard, with the guard installed (and checked) before
  the context is constructed, so on every exit the override clears first, then the context goes, then the
  backends; a `llama_measure_unsupported` is caught before the generic catch and is `unsupported`, not a
  refusal;
- every refusal goes through the one `[LOAD-PLAN] compute-slot measure failed at ... (refused)` text, which
  occurs once in src/;
- the shape is the auto-ubatch ladder's bottom rung (no literal 512) and the caller's n_ctx (the training
  context for 0);
- the compute term is the one per-chunk peak helper of llama-measure-plan.h (no second reduction);
- `llama_load_late_check` measures only when the backend exports the L4 entry points, puts the weight
  stand-ins up before the measure, measures at stage (c), and hands the devices to the one fold, in which
  NOT_RECORDED is its own list and never an EQUAL, and the first REFUSED is the load's refusal;
- the loader calls the late check exactly once, after the dev_layer sync and before the mappings are
  initialised; it WARNs for an unsupported model and for every device nothing was compared on, and throws
  a refusal.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEXT_CPP = (SRC / "llama-context.cpp").read_text()
MODEL_CPP = (SRC / "llama-model.cpp").read_text()
MEASURE_H = (SRC / "llama-load-measure.h").read_text()

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
_RUN = (
    "llama_load_measure_result llama_load_measure_run(const llama_model & model, llama_measure_context_args & args, "
    "const llama_measure_override_procs & procs, uint32_t n_ctx, uint64_t load_txn, "
    "enum ggml_sycl_measure_stage stage, int first_device)"
)
_LATE = (
    "llama_late_check_result llama_load_late_check(const llama_model & model, uint32_t n_ctx, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_measure_dummy_entry> & weights)"
)
_REFUSAL = (
    "inline std::string llama_load_measure_refusal_text(enum ggml_sycl_measure_stage stage, int device, "
    "const std::string & reason)"
)
_PARAMS = "inline llama_context_params llama_load_measure_context_params(uint32_t n_ctx, uint32_t n_ctx_train)"
_FOLD = (
    "inline llama_late_check_result llama_late_check_fold(const llama_sycl_l4_procs & procs, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_load_measure_device> & devices)"
)
_PEAK = "static void llama_context_peak_chunks(const sched_measure_buft & entry, llama_tenant_buft_caps & c)"


def measure_ok(code: str) -> bool:
    """The outer function: the backends go into `args`, the CPU last, then the one run."""
    b = function_body(code, _MEASURE)
    order = [
        "llama_measure_context_args args;",
        "llama_context_sycl_measure_backend_init(",
        "ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)",
        "args.backends.emplace_back(cpu);",
        "return llama_load_measure_run(model, args, llama_context_sycl_measure_override_procs(sycl_devs[0]), n_ctx, "
        "load_txn, stage, first_device);",
    ]
    pos = [b.find(z(t)) for t in order]
    if -1 in pos or pos != sorted(pos):
        return False
    # the context is built in the run, not here
    return "llama_context(" not in b and "holder" not in b and "llama_measure_plan_override" not in b


def run_ok(code: str) -> bool:
    b = function_body(code, _RUN)
    order = [
        "llama_measure_unsupported_reason(model)",
        "std::unique_ptr<llama_context> holder;",
        "llama_measure_plan_override guard(procs, load_txn, stage);",
        "if (!guard.installed())",
        "args.stage = stage;",
        "llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train)",
        "holder.reset(new llama_context(model, params, &args));",
        "catch (const llama_measure_unsupported & e)",
        "catch (const std::exception & e)",
        "status.status != sched_reserve_status::OK",
        "out.n_splits = holder->get_measure_plan().n_splits_max;",
    ]
    pos = [b.find(z(t)) for t in order]
    if -1 in pos or pos != sorted(pos):
        return False
    # `args` is a parameter: it outlives every local, so the backends go last
    if z("llama_measure_context_args") in b[b.index("{") :]:
        return False
    # unsupported is its own outcome, never a refusal
    if b.count(z("out.unsupported = true;")) != 2:
        return False
    # a refusal is never built outside the one text
    return b.count(z("out.refusal = ")) == b.count(z("llama_load_measure_refusal_text(")) and "[LOAD-PLAN]" not in b


def refusal_ok(code: str) -> bool:
    b = function_body(code, _REFUSAL)
    return (
        z('"[LOAD-PLAN] compute-slot measure failed at "') in b
        and z('" on device "') in b
        and z('" (refused)"') in b
        and all(z(f'return "{s}";') in code for s in ("probe", "admitted", "late"))
    )


def refusal_text_is_once() -> bool:
    n = 0
    for path in SRC.rglob("*"):
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}:
            n += _gate.strip_comments(path.read_text(errors="ignore")).count("compute-slot measure failed at")
    return n == 1


def params_ok(code: str) -> bool:
    b = function_body(code, _PARAMS)
    return (
        z("params.n_ubatch = llama_auto_ubatch_ladder[0];") in b
        and z("params.n_ctx = n_ctx != 0 ? n_ctx : n_ctx_train;") in b
        and z("params.n_batch = std::max(params.n_batch, params.n_ubatch);") in b
        and "512" not in b
        and '#include"llama-auto-ubatch.h"' in code
    )


def peak_ok(code: str) -> bool:
    """One per-chunk peak: the tenant caps read the helper, and no second reduction over `peaks` is written."""
    b = function_body(code, _PEAK)
    return (
        z("c.chunk_bytes = llama_measure_peak_per_chunk(entry.peaks);") in b
        and z("c.total = llama_measure_peak_total(c.chunk_bytes);") in b
        and "llama_context_worst_chunks" not in code
        and "for(constauto&graph:entry.peaks)" not in code
    )


def late_ok(code: str) -> bool:
    b = function_body(code, _LATE)
    order = [
        "if (!procs.available())",
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        "llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)",
        "if (!measured.ok) { (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal; return out; }",
        "llama_late_check_fold(procs, txn, measured.devices)",
    ]
    pos = [b.find(z(t)) for t in order]
    return -1 not in pos and pos == sorted(pos)


def fold_ok(code: str) -> bool:
    b = function_body(code, _FOLD)
    return (
        z("case GGML_SYCL_LATE_CHECK_NOT_RECORDED: out.not_recorded.push_back(d.device); break;") in b
        and z("case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED: out.shrunk.push_back(d.device); break;") in b
        and z("case GGML_SYCL_LATE_CHECK_EQUAL: break;") in b
        and z("case GGML_SYCL_LATE_CHECK_REFUSED:") in b
        and "default:" not in b
        and z("if (d.host) { continue; }") in b
        and b.count("return") == 2
        and z("llama_load_measure_refusal_text( GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, d.device,") in b
    )


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
    seg = c[call:prec]
    # an unsupported model is a WARN and the load goes on; a device nothing was compared on is a WARN; a
    # refusal is thrown. None of the three is a pass in disguise: the refusal is the only throw.
    return (
        "sycl_model_loading_guard.txn" in c[call : call + 400]
        and z("if (!late.unsupported.empty()) {") in seg
        and z("for (const int32_t device : late.not_recorded) {") in seg
        and "nothing was compared" in seg
        and z("if (!late.refusal.empty()) { throw std::runtime_error(late.refusal); }") in seg
        and seg.count("throw") == 1
        and "LLAMA_LOG_WARN" in seg
    )


def test_the_measure_builds_backends_and_hands_them_to_the_run():
    assert measure_ok(code_of(CONTEXT_CPP))


def test_the_run_orders_its_locals_for_unwind():
    assert run_ok(code_of(CONTEXT_CPP))


def test_run_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _RUN)
    for name, old, new in [
        ("the guard declared before the holder", "std::unique_ptr<llama_context> holder;", ""),
        ("the guard unchecked", "if (!guard.installed())", "if (false)"),
        ("stage not carried", "args.stage = stage;", ""),
        ("status unread", "status.status != sched_reserve_status::OK", "false"),
        ("unsupported read as a refusal", "catch (const llama_measure_unsupported & e) {\n        out.unsupported = true;\n        out.refusal", "catch (const llama_measure_unsupported & e) {\n        out.refusal"),
        ("a refusal outside the text", "out.refusal = llama_load_measure_refusal_text(stage, first_device, e.what());\n        return out;\n    }\n\n    const", "out.refusal = e.what();\n        return out;\n    }\n\n    const"),
        ("the shape hand-written", "llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train)", "llama_context_default_params()"),
    ]:
        assert not run_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_measure_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _MEASURE)
    for name, old, new in [
        ("cpu not last", "args.backends.emplace_back(cpu);", "args.backends.insert(args.backends.begin(), ggml_backend_ptr(cpu));"),
        ("the context built here", "return llama_load_measure_run(", "std::unique_ptr<llama_context> holder; return llama_load_measure_run("),
    ]:
        assert not measure_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_refusal_text_is_one():
    assert refusal_ok(code_of(MEASURE_H))
    assert refusal_text_is_once()
    code = code_of(MEASURE_H)
    b = function_body(code, _REFUSAL)
    assert not refusal_ok(code.replace(b, mutate(b, "(refused)", "(failed)"), 1))
    assert not refusal_ok(code.replace(z('return "late";'), z('return "later";'), 1))


def test_the_measure_shape_comes_from_the_ladder():
    assert params_ok(code_of(MEASURE_H))
    code = code_of(MEASURE_H)
    b = function_body(code, _PARAMS)
    assert not params_ok(code.replace(b, mutate(b, "llama_auto_ubatch_ladder[0]", "512"), 1))
    assert not params_ok(code.replace(b, mutate(b, "std::max(params.n_batch, params.n_ubatch)", "params.n_batch"), 1))


def test_one_per_chunk_peak():
    assert peak_ok(code_of(CONTEXT_CPP))
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _PEAK)
    assert not peak_ok(code.replace(b, mutate(b, "llama_measure_peak_total(c.chunk_bytes)", "0"), 1))
    assert not peak_ok(code + z("static void llama_context_worst_chunks();"))


def test_the_late_check_measures_only_with_l4_and_at_stage_c():
    assert late_ok(code_of(CONTEXT_CPP))


def test_late_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _LATE)
    for name, old, new in [
        ("no L4 gate", "if (!procs.available())", "if (false)"),
        ("measured at stage b", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)"),
        ("a failed measure ignored", "if (!measured.ok) {\n        (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal;\n        return out;\n    }", ""),
        ("the fold bypassed", "out = llama_late_check_fold(procs, txn, measured.devices);", ""),
        ("stand-ins after the measure", "llama_measure_dummy_scope dummies(weights);", ""),
    ]:
        assert not late_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_fold_keeps_not_recorded_apart():
    assert fold_ok(code_of(MEASURE_H))


def test_fold_mutants():
    code = code_of(MEASURE_H)
    b = function_body(code, _FOLD)
    for name, old, new in [
        ("not recorded read as equal", "case GGML_SYCL_LATE_CHECK_NOT_RECORDED:\n                out.not_recorded.push_back(d.device);\n                break;", "case GGML_SYCL_LATE_CHECK_NOT_RECORDED:\n                break;"),
        ("shrink lost", "out.shrunk.push_back(d.device);", ""),
        ("a default arm", "case GGML_SYCL_LATE_CHECK_EQUAL:\n                break;", "case GGML_SYCL_LATE_CHECK_EQUAL:\n                break;\n            default:\n                break;"),
        ("the host tier compared", "if (d.host) {\n            continue;\n        }", ""),
    ]:
        assert not fold_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_loader_calls_the_late_check_once_before_the_mappings():
    assert site_ok(code_of(MODEL_CPP))


def test_site_mutants():
    code = code_of(MODEL_CPP)
    assert not site_ok(code.replace(z("throw std::runtime_error(late.refusal);"), "", 1))
    assert not site_ok(code.replace(z("for (const int32_t device : late.not_recorded) {"), z("for (const int32_t device : late.shrunk) {"), 1))
    assert not site_ok(code.replace(z("if (!late.unsupported.empty()) {"), z("if (false) {"), 1))
    assert not site_ok(code + z("llama_load_late_check(*this, 0, sycl_model_loading_guard.txn, {});"))
    moved = code.replace(z("ml.init_mappings(true,"), z("llama_load_late_check(") + z("ml.init_mappings(true,"), 1)
    assert not site_ok(moved)
