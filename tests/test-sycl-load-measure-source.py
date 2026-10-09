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
  initialised; it WARNs for an unsupported model (the WARN pinned inside its own block) and, through one
  text helper that carries the ubatch and the device's measured compute term (the fold keeps it beside the
  device), for every device nothing was compared on, and throws a refusal;
- "this model needs ctx_other" is one predicate, `llama_model_needs_ctx_other`, used by the context
  constructor and by the unsupported reason alike;
- llama_load_measure asks the unsupported question before it creates any backend;
- `llama_late_check_result` carries only what production reads (no `checked`, no `shrunk`; the measured
  term of each not-recorded device is read by the WARN), and the ubatch reaches it through the fold, where a
  host test can see it.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
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
    "struct ggml_sycl_load_txn txn, const std::vector<llama_load_measure_device> & devices, uint32_t n_ubatch)"
)
_PEAK = (
    "inline void llama_tenant_caps_set_peaks(llama_tenant_buft_caps & c, "
    "const std::vector<std::vector<size_t>> & peaks)"
)


def measure_ok(code: str) -> bool:
    """The outer function: the backends go into `args`, the CPU last, then the one run."""
    b = function_body(code, _MEASURE)
    order = [
        "llama_measure_unsupported_reason(model)",
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
        # llama.cpp-p6i0: the params come first, so the override re-fits the plan's KV for their KV shape
        "const llama_context_params params = llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train);",
        "const ggml_sycl_measure_kv_shape kv_shape = llama_load_measure_kv_shape(params);",
        "llama_measure_plan_override guard(procs, load_txn, stage, &kv_shape);",
        "if (!guard.installed())",
        "args.stage = stage;",
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
        and z("params.n_ctx = llama_load_measure_n_ctx(n_ctx, n_ctx_train);") in b
        and z("params.n_batch = std::max(params.n_batch, params.n_ubatch);") in b
        and "512" not in b
        and '#include"llama-auto-ubatch.h"' in code
    )


def peak_ok(tenant_code: str, ctx_code: str) -> bool:
    """One per-chunk peak: the tenant caps read the helper (defined once, in llama-context-tenant.h, from the
    two llama-measure-plan.h functions), the context calls it for both tiers, and no second reduction over
    `peaks` is written."""
    b = function_body(tenant_code, _PEAK)
    return (
        z("c.chunk_bytes = llama_measure_peak_per_chunk(peaks);") in b
        and z("c.total = llama_measure_peak_total(c.chunk_bytes);") in b
        and ctx_code.count(z("llama_tenant_caps_set_peaks(c, entry.peaks);")) == 2
        and "llama_context_worst_chunks" not in ctx_code
        and "llama_context_peak_chunks" not in ctx_code
        and "for(constauto&graph:entry.peaks)" not in ctx_code
    )


def late_ok(code: str) -> bool:
    b = function_body(code, _LATE)
    order = [
        "if (!procs.available())",
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        "llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)",
        "if (!measured.ok) { (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal; return out; }",
        "llama_late_check_fold(procs, txn, measured.devices, n_ubatch)",
    ]
    pos = [b.find(z(t)) for t in order]
    return (
        -1 not in pos
        and pos == sorted(pos)
        and z("const uint32_t n_ubatch = llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train).n_ubatch;")
        in b
    )


def fold_ok(code: str) -> bool:
    b = function_body(code, _FOLD)
    return (
        z(
            "case GGML_SYCL_LATE_CHECK_NOT_RECORDED: out.not_recorded.push_back(d.device); "
            "out.not_recorded_bytes.push_back(d.total); break;"
        )
        in b
        and z("case GGML_SYCL_LATE_CHECK_EQUAL: case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED: break;") in b
        and z("case GGML_SYCL_LATE_CHECK_REFUSED:") in b
        and "default:" not in b
        and z("if (d.host) { continue; }") in b
        and z("out.n_ubatch = n_ubatch;") in b
        and b.count("return") == 2
        and z("llama_load_measure_refusal_text( GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, d.device,") in b
    )


def block_after(code: str, opener: str) -> str:
    """The brace block that follows `opener` (z-spelled), or ''."""
    i = code.find(z(opener))
    if i == -1:
        return ""
    j = code.index("{", i)
    depth = 0
    for k in range(j, len(code)):
        depth += (code[k] == "{") - (code[k] == "}")
        if depth == 0:
            return code[j : k + 1]
    return ""


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
    # an unsupported model is a WARN and the load goes on, the WARN inside its own block; a device nothing was
    # compared on is a WARN through the one text helper; a refusal is thrown. The refusal is the only throw.
    unsupported = block_after(seg, "if (!late.unsupported.empty())")
    not_recorded = block_after(seg, "for (size_t i = 0; i < late.not_recorded.size(); ++i)")
    return (
        "sycl_model_loading_guard.txn" in c[call : call + 400]
        and "LLAMA_LOG_WARN(" in unsupported
        and "unplanned path" in unsupported
        and "late.unsupported.c_str()" in unsupported
        and "throw" not in unsupported
        and "LLAMA_LOG_WARN(" in not_recorded
        and z("llama_late_check_not_recorded_text(late.not_recorded[i], late.n_ubatch, late.not_recorded_bytes[i])")
        in not_recorded
        and z("if (!late.refusal.empty()) { throw std::runtime_error(late.refusal); }") in seg
        and seg.count("throw") == 1
    )


_PROBE = (
    "llama_load_probe_result llama_load_probe_bound(const llama_model & model, uint32_t n_ctx, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_measure_dummy_entry> & weights)"
)
_ADMIT = (
    "llama_admitted_check_result llama_load_admitted_check(const llama_model & model, uint32_t n_ctx, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_measure_dummy_entry> & weights, "
    "const llama_load_probe_result & probe)"
)
_N_CTX = "inline uint32_t llama_load_measure_n_ctx(uint32_t n_ctx, uint32_t n_ctx_train)"


def probe_ok(code: str) -> bool:
    """Stage (a), llama.cpp-p6i0: C-hat is measured at PROBE over the weight stand-ins, and the measured devices are
    handed to llama_load_probe_reserve at the n_ctx the measure ran at (never the envelope's 0), which reserves each
    SYCL device's state then its compute term and skips the host tier (pinned by test-load-measure-guards)."""
    b = function_body(code, _PROBE)
    order = [
        "if (!procs.load_terms_available())",
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        "llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_PROBE)",
        "if (!measured.ok) { (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal; return out; }",
        "const uint32_t measured_n_ctx = llama_load_measure_n_ctx(n_ctx, model.hparams.n_ctx_train);",
        "out.not_reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);",
    ]
    pos = [b.find(z(t)) for t in order]
    return (
        -1 not in pos
        and pos == sorted(pos)
        and z("out.devices = measured.devices;") in b
        and z("out.measured = true;") in b
        and b.count(z("llama_load_probe_reserve(")) == 1
        # every reservation goes through the helper, so its order cannot be bypassed here
        and "reserve_compute_term" not in b
        and "reserve_state_term" not in b
        # the probe never writes the ledger: c(P) is recorded at the admitted placement only
        and "record" not in b
        and "GGML_SYCL_MEASURE_STAGE_CANDIDATE" not in b
    )


def admit_ok(code: str) -> bool:
    """Stage (b), llama.cpp-p6i0: c(P) is measured at the admitted placement, folded against the probe bound, and
    only an admitted result is recorded, at the measure's n_ctx."""
    b = function_body(code, _ADMIT)
    order = [
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        "llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)",
        "if (!measured.ok)",
        "const uint32_t measured_n_ctx = llama_load_measure_n_ctx(n_ctx, model.hparams.n_ctx_train);",
        "llama_admitted_check_fold(procs, probe.devices, probe.not_reserved, measured.devices, measured_n_ctx,",
        "if (!out.refusal.empty()) { return out; }",
        "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);",
    ]
    pos = [b.find(z(t)) for t in order]
    return (
        -1 not in pos
        and pos == sorted(pos)
        and b.count(z("llama_admitted_record(")) == 1
        and "GGML_SYCL_MEASURE_STAGE_PROBE" not in b
        and "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C" not in b
        # the admitted stage reserves nothing: the probe's reservation is the one it compares with
        and "reserve_compute_term" not in b
        and "reserve_term" not in b
        and "reserve_state" not in b
        and "llama_load_probe_reserve" not in b
    )


def n_ctx_ok(code: str) -> bool:
    """One n_ctx rule: the measure's params and the recorded/reserved n_ctx read the same helper."""
    b = function_body(code, _N_CTX)
    p = function_body(code, _PARAMS)
    return z("return n_ctx != 0 ? n_ctx : n_ctx_train;") in b and z(
        "params.n_ctx = llama_load_measure_n_ctx(n_ctx, n_ctx_train);"
    ) in p


def stage_order_ok(code: str) -> bool:
    """The loader's order (llama.cpp-p6i0): the tensors exist, the probe bound is measured and handed to the planner,
    the late inventory packs, c(P) is measured and recorded at the admitted placement, the dev_layer sync runs, and
    the late check compares. Each new call is made once, with the same n_ctx and transaction the late check passes,
    and each refusal is thrown."""
    c = code
    done = c.find(z("ml.done_getting_tensors();"))
    probe = c.find(z("llama_load_probe_bound("))
    late_inv = c.find(z("llama_model_sycl_set_late_inventory(ml, hparams, __func__);"))
    admit = c.find(z("llama_load_admitted_check("))
    sync = c.find(z("dev_layer sync: corrected"))
    late = c.find(z("llama_load_late_check("))
    if min(done, probe, late_inv, admit, sync, late) == -1:
        return False
    if not (done < probe < late_inv < admit < sync < late):
        return False
    if c.count(z("llama_load_probe_bound(")) != 1 or c.count(z("llama_load_admitted_check(")) != 1:
        return False
    if c.count(z("llama_model_sycl_set_late_inventory(ml, hparams, __func__);")) != 1:
        return False
    args = z("llama_model_sycl_make_placement_envelope().n_ctx, sycl_model_loading_guard.txn,")
    probe_seg = c[probe:late_inv]
    admit_seg = c[admit:sync]
    unsupported = block_after(probe_seg, "if (!probe.unsupported.empty())")
    return (
        args in c[probe : probe + 300]
        and args in c[admit : admit + 300]
        and z("admitted_weights, probe);") in c[admit : admit + 400]
        and "LLAMA_LOG_WARN(" in unsupported
        and "unplanned path" in unsupported
        and "throw" not in unsupported
        and z("if (!probe.refusal.empty()) { throw std::runtime_error(probe.refusal); }") in probe_seg
        and z("if (!admitted.refusal.empty()) { throw std::runtime_error(admitted.refusal); }") in admit_seg
        # stage (b) runs only when the probe measured: an unsupported model stays on master's unplanned path
        and z("if (probe.measured)") in c[late_inv:admit]
    )


_NOT_RECORDED_TEXT = (
    "inline std::string llama_late_check_not_recorded_text(int32_t device, uint32_t n_ubatch, size_t measured_bytes)"
)


def not_recorded_text_ok(code: str) -> bool:
    b = function_body(code, _NOT_RECORDED_TEXT)
    return (
        "nothing was compared" in b
        and "std::to_string(n_ubatch)" in b
        and "std::to_string(device)" in b
        # the measured term, in MiB with one decimal, reaches the returned text
        and z('std::snprintf(mib, sizeof(mib), "%.1f", measured_bytes / 1024.0 / 1024.0);') in b
        and z('"; measured compute term " + mib + " MiB on device "') in b
    )


def result_fields_ok(raw: str) -> bool:
    """llama_late_check_result carries what production reads and nothing it does not."""
    code = _gate.strip_comments(raw)
    i = code.index("struct llama_late_check_result {")
    body = code[i : code.index("};", i)]
    names = set(re.findall(r"(?:std::string|uint32_t|std::vector<int32_t>|std::vector<size_t>|bool)\s+(\w+)", body))
    return names == {"refusal", "unsupported", "n_ubatch", "not_recorded", "not_recorded_bytes"}


_NEEDS_OTHER = "bool llama_model_needs_ctx_other(const llama_model & model)"
_REASON = "std::string llama_measure_unsupported_reason(const llama_model & model)"


_CTOR_SIG = (
    "llama_context::llama_context(const llama_model & model, llama_context_params params, "
    "llama_measure_context_args * measure)"
)


def ctor_body(code: str) -> str:
    at = code.find(z(_CTOR_SIG))
    return block_after(code[at:], _CTOR_SIG) if at != -1 else ""


def needs_other_ok(code: str) -> bool:
    b = function_body(code, _NEEDS_OTHER)
    # the predicate's exact form: the assistant always, a draft arch only when it carries no token embedding
    # or no output of its own (test-measure-context checks the EAGLE3 and assistant answers behaviourally)
    if z(
        "return model.arch == LLM_ARCH_GEMMA4_ASSISTANT || ((model.arch == LLM_ARCH_EAGLE3 || "
        "model.arch == LLM_ARCH_DFLASH) && (model.tok_embd == nullptr || model.output == nullptr));"
    ) not in z(b):
        return False
    # the predicate is stated once: its arch names appear in neither user
    reason = function_body(code, _REASON)
    ctor = ctor_body(code)
    if not ctor:
        return False
    for user in (reason, ctor):
        if "LLM_ARCH_GEMMA4_ASSISTANT" in user or "LLM_ARCH_EAGLE3" in user:
            return False
        if z("llama_model_needs_ctx_other(model)") not in user:
            return False
    # the constructor's use of it: a missing ctx_other throws, and a given one is what the graph reads
    if z(
        "if (llama_model_needs_ctx_other(model)) { if (params.ctx_other == nullptr) { throw std::runtime_error("
    ) not in z(ctor):
        return False
    after = z(ctor).split(z("if (llama_model_needs_ctx_other(model)) {"), 1)[1]
    if z("cparams.ctx_other = params.ctx_other; }") not in after.split(z("if (params.ctx_other == nullptr) {"), 1)[1]:
        return False
    return True


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
        ("no KV shape for the re-fit", "guard(procs, load_txn, stage, &kv_shape);", "guard(procs, load_txn, stage, nullptr);"),
        ("the KV shape not the context's", "llama_load_measure_kv_shape(params);", "llama_load_measure_kv_shape(llama_context_default_params());"),
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


_KV_SHAPE = "inline struct ggml_sycl_measure_kv_shape llama_load_measure_kv_shape(const llama_context_params & params)"


def kv_shape_ok(code: str) -> bool:
    """llama.cpp-p6i0: the re-fit's KV shape is every KV field of the measure context's params, and nothing else."""
    b = function_body(code, _KV_SHAPE)
    want = [
        "shape.n_ctx = params.n_ctx;",
        "shape.n_ubatch = params.n_ubatch;",
        "shape.n_seq_max = params.n_seq_max;",
        "shape.kv_unified = params.kv_unified;",
        "shape.swa_full = params.swa_full;",
    ]
    return all(b.count(z(w)) == 1 for w in want) and b.count("shape.") == len(want)


def test_the_kv_shape_is_the_contexts():
    code = code_of(MEASURE_H)
    assert kv_shape_ok(code)
    b = function_body(code, _KV_SHAPE)
    for old, new in [
        ("shape.n_ctx = params.n_ctx;", "shape.n_ctx = 512;"),
        ("shape.swa_full = params.swa_full;", "shape.swa_full = false;"),
        ("shape.n_seq_max = params.n_seq_max;", "shape.n_seq_max = 1;"),
    ]:
        assert not kv_shape_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {old!r} slipped through"


def test_the_measure_shape_comes_from_the_ladder():
    assert params_ok(code_of(MEASURE_H))
    code = code_of(MEASURE_H)
    b = function_body(code, _PARAMS)
    assert not params_ok(code.replace(b, mutate(b, "llama_auto_ubatch_ladder[0]", "512"), 1))
    assert not params_ok(code.replace(b, mutate(b, "std::max(params.n_batch, params.n_ubatch)", "params.n_batch"), 1))


def test_one_per_chunk_peak():
    tenant = code_of((SRC / "llama-context-tenant.h").read_text())
    ctx = code_of(CONTEXT_CPP)
    assert peak_ok(tenant, ctx)
    b = function_body(tenant, _PEAK)
    assert not peak_ok(tenant.replace(b, mutate(b, "llama_measure_peak_total(c.chunk_bytes)", "0"), 1), ctx)
    assert not peak_ok(tenant, ctx + z("static void llama_context_worst_chunks();"))
    assert not peak_ok(tenant, ctx.replace(z("llama_tenant_caps_set_peaks(c, entry.peaks);"), "", 1))


def test_the_late_check_measures_only_with_l4_and_at_stage_c():
    assert late_ok(code_of(CONTEXT_CPP))


def test_late_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _LATE)
    for name, old, new in [
        ("no L4 gate", "if (!procs.available())", "if (false)"),
        ("measured at stage b", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)"),
        ("a failed measure ignored", "if (!measured.ok) {\n        (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal;\n        return out;\n    }", ""),
        ("the fold bypassed", "return llama_late_check_fold(procs, txn, measured.devices, n_ubatch);", "return out;"),
        ("stand-ins after the measure", "llama_measure_dummy_scope dummies(weights);", ""),
    ]:
        assert not late_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_fold_keeps_not_recorded_apart():
    assert fold_ok(code_of(MEASURE_H))


def test_fold_mutants():
    code = code_of(MEASURE_H)
    b = function_body(code, _FOLD)
    for name, old, new in [
        ("not recorded read as equal", "case GGML_SYCL_LATE_CHECK_NOT_RECORDED:\n                out.not_recorded.push_back(d.device);\n                out.not_recorded_bytes.push_back(d.total);\n                break;", "case GGML_SYCL_LATE_CHECK_NOT_RECORDED:\n                break;"),
        ("the measured term dropped", "out.not_recorded_bytes.push_back(d.total);", ""),
        ("shrink unhandled", "case GGML_SYCL_LATE_CHECK_EQUAL:\n            case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:", "case GGML_SYCL_LATE_CHECK_EQUAL:"),
        ("a default arm", "case GGML_SYCL_LATE_CHECK_REFUSED:", "default:\n                break;\n            case GGML_SYCL_LATE_CHECK_REFUSED:"),
        ("the host tier compared", "if (d.host) {\n            continue;\n        }", ""),
    ]:
        assert not fold_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_loader_calls_the_late_check_once_before_the_mappings():
    assert site_ok(code_of(MODEL_CPP))


def test_site_mutants():
    code = code_of(MODEL_CPP)
    assert not site_ok(code.replace(z("throw std::runtime_error(late.refusal);"), "", 1))
    assert not site_ok(code.replace(z("for (size_t i = 0; i < late.not_recorded.size(); ++i) {"), z("for (size_t i = 0; i < late.shrunk.size(); ++i) {"), 1))
    assert not site_ok(code.replace(z("if (!late.unsupported.empty()) {"), z("if (false) {"), 1))
    assert not site_ok(code + z("llama_load_late_check(*this, 0, sycl_model_loading_guard.txn, {});"))
    moved = code.replace(z("ml.init_mappings(true,"), z("llama_load_late_check(") + z("ml.init_mappings(true,"), 1)
    assert not site_ok(moved)


def test_the_unsupported_warn_is_pinned_in_its_block():
    assert site_ok(code_of(MODEL_CPP))
    code = code_of(MODEL_CPP)
    # the WARN replaced by nothing, with the sibling WARN still there
    old = z("if (!late.unsupported.empty()) {")
    blk = block_after(code, "if (!late.unsupported.empty())")
    assert "LLAMA_LOG_WARN(" in blk
    quiet = blk.replace("LLAMA_LOG_WARN(", "(void)sizeof(", 1)
    assert not site_ok(code.replace(blk, quiet, 1))
    # the not-recorded WARN with a hand-written text, losing the ubatch and the measured term
    nr = block_after(code, "for (size_t i = 0; i < late.not_recorded.size(); ++i)")
    call = z("llama_late_check_not_recorded_text(late.not_recorded[i], late.n_ubatch, late.not_recorded_bytes[i])")
    assert not site_ok(code.replace(nr, nr.replace(call, '"x"', 1), 1))
    # the measured term of another device
    wrong = z("llama_late_check_not_recorded_text(late.not_recorded[i], late.n_ubatch, late.not_recorded_bytes[0])")
    assert not site_ok(code.replace(nr, nr.replace(call, wrong, 1), 1))


def test_the_not_recorded_text_carries_the_ubatch():
    assert not_recorded_text_ok(code_of(MEASURE_H))
    code = code_of(MEASURE_H)
    b = function_body(code, _NOT_RECORDED_TEXT)
    assert not not_recorded_text_ok(code.replace(b, mutate(b, "std::to_string(n_ubatch)", '"0"'), 1))
    # the measured term left out of the text, or printed unscaled
    assert not not_recorded_text_ok(code.replace(b, mutate(b, '" + mib + "', ''), 1))
    assert not not_recorded_text_ok(code.replace(b, mutate(b, "measured_bytes / 1024.0 / 1024.0", "measured_bytes"), 1))


def test_the_result_carries_only_what_production_reads():
    assert result_fields_ok(MEASURE_H)
    assert not result_fields_ok(MEASURE_H.replace("std::vector<int32_t> not_recorded;", "std::vector<int32_t> not_recorded;\n    bool checked = false;", 1))
    assert not result_fields_ok(MEASURE_H.replace("std::vector<size_t>  not_recorded_bytes;", "", 1))


def test_the_fold_sets_the_ubatch_and_the_caller_passes_it():
    assert fold_ok(code_of(MEASURE_H))
    code = code_of(MEASURE_H)
    b = function_body(code, _FOLD)
    assert not fold_ok(code.replace(b, mutate(b, "out.n_ubatch = n_ubatch;", ""), 1))
    ctx = code_of(CONTEXT_CPP)
    lb = function_body(ctx, _LATE)
    assert not late_ok(ctx.replace(lb, mutate(lb, "llama_late_check_fold(procs, txn, measured.devices, n_ubatch)", "llama_late_check_fold(procs, txn, measured.devices, 0)"), 1))


def test_needs_ctx_other_is_one_predicate():
    assert needs_other_ok(code_of(CONTEXT_CPP))
    code = code_of(CONTEXT_CPP)
    reason = function_body(code, _REASON)
    assert not needs_other_ok(code.replace(reason, mutate(reason, "llama_model_needs_ctx_other(model)", "false"), 1))
    ctor = ctor_body(code)
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, "llama_model_needs_ctx_other(model)", "(model.arch == LLM_ARCH_EAGLE3)"), 1))
    pred = function_body(code, _NEEDS_OTHER)
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_GEMMA4_ASSISTANT", "LLM_ARCH_GEMMA4"), 1))
    # the inner or, the arch list and the outer or are each pinned
    assert not needs_other_ok(code.replace(pred, mutate(pred, "model.tok_embd == nullptr ||", "model.tok_embd == nullptr &&"), 1))
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_EAGLE3 ||", "LLM_ARCH_EAGLE3 &&"), 1))
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_GEMMA4_ASSISTANT ||", "LLM_ARCH_GEMMA4_ASSISTANT &&"), 1))
    # the constructor's throw and the ctx_other it hands the graph
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, "if (params.ctx_other == nullptr) {", "if (false) {"), 1))
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, "cparams.ctx_other = params.ctx_other;", "cparams.ctx_other = nullptr;"), 1))


def test_the_measure_asks_the_unsupported_question_before_any_backend():
    code = code_of(CONTEXT_CPP)
    assert measure_ok(code)
    b = function_body(code, _MEASURE)
    moved = mutate(b, "if (const std::string why = llama_measure_unsupported_reason(model); !why.empty()) {", "if (false) {")
    assert not measure_ok(code.replace(b, moved, 1))


def test_the_probe_bound_is_measured_and_reserved():
    assert probe_ok(code_of(CONTEXT_CPP))


def test_probe_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _PROBE)
    for name, old, new in [
        ("measured at the admitted stage", "GGML_SYCL_MEASURE_STAGE_PROBE)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)"),
        ("no reservation", "out.not_reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);", "(void) measured_n_ctx;"),
        ("the envelope's n_ctx reserved", "llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx)", "llama_load_probe_reserve(procs, txn, measured.devices, n_ctx)"),
        ("the declined devices dropped", "out.not_reserved = llama_load_probe_reserve(", "(void) llama_load_probe_reserve("),
        ("the compute term reserved past the helper", "out.not_reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);", "out.not_reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);\n    (void) llama_sycl_l4_reserve_compute_term(procs, txn, measured.devices[0], measured_n_ctx);"),
        ("no proc gate", "if (!procs.load_terms_available())", "if (false)"),
        ("stand-ins dropped", "llama_measure_dummy_scope dummies(weights);", ""),
    ]:
        assert not probe_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_admitted_term_is_checked_then_recorded():
    assert admit_ok(code_of(CONTEXT_CPP))


def test_admit_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _ADMIT)
    for name, old, new in [
        ("measured at the late stage", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C)"),
        ("recorded despite a refusal", "if (!out.refusal.empty()) {\n        return out;\n    }", ""),
        ("never recorded", "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);", ""),
        ("recorded at n_ctx 0", "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);", "out.n_recorded = llama_admitted_record(procs, txn, out, n_ctx);"),
        ("the fold bypassed", "llama_admitted_check_fold(procs, probe.devices, probe.not_reserved, measured.devices,", "llama_admitted_check_fold(procs, measured.devices, probe.not_reserved, measured.devices,"),
        ("the admitted stage reserves", "    if (!out.refusal.empty()) {\n        return out;\n    }", "    (void) llama_sycl_l4_reserve_compute_term(procs, txn, measured.devices[0], measured_n_ctx);\n    if (!out.refusal.empty()) {\n        return out;\n    }"),
        ("unreserved devices compared", "llama_admitted_check_fold(procs, probe.devices, probe.not_reserved, measured.devices,", "llama_admitted_check_fold(procs, probe.devices, {}, measured.devices,"),
        ("the admitted stage reserves state", "    if (!out.refusal.empty()) {\n        return out;\n    }", "    (void) llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);\n    if (!out.refusal.empty()) {\n        return out;\n    }"),
    ]:
        assert not admit_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_one_n_ctx_rule():
    code = code_of(MEASURE_H)
    assert n_ctx_ok(code)
    p = function_body(code, _PARAMS)
    assert not n_ctx_ok(code.replace(p, mutate(p, "llama_load_measure_n_ctx(n_ctx, n_ctx_train)", "n_ctx"), 1))
    b = function_body(code, _N_CTX)
    assert not n_ctx_ok(code.replace(b, mutate(b, "return n_ctx != 0 ? n_ctx : n_ctx_train;", "return n_ctx;"), 1))


def test_the_loader_orders_probe_pack_admit_sync_late():
    assert stage_order_ok(code_of(MODEL_CPP))


def test_stage_order_mutants():
    code = code_of(MODEL_CPP)
    probe_call_start = code.find(z("llama_load_probe_bound("))
    admit_call_start = code.find(z("llama_load_admitted_check("))
    assert probe_call_start != -1 and admit_call_start != -1
    late_inv = z("llama_model_sycl_set_late_inventory(ml, hparams, __func__);")
    # the probe after the pack: the late inventory moved in front of the probe
    moved_pack = code.replace(late_inv, "", 1)
    moved_pack = moved_pack.replace(z("ml.done_getting_tensors();"), z("ml.done_getting_tensors();") + late_inv, 1)
    assert not stage_order_ok(moved_pack), "the pack before the probe slipped through"
    # the admitted measure after the dev_layer sync
    sync = z("dev_layer sync: corrected")
    moved_admit = code.replace(z("llama_load_admitted_check("), z("llama_load_admitted_check_x("), 1)
    moved_admit = moved_admit.replace(sync, sync + z("llama_load_admitted_check("), 1)
    assert not stage_order_ok(moved_admit), "the admitted measure after the sync slipped through"
    # a refusal swallowed
    assert not stage_order_ok(code.replace(z("throw std::runtime_error(probe.refusal);"), "", 1))
    assert not stage_order_ok(code.replace(z("throw std::runtime_error(admitted.refusal);"), "", 1))
    # the admitted check run for a model the probe could not measure
    assert not stage_order_ok(code.replace(z("if (probe.measured)"), z("if (true)"), 1))
    # a second probe call
    assert not stage_order_ok(code + z("llama_load_probe_bound(*this, 0, sycl_model_loading_guard.txn, w);"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
