"""Source gate for the load-time measure and its late-check call site (zhcn design 2.10, C7h-3, and the
C7 review folds I2, I3, I4, I5, I6, M1, M3).

`llama_load_measure` builds the measure backends and hands them to `llama_load_measure_run`, which builds
one measure-only context over a load's placement. This gate pins, on comment-stripped text:

- the unwind order of the run's locals: the caller's `args` (with the measure backends, the CPU backend last)
  outlive the context holder, then the plan-override guard, with the guard installed (and checked) before
  the context is constructed, so on every exit the override clears first, then the context goes, then the
  backends; a `llama_measure_unsupported` is caught before the generic catch and is `unsupported`, not a
  refusal;
- the run asks the backend whether a plan is staged for its stage (`llama_load_measure_plan_absent`) before it
  builds the guard, and a stage with none is `unsupported`, so the install never refuses a load for a plan the
  early pass could not stage;
- every refusal goes through the one `[LOAD-PLAN] compute-slot measure failed at ... (refused)` text, which
  occurs once in src/;
- the shape is the one the caller's hints resolve to (`llama_load_measure_shape_resolve`): the context and
  micro-batch the caller told the load, the training context and the auto-ubatch ladder's bottom rung (no literal
  512) where it told none; the loader resolves it once per measure scope, prints it on the compute-slot line, and
  sends the same resolver's micro-batch to the backend in the inventory;
- the compute term is the one per-chunk peak helper of llama-measure-plan.h (no second reduction);
- the probe, admitted and late stages measure through one helper, `llama_load_stage_measure`, which puts the
  weight stand-ins up before the measure and logs its trace, and split a failed measure through one
  `llama_load_stage_fail`; each stage keeps only its gate and its fold;
- `llama_load_late_check` measures only when the backend exports the L4 entry points, measures at stage (c),
  and hands the devices to the one fold, in which
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
  host test can see it;
- (llama.cpp-p6i0) every RUNTIME-first consumer at context time is named in the reserve, and the recurrent state
  is one: the measure reads the context memory's size per buffer type (memory_breakdown) into each SYCL device's
  caps, the run carries it on the measured device and never on the host tier, the probe reserve hands it to the
  state reservation before the compute term and lists each reserved state apart from the declined compute terms, the
  admitted stage marks and records each reserved state under its own name whatever its compute term did, and the
  late fold checks it with the state bytes, never the compute total;
- (llama.cpp-p6i0) the load measures that state at n_seq_max 1, so the context constructor compares, once and right
  after its memory exists, each SYCL device's state (the memory's bytes for that device's buffer type) with the
  planned state term the backend holds, and WARNs by name through one text helper when the state is larger, on a
  line of its own when the load planned no state term at all. It is never a refusal: neither the comparison nor the helper throws, returns a failure, or reaches the scheduler.

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
    "llama_load_measure_result llama_load_measure(const llama_model & model, const llama_load_measure_shape & shape, "
    "uint64_t load_txn, enum ggml_sycl_measure_stage stage)"
)
_RUN = (
    "llama_load_measure_result llama_load_measure_run(const llama_model & model, llama_measure_context_args & args, "
    "const llama_measure_override_procs & procs, const llama_load_measure_shape & shape, uint64_t load_txn, "
    "enum ggml_sycl_measure_stage stage, int first_device)"
)
_PLAN_ABSENT_BLOCK = (
    "if (const char * why = llama_load_measure_plan_absent(procs, load_txn, stage); why != nullptr) { "
    "out.unsupported = true; out.refusal = llama_load_measure_refusal_text(stage, first_device, why); return out; }"
)
_LATE = (
    "llama_late_check_result llama_load_late_check(const llama_model & model, const llama_load_measure_shape & shape, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_measure_dummy_entry> & weights)"
)
_STAGE_MEASURE = (
    "static llama_load_measure_result llama_load_stage_measure(const llama_model & model, "
    "const llama_load_measure_shape & shape, uint64_t load_txn, enum ggml_sycl_measure_stage stage, "
    "const std::vector<llama_measure_dummy_entry> & weights)"
)
_STAGE_FAIL = (
    "static void llama_load_stage_fail(const llama_load_measure_result & measured, std::string & unsupported, "
    "std::string & refusal)"
)
# every stage's procs, and every stage's failed measure, go through the one helper each
_STAGE_PROCS = "const llama_sycl_l4_procs procs = llama_load_stage_procs(model);"
_STAGE_FAILED = "if (!measured.ok) { llama_load_stage_fail(measured, out.unsupported, out.refusal); return out; }"
_REFUSAL = (
    "inline std::string llama_load_measure_refusal_text(enum ggml_sycl_measure_stage stage, int device, "
    "const std::string & reason)"
)
_PARAMS = "inline llama_context_params llama_load_measure_context_params(const llama_load_measure_shape & shape)"
_SHAPE = (
    "inline llama_load_measure_shape llama_load_measure_shape_resolve(uint32_t n_ctx_hint, uint32_t n_ubatch_hint, "
    "uint32_t n_ctx_train)"
)
_SHAPE_FALLBACK = "const uint32_t n_ubatch = n_ubatch_hint != 0 ? n_ubatch_hint : llama_auto_ubatch_ladder[0];"
_SHAPE_N_CTX = "const uint32_t n_ctx = llama_load_measure_n_ctx(named ? n_ctx_hint : 0, n_ctx_train);"
# the loader's one shape per measure scope, from the model's hints
_MODEL_SHAPE = (
    "static llama_load_measure_shape llama_model_load_measure_shape(const llama_model & model)"
)
_MEASURE_SHAPE_LOCAL = "const llama_load_measure_shape measure_shape = llama_model_load_measure_shape(*this);"
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
        "return llama_load_measure_run(model, args, llama_context_sycl_measure_override_procs(sycl_devs[0]), shape, "
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
        # a stage with no plan staged is unsupported, asked before the guard so the install never refuses it
        _PLAN_ABSENT_BLOCK,
        "std::unique_ptr<llama_context> holder;",
        # llama.cpp-p6i0: the params come first, so the override re-fits the plan's KV for their KV shape
        "const llama_context_params params = llama_load_measure_context_params(shape);",
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
    if b.count(z("out.unsupported = true;")) != 3:
        return False
    # a refusal is never built outside the one text
    return b.count(z("out.refusal = ")) == b.count(z("llama_load_measure_refusal_text(")) and "[LOAD-PLAN]" not in b


_PLAN_ABSENT = (
    "inline const char * llama_load_measure_plan_absent(const llama_measure_override_procs & procs, "
    "uint64_t load_txn, enum ggml_sycl_measure_stage stage)"
)
_OVERRIDE_PROCS = "llama_measure_override_procs llama_context_sycl_measure_override_procs(ggml_backend_dev_t dev)"


def plan_absent_ok(code: str) -> bool:
    """The query is asked only when the backend has it: a backend without it leaves the install to answer."""
    b = function_body(code, _PLAN_ABSENT)
    return (
        z("if (procs.staged == nullptr || procs.staged(load_txn, stage)) { return nullptr; }") in b
        and b.count("procs.install") == 0
        and z('"no placement plan is staged for this load at this stage') in b
    )


def override_procs_ok(code: str, sycl: str) -> bool:
    """Both builds take the query from the backend, the DL build by the name the backend's proc table answers."""
    b = function_body(code, _OVERRIDE_PROCS)
    return (
        z("procs.staged = &ggml_backend_sycl_measure_plan_staged;") in b
        and z('llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_measure_plan_staged")') in b
        and z('if (strcmp(name, "ggml_backend_sycl_measure_plan_staged") == 0) '
              "{ return (void *) ggml_backend_sycl_measure_plan_staged; }") in sycl
    )


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
    """The measure's params are the resolved shape, and the shape falls back to the ladder's bottom rung (from the
    helper the real context's trial reads), never a literal 512."""
    b = function_body(code, _PARAMS)
    s = function_body(code, _SHAPE)
    return (
        z("params.n_ubatch = shape.n_ubatch;") in b
        and z("params.n_ctx = shape.n_ctx;") in b
        and z("params.n_batch = std::max(params.n_batch, params.n_ubatch);") in b
        and "512" not in b
        and z(_SHAPE_FALLBACK) in s
        and "512" not in s
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


def stage_measure_ok(code: str) -> bool:
    """The one stage measure: the stand-ins go up before the measure (a refused one refuses by name), the measure
    runs at the caller's stage and txn, its trace is logged, and every stage calls it rather than its own scaffold.
    The failure split sends only an unsupported measure to `unsupported`."""
    b = function_body(code, _STAGE_MEASURE)
    order = [
        "llama_measure_dummy_scope dummies(weights);",
        "if (dummies.failed())",
        'out.refusal = llama_load_measure_refusal_text(stage, -1, "a weight stand-in buffer was refused"); return out;',
        "llama_load_measure_result measured = llama_load_measure(model, shape, load_txn, stage);",
        "llama_load_measure_log_trace(measured);",
        "return measured;",
    ]
    pos = [b.find(z(t)) for t in order]
    fail = function_body(code, _STAGE_FAIL)
    stages = [function_body(code, sig) for sig in (_PROBE, _ADMIT, _LATE)]
    return (
        -1 not in pos
        and pos == sorted(pos)
        and fail.endswith("{" + z("(measured.unsupported ? unsupported : refusal) = measured.refusal;") + "}")
        and all(
            z(_STAGE_PROCS) in st
            and st.count(z("llama_load_stage_measure(")) == 1
            and z(_STAGE_FAILED) in st
            and "llama_measure_dummy_scope" not in st
            and z("llama_load_measure(") not in st
            and "llama_load_measure_log_trace" not in st
            and "llama_context_sycl_l4_procs_for_dev" not in st
            for st in stages
        )
    )


def late_ok(code: str) -> bool:
    b = function_body(code, _LATE)
    order = [
        _STAGE_PROCS,
        "if (!procs.available())",
        "llama_load_stage_measure(model, shape, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, weights)",
        _STAGE_FAILED,
        "llama_late_check_fold(procs, txn, measured.devices, n_ubatch)",
    ]
    pos = [b.find(z(t)) for t in order]
    return -1 not in pos and pos == sorted(pos) and z("const uint32_t n_ubatch = shape.n_ubatch;") in b


def fold_ok(code: str) -> bool:
    b = function_body(code, _FOLD)
    return (
        z(
            "case GGML_SYCL_LATE_CHECK_NOT_RECORDED: out.not_recorded.push_back(d.device); "
            "out.not_recorded_bytes.push_back(d.total); break;"
        )
        in b
        # each switch, the compute term's and the state term's, admits EQUAL and SHRINK and handles REFUSED itself
        and z(
            "switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) { case GGML_SYCL_LATE_CHECK_EQUAL: "
            "case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED: break; case GGML_SYCL_LATE_CHECK_REFUSED:"
        )
        in b
        and z(
            "switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) { "
            "case GGML_SYCL_LATE_CHECK_EQUAL: case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED: break; "
            "case GGML_SYCL_LATE_CHECK_REFUSED:"
        )
        in b
        and "default:" not in b
        and z("if (d.host) { continue; }") in b
        and z("out.n_ubatch = n_ubatch;") in b
        # the compute refusal and the state refusal (llama.cpp-p6i0) each end the fold; the end returns the rest
        and b.count("return") == 3
        and z("llama_load_measure_refusal_text( GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, d.device,") in b
        and z('"the final placement needs more recurrent state than the reserved state term"); return out;') in b
        # a state nothing was compared with is its own list, never a compute miss and never a pass
        and z(
            "case GGML_SYCL_LATE_CHECK_NOT_RECORDED: out.state_not_recorded.push_back(d.device); "
            "out.state_not_recorded_bytes.push_back(d.state_bytes); break;"
        )
        in b
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
    maps = c.find(z("ml.init_mappings(prefetch_mappings,"))
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
        # true whichever stage was skipped before it: the weights may be placed, the late compute term is not compared
        and "the compute slot is not checked at load" in unsupported
        and "unplanned path" not in unsupported
        and "late.unsupported.c_str()" in unsupported
        and "throw" not in unsupported
        and "LLAMA_LOG_WARN(" in not_recorded
        and z("llama_late_check_not_recorded_text(late.not_recorded[i], late.n_ubatch, late.not_recorded_bytes[i])")
        in not_recorded
        and z("if (!late.refusal.empty()) { throw std::runtime_error(late.refusal); }") in seg
        and seg.count("throw") == 1
    )


_PROBE = (
    "llama_load_probe_result llama_load_probe_bound(const llama_model & model, const llama_load_measure_shape & shape, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_measure_dummy_entry> & weights)"
)
_ADMIT = (
    "llama_admitted_check_result llama_load_admitted_check(const llama_model & model, "
    "const llama_load_measure_shape & shape, struct ggml_sycl_load_txn txn, "
    "const std::vector<llama_measure_dummy_entry> & weights, const llama_load_probe_result & probe)"
)
_N_CTX = "inline uint32_t llama_load_measure_n_ctx(uint32_t n_ctx, uint32_t n_ctx_train)"


def probe_ok(code: str) -> bool:
    """Stage (a), llama.cpp-p6i0: C-hat is measured at PROBE over the weight stand-ins, and the measured devices are
    handed to llama_load_probe_reserve at the shape's n_ctx, the one the measure ran at, which reserves each SYCL
    device's state then its compute term and skips the host tier (pinned by test-load-measure-guards)."""
    b = function_body(code, _PROBE)
    order = [
        _STAGE_PROCS,
        "if (!procs.available() || !procs.load_terms_available())",
        "llama_load_stage_measure(model, shape, txn.id, GGML_SYCL_MEASURE_STAGE_PROBE, weights)",
        _STAGE_FAILED,
        "const uint32_t measured_n_ctx = shape.n_ctx;",
        "llama_load_probe_reservations reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);",
        # both outcomes kept: the declined compute terms and, apart from them, the reserved states
        "out.compute_declined = std::move(reserved.compute_declined);",
        "out.state_reserved = std::move(reserved.state_reserved);",
    ]
    pos = [b.find(z(t)) for t in order]
    return (
        -1 not in pos
        and pos == sorted(pos)
        and z("out.devices = measured.devices;") in b
        and z("out.measured = true;") in b
        # the admitted fold compares the residency the probe saw with its own
        and z("out.kv = measured.kv;") in b
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
    only an admitted result is recorded, at the shape's n_ctx and micro-batch, the ones the measure ran at."""
    b = function_body(code, _ADMIT)
    order = [
        _STAGE_PROCS,
        "if (!procs.available() || !procs.load_terms_available())",
        "llama_load_stage_measure(model, shape, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, weights)",
        _STAGE_FAILED,
        "const uint32_t measured_n_ctx = shape.n_ctx;",
        "const uint32_t n_ubatch = shape.n_ubatch;",
        "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.state_reserved, measured.devices, measured_n_ctx,",
        "if (!out.refusal.empty()) { return out; }",
        "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);",
        # llama.cpp-p6i0: the reserved state under its own name, at the same n_ctx
        "out.n_state_recorded = llama_admitted_record_state(procs, txn, out, measured_n_ctx);",
    ]
    pos = [b.find(z(t)) for t in order]
    return (
        -1 not in pos
        and pos == sorted(pos)
        # the KV-residency delta is judged on the residencies the two measures saw, probe first
        and z("measured_n_ctx, n_ubatch, probe.kv, measured.kv);") in b
        and b.count(z("llama_admitted_record(")) == 1
        and b.count(z("llama_admitted_record_state(")) == 1
        and "GGML_SYCL_MEASURE_STAGE_PROBE" not in b
        and "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C" not in b
        # the admitted stage reserves nothing: the probe's reservation is the one it compares with
        and "reserve_compute_term" not in b
        and "reserve_term" not in b
        and "reserve_state" not in b
        and "llama_load_probe_reserve" not in b
    )


def n_ctx_ok(code: str) -> bool:
    """One n_ctx rule: the measure's params read the shape, and the shape reads the training-context fallback."""
    b = function_body(code, _N_CTX)
    s = function_body(code, _SHAPE)
    p = function_body(code, _PARAMS)
    return (
        z("return n_ctx != 0 ? n_ctx : n_ctx_train;") in b
        and z(_SHAPE_N_CTX) in s
        and z("params.n_ctx = shape.n_ctx;") in p
    )


def stage_order_ok(code: str) -> bool:
    """The loader's order (llama.cpp-p6i0): the tensors exist, the probe bound is measured and handed to the planner,
    the late inventory packs, c(P) is measured and recorded at the admitted placement, the dev_layer sync runs, and
    the late check compares. Each new call is made once, with the same shape and transaction the late check passes,
    and each refusal is thrown. The shape is resolved once per measure scope from the model's hints, never the
    placement envelope's n_ctx of 0."""
    c = code
    done = c.find(z("ml.done_getting_tensors();"))
    probe = c.find(z("llama_load_probe_bound("))
    late_inv = c.find(z("llama_model_sycl_set_late_inventory(ml, *this, use_mlock, __func__);"))
    admit = c.find(z("llama_load_admitted_check("))
    sync = c.find(z("dev_layer sync: corrected"))
    late = c.find(z("llama_load_late_check("))
    if min(done, probe, late_inv, admit, sync, late) == -1:
        return False
    if not (done < probe < late_inv < admit < sync < late):
        return False
    if c.count(z("llama_load_probe_bound(")) != 1 or c.count(z("llama_load_admitted_check(")) != 1:
        return False
    if c.count(z("llama_model_sycl_set_late_inventory(ml, *this, use_mlock, __func__);")) != 1:
        return False
    args = z("measure_shape, sycl_model_loading_guard.txn,")
    probe_seg = c[probe:late_inv]
    admit_seg = c[admit:sync]
    unsupported = block_after(probe_seg, "if (!probe.unsupported.empty())")
    # one local shape for the probe and admitted calls, one for the late call, both from the one model helper
    shape_local = z(_MEASURE_SHAPE_LOCAL)
    first_shape = c.find(shape_local)
    return (
        c.count(shape_local) == 2
        and -1 < first_shape < probe
        and c.find(shape_local, first_shape + 1) in range(sync, late)
        and z("llama_model_sycl_make_placement_envelope().n_ctx, sycl_model_loading_guard.txn") not in c
        and args in c[probe : probe + 300]
        and args in c[admit : admit + 300]
        and args in c[late : late + 300]
        and z("admitted_weights, probe);") in c[admit : admit + 400]
        and "LLAMA_LOG_WARN(" in unsupported
        # the weights may still be placed by the late plan; only the compute slot goes unreserved
        and "the compute slot is not planned at load" in unsupported
        and "unplanned path" not in unsupported
        and "throw" not in unsupported
        and z("if (!probe.refusal.empty()) { throw std::runtime_error(probe.refusal); }") in probe_seg
        and z("if (!admitted.refusal.empty()) { throw std::runtime_error(admitted.refusal); }") in admit_seg
        # stage (b) runs only when the probe measured: an unsupported probe reserves no compute term to admit
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
    return names == {"refusal", "unsupported", "n_ubatch", "not_recorded", "not_recorded_bytes", "state_not_recorded",
                     "state_not_recorded_bytes"}


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
    # the write closes the predicate's own block, right after the throw: a later block (upstream's DFlash one also
    # writes ctx_other) must not satisfy it
    after = z(ctor).split(z("if (llama_model_needs_ctx_other(model)) {"), 1)[1]
    block = after.split(z("if (params.ctx_other == nullptr) {"), 1)[1]
    if not re.match(r"throw std::runtime_error\([^;]*\);\}" + re.escape(z("cparams.ctx_other = params.ctx_other; }")), block):
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
        ("the shape hand-written", "llama_load_measure_context_params(shape)", "llama_context_default_params()"),
        ("no KV shape for the re-fit", "guard(procs, load_txn, stage, &kv_shape);", "guard(procs, load_txn, stage, nullptr);"),
        ("the KV shape not the context's", "llama_load_measure_kv_shape(params);", "llama_load_measure_kv_shape(llama_context_default_params());"),
        ("no staged-plan query", _PLAN_ABSENT_BLOCK, ""),
        ("a missing plan read as a refusal", _PLAN_ABSENT_BLOCK, _PLAN_ABSENT_BLOCK.replace("out.unsupported = true; ", "")),
    ]:
        assert not run_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    # the query asked after the guard, where the install has already refused
    moved = mutate(b, _PLAN_ABSENT_BLOCK, "")
    moved = moved.replace(z("args.stage = stage;"), z(_PLAN_ABSENT_BLOCK) + z("args.stage = stage;"), 1)
    assert not run_ok(code.replace(b, moved, 1)), "mutant 'the query after the guard' slipped through"


def test_the_plan_absent_query_defers_to_a_backend_without_it():
    code = code_of(MEASURE_H)
    assert plan_absent_ok(code)
    b = function_body(code, _PLAN_ABSENT)
    for name, old, new in [
        ("a backend without the query read as having no plan", "procs.staged == nullptr ||", "procs.staged != nullptr &&"),
        ("the query's answer ignored", "|| procs.staged(load_txn, stage)", ""),
        ("the install asked instead", "procs.staged(load_txn, stage)", "procs.install(load_txn, stage, nullptr)"),
    ]:
        assert not plan_absent_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_both_builds_take_the_query_from_the_backend():
    code = code_of(CONTEXT_CPP)
    sycl = code_of((ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text())
    assert override_procs_ok(code, sycl)
    b = function_body(code, _OVERRIDE_PROCS)
    for name, old, new in [
        ("the static build without the query", "procs.staged = &ggml_backend_sycl_measure_plan_staged;", ""),
        ("the DL build looking up another name", '"ggml_backend_sycl_measure_plan_staged"', '"ggml_backend_sycl_measure_plan_staged_v0"'),
    ]:
        assert not override_procs_ok(code.replace(b, mutate(b, old, new), 1), sycl), f"mutant {name!r} slipped through"
    entry = z('if (strcmp(name, "ggml_backend_sycl_measure_plan_staged") == 0) '
              "{ return (void *) ggml_backend_sycl_measure_plan_staged; }")
    assert not override_procs_ok(code, sycl.replace(entry, "", 1)), "mutant 'the proc table without the name' slipped through"


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
    s = function_body(code, _SHAPE)
    assert not params_ok(code.replace(s, mutate(s, "llama_auto_ubatch_ladder[0]", "512"), 1))
    assert not params_ok(code.replace(b, mutate(b, "std::max(params.n_batch, params.n_ubatch)", "params.n_batch"), 1))
    # the caller's micro-batch dropped for the fixed rung the measure ran at before the shape reached it
    assert not params_ok(code.replace(b, mutate(b, "params.n_ubatch = shape.n_ubatch;",
                                                "params.n_ubatch = llama_auto_ubatch_ladder[0];"), 1))


def test_one_per_chunk_peak():
    tenant = code_of((SRC / "llama-context-tenant.h").read_text())
    ctx = code_of(CONTEXT_CPP)
    assert peak_ok(tenant, ctx)
    b = function_body(tenant, _PEAK)
    assert not peak_ok(tenant.replace(b, mutate(b, "llama_measure_peak_total(c.chunk_bytes)", "0"), 1), ctx)
    assert not peak_ok(tenant, ctx + z("static void llama_context_worst_chunks();"))
    assert not peak_ok(tenant, ctx.replace(z("llama_tenant_caps_set_peaks(c, entry.peaks);"), "", 1))


def test_the_stages_share_one_measure():
    assert stage_measure_ok(code_of(CONTEXT_CPP))


def test_stage_measure_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _STAGE_MEASURE)
    for name, old, new in [
        ("stand-ins dropped", "llama_measure_dummy_scope dummies(weights);", ""),
        ("a refused stand-in ignored", "if (dummies.failed()) {", "if (false) {"),
        ("the measure at a fixed stage", "llama_load_measure(model, shape, load_txn, stage);", "llama_load_measure(model, shape, load_txn, GGML_SYCL_MEASURE_STAGE_PROBE);"),
        ("the trace not logged", "llama_load_measure_log_trace(measured);", ""),
    ]:
        assert not stage_measure_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    f = function_body(code, _STAGE_FAIL)
    swapped = mutate(f, "measured.unsupported ? unsupported : refusal", "measured.unsupported ? refusal : unsupported")
    assert not stage_measure_ok(code.replace(f, swapped, 1)), "a swapped failure split slipped through"
    # a stage that rebuilds its own scaffold instead of calling the helper
    late = function_body(code, _LATE)
    own = mutate(late, "const llama_sycl_l4_procs procs = llama_load_stage_procs(model);", "const llama_sycl_l4_procs procs = llama_load_stage_procs(model);\n    llama_measure_dummy_scope dummies(weights);")
    assert not stage_measure_ok(code.replace(late, own, 1)), "a stage's own stand-in scope slipped through"
    admit = function_body(code, _ADMIT)
    lookup = mutate(admit, "llama_load_stage_procs(model);", "llama_context_sycl_l4_procs_for_dev(model.devices[0].dev);")
    assert not stage_measure_ok(code.replace(admit, lookup, 1)), "a stage's own procs lookup slipped through"


_WEIGHTS = "static std::vector<llama_measure_dummy_entry> llama_model_measure_weights(const llama_model_loader & ml)"


def weights_ok(code: str) -> bool:
    """The loader builds the stages' stand-in list in one helper, over every (buffer type, context), and each of
    the three stages takes its list from it."""
    b = function_body(code, _WEIGHTS)
    return (
        z("for (const auto & [ctx_key, ctx_ptr] : ml.ctx_map) { weights.push_back({ ctx_key.buft, ctx_ptr.get() }); }")
        in b
        and code.count(z(".push_back({ ctx_key.buft, ctx_ptr.get() });")) == 1
        and all(
            z(f"const std::vector<llama_measure_dummy_entry> {v} = llama_model_measure_weights(ml);") in code
            for v in ("probe_weights", "admitted_weights", "late_weights")
        )
    )


def test_the_stages_share_one_stand_in_list():
    assert weights_ok(code_of(MODEL_CPP))


def test_weights_mutants():
    code = code_of(MODEL_CPP)
    b = function_body(code, _WEIGHTS)
    assert not weights_ok(code.replace(b, mutate(b, "weights.push_back({ ctx_key.buft, ctx_ptr.get() });", ""), 1))
    copy = mutate(
        code,
        "const std::vector<llama_measure_dummy_entry> late_weights = llama_model_measure_weights(ml);",
        "std::vector<llama_measure_dummy_entry> late_weights;\n"
        "for (const auto & [ctx_key, ctx_ptr] : ml.ctx_map) { late_weights.push_back({ ctx_key.buft, ctx_ptr.get() }); }",
    )
    assert not weights_ok(copy), "a stage's own copy of the stand-in list slipped through"


def test_the_late_check_measures_only_with_l4_and_at_stage_c():
    assert late_ok(code_of(CONTEXT_CPP))


def test_late_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _LATE)
    for name, old, new in [
        ("no L4 gate", "if (!procs.available())", "if (false)"),
        ("measured at stage b", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, weights)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, weights)"),
        ("a failed measure ignored", "if (!measured.ok) {\n        llama_load_stage_fail(measured, out.unsupported, out.refusal);\n        return out;\n    }", ""),
        ("the fold bypassed", "return llama_late_check_fold(procs, txn, measured.devices, n_ubatch);", "return out;"),
        ("no procs", "const llama_sycl_l4_procs procs = llama_load_stage_procs(model);", "const llama_sycl_l4_procs procs = {};"),
        ("the fold given the ladder's rung, not the shape's", "const uint32_t n_ubatch = shape.n_ubatch;", "const uint32_t n_ubatch = llama_auto_ubatch_ladder[0];"),
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
        ("shrink unhandled", "switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) {\n            case GGML_SYCL_LATE_CHECK_EQUAL:\n            case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:", "switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) {\n            case GGML_SYCL_LATE_CHECK_EQUAL:"),
        ("the state's shrink unhandled", "switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) {\n            case GGML_SYCL_LATE_CHECK_EQUAL:\n            case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:", "switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) {\n            case GGML_SYCL_LATE_CHECK_EQUAL:"),
        ("a default arm", "switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) {", "switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) {\n            default:\n                break;"),
        ("a default arm in the state switch", "switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) {", "switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) {\n            default:\n                break;"),
        ("the host tier compared", "if (d.host) {\n            continue;\n        }", ""),
        ("a state refusal does not end the fold", "the reserved state term\");\n                return out;", "the reserved state term\");\n                break;"),
        ("a state miss read as equal", "                out.state_not_recorded.push_back(d.device);\n", ""),
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
    moved = code.replace(z("ml.init_mappings(prefetch_mappings,"), z("llama_load_late_check(") + z("ml.init_mappings(prefetch_mappings,"), 1)
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
    assert not result_fields_ok(MEASURE_H.replace("std::vector<size_t>  state_not_recorded_bytes;", "", 1))


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
    # the constructor asks the predicate twice since the 19bc41dbb merge (the throw block, and the flash-attn follow
    # of a model that shares ctx_other's KV); the mutants anchor on the throw block
    guard = "if (llama_model_needs_ctx_other(model)) { if (params.ctx_other == nullptr) {"
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, guard, "if ((model.arch == LLM_ARCH_EAGLE3)) { if (params.ctx_other == nullptr) {"), 1))
    pred = function_body(code, _NEEDS_OTHER)
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_GEMMA4_ASSISTANT", "LLM_ARCH_GEMMA4"), 1))
    # the inner or, the arch list and the outer or are each pinned
    assert not needs_other_ok(code.replace(pred, mutate(pred, "model.tok_embd == nullptr ||", "model.tok_embd == nullptr &&"), 1))
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_EAGLE3 ||", "LLM_ARCH_EAGLE3 &&"), 1))
    assert not needs_other_ok(code.replace(pred, mutate(pred, "LLM_ARCH_GEMMA4_ASSISTANT ||", "LLM_ARCH_GEMMA4_ASSISTANT &&"), 1))
    # the constructor's throw and the ctx_other it hands the graph
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, "if (params.ctx_other == nullptr) {", "if (false) {"), 1))
    handed = '(this warning is normal during memory fitting)"); } cparams.ctx_other = params.ctx_other; }'
    assert not needs_other_ok(code.replace(ctor, mutate(ctor, handed, handed.replace("= params.ctx_other;", "= nullptr;")), 1))


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
        ("measured at the admitted stage", "GGML_SYCL_MEASURE_STAGE_PROBE, weights)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, weights)"),
        ("no reservation", "llama_load_probe_reservations reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);", "(void) measured_n_ctx;"),
        ("n_ctx 0 reserved", "llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx)", "llama_load_probe_reserve(procs, txn, measured.devices, 0)"),
        ("the micro-batch reserved as the context", "const uint32_t measured_n_ctx = shape.n_ctx;", "const uint32_t measured_n_ctx = shape.n_ubatch;"),
        ("the probe's residency not kept", "out.kv = measured.kv;", ""),
        ("the declined devices dropped", "out.compute_declined = std::move(reserved.compute_declined);", ""),
        ("the reserved states dropped", "out.state_reserved = std::move(reserved.state_reserved);", ""),
        ("the reserved states read from the declined list", "out.state_reserved = std::move(reserved.state_reserved);", "out.state_reserved = reserved.compute_declined;"),
        ("the compute term reserved past the helper", "llama_load_probe_reservations reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);", "llama_load_probe_reservations reserved = llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);\n    (void) llama_sycl_l4_reserve_compute_term(procs, txn, measured.devices[0], measured_n_ctx);"),
        ("no proc gate", "if (!procs.available() || !procs.load_terms_available())", "if (false)"),
        ("no load-term gate", "if (!procs.available() || !procs.load_terms_available())", "if (!procs.available())"),
    ]:
        assert not probe_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_admitted_term_is_checked_then_recorded():
    assert admit_ok(code_of(CONTEXT_CPP))


def test_admit_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _ADMIT)
    for name, old, new in [
        ("measured at the late stage", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, weights)", "GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, weights)"),
        ("recorded despite a refusal", "if (!out.refusal.empty()) {\n        return out;\n    }", ""),
        ("never recorded", "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);", ""),
        ("recorded at n_ctx 0", "out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);", "out.n_recorded = llama_admitted_record(procs, txn, out, 0);"),
        ("the fold given the ladder's rung, not the shape's", "const uint32_t n_ubatch = shape.n_ubatch;", "const uint32_t n_ubatch = llama_auto_ubatch_ladder[0];"),
        ("the fold bypassed", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.state_reserved, measured.devices,", "llama_admitted_check_fold(procs, measured.devices, probe.compute_declined, probe.state_reserved, measured.devices,"),
        ("the admitted stage reserves", "    if (!out.refusal.empty()) {\n        return out;\n    }", "    (void) llama_sycl_l4_reserve_compute_term(procs, txn, measured.devices[0], measured_n_ctx);\n    if (!out.refusal.empty()) {\n        return out;\n    }"),
        ("unreserved devices compared", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.state_reserved, measured.devices,", "llama_admitted_check_fold(procs, probe.devices, {}, probe.state_reserved, measured.devices,"),
        ("the fold given no reserved state", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.state_reserved, measured.devices,", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, {}, measured.devices,"),
        ("the fold given the declined list as the states", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.state_reserved, measured.devices,", "llama_admitted_check_fold(procs, probe.devices, probe.compute_declined, probe.compute_declined, measured.devices,"),
        ("the residencies swapped", "probe.kv, measured.kv);", "measured.kv, probe.kv);"),
        ("the probe's residency judged against itself", "probe.kv, measured.kv);", "probe.kv, probe.kv);"),
        ("the admitted stage reserves state", "    if (!out.refusal.empty()) {\n        return out;\n    }", "    (void) llama_load_probe_reserve(procs, txn, measured.devices, measured_n_ctx);\n    if (!out.refusal.empty()) {\n        return out;\n    }"),
        ("the state never recorded", "out.n_state_recorded = llama_admitted_record_state(procs, txn, out, measured_n_ctx);", ""),
        ("the state recorded at n_ctx 0", "out.n_state_recorded = llama_admitted_record_state(procs, txn, out, measured_n_ctx);", "out.n_state_recorded = llama_admitted_record_state(procs, txn, out, 0);"),
        ("the state recorded before the refusal check", "    if (!out.refusal.empty()) {\n        return out;\n    }\n    out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);\n    out.n_state_recorded = llama_admitted_record_state(procs, txn, out, measured_n_ctx);", "    out.n_state_recorded = llama_admitted_record_state(procs, txn, out, measured_n_ctx);\n    if (!out.refusal.empty()) {\n        return out;\n    }\n    out.n_recorded = llama_admitted_record(procs, txn, out, measured_n_ctx);"),
    ]:
        assert not admit_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


_CAPS = "std::vector<llama_tenant_buft_caps> llama_context::measure_tenant_caps(const sched_measure_plan & plan) const"
_PROBE_RESERVE = (
    "inline llama_load_probe_reservations llama_load_probe_reserve(const llama_sycl_l4_procs & procs, "
    "struct ggml_sycl_load_txn txn, const std::vector<llama_load_measure_device> & devices, uint32_t n_ctx)"
)
_RECORD_STATE = (
    "inline size_t llama_admitted_record_state(const llama_sycl_l4_procs & procs, struct ggml_sycl_load_txn txn, "
    "llama_admitted_check_result & admitted, uint32_t n_ctx)"
)


def state_consumer_ok(ctx_code: str, measure_code: str) -> bool:
    """llama.cpp-p6i0: the recurrent state is a named RUNTIME-first consumer of the reserve, at every hop from the
    measure to the late check. A hop that drops it leaves the state allocated RUNTIME-first with nothing reserved
    for it, and a compute chunk lands outside the RUNTIME zone (Qwen3.8 on the B70)."""
    caps = function_body(ctx_code, _CAPS)
    run = function_body(ctx_code, _RUN)
    res = function_body(measure_code, _PROBE_RESERVE)
    rec = function_body(measure_code, _RECORD_STATE)
    fold = function_body(measure_code, _FOLD)
    # the measure: the memory's size per buffer type, read once, then each SYCL device's caps take its buft's entry
    caps_order = [
        "memory_bytes = memory->memory_breakdown();",
        "for (const auto & entry : plan.bufts)",
        "const auto state = memory_bytes.find(entry.buft);",
        "c.state_bytes = state != memory_bytes.end() ? state->second : 0;",
        "out.push_back(c);",
    ]
    pos = [caps.find(z(t)) for t in caps_order]
    if -1 in pos or pos != sorted(pos) or caps.count(z("c.state_bytes =")) != 1:
        return False
    # the run: on the measured device, never on the host tier
    if z("d.state_bytes = c.host ? 0 : c.state_bytes;") not in run:
        return False
    # the probe reserve: the state's own reservation, before the compute term, for a device that has state
    st = res.find(z("if (d.state_bytes != 0) {if (!llama_sycl_l4_reserve_state_term(procs, txn, d.device, d.state_bytes)) {"))
    ls = res.find(z("out.state_reserved.push_back(d.device);"))
    co = res.find(z("llama_sycl_l4_reserve_compute_term(procs, txn, d, n_ctx)"))
    if -1 in (st, ls, co) or not st < ls < co or res.count(z("out.state_reserved.push_back(")) != 1:
        return False
    # the state reservation is its own outcome: the fold marks it from the reserved-state list, and the record follows
    # it, never the compute term's (a state reserved beside a declined compute term is still recorded)
    if z("t.state_reserved = t.state_bytes != 0 && std::find(state_reserved.begin(), state_reserved.end(), d.device) "
         "!= state_reserved.end();") not in measure_code:
        return False
    if z("t.state_recorded = t.state_reserved && t.state_bytes != 0 &&") not in rec:
        return False
    # the admitted fold carries the probe's state (the reserved one); the record names it as the state term
    if z("t.state_bytes = bound->state_bytes;") not in measure_code:
        return False
    if z("llama_sycl_l4_record_state_term(procs, txn, t.device, t.state_bytes, n_ctx)") not in rec:
        return False
    # the late fold: the state check with the state bytes, never the compute total
    return z("llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)") in fold


def test_the_recurrent_state_is_a_named_runtime_consumer():
    assert state_consumer_ok(code_of(CONTEXT_CPP), code_of(MEASURE_H))


def test_state_consumer_mutants():
    ctx = code_of(CONTEXT_CPP)
    meas = code_of(MEASURE_H)
    caps = function_body(ctx, _CAPS)
    run = function_body(ctx, _RUN)
    res = function_body(meas, _PROBE_RESERVE)
    rec = function_body(meas, _RECORD_STATE)
    fold = function_body(meas, _FOLD)
    for name, body, old, new, in_ctx in [
        ("the measure never reads the memory", caps, "memory_bytes = memory->memory_breakdown();", "", True),
        ("the caps drop the state", caps, "c.state_bytes = state != memory_bytes.end() ? state->second : 0;", "", True),
        ("the caps take another buft's state", caps, "const auto state = memory_bytes.find(entry.buft);", "const auto state = memory_bytes.begin();", True),
        ("the run drops the state", run, "d.state_bytes = c.host ? 0 : c.state_bytes;", "", True),
        ("the run carries the host tier's state", run, "d.state_bytes = c.host ? 0 : c.state_bytes;", "d.state_bytes = c.state_bytes;", True),
        ("the probe reserve drops the state", res, "if (d.state_bytes != 0) {", "if (false) {", False),
        ("the probe reserve never lists the state", res, "out.state_reserved.push_back(d.device);", "", False),
        ("the probe reserve lists the state before it is reserved", res, "if (d.state_bytes != 0) {", "if (d.state_bytes != 0) {out.state_reserved.push_back(d.device);", False),
        ("the fold marks the state from the compute term", meas, "t.state_reserved = t.state_bytes != 0 &&", "t.state_reserved = t.reserved && t.state_bytes != 0 &&", False),
        ("the record follows the compute reservation", rec, "t.state_recorded = t.state_reserved && t.state_bytes != 0 &&", "t.state_recorded = t.reserved && t.state_bytes != 0 &&", False),
        ("the probe reserve hands over the compute total", res, "llama_sycl_l4_reserve_state_term(procs, txn, d.device, d.state_bytes)", "llama_sycl_l4_reserve_state_term(procs, txn, d.device, d.total)", False),
        ("the admitted fold carries the admitted state", meas, "t.state_bytes = bound->state_bytes;", "t.state_bytes = d.state_bytes;", False),
        ("the record drops the state", rec, "llama_sycl_l4_record_state_term(procs, txn, t.device, t.state_bytes, n_ctx)", "false", False),
        ("the late fold checks the compute total", fold, "llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)", "llama_sycl_l4_late_check_state(procs, txn, d.device, d.total)", False),
    ]:
        if body is meas:
            c2, m2 = ctx, mutate(meas, old, new)
        elif in_ctx:
            c2, m2 = ctx.replace(body, mutate(body, old, new), 1), meas
        else:
            c2, m2 = ctx, meas.replace(body, mutate(body, old, new), 1)
        assert not state_consumer_ok(c2, m2), f"mutant {name!r} slipped through"


_WARN_EXCESS = (
    "static void llama_context_sycl_warn_state_excess(const llama_sycl_l4_procs & procs, "
    "const std::vector<ggml_backend_t> & backend_ptrs, const std::vector<ggml_backend_buffer_type_t> & backend_buft, "
    "const std::map<ggml_backend_buffer_type_t, size_t> & memory_bytes, uint32_t n_seq_max)"
)
_EXCESS_TEXT = (
    "inline std::string llama_context_state_excess_text(int32_t device, uint32_t n_seq_max, uint64_t planned_bytes, "
    "uint64_t real_bytes)"
)
_EXCESS_CALL = (
    "llama_context_sycl_warn_state_excess(llama_context_sycl_l4_procs_for(backends), backend_ptrs, backend_buft, "
    "memory->memory_breakdown(), cparams.n_seq_max);"
)
_REFUSALS = ("throw", "return false", "GGML_ABORT", "abort(", "measure_status", "sched_reserve")


def state_excess_ok(ctx_code: str, measure_code: str) -> bool:
    """llama.cpp-p6i0: a context with more sequences than the load measured is placed and warned, never refused."""
    ctor = ctor_body(ctx_code)
    warn = function_body(ctx_code, _WARN_EXCESS)
    text = function_body(measure_code, _EXCESS_TEXT)
    # once, right after the memory exists, for a context that allocates (not the measure itself)
    mem = ctor.find(z("memory.reset(model.create_memory(params_mem, cparams, measure_only));"))
    gate = ctor.find(z("if (!measure_only && memory && llama_context_has_sycl_backend(backends)) {"))
    call = ctor.find(z(_EXCESS_CALL))
    if -1 in (mem, gate, call) or not mem < gate < call or ctx_code.count(z(_EXCESS_CALL)) != 1:
        return False
    if ctor[mem:call].count(";") != 1:  # nothing between the memory and its comparison but the gate
        return False
    # the comparison: the device's own buft's bytes, the backend's planned term, the one text, at WARN
    for t in (
        "const auto state = memory_bytes.find(backend_buft[i]);",
        "llama_sycl_l4_planned_state_term(procs, device, &planned)",
        "const std::string warn = llama_context_state_excess_text(device, n_seq_max, planned, real);",
        'LLAMA_LOG_WARN("%s: %s\\n", __func__, warn.c_str());',
    ):
        if z(t) not in warn:
            return False
    # never a refusal, in the comparison or the helper
    if any(r in warn or r in text for r in _REFUSALS):
        return False
    # a load that planned no state term has its own line, after the fit check: it never blames the n_seq_max 1 measure
    fit = text.find(z("if (real_bytes <= planned_bytes) {return {};}"))
    unplanned = text.find(z('if (planned_bytes == 0) {return "[LOAD-PLAN] state term not planned on device "'))
    return -1 not in (fit, unplanned) and fit < unplanned


def test_a_larger_state_is_warned_never_refused():
    assert state_excess_ok(code_of(CONTEXT_CPP), code_of(MEASURE_H))


def test_state_excess_mutants():
    ctx = code_of(CONTEXT_CPP)
    meas = code_of(MEASURE_H)
    warn = function_body(ctx, _WARN_EXCESS)
    text = function_body(meas, _EXCESS_TEXT)
    ctor = ctor_body(ctx)
    for name, body, old, new, in_ctx in [
        ("the call is dropped", ctor, _EXCESS_CALL, "", True),
        ("the measure context warns too", ctor, "if (!measure_only && memory && llama_context_has_sycl_backend(backends)) {", "if (memory && llama_context_has_sycl_backend(backends)) {", True),
        ("the call passes n_seq_max 1", ctor, _EXCESS_CALL, _EXCESS_CALL.replace("cparams.n_seq_max", "1"), True),
        ("the WARN is dropped", warn, 'LLAMA_LOG_WARN("%s: %s\\n", __func__, warn.c_str());', "", True),
        ("the WARN is demoted to INFO", warn, 'LLAMA_LOG_WARN("%s: %s\\n", __func__, warn.c_str());', 'LLAMA_LOG_INFO("%s: %s\\n", __func__, warn.c_str());', True),
        ("the WARN becomes a refusal", warn, 'LLAMA_LOG_WARN("%s: %s\\n", __func__, warn.c_str());', "throw std::runtime_error(warn);", True),
        ("another buft's bytes", warn, "const auto state = memory_bytes.find(backend_buft[i]);", "const auto state = memory_bytes.begin();", True),
        ("the planned term is not read", warn, "llama_sycl_l4_planned_state_term(procs, device, &planned)", "true", True),
        ("the helper refuses", text, "if (real_bytes <= planned_bytes) {", "if (real_bytes > planned_bytes) {throw std::runtime_error(\"state\");}if (real_bytes <= planned_bytes) {", False),
        ("the helper compares loosely", text, "if (real_bytes <= planned_bytes) {", "if (real_bytes < planned_bytes) {", False),
        ("the unplanned line is dropped", text, "if (planned_bytes == 0) {", "if (false) {", False),
    ]:
        if in_ctx:
            c2, m2 = ctx.replace(body, mutate(body, old, new), 1), meas
        else:
            c2, m2 = ctx, meas.replace(body, mutate(body, old, new), 1)
        assert not state_excess_ok(c2, m2), f"mutant {name!r} slipped through"


def test_one_n_ctx_rule():
    code = code_of(MEASURE_H)
    assert n_ctx_ok(code)
    p = function_body(code, _PARAMS)
    assert not n_ctx_ok(code.replace(p, mutate(p, "params.n_ctx = shape.n_ctx;", "params.n_ctx = 0;"), 1))
    s = function_body(code, _SHAPE)
    assert not n_ctx_ok(code.replace(s, mutate(s, "llama_load_measure_n_ctx(named ? n_ctx_hint : 0, n_ctx_train)",
                                               "(named ? n_ctx_hint : 0)"), 1))
    b = function_body(code, _N_CTX)
    assert not n_ctx_ok(code.replace(b, mutate(b, "return n_ctx != 0 ? n_ctx : n_ctx_train;", "return n_ctx;"), 1))


def test_the_loader_orders_probe_pack_admit_sync_late():
    assert stage_order_ok(code_of(MODEL_CPP))


def test_stage_order_mutants():
    code = code_of(MODEL_CPP)
    probe_call_start = code.find(z("llama_load_probe_bound("))
    admit_call_start = code.find(z("llama_load_admitted_check("))
    assert probe_call_start != -1 and admit_call_start != -1
    late_inv = z("llama_model_sycl_set_late_inventory(ml, *this, use_mlock, __func__);")
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
    # a call given the placement envelope's n_ctx of 0 again
    envelope = z("llama_model_sycl_make_placement_envelope().n_ctx, sycl_model_loading_guard.txn,")
    for call in ("llama_load_probe_bound(", "llama_load_admitted_check(", "llama_load_late_check("):
        at = code.find(z(call))
        seg = code[at : at + 300]
        back = seg.replace(z("measure_shape, sycl_model_loading_guard.txn,"), envelope, 1)
        assert back != seg, call
        assert not stage_order_ok(code.replace(seg, back, 1)), f"the envelope's n_ctx at {call} slipped through"
    local = z(_MEASURE_SHAPE_LOCAL)
    # a third shape resolved apart from the two scopes
    assert not stage_order_ok(code.replace(local, local + local, 1)), "a third measure shape slipped through"
    # the late scope with no shape of its own
    last = code.rfind(local)
    assert not stage_order_ok(code[:last] + code[last + len(local):]), "a late scope with no shape slipped through"


def test_the_load_line_prints_the_measured_shape():
    """The compute-slot WARN prints the shape the admitted measure ran at and no longer says the caller's shape is
    not transported."""
    code = z(MODEL_CPP)
    assert z("measured at n_ctx %u ubatch %u, the context and micro-batch this load was told") in code
    assert "are not transported" not in MODEL_CPP
    assert "do not reach the load (fkpg)" not in MEASURE_H


def inventory_shape_ok(model_code: str) -> bool:
    """The micro-batch the backend credits the term at comes from the resolver the measures use, over the same hints
    and the same training context."""
    helper = function_body(model_code, _MODEL_SHAPE)
    return (
        z("inventory.n_ubatch_measure = llama_load_measure_shape_resolve(n_ctx_hint, n_ubatch_hint, "
          "hparams.n_ctx_train).n_ubatch;") in model_code
        and model_code.count(z("inventory.n_ubatch_measure =")) == 1
        and helper.endswith(z("{ return llama_load_measure_shape_resolve(model.get_n_ctx_hint(), "
                              "model.get_n_ubatch_hint(), model.hparams.n_ctx_train); }"))
    )


def test_the_inventory_carries_the_measured_micro_batch():
    code = code_of(MODEL_CPP)
    assert inventory_shape_ok(code)
    sent = "inventory.n_ubatch_measure = llama_load_measure_shape_resolve(n_ctx_hint, n_ubatch_hint, hparams.n_ctx_train).n_ubatch;"
    for name, new in [
        ("the planning rung sent", "inventory.n_ubatch_measure = inventory.n_ubatch;"),
        ("the raw hint sent", "inventory.n_ubatch_measure = n_ubatch_hint;"),
        ("the field never set", ""),
    ]:
        assert not inventory_shape_ok(mutate(code, sent, new)), f"mutant {name!r} slipped through"
    helper = function_body(code, _MODEL_SHAPE)
    assert not inventory_shape_ok(code.replace(helper, mutate(helper, "model.get_n_ubatch_hint()", "0"), 1))


def test_mutant_measure_rides_the_envelope_again_fails():
    """A measure fed the envelope's n_ctx (0, the training context) instead of the load's requested context."""
    code = code_of(MODEL_CPP)
    new = z("*this, measure_shape, sycl_model_loading_guard.txn,")
    old = z("*this, llama_model_sycl_make_placement_envelope().n_ctx, sycl_model_loading_guard.txn,")
    assert code.count(new) == 3, "the probe, admitted and late calls must each carry the requested shape"
    at = -1
    for site in ("probe", "admitted", "late"):
        at = code.index(new, at + 1)
        mutant = code[:at] + old + code[at + len(new) :]
        assert not stage_order_ok(mutant), f"the {site} measure on the envelope's n_ctx slipped through"
    # the shape local itself read from the envelope again
    request = "llama_model_load_measure_shape(*this);"
    envelope_shape = "llama_load_measure_shape{ llama_model_sycl_make_placement_envelope().n_ctx, 512 };"
    assert z(request) in code, "the measure shape local is no longer resolved from the model's hints"
    assert not stage_order_ok(
        code.replace(z(request), z(envelope_shape), 1)
    ), "the requested shape read from the envelope slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
