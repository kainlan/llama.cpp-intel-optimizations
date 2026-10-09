"""Source gate for the tenant section of a planned reserve (zhcn design 2.2, 2.5, 3.3).

A planned context's transaction measures, then derives the tenant section from the measured
chunk caps, and only then publishes. This gate pins, on comment-stripped text:

- `llama_context::measure_tenant_caps` maps each measured compute buft to the device it
  belongs to (a SYCL device's own buft, by the device index its name carries) or to the host
  tier (the host buffer type the CPU backend computes in), and drops every other buft: a CPU
  buft owns no SYCL slot;
- `sched_reserve_transaction` builds the section from `storage.plan` after the MEASURE and
  before the publish, refuses by name when the builder refuses, records the section and its
  key on the context, and prints the plan line for each device the section names;
- the plan line is printed at INFO through `llama_tenant_plan_line`, one per device, and
  carries the transaction's `n_ubatch`;
- the context holds the section, its key, each device's last compute load and the republish
  and covered counters.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEXT_CPP = (SRC / "llama-context.cpp").read_text()
CONTEXT_H = (SRC / "llama-context.h").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_TXN = "sched_reserve_result llama_context::sched_reserve_transaction()"
_CAPS = "std::vector<llama_tenant_buft_caps> llama_context::measure_tenant_caps(const sched_measure_plan & plan) const"
_REPORT = "void llama_context::tenant_plan_report(const sched_measure_plan & plan, uint32_t n_ubatch)"


def txn_ok(code: str) -> bool:
    b = function_body(code, _TXN)
    measure = b.find(z("sched_reserve_impl(sched_reserve_mode::MEASURE, measure_state)"))
    caps = b.find(z("measure_tenant_caps(storage.plan)"))
    build = b.find(z("llama_tenant_section_from_caps("))
    report = b.find(z("tenant_plan_report(storage.plan, storage.cparams.n_ubatch)"))
    publish = b.find(z("sycl_publish_runtime_context("))
    if min(measure, caps, build, report, publish) == -1 or not (measure < caps < build < report < publish):
        return False
    refuse = z("{ sched_reserve_status::REFUSED, tenant_reason }")
    if b.count(refuse) != 1 or not (build < b.find(refuse) < report):
        return False
    return z("tenant_key = llama_tenant_key_digest(tenant_section);") in b


def test_the_transaction_builds_the_section_between_measure_and_publish():
    assert txn_ok(code_of(CONTEXT_CPP))


def test_transaction_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _TXN)
    for name, old, new in [
        ("the caps never read", "measure_tenant_caps(storage.plan)", "std::vector<llama_tenant_buft_caps>()"),
        ("the builder refusal ignored", "{ sched_reserve_status::REFUSED, tenant_reason }", "{ sched_reserve_status::OK, \"\" }"),
        ("the key not recorded", "tenant_key = llama_tenant_key_digest(tenant_section);", ""),
        ("the plan line never printed", "tenant_plan_report(storage.plan, storage.cparams.n_ubatch);", ""),
    ]:
        assert not txn_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    # the section built after the publish
    moved = b.replace(z("if (!llama_tenant_section_from_caps("), z("if (!true || !llama_tenant_section_from_caps("), 1)
    assert moved != b
    publish = z("const sched_reserve_result published = sycl_publish_runtime_context(storage.cparams.flash_attn);")
    assert publish in b
    swapped = b.replace(publish, "", 1).replace(z("const std::vector<llama_tenant_buft_caps> tenant_caps"), publish + z("const std::vector<llama_tenant_buft_caps> tenant_caps"), 1)
    assert not txn_ok(code.replace(b, swapped, 1)), "mutant 'publish before the section' slipped through"


def caps_ok(code: str) -> bool:
    b = function_body(code, _CAPS)
    needs = [
        "llama_context_dev_is_sycl(dev)",
        "llama_context_sycl_device_index(dev, (int) sycl_ordinal)",
        "ggml_backend_dev_host_buffer_type(",
        "if (entry.buft == backend_buft[i]) { llama_tenant_buft_caps c;",
        "c.host = true;",
        "c.host = false;",
    ]
    return all(z(n) in b for n in needs) and b.count(z("llama_tenant_buft_caps")) >= 1


def test_the_caps_map_each_buft_to_its_tier():
    assert caps_ok(code_of(CONTEXT_CPP))


def test_caps_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _CAPS)
    for name, old, new in [
        ("a device buft matched by position", "if (entry.buft == backend_buft[i]) { llama_tenant_buft_caps c;", "if (true) { llama_tenant_buft_caps c;"),
        ("the host tier never recognised", "ggml_backend_dev_host_buffer_type(", "ggml_backend_dev_buffer_type("),
        ("the device index guessed", "llama_context_sycl_device_index(dev, (int) sycl_ordinal)", "(int) sycl_ordinal"),
    ]:
        assert not caps_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def report_ok(code: str) -> bool:
    b = function_body(code, _REPORT)
    needs = [
        "llama_tenant_plan_line(",
        "LLAMA_LOG_INFO(\"%s\\n\"",
        "f.n_ubatch = n_ubatch;",
        "f.n_measured = plan.n_measured;",
        "f.measure_ms = plan.measure_ms;",
        "f.republish = tenant_republish;",
        "f.covered = tenant_covered;",
        "tenant_compute_load[dev] = f.compute_load;",
    ]
    return all(z(n) in b for n in needs)


def test_the_report_prints_one_plan_line_per_device():
    assert report_ok(code_of(CONTEXT_CPP))


def test_report_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _REPORT)
    for name, old, new in [
        ("the line at DEBUG", "LLAMA_LOG_INFO(", "LLAMA_LOG_DEBUG("),
        ("the delta never carried forward", "tenant_compute_load[dev] = f.compute_load;", ""),
        ("n_ubatch left at zero", "f.n_ubatch = n_ubatch;", ""),
        ("the covered counter not read", "f.covered = tenant_covered;", ""),
    ]:
        assert not report_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_context_holds_the_section_and_its_counters():
    h = code_of(CONTEXT_H)
    for decl in [
        "std::vector<ggml_sycl_context_tenant_desc> tenant_section;",
        "uint64_t tenant_key = 0;",
        "std::map<int32_t, uint64_t> tenant_compute_load;",
        "uint32_t tenant_republish = 0;",
        "uint32_t tenant_covered = 0;",
        "std::vector<llama_tenant_buft_caps> measure_tenant_caps(const sched_measure_plan & plan) const;",
        "void tenant_plan_report(const sched_measure_plan & plan, uint32_t n_ubatch);",
    ]:
        assert z(decl) in h, decl


# --- the L4 procs table: one resolution path, no weak symbols ----------------

_L4 = "static llama_sycl_l4_procs llama_context_sycl_l4_procs_for_dev(ggml_backend_dev_t dev)"
_L4_NAMES = {
    "publish": "GGML_SYCL_PROC_SET_RUNTIME_CONTEXT_DESC",
    "coverage": "GGML_SYCL_PROC_TENANT_COVERAGE",
    "late_check": "GGML_SYCL_PROC_LOAD_LATE_CHECK",
    "probe_residency": "GGML_SYCL_PROC_PROBE_RESIDENCY",
    "record_term": "GGML_SYCL_PROC_LOAD_RECORD_COMPUTE_TERM",  # llama.cpp-p6i0
    "reserve_term": "GGML_SYCL_PROC_LOAD_RESERVE_COMPUTE_TERM",  # llama.cpp-p6i0
    "term_bytes": "GGML_SYCL_PROC_LOAD_COMPUTE_TERM_BYTES",  # llama.cpp-p6i0
    "reserve_state": "GGML_SYCL_PROC_LOAD_RESERVE_STATE_TERM",  # llama.cpp-p6i0
    "record_state": "GGML_SYCL_PROC_LOAD_RECORD_STATE_TERM",  # llama.cpp-p6i0
    "late_check_state": "GGML_SYCL_PROC_LOAD_LATE_CHECK_STATE",  # llama.cpp-p6i0
}


def no_weak(code: str) -> bool:
    # a weak reference is a second resolution path, and MSVC has no weak symbols
    return z("__attribute__((weak))") not in code and z("#pragma weak") not in code


def l4_ok(code: str) -> bool:
    if not no_weak(code):
        return False
    start = code.find(z(_L4))
    if start == -1:
        return False
    b = function_body(code, _L4)
    # every proc comes through the reg's proc address, by the header's name constant, in both link modes
    for field, name in _L4_NAMES.items():
        if z(f"procs.{field} = reinterpret_cast<decltype(procs.{field})>(llama_context_sycl_proc_addr(dev, {name}));") not in b:
            return False
    # the per-context table is that same one path, over the first SYCL backend's device
    if z("return llama_context_sycl_l4_procs_for_dev(dev);") not in code:
        return False
    return "#if" not in b and "&ggml_backend_sycl_" not in b and '"ggml_backend_sycl_' not in b


def test_the_l4_table_is_filled_through_the_reg_in_both_link_modes():
    assert l4_ok(code_of(CONTEXT_CPP))


def test_l4_table_mutants():
    code = code_of(CONTEXT_CPP)
    assert not l4_ok(code + z("#pragma weak ggml_backend_sycl_tenant_coverage\n"))
    assert not l4_ok(code + z("__attribute__((weak)) void f();"))
    b = function_body(code, _L4)
    for name, old, new in [
        ("a direct reference", "llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_TENANT_COVERAGE)", "&ggml_backend_sycl_tenant_coverage"),
        ("a string literal name", "GGML_SYCL_PROC_LOAD_LATE_CHECK)", '"ggml_backend_sycl_load_late_check")'),
        ("a link-mode split", "procs.publish = reinterpret_cast", "\n#ifdef GGML_USE_SYCL\n procs.publish = reinterpret_cast"),
    ]:
        assert not l4_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"
    # the header owns the names
    hdr = (ROOT / "ggml/include/ggml-sycl-l4-procs.h").read_text()
    for name in _L4_NAMES.values():
        assert name in hdr, name


# --- the cohort ids are ABI -------------------------------------------------

_COHORTS = [
    ("COMPUTE", 0),
    ("COMPUTE_HOST", 1),
    ("FATTN_MATERIALIZE", 2),
    ("NONFA_STAGE", 3),
    ("GRAPH_STAGE", 4),
]


def cohorts_ok(text: str) -> bool:
    code = code_of(text)
    if "APPEND-ONLY" not in text:
        return False
    for name, value in _COHORTS:
        if z(f"GGML_SYCL_CONTEXT_COHORT_{name} = {value},") not in code:
            return False
    # the count is last, so a new id goes before it and takes the next value
    return code.rstrip().endswith(z("GGML_SYCL_CONTEXT_COHORT_COUNT };"))


def test_the_cohort_ids_are_pinned_and_append_only():
    text = (ROOT / "ggml/include/ggml-sycl-cohort.h").read_text()
    assert cohorts_ok(text)
    assert not cohorts_ok(text.replace("APPEND-ONLY", "append-only"))
    assert not cohorts_ok(text.replace("COHORT_COMPUTE_HOST      = 1", "COHORT_COMPUTE_HOST      = 5"))
    assert not cohorts_ok(text.replace("    GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE       = 3,", ""))
    assert not cohorts_ok(text.replace("    GGML_SYCL_CONTEXT_COHORT_COUNT\n};", "    GGML_SYCL_CONTEXT_COHORT_COUNT,\n    GGML_SYCL_CONTEXT_COHORT_LATE = 9\n};"))


# --- the host tier's HOLD: one source for R_h, folded over the rung set ----------------------------------------

_HOLD_FOLD = "void llama_context::tenant_host_hold_measure_and_fold(const std::vector<ggml_sycl_context_tenant_desc> & current)"
_SELECT = "void llama_context::sycl_select_auto_ubatch()"
# llama.cpp-7gno: the rung set is made, and handed to the hold, by the trial's hoisted block, which the constructor runs
# before the memory module exists; the ladder half only reads it.
_PREPARE = "void llama_context::sycl_auto_ubatch_prepare(ggml_type type_k, ggml_type type_v)"


def hold_txn_ok(code: str) -> bool:
    """The transaction folds R_h once, from the section it just built, and applies it before the key is taken."""
    b = function_body(code, _TXN)
    build = b.find(z("llama_tenant_section_from_caps("))
    fold = b.find(z("if (!tenant_host_hold_ready) { tenant_host_hold_measure_and_fold(tenant_section); }"))
    apply = b.find(z("llama_tenant_section_apply_host_hold(tenant_section, tenant_host_hold);"))
    key = b.find(z("tenant_key = llama_tenant_key_digest(tenant_section);"))
    if min(build, fold, apply, key) == -1 or not (build < fold < apply < key):
        return False
    return b.count(z("llama_tenant_section_apply_host_hold(")) == 1 and b.count(z("tenant_host_hold_measure_and_fold(")) == 1


def hold_fold_ok(code: str) -> bool:
    """R_h is read from the sections the builder produced: the current rung's, and each other rung's own MEASURE.
    A rung that refuses or throws is left out, and the hold is ready only after every rung has been tried."""
    b = function_body(code, _HOLD_FOLD)
    needs = [
        "llama_tenant_host_hold hold;",
        "llama_tenant_host_hold_fold(hold, current);",
        "for (const uint32_t rung : tenant_rung_set) {",
        "if (rung == cparams.n_ubatch) { continue; }",
        "try {",
        "storage.cparams.n_ubatch = rung;",
        "sched_reserve_impl(sched_reserve_mode::MEASURE, state);",
        # a refused MEASURE is skipped, by name, with its own continue
        'if (measured.status != sched_reserve_status::OK) { LLAMA_LOG_DEBUG("%s: rung %u left out of the host hold: %s\\n", __func__, rung, measured.reason.c_str()); continue; }',
        # a section the builder refuses is skipped, with its own continue
        'if (!llama_tenant_section_from_caps(measure_tenant_caps(storage.plan), rung_section, reason)) { LLAMA_LOG_DEBUG("%s: rung %u left out of the host hold: %s\\n", __func__, rung, reason.c_str()); continue; }',
        "llama_tenant_host_hold_fold(hold, rung_section);",
        # a throwing MEASURE is skipped too, at WARN, naming the rung
        '} catch (const std::exception & err) { LLAMA_LOG_WARN("%s: rung %u left out of the host hold: %s\\n", __func__, rung, err.what()); }',
        "tenant_host_hold = hold;",
        "tenant_host_hold_ready = true;",
    ]
    if not all(b.count(z(n)) == 1 for n in needs if n not in ("try {",)) or b.count(z("try {")) != 1:
        return False
    # the rung is folded after its measure succeeded, inside the try, before the catch
    measure = b.index(z("sched_reserve_impl(sched_reserve_mode::MEASURE, state);"))
    fold = b.index(z("llama_tenant_host_hold_fold(hold, rung_section);"))
    catch = b.index(z("} catch (const std::exception & err) {"))
    if not (b.index(z("try {")) < measure < fold < catch):
        return False
    # the hold is recorded, and marked ready, only after the loop and its catch: never before a rung is tried
    record = b.index(z("tenant_host_hold = hold;"))
    ready = b.index(z("tenant_host_hold_ready = true;"))
    if not (catch < record < ready):
        return False
    # one source: R_h has no writer but the fold, and the context's hold is assigned only from the local
    c = code
    if c.count(z("llama_tenant_host_hold_fold(")) != 2:  # the current rung's fold and each other rung's
        return False
    if c.count(z("tenant_host_hold = ")) != 1 or c.count(z("tenant_host_hold_ready = true;")) != 1:
        return False
    return "tenant_host_hold.bytes" not in c and "tenant_host_hold.n_rungs" not in c


def hold_select_ok(code: str) -> bool:
    b = function_body(code, _PREPARE)
    set_at = b.find(z("llama_auto_ubatch_rung_set("))
    assign = b.find(z("tenant_rung_set.assign(rung_set, rung_set + n_rung_set);"))
    # one writer: the ladder half does not hand over (or change) the set
    return set_at != -1 and assign != -1 and set_at < assign and z("tenant_rung_set") not in function_body(code, _SELECT)


def test_the_transaction_applies_the_host_hold_before_the_key():
    assert hold_txn_ok(code_of(CONTEXT_CPP))


def test_the_fold_reads_r_h_from_the_measured_sections():
    assert hold_fold_ok(code_of(CONTEXT_CPP))


def test_the_ladder_hands_its_rung_set_to_the_hold():
    assert hold_select_ok(code_of(CONTEXT_CPP))


def test_the_context_declares_the_hold():
    h = code_of(CONTEXT_H)
    for decl in [
        "llama_tenant_host_hold tenant_host_hold;",
        "bool tenant_host_hold_ready = false;",
        "std::vector<uint32_t> tenant_rung_set;",
        "void tenant_host_hold_measure_and_fold(const std::vector<ggml_sycl_context_tenant_desc> & current);",
    ]:
        assert z(decl) in h, decl


def test_hold_mutants():
    code = code_of(CONTEXT_CPP)
    t = function_body(code, _TXN)
    for name, old, new in [
        ("the hold never applied", "llama_tenant_section_apply_host_hold(tenant_section, tenant_host_hold);", ""),
        ("the hold applied after the key", "llama_tenant_section_apply_host_hold(tenant_section, tenant_host_hold);",
         ""),
        ("the hold folded every transaction", "if (!tenant_host_hold_ready) { tenant_host_hold_measure_and_fold(tenant_section); }",
         "tenant_host_hold_measure_and_fold(tenant_section);"),
        ("the hold never folded", "if (!tenant_host_hold_ready) { tenant_host_hold_measure_and_fold(tenant_section); }", ""),
    ]:
        mutated = mutate(t, old, new)
        if name == "the hold applied after the key":
            mutated = mutated.replace(z("tenant_key = llama_tenant_key_digest(tenant_section);"),
                                      z("tenant_key = llama_tenant_key_digest(tenant_section); llama_tenant_section_apply_host_hold(tenant_section, tenant_host_hold);"), 1)
        assert not hold_txn_ok(code.replace(t, mutated, 1)), f"mutant {name!r} slipped through the transaction gate"

    f = function_body(code, _HOLD_FOLD)
    for name, old, new in [
        ("the current rung's section never folded", "llama_tenant_host_hold_fold(hold, current);", ""),
        ("every rung measured at the context's own n_ubatch", "storage.cparams.n_ubatch = rung;", ""),
        ("the current rung measured twice", "if (rung == cparams.n_ubatch) { continue; }", ""),
        ("a failed measure folded anyway", "if (measured.status != sched_reserve_status::OK) {", "if (false) {"),
        ("a refused measure no longer skipped (continue deleted)",
         'measured.reason.c_str()); continue; }', 'measured.reason.c_str()); }'),
        ("a refused section no longer skipped (continue deleted)",
         'rung, reason.c_str()); continue; }', 'rung, reason.c_str()); }'),
        ("the rung's section never folded", "llama_tenant_host_hold_fold(hold, rung_section);", ""),
        ("the rung's host bytes derived a second way", "llama_tenant_section_from_caps(measure_tenant_caps(storage.plan), rung_section, reason)",
         "(rung_section.push_back(ggml_sycl_context_tenant_desc()), true)"),
        ("the rung set ignored", "for (const uint32_t rung : tenant_rung_set) {", "for (const uint32_t rung : std::vector<uint32_t>()) {"),
        ("a throwing rung measure unhandled", "} catch (const std::exception & err) {", "} catch (const std::logic_error & err) {"),
        ("a throwing rung measure swallowed silently",
         'LLAMA_LOG_WARN("%s: rung %u left out of the host hold: %s\\n", __func__, rung, err.what());', ""),
        ("the hold never recorded", "tenant_host_hold = hold;", ""),
    ]:
        assert not hold_fold_ok(code.replace(f, mutate(f, old, new), 1)), f"mutant {name!r} slipped through the fold gate"
    # ready before the loop: marked ready first, so a throw from a rung leaves ready=true with a partial hold
    early = mutate(f, "tenant_host_hold_ready = true;", "")
    early = early.replace(z("llama_tenant_host_hold_fold(hold, current);"),
                          z("tenant_host_hold_ready = true; llama_tenant_host_hold_fold(hold, current);"), 1)
    assert not hold_fold_ok(code.replace(f, early, 1)), "mutant 'ready before the loop' slipped through"
    # recorded straight into the context's hold while the rungs are folded
    assert not hold_fold_ok(code.replace(f, f.replace(z("llama_tenant_host_hold_fold(hold, rung_section);"),
                                                      z("llama_tenant_host_hold_fold(hold, rung_section); tenant_host_hold = hold;"), 1), 1)), \
        "mutant 'a second write of the context hold' slipped through"
    # a second writer of R_h
    assert not hold_fold_ok(code + z("void x() { tenant_host_hold.bytes.push_back(1); }"))

    s = function_body(code, _PREPARE)
    assign = "tenant_rung_set.assign(rung_set, rung_set + n_rung_set);"
    assert not hold_select_ok(code.replace(s, mutate(s, assign, ""), 1)), "mutant 'the rung set never handed over' slipped through"
    # assigned before the set is computed
    early = mutate(s, assign, "").replace(z("llama_auto_ubatch_rung_set("), z(assign + " llama_auto_ubatch_rung_set("), 1)
    assert not hold_select_ok(code.replace(s, early, 1)), "mutant 'the rung set handed over before it exists' slipped through"
    # a second writer in the ladder half
    sel = function_body(code, _SELECT)
    assert not hold_select_ok(code.replace(sel, sel.replace(z("sycl_hold_spill_validated_ub=0;"), z("sycl_hold_spill_validated_ub=0;" + assign), 1), 1)), \
        "mutant 'the ladder half writes the hold's set' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
