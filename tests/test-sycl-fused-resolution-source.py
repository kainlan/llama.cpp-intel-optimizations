"""Source gate for the fused-op resolution report (zhcn design gate 37, the
report-site clauses, as far as the code before the fixpoint can satisfy them).

`resolve_fused_ops` no longer logs: it records its outcome in the reserve
state's `fused_resolution`, one entry per group header when the group starts and
one per probe when it completes. One member, `fused_resolution_report`, prints a
record through the pure `fused_resolution_emit` with a per-entry printed mark
that belongs to the context, so a retried reserve never prints an entry twice.
This gate pins, on comment-stripped text:

- `resolve_fused_ops` contains no log statement and no `__func__`, and writes
  only through `state.resolution`;
- the report's body is the one statement
  `if (!measure_only) { fused_resolution_emit(record, fused_resolution_printed,
  "resolve_fused_ops", fused_resolution_sink); }`, `fused_resolution_emit` has no
  other caller, and the prefix is a literal;
- the two forms of the report call: the MEASURE form, once, as the statement right
  before the publish in the reserve transaction, and the ALLOC form, once, as the
  statement right after `resolve_fused_ops(` in the impl; no other call but the
  guard's trampoline;
- the unwind guard: constructed at the entry of `sched_reserve_impl` over the
  state's record, `guard_arg` declared before `unwind_guard`, its class catching
  what the report throws and counting `std::uncaught_exceptions()`, its lost
  callback `noexcept`, and the lost line one fixed literal at WARN under
  `!measure_only`;
- the marks are a non-static, non-thread_local `llama_context` data member passed
  to the emit call by its bare name.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
CONTEXT_H = (ROOT / "src/llama-context.h").read_text()
RESOLUTION_H = (ROOT / "src/llama-fused-resolution.h").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body
strip_comments = _gate.strip_comments

_RESOLVE = _gate._RESOLVE
_IMPL = _gate._IMPL
_REPORT = "void llama_context::fused_resolution_report(const fused_resolution & record)"
_LOST = "void llama_context::fused_resolution_report_lost() noexcept"
_TXN = "sched_reserve_result llama_context::sched_reserve_transaction()"

_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


# --- resolve_fused_ops records, it does not log ---------------------------------


def resolve_ok(code: str) -> bool:
    b = function_body(code, _RESOLVE)
    proper = _STRING_RE.sub('""', b[b.index("{") :])
    if "LLAMA_LOG_" in proper or "__func__" in proper or "llama_log" in proper:
        return False
    # it writes the record through the state, one assignment per probe and one header per group
    needed = [
        "GGML_ASSERT(state.resolution != nullptr);",
        "fused_resolution & resolution = *state.resolution;",
        "resolution.entry[id] = entry;",
        "resolution.entry[id].present = true;",
        "resolution.entry[id].header = header;",
    ]
    if not all(z(n) in b for n in needed):
        return False
    # the headers of the three groups, as upstream printed them
    for header in (
        '"resolving fused Gated Delta Net support:"',
        '"resolving fused Lightning Indexer support:"',
        '"resolving fused DeepSeek V4 HC support:"',
    ):
        if b.count(z(header)) != 1:
            return False
    # every probe has its own entry id
    for entry in (
        "FA", "GDN_AR", "GDN_CH", "LID", "HC_PRE", "HC_COMB", "HC_POST",
    ):
        if len(re.findall(rf"FUSED_RESOLUTION_ENTRY_{entry}\b", b)) != 1:
            return False
    return True


def test_resolve_records_and_never_logs():
    assert resolve_ok(code_of(CONTEXT_CPP)), (
        "resolve_fused_ops must contain no log statement or __func__, record every probe and group header "
        "through state.resolution, and give each probe its own entry"
    )


def test_resolve_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _RESOLVE)
    mutants = [
        ("a log statement restored", "enabled = entry.enabled;", 'LLAMA_LOG_INFO("%s enabled\\n", probe.name); enabled = entry.enabled;'),
        ("the upstream func restored", "const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;", "const char * func = __func__; const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;"),
        ("the record never written", "resolution.entry[id] = entry;", ""),
        ("a header never recorded", 'start_group(FUSED_RESOLUTION_ENTRY_LID_HEADER, "resolving fused Lightning Indexer support:");', ""),
        ("two probes share an entry", "resolve(llm_fused_op_gdn_ch_probe, state.cparams.fused_gdn_ch, FUSED_RESOLUTION_ENTRY_GDN_CH);", "resolve(llm_fused_op_gdn_ch_probe, state.cparams.fused_gdn_ch, FUSED_RESOLUTION_ENTRY_GDN_AR);"),
    ]
    for name, old, new in mutants:
        assert not resolve_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


# --- one report site -----------------------------------------------------------


def report_ok(code: str) -> bool:
    b = function_body(code, _REPORT)
    want = z(
        _REPORT
        + ' { if (!measure_only) { fused_resolution_emit(record, fused_resolution_printed, "resolve_fused_ops", fused_resolution_sink); } }'
    )
    if b != want:
        return False
    # the emit function has exactly one caller in src/ outside the header that defines it
    if code.count("fused_resolution_emit(") != 1:
        return False
    # the prefix literal appears where the emit is called, never as __func__
    return "__func__" not in b


def test_report_has_one_site():
    assert report_ok(code_of(CONTEXT_CPP)), (
        "fused_resolution_report's body must be the one `if (!measure_only) { fused_resolution_emit(...) }` statement "
        'with the literal prefix "resolve_fused_ops", and nothing else may call fused_resolution_emit'
    )


def test_report_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _REPORT)
    mutants = [
        ("the measure-only guard dropped", "if (!measure_only) {", "if (true) {"),
        ("the prefix is __func__", '"resolve_fused_ops"', "__func__"),
        ("the marks are a temporary", "fused_resolution_printed,", "std::string(),"),
    ]
    for name, old, new in mutants:
        mutated = code.replace(b, mutate(b, old, new), 1) if name != "the marks are a temporary" else code.replace(
            b, b.replace(z("fused_resolution_printed,"), z("marks_copy,"), 1), 1
        )
        assert mutated != code, name
        assert not report_ok(mutated), f"mutant {name!r} slipped through"
    # a second caller of the emit function
    second = code.replace(
        z("void llama_context::fused_resolution_report_lost() noexcept {"),
        z('void stray() { fused_resolution_emit(rec, marks, "x", s); }') + z("void llama_context::fused_resolution_report_lost() noexcept {"),
        1,
    )
    assert not report_ok(second), "mutant 'a second emit caller' slipped through"


# --- the two forms of the report call ------------------------------------------


def forms_ok(code: str) -> bool:
    # every call of the report, wherever it is
    calls = [m.start() for m in re.finditer(re.escape("fused_resolution_report("), code)]
    # the definition and the trampoline's call are the two non-form occurrences
    txn = function_body(code, _TXN)
    impl = function_body(code, _IMPL)
    measure_form = z("fused_resolution_report(storage.resolution);") + z("const sched_reserve_result published = sycl_publish_runtime_context(")
    alloc_form = z("resolve_fused_ops(state, mctx.get(), n_seqs);") + z("fused_resolution_report(*state.resolution);")
    if txn.count(measure_form) != 1 or impl.count(alloc_form) != 1:
        return False
    # nowhere else: the transaction, the impl, the definition and the trampoline
    return len(calls) == 4


def test_report_call_forms():
    assert forms_ok(code_of(CONTEXT_CPP)), (
        "the report is called once right before the publish (MEASURE form) and once right after resolve_fused_ops in "
        "the impl (ALLOC form), and by the guard's trampoline only otherwise"
    )


def test_form_mutants():
    code = code_of(CONTEXT_CPP)
    txn = function_body(code, _TXN)
    impl = function_body(code, _IMPL)
    mm = z("fused_resolution_report(storage.resolution);")
    am = z("fused_resolution_report(*state.resolution);")
    cases = [
        ("the MEASURE-form call dropped", code.replace(txn, txn.replace(mm, "", 1), 1)),
        ("the ALLOC-form call dropped", code.replace(impl, impl.replace(am, "", 1), 1)),
        ("a second MEASURE-form call", code.replace(txn, txn.replace(mm, mm + mm, 1), 1)),
        ("the ALLOC-form call moved after the first reserve", code.replace(impl, impl.replace(am, "", 1).replace(z("int n_splits_pp = -1;"), z("int n_splits_pp = -1;") + am, 1), 1)),
    ]
    for name, mutated in cases:
        assert mutated != code, name
        assert not forms_ok(mutated), f"mutant {name!r} slipped through"


# --- the unwind guard ------------------------------------------------------------


def guard_site_ok(code: str) -> bool:
    impl = function_body(code, _IMPL)
    pre = (
        z("GGML_ASSERT(state.resolution != nullptr);")
        + z("fused_resolution_guard_arg guard_arg = { this, state.resolution };")
        + z("fused_resolution_unwind_guard unwind_guard(fused_resolution_report_trampoline, fused_resolution_lost_trampoline, &guard_arg);")
    )
    start = impl[impl.index("{") + 1 :]
    if not start.startswith(pre):
        return False
    # the trampolines: the lost one is noexcept
    if z("static void fused_resolution_lost_trampoline(void * arg) noexcept {") not in code:
        return False
    if z("static void fused_resolution_report_trampoline(void * arg) { auto * a = static_cast<fused_resolution_guard_arg *>(arg); a->ctx->fused_resolution_report(*a->record); }") not in code:
        return False
    # the guard struct holds the context and the record
    return z("struct fused_resolution_guard_arg { llama_context * ctx; const fused_resolution * record; };") in code


def test_guard_site():
    assert guard_site_ok(code_of(CONTEXT_CPP)), (
        "sched_reserve_impl must construct guard_arg, then unwind_guard, as its first statements over the state's record"
    )


def test_guard_site_mutants():
    code = code_of(CONTEXT_CPP)
    impl = function_body(code, _IMPL)
    ga = z("fused_resolution_guard_arg guard_arg = { this, state.resolution };")
    ug = z("fused_resolution_unwind_guard unwind_guard(fused_resolution_report_trampoline, fused_resolution_lost_trampoline, &guard_arg);")
    cases = [
        ("no guard", code.replace(impl, impl.replace(ug, "", 1), 1)),
        ("the guard declared before its argument", code.replace(impl, impl.replace(ga + ug, ug + ga, 1), 1)),
        ("the guard over a local record", code.replace(impl, impl.replace(ga, z("fused_resolution_guard_arg guard_arg = { this, nullptr };"), 1), 1)),
        ("the lost trampoline may throw", code.replace(z("static void fused_resolution_lost_trampoline(void * arg) noexcept {"), z("static void fused_resolution_lost_trampoline(void * arg) {"), 1)),
    ]
    for name, mutated in cases:
        assert mutated != code, name
        assert not guard_site_ok(mutated), f"mutant {name!r} slipped through"


def guard_class_ok(h: str) -> bool:
    c = code_of(h)
    return (
        z("n_at_construction_(std::uncaught_exceptions())") in c
        and z("if (std::uncaught_exceptions() > n_at_construction_) { try { report_(arg_); } catch (...) { lost_(arg_); } }") in c
        and z("typedef void (*lost_fn)(void * arg) noexcept;") in c
        and z("fused_resolution_unwind_guard(const fused_resolution_unwind_guard &) = delete;") in c
    )


def test_guard_class():
    assert guard_class_ok(RESOLUTION_H)


def test_guard_class_mutants():
    for name, old, new in [
        ("reports on every destruction", "if (std::uncaught_exceptions() > n_at_construction_) {", "if (true) {"),
        ("the report may throw out of the destructor", "} catch (...) { lost_(arg_); }", "}"),
        ("the lost callback may throw", "typedef void (*lost_fn)(void * arg) noexcept;", "typedef void (*lost_fn)(void * arg);"),
        ("the count is not captured", "n_at_construction_(std::uncaught_exceptions())", "n_at_construction_(0)"),
        ("the guard is copyable", "fused_resolution_unwind_guard(const fused_resolution_unwind_guard &) = delete;", ""),
    ]:
        assert not guard_class_ok_z(mutate(code_of(RESOLUTION_H), old, new)), f"mutant {name!r} slipped through"


def guard_class_ok_z(c: str) -> bool:
    return (
        z("n_at_construction_(std::uncaught_exceptions())") in c
        and z("if (std::uncaught_exceptions() > n_at_construction_) { try { report_(arg_); } catch (...) { lost_(arg_); } }") in c
        and z("typedef void (*lost_fn)(void * arg) noexcept;") in c
        and z("fused_resolution_unwind_guard(const fused_resolution_unwind_guard &) = delete;") in c
    )


# --- the lost line ---------------------------------------------------------------


def lost_ok(code: str) -> bool:
    b = function_body(code, _LOST)
    want = z(
        _LOST
        + ' { if (!measure_only) { LLAMA_LOG_WARN("resolve_fused_ops: resolution lines lost: exception while unwinding\\n"); } }'
    )
    if b != want:
        return False
    # the fixed line fits the logger's stack buffer
    lit = re.search(r'LLAMA_LOG_WARN\("([^"]*)"', strip_comments(CONTEXT_CPP[CONTEXT_CPP.index("fused_resolution_report_lost() noexcept {") :]))
    return lit is not None and len(lit.group(1)) < 128


def test_lost_line():
    assert lost_ok(code_of(CONTEXT_CPP)), "the lost line must be one fixed literal at WARN under !measure_only"


def test_lost_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _LOST)
    for name, old, new in [
        ("lost prints in a measure-only context", "if (!measure_only) {", "if (true) {"),
        ("lost at INFO", "LLAMA_LOG_WARN(", "LLAMA_LOG_INFO("),
        ("lost formats a value", 'LLAMA_LOG_WARN("resolve_fused_ops: resolution lines lost: exception while unwinding\\n");', 'LLAMA_LOG_WARN("resolve_fused_ops: resolution lines lost: %s\\n", "x");'),
    ]:
        assert not lost_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


# --- the marks -------------------------------------------------------------------


def marks_ok(header: str) -> bool:
    h = strip_comments(header)
    m = re.search(r"\n\s*(?P<decl>[^\n;]*fused_resolution_printed\s*\[[^\]]*\])\s*;", h)
    if not m:
        return False
    decl = m.group("decl")
    if "static" in decl or "thread_local" in decl or "const " in decl:
        return False
    # a data member of the context: the declaration sits inside `struct llama_context {`
    return h.index("struct llama_context {") < m.start()


def test_marks_are_a_context_member():
    assert marks_ok(CONTEXT_H)


def test_marks_mutants():
    for name, old, new in [
        ("static marks", "std::string fused_resolution_printed[", "static std::string fused_resolution_printed["),
        ("thread_local marks", "std::string fused_resolution_printed[", "thread_local std::string fused_resolution_printed["),
        ("const marks", "std::string fused_resolution_printed[", "const std::string fused_resolution_printed["),
    ]:
        assert CONTEXT_H.count(old) == 1
        assert not marks_ok(CONTEXT_H.replace(old, new, 1)), f"mutant {name!r} slipped through"
    # moved out of the context
    moved = CONTEXT_H.replace("std::string fused_resolution_printed[FUSED_RESOLUTION_N_ENTRIES];", "", 1)
    moved = "std::string fused_resolution_printed[FUSED_RESOLUTION_N_ENTRIES];\n" + moved
    assert not marks_ok(moved), "mutant 'the marks outside the context' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
