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
