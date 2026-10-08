"""Source gate for the constructor's planned-reserve wiring (llama.cpp-7gno, zhcn design 2.4, 2.7).

The pure decisions live in `src/llama-residency-fixpoint.h` and run in test-residency-fixpoint; this gate pins where the
constructor calls them, on comment-stripped text, because a host test cannot reach `llama_context`'s constructor:

- after the hoisted block (`sycl_auto_ubatch_prepare`, pinned by test-sycl-auto-ubatch-hoist-source.py) and before
  `model.create_memory(`, the constructor decides once with `llama_plan_caps_decide` over the context's own facts (a SYCL
  backend, the active-plan predicate, the two cap procs, `llama_context_l4_ready`);
- a refusal is the named text of `llama_plan_caps_missing_procs_reason()`, thrown before anything is acquired;
- the copy is acquired straight into the member, once in the whole file, under the ACQUIRE arm, with the looked-up `_free`
  in the deleter and a null result refused by name; the fixpoint follows it, still under that arm;
- the planned transaction stays production-unreachable: `llama_context_l4_ready` needs the residency probe to be wired
  (`llama_context_residency_probe_wired`, false until moua's L4 probe exists and is looked up) AND moua's three L4 procs,
  and the fixpoint member refuses by name rather than publishing an unchecked plan.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX_CPP = (ROOT / "src/llama-context.cpp").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_CTOR_HEAD = (
    "llama_context::llama_context(const llama_model & model, llama_context_params params, "
    "llama_measure_context_args * measure) :"
)
_CTOR_END = "llama_context::~llama_context()"
_FIXPOINT = "void llama_context::sched_residency_fixpoint()"
_L4_READY = "static bool llama_context_l4_ready(const std::vector<ggml_backend_ptr> & backends)"

_DECIDE = (
    "llama_plan_caps_decide(llama_context_has_sycl_backend(backends), plan_procs.plan_active, "
    "plan_procs.caps_new != nullptr, plan_procs.caps_free != nullptr, plan_l4_ready);"
)
# the proc lookups run only once L4 is ready: until then decide() answers UNPLANNED before it reads them, so a SYCL
# context pays for none of them
_L4_FIRST = "const bool plan_l4_ready = llama_context_l4_ready(backends);"
_LAZY_PROCS = (
    "const llama_context_sycl_plan_caps_procs plan_procs = "
    "plan_l4_ready ? llama_context_sycl_plan_caps_procs_for(backends) : llama_context_sycl_plan_caps_procs{};"
)
_ACQUIRE = "plan_caps = llama_plan_caps_ptr(plan_procs.caps_new(), llama_plan_caps_deleter{ plan_procs.caps_free });"


def ctor_text(code: str) -> str:
    start = code.index(z(_CTOR_HEAD))
    return code[start : code.index(z(_CTOR_END), start)]


def arm_of(text: str, head: str) -> str:
    """The braced block that follows `head` (z-spelled), through its matching brace."""
    at = text.index(z(head))
    i = text.index("{", at)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
    raise AssertionError(f"unbalanced braces after {head!r}")


def order_ok(code: str) -> bool:
    ctor = ctor_text(code)
    needles = [
        "sycl_auto_ubatch_prepare(params.type_k, params.type_v);",
        "llama_plan_caps_decide(",
        "llama_plan_caps_missing_procs_reason()",
        _ACQUIRE,
        "sched_residency_fixpoint();",
        "model.create_memory(",
    ]
    pos = []
    for n in needles:
        if ctor.count(z(n)) != 1:
            return False
        pos.append(ctor.index(z(n)))
    return pos == sorted(pos)


def decision_ok(code: str) -> bool:
    ctor = ctor_text(code)
    return (
        ctor.count(z("llama_plan_caps_decide(")) == 1
        and z("plan_decision = " + _DECIDE) in ctor
        and ctor.count(z("llama_context_sycl_plan_caps_procs_for(")) == 1
        and z(_L4_FIRST + " " + _LAZY_PROCS) in ctor
    )


def refusal_ok(code: str) -> bool:
    arm = arm_of(ctor_text(code), "if (plan_decision == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS)")
    return z("throw std::runtime_error(llama_plan_caps_missing_procs_reason());") in arm and "plan_caps=" not in arm


def acquisition_ok(code: str) -> bool:
    # the member is assigned once in the whole file, and never reset anywhere else
    if len(re.findall(r"(?<![\w.>])plan_caps\s*=\s*llama_plan_caps_ptr\(", code)) != 1 or "plan_caps.reset(" in code:
        return False
    arm = arm_of(ctor_text(code), "if (plan_decision == LLAMA_PLAN_CAPS_ACQUIRE)")
    return (
        z(_ACQUIRE) in arm
        and z("if (!plan_caps) { throw std::runtime_error(") in arm
        and arm.index(z(_ACQUIRE)) < arm.index(z("sched_residency_fixpoint();"))
    )


_GUARD = "if (!hparams.vocab_only && !measure_only)"


def guarded_ok(code: str) -> bool:
    """The decision, the refusal, the acquisition and the fixpoint all sit inside ONE `!vocab_only && !measure_only` block:
    a measure-only context (the plan override's own transaction) must not be refused, and must not acquire a copy."""
    ctor = ctor_text(code)
    head = z(_GUARD)
    at = ctor.find(head)
    while at != -1:
        arm = arm_of(ctor[at:], _GUARD)
        if all(
            z(n) in arm
            for n in (
                "llama_plan_caps_decide(",
                "if (plan_decision == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS)",
                "if (plan_decision == LLAMA_PLAN_CAPS_ACQUIRE)",
                "sched_residency_fixpoint();",
            )
        ):
            return True
        at = ctor.find(head, at + 1)
    return False


def inert_ok(code: str) -> bool:
    if z("static constexpr bool llama_context_residency_probe_wired = false;") not in code:
        return False
    ready = function_body(code, _L4_READY)
    if z("return llama_context_residency_probe_wired && llama_context_sycl_l4_procs_for(backends).available();") not in ready:
        return False
    # the fixpoint member refuses by name; it publishes nothing and returns nothing
    body = function_body(code, _FIXPOINT)
    return body.count("throw") == 1 and "publish" not in body and "return" not in body


def test_decision_and_acquisition_order():
    assert order_ok(code_of(CTX_CPP))


def test_decision_reads_the_contexts_own_facts():
    assert decision_ok(code_of(CTX_CPP))


def test_refusal_is_named_and_acquires_nothing():
    assert refusal_ok(code_of(CTX_CPP))


def test_acquisition_is_the_only_assignment_under_the_acquire_arm():
    assert acquisition_ok(code_of(CTX_CPP))


def test_the_decision_is_inside_the_not_measure_only_block():
    assert guarded_ok(code_of(CTX_CPP))


def test_the_planned_reserve_is_production_unreachable_until_the_probe_is_wired():
    assert inert_ok(code_of(CTX_CPP))


def test_mutants():
    code = code_of(CTX_CPP)
    ctor = ctor_text(code)

    def with_ctor(new: str) -> str:
        return code.replace(ctor, new, 1)

    # order: the fixpoint call dropped, the acquisition after the memory module, a second decision
    assert not order_ok(with_ctor(mutate(ctor, "sched_residency_fixpoint();", ""))), "mutant 'no fixpoint call' slipped through"
    moved = mutate(ctor, _ACQUIRE, "").replace(z("if (!plan_caps)"), z("if (false)"), 1)
    moved = moved.replace(z("model.create_memory("), z(_ACQUIRE + " model.create_memory("), 1)
    assert not order_ok(with_ctor(moved)), "mutant 'acquired after the memory module' slipped through"

    # the decision
    assert not decision_ok(with_ctor(mutate(ctor, "plan_procs.plan_active,", "true,"))), \
        "mutant 'the plan predicate is a constant' slipped through"
    assert not decision_ok(with_ctor(mutate(ctor, "plan_procs.caps_free != nullptr, plan_l4_ready);",
                                            "plan_procs.caps_free != nullptr, true);"))), \
        "mutant 'L4 readiness is a constant' slipped through"
    assert not decision_ok(with_ctor(mutate(ctor, "plan_l4_ready ? llama_context_sycl_plan_caps_procs_for(backends) :",
                                            "llama_context_sycl_plan_caps_procs_for(backends); true ? llama_context_sycl_plan_caps_procs_for(backends) :"))), \
        "mutant 'the procs are looked up before L4 is known ready' slipped through"
    assert not decision_ok(with_ctor(mutate(ctor, _L4_FIRST, "const bool plan_l4_ready = true;"))), \
        "mutant 'readiness is not asked of llama_context_l4_ready' slipped through"
    assert not decision_ok(with_ctor(mutate(ctor, "plan_procs.caps_new != nullptr,", "true,"))), \
        "mutant 'the new proc is not consulted' slipped through"

    # the refusal
    assert not refusal_ok(with_ctor(mutate(ctor, "throw std::runtime_error(llama_plan_caps_missing_procs_reason());",
                                           'throw std::runtime_error("no caps");'))), \
        "mutant 'the refusal text is reworded' slipped through"
    assert not refusal_ok(with_ctor(mutate(ctor, "throw std::runtime_error(llama_plan_caps_missing_procs_reason());",
                                           "LLAMA_LOG_WARN(\"x\");"))), "mutant 'the refusal does not throw' slipped through"

    # the acquisition
    assert not acquisition_ok(with_ctor(mutate(ctor, "llama_plan_caps_deleter{ plan_procs.caps_free }", "llama_plan_caps_deleter{}"))), \
        "mutant 'the deleter has no free' slipped through"
    assert not acquisition_ok(code + z("void x() { plan_caps = llama_plan_caps_ptr(nullptr, llama_plan_caps_deleter{}); }")), \
        "mutant 'a second acquisition site' slipped through"
    assert not acquisition_ok(with_ctor(ctor.replace(z("if (!plan_caps) {"), z("if (false) {"), 1))), \
        "mutant 'a null copy is not refused' slipped through"

    # inertness
    assert not inert_ok(code.replace(z("llama_context_residency_probe_wired = false;"),
                                     z("llama_context_residency_probe_wired = true;"), 1)), \
        "mutant 'the probe is declared wired' slipped through"
    ready = function_body(code, _L4_READY)
    assert not inert_ok(code.replace(ready, mutate(ready, "return llama_context_residency_probe_wired && llama_context_sycl_l4_procs_for(backends).available();",
                                                   "return llama_context_sycl_l4_procs_for(backends).available();"), 1)), \
        "mutant 'readiness ignores the probe' slipped through"
    fix = function_body(code, _FIXPOINT)
    assert not inert_ok(code.replace(fix, fix.replace("throw", "return;", 1), 1)), \
        "mutant 'the fixpoint member returns without running' slipped through"

    # the guard
    # (the decision's block is the one that opens right after `sycl_auto_ubatch_trial` is declared)
    opens = "bool sycl_auto_ubatch_trial = false; " + _GUARD
    assert not guarded_ok(with_ctor(mutate(ctor, opens, "bool sycl_auto_ubatch_trial = false; if (!hparams.vocab_only)"))), \
        "mutant 'a measure-only context takes the planned decision' slipped through"
    assert not guarded_ok(with_ctor(mutate(ctor, opens, "bool sycl_auto_ubatch_trial = false; if (!measure_only)"))), \
        "mutant 'the planned decision is not skipped for a vocab-only context' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
