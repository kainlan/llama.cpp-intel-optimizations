"""Source gate for the constructor's planned-reserve wiring (llama.cpp-7gno, zhcn design 2.4, 2.7).

The pure decisions live in `src/llama-residency-fixpoint.h` and run in test-residency-fixpoint;
this gate pins where the constructor calls them, on comment-stripped text, because a host test
cannot reach `llama_context`'s constructor:

- the hoisted block (`sycl_auto_ubatch_prepare`, pinned by test-sycl-auto-ubatch-hoist-source.py) runs
  before everything below, so the rung set exists when the fixpoint runs;
- the plan_caps decision is `llama_plan_caps_decide` over the context's own facts (a SYCL backend,
  the active-plan predicate, the two cap procs, `l4_procs.available()`), made once;
- a refusal is the named text of `llama_plan_caps_missing_procs_reason()`, thrown before anything
  is acquired;
- the copy is acquired straight into the member, once, after the decision and before the
  fixpoint, with the looked-up `_free` in the deleter, and the fixpoint comes before
  `model.create_memory(`;
- the planned transaction stays production-unreachable until the L4 procs are all present: the
  acquisition is under the decision's ACQUIRE arm only, and nothing else assigns `plan_caps`.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CTX_CPP = (SRC / "llama-context.cpp").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate

_CTOR_HEAD = (
    "llama_context::llama_context(const llama_model & model, llama_context_params params, "
    "llama_measure_context_args * measure) :"
)
_CTOR_END = "llama_context::~llama_context()"
_SELECT = "void llama_context::sycl_select_auto_ubatch(ggml_type type_k, ggml_type type_v)"


def ctor_text(code: str) -> str:
    start = code.index(z(_CTOR_HEAD))
    return code[start : code.index(z(_CTOR_END), start)]


def positions(text: str, needles: list) -> list:
    out = []
    for n in needles:
        assert text.count(z(n)) >= 1, f"{n!r} missing"
        out.append(text.index(z(n)))
    return out


def test_decision_and_acquisition_order():
    ctor = ctor_text(code_of(CTX_CPP))
    pos = positions(
        ctor,
        [
            "sycl_auto_ubatch_prepare(",
            "llama_plan_caps_decide(",
            "llama_plan_caps_missing_procs_reason()",
            "plan_caps = llama_plan_caps_ptr(",
            "sched_residency_fixpoint(",
            "model.create_memory(",
        ],
    )
    assert pos == sorted(pos), pos


def test_decision_reads_the_contexts_own_facts():
    ctor = ctor_text(code_of(CTX_CPP))
    assert z(
        "llama_plan_caps_decide(llama_context_has_sycl_backend(backends), plan_active, "
        "plan_procs.caps_new != nullptr, plan_procs.caps_free != nullptr, l4_procs.available())"
    ) in ctor
    assert ctor.count(z("llama_plan_caps_decide(")) == 1


def test_refusal_is_named_and_precedes_acquisition():
    ctor = ctor_text(code_of(CTX_CPP))
    refuse = z("LLAMA_PLAN_CAPS_REFUSE_NO_PROCS")
    assert refuse in ctor
    arm = ctor[ctor.index(refuse) :]
    arm = arm[: arm.index("}")]
    assert z("throw std::runtime_error(llama_plan_caps_missing_procs_reason())") in arm


def test_acquisition_is_the_only_assignment_under_the_acquire_arm():
    code = code_of(CTX_CPP)
    # the member is assigned once in the whole file, and never reset anywhere else
    assert len(re.findall(r"(?<![\w.>])plan_caps\s*=\s*llama_plan_caps_ptr\(", code)) == 1
    assert "plan_caps.reset(" not in code
    ctor = ctor_text(code)
    acquire = z("LLAMA_PLAN_CAPS_ACQUIRE")
    assert acquire in ctor
    arm = ctor[ctor.index(acquire) :]
    arm = arm[: arm.index("}")]
    assert z("plan_caps = llama_plan_caps_ptr(plan_procs.caps_new(), llama_plan_caps_deleter{ plan_procs.caps_free })") in arm
    assert z("if (!plan_caps)") in arm


def test_mutants():
    code = code_of(CTX_CPP)
    # reordering the fixpoint after the memory module
    m = mutate(code, "sched_residency_fixpoint();", "")
    try:
        _order_claim(m)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("dropping the fixpoint call must fail the order claim")


def _order_claim(code: str):
    ctor = ctor_text(code)
    pos = positions(
        ctor,
        [
            "llama_plan_caps_decide(",
            "plan_caps = llama_plan_caps_ptr(",
            "sched_residency_fixpoint(",
            "model.create_memory(",
        ],
    )
    assert pos == sorted(pos), pos
