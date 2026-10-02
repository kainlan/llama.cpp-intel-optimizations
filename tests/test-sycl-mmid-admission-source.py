#!/usr/bin/env python3
"""Source contract: MUL_MAT_ID admission keys on MMID coverage, ADD_ID does not.

llama.cpp-yitq. `ggml_backend_sycl_device_supports_op` admits ADD_ID and
MUL_MAT_ID in one branch. Until this gate landed, both were type-gated on
`ggml_sycl_mul_mat_type_supported()` -- the DENSE MUL_MAT allowlist -- which is
legitimately true for F32/F16/IQ1_S..IQ4_XS because those types have real dense
kernels. None of them has an `_id` kernel family, so admission handed the
scheduler 234 census cases this backend cannot compute. The route oracle did
refuse them, but that refusal escapes as `ggml_sycl_fallback_error` ->
`GGML_STATUS_FAILED`, and at the time this gate landed
`ggml_backend_compare_graph_backend` discarded that status -- so the op reported
uncorrelated numbers (ERR 86-99) instead of falling back to the CPU backend.
(That discard is fixed as of llama.cpp-t98c: ggml-backend.cpp now returns false
when either compute fails, so such a refusal reports as "compare failed". This
gate is still the load-bearing one -- reporting a refusal honestly is not the
same as not needing the refusal, and admission is what avoids it.)

Two halves, and the second is the one a well-meaning change breaks:

1. MUL_MAT_ID must gate on `moe_mmvq_admission_supports_type()`.
2. ADD_ID must NOT. Its `src[0]` is the F32 activation (`ggml_add_id` ->
   `ggml_dup_tensor(ctx, a)`), not an expert weight, so gating it on an MMID
   expert-weight coverage table would refuse F32 and send every MoE bias-add to
   the CPU backend -- a graph split per layer, in every MoE model.

The population itself is gated by tests/test-sycl-moe-mmvq-tables.cpp (host-only,
no GPU). This file gates only that the predicate is actually WIRED IN: without it
the predicate can be added, left unconsulted, and that test still passes green.

Why the anchors are two-level: the branch's `if` line appears more than once in
ggml-sycl.cpp, so a whole-file `.index()`/`.replace()` would select by FILE
POSITION and silently latch onto the wrong copy the moment anyone adds another
one above it. Every anchor below is resolved INSIDE the enclosing function and
asserted unique. Same lesson as test-sycl-supports-op-foreign-buffer-source.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

FUNC_START = "static bool ggml_backend_sycl_device_supports_op"
FUNC_END = "\n    switch (op->op) {"
BRANCH_START = "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) {"
BRANCH_END = "        return true;\n    }"

MMID_PREDICATE = "moe_mmvq_admission_supports_type(indexed_a_type)"
DENSE_ALLOWLIST = "ggml_sycl_mul_mat_type_supported(indexed_a_type)"

# The inner guard that scopes the MMID predicate to MUL_MAT_ID, matched as the
# WHOLE condition. Matching the bare substring `op->op == GGML_OP_MUL_MAT_ID`
# instead would be satisfied by the branch HEADER (`ADD_ID || MUL_MAT_ID`) and so
# would survive widening the inner guard to cover ADD_ID -- measured: that mutant
# passed until this was tightened.
MMID_ONLY_GUARD = "if (op->op == GGML_OP_MUL_MAT_ID) {"


def section(text: str, start: str, end: str, what: str) -> str:
    """The slice between two anchors, with both asserted present and the start unique."""
    if text.count(start) != 1:
        raise AssertionError(
            f"{what}: start anchor occurs {text.count(start)} times, expected exactly 1. "
            f"An anchor that is not unique selects by position, not by identity."
        )
    begin = text.index(start)
    if end not in text[begin:]:
        raise AssertionError(f"{what}: end anchor {end!r} not found after the start anchor")
    return text[begin : begin + text[begin:].index(end) + len(end)]


def admission_branch(text: str) -> str:
    func = section(text, FUNC_START, FUNC_END, "supports_op body")
    return section(func, BRANCH_START, BRANCH_END, "ADD_ID/MUL_MAT_ID admission branch")


def contract(text: str) -> bool:
    """True when MUL_MAT_ID gates on MMID coverage and ADD_ID keeps the dense allowlist."""
    try:
        branch = admission_branch(text)
    except AssertionError:
        return False

    # Half 1: the MMID predicate is consulted, under a MUL_MAT_ID-only condition.
    if MMID_PREDICATE not in branch:
        return False
    if MMID_ONLY_GUARD not in branch:
        return False

    # Half 2: the dense allowlist survives for the ADD_ID side. If it is gone
    # entirely, ADD_ID has either been gated on the MMID tables (refusing F32) or
    # left ungated (admitting anything) -- both regressions.
    if DENSE_ALLOWLIST not in branch:
        return False

    # The MMID predicate must be reached only for MUL_MAT_ID: the guard opens
    # before the predicate is consulted.
    return branch.index(MMID_ONLY_GUARD) < branch.index(MMID_PREDICATE)


def mutate_in_branch(old: str, new: str) -> str:
    """Replace `old` ONLY inside the admission branch, spliced back into the file.

    Scoped by containment, never by occurrence index: the needle text also appears
    elsewhere in this 107k-line file, and a whole-file replace(needle, x, 1) would
    mutate whichever copy sorts first by position -- disarming the mutant while
    reading as a code failure.
    """
    branch = admission_branch(SOURCE)
    assert old in branch, f"mutation target {old!r} is no longer in the admission branch"
    return SOURCE.replace(branch, branch.replace(old, new, 1), 1)


def test_mmid_admission_is_wired_to_the_mmid_tables() -> None:
    assert contract(SOURCE), (
        "MUL_MAT_ID admission in ggml_backend_sycl_device_supports_op no longer gates on "
        f"{MMID_PREDICATE} (or ADD_ID lost the dense allowlist). See llama.cpp-yitq."
    )


def test_mutations_are_rejected() -> None:
    """Positive controls. A contract that has never been observed to fail is not evidence."""
    mutants = (
        # Reverting to the shared dense allowlist -- the exact pre-fix state.
        ("revert to the dense MUL_MAT allowlist", MMID_PREDICATE, DENSE_ALLOWLIST),
        # Dropping the MMID gate entirely.
        ("drop the MMID gate", f"!{MMID_PREDICATE}", "false"),
        # Applying the MMID gate to ADD_ID too, by widening the inner guard. This
        # is the trap half: it reads as a tightening and refuses every F32 MoE
        # bias-add to the CPU backend.
        ("widen the MMID gate onto ADD_ID", MMID_ONLY_GUARD, "if (op->op != GGML_OP_NONE) {"),
        # Removing the dense allowlist, leaving the ADD_ID side ungated.
        ("drop the ADD_ID dense allowlist", f"!{DENSE_ALLOWLIST}", "false"),
    )
    survivors = [name for name, old, new in mutants if contract(mutate_in_branch(old, new))]
    assert not survivors, (
        "these mutations did NOT break the contract, so it does not detect them: "
        + "; ".join(survivors)
    )


def test_anchors_are_identity_not_position() -> None:
    """A second copy of the branch's `if` line must not move what we resolve.

    The real file already contains more than one; this asserts the two-level
    anchoring tolerates that, so the gate cannot be disarmed by an unrelated
    addition elsewhere in the file.
    """
    assert SOURCE.count(BRANCH_START) >= 1
    branch = admission_branch(SOURCE)
    decoy = "    if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID) {\n        return false;\n    }\n"
    assert admission_branch(decoy + SOURCE) == branch, (
        "adding a decoy copy of the branch header above the real one changed which section "
        "was resolved -- the anchors are position-selected and this gate is disarmable"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        failures = 0
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    print(f"ok   {name}")
                except AssertionError as exc:
                    failures += 1
                    print(f"FAIL {name}: {exc}")
        print(
            "test-sycl-mmid-admission-source: "
            + ("OK" if failures == 0 else f"FAILED ({failures})")
        )
        sys.exit(1 if failures else 0)
    test_mmid_admission_is_wired_to_the_mmid_tables()
    test_mutations_are_rejected()
    test_anchors_are_identity_not_position()
    print("test-sycl-mmid-admission-source: OK")
