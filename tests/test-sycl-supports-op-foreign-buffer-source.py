#!/usr/bin/env python3
"""Source contract for llama.cpp-zviv foreign-buffer residency handling."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()


# Anchors are the DEFINITIONS (signature plus opening brace), never a bare
# name: llama.cpp-9qjy added a forward declaration of
# ggml_sycl_weight_residency_is_observable ~105k lines above its definition,
# and a name-only anchor then opened the section there, so the "section" held
# most of the file and every check below passed on unrelated code.
OBSERVABLE_DEF = "static bool ggml_sycl_weight_residency_is_observable(const ggml_tensor * tensor) {"
EXECUTES_DEF = "static bool ggml_sycl_weight_executes_on_host(const ggml_tensor * tensor, int device) {"
LAYER_PLAN_DEF = "static bool ggml_sycl_layer_plan_applies_to_op(const ggml_tensor * op) {"
OBSERVABLE_BOUNDS = (OBSERVABLE_DEF, EXECUTES_DEF)
EXECUTES_BOUNDS = (EXECUTES_DEF, LAYER_PLAN_DEF)
# Both predicates are a few dozen lines; a section past this is a mis-anchor.
MAX_SECTION_LINES = 120


def section(text: str, start: str, end: str) -> str:
    for anchor in (start, end):
        if text.count(anchor) != 1:
            raise ValueError(f"anchor must occur exactly once, found {text.count(anchor)}: {anchor}")
    begin = text.index(start)
    finish = text.index(end, begin)
    body = text[begin:finish]
    if body.count("\n") > MAX_SECTION_LINES:
        raise ValueError(f"section from {start!r} spans {body.count(chr(10))} lines (> {MAX_SECTION_LINES})")
    return body


def contract(text: str) -> bool:
    try:
        observable = section(text, *OBSERVABLE_BOUNDS)
        executes = section(text, *EXECUTES_BOUNDS)
    except ValueError:
        return False

    known_storage = (
        "ggml_backend_buffer_is_host(tensor->buffer)" in observable
        and "ggml_backend_buffer_has_sycl_context(tensor->buffer)" in observable
        and "ggml_backend_buffer_is_sycl_split(tensor->buffer)" in observable
        and "ggml_backend_buffer_is_sycl_tp(tensor->buffer)" in observable
    )
    guard = "if (!ggml_sycl_weight_residency_is_observable(tensor))"
    if guard not in executes:
        return False
    guard_pos = executes.index(guard)
    return (
        known_storage
        and "return false;" in executes[guard_pos:]
        and guard_pos < executes.index("ggml_sycl_weight_is_planned_on_host")
        and guard_pos < executes.index("ggml_sycl_resolve(tensor, device)")
    )


def test_foreign_buffers_are_not_claimed_as_host_resident() -> None:
    assert contract(SOURCE)




def mutate_in_section(bounds: tuple, old: str, new: str) -> str:
    """Rewrite `old` -> `new` only inside the section the contract reads.

    A bare SOURCE.replace(old, new, 1) rewrites the FIRST occurrence in a
    106k-line file, so any unrelated earlier use of the same call silently
    disarms the mutant: the mutation lands somewhere the contract never looks,
    the real predicate survives, contract() still returns True, and only this
    test notices. That happened -- llama.cpp-srti added
    ggml_backend_buffer_has_sycl_context(tensor->buffer) to a storage-miss
    diagnostic ~55k lines ABOVE the predicate this gate guards. Scoping the
    mutation keeps it aimed at the code under contract, and asserting the
    string is present in the section first means a rename cannot make this
    pass vacuously either.
    """
    body = section(SOURCE, *bounds)
    assert old in body, f"mutation target {old!r} is no longer in this section"
    return SOURCE.replace(body, body.replace(old, new, 1), 1)


def test_mutations_are_rejected() -> None:
    assert not contract(mutate_in_section(
        EXECUTES_BOUNDS,
        "if (!ggml_sycl_weight_residency_is_observable(tensor))",
        "if (false)",
    ))
    assert not contract(mutate_in_section(
        OBSERVABLE_BOUNDS,
        "ggml_backend_buffer_has_sycl_context(tensor->buffer)",
        "false",
    ))


def test_anchors_bind_to_the_definitions() -> None:
    # Controls for the anchors themselves, so a mis-anchor is a FAIL rather
    # than a gate that quietly reads the wrong code.
    observable = section(SOURCE, *OBSERVABLE_BOUNDS)
    executes = section(SOURCE, *EXECUTES_BOUNDS)
    assert observable.startswith(OBSERVABLE_DEF) and "return ggml_backend_buffer_is_host" in observable
    assert executes.startswith(EXECUTES_DEF)

    # A forward declaration (the 9qjy shape) plus an unrelated use of a
    # checked predicate above the definition must not move the section: the
    # contract still holds and the scoped mutant is still rejected.
    decl = "static bool ggml_sycl_weight_residency_is_observable(const ggml_tensor * tensor);\n"
    decoy = "static bool decoy(ggml_tensor * tensor) { return ggml_backend_buffer_has_sycl_context(tensor->buffer); }\n"
    with_decl = decl + decoy + SOURCE
    assert contract(with_decl)
    body = section(with_decl, *OBSERVABLE_BOUNDS)
    disarmed = body.replace("ggml_backend_buffer_has_sycl_context(tensor->buffer)", "false", 1)
    assert disarmed != body
    assert not contract(with_decl.replace(body, disarmed, 1))

    # A second definition-shaped anchor, a renamed anchor, or a section that
    # has grown past the predicate are each refused.
    assert not contract(OBSERVABLE_DEF + "\n}\n" + SOURCE)
    assert not contract(SOURCE.replace(EXECUTES_DEF, EXECUTES_DEF.replace("executes_on_host", "runs_on_host"), 1))
    bloated = SOURCE.replace(observable, observable + "// filler\n" * (MAX_SECTION_LINES + 1), 1)
    assert not contract(bloated)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
