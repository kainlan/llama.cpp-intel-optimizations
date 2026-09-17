#!/usr/bin/env python3
"""Source contract for llama.cpp-zviv foreign-buffer residency handling."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()


def section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def contract(text: str) -> bool:
    try:
        observable = section(
            text,
            "static bool ggml_sycl_weight_residency_is_observable",
            "static bool ggml_sycl_weight_executes_on_host",
        )
        executes = section(
            text,
            "static bool ggml_sycl_weight_executes_on_host(const ggml_tensor * tensor, int device) {",
            "static bool ggml_sycl_layer_plan_applies_to_op",
        )
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


OBSERVABLE_BOUNDS = (
    "static bool ggml_sycl_weight_residency_is_observable",
    "static bool ggml_sycl_weight_executes_on_host",
)
EXECUTES_BOUNDS = (
    "static bool ggml_sycl_weight_executes_on_host(const ggml_tensor * tensor, int device) {",
    "static bool ggml_sycl_layer_plan_applies_to_op",
)


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
