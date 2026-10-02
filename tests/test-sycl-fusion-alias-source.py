#!/usr/bin/env python3
"""Every fused-kernel site that reads an input while it writes an output must call the aliasing gate
(ggml/src/ggml-sycl/fusion-alias.hpp) before it launches -- source gated because the call is a
DECISION a later "simplify" pass could delete without any functional test noticing: the unfused
and the fused path print the same tokens on every layout except the one that races.

llama.cpp-rb2h: ggml_gallocr allocates in unfused order, so a fused chain's output can land
partially over a dead input (the fused ADD+RMS_NORM of a Qwen3.6-27B layer sat 8192 B below the ADD
operand it overlapped, and B50 perplexity varied run to run). Each site declines the fusion when the
gate does not admit its layout; the unfused kernels are correct for any layout.

Gated sites (the host test tests/test-sycl-fusion-alias.cpp pins what each admits and declines):
  bit1 RMS_NORM+MUL+ADD, bit4 RMS_NORM+MUL, bit5 ADD+RMS_NORM, bit6 MUL+ADD   (chains in graph_compute_impl)
  bit0 MUL_MAT+ADD (ggml_sycl_try_fuse_tg_mul_mat_add), bit0 router (ggml_sycl_try_fuse_tg_router_f32_add_argsort)
Not gated, by design and stated in the commit: bit2/bit3 (the activation is quantised to scratch by a
separate kernel before the MMQ kernel runs), and the opt-in fusions (MoE down-weighted-sum,
GGML_SYCL_FFN_FUSION, GGML_SYCL_PERSISTENT_TG_RMS_FUSION), which are tracked on a follow-up ticket.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

# (site enumerator, launch marker, enclosing function signature or None for graph_compute_impl,
#  extra markers that must also come after the gate: a drain or state change that precedes the launch)
CHAIN_SITES = [
    ("GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD", "ggml_sycl_op_rms_norm_fused_add(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_RMS_NORM_MUL", "ggml_sycl_op_rms_norm_fused(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_ADD_RMS_NORM", "ggml_sycl_op_add_rms_norm_fused(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_MUL_ADD", "ggml_sycl_op_mul_add_fused(*sycl_ctx"),
]
FUNCTION_SITES = [
    (
        "GGML_SYCL_FUSION_SITE_MUL_MAT_ADD",
        "static bool ggml_sycl_try_fuse_tg_mul_mat_add(",
        ["split_merge_drain();", "ggml_sycl_mmvq_set_fused_add("],
    ),
    (
        "GGML_SYCL_FUSION_SITE_ROUTER",
        "static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort(",
        ["split_merge_drain();", "ggml_sycl_router_f32_bias_argsort_sycl(*ctx.stream()"],
    ),
]


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy of the walker in this fork's
    other source-gate tests -- the convention is one self-contained file per check, not a shared import."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError("unclosed brace")


def function(text: str, signature: str) -> str | None:
    if signature not in text:
        return None
    start = text.index(signature)
    brace = text.index("{", start)
    return text[start:matching_brace(text, brace) + 1]


def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments so a marker quoted in prose is not mistaken for a call."""
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


def gate_call(site: str) -> re.Pattern:
    return re.compile(r"ggml_sycl_fusion_alias_admit(?:_chain)?\(\s*" + re.escape(site) + r"\b")


def gate_violations(source: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(source)

    graph = function(code, "static void ggml_backend_sycl_graph_compute_impl(")
    if graph is None:
        # the graph loop has been renamed; search the whole file rather than silently passing
        graph = code

    for site, launch in CHAIN_SITES:
        launches = [m.start() for m in re.finditer(re.escape(launch), graph)]
        gates = [m.start() for m in gate_call(site).finditer(graph)]
        if len(launches) != 1:
            found.append(
                f"{site}: expected exactly one launch `{launch}`, found {len(launches)} -- a second launch of "
                "the fused kernel needs its own gate call"
            )
            continue
        if len(gates) != 1:
            found.append(f"{site}: expected exactly one gate call for this site, found {len(gates)}")
            continue
        if not gates[0] < launches[0]:
            found.append(f"{site}: the gate call must precede the launch `{launch}`")
            continue
        # The gate must be part of the condition that guards the launch: no `continue;` or closing of the
        # enclosing statement between them, and the call must sit inside an `if (` ... `)` condition.
        between = graph[gates[0]:launches[0]]
        if "continue;" in between:
            found.append(f"{site}: a `continue;` sits between the gate call and the launch")
        head = graph[max(0, gates[0] - 1200):gates[0]]
        last_if = head.rfind("if (")
        if last_if < 0 or head[last_if:].count("(") - head[last_if:].count(")") < 1:
            found.append(f"{site}: the gate call is not inside an `if (...)` condition guarding the launch")

    for site, signature, markers in FUNCTION_SITES:
        body = function(code, signature)
        if body is None:
            found.append(f"{site}: {signature} is missing")
            continue
        gates = [m.start() for m in gate_call(site).finditer(body)]
        if len(gates) != 1:
            found.append(f"{site}: expected exactly one gate call inside {signature}, found {len(gates)}")
            continue
        for marker in markers:
            at = body.find(marker)
            if at < 0:
                found.append(f"{site}: `{marker}` is missing from {signature}; update this gate with the launch")
            elif not gates[0] < at:
                found.append(f"{site}: the gate call must precede `{marker}` in {signature}")
        # A declined fusion must return false (the router's `reject` helper returns false) so the unfused
        # kernels run.
        tail = body[gates[0]:gates[0] + 700]
        if "return false;" not in tail and "return reject(" not in tail:
            found.append(f"{site}: no `return false;` follows the gate call, so a decline would still launch")

    return found


def test_every_fused_site_calls_the_gate() -> None:
    assert gate_violations(SOURCE.read_text()) == []


def mutate_remove_gate(source: str, site: str) -> str:
    """Replace the gate call at `site` with `true` / `(void) 0`, i.e. the fused kernel launches unconditionally."""
    pattern = re.compile(
        r"ggml_sycl_fusion_alias_admit(?:_chain)?\(\s*" + re.escape(site) + r"\b[^;{]*?\)(?=\s*(?:\)|\{|&&|;|\|\|))",
        re.S,
    )
    mutated, n = pattern.subn("true", source, count=1)
    assert n == 1, f"could not isolate the gate call for {site} to remove it"
    return mutated


def test_removing_each_gate_is_witnessed() -> None:
    """Delete the gate call at each site in turn and confirm the checker names that site: a positive control
    against the check itself being vacuous."""
    source = SOURCE.read_text()
    for site, _ in CHAIN_SITES:
        violations = gate_violations(mutate_remove_gate(source, site))
        assert any(v.startswith(site + ":") for v in violations), f"removing the {site} gate was not witnessed: {violations}"
    for site, _, _ in FUNCTION_SITES:
        violations = gate_violations(mutate_remove_gate(source, site))
        assert any(v.startswith(site + ":") for v in violations), f"removing the {site} gate was not witnessed: {violations}"


def test_a_second_ungated_launch_is_witnessed() -> None:
    source = SOURCE.read_text()
    for site, launch in CHAIN_SITES:
        mutated = source.replace(launch, launch + ");\n    " + launch, 1)
        assert mutated != source
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ": expected exactly one launch") for v in violations), (
            f"a duplicated {site} launch was not witnessed: {violations}"
        )


def test_gate_after_launch_is_witnessed() -> None:
    """Move the bit0 gate calls below the launch: the check must notice the order, not just the presence."""
    source = SOURCE.read_text()
    for site, signature, markers in FUNCTION_SITES:
        body = function(source, signature)
        assert body is not None
        gate = gate_call(site).search(body)
        assert gate is not None
        # Take the gate call out of its place before the launch and append an equivalent call at the end of the
        # function, after the launch, so the call is present once but too late.
        moved = body.replace(gate.group(0), "(void) 0; (void) 0", 1)
        close = moved.rindex("}")
        moved = moved[:close] + gate.group(0) + "{}, 0, nullptr, 0);\n" + moved[close:]
        mutated_body = moved
        mutated = source.replace(body, mutated_body, 1)
        assert mutated != source
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ":") and "precede" in v for v in violations), (
            f"a {site} gate after the launch was not witnessed: {violations}"
        )


def test_an_ignored_decline_is_witnessed() -> None:
    """The gate is called but its answer is dropped, so a decline would still launch the fused kernel."""
    source = SOURCE.read_text()
    for site, signature, _ in FUNCTION_SITES:
        body = function(source, signature)
        assert body is not None
        gate = gate_call(site).search(body)
        assert gate is not None
        tail = body[gate.start():gate.start() + 700]
        mutated_tail = re.sub(r"return (?:false|reject\(\"alias\"\));", "(void) 0;", tail, count=1)
        assert mutated_tail != tail, f"could not find the decline return after the {site} gate"
        mutated = source.replace(body, body.replace(tail, mutated_tail, 1), 1)
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ":") and "return false" in v for v in violations), (
            f"an ignored {site} decline was not witnessed: {violations}"
        )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
