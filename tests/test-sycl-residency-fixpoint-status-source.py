"""Source gate for the residency probe's failure channel (llama.cpp-71hq).

`llama_residency_fixpoint` (src/llama-residency-fixpoint.h) used to take a probe that returned a bare residency, so a
backend that could not answer had only an all-zero residency to give, and the iteration read that as "every layer is
device-resident" and converged. The probe now returns a `llama_residency_probe_answer` with a status, and the host test
(test-residency-fixpoint) runs the behaviour; this gate pins the structure that keeps it fail-closed, on comment-stripped
text, because a later edit that restores the old shape would still compile and still pass the cases that never fail:

- the probe is called in exactly one place, the `ask` lambda, and nowhere else in the template body;
- `ask` reads the status BEFORE it looks at the residency (so a refusing answer with a wrong-length residency is
  PROBE_FAILED, not BUG), refuses on anything but OK, and only then length-checks and normalises;
- the three call sites (the initial call, a round, the verify's re-probe) each end the function with the result the
  moment `ask` is false: none ignores the answer and none drops it into a local that is read anyway;
- the verify compares the answer it asked for (`kept`), not a stale one;
- the default answer is NOT_ANSWERED, and the constructor's helper refuses every status but OK, so a status this build
  does not know is refused too, and the refusal text carries the reason under a greppable prefix.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src/llama-residency-fixpoint.h").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_TEMPLATE = (
    "template <typename Probe, typename Measure> llama_residency_fixpoint_result "
    "llama_residency_fixpoint(size_t n_layer, Probe probe, Measure measure)"
)
_ASK_HEAD = "const auto ask = [&](const llama_tenants * with, const char * site, llama_residency & into)"
_REFUSED = "inline bool llama_residency_fixpoint_refused(llama_residency_fixpoint_status status)"
_REFUSAL_TEXT = (
    "inline std::string llama_residency_fixpoint_refusal_text(const llama_residency_fixpoint_result & result)"
)

_SITES = [
    'if (!ask(nullptr, "initial call", residency)) { return out; }',
    'if (!ask(&tenants, "round", answer)) { return out; }',
    'if (!ask(&shrunk, "verify", kept)) { return out; }',
]

_STATUS_CHECK = "if (a.status != LLAMA_RESIDENCY_PROBE_OK) {"
_LENGTH_CHECK = "if (a.residency.size() != n_layer) {"


def block_after(text: str, head: str) -> str:
    """The braced block that follows `head` (z-spelled), through its matching brace."""
    at = text.index(z(head))
    i = text.index("{", at + len(z(head)) - 1) if z(head).endswith("{") else text.index("{", at)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
    raise AssertionError(f"unbalanced braces after {head!r}")


def template_body(code: str) -> str:
    return function_body(code, _TEMPLATE)


def ask_text(code: str) -> str:
    return block_after(template_body(code), _ASK_HEAD)


def probe_once_ok(code: str) -> bool:
    """One raw `probe(` call in the template body, and it is inside `ask`."""
    body = template_body(code)
    raw = re.compile(r"(?<![\w.])probe\(")
    return len(raw.findall(body)) == 1 and len(raw.findall(ask_text(code))) == 1


def ask_ok(code: str) -> bool:
    ask = ask_text(code)
    status_at = ask.find(z(_STATUS_CHECK))
    length_at = ask.find(z(_LENGTH_CHECK))
    if status_at == -1 or length_at == -1 or not status_at < length_at:
        return False
    refusal = block_after(ask, _STATUS_CHECK)
    if z("out.status = LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;") not in refusal or z("return false;") not in refusal:
        return False
    # the answer's residency is read only after both checks, and a success is the only `return true`
    read_at = ask.find(z("into = normalise(std::move(a.residency));"))
    return (
        read_at > length_at
        and ask.count(z("return true;")) == 1
        and ask.index(z("return true;")) > read_at
        and ask.count("a.residency") == 2  # the length check and the one read
    )


def sites_ok(code: str) -> bool:
    """Three `ask(` call sites outside the lambda, in order, each returning the result when `ask` is false."""
    body = template_body(code)
    rest = body.replace(ask_text(code), "", 1)
    if len(re.findall(r"(?<!\w)ask\(", rest)) != len(_SITES):
        return False
    pos = []
    for site in _SITES:
        if rest.count(z(site)) != 1:
            return False
        pos.append(rest.index(z(site)))
    # the verify compares the answer it just asked for
    return pos == sorted(pos) and rest.count(z("if (kept == raw) {")) == 1


def helper_ok(code: str) -> bool:
    refused = function_body(code, _REFUSED)
    text = function_body(code, _REFUSAL_TEXT)
    return (
        refused == z(_REFUSED) + "{" + z("return status != LLAMA_RESIDENCY_FIXPOINT_OK;") + "}"
        and z("if (!llama_residency_fixpoint_refused(result.status)) {") in text
        and z('return "SYCL residency fixpoint refused: " + result.reason;') in text
    )


def defaults_ok(code: str) -> bool:
    return (
        z("LLAMA_RESIDENCY_PROBE_NOT_ANSWERED = 0,") in code
        and z("llama_residency_probe_status status = LLAMA_RESIDENCY_PROBE_NOT_ANSWERED;") in code
    )


GATES = {
    "the probe has one call, in ask": probe_once_ok,
    "ask reads the status first and refuses on anything but OK": ask_ok,
    "each of the three call sites ends the function when ask is false": sites_ok,
    "the constructor's helper refuses every status but OK and carries the reason": helper_ok,
    "an unset answer is NOT_ANSWERED": defaults_ok,
}

CODE = code_of(HEADER)


def test_real_header_passes_every_clause():
    for name, ok in GATES.items():
        assert ok(CODE), f"clause failed on the real header: {name}"


def test_each_call_site_has_a_mutant_that_ignores_the_status():
    for site in _SITES:
        ignored = site.replace("if (!", "(", 1).replace(") { return out; }", ");", 1)
        assert not sites_ok(mutate(CODE, site, ignored)), f"mutant 'status ignored at {site}' slipped through"
        no_return = site.replace("return out;", "")
        assert not sites_ok(mutate(CODE, site, no_return)), f"mutant 'refusal does not return at {site}' slipped through"
    # a refusal that carries on to the next round, or breaks out of the loop into the verify, is the same fault
    round_site = _SITES[1]
    assert not sites_ok(mutate(CODE, round_site, round_site.replace("return out;", "break;")))
    assert not sites_ok(mutate(CODE, round_site, round_site.replace("return out;", "continue;")))


def test_a_second_probe_call_outside_ask_is_caught():
    bypass = mutate(CODE, _SITES[0], "probe(nullptr); " + _SITES[0])
    assert not probe_once_ok(bypass), "mutant 'a direct probe call beside ask' slipped through"
    # a residency read straight from the probe, the shape the old code had, is the same fault
    direct = mutate(CODE, _SITES[0], "residency = probe(nullptr).residency;")
    assert not probe_once_ok(direct), "mutant 'the initial residency read straight from the probe' slipped through"
    assert not sites_ok(direct), "mutant 'the initial site does not go through ask' slipped through"


def test_ask_mutants():
    # the status ignored
    assert not ask_ok(mutate(CODE, _STATUS_CHECK, "if (false) {")), "mutant 'status never checked' slipped through"
    # only OK passes, so inverting the comparison would accept a refusing probe
    assert not ask_ok(mutate(CODE, _STATUS_CHECK, "if (a.status == LLAMA_RESIDENCY_PROBE_OK) {")), \
        "mutant 'status check inverted' slipped through"
    # the length checked before the status, so a refusing answer of the wrong length is a BUG
    length_first = mutate(
        CODE,
        _STATUS_CHECK,
        'if (a.residency.size() != n_layer) { bug("x"); return false; } ' + _STATUS_CHECK,
    )
    assert not ask_ok(length_first), "mutant 'the length checked before the status' slipped through"
    # the refusal reporting the wrong status
    assert not ask_ok(
        mutate(CODE, "out.status = LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;", "out.status = LLAMA_RESIDENCY_FIXPOINT_OK;")
    ), "mutant 'PROBE_FAILED mapped to OK in ask' slipped through"
    assert not ask_ok(
        mutate(CODE, "out.status = LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;", "out.status = LLAMA_RESIDENCY_FIXPOINT_BUG;")
    ), "mutant 'a refusing probe reported as a BUG' slipped through"
    # the refusal falls through instead of returning false
    assert not ask_ok(
        mutate(CODE, "(a.reason.empty() ? \"no reason given\" : a.reason); return false;", "(a.reason.empty() ? \"no reason given\" : a.reason);")
    ), "mutant 'refusal falls through to the residency' slipped through"
    # the residency read before the checks
    early = mutate(CODE, "llama_residency_probe_answer a = probe(with);", "llama_residency_probe_answer a = probe(with); into = a.residency;")
    assert not ask_ok(early), "mutant 'the residency read before the status' slipped through"


def test_helper_and_default_mutants():
    ref = "return status != LLAMA_RESIDENCY_FIXPOINT_OK;"
    # a consumer that maps PROBE_FAILED to a pass: the exact fail-open this ticket closes
    assert not helper_ok(
        mutate(CODE, ref, "return status != LLAMA_RESIDENCY_FIXPOINT_OK && status != LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;")
    ), "mutant 'the helper passes PROBE_FAILED' slipped through"
    # a helper that lists the refusing statuses instead of excluding OK: a status this build does not know passes
    assert not helper_ok(
        mutate(
            CODE,
            ref,
            "return status == LLAMA_RESIDENCY_FIXPOINT_MEASURE_FAILED || status == LLAMA_RESIDENCY_FIXPOINT_BUG;",
        )
    ), "mutant 'the helper lists refusals and forgets PROBE_FAILED' slipped through"
    assert not helper_ok(
        mutate(CODE, ref, "return status == LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;")
    ), "mutant 'the helper refuses only PROBE_FAILED' slipped through"
    assert not helper_ok(
        mutate(CODE, "if (!llama_residency_fixpoint_refused(result.status)) {", "if (true) {")
    ), "mutant 'the refusal text is always empty' slipped through"
    assert not helper_ok(
        mutate(CODE, 'return "SYCL residency fixpoint refused: " + result.reason;', "return std::string();")
    ), "mutant 'the refusal text drops the reason' slipped through"
    # the default answer
    assert not defaults_ok(
        mutate(
            CODE,
            "llama_residency_probe_status status = LLAMA_RESIDENCY_PROBE_NOT_ANSWERED;",
            "llama_residency_probe_status status = LLAMA_RESIDENCY_PROBE_OK;",
        )
    ), "mutant 'an unset answer is OK' slipped through"
    assert not defaults_ok(
        mutate(CODE, "LLAMA_RESIDENCY_PROBE_NOT_ANSWERED = 0,", "LLAMA_RESIDENCY_PROBE_OK = 0,")
    ), "mutant 'zero is OK' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
