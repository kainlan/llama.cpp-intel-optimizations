#!/usr/bin/env python3
"""GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW's three-way mapping, source-asserted so a
silent revert of the round-9 default flip is caught without a live SYCL
device.

llama.cpp-dyi3 round 9: the owner approved flipping the default -- UNSET now
means AUTO (the observation-gated mode), not FORCE_OFF -- now that decode
graph replay is proven correct on hardware (dyi3 c-z3di: llama-server
/completion chosen-token logprob deltas measured with a nondeterminism
control on both cards; forced graph sits inside the machine's own
nondeterminism envelope). Rounds 5-8 held UNSET at FORCE_OFF pending that
proof, and nothing in the test suite ever asserted the mapping either way --
this file exists so a future edit that silently reverts the default (or
breaks any of the other three cases: the explicit "auto" spelling, a numeric
value, or an unrecognized value) fails a test instead of shipping quietly.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

ALLOW_MODE_FN = "static ggml_sycl_fa_graph_allow_mode ggml_sycl_flash_attn_graph_allow_mode() {"


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy
    of the walker in test-sycl-graph-weight-layout-fallback-source.py --
    this fork's source-gate convention is one self-contained file per check,
    not a shared import, so each file carries its own copy."""
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


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    return text[start:matching_brace(text, brace) + 1]


def allow_mode_violations(source: str) -> list[str]:
    """Assert the exact three-way mapping, including the round-9 default."""
    found: list[str] = []

    if ALLOW_MODE_FN not in source:
        return [f"{ALLOW_MODE_FN} is missing"]

    body = function(source, ALLOW_MODE_FN)

    # Case 1: unset (env is null or empty) -> AUTO. This is the round-9
    # change and the whole point of this file: catch a silent revert of it.
    unset_match = re.search(
        r"if \(!env \|\| env\[0\] == '\\0'\) \{\s*return ggml_sycl_fa_graph_allow_mode::(\w+);",
        body,
    )
    if not unset_match:
        found.append("could not find the unset-env branch (env null/empty check)")
    elif unset_match.group(1) != "AUTO":
        found.append(
            f"unset GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW returns {unset_match.group(1)}, not AUTO -- "
            "this is the round-9 owner-approved default flip (dyi3 c-z3di); a silent revert to "
            "FORCE_OFF here re-disables graph replay for everyone who never sets the variable"
        )

    # Case 2: the literal string "auto" -> AUTO, unconditionally, as an
    # explicit spelling of the default.
    auto_match = re.search(
        r'if \(std::strcmp\(env, "auto"\) == 0\) \{\s*return ggml_sycl_fa_graph_allow_mode::(\w+);',
        body,
    )
    if not auto_match:
        found.append('could not find the explicit "auto" branch')
    elif auto_match.group(1) != "AUTO":
        found.append(f'explicit "auto" returns {auto_match.group(1)}, not AUTO')

    # Case 3 & 4: the final ternary maps a nonzero integer -> FORCE_ON, and
    # everything else (an explicit "0", or an unrecognized non-numeric value
    # that std::atoi also reads as 0) -> FORCE_OFF.
    ternary_match = re.search(
        r"return std::atoi\(env\) != 0 \? ggml_sycl_fa_graph_allow_mode::(\w+) : "
        r"ggml_sycl_fa_graph_allow_mode::(\w+);",
        body,
    )
    if not ternary_match:
        found.append("could not find the final nonzero-integer/fallback ternary")
    else:
        nonzero_mode, fallback_mode = ternary_match.group(1), ternary_match.group(2)
        if nonzero_mode != "FORCE_ON":
            found.append(f"a nonzero integer value returns {nonzero_mode}, not FORCE_ON")
        if fallback_mode != "FORCE_OFF":
            found.append(f'"0" and an unrecognized value return {fallback_mode}, not FORCE_OFF')

    return found


def test_allow_mode_three_way_mapping() -> None:
    assert allow_mode_violations(SOURCE.read_text()) == []


def test_mutation_is_witnessed() -> None:
    """Reintroduce the round-5-through-8 default (unset -> FORCE_OFF) as a
    source mutation and confirm the checker catches it -- a positive control
    against the check itself being vacuous."""
    source = SOURCE.read_text()
    reverted = source.replace(
        "        if (!env || env[0] == '\\0') {\n"
        "            return ggml_sycl_fa_graph_allow_mode::AUTO;\n"
        "        }",
        "        if (!env || env[0] == '\\0') {\n"
        "            return ggml_sycl_fa_graph_allow_mode::FORCE_OFF;\n"
        "        }",
        1,
    )
    assert reverted != source, "revert mutation did not change the source"
    violations = allow_mode_violations(reverted)
    assert any("round-9 owner-approved default flip" in v for v in violations), (
        f"revert mutation was not witnessed: {violations}"
    )
