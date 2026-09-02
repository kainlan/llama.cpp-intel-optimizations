#!/usr/bin/env python3
"""The decode-shape FLASH_ATTN_EXT graph-replay gate's verified-safe kernel
set must stay a POSITIVE, NAMED allowlist -- source-asserted so a later
"just weaken the check" fix cannot silently turn it into a blanket relax
without a test catching it.

llama.cpp-86a7: fa_decode_kernel_observation (common.hpp) classifies every
observed decode-shape (ne01<=1) FLASH_ATTN_EXT dispatch into a named
kernel_family (ESIMD_PARTITIONED, D512_TILE, or the unverified fallback
OTHER) and only engages SYCL command-graph replay in AUTO mode when EVERY
observation fell into a verified-safe family -- other_kernel_count == 0.
D512_TILE (gemma4's D=512 global-attention layers, launch_fattn_tile_d512)
was added to the allowlist in this round after individually establishing
its launch has none of the graph-hostile properties that excluded it
before (no wait()/malloc at submission, shared graph-input-staging pointer
resolution, and oneDNN -- the one dispatch-time-only decline in this
kernel's neighborhood -- structurally cannot reach decode at all, ne01=1
being below its own MIN_NCOLS floor).

The two failure modes this guards against are specific and have specific
fixes elsewhere if they are ever genuinely needed: (1) classify() stops
being driven by named string comparisons and starts returning a "safe"
family unconditionally (a blanket relax that would silently readmit any
future or misidentified kernel), and (2) all_verified_safe() drops its
other_kernel_count == 0 requirement (turning a closed allowlist into an
open one). Widening the allowlist legitimately means adding a new,
individually-verified enumerator and a classify() case for it -- exactly
what this round did for D512_TILE -- not touching either of these two
properties.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/common.hpp"

OBSERVATION_STRUCT = "struct fa_decode_kernel_observation {"
CLASSIFY_FN = "static kernel_family classify(const char * kernel) {"
ALL_VERIFIED_SAFE_FN = "bool all_verified_safe() const {"


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy
    of the walker in the fork's other source-gate tests (e.g.
    test-sycl-graph-weight-layout-fallback-source.py) -- this fork's
    convention is one self-contained file per check, not a shared import."""
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


def function_or_none(text: str, signature: str) -> str | None:
    if signature not in text:
        return None
    return function(text, signature)


def allowlist_violations(source: str) -> list[str]:
    """The gate must stay a positive, named allowlist: classify() driven by
    real string comparisons with an unverified OTHER fallback, and
    all_verified_safe() requiring other_kernel_count == 0."""
    found: list[str] = []

    struct_body = function_or_none(source, OBSERVATION_STRUCT)
    if struct_body is None:
        return [f"{OBSERVATION_STRUCT} is missing"]

    # The named set itself: at minimum the two verified-safe families this
    # round established, plus the unverified fallback they are checked
    # against. A named enum, not a bare integer threshold.
    for member in ("ESIMD_PARTITIONED", "D512_TILE", "OTHER"):
        if member not in struct_body:
            found.append(f"kernel_family is missing the '{member}' enumerator")

    classify_body = function_or_none(struct_body, CLASSIFY_FN)
    if classify_body is None:
        found.append(f"{CLASSIFY_FN} is missing")
    else:
        # Both verified-safe families must be reached through an actual
        # named string comparison, not assumed.
        for kernel_name, family in (("esimd_f16", "ESIMD_PARTITIONED"), ("d512_tile", "D512_TILE")):
            guard = re.search(
                r'if \(std::strcmp\(kernel, "' + re.escape(kernel_name) + r'"\) == 0\) \{\s*'
                r"return kernel_family::" + family + r";",
                classify_body,
            )
            if not guard:
                found.append(
                    f'classify() no longer guards "{family}" behind an std::strcmp(kernel, "{kernel_name}") '
                    "check -- this is the blanket-relax failure mode: an unconditional or unguarded return "
                    "here would silently readmit any kernel name, not just the two verified ones"
                )

        # The fallback for anything not explicitly matched above must be the
        # unverified family. This is the single line that keeps a future,
        # unrecognized kernel name excluded by default.
        if not re.search(r"return kernel_family::OTHER;\s*\}\s*$", classify_body):
            found.append(
                "classify()'s fallback return is not kernel_family::OTHER -- an unrecognized kernel name "
                "would be silently admitted into the verified-safe set instead of correctly excluded"
            )

    safe_body = function_or_none(struct_body, ALL_VERIFIED_SAFE_FN)
    if safe_body is None:
        found.append(f"{ALL_VERIFIED_SAFE_FN} is missing")
    elif "other_kernel_count == 0" not in safe_body:
        found.append(
            "all_verified_safe() no longer requires other_kernel_count == 0 -- this is the open-vs-closed "
            "allowlist failure mode: without it, any number of unverified-kernel observations would still "
            "report the decode graph as safe to auto-engage"
        )

    return found


def test_decode_kernel_allowlist_is_positive_and_named() -> None:
    assert allowlist_violations(SOURCE.read_text()) == []


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()

    # Blanket-relax mutation: classify()'s OTHER fallback silently starts
    # reporting D512_TILE (verified-safe) instead of the unverified default.
    # This is the exact shape a careless future "just make gemma4 always
    # engage" fix could take.
    relaxed = source.replace(
        "            return kernel_family::OTHER;\n        }",
        "            return kernel_family::D512_TILE;\n        }",
        1,
    )
    assert relaxed != source, "blanket-relax mutation did not change the source"
    relaxed_violations = allowlist_violations(relaxed)
    assert any("fallback return is not kernel_family::OTHER" in v for v in relaxed_violations), (
        f"blanket-relax mutation was not witnessed: {relaxed_violations}"
    )

    # Open-allowlist mutation: all_verified_safe() drops its exclusion
    # requirement, so any number of unverified-kernel observations no
    # longer blocks auto-engagement.
    opened = source.replace(
        "        bool all_verified_safe() const {\n"
        "            return (esimd_partitioned_count + d512_tile_count) > 0 && other_kernel_count == 0;\n"
        "        }",
        "        bool all_verified_safe() const {\n"
        "            return (esimd_partitioned_count + d512_tile_count) > 0;\n"
        "        }",
        1,
    )
    assert opened != source, "open-allowlist mutation did not change the source"
    opened_violations = allowlist_violations(opened)
    assert any("no longer requires other_kernel_count == 0" in v for v in opened_violations), (
        f"open-allowlist mutation was not witnessed: {opened_violations}"
    )
