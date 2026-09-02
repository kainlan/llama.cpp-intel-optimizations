#!/usr/bin/env python3
"""The RMSNorm->MUL_MAT 3-way fusion (FUSION_MASK bit3) must decline
whenever the unfused MUL_MAT for the same (W, x) would take the oneDNN PP
route, rather than silently landing its GEMM on mmq_generic -- source
gated because the whole point of this check is a route DECISION that a
later "simplify" pass could delete without any functional test noticing
(mmq_generic still produces correct numbers, just ~7.7x slower).

llama.cpp-1etg: S6/qmen's profiler coverage (2026-09-01, gemma4 B70 pp512)
measured the fusion's 18 GEMMs/ubatch running on mmq_generic at ~1.82 ms
each (32.7 ms/ubatch), while the identical m=10240 shape runs unfused
through oneDNN WOQ at 0.235 ms (~114 TFLOPS) -- the fusion was COSTING
~29 ms of a 197 ms ubatch by displacing a route 7.7x faster than the one
it lands on. The fix is a route predicate
(ggml_sycl_can_fuse_rmsnorm_mulmat calls the same
ggml_sycl_onednn_pp_candidate the unfused dispatcher itself uses, on
mulmat->src[1] -- the exact tensor the unfused MUL_MAT would consume),
not a blanket mask or an M-threshold, so that fusion is preserved for
M in [8,16) and for weight types oneDNN PP skips.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

PREDICATE_FN = "static bool ggml_sycl_can_fuse_rmsnorm_mulmat(const ggml_tensor * rms_norm,"
CALL_SITE = "ggml_sycl_can_fuse_rmsnorm_mulmat(node, mul_node, mulmat_node, sycl_ctx->device)"
ROUTE_DECLINE = "if (ggml_sycl_onednn_pp_candidate(mulmat->src[0], mulmat->src[1], mulmat, device)) {"


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy
    of the walker in this fork's other source-gate tests -- the convention
    is one self-contained file per check, not a shared import."""
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


def route_decline_violations(source: str) -> list[str]:
    """The predicate must exist, must decline via the shared
    ggml_sycl_onednn_pp_candidate check (not a re-derived condition), and
    must actually be consulted at the 3-way fusion's admission site."""
    found: list[str] = []

    body = function_or_none(source, PREDICATE_FN)
    if body is None:
        return [f"{PREDICATE_FN} is missing"]

    # The route-decline check must be present and must return false --
    # i.e. it is a decline, not merely observed and ignored.
    decline = re.search(
        r"if \(ggml_sycl_onednn_pp_candidate\(mulmat->src\[0\], mulmat->src\[1\], mulmat, device\)\) \{\s*"
        r"return false;\s*\}",
        body,
    )
    if not decline:
        found.append(
            "ggml_sycl_can_fuse_rmsnorm_mulmat no longer declines via "
            "ggml_sycl_onednn_pp_candidate(mulmat->src[0], mulmat->src[1], mulmat, device) -- this is the exact "
            "check that stops the fusion from displacing the faster unfused oneDNN PP route; without it the "
            "fusion silently lands its GEMM on mmq_generic again, ~7.7x slower, with no test catching it because "
            "mmq_generic still produces correct numbers"
        )

    # The predicate must be genuinely reachable from the fusion admission
    # site -- a correct-looking check nobody calls protects nothing.
    if CALL_SITE not in source:
        found.append(
            f"the 3-way fusion admission call site no longer reads: {CALL_SITE} -- either the predicate is no "
            "longer consulted, or the device argument (needed by ggml_sycl_onednn_pp_candidate's placement-safety "
            "check) was dropped"
        )

    return found


def test_fusion_declines_the_faster_route() -> None:
    assert route_decline_violations(SOURCE.read_text()) == []


def test_mutation_is_witnessed() -> None:
    """Reintroduce the pre-1etg predicate (no route decline at all) as a
    source mutation and confirm the checker catches it -- a positive
    control against the check itself being vacuous."""
    source = SOURCE.read_text()

    body = function(source, PREDICATE_FN)
    decline_match = re.search(
        r"    // llama\.cpp-1etg:.*?\n    if \(ggml_sycl_onednn_pp_candidate\(mulmat->src\[0\], mulmat->src\[1\], "
        r"mulmat, device\)\) \{\n        return false;\n    \}\n",
        body,
        re.S,
    )
    assert decline_match, "could not isolate the route-decline block to remove for the mutation"
    reverted_body = body.replace(decline_match.group(0), "", 1)
    reverted_source = source.replace(body, reverted_body, 1)
    assert reverted_source != source, "route-decline-removal mutation did not change the source"
    violations = route_decline_violations(reverted_source)
    assert any("no longer declines via ggml_sycl_onednn_pp_candidate" in v for v in violations), (
        f"route-decline-removal mutation was not witnessed: {violations}"
    )
