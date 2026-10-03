#!/usr/bin/env python3
"""Source gate: every MMVQ consumer switch must cover the capability set.

Why this exists (llama.cpp-gx30). tests/test-sycl-moe-mmvq-tables.cpp asserts
capability is a subset of the executors that EXIST, by launcher existence. That
is not sufficient: a consumer's own `switch (src0->type)` can be narrower than
the launchers available to it, so a type the capability query admits reaches a
consumer that never enumerated it and lands in `default: GGML_ABORT`. Census 9b
died exactly that way -- a q1_0 MUL_MAT_ID case hit the default arm of the
non-indexed roster switch, which had no Q1_0 case because until scope A those
types were refused at capability and could never arrive.

So this gate reads the other direction: for each consumer switch, does it
enumerate every type capability admits?

Design notes, both learned the hard way on this ticket:
  * The switch list is ENUMERATED FROM SOURCE, never hardcoded by line number --
    line numbers drift every commit and a stale list silently checks nothing.
  * Exemptions carry a PREMISE ASSERT: each exempt site must still be found. An
    exemption that stops matching would otherwise exempt nothing while looking
    like it still applies, which is how a control quietly dies.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "ggml" / "src" / "ggml-sycl" / "moe-mmvq-tables.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"

# Switches that legitimately need not enumerate the capability set. Keyed by the
# abort message, which is stable across edits in a way line numbers are not.
#
# Each entry carries a reason AND a `premise`: a regex over mmvq.cpp asserting
# the structural fact the exemption rests on. An exemption whose premise stops
# holding is worse than no exemption -- it keeps a real gap quiet -- so the
# premise failing turns the gate red rather than silently continuing to excuse.
EXEMPT = {
    "MMVQ streaming: unsupported layout/type": {
        "reason":
            "mmvq_build_stream_segments() returns early for GGML_LAYOUT_AOS, so this "
            "switch is reached only for non-AoS layouts. Every type gx30 added is "
            "admitted by capability for AoS ONLY, so none of them can arrive here. "
            "This is a segment-sizing helper, not a kernel dispatch -- there is no "
            "submit to wire in, and inventing per-type byte layouts to satisfy the "
            "gate would be a forced fit.",
        # If that early return goes away, AoS-only types can reach the switch and
        # the exemption is void.
        "premise": r"if\s*\(\s*layout\s*==\s*GGML_LAYOUT_AOS\s*\)\s*\{[^}]*return\s+false\s*;",
    },
}


def capability_types(text):
    """Types admitted by moe_mmvq_capability_supports_layout."""
    m = re.search(
        r"moe_mmvq_capability_supports_layout\s*\([^)]*\)\s*\{(.*?)\n\}",
        text, re.S)
    if not m:
        return None
    return set(re.findall(r"case\s+(GGML_TYPE_[A-Z0-9_]+)\s*:", m.group(1)))


def consumer_switches(text):
    """Every `switch (src0->type)` whose default arm aborts.

    Returns [(abort_message, {types})]. Brace-matched rather than regexed to the
    closing brace, because these switches contain nested blocks.
    """
    out = []
    for m in re.finditer(r"switch\s*\(\s*src0->type\s*\)\s*\{", text):
        i = m.end() - 1
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    body = text[i:j]
                    break
        else:
            continue
        # Only the default arm's abort counts; an abort inside a case is a
        # layout/shape refusal for a type the switch DOES handle. `body` is the
        # brace-matched interior, so the default arm runs to the end of it --
        # do not anchor on a trailing brace that is by construction not here.
        dm = re.search(r"\bdefault\s*:(.*)$", body, re.S)
        if not dm:
            continue
        am = re.search(r'GGML_ABORT\(\s*"([^"]*)"', dm.group(1))
        if not am:
            continue
        types = set(re.findall(r"case\s+(GGML_TYPE_[A-Z0-9_]+)\s*:", body))
        out.append((am.group(1), types))
    return out


# Capability types served by a dedicated launcher rather than the generic AoS
# submit helpers. Q4_0/Q8_0 have mul_mat_vec_q{4_0,8_0}_q8_1_id_sycl; MXFP4 has its
# own batched executors.
DEDICATED_LAUNCHER_TYPES = {"GGML_TYPE_Q4_0", "GGML_TYPE_Q8_0", "GGML_TYPE_MXFP4"}


def submit_helper_types(text):
    """Types the generic AoS-id submit helpers (mmvq_submit_q1_nvfp4_aos_id,
    mmvq_submit_quant_aos_id) can launch: the case labels of their definitions.

    Consumer switches above only prove the consumer REACHES a launcher. A type
    advertised by capability whose helper has no case returns false there and
    the consumer aborts (llama.cpp-s36q), which is the same gap from the other
    side.
    """
    types = set()
    for name in ("mmvq_submit_q1_nvfp4_aos_id", "mmvq_submit_quant_aos_id"):
        for m in re.finditer(r"\bbool\s+" + name + r"\s*\(", text):
            j = m.end()
            while j < len(text) and text[j] not in "{;":
                j += 1
            if j >= len(text) or text[j] != "{":
                continue
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        types |= set(re.findall(r"case\s+(GGML_TYPE_[A-Z0-9_]+)\s*:", text[j:k]))
                        break
    return types


def main():
    failures = []

    for p in (TABLES, MMVQ):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    cap = capability_types(TABLES.read_text(encoding="utf-8"))
    if not cap:
        print("FAIL: could not parse moe_mmvq_capability_supports_layout. "
              "The parser matched nothing, so every check below would pass "
              "vacuously.")
        return 1

    switches = consumer_switches(MMVQ.read_text(encoding="utf-8"))

    # Premise 1: the enumeration must actually find switches. A regex that stops
    # matching (a rename, a reformat) would otherwise report a clean sweep over
    # an empty set.
    if len(switches) < 2:
        print(f"FAIL: found only {len(switches)} consumer switch(es) in mmvq.cpp. "
              "The enumeration is broken -- it is not credible that the backend "
              "has fewer than two `switch (src0->type)` dispatch sites.")
        return 1

    # Premise 2: every exemption must still match a switch, AND the structural
    # fact it rests on must still hold.
    mmvq_src = MMVQ.read_text(encoding="utf-8")
    seen = {msg for msg, _ in switches}
    for msg, entry in EXEMPT.items():
        if msg not in seen:
            failures.append(
                f"exemption premise broken: no consumer switch aborts with "
                f'"{msg}". The exemption now covers nothing -- either the site '
                f"was renamed (update the key) or removed (drop the entry).")
            continue
        if not re.search(entry["premise"], mmvq_src, re.S):
            failures.append(
                f'exemption premise broken for "{msg}": the structural guard it '
                f"depends on is gone. Reason on file was: {entry['reason']} "
                f"Re-verify reachability before restoring the exemption.")

    helper_types = submit_helper_types(mmvq_src)
    if not helper_types:
        print("FAIL: could not find the generic AoS-id submit helpers in mmvq.cpp; "
              "the launcher check below would pass vacuously.")
        return 1
    unlaunchable = sorted(cap - DEDICATED_LAUNCHER_TYPES - helper_types)
    if unlaunchable:
        failures.append(
            f"{', '.join(unlaunchable)} admitted by moe_mmvq_capability_supports_layout "
            f"but no generic AoS-id submit helper has a case for it, so the consumer's "
            f"submit call returns false and aborts.")

    for msg, types in switches:
        if msg in EXEMPT:
            continue
        missing = sorted(cap - types)
        if missing:
            failures.append(
                f'consumer switch "{msg}" does not enumerate '
                f"{', '.join(missing)}, which moe_mmvq_capability_supports_layout "
                f"admits. A type admitted by capability that reaches this switch "
                f"hits its default abort.")

    if failures:
        for f in failures:
            print("FAIL:", f)
        print(f"\ncapability set ({len(cap)}): {', '.join(sorted(cap))}")
        for msg, types in switches:
            tag = " [exempt]" if msg in EXEMPT else ""
            print(f'  switch "{msg}"{tag}: {len(types)} cases')
        return 1

    print(f"test-sycl-moe-mmvq-consumer-coverage: OK "
          f"({len(cap)} capability types, {len(switches)} consumer switches, "
          f"{len(EXEMPT)} exempt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
