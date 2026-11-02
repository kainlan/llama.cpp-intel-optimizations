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

REACHABILITY IS NOT PROVEN HERE (llama.cpp-s36q review M1). mmvq.cpp has two consumer
switches over the same helpers: the one in mmvq_moe_batched_dispatch is live, the one
inside ggml_sycl_mul_mat_id_vec_q is not -- that function returns false at its
"type_unsupported" refusal for every type outside {Q4_0, Q8_0, MXFP4}, before the switch
is reached. This gate checks that BOTH switches enumerate the capability set and call a
helper that launches each of their labels; a type passing it in the dead switch is not
evidence that the type is served, only that the dead arm is consistent.

Design notes, both learned the hard way on this ticket:
  * The switch list is ENUMERATED FROM SOURCE, never hardcoded by line number --
    line numbers drift every commit and a stale list silently checks nothing.
  * Exemptions carry a PREMISE ASSERT: each exempt site must still be found. An
    exemption that stops matching would otherwise exempt nothing while looking
    like it still applies, which is how a control quietly dies.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "ggml" / "src" / "ggml-sycl" / "moe-mmvq-tables.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
# Mutation-check hook: point the gate at a doctored copy of mmvq.cpp.
if os.environ.get("GGML_SYCL_S36Q_MMVQ_SOURCE"):
    MMVQ = Path(os.environ["GGML_SYCL_S36Q_MMVQ_SOURCE"])

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


LIVE_KEY = "mmvq_moe_batched_dispatch: default returns false"


def consumer_switch_bodies(text):
    """[(key, body)] for every `switch (src0->type)` that is a MoE consumer.

    A consumer is a switch whose default arm aborts, OR (llama.cpp-s36q review M1) one
    whose arms call a generic AoS-id submit helper. The second form is the live
    consumer in mmvq_moe_batched_dispatch: its default is `return false`, so a missing
    case does not abort, it silently drops the op onto the fallback -- and until then
    this gate enumerated only the abort-defaulted switches and never saw it.

    Brace-matched rather than regexed to the closing brace, because these switches
    contain nested blocks.
    """
    out = []
    for m in re.finditer(r"switch\s*\(\s*src0->type\s*\)\s*\{", text):
        body = brace_body(text, m.end() - 1)
        if body is None:
            continue
        # Only the default arm's abort counts; an abort inside a case is a
        # layout/shape refusal for a type the switch DOES handle. `body` is the
        # brace-matched interior, so the default arm runs to the end of it --
        # do not anchor on a trailing brace that is by construction not here.
        dm = re.search(r"\bdefault\s*:(.*)$", body, re.S)
        if not dm:
            continue
        am = re.search(r'GGML_ABORT\(\s*"([^"]*)"', dm.group(1))
        if am:
            out.append((am.group(1), body))
        elif any(re.search(r"\b" + h + r"\s*\(", body) for h in HELPERS):
            out.append((LIVE_KEY, body))
    return out


def consumer_switches(text):
    """[(key, {types})] for every consumer switch."""
    return [(key, set(re.findall(r"case\s+(GGML_TYPE_[A-Z0-9_]+)\s*:", body)))
            for key, body in consumer_switch_bodies(text)]


# Capability types served by a dedicated launcher rather than the generic AoS
# submit helpers. Q4_0/Q8_0 have mul_mat_vec_q{4_0,8_0}_q8_1_id_sycl; MXFP4 has its
# own batched executors.
DEDICATED_LAUNCHER_TYPES = {"GGML_TYPE_Q4_0", "GGML_TYPE_Q8_0", "GGML_TYPE_MXFP4"}


HELPERS = ("mmvq_submit_q1_nvfp4_aos_id", "mmvq_submit_quant_aos_id")


def brace_body(text, open_idx):
    """Interior of the brace block opening at text[open_idx], or None."""
    depth = 0
    for k in range(open_idx, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:k]
    return None


LABEL = re.compile(r"\bcase\s+(GGML_TYPE_[A-Z0-9_]+)\s*:|\bdefault\s*:")


def switch_arms(body):
    """Split a switch body into arms: [(labels, arm_text)].

    Stacked labels (`case A: case B:` with nothing between) share one arm. An arm's
    text runs to the next label group.
    """
    marks = list(LABEL.finditer(body))
    arms = []
    i = 0
    while i < len(marks):
        labels = []
        j = i
        while j < len(marks):
            labels.append(marks[j].group(1) or "default")
            nxt = marks[j + 1] if j + 1 < len(marks) else None
            if nxt is None or body[marks[j].end():nxt.start()].strip():
                break
            j += 1
        end = marks[j + 1].start() if j + 1 < len(marks) else len(body)
        arms.append((labels, body[marks[j].end():end]))
        i = j + 1
    return arms


def helper_launches(text):
    """{helper: {type: launched}} for the generic AoS-id submit helpers.

    A type counts as launchable by a helper only if one of the helper's switch arms
    labelled with that type contains a launch `mmvq_submit_aos_id_impl<THAT_TYPE,`. A
    label with no launch (a renamed case, a body that is just `return false;`, a body
    launching a different type) is not a launcher, which is the same gap from the
    other side as a missing consumer case: the consumer's submit call returns false
    and aborts (llama.cpp-s36q).
    """
    out = {h: set() for h in HELPERS}
    for name in HELPERS:
        for m in re.finditer(r"\bbool\s+" + name + r"\s*\(", text):
            j = m.end()
            while j < len(text) and text[j] not in "{;":
                j += 1
            if j >= len(text) or text[j] != "{":
                continue
            fbody = brace_body(text, j)
            if fbody is None:
                continue
            sm = re.search(r"switch\s*\(\s*weight_type\s*\)\s*\{", fbody)
            if not sm:
                continue  # the if/else overload; the vector-deps one is the switch form
            sbody = brace_body(fbody, sm.end() - 1)
            for labels, arm in switch_arms(sbody):
                launched = set(re.findall(r"mmvq_submit_aos_id_impl\s*<\s*(GGML_TYPE_[A-Z0-9_]+)\s*,", arm))
                out[name] |= {l for l in labels if l in launched}
    return out


def consumer_arm_helpers(text):
    """[(switch key, [(labels, helper)])]: which helper each consumer arm calls."""
    out = []
    for key, body in consumer_switch_bodies(text):
        arms = []
        for labels, arm in switch_arms(body):
            for h in HELPERS:
                if re.search(r"\b" + h + r"\s*\(", arm):
                    arms.append((labels, h))
        out.append((key, arms))
    return out


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

    launches = helper_launches(mmvq_src)
    helper_types = set().union(*launches.values())
    if not helper_types:
        print("FAIL: found no launch in the generic AoS-id submit helpers in mmvq.cpp; "
              "the launcher check below would pass vacuously.")
        return 1
    unlaunchable = sorted(cap - DEDICATED_LAUNCHER_TYPES - helper_types)
    if unlaunchable:
        failures.append(
            f"{', '.join(unlaunchable)} admitted by moe_mmvq_capability_supports_layout "
            f"but no generic AoS-id submit helper arm launches mmvq_submit_aos_id_impl<that "
            f"type, ...>, so the consumer's submit call returns false and aborts.")

    # Each consumer arm that calls a helper must call one that launches every label
    # on the arm: a type routed to the wrong helper returns false there.
    arm_calls = consumer_arm_helpers(mmvq_src)
    if not any(arms for _, arms in arm_calls):
        print("FAIL: found no consumer arm calling a generic AoS-id submit helper; "
              "the routing check below would pass vacuously.")
        return 1
    for msg, arms in arm_calls:
        for labels, helper in arms:
            wrong = sorted(l for l in labels if l != "default" and l not in launches[helper])
            if wrong:
                failures.append(
                    f'consumer switch "{msg}" routes {", ".join(wrong)} to {helper}, '
                    f"which has no arm launching it.")
    routed = set()
    for _, arms in arm_calls:
        for labels, _h in arms:
            routed |= set(labels)
    unrouted = sorted(cap - DEDICATED_LAUNCHER_TYPES - routed)
    if unrouted:
        failures.append(
            f"{', '.join(unrouted)} admitted by capability but no consumer arm calls a "
            f"generic AoS-id submit helper for it.")

    for msg, types in switches:
        if msg in EXEMPT:
            continue
        missing = sorted(cap - types)
        if missing:
            failures.append(
                f'consumer switch "{msg}" does not enumerate '
                f"{', '.join(missing)}, which moe_mmvq_capability_supports_layout "
                f"admits. A type admitted by capability that reaches this switch "
                f"hits its default arm (abort, or a silent `return false` fallback).")

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
