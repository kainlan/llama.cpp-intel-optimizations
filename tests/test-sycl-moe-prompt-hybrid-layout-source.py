#!/usr/bin/env python3
"""A prompt-phase MoE layout over a MIXED tensor is admitted by what is loaded (llama.cpp-f6zo).

THE DEFECT.  GPT-OSS 20B on the B50 at GGML_SYCL_VRAM_BUDGET_PCT=60, prompt processing (ubatch > 1),
aborts in ggml_sycl_select_moe_planned_graph_layout:

  [GRAPH-MOE-LAYOUT] prompt MoE plan selected incomplete executable layout
  tensor=blk.12.ffn_gate_exps.weight device=0 layout=soa local=0 secondary=0 host=0 missing=1

blk.12 is the one MIXED tensor: 12 experts on the device (layout xmx_tiled, the single layout the
planner budgets for MXFP4 gate/up -- there is no SOA alternate), 20 on the host (AOS, executed on the
CPU).  Layers 13-22 are all host, layers 0-11 all device.

THE CHAIN.
  1. The prompt gate/up admission asks for XMX_TILED and takes it only when the probe says
     `local == n_experts && host == 0` -- every expert on the device.  For the mixed tensor the probe
     says local=12 host=20 missing=0: every expert is accounted for, but not all on the device.
  2. It falls through to the SOA probe.  The device entries are xmx_tiled, so asking for SOA finds no
     route (allow_materialize is false) and the probe stops at the first device expert: local=0
     host=0 missing=1.  The all-host layers survive this step only because host entries need no
     layout.
  3. The candidate loop skips XMX_TILED for prompt (prompt-xmx-diagnostic-only) and keeps SOA as the
     "best" layout with missing > 0 -> GGML_ABORT "planner must budget a complete PP layout".
  4. Even past 1-3, ggml_sycl_moe_layout_for_selected_rows repeats the strict test, finds it false
     and rewrites XMX_TILED to SOA -- the same wrong question one function later.

The fault is the question, not the planner's budget.  The loaded layout is the answer: the device
entries are in xmx_tiled and the host entries are host AOS run by the CPU, and the hybrid executor
already partitions by operand residency and dispatches each operand by its actual layout.  Budgeting
an extra SOA copy would duplicate weights the single-layout planner deliberately does not duplicate.

THE CONTRACT.
  H.  moe_mmvq_prompt_layout_cover_executable (pure, moe-mmvq-tables.hpp, unit-tested by
      test-sycl-moe-mmvq-tables) is the one definition of "executable": nothing missing, nothing on a
      secondary device, at least one device entry, and device + host account for every expert.  The
      backend wraps it as ggml_sycl_moe_prompt_layout_executable(src0, device, layout) over the
      probe AT THE REQUESTED LAYOUT (pin H).
  A1. the prompt gate/up XMX_TILED admission decides through the predicate (no `local == n_experts`,
      no `host == 0`).
  A2. ggml_sycl_moe_layout_for_selected_rows decides its `keep=xmx_tiled` the same way, so it does not
      rewrite the admitted layout to SOA.
  A3. the cached planned layout is revalidated by the same predicate, or a mixed tensor would be
      invalidated and recomputed on every call.
  A4. the prompt DOWN planned-primary admission uses it too.
  A2b. selected-rows keeps XMX_TILED for DOWN exactly where admission could have chosen it for DOWN
      (GGML_SYCL_MOE_DOWN_XMX_TILED), or a mixed DOWN admitted at XMX_TILED is rewritten to a SOA route
      over xmx-only entries.
  K.  the abort prints the cover at xmx_tiled -- the layout the planner loaded -- next to the SOA
      candidate's, whose "local=0 missing=1" alone is what sent the first report astray.
  L.  an all-host tensor is logged as `no-device-entries`, not `incomplete` (it has missing=0).
  C.  counts that make a NEW strict or loose spelling fail the gate: the strict
      ggml_sycl_moe_planned_layout_complete keeps exactly its current callers (the fused executors and
      the materializers), the loose wrapper keeps exactly its two callers, and the strict probe
      spelling `.local == static_cast<size_t>` appears only where decode needs it.
  S.  ggml_sycl_moe_planned_layout_complete stays STRICT (all experts local): the fused PP executors
      and the materializers read it as "a single device pointer table exists" and must keep
      declining a mixed tensor.
  G.  the abort stays.  A prompt layout with a missing expert must still refuse loudly.

WHAT THIS DOES NOT PROVE.  It reads source text.  It does not show that the grouped xmx_tiled kernel
produces right numbers for a prompt over a partial cover, nor that the CPU arm's scatter is right for
ubatch > 1 -- that is the GPU repro at PCT=60 (-p 512 -n 0, then eval-callback against full budget),
run by the lead.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real source and requires each to
FAIL on its OWN pin, then requires the unmodified tree to pass.

THE RED, against 79ebe0d9b:
  FAIL [pin A1]: the prompt gate/up XMX_TILED admission is not decided by ... (llama.cpp-f6zo)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
TAG = "(llama.cpp-f6zo)"


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def squash(text: str) -> str:
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", text))


def region(code: str, start_marker: str, end_marker: str) -> str:
    a = code.find(start_marker)
    if a < 0:
        raise ContractError(f"FAIL: anchor not found: {start_marker!r} {TAG}")
    b = code.find(end_marker, a)
    if b < 0:
        raise ContractError(f"FAIL: end anchor not found after {start_marker!r}: {end_marker!r} {TAG}")
    return squash(code[a:b])


def predicate_call(probe: str, n_expr: str = r"[^;]*?") -> str:
    return (
        rf"moe_mmvq_prompt_layout_cover_executable\({probe}\.local, {probe}\.secondary, {probe}\.host, "
        rf"{probe}\.missing, {n_expr}\)"
    )


def check(src: str) -> None:
    code = blank_comments(src)

    # A1.  Prompt gate/up XMX_TILED admission.
    gateup = region(
        code,
        "const moe_planned_layout_probe gateup_xmx_probe =",
        "prompt-gateup-xmx-tiled-incomplete",
    )
    if not re.search(r"if \(" + predicate_call("gateup_xmx_probe") + r"\) \{", gateup) or re.search(
        r"if \([^{]*(gateup_xmx_probe\.host == 0|gateup_xmx_probe\.local == static_cast)", gateup
    ):
        raise ContractError(
            "FAIL [pin A1]: the prompt gate/up XMX_TILED admission is not decided by "
            "moe_mmvq_prompt_layout_cover_executable(local, secondary, host, missing, n_experts) over its own "
            f"probe; `local == n_experts && host == 0` rejects a mixed tensor, which then aborts {TAG}"
        )

    # A2.  moe_layout_for_selected_rows must not rewrite the admitted layout to SOA.
    keep = region(
        code,
        "const moe_planned_layout_probe xmx_probe = ggml_sycl_probe_moe_planned_layout(src0, device, layout);",
        "reason=prompt-planned-xmx-complete",
    )
    if not re.search(r"if \(" + predicate_call("xmx_probe") + r"\) \{", keep) or re.search(
        r"xmx_probe\.host == 0|xmx_probe\.local == static_cast", keep
    ):
        raise ContractError(
            "FAIL [pin A2]: ggml_sycl_moe_layout_for_selected_rows keeps xmx_tiled for a prompt only when every expert "
            "is local; it must judge the probe with moe_mmvq_prompt_layout_cover_executable, or it rewrites the "
            f"admitted layout of a mixed tensor to SOA {TAG}"
        )

    # A3.  The cached planned layout is revalidated by the same predicate.
    cached = region(
        code,
        "} else if (!prompt_mxfp4_moe || cached_layout == GGML_LAYOUT_AOS ||",
        "return cached_layout;",
    )
    if "ggml_sycl_moe_prompt_layout_executable(src0, device, cached_layout)" not in cached or (
        "ggml_sycl_moe_planned_layout_complete(src0, device, cached_layout)" in cached
    ):
        raise ContractError(
            "FAIL [pin A3]: the cached planned layout is revalidated with the strict all-local test; a mixed tensor "
            f"is invalidated and recomputed on every call {TAG}"
        )

    # A4.  The prompt DOWN planned-primary admission.
    down = region(
        code,
        "} else if (adjusted_planned == planned_layout && planned_layout != GGML_LAYOUT_AOS &&",
        "reason=prompt-planner-down-layout",
    )
    if "ggml_sycl_moe_prompt_layout_executable(src0, device, planned_layout)" not in down or (
        "ggml_sycl_moe_planned_layout_complete(src0, device, planned_layout)" in down
    ):
        raise ContractError(
            "FAIL [pin A4]: the prompt down planned-primary admission still requires every expert local "
            f"(ggml_sycl_moe_planned_layout_complete) {TAG}"
        )

    # A2b.  selected-rows keeps XMX_TILED for the roles admission can choose it for.
    keep_role = re.search(r"const bool keep_xmx_role = ([^;]*);", squash(code))
    if (
        not keep_role
        or "MOE_TENSOR_GATE" not in keep_role.group(1)
        or "MOE_TENSOR_UP" not in keep_role.group(1)
        or not re.search(r"MOE_TENSOR_DOWN && ggml_sycl_moe_down_xmx_tiled_enabled\(\)", keep_role.group(1))
        or not re.search(r"authoritative_residency_active\(device\) && keep_xmx_role\) \{", squash(code))
    ):
        raise ContractError(
            "FAIL [pin A2b]: ggml_sycl_moe_layout_for_selected_rows does not keep XMX_TILED for a DOWN tensor that "
            "admission may have chosen it for (keep_xmx_role = GATE || UP || DOWN && "
            f"ggml_sycl_moe_down_xmx_tiled_enabled()); a mixed DOWN becomes a SOA route over xmx-only entries {TAG}"
        )

    # K.  The abort names the cover at the layout the planner loaded.
    abort_body = region(code, "const moe_planned_layout_probe loaded_xmx_probe =", "const bool decode_incomplete_layout")
    if "ggml_sycl_probe_moe_planned_layout(src0, device, GGML_LAYOUT_XMX_TILED)" not in abort_body or not re.search(
        r"at xmx_tiled local=%zu secondary=%zu (?:\" \")?host=%zu missing=%zu.*loaded_xmx_probe\.local, loaded_xmx_probe\.secondary, "
        r"loaded_xmx_probe\.host, loaded_xmx_probe\.missing",
        abort_body,
    ):
        raise ContractError(
            "FAIL [pin K]: the incomplete-prompt-layout abort prints only the SOA candidate's probe (local=0 missing=1 "
            f"for a tensor whose entries are xmx_tiled); it must also print the cover at xmx_tiled {TAG}"
        )

    # L.  An all-host tensor is not "incomplete".
    if "prompt-gateup-xmx-tiled-no-device-entries" not in code or not re.search(
        r"gateup_xmx_probe\.local == 0 && gateup_xmx_probe\.missing == 0 \? \"prompt-gateup-xmx-tiled-no-device-entries\" : "
        r"\"prompt-gateup-xmx-tiled-incomplete\"",
        squash(code),
    ):
        raise ContractError(
            "FAIL [pin L]: an all-host tensor (local=0 missing=0) is logged as `prompt-gateup-xmx-tiled-incomplete`; "
            f"it must say `no-device-entries` {TAG}"
        )

    # S.  The strict test stays strict.
    strict = region(
        code,
        "static bool ggml_sycl_moe_planned_layout_complete(",
        "static bool ggml_sycl_moe_prompt_specialized_layouts_enabled(",
    )
    if (
        "return probe.ok && probe.local == static_cast<size_t>(n_experts) && probe.secondary == 0 && probe.host == 0;"
        not in strict
    ):
        raise ContractError(
            "FAIL [pin S]: ggml_sycl_moe_planned_layout_complete no longer means 'every expert is a local device "
            "entry'; the fused PP executors and the materializers read it as 'one device pointer table exists' "
            f"{TAG}"
        )

    # G.  The abort stays: a prompt layout with a missing expert must still refuse loudly.
    if len(re.findall(r"prompt MoE plan selected incomplete executable layout", code)) != 1 or not re.search(
        r"if \(prompt_incomplete_layout\) \{ const moe_planned_layout_probe loaded_xmx_probe = [^;]*; GGML_ABORT\(\"\[GRAPH-MOE-LAYOUT\] prompt MoE plan selected",
        squash(code),
    ):
        raise ContractError(
            f"FAIL [pin G]: the incomplete-prompt-layout abort was removed or weakened; the fix is to ask the loaded "
            f"layout, not to stop refusing a missing expert {TAG}"
        )

    # C.  Counts: a new strict or loose spelling fails the gate and must be decided on purpose.
    flat = squash(code)
    n_strict_callers = len(re.findall(r"ggml_sycl_moe_planned_layout_complete\(", flat))
    if n_strict_callers != 18:
        raise ContractError(
            f"FAIL [pin C1]: ggml_sycl_moe_planned_layout_complete( appears {n_strict_callers} times, expected 18 "
            "(its definition and 17 fused-executor / materializer callers); a caller that moved to the loose wrapper "
            "would let the fused path take a mixed tensor, a new caller must decide strict vs "
            f"ggml_sycl_moe_prompt_layout_executable on purpose and update this count {TAG}"
        )
    n_loose = len(re.findall(r"ggml_sycl_moe_prompt_layout_executable\(", flat))
    if n_loose != 3:
        raise ContractError(
            f"FAIL [pin C2]: ggml_sycl_moe_prompt_layout_executable( appears {n_loose} times, expected 3 (its "
            f"definition, the cached-layout revalidation and the DOWN planned-primary admission) {TAG}"
        )
    n_spell = len(re.findall(r"\.local == static_cast<size_t>", flat))
    if n_spell != 3:
        raise ContractError(
            f"FAIL [pin C3]: the strict probe spelling `.local == static_cast<size_t>` appears {n_spell} times, expected "
            "3 (ggml_sycl_moe_planned_layout_complete and the two decode phase-complete admissions); a new one in a "
            f"prompt admission path is the defect this ticket removed {TAG}"
        )

    # H.  The backend wrapper: the probe at the REQUESTED layout, judged by the predicate.
    helper = region(
        code,
        "static bool ggml_sycl_moe_prompt_layout_executable(",
        "static bool ggml_sycl_moe_prompt_specialized_layouts_enabled(",
    )
    if "ggml_sycl_probe_moe_planned_layout(src0, device, layout)" not in helper or not re.search(
        predicate_call("probe", r"static_cast<size_t>\(std::max<int64_t>\(0, n_experts\)\)"), helper
    ):
        raise ContractError(
            "FAIL [pin H]: ggml_sycl_moe_prompt_layout_executable must probe the REQUESTED layout and judge the probe "
            f"with moe_mmvq_prompt_layout_cover_executable over the tensor's n_experts {TAG}"
        )


def self_test(src: str) -> int:
    failures: list[str] = []

    def expect_fail(name: str, mutated: str, pin: str) -> None:
        if mutated == src:
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mutated)
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: failed, but not on pin {pin}: {str(e)[:120]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:60]}...)")
                return
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: mutant survived")
        print(f"  mutant {name}: SURVIVED")

    def sub(old: str, new: str) -> str:
        return src.replace(old, new, 1)

    def resub(pattern: str, repl: str) -> str:
        return re.sub(pattern, repl, src, count=1, flags=re.S)

    expect_fail(
        "gateup-back-to-all-local",
        resub(
            r"if \(moe_mmvq_prompt_layout_cover_executable\(\s*gateup_xmx_probe\.local,.*?n_experts_gateup\)\)\)\) \{",
            "if (gateup_xmx_probe.ok && gateup_xmx_probe.local == static_cast<size_t>(n_experts_gateup) && "
            "gateup_xmx_probe.secondary == 0 && gateup_xmx_probe.host == 0) {",
        ),
        "A1",
    )
    expect_fail(
        "gateup-ignores-secondary",
        resub(
            r"(moe_mmvq_prompt_layout_cover_executable\(\s*gateup_xmx_probe\.local,\s*)gateup_xmx_probe\.secondary",
            r"\g<1>0",
        ),
        "A1",
    )
    expect_fail(
        "selected-rows-back-to-all-local",
        sub(
            "if (moe_mmvq_prompt_layout_cover_executable(xmx_probe.local, xmx_probe.secondary, xmx_probe.host,\n"
            "                                                        xmx_probe.missing,\n"
            "                                                        static_cast<size_t>(std::max<int64_t>(0, "
            "n_experts)))) {",
            "if (xmx_probe.ok && xmx_probe.local == static_cast<size_t>(std::max<int64_t>(0, n_experts)) &&\n"
            "                xmx_probe.secondary == 0 && xmx_probe.host == 0) {",
        ),
        "A2",
    )
    expect_fail(
        "cached-revalidated-strictly",
        sub(
            "ggml_sycl_moe_prompt_layout_executable(src0, device, cached_layout)) {",
            "ggml_sycl_moe_planned_layout_complete(src0, device, cached_layout)) {",
        ),
        "A3",
    )
    expect_fail(
        "down-back-to-all-local",
        sub(
            "ggml_sycl_moe_prompt_layout_executable(src0, device, planned_layout)) {\n"
            "                    if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {\n"
            "                        fprintf(stderr,\n"
            "                                \"[GRAPH-MOE-LAYOUT] tensor=%s device=%d plan=1 selected=%s host_weights=%d \"\n"
            "                                \"reason=prompt-planner-down-layout\\n\",",
            "ggml_sycl_moe_planned_layout_complete(src0, device, planned_layout)) {\n"
            "                    if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {\n"
            "                        fprintf(stderr,\n"
            "                                \"[GRAPH-MOE-LAYOUT] tensor=%s device=%d plan=1 selected=%s host_weights=%d \"\n"
            "                                \"reason=prompt-planner-down-layout\\n\",",
        ),
        "A4",
    )
    expect_fail(
        "strict-complete-loosened",
        sub(
            "return probe.ok && probe.local == static_cast<size_t>(n_experts) && probe.secondary == 0 && probe.host == 0;",
            "return probe.ok && probe.secondary == 0;",
        ),
        "S",
    )
    expect_fail(
        "helper-probes-soa",
        resub(
            r"(ggml_sycl_moe_prompt_layout_executable\(.*?ggml_sycl_probe_moe_planned_layout\(src0, device, )layout\)",
            r"\g<1>GGML_LAYOUT_SOA)",
        ),
        "H",
    )
    expect_fail(
        "helper-drops-the-predicate",
        re.sub(
            r"return moe_mmvq_prompt_layout_cover_executable\(probe\.local,[^;]*;",
            "return probe.ok;",
            src,
            count=1,
        ),
        "H",
    )
    expect_fail(
        "selected-rows-keeps-gateup-only",
        resub(
            r"(keep_xmx_role\s*=\s*moe_kind == MOE_TENSOR_GATE \|\| moe_kind == MOE_TENSOR_UP)\s*\|\|\s*\(moe_kind == MOE_TENSOR_DOWN"
            r" && ggml_sycl_moe_down_xmx_tiled_enabled\(\)\);",
            r"\g<1>;",
        ),
        "A2b",
    )
    expect_fail(
        "abort-drops-the-xmx-cover",
        resub(r"at xmx_tiled local=%zu secondary=%zu \"\s*\"host=%zu missing=%zu; ", ""),
        "K",
    )
    expect_fail(
        "all-host-still-called-incomplete",
        resub(r"gateup_xmx_probe\.local == 0 && gateup_xmx_probe\.missing == 0 \?\s*\"prompt-gateup-xmx-tiled-no-device-entries\" :\s*",
              ""),
        "L",
    )
    expect_fail(
        "fused-caller-moved-to-loose-wrapper",
        sub("!ggml_sycl_moe_planned_layout_complete(src0, device, GGML_LAYOUT_XMX_TILED)) {\n        return false;",
            "!ggml_sycl_moe_prompt_layout_executable(src0, device, GGML_LAYOUT_XMX_TILED)) {\n        return false;"),
        "C1",
    )
    expect_fail(
        "new-loose-caller",
        sub("static bool ggml_sycl_moe_prompt_specialized_layouts_enabled() {",
            "static bool ggml_sycl_moe_prompt_specialized_layouts_enabled() {\n"
            "    (void) ggml_sycl_moe_prompt_layout_executable(nullptr, 0, GGML_LAYOUT_SOA);"),
        "C2",
    )
    expect_fail(
        "new-strict-spelling",
        sub("static bool ggml_sycl_moe_prompt_specialized_layouts_enabled() {",
            "static bool ggml_sycl_moe_prompt_specialized_layouts_enabled() {\n"
            "    (void) (moe_planned_layout_probe{}.local == static_cast<size_t>(1));"),
        "C3",
    )
    expect_fail(
        "abort-removed",
        sub("GGML_ABORT(\n                \"[GRAPH-MOE-LAYOUT] prompt MoE plan selected",
            "GGML_LOG_ERROR(\n                \"[GRAPH-MOE-LAYOUT] prompt MoE plan selected"),
        "G",
    )

    try:
        check(src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 15 mutants caught, each on its own pin; unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    src = BACKEND.read_text()
    if "--self-test" in argv:
        return self_test(src)
    try:
        check(src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: a prompt-phase MoE layout over a mixed tensor is admitted by the layout that is loaded "
        "(device entries) plus the host entries the CPU executes, at every admission site " + TAG
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
