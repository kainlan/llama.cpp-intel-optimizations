#!/usr/bin/env python3
"""Host MoE experts must read the activation row of their own slot (llama.cpp-4hg7).

THE DEFECT.  GPT-OSS 20B on the B50 with GGML_SYCL_VRAM_BUDGET_PCT=60 places the
experts of layers 12-22 on the host, and decode then emits garbage that is
byte-identical from run to run.  The same command at full budget is coherent.

The CPU-TG route in ggml_sycl_mul_mat_id (ggml-sycl.cpp, the dispatch_cpu_compute
lambda) had a "every expert shares one activation" shortcut keyed on
cpu_expert_tg_active alone:

  * a single D2H copied K floats from offset 0 of src1 into the staging buffer, and
  * every task's act_host pointed at that one copy (and the CPU batch then
    deduplicated the Q8 quantization on act_host pointer equality).

That is true of GATE and UP, whose src1 is [n_embd, 1, n_tokens] (ne11 == 1).  It is
false of DOWN, whose src1 is [n_ff, n_used, n_tokens]: ne11 == n_used (4 on
GPT-OSS), one activation row per expert slot.  So every host expert in the down
projection multiplied row 0 instead of the row of its own slot.  It only shows
when some experts are on the host, which is why full budget never saw it.  The
per-expert branch (row i11 = entry.id % ne11, destination ci * K) was already right;
it simply was not reached.

THE CONTRACT.  There is ONE fact -- the dispatch has a single activation row iff
ne11 == 1 -- and ONE bool carrying it, cpu_shared_act = cpu_expert_tg_active &&
ne11 == 1.  Inside dispatch_cpu_compute (comments blanked):

  1. cpu_shared_act is defined exactly once, as cpu_expert_tg_active && ne11 == 1;
  2. cpu_expert_tg_active is not read there at all -- every "shared" decision goes
     through cpu_shared_act, so there is no second source for the fact;
  3. the single-D2H branch is conditioned on cpu_shared_act (and n_cpu > 1);
  4. the shared pointer choice is conditioned on cpu_shared_act at BOTH task
     builders (task.activations for the Q1_0/NVFP4 recipe path, t.act_host);
  5. the per-expert fall-through survives: it copies row (entry.id % ne11) of token
     entry.iid1 to ci * K, and both builders still fall back to act_pinned + ci * K.

WHAT THIS DOES NOT PROVE.  It reads source text.  It shows the shortcut can no longer
be taken when ne11 > 1 and that the per-expert path it falls to is still the
row-selecting one.  It does not show that the numbers are right, that the CPU kernel
consumes the staged rows correctly, or that decode is coherent -- that needs the GPU
repro (GGML_SYCL_VRAM_BUDGET_PCT=60 on the B50), which is run by the lead.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real source and
requires each to FAIL, then requires the unmodified tree to pass.  Run it that way
(it is how ctest registers it): a text check whose anchors stop matching passes
vacuously, and the mutants prove each anchor is still load-bearing.

THE ORIGINAL RED, recorded before the fix (against master d8a67422d):
  FAIL: cpu_shared_act is not defined exactly once, ahead of dispatch_cpu_compute, as
  `cpu_expert_tg_active && ne11 == 1` in ggml_sycl_mul_mat_id (llama.cpp-4hg7)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

LAMBDA_MARKER = "auto dispatch_cpu_compute = [&]("
SHARED = "cpu_shared_act"
TG_ACTIVE = "cpu_expert_tg_active"

DEFINITION = re.compile(r"const\s+bool\s+cpu_shared_act\s*=\s*cpu_expert_tg_active\s*&&\s*ne11\s*==\s*1\s*;")


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines."""
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


def brace_block_from(src: str, start: int) -> str:
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError("FAIL: expected a brace block and found none")
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
    raise ContractError("FAIL: brace block is unbalanced")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check(backend_src: str) -> None:
    code = blank_comments(backend_src)

    # 1. one definition of the fact.  It must sit before the lambda (the lambda captures it).
    lam_at = code.find(LAMBDA_MARKER)
    if lam_at < 0:
        raise ContractError("FAIL: dispatch_cpu_compute lambda not found in ggml-sycl.cpp (llama.cpp-4hg7)")
    defs = list(DEFINITION.finditer(code))
    if len(defs) != 1 or defs[0].start() > lam_at:
        raise ContractError(
            "FAIL: cpu_shared_act is not defined exactly once, ahead of dispatch_cpu_compute, as "
            "`cpu_expert_tg_active && ne11 == 1` in ggml_sycl_mul_mat_id (llama.cpp-4hg7)"
        )

    body = squash(brace_block_from(code, lam_at))

    # 2. the lambda never consults cpu_expert_tg_active directly.
    if re.search(r"\b" + TG_ACTIVE + r"\b", body):
        raise ContractError(
            "FAIL: dispatch_cpu_compute reads cpu_expert_tg_active directly, so the 'all experts share one "
            "activation' decision has a second source that ignores ne11 (llama.cpp-4hg7)"
        )

    # 3. the single-D2H branch.
    single = re.search(r"else if \(\s*cpu_shared_act\s*&&\s*n_cpu\s*>\s*1\s*\)", body)
    if not single:
        raise ContractError(
            "FAIL: the single-D2H branch (one activation copied for all host experts) is not conditioned on "
            "`cpu_shared_act && n_cpu > 1`, so with ne11 > 1 every host expert reads row 0 (llama.cpp-4hg7)"
        )

    # 4. both task builders choose the shared pointer on cpu_shared_act only.
    for what, lhs in (("task.activations", r"task\.activations"), ("t.act_host", r"t\.act_host")):
        pat = re.compile(
            lhs + r"\s*=\s*cpu_shared_act\s*\?\s*\(\s*act_on_host\s*\?\s*shared_act_host\s*:\s*act_pinned\s*\)\s*:\s*"
            r"act_pinned\s*\+\s*ci\s*\*\s*static_cast<size_t>\(K\)\s*;"
        )
        if not pat.search(body):
            raise ContractError(
                f"FAIL: {what} does not pick the shared activation on `cpu_shared_act` with an "
                "`act_pinned + ci * K` per-expert fallback (llama.cpp-4hg7)"
            )

    # 5. the per-expert copy selects the slot's row, of the entry's token, into slot ci.
    per_expert = re.search(
        r"const int64_t i11\s*=\s*entry\.id\s*%\s*ne11\s*;\s*const int64_t i12\s*=\s*entry\.iid1\s*;"
        r".*?const size_t dst_off\s*=\s*ci\s*\*\s*static_cast<size_t>\(K\)\s*\*\s*sizeof\(float\)\s*;",
        body,
    )
    if not per_expert:
        raise ContractError(
            "FAIL: the per-expert D2H no longer copies row (entry.id % ne11) of token entry.iid1 into slot "
            "ci * K (llama.cpp-4hg7)"
        )
    if "src_off" not in body or not re.search(r"static_cast<size_t>\(i11\)\s*\*\s*nb11", body):
        raise ContractError("FAIL: the per-expert D2H source offset no longer uses i11 * nb11 (llama.cpp-4hg7)")


def self_test(backend_src: str) -> int:
    failures: list[str] = []

    def expect_fail(name: str, mutated: str) -> None:
        if mutated == backend_src:
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mutated)
        except ContractError:
            print(f"  mutant {name}: caught")
            return
        failures.append(f"{name}: mutant survived")
        print(f"  mutant {name}: SURVIVED")

    def sub(old: str, new: str, in_lambda: bool = True) -> str:
        """Replace the first `old` at or after the dispatch_cpu_compute lambda.

        The secondary-GPU dispatch earlier in the file carries look-alike lines
        (i11/i12/dst_off); a mutant must land in the lambda under test.
        """
        start = backend_src.find(LAMBDA_MARKER) if in_lambda else 0
        at = backend_src.find(old, max(start, 0))
        if at < 0:
            return backend_src
        return backend_src[:at] + new + backend_src[at + len(old) :]

    # m1: the fact drops ne11 (the original defect).
    expect_fail("fact-drops-ne11", sub("const bool cpu_shared_act = cpu_expert_tg_active && ne11 == 1;",
                                       "const bool cpu_shared_act = cpu_expert_tg_active;",
                                       in_lambda=False))
    # m2: the single-D2H branch reverts to the TG flag alone.
    expect_fail("single-d2h-on-tg-flag", sub("else if (cpu_shared_act && n_cpu > 1)",
                                             "else if (cpu_expert_tg_active && n_cpu > 1)"))
    # m3: the CPU task pointer reverts to the TG flag alone.
    expect_fail("task-act-host-on-tg-flag", sub("t.act_host     = cpu_shared_act ?",
                                                "t.act_host     = cpu_expert_tg_active ?"))
    # m4: the recipe task pointer reverts to the TG flag alone.
    expect_fail("recipe-activations-on-tg-flag", sub("task.activations               = cpu_shared_act ?",
                                                     "task.activations               = cpu_expert_tg_active ?"))
    # m5: the per-expert fallback loses its slot offset (every task would read slot 0's copy).
    expect_fail("task-act-host-no-slot-offset", sub(
        "act_pinned + ci * static_cast<size_t>(K);\n                    t.output_host",
        "act_pinned;\n                    t.output_host"))
    # m6: the per-expert D2H stops selecting the slot's row.
    expect_fail("per-expert-row-fixed", sub("const int64_t i11     = entry.id % ne11;",
                                            "const int64_t i11     = 0;"))
    # m7: the per-expert D2H stops selecting the token.
    expect_fail("per-expert-token-fixed", sub("const int64_t i12     = entry.iid1;",
                                              "const int64_t i12     = 0;"))
    # m8: the per-expert D2H writes every slot to the same staging offset.
    expect_fail("per-expert-dst-collapsed", sub(
        "const size_t dst_off = ci * static_cast<size_t>(K) * sizeof(float);",
        "const size_t dst_off = 0;"))
    # m9: a second, unconditioned shared decision appears in the lambda.
    expect_fail("second-source-in-lambda", sub(
        "if (act_on_host && cpu_shared_act) {",
        "if (act_on_host && cpu_expert_tg_active) {"))

    try:
        check(backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 9 mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    backend_src = BACKEND.read_text()
    if "--self-test" in argv:
        return self_test(backend_src)
    try:
        check(backend_src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: the host-expert CPU dispatch shares one activation only when ne11 == 1, and otherwise copies "
        "each slot's own row (llama.cpp-4hg7)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
