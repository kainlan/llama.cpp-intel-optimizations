#!/usr/bin/env python3
"""A MUL_MAT_ID refusal must say which clause refused, and its fields must be set.

Census 5 produced 264 decode refusals reading `reason=recipe_missing
source_reason=0`.  Two defects made that pair uninformative:

  1. `source_reason` was never ASSIGNED on the RECIPE_MISSING path, so 0 was the
     struct's default -- and 0 also happens to be a real value,
     expert_resolve_reason::FOUND.  An unset field that ties with a live enum
     value is indistinguishable from a measurement, and it invites exactly the
     wrong reading ("residency succeeded").
  2. `recipe_reason` -- which names the capability clause that declined -- was
     computed, stored on the route, and then dropped before it reached the log.

Both are fixed at the source of the number rather than in the reader, so every
future refusal arrives with reasons instead of archaeology.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
BATCH = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"


def violations(batch: str, sycl: str) -> list[str]:
    found: list[str] = []

    if not re.search(r"struct moe_resolved_batch_result \{", batch):
        return ["moe_resolved_batch_result is missing"]
    if not re.search(r"const char \*\s+recipe_reason\s*=\s*nullptr\s*;", batch):
        found.append("moe_resolved_batch_result does not carry recipe_reason")

    # Every early return that sets `reject` must also set both diagnostic
    # fields. A path that sets one and not the other is how the sentinel got
    # published in the first place.
    # Column spacing inside these assignments is clang-format alignment and
    # shifts whenever a neighbouring line changes width. Matching it literally
    # makes the checker report "the path is missing" for a path that is present
    # and correct -- a false label on a real finding, which is worse than either.
    for branch, marker in (("validate", r"out\.reject\s+= reject;"),
                           ("recipe", r"out\.reject\s+= moe_batch_reject_reason::RECIPE_MISSING;")):
        m = re.search(marker, batch)
        if m is None:
            found.append(f"the {branch} reject path is missing")
            continue
        window = batch[m.start():m.start() + 420]
        if not re.search(r"out\.source_reason\s*=\s*route\.source_reason;", window):
            found.append(f"the {branch} reject path does not assign source_reason")
        if not re.search(r"out\.recipe_reason\s*=\s*route\.recipe_reason;", window):
            found.append(f"the {branch} reject path does not assign recipe_reason")

    for tag in ("MOE-PROMPT-REFUSAL", "MOE-DECODE-REFUSAL"):
        at = sycl.find(f"[{tag}] tensor=")
        if at < 0:
            found.append(f"the {tag} log line is missing")
            continue
        window = sycl[at:at + 700]
        if "recipe_reason=%s" not in window:
            found.append(f"{tag} does not report recipe_reason")
        if "source_reason=%d" not in window:
            found.append(f"{tag} does not report source_reason")

    return found


def test_anchors_exist() -> None:
    # Methodology control, first: with the struct or the log lines renamed,
    # every check below is vacuous rather than failing.
    batch = BATCH.read_text()
    sycl = SYCL.read_text()
    assert "struct moe_resolved_batch_result {" in batch
    assert "[MOE-PROMPT-REFUSAL] tensor=" in sycl and "[MOE-DECODE-REFUSAL] tensor=" in sycl


def test_refusals_carry_their_reasons() -> None:
    assert violations(BATCH.read_text(), SYCL.read_text()) == []


def test_mutations_are_witnessed() -> None:
    batch = BATCH.read_text()
    sycl = SYCL.read_text()
    batch_mutations = [
        # Restoring the sentinel: RECIPE_MISSING with source_reason left at 0.
        re.sub(r"(out\.reject        = moe_batch_reject_reason::RECIPE_MISSING;\n"
               r"            out\.occurrence    = i;\n"
               r"            out\.expert_id     = expert_id;\n)"
               r"            out\.source_reason = route\.source_reason;\n",
               r"\1", batch, count=1),
        batch.replace("    const char *            recipe_reason = nullptr;", "", 1),
        batch.replace("            out.recipe_reason = route.recipe_reason;\n", "", 1),
    ]
    for index, mutated in enumerate(batch_mutations):
        assert mutated != batch, f"batch mutation {index} did not change the source"
        assert violations(mutated, sycl), f"batch mutation {index} was not witnessed"

    sycl_mutations = [
        sycl.replace(" recipe_reason=%s\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_prompt_batch_result.occurrence", "\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_prompt_batch_result.occurrence", 1),
        sycl.replace(" recipe_reason=%s\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_decode_batch_result.occurrence", "\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_decode_batch_result.occurrence", 1),
    ]
    for index, mutated in enumerate(sycl_mutations):
        assert mutated != sycl, f"log mutation {index} did not change the source"
        assert violations(batch, mutated), f"log mutation {index} was not witnessed"
