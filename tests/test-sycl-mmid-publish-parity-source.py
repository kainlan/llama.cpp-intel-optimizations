#!/usr/bin/env python3
"""Both MUL_MAT_ID admissions publish allocation-owned experts before resolving.

ggml_sycl_mul_mat_id has two admissions -- a prompt one (ne12 != 1) and a decode
one (ne12 == 1) -- and each runs the non-materializing retained resolver.  That
resolver's first authority is not the unified cache: it is
ggml_sycl_try_moe_storage_handle_route reading the per-expert mem_handle slices
held on the tensor extra.  Those slices exist only because something published
them, and an ordinary backend allocation (test-backend-ops included) never goes
through the set_tensor path that normally does.

The decode admission has always published for exactly that reason.  The prompt
one did not, so its operands had no storage records at all and every prompt
occurrence refused with route_unavailable before it ever reached a key.  The two
admissions are ~1500 lines apart in one function, which is precisely how they
drifted apart in the first place, so the parity is asserted here rather than
left to whoever next edits one of them.

This gate is about PARITY and about the publication's own gate.  It deliberately
also pins allow_materialize=false at both admissions: publishing is only safe
because it hands the resolver something already owned, and an admission that
started materializing would be a different change wearing this one's clothes.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

PUBLISH = "ggml_sycl_publish_mmid_canonical_aos_experts("

PROMPT_START = "if (!ggml_sycl_copy_ids_to_host(ctx, ids, prompt_ids_snapshot)) {"
PROMPT_END = "retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch("
DECODE_START = "ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;"
DECODE_END = "retained_decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch("

# The publication's own refusal, in the helper.  A non-AoS request must not be
# able to publish, because the record it would install is a view of the tensor's
# own AoS bytes and would alias whatever layout was asked for.
AOS_GATE = ("if (route_layout != GGML_LAYOUT_AOS || !src0 || !src0->buffer || !src0->buffer->context ||\n"
            "        ggml_backend_buffer_is_host(src0->buffer)) {\n"
            "        return false;\n"
            "    }")


def admission(source: str, start: str, end: str) -> str | None:
    """One admission, from its entry anchor through its whole resolver call.

    The region must run to the call's closing paren, not to its opening one:
    allow_materialize is an ARGUMENT, so a region ending at the call name
    excludes it and the check that pins it would pass on any tree at all.
    """
    if start not in source:
        return None
    begin = source.index(start)
    if end not in source[begin:]:
        return None
    stop = source.index(end, begin)
    close = source.find(");", stop)
    if close < 0:
        return None
    return source[begin:close + 2]


def regions(source: str) -> tuple[str | None, str | None]:
    return (admission(source, PROMPT_START, PROMPT_END),
            admission(source, DECODE_START, DECODE_END))


def violations(source: str) -> list[str]:
    found: list[str] = []

    prompt, decode = regions(source)
    if prompt is None:
        return ["the MUL_MAT_ID prompt admission is missing"]
    if decode is None:
        return ["the MUL_MAT_ID decode admission is missing"]

    # The parity itself, named per side: "one of them lost it" is not actionable.
    if PUBLISH not in prompt:
        found.append("the prompt admission resolves without publishing allocation-owned experts")
    if PUBLISH not in decode:
        found.append("the decode admission resolves without publishing allocation-owned experts")

    # Publishing is only safe while it stays non-materializing at the seam.
    if "/*allow_materialize=*/false" not in prompt:
        found.append("the prompt admission no longer resolves with allow_materialize=false")
    if "/*allow_materialize=*/false" not in decode:
        found.append("the decode admission no longer resolves with allow_materialize=false")

    # The helper's own gate.  Checked against the whole file: it lives outside
    # both regions, and a weakened gate would let either admission publish an
    # AoS view under a layout that is not AoS.
    if AOS_GATE not in source:
        found.append("the publication no longer refuses a non-AoS or host-buffer request")

    return found


def test_both_admissions_exist() -> None:
    # Methodology control, ordered FIRST: with either region missing every check
    # below is vacuous, and the checker would pass on a restructured file.
    prompt, decode = regions(SOURCE.read_text())
    assert prompt is not None, "no prompt admission region"
    assert decode is not None, "no decode admission region"


def test_regions_are_disjoint_and_bounded() -> None:
    # Second control.  A too-greedy prompt region that swallowed the decode one
    # would make the parity check pass on a tree where only decode publishes --
    # the exact regression this gate exists to catch.
    source = SOURCE.read_text()
    prompt, decode = regions(source)
    assert prompt is not None and decode is not None
    assert DECODE_START not in prompt, "the prompt region swallowed the decode admission"
    assert PROMPT_END not in decode, "the decode region swallowed the prompt admission"


def test_both_admissions_publish_before_resolving() -> None:
    assert violations(SOURCE.read_text()) == []


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()
    prompt, decode = regions(source)
    assert prompt is not None and decode is not None

    # Mutations are applied INSIDE the region they target.  The two admissions
    # call the same helper, so a file-wide replace would hit the prompt one
    # first and the decode mutation would silently test the prompt region.
    prompt_unpublished = prompt.replace(PUBLISH, "false && ggml_sycl_never_published(", 1)
    assert prompt_unpublished != prompt, "the publish call was not found in the prompt region"
    decode_unpublished = decode.replace(PUBLISH, "false && ggml_sycl_never_published(", 1)
    assert decode_unpublished != decode, "the publish call was not found in the decode region"

    prompt_materializing = prompt.replace("/*allow_materialize=*/false", "/*allow_materialize=*/true", 1)
    assert prompt_materializing != prompt, "allow_materialize=false was not found in the prompt region"

    mutations = [
        # The regression this gate exists for: decode publishes, prompt does not.
        source.replace(prompt, prompt_unpublished, 1),
        # The mirror image, so the check cannot be satisfied by one side alone.
        source.replace(decode, decode_unpublished, 1),
        # Publishing while materializing at the seam.
        source.replace(prompt, prompt_materializing, 1),
        # Weakening the helper's gate so a non-AoS request can publish an alias.
        source.replace(AOS_GATE, "if (!src0 || !src0->buffer) {\n        return false;\n    }", 1),
        # Deleting that gate outright.
        re.sub(re.escape(AOS_GATE), "", source, count=1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"mutation {index} did not change the source"
        assert violations(mutated), f"mutation {index} was not witnessed"
