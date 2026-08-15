#!/usr/bin/env python3
"""The MoE prestaging planner only plans for tensors that carry a layer number.

moe_extract_block_number() returns -1 for any name without "blk.N.".  The group
registry has always skipped those -- a negative block number is not a layer, and
every such meta would share one layer -1 bucket -- but the metadata list it is
derived from had no such guard.  While the canonical provenance gate happened to
drop all non-model tensors upstream that was latent; universal provenance opens
that gate on purpose, so the planner precondition has to be stated where the
metadata is built.

This is a PRECONDITION OF THE PRESTAGING PLANNER, not a provenance exemption.
Admission to compute is decided by meta.canonical_key above it and keeps its
full strength; this decides only what the planner plans for.  A checker that let
the canonical-key gate be weakened would defeat the point, so the key check is
asserted here too.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

KEY_GATE = "meta.canonical_key = ggml_sycl_get_moe_expert_cache_key"
BLOCK_ASSIGN = "meta.block_num   = moe_extract_block_number"
PUSH = "new_expert_meta.push_back"
GROUP_GATE = "if (meta.block_num < 0 || !meta.canonical_key.valid) { continue; }"


def region(source: str) -> str | None:
    """The metadata-building loop, from the key gate to the push_back."""
    if KEY_GATE not in source or PUSH not in source:
        return None
    start = source.index(KEY_GATE)
    end = source.index(PUSH, start)
    return source[start:end + len(PUSH)]


def violations(source: str) -> list[str]:
    found: list[str] = []

    body = region(source)
    if body is None:
        return ["the MoE expert metadata loop is missing"]

    # Ordered: the canonical key is still the admission authority, and the
    # planner guard sits after it. Checking the guard alone would pass on a
    # tree where the key gate had been removed.
    if not re.search(r"if \(!meta\.canonical_key\.valid\) \{\s*\n\s*continue;", body):
        found.append("the metadata loop no longer refuses an invalid canonical key")
    if BLOCK_ASSIGN not in body:
        found.append("the metadata loop no longer computes a block number")
    if not re.search(r"if \(meta\.block_num < 0\) \{\s*\n\s*continue;\s*\n\s*\}", body):
        found.append("the metadata loop admits tensors with no layer number to the prestaging planner")

    # The group registry's own guard is what this mirrors; losing it would put
    # the collision back one level down.
    if not re.search(r"if \(meta\.block_num < 0 \|\| !meta\.canonical_key\.valid\) \{\s*\n?\s*continue;", source):
        found.append("the group registry no longer guards its block number")

    return found


def test_region_anchor_exists() -> None:
    # Methodology control, ordered first: with no region every check below is
    # vacuous, and the checker would pass on a file that had been restructured.
    assert region(SOURCE.read_text()) is not None


def test_prestage_metadata_requires_a_layer_number() -> None:
    assert violations(SOURCE.read_text()) == []


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()
    # Mutations are applied INSIDE the region. A file-wide re.sub for the same
    # text hits the prestaging loop's own block_num guard first (ggml-sycl.cpp
    # has three), so the mutation would change a line nothing here checks and
    # then read as "not witnessed" against an unmutated region.
    body = region(source)
    assert body is not None, "no region to mutate"
    without_guard = body.replace("            if (meta.block_num < 0) {\n                continue;\n            }\n", "", 1)
    assert without_guard != body, "the planner guard was not found inside the region"
    mutations = [
        # Dropping the planner guard: every layer-less expert lands in one bucket.
        source.replace(body, without_guard, 1),
        # Weakening the admission gate it must NOT be confused with.
        source.replace("        if (!meta.canonical_key.valid) {\n                continue;\n            }",
                       "        if (false) {\n                continue;\n            }", 1),
        re.sub(r"            if \(!meta\.canonical_key\.valid\) \{\n                continue;\n            \}\n",
               "", source, count=1),
        source.replace(GROUP_GATE, "", 1) if GROUP_GATE in source else source.replace(
            "            if (meta.block_num < 0 || !meta.canonical_key.valid) {\n                continue;\n            }", "", 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"mutation {index} did not change the source"
        assert violations(mutated), f"mutation {index} was not witnessed"
