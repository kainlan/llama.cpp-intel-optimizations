"""The batched MoE prompt executor takes every activation and output offset from one helper each.

The executor lambda try_pp_onednn_f16_batched (ggml/src/ggml-sycl/ggml-sycl.cpp) lays its activation and output
scratch out per GEMM group, from moe_pp_pack_gemm_groups (moe-pp-pack.hpp). Every reader of that scratch that addresses
one expert takes its offsets from act_base and out_base, so they cannot disagree about where an expert's rows are, and
none of the old rectangle's names remains. Those readers are counted exactly: the two GEMM arms (WOQ and f16) and the
debug compare, three calls of each helper. The batched gather and scatter (Inkling Task 25) address the scratch from row
0 through the packed-row table, whose rows come from the same layout (slot_base_row, moe_pp_pack_row_table): each takes
its pointer once, in one pinned spelling, inside the submit that launches its kernel.

Host-only text assertions. llama_test_pytest hands this file to pytest, so every check lives inside a test_*()
function: a file with none is "no tests collected" (exit 5), which ctest scores as a failure. Checks run on
comment-stripped text, so prose that quotes a name or a call cannot satisfy or trip them.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Plain file I/O: the codescout index is blind inside ggml-sycl.cpp.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _executor_body():
    """The executor lambda's text. A lambda that is not found fails the calling test rather than skipping it."""
    text = _strip_comments(GGML_SYCL_CPP)
    begin = text.find("auto try_pp_onednn_f16_batched = [&]() -> bool {")
    end = text.find("if (try_pp_onednn_f16_batched()) {")
    assert begin >= 0 and end >= begin, "the executor lambda try_pp_onednn_f16_batched was not found"
    return text[begin:end]


def test_the_rectangle_names_are_gone():
    body = _executor_body()
    # Whole words only: the header's own type, moe_pp_gemm_group, contains the retired pp_gemm_group, so a substring
    # test would fire on the correct executor.
    failures = [
        f"{name}: the rectangle's name is still in the executor"
        for name in ("max_rows_padded", "act_slot_elems", "out_slot_elems", "pp_gemm_group")
        if re.search(rf"\b{re.escape(name)}\b", body)
    ]
    assert not failures, "\n".join(failures)


def test_the_groups_come_from_the_header():
    body = _executor_body()
    assert "moe_pp_pack_gemm_groups(" in body, "the executor does not take its groups from moe_pp_pack_gemm_groups"


@pytest.mark.parametrize("pointer,helper,uses", [("batched_acts", "act_base", 3), ("batched_out", "out_base", 3)])
def test_every_offset_comes_from_one_helper(pointer, helper, uses):
    body = _executor_body()
    failures = []
    offsets = len(re.findall(rf"\b{pointer}\s*\+", body))
    if offsets != 1:
        failures.append(f"{pointer}: {offsets} pointer offsets, want exactly one (inside {helper})")
    calls = len(re.findall(rf"\b{helper}\(", body))
    if calls != uses:
        failures.append(f"{helper}: {calls} calls, want exactly {uses} (the WOQ GEMM arm, the f16 GEMM arm and the debug "
                        f"compare); a new reader of one expert's rows is a deliberate edit here")
    assert not failures, "\n".join(failures)


# The batched gather's and scatter's one read of each scratch pointer (Inkling Task 25): the packed-row table's rows
# are absolute scratch rows, so these two kernels take the scratch from row 0. Each read must sit inside the submit that
# launches its kernel: after it, the kernel's name comes before any other submit.
POINTER_CASES = [
    ("batched_acts", "sycl::half", "activation", "act_base",
     r"\bsycl::half\s*\*\s*__restrict\s+act_get\s*=\s*batched_acts\s*;", "k_copy_src1_to_packed_f16"),
    ("batched_out", "float", "output", "out_base",
     r"\bconst\s+float\s*\*\s*__restrict\s+out_get\s*=\s*batched_out\s*;", "k_copy_dst_from_packed"),
]
# A place in the executor outside every submit, where the positive controls put their mutations.
MUTATION_ANCHOR = "batched_scratch_claim.mark_used();"


@pytest.mark.parametrize("case", POINTER_CASES, ids=[c[0] for c in POINTER_CASES])
def test_the_scratch_pointers_are_read_only_through_their_helpers(case):
    failures = _pointer_failures(_executor_body(), *case)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("case", POINTER_CASES, ids=[c[0] for c in POINTER_CASES])
def test_a_third_bare_reader_of_the_scratch_fails(case):
    # Positive control: a reader that takes the pointer outside the pinned places, beside the kernel's own read.
    pointer, pointee = case[0], case[1]
    body = _executor_body()
    assert body.count(MUTATION_ANCHOR) == 1, "the mutation anchor is missing"
    mutated = body.replace(MUTATION_ANCHOR, f"{pointee} * stray_reader = {pointer}; " + MUTATION_ANCHOR, 1)
    assert not _pointer_failures(body, *case)
    assert _pointer_failures(mutated, *case)


@pytest.mark.parametrize("case", POINTER_CASES, ids=[c[0] for c in POINTER_CASES])
def test_the_row0_read_moved_out_of_its_submit_fails(case):
    # Positive control: the pinned row-0 read itself, moved out of the submit that launches its kernel to a place
    # outside every submit. Its spelling and the use count are unchanged, so only the anchoring can catch it.
    row0 = case[4]
    body = _executor_body()
    assert body.count(MUTATION_ANCHOR) == 1, "the mutation anchor is missing"
    m = re.search(row0, body)
    assert m, "the row-0 read is missing"
    moved = body[: m.start()] + body[m.end() :]
    moved = moved.replace(MUTATION_ANCHOR, m.group(0) + " " + MUTATION_ANCHOR, 1)
    assert not _pointer_failures(body, *case)
    assert _pointer_failures(moved, *case)


def _row0_inside_its_submit(body, row0, kernel):
    # The read is inside the submit that launches `kernel` when the kernel's name follows it with no other submit in
    # between, and the kernel is launched once in the executor.
    m = re.search(row0, body)
    if not m or len(re.findall(rf"\b{kernel}\(", body)) != 1:
        return False
    launch = body.find(kernel + "(", m.end())
    return launch >= 0 and "submit(" not in body[m.end() : launch]


def _pointer_failures(body, pointer, pointee, field, helper, row0, kernel):
    # A reader that takes the pointer itself assumes where an expert's rows are instead of asking, and the offset count
    # above cannot see it when the read is at offset 0 (no `+`). So the pointer may appear only where it is declared,
    # where the reservation sets it, inside its helper and in the batched gather's or scatter's one pinned read, and the
    # reservation's field only where it sets the pointer.
    t = re.escape(pointee)
    allowed = {
        "its declaration": rf"\b{t}\s*\*\s*{pointer}\s*=\s*nullptr\s*;",
        "its assignment from the reservation": (
            rf"\b{pointer}\s*=\s*static_cast<\s*{t}\s*\*\s*>\(\s*reserved\.{field}\s*\)\s*;"
        ),
        helper: (
            rf"\bconst\s+auto\s+{helper}\s*=\s*\[&\]\s*\(\s*size_t\s+slot\s*\)\s*->\s*{t}\s*\*\s*\{{\s*"
            rf"return\s+{pointer}\s*\+"
        ),
        "the batched gather's or scatter's read from row 0": row0,
    }
    failures = []
    if not _row0_inside_its_submit(body, row0, kernel):
        failures.append(f"{pointer}: the row-0 read is not inside the one submit that launches {kernel}")
    for where, pattern in allowed.items():
        found = len(re.findall(pattern, body))
        if found != 1:
            failures.append(f"{pointer}: {found} matches for {where}, want exactly one")
    uses = len(re.findall(rf"\b{pointer}\b", body))
    if uses != len(allowed):
        failures.append(
            f"{pointer}: {uses} uses in the executor, want {len(allowed)} (its declaration, its assignment from the "
            f"reservation, {helper} and the batched gather's or scatter's read); every other reader must ask {helper}"
        )
    # The reservation's own field is the same address under another name, so a reader that takes it bypasses the
    # helper just as the bare pointer would. Whole words, so reserved.<field>_size does not count.
    field_uses = len(re.findall(rf"\breserved\.{field}\b", body))
    if field_uses != 1:
        failures.append(
            f"reserved.{field}: {field_uses} uses in the executor, want exactly one (the assignment to {pointer}); "
            f"every other reader must ask {helper}"
        )
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
