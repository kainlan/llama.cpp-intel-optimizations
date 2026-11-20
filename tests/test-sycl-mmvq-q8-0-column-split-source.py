#!/usr/bin/env python3
"""Source contract: the column-split form of the Q8_0 matvec, and the rule that picks it.

Host-only: reads ggml/src/ggml-sycl/mmvq.cpp and mmvq-launch-geometry.hpp, builds nothing, loads no model and touches
no device.

A Q8_0 matvec with few rows and long rows used to run one sub-group per row (4 x 10240: four sub-groups while the card
idles). The SOA and coalesced Q8_0 launchers now pick a column-split form from the shape. The host test
(test-sycl-mmvq-column-split-rule) proves the rule's arithmetic and the device test (test-sycl-mmvq-q8-0-column-split)
the numbers. Neither can see whether production consults the rule with the right inputs, whether the row form is still
what the other shapes launch, or whether the split kernels keep the properties a single device run cannot show. This
gate holds those by text:

  R1  the rule (mmvq_column_split_chunks) keeps the row form for rows >= compute units and for an unknown device, and
      caps the chunks at the work-group's 16 sub-groups and at the row's steps; the header includes nothing;
  D1  the SOA launcher computes the chunks from (nrows, ncols / QK8_0, its row form's sub-group step, compute_units),
      the step being mul_mat_vec_q_reorder's blocks_per_subgroup, and launches the split kernel and returns before the
      unchanged row-form launch;
  D2  the same for the coalesced launcher, whose step is one MMVQ_COALESCED_TILE_BLOCKS tile;
  D3  the dispatcher passes the card's compute-unit count (ggml_sycl_info().devices[id].nsm, 0 for an unknown index)
      to both launchers, and nothing in either launcher's rule call names a tensor;
  W1  both split kernels walk a row through the one shared step walk the host test sweeps
      (mmvq_column_split_first_step / mmvq_column_split_step_stride): the coalesced kernel calls it directly over tile
      indices, the SOA kernel through mmvq_column_split_first_block / block_stride, which the header defines through
      it;
  K1  each split kernel reduces its sub-group, publishes one partial per sub-group to local memory, passes a
      work-group barrier with no early return before it, and only then lets the row's chunk-0 sub-group add the
      partials in chunk order and the fused addend once;
  K2  the row forms still walk what they walked: the SOA row kernel's block loop, and the coalesced row kernel's
      every-tile loop through the shared per-tile body;
  E1  no host wait and no new dependency in either launcher: both forms go through the one ggml_sycl_profile_submit
      on the caller's stream, with no wait and no depends_on;
  A1  the split path allocates nothing: its only memory is a 16-float local_accessor; no malloc and no unified-cache
      call in either launcher.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources;
each names the pin that must catch it and is reported if it survives, if another pin catches it, or if its anchor no
longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text: that the kernels compute the right numbers is the device test's job, that the
rule's arithmetic is right on every shape is the host test's. It cannot see a decision made from a tensor name through
a variable the rule call does not spell, only one spelled in the call itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml/src/ggml-sycl/mmvq.cpp"
HEADER = ROOT / "ggml/src/ggml-sycl/mmvq-launch-geometry.hpp"

_COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def blank(text: str) -> str:
    """Comments blanked (line structure kept), string and character literals kept."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def squeeze(text: str) -> str:
    """Runs of blanks to one space, newlines to spaces: clang-format's alignment and wrapping cannot move a needle."""
    return re.sub(r"\s+", " ", text)


def body(code: str, signature: str, pin: str) -> str:
    """The text of the function whose definition starts with `signature`, through its matching closing brace."""
    n = code.count(signature)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{signature}` appears {n} times, not once")
    start = code.index(signature)
    open_at = code.index("{", code.index(")", start))
    depth = 0
    for k in range(open_at, len(code)):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                return code[start : k + 1]
    raise ContractError(f"FAIL [pin {pin}]: `{signature}` has no closing brace")


def ordered(text: str, *needles: str) -> bool:
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def need(ok: bool, pin: str, what: str) -> None:
    if not ok:
        raise ContractError(f"FAIL [pin {pin}]: {what}")


SOA_LAUNCHER = "static void reorder_mul_mat_vec_q8_0_q8_1_sycl("
COALESCED_LAUNCHER = "static void coalesced_mul_mat_vec_q8_0_q8_1_sycl("
SOA_SPLIT = "static void mul_mat_vec_q_reorder_column_split("
COALESCED_SPLIT = "static void mul_mat_vec_q8_0_coalesced_column_split("
SOA_ROW = "static void mul_mat_vec_q_reorder("
COALESCED_ROW = "static void mul_mat_vec_q8_0_coalesced("
DISPATCH = "static void ggml_sycl_mmvq_dispatch("
UNITS = "static int mmvq_device_compute_units("
ROW_ANCHOR = "constexpr size_t num_subgroups = 16;"


def check_launcher(launcher: str, pin: str, step_decl: str, split_kernel_call: str, row_kernel_call: str) -> None:
    need(step_decl in launcher, pin, f"the launcher's sub-group step is not `{step_decl}`")
    # D3 before the exact call: a rule call that reads the tensor must be named as that, not as a changed call.
    rule_call = re.search(r"mmvq_column_split_chunks\(([^;]*)\);", launcher)
    need(rule_call is not None and not re.search(r"name|src0|tensor", rule_call.group(1)), "D3",
         "the rule call reads something other than the shape and the compute units")
    need(
        "ggml_sycl::mmvq_column_split_chunks(nrows, ncols / QK8_0, blocks_per_step, compute_units)" in launcher,
        pin,
        "the launcher does not ask the rule with (nrows, ncols / QK8_0, blocks_per_step, compute_units)",
    )
    need(ROW_ANCHOR in launcher, pin, "the row-form launch geometry is gone")
    split_at = launcher.find("if (chunks > 1) { constexpr int subgroups")
    need(split_at >= 0, pin, "no `if (chunks > 1)` split branch opening with the work-group's sub-groups")
    row_at = launcher.index(ROW_ANCHOR)
    need(split_at < row_at, pin, "the split branch does not come before the row-form launch")
    branch = launcher[split_at:row_at]
    need(
        ordered(branch, "sycl::local_accessor<float, 1> row_partials(", split_kernel_call, "return; }"),
        pin,
        "the split branch does not launch the split kernel and return before the row form",
    )
    need(row_kernel_call in launcher[row_at:], pin, "the row-form launch no longer calls the row kernel")
    need("chunks_out" in launcher and "*chunks_out = chunks;" in launcher, pin, "the launcher does not report chunks")
    # E1: one submit wrapper per form, on the caller's stream, no wait, no dependency.
    need(launcher.count("ggml_sycl_profile_submit(*stream, profile_label,") == 2, "E1",
         "the launcher does not submit both forms through ggml_sycl_profile_submit on its stream")
    need(not re.search(r"(?:\.|->)\s*wait\w*\s*\(|\bdepends_on\b", launcher), "E1",
         "a wait or a dependency in the launcher")
    # A1: no allocation on the split path; the local accessor is its only memory.
    need(not re.search(r"\b(?:sycl::)?malloc\w*\s*[<(]|\bunified_\w+\s*\(|\bsycl::free\s*\(", launcher), "A1",
         "an allocation in the launcher")
    need(launcher.count("sycl::local_accessor<") == 1, "A1", "the split path holds memory other than one accessor")


def check_split_kernel(kernel: str, pin: str) -> None:
    barrier = "nd_item.barrier(sycl::access::fence_space::local_space);"
    need(
        ordered(
            kernel,
            "sycl::reduce_over_group(sg, partial_sum, std::plus<>());",
            "if (sg.leader()) { row_partials[sg_id] = sum; }",
            barrier,
            "if (chunk == 0 && row < nrows && sg.leader()) {",
            "for (int c = 0; c < chunks; ++c) { row_sum += row_partials[sg_id + c]; }",
            "dst[row] = row_sum + mmvq_fused_add_value(fused_add, fused_add_ne0, fused_add_nb0, "
            "fused_add_row_base + row);",
        ),
        pin,
        "reduce, publish, barrier, then the chunk-0 sum and the fused addend, in that order",
    )
    need(kernel.count(barrier) == 1, pin, "the split kernel does not have exactly one barrier")
    need("return" not in kernel[: kernel.index(barrier)], pin, "a return before the barrier: a work-group would hang")
    need(kernel.count("dst[") == 1, pin, "the split kernel writes dst more than once")
    need("if (row < nrows) {" in kernel, pin, "padded rows are not kept from computing")


def check(mmvq_raw: str, header_raw: str) -> None:
    code = squeeze(blank(mmvq_raw))
    header = squeeze(blank(header_raw))

    # R1 ---------------------------------------------------------------------------------------------------------------
    need("#include" not in header, "R1", "the geometry header includes something")
    need("constexpr int MMVQ_COLUMN_SPLIT_SUBGROUPS = 16;" in header, "R1", "the split work-group is not 16 sub-groups")
    rule = body(header, "inline int mmvq_column_split_chunks(", "R1")
    need("if (rows <= 0 || compute_units <= 0 || rows >= compute_units) { return 1; }" in rule, "R1",
         "the rule does not keep the row form for rows >= compute units or an unknown device")
    need("while (chunks * 2 <= MMVQ_COLUMN_SPLIT_SUBGROUPS && chunks * 2 <= steps) { chunks *= 2; }" in rule, "R1",
         "the chunks are not the largest power of two within the work-group and the row's steps")
    steps = body(header, "inline int mmvq_column_split_steps(", "R1")
    need("(blocks_per_row + blocks_per_step - 1) / blocks_per_step" in steps, "R1", "the steps do not round up")

    # D1 / D2 ----------------------------------------------------------------------------------------------------------
    check_launcher(
        body(code, SOA_LAUNCHER, "D1"),
        "D1",
        "constexpr int blocks_per_step = ceil_div(q8_0_traits::vdr_mmvq * WARP_SIZE, q8_0_traits::qi);",
        "mul_mat_vec_q_reorder_column_split<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(",
        "mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(",
    )
    check_launcher(
        body(code, COALESCED_LAUNCHER, "D2"),
        "D2",
        "constexpr int blocks_per_step = MMVQ_COALESCED_TILE_BLOCKS;",
        "mul_mat_vec_q8_0_coalesced_column_split(",
        "mul_mat_vec_q8_0_coalesced(",
    )

    # D3 ---------------------------------------------------------------------------------------------------------------
    units = body(code, UNITS, "D3")
    need(
        ordered(units, "if (device_id < 0 || device_id >= ggml_sycl_info().device_count) { return 0; }",
                "return ggml_sycl_info().devices[device_id].nsm;"),
        "D3",
        "the compute units are not the device's nsm with 0 for an unknown index",
    )
    dispatch = body(code, DISPATCH, "D3")
    need(
        re.search(r"coalesced_mul_mat_vec_q8_0_q8_1_sycl\([^;]*mmvq_device_compute_units\(device_id\)[^;]*\);",
                  dispatch) is not None,
        "D3",
        "the dispatcher does not pass the card's compute units to the coalesced launcher",
    )
    need(
        re.search(r"reorder_mul_mat_vec_q8_0_q8_1_sycl\([^;]*mmvq_device_compute_units\(device_id\)[^;]*\);",
                  dispatch) is not None,
        "D3",
        "the dispatcher does not pass the card's compute units to the SOA launcher",
    )

    # W1 ---------------------------------------------------------------------------------------------------------------
    first_block = body(header, "inline int mmvq_column_split_first_block(", "W1")
    need("return mmvq_column_split_first_step(chunk) * blocks_per_step + lane_block;" in first_block, "W1",
         "the SOA block walk does not start from the shared step walk")
    block_stride = body(header, "inline int mmvq_column_split_block_stride(", "W1")
    need("return mmvq_column_split_step_stride(chunks) * blocks_per_step;" in block_stride, "W1",
         "the SOA block stride is not the shared step stride")
    soa_split = body(code, SOA_SPLIT, "W1")
    need(
        ordered(
            soa_split,
            "ggml_sycl::mmvq_column_split_block_stride(chunks, blocks_per_subgroup);",
            "for (int i = ggml_sycl::mmvq_column_split_first_block(chunk, lane / block_elements_per_subgroup, "
            "blocks_per_subgroup); i < blocks_per_row; i += block_stride) {",
        ),
        "W1",
        "the SOA split kernel does not walk its chunk's blocks through the shared walk",
    )
    coalesced_split = body(code, COALESCED_SPLIT, "W1")
    need(
        ordered(
            coalesced_split,
            "const int tile_stride = ggml_sycl::mmvq_column_split_step_stride(chunks);",
            "for (int tile = ggml_sycl::mmvq_column_split_first_step(chunk); tile < tiles_per_row; tile += tile_stride) {",
            "mmvq_q8_0_coalesced_accumulate_tile(x_qs, x_d, y_qs, y_ds, row, tile, x_row_stride, blocks_per_row, lane,"
            " partial_sum);",
        ),
        "W1",
        "the coalesced split kernel does not walk its chunk's tiles through the shared walk",
    )

    # K1 ---------------------------------------------------------------------------------------------------------------
    check_split_kernel(soa_split, "K1")
    check_split_kernel(coalesced_split, "K1")

    # K2 ---------------------------------------------------------------------------------------------------------------
    need(
        "for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; "
        "i += blocks_per_subgroup) {" in body(code, SOA_ROW, "K2"),
        "K2",
        "the SOA row kernel no longer walks every block of its row",
    )
    need(
        "for (int tile = 0; tile < tiles_per_row; tile++) { mmvq_q8_0_coalesced_accumulate_tile(x_qs, x_d, y_qs, y_ds, "
        "row, tile, x_row_stride, blocks_per_row, lane_id, partial_sum); }" in body(code, COALESCED_ROW, "K2"),
        "K2",
        "the coalesced row kernel no longer walks every tile through the shared body",
    )


def mutate(raw: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        n = raw.count(old)
        if n != 1:
            raise MutantNotApplied(f"NOT APPLIED: anchor appears {n} times, not once: {old[:70]!r}")
        raw = raw.replace(old, new)
    return raw


# The split kernels' one write to dst, as the source spells it (clang-format keeps it on one line).
SPLIT_DST = ("        dst[row] = row_sum + mmvq_fused_add_value(fused_add, fused_add_ne0, fused_add_nb0, "
             "fused_add_row_base + row);")

# Each mutant: (name, pin that must catch it, file, edits). Anchors are exact source text, so a mutant whose anchor
# drifts reports NOT APPLIED instead of passing.
MUTANTS = [
    ("the rule splits a matrix the row form already spreads", "R1", "header",
     [("rows >= compute_units) {", "rows >= 4 * compute_units) {")]),
    ("the rule ignores the row's step count", "R1", "header",
     [("chunks * 2 <= MMVQ_COLUMN_SPLIT_SUBGROUPS && chunks * 2 <= steps", "chunks * 2 <= MMVQ_COLUMN_SPLIT_SUBGROUPS")]),
    ("the split work-group grows past the row form's", "R1", "header",
     [("constexpr int MMVQ_COLUMN_SPLIT_SUBGROUPS = 16;", "constexpr int MMVQ_COLUMN_SPLIT_SUBGROUPS = 32;")]),
    ("the SOA launcher asks the rule with the coalesced step", "D1", "mmvq",
     [("    constexpr int blocks_per_step = ceil_div(q8_0_traits::vdr_mmvq * WARP_SIZE, q8_0_traits::qi);",
       "    constexpr int blocks_per_step = MMVQ_COALESCED_TILE_BLOCKS + 0 * q8_0_traits::qi;")]),
    ("the SOA split branch falls through into the row form", "D1", "mmvq",
     [("                            row_partials.get_pointer(), fused_add, fused_add_ne0, fused_add_nb0, "
       "fused_add_row_base);\n                    });\n            });\n        });\n        return;\n",
       "                            row_partials.get_pointer(), fused_add, fused_add_ne0, fused_add_nb0, "
       "fused_add_row_base);\n                    });\n            });\n        });\n")]),
    ("the coalesced launcher asks the rule with the SOA step", "D2", "mmvq",
     [("    constexpr int blocks_per_step = MMVQ_COALESCED_TILE_BLOCKS;",
       "    constexpr int blocks_per_step = 8;")]),
    ("the dispatcher keeps the SOA launcher on the row form", "D3", "mmvq",
     [("row_low, mmvq_device_compute_units(device_id), stream,", "row_low, 0, stream,")]),
    ("the compute units come from somewhere other than the device", "D3", "mmvq",
     [("    return ggml_sycl_info().devices[device_id].nsm;", "    return 256;")]),
    ("the rule call reads the tensor", "D3", "mmvq",
     [("const int     chunks = ggml_sycl::mmvq_column_split_chunks(nrows, ncols / QK8_0, blocks_per_step, compute_units);\n"
       "    if (chunks_out) {\n        *chunks_out = chunks;\n    }\n\n    // P4 TG-cost-visibility (llama.cpp-os8k): SOA",
       "const int     chunks = ggml_sycl::mmvq_column_split_chunks(nrows, ncols / QK8_0, blocks_per_step, compute_units"
       " + (int) sizeof(src0_name));\n"
       "    if (chunks_out) {\n        *chunks_out = chunks;\n    }\n\n"
       "    // P4 TG-cost-visibility (llama.cpp-os8k): SOA")]),
    ("the SOA split kernel loses its barrier", "K1", "mmvq",
     [("        row_partials[sg_id] = sum;\n    }\n    nd_item.barrier(sycl::access::fence_space::local_space);\n\n"
       "    if (chunk == 0 && row < nrows && sg.leader()) {\n        float row_sum = 0.0f;\n"
       "        for (int c = 0; c < chunks; ++c) {\n            row_sum += row_partials[sg_id + c];\n        }\n"
       + SPLIT_DST
       + "\n    }\n}\n\n// Multi-row",
       "        row_partials[sg_id] = sum;\n    }\n\n"
       "    if (chunk == 0 && row < nrows && sg.leader()) {\n        float row_sum = 0.0f;\n"
       "        for (int c = 0; c < chunks; ++c) {\n            row_sum += row_partials[sg_id + c];\n        }\n"
       + SPLIT_DST
       + "\n    }\n}\n\n// Multi-row")]),
    ("the coalesced split kernel returns early for padded rows", "K1", "mmvq",
     [("    const int blocks_per_row = ncols / QK8_0;\n    const int tiles_per_row  = ggml_sycl_coalesced_fixed_tile_count"
       "(blocks_per_row);\n\n    const uint8_t *     x_qs",
       "    if (row >= nrows) {\n        return;\n    }\n    const int blocks_per_row = ncols / QK8_0;\n"
       "    const int tiles_per_row  = ggml_sycl_coalesced_fixed_tile_count(blocks_per_row);\n\n"
       "    const uint8_t *     x_qs")]),
    ("the coalesced split kernel drops the fused addend", "K1", "mmvq",
     [("            row_sum += row_partials[sg_id + c];\n        }\n"
       + SPLIT_DST
       + "\n    }\n}\n\n// GPU kernel to convert Q8_0",
       "            row_sum += row_partials[sg_id + c];\n        }\n        dst[row] = row_sum;\n    }\n}\n\n"
       "// GPU kernel to convert Q8_0")]),
    ("the SOA split kernel sums the first row's partials for every row", "K1", "mmvq",
     [("            row_sum += row_partials[sg_id + c];\n        }\n"
       + SPLIT_DST
       + "\n    }\n}\n\n// Multi-row",
       "            row_sum += row_partials[c];\n        }\n"
       + SPLIT_DST
       + "\n    }\n}\n\n// Multi-row")]),
    ("the SOA split chunks all start at the row's first step", "W1", "mmvq",
     [("ggml_sycl::mmvq_column_split_first_block(chunk, lane / block_elements_per_subgroup,",
       "ggml_sycl::mmvq_column_split_first_block(0, lane / block_elements_per_subgroup,")]),
    ("the coalesced split kernel walks its tiles with its own arithmetic", "W1", "mmvq",
     [("        for (int tile = ggml_sycl::mmvq_column_split_first_step(chunk); tile < tiles_per_row; "
       "tile += tile_stride) {",
       "        for (int tile = chunk; tile < tiles_per_row; tile += chunks) {")]),
    ("the SOA block walk stops going through the shared step walk", "W1", "header",
     [("    return mmvq_column_split_first_step(chunk) * blocks_per_step + lane_block;",
       "    return chunk * blocks_per_step + lane_block;")]),
    ("the coalesced row form skips its first tile", "K2", "mmvq",
     [("    for (int tile = 0; tile < tiles_per_row; tile++) {\n        mmvq_q8_0_coalesced_accumulate_tile(",
       "    for (int tile = 1; tile < tiles_per_row; tile++) {\n        mmvq_q8_0_coalesced_accumulate_tile(")]),
    ("the coalesced split branch waits on the host", "E1", "mmvq",
     [("                                                                fused_add_nb0, fused_add_row_base);\n"
       "                    });\n            });\n        });\n        return;\n",
       "                                                                fused_add_nb0, fused_add_row_base);\n"
       "                    });\n            });\n        });\n        stream->wait();\n        return;\n")]),
    ("the SOA split branch allocates a device scratch", "A1", "mmvq",
     [("        const sycl::range<3> split_workgroup(1, 1, (size_t) subgroups * WARP_SIZE);\n\n"
       "        (void) ggml_sycl_profile_submit(*stream, profile_label, [&](sycl::queue & profiled_queue) {\n"
       "            return profiled_queue.submit([&](sycl::handler & cgh) {\n"
       "                sycl::local_accessor<float, 1> row_partials(sycl::range<1>(subgroups), cgh);\n"
       "                cgh.parallel_for<mmvq_reorder_column_split_kernel_name",
       "        const sycl::range<3> split_workgroup(1, 1, (size_t) subgroups * WARP_SIZE);\n"
       "        float * scratch = sycl::malloc_device<float>(16, *stream);\n\n"
       "        (void) ggml_sycl_profile_submit(*stream, profile_label, [&](sycl::queue & profiled_queue) {\n"
       "            return profiled_queue.submit([&](sycl::handler & cgh) {\n"
       "                sycl::local_accessor<float, 1> row_partials(sycl::range<1>(subgroups), cgh);\n"
       "                cgh.parallel_for<mmvq_reorder_column_split_kernel_name")]),
]


def self_test(mmvq_raw: str, header_raw: str) -> int:
    failures: list[str] = []
    for name, pin, target, edits in MUTANTS:
        try:
            m = mutate(mmvq_raw, edits) if target == "mmvq" else mmvq_raw
            h = mutate(header_raw, edits) if target == "header" else header_raw
        except MutantNotApplied as e:
            failures.append(f"{name}: {e}")
            continue
        try:
            check(m, h)
        except ContractError as e:
            caught = re.search(r"\[pin (\w+)\]", str(e))
            if not caught or caught.group(1) != pin:
                failures.append(f"{name}: caught by {caught.group(1) if caught else '?'}, not {pin}: {e}")
            continue
        failures.append(f"{name}: SURVIVED (pin {pin} did not fire)")

    # The NOT APPLIED report must itself fire, or a drifted anchor would read as a kill.
    try:
        mutate(mmvq_raw, [("this text is not in mmvq.cpp", "")])
        failures.append("NOT APPLIED does not fire on a missing anchor")
    except MutantNotApplied:
        pass

    for f in failures:
        print("SELF-TEST FAIL: " + f)
    if failures:
        return 1
    print(f"mmvq q8_0 column-split source contract: self-test PASS ({len(MUTANTS)} mutants, each caught by its pin)")
    return 0


def main() -> int:
    mmvq_raw = MMVQ.read_text(encoding="utf-8")
    header_raw = HEADER.read_text(encoding="utf-8")
    try:
        check(mmvq_raw, header_raw)
    except ContractError as e:
        print(e)
        return 1
    print("mmvq q8_0 column-split source contract: PASS")
    if "--self-test" in sys.argv[1:]:
        return self_test(mmvq_raw, header_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
