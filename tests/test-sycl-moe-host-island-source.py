#!/usr/bin/env python3
"""Source contract: the prompt's CPU island moves its rows across the device by its copy plan, not one copy per row.

Host-only: reads ggml-sycl.cpp, builds nothing, loads no model and touches no device.

A MUL_MAT_ID whose experts live on the host runs them on the CPU through the lambda dispatch_host_expert_groups in
ggml_sycl_mul_mat_id (the "PP CPU island"). Every (token, slot) row reads its token's activation row of src1 and
writes one result row of dst. It used to copy each row on its own in both directions: on a 2048-token prompt of a
host-expert MoE that was millions of copies each way, almost every copy of the run. The copy plan in
ggml-sycl/moe-host-island.hpp (tested by tests/test-sycl-moe-host-island.cpp) gathers each distinct activation row once,
in runs of adjacent rows, and lays the results out in destination order so adjacent rows return in one copy.

This gate holds the lambda to that plan. It is the only gate on the lambda: no other source pin reads it.

  P0  the lambda exists exactly once;
  P1  its body builds the plan with ggml_sycl::moe_island_plan_build( exactly once;
  P2  its body walks island.gather and island.scatter, one loop each;
  P3  its body has exactly two mem_copy_async( calls: the device-to-host copy of a gather run (src1 into the
      activation block, at the run's offsets) inside the gather loop, and the host-to-device copy of a scatter run
      (the result block into dst, at the run's offsets) inside the scatter loop. A per-row loop in either direction
      is a third call;
  P4  the per-row destination list scatter_dst_offsets is gone;
  P5  the activation block is sized from the plan's distinct rows (island.n_act_rows), not from the row count;
  P6  each CPU task reads the activation block row and writes the result block row the plan assigned to it, indexed
      by the task loop's one row counter flat_row: act_index for the activation, out_index for the result.

Checks run on comment-blanked text with whitespace collapsed, so a clang-format re-wrap moves no claim.

NOT VACUOUS. --self-test runs the gate on in-memory mutants of the real source. Each mutant names the pin that must
catch it, and is reported if it survives, if a different pin catches it, or if its anchor no longer matches exactly
once in the lambda (NOT APPLIED). The self-test also proves that last report fires, on an anchor that cannot match.
ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the plan maps rows correctly is the host test's job; that the island's
numbers are right is the lead's correctness gate on a device.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

LAMBDA_MARKER = "auto dispatch_host_expert_groups = [&]"
PLAN_CALL = "ggml_sycl::moe_island_plan_build("
GATHER_LOOP = "for (const ggml_sycl::moe_gather_run & run : island.gather)"
SCATTER_LOOP = "for (const ggml_sycl::moe_scatter_run & run : island.scatter)"
GATHER_COPY = ("d2h_events.push_back(ggml_sycl::mem_copy_async(act_handle, run.dst_offset, src1_storage.handle, "
               "run.src_offset, run.bytes, *stream));")
SCATTER_COPY = ("h2d_events.push_back(ggml_sycl::mem_copy_async(dst_storage.handle, run.dst_offset, out_handle, "
                "run.src_offset, run.bytes, *stream));")
COPY_CALL = "mem_copy_async("
ACT_SIZE = "act_req.size = island.n_act_rows * static_cast<size_t>(K) * sizeof(float);"
TASK_ACT = "task.act_host = act_pinned + static_cast<size_t>(island.act_index[flat_row]) * static_cast<size_t>(K);"
TASK_OUT = "task.output_host = out_pinned + static_cast<size_t>(island.out_index[flat_row]) * static_cast<size_t>(N);"


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines. String literals are kept."""
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
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`; string literals are skipped."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError("FAIL: expected a brace block and found none")
    depth = 0
    k = open_at
    n = len(src)
    while k < n:
        c = src[k]
        if c == '"':
            k += 1
            while k < n and src[k] != '"':
                k += 2 if src[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
        k += 1
    raise ContractError("FAIL: brace block is unbalanced")


def squash(text: str) -> str:
    """Collapse whitespace, and drop it inside parentheses' edges so a wrapped call reads like an unwrapped one."""
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def lambda_body(backend_src: str) -> str:
    code = blank_comments(backend_src)
    n_markers = code.count(LAMBDA_MARKER)
    if n_markers != 1:
        raise ContractError(f"FAIL [pin P0]: `{LAMBDA_MARKER}` appears {n_markers} times in ggml-sycl.cpp, not once")
    return squash(brace_block_from(code, code.find(LAMBDA_MARKER)))


def loop_block(body: str, header: str, pin: str) -> str:
    at = body.find(header)
    if at < 0 or body.count(header) != 1:
        raise ContractError(f"FAIL [pin {pin}]: the island does not walk its plan with exactly one `{header}`")
    return brace_block_from(body, at + len(header))


def check(backend_src: str) -> None:
    body = lambda_body(backend_src)

    n_plans = body.count(PLAN_CALL)
    if n_plans != 1:
        raise ContractError(f"FAIL [pin P1]: the island builds its copy plan {n_plans} times, not once ({PLAN_CALL})")

    gather = loop_block(body, GATHER_LOOP, "P2")
    scatter = loop_block(body, SCATTER_LOOP, "P2")

    n_copies = body.count(COPY_CALL)
    if n_copies != 2:
        raise ContractError(
            f"FAIL [pin P3]: the island issues {n_copies} mem_copy_async calls, not two (one per gather run, one per "
            "scatter run): a per-row copy loop is back"
        )
    if gather.count(COPY_CALL) != 1 or GATHER_COPY not in gather:
        raise ContractError(f"FAIL [pin P3]: the gather loop does not copy each run with `{GATHER_COPY}`")
    if scatter.count(COPY_CALL) != 1 or SCATTER_COPY not in scatter:
        raise ContractError(f"FAIL [pin P3]: the scatter loop does not copy each run with `{SCATTER_COPY}`")

    if re.search(r"\bscatter_dst_offsets\b", body):
        raise ContractError("FAIL [pin P4]: the per-row destination list scatter_dst_offsets is back in the island")

    if body.count("act_req.size =") != 1 or ACT_SIZE not in body:
        raise ContractError(f"FAIL [pin P5]: the activation block is not sized once as `{ACT_SIZE}`")

    if body.count(TASK_ACT) != 1 or body.count(TASK_OUT) != 1:
        raise ContractError(
            "FAIL [pin P6]: a CPU task does not read and write the block rows the plan assigned to it "
            f"(`{TASK_ACT}` and `{TASK_OUT}`, once each)"
        )
    if len(re.findall(r"\bsize_t flat_row = 0;", body)) != 1 or len(re.findall(r"\+\+flat_row;", body)) != 1:
        raise ContractError(
            "FAIL [pin P6]: flat_row is not the task loop's one row counter (declared once from 0, advanced once)"
        )


def self_test(backend_src: str) -> int:
    failures: list[str] = []
    n_mutants = 0
    # Mutants land in the lambda's raw text. Blanking comments preserves offsets, so the block found on the blanked
    # text spans the same characters of the raw text.
    code = blank_comments(backend_src)
    lam_at = code.find(LAMBDA_MARKER)
    if lam_at < 0:
        print("  self-test: lambda marker not found; cannot place mutants")
        return 1
    lam_end = code.find("{", lam_at) + len(brace_block_from(code, lam_at))

    def sub(old: str, new: str, after: str | None = None) -> str:
        """Replace `old` with `new` in the raw text of the lambda. `old` must occur exactly once there, or exactly once
        after the lambda's one `after` when that is given."""
        start = lam_at
        if after is not None:
            region = backend_src[lam_at:lam_end]
            if region.count(after) != 1:
                raise MutantNotApplied(f"`after` matches {region.count(after)} times in the lambda: {after[:70]!r}")
            start = lam_at + region.find(after)
        region = backend_src[start:lam_end]
        if region.count(old) != 1:
            raise MutantNotApplied(f"anchor matches {region.count(old)} times in the lambda: {old[:70]!r}")
        return backend_src[:start] + region.replace(old, new, 1) + backend_src[lam_end:]

    def expect_fail(name: str, pin: str, old: str, new: str, after: str | None = None) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = sub(old, new, after)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(mutated)
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught, but not by pin {pin}: {str(e)[:120]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:90]}...)")
                return
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: survived")
        print(f"  mutant {name}: SURVIVED")

    plan_call = "ggml_sycl::moe_island_plan_build(island_rows,"
    gather_copy = "d2h_events.push_back(ggml_sycl::mem_copy_async(act_handle, run.dst_offset, src1_storage.handle,"
    scatter_copy = "h2d_events.push_back(ggml_sycl::mem_copy_async(dst_storage.handle, run.dst_offset, out_handle,"
    gather_wait = "sycl::event::wait(d2h_events);"
    scatter_wait = "sycl::event::wait(h2d_events);"
    per_row_d2h = ("for (size_t ri = 0; ri < n_rows; ++ri) { d2h_events.push_back(ggml_sycl::mem_copy_async(act_handle, "
                   "ri * static_cast<size_t>(K) * sizeof(float), src1_storage.handle, ri * nb12, "
                   "static_cast<size_t>(K) * sizeof(float), *stream)); } ")
    per_row_h2d = ("for (size_t ri = 0; ri < n_rows; ++ri) { h2d_events.push_back(ggml_sycl::mem_copy_async("
                   "dst_storage.handle, ri * nb1, out_handle, ri * static_cast<size_t>(N) * sizeof(float), "
                   "static_cast<size_t>(N) * sizeof(float), *stream)); } ")

    expect_fail("lambda-renamed", "P0", "auto dispatch_host_expert_groups = [&]", "auto dispatch_host_rows = [&]")
    expect_fail("plan-not-built", "P1", plan_call, "build_island_plan_elsewhere(island_rows,")
    expect_fail("plan-call-commented-out", "P1", plan_call, "/* ggml_sycl::moe_island_plan_build( */ (island_rows,")
    expect_fail("plan-built-twice", "P1", "ggml_sycl::moe_island_plan island;",
                "ggml_sycl::moe_island_plan island; (void) ggml_sycl::moe_island_plan_build(island_rows, 1, 1, &island);")
    expect_fail("gather-loop-gone", "P2", "run : island.gather)", "run : island.scatter_unused)")
    expect_fail("scatter-loop-gone", "P2", "run : island.scatter)", "run : island.gather_unused)")
    expect_fail("per-row-d2h-loop-restored", "P3", gather_wait, per_row_d2h + gather_wait)
    expect_fail("per-row-h2d-loop-restored", "P3", scatter_wait, per_row_h2d + scatter_wait)
    expect_fail("gather-copy-offsets-swapped", "P3", gather_copy,
                "d2h_events.push_back(ggml_sycl::mem_copy_async(act_handle, run.src_offset, src1_storage.handle,")
    expect_fail("gather-copy-wrong-handle", "P3", gather_copy,
                "d2h_events.push_back(ggml_sycl::mem_copy_async(out_handle, run.dst_offset, src1_storage.handle,")
    expect_fail("scatter-copy-direction-reversed", "P3", scatter_copy,
                "h2d_events.push_back(ggml_sycl::mem_copy_async(out_handle, run.dst_offset, dst_storage.handle,")
    expect_fail("scatter-copy-one-row", "P3", "run.bytes, *stream)", "static_cast<size_t>(N) * sizeof(float), *stream)",
                after="run : island.scatter)")
    expect_fail("per-row-destinations-back", "P4", "std::vector<cpu_expert_task> tasks;",
                "std::vector<cpu_expert_task> tasks; std::vector<size_t> scatter_dst_offsets;")
    expect_fail("act-block-sized-by-rows", "P5", "island.n_act_rows * static_cast<size_t>(K) * sizeof(float);\n",
                "n_rows * static_cast<size_t>(K) * sizeof(float);\n")
    expect_fail("task-reads-its-flat-row", "P6", "static_cast<size_t>(island.act_index[flat_row])", "flat_row")
    expect_fail("task-writes-its-flat-row", "P6", "static_cast<size_t>(island.out_index[flat_row])", "flat_row")
    expect_fail("task-act-reads-out-index", "P6", "island.act_index[flat_row]", "island.out_index[flat_row]")
    expect_fail("task-second-row-counter", "P6", "++flat_row;", "++flat_row; ++flat_row;")

    # The NOT APPLIED report must fire on an anchor the lambda does not hold, or a stale mutant would pass unseen.
    try:
        sub("this anchor is in no lambda", "x")
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
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
    print("PASS: the prompt CPU island copies by its plan: each distinct activation row once in runs, results in "
          "destination order, no per-row copy loop")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
