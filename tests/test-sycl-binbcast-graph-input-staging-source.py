#!/usr/bin/env python3
"""Source gate: a recorded bin-broadcast reads a staged graph input, not host memory.

Under spread placement a graph input (an attention bias leaf) stays in host
memory, and the graph pre-stage copies it to a stable device buffer keyed by
the tensor. Bin-broadcast's raw host staging cannot be recorded, so a segment
whose ADD read the host bytes ran directly, and the staging copy's wait drained
the stream once per full-attention layer.

The device test (test-sycl-kv-view-resolution,
test_binbcast_reads_staged_graph_input_while_recording) proves what the helper
answers. This gate pins how stage_raw_host_source uses it, which that test
cannot reach without a GPU:
  - the lookup runs after the "no staging needed" and zero-span returns and
    before any staging allocation, the weight cache path and the throw, so a
    hit allocates and copies nothing;
  - a hit keeps the staging handle in the op's terminal owners (`stage`) and
    does not set staged_raw_host, whose end-of-op wait is the drain removed;
  - a miss while recording still throws, with the message unchanged;
  - the helper answers only while this thread records.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINBCAST = ROOT / "ggml" / "src" / "ggml-sycl" / "binbcast.cpp"

HELPER = "ggml_sycl_binbcast_graph_input_staged_ptr("
THROW = 'throw std::runtime_error("bin-broadcast raw host staging is not graph-recordable");'


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


class UnbalancedBraces(Exception):
    pass


def block_end(code: str, brace: int) -> int:
    """Index just past the brace that closes the one at `brace`, skipping string and char literals."""
    depth, i, n = 1, brace + 1, len(code)
    while depth:
        if i >= n:
            raise UnbalancedBraces(f"no closing brace for the block at offset {brace}")
        ch = code[i]
        if ch in "\"'":
            i += 1
            while i < n and code[i] != ch:
                i += 2 if code[i] == "\\" else 1
            if i >= n:
                raise UnbalancedBraces(f"unterminated literal in the block at offset {brace}")
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return i


def lambda_body(code: str) -> str:
    start = code.index("auto stage_raw_host_source = [&]")
    brace = code.index("{", code.index("->", start))
    return code[brace:block_end(code, brace)]


def helper_body(code: str) -> str:
    start = code.index("static void * " + HELPER)
    brace = code.index("{", start)
    return code[brace:block_end(code, brace)]


def main() -> int:
    code = strip_comments(BINBCAST.read_text(encoding="utf-8"))
    try:
        body = lambda_body(code)
        helper = helper_body(code)
    except ValueError:
        print("FAIL: binbcast.cpp no longer has stage_raw_host_source or " + HELPER.rstrip("("))
        return 1
    except UnbalancedBraces as exc:
        print(f"FAIL: binbcast.cpp could not be scanned: {exc}")
        return 1
    failures = []

    call = body.find(HELPER)
    if call < 0:
        failures.append("stage_raw_host_source no longer looks up a staged graph input")
    else:
        for pattern, what in (
            (r"ggml_sycl_binbcast_needs_raw_host_staging\(", "the no-staging-needed return"),
            (r"if\s*\(\s*span_bytes\s*==\s*0u?\s*\)", "the zero-span return"),
        ):
            m = re.search(pattern, body)
            pos = m.start() if m else -1
            if pos < 0 or pos > call:
                failures.append(f"the staged-input lookup must come after {what}")
        for marker, what in (
            ("ggml_sycl_tensor_is_weight(tensor)", "the weight cache path"),
            ("ggml_sycl::unified_allocate(req)", "the raw staging allocation"),
            (THROW, "the recording throw"),
        ):
            pos = body.find(marker)
            if pos < 0:
                failures.append(f"stage_raw_host_source lost {what}")
            elif pos < call:
                failures.append(f"the staged-input lookup must come before {what}")

        hit_open = body.find("{", call)
        try:
            hit_end = block_end(body, hit_open) if hit_open >= 0 else -1
        except UnbalancedBraces as exc:
            print(f"FAIL: the staged-input hit block could not be scanned: {exc}")
            return 1
        if hit_end < 0:
            print("FAIL: the staged-input lookup has no hit block")
            return 1
        hit = body[hit_open:hit_end]
        # A miss while recording must reach the throw: a return between the hit block and the weight path would let
        # a recorded kernel bake the host pointer. (The allocation-failure return further down is not on this
        # stretch.)
        weight = body.find("if (ggml_sycl_tensor_is_weight(tensor)", hit_end)
        if weight < 0:
            failures.append("stage_raw_host_source lost the weight cache path after the staged-input lookup")
        elif re.search(r"\breturn\b", body[hit_end:weight]):
            failures.append("a staged-input miss must not return before the weight path (a recording would bake "
                            "the host pointer instead of throwing)")
        if "staged_raw_host" in hit:
            failures.append("a staged-input hit must not set staged_raw_host (its end-of-op wait is the drain)")
        if not re.search(r"\bstage\s*=\s*std::move\(", hit):
            failures.append("a staged-input hit must keep the staging handle in the op's terminal owners (stage)")
        if "return" not in hit:
            failures.append("a staged-input hit must return the staged pointer")

    if "ggml_sycl_graph_recording_this_thread()" not in helper:
        failures.append("the helper must answer only while this thread records")
    if "graph_input_stage_lookup(" not in helper:
        failures.append("the helper must look the copy up in the context's input staging")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("PASS: bin-broadcast reads a staged graph input while recording")
    return 0


if __name__ == "__main__":
    sys.exit(main())
