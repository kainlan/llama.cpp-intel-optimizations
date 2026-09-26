#!/usr/bin/env python3
"""Source gate for llama.cpp-2lxw: host-side trims of the dense split executor.

Input refresh on a plan-cache hit (item D). The executor copies only the
graph inputs whose host bytes moved since its own last copy into each input's
stable device copy (ggml_backend_sycl_context::graph_input_staging). That
skip is sound only while nothing else has written or replaced that copy, which
the context's graph_input_staging_generation tells it. So:

  1. every member function that writes or replaces a staging entry bumps the
     generation, and nothing outside those functions touches the map;
  2. the executor takes the moved-only refresh only on a plan-cache hit that
     records nothing, and checks the generation and the input list first.

Crossings (item C). The planner lays each crossing's slices out next to each
other in the arena and hands out runs (block-exec-dense.hpp,
dense_exec_coalesce), so:

  3. the arena side of every crossing copies whole runs, never a slice at a
     time;
  4. leaving a range waits on its queue only when the crossing did not, and
     the crossing reports that it waited only after it has.

Runs under pytest and as a plain script. GGML_SYCL_2LXW_BACKEND_SOURCE and
GGML_SYCL_2LXW_COMMON_SOURCE point it at other copies (to see it fail on a
broken tree). No SYCL device or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_2LXW_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))
COMMON = Path(os.environ.get("GGML_SYCL_2LXW_COMMON_SOURCE", str(ROOT / "ggml/src/ggml-sycl/common.hpp")))

backend = BACKEND.read_text()
common = COMMON.read_text()

GENERATION = "graph_input_staging_generation"

# The members that write or replace an entry, and the one that only reads.
STAGING_WRITERS = ("graph_input_stage(", "graph_input_refresh(", "graph_input_staging_clear(")
STAGING_READERS = ("graph_input_stage_lookup(",)


def matching_brace(text, open_idx):
    """Comment/string-aware brace match."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "chr"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state in ("str", "chr"):
            if ch == "\\":
                i += 2
                continue
            if (state == "str" and ch == '"') or (state == "chr" and ch == "'"):
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def member_body(text, name):
    """Body of the member function `name(` defined in `text` (the first
    definition: a name followed by a parameter list and an opening brace)."""
    m = re.search(r"\b" + re.escape(name) + r"[^;{}]*\)\s*(const\s*)?\{", text)
    assert m, f"missing definition: {name}"
    open_idx = m.end() - 1
    return text[m.start() : matching_brace(text, open_idx) + 1]


def test_every_staging_writer_bumps_the_generation():
    code = strip_comments(common)
    assert re.search(r"uint64_t\s+" + GENERATION + r"\s*=\s*0\s*;", code), f"{GENERATION} must be a context member"
    for writer in STAGING_WRITERS:
        body = strip_comments(member_body(common, writer))
        assert GENERATION + "++" in body, f"{writer} writes a staging entry without bumping {GENERATION}"
    # graph_input_stage writes an entry on two paths: reusing it, and
    # replacing it. Each must bump.
    stage = strip_comments(member_body(common, "graph_input_stage("))
    reuse = stage.find("mem_copy(it->second.handle")
    assert reuse >= 0, "graph_input_stage no longer reuses an entry in place"
    assert GENERATION + "++" in stage[reuse : stage.find("return", reuse)], "reusing an entry must bump the generation"


def test_no_other_code_touches_the_staging_map():
    allowed = [member_body(common, name) for name in STAGING_WRITERS + STAGING_READERS]
    rest = strip_comments(common)
    for body in allowed:
        rest = rest.replace(strip_comments(body), "")
    touches = re.findall(r"\bgraph_input_staging\b(?!_)", rest)
    # The declaration itself is the only remaining mention.
    assert len(touches) == 1, f"graph_input_staging is touched outside its members ({len(touches)} mentions)"
    for path, text in (("ggml-sycl.cpp", backend),):
        assert not re.search(r"\bgraph_input_staging\b(?!_)", strip_comments(text)), (
            f"{path} touches graph_input_staging directly; go through the context's members"
        )


def test_moved_only_refresh_is_gated_on_a_recordless_plan_cache_hit():
    prepare = strip_comments(member_body(backend, "void prepare_graphs("))
    call = prepare.find("refresh_moved_inputs()")
    assert call >= 0, "prepare_graphs must refresh through refresh_moved_inputs on a plan-cache hit"
    guard = prepare[prepare.rfind("if", 0, call) : call]
    assert "plan_cache_result::HIT" in guard and "any_missing" in guard, (
        "the moved-only refresh must run only on a plan-cache hit that records nothing"
    )
    full = prepare.find("graph_refresh_input_tensors(&ctx_, cgraph_)", call)
    assert full >= 0, "a refused moved-only refresh must fall back to the full refresh"
    assert "remember_refresh()" in prepare[full:], "the full refresh must be remembered for the next hit"

    moved = strip_comments(member_body(backend, "bool refresh_moved_inputs("))
    for fact in (GENERATION, "cached_input_tensors", "input_tensors_cached"):
        assert fact in moved, f"refresh_moved_inputs must check {fact} before trusting its list"
    assert moved.find(GENERATION) < moved.find("mem_copy_async"), "check the generation before copying"
    assert "dense_exec_input_moved" in moved, "only moved inputs are copied"
    assert "in.written" in moved, "an input a node writes is copied every graph"


def test_crossings_copy_whole_runs():
    stage = strip_comments(member_body(backend, "bool stage_range("))
    assert "io.stage_in_runs" in stage and "runs_[idx].stage_in" in stage, (
        "stage_range must copy into the arena by the plan's stage-in runs"
    )
    assert "slices_[" not in stage, "stage_range must not copy into the arena one slice at a time"

    leave = strip_comments(member_body(backend, "void leave_range("))
    for runs in ("io.copy_out_src_runs", "io.copy_out_dst_runs", "runs.copy_out_src", "runs.copy_out_dst"):
        assert runs in leave, f"leave_range must copy out by the plan's runs ({runs})"
    assert "slices_[" not in leave, "leave_range must not copy out one slice at a time"


def test_leaving_a_range_waits_once():
    leave = strip_comments(member_body(backend, "void leave_range("))
    def blocks(pattern):
        spans = []
        for m in re.finditer(pattern, leave):
            open_idx = leave.index("{", m.end() - 1)
            spans.append((open_idx, matching_brace(leave, open_idx)))
        return spans

    traced = blocks(r"if\s*\(\s*phase_trace_\s*\)\s*\{")
    waits = [m.start() for m in re.finditer(r"q_exec->wait_and_throw\(\)", leave)]
    untraced = [w for w in waits if not any(a < w < b for a, b in traced)]
    assert len(untraced) == 1, f"leave_range must wait on its queue in exactly one untraced place ({len(untraced)})"
    guarded = blocks(r"if\s*\(\s*!\s*cross\([^{]*\{")
    assert any(a < untraced[0] < b for a, b in guarded), "leave_range must wait only when the crossing did not"

    cross = strip_comments(member_body(backend, "bool cross("))
    wait = cross.find("q_from->wait_and_throw()")
    assert wait >= 0, "a crossing waits for its source queue"
    assert "return true" in cross and cross.find("return true") > wait, "cross reports a wait only after it"
    assert all(m.start() < wait for m in re.finditer(r"return false", cross)), "cross must not report false after waiting"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    raise SystemExit(1 if failed else 0)
