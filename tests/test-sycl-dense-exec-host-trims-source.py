#!/usr/bin/env python3
"""Source gate for llama.cpp-2lxw: host-side trims of the dense split executor.

Input refresh on a plan-cache hit (item D). The executor copies only the
graph inputs whose host bytes moved since its own last copy into each input's
stable device copy (ggml_backend_sycl_context::graph_input_staging). That
skip is sound only while nothing else has written or replaced that copy, which
the context's graph_input_staging_generation tells it. So:

  1. every member function that writes or replaces a staging entry bumps the
     generation -- on each path that touches the entry -- and nothing in any
     ggml-sycl source outside those functions touches the map;
  2. the executor takes the moved-only refresh only on a plan-cache hit that
     records nothing, checks the generation and the input list first, and
     skips exactly the inputs whose bytes did not move and that no node
     writes. Every other input is copied from its host bytes into its
     staging copy, or, without one, refreshed in full;
  2a. the executor's memo of that refresh holds no staging handle: a copy
     looks the entry up when it runs, so nothing keeps an allocation alive
     after the staging map has dropped it.

What makes the skip sound is (1): no node writes a staging copy. Kernels
only read them, through graph_input_stage_lookup, and a node that writes a
graph input writes the tensor's own storage, not its staging copy. So a
copy's content changes only through the members in (1), or through the
executor's own copy, which it records. The executor's `written` flag, which
it sets for written graph-input (CONTROL) roots, is an extra margin on top
of that and not what the skip relies on, which is why it does not have to
cover every input, such as an INPUT-flagged view of a non-input root.
Lookups hand a caller the copy's handle, so every caller that takes the
handle, rather than only the pointer, is listed here and must be a reader.

Crossings (item C). The planner lays each crossing's slices out next to each
other in the arena and hands out runs (block-exec-dense.hpp,
dense_exec_coalesce), so:

  3. the arena side of every crossing copies whole runs, never a slice at a
     time;
  4. leaving a range waits on its queue only when the crossing did not, and
     the crossing reports that it waited only after it has;
  4a. a plan-cache hit shares the prepared plan, slices and run views with
     the cache entry instead of copying their handles.

Publications (item A-lite). Every storage publish and restore starts a new
data-pointer cache, and a split token publishes and restores dozens of
slices. unordered_map::clear() writes the whole bucket array even when the
map is empty, so:

  5. the cache is cleared only when it holds something, and nothing on the
     publish/restore path fills it, so a batch clears it once.

Runs under pytest and as a plain script. GGML_SYCL_2LXW_BACKEND_SOURCE,
GGML_SYCL_2LXW_COMMON_SOURCE and GGML_SYCL_2LXW_SYCL_DIR (the other ggml-sycl
sources) point it at other copies (to see it fail on a broken tree). No SYCL
device or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_2LXW_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))
COMMON = Path(os.environ.get("GGML_SYCL_2LXW_COMMON_SOURCE", str(ROOT / "ggml/src/ggml-sycl/common.hpp")))

backend = BACKEND.read_text()
common = COMMON.read_text()

SYCL_DIR = Path(os.environ.get("GGML_SYCL_2LXW_SYCL_DIR", str(ROOT / "ggml/src/ggml-sycl")))


def sycl_sources():
    """Every ggml-sycl source, with the backend and common overrides applied."""
    out = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h") or not path.is_file():
            continue
        rel = path.relative_to(SYCL_DIR).as_posix()
        if rel == "ggml-sycl.cpp":
            out[rel] = backend
        elif rel == "common.hpp":
            out[rel] = common
        else:
            out[rel] = path.read_text(errors="replace")
    return out

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


def struct_body(text, name):
    """Body of the struct `name` defined in `text` (not a forward declaration)."""
    m = re.search(r"\bstruct\s+" + re.escape(name) + r"\s*\{", text)
    assert m, f"missing definition: struct {name}"
    return text[m.start() : matching_brace(text, m.end() - 1) + 1]


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
    # The create/replace path resets any old entry, then may give up (the
    # allocation or its resolve fails) before it stores a new one. The bump
    # must come between the reset and the first way out after it.
    reset = re.search(r"it->second\.handle\s*=\s*ggml_sycl::mem_handle\{\}", stage)
    assert reset, "graph_input_stage no longer resets the entry it replaces"
    first_exit = stage.find("return", reset.end())
    assert first_exit > 0 and GENERATION + "++" in stage[reset.end() : first_exit], (
        "replacing or creating an entry must bump the generation before any return"
    )


def test_no_other_code_touches_the_staging_map():
    allowed = [member_body(common, name) for name in STAGING_WRITERS + STAGING_READERS]
    rest = strip_comments(common)
    for body in allowed:
        rest = rest.replace(strip_comments(body), "")
    touches = re.findall(r"\bgraph_input_staging\b(?!_)", rest)
    # The declaration itself is the only remaining mention.
    assert len(touches) == 1, f"graph_input_staging is touched outside its members ({len(touches)} mentions)"
    sources = sycl_sources()
    assert len(sources) > 50 and "ggml-sycl.cpp" in sources and "getrows.cpp" in sources, "ggml-sycl sources not found"
    for path, text in sources.items():
        if path == "common.hpp":
            continue
        assert not re.search(r"\bgraph_input_staging\b(?!_)", strip_comments(text)), (
            f"{path} touches graph_input_staging directly; go through the context's members"
        )


# Lookups that take the staging copy's handle, not only its pointer. Each is
# a reader, except the executor's memo, which writes through its handle and
# records what it wrote. A new one is a writer until someone shows otherwise:
# add it here only if it reads, or make it bump the generation.
LOOKUP_HANDLE_TAKERS = {
    ("getrows.cpp", "&out_handle"),  # pre-staged get_rows indices, read by the kernel
    ("ggml-sycl.cpp", "&dst"),  # the executor's moved-only copy, held for that copy only
}


def test_every_lookup_that_takes_the_handle_is_known():
    found = set()
    for path, text in sycl_sources().items():
        code = strip_comments(text)
        if path == "common.hpp":
            # The staging members themselves: the writers bump, the lookup is
            # the definition.
            for name in STAGING_WRITERS + STAGING_READERS:
                code = code.replace(strip_comments(member_body(text, name)), "")
        for m in re.finditer(r"\bgraph_input_stage_lookup\s*\(", code):
            depth, i, args, arg = 1, m.end(), [], ""
            while depth:
                ch = code[i]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if depth == 1 and ch == ",":
                    args.append(arg.strip())
                    arg = ""
                elif depth:
                    arg += ch
                i += 1
            args.append(arg.strip())
            assert len(args) == 5, f"{path}: unexpected graph_input_stage_lookup call {args}"
            if args[3] != "nullptr":
                found.add((path, args[3]))
    unknown = found - LOOKUP_HANDLE_TAKERS
    assert not unknown, f"a lookup takes a staging copy's handle and may write through it: {sorted(unknown)}"
    assert found == LOOKUP_HANDLE_TAKERS, f"a listed handle taker is gone; update the list: {sorted(LOOKUP_HANDLE_TAKERS - found)}"


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
    skip = re.findall(
        r"if\s*\(\s*!\s*in\.written\s*&&\s*!\s*ggml_sycl::dense_exec_input_moved\(\s*in\.snapshot\s*,\s*t->data\s*,"
        r"\s*in\.bytes\s*\)\s*\)\s*\{\s*refresh_skipped_\+\+\s*;\s*continue\s*;\s*\}",
        moved,
    )
    assert len(skip) == 1, "skip exactly the inputs that no node writes and whose bytes did not move"
    assert moved.count("refresh_skipped_++") == 1 and moved.count("continue") == 2, (
        "the unmoved skip and the full refresh are the only ways past the copy"
    )
    full = re.findall(
        r"if\s*\(\s*!\s*in\.staged\s*\|\|\s*t\s*==\s*nullptr\s*\|\|\s*t->data\s*==\s*nullptr\s*\|\|"
        r"\s*ggml_nbytes\(t\)\s*!=\s*in\.bytes\s*\)\s*\{\s*in\.snapshot\s*=\s*ggml_sycl::dense_exec_input_snapshot\{\}\s*;"
        r"\s*refresh_full_\s*\+=\s*graph_refresh_input_tensor\(&ctx_,\s*t,\s*q\)\s*\?\s*1\s*:\s*0\s*;\s*continue\s*;\s*\}",
        moved,
    )
    assert len(full) == 1, "an input without a staging copy is refreshed in full"
    # Past the skip, the input is copied: its entry looked up (a missing one
    # refuses the whole moved-only refresh), then host to device, then
    # recorded. Nothing may condition or reorder the copy.
    copy = re.findall(
        r"refresh_skipped_\+\+\s*;\s*continue\s*;\s*\}\s*"
        r"ggml_sycl::mem_handle\s+dst\s*;\s*void\s*\*\s*dst_ptr\s*=\s*nullptr\s*;\s*"
        r"if\s*\(\s*!\s*ctx_\.graph_input_stage_lookup\(\s*t\s*,\s*in\.bytes\s*,\s*original_device_\s*,\s*&dst\s*,"
        r"\s*&dst_ptr\s*\)\s*\)\s*\{\s*return\s+false\s*;\s*\}\s*"
        r"const\s+ggml_sycl::mem_handle\s+src\s*=\s*ggml_sycl::mem_handle::from_direct\(\s*t->data\s*,[^;]*,\s*in\.bytes\s*\)\s*;\s*"
        r"\(void\)\s*ggml_sycl::mem_copy_async\(\s*dst\s*,\s*src\s*,\s*in\.bytes\s*,\s*q\s*\)\s*;\s*"
        r"ggml_sycl::dense_exec_input_record\(\s*in\.snapshot\s*,\s*t->data\s*,\s*in\.bytes\s*\)\s*;\s*"
        r"refresh_copied_\+\+\s*;\s*\}",
        moved,
    )
    assert len(copy) == 1, "every input past the skip is copied from its host bytes into its staging copy, then recorded"
    assert moved.count("mem_copy_async") == 1, "the moved-only refresh copies in exactly one place"


def test_the_refresh_memo_holds_no_staging_handle():
    memo = strip_comments(struct_body(backend, "refresh_input"))
    assert "mem_handle" not in memo, "the refresh memo must not hold a staging copy's handle"
    remember = strip_comments(member_body(backend, "void remember_refresh("))
    lookups = re.findall(r"graph_input_stage_lookup\(([^;]*)\)", remember)
    assert len(lookups) == 1 and lookups[0].split(",")[3].strip() == "nullptr", (
        "remembering a refresh asks only whether an input has a staging copy, not for its handle"
    )


def test_crossings_copy_whole_runs():
    stage = strip_comments(member_body(backend, "bool stage_range("))
    assert "io.stage_in_runs" in stage and "built_->runs[idx].stage_in" in stage, (
        "stage_range must copy into the arena by the plan's stage-in runs"
    )
    assert "built_->slices[" not in stage, "stage_range must not copy into the arena one slice at a time"

    leave = strip_comments(member_body(backend, "void leave_range("))
    for runs in ("io.copy_out_src_runs", "io.copy_out_dst_runs", "runs.copy_out_src", "runs.copy_out_dst"):
        assert runs in leave, f"leave_range must copy out by the plan's runs ({runs})"
    assert "built_->slices[" not in leave, "leave_range must not copy out one slice at a time"

    # Each run's view is on the device whose arena that side of the crossing
    # copies: the range's device for the stage in and the copy out's source,
    # the backend's device for the copy out's destination.
    allocate = strip_comments(
        member_body(backend, "bool allocate(const std::shared_ptr<const ggml_sycl::placement_plan> & plan_owner,")
    )
    for device, runs, views in (
        ("d", "io.stage_in_runs", "build.runs[r].stage_in"),
        ("d", "io.copy_out_src_runs", "build.runs[r].copy_out_src"),
        ("original_device_", "io.copy_out_dst_runs", "build.runs[r].copy_out_dst"),
    ):
        call = r"views\(\s*" + device + r"\s*,\s*" + re.escape(runs) + r"\s*,\s*" + re.escape(views) + r"\s*\)"
        assert len(re.findall(call, allocate)) == 1, f"{views} must be views of device {device}'s arena"
    assert re.search(r"const\s+int\s+d\s*=\s*build\.plan\.ranges\[r\]\.device\s*;", allocate), (
        "d must be the range's device"
    )


def test_a_plan_cache_hit_shares_the_prepared_plan():
    prepared = strip_comments(struct_body(backend, "prepared_plan"))
    assert re.search(r"std::shared_ptr<const\s+ggml_sycl_block_exec_dense_prepared>\s+built\s*;", prepared), (
        "the cache entry must hold the prepared plan by a shared, immutable reference"
    )
    for copied in ("dense_exec_plan", "mem_handle", "range_runs"):
        assert copied not in prepared, f"the cache entry holds its own {copied} rather than sharing the prepared one"
    reuse = strip_comments(member_body(backend, "bool reuse_prepared("))
    tail = reuse[reuse.rfind("return false") :]
    assert re.search(r"return\s+false\s*;(?:\s*\})+\s*built_\s*=\s*p\.built\s*;\s*return\s+true\s*;\s*\}\s*$", tail), (
        "a plan-cache hit must take the entry's prepared plan by reference, copying nothing"
    )


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


def test_data_ptr_cache_clears_only_when_filled():
    body = strip_comments(member_body(backend, "void ggml_sycl_data_ptr_cache_new_graph("))
    clears = [m.start() for m in re.finditer(r"g_data_ptr_cache\.clear\(\)", body)]
    assert len(clears) == 1, f"one clear of the data-pointer cache ({len(clears)})"
    guard = re.search(r"if\s*\(\s*!\s*g_data_ptr_cache\.empty\(\)\s*\)\s*\{", body)
    assert guard and guard.end() <= clears[0] < matching_brace(body, guard.end() - 1), (
        "the data-pointer cache must be cleared only when it holds something"
    )
    code = strip_comments(backend)
    assert len(re.findall(r"g_data_ptr_cache\.clear\(\)", code)) == 1, (
        "every clear of the data-pointer cache goes through ggml_sycl_data_ptr_cache_new_graph"
    )

    # One clear per batch holds only while publishing and restoring fill
    # nothing: the cache is filled by ggml_sycl_get_data_ptr_slow alone.
    batch = [
        member_body(backend, "static bool ggml_sycl_publish_existing_storage_handle_for_device("),
        struct_body(backend, "block_exec_tensor_storage_slot_snapshot"),
        struct_body(backend, "block_exec_scoped_tensor_storage_publication"),
    ]
    for body in batch:
        body = strip_comments(body)
        assert "get_data_ptr" not in body and "g_data_ptr_cache[" not in body, (
            "the publish/restore path must not fill the data-pointer cache"
        )

    # One call level down: every function those bodies call, in every
    # ggml-sycl source that defines it (all overloads). The map is static in
    # ggml-sycl.cpp, so code elsewhere can fill it only through get_data_ptr.
    local = {"capture", "restore", "publish", "ensure_extra", "block_exec_scoped_tensor_storage_publication",
             "ggml_sycl_data_ptr_cache_new_graph", "ggml_nbytes"}
    keywords = {"if", "for", "while", "switch", "return", "sizeof", "static_cast", "const_cast", "reinterpret_cast"}
    callees = set()
    for body in batch:
        body = re.sub(r'"(?:\\.|[^"\\])*"', '""', strip_comments(body))
        callees |= {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body)}
    callees -= local | keywords
    for known in ("ggml_sycl_init_layout_info", "release_extra_gpu", "ggml_sycl_view_root_and_offset", "resolve"):
        assert known in callees, f"the publish/restore path no longer calls {known}; re-check this census"
    sources = {path: strip_comments(text) for path, text in sycl_sources().items()}
    for name in sorted(callees):
        definition = re.compile(
            r"^[ \t]*(?:(?:static|inline|constexpr|virtual|explicit|friend)\s+)*[A-Za-z_][\w:<>,\*&\s]*?[\s\*&:]"
            r"(?:\w+::)*" + re.escape(name) + r"\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{",
            re.M,
        )
        defined = 0
        for path, code in sources.items():
            for m in definition.finditer(code):
                defined += 1
                callee = code[m.start() : matching_brace(code, m.end() - 1) + 1]
                assert "get_data_ptr" not in callee and "g_data_ptr_cache[" not in callee, (
                    f"{name} ({path}), called while publishing or restoring, fills the data-pointer cache"
                )
        assert defined, f"no definition of {name}, called while publishing or restoring, in the ggml-sycl sources"


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
