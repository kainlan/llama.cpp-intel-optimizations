#!/usr/bin/env python3
"""pool_leg's arena-off free() must route through pool_legacy_release() with
the live graph-recording state, source-asserted so it is checked without a
SYCL device.

The release rule itself is covered by the host-only test
sycl-pool-legacy-release, which drives pool-legacy-release.hpp. That test
cannot see the call site: pool_leg passing `false` for graph_recording, or
writing its free list directly again, would leave it green while a block freed
during graph recording goes back on the free list and a later alloc() reuses
memory the recorded graph still references. This file pins the call site.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

POOL_LEG = "struct ggml_sycl_pool_leg : public ggml_sycl_pool {"
FREE_FN = "void free(void * ptr, size_t size) override {"
DTOR_FN = "~ggml_sycl_pool_leg() {"


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy
    of the walker in test-sycl-graph-weight-layout-fallback-source.py -- this
    fork's source-gate convention is one self-contained file per check, not a
    shared import, so each file carries its own copy."""
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
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
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
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError("unclosed brace")


def block(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    return text[start:matching_brace(text, brace) + 1]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def pool_leg_violations(source: str) -> list[str]:
    if POOL_LEG not in source:
        return [f"{POOL_LEG} is missing"]
    pool_leg = block(source, POOL_LEG)
    found: list[str] = []

    if FREE_FN not in pool_leg:
        found.append("pool_leg has no free() override")
    else:
        free_code = strip_comments(block(pool_leg, FREE_FN))
        calls = re.findall(r"ggml_sycl::pool_legacy_release\(\s*ptr\s*,\s*([^,]+?)\s*,", free_code)
        if len(calls) != 1:
            found.append(f"free() calls ggml_sycl::pool_legacy_release {len(calls)} times, expected once")
        elif calls[0] != "ggml_sycl_graph_recording_active()":
            found.append(
                f"free() passes `{calls[0]}` as graph_recording, not ggml_sycl_graph_recording_active() -- "
                "a block freed during graph recording would go back on the free list"
            )
        if re.search(r"buffer_pool\s*\[", free_code):
            found.append("free() writes buffer_pool directly instead of through pool_legacy_release()")

    if DTOR_FN not in pool_leg:
        found.append("pool_leg has no destructor")
    else:
        dtor_code = strip_comments(block(pool_leg, DTOR_FN))
        release_at = dtor_code.find("release_graph_retained();")
        arena_at = dtor_code.find("if (arena_mode_)")
        if release_at < 0 or arena_at < 0 or release_at > arena_at:
            found.append(
                "the destructor does not call release_graph_retained() before its arena_mode_ branch -- "
                "arena-off graph-retained handles would be released only by member destruction"
            )

    return found


def test_pool_leg_free_routes_through_the_release_rule() -> None:
    assert pool_leg_violations(SOURCE.read_text()) == []


def test_recording_argument_mutation_is_witnessed() -> None:
    """Pass `false` for graph_recording, the edit that would leave the host
    test green, and confirm this checker catches it."""
    source = SOURCE.read_text()
    mutated = re.sub(
        r"(ggml_sycl::pool_legacy_release\(\s*ptr\s*,\s*)ggml_sycl_graph_recording_active\(\)",
        r"\1false",
        source,
        count=1,
    )
    assert mutated != source, "recording-argument mutation did not change the source"
    violations = pool_leg_violations(mutated)
    assert any("as graph_recording" in v for v in violations), f"mutation was not witnessed: {violations}"


def test_destructor_mutation_is_witnessed() -> None:
    """Move release_graph_retained() back inside the arena-only branch."""
    source = SOURCE.read_text()
    dtor = block(block(source, POOL_LEG), DTOR_FN)
    mutated_dtor = dtor.replace("        release_graph_retained();\n", "", 1).replace(
        "if (arena_mode_) {\n", "if (arena_mode_) {\n            release_graph_retained();\n", 1
    )
    assert mutated_dtor != dtor, "destructor mutation did not change the source"
    violations = pool_leg_violations(source.replace(dtor, mutated_dtor, 1))
    assert any("release_graph_retained() before" in v for v in violations), (
        f"mutation was not witnessed: {violations}"
    )
