#!/usr/bin/env python3
"""Source gate for llama.cpp-pktr: the Q8_0 dense-weight coalesced/SOA
tile-alignment predicate (ggml_sycl_q8_0_coalesced_tile_aligned,
ggml/src/ggml-sycl/q8-dense-layout-rule.hpp) must be defined exactly once and
every chokepoint that resolves a Q8_0 dense weight's layout must call it
instead of repeating the alignment arithmetic:

  1. ggml_sycl_adjust_layout_for_tensor (ggml-sycl.cpp) -- the RUNTIME
     chokepoint that decides the materialized layout a Q8_0
     ATTENTION_WEIGHT/FFN_WEIGHT/EMBEDDING/OUTPUT_WEIGHT tensor actually gets.
  2. planner_default_device_layout, BOTH overloads (unified-cache.cpp) -- the
     PLANNING chokepoint that charges bytes and decides re-placement layout
     for the same tensor classes; if these disagree with (1), the planner's
     byte-charging and re-placement decisions target a layout runtime never
     materializes (see the llama.cpp-pktr comment on the tensor_info overload
     in unified-cache.cpp for the mismatch this would reintroduce).

This file exists so a future edit that re-derives the alignment condition by
hand at a new or existing call site (instead of calling the shared predicate)
fails a test instead of silently drifting from the other chokepoint.

Runs under pytest (llama_test_pytest registration) and as a plain script.
Point it at alternate copies (to exercise the RED path with a deliberately
broken tree) via GGML_SYCL_PKTR_BACKEND_SOURCE / GGML_SYCL_PKTR_CACHE_SOURCE /
GGML_SYCL_PKTR_RULE_HEADER_SOURCE; defaults are the in-tree files. No SYCL
device or build is touched.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_PKTR_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))
CACHE = Path(os.environ.get("GGML_SYCL_PKTR_CACHE_SOURCE", str(ROOT / "ggml/src/ggml-sycl/unified-cache.cpp")))
RULE_HEADER = Path(
    os.environ.get("GGML_SYCL_PKTR_RULE_HEADER_SOURCE", str(ROOT / "ggml/src/ggml-sycl/q8-dense-layout-rule.hpp"))
)
COMMON = Path(os.environ.get("GGML_SYCL_PKTR_COMMON_SOURCE", str(ROOT / "ggml/src/ggml-sycl/common.hpp")))

backend = BACKEND.read_text()
cache = CACHE.read_text()
rule_header = RULE_HEADER.read_text()
common = COMMON.read_text()

PREDICATE = "ggml_sycl_q8_0_coalesced_tile_aligned"
ADJUST_SIG = "layout_mode ggml_sycl_adjust_layout_for_tensor(const ggml_tensor * tensor, layout_mode target, int device) {"
PLANNER_TENSOR_INFO_SIG = "static ggml_layout_mode planner_default_device_layout(const placement_tensor_info & tensor,"
PLANNER_ENTRY_SIG = "static ggml_layout_mode planner_default_device_layout(const placement_entry & entry, int device_id) {"


def matching_brace(text, open_idx):
    """Comment/string-aware brace match (self-contained copy, per this fork's
    one-file-per-gate convention)."""
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
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def function_body(text, signature):
    idx = text.find(signature)
    assert idx >= 0, f"missing definition: {signature}"
    open_idx = text.find("{", idx)
    return text[open_idx : matching_brace(text, open_idx) + 1]


def test_predicate_defined_exactly_once():
    # Defined in the standalone header; ggml-sycl.cpp and unified-cache.cpp
    # must each CALL it, never redefine it (both include common.hpp, which
    # includes the header).
    assert (
        rule_header.count(f"bool {PREDICATE}(int64_t ne00)") == 1
    ), f"{PREDICATE} must be defined exactly once, in q8-dense-layout-rule.hpp"
    for path, text in (("ggml-sycl.cpp", backend), ("unified-cache.cpp", cache)):
        assert (
            f"bool {PREDICATE}(int64_t" not in text
        ), f"{path} must not redefine {PREDICATE} -- it must call the shared helper"


def test_common_hpp_guards_the_local_copies_with_static_asserts():
    # q8-dense-layout-rule.hpp keeps local copies of QK8_0 and
    # MMVQ_COALESCED_TILE_BLOCKS to stay free of ggml/SYCL includes;
    # common.hpp must static_assert they still match the canonical values.
    assert "GGML_SYCL_Q8_DENSE_LAYOUT_RULE_QK == QK8_0" in common, "missing static_assert guarding the local QK copy"
    assert (
        "GGML_SYCL_Q8_DENSE_LAYOUT_RULE_TILE_BLOCKS == MMVQ_COALESCED_TILE_BLOCKS" in common
    ), "missing static_assert guarding the local tile-blocks copy"
    assert 'include "q8-dense-layout-rule.hpp"' in common, "common.hpp must include q8-dense-layout-rule.hpp"


def test_adjust_layout_for_tensor_calls_the_predicate_for_the_dense_usage_set():
    body = function_body(backend, ADJUST_SIG)
    assert f"{PREDICATE}(tensor->ne[0])" in body, "ggml_sycl_adjust_layout_for_tensor must call the shared predicate"
    # The dense-usage branch (llama.cpp-os8k, extended by llama.cpp-pktr) must
    # cover all four usages the predicate now applies to.
    usage_idx = body.find(
        "(usage == tensor_usage::EMBEDDING || usage == tensor_usage::OUTPUT_WEIGHT ||\n"
        "         usage == tensor_usage::ATTENTION_WEIGHT || usage == tensor_usage::FFN_WEIGHT)"
    )
    assert usage_idx > 0, "the dense-usage branch must cover EMBEDDING, OUTPUT_WEIGHT, ATTENTION_WEIGHT, FFN_WEIGHT"
    predicate_idx = body.find(f"{PREDICATE}(tensor->ne[0])")
    assert usage_idx < predicate_idx, "the predicate call must be inside the widened dense-usage branch"
    # No hand-rolled re-derivation of the arithmetic alongside the call.
    assert "% MMVQ_COALESCED_TILE_BLOCKS) != 0" not in body[usage_idx:], (
        "ggml_sycl_adjust_layout_for_tensor must not re-derive the tile-alignment arithmetic inline "
        "next to calling the shared predicate"
    )


def test_planner_default_device_layout_calls_the_predicate_in_both_overloads():
    tensor_info_body = function_body(cache, PLANNER_TENSOR_INFO_SIG)
    entry_body = function_body(cache, PLANNER_ENTRY_SIG)
    assert (
        f"{PREDICATE}(tensor.ne[0])" in tensor_info_body
    ), "the placement_tensor_info overload of planner_default_device_layout must call the shared predicate"
    assert (
        f"{PREDICATE}(entry.ne[0])" in entry_body
    ), "the placement_entry overload of planner_default_device_layout must call the shared predicate"
    for label, body in (("tensor_info", tensor_info_body), ("entry", entry_body)):
        assert "% MMVQ_COALESCED_TILE_BLOCKS) != 0" not in body, (
            f"planner_default_device_layout({label}) must not re-derive the tile-alignment arithmetic inline "
            "next to calling the shared predicate"
        )


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
