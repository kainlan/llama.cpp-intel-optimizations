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
import re
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
# Short prefix only (not the full signature): the parameter list wraps
# across lines and clang-format may reflow it differently in the future.
PLANNER_DEMOTE_SIG = "static ggml_layout_mode planner_demote_coalesced_if_misaligned("

# Matches a hand-rolled re-derivation in the `% <const> <cmp> 0` family of
# spellings: the named constant or the literal 32 it currently equals,
# `%`'d and compared against 0 with any comparison operator (!=, ==, >, <,
# >=, <=), in any spacing. Deliberately broader than the one exact string
# ("% MMVQ_COALESCED_TILE_BLOCKS) != 0") the shared predicate itself
# happens to use, so a differently-spelled re-derivation WITHIN THIS FAMILY
# is caught (spec review finding 6, rev-pktr-spec-1; widened again to cover
# comparison operators other than != / == in finding 3, rev-pktr-spec-2 --
# "% 32 > 0" is an equally plausible re-derivation and the narrower
# alternation missed it). NOT a catch-all for every possible re-derivation
# (wording corrected, finding 4, rev-pktr-spec-3): a bitwise spelling such
# as `& (MMVQ_COALESCED_TILE_BLOCKS - 1)` uses neither `%` nor a comparison
# token this regex looks for and would pass undetected.
RE_DERIVATION_RE = re.compile(r"%\s*(MMVQ_COALESCED_TILE_BLOCKS|32)\s*[<>=!]=?\s*0")


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
    # Whitespace-flexible (spec review nit 4, rev-pktr-spec-4): a reflowed
    # signature (e.g. a long parameter list clang-format wraps differently)
    # must not read as "missing definition".
    idx = ws_find(text, signature)
    assert idx >= 0, f"missing definition: {signature}"
    open_idx = text.find("{", idx)
    return text[open_idx : matching_brace(text, open_idx) + 1]


def ws_pattern(needle):
    """Compile a regex matching `needle` where a line wrap is tolerated at
    ANY boundary, not just an EXISTING whitespace run: a boundary next to
    structural punctuation ( `(` `)` `,` `*` `&` ) allows zero or more
    whitespace, while a boundary between two adjacent identifiers/keywords
    requires at least one (real C++ needs the separator; a punctuation
    boundary does not). This matters for a single-parameter signature like
    `bool NAME(int64_t ne00)`: a real formatter can place that sole
    parameter on its own line, wrapping immediately after `(` -- a spot
    with NO whitespace at all in the canonical string, which a simpler
    tokenizer that only allows flex at existing whitespace runs (as this
    file's own predecessor did, and as the sibling woq gate's ws_pattern
    still does) cannot represent (spec review should-fix, rev-pktr-spec-6:
    the exact-count check false-failed on that reflow, and the same
    exact-substring style in the negative "must not redefine" check let a
    reflowed redefinition slip past undetected)."""
    marked = re.sub(r"([(),*&])", r" \1 ", needle)
    raw_tokens = marked.split()
    assert raw_tokens, "empty needle"

    def is_word(tok):
        return re.fullmatch(r"\w+", tok) is not None

    parts = [re.escape(raw_tokens[0])]
    for i in range(1, len(raw_tokens)):
        sep = r"\s+" if is_word(raw_tokens[i - 1]) and is_word(raw_tokens[i]) else r"\s*"
        parts.append(sep)
        parts.append(re.escape(raw_tokens[i]))
    return re.compile("".join(parts))


def ws_find(text, needle, start=0):
    """First match position of `needle` in `text`, boundary-flexible (see
    ws_pattern). Returns -1 if not found; the returned position is a real
    offset into the ORIGINAL text."""
    m = ws_pattern(needle).search(text, start)
    return m.start() if m else -1


def ws_in(text, needle):
    """Boundary-flexible containment check (see ws_pattern)."""
    return ws_pattern(needle).search(text) is not None


def ws_count(text, needle):
    """Number of boundary-flexible matches of `needle` in `text` (see
    ws_pattern) -- for a "defined exactly once" style check."""
    return len(list(ws_pattern(needle).finditer(text)))


def test_predicate_defined_exactly_once():
    # Defined in the standalone header; ggml-sycl.cpp and unified-cache.cpp
    # must each CALL it, never redefine it (both include common.hpp, which
    # includes the header). Boundary-flexible (spec review should-fix,
    # rev-pktr-spec-6): the previous exact-string forms false-failed the
    # positive check on a reflowed single-parameter signature (a real
    # formatter can place `int64_t ne00` alone on the next line, wrapping
    # right after `(`) and, worse, let a reflowed REDEFINITION slip past
    # the negative check undetected (fail-open).
    assert (
        ws_count(rule_header, f"bool {PREDICATE}(int64_t ne00)") == 1
    ), f"{PREDICATE} must be defined exactly once, in q8-dense-layout-rule.hpp"
    for path, text in (("ggml-sycl.cpp", backend), ("unified-cache.cpp", cache)):
        assert not ws_in(
            text, f"bool {PREDICATE}(int64_t"
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
    usage_idx = ws_find(
        body,
        "(usage == tensor_usage::EMBEDDING || usage == tensor_usage::OUTPUT_WEIGHT || "
        "usage == tensor_usage::ATTENTION_WEIGHT || usage == tensor_usage::FFN_WEIGHT)",
    )
    assert usage_idx > 0, "the dense-usage branch must cover EMBEDDING, OUTPUT_WEIGHT, ATTENTION_WEIGHT, FFN_WEIGHT"
    predicate_idx = body.find(f"{PREDICATE}(tensor->ne[0])")
    assert usage_idx < predicate_idx, "the predicate call must be inside the widened dense-usage branch"
    # No hand-rolled re-derivation of the arithmetic alongside the call, in
    # the `% <const> <cmp> 0` family of spellings -- spec review finding 6
    # (rev-pktr-spec-1): the original form of this check matched only the
    # one exact spelling ("% MMVQ_COALESCED_TILE_BLOCKS) != 0") the
    # predicate itself happens to use, so a re-derivation written with
    # "== 0" instead of "!= 0", with the literal 32 instead of the named
    # constant, with a different spacing, or with a different comparison
    # operator (finding 3, rev-pktr-spec-2) would pass unnoticed. This is
    # NOT a catch-all for every possible re-derivation, wording corrected
    # in finding 4, rev-pktr-spec-3 -- a re-derivation spelled as a bitwise
    # test (e.g. `& (MMVQ_COALESCED_TILE_BLOCKS - 1)`) uses no `%`/comparison
    # token this regex looks for and stays undetected.
    assert not RE_DERIVATION_RE.search(
        body[usage_idx:]
    ), "ggml_sycl_adjust_layout_for_tensor must not re-derive the tile-alignment arithmetic inline next to calling the shared predicate"


def test_planner_default_device_layout_calls_the_predicate_in_both_overloads():
    # llama.cpp-pktr spec review nit 9 (rev-pktr-spec-3): both overloads used
    # to call the predicate directly and identically; that duplicate
    # 4-line block is now factored into planner_demote_coalesced_if_misaligned,
    # which each overload calls instead. Check the delegation from each
    # overload, and that the shared helper itself calls the predicate
    # exactly once with no re-derivation anywhere.
    tensor_info_body = function_body(cache, PLANNER_TENSOR_INFO_SIG)
    entry_body = function_body(cache, PLANNER_ENTRY_SIG)
    helper_body = function_body(cache, PLANNER_DEMOTE_SIG)
    assert (
        "planner_demote_coalesced_if_misaligned(" in tensor_info_body and "tensor.ne[0]" in tensor_info_body
    ), "the placement_tensor_info overload must delegate to planner_demote_coalesced_if_misaligned(..., tensor.ne[0])"
    assert (
        "planner_demote_coalesced_if_misaligned(" in entry_body and "entry.ne[0]" in entry_body
    ), "the placement_entry overload must delegate to planner_demote_coalesced_if_misaligned(..., entry.ne[0])"
    assert (
        helper_body.count(f"{PREDICATE}(") == 1
    ), "planner_demote_coalesced_if_misaligned must call the shared predicate exactly once"
    for label, body in (
        ("tensor_info", tensor_info_body),
        ("entry", entry_body),
        ("planner_demote_coalesced_if_misaligned", helper_body),
    ):
        assert not RE_DERIVATION_RE.search(body), (
            f"{label} must not re-derive the tile-alignment arithmetic inline next to calling the shared predicate"
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
