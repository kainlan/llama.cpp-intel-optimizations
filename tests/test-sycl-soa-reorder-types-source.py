#!/usr/bin/env python3
"""Source gate for llama.cpp-76os: one table says which weight types have an
AOS->SOA reorder, and the planner, the runtime clamp and the fill all read it.

ggml_sycl_soa_reorder_supported_type (ggml/src/ggml-sycl/soa-reorder-types.hpp)
is the single list. Its readers:

  1. ggml_sycl_layout_supports_soa (ggml-sycl.cpp): the runtime clamp that
     ggml_sycl_adjust_layout_for_tensor applies to every SOA request.
  2. unified_cache::load_partial_rows (unified-cache.cpp): the partial-row fill.
  3. layout_policy::get_optimal (common.hpp): the planner's layout source. Its
     SOA fallthroughs ("SOA is safe for all quantized types") are what planned
     SOA for IQ*/Q2_0 expert types while the runtime clamped them to AOS.

The fill itself is the materializer: every per-type switch of reorder kernels
(reorder_aos_to_soa_device, reorder_rows_to_soa, reorder_data_internal_,
ggml_sycl_reorder_weight_gpu, ggml_sycl_reorder_weight_cpu and the size helper
ggml_sycl_reorder_expected_size) must carry exactly the table's types, so the
table cannot name a type the fill has no kernel for (the planner would then plan
a layout nothing materializes) nor omit one it has (a kernel nothing reaches).

The other readers are the buffer-side eligibility checks (init_tensor, set_tensor,
should_cpu_reorder, reorder_tensor_to_soa, the dispatch's type_has_reorder_support):
none may keep a private `type == Q4_0 || ... || type == MXFP4` chain. A statement
that compares against all five table types is flagged wherever it appears.

The compiled counterpart (run_planned_layout_materializable_test in
tests/test-sycl-layout-choice.cpp) drives get_optimal and the real planner over
every quantized type; this gate is the structural half: the readers consult the
table instead of keeping a private copy.

Runs under pytest and as a plain script. Alternate copies (for a deliberately
broken tree) via GGML_SYCL_76OS_{HEADER,BACKEND,CACHE,COMMON}_SOURCE. No device
or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
HEADER = Path(os.environ.get("GGML_SYCL_76OS_HEADER_SOURCE", str(SYCL / "soa-reorder-types.hpp")))
BACKEND = Path(os.environ.get("GGML_SYCL_76OS_BACKEND_SOURCE", str(SYCL / "ggml-sycl.cpp")))
CACHE = Path(os.environ.get("GGML_SYCL_76OS_CACHE_SOURCE", str(SYCL / "unified-cache.cpp")))
COMMON = Path(os.environ.get("GGML_SYCL_76OS_COMMON_SOURCE", str(SYCL / "common.hpp")))

PREDICATE = "ggml_sycl_soa_reorder_supported_type"
CASE_RE = re.compile(r"\bcase\s+(GGML_TYPE_[A-Z0-9_]+)\s*:")


def strip_comments(text):
    """Blank out // and /* */ comments (string-aware) keeping offsets stable."""
    out = []
    i = 0
    state = "code"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                out.append("  ")
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                state = "str"
            out.append(ch)
        elif state == "line":
            if ch == "\n":
                state = "code"
                out.append(ch)
            else:
                out.append(" ")
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
        elif state == "str":
            if ch == "\\":
                out.append(ch + nxt)
                i += 2
                continue
            if ch == '"':
                state = "code"
            out.append(ch)
        i += 1
    return "".join(out)


def matching_brace(text, open_idx):
    assert text[open_idx] == "{"
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def function_body(text, signature_re):
    """Body of the DEFINITION matching signature_re (a declaration ending in
    ';' before any '{' is skipped). `text` must be comment-stripped."""
    for m in re.finditer(signature_re, text):
        j = m.end()
        while j < len(text) and text[j] not in "{;":
            j += 1
        if j < len(text) and text[j] == "{":
            return text[j : matching_brace(text, j) + 1]
    raise AssertionError(f"missing definition: {signature_re}")


def case_types(body):
    return set(CASE_RE.findall(body))


header = strip_comments(HEADER.read_text())
backend = strip_comments(BACKEND.read_text())
cache = strip_comments(CACHE.read_text())
common = strip_comments(COMMON.read_text())

TABLE_SIG = r"inline\s+bool\s+" + PREDICATE + r"\s*\(\s*ggml_type\s+type\s*\)"
SUPPORTS_SOA_SIG = r"static\s+bool\s+ggml_sycl_layout_supports_soa\s*\(\s*ggml_type\s+type\s*\)"
DEVICE_REORDER_SIG = r"static\s+bool\s+reorder_aos_to_soa_device\s*\("
ROWS_REORDER_SIG = r"\bbool\s+reorder_rows_to_soa\s*\("
PARTIAL_ROWS_SIG = r"void\s*\*\s*unified_cache::load_partial_rows\s*\("
GET_OPTIMAL_SIG = r"static\s+layout_mode\s+get_optimal\s*\("
# Every per-type switch that selects an AOS->SOA reorder kernel or sizes its output.
FILL_SWITCH_SIGS = {
    "reorder_aos_to_soa_device": (r"static\s+bool\s+reorder_aos_to_soa_device\s*\(", "backend"),
    "reorder_rows_to_soa": (r"\bbool\s+reorder_rows_to_soa\s*\(", "backend"),
    "reorder_data_internal_": (r"static\s+bool\s+reorder_data_internal_\s*\(", "backend"),
    "ggml_sycl_reorder_weight_gpu": (r"static\s+sycl::event\s+ggml_sycl_reorder_weight_gpu\s*\(", "backend"),
    "ggml_sycl_reorder_weight_cpu": (r"static\s+bool\s+ggml_sycl_reorder_weight_cpu\s*\(", "backend"),
    "ggml_sycl_reorder_expected_size": (r"static\s+bool\s+ggml_sycl_reorder_expected_size\s*\(", "backend"),
}


def table():
    types = case_types(function_body(header, TABLE_SIG))
    assert types, "the SOA reorder table is empty"
    return types


def test_table_is_defined_once_and_headers_include_it():
    table()
    assert header.count(PREDICATE + "(") == 1, "the table must be defined exactly once in its header"
    assert "soa-reorder-types.hpp" in common, "common.hpp (layout_policy) must include the table header"


def test_every_fill_switch_matches_the_table():
    # Each of these is a fill the weight-load path runs for a SOA layout: a type in
    # the table without a case is a layout the fill cannot materialize, and a case
    # outside the table is a kernel nothing reaches.
    for name, (sig, _) in sorted(FILL_SWITCH_SIGS.items()):
        got = case_types(function_body(backend, sig))
        assert got == table(), f"{name} cases {sorted(got)} != table {sorted(table())}"


def test_runtime_clamp_reads_the_table():
    body = function_body(backend, SUPPORTS_SOA_SIG)
    assert PREDICATE + "(" in body, "ggml_sycl_layout_supports_soa must delegate to the table"
    assert not case_types(body), "ggml_sycl_layout_supports_soa must not keep a private type list"


def test_unified_reorder_entry_reads_the_table():
    # reorder_tensor_to_soa is the single entry every SOA conversion funnels through; its
    # "type supported" precheck used to be a switch over the five types.
    body = function_body(backend, r"\bbool\s+reorder_tensor_to_soa\s*\(")
    assert PREDICATE + "(" in body, "reorder_tensor_to_soa must gate on the table"
    assert not case_types(body), "reorder_tensor_to_soa must not keep a private type list"


def test_partial_row_fill_reads_the_table():
    body = function_body(cache, PARTIAL_ROWS_SIG)
    assert PREDICATE + "(" in body, "load_partial_rows must gate on the table"
    assert not case_types(body), "load_partial_rows must not keep a private type list"


def private_chains(text):
    """Statements that compare against every table type with == / != : a private copy
    of the table spelled as a chain. Switches are not flagged here (the fill switches
    are checked above; other switches name a different concept)."""
    hits = []
    for m in re.finditer(r"[^;{}]+", text):
        chunk = m.group(0)
        names = set(re.findall(r"(?:==|!=)\s*(GGML_TYPE_[A-Z0-9_]+)", chunk))
        if table() <= names:
            hits.append(" ".join(chunk.split())[:100])
    return hits


def test_no_private_copy_of_the_table_as_a_comparison_chain():
    for label, text in (("ggml-sycl.cpp", backend), ("unified-cache.cpp", cache), ("common.hpp", common)):
        hits = private_chains(text)
        assert not hits, f"{label} keeps a private copy of the SOA reorder table: {hits}"


def test_planner_layout_source_never_defaults_to_unmaterializable_soa():
    # get_optimal is the planner's layout source. Every SOA it returns must be either
    # (a) the table-guarded form `table(type) ? SOA : AOS`, or (b) a bare `return SOA` inside
    # a branch that has just compared the type against a table member (the Q4_0/Q8_0/MXFP4
    # carve-outs). The two fallthroughs (OUTPUT_WEIGHT quantized, final default) must be (a).
    body = function_body(common, GET_OPTIMAL_SIG)
    guarded = re.findall(
        r"return\s+" + PREDICATE + r"\(\s*qtype\s*\)\s*\?\s*GGML_LAYOUT_SOA\s*:\s*GGML_LAYOUT_AOS\s*;", body
    )
    assert len(guarded) == 2, f"expected the OUTPUT_WEIGHT and default arms to be table-guarded, found {len(guarded)}"
    for m in re.finditer(r"return\s+GGML_LAYOUT_SOA\s*;", body):
        window = body[body.rfind("if (", 0, m.start()) : m.start()]
        mentioned = set(re.findall(r"qtype\s*(?:==|!=)\s*(GGML_TYPE_[A-Z0-9_]+)", window))
        assert mentioned and mentioned <= table(), (
            f"bare `return GGML_LAYOUT_SOA` at offset {m.start()} is not preceded by a comparison against "
            f"a SOA-table type (saw {sorted(mentioned)})"
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
