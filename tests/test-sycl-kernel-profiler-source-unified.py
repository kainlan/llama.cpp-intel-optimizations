#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "ggml" / "src" / "ggml-sycl" / "unified-kernel.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def micro_submit_body(src: str, name: str, next_name: str) -> str:
    return slice_between(src, f"static void {name}", f"static void {next_name}")


def assert_micro_submit_profiled(src: str, name: str, next_name: str, label: str, op: str) -> None:
    body = micro_submit_body(src, name, next_name)
    assert label in body
    assert "ggml_sycl_profile_label" in body
    assert "ggml_sycl_profile_submit" in body
    assert "profiled_queue.submit" in body
    assert body.count(".submit(") == body.count("profiled_queue.submit(")
    assert "role=unified_micro" in body
    assert "micro_submit=1" in body
    assert "graph_recorded=1" in body
    assert f"op={op}" in body
    assert "__builtin_FILE()" in body
    assert "__builtin_LINE()" in body
    assert "__builtin_FUNCTION()" in body
    assert "file, line, function" in body
    assert ".wait(" not in body


def test_unified_micro_submits_are_named_profiled() -> None:
    src = UNIFIED.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    assert "static ggml_sycl_profile_label unified_micro_profile_label" in src
    assert_micro_submit_profiled(src, "micro_submit_mul", "micro_submit_rms_norm", "sycl.unified.mul", "MUL")
    assert_micro_submit_profiled(src, "micro_submit_rope", "micro_submit_strided_copy", "sycl.unified.rope", "ROPE")
    assert_micro_submit_profiled(
        src,
        "micro_submit_softmax",
        "micro_submit_set_rows",
        "sycl.unified.softmax",
        "SOFT_MAX",
    )
    assert_micro_submit_profiled(
        src,
        "micro_submit_set_rows",
        "micro_submit_attention",
        "sycl.unified.set_rows",
        "SET_ROWS",
    )


def test_unified_matmul_profile_labels_remain_unchanged() -> None:
    src = UNIFIED.read_text(encoding="utf-8")
    for label in [
        "unified.matmul.xmx",
        "unified.matmul.esimd_int8",
        "unified.matmul.esimd_cooperative",
        "unified.matmul.esimd_fp16_double_buffered",
        "unified.matmul.esimd_fp16",
        "unified.matmul.dmmv",
        "unified.matmul.scalar",
    ]:
        assert label in src
