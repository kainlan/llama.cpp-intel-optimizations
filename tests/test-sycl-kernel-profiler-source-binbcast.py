#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINBCAST = ROOT / "ggml" / "src" / "ggml-sycl" / "binbcast.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def binbcast_event_helper() -> str:
    src = BINBCAST.read_text(encoding="utf-8")
    return slice_between(
        src,
        "static sycl::event ggml_sycl_submit_binbcast_event",
        "static inline const char * ggml_sycl_layout_mode_name",
    )


def test_binbcast_includes_kernel_profiler_header() -> None:
    src = BINBCAST.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src


def test_binbcast_event_submit_is_named_profiled_and_wait_free() -> None:
    body = binbcast_event_helper()
    assert "ggml_sycl_profile_label" in body
    assert "sycl.binbcast.event" in body
    assert 'label.category = "binbcast"' in body or 'label.category   = "binbcast"' in body
    assert 'label.queue_kind = "compute"' in body or 'label.queue_kind = "compute"' in body
    assert "role=binbcast" in body
    assert "mode=" in body
    assert "ggml_sycl_profile_submit" in body
    assert "ggml_sycl_profile_record_returned_event" in body
    assert ".wait(" not in body


def test_binbcast_event_helper_forwards_caller_source_location() -> None:
    body = binbcast_event_helper()
    assert "__builtin_FILE()" in body
    assert "__builtin_LINE()" in body
    assert "__builtin_FUNCTION()" in body
    assert "const char *" in body and "file" in body
    assert "int" in body and "line" in body
    assert "function" in body
    assert "file, line, function" in body


def test_binbcast_real_queue_submit_uses_profile_submit_not_returned_event_only() -> None:
    body = binbcast_event_helper()
    submit_index = body.index(".submit(")
    wrapper_index = body.rindex("ggml_sycl_profile_submit", 0, submit_index)
    assert wrapper_index < submit_index
    assert "profiled_queue.submit" in body
    assert "ggml_sycl_profile_record_returned_event(label, q.submit" not in body
    assert "ggml_sycl_profile_record_returned_event(label, profiled_queue.submit" not in body
