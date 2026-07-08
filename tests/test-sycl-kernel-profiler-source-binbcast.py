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
        "template <float (*bin_op)(const float, const float)>",
    )


def binbcast_kernel_submit_helper() -> str:
    src = BINBCAST.read_text(encoding="utf-8")
    return slice_between(
        src,
        "static sycl::event ggml_sycl_submit_binbcast_kernel",
        "static inline const char * ggml_sycl_layout_mode_name",
    )


def binbcast_kernel_launcher() -> str:
    src = BINBCAST.read_text(encoding="utf-8")
    return slice_between(
        src,
        "template <float (*bin_op)(const float, const float)> struct bin_bcast_sycl",
        "template <class op>",
    )


def binbcast_op_helper() -> str:
    src = BINBCAST.read_text(encoding="utf-8")
    return slice_between(src, "template <class op>", "inline void ggml_sycl_op_add")


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
    assert "ggml_sycl_profile_record_returned_event" not in body
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


def test_binbcast_barrier_event_forwards_callsite_through_profile_submit() -> None:
    body = binbcast_event_helper()
    barrier_index = body.index("ext_oneapi_submit_barrier")
    wrapper_index = body.rindex("ggml_sycl_profile_submit", 0, barrier_index)
    assert wrapper_index < barrier_index
    assert "profiled_queue.ext_oneapi_submit_barrier()" in body
    assert "file, line, function" in body
    assert "ggml_sycl_profile_record_returned_event" not in body


def test_binbcast_kernel_submit_helper_is_named_and_forwards_callsite() -> None:
    body = binbcast_kernel_submit_helper()
    assert "ggml_sycl_profile_label" in body
    assert 'label.category   = "binbcast"' in body
    assert 'label.queue_kind = "compute"' in body
    assert "ggml_sycl_profile_submit(q" in body
    assert "ggml_sycl_kernel_profile_enabled() ? ggml_sycl_get_device_id_from_queue(q) : -1" in body
    assert "__builtin_FILE()" in body
    assert "__builtin_LINE()" in body
    assert "__builtin_FUNCTION()" in body
    assert "file, line, function" in body
    assert ".wait(" not in body


def test_generic_binbcast_parallel_for_submits_are_named_profiled_and_wait_free() -> None:
    src = BINBCAST.read_text(encoding="utf-8")
    body = binbcast_kernel_launcher()
    assert "ggml_sycl_binbcast_kernel_profile_name<bin_op>()" in body
    assert "ggml_sycl_binbcast_kernel_profile_metadata<bin_op>" in body
    assert "ggml_sycl_binbcast_kernel_variant::UNRAVEL" in body
    assert "ggml_sycl_binbcast_kernel_variant::ND" in body
    assert body.count("ggml_sycl_submit_binbcast_kernel(\n") >= 2
    assert "sycl.binbcast.add" in src
    assert "sycl.binbcast.mul" in src
    assert "role=binbcast;mode=kernel;op=add;variant=nd" in src
    assert "role=binbcast;mode=kernel;op=mul;variant=nd" in src
    assert "profiled_queue.parallel_for" in body
    assert "stream->parallel_for(" not in body
    assert ".wait(" not in body


def test_generic_binbcast_submit_forwards_add_mul_callsite() -> None:
    launcher = binbcast_kernel_launcher()
    helper = binbcast_op_helper()
    for body in (launcher, helper):
        assert "__builtin_FILE()" in body
        assert "__builtin_LINE()" in body
        assert "__builtin_FUNCTION()" in body
        assert "file, line, function" in " ".join(body.split())


def test_binbcast_file_has_no_unprofiled_stream_parallel_for_submits() -> None:
    src = BINBCAST.read_text(encoding="utf-8")
    assert "stream->parallel_for(" not in src
    assert "sycl.binbcast.add1" in src
    assert "sycl.binbcast.mul_add_fused" in src
