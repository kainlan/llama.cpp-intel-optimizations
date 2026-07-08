#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
GETROWS = ROOT / "ggml" / "src" / "ggml-sycl" / "getrows.cpp"
SETROWS = ROOT / "ggml" / "src" / "ggml-sycl" / "set_rows.cpp"
MEMOPS = ROOT / "ggml" / "src" / "ggml-sycl" / "mem-ops.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_getrows_marker_and_slice_submits_are_named_and_bracketed() -> None:
    src = GETROWS.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    assert "sycl.get_rows.marker" in src
    assert "sycl.get_rows.slice" in src
    assert "role=get_rows;kind=marker" in src
    assert "role=get_rows;kind=stream_slice" in src

    helper = slice_between(src, "static sycl::event ggml_sycl_get_rows_profile_marker", "static const ggml_tensor * get_storage_tensor")
    assert "__builtin_FILE()" in helper
    assert "__builtin_LINE()" in helper
    assert "__builtin_FUNCTION()" in helper
    assert "ggml_sycl_profile_submit(" in helper
    assert "queue, label" in helper
    assert "file," in helper and "line, function" in helper
    assert "ggml_sycl_submit_marker<MarkerKernel>" in helper
    assert "ggml_sycl_profile_record_returned_event" not in helper
    assert ".wait(" not in helper


def test_getrows_documents_no_returned_event_for_real_marker_submits() -> None:
    src = GETROWS.read_text(encoding="utf-8")
    helper = slice_between(src, "static sycl::event ggml_sycl_get_rows_profile_marker", "static const ggml_tensor * get_storage_tensor")
    assert "GET_ROWS markers submit real marker/barrier commands" in helper
    assert "record_returned_event" in helper
    assert "ggml_sycl_profile_record_returned_event" not in src


def test_set_rows_submits_are_named_and_forward_callsite() -> None:
    src = SETROWS.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    for label in ["sycl.set_rows.generic", "sycl.set_rows.fp8", "sycl.set_rows.quantized"]:
        assert label in src
    assert "role=set_rows;kind=generic" in src
    assert "role=set_rows;kind=fp8" in src
    assert "role=set_rows;kind=quantized" in src

    helper = slice_between(src, "static sycl::event ggml_sycl_set_rows_profile_submit", "static int ggml_sycl_set_rows_ptr_device")
    assert "__builtin_FILE()" in helper
    assert "__builtin_LINE()" in helper
    assert "__builtin_FUNCTION()" in helper
    assert "ggml_sycl_profile_submit(queue" in helper
    assert "file, line, function" in helper
    assert "ggml_sycl_profile_record_returned_event" not in helper
    assert ".wait(" not in helper

    generic_body = slice_between(src, "static sycl::event set_rows_sycl(const char *  src0_d", "// FP8 E4M3 specific kernel")
    assert "ggml_sycl_set_rows_profile_submit" in generic_body
    assert "profiled_queue.parallel_for" in generic_body

    fp8_body = slice_between(src, "static sycl::event set_rows_sycl_fp8", "template <typename TIn, typename TIdx>")
    assert "ggml_sycl_set_rows_profile_submit" in fp8_body
    assert "profiled_queue.parallel_for" in fp8_body


def test_mem_ops_copy_submits_are_named_bracketed_and_preserve_deps() -> None:
    src = MEMOPS.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    for label in ["sycl.memcpy.mem_ops", "sycl.memcpy.cross_device"]:
        assert label in src
    assert "role=memcpy;path=mem_ops" in src
    assert "role=memcpy;path=cross_device" in src

    helper = slice_between(src, "static sycl::event ggml_sycl_memcpy_profile_submit", "static bool host_ptr_is_usm_accessible")
    assert "__builtin_FILE()" in helper
    assert "__builtin_LINE()" in helper
    assert "__builtin_FUNCTION()" in helper
    assert "ggml_sycl_profile_submit(queue" in helper
    assert "file, line, function" in helper
    assert "queue_kind" in helper
    assert "ggml_sycl_profile_record_returned_event" not in helper
    assert ".wait(" not in helper

    submit_body = slice_between(src, "static sycl::event mem_copy_direct_submit", "static sycl::event mem_copy_submit")
    assert "ggml_sycl_memcpy_profile_submit" in submit_body
    assert "add_deps(cgh, deps)" in submit_body
    assert "cgh.memcpy(dst_ptr, src_ptr, size)" in submit_body
    assert "queue.submit" not in submit_body
    assert "ggml_sycl_profile_record_returned_event" not in submit_body


def test_mem_ops_fill_submit_is_named_bracketed_and_preserves_deps() -> None:
    src = MEMOPS.read_text(encoding="utf-8")
    body = slice_between(src, "static sycl::event mem_fill_direct_submit", "static sycl::event mem_fill_submit")
    assert "sycl.memcpy.mem_fill" in body
    assert "role=memfill;path=mem_ops" in body
    assert "__builtin_FILE()" in body
    assert "__builtin_LINE()" in body
    assert "__builtin_FUNCTION()" in body
    assert "file, line, function" in body
    assert "ggml_sycl_memcpy_profile_submit" in body
    assert "add_deps(cgh, deps)" in body
    assert "cgh.memset(ptr, value, size)" in body
    assert "queue.submit" not in body
    assert "ggml_sycl_profile_record_returned_event" not in body
