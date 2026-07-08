#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
FATTN = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_submit_wrapper_api_is_default_off_and_wait_free() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "template <typename SubmitFn>" in header
    assert "ggml_sycl_profile_submit_impl" in header
    body = slice_between(header, "ggml_sycl_profile_submit_impl", "template <typename Fn>")
    assert "submit_fn(q)" in body
    assert "ggml_sycl_kernel_profile_enabled()" in body
    assert ".wait(" not in body
    assert "ggml_sycl_kernel_profile_record_event" in body


def test_profile_submit_captures_callsite_without_macro_and_stays_wait_free() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "ggml_sycl_profile_submit_impl" in header
    assert "#define ggml_sycl_profile_submit" not in header

    impl = slice_between(header, "ggml_sycl_profile_submit_impl", "// Test-only helper")
    assert "submit_fn(q)" in impl
    assert "ggml_sycl_kernel_profile_record_event" in impl
    assert ".wait(" not in impl
    assert "__builtin_FILE()" in impl
    assert "__builtin_LINE()" in impl
    assert "__builtin_FUNCTION()" in impl
    assert "ggml_sycl::sycl_timeline_callsite{ file, line, function }" in impl


def test_profiler_env_names_are_stable() -> None:
    cpp = CPP.read_text(encoding="utf-8")
    for name in [
        "GGML_SYCL_KERNEL_PROFILE",
        "GGML_SYCL_KERNEL_PROFILE_OUTPUT",
        "GGML_SYCL_KERNEL_PROFILE_FORMAT",
        "GGML_SYCL_KERNEL_PROFILE_TOP_N",
        "GGML_SYCL_KERNEL_PROFILE_RAW",
        "GGML_SYCL_KERNEL_PROFILE_FLUSH",
    ]:
        assert name in cpp


def test_sycl_docs_describe_named_kernel_profiler_contract() -> None:
    doc = (ROOT / "docs" / "backend" / "SYCL.md").read_text(encoding="utf-8")
    assert "GGML_SYCL_KERNEL_PROFILE" in doc
    assert "GGML_SYCL_KERNEL_PROFILE_OUTPUT" in doc
    assert "GGML_SYCL_KERNEL_PROFILE_FLUSH" in doc
    assert "SYCL event profiling timestamps" in doc
    assert "VTune computing-task attribution is not the source of truth" in doc
