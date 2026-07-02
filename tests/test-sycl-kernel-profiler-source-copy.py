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


def test_graph_safe_memcpy_is_profiled_without_changing_graph_return_contract() -> None:
    common = COMMON.read_text(encoding="utf-8")
    body = slice_between(common, "inline sycl::event ggml_sycl_graph_safe_memcpy", "inline bool ggml_sycl_graph_recording_active")
    assert "sycl-kernel-profiler.hpp" in common
    assert "sycl.memcpy.graph_safe" in body
    assert "ggml_sycl_profile_record_returned_event" in body
    assert "return sycl::event{};" in body
    assert "ggml_sycl::mem_copy_async" in body


def test_profile_flush_points_are_explicit_not_atexit() -> None:
    backend = (ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp").read_text(encoding="utf-8")
    free_body = slice_between(backend, "static void ggml_backend_sycl_free", "static void ggml_backend_sycl_set_tensor_async")
    assert "ggml_sycl_kernel_profile_flush(true, \"backend-free\")" in free_body
    assert free_body.index("ggml_sycl_kernel_profile_flush(true, \"backend-free\")") < free_body.index("delete sycl_ctx;")
    profiler = CPP.read_text(encoding="utf-8")
    assert "std::atexit" not in profiler
    bench = (ROOT / "tools" / "sycl-kernel-bench" / "main.cpp").read_text(encoding="utf-8")
    assert "ggml_sycl_kernel_profile_flush(true, \"sycl-kernel-bench\")" in bench
