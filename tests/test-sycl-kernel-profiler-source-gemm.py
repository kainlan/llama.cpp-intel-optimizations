#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEMM = ROOT / "ggml" / "src" / "ggml-sycl" / "gemm.hpp"
GGML_SYCL = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_gemm_hpp_includes_profiler_header() -> None:
    gemm = GEMM.read_text(encoding="utf-8")
    assert '"sycl-kernel-profiler.hpp"' in gemm


def test_unbatched_gemm_execute_sites_are_wrapped_and_wait_free() -> None:
    # DnnlGemmWrapper::gemm() (the un-batched primitive, "variant=0") is the
    # oneDNN WOQ Q8_0/Q4_0 MUL_MAT choke point reached via row_gemm() from
    # ggml_sycl_op_mul_mat_sycl -- llama.cpp-qmen (S6/I1), the profiler
    # completeness task that closed kprof-pp-b70.csv's 219/237 dark GEMMs.
    gemm = GEMM.read_text(encoding="utf-8")
    body = slice_between(
        gemm,
        "static sycl::event gemm(ggml_backend_sycl_context & ctx,",
        "static sycl::event row_gemm(ggml_backend_sycl_context & ctx,",
    )
    assert body.count('"mulmat.onednn_woq.execute"') == 2
    assert body.count("ggml_sycl_profile_submit(*q,") == 2
    assert body.count("dnnl::sycl_interop::execute(") == 2
    assert "variant=fallback_create" in body
    assert "variant=cached" in body
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body
    # gemm() must return the primitive's event, not discard it (void before
    # this task -- neither execute call site had a profiler wrap at all).
    assert "static sycl::event gemm(ggml_backend_sycl_context & ctx," in gemm


def test_gemm_accepts_deps_and_optional_op_context_with_safe_defaults() -> None:
    gemm = GEMM.read_text(encoding="utf-8")
    sig = slice_between(
        gemm,
        "static sycl::event gemm(ggml_backend_sycl_context & ctx,",
        "std::lock_guard<std::mutex> lock(exec_mutex(q));",
    )
    assert "const std::vector<sycl::event> & deps = {}" in sig
    assert "const char *                op_context = nullptr" in sig


def test_row_gemm_forwards_op_context_to_gemm() -> None:
    gemm = GEMM.read_text(encoding="utf-8")
    body = slice_between(
        gemm,
        "static sycl::event row_gemm(ggml_backend_sycl_context & ctx,",
        "// WoQ GEMM for Q4_0 weights",
    )
    assert "const char *                op_context = nullptr" in body
    assert "op_context);" in body
    assert "return gemm(" in body


def test_ggml_sycl_cpp_passes_src0_type_name_at_the_q8_0_row_gemm_call_sites() -> None:
    ggml_sycl = GGML_SYCL.read_text(encoding="utf-8")
    assert ggml_sycl.count("DnnlGemmWrapper::row_gemm(") >= 2
    assert ggml_sycl.count("ggml_type_name(src0->type));") >= 2
