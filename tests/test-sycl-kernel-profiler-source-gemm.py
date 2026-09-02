#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re

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
    # "static sycl::event gemm(" is distinct from "static sycl::event row_gemm("
    # and contains no variable whitespace, so the literal anchors the slice.
    body = slice_between(gemm, "static sycl::event gemm(", "static sycl::event row_gemm(")
    assert body.count('"mulmat.onednn_gemm.unbatched"') == 2
    assert body.count("ggml_sycl_profile_submit(*q,") == 2
    assert body.count("dnnl::sycl_interop::execute(") == 2
    assert "variant=fallback_create" in body
    assert "variant=cached" in body
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body


def test_gemm_metadata_construction_is_gated_and_wraps_return_events() -> None:
    # Spec-review fix round, finding 6: the metadata std::string must be
    # built ONLY under ggml_sycl_kernel_profile_enabled() -- building it
    # unconditionally is a heap allocation on every call even with the
    # profiler off, which contradicts docs/backend/sycl-env-vars.md's
    # "zero overhead when unset" claim.
    gemm = GEMM.read_text(encoding="utf-8")
    body = slice_between(
        gemm,
        "static sycl::event gemm(",
        "static sycl::event row_gemm(",
    )
    assert body.count("std::string label_metadata;") == 2
    assert body.count("if (ggml_sycl_kernel_profile_enabled()) {") == 2
    # The metadata assignment and the .metadata pointer-store must both live
    # INSIDE the gate, not just the declaration -- find each gated block and
    # confirm both lines are between its opening and closing brace.
    for gate_start in [m.start() for m in re.finditer(r"if \(ggml_sycl_kernel_profile_enabled\(\)\) \{", body)]:
        depth = 0
        brace_open = body.index("{", gate_start)
        i = brace_open
        while True:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        gated = body[brace_open:i]
        assert re.search(r"label_metadata\s*=", gated)
        assert re.search(r"gemm_label\.metadata\s*=\s*label_metadata\.c_str\(\);", gated)
    # gemm() must return the primitive's event, not discard it (void before
    # this task -- neither execute call site had a profiler wrap at all).
    assert re.search(r"static sycl::event gemm\(ggml_backend_sycl_context", gemm)


def test_gemm_accepts_deps_and_optional_op_context_with_safe_defaults() -> None:
    gemm = GEMM.read_text(encoding="utf-8")
    sig = slice_between(
        gemm,
        "static sycl::event gemm(",
        "std::lock_guard<std::mutex> lock(exec_mutex(q));",
    )
    # Whitespace-tolerant: clang-format is free to realign the parameter
    # column, so anchor on the token content, not the spacing.
    assert re.search(r"const std::vector<sycl::event>\s*&\s*deps\s*=\s*\{\},", sig)
    assert re.search(r"const char\s*\*\s*op_context\s*=\s*nullptr\)\s*\{", sig)


def test_row_gemm_forwards_op_context_to_gemm() -> None:
    gemm = GEMM.read_text(encoding="utf-8")
    body = slice_between(
        gemm,
        "static sycl::event row_gemm(",
        "// WoQ GEMM for Q4_0 weights",
    )
    assert re.search(r"const char\s*\*\s*op_context\s*=\s*nullptr\)\s*\{", body)
    assert "op_context);" in body
    assert "return gemm(" in body


def test_ggml_sycl_cpp_passes_src0_type_name_at_the_row_gemm_q8_0_call_sites() -> None:
    # Anchored to row_gemm's own call sites (not a bare substring match
    # anywhere in the 60k-line file) -- spec review flagged the previous
    # version of this test as an unanchored `ggml_type_name(src0->type));`
    # count that could pass without the arg actually reaching a row_gemm
    # call. DnnlGemmWrapper::gemm()/row_gemm() have 9 call sites total
    # (outprod.cpp:1, ggml-sycl.cpp: 2 row_gemm here + 3 gemm in the F16
    # batch launcher + 4 more row_gemm elsewhere) -- only the two Q8_0
    # dequant-then-GEMM sites in ggml_sycl_op_mul_mat_sycl pass op_context;
    # the other 7 are unaffected (default nullptr) and untouched by this
    # task.
    ggml_sycl = GGML_SYCL.read_text(encoding="utf-8")
    row_gemm_calls = [m.start() for m in re.finditer(r"DnnlGemmWrapper::row_gemm\(", ggml_sycl)]
    assert len(row_gemm_calls) >= 2
    type_name_calls = 0
    for call_start in row_gemm_calls:
        call_end = ggml_sycl.index(";", call_start)
        if "ggml_type_name(src0->type)" in ggml_sycl[call_start:call_end]:
            type_name_calls += 1
    assert type_name_calls == 2
