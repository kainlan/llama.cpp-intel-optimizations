#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
NORM = ROOT / "ggml" / "src" / "ggml-sycl" / "norm.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_norm_cpp_includes_common_hpp_for_profiler_access() -> None:
    # ggml_sycl_profile_submit/ggml_sycl_profile_label come from
    # sycl-kernel-profiler.hpp, reached transitively through common.hpp --
    # norm.cpp does not include the profiler header directly.
    norm = NORM.read_text(encoding="utf-8")
    assert '"ggml-sycl/common.hpp"' in norm


def test_rms_norm_f32_sycl_submits_are_wrapped_and_wait_free() -> None:
    # llama.cpp-qmen (S6/I1 profiler completeness): the load-bearing wrap --
    # RMS_NORM was raw-submit and entirely dark to the kernel profiler
    # (~300 launches/decode token per docs/plans/2026-09-01-sycl-utilization-
    # plan.md's decode census).
    norm = NORM.read_text(encoding="utf-8")
    body = slice_between(
        norm,
        "static void rms_norm_f32_sycl(const float * x,",
        "// Maximum ncols for SLM caching",
    )
    assert body.count('"norm.rms_norm"') >= 1
    assert body.count("ggml_sycl_profile_submit(*stream, rms_norm_label") == 2
    assert body.count("stream->submit(") == 2
    assert "ncols=" in body
    assert "nrows=" in body
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body


def test_rms_norm_mul_f32_sycl_submits_are_wrapped_and_preserve_evt() -> None:
    norm = NORM.read_text(encoding="utf-8")
    body = slice_between(
        norm,
        "static void rms_norm_mul_f32_sycl(const float * x,",
        "// Fused RMS norm + multiply + add SYCL host function",
    )
    assert body.count('"norm.rms_norm_mul"') >= 1
    assert body.count("evt = ggml_sycl_profile_submit(*stream, rms_norm_mul_label") == 3
    # The wait-after-mul opt-in still reads the same `evt` the wrapped
    # submit now produces -- confirms the wrap didn't drop the barrier chain.
    assert "ggml_sycl_wait_after_rms_norm_mul()" in body
    assert "stream->ext_oneapi_submit_barrier({ evt })" in body
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body


def test_rms_norm_mul_add_f32_sycl_submits_are_wrapped() -> None:
    norm = NORM.read_text(encoding="utf-8")
    body = slice_between(
        norm,
        "static void rms_norm_mul_add_f32_sycl(const float * x,",
        "// Fused ADD + RMS norm SYCL host function",
    )
    assert body.count('"norm.rms_norm_mul_add"') >= 1
    assert body.count("ggml_sycl_profile_submit(*stream, rms_norm_mul_add_label") == 3
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body


def test_add_rms_norm_f32_sycl_submits_are_wrapped() -> None:
    norm = NORM.read_text(encoding="utf-8")
    body = slice_between(
        norm,
        "static void add_rms_norm_f32_sycl(const float * x,",
        "static void l2_norm_f32_sycl(const float * x,",
    )
    assert body.count('"norm.add_rms_norm"') >= 1
    assert body.count("ggml_sycl_profile_submit(*stream, add_rms_norm_label") == 3
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body


def test_plain_norm_and_group_norm_submits_are_also_wrapped() -> None:
    # The task named RMS_NORM as the primary target and NORM/GROUP_NORM as
    # "trivially adjacent" -- both share the exact raw-submit pattern this
    # closes.
    norm = NORM.read_text(encoding="utf-8")
    norm_body = slice_between(
        norm,
        "static void norm_f32_sycl(const float * x,",
        "static void group_norm_f32_sycl(const float * x,",
    )
    assert norm_body.count('"norm.norm"') >= 1
    assert norm_body.count("ggml_sycl_profile_submit(*stream, norm_label") == 2

    group_body = slice_between(
        norm,
        "static void group_norm_f32_sycl(const float * x,",
        "static void rms_norm_f32_sycl(const float * x,",
    )
    assert group_body.count('"norm.group_norm"') >= 1
    assert group_body.count("ggml_sycl_profile_submit(*stream, group_norm_label") == 2


def test_metadata_construction_is_gated_for_every_wrapped_norm_function() -> None:
    # Each of the 6 wrapped functions' metadata std::string must be built ONLY under
    # ggml_sycl_kernel_profile_enabled() -- building it unconditionally is
    # a heap allocation on every launch even with the profiler off
    # (~300/decode token across this family), which contradicts
    # docs/backend/sycl-env-vars.md's "zero overhead when unset" claim.
    norm = NORM.read_text(encoding="utf-8")
    for label_var, metadata_var in [
        ("norm_label", "norm_metadata"),
        ("group_norm_label", "group_norm_metadata"),
        ("rms_norm_label", "rms_norm_metadata"),
        ("rms_norm_mul_label", "rms_norm_mul_metadata"),
        ("rms_norm_mul_add_label", "rms_norm_mul_add_metadata"),
        ("add_rms_norm_label", "add_rms_norm_metadata"),
    ]:
        decl = f"std::string {metadata_var};"
        assert decl in norm, decl
        gate = norm.index("if (ggml_sycl_kernel_profile_enabled()) {", norm.index(decl))
        # Match the gate's own closing brace by depth, not by indentation, so
        # a reformat cannot move the slice boundary.
        brace_open = norm.index("{", gate)
        depth = 0
        i = brace_open
        while True:
            if norm[i] == "{":
                depth += 1
            elif norm[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        gated = norm[brace_open:i]
        assert re.search(rf"{metadata_var}\s*=", gated), metadata_var
        assert re.search(rf"{label_var}\.metadata\s*=\s*{metadata_var}\.c_str\(\);", gated), label_var


def test_l2_norm_is_intentionally_out_of_scope_and_left_raw() -> None:
    # Not named by the task; confirms the scoping decision rather than an
    # oversight -- l2_norm_f32_sycl's two raw submits are untouched.
    norm = NORM.read_text(encoding="utf-8")
    body = slice_between(
        norm,
        "static void l2_norm_f32_sycl(const float * x,",
        "void ggml_sycl_op_norm(ggml_backend_sycl_context & ctx",
    )
    assert body.count("stream->submit(") == 2
    assert "ggml_sycl_profile_submit(" not in body
