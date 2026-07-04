from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROPE = ROOT / "ggml/src/ggml-sycl/rope.cpp"
SOFTMAX = ROOT / "ggml/src/ggml-sycl/softmax.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_rope_submit_has_named_profile_label() -> None:
    src = ROPE.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    assert "sycl.rope" in src
    assert "role=rope" in src
    assert "ggml_sycl_profile_submit" in src
    rope_body = src[src.index("inline void ggml_sycl_op_rope") : src.index("void ggml_sycl_rope", src.index("inline void ggml_sycl_op_rope"))]
    assert ".wait(" not in rope_body


def test_rope_profile_helpers_forward_callsite() -> None:
    src = ROPE.read_text(encoding="utf-8")
    launch_body = slice_between(src, "static sycl::event rope_norm_sycl", "inline void ggml_sycl_op_rope")
    assert "__builtin_FILE()" in launch_body
    assert "__builtin_LINE()" in launch_body
    assert "__builtin_FUNCTION()" in launch_body
    assert "file, line, function" in launch_body


def test_softmax_submits_have_named_profile_labels() -> None:
    src = SOFTMAX.read_text(encoding="utf-8")
    assert '#include "sycl-kernel-profiler.hpp"' in src
    assert "sycl.softmax.forward" in src
    assert "sycl.softmax.backward" in src
    assert "role=softmax;direction=forward" in src
    assert "role=softmax;direction=backward" in src
    assert "ggml_sycl_profile_submit" in src


def test_softmax_profile_helpers_forward_callsite() -> None:
    src = SOFTMAX.read_text(encoding="utf-8")
    forward_body = slice_between(src, "static sycl::event launch_soft_max_kernels", "template <typename T>\nstatic sycl::event soft_max_f32_sycl")
    helper_body = slice_between(src, "template <typename T>\nstatic sycl::event soft_max_f32_sycl", "void ggml_sycl_op_soft_max")
    assert "__builtin_FILE()" in forward_body
    assert "__builtin_LINE()" in forward_body
    assert "__builtin_FUNCTION()" in forward_body
    assert "file, line, function" in forward_body
    assert "__builtin_FILE()" in helper_body
    assert "__builtin_LINE()" in helper_body
    assert "__builtin_FUNCTION()" in helper_body
    assert "file, line, function" in helper_body
