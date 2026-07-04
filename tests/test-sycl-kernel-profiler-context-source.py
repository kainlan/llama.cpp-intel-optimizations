from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml/src/ggml-sycl/sycl-kernel-profiler.cpp"


def test_profiler_node_context_api_is_declared() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "struct ggml_sycl_kernel_profile_node_context" in header
    assert "class ggml_sycl_kernel_profile_node_scope" in header
    assert "ggml_sycl_kernel_profile_set_node_context" in header
    assert "ggml_sycl_kernel_profile_clear_node_context" in header


def test_timeline_metadata_includes_node_context_without_touching_label_key() -> None:
    cpp = CPP.read_text(encoding="utf-8")
    assert "thread_local ggml_sycl_kernel_profile_node_context" in cpp
    assert "node_step=" in cpp
    assert "node_idx=" in cpp
    assert "node_op=" in cpp
    assert "node_tensor=" in cpp
    assert "node_count=" in cpp
    assert "raw_event.node_context" in cpp
