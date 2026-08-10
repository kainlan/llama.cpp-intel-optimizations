from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml/src/ggml-sycl/sycl-kernel-profiler.cpp"
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


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


def test_graph_loop_wraps_compute_forward_in_profiler_node_scope() -> None:
    src = GGML_SYCL.read_text(encoding="utf-8")
    # Two dispatch sites now carry this timeline scope on one line: the main graph
    # loop, and the MUL_MAT_ID recording-pause site above it. Both have had the
    # scope since efd840363; c3bfd71c4 only reflowed the recording-pause copy from
    # a wrapped call onto a single line, which moved this marker's FIRST match to a
    # site that never carried a profiler node scope. Anchoring on `index()` alone
    # therefore reported a property loss that never happened (llama.cpp-pp72), so
    # select by the property instead of by position.
    starts = [m.start() for m in re.finditer(re.escape('node_timeline_scope.emplace("ggml.op", "compute_forward_node"'), src)]
    assert starts, "no compute_forward_node timeline scope found"
    windows = [src[start : start + 1200] for start in starts]
    scoped = [w for w in windows if "ggml_sycl_kernel_profile_node_scope" in w]
    assert scoped, "no compute_forward_node dispatch is wrapped in a profiler node scope"
    for window in scoped:
        assert "ggml_sycl::sycl_timeline_current_graph_compute_step()" in window
        assert "ggml_op_name(node->op)" in window
        assert "node->name" in window
        assert window.index("ggml_sycl_kernel_profile_node_scope") < window.index(
            "ggml_sycl_compute_forward(*sycl_ctx, node)"
        )
