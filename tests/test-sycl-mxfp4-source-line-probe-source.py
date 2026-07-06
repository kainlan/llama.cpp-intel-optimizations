#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tools" / "sycl-kernel-bench" / "CMakeLists.txt"
SOURCE = ROOT / "tools" / "sycl-kernel-bench" / "mxfp4_source_line_probe.cpp"
INLINE_DOT = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"


def test_mxfp4_source_line_probe_target_is_narrow() -> None:
    text = CMAKE.read_text(encoding="utf-8")
    assert "sycl-mxfp4-source-line-probe" in text
    normal_block = text[
        text.index("add_executable(${TARGET}") : text.index("if (GGML_SYCL)\n    add_executable(sycl-mxfp4-source-line-probe")
    ]
    assert "kernels/reference/mxfp4_inline_dot.cpp" in normal_block
    assert "SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY" not in normal_block

    block = text[text.index("add_executable(sycl-mxfp4-source-line-probe") : text.index("if (LLAMA_TOOLS_INSTALL)")]
    assert "mxfp4_source_line_probe.cpp" in block
    assert "kernels/reference/mxfp4_inline_dot.cpp" in block
    assert "SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY=1" in block
    assert "kernels/reference/onednn_fp16_gemm.cpp" not in block
    assert "kernels/tier1/mmvq_aos_baseline.cpp" not in block
    assert "-fsycl-device-code-split=per_kernel" in block
    assert "-ffunction-sections" in block
    assert "-Wl,--gc-sections" in block
    assert "GGML_SYCL_PROFILING_DEBUG" in block


def test_mxfp4_source_line_probe_runs_pair_glu_directly() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "run_mxfp4_pair_glu" in text
    assert "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias" in text
    assert "generate_quantized_weights(GGML_TYPE_MXFP4, GGML_LAYOUT_SOA" in text
    assert "generate_activations(token_rows" in text
    assert "sycl::gpu_selector_v" in text
    assert "benchmark_harness.hpp" not in text
    assert "run_single_benchmark" not in text


def test_mxfp4_inline_dot_probe_guard_keeps_only_pair_glu_path() -> None:
    text = INLINE_DOT.read_text(encoding="utf-8")
    guard_open = "#ifndef SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY"
    guard_close = "#endif  // !SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY"
    assert text.count(guard_open) == 2
    assert text.count(guard_close) == 2

    first_guard = text.index(guard_open)
    first_end = text.index(guard_close, first_guard)
    pair_glu = text.index("bool run_mxfp4_pair_glu")
    second_guard = text.index(guard_open, first_end)
    second_end = text.index(guard_close, second_guard)
    namespace_close = text.index("}  // namespace sycl_bench", second_end)

    for helper in (
        "static size_t sparse_expert_slot",
        "static bool make_mxfp4_soa_scale_stride_layout",
        "static bool make_mxfp4_xmx_tiled_layout",
        "static bool select_mxfp4_xmx_tiles_n",
    ):
        assert text.index(helper) < first_guard

    assert first_guard < text.index("static bool validate_inline_dot") < first_end
    assert first_end < pair_glu < second_guard
    assert second_guard < text.index("bool run_mxfp4_layer_glu_down") < second_end
    assert second_end < namespace_close
