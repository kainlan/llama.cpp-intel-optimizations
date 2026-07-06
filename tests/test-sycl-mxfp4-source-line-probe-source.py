#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tools" / "sycl-kernel-bench" / "CMakeLists.txt"
SOURCE = ROOT / "tools" / "sycl-kernel-bench" / "mxfp4_source_line_probe.cpp"


def test_mxfp4_source_line_probe_target_is_narrow() -> None:
    text = CMAKE.read_text(encoding="utf-8")
    assert "sycl-mxfp4-source-line-probe" in text
    block = text[text.index("add_executable(sycl-mxfp4-source-line-probe") : text.index("if (LLAMA_TOOLS_INSTALL)")]
    assert "mxfp4_source_line_probe.cpp" in block
    assert "kernels/reference/mxfp4_inline_dot.cpp" in block
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
