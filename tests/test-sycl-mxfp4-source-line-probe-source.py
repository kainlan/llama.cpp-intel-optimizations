#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tools" / "sycl-kernel-bench" / "CMakeLists.txt"
SOURCE = ROOT / "tools" / "sycl-kernel-bench" / "mxfp4_source_line_probe.cpp"
INLINE_DOT = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"
DEVICE_TU = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_pair_glu_source_line_device.cpp"


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
    assert "kernels/reference/mxfp4_pair_glu_source_line_device.cpp" in block
    link_line = next(line for line in block.splitlines() if "target_link_libraries(sycl-mxfp4-source-line-probe" in line)
    link_tokens = link_line.replace(")", " ").split()
    assert "ggml-base" in link_tokens
    assert "llama-common" not in link_tokens
    assert "ggml" not in link_tokens
    assert "ggml-sycl" not in block
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


def test_source_line_device_tu_is_minimal_and_probe_gated() -> None:
    assert DEVICE_TU.exists()
    text = DEVICE_TU.read_text(encoding="utf-8")
    assert "#ifdef SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY" in text
    assert "#else" in text
    assert "source-line MXFP4 device TU must only be compiled for sycl-mxfp4-source-line-probe" in text
    assert "bool ggml_sycl_mxfp4_pair_glu_bench_launch(const mxfp4_pair_glu_bench_args & args)" in text
    assert text.index("mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel") < text.index("namespace ggml_sycl {")
    assert "GGML_GLU_OP_SWIGLU_OAI" in text
    assert "mxfp4_dpas_pack_q8_source_line_sycl" in text
    assert "mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl<8, GGML_GLU_OP_SWIGLU_OAI, false, false>" in text
    assert "line 9107 \"ggml/src/ggml-sycl/mmvq.cpp\"" in text
    assert "line 9730 \"ggml/src/ggml-sycl/mmvq.cpp\"" in text
    assert "args.xmx_tiled && args.xmx_tiled_pack_q8 && args.rows_per_wg == 8" in text
    assert "args.xmx_tiled_m_tiles == 2" in text
    assert "args.xmx_tiles_n == 1" in text
    assert "args.glu_op == GGML_GLU_OP_SWIGLU_OAI" in text
    assert "args.subgroup_size == 32" in text
    for unsupported in (
        "args.direct_xmx",
        "args.xmx_tiled_grouped",
        "args.xmx_tiled_prefetch",
        "args.xmx_tiled_v2",
        "args.xmx_tiled_bundle4",
        "args.split_gate_up",
        "args.single_column_gateup",
        "args.multi_rhs_gateup",
        "args.predecoded_i8",
        "args.vector_qs_load",
        "args.scale_stride_blocks != 0",
        "args.down_q8_soa != nullptr",
    ):
        assert unsupported in text
    assert "GGML_SYCL_PROFILING_DEBUG" not in text
    assert "ggml_sycl_op_profiler" not in text
    assert "ggml_sycl_mxfp4_mmv_id" not in text
    assert "mxfp4_layer_glu_down" not in text
    assert "mxfp4_pair_glu_xmx_tiled_grouped" not in text
    assert "mxfp4_pair_glu_xmx_tiled_v2" not in text
    assert "mxfp4_pair_glu_xmx_tiled_bundle4" not in text
    assert "mxfp4_pair_glu_xmx_tiled_dpas_m4" not in text


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


def test_mxfp4_inline_dot_probe_mode_avoids_backend_helpers_and_uses_cpu_reference() -> None:
    text = INLINE_DOT.read_text(encoding="utf-8")
    selector_start = text.index("static bool select_mxfp4_xmx_tiles_n")
    selector_end = text.index("static inline float mxfp4_e8m0_to_fp32_device", selector_start)
    selector = text[selector_start:selector_end]
    assert "#ifdef SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY" in selector
    probe_selector = selector.split("#else", 1)[0]
    assert "ggml_sycl_info" not in probe_selector
    assert "ggml_sycl_get_device_id_from_queue" not in probe_selector

    assert "validate_pair_glu_cpu_reference" in text
    pair_glu_start = text.index("bool run_mxfp4_pair_glu")
    pair_glu_end = text.index("bool run_mxfp4_layer_glu_down")
    pair_glu = text[pair_glu_start:pair_glu_end]
    cpu_ref = pair_glu.index("validate_pair_glu_cpu_reference")
    guard_start = pair_glu.rindex("#ifdef SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY", 0, cpu_ref)
    guard_else = pair_glu.index("#else", guard_start)
    guard_end = pair_glu.index("#endif", guard_else)
    probe_pair_glu = pair_glu[guard_start:guard_else]
    normal_pair_glu = pair_glu[guard_else:guard_end]
    assert "validate_pair_glu_cpu_reference" in probe_pair_glu
    assert "ggml_sycl_mxfp4_pair_glu_bench_launch(ref_args)" not in probe_pair_glu
    assert "ggml_sycl_mxfp4_pair_glu_bench_launch(ref_args)" in normal_pair_glu
