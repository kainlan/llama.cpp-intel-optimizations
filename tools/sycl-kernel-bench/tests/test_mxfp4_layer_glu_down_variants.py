from __future__ import annotations

import pathlib

BENCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parents[1]
BENCH_HPP = BENCH_ROOT / "benchmark_harness.hpp"
BENCH_CPP = BENCH_ROOT / "kernels" / "reference" / "mxfp4_inline_dot.cpp"
SYCL_BENCH_HPP = REPO_ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
MMVQ = REPO_ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"


def test_layer_glu_down_harness_parses_down_q8_dpas_tile_names() -> None:
    text = BENCH_HPP.read_text(encoding="utf-8")
    assert "down_q8_dpas_tile_rows" in text
    assert 'config.kernel_name.find("_q8_dpas_tile2")' in text
    assert 'config.kernel_name.find("_q8_dpas_tile4")' in text


def test_layer_glu_down_bench_args_carry_down_q8_dpas_tile_rows() -> None:
    text = SYCL_BENCH_HPP.read_text(encoding="utf-8")
    assert "int   down_q8_dpas_tile_rows" in text


def test_layer_glu_down_launch_passes_down_q8_dpas_tile_rows_to_runtime() -> None:
    text = BENCH_CPP.read_text(encoding="utf-8")
    runtime = MMVQ.read_text(encoding="utf-8")
    assert "args.down_q8_dpas_tile_rows" in text
    assert "down_args.down_q8_dpas_tile_rows" in runtime


def test_runtime_bench_rejects_invalid_down_q8_dpas_tile_rows() -> None:
    text = MMVQ.read_text(encoding="utf-8")
    assert "args.down_q8_dpas_tile_rows != 0" in text
    assert "args.down_q8_dpas_tile_rows != 2" in text
    assert "args.down_q8_dpas_tile_rows != 4" in text
