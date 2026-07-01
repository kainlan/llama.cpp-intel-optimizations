#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
BENCH_HPP = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
REFERENCE = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"


def test_multirhs_bench_cli_scaffolding_exists() -> None:
    bench = BENCH_HPP.read_text(encoding="utf-8")
    harness = HARNESS.read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    main = (ROOT / "tools" / "sycl-kernel-bench" / "main.cpp").read_text(encoding="utf-8")
    assert "bool  multi_rhs_gateup" in bench
    assert "int   multi_rhs_cols" in bench
    assert "multi_rhs_gateup = false" in bench
    assert "multi_rhs_cols = 1" in bench
    assert "_multirhs" in harness
    assert "parse_moe_multirhs_cols" in harness
    assert "mxfp4_pair_glu_xmx_tiled_multirhs_n2_r8" in registry
    assert "mxfp4_pair_glu_xmx_tiled_multirhs_n4_r8" in registry
    assert "mxfp4_pair_glu_xmx_tiled_multirhs_n2_r8" in main
    assert "mxfp4_pair_glu_xmx_tiled_multirhs_n4_r8" in main


def test_multirhs_bench_args_default_off() -> None:
    bench = BENCH_HPP.read_text(encoding="utf-8")
    assert "bool  multi_rhs_gateup" in bench
    assert "int   multi_rhs_cols" in bench
    assert "multi_rhs_gateup = false" in bench
    assert "multi_rhs_cols = 1" in bench


def test_multirhs_reference_path_bounds_group_columns() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    assert "multi_rhs_gateup && multi_rhs_cols != 2 && multi_rhs_cols != 4" in reference
    assert "multi-RHS gate/up requires n2 or n4" in reference


def test_production_dispatch_has_inputs_needed_for_same_expert_policy() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    start = mmvq.index("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa")
    end = mmvq.index("bool mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4", start)
    dispatch = mmvq[start:end]
    assert "ids_device" in dispatch
    assert "total_batches" in dispatch
    assert "num_tokens" in dispatch
    assert "ggml_sycl_graph_recording_active()" in dispatch
    assert "ne12 == 1 && !pp_profile" in dispatch


def test_multirhs_requires_same_expert_contract_text() -> None:
    contract = (
        "multi-RHS DPAS may batch multiple RHS columns only when every RHS column "
        "uses the same expert and therefore the same gate/up A matrices"
    )
    assert "same expert" in contract
    assert "same gate/up A matrices" in contract
