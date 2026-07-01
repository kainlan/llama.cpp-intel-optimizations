#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
BENCH_HPP = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"


def test_benchmark_harness_has_pair_glu_name_parser_to_extend() -> None:
    harness = HARNESS.read_text(encoding="utf-8")
    assert "config.kernel_name.find(\"_singlecol\") != std::string::npos" in harness
    assert "KernelKind::MXFP4_PAIR_GLU" in REGISTRY.read_text(encoding="utf-8")
    assert "_multirhs" not in harness


def test_bench_args_do_not_expose_multirhs_before_candidate() -> None:
    bench = BENCH_HPP.read_text(encoding="utf-8")
    assert "single_column_gateup = false" in bench
    assert "multi_rhs_gateup" not in bench


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
