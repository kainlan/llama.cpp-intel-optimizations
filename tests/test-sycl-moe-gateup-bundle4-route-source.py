#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
MAIN = ROOT / "tools" / "sycl-kernel-bench" / "main.cpp"
HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"


def test_bundle4_non_bias_route_is_registered_and_help_listed() -> None:
    route = "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2"
    registry = REGISTRY.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert route in registry
    assert f"{route}|" in main
    assert "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2_sparse32_bias" in registry
    assert registry.index(route) < registry.index("mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2_sparse32_bias")


def test_bundle4_non_bias_route_uses_existing_bundle4_parser_without_bias_or_sparse_flags() -> None:
    route = "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2"
    harness = HARNESS.read_text(encoding="utf-8")

    assert "parse_moe_xmx_tiled_bundle4" in harness
    assert "const bool xmx_tiled_bundle4             = parse_moe_xmx_tiled_bundle4(config.kernel_name)" in harness
    assert "const bool sparse_expert_slots      = config.kernel_name.find(\"_sparse32\") != std::string::npos" in harness
    assert "const bool use_bias                 = config.kernel_name.find(\"_bias\") != std::string::npos" in harness
    assert "_bias" not in route
    assert "_sparse32" not in route
