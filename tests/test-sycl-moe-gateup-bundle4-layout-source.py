#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
BENCH_ARGS = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
REFERENCE_HEADER = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "reference_kernels.hpp"
HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
MAIN = ROOT / "tools" / "sycl-kernel-bench" / "main.cpp"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def test_bundle4_layout_formula_preserves_baseline_bytes_and_aligns_payloads() -> None:
    tile_n_total = 16
    packed_bytes = 16
    bundle_groups = 4
    baseline_group_bytes = tile_n_total * (1 + packed_bytes)
    payload_group_bytes = tile_n_total * packed_bytes
    payload_slab_bytes = bundle_groups * payload_group_bytes
    scale_slab_bytes = bundle_groups * tile_n_total
    bundle_bytes = payload_slab_bytes + scale_slab_bytes
    assert baseline_group_bytes == 272
    assert bundle_bytes == 1088
    assert bundle_bytes == bundle_groups * baseline_group_bytes
    assert [g * payload_group_bytes for g in range(bundle_groups)] == [0, 256, 512, 768]
    assert all((g * payload_group_bytes) % 64 == 0 for g in range(bundle_groups))
    assert payload_slab_bytes == 1024


def test_reference_bundle4_layout_helper_exists() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    helper = slice_between(reference, "make_xmx_tiled_bundle4_payload_layout", "static int normalize_supported_xmx_tiles_n")
    assert "constexpr size_t bundle_groups" in helper
    assert "const size_t     payload_group_bytes" in helper
    assert "const size_t     payload_slab_bytes" in helper
    assert "const size_t     scale_slab_bytes" in helper
    assert "const size_t     bundle_bytes" in helper
    assert "bundle_bytes == bundle_groups * old_group_bytes" in helper
    assert "group_n / bundle_groups" in helper
    assert "group_n % bundle_groups" in helper
    normalized = " ".join(helper.split())
    assert "new_payload = new_bundle + group_in_bundle * payload_group_bytes" in normalized
    assert "new_scale = new_bundle + payload_slab_bytes + group_in_bundle * tile_n_total" in normalized


def test_bundle4_benchmark_route_is_registered_and_parsed() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")
    harness = HARNESS.read_text(encoding="utf-8")
    bench_args = BENCH_ARGS.read_text(encoding="utf-8")
    reference_header = REFERENCE_HEADER.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")
    route = "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2_sparse32_bias"
    assert route in registry
    assert route in main
    assert "parse_moe_xmx_tiled_bundle4" in harness
    assert "const bool xmx_tiled_bundle4             = parse_moe_xmx_tiled_bundle4(config.kernel_name)" in harness
    assert "const int  xmx_tiled_bundle4_group_bytes = xmx_tiled_bundle4 ? 1088 : 0" in harness
    assert "bool  xmx_tiled_bundle4" in bench_args
    assert "int   xmx_tiled_bundle4_group_bytes" in bench_args
    assert "bool                         xmx_tiled_bundle4" in reference_header
    assert "int                          xmx_tiled_bundle4_group_bytes" in reference_header
    assert "xmx_tiled_bundle4, xmx_tiled_bundle4_group_bytes" in reference


def test_reference_bundle4_route_is_fail_closed_before_layout_allocation() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    invalid = slice_between(
        reference,
        "if (xmx_tiled_bundle4 &&",
        "const size_t         selected_count",
    )
    assert "!xmx_tiled" in invalid
    assert "!xmx_tiled_pack_q8" in invalid
    assert "xmx_tiled_grouped" in invalid
    assert "xmx_tiled_prefetch" in invalid
    assert "xmx_tiled_m_tiles != 2" in invalid
    assert "rows_per_wg != 8" in invalid
    assert "xmx_tiles_n != 1" in invalid
    assert "xmx_tiled_bundle4_group_bytes != 1088" in invalid
    assert "mxfp4_pair_glu XMX_TILED_BUNDLE4 requires packed XMX_TILED r8 m2 and 1088-byte bundles." in invalid
    layout_block = slice_between(reference, "if (xmx_tiled_v2)", "const size_t  expert_bytes")
    assert "if (xmx_tiled_bundle4)" in layout_block
    assert "make_xmx_tiled_bundle4_payload_layout(launch_layout, m, k)" in layout_block


def test_sycl_bundle4_bench_launcher_fails_closed_before_v2_and_packed_dispatch() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    launcher = slice_between(
        mmvq,
        "bool ggml_sycl_mxfp4_pair_glu_bench_launch",
        "bool ggml_sycl_mxfp4_mmv_id_bench_launch",
    )
    bundle4_pos = launcher.index("if (args.xmx_tiled_bundle4)")
    v2_pos = launcher.index("if (args.xmx_tiled_v2)")
    packed_pos = launcher.index("if (args.xmx_tiled_pack_q8)")
    bundle4_guard = slice_between(launcher, "if (args.xmx_tiled_bundle4)", "if (args.xmx_tiled_v2)")
    assert bundle4_pos < v2_pos < packed_pos
    assert "return false" in bundle4_guard
