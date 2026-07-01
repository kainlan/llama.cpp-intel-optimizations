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
