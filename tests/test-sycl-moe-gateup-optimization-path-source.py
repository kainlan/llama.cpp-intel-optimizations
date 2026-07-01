#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
PLAN = ROOT / "docs" / "plans" / "2026-06-30-sycl-mxfp4-multirhs-gateup-dpas-work-reduction.md"
SYCL_DOC = ROOT / "docs" / "backend" / "SYCL.md"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def test_current_xmx_tiled_a_load_layout_is_scale_prefix_plus_payload() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    helper = slice_between(
        mmvq,
        "mxfp4_xmx_tiled_load_a_vec_from_group",
        "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec(",
    )
    assert "constexpr int packed_bytes  = k_per / 2" in helper
    assert "const uint8_t * scale_ptr  = group + xmx_row_in_group" in helper
    assert "const uint8_t * packed_ptr = group + tile_n_total + xmx_row_in_group * packed_bytes" in helper
    assert "block_load<uint8_t, Repeat>(scale_ptr)" in helper
    assert "block_load<uint8_t, compact_bytes>(packed_ptr)" in helper


def test_current_packed_m2_route_uses_272_byte_groups_for_tile_n_16() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    body = slice_between(
        mmvq,
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
        "template <int Repeat, int GLU_OP>\nstatic sycl::event mxfp4_pair_glu_gateup_prepack_dpas_sycl",
    )
    assert "const int64_t group_bytes     = tile_n_total * (1 + k_per / 2)" in body
    assert "const int64_t kt_group_stride = n_tile_groups_n * group_bytes" in body
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0" in body
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group0" in body
    registry = REGISTRY.read_text(encoding="utf-8")
    assert "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias" in registry


def test_single_stream_tg_cannot_use_existing_grouped_gateup_without_more_rows() -> None:
    common = COMMON.read_text(encoding="utf-8")
    assert "static constexpr size_t GGML_SYCL_MXFP4_MOE_XMX_N  = 16" in common
    mmvq = MMVQ.read_text(encoding="utf-8")
    dispatch = slice_between(
        mmvq,
        "const bool grouped_decode_shape = xmx_tiled_grouped_eligible",
        "if (device_grouped_shape)",
    )
    assert "total_batches >= exec_n" in dispatch
    assert "ids_host_count == total_batches" in dispatch


def test_rejected_dpas_column_candidates_stay_rejected() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    assert "Decision: `runtime-rejected`" in plan
    assert "235.588515" in plan
    assert "605.034755" in plan
    assert "1323.299220" in plan
    sycl_doc = SYCL_DOC.read_text(encoding="utf-8")
    assert "No production" in sycl_doc
    assert "GGML_SYCL_MOE_GATEUP_MULTIRHS" in sycl_doc
    assert "route is authorized or wired" in sycl_doc


def v2_payload_offset(row: int, packed_bytes: int = 16) -> int:
    return row * packed_bytes


def v2_scale_offset(row: int, tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    return tile_n_total * packed_bytes + row


def v2_group_bytes(tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    payload_bytes = tile_n_total * packed_bytes
    scale_slab_bytes = 64
    return payload_bytes + scale_slab_bytes


def test_v2_aligned_payload_layout_contract() -> None:
    assert v2_group_bytes() == 320
    assert v2_payload_offset(0) == 0
    assert v2_payload_offset(8) == 128
    assert v2_payload_offset(0) % 64 == 0
    assert v2_payload_offset(8) % 64 == 0
    assert v2_scale_offset(0) == 256
    assert v2_scale_offset(15) == 271
    assert v2_scale_offset(0) >= 256


def test_v2_benchmark_cli_scaffolding_exists() -> None:
    bench = (ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp").read_text(encoding="utf-8")
    harness = (ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp").read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    main = (ROOT / "tools" / "sycl-kernel-bench" / "main.cpp").read_text(encoding="utf-8")
    reference = (ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp").read_text(
        encoding="utf-8"
    )
    assert re.search(r"\bbool\s+xmx_tiled_v2\s*=\s*false\s*;", bench)
    assert re.search(r"\bint\s+xmx_tiled_v2_group_bytes\s*=\s*320\s*;", bench)
    assert "parse_moe_xmx_tiled_v2" in harness
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in registry
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in main
    assert "args.xmx_tiled_v2" in reference
    assert "args.xmx_tiled_v2_group_bytes" in reference
