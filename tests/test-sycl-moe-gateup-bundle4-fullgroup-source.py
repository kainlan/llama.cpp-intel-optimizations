#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def test_bundle4_full_group_loader_exists_and_loads_one_16_row_group() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    helper = slice_between(
        mmvq,
        "mxfp4_xmx_tiled_bundle4_load_a_full_group_from_bundle",
        "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_v2_load_a_vec_from_group",
    )

    assert "static_assert(Repeat == 8" in helper
    normalized = " ".join(helper.split())
    assert "constexpr int rows = 2 * Repeat" in normalized
    assert "constexpr int payload_group_bytes = tile_n_total * packed_bytes" in helper
    assert "constexpr int payload_slab_bytes  = bundle_groups * payload_group_bytes" in helper
    assert "simd<uint8_t, 256> packed" in helper
    assert "simd<uint8_t, 16>  scale_bytes" in helper
    assert "block_load<uint8_t, 256>(packed_ptr)" in helper
    assert "block_load<uint8_t, 16>(scale_ptr)" in helper
    assert "for (int r = 0; r < rows; ++r)" in helper
    assert "a_vec.template select<k_per, 1>(r * k_per)" in helper


def test_bundle4_kernel_uses_full_group_loader_with_half_group_fallback() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    kernel = slice_between(
        mmvq,
        "mxfp4_pair_glu_xmx_tiled_bundle4_dpas_m2_sycl",
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
    )

    assert "const bool use_full_group_load" in kernel
    assert "xmx_group_n0 == xmx_group_n1" in kernel
    assert "xmx_row_in_group0 == 0" in kernel
    assert "xmx_row_in_group1 == Repeat" in kernel
    assert "mxfp4_xmx_tiled_bundle4_load_a_full_group_from_bundle<Repeat>" in kernel
    assert "gate_a_vec_full.template select<an, 1>(0)" in kernel
    assert "gate_a_vec_full.template select<an, 1>(an)" in kernel
    assert "up_a_vec_full.template select<an, 1>(0)" in kernel
    assert "up_a_vec_full.template select<an, 1>(an)" in kernel
    assert "if (use_full_group_load)" in kernel
    assert "else" in kernel
    fallback = kernel[kernel.index("else"):kernel.index("simd<int, Repeat * exec_n> gate_part0")]
    assert "mxfp4_xmx_tiled_bundle4_load_a_vec_from_bundle<Repeat>" in fallback
