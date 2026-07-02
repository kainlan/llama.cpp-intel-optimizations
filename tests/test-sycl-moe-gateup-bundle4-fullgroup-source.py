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
