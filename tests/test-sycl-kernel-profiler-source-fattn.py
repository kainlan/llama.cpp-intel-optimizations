#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
FATTN = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn.cpp"
FATTN_XMX = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn-xmx-f16.hpp"
FATTN_XMX_V2 = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn-xmx-f16-v2.hpp"
UNIFIED = ROOT / "ggml" / "src" / "ggml-sycl" / "unified-kernel.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_fattn_major_submits_have_named_profile_labels() -> None:
    fattn = FATTN.read_text(encoding="utf-8")
    assert "#include \"sycl-kernel-profiler.hpp\"" in fattn
    assert "fattn.xmx_pack_k_set_rows" in fattn
    assert "fattn.pack" in fattn
    set_rows = slice_between(
        fattn,
        "static sycl::event ggml_sycl_fattn_xmx_submit_set_rows_update",
        "}  // namespace",
    )
    assert "ggml_sycl_profile_submit(*stream" in set_rows
    assert "cgh.depends_on(set_rows_event)" in set_rows
    assert ".wait(" not in set_rows

    pack_body = slice_between(
        fattn,
        "bool ggml_sycl_fattn_xmx_materialize_packed_k",
        "static bool ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer",
    )
    assert "fattn.pack" in pack_body
    assert "ggml_sycl_profile_submit(*stream" in pack_body
    assert "cgh.depends_on(zero_event)" in pack_body
    assert ".wait(" not in pack_body


def test_fattn_compute_headers_have_named_profile_labels() -> None:
    legacy = FATTN_XMX.read_text(encoding="utf-8")
    v2 = FATTN_XMX_V2.read_text(encoding="utf-8")

    assert "fattn.compute.xmx_f16" in legacy
    assert "ggml_sycl_profile_submit(*stream" in legacy

    for label in [
        "fattn.compute.xmx_v2",
        "fattn.compute.xmx_v2_decode_m1n64",
        "fattn.compute.xmx_v2_decode_gqa",
        "fattn.compute.xmx_v2_decode_gqa_split_first",
        "fattn.compute.xmx_v2_decode_gqa_split_merge",
    ]:
        assert label in v2
    assert "ggml_sycl_profile_submit(*stream" in v2
    assert ".wait(" not in slice_between(v2, "static void launch_fattn_xmx_v2_f16_leaf", "template <int D, bool use_logit_softcap, typename Q_type>")


def test_unified_matmul_dispatches_have_named_profile_labels() -> None:
    unified = UNIFIED.read_text(encoding="utf-8")
    for label in [
        "unified.matmul.xmx",
        "unified.matmul.esimd_int8",
        "unified.matmul.esimd_cooperative",
        "unified.matmul.esimd_fp16_double_buffered",
        "unified.matmul.esimd_fp16",
        "unified.matmul.dmmv",
        "unified.matmul.scalar",
    ]:
        assert label in unified
    launch_body = slice_between(unified, "void launch_unified_matmul", "}  // namespace ggml_sycl_unified")
    assert "ggml_sycl_profile_submit(q" in launch_body
