#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
FATTN = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_mmvq_mxfp4_hot_submits_have_named_profile_labels() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "#include \"sycl-kernel-profiler.hpp\"" in mmvq
    for label in [
        "mxfp4.gateup.xmx_tiled_dpas_m2",
        "mxfp4.gateup.xmx_tiled_dpas_m4",
        "mxfp4.gateup.xmx_tiled_bundle4_m2",
        "mxfp4.down.direct_final_i8",
        "mxfp4.down.direct_final_dpas",
        "mxfp4.down.same_expert_grouped",
        "mxfp4.soa.batched",
        "mxfp4.soa.pair_glu_batched",
    ]:
        assert label in mmvq
    assert mmvq.count("ggml_sycl_profile_submit(") >= 8


def test_active_packed_q8_m2_metadata_preserves_route_context() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    body = slice_between(
        mmvq,
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
        "template <int Repeat, int GLU_OP>\nstatic sycl::event mxfp4_pair_glu_gateup_prepack_dpas_sycl",
    )
    assert "mxfp4.gateup.xmx_tiled_dpas_m2" in body
    assert "path=packed-q8-m2" in body
    assert "tiles=" in body
    assert "total_batches=" in body
    assert "ggml_sycl_profile_submit(queue" in body
    assert "h.depends_on(pack_event)" in body


def test_mmvq_copy_helper_records_named_copy_event() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    body = slice_between(mmvq, "static sycl::event mmvq_submit_memcpy_with_deps", "static void mmvq_memcpy_sync")
    assert "sycl.memcpy.mmvq_with_deps" in body
    assert "ggml_sycl_profile_record_returned_event" in body
    assert "ggml_sycl::mem_copy_async" in body


def test_mmvq_active_blind_spots_have_named_profile_labels() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    for label in [
        "mxfp4.quantize.activation_q8_soa",
        "mxfp4.pack_q8.single_col",
        "mxfp4.pack_q8.grouped_chunks",
        "mxfp4.down.zero",
        "mxfp4.down.q8_soa",
        "mxfp4.down.q8_soa_atomic",
        "mxfp4.down.q8_soa_row_group",
        "mxfp4.down.weighted_tmp_reduce",
    ]:
        assert label in mmvq

    pack_body = slice_between(
        mmvq,
        "static sycl::event mxfp4_dpas_pack_q8_single_col_groups_sycl",
        "static bool mxfp4_grouped_pack_q8_vector_enabled",
    )
    assert "ggml_sycl_profile_submit(queue" in pack_body
    assert "path=packed-q8;role=pack" in pack_body
    assert ".wait(" not in pack_body

    down_body = slice_between(
        mmvq,
        "class mxfp4_down_sum_q8_soa_kernel;",
        "bool mmvq_moe_batched_dispatch_down_sum_from_cached_q8_mxfp4",
    )
    assert "ggml_sycl_profile_submit(queue" in down_body
    assert "path=q8-soa;role=down" in down_body
    assert ".wait(" not in down_body


def test_mmvq_active_direct_q8_soa_down_honors_row_group_variants() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    direct_body = slice_between(
        mmvq,
        "static const bool atomic_reduce = []() {\n        const char * env = std::getenv(\"GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC\");",
        "    if (completion_event) {\n        *completion_event = event;\n    }",
    )
    assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC" in direct_body
    assert direct_body.index("if (atomic_reduce)") < direct_body.index("mxfp4_down_sum_q8_soa_atomic_sycl")

    assert "mxfp4_moe_down_sum_q8_soa_tg_active_rows_per_group" in direct_body
    assert "/*is_down_role=*/true, n_tokens" in direct_body
    assert "mxfp4_down_sum_q8_soa_row_group_sycl<2>" in direct_body
    assert "mxfp4_down_sum_q8_soa_row_group_sycl<4>" in direct_body
    assert "mxfp4_down_sum_q8_soa_sycl" in direct_body
    assert direct_body.index("mxfp4_down_sum_q8_soa_atomic_sycl") < direct_body.index(
        "mxfp4_moe_down_sum_q8_soa_tg_active_rows_per_group"
    )
    assert direct_body.index("mxfp4_down_sum_q8_soa_row_group_sycl<2>") < direct_body.index(
        "mxfp4_down_sum_q8_soa_sycl"
    )
