#!/usr/bin/env python3
"""Host-only regression checks for SYCL supports_op/kernel contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
CONCAT_SOURCE = ROOT / "ggml/src/ggml-sycl/concat.cpp"


def _supports_source() -> str:
    source = SUPPORT_SOURCE.read_text(encoding="utf-8")
    return source[source.index("static bool ggml_backend_sycl_device_supports_op") :]


def _case(source: str, op: str, next_op: str) -> str:
    start = source.index(f"case GGML_OP_{op}:")
    end = source.index(f"case GGML_OP_{next_op}:", start)
    return source[start:end]


def test_pool_2d_does_not_inherit_acc_restrictions() -> None:
    source = _supports_source()
    pool = _case(source, "POOL_2D", "ACC")
    assert "return true;" in pool
    assert "GGML_TYPE_F32" not in pool


def test_group_norm_declines_batches_and_preserves_contiguous_f32_control() -> None:
    source = _supports_source()
    predicate = _case(source, "GROUP_NORM", "RMS_NORM_BACK")
    for required in (
        "op->type == GGML_TYPE_F32",
        "op->src[0]->type == GGML_TYPE_F32",
        "op->src[0]->ne[3] == 1",
        "ggml_is_contiguous(op->src[0])",
        "ggml_is_contiguous(op)",
    ):
        assert required in predicate

    supports = lambda src_f32, dst_f32, batches, src_contig, dst_contig: (
        src_f32 and dst_f32 and batches == 1 and src_contig and dst_contig
    )
    assert supports(True, True, 1, True, True)  # normal inference path
    assert not supports(True, True, 2, True, True)


def test_acc_declines_each_unsupported_4d_operand_and_keeps_3d_control() -> None:
    source = _supports_source()
    predicate = _case(source, "ACC", "PAD")
    for required in (
        "op->ne[3] == 1",
        "op->src[1]->ne[3] == 1",
        "ggml_is_contiguous(op->src[0])",
        "ggml_is_contiguous(op->src[1])",
        "ggml_is_contiguous(op)",
    ):
        assert required in predicate

    supports = lambda dst_batches, src1_batches: dst_batches == 1 and src1_batches == 1
    assert supports(1, 1)  # normal 3D inference path
    assert not supports(2, 1)
    assert not supports(1, 2)


def test_norm_family_and_concat_keep_exact_f32_or_instantiated_type_gates() -> None:
    source = _supports_source()
    for op, next_op in (
        ("NORM", "RMS_NORM"),
        ("RMS_NORM", "L2_NORM"),
        ("L2_NORM", "GROUP_NORM"),
        ("RMS_NORM_BACK", "SCALE"),
    ):
        predicate = _case(source, op, next_op)
        assert "op->type == GGML_TYPE_F32" in predicate
        assert "op->src[0]->type == GGML_TYPE_F32" in predicate

    concat = _case(source, "CONCAT", "DUP")
    assert "ggml_sycl_concat_type_supported(op->type)" in concat
    assert "op->src[0]->type == op->type" in concat
    assert "op->src[1]->type == op->type" in concat

    concat_impl = CONCAT_SOURCE.read_text(encoding="utf-8")
    helper_start = concat_impl.index("bool ggml_sycl_concat_type_supported")
    helper_end = concat_impl.index("void ggml_sycl_op_concat", helper_start)
    helper = concat_impl[helper_start:helper_end]
    for supported in ("F32", "F16", "I32", "I16", "I64", "I8"):
        assert f"case GGML_TYPE_{supported}:" in helper
    assert "default:\n            return false;" in helper
