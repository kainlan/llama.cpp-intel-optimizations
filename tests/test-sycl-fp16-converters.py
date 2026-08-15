#!/usr/bin/env python3
"""Host/source contracts and device-reference tests for SYCL Q1_0/NVFP4 FP16 conversion."""

import math
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONVERT = ROOT / "ggml/src/ggml-sycl/convert.cpp"
COMMON = ROOT / "ggml/src/ggml-common.h"
SUPPORT = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
SOURCE = CONVERT.read_text(encoding="utf-8")
SUPPORT_SOURCE = SUPPORT.read_text(encoding="utf-8")
KVALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"unterminated function: {signature}")


def _half(value: float) -> bytes:
    """Compare IEEE binary16 encodings, including signed zero."""
    return struct.pack("<e", value)


def _ue4m3(value: int) -> float:
    if value in (0, 0x7F, 0xFF):
        return 0.0
    exp = (value >> 3) & 0xF
    man = value & 7
    raw = math.ldexp(float(man), -9) if exp == 0 else math.ldexp(1.0 + man / 8.0, exp - 7)
    return raw * 0.5


def _cpu_reference(blocks):
    out = []
    for scales, packed in blocks:
        block = [None] * 64
        for sub in range(4):
            d = _ue4m3(scales[sub])
            for j in range(8):
                q = packed[sub * 8 + j]
                block[sub * 16 + j] = _half(d * KVALUES[q & 0xF])
                block[sub * 16 + j + 8] = _half(d * KVALUES[q >> 4])
        out.extend(block)
    return out


def _device_mapping(blocks, guard=13):
    sentinel = b"GU"
    out = [sentinel] * (guard + len(blocks) * 64 + guard)
    for ib, (scales, packed) in enumerate(blocks):
        for tid in range(32):
            sub, j = divmod(tid, 8)
            q = packed[sub * 8 + j]
            d = _ue4m3(scales[sub])
            iy = guard + ib * 64 + sub * 16 + j
            out[iy] = _half(d * KVALUES[q & 0xF])
            out[iy + 8] = _half(d * KVALUES[q >> 4])
    return out


def _blocks(count):
    result = []
    for ib in range(count):
        # Every subblock has a distinct scale and every byte has asymmetric nibbles.
        scales = tuple((0x38, 0x40, 0x44, 0x48)[(sub + ib) % 4] for sub in range(4))
        packed = tuple((((15 - j - 2 * sub - ib) & 15) << 4) | ((j + 3 * sub + ib) & 15)
                       for sub in range(4) for j in range(8))
        result.append((scales, packed))
    return result


def _contract(convert_source: str, support_source: str = SUPPORT_SOURCE) -> bool:
    try:
        getter = _function(convert_source, "to_fp16_sycl_t ggml_get_to_fp16_sycl(")
        kernel = _function(convert_source, "static void dequantize_block_nvfp4_fp16(")
        launch = _function(convert_source, "static void dequantize_row_nvfp4_fp16_sycl(")
        dense = _function(support_source, "static bool ggml_sycl_mul_mat_type_supported(")
        supports = _function(support_source, "static bool ggml_backend_sycl_device_supports_op(")
    except (ValueError, AssertionError):
        return False

    required_kernel = (
        "const int     sub = tid / 8;",
        "const int     j   = tid % 8;",
        "xb.qs[sub * 8 + j]",
        "xb.d[sub]",
        "sub * QK_NVFP4_SUB + j",
        "y[iy + 8]",
        "q & 0x0f",
        "q >> 4",
    )
    mmid_guard = "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID)"
    if mmid_guard not in supports:
        return False
    mmid_body = supports[supports.index(mmid_guard) : supports.index("}", supports.index(mmid_guard)) + 1]
    return (
        "case GGML_TYPE_Q1_0:\n            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;" in getter
        and "case GGML_TYPE_NVFP4:\n            return dequantize_row_nvfp4_fp16_sycl;" in getter
        and "dequantize_block_sycl<QK_NVFP4" not in getter
        and all(token in kernel for token in required_kernel)
        and "const sycl::nd_item<3> & item_ct1" in kernel
        and "sycl::range<3>(1, 1, nb * 32)" in launch
        and "sycl::range<3>(1, 1, 32)" in launch
        and "GGML_ASSERT(k % QK_NVFP4 == 0);" in launch
        and "dequantize_block_nvfp4_fp16(vx, y, item_ct1)" in launch
        and "GGML_TYPE_Q1_0" not in dense
        and "GGML_TYPE_NVFP4" not in dense
        and "return true;" in mmid_body
    )


def test_layout_constants_and_fp16_only_registration() -> None:
    common = COMMON.read_text(encoding="utf-8")
    assert re.search(r"#define\s+QK1_0\s+128\b", common)
    assert re.search(r"#define\s+QR1_0\s+1\b", common)
    assert re.search(r"#define\s+QK_NVFP4\s+64\b", common)
    assert re.search(r"#define\s+QK_NVFP4_SUB\s+16\b", common)
    assert re.search(r"uint8_t\s+d\[QK_NVFP4/QK_NVFP4_SUB\]", common)
    assert re.search(r"uint8_t\s+qs\[QK_NVFP4/2\]", common)
    assert _contract(SOURCE)

    fp32 = _function(SOURCE, "to_fp32_sycl_t ggml_get_to_fp32_sycl(")
    assert "case GGML_TYPE_Q1_0:" not in fp32
    assert "case GGML_TYPE_NVFP4:" not in fp32


@pytest.mark.parametrize("count", [1, 2, 5])
def test_nvfp4_device_mapping_matches_cpu_reference_with_guards(count: int) -> None:
    blocks = _blocks(count)
    guard = 13
    actual = _device_mapping(blocks, guard)
    assert actual[:guard] == [b"GU"] * guard
    assert actual[-guard:] == [b"GU"] * guard
    assert actual[guard:-guard] == _cpu_reference(blocks)


def test_generic_nibble_adapter_is_not_nvfp4_subblock_mapping() -> None:
    # A conventional generic nibble adapter treats all 32 low nibbles as the
    # first half and all high nibbles as the second. NVFP4 instead interleaves
    # four independently-scaled 16-value subblocks, so it must stay custom.
    scales, packed = _blocks(1)[0]
    generic = [None] * 64
    for j, q in enumerate(packed):
        d = _ue4m3(scales[j // 8])
        generic[j] = _half(d * KVALUES[q & 0xF])
        generic[j + 32] = _half(d * KVALUES[q >> 4])
    reference = _cpu_reference([(scales, packed)])
    assert generic != reference
    assert any(generic[i] != reference[i] for i in range(33, 64))


def test_source_contract_mutations_fail_closed() -> None:
    assert _contract(SOURCE)
    mutations = (
        (
            "q1_0 registration renamed away",
            SOURCE.replace(
                "case GGML_TYPE_Q1_0:\n            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;",
                "case GGML_TYPE_Q2_0:\n            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;",
                1,
            ),
        ),
        (
            "nvfp4 downgraded to the generic adapter",
            SOURCE.replace("return dequantize_row_nvfp4_fp16_sycl;", "return dequantize_block_sycl<QK_NVFP4, 2, dequantize_nvfp4>;", 1),
        ),
        ("nvfp4 sub-block index collapsed", SOURCE.replace("const int     sub = tid / 8;", "const int     sub = tid / 16;", 1)),
        ("nvfp4 lane index widened", SOURCE.replace("const int     j   = tid % 8;", "const int     j   = tid % 16;", 1)),
        ("nvfp4 high-nibble offset moved off the sub-block", SOURCE.replace("y[iy + 8]", "y[iy + 32]", 1)),
        ("nvfp4 launch geometry doubled", SOURCE.replace("sycl::range<3>(1, 1, nb * 32)", "sycl::range<3>(1, 1, nb * 64)", 1)),
    )
    for label, mutated in mutations:
        assert mutated != SOURCE, f"mutation no longer applies to the source: {label}"
        assert not _contract(mutated), f"contract still passed under mutation: {label}"

    dense_fn = _function(SUPPORT_SOURCE, "static bool ggml_sycl_mul_mat_type_supported(")
    dense_open_fn = dense_fn.replace(
        "        case GGML_TYPE_F32:",
        "        case GGML_TYPE_Q1_0:\n        case GGML_TYPE_NVFP4:\n        case GGML_TYPE_F32:",
        1,
    )
    dense_open = SUPPORT_SOURCE.replace(dense_fn, dense_open_fn, 1)
    assert not _contract(SOURCE, dense_open)

    mmid_closed = SUPPORT_SOURCE.replace(
        "if (op->op == GGML_OP_ADD_ID || op->op == GGML_OP_MUL_MAT_ID)",
        "if (op->op == GGML_OP_ADD_ID)",
        1,
    )
    assert not _contract(SOURCE, mmid_closed)
