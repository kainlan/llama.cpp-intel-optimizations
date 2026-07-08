from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
DPAS_HPP = pathlib.Path("/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp")


def strip_cpp_comments(source: str) -> str:
    output: list[str] = []
    i = 0
    state = "code"
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            output.append(ch)
            if ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
        elif state == "line_comment":
            if ch == "\n":
                output.append(ch)
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            if ch == "\n":
                output.append(ch)
        elif state == "string":
            output.append(ch)
            if ch == "\\" and i + 1 < len(source):
                output.append(source[i + 1])
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            output.append(ch)
            if ch == "\\" and i + 1 < len(source):
                output.append(source[i + 1])
                i += 1
            elif ch == "'":
                state = "code"
        i += 1
    return "".join(output)


def current_m2_body() -> str:
    mmvq = MMVQ.read_text(encoding="utf-8")
    start = mmvq.index("mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl")
    end = mmvq.index("mxfp4_pair_glu_xmx_tiled_dpas_m4_sycl", start)
    return strip_cpp_comments(mmvq[start:end])


def test_dpas_repeat_count_is_already_capped_at_8() -> None:
    if not DPAS_HPP.exists():
        pytest.skip(f"oneAPI DPAS header not found at {DPAS_HPP}")
    dpas = DPAS_HPP.read_text(encoding="utf-8")
    assert "RepeatCount >= 1 && RepeatCount <= 8" in dpas
    assert "Repeat count must be within 1 to 8 range" in dpas


def test_current_m2_uses_distinct_gate_and_up_a_matrices() -> None:
    body = current_m2_body()
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0" in body
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group0" in body
    assert re.search(
        r"gate_part0\s*=\s*xmx::dpas<8,\s*Repeat,\s*int,\s*int,\s*int8_t,\s*int8_t>"
        r"\(gate_part0,\s*b_vec,\s*gate_a_vec0\)",
        body,
    )
    assert re.search(
        r"up_part0\s*=\s*xmx::dpas<8,\s*Repeat,\s*int,\s*int,\s*int8_t,\s*int8_t>"
        r"\(up_part0,\s*b_vec,\s*up_a_vec0\)",
        body,
    )


def test_current_m2_consumes_rhs_column_zero_not_role_columns() -> None:
    body = current_m2_body()
    assert "gate_part0.template select<1, 1>(r * exec_n)" in body
    assert "up_part0.template select<1, 1>(r * exec_n)" in body
    assert "gate_part0.template select<1, 1>(r * exec_n + 1)" not in body
    assert "up_part0.template select<1, 1>(r * exec_n + 1)" not in body


def test_role_column_route_is_not_implemented() -> None:
    mmvq = strip_cpp_comments(MMVQ.read_text(encoding="utf-8"))
    assert "GGML_SYCL_MOE_GATEUP_ROLECOL" not in mmvq
    assert "rolecol-gateup" not in mmvq
