#!/usr/bin/env python3
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
BENCH_HPP = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
E2E_PLAN_REL = pathlib.Path("docs/plans/2026-06-30-sycl-gptoss-mxfp4-e2e-decode-profiling.md")
E2E_PLAN = ROOT / E2E_PLAN_REL
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
            if ch == "\\":
                if i + 1 < len(source):
                    output.append(source[i + 1])
                    i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            output.append(ch)
            if ch == "\\":
                if i + 1 < len(source):
                    output.append(source[i + 1])
                    i += 1
            elif ch == "'":
                state = "code"
        i += 1
    return "".join(output)


def read_e2e_plan_text() -> str:
    assert E2E_PLAN.exists(), f"missing local E2E evidence plan: {E2E_PLAN}"
    return E2E_PLAN.read_text(encoding="utf-8")


def test_dpas_repeat_count_limit_blocks_repeat16_role_fusion() -> None:
    if not DPAS_HPP.exists():
        pytest.skip(f"oneAPI DPAS header not found at {DPAS_HPP}")
    dpas = DPAS_HPP.read_text(encoding="utf-8")
    assert "RepeatCount >= 1 && RepeatCount <= 8" in dpas
    assert "Repeat count must be within 1 to 8 range" in dpas


def test_current_m2_gateup_uses_separate_gate_and_up_dpas_calls() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    start = mmvq.index("mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl")
    end = mmvq.index("mxfp4_pair_glu_xmx_tiled_dpas_m4_sycl", start)
    body = strip_cpp_comments(mmvq[start:end])
    assert "gate_part0 = xmx::dpas" in body
    assert "up_part0   = xmx::dpas" in body
    assert "gate_part0.template select<1, 1>(r * exec_n)" in body
    assert "up_part0.template select<1, 1>(r * exec_n)" in body


def test_singlecol_route_is_absent_before_candidate_task() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    bench = BENCH_HPP.read_text(encoding="utf-8")
    assert "GGML_SYCL_MOE_GATEUP_SINGLECOL" not in mmvq
    assert "singlecol-gateup" not in mmvq
    assert "single_column_gateup" not in bench


def test_e2e_evidence_names_moe_as_dominant_bucket() -> None:
    plan = read_e2e_plan_text()
    assert "moe host 7112.910 ms" in plan
    assert "attention 685.619 ms" in plan
    assert "KV 140.835 ms" in plan
    assert "Next target must remain MoE gate/up DPAS work-reduction" in plan
