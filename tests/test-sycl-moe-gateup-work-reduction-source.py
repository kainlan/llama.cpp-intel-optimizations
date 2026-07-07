#!/usr/bin/env python3
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
BENCH_HPP = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
KERNEL_REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
BENCHMARK_HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"
REFERENCE_INLINE_DOT = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"
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


def test_singlecol_route_label_and_env_are_default_off() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "singlecol-gateup" in mmvq
    assert "GGML_SYCL_MOE_GATEUP_SINGLECOL" in mmvq
    assert "static bool mxfp4_moe_gateup_singlecol_enabled()" in mmvq
    helper_start = mmvq.index("static bool mxfp4_moe_gateup_singlecol_enabled()")
    helper_end = mmvq.index("}", helper_start)
    helper = mmvq[helper_start:helper_end]
    assert "getenv(\"GGML_SYCL_MOE_GATEUP_SINGLECOL\")" in helper
    assert "std::atoi" in helper


def test_singlecol_candidate_has_its_own_kernel_not_prepack_only() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "mxfp4_pair_glu_singlecol_sycl" in mmvq
    assert "mxfp4_pair_glu_singlecol_submit" in mmvq
    singlecol_start = mmvq.index("mxfp4_pair_glu_singlecol_sycl")
    singlecol_end = mmvq.index("mxfp4_pair_glu_singlecol_submit", singlecol_start)
    singlecol_body = strip_cpp_comments(mmvq[singlecol_start:singlecol_end])
    assert "mxfp4_xmx_tiled_load_a_vec_from_group" in singlecol_body
    assert "mmvq_moe_apply_pair_glu_esimd" in singlecol_body
    assert "prepack" not in singlecol_body.lower()


def test_bench_args_expose_singlecol_without_runtime_default() -> None:
    bench = BENCH_HPP.read_text(encoding="utf-8")
    assert "bool  single_column_gateup" in bench
    singlecol_start = bench.index("single_column_gateup")
    assert "= false" in bench[singlecol_start : singlecol_start + 80]


def test_e2e_evidence_names_moe_as_dominant_bucket() -> None:
    plan = read_e2e_plan_text()
    assert "moe host 7112.910 ms" in plan
    assert "attention 685.619 ms" in plan
    assert "KV 140.835 ms" in plan
    assert "Next target must remain MoE gate/up DPAS work-reduction" in plan


def test_runtime_dispatch_is_default_off_and_records_route() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    dispatch_start = mmvq.index("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa")
    dispatch_end = mmvq.index("bool mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4", dispatch_start)
    dispatch = strip_cpp_comments(mmvq[dispatch_start:dispatch_end])
    assert "mxfp4_moe_gateup_singlecol_enabled()" in dispatch
    assert "test_moe_gateup_singlecol_policy" in dispatch
    assert "singlecol_policy_in.is_tg" in dispatch and "ne12 == 1" in dispatch and "!pp_profile" in dispatch
    assert "singlecol_policy_in.graph_recording" in dispatch and "ggml_sycl_graph_recording_active()" in dispatch
    branch_start = dispatch.index("if (singlecol_policy.accepted)")
    branch_end = dispatch.index("} else {", branch_start)
    singlecol_branch = " ".join(dispatch[branch_start:branch_end].split())
    assert "kernel_event = mxfp4_pair_glu_singlecol_submit" in singlecol_branch
    assert "used_singlecol_gateup = true" in singlecol_branch
    assert "xmx_tiled_path = MXFP4_MOE_GATEUP_SINGLECOL_ROUTE" in singlecol_branch
    assert "profile_path = MXFP4_MOE_GATEUP_SINGLECOL_ROUTE" in singlecol_branch
    assert "profile_layout = GGML_LAYOUT_SOA" in singlecol_branch
    profile_call_start = dispatch.index("mmvq_moe_tg_profile_record", branch_end)
    profile_call_end = dispatch.index(";", profile_call_start)
    assert "profile_layout" in dispatch[profile_call_start:profile_call_end]


def test_gateup_loadv2_runtime_branch_is_not_default_on() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "GGML_SYCL_MOE_GATEUP_M2_LOADV2" not in mmvq
    assert "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_loadv2" in KERNEL_REGISTRY.read_text(encoding="utf-8")


def test_gateup_loadv2_bench_path_reuses_xmx_tiled_layout_without_bundle4() -> None:
    bench = BENCH_HPP.read_text(encoding="utf-8")
    harness = BENCHMARK_HARNESS.read_text(encoding="utf-8")
    reference = REFERENCE_INLINE_DOT.read_text(encoding="utf-8")
    assert "bool  xmx_tiled_loadv2" in bench
    assert "xmx_tiled_loadv2         = false" in bench
    assert "xmx_tiled_loadv2" in harness
    assert 'config.kernel_name.find("_loadv2") != std::string::npos' in harness
    assert harness.index("xmx_tiled_prefetch") < harness.index("xmx_tiled_loadv2, xmx_tiled_m_tiles")
    assert "bool                         xmx_tiled_loadv2," in reference
    assert "args.xmx_tiled_loadv2" in reference
    assert "if (xmx_tiled_loadv2 &&" in reference
    loadv2_validation_start = reference.index("if (xmx_tiled_loadv2 &&")
    loadv2_validation_end = reference.index("if (xmx_tiled_m_tiles != 1", loadv2_validation_start)
    loadv2_validation = reference[loadv2_validation_start:loadv2_validation_end]
    assert "xmx_tiled_bundle4" in loadv2_validation
    assert "make_xmx_tiled_bundle4_payload_layout" not in loadv2_validation
    assert "make_xmx_tiled_loadv2" not in reference
