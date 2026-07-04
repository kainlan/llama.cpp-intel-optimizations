#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-full-attribution-profile.sh"

HARNESS_OVERRIDE_ENVS = [
    "SYCL_GPTOSS_FULL_ATTRIBUTION_OUT",
    "SYCL_LLAMA_BENCH",
    "SYCL_GPTOSS_MODEL",
]

REQUIRED_DRY_RUN_STRINGS = [
    "DRY RUN",
    "--execute --i-understand-this-runs-gpu-models-and-profilers",
    "./build/bin/llama-bench",
    "/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf",
    "ONEAPI_DEVICE_SELECTOR=level_zero:1",
    "GGML_SYCL_E2E_TG_PROFILE=1",
    "GGML_SYCL_TIMELINE=timeline+events",
    "GGML_SYCL_TIMELINE_OUTPUT=",
    "GGML_SYCL_KERNEL_PROFILE=1",
    "GGML_SYCL_KERNEL_PROFILE_RAW=1",
    "GGML_SYCL_VTUNE_ITT=1",
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
    "ZE_ENABLE_TRACING_LAYER=1",
    "SYCL_UR_TRACE=2",
    "PTI_L0_TRACE_OUTPUT=",
    "SYCL_UR_TRACE_LOG=",
    "-fa 1",
    "raw/timeline/sycl-timeline.json",
    "raw/kernel/sycl-kernels.csv",
    "pti/level-zero-api.jsonl",
    "ur/sycl-ur-trace.log",
    "vtune/exported-kernels.csv",
    "vtune/exported-source-lines.csv",
    "source-line/build-matrix",
    "parsed/timeline.parse",
    "parsed/kernel-cost.parse",
    "parsed/l0.parse",
    "parsed/ur.parse",
    "parsed/vtune.parse",
    "parsed/layer-ledger.parse",
    "parsed/source-line.parse",
    "parsed/source-attribution.parse",
    "scripts/parse-sycl-timeline.py",
    "scripts/parse-sycl-kernel-profile.py",
    "scripts/parse-sycl-pti-l0.py",
    "scripts/parse-sycl-ur-trace.py",
    "scripts/parse-sycl-vtune-exports.py",
    "scripts/parse-sycl-layer-ledger.py",
    "scripts/check-sycl-vtune-source-lines.py",
    "scripts/parse-sycl-source-attribution.py",
    "scripts/sycl-source-line-debug-matrix.sh",
    "vtune -collect gpu-hotspots",
    "dump-compute-task-binaries=true",
]

EXECUTE_BRANCH_STRINGS = [
    "mkdir -p \"${raw_timeline_dir}\" \"${raw_kernel_dir}\" \"${pti_dir}\" \"${ur_dir}\" \"${vtune_dir}\" \"${source_line_dir}\" \"${parsed_dir}\"",
    "set +u",
    "source /opt/intel/oneapi/setvars.sh --force >\"${OUT_ROOT}/setvars.log\" 2>&1",
    "set -u",
    "-- env \"${env_args[@]}\" \"${bench_args[@]}\" >\"${OUT_ROOT}/bench.stdout\" 2>\"${OUT_ROOT}/bench.stderr\"",
    "require_file \"${raw_timeline_dir}/sycl-timeline.json\"",
    "require_file \"${raw_kernel_dir}/sycl-kernels.csv\"",
    "require_file \"${pti_dir}/level-zero-api.jsonl\"",
    "require_file \"${ur_dir}/sycl-ur-trace.log\"",
    "require_file \"${vtune_dir}/exported-kernels.csv\"",
    "require_file \"${vtune_dir}/exported-source-lines.csv\"",
    "python3 scripts/parse-sycl-timeline.py \"${raw_timeline_dir}/sycl-timeline.json\" >\"${parsed_dir}/timeline.parse\"",
    "python3 scripts/parse-sycl-kernel-profile.py --top-kernels 40 \"${raw_kernel_dir}/sycl-kernels.csv\" >\"${parsed_dir}/kernel-cost.parse\"",
    "python3 scripts/parse-sycl-pti-l0.py \"${pti_dir}/level-zero-api.jsonl\" >\"${parsed_dir}/l0.parse\"",
    "python3 scripts/parse-sycl-ur-trace.py \"${ur_dir}/sycl-ur-trace.log\" >\"${parsed_dir}/ur.parse\"",
    "python3 scripts/parse-sycl-vtune-exports.py \\",
    "python3 scripts/parse-sycl-layer-ledger.py \\",
    "python3 scripts/check-sycl-vtune-source-lines.py \\",
    "if [[ -f scripts/parse-sycl-source-attribution.py && -f \"${source_region_map}\" ]]; then",
    "python3 scripts/parse-sycl-source-attribution.py \\",
    "source_attribution.status missing_parser",
]


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _run_script(*args: str, out_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in HARNESS_OVERRIDE_ENVS:
        env.pop(name, None)
    env["SYCL_GPTOSS_FULL_ATTRIBUTION_OUT"] = str(out_dir)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_full_attribution_profile_script_defaults_to_dry_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "dry-run-output"
    result = _run_script(out_dir=out_dir)
    assert result.returncode == 0, result.stdout
    assert not out_dir.exists()
    assert "setvars.sh" not in result.stdout
    for required in REQUIRED_DRY_RUN_STRINGS:
        assert required in result.stdout


def test_full_attribution_profile_script_refuses_execute_without_full_ack(tmp_path: Path) -> None:
    result = _run_script("--execute", out_dir=tmp_path / "execute-output")
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-models-and-profilers" in result.stdout


def test_full_attribution_profile_script_rejects_shorter_ack(tmp_path: Path) -> None:
    result = _run_script(
        "--execute",
        "--i-understand-this-runs-gpu-models",
        out_dir=tmp_path / "execute-output",
    )
    assert result.returncode == 2
    assert "unknown argument: --i-understand-this-runs-gpu-models" in result.stdout


def test_execute_branch_wires_full_artifact_layout_and_parsers() -> None:
    text = _script_text()
    execute_start = 'mkdir -p "${raw_timeline_dir}"'
    assert execute_start in text
    execute_branch = text[text.index(execute_start) :]
    for required in EXECUTE_BRANCH_STRINGS:
        assert required in execute_branch


def test_execute_branch_sets_required_profile_envs_and_fa_on() -> None:
    text = _script_text()
    for required in [
        "GGML_SYCL_E2E_TG_PROFILE=1",
        "GGML_SYCL_TIMELINE=timeline+events",
        "GGML_SYCL_KERNEL_PROFILE=1",
        "GGML_SYCL_KERNEL_PROFILE_RAW=1",
        "GGML_SYCL_VTUNE_ITT=1",
        "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
        "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
        "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
        "ZE_ENABLE_TRACING_LAYER=1",
        "SYCL_UR_TRACE=2",
        "PTI_L0_TRACE_OUTPUT=",
        "SYCL_UR_TRACE_LOG=",
        "-fa 1",
    ]:
        assert required in text


def test_execute_branch_runs_bench_command_once_and_fails_for_missing_external_traces() -> None:
    text = _script_text()
    assert text.count('-- env "${env_args[@]}" "${bench_args[@]}"') == 1
    assert "grep '^UR_TRACE ' \"${OUT_ROOT}/bench.stderr\" >\"${ur_trace}\" || true" in text
    for path in [
        "${pti_dir}/level-zero-api.jsonl",
        "${ur_dir}/sycl-ur-trace.log",
        "${vtune_dir}/exported-kernels.csv",
        "${vtune_dir}/exported-source-lines.csv",
    ]:
        assert f'require_file "{path}"' in text
