#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-vtune-source-line-feasibility.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_source_line_feasibility_script_is_dry_run_by_default() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 0, result.stdout
    assert "DRY RUN" in result.stdout
    assert "-DGGML_SYCL_PROFILING_DEBUG=ON" in result.stdout
    assert "sycl-kernel-bench" in result.stdout
    assert "gpu-profiling-mode=source-analysis" in result.stdout
    assert "source-analysis=mem-latency" in result.stdout
    assert "dump-compute-task-binaries=true" in result.stdout
    assert "check-sycl-vtune-source-lines.py" in result.stdout
    assert "/Storage" not in result.stdout
    assert "llama-bench" not in result.stdout


def test_mxfp4_feasibility_vtune_target_gpu_is_explicit_opt_in() -> None:
    explicit = subprocess.run(
        ["bash", str(SCRIPT), "--vtune-target-gpu", "0:7:0.0"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert explicit.returncode == 0, explicit.stdout
    assert "target-gpu=0:7:0.0" in explicit.stdout

    default = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert default.returncode == 0, default.stdout
    assert "target-gpu=" not in default.stdout


def test_source_line_feasibility_script_handles_absolute_build_dir_in_dry_run(tmp_path: Path) -> None:
    build_dir = tmp_path / "source line build"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--build-dir", str(build_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    expected_bench = build_dir / "bin" / "sycl-kernel-bench"
    assert str(expected_bench).replace(" ", "\\ ") in result.stdout
    assert ".//tmp" not in result.stdout


def test_source_line_feasibility_script_refuses_execute_without_ack() -> None:
    result = subprocess.run(["bash", str(SCRIPT), "--execute"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-microbenchmarks" in result.stdout


def test_source_line_feasibility_script_supports_matrix_pass_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        matrix_parse = Path(tmp_raw) / "source-line-feasibility.parse"
        result = subprocess.run(
            ["bash", str(SCRIPT), "--require-matrix-pass", str(matrix_parse)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "DRY RUN" in result.stdout
        assert str(matrix_parse) in result.stdout
        assert "matrix gate" in result.stdout
        assert "grep -qx" in result.stdout
        assert "source_line.status pass" in result.stdout
        matrix_index = result.stdout.index("matrix gate")
        assert matrix_index < result.stdout.index("cmake -S")
        assert matrix_index < result.stdout.index("profiling-debug-build.log")
        assert "/Storage" not in result.stdout
        assert "llama-bench" not in result.stdout


def test_source_line_feasibility_script_propagates_target_kernel_to_checker() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--target-kernel", "custom_kernel"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "--kernel=custom_kernel" in result.stdout
    assert "computing-tasks-of-interest=\\*custom_kernel\\*#1#1#20" in result.stdout
    assert "--require-kernel custom_kernel" in result.stdout


def test_source_line_feasibility_execute_branch_uses_pipe_free_zebin_probe() -> None:
    text = script_text()
    assert "-print -quit" in text
    assert "| head -n 1" not in text
    assert "error: no .zebin found" in text


def test_source_line_feasibility_execute_branch_writes_expected_artifacts() -> None:
    text = script_text()
    assert "profiling-debug-build.log" in text
    assert "vtune-gpu-source-line.csv" in text
    assert "zebin-debug-sections.txt" in text
    assert "source-line-feasibility.parse" in text
    assert "vtune -collect gpu-hotspots" in text
    assert "vtune -report hotspots" in text
    assert "readelf -S" in text
    assert "scripts/check-sycl-vtune-source-lines.py" in text
    assert "REQUIRE_MATRIX_PASS" in text
    assert "! grep -qx \"source_line.status pass\" \"${REQUIRE_MATRIX_PASS}\"" in text
    assert "MXFP4 source-line matrix gate failed" in text
