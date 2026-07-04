#!/usr/bin/env python3
from __future__ import annotations

import subprocess
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


def test_source_line_feasibility_script_refuses_execute_without_ack() -> None:
    result = subprocess.run(["bash", str(SCRIPT), "--execute"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-microbenchmarks" in result.stdout


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
