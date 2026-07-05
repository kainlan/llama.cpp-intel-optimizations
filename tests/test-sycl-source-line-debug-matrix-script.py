#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-source-line-debug-matrix.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_debug_matrix_zebin_lookup_is_pipe_free() -> None:
    text = script_text()
    assert "-print -quit" in text
    assert "| head -n 1" not in text


def test_debug_matrix_script_is_dry_run_by_default() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 0, result.stdout
    for required in ("DRY RUN", "release_split", "debug_line_tables", "debug_full", "debug_no_inline", "sycl-source-line-probe", "readelf -S", "gpu-source-line"):
        assert required in result.stdout
    assert "vtune-computing-tasks.csv" in result.stdout
    assert "parse-sycl-vtune-tasks.py" in result.stdout
    assert "zebin-debug-line.txt" in result.stdout
    assert "--dwarf-line-dump" in result.stdout
    assert "--require-source-path main.cpp" in result.stdout
    assert "tools/sycl-source-line-probe/main.cpp" not in result.stdout
    assert "computing-tasks-of-interest" not in result.stdout
    assert "/Storage" not in result.stdout
    assert "llama-bench" not in result.stdout
    assert "sycl-kernel-bench" not in result.stdout


def test_debug_matrix_vtune_target_gpu_is_explicit_opt_in() -> None:
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


def test_debug_matrix_script_handles_absolute_out_root_in_dry_run(tmp_path: Path) -> None:
    out_root = tmp_path / "source line matrix"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--out-root", str(out_root)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    expected_probe = out_root / "build-matrix" / "release_split" / "build" / "bin" / "sycl-source-line-probe"
    assert str(expected_probe).replace(" ", "\\ ") in result.stdout
    assert ".//tmp" not in result.stdout


def test_debug_matrix_script_refuses_execute_without_ack() -> None:
    result = subprocess.run(["bash", str(SCRIPT), "--execute"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-source-probe" in result.stdout


def test_debug_matrix_task_filter_is_opt_in() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--task-glob", "*sycl_source_line_probe*"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "computing-tasks-of-interest" in result.stdout
    assert "\\*sycl_source_line_probe\\*#1#1#20" in result.stdout


def test_debug_matrix_task_glob_with_single_quote_is_shell_escaped() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--task-glob", "a'b"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "computing-tasks-of-interest=a\\'b#1#1#20" in result.stdout
    assert "computing-tasks-of-interest='a'b#1#1#20'" not in result.stdout


def test_debug_matrix_requires_probe_dwarf_basename() -> None:
    text = script_text()
    assert '"tools/sycl-source-line-probe/main.cpp"' not in text
    assert '--require-source-path "main.cpp"' in text
    assert '"main.cpp" "${dir}/source-line-feasibility.parse"' in text


def test_debug_matrix_execute_branch_writes_expected_artifacts() -> None:
    text = script_text()
    for required in (
        "source-line/build-matrix",
        "zebin-debug-sections.txt",
        "vtune-computing-tasks.csv",
        "vtune-task.parse",
        "parse-sycl-vtune-tasks.py",
        "zebin-debug-line.txt",
        "vtune-gpu-source-line.csv",
        "source-line-feasibility.parse",
        "check-sycl-vtune-source-lines.py",
        "--dwarf-line-dump",
        "--require-source-path",
        "VTune gpu-source-line report failed",
        "source-line checker reported failure",
        "set +u",
        "source /opt/intel/oneapi/setvars.sh --force",
        "set -u",
    ):
        assert required in text
