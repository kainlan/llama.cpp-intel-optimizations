#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-vtune-tasks.py"


def run_parser(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PARSER), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_vtune_task_parser_selects_matching_csv_task() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "tasks.csv"
        path.write_text(
            "Computing Task,Computing Task:Total Time\n"
            "other_kernel,0.250\n"
            "sycl_source_line_probe_kernel,1.500\n",
            encoding="utf-8",
        )
        result = run_parser(str(path), "--match", "sycl_source_line_probe")
        assert result.returncode == 0, result.stdout
        assert "vtune_task.status ok" in result.stdout
        assert "vtune_task.selected sycl_source_line_probe_kernel" in result.stdout
        assert "vtune_task.selected_time_ms_x1000 1500" in result.stdout


def test_vtune_task_parser_allows_missing_or_blank_time_as_zero() -> None:
    cases = {
        "missing_time.csv": "Source Computing Task\nsycl_source_line_probe_kernel\n",
        "blank_time.csv": "Source Computing Task,GPU Time\nsycl_source_line_probe_kernel, \n",
    }
    with tempfile.TemporaryDirectory() as tmp_raw:
        for filename, content in cases.items():
            path = pathlib.Path(tmp_raw) / filename
            path.write_text(content, encoding="utf-8")
            result = run_parser(str(path), "--match", "sycl_source_line_probe")
            assert result.returncode == 0, result.stdout
            assert "vtune_task.selected sycl_source_line_probe_kernel" in result.stdout
            assert "vtune_task.selected_time_ms_x1000 0" in result.stdout


def test_vtune_task_parser_rejects_non_numeric_time_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "tasks.tsv"
        path.write_text("Task\tGPU Time\nsycl_source_line_probe_kernel\tnot-a-number\n", encoding="utf-8")
        result = run_parser(str(path), "--match", "sycl_source_line_probe")
        assert result.returncode == 2
        assert "failed to parse VTune tasks" in result.stdout
        assert "Traceback" not in result.stdout


def test_vtune_task_parser_reports_missing_match_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "tasks.tsv"
        path.write_text("Task\tGPU Time\nalpha\t0.125\n", encoding="utf-8")
        result = run_parser(str(path), "--match", "missing")
        assert result.returncode == 2
        assert "failed to parse VTune tasks" in result.stdout
        assert "no task matched missing" in result.stdout
        assert "Traceback" not in result.stdout
