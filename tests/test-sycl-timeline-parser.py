#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-timeline.py"


def run_parser(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_trace(tmp_path: pathlib.Path, raw: str) -> pathlib.Path:
    path = tmp_path / "timeline.json"
    path.write_text(raw, encoding="utf-8")
    return path


def test_wall_ms_nan_reports_clean_argparse_error(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"traceEvents": []}')

    result = run_parser(path, "--wall-ms", "nan")

    assert result.returncode == 2
    assert "--wall-ms must be finite and greater than zero" in result.stdout
    assert "Traceback" not in result.stdout


def test_malformed_json_reports_clean_timeline_error(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"traceEvents": [')

    result = run_parser(path)

    assert result.returncode == 2
    assert "failed to parse timeline" in result.stdout
    assert "Traceback" not in result.stdout


def test_missing_trace_events_reports_clean_timeline_error(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"metadata": []}')

    result = run_parser(path)

    assert result.returncode == 2
    assert "failed to parse timeline" in result.stdout
    assert "Traceback" not in result.stdout


def test_top_abbreviation_is_rejected(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"traceEvents": []}')

    result = run_parser(path, "--top", "20")

    assert result.returncode == 2
    assert "unrecognized arguments: --top" in result.stdout
    assert "Traceback" not in result.stdout


def test_parser_summarizes_wall_categories_and_callsites() -> None:
    trace = {
        "traceEvents": [
            {
                "name": "ggml graph compute",
                "cat": "ggml.graph",
                "ph": "X",
                "ts": 1000,
                "dur": 10000,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 78351,
                    "function": "ggml_backend_sycl_graph_compute_impl",
                },
            },
            {
                "name": "submit",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 2000,
                "dur": 100,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 120,
                    "function": "submit_kernel",
                },
            },
            {
                "name": "wait",
                "cat": "sycl.wait",
                "ph": "X",
                "ts": 9000,
                "dur": 1500,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 130,
                    "function": "wait_kernel",
                },
            },
            {
                "name": "device event 1",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 3000,
                "dur": 3000,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 140,
                    "function": "device_event",
                    "device": 1,
                    "queue_kind": "compute",
                    "device_start_ns": 10000,
                    "device_end_ns": 13000,
                },
            },
            {
                "name": "device event 2",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 6000,
                "dur": 2000,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 140,
                    "function": "device_event",
                    "device": 1,
                    "queue_kind": "compute",
                    "device_start_ns": 16000,
                    "device_end_ns": 18000,
                },
            },
            {"name": "instant", "cat": "sycl.submit", "ph": "i", "ts": 4000},
        ]
    }

    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "timeline.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        result = run_parser(path, "--wall-ms", "10")

    assert result.returncode == 0, result.stdout
    assert "timeline.wall_ms_x1000 10000" in result.stdout
    assert "timeline.gpu_event_total_ms_x1000 5" in result.stdout
    assert "timeline.gpu_event_coverage_pct_x1000 50" in result.stdout
    assert "timeline.unattributed_ms_x1000 9995" in result.stdout
    assert "gap.device1.compute.count 1" in result.stdout
    assert "gap.device1.compute.total_ms_x1000 3" in result.stdout
    assert "category.ggml.graph.host_ms_x1000 10000" in result.stdout
    assert "category.sycl.submit.host_ms_x1000 100" in result.stdout
    assert "category.sycl.wait.host_ms_x1000 1500" in result.stdout
    assert (
        "callsite.ggml/src/ggml-sycl/ggml-sycl.cpp:78351:ggml_backend_sycl_graph_compute_impl.host_ms_x1000 10000"
        in result.stdout
    )
    assert "category.sycl.event" not in result.stdout
