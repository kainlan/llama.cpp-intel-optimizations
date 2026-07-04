#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-profile-ledger.py"


def run_parser(timeline: pathlib.Path, kernels: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args, str(timeline), str(kernels)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_inputs(tmp: pathlib.Path, kernel_total_ns: int) -> tuple[pathlib.Path, pathlib.Path]:
    timeline = {
        "traceEvents": [
            {"name": "graph", "cat": "ggml.graph", "ph": "X", "ts": 0, "dur": 1000, "args": {}},
            {
                "name": "a.kernel",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 100,
                "dur": 10,
                "args": {"metadata": "event_id=1"},
            },
            {
                "name": "compute_forward_node",
                "cat": "ggml.op",
                "ph": "X",
                "ts": 110,
                "dur": 300,
                "args": {"metadata": "op=MUL_MAT"},
            },
            {
                "name": "b.kernel",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 600,
                "dur": 10,
                "args": {"metadata": "event_id=2"},
            },
            {
                "name": "a.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {"metadata": "event_id=1;device=0;queue_kind=compute;device_start_ns=0;device_end_ns=100000"},
            },
            {
                "name": "b.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {"metadata": "event_id=2;device=0;queue_kind=compute;device_start_ns=400000;device_end_ns=500000"},
            },
        ]
    }
    timeline_path = tmp / "timeline.json"
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    kernel_path = tmp / "kernels.csv"
    kernel_path.write_text(
        "name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded\n"
        f"a.kernel,test,,0,compute,1,{kernel_total_ns},{kernel_total_ns},{kernel_total_ns},{kernel_total_ns},{kernel_total_ns},{kernel_total_ns},0,0,0\n",
        encoding="utf-8",
    )
    return timeline_path, kernel_path


def test_ledger_reports_coverage_mismatch_and_unknown_residual() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        timeline, kernels = write_inputs(pathlib.Path(tmp_raw), kernel_total_ns=1000000)
        result = run_parser(timeline, kernels)
        assert result.returncode == 0, result.stdout
        assert "ledger.wall_ms_x1000 1000" in result.stdout
        assert "ledger.timeline_gpu_event_ms_x1000 200" in result.stdout
        assert "ledger.kernel_profile_total_ms_x1000 1000" in result.stdout
        assert "ledger.timeline_kernel_delta_ms_x1000 800" in result.stdout
        assert "ledger.timeline_kernel_ratio_pct_x1000 20000" in result.stdout
        assert "ledger.coverage_status coverage_mismatch" in result.stdout
        assert "ledger.gap_class.host_overlap_ms_x1000 300" in result.stdout
        assert "ledger.gap_class.queue_serialization_ms_x1000 0" in result.stdout
        assert "ledger.gap_class.runtime_idle_ms_x1000 0" in result.stdout
        assert "ledger.unknown_wall_residual_ms_x1000 500" in result.stdout


def test_ledger_reports_ok_when_timeline_matches_kernel_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        timeline, kernels = write_inputs(pathlib.Path(tmp_raw), kernel_total_ns=200000)
        result = run_parser(timeline, kernels, "--coverage-ratio-threshold", "0.90")
        assert result.returncode == 0, result.stdout
        assert "ledger.coverage_status ok" in result.stdout
        assert "ledger.timeline_kernel_ratio_pct_x1000 100000" in result.stdout


def test_ledger_reports_malformed_input_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        timeline = tmp / "bad.json"
        kernels = tmp / "kernels.csv"
        timeline.write_text("{", encoding="utf-8")
        kernels.write_text("name,total_ns\n", encoding="utf-8")
        result = run_parser(timeline, kernels)
        assert result.returncode == 2
        assert "failed to parse profile ledger" in result.stdout
        assert "Traceback" not in result.stdout


def test_ledger_reports_malformed_kernel_profile_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        timeline = tmp / "timeline.json"
        kernels = tmp / "kernels.json"
        timeline.write_text('{"traceEvents": []}', encoding="utf-8")
        kernels.write_text("[]", encoding="utf-8")
        result = run_parser(timeline, kernels)
        assert result.returncode == 2
        assert "failed to parse profile ledger" in result.stdout
        assert "Traceback" not in result.stdout
