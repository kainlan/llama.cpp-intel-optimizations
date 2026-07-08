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


def test_top_gaps_negative_is_rejected(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"traceEvents": []}')

    result = run_parser(path, "--top-gaps", "-1")

    assert result.returncode == 2
    assert "--top-gaps must be non-negative" in result.stdout
    assert "Traceback" not in result.stdout


def test_top_host_gap_overlaps_negative_is_rejected(tmp_path: pathlib.Path) -> None:
    path = write_trace(tmp_path, '{"traceEvents": []}')

    result = run_parser(path, "--top-host-gap-overlaps", "-1")

    assert result.returncode == 2
    assert "--top-host-gap-overlaps must be non-negative" in result.stdout
    assert "Traceback" not in result.stdout


def test_parser_reads_device_ranges_from_timeline_metadata(tmp_path: pathlib.Path) -> None:
    trace = {
        "traceEvents": [
            {
                "name": "event-a",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 3,
                "args": {
                    "metadata": "device=2;queue_kind=copy;device_start_ns=1000;device_end_ns=4000",
                },
            },
            {
                "name": "event-b",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 2,
                "args": {
                    "metadata": "device=2;queue_kind=copy;device_start_ns=7000;device_end_ns=9000",
                },
            },
        ]
    }
    path = tmp_path / "metadata-timeline.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = run_parser(path, "--wall-ms", "1")

    assert result.returncode == 0, result.stdout
    assert "timeline.gpu_event_total_ms_x1000 5" in result.stdout
    assert "gap.device2.copy.count 1" in result.stdout
    assert "gap.device2.copy.total_ms_x1000 3" in result.stdout
    assert "gap_transition." not in result.stdout


def test_parser_reports_top_gap_transitions(tmp_path: pathlib.Path) -> None:
    trace = {
        "traceEvents": [
            {
                "name": "a.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "device=0;queue_kind=compute;device_start_ns=1000;device_end_ns=2000",
                },
            },
            {
                "name": "b.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "device=0;queue_kind=compute;device_start_ns=7000;device_end_ns=9000",
                },
            },
            {
                "name": "a.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "device=0;queue_kind=compute;device_start_ns=10000;device_end_ns=11000",
                },
            },
            {
                "name": "b.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "device=0;queue_kind=compute;device_start_ns=15000;device_end_ns=16000",
                },
            },
            {
                "name": "copy.kernel",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "device=1;queue_kind=copy;device_start_ns=2000;device_end_ns=3000",
                },
            },
        ]
    }
    path = tmp_path / "gap-transition-timeline.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = run_parser(path, "--wall-ms", "1", "--top-gaps", "10")

    assert result.returncode == 0, result.stdout
    assert "gap.device0.compute.count 3" in result.stdout
    assert "gap_transition.device0.compute.a.kernel--to--b.kernel.count 2" in result.stdout
    assert "gap_transition.device0.compute.a.kernel--to--b.kernel.total_ms_x1000 9" in result.stdout
    assert "gap_transition.device0.compute.a.kernel--to--b.kernel.max_ms_x1000 5" in result.stdout
    assert "gap_transition.device0.compute.b.kernel--to--a.kernel.count 1" in result.stdout
    assert "gap_transition.device0.compute.b.kernel--to--a.kernel.total_ms_x1000 1" in result.stdout
    assert result.stdout.index("gap_transition.device0.compute.a.kernel--to--b.kernel.count") < result.stdout.index(
        "gap_transition.device0.compute.b.kernel--to--a.kernel.count"
    )
    assert "gap_transition.device1.copy" not in result.stdout


def test_parser_reports_host_gap_node_overlaps(tmp_path: pathlib.Path) -> None:
    trace = {
        "traceEvents": [
            {
                "name": "first.kernel",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 1000,
                "dur": 100,
                "args": {"file": "a.cpp", "line": 1, "function": "first", "metadata": "event_id=1"},
            },
            {
                "name": "compute_forward_node",
                "cat": "ggml.op",
                "ph": "X",
                "ts": 1200,
                "dur": 300,
                "args": {"metadata": "device=0;op=ADD_ID;tensor=x;node_idx=7;nodes=9"},
            },
            {
                "name": "compute_forward_node",
                "cat": "ggml.op",
                "ph": "X",
                "ts": 1400,
                "dur": 500,
                "args": {"metadata": "device=0;op=ROPE;tensor=y;node_idx=8;nodes=9"},
            },
            {
                "name": "compute_forward_node",
                "cat": "ggml.op",
                "ph": "X",
                "ts": 2200,
                "dur": 100,
                "args": {"metadata": "device=0;op=NO_OVERLAP;tensor=z;node_idx=9;nodes=9"},
            },
            {
                "name": "second.kernel",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 2000,
                "dur": 100,
                "args": {"file": "b.cpp", "line": 2, "function": "second", "metadata": "event_id=2"},
            },
        ]
    }
    path = tmp_path / "host-overlap-timeline.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = run_parser(path, "--wall-ms", "2", "--top-host-gap-overlaps", "10")

    assert result.returncode == 0, result.stdout
    rope_line = "host_gap_overlap.first.kernel--to--second.kernel.ROPE.host_ms_x1000 500"
    add_line = "host_gap_overlap.first.kernel--to--second.kernel.ADD_ID.host_ms_x1000 300"
    assert rope_line in result.stdout
    assert add_line in result.stdout
    assert result.stdout.index(rope_line) < result.stdout.index(add_line)
    assert "NO_OVERLAP" not in result.stdout


def test_parser_reports_gap_classification_buckets(tmp_path: pathlib.Path) -> None:
    trace = {
        "traceEvents": [
            {
                "name": "prev.host",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 1000,
                "dur": 100,
                "args": {"metadata": "event_id=1"},
            },
            {
                "name": "next.host",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 1120,
                "dur": 10,
                "args": {"metadata": "event_id=2"},
            },
            {
                "name": "next.serial",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 1200,
                "dur": 10,
                "args": {"metadata": "event_id=3;depends_on=2"},
            },
            {
                "name": "next.idle",
                "cat": "sycl.submit",
                "ph": "X",
                "ts": 1300,
                "dur": 10,
                "args": {"metadata": "event_id=4"},
            },
            {
                "name": "compute_forward_node",
                "cat": "ggml.op",
                "ph": "X",
                "ts": 1100,
                "dur": 10,
                "args": {"metadata": "device=0;op=MUL_MAT_ID;tensor=x;node_idx=1;nodes=4"},
            },
            {
                "name": "prev.host",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "event_id=1;device=0;queue_kind=compute;device_start_ns=0;device_end_ns=1000",
                },
            },
            {
                "name": "next.host",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "event_id=2;device=0;queue_kind=compute;device_start_ns=11000;device_end_ns=12000",
                },
            },
            {
                "name": "next.serial",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "event_id=3;depends_on=2;device=0;queue_kind=compute;device_start_ns=33000;device_end_ns=34000",
                },
            },
            {
                "name": "next.idle",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {
                    "metadata": "event_id=4;device=0;queue_kind=compute;device_start_ns=74000;device_end_ns=75000",
                },
            },
        ]
    }
    path = tmp_path / "gap-class-timeline.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = run_parser(path, "--wall-ms", "1")

    assert result.returncode == 0, result.stdout
    assert "gap.device0.compute.total_ms_x1000 71" in result.stdout
    assert "gap_class.device0.compute.host_overlap.total_ms_x1000 10" in result.stdout
    assert "gap_class.device0.compute.queue_serialization.total_ms_x1000 21" in result.stdout
    assert "gap_class.device0.compute.runtime_idle.total_ms_x1000 40" in result.stdout

    metrics = dict(line.rsplit(" ", 1) for line in result.stdout.splitlines() if " " in line)
    assert sum(
        int(metrics[f"gap_class.device0.compute.{gap_class}.total_ms_x1000"])
        for gap_class in ("host_overlap", "queue_serialization", "runtime_idle")
    ) == int(metrics["gap.device0.compute.total_ms_x1000"])


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
