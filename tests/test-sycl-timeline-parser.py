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
                "name": "device event",
                "cat": "sycl.event",
                "ph": "X",
                "ts": 3000,
                "dur": 5000,
                "args": {
                    "file": "ggml/src/ggml-sycl/ggml-sycl.cpp",
                    "line": 140,
                    "function": "device_event",
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
    assert "category.ggml.graph.host_ms_x1000 10000" in result.stdout
    assert "category.sycl.submit.host_ms_x1000 100" in result.stdout
    assert "category.sycl.wait.host_ms_x1000 1500" in result.stdout
    assert (
        "callsite.ggml/src/ggml-sycl/ggml-sycl.cpp:78351:ggml_backend_sycl_graph_compute_impl.host_ms_x1000 10000"
        in result.stdout
    )
    assert "category.sycl.event" not in result.stdout
