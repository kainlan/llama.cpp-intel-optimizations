#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-kernel-profile.py"

CSV_TEXT = """name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded
mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2,1,compute,2,4000000,2000000,1000000,1000000,3000000,3000000,8192,0,0
fattn.pack,fattn,role=pack,1,compute,1,500000,500000,500000,500000,500000,500000,0,0,0
"""

CSV_WITH_DUPLICATE_KERNEL = """name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded
mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2,1,compute,2,4000000,2000000,1000000,1000000,3000000,3000000,8192,0,0
mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2;shape=alt,1,compute,1,1000000,1000000,1000000,1000000,1000000,1000000,4096,1,0
"""

CSV_WITH_BAD_INTEGER = """name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded
mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2,1,compute,not-an-int,4000000,2000000,1000000,1000000,3000000,3000000,8192,0,0
"""

JSON_OBJ = {
    "kernels": [
        {
            "name": "mxfp4.gateup.xmx_tiled_dpas_m2",
            "category": "mmvq",
            "metadata": "path=packed-q8-m2",
            "device": 1,
            "queue_kind": "compute",
            "count": 2,
            "total_ns": 4000000,
            "mean_ns": 2000000,
            "min_ns": 1000000,
            "p50_ns": 1000000,
            "p95_ns": 3000000,
            "max_ns": 3000000,
            "bytes": 8192,
            "failed_timestamps": 0,
            "graph_recorded": 0,
        }
    ]
}


def run_parser(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_parser_summarizes_csv_and_enforces_requirements() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.csv"
        path.write_text(CSV_TEXT, encoding="utf-8")
        result = run_parser(
            path,
            "--require-kernel",
            "mxfp4.gateup.xmx_tiled_dpas_m2",
            "--min-total-ms",
            "mxfp4.gateup.xmx_tiled_dpas_m2=3.5",
        )
        assert result.returncode == 0, result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.count 2" in result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.total_ms_x1000 4000" in result.stdout
        assert "category.mm-vq" not in result.stdout
        assert "category.mmvq.total_ms_x1000 4000" in result.stdout


def test_parser_summarizes_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.json"
        path.write_text(json.dumps(JSON_OBJ), encoding="utf-8")
        result = run_parser(path, "--require-kernel", "mxfp4.gateup.xmx_tiled_dpas_m2")
        assert result.returncode == 0, result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.count 2" in result.stdout
        assert "category.mmvq.total_ms_x1000 4000" in result.stdout


def test_parser_fails_when_required_kernel_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.csv"
        path.write_text(CSV_TEXT, encoding="utf-8")
        result = run_parser(path, "--require-kernel", "missing.kernel")
        assert result.returncode == 2
        assert "missing required kernel: missing.kernel" in result.stdout


def test_parser_aggregates_duplicate_kernel_names_for_thresholds() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.csv"
        path.write_text(CSV_WITH_DUPLICATE_KERNEL, encoding="utf-8")
        result = run_parser(path, "--min-total-ms", "mxfp4.gateup.xmx_tiled_dpas_m2=4.5")
        assert result.returncode == 0, result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.count 3" in result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.total_ms_x1000 5000" in result.stdout
        assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.failed_timestamps 1" in result.stdout


def test_parser_reports_malformed_integer_fields_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.csv"
        path.write_text(CSV_WITH_BAD_INTEGER, encoding="utf-8")
        result = run_parser(path)
        assert result.returncode == 2
        assert "failed to parse profile: invalid integer field count" in result.stdout
        assert "Traceback" not in result.stdout
