#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-layer-ledger.py"


def run_parser(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PARSER), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def write_inputs(tmp: pathlib.Path) -> dict[str, pathlib.Path]:
    timeline = tmp / "timeline.json"
    timeline.write_text(json.dumps({"traceEvents": [
        {"name":"graph", "cat":"ggml.graph", "ph":"X", "ts":0, "dur":1000, "args":{}},
        {"name":"mxfp4.gateup", "cat":"sycl.submit", "ph":"X", "ts":100, "dur":20, "args":{"metadata":"event_id=1"}},
        {"name":"mxfp4.gateup", "cat":"sycl.event", "ph":"X", "ts":0, "dur":1, "args":{"metadata":"event_id=1;device=0;queue_kind=compute;device_start_ns=0;device_end_ns=300000"}},
    ]}), encoding="utf-8")
    kernels = tmp / "kernels.csv"
    kernels.write_text("name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded\nmxfp4.gateup,mmvq,,0,compute,1,300000,300000,300000,300000,300000,300000,0,0,0\n", encoding="utf-8")
    l0 = tmp / "l0.parse"
    l0.write_text("l0.total_ms_x1000 80\nl0.bucket.queue_submit.ms_x1000 70\n", encoding="utf-8")
    ur = tmp / "ur.parse"
    ur.write_text("ur.total_ms_x1000 50\nur.bucket.enqueue.ms_x1000 45\n", encoding="utf-8")
    vtune = tmp / "vtune.parse"
    vtune.write_text(
        "vtune.kernel_total_ms_x1000 310\n"
        "vtune.api_total_ms_x1000 120\n"
        "vtune.source.known_rows 0\n"
        "vtune.kernel.rank.1.name kernel name with spaces\n",
        encoding="utf-8",
    )
    stderr = tmp / "bench.stderr"
    stderr.write_text("[SYCL-E2E-TG-STAGE] stage=moe calls=72 host=200.000 ms device=0.000 ms bytes=0 last_path=MUL_MAT_ID\n", encoding="utf-8")
    return {"timeline": timeline, "kernels": kernels, "l0": l0, "ur": ur, "vtune": vtune, "stderr": stderr}


def test_layer_ledger_closes_wall_time_with_explicit_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_inputs(pathlib.Path(tmp_raw))
        result = run_parser(
            "--timeline", str(p["timeline"]),
            "--kernel-profile", str(p["kernels"]),
            "--l0-summary", str(p["l0"]),
            "--ur-summary", str(p["ur"]),
            "--vtune-summary", str(p["vtune"]),
            "--bench-stderr", str(p["stderr"]),
        )
        assert result.returncode == 0, result.stdout
        assert "layer.wall_ms_x1000 1000" in result.stdout
        assert "layer.app_host_ms_x1000 200000" not in result.stdout
        assert "layer.app_host_ms_x1000 200" in result.stdout
        assert "layer.sycl_submit_host_ms_x1000 20" in result.stdout
        assert "layer.ur_api_ms_x1000 50" in result.stdout
        assert "layer.level_zero_api_ms_x1000 80" in result.stdout
        assert "layer.gpu_kernel_ms_x1000 300" in result.stdout
        assert "layer.vtune_gpu_ms_x1000 310" in result.stdout
        assert "layer.unknown_wall_ms_x1000 550" in result.stdout
        assert "coverage.layer_status ok" in result.stdout


def test_layer_ledger_reports_missing_optional_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_inputs(pathlib.Path(tmp_raw))
        result = run_parser("--timeline", str(p["timeline"]), "--kernel-profile", str(p["kernels"]))
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status missing_layers" in result.stdout
        assert "coverage.missing_layer l0" in result.stdout
        assert "coverage.missing_layer ur" in result.stdout
        assert "coverage.missing_layer vtune" in result.stdout
