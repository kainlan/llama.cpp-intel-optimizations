#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-kernel-profile.py"

CSV_TEXT = """name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded
mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2,1,compute,2,8000000,4000000,3000000,4000000,5000000,5000000,0,0,0
mxfp4.down.q8_soa,mmvq,path=q8-soa,1,compute,5,2000000,400000,300000,400000,500000,500000,0,1,0
"""


def run_parser(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_parser_for_csv(*args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "profile.csv"
        path.write_text(CSV_TEXT, encoding="utf-8")
        return run_parser(path, *args)


def test_parser_emits_kernel_sum_total() -> None:
    result = run_parser_for_csv()

    assert result.returncode == 0, result.stdout
    assert "profile.kernel_sum_total_ms_x1000 10000" in result.stdout
    assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.count 2" in result.stdout
    assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.total_ms_x1000 8000" in result.stdout
    assert "category.mmvq.total_ms_x1000 10000" in result.stdout


def test_parser_emits_decode_wall_and_kernel_coverage() -> None:
    result = run_parser_for_csv("--wall-ms", "27.05")

    assert result.returncode == 0, result.stdout
    assert "profile.decode_wall_ms_x1000 27050" in result.stdout
    assert "profile.kernel_coverage_pct_x1000 36969" in result.stdout


def test_parser_emits_achieved_bandwidth_for_requested_kernel_bytes() -> None:
    result = run_parser_for_csv(
        "--kernel-bytes",
        "mxfp4.gateup.xmx_tiled_dpas_m2=4000000",
        "--kernel-bytes",
        "mxfp4.down.q8_soa=100000",
    )

    assert result.returncode == 0, result.stdout
    assert "kernel.mxfp4.gateup.xmx_tiled_dpas_m2.achieved_gbps_x1000 1000" in result.stdout
    assert "kernel.mxfp4.down.q8_soa.achieved_gbps_x1000 250" in result.stdout
    assert "kernel.mxfp4.down.q8_soa.failed_timestamps 1" in result.stdout


def test_parser_reports_missing_kernel_bytes_as_exit_2() -> None:
    result = run_parser_for_csv("--kernel-bytes", "missing.kernel=128")

    assert result.returncode == 2
    assert "missing bytes kernel: missing.kernel" in result.stdout
    assert "Traceback" not in result.stdout
