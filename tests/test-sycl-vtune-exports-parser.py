#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-vtune-exports.py"


def run_parser(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PARSER), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_vtune_parser_normalizes_kernel_api_and_source_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        kernels = tmp / "kernels.csv"
        api = tmp / "api.csv"
        source = tmp / "source.csv"
        kernels.write_text("Computing Task,Computing Task:Total Time\nmxfp4.gateup,1.250\nfattn.compute,0.250\n", encoding="utf-8")
        api.write_text("Function,CPU Time\nzeCommandQueueExecuteCommandLists,0.400\nurEnqueueKernelLaunch,0.100\n", encoding="utf-8")
        source.write_text("Source Line,Source Computing Task\nmmvq.cpp:9730,mxfp4.gateup\n[Unknown],mxfp4.gateup\n", encoding="utf-8")
        result = run_parser("--kernel-csv", str(kernels), "--api-csv", str(api), "--source-csv", str(source), "--require-kernel", "mxfp4")
        assert result.returncode == 0, result.stdout
        assert "vtune.kernel_total_ms_x1000 1500" in result.stdout
        assert "vtune.api_total_ms_x1000 500" in result.stdout
        assert "vtune.source.known_rows 1" in result.stdout
        assert "vtune.source.unknown_rows 1" in result.stdout
        assert "vtune.kernel.rank.1.name mxfp4.gateup" in result.stdout


def test_vtune_parser_reports_malformed_csv_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "bad.csv"
        path.write_text("Computing Task,Total Time\na,1,extra\n", encoding="utf-8")
        result = run_parser("--kernel-csv", str(path))
        assert result.returncode == 2
        assert "failed to parse VTune exports" in result.stdout
        assert "Traceback" not in result.stdout
