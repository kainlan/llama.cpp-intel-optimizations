#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "scripts" / "convert-sycl-ur-stderr.py"
PARSER = ROOT / "scripts" / "parse-sycl-ur-trace.py"


def run_converter(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CONVERTER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_converter_normalizes_real_ur_trace_lines_for_existing_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        raw = tmp / "bench.stderr"
        normalized = tmp / "ur.trace"
        raw.write_text(
            "   ---> urQueueFinish\n"
            "   <--- urQueueFinish(.hQueue = 0x4d2e970) -> UR_RESULT_SUCCESS;\n"
            "   ---> urEnqueueKernelLaunch\n"
            "   <--- urEnqueueKernelLaunch(.hQueue = 0x1) -> UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;\n",
            encoding="utf-8",
        )
        result = run_converter(raw)
        assert result.returncode == 0, result.stdout
        assert "UR_TRACE name=urQueueFinish dur_us=0 evidence=counts_only result=UR_RESULT_SUCCESS" in result.stdout
        assert "UR_TRACE name=urEnqueueKernelLaunch dur_us=0 evidence=counts_only result=UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY" in result.stdout
        normalized.write_text(result.stdout, encoding="utf-8")
        parsed = subprocess.run([sys.executable, str(PARSER), str(normalized)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert parsed.returncode == 0, parsed.stdout
        assert "ur.total_ms_x1000 0" in parsed.stdout
        assert "ur.api.urQueueFinish.count 1" in parsed.stdout
        assert "ur.api.urEnqueueKernelLaunch.count 1" in parsed.stdout


def test_converter_rejects_files_without_ur_rows_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        raw = pathlib.Path(tmp_raw) / "bench.stderr"
        raw.write_text("no unified runtime trace lines\n", encoding="utf-8")
        result = run_converter(raw)
        assert result.returncode == 2
        assert "failed to convert UR stderr" in result.stdout
        assert "no UR API rows found" in result.stdout
        assert "Traceback" not in result.stdout
