#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-ur-trace.py"


def run_parser(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PARSER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_ur_parser_buckets_runtime_api_time() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "ur.log"
        path.write_text(
            "UR_TRACE name=urEnqueueKernelLaunch begin_us=0 end_us=12\n"
            "UR_TRACE name=urEnqueueMemBufferWrite dur_us=8\n"
            "UR_TRACE name=urEventWait begin_us=30 end_us=35\n"
            "UR_TRACE name=urProgramBuild begin_us=40 end_us=43\n",
            encoding="utf-8",
        )
        result = run_parser(path)
        assert result.returncode == 0, result.stdout
        assert "ur.total_ms_x1000 28" in result.stdout
        assert "ur.bucket.enqueue.ms_x1000 12" in result.stdout
        assert "ur.bucket.memory.ms_x1000 8" in result.stdout
        assert "ur.bucket.wait.ms_x1000 5" in result.stdout
        assert "ur.bucket.program_kernel.ms_x1000 3" in result.stdout
        assert "ur.api.urEnqueueKernelLaunch.count 1" in result.stdout


def test_ur_parser_rejects_bad_duration_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "bad.log"
        path.write_text("UR_TRACE name=urEventWait begin_us=9 end_us=4\n", encoding="utf-8")
        result = run_parser(path)
        assert result.returncode == 2
        assert "failed to parse UR trace" in result.stdout
        assert "Traceback" not in result.stdout
