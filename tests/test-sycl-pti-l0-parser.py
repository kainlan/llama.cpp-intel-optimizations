#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-pti-l0.py"


def run_parser(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PARSER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_l0_parser_buckets_driver_api_time() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "l0.jsonl"
        path.write_text(
            '{"name":"zeCommandQueueExecuteCommandLists","begin_us":10,"end_us":25}\n'
            '{"name":"zeCommandListAppendMemoryCopy","ts_us":30,"dur_us":7}\n'
            '{"name":"zeModuleCreate","start_us":40,"end_us":43}\n'
            '{"name":"zeEventHostSynchronize","begin_us":50,"end_us":55}\n',
            encoding="utf-8",
        )
        result = run_parser(path)
        assert result.returncode == 0, result.stdout
        assert "l0.total_ms_x1000 30" in result.stdout
        assert "l0.bucket.queue_submit.ms_x1000 15" in result.stdout
        assert "l0.bucket.memory.ms_x1000 7" in result.stdout
        assert "l0.bucket.module_kernel.ms_x1000 3" in result.stdout
        assert "l0.bucket.event_wait.ms_x1000 5" in result.stdout
        assert "l0.api.zeCommandQueueExecuteCommandLists.count 1" in result.stdout


def test_l0_parser_rejects_malformed_rows_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        path = pathlib.Path(tmp_raw) / "bad.jsonl"
        path.write_text('{"name":"zeCommandQueueExecuteCommandLists","begin_us":30,"end_us":10}\n', encoding="utf-8")
        result = run_parser(path)
        assert result.returncode == 2
        assert "failed to parse Level Zero trace" in result.stdout
        assert "Traceback" not in result.stdout


def test_l0_parser_rejects_bad_json_and_non_object_rows_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        bad_json = pathlib.Path(tmp_raw) / "bad-json.jsonl"
        bad_json.write_text('{"name":"zeEventHostSynchronize",\n', encoding="utf-8")
        result = run_parser(bad_json)
        assert result.returncode == 2
        assert "failed to parse Level Zero trace" in result.stdout
        assert "Traceback" not in result.stdout

        non_object = pathlib.Path(tmp_raw) / "non-object.jsonl"
        non_object.write_text('["zeEventHostSynchronize"]\n', encoding="utf-8")
        result = run_parser(non_object)
        assert result.returncode == 2
        assert "failed to parse Level Zero trace" in result.stdout
        assert "Traceback" not in result.stdout
