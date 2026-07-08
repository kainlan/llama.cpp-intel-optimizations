#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "scripts" / "convert-sycl-vtune-l0-host-tasks.py"
PARSER = ROOT / "scripts" / "parse-sycl-pti-l0.py"


def test_converter_turns_vtune_host_tasks_into_l0_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        vtune = tmp / "vtune-summary.txt"
        jsonl = tmp / "l0.jsonl"
        vtune.write_text(
            "Hottest Host Tasks\n"
            "Host Task                        Task Time  % of Elapsed Time(%)  Task Count\n"
            "-------------------------------  ---------  --------------------  ----------\n"
            "zeCommandListAppendLaunchKernel   302.028s                 68.9%           1\n"
            "zeMemAllocHost                      3.344s                  0.8%          16\n"
            "[Others]                            0.050s                  0.0%         798\n",
            encoding="utf-8",
        )
        result = subprocess.run([sys.executable, str(CONVERTER), str(vtune)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 0, result.stdout
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert rows[0]["name"] == "zeCommandListAppendLaunchKernel"
        assert rows[0]["dur_us"] == 302028000
        assert rows[0]["ts_us"] == 0
        assert rows[0]["source"] == "vtune_host_task_summary"
        jsonl.write_text(result.stdout, encoding="utf-8")
        parsed = subprocess.run([sys.executable, str(PARSER), str(jsonl)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert parsed.returncode == 0, parsed.stdout
        assert "l0.total_ms_x1000 305372000" in parsed.stdout
        assert "l0.api.zeCommandListAppendLaunchKernel.count 1" in parsed.stdout
        assert "l0.api.zeMemAllocHost.count 1" in parsed.stdout


def test_converter_rejects_missing_ze_rows_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        vtune = pathlib.Path(tmp_raw) / "vtune-summary.txt"
        vtune.write_text("Hottest Host Tasks\n[Others] 0.050s 0.0% 798\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(CONVERTER), str(vtune)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 2
        assert "failed to convert VTune L0 host tasks" in result.stdout
        assert "no Level Zero host task rows found" in result.stdout
        assert "Traceback" not in result.stdout
