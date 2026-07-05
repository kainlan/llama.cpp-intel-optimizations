#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-zebin-line-table.py"


def run_parser(input_text: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_zebin_line_table_reports_required_probe_source_path() -> None:
    decoded_dump = """
.debug_line contents:
include_directories[  1] = /Apps/llama.cpp/tools/sycl-source-line-probe
file_names[  1]:
           name: main.cpp
      dir_index: 1
Address            Line   Column File   ISA Discriminator Flags
------------------ ------ ------ ------ --- ------------- -------------
0x0000000000000040  150      17     1     0             0  is_stmt
"""
    result = run_parser(decoded_dump, "--require-path", "tools/sycl-source-line-probe/main.cpp")

    assert result.returncode == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
    assert data["source_rows"] == 1
    assert data["required_path_present"] is True
    assert data["files"] == ["/Apps/llama.cpp/tools/sycl-source-line-probe/main.cpp"]


def test_zebin_line_table_reports_missing_source_rows_without_traceback() -> None:
    decoded_dump = """
.debug_line contents:
include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl
file_names[  1]:
           name: mmvq.cpp
      dir_index: 1
Address            Line   Column File   ISA Discriminator Flags
------------------ ------ ------ ------ --- ------------- -------------
"""
    result = run_parser(decoded_dump, "--require-path", "mmvq.cpp")

    assert result.returncode == 2
    assert "failed to parse ZEBin line table" in result.stdout
    assert "no source rows found" in result.stdout
    assert "Traceback" not in result.stdout
