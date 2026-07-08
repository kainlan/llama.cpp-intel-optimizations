#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-sycl-vtune-source-lines.py"
CONVERTER = ROOT / "scripts" / "convert-sycl-zebin-line-table-to-source-csv.py"


def run_converter(input_text: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CONVERTER), *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_converter_writes_checker_compatible_dwarf_line_table_rows() -> None:
    decoded_dump = """
.debug_line contents:
include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl
file_names[  1]:
           name: mmvq.cpp
      dir_index: 1
Address            Line   Column File   ISA Discriminator Flags
------------------ ------ ------ ------ --- ------------- -------------
0x0000000000000040  9730     17     1     0             0  is_stmt
"""
    result = run_converter(decoded_dump, "--source-computing-task", "mxfp4_pair_glu_xmx_tiled")

    assert result.returncode == 0, result.stdout
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == 1
    assert rows[0]["Source Line"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:9730"
    assert rows[0]["Source File"] == "mmvq.cpp"
    assert rows[0]["Source File Path"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp"
    assert rows[0]["Source Computing Task"] == "mxfp4_pair_glu_xmx_tiled"
    assert rows[0]["Address"] == "0x0000000000000040"
    assert rows[0]["Line"] == "9730"
    assert rows[0]["Column"] == "17"
    assert rows[0]["Source Attribution Mode"] == "dwarf-line-table"
    assert rows[0]["Source Attribution Status"] == "dwarf_line_table_only"


def test_converter_reports_empty_line_table_without_traceback() -> None:
    result = run_converter(".debug_line contents:\n")

    assert result.returncode == 2
    assert "failed to convert ZEBin line table: no source rows found" in result.stdout
    assert "Traceback" not in result.stdout


def test_converter_output_drives_checker_dwarf_line_table_only_mode() -> None:
    decoded_dump = """
.debug_line contents:
include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl
file_names[  1]:
           name: mmvq.cpp
      dir_index: 1
Address            Line   Column File   ISA Discriminator Flags
------------------ ------ ------ ------ --- ------------- -------------
0x0000000000000040  9730     17     1     0             0  is_stmt
"""
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        dwarf_csv = tmp / "dwarf-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        convert_result = run_converter(
            decoded_dump,
            "--output",
            str(dwarf_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert convert_result.returncode == 0, convert_result.stdout

        check_result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--dwarf-source-lines-csv",
                str(dwarf_csv),
                "--allow-dwarf-line-table-only",
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    assert check_result.returncode == 0, check_result.stdout
    assert "source_line.dwarf_source_line_rows 1" in check_result.stdout
    assert "source_line.source_attribution_mode dwarf-line-table" in check_result.stdout
    assert "source_line.status dwarf-line-table-only" in check_result.stdout
