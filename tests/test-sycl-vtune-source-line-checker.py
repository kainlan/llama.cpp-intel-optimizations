#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-sycl-vtune-source-lines.py"


def run_checker(sections: pathlib.Path, csv: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--readelf-sections", str(sections), "--vtune-csv", str(csv), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_checker_passes_with_debug_line_and_non_unknown_source_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\tComputing Task:Total Time\nmmvq.cpp:123\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\t0.1\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 0, result.stdout
        assert "source_line.debug_line_present 1" in result.stdout
        assert "source_line.non_unknown_rows 1" in result.stdout
        assert "source_line.required_kernel mxfp4_pair_glu_xmx_tiled" in result.stdout
        assert "source_line.blocker none" in result.stdout
        assert "source_line.status pass" in result.stdout


def test_checker_accepts_comma_separated_vtune_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line,Source Computing Task\n"
            "mmvq.cpp:123,mxfp4_pair_glu_xmx_tiled_packed_r8_m2\n",
            encoding="utf-8",
        )
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 1" in result.stdout
        assert "source_line.status pass" in result.stdout


def test_checker_fails_when_source_rows_are_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_reports_blocker_reason_for_unknown_vtune_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line\tSource File\tSource File Path\tSource Computing Task\n"
            "[Unknown]\t[Unknown source file]\t\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n",
            encoding="utf-8",
        )
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.required_kernel mxfp4_pair_glu_xmx_tiled" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_fails_without_debug_line() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .ze_info LOUSER\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\nmmvq.cpp:123\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.debug_line_present 0" in result.stdout
        assert "source_line.blocker missing_debug_line" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_does_not_treat_debug_line_str_as_debug_line() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line_str PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\nmmvq.cpp:123\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.debug_line_present 0" in result.stdout
        assert "source_line.blocker missing_debug_line" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_reports_blocker_reason_for_missing_debug_line() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .ze_info LOUSER\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\nmmvq.cpp:123\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.required_kernel mxfp4_pair_glu_xmx_tiled" in result.stdout
        assert "source_line.blocker missing_debug_line" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_reports_empty_vtune_csv_as_unknown_source_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.debug_line_present 1" in result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout
        assert "Traceback" not in result.stdout


def test_checker_reports_malformed_surplus_csv_fields_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line,Source Computing Task\nmmvq.cpp:123,mxfp4_pair_glu_xmx_tiled_packed_r8_m2,extra-field\n", encoding="utf-8")
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "failed to check source lines" in result.stdout
        assert "Traceback" not in result.stdout


def test_checker_reports_useful_dwarf_table_when_vtune_rows_are_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        dwarf = tmp / "dwarf.txt"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        dwarf.write_text(
            "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl\n"
            "file_names[  1]:\n"
            "           name: mmvq.cpp\n"
            "      dir_index: 1\n"
            "Address Line Column File ISA Discriminator Flags\n"
            "0x00000040 9730 1 1 0 0 is_stmt\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            csv,
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
            "--dwarf-line-dump",
            str(dwarf),
            "--require-source-path",
            "ggml/src/ggml-sycl/mmvq.cpp",
        )
        assert result.returncode == 2
        assert "source_line.dwarf_status ok" in result.stdout
        assert "source_line.dwarf_source_rows 1" in result.stdout
        assert "source_line.dwarf_required_path_present 1" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_reports_missing_dwarf_source_path_when_line_table_lacks_required_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        dwarf = tmp / "dwarf.txt"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        dwarf.write_text(
            "include_directories[  1] = /tmp/generated\n"
            "file_names[  1]:\n"
            "           name: generated.cpp\n"
            "      dir_index: 1\n"
            "Address Line Column File ISA Discriminator Flags\n"
            "0x00000040 1 1 1 0 0 is_stmt\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            csv,
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
            "--dwarf-line-dump",
            str(dwarf),
            "--require-source-path",
            "ggml/src/ggml-sycl/mmvq.cpp",
        )
        assert result.returncode == 2
        assert "source_line.dwarf_required_path_present 0" in result.stdout
        assert "source_line.blocker missing_dwarf_source_path" in result.stdout
        assert "source_line.status fail" in result.stdout
