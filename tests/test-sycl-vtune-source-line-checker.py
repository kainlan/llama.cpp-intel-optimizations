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


def test_checker_requires_concrete_source_line_not_only_source_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line\tSource File\tSource File Path\tSource Computing Task\n"
            "[Unknown]\tmmvq.cpp\t/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp\tmxfp4_pair_glu_xmx_tiled_packed_r8_m2\n",
            encoding="utf-8",
        )
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
        assert "source_line.vtune_no_gpu_side_trace 0" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout
        assert "Traceback" not in result.stdout


def test_checker_reports_vtune_no_gpu_side_trace_from_logs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        stdout = tmp / "probe.stdout"
        stderr = tmp / "probe.stderr"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("", encoding="utf-8")
        stdout.write_text(
            '"Trace GPU programming APIs" option was turned ON, but no GPU-side trace data was collected.\n',
            encoding="utf-8",
        )
        stderr.write_text(
            "GTPin: kernel: Not enough free registers while memory-mapped registers (SREGs) are disabled\n"
            "GTPin didn't find any kernels... Exiting without doing anything.\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            csv,
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
            "--vtune-stdout",
            str(stdout),
            "--vtune-stderr",
            str(stderr),
        )
        assert result.returncode == 2
        assert "source_line.vtune_no_gpu_side_trace 1" in result.stdout
        assert "source_line.gtpin_no_kernels 1" in result.stdout
        assert "source_line.gtpin_register_pressure 1" in result.stdout
        assert "source_line.blocker vtune_no_gpu_side_trace" in result.stdout
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


def test_checker_reports_empty_dwarf_line_dump_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        dwarf = tmp / "dwarf.txt"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        dwarf.write_text("", encoding="utf-8")
        result = run_checker(
            sections,
            csv,
            "--dwarf-line-dump",
            str(dwarf),
            "--require-source-path",
            "ggml/src/ggml-sycl/mmvq.cpp",
        )
        assert result.returncode == 2
        assert "source_line.dwarf_status error" in result.stdout
        assert "source_line.dwarf_error no source rows found" in result.stdout
        assert "source_line.blocker dwarf_no_source_rows_found" in result.stdout
        assert "source_line.status fail" in result.stdout
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


def test_checker_does_not_require_dwarf_source_path_when_argument_is_omitted() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        dwarf = tmp / "dwarf.txt"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\nmmvq.cpp:123\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
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
        )
        assert result.returncode == 0, result.stdout
        assert "source_line.dwarf_required_path_present 1" in result.stdout
        assert "source_line.blocker none" in result.stdout
        assert "source_line.status pass" in result.stdout


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


def test_checker_does_not_count_dwarf_line_table_csv_as_vtune_sampled_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:9730,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,dwarf-line-table,dwarf_line_table_only\n",
            encoding="utf-8",
        )
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.vtune_sampled_non_unknown_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_allows_dwarf_line_table_only_with_explicit_flag_and_no_vtune_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        dwarf_csv = tmp / "dwarf-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        dwarf_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:9730,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,dwarf-line-table,dwarf_line_table_only\n",
            encoding="utf-8",
        )
        result = subprocess.run(
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
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.vtune_sampled_non_unknown_rows 0" in result.stdout
        assert "source_line.dwarf_source_line_rows 1" in result.stdout
        assert "source_line.allow_dwarf_line_table_only 1" in result.stdout
        assert "source_line.source_attribution_mode dwarf-line-table" in result.stdout
        assert "source_line.blocker none" in result.stdout
        assert "source_line.status dwarf-line-table-only" in result.stdout


def test_checker_requires_explicit_allow_for_dwarf_line_table_only_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        dwarf_csv = tmp / "dwarf-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        dwarf_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:9730,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,dwarf-line-table,dwarf_line_table_only\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            csv,
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
            "--dwarf-source-lines-csv",
            str(dwarf_csv),
        )
        assert result.returncode == 2
        assert "source_line.dwarf_source_line_rows 1" in result.stdout
        assert "source_line.allow_dwarf_line_table_only 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.blocker vtune_unknown_source" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_allows_asm_line_static_cost_with_explicit_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        asm_csv = tmp / "asm-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        asm_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Static Instruction Count,Static Score,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2,14,asm-line-static,asm_line_static_cost\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--asm-source-lines-csv",
                str(asm_csv),
                "--allow-asm-line-static-cost",
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.vtune_sampled_non_unknown_rows 0" in result.stdout
        assert "source_line.asm_source_line_rows 1" in result.stdout
        assert "source_line.allow_asm_line_static_cost 1" in result.stdout
        assert "source_line.asm_top_source_line /Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800" in result.stdout
        assert "source_line.asm_top_static_score 14" in result.stdout
        assert "source_line.asm_top_instruction_count 2" in result.stdout
        assert "source_line.source_attribution_mode asm-line-static" in result.stdout
        assert "source_line.blocker none" in result.stdout
        assert "source_line.status asm-line-static-cost" in result.stdout


def test_checker_requires_explicit_allow_for_asm_line_static_cost() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        vtune_csv = tmp / "vtune-source.csv"
        asm_csv = tmp / "asm-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        vtune_csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        asm_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Static Instruction Count,Static Score,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2,14,asm-line-static,asm_line_static_cost\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            vtune_csv,
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
            "--asm-source-lines-csv",
            str(asm_csv),
        )
        assert result.returncode == 2
        assert "source_line.asm_source_line_rows 1" in result.stdout
        assert "source_line.allow_asm_line_static_cost 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_missing_optional_asm_csv_does_not_block_dwarf_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        dwarf_csv = tmp / "dwarf-source.csv"
        missing_asm_csv = tmp / "missing-asm-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        dwarf_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,dwarf-line-table,dwarf_line_table_only\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--asm-source-lines-csv",
                str(missing_asm_csv),
                "--allow-asm-line-static-cost",
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
        assert result.returncode == 0, result.stdout
        assert "source_line.asm_source_line_rows 0" in result.stdout
        assert "source_line.dwarf_source_line_rows 1" in result.stdout
        assert "source_line.status dwarf-line-table-only" in result.stdout


def test_checker_prefers_asm_static_cost_over_dwarf_line_table_only() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        asm_csv = tmp / "asm-source.csv"
        dwarf_csv = tmp / "dwarf-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        asm_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Static Instruction Count,Static Score,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2,14,asm-line-static,asm_line_static_cost\n",
            encoding="utf-8",
        )
        dwarf_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,dwarf-line-table,dwarf_line_table_only\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--asm-source-lines-csv",
                str(asm_csv),
                "--allow-asm-line-static-cost",
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
        assert result.returncode == 0, result.stdout
        assert "source_line.dwarf_source_line_rows 1" in result.stdout
        assert "source_line.asm_source_line_rows 1" in result.stdout
        assert "source_line.source_attribution_mode asm-line-static" in result.stdout
        assert "source_line.status asm-line-static-cost" in result.stdout
        assert "source_line.status dwarf-line-table-only" not in result.stdout


def test_checker_accepts_gtpin_bbl_runtime_cost_with_explicit_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        gtpin_csv = tmp / "gtpin-bbl-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        gtpin_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,720,gtpin-bbl-instruction-exec-count,gtpin-bbl-line,gtpin_bbl_runtime_cost,720,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--gtpin-bbl-source-lines-csv",
                str(gtpin_csv),
                "--allow-gtpin-bbl-runtime-cost",
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.vtune_sampled_non_unknown_rows 0" in result.stdout
        assert "source_line.gtpin_bbl_source_line_rows 1" in result.stdout
        assert "source_line.allow_gtpin_bbl_runtime_cost 1" in result.stdout
        assert "source_line.gtpin_bbl_top_source_line /Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233" in result.stdout
        assert "source_line.gtpin_bbl_top_sample_count 720" in result.stdout
        assert "source_line.source_attribution_mode gtpin-bbl-line" in result.stdout
        assert "source_line.status gtpin-bbl-runtime-cost" in result.stdout


def test_checker_accepts_pti_instcount_runtime_cost_with_explicit_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        pti_csv = tmp / "pti-instcount-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        pti_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2073600,pti-instcount-instruction-exec-count,pti-instcount-line,pti_instcount_runtime_cost,2073600,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--pti-instcount-source-lines-csv",
                str(pti_csv),
                "--allow-pti-instcount-runtime-cost",
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.pti_instcount_source_line_rows 1" in result.stdout
        assert "source_line.allow_pti_instcount_runtime_cost 1" in result.stdout
        assert "source_line.pti_instcount_top_source_line /Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233" in result.stdout
        assert "source_line.pti_instcount_top_sample_count 2073600" in result.stdout
        assert "source_line.source_attribution_mode pti-instcount-line" in result.stdout
        assert "source_line.status pti-instcount-runtime-cost" in result.stdout


def test_checker_requires_explicit_allow_for_pti_instcount_runtime_cost() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        vtune_csv = tmp / "vtune.csv"
        pti_csv = tmp / "pti-instcount-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        vtune_csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        pti_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2073600,pti-instcount-instruction-exec-count,pti-instcount-line,pti_instcount_runtime_cost,2073600,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            vtune_csv,
            "--pti-instcount-source-lines-csv",
            str(pti_csv),
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "source_line.pti_instcount_source_line_rows 1" in result.stdout
        assert "source_line.allow_pti_instcount_runtime_cost 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_requires_explicit_allow_for_gtpin_bbl_runtime_cost() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        vtune_csv = tmp / "vtune.csv"
        gtpin_csv = tmp / "gtpin-bbl-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        vtune_csv.write_text("Source Line\tSource Computing Task\n[Unknown]\tmxfp4_pair_glu_xmx_tiled\n", encoding="utf-8")
        gtpin_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:7233,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,720,gtpin-bbl-instruction-exec-count,gtpin-bbl-line,gtpin_bbl_runtime_cost,720,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = run_checker(
            sections,
            vtune_csv,
            "--gtpin-bbl-source-lines-csv",
            str(gtpin_csv),
            "--require-kernel",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "source_line.gtpin_bbl_source_line_rows 1" in result.stdout
        assert "source_line.allow_gtpin_bbl_runtime_cost 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_does_not_promote_gtpin_bbl_csv_to_vtune_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "mmvq.cpp:7233,mxfp4_pair_glu_xmx_tiled,gtpin-bbl-line,gtpin_bbl_runtime_cost\n",
            encoding="utf-8",
        )
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_accepts_sampled_pc_line_cost_from_positive_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        sampled_csv = tmp / "sampled-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        sampled_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,7,cycles,sampled-pc-line,sampled_line_cost,7,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--sampled-source-lines-csv",
                str(sampled_csv),
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.vtune_sampled_non_unknown_rows 0" in result.stdout
        assert "source_line.sampled_source_line_rows 1" in result.stdout
        assert "source_line.sampled_top_sample_count 7" in result.stdout
        assert "source_line.source_attribution_mode sampled-pc-line" in result.stdout
        assert "source_line.status sampled-line-cost" in result.stdout


def test_checker_rejects_sampled_pc_line_cost_without_positive_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        sampled_csv = tmp / "sampled-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        sampled_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,0,cycles,sampled-pc-line,sampled_line_cost,0,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--sampled-source-lines-csv",
                str(sampled_csv),
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 2
        assert "source_line.sampled_source_line_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_rejects_contradictory_sampled_pc_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        sampled_csv = tmp / "sampled-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        sampled_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Sample Count,Sample Kind,Source Attribution Mode,Source Attribution Status,sample_count,kernel\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,7,cycles,sampled-pc-line,asm_line_static_cost,7,mxfp4_pair_glu_xmx_tiled\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--sampled-source-lines-csv",
                str(sampled_csv),
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 2
        assert "source_line.sampled_source_line_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_does_not_promote_contradictory_sampled_pc_vtune_row_to_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        csv = tmp / "source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        csv.write_text(
            "Source Line,Source Computing Task,Source Attribution Mode,Source Attribution Status\n"
            "mmvq.cpp:123,mxfp4_pair_glu_xmx_tiled,sampled-pc-line,asm_line_static_cost\n",
            encoding="utf-8",
        )
        result = run_checker(sections, csv, "--require-kernel", "mxfp4_pair_glu_xmx_tiled")
        assert result.returncode == 2
        assert "source_line.non_unknown_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout


def test_checker_does_not_promote_contradictory_sampled_pc_row_to_asm_static() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = tmp / "sections.txt"
        asm_csv = tmp / "asm-source.csv"
        sections.write_text("[12] .debug_line PROGBITS\n", encoding="utf-8")
        asm_csv.write_text(
            "Source Line,Source File,Source File Path,Source Computing Task,Static Dpas Count,Static Score,Source Attribution Mode,Source Attribution Status\n"
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800,mmvq.cpp,/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp,mxfp4_pair_glu_xmx_tiled,2,14,sampled-pc-line,asm_line_static_cost\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--readelf-sections",
                str(sections),
                "--asm-source-lines-csv",
                str(asm_csv),
                "--allow-asm-line-static-cost",
                "--require-kernel",
                "mxfp4_pair_glu_xmx_tiled",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 2
        assert "source_line.asm_source_line_rows 0" in result.stdout
        assert "source_line.source_attribution_mode none" in result.stdout
        assert "source_line.status fail" in result.stdout
