#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts" / "resolve-sycl-zebin-asm-source-lines.py"


def run_resolver(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RESOLVER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_fixture(tmp: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    dwarf = tmp / "zebin-debug-line.txt"
    dwarf.write_text(
        "\n".join(
            [
                ".debug_line contents:",
                "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl",
                "file_names[  1]:",
                "           name: mmvq.cpp",
                "      dir_index: 1",
                "Address            Line   Column File   ISA Discriminator Flags",
                "------------------ ------ ------ ------ --- ------------- -------------",
                "0x0000000000000040  6800     12     1     0             0  is_stmt",
                "0x0000000000000080  6801     20     1     0             0  is_stmt",
                "0x00000000000000c0  6802     28     1     0             0  is_stmt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asm = tmp / "kernel.asm"
    asm.write_text(
        "\n".join(
            [
                "// Kernel: mxfp4_pair_glu_xmx_tiled",
                "0x00000020: add (1|M0) r0:d r0:d r0:d",
                "0x00000040: dpas.8x8 (16|M0) r28:d null:d r52:b r24.0:b",
                "0x00000050: send.ugm (1|M0) r52 r49 null:0 0x0 0x0240F580 // wr:1+0, rd:4; load.ugm.d32x64t.a64",
                "0x00000080: math.exp (1|M0) r1.2<1>:f r1.2<0;1,0>:f",
                "0x000000c0: add (1|M0) r2:d r3:d r4:d",
                "0x000000c8: send.ugm (1|M0) null r6 r62:1 0x0 0x04000584 // wr:2+1, rd:0; store.ugm.d32.a64",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dwarf, asm


def test_resolver_maps_asm_addresses_to_dwarf_source_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, asm = write_fixture(tmp)
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["Source Line"] for row in rows] == [
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800",
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6802",
            "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6801",
        ]
        first = rows[0]
        assert first["Source Computing Task"] == "mxfp4_pair_glu_xmx_tiled"
        assert first["kernel"] == "mxfp4_pair_glu_xmx_tiled"
        assert first["source_file"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp"
        assert first["source_line"] == "6800"
        assert first["Static Instruction Count"] == "2"
        assert first["instruction_count"] == "2"
        assert first["Static Dpas Count"] == "1"
        assert first["Static Send Ugm Count"] == "1"
        assert first["Static Score"] == "14"
        assert first["static_score"] == "14"
        assert first["Source Attribution Mode"] == "asm-line-static"
        assert first["Source Attribution Status"] == "asm_line_static_cost"
        final_row = rows[1]
        assert final_row["Source Line"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6802"
        assert final_row["Static Instruction Count"] == "2"
        assert final_row["Static Send Ugm Count"] == "1"
        assert final_row["Static Score"] == "6"


def test_resolver_aggregates_same_file_and_line_across_different_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = tmp / "zebin-debug-line.txt"
        dwarf.write_text(
            "\n".join(
                [
                    ".debug_line contents:",
                    "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl",
                    "file_names[  1]:",
                    "           name: mmvq.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040  6800     12     1     0             0  is_stmt",
                    "0x0000000000000080  6800     44     1     0             0  is_stmt",
                    "0x00000000000000c0  6801      1     1     0             0  is_stmt",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        asm = tmp / "kernel.asm"
        asm.write_text(
            "// Kernel: mxfp4_pair_glu_xmx_tiled\n"
            "0x00000040: dpas.8x8 (16|M0) r28:d null:d r52:b r24.0:b\n"
            "0x00000080: math.exp (1|M0) r1.2<1>:f r1.2<0;1,0>:f\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == 1
        assert rows[0]["Source Line"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800"
        assert rows[0]["Column"] == "12"
        assert rows[0]["Static Instruction Count"] == "2"
        assert rows[0]["Static Dpas Count"] == "1"
        assert rows[0]["Static Math Count"] == "1"
        assert rows[0]["Static Score"] == "12"


def test_resolver_ignores_unknown_and_out_of_range_instruction_addresses() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, asm = write_fixture(tmp)
        summary = tmp / "asm-source-lines.parse"
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--summary-output",
            str(summary),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        rows_by_source_line = {row["Source Line"]: row for row in rows}
        final_row = rows_by_source_line["/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6802"]
        assert final_row["Static Instruction Count"] == "2"
        assert final_row["Static Send Ugm Count"] == "1"
        summary_text = summary.read_text(encoding="utf-8")
        assert "asm_source.status ok" in summary_text
        assert "asm_source.mapped_instruction_count 5" in summary_text
        assert "asm_source.unmapped_instruction_count 1" in summary_text
        assert "asm_source.source_line_rows 3" in summary_text
        assert "asm_source.top_source_line /Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800" in summary_text
        assert "asm_source.top_static_score 14" in summary_text


def test_resolver_filters_to_required_source_path_and_writes_output_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, asm = write_fixture(tmp)
        out_csv = tmp / "asm-source-lines.csv"
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--output",
            str(out_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
            "--require-source-path",
            "mmvq.cpp",
        )
        assert result.returncode == 0, result.stdout
        assert result.stdout == ""
        rows = list(csv.DictReader(io.StringIO(out_csv.read_text(encoding="utf-8"))))
        assert len(rows) == 3
        assert all(row["Source Computing Task"] == "mxfp4_pair_glu_xmx_tiled" for row in rows)


def test_resolver_filters_dwarf_and_asm_sections_by_source_computing_task() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = tmp / "zebin-debug-line.txt"
        dwarf.write_text(
            "\n".join(
                [
                    "// Kernel: selected_kernel",
                    ".debug_line contents:",
                    "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl",
                    "file_names[  1]:",
                    "           name: selected.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040  7000     12     1     0             0  is_stmt",
                    "0x0000000000000080  7001     20     1     0             0  is_stmt",
                    "// Kernel: other_kernel",
                    ".debug_line contents:",
                    "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl",
                    "file_names[  1]:",
                    "           name: other.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040  8000     12     1     0             0  is_stmt",
                    "0x0000000000000080  8001     20     1     0             0  is_stmt",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        asm = tmp / "kernels.asm"
        asm.write_text(
            "\n".join(
                [
                    "// Kernel: selected_kernel",
                    "0x00000040: dpas.8x8 (16|M0) r28:d null:d r52:b r24.0:b",
                    "0x00000050: send.ugm (1|M0) r52 r49 null:0 0x0 0x0240F580 // wr:1+0, rd:4; load.ugm.d32x64t.a64",
                    "// Kernel: other_kernel",
                    "0x00000040: math.exp (1|M0) r1.2<1>:f r1.2<0;1,0>:f",
                    "0x00000050: add (1|M0) r2:d r3:d r4:d",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--source-computing-task",
            "selected_kernel",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == 1
        assert rows[0]["Source Line"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/selected.cpp:7000"
        assert rows[0]["Source Computing Task"] == "selected_kernel"
        assert rows[0]["kernel"] == "selected_kernel"
        assert rows[0]["Static Instruction Count"] == "2"
        assert rows[0]["Static Dpas Count"] == "1"
        assert rows[0]["Static Send Ugm Count"] == "1"
        assert rows[0]["Static Math Count"] == "0"
        assert rows[0]["Static Score"] == "14"
        assert "other.cpp" not in result.stdout


def test_resolver_matches_source_computing_task_exactly_not_by_substring() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = tmp / "zebin-debug-line.txt"
        dwarf.write_text(
            "\n".join(
                [
                    "// Kernel: foo",
                    ".debug_line contents:",
                    "include_directories[  1] = /tmp/src",
                    "file_names[  1]:",
                    "           name: foo.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040   100      1     1     0             0  is_stmt",
                    "// Kernel: foobar",
                    ".debug_line contents:",
                    "include_directories[  1] = /tmp/src",
                    "file_names[  1]:",
                    "           name: foobar.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040   200      1     1     0             0  is_stmt",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        asm = tmp / "kernels.asm"
        asm.write_text(
            "\n".join(
                [
                    "// Kernel: foo",
                    "0x00000040: add (1|M0) r2:d r3:d r4:d",
                    "// Kernel: foobar",
                    "0x00000040: send.ugm (1|M0) null r6 r62:1 0x0 0x04000584 // wr:2+1, rd:0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--source-computing-task",
            "foo",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == 1
        assert rows[0]["Source Line"] == "/tmp/src/foo.cpp:100"
        assert "foobar.cpp" not in result.stdout


def test_resolver_writes_empty_csv_and_no_match_summary_for_missing_task() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = tmp / "zebin-debug-line.txt"
        dwarf.write_text(
            "\n".join(
                [
                    "// Kernel: selected_kernel",
                    ".debug_line contents:",
                    "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl",
                    "file_names[  1]:",
                    "           name: selected.cpp",
                    "      dir_index: 1",
                    "Address            Line   Column File   ISA Discriminator Flags",
                    "0x0000000000000040  7000     12     1     0             0  is_stmt",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        asm = tmp / "kernels.asm"
        asm.write_text(
            "// Kernel: selected_kernel\n0x00000040: dpas.8x8 (16|M0) r28:d null:d r52:b r24.0:b\n",
            encoding="utf-8",
        )
        summary = tmp / "asm-source-lines.parse"
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--summary-output",
            str(summary),
            "--source-computing-task",
            "missing_kernel",
        )
        assert result.returncode == 2, result.stdout
        assert result.stdout.startswith("Source Line,Source File")
        assert "no mapped ASM source rows" in result.stdout
        assert "no_asm_source_matches" in result.stdout
        summary_text = summary.read_text(encoding="utf-8")
        assert "asm_source.status no_asm_source_matches" in summary_text
        assert "asm_source.blocker no_asm_source_matches" in summary_text
        assert "asm_source.mapped_instruction_count 0" in summary_text
        assert "asm_source.unmapped_instruction_count 0" in summary_text
        assert "Traceback" not in result.stdout


def test_resolver_writes_empty_csv_and_no_match_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        asm = tmp / "kernel.asm"
        asm.write_text(
            "// Kernel: mxfp4_pair_glu_xmx_tiled\n"
            "0x00000010: dpas.8x8 r1:d null:d r2:b r3:b\n",
            encoding="utf-8",
        )
        summary = tmp / "asm-source-lines.parse"
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--summary-output",
            str(summary),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2, result.stdout
        assert result.stdout.startswith("Source Line,Source File")
        assert "no mapped ASM source rows" in result.stdout
        assert "no_asm_source_matches" in result.stdout
        summary_text = summary.read_text(encoding="utf-8")
        assert "asm_source.status no_asm_source_matches" in summary_text
        assert "asm_source.blocker no_asm_source_matches" in summary_text
        assert "asm_source.mapped_instruction_count 0" in summary_text
        assert "asm_source.unmapped_instruction_count 1" in summary_text
        assert "Traceback" not in result.stdout


def test_resolver_rejects_unmarked_asm_when_task_is_required() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        asm = tmp / "kernel.asm"
        asm.write_text("0x00000040: add (1|M0) r2:d r3:d r4:d\n", encoding="utf-8")
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "ASM does not identify required source-computing-task mxfp4_pair_glu_xmx_tiled" in result.stdout
        assert "Traceback" not in result.stdout


def test_resolver_fails_closed_on_missing_required_input() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(tmp / "missing.asm"),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "failed to resolve ZEBin ASM source lines: ASM file does not exist" in result.stdout
        assert "Traceback" not in result.stdout


def test_resolver_maps_iga_pc_rows_to_dwarf_source_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        iga_csv = tmp / "iga-pc.csv"
        iga_csv.write_text(
            "kernel,pc,pc_hex,opcode,text,raw,send_comment,source\n"
            "mxfp4_pair_glu_xmx_tiled,0,0x0,dpas.8x8,dpas.8x8 r1 r2 r3,raw,,iga-json\n"
            "mxfp4_pair_glu_xmx_tiled,16,0x10,send.ugm,send.ugm r4 r5,raw,wr:1+0; rd:4,iga-json\n"
            "mxfp4_pair_glu_xmx_tiled,136,0x88,send.ugm,send.ugm r6 r7,raw,wr:1+0; rd:0,iga-json\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--iga-instructions-csv",
            str(iga_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
            "--pc-base",
            "0x40",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert rows[0]["Source Line"] == "/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:6800"
        assert rows[0]["Static Dpas Count"] == "1"
        assert rows[0]["Static Send Ugm Count"] == "1"
        assert any(row["Source Line"].endswith("mmvq.cpp:6802") for row in rows)


def test_resolver_rejects_iga_rows_for_wrong_kernel() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        iga_csv = tmp / "iga-pc.csv"
        iga_csv.write_text(
            "kernel,pc,pc_hex,opcode,text,raw,send_comment,source\n"
            "mxfp4_pair_glu_xmx_tiled,0,0x0,dpas.8x8,dpas.8x8 r1 r2 r3,raw,,iga-json\n"
            "other_kernel,16,0x10,send.ugm,send.ugm r4 r5,raw,wr:1+0; rd:4,iga-json\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--iga-instructions-csv",
            str(iga_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert (
            "IGA PC instruction CSV contains kernel other_kernel but expected mxfp4_pair_glu_xmx_tiled"
            in result.stdout
        )
        assert "Traceback" not in result.stdout


def test_resolver_rejects_iga_rows_with_no_source_matches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, _ = write_fixture(tmp)
        iga_csv = tmp / "iga-pc.csv"
        iga_csv.write_text(
            "kernel,pc,pc_hex,opcode,text,raw,send_comment,source\n"
            "mxfp4_pair_glu_xmx_tiled,0,0x0,dpas.8x8,dpas.8x8 r1 r2 r3,raw,,iga-json\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--iga-instructions-csv",
            str(iga_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "no mapped ASM source rows (no_asm_source_matches)" in result.stdout
        assert "Traceback" not in result.stdout


def test_resolver_rejects_both_asm_and_iga_instruction_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf, asm = write_fixture(tmp)
        iga_csv = tmp / "iga-pc.csv"
        iga_csv.write_text(
            "kernel,pc,pc_hex,opcode,text,raw,send_comment,source\n"
            "mxfp4_pair_glu_xmx_tiled,0,0x0,dpas.8x8,dpas.8x8 r1 r2 r3,raw,,iga-json\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--asm",
            str(asm),
            "--iga-instructions-csv",
            str(iga_csv),
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled",
        )
        assert result.returncode == 2
        assert "pass exactly one of --asm or --iga-instructions-csv" in result.stdout
        assert "Traceback" not in result.stdout
