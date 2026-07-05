#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts" / "resolve-sycl-pc-samples-to-source-lines.py"


def run_resolver(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RESOLVER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_dwarf(tmp: pathlib.Path) -> pathlib.Path:
    p = tmp / "zebin-debug-line.txt"
    p.write_text(
        ".debug_line contents:\n"
        "include_directories[  1] = /Apps/llama.cpp/ggml/src/ggml-sycl\n"
        "file_names[  1]:\n"
        "           name: mmvq.cpp\n"
        "      dir_index: 1\n"
        "Address            Line   Column File   ISA Discriminator Flags\n"
        "0x0000000000000040  6800     12     1     0             0  is_stmt\n"
        "0x0000000000000080  6801     20     1     0             0  is_stmt\n",
        encoding="utf-8",
    )
    return p


def test_sample_resolver_maps_pc_samples_to_source_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = write_dwarf(tmp)
        samples = tmp / "pc-samples.csv"
        samples.write_text(
            "kernel,pc,sample_count,sample_kind\n"
            "target_kernel,64,7,cycles\n"
            "target_kernel,80,3,cycles\n",
            encoding="utf-8",
        )
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--pc-samples",
            str(samples),
            "--source-computing-task",
            "target_kernel",
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert rows[0]["Source Attribution Mode"] == "sampled-pc-line"
        assert rows[0]["Source Attribution Status"] == "sampled_line_cost"
        assert rows[0]["Sample Count"] == "7"
        assert rows[0]["Source Line"].endswith("mmvq.cpp:6800")


def test_sample_resolver_requires_sample_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = write_dwarf(tmp)
        samples = tmp / "pc-samples.csv"
        samples.write_text("kernel,pc,sample_count\ntarget_kernel,64,7\n", encoding="utf-8")
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--pc-samples",
            str(samples),
            "--source-computing-task",
            "target_kernel",
        )
        assert result.returncode == 2
        assert "missing required columns: sample_kind" in result.stdout
        assert "Traceback" not in result.stdout


def test_sample_resolver_rejects_zero_sample_count() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        dwarf = write_dwarf(tmp)
        samples = tmp / "pc-samples.csv"
        samples.write_text("kernel,pc,sample_count,sample_kind\ntarget_kernel,64,0,cycles\n", encoding="utf-8")
        result = run_resolver(
            "--dwarf-line-dump",
            str(dwarf),
            "--pc-samples",
            str(samples),
            "--source-computing-task",
            "target_kernel",
        )
        assert result.returncode == 2
        assert "sample_count must be positive" in result.stdout
        assert "Traceback" not in result.stdout
