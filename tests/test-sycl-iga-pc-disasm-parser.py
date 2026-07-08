#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-iga-pc-disasm.py"


def run_parser(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_parser_reads_iga_json_pc_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        p = tmp / "kernel.iga.json"
        p.write_text(
            json.dumps(
                {
                    "kernels": [
                        {
                            "name": "target_kernel",
                            "elems": [
                                {"kind": "label", "id": 0, "name": "L0"},
                                {"kind": "inst", "pc": 64, "op": "dpas.8x8", "syntax": "dpas.8x8 r1 r2 r3"},
                                {"kind": "inst", "pc": "0x50", "op": "send.ugm", "syntax": "send.ugm r4 r5 // wr:1+0, rd:4"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        result = run_parser("--input", str(p), "--format", "json", "--kernel", "target_kernel")
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["pc_hex"] for row in rows] == ["0x40", "0x50"]
        assert rows[0]["opcode"] == "dpas.8x8"
        assert rows[1]["send_comment"] == "wr:1+0, rd:4"
        assert rows[0]["kernel"] == "target_kernel"
        assert rows[0]["source"] == "iga-json"


def test_parser_reads_actual_iga_json_v2_top_level_elems() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        p = tmp / "kernel.iga.json"
        p.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "platform": "xe2",
                    "elems": [
                        {"kind": "L", "id": 1, "pc": 0, "symbol": "L0000"},
                        {"kind": "I", "id": 1, "pc": 0, "op": "mov", "dst": {"kind": "RD"}},
                        {"kind": "I", "id": 2, "pc": 16, "op": "send", "srcs": []},
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = run_parser("--input", str(p), "--format", "json", "--kernel", "target_kernel")
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["pc"] for row in rows] == ["0", "16"]
        assert [row["opcode"] for row in rows] == ["mov", "send"]
        assert rows[0]["kernel"] == "target_kernel"
        assert rows[0]["source"] == "iga-json"


def test_parser_rejects_label_only_text_without_pc_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        p = tmp / "kernel.asm"
        p.write_text("L0:\n(W) mov (16|M0) r1.0<1>:ud 0x0:ud\nL304:\n", encoding="utf-8")
        result = run_parser("--input", str(p), "--format", "text", "--kernel", "target_kernel")
        assert result.returncode == 2
        assert "iga_pc.status no_pc_rows" in result.stdout
        assert "Traceback" not in result.stdout


def test_parser_reads_explicit_text_pc_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        p = tmp / "kernel.asm"
        p.write_text(
            "// Kernel: target_kernel\n"
            "0x40: dpas.8x8 r1 r2 r3\n"
            "PC=0x50 send.ugm r4 r5 // wr:1+0, rd:4\n",
            encoding="utf-8",
        )
        result = run_parser("--input", str(p), "--format", "text", "--kernel", "target_kernel")
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["pc"] for row in rows] == ["64", "80"]
        assert rows[1]["opcode"] == "send.ugm"
