#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-vtune-kernel-asm.py"


def run_parser(*args: str) -> dict:
    out = subprocess.check_output([sys.executable, str(PARSER), *args], text=True)
    return json.loads(out)


def test_asm_parser_counts_dpas_sends_and_send_comments() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        asm = pathlib.Path(tmp_raw) / "kernel.asm"
        asm.write_text(
            """
L0:
        send.ugm (1|M0)          r52      r49  null:0  0x0 0x0240F580 // wr:1+0, rd:4; load.ugm.d32x64t.a64
        dpas.8x8 (16|M0)         r28:d    null:d       r52:b r24.0:b
        dpas.8x8 (16|M0)         r36:d    null:d       r52:b r18.0:b
        math.exp (1|M0)          r1.2<1>:f r1.2<0;1,0>:f
        send.ugm (1|M0)          null     r6   r62:1 0x0 0x04000584 // wr:2+1, rd:0; store.ugm.d32.a64
""".strip()
            + "\n"
        )
        data = run_parser("--asm", str(asm))
        assert data["asm"]["opcodes"]["dpas.8x8"] == 2
        assert data["asm"]["opcodes"]["send.ugm"] == 2
        assert data["asm"]["opcodes"]["math.exp"] == 1
        assert data["asm"]["send_comments"]["wr:1+0, rd:4; load.ugm.d32x64t.a64"] == 1
        assert data["asm"]["send_comments"]["wr:2+1, rd:0; store.ugm.d32.a64"] == 1


def test_tsv_parser_extracts_gpu_instruction_count() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tsv = pathlib.Path(tmp_raw) / "instr.tsv"
        tsv.write_text(
            "Computing Task\tGPU Instructions Executed\tComputing Task:SIMD Width\tComputing Task:Spill Memory Size\n"
            "mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<(int)8, (int)3, (bool)0>\t100224000\t1\t0\n"
        )
        data = run_parser("--instr-tsv", str(tsv))
        assert data["instr_tsv"][0]["task"].startswith("mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel")
        assert data["instr_tsv"][0]["gpu_instructions"] == 100224000
        assert data["instr_tsv"][0]["simd_width"] == 1
        assert data["instr_tsv"][0]["spill_memory_size"] == 0


def test_asm_parser_classifies_high_send_dpas_as_memory_or_address_bound() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        asm = pathlib.Path(tmp_raw) / "kernel.asm"
        asm.write_text(
            "\n".join(
                ["send.ugm r1 r2 r3 // wr:1+0, rd:4; load.ugm.d32x64t.a64"] * 40
                + ["dpas.8x8 r4:d null:d r5:b r6:b"] * 2
            )
            + "\n",
            encoding="utf-8",
        )
        data = run_parser("--classify-bound", "--asm", str(asm))
        assert data["asm"]["bound"]["verdict"] == "memory_or_address_bound"
        assert data["asm"]["bound"]["dpas"] == 2
        assert data["asm"]["bound"]["send_ugm"] == 40


def test_asm_parser_classifies_dpas_heavy_low_send_as_compute_bound() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        asm = pathlib.Path(tmp_raw) / "kernel.asm"
        asm.write_text(
            "\n".join(["dpas.8x8 r4:d null:d r5:b r6:b"] * 16 + ["send.ugm r1 r2 r3"] * 2) + "\n",
            encoding="utf-8",
        )
        data = run_parser("--classify-bound", "--asm", str(asm))
        assert data["asm"]["bound"]["verdict"] == "compute_xmx_bound"


def test_asm_parser_classifies_send_only_as_memory_bound() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        asm = pathlib.Path(tmp_raw) / "kernel.asm"
        asm.write_text("send.ugm r1 r2 r3\nsend.ugm r4 r5 r6\n", encoding="utf-8")
        data = run_parser("--classify-bound", "--asm", str(asm))
        assert data["asm"]["bound"]["verdict"] == "memory_bound"


def test_asm_parser_emits_addressed_instruction_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        asm = pathlib.Path(tmp_raw) / "kernel.asm"
        asm.write_text(
            "\n".join(
                [
                    "0x00000040: dpas.8x8 (16|M0) r28:d null:d r52:b r24.0:b",
                    "40: mov (1|M0) r1:d r2:d",
                    "00000050: send.ugm (1|M0) r52 r49 null:0 0x0 0x0240F580 // wr:1+0, rd:4; load.ugm.d32x64t.a64",
                    "/* 0x00000058 */ math.exp (1|M0) r1.2<1>:f r1.2<0;1,0>:f",
                    "[0x00000060] add (1|M0) r2:d r3:d r4:d",
                    "send.ugm r5 r6 r7 // wr:2+1, rd:0; store.ugm.d32.a64",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        data = run_parser("--emit-instructions", "--asm", str(asm))
        assert [row["address"] for row in data["instructions"]] == [64, 64, 80, 88, 96]
        assert [row["address_hex"] for row in data["instructions"]] == ["0x40", "0x40", "0x50", "0x58", "0x60"]
        assert [row["opcode"] for row in data["instructions"]] == ["dpas.8x8", "mov", "send.ugm", "math.exp", "add"]
        assert data["instructions"][0]["raw"].startswith("0x00000040:")
        assert data["instructions"][1]["text"] == "mov (1|M0) r1:d r2:d"
        assert data["instructions"][2]["send_comment"] == "wr:1+0, rd:4; load.ugm.d32x64t.a64"
        assert data["asm"]["opcodes"]["send.ugm"] == 2
