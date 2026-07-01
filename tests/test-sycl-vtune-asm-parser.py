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
