#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts" / "extract-sycl-gtpin-bbl-pc-counts.py"


def run_extractor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXTRACTOR), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def pack_u32(values: list[int]) -> bytes:
    return b"".join(struct.pack("<I", value) for value in values)


def send_descriptor_20(offset: int, payload_len: int = 0, addr_width: int = 64) -> list[int]:
    words = [0] * 20
    words[0] = offset
    words[9] = addr_width
    words[12] = payload_len
    words[17] = 1  # numOfElements
    words[18] = 1  # execSize
    return words


def write_trace(tmp: pathlib.Path) -> pathlib.Path:
    data: list[int] = []
    data.append(2)  # BBL header count
    data.extend([3, 1])  # BBL 3 has one send at PC 0x20
    data.extend(send_descriptor_20(0x20))
    data.extend([4, 0])  # BBL 4 has no sends
    data.extend([64, 0, 1])  # 64-byte GRF, no timestamps, one tile
    data.extend([0, 1])  # tile id, one profiled hardware thread
    data.extend([0, 0, 0, 0, 0])  # TID tuple
    data.append(3)  # records
    data.extend([3, 0xFFFFFFFF])
    data.extend([3, 0xFFFFFFFF])
    data.extend([4, 0xFFFFFFFF])
    path = tmp / "memorytrace_compressed.bin"
    path.write_bytes(pack_u32(data))
    return path


def write_session_asm(tmp: pathlib.Path) -> pathlib.Path:
    path = tmp / "kernel.asm"
    path.write_text(
        "// kernel name: mangled_kernel\n"
        "\n"
        "// BBL3\n"
        "[ 10] mov (1|M0) r1 r2\n"
        "[ 11] send.ugm (1|M0) r3 r4\n"
        "// BBL4\n"
        "[ 12] add (1|M0) r5 r6 r7\n",
        encoding="utf-8",
    )
    return path


def write_app_report(tmp: pathlib.Path) -> pathlib.Path:
    path = tmp / "app.report.json"
    path.write_text(
        json.dumps(
            {
                "kernels": {
                    "1 mangled_kernel": {
                        "name": "1 mangled_kernel",
                        "asm": [
                            [11, 0x10, "mov (1|M0) r1 r2"],
                            [12, 0x20, "send.ugm (1|M0) r3 r4"],
                            [13, 0x30, "add (1|M0) r5 r6 r7"],
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gtpin_bbl_extractor_emits_positive_pc_counts_and_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        trace = write_trace(tmp)
        session_asm = write_session_asm(tmp)
        app_report = write_app_report(tmp)
        summary = tmp / "summary.parse"
        bbl_csv = tmp / "bbl.csv"
        result = run_extractor(
            "--trace",
            str(trace),
            "--session-asm",
            str(session_asm),
            "--app-report-json",
            str(app_report),
            "--kernel-match",
            "mangled_kernel",
            "--source-computing-task",
            "friendly_kernel",
            "--summary-output",
            str(summary),
            "--bbl-output",
            str(bbl_csv),
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        by_pc = {int(row["pc"]): row for row in rows}
        assert by_pc[0x10]["sample_count"] == "2"
        assert by_pc[0x10]["sample_kind"] == "gtpin-bbl-instruction-exec-count"
        assert by_pc[0x20]["sample_count"] == "2"
        assert by_pc[0x30]["sample_count"] == "1"
        assert by_pc[0x20]["bbl_id"] == "3"
        text = summary.read_text(encoding="utf-8")
        assert "gtpin_bbl_pc.status ok" in text
        assert "gtpin_bbl_pc.trace_send_descriptor_u32_count 20" in text
        assert "gtpin_bbl_pc.dynamic_bbl_records 3" in text
        assert "gtpin_bbl_pc.dynamic_instruction_count 5" in text
        assert "gtpin_bbl_pc.id_shift 1" in text
        bbl_rows = list(csv.DictReader(bbl_csv.open(newline="", encoding="utf-8")))
        assert bbl_rows[0]["bbl_id"] == "3"
        assert bbl_rows[0]["dynamic_count"] == "2"


def test_gtpin_bbl_extractor_fails_closed_on_bad_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        bad_trace = tmp / "memorytrace_compressed.bin"
        bad_trace.write_bytes(b"not a valid trace")
        result = run_extractor(
            "--trace",
            str(bad_trace),
            "--session-asm",
            str(write_session_asm(tmp)),
            "--app-report-json",
            str(write_app_report(tmp)),
            "--kernel-match",
            "mangled_kernel",
            "--source-computing-task",
            "friendly_kernel",
        )
        assert result.returncode == 2
        assert "failed to extract GTPin BBL PC counts" in result.stdout
        assert "Traceback" not in result.stdout
