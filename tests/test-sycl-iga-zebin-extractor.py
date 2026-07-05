#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare-sycl-iga-disasm-inputs.py"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_sections(tmp: pathlib.Path, body: str) -> pathlib.Path:
    p = tmp / "sections.txt"
    p.write_text(body, encoding="utf-8")
    return p


def test_prepare_outputs_manifest_and_commands_for_single_section() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = write_sections(
            tmp,
            "[ 1] .text._ZTS_target_kernel PROGBITS 0000000000000040 000040 000080 00 AX 0 0 16\n"
            "[ 2] .debug_line PROGBITS 0000000000000000 0000c0 000040 00 0 0 1\n",
        )
        out_dir = tmp / "iga"
        result = run_script(
            "--readelf-sections",
            str(sections),
            "--zebin",
            str(tmp / "kernel.zebin"),
            "--kernel-match",
            "target_kernel",
            "--platform",
            "xe2",
            "--out-dir",
            str(out_dir),
        )
        assert result.returncode == 0, result.stdout
        manifest = json.loads((out_dir / "iga-disasm-manifest.json").read_text(encoding="utf-8"))
        assert manifest["extract.status"] == "ok"
        assert manifest["extract.section"] == ".text._ZTS_target_kernel"
        assert manifest["extract.section_addr"] == "0x40"
        command_text = (out_dir / "run-iga-disasm.sh").read_text(encoding="utf-8")
        assert "llvm-objcopy" in command_text
        assert "--dump-section" in command_text
        assert "iga64" in command_text
        assert "-Xprint-json" in command_text
        assert "-Xprint-pc" in command_text


def test_prepare_fails_closed_on_missing_sections() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = write_sections(
            tmp,
            "[ 1] .debug_line PROGBITS 0000000000000000 000040 000080 00 0 0 1\n",
        )
        result = run_script(
            "--readelf-sections",
            str(sections),
            "--zebin",
            str(tmp / "kernel.zebin"),
            "--kernel-match",
            "target_kernel",
            "--platform",
            "xe2",
            "--out-dir",
            str(tmp / "iga"),
        )
        assert result.returncode == 2
        assert "extract.status missing_kernel_text_section" in result.stdout
        assert "Traceback" not in result.stdout


def test_prepare_fails_closed_on_ambiguous_sections() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        sections = write_sections(
            tmp,
            "[ 1] .text._ZTS_target_kernel_a PROGBITS 0000000000000040 000040 000080 00 AX 0 0 16\n"
            "[ 2] .text._ZTS_target_kernel_b PROGBITS 00000000000000c0 0000c0 000080 00 AX 0 0 16\n",
        )
        result = run_script(
            "--readelf-sections",
            str(sections),
            "--zebin",
            str(tmp / "kernel.zebin"),
            "--kernel-match",
            "target_kernel",
            "--platform",
            "xe2",
            "--out-dir",
            str(tmp / "iga"),
        )
        assert result.returncode == 2
        assert "extract.status ambiguous_kernel_text_section" in result.stdout
        assert "Traceback" not in result.stdout
