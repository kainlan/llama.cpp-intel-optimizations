#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-source-line-debug-matrix.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_debug_matrix_zebin_lookup_is_pipe_free() -> None:
    text = script_text()
    assert "-print -quit" in text
    assert "| head -n 1" not in text


def test_debug_matrix_script_is_dry_run_by_default() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 0, result.stdout
    for required in ("DRY RUN", "release_split", "debug_line_tables", "debug_full", "debug_no_inline", "sycl-source-line-probe", "llvm-readelf --sections --wide", "gpu-source-line"):
        assert required in result.stdout
    assert "vtune-computing-tasks.csv" in result.stdout
    assert "parse-sycl-vtune-tasks.py" in result.stdout
    assert "zebin-debug-line.txt" in result.stdout
    assert "dwarf-source-lines.csv" in result.stdout
    assert "convert-sycl-zebin-line-table-to-source-csv.py" in result.stdout
    assert "--source-computing-task sycl_source_line_probe" in result.stdout
    for required in (
        "prepare-sycl-iga-disasm-inputs.py",
        "iga64",
        "-Xprint-json",
        "-Xprint-pc",
        "parse-sycl-iga-pc-disasm.py",
        "iga-pc-instructions.csv",
        "--iga-instructions-csv",
        "--pc-base",
        "--iga-platform",
        "llvm-readelf --sections --wide",
    ):
        assert required in result.stdout
    assert "ocloc disasm -file kernel.zebin" in result.stdout
    assert "asm-source-lines.csv" in result.stdout
    assert "resolve-sycl-zebin-asm-source-lines.py" in result.stdout
    assert "--asm-source-lines-csv" in result.stdout
    assert "--allow-asm-line-static-cost" in result.stdout
    assert "--dwarf-line-dump" in result.stdout
    assert "--dwarf-source-lines-csv" in result.stdout
    assert "--allow-dwarf-line-table-only" in result.stdout
    assert "--require-source-path main.cpp" in result.stdout
    assert "--vtune-stdout" in result.stdout
    assert "--vtune-stderr" in result.stdout
    assert "tools/sycl-source-line-probe/main.cpp" not in result.stdout
    assert "computing-tasks-of-interest" not in result.stdout
    assert "/Storage" not in result.stdout
    assert "llama-bench" not in result.stdout
    assert "sycl-kernel-bench" not in result.stdout


def test_debug_matrix_vtune_target_gpu_is_explicit_opt_in() -> None:
    explicit = subprocess.run(
        ["bash", str(SCRIPT), "--vtune-target-gpu", "0:7:0.0"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert explicit.returncode == 0, explicit.stdout
    assert "target-gpu=0:7:0.0" in explicit.stdout

    default = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert default.returncode == 0, default.stdout
    assert "target-gpu=" not in default.stdout


def test_debug_matrix_script_handles_absolute_out_root_in_dry_run(tmp_path: Path) -> None:
    out_root = tmp_path / "source line matrix"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--out-root", str(out_root)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    expected_probe = out_root / "build-matrix" / "release_split" / "build" / "bin" / "sycl-source-line-probe"
    assert str(expected_probe).replace(" ", "\\ ") in result.stdout
    assert ".//tmp" not in result.stdout


def test_debug_matrix_script_refuses_execute_without_ack() -> None:
    result = subprocess.run(["bash", str(SCRIPT), "--execute"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-source-probe" in result.stdout


def test_debug_matrix_task_filter_is_opt_in() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--task-glob", "*sycl_source_line_probe*"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "computing-tasks-of-interest" in result.stdout
    assert "\\*sycl_source_line_probe\\*#1#1#20" in result.stdout


def test_debug_matrix_task_glob_with_single_quote_is_shell_escaped() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--task-glob", "a'b"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "computing-tasks-of-interest=a\\'b#1#1#20" in result.stdout
    assert "computing-tasks-of-interest='a'b#1#1#20'" not in result.stdout


def test_debug_matrix_requires_probe_dwarf_basename() -> None:
    text = script_text()
    assert '"tools/sycl-source-line-probe/main.cpp"' not in text
    assert '--require-source-path "main.cpp"' in text
    assert '--require-source-path "main.cpp"' in text
    assert 'grep -qx \'asm_source.status ok\' "${dir}/asm-source-lines.parse"' in text
    assert 'checker_args+=(--asm-source-lines-csv "${dir}/asm-source-lines.csv" --allow-asm-line-static-cost)' in text
    assert '--dwarf-source-lines-csv "${dir}/dwarf-source-lines.csv"' in text
    assert '--allow-dwarf-line-table-only' in text
    assert '--vtune-stdout "${dir}/probe.stdout"' in text
    assert '--vtune-stderr "${dir}/probe.stderr"' in text
    assert 'python3 scripts/check-sycl-vtune-source-lines.py "${checker_args[@]}" >"${dir}/source-line-feasibility.parse"' in text


def test_debug_matrix_execute_branch_writes_expected_artifacts() -> None:
    text = script_text()
    for required in (
        "source-line/build-matrix",
        "zebin-debug-sections.txt",
        "vtune-computing-tasks.csv",
        "vtune-task.parse",
        "parse-sycl-vtune-tasks.py",
        "zebin-debug-line.txt",
        "dwarf-source-lines.csv",
        "asm-source-lines.csv",
        "asm-source-lines.parse",
        "zebin-disasm",
        "iga-disasm",
        "run-iga-disasm.sh",
        "iga-pc-instructions.csv",
        "prepare-sycl-iga-disasm-inputs.py",
        "parse-sycl-iga-pc-disasm.py",
        "--iga-instructions-csv",
        "--pc-base",
        "llvm-readelf --sections --wide",
        "warning: IGA PC disassembly failed",
        "checker will use ocloc/DWARF evidence if available",
        "ocloc disasm -file kernel.zebin",
        "resolve-sycl-zebin-asm-source-lines.py",
        "vtune-gpu-source-line.csv",
        "source-line-feasibility.parse",
        "convert-sycl-zebin-line-table-to-source-csv.py",
        "check-sycl-vtune-source-lines.py",
        "--dwarf-line-dump",
        "--asm-source-lines-csv",
        "--allow-asm-line-static-cost",
        "--dwarf-source-lines-csv",
        "--allow-dwarf-line-table-only",
        "--require-source-path",
        "--vtune-stdout",
        "--vtune-stderr",
        "VTune gpu-source-line report failed",
        "source-line checker reported failure",
        "set +u",
        "source /opt/intel/oneapi/setvars.sh --force",
        "set -u",
    ):
        assert required in text
