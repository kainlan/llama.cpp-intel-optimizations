#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-staged-attribution-profile.sh"


def run_script(*args: str, out_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SYCL_GPTOSS_STAGED_OUT"] = str(out_root)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_staged_runner_is_dry_run_by_default(tmp_path: Path) -> None:
    out_root = tmp_path / "staged"
    result = run_script(out_root=out_root)
    assert result.returncode == 0, result.stdout
    assert "DRY RUN" in result.stdout
    assert "stage=all" in result.stdout
    assert "base/stage-manifest.json" in result.stdout
    assert "l0/stage-manifest.json" in result.stdout
    assert "ur/stage-manifest.json" in result.stdout
    assert "vtune-source/stage-manifest.json" in result.stdout
    assert "ablation/stage-manifest.json" in result.stdout
    assert "scripts/merge-sycl-staged-ledger.py" in result.stdout
    assert not out_root.exists()


def test_staged_runner_refuses_execute_without_ack(tmp_path: Path) -> None:
    result = run_script("--execute", out_root=tmp_path / "staged")
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-staged-gpu-profiling" in result.stdout


def test_staged_runner_execute_branch_uses_safe_gptoss_knobs_without_monolithic_vtune() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
        "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
        "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
        "-fa 1",
        "write_manifest",
        "merge-sycl-staged-ledger.py",
    ):
        assert required in text
    assert "vtune -collect gpu-hotspots" not in text[text.index("run_base_stage") : text.index("run_ur_stage")]


def test_vtune_source_dry_run_reports_matrix_artifact_contract(tmp_path: Path) -> None:
    out_root = tmp_path / "staged"
    result = run_script("--stage", "vtune-source", out_root=out_root)
    assert result.returncode == 0, result.stdout
    assert "source-line-matrix/build-matrix/<case>/source-line-feasibility.parse" in result.stdout
    assert "source-line-matrix/build-matrix/<case>/zebin-debug-sections.txt" in result.stdout
    assert "source-line-matrix/build-matrix/<case>/zebin-debug-line.txt" in result.stdout
    assert "source-line-matrix/build-matrix/<case>/dwarf-source-lines.csv" in result.stdout
    assert "source-line-matrix/build-matrix/<case>/asm-source-lines.csv" in result.stdout
    assert "source-line-matrix/build-matrix/<case>/vtune-gpu-source-line.csv" in result.stdout
    assert "convert-sycl-zebin-line-table-to-source-csv.py" in result.stdout
    assert "ocloc disasm -file kernel.zebin" in result.stdout
    assert "asm-source-lines.csv" in result.stdout
    assert "resolve-sycl-zebin-asm-source-lines.py" in result.stdout
    assert "--asm-source-lines-csv" in result.stdout
    assert "--allow-asm-line-static-cost" in result.stdout
    assert "--dwarf-source-lines-csv" in result.stdout
    assert "--allow-dwarf-line-table-only" in result.stdout
    assert "source_line.status pass" in result.stdout
    assert "source_line.status asm-line-static-cost" in result.stdout
    assert "source_line.status dwarf-line-table-only" in result.stdout
    assert "DWARF line-table fallback" in result.stdout
    assert "source_line.blocker vtune_no_gpu_side_trace or vtune_unknown_source" in result.stdout
    assert "exported-kernels.csv" not in result.stdout
    assert "exported-source-lines.csv" not in result.stdout
    assert "source-line-matrix/readelf-sections.txt" not in result.stdout
    assert "source-line-matrix/vtune-source-lines.csv" not in result.stdout
    assert not out_root.exists()


def test_vtune_source_dry_run_quotes_asm_paths_with_spaces(tmp_path: Path) -> None:
    out_root = tmp_path / "source line staged"
    result = run_script("--stage", "vtune-source", "--out-root", str(out_root), out_root=tmp_path / "ignored")
    assert result.returncode == 0, result.stdout

    asm_lines = [line for line in result.stdout.splitlines() if "matrix ASM" in line or "matrix disasm" in line]
    assert asm_lines
    raw_root_prefix = f"{out_root}/"
    raw_asm_dir = f"{out_root}/vtune-source/source-line-matrix/build-matrix/debug_full/zebin-disasm"
    escaped_asm_dir = raw_asm_dir.replace(" ", "\\ ")
    for line in asm_lines:
        assert raw_root_prefix not in line
    assert any(f"mkdir -p {escaped_asm_dir}" in line for line in asm_lines)
    assert any(f'first_asm="$(find {escaped_asm_dir} -type f -name' in line for line in asm_lines)
    assert any('--asm "${first_asm}"' in line for line in asm_lines)
    assert not out_root.exists()


def test_vtune_source_execute_branch_consumes_matrix_artifacts() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "source-line-matrix/build-matrix",
        "source-line-feasibility.parse",
        "vtune-gpu-source-line.csv",
        "dwarf-source-lines.csv",
        "asm-source-lines.csv",
        "source_line.status pass",
        "source_line.status asm-line-static-cost",
        "source_line.status dwarf-line-table-only",
        "asm_line_static_parse",
        "dwarf_line_table_parse",
        "source_line.blocker vtune_no_gpu_side_trace",
        "source_line.blocker vtune_unknown_source",
        "found no matrix source-line-feasibility.parse files",
        "cp \"${selected_parse}\" \"${root}/source-line.parse\"",
        "--source-csv \"${selected_source_csv}\" >\"${root}/vtune.parse\"",
    ):
        assert required in text
    for invented in (
        "exported-kernels.csv",
        "exported-source-lines.csv",
        "source-line-matrix/readelf-sections.txt",
        "source-line-matrix/vtune-source-lines.csv",
    ):
        assert invented not in text
