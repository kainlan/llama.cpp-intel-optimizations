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
    assert "source-line-matrix/build-matrix/<case>/vtune-gpu-source-line.csv" in result.stdout
    assert "source_line.status pass" in result.stdout
    assert "source_line.blocker vtune_unknown_source" in result.stdout
    assert "exported-kernels.csv" not in result.stdout
    assert "exported-source-lines.csv" not in result.stdout
    assert "source-line-matrix/readelf-sections.txt" not in result.stdout
    assert "source-line-matrix/vtune-source-lines.csv" not in result.stdout
    assert not out_root.exists()


def test_vtune_source_execute_branch_consumes_matrix_artifacts() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "source-line-matrix/build-matrix",
        "source-line-feasibility.parse",
        "vtune-gpu-source-line.csv",
        "source_line.status pass",
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
