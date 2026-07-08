#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-moe-gateup-work-reduction-gates.sh"


def run_script(*args: str, out_dir: pathlib.Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OUT_DIR"] = str(out_dir)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_default_dry_run_has_no_side_effects(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "jd32-dryrun"
    result = run_script(out_dir=out_dir)
    assert result.returncode == 0, result.stdout
    assert not out_dir.exists()
    assert "# case: baseline" in result.stdout
    assert "# case: candidate_singlecol" in result.stdout
    assert "GGML_SYCL_MOE_GATEUP_SINGLECOL=1" in result.stdout
    assert "--require-mxfp4-tg-path singlecol-gateup" in result.stdout
    assert "--require-mxfp4-gateup-max-ms 4.2" in result.stdout
    assert "--require-bench-min pp512=1200" in result.stdout
    assert "--require-bench-min tg128=45" in result.stdout


def test_baseline_dry_run_does_not_enable_singlecol_candidate(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "jd32-dryrun"
    result = run_script("--dry-run", out_dir=out_dir)
    assert result.returncode == 0, result.stdout
    baseline = result.stdout.split("# case: baseline", 1)[1].split("# case: candidate_singlecol", 1)[0]
    assert "GGML_SYCL_MOE_GATEUP_SINGLECOL=1" not in baseline


def test_run_requires_ack(tmp_path: pathlib.Path) -> None:
    result = run_script("--run", out_dir=tmp_path / "real")
    assert result.returncode == 2
    assert "real execution requires --i-understand-this-runs-gpu-models" in result.stdout


def test_missing_option_value_fails(tmp_path: pathlib.Path) -> None:
    result = run_script("--model", out_dir=tmp_path / "missing")
    assert result.returncode == 2
    assert "--model requires a value" in result.stdout


def test_dry_run_forbids_probe_commands(tmp_path: pathlib.Path) -> None:
    result = run_script("--dry-run", out_dir=tmp_path / "dry")
    assert result.returncode == 0, result.stdout
    forbidden = ("sycl-ls", "/dev/dri", "fdinfo", "lsof", "direct P2P")
    for needle in forbidden:
        assert needle not in result.stdout
