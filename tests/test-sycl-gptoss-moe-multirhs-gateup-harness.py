from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-moe-multirhs-gateup-gates.sh"


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True, env=merged_env)


def test_multirhs_harness_defaults_to_dry_run_without_side_effects() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        out_dir = pathlib.Path(tmp_raw) / "must_not_exist"
        result = run_script("--out-dir", str(out_dir))
        assert result.returncode == 0, result.stderr
        assert not out_dir.exists()
        assert "candidate_multirhs" in result.stdout


def test_multirhs_candidate_env_and_parser_gates_are_candidate_only() -> None:
    result = run_script("--dry-run")
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    baseline = stdout.split("# case: baseline", 1)[1].split("# case: candidate_multirhs", 1)[0]
    candidate = stdout.split("# case: candidate_multirhs", 1)[1]
    assert "GGML_SYCL_MOE_GATEUP_MULTIRHS=1" not in baseline
    assert "GGML_SYCL_MOE_GATEUP_MULTIRHS=1" in candidate
    assert "--require-mxfp4-tg-path multirhs-gateup" in candidate
    assert "--require-mxfp4-gateup-max-ms 4.2" in candidate
    assert "--require-bench-min pp512=1200" in candidate
    assert "--require-bench-min tg128=45" in candidate


def test_multirhs_real_run_requires_ack() -> None:
    result = run_script("--run")
    assert result.returncode == 2
    assert "real execution requires --i-understand-this-runs-gpu-models" in result.stderr


def test_multirhs_dry_run_omits_forbidden_probe_text() -> None:
    result = run_script("--dry-run")
    assert result.returncode == 0, result.stderr
    forbidden = ["sycl-ls", "/dev/dri", "fdinfo", "lsof", "P2P", "peer-to-peer"]
    for needle in forbidden:
        assert needle not in result.stdout
