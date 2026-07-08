import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "run-sycl-mxfp4-tg-microbenches.py"


def test_orchestrator_dry_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    result = subprocess.run(
        ["python3", str(SCRIPT), "--dry-run", "--out-dir", str(out_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "baseline.jsonl").exists()
    assert (out_dir / "prepack.jsonl").exists()
    assert (out_dir / "summary.txt").exists()
    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "route ranking by saving_vs_baseline_ms" in summary
    assert "baseline" in summary
    assert "prepack" in summary
