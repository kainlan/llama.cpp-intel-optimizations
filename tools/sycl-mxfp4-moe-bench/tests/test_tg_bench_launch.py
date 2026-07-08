import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"


def test_launch_dry_run_modes(tmp_path: Path) -> None:
    out = tmp_path / "launch.jsonl"
    result = subprocess.run([str(BIN), "--route=launch", "--dry-run", "--output-jsonl", str(out)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    expected = [
        ("raw-queue", 2400.0, 2.4, 0.0),
        ("command-graph", 900.0, 0.9, 1.5),
        ("persistent-descriptor", 700.0, 0.7, 1.7),
    ]
    assert [record["mode"] for record in records] == [mode for mode, _, _, _ in expected]
    assert all(record["metrics"]["launch_us"] > 0.0 for record in records)
    assert all(record["evidence"]["path"] == "launch-reduction" for record in records)
    for record, (_, launch_us, total_ms, saving_ms) in zip(records, expected):
        assert record["metrics"]["launch_us"] == launch_us
        assert record["metrics"]["total_gateup_equiv_ms"] == total_ms
        assert record["metrics"]["saving_vs_baseline_ms"] == saving_ms
