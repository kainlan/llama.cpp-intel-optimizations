import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"
PARSER = ROOT / "scripts" / "parse-sycl-mxfp4-tg-bench.py"


def test_baseline_dry_run_contract(tmp_path: Path) -> None:
    out = tmp_path / "baseline.jsonl"
    result = subprocess.run(
        [
            str(BIN),
            "--route=baseline",
            "--dry-run",
            "--ncols=2880",
            "--topk=4",
            "--layers=24",
            "--tokens=128",
            "--output-jsonl",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["route"] == "baseline"
    assert record["mode"] == "dry-run"
    assert record["evidence"] == {"path": "packed-q8-m2", "dry_run": True, "device": "none"}
    assert record["shape"] == {"ncols": 2880, "hidden": 2880, "topk": 4, "layers": 24, "tokens": 128}
    assert record["metrics"]["compute_us"] == 6000.0
    assert record["metrics"]["total_gateup_equiv_ms"] > 0.0
    assert record["metrics"]["saving_vs_baseline_ms"] == 0.0
    assert record["metrics"]["p50_us"] == 6000.0
    assert record["metrics"]["p90_us"] == 6000.0
    assert record["metrics"]["p99_us"] == 6000.0
    assert record["fatal"]["total"] == 0

    parsed = subprocess.run(
        ["python3", str(PARSER), str(out), "--require-route", "baseline"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert "records.total 1" in parsed.stdout
