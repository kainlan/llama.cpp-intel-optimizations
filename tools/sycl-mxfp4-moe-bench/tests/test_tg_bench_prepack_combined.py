import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"
PARSER = ROOT / "scripts" / "parse-sycl-mxfp4-tg-bench.py"


def test_prepack_combined_emits_three_modes(tmp_path: Path) -> None:
    out = tmp_path / "prepack.jsonl"
    result = subprocess.run(
        [str(BIN), "--route=prepack", "--dry-run", "--tokens=128", "--output-jsonl", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["mode"] for record in records] == ["prepack-only", "compute-only", "cache-combined"]

    for record in records:
        assert record["route"] == "prepack"
        assert record["evidence"]["path"] == "selected-expert-prepack"
        assert record["evidence"]["dry_run"] is True

    expected_metrics = [
        {
            "prepack_us": 1200.0,
            "compute_us": 0.0,
            "total_gateup_equiv_ms": 6.0,
            "saving_vs_baseline_ms": 0.0,
            "p50_us": 6000.0,
            "p90_us": 6000.0,
            "p99_us": 6000.0,
        },
        {
            "prepack_us": 0.0,
            "compute_us": 2500.0,
            "total_gateup_equiv_ms": 2.5,
            "saving_vs_baseline_ms": 3.5,
            "p50_us": 2500.0,
            "p90_us": 2500.0,
            "p99_us": 2500.0,
        },
        {
            "prepack_us": 200.0,
            "compute_us": 2500.0,
            "total_gateup_equiv_ms": 2.7,
            "saving_vs_baseline_ms": 3.3,
            "p50_us": 2700.0,
            "p90_us": 2700.0,
            "p99_us": 2700.0,
        },
    ]
    for record, expected in zip(records, expected_metrics):
        for key, value in expected.items():
            assert record["metrics"][key] == value

    parsed = subprocess.run(
        ["python3", str(PARSER), str(out), "--require-route", "prepack"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert "records.total 3" in parsed.stdout
    assert "route.prepack.count 3" in parsed.stdout
