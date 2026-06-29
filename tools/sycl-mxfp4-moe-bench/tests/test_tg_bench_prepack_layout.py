import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"


def test_prepack_dry_run_labels_route(tmp_path: Path) -> None:
    out = tmp_path / "prepack.jsonl"
    result = subprocess.run(
        [str(BIN), "--route=prepack", "--dry-run", "--output-jsonl", str(out)],
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
        assert record["metrics"]["total_gateup_equiv_ms"] > 0.0
        assert record["fatal"]["total"] == 0

    record = records[0]
    assert record["metrics"]["prepack_us"] == 1200.0
    assert record["metrics"]["compute_us"] == 0.0
    assert record["metrics"]["p50_us"] == 6000.0
    assert record["metrics"]["p90_us"] == 6000.0
    assert record["metrics"]["p99_us"] == 6000.0
