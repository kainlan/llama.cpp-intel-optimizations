import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"

EXPECTED = [
    {
        "mode": "fp32-accum",
        "total_gateup_equiv_ms": 2.2,
        "saving_vs_baseline_ms": 3.8,
        "max_abs": 0.00001,
        "mean_abs": 0.000001,
        "rel_l2": 0.000001,
    },
    {
        "mode": "fp16-accum",
        "total_gateup_equiv_ms": 2.0,
        "saving_vs_baseline_ms": 4.0,
        "max_abs": 0.0002,
        "mean_abs": 0.00002,
        "rel_l2": 0.00002,
    },
    {
        "mode": "quant-accum",
        "total_gateup_equiv_ms": 1.8,
        "saving_vs_baseline_ms": 4.2,
        "max_abs": 0.0005,
        "mean_abs": 0.00005,
        "rel_l2": 0.00005,
    },
]


def test_fused_layer_dry_run_reports_drift(tmp_path: Path) -> None:
    out = tmp_path / "fused.jsonl"
    result = subprocess.run(
        [str(BIN), "--route=fused-layer", "--dry-run", "--output-jsonl", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["mode"] for record in records] == [expected["mode"] for expected in EXPECTED]
    assert all(record["evidence"]["path"] == "fused-gateup-glu-down" for record in records)

    for record, expected in zip(records, EXPECTED):
        assert record["metrics"]["total_gateup_equiv_ms"] == expected["total_gateup_equiv_ms"]
        assert record["metrics"]["saving_vs_baseline_ms"] == expected["saving_vs_baseline_ms"]
        assert record["correct"]["max_abs"] == expected["max_abs"]
        assert record["correct"]["mean_abs"] == expected["mean_abs"]
        assert record["correct"]["rel_l2"] == expected["rel_l2"]
        assert record["correct"]["rel_l2"] < 1e-3
