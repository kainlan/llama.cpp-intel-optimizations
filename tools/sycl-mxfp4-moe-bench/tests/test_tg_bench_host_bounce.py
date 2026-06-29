import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"


def test_host_bounce_dry_run_modes(tmp_path: Path) -> None:
    out = tmp_path / "bounce.jsonl"
    result = subprocess.run(
        [str(BIN), "--route=host-bounce", "--dry-run", "--output-jsonl", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["mode"] for record in records] == ["copy-only", "remote-expert", "overlap-sim"]
    assert all(record["evidence"]["path"] == "host-bounce-no-p2p" for record in records)
    assert records[0]["metrics"]["host_bounce_us"] > 0.0
    assert records[1]["metrics"]["host_bounce_us"] > 0.0
    assert records[0]["metrics"]["host_bounce_us"] == 500.0
    assert records[0]["metrics"]["total_gateup_equiv_ms"] == 0.5
    assert records[0]["metrics"]["saving_vs_baseline_ms"] == 0.0
    assert records[1]["metrics"]["host_bounce_us"] == 700.0
    assert records[1]["metrics"]["compute_us"] == 2500.0
    assert records[1]["metrics"]["total_gateup_equiv_ms"] == 3.2
    assert records[1]["metrics"]["saving_vs_baseline_ms"] == 2.8
    assert records[2]["metrics"]["host_bounce_us"] == 700.0
    assert records[2]["metrics"]["compute_us"] == 2500.0
    assert records[2]["metrics"]["total_gateup_equiv_ms"] == 2.1
    assert records[2]["metrics"]["saving_vs_baseline_ms"] == 3.9
