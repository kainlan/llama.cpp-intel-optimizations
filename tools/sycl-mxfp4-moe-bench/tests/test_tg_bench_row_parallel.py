import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"


def test_row_parallel_dry_run_modes(tmp_path: Path) -> None:
    out = tmp_path / "row.jsonl"
    result = subprocess.run(
        [str(BIN), "--route=row-parallel", "--dry-run", "--output-jsonl", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["mode"] for record in records] == ["row-dot", "gate-up-glu", "hybrid-tail"]
    assert all(record["evidence"]["path"] == "row-parallel-non-xmx" for record in records)
    assert all(record["correct"]["rel_l2"] <= 1e-6 for record in records)
