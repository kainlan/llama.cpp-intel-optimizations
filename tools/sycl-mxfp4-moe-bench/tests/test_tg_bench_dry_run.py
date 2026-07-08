import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"
PARSER = ROOT / "scripts" / "parse-sycl-mxfp4-tg-bench.py"


def test_dry_run_emits_valid_jsonl(tmp_path: Path) -> None:
    if not BIN.exists():
        raise AssertionError(f"missing binary: {BIN}")
    out = tmp_path / "baseline.jsonl"
    run = subprocess.run(
        [str(BIN), "--route=baseline", "--dry-run", "--output-jsonl", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["route"] == "baseline"
    assert records[0]["evidence"]["dry_run"] is True
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
