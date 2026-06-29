import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARSER = ROOT / "scripts" / "parse-sycl-mxfp4-tg-bench.py"


def run_parser(path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(PARSER), str(path), *extra],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def valid_record(route: str = "baseline") -> dict:
    return {
        "route": route,
        "mode": "dry-run",
        "shape": {"ncols": 2880, "hidden": 2880, "topk": 4, "layers": 24, "tokens": 128},
        "metrics": {
            "prepack_us": 0.0,
            "compute_us": 0.0,
            "launch_us": 0.0,
            "host_bounce_us": 0.0,
            "total_gateup_equiv_ms": 6.0,
            "saving_vs_baseline_ms": 0.0,
            "p50_us": 0.0,
            "p90_us": 0.0,
            "p99_us": 0.0,
        },
        "correct": {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0},
        "fatal": {"total": 0},
        "evidence": {"path": "packed-q8-m2", "dry_run": True, "device": "none"},
    }


def write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_accepts_valid_record(tmp_path: Path) -> None:
    result = run_parser(write_jsonl(tmp_path, [valid_record()]), "--require-route", "baseline")
    assert result.returncode == 0, result.stderr
    assert "records.total 1" in result.stdout
    assert "route.baseline.total_gateup_equiv_ms 6.000000" in result.stdout


def test_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = run_parser(empty)
    assert result.returncode != 0
    assert "error: empty microbench JSONL" in result.stderr


def test_rejects_missing_route(tmp_path: Path) -> None:
    record = valid_record()
    del record["route"]
    result = run_parser(write_jsonl(tmp_path, [record]))
    assert result.returncode != 0
    assert "error: record 1 missing key: route" in result.stderr


def test_rejects_zero_total_gateup(tmp_path: Path) -> None:
    record = valid_record()
    record["metrics"]["total_gateup_equiv_ms"] = 0.0
    result = run_parser(write_jsonl(tmp_path, [record]))
    assert result.returncode != 0
    assert "error: record 1 has non-positive metrics.total_gateup_equiv_ms" in result.stderr


def test_rejects_fatal_marker(tmp_path: Path) -> None:
    record = valid_record()
    record["fatal"]["total"] = 1
    result = run_parser(write_jsonl(tmp_path, [record]))
    assert result.returncode != 0
    assert "error: record 1 fatal.total is non-zero" in result.stderr


def test_requires_named_route(tmp_path: Path) -> None:
    result = run_parser(write_jsonl(tmp_path, [valid_record("row-parallel")]), "--require-route", "baseline")
    assert result.returncode != 0
    assert "error: required route missing: baseline" in result.stderr
