#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "bin" / "sycl-mxfp4-moe-bench"
PARSER = ROOT / "scripts" / "parse-sycl-mxfp4-tg-bench.py"
ROUTES = ("baseline", "prepack", "row-parallel", "fused-layer", "launch", "host-bounce")

KILL_THRESHOLD_RECOMMENDATIONS = (
    "baseline: reference route for saving_vs_baseline_ms comparisons.",
    "prepack: kill unless combined prepack+compute/cache evidence saves at least 3.0 ms/token against baseline gate/up+GLU.",
    "row-parallel: kill if full equivalent is slower than packed-q8-m2 by more than 10%; continue only at <= 4.0 ms gate/up+GLU equivalent or with a measured path to <= 3.0 ms.",
    "fused-layer: kill if synthetic drift exceeds 2x calibrated baseline max_abs or mean_abs, or relative L2 exceeds 1e-3 before model validation.",
    "launch: kill on hang, timeout, replay exception, correctness mismatch, or launch-only saving < 2.0 ms/token equivalent.",
    "host-bounce: kill if p99 bounce latency exceeds possible saved local compute or if the route requires direct P2P.",
)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def print_child_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="", file=sys.stderr)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def load_route_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    records.append(parsed)
    return records


def route_average_saving(records: list[dict[str, Any]]) -> float:
    return sum(float(record["metrics"]["saving_vs_baseline_ms"]) for record in records) / len(records)


def write_summary(out_dir: Path, route_records: dict[str, list[dict[str, Any]]]) -> Path:
    ranked = sorted(
        ((route, route_average_saving(records), len(records)) for route, records in route_records.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    lines = ["route ranking by saving_vs_baseline_ms", ""]
    for index, (route, saving, count) in enumerate(ranked, start=1):
        lines.append(f"{index}. {route} saving_vs_baseline_ms {saving:.6f} records {count}")

    lines.extend(["", "kill-threshold recommendations"])
    lines.extend(f"- {recommendation}" for recommendation in KILL_THRESHOLD_RECOMMENDATIONS)
    lines.append("")

    summary = out_dir / "summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dry-run SYCL MXFP4 TG microbenches and summarize evidence")
    parser.add_argument("--dry-run", action="store_true", help="run worker-safe dry-run routes only")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for route JSONL files and summary.txt")
    args = parser.parse_args()

    if not args.dry_run:
        return fail("only --dry-run is supported by this orchestrator")

    if not BIN.exists():
        return fail("missing sycl-mxfp4-moe-bench binary; run ./scripts/sycl-build.sh sycl-mxfp4-moe-bench")
    if not BIN.is_file():
        return fail(f"sycl-mxfp4-moe-bench path is not a file: {BIN}")
    if not PARSER.exists():
        return fail(f"missing parser: {PARSER}")

    out_dir = args.out_dir
    if out_dir.exists() and not out_dir.is_dir():
        return fail(f"out-dir is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    route_records: dict[str, list[dict[str, Any]]] = {}
    for route in ROUTES:
        output = out_dir / f"{route}.jsonl"
        if output.exists():
            output.unlink()

        bench = run_command([str(BIN), f"--route={route}", "--dry-run", "--output-jsonl", str(output)])
        if bench.returncode != 0:
            print_child_output(bench)
            return fail(f"route {route} dry-run failed with exit code {bench.returncode}")
        if not output.exists():
            return fail(f"missing route output for {route}: {output}")
        if output.stat().st_size == 0:
            return fail(f"empty route output for {route}: {output}")

        parsed = run_command(["python3", str(PARSER), str(output), "--require-route", route])
        if parsed.returncode != 0:
            print_child_output(parsed)
            return fail(f"parser failed for route {route} with exit code {parsed.returncode}")

        try:
            records = load_route_records(output)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return fail(f"failed to load parsed route output for {route}: {exc}")
        if not records:
            return fail(f"empty route output for {route}: {output}")
        route_records[route] = records

    summary = write_summary(out_dir, route_records)
    print(f"wrote summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
