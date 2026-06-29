#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_TOP = ("route", "mode", "shape", "metrics", "correct", "fatal", "evidence")
REQUIRED_SHAPE = ("ncols", "hidden", "topk", "layers", "tokens")
REQUIRED_METRICS = (
    "prepack_us",
    "compute_us",
    "launch_us",
    "host_bounce_us",
    "total_gateup_equiv_ms",
    "saving_vs_baseline_ms",
    "p50_us",
    "p90_us",
    "p99_us",
)
REQUIRED_CORRECT = ("max_abs", "mean_abs", "rel_l2")
REQUIRED_FATAL = ("total",)
REQUIRED_EVIDENCE = ("path", "dry_run", "device")


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def require_mapping(record: dict[str, Any], key: str, index: int) -> dict[str, Any]:
    value = record[key]
    if not isinstance(value, dict):
        raise ValueError(f"record {index} key is not an object: {key}")
    return value


def require_keys(obj: dict[str, Any], keys: tuple[str, ...], prefix: str, index: int) -> None:
    for key in keys:
        if key not in obj:
            name = key if not prefix else f"{prefix}.{key}"
            raise ValueError(f"record {index} missing key: {name}")


def validate_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"record {index} is not a JSON object")
    require_keys(record, REQUIRED_TOP, "", index)

    shape = require_mapping(record, "shape", index)
    metrics = require_mapping(record, "metrics", index)
    correct = require_mapping(record, "correct", index)
    fatal = require_mapping(record, "fatal", index)
    evidence = require_mapping(record, "evidence", index)

    require_keys(shape, REQUIRED_SHAPE, "shape", index)
    require_keys(metrics, REQUIRED_METRICS, "metrics", index)
    require_keys(correct, REQUIRED_CORRECT, "correct", index)
    require_keys(fatal, REQUIRED_FATAL, "fatal", index)
    require_keys(evidence, REQUIRED_EVIDENCE, "evidence", index)

    if not isinstance(record["route"], str) or not record["route"]:
        raise ValueError(f"record {index} has invalid route")

    try:
        total_gateup = float(metrics["total_gateup_equiv_ms"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {index} has invalid metrics.total_gateup_equiv_ms") from exc
    if total_gateup <= 0.0:
        raise ValueError(f"record {index} has non-positive metrics.total_gateup_equiv_ms")

    try:
        fatal_total = int(fatal["total"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {index} has invalid fatal.total") from exc
    if fatal_total != 0:
        raise ValueError(f"record {index} fatal.total is non-zero")

    return record


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"record {line_no} malformed JSON: {exc.msg}") from exc
            records.append(validate_record(parsed, line_no))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse SYCL MXFP4 TG microbench JSONL evidence")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--require-route", action="append", default=[])
    args = parser.parse_args()

    if not args.jsonl.exists():
        return fail(f"missing microbench JSONL: {args.jsonl}")
    if not args.jsonl.is_file():
        return fail(f"microbench JSONL is not a file: {args.jsonl}")

    try:
        records = load_records(args.jsonl)
    except ValueError as exc:
        return fail(str(exc))

    if not records:
        return fail("empty microbench JSONL")

    routes: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        routes.setdefault(record["route"], []).append(record)

    for route in args.require_route:
        if route not in routes:
            return fail(f"required route missing: {route}")

    print(f"records.total {len(records)}")
    for route in sorted(routes):
        route_records = routes[route]
        total_gateup = sum(float(record["metrics"]["total_gateup_equiv_ms"]) for record in route_records) / len(route_records)
        saving = sum(float(record["metrics"]["saving_vs_baseline_ms"]) for record in route_records) / len(route_records)
        print(f"route.{route}.count {len(route_records)}")
        print(f"route.{route}.total_gateup_equiv_ms {total_gateup:.6f}")
        print(f"route.{route}.saving_vs_baseline_ms {saving:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
