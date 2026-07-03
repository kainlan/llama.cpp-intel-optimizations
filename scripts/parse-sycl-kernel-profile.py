#!/usr/bin/env python3
"""Summarize SYCL named kernel profiler CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from typing import Any


def metric_name(raw: str) -> str:
    return raw.replace(" ", "_")


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" or text.lstrip().startswith("{"):
        obj = json.loads(text)
        return [dict(row) for row in obj.get("kernels", [])]
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines()):
        rows.append(dict(row))
    return rows


def int_field(row: dict[str, Any], name: str) -> int:
    raw = row.get(name, 0)
    if raw in (None, ""):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        kernel = row.get("name", "unknown")
        raise ValueError(f"invalid integer field {name} for kernel {kernel}: {raw!r}") from exc


def ns_to_ms_x1000(ns: int) -> int:
    return int(round(ns / 1000.0))


def parse_min_total(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=MS")
    name, value = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("expected NAME=MS")
    try:
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid millisecond value: {raw}") from exc


def parse_wall_ms(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid wall millisecond value: {raw}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("--wall-ms must be greater than zero")
    return value


def parse_kernel_bytes(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=BYTES")
    name, value = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("expected NAME=BYTES")
    try:
        bytes_per_event = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte count: {raw}") from exc
    if bytes_per_event < 0:
        raise argparse.ArgumentTypeError("kernel bytes must be non-negative")
    return name, bytes_per_event


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Counter[str]], Counter[str]]:
    """Aggregate rows by kernel name and category.

    Profiler artifacts are emitted at `(name, category, metadata)` granularity,
    but this parser's `--require-kernel NAME` and counter output are intentionally
    name-based so one kernel can be checked even when several metadata variants
    appear in a single artifact.
    """

    kernel_totals: dict[str, Counter[str]] = {}
    category_totals: Counter[str] = Counter()
    for row in rows:
        name = str(row.get("name", "unknown"))
        category = metric_name(str(row.get("category", "unknown")))
        count = int_field(row, "count")
        total_ns = int_field(row, "total_ns")
        failed = int_field(row, "failed_timestamps")
        totals = kernel_totals.setdefault(name, Counter())
        totals["count"] += count
        totals["total_ns"] += total_ns
        totals["failed_timestamps"] += failed
        category_totals[category] += total_ns
    return kernel_totals, category_totals


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize SYCL named kernel profile artifacts")
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument("--require-kernel", action="append", default=[])
    parser.add_argument("--min-total-ms", action="append", type=parse_min_total, default=[])
    parser.add_argument("--wall-ms", type=parse_wall_ms)
    parser.add_argument("--kernel-bytes", action="append", type=parse_kernel_bytes, default=[])
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.profile)
        kernel_totals, category_totals = aggregate_rows(rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse profile: {exc}")
        return 2

    ok = True

    for name in args.require_kernel:
        if name not in kernel_totals:
            print(f"missing required kernel: {name}")
            ok = False

    for name, threshold_ms in args.min_total_ms:
        totals = kernel_totals.get(name)
        if totals is None:
            print(f"missing threshold kernel: {name}")
            ok = False
            continue
        total_ms = totals["total_ns"] / 1_000_000.0
        if total_ms < threshold_ms:
            print(f"kernel below total_ms threshold: {name} observed={total_ms:.3f} required={threshold_ms:.3f}")
            ok = False

    kernel_bytes = dict(args.kernel_bytes)
    for name in kernel_bytes:
        if name not in kernel_totals:
            print(f"missing bytes kernel: {name}")
            ok = False

    kernel_sum_total_ns = sum(totals["total_ns"] for totals in kernel_totals.values())
    print(f"profile.kernel_sum_total_ms_x1000 {ns_to_ms_x1000(kernel_sum_total_ns)}")
    if args.wall_ms is not None:
        kernel_sum_ms = kernel_sum_total_ns / 1_000_000.0
        print(f"profile.decode_wall_ms_x1000 {int(round(args.wall_ms * 1000.0))}")
        print(f"profile.kernel_coverage_pct_x1000 {int(round(100000.0 * kernel_sum_ms / args.wall_ms))}")

    for name, totals in sorted(kernel_totals.items(), key=lambda item: item[1]["total_ns"], reverse=True):
        metric = metric_name(name)
        print(f"kernel.{metric}.count {totals['count']}")
        print(f"kernel.{metric}.total_ms_x1000 {ns_to_ms_x1000(totals['total_ns'])}")
        print(f"kernel.{metric}.failed_timestamps {totals['failed_timestamps']}")
        if name in kernel_bytes:
            total_ns = totals["total_ns"]
            # `--kernel-bytes` is bytes per profiler event. Achieved GB/s is
            # (bytes_per_event * event_count) / total_ns because bytes/ns has
            # the same 10^9 scale as GB/s; the printed metric is x1000 scaled.
            achieved_gbps_x1000 = (
                0 if total_ns == 0 else int(round(1000.0 * kernel_bytes[name] * totals["count"] / total_ns))
            )
            print(f"kernel.{metric}.achieved_gbps_x1000 {achieved_gbps_x1000}")

    for category, total_ns in sorted(category_totals.items()):
        print(f"category.{category}.total_ms_x1000 {ns_to_ms_x1000(total_ns)}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
