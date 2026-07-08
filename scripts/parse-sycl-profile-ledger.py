#!/usr/bin/env python3
"""Reconcile SYCL timeline and named-kernel profile totals into a wall ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timeline_parser = load_module("parse_sycl_timeline", ROOT / "scripts" / "parse-sycl-timeline.py")
kernel_parser = load_module("parse_sycl_kernel_profile", ROOT / "scripts" / "parse-sycl-kernel-profile.py")


def parse_ratio(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid coverage ratio threshold: {raw}") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("--coverage-ratio-threshold must be between 0 and 1")
    return value


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Reconcile SYCL timeline and named-kernel profile totals")
    parser.add_argument("timeline", type=pathlib.Path)
    parser.add_argument("kernel_profile", type=pathlib.Path)
    parser.add_argument("--coverage-ratio-threshold", type=parse_ratio, default=0.90)
    args = parser.parse_args(argv)

    try:
        events = timeline_parser.load_trace_events(args.timeline)
        _, _, wall_us, timeline_gpu_total, _, _ = timeline_parser.summarize_events(events)
        gap_classes_by_queue = timeline_parser.summarize_queue_gap_classes(events)
        rows = kernel_parser.load_rows(args.kernel_profile)
        kernel_totals, _ = kernel_parser.aggregate_rows(rows)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError, AttributeError) as exc:
        print(f"failed to parse profile ledger: {exc}")
        return 2

    wall_total = timeline_parser.us_to_ms_x1000(wall_us)
    kernel_total = kernel_parser.ns_to_ms_x1000(sum(totals["total_ns"] for totals in kernel_totals.values()))
    delta = abs(kernel_total - timeline_gpu_total)
    ratio = 100000 if kernel_total == 0 and timeline_gpu_total == 0 else 0
    if kernel_total > 0:
        ratio = int(round(100000.0 * timeline_gpu_total / kernel_total))
    status = "ok" if ratio >= int(round(args.coverage_ratio_threshold * 100000.0)) else "coverage_mismatch"

    aggregate_gap_classes: Counter[str] = Counter()
    for gap_classes in gap_classes_by_queue.values():
        aggregate_gap_classes.update(gap_classes)
    host_overlap = aggregate_gap_classes["host_overlap"]
    queue_serialization = aggregate_gap_classes["queue_serialization"]
    runtime_idle = aggregate_gap_classes["runtime_idle"]
    unknown_residual = max(0, wall_total - timeline_gpu_total - host_overlap - queue_serialization - runtime_idle)

    print(f"ledger.wall_ms_x1000 {wall_total}")
    print(f"ledger.timeline_gpu_event_ms_x1000 {timeline_gpu_total}")
    print(f"ledger.kernel_profile_total_ms_x1000 {kernel_total}")
    print(f"ledger.timeline_kernel_delta_ms_x1000 {delta}")
    print(f"ledger.timeline_kernel_ratio_pct_x1000 {ratio}")
    print(f"ledger.coverage_status {status}")
    print(f"ledger.gap_class.host_overlap_ms_x1000 {host_overlap}")
    print(f"ledger.gap_class.queue_serialization_ms_x1000 {queue_serialization}")
    print(f"ledger.gap_class.runtime_idle_ms_x1000 {runtime_idle}")
    print(f"ledger.unknown_wall_residual_ms_x1000 {unknown_residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
