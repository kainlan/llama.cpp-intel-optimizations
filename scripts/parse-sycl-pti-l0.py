#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import Counter
from typing import Any


def sanitize(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return token or "unknown"


def ms_x1000_from_us(value: float) -> int:
    return int(round(value))


def finite_number(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def row_duration_us(row: dict[str, Any]) -> float:
    begin = finite_number(row, "begin_us")
    if begin is None:
        begin = finite_number(row, "start_us")
    end = finite_number(row, "end_us")
    dur = finite_number(row, "dur_us")
    if begin is not None and end is not None:
        if end < begin:
            raise ValueError(f"negative duration for {row.get('name', 'unknown')}")
        return end - begin
    if dur is not None:
        if dur < 0:
            raise ValueError(f"negative duration for {row.get('name', 'unknown')}")
        return dur
    raise ValueError(f"missing timestamp fields for {row.get('name', 'unknown')}")


def bucket_for(name: str) -> str:
    if "CommandQueueExecute" in name or "CommandListClose" in name or "CommandListReset" in name:
        return "queue_submit"
    if "Memory" in name or "Mem" in name:
        return "memory"
    if "Module" in name or "Kernel" in name:
        return "module_kernel"
    if "Event" in name or "Synchronize" in name:
        return "event_wait"
    return "other"


def parse_rows(path: pathlib.Path) -> tuple[Counter[str], Counter[str]]:
    bucket_totals: Counter[str] = Counter()
    api_counts: Counter[str] = Counter()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no} is not a JSON object")
        name = str(row.get("name") or row.get("api") or row.get("function") or "unknown")
        duration = row_duration_us(row)
        bucket_totals[bucket_for(name)] += ms_x1000_from_us(duration)
        api_counts[sanitize(name)] += 1
    return bucket_totals, api_counts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize Level Zero/PTI API trace JSONL")
    parser.add_argument("trace", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        buckets, api_counts = parse_rows(args.trace)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse Level Zero trace: {exc}")
        return 2
    total = sum(buckets.values())
    print(f"l0.total_ms_x1000 {total}")
    for bucket in ("queue_submit", "memory", "module_kernel", "event_wait", "other"):
        print(f"l0.bucket.{bucket}.ms_x1000 {buckets[bucket]}")
    for api, count in sorted(api_counts.items()):
        print(f"l0.api.{api}.count {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
