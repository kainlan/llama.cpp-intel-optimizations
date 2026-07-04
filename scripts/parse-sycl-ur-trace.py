#!/usr/bin/env python3
"""Summarize Unified Runtime / XPTI-style SYCL runtime trace logs."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
from collections import Counter

BUCKET_ORDER = ("enqueue", "memory", "wait", "program_kernel", "other")


def sanitize_metric_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return token or "unknown"


def parse_number(raw: str | None, field: str, name: str) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {field} for {name}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"invalid non-finite {field} for {name}: {raw!r}")
    if value < 0:
        raise ValueError(f"negative {field} for {name}")
    return value


def duration_us(fields: dict[str, str]) -> float:
    name = fields.get("name")
    if not name:
        raise ValueError("UR_TRACE line missing name")

    begin = parse_number(fields.get("begin_us"), "begin_us", name)
    end = parse_number(fields.get("end_us"), "end_us", name)
    dur = parse_number(fields.get("dur_us"), "dur_us", name)

    if begin is not None or end is not None:
        if begin is None or end is None:
            raise ValueError(f"missing begin_us/end_us pair for {name}")
        if end < begin:
            raise ValueError(f"negative duration for {name}")
        return end - begin

    if dur is None:
        raise ValueError(f"missing duration for {name}")
    if dur < 0:
        raise ValueError(f"negative duration for {name}")
    return dur


def bucket_for(name: str) -> str:
    lower = name.lower()
    if "mem" in lower or "usm" in lower:
        return "memory"
    if "enqueue" in lower or "queue" in lower:
        return "enqueue"
    if "wait" in lower or "event" in lower:
        return "wait"
    if "program" in lower or "kernel" in lower:
        return "program_kernel"
    return "other"


def parse_kv_line(raw: str) -> dict[str, str] | None:
    line = raw.strip()
    if not line or not line.startswith("UR_TRACE"):
        return None

    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key:
            continue
        fields[key] = value
    if "name" not in fields:
        raise ValueError("UR_TRACE line missing name")
    return fields


def parse_trace(path: pathlib.Path) -> tuple[Counter[str], Counter[str]]:
    bucket_totals: Counter[str] = Counter()
    api_counts: Counter[str] = Counter()

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = parse_kv_line(raw)
        if fields is None:
            continue
        name = fields["name"]
        bucket_totals[bucket_for(name)] += int(round(duration_us(fields)))
        api_counts[sanitize_metric_token(name)] += 1

    return bucket_totals, api_counts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize Unified Runtime / XPTI SYCL runtime trace logs")
    parser.add_argument("trace", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        bucket_totals, api_counts = parse_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"failed to parse UR trace: {exc}")
        return 2

    print(f"ur.total_ms_x1000 {sum(bucket_totals.values())}")
    for bucket in BUCKET_ORDER:
        print(f"ur.bucket.{bucket}.ms_x1000 {bucket_totals[bucket]}")
    for name, count in sorted(api_counts.items()):
        print(f"ur.api.{name}.count {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
