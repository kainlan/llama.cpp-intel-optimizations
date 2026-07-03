#!/usr/bin/env python3
"""Summarize host-side SYCL timeline Chrome Trace artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any


def parse_wall_ms(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid wall millisecond value: {raw}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("--wall-ms must be finite and non-negative")
    return value


def load_trace_events(path: pathlib.Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("trace JSON must be a top-level object")
    events = obj.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("trace JSON missing top-level traceEvents list")
    return [event for event in events if isinstance(event, dict)]


def numeric_field(event: dict[str, Any], name: str) -> float:
    raw = event.get(name)
    if isinstance(raw, bool):
        raise ValueError(f"invalid boolean {name} for event {event.get('name', 'unknown')}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric {name} for event {event.get('name', 'unknown')}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"invalid non-finite {name} for event {event.get('name', 'unknown')}: {raw!r}")
    return value


def is_sycl_event_category(category: str) -> bool:
    return "sycl.event" in {part.strip() for part in category.split(",")}


def callsite_key(event: dict[str, Any]) -> str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    file = args.get("file")
    line = args.get("line")
    function = args.get("function")
    if file in (None, "") or line in (None, "") or function in (None, ""):
        return None
    return f"{file}:{line}:{function}"


def us_to_ms_x1000(us: float) -> int:
    # Chrome Trace timestamps and complete-event durations are microseconds.
    # Converting us -> ms -> x1000-ms returns the original microsecond scale.
    return int(round(us))


def summarize_events(events: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], float]:
    category_totals: Counter[str] = Counter()
    callsite_totals: Counter[str] = Counter()
    span_start: float | None = None
    span_end: float | None = None

    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", "unknown"))
        if is_sycl_event_category(category):
            continue
        dur_us = numeric_field(event, "dur")
        ts_us = numeric_field(event, "ts")

        category_totals[category] += us_to_ms_x1000(dur_us)
        callsite = callsite_key(event)
        if callsite is not None:
            callsite_totals[callsite] += us_to_ms_x1000(dur_us)

        start = ts_us
        end = ts_us + dur_us
        span_start = start if span_start is None else min(span_start, start)
        span_end = end if span_end is None else max(span_end, end)

    if span_start is None or span_end is None:
        wall_us = 0.0
    else:
        wall_us = max(0.0, span_end - span_start)
    return category_totals, callsite_totals, wall_us


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize SYCL Chrome Trace timeline complete events")
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--wall-ms", type=parse_wall_ms)
    parser.add_argument("--top-callsites", type=int, default=20, help="number of callsite totals to print; use 0 for all")
    args = parser.parse_args(argv)

    if args.top_callsites < 0:
        parser.error("--top-callsites must be non-negative")

    try:
        events = load_trace_events(args.trace)
        category_totals, callsite_totals, envelope_wall_us = summarize_events(events)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse timeline: {exc}")
        return 2

    wall_us = args.wall_ms * 1000.0 if args.wall_ms is not None else envelope_wall_us
    print(f"timeline.wall_ms_x1000 {us_to_ms_x1000(wall_us)}")

    for category, total in sorted(category_totals.items()):
        print(f"category.{category}.host_ms_x1000 {total}")

    sorted_callsites = sorted(callsite_totals.items(), key=lambda item: (-item[1], item[0]))
    if args.top_callsites > 0:
        sorted_callsites = sorted_callsites[: args.top_callsites]
    for callsite, total in sorted_callsites:
        print(f"callsite.{callsite}.host_ms_x1000 {total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
