#!/usr/bin/env python3
"""Summarize host-side SYCL timeline Chrome Trace artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any


GAP_CLASS_NAMES = ("host_overlap", "queue_serialization", "runtime_idle")
HOST_OVERLAP_MIN_FRACTION = 0.5
QUEUE_SERIALIZATION_EPSILON_NS = 1000.0
DEPENDENCY_FIELD_NAMES = (
    "depends_on",
    "depends_on_event_id",
    "depends_on_event_ids",
    "dependency_event_id",
    "dependency_event_ids",
    "dep_event_id",
    "dep_event_ids",
    "deps",
)


def parse_wall_ms(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid wall millisecond value: {raw}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("--wall-ms must be finite and greater than zero")
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


def ns_to_ms_x1000(ns: float) -> int:
    # Device timestamps are reported in nanoseconds.  x1000-ms is the same
    # scale as microseconds, so convert nanoseconds to microseconds.
    return int(round(ns / 1000.0))


def metadata_arg_fields(event: dict[str, Any]) -> dict[str, str]:
    args = event.get("args")
    if not isinstance(args, dict):
        return {}
    metadata = args.get("metadata")
    if not isinstance(metadata, str):
        return {}

    fields: dict[str, str] = {}
    for part in metadata.split(";"):
        key, separator, value = part.partition("=")
        if separator and key:
            fields.setdefault(key, value)
    return fields


def arg_or_metadata_field(event: dict[str, Any], name: str) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    raw = args.get(name)
    if raw not in (None, ""):
        return raw
    return metadata_arg_fields(event).get(name)


def metric_identifier(raw: Any) -> str | None:
    if raw in (None, "") or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        if not math.isfinite(raw):
            return None
        return str(int(raw)) if raw.is_integer() else str(raw)
    return str(raw)


def string_arg_field(event: dict[str, Any], name: str) -> str | None:
    return metric_identifier(arg_or_metadata_field(event, name))


def event_id(event: dict[str, Any]) -> str | None:
    return string_arg_field(event, "event_id")


def event_dependency_ids(event: dict[str, Any]) -> set[str]:
    dependencies: set[str] = set()
    for name in DEPENDENCY_FIELD_NAMES:
        raw = arg_or_metadata_field(event, name)
        if raw in (None, "") or isinstance(raw, bool):
            continue
        if isinstance(raw, list):
            for item in raw:
                item_id = metric_identifier(item)
                if item_id is not None:
                    dependencies.add(item_id)
            continue
        raw_id = metric_identifier(raw)
        if raw_id is None:
            continue
        for token in raw_id.replace(",", " ").replace("|", " ").replace("[", " ").replace("]", " ").split():
            if token:
                dependencies.add(token)
    return dependencies


def numeric_arg_field(event: dict[str, Any], name: str) -> float:
    raw = arg_or_metadata_field(event, name)
    if isinstance(raw, bool):
        raise ValueError(f"invalid boolean args.{name} for event {event.get('name', 'unknown')}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric args.{name} for event {event.get('name', 'unknown')}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"invalid non-finite args.{name} for event {event.get('name', 'unknown')}: {raw!r}")
    return value


def device_range(event: dict[str, Any]) -> tuple[str, str, float, float] | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    device_raw = arg_or_metadata_field(event, "device")
    queue_kind_raw = arg_or_metadata_field(event, "queue_kind")
    if any(value in (None, "") for value in (device_raw, queue_kind_raw,
                                             arg_or_metadata_field(event, "device_start_ns"),
                                             arg_or_metadata_field(event, "device_end_ns"))):
        return None

    device = str(device_raw)
    queue_kind = str(queue_kind_raw)
    start_ns = numeric_arg_field(event, "device_start_ns")
    end_ns = numeric_arg_field(event, "device_end_ns")
    if end_ns < start_ns:
        return None
    return device, queue_kind, start_ns, end_ns


def summarize_queue_gaps(
    ranges_by_queue: dict[tuple[str, str], list[tuple[float, float]]],
) -> dict[tuple[str, str], tuple[int, int]]:
    gaps: dict[tuple[str, str], tuple[int, int]] = {}
    for queue, ranges in ranges_by_queue.items():
        if not ranges:
            continue
        sorted_ranges = sorted(ranges)
        previous_end = sorted_ranges[0][1]
        gap_count = 0
        gap_ns = 0.0
        for start_ns, end_ns in sorted_ranges[1:]:
            if start_ns > previous_end:
                gap_count += 1
                gap_ns += start_ns - previous_end
            previous_end = max(previous_end, end_ns)
        gaps[queue] = (gap_count, ns_to_ms_x1000(gap_ns))
    return gaps


def sanitize_metric_token(value: str) -> str:
    result = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            result.append(ch)
        else:
            result.append("_")
    return "".join(result) if result else "unknown"


def event_metadata_op(event: dict[str, Any]) -> str:
    return sanitize_metric_token(metadata_arg_fields(event).get("op") or "unknown_op")


def collect_submit_spans(
    events: list[dict[str, Any]],
) -> tuple[
    list[tuple[float, float, str, dict[str, Any], str | None]],
    dict[str, tuple[float, float, str, dict[str, Any], str | None]],
]:
    submits: list[tuple[float, float, str, dict[str, Any], str | None]] = []
    submits_by_event_id: dict[str, tuple[float, float, str, dict[str, Any], str | None]] = {}
    for event in events:
        if event.get("ph") != "X" or str(event.get("cat", "unknown")) != "sycl.submit":
            continue
        ts = numeric_field(event, "ts")
        dur = numeric_field(event, "dur")
        submit_event_id = event_id(event)
        submit = (ts, ts + dur, sanitize_metric_token(str(event.get("name", "unknown"))), event, submit_event_id)
        submits.append(submit)
        if submit_event_id is not None:
            submits_by_event_id.setdefault(submit_event_id, submit)
    submits.sort(key=lambda item: (item[0], item[1], item[2]))
    return submits, submits_by_event_id


def collect_host_nodes(events: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    nodes: list[tuple[float, float, str]] = []
    for event in events:
        if (
            event.get("ph") != "X"
            or str(event.get("cat", "unknown")) != "ggml.op"
            or str(event.get("name", "unknown")) != "compute_forward_node"
        ):
            continue
        ts = numeric_field(event, "ts")
        dur = numeric_field(event, "dur")
        nodes.append((ts, ts + dur, event_metadata_op(event)))
    nodes.sort(key=lambda item: (item[0], item[1], item[2]))
    return nodes


def host_node_overlaps_by_op(nodes: list[tuple[float, float, str]], start_us: float, end_us: float) -> Counter[str]:
    totals: Counter[str] = Counter()
    if end_us <= start_us:
        return totals
    for node_start, node_end, op in nodes:
        if node_end <= start_us:
            continue
        if node_start >= end_us:
            break
        overlap = max(0.0, min(node_end, end_us) - max(node_start, start_us))
        overlap_total = us_to_ms_x1000(overlap)
        if overlap_total > 0:
            totals[op] += overlap_total
    return totals


def max_host_node_overlap_us(nodes: list[tuple[float, float, str]], start_us: float, end_us: float) -> float:
    max_overlap = 0.0
    if end_us <= start_us:
        return max_overlap
    for node_start, node_end, _ in nodes:
        if node_end <= start_us:
            continue
        if node_start >= end_us:
            break
        max_overlap = max(max_overlap, min(node_end, end_us) - max(node_start, start_us))
    return max_overlap


def summarize_host_gap_overlaps(events: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    submits, _ = collect_submit_spans(events)
    nodes = collect_host_nodes(events)
    totals: Counter[tuple[str, str, str]] = Counter()
    for (_, previous_end, previous_name, _, _), (next_start, _, next_name, _, _) in zip(submits, submits[1:]):
        if next_start <= previous_end:
            continue
        for op, total in host_node_overlaps_by_op(nodes, previous_end, next_start).items():
            totals[(previous_name, next_name, op)] += total
    return totals


def submit_span_for_event(
    event: dict[str, Any],
    submits_by_event_id: dict[str, tuple[float, float, str, dict[str, Any], str | None]],
) -> tuple[float, float, str, dict[str, Any], str | None] | None:
    current_event_id = event_id(event)
    if current_event_id is not None and current_event_id in submits_by_event_id:
        return submits_by_event_id[current_event_id]

    begin_raw = arg_or_metadata_field(event, "host_submit_begin_us")
    end_raw = arg_or_metadata_field(event, "host_submit_end_us")
    if begin_raw in (None, "") or end_raw in (None, ""):
        return None
    begin_us = numeric_arg_field(event, "host_submit_begin_us")
    end_us = numeric_arg_field(event, "host_submit_end_us")
    if end_us < begin_us:
        return None
    return (begin_us, end_us, sanitize_metric_token(str(event.get("name", "unknown"))), event, current_event_id)


def device_gap_has_host_overlap(
    previous_event: dict[str, Any],
    next_event: dict[str, Any],
    gap_ns: float,
    submits_by_event_id: dict[str, tuple[float, float, str, dict[str, Any], str | None]],
    nodes: list[tuple[float, float, str]],
) -> bool:
    previous_submit = submit_span_for_event(previous_event, submits_by_event_id)
    next_submit = submit_span_for_event(next_event, submits_by_event_id)
    if previous_submit is None or next_submit is None:
        return False
    previous_end_us = previous_submit[1]
    next_start_us = next_submit[0]
    if next_start_us <= previous_end_us:
        return False
    required_overlap_us = (gap_ns / 1000.0) * HOST_OVERLAP_MIN_FRACTION
    return max_host_node_overlap_us(nodes, previous_end_us, next_start_us) >= required_overlap_us


def device_gap_has_dependency(
    previous_event: dict[str, Any],
    next_event: dict[str, Any],
    submits_by_event_id: dict[str, tuple[float, float, str, dict[str, Any], str | None]],
) -> bool:
    previous_event_id = event_id(previous_event)
    if previous_event_id is None:
        return False
    if previous_event_id == event_id(next_event):
        return True
    dependency_ids = event_dependency_ids(next_event)
    next_submit = submit_span_for_event(next_event, submits_by_event_id)
    if next_submit is not None:
        dependency_ids.update(event_dependency_ids(next_submit[3]))
    return previous_event_id in dependency_ids


def summarize_queue_gap_classes(events: list[dict[str, Any]]) -> dict[tuple[str, str], Counter[str]]:
    _, submits_by_event_id = collect_submit_spans(events)
    nodes = collect_host_nodes(events)
    ranges_by_queue: dict[tuple[str, str], list[tuple[float, float, str, dict[str, Any]]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or not is_sycl_event_category(str(event.get("cat", "unknown"))):
            continue
        event_range = device_range(event)
        if event_range is None:
            continue
        device, queue_kind, start_ns, end_ns = event_range
        ranges_by_queue[(device, queue_kind)].append((start_ns, end_ns, str(event.get("name", "unknown")), event))

    result: dict[tuple[str, str], Counter[str]] = {}
    for queue, ranges in ranges_by_queue.items():
        totals: Counter[str] = Counter()
        sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1], item[2]))
        if not sorted_ranges:
            result[queue] = totals
            continue
        previous_end = sorted_ranges[0][1]
        previous_event = sorted_ranges[0][3]
        for start_ns, end_ns, _, event in sorted_ranges[1:]:
            if start_ns > previous_end:
                gap_ns = start_ns - previous_end
                gap_total = ns_to_ms_x1000(gap_ns)
                if device_gap_has_host_overlap(previous_event, event, gap_ns, submits_by_event_id, nodes):
                    gap_class = "host_overlap"
                elif gap_ns <= QUEUE_SERIALIZATION_EPSILON_NS or device_gap_has_dependency(
                    previous_event, event, submits_by_event_id
                ):
                    gap_class = "queue_serialization"
                else:
                    gap_class = "runtime_idle"
                totals[gap_class] += gap_total
            if end_ns >= previous_end:
                previous_end = end_ns
                previous_event = event
        result[queue] = totals
    return result


def summarize_queue_gap_transitions(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[tuple[str, str], tuple[int, int, int]]]:
    ranges_by_queue: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or not is_sycl_event_category(str(event.get("cat", "unknown"))):
            continue
        event_range = device_range(event)
        if event_range is None:
            continue
        device, queue_kind, start_ns, end_ns = event_range
        ranges_by_queue[(device, queue_kind)].append((start_ns, end_ns, str(event.get("name", "unknown"))))

    result: dict[tuple[str, str], dict[tuple[str, str], tuple[int, int, int]]] = {}
    for queue, ranges in ranges_by_queue.items():
        rows: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        sorted_ranges = sorted(ranges)
        if not sorted_ranges:
            continue
        previous_end = sorted_ranges[0][1]
        previous_name = sorted_ranges[0][2]
        for start_ns, end_ns, name in sorted_ranges[1:]:
            if start_ns > previous_end:
                gap_total = ns_to_ms_x1000(start_ns - previous_end)
                row = rows[(previous_name, name)]
                row[0] += 1
                row[1] += gap_total
                row[2] = max(row[2], gap_total)
            if end_ns >= previous_end:
                previous_end = end_ns
                previous_name = name
        result[queue] = {key: (value[0], value[1], value[2]) for key, value in rows.items()}
    return result


def summarize_events(
    events: list[dict[str, Any]],
) -> tuple[
    Counter[str],
    Counter[str],
    float,
    int,
    dict[tuple[str, str], tuple[int, int]],
    dict[tuple[str, str], dict[tuple[str, str], tuple[int, int, int]]],
]:
    category_totals: Counter[str] = Counter()
    callsite_totals: Counter[str] = Counter()
    ranges_by_queue: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    gpu_event_total_us = 0.0
    span_start: float | None = None
    span_end: float | None = None

    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", "unknown"))
        if is_sycl_event_category(category):
            event_range = device_range(event)
            if event_range is not None:
                device, queue_kind, start_ns, end_ns = event_range
                ranges_by_queue[(device, queue_kind)].append((start_ns, end_ns))
                gpu_event_total_us += max(0.0, end_ns - start_ns) / 1000.0
            else:
                gpu_event_total_us += numeric_field(event, "dur")
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
    return (
        category_totals,
        callsite_totals,
        wall_us,
        us_to_ms_x1000(gpu_event_total_us),
        summarize_queue_gaps(ranges_by_queue),
        summarize_queue_gap_transitions(events),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize SYCL Chrome Trace timeline complete events", allow_abbrev=False)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--wall-ms", type=parse_wall_ms)
    parser.add_argument("--top-callsites", type=int, default=20, help="number of callsite totals to print; use 0 for all")
    parser.add_argument(
        "--top-gaps",
        type=int,
        default=0,
        help="number of per-queue event-transition gap totals to print; 0 disables",
    )
    parser.add_argument(
        "--top-host-gap-overlaps",
        type=int,
        default=0,
        help="number of submit-gap host op overlap totals to print; 0 disables",
    )
    args = parser.parse_args(argv)

    if args.top_callsites < 0:
        parser.error("--top-callsites must be non-negative")
    if args.top_gaps < 0:
        parser.error("--top-gaps must be non-negative")
    if args.top_host_gap_overlaps < 0:
        parser.error("--top-host-gap-overlaps must be non-negative")

    try:
        events = load_trace_events(args.trace)
        category_totals, callsite_totals, envelope_wall_us, gpu_event_total, queue_gaps, queue_gap_transitions = summarize_events(
            events
        )
        queue_gap_classes = summarize_queue_gap_classes(events)
        host_gap_overlaps: Counter[tuple[str, str, str]] = Counter()
        if args.top_host_gap_overlaps > 0:
            host_gap_overlaps = summarize_host_gap_overlaps(events)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse timeline: {exc}")
        return 2

    wall_us = args.wall_ms * 1000.0 if args.wall_ms is not None else envelope_wall_us
    wall_total = us_to_ms_x1000(wall_us)
    gpu_event_coverage = int(round(gpu_event_total / wall_total * 100000.0)) if wall_total > 0 else 0
    unattributed_total = max(0, wall_total - gpu_event_total)

    print(f"timeline.wall_ms_x1000 {wall_total}")
    print(f"timeline.gpu_event_total_ms_x1000 {gpu_event_total}")
    print(f"timeline.gpu_event_coverage_pct_x1000 {gpu_event_coverage}")
    print(f"timeline.unattributed_ms_x1000 {unattributed_total}")

    for (device, queue_kind), (gap_count, gap_total) in sorted(queue_gaps.items()):
        print(f"gap.device{device}.{queue_kind}.count {gap_count}")
        print(f"gap.device{device}.{queue_kind}.total_ms_x1000 {gap_total}")
        gap_classes = queue_gap_classes.get((device, queue_kind), Counter())
        gap_class_totals = {gap_class: gap_classes[gap_class] for gap_class in GAP_CLASS_NAMES}
        rounding_delta = gap_total - sum(gap_class_totals.values())
        if rounding_delta != 0:
            adjustment_class = "runtime_idle"
            if gap_class_totals[adjustment_class] + rounding_delta < 0:
                adjustment_class = max(GAP_CLASS_NAMES, key=lambda gap_class: gap_class_totals[gap_class])
            gap_class_totals[adjustment_class] += rounding_delta
        for gap_class in GAP_CLASS_NAMES:
            print(f"gap_class.device{device}.{queue_kind}.{gap_class}.total_ms_x1000 {gap_class_totals[gap_class]}")

    if args.top_gaps > 0:
        for (device, queue_kind), transitions in sorted(queue_gap_transitions.items()):
            rows = sorted(transitions.items(), key=lambda item: (-item[1][1], item[0][0], item[0][1]))[: args.top_gaps]
            for (previous_name, next_name), (count, total, max_gap) in rows:
                previous_token = sanitize_metric_token(previous_name)
                next_token = sanitize_metric_token(next_name)
                prefix = f"gap_transition.device{device}.{queue_kind}.{previous_token}--to--{next_token}"
                print(f"{prefix}.count {count}")
                print(f"{prefix}.total_ms_x1000 {total}")
                print(f"{prefix}.max_ms_x1000 {max_gap}")

    for category, total in sorted(category_totals.items()):
        print(f"category.{category}.host_ms_x1000 {total}")

    sorted_callsites = sorted(callsite_totals.items(), key=lambda item: (-item[1], item[0]))
    if args.top_callsites > 0:
        sorted_callsites = sorted_callsites[: args.top_callsites]
    for callsite, total in sorted_callsites:
        print(f"callsite.{callsite}.host_ms_x1000 {total}")

    if args.top_host_gap_overlaps > 0:
        rows = sorted(host_gap_overlaps.items(), key=lambda item: (-item[1], item[0]))[: args.top_host_gap_overlaps]
        for (previous_name, next_name, op), total in rows:
            print(f"host_gap_overlap.{previous_name}--to--{next_name}.{op}.host_ms_x1000 {total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
