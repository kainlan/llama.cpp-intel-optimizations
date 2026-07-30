#!/usr/bin/env python3
"""Split each SYCL device gap into "the host had not submitted yet" and
"submitted, but the queue would not start it", both on the *device* clock.

`parse-sycl-gap-causes.py` answers *why a gap failed the `host_overlap` test*.
It cannot answer the question that decides whether a `truly_idle` gap is
actionable: **was the device actually idle?**  It only ever sees the kernels the
profiler was asked to label, and labelling is opt-in per submission site
(`ggml_sycl_profile_label`).  On a GPT-OSS decode step 339 of the backend's
compute-queue kernels per step carry a label and the dense matmuls do not, so a
gap between two labelled kernels routinely contains an unlabelled one -- device
work that the gap classifier necessarily reads as idleness.

This script separates the two using only the three device-clock fields the
profiler already writes (`device_submit_ns`, `device_start_ns`,
`device_end_ns` -- never `host_submit_*_us`, which is a different clock):

    A = clamp(device_submit_ns(next) - device_end_ns(prev), 0, gap)
        The successor was not yet submitted.  The host is behind; whether the
        device was idle during A is NOT decided by this term.
    B = gap - A
        The successor was already submitted and still did not start.  Callers
        form gaps per (device, queue_kind) and every ggml-sycl GPU queue is
        in-order, so on this backend B can only be earlier queued work still
        executing, plus the bare submit-to-start dispatch latency.

**B is the finding.** Calibrate the dispatch-latency floor from transitions
where no graph node intervenes -- consecutive kernels of one fused dispatch, e.g.
`mxfp4.quantize.activation_q8_soa --to-- mxfp4.pack_q8.single_col`, whose
`node_idx` are equal.  On the 2026-07-30 B70 captures that floor is 0.35-0.7 us,
so any B of tens of microseconds is unrecorded device execution rather than
latency.

`hostint` / `hostcov` are reported alongside as the *host*-clock counterpart:
the interval between the two `sycl.submit` spans, and how much of it lies inside
an instrumented `ggml.op compute_forward_node` span.  A low `hostcov` does not
mean the host was idle -- the node loop `continue`s past that timeline scope on
its fusion fast-paths -- it means the host work there is uninstrumented.  Do not
read `hostint - hostcov` as idle host time.

Read-only diagnostic.  It reclassifies nothing and contradicts no figure printed
by the sibling parsers; it adds a column they do not have.

Usage:
    parse-sycl-gap-device-split.py <trace.json> [--queue compute] [--steps 100]
                                   [--top-transitions 22]

Note the trace is Chrome Trace format: events live under `traceEvents`, not
`events`.  A parse keyed on `events` returns zero and is indistinguishable from
an empty capture.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any


def ns_to_ms_x1000(value_ns: float) -> int:
    """Same convention (and formula) as parse-sycl-timeline.py's helper of the
    same name: nanoseconds -> microseconds, which is the "ms x1000" integer
    scale the sibling parsers use for every greppable metric. Duplicated
    locally rather than imported, matching this file's existing standalone
    style (metadata_fields / load_trace_events are already self-contained
    reimplementations, not imports of the sibling modules)."""
    return int(round(value_ns / 1000.0))


def sanitize_metric_token(value: str) -> str:
    """Matches parse-sycl-timeline.py's helper of the same name, so a
    transition token stays a single space-free `key value` token."""
    result = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            result.append(ch)
        else:
            result.append("_")
    return "".join(result) if result else "unknown"


def metadata_fields(event: dict[str, Any]) -> dict[str, str]:
    """The profiler packs its per-event fields into one `;`-separated string."""
    args = event.get("args")
    if not isinstance(args, dict):
        return {}
    fields: dict[str, str] = {}
    raw = args.get("metadata")
    if isinstance(raw, str):
        for part in raw.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields.setdefault(key, value)
    return fields


def load_trace_events(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        document = json.load(handle)
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"{path}: no traceEvents array (Chrome trace format expected)")
    return events


def collect_submit_spans(events: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    spans: dict[str, tuple[float, float]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "sycl.submit":
            continue
        event_id = metadata_fields(event).get("event_id")
        if event_id is None or event_id in spans:
            continue
        # Same field-presence discipline as collect_device_events: a submit
        # span missing ts/dur is skipped, not a crash -- one convention for
        # malformed events across the file rather than two.
        ts, dur = event.get("ts"), event.get("dur")
        if ts in (None, "") or dur in (None, ""):
            continue
        try:
            start = float(ts)
            spans[event_id] = (start, start + float(dur))
        except (TypeError, ValueError):
            continue
    return spans


def collect_host_nodes(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    nodes: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "ggml.op":
            continue
        if event.get("name") != "compute_forward_node":
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if ts in (None, "") or dur in (None, ""):
            continue
        try:
            start = float(ts)
            nodes.append((start, start + float(dur)))
        except (TypeError, ValueError):
            continue
    nodes.sort()
    return nodes


def host_node_coverage_us(nodes: list[tuple[float, float]], start_us: float, end_us: float) -> float:
    """Merged host-node coverage of [start_us, end_us); spans are already sorted
    and non-overlapping in practice, so a plain clipped sum is the union."""
    total = 0.0
    if end_us <= start_us:
        return total
    for node_start, node_end in nodes:
        if node_end <= start_us:
            continue
        if node_start >= end_us:
            break
        total += min(node_end, end_us) - max(node_start, start_us)
    return total


def collect_device_events(
    events: list[dict[str, Any]], queue_kind: str
) -> list[tuple[float, float, float, str, int, str | None]]:
    device_events: list[tuple[float, float, float, str, int, str | None]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "sycl.event":
            continue
        fields = metadata_fields(event)
        if queue_kind != "all" and fields.get("queue_kind") != queue_kind:
            continue
        if any(fields.get(name) in (None, "") for name in ("device_start_ns", "device_end_ns", "device_submit_ns")):
            continue
        device_events.append(
            (
                float(fields["device_start_ns"]),
                float(fields["device_end_ns"]),
                float(fields["device_submit_ns"]),
                str(event.get("name", "unknown")),
                int(fields.get("node_idx", -1)),
                fields.get("event_id"),
            )
        )
    device_events.sort(key=lambda item: (item[0], item[1], item[3]))
    return device_events


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Split SYCL device gaps into host-not-submitted (A) and queue-would-not-start (B)",
        allow_abbrev=False,
    )
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--queue", default="compute", help="queue kind to report; 'all' for every queue")
    parser.add_argument("--steps", type=int, default=0, help="decode steps in the window; >0 prints per-step figures")
    parser.add_argument("--top-transitions", type=int, default=22, help="rows to print, ranked by total gap")
    parser.add_argument(
        "--format",
        choices=("table", "metrics"),
        default="table",
        help="'table' (default, unchanged) prints the human-readable columns the findings doc cites; "
        "'metrics' prints greppable 'key value' lines in the sibling parsers' _ms_x1000 style, "
        "for scripted before/after comparison (see llama.cpp-ejjq)",
    )
    args = parser.parse_args(argv)

    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.top_transitions < 0:
        parser.error("--top-transitions must be non-negative")

    try:
        events = load_trace_events(args.trace)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse timeline: {exc}")
        return 2

    submits = collect_submit_spans(events)
    host_nodes = collect_host_nodes(events)
    device_events = collect_device_events(events, args.queue)
    if not device_events:
        print(f"no sycl.event records with device timestamps on queue {args.queue!r}")
        return 2

    rows: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "gap": 0.0, "a": 0.0, "b": 0.0, "hostint": 0.0, "hostcov": 0.0, "hostn": 0, "dnode": []}
    )
    previous = device_events[0]
    for current in device_events[1:]:
        if current[0] > previous[1]:
            gap_ns = current[0] - previous[1]
            a_ns = min(max(current[2] - previous[1], 0.0), gap_ns)
            row = rows[(previous[3], current[3])]
            row["n"] += 1
            row["gap"] += gap_ns
            row["a"] += a_ns
            row["b"] += gap_ns - a_ns
            row["dnode"].append(current[4] - previous[4])
            previous_submit = submits.get(previous[5] or "")
            next_submit = submits.get(current[5] or "")
            if previous_submit and next_submit and next_submit[0] > previous_submit[1]:
                row["hostint"] += next_submit[0] - previous_submit[1]
                row["hostcov"] += host_node_coverage_us(host_nodes, previous_submit[1], next_submit[0])
                row["hostn"] += 1
        if current[1] >= previous[1]:
            previous = current

    divisor = args.steps if args.steps > 0 else 1
    unit = "ms/step" if args.steps > 0 else "ms"
    ranked = sorted(rows.items(), key=lambda item: -item[1]["gap"])
    selected = ranked[: args.top_transitions or None]

    if args.format == "table":
        print(
            f'{"transition":58} {"n":>8} {"gap " + unit:>11} {"mean_us":>8} {"A_us":>8} {"B_us":>8} '
            f'{"B%":>5} {"hostint_us":>10} {"hostcov_us":>10} {"dnodes":>6}'
        )
        for (previous_name, next_name), row in selected:
            count = row["n"]
            host_interval = row["hostint"] / row["hostn"] if row["hostn"] else float("nan")
            host_coverage = row["hostcov"] / row["hostn"] if row["hostn"] else float("nan")
            print(
                f'{previous_name + " -> " + next_name:58} {count / divisor:8.2f} {row["gap"] / divisor / 1e6:11.3f} '
                f'{row["gap"] / count / 1e3:8.2f} {row["a"] / count / 1e3:8.2f} {row["b"] / count / 1e3:8.2f} '
                f'{100 * row["b"] / row["gap"]:5.1f} {host_interval:10.2f} {host_coverage:10.2f} '
                f'{statistics.median(row["dnode"]):6.0f}'
            )
    else:
        for (previous_name, next_name), row in selected:
            token = f"{sanitize_metric_token(previous_name)}--to--{sanitize_metric_token(next_name)}"
            prefix = f"gap_device_split.{args.queue}.transition.{token}"
            print(f"{prefix}.n {row['n']}")
            print(f"{prefix}.gap_ms_x1000 {ns_to_ms_x1000(row['gap'])}")
            print(f"{prefix}.a_ms_x1000 {ns_to_ms_x1000(row['a'])}")
            print(f"{prefix}.b_ms_x1000 {ns_to_ms_x1000(row['b'])}")
            print(f"{prefix}.b_pct_x10 {round(1000 * row['b'] / row['gap'])}")
            print(f"{prefix}.hostn {row['hostn']}")
            print(f"{prefix}.hostint_ms_x1000 {int(round(row['hostint'])) if row['hostn'] else 0}")
            print(f"{prefix}.hostcov_ms_x1000 {int(round(row['hostcov'])) if row['hostn'] else 0}")
            print(f"{prefix}.dnode_median {int(round(statistics.median(row['dnode'])))}")

    gap_total = sum(row["gap"] for _, row in ranked)
    a_total = sum(row["a"] for _, row in ranked)
    b_total = sum(row["b"] for _, row in ranked)
    count_total = sum(row["n"] for _, row in ranked)

    # A trace can have device events on this queue but NO gap between any of
    # them -- one event total, or a run of events each fully contained in its
    # predecessor (`current[0] > previous[1]` never fires). `rows` is then
    # empty and gap_total is 0.0: the percentage below would divide by zero.
    # Same shape as the "no device events at all" guard above: clean message,
    # return 2, one convention for "nothing to report" rather than two.
    if gap_total == 0:
        print(
            f"no device gaps found on queue {args.queue!r} "
            f"({len(device_events)} device event(s), all back-to-back or singular)"
        )
        return 2

    busy_total = sum(end - start for start, end, _, _, _, _ in device_events)

    if args.format == "table":
        print(
            f"\nall gaps on queue {args.queue!r}: {gap_total / divisor / 1e6:.3f} {unit}   "
            f"A={a_total / divisor / 1e6:.3f}  B={b_total / divisor / 1e6:.3f} ({100 * b_total / gap_total:.1f} % B)   "
            f"n={count_total / divisor:.1f}"
        )
        print(f"device busy on that queue: {busy_total / divisor / 1e6:.3f} {unit} over {len(device_events) / divisor:.1f} events")
    else:
        prefix = f"gap_device_split.{args.queue}.total"
        print(f"{prefix}.n {count_total}")
        print(f"{prefix}.gap_ms_x1000 {ns_to_ms_x1000(gap_total)}")
        print(f"{prefix}.a_ms_x1000 {ns_to_ms_x1000(a_total)}")
        print(f"{prefix}.b_ms_x1000 {ns_to_ms_x1000(b_total)}")
        print(f"{prefix}.b_pct_x10 {round(1000 * b_total / gap_total)}")
        print(f"{prefix}.busy_ms_x1000 {ns_to_ms_x1000(busy_total)}")
        print(f"{prefix}.events {len(device_events)}")
        if args.steps > 0:
            print(f"{prefix}.gap_per_step_ms_x1000 {ns_to_ms_x1000(gap_total / args.steps)}")
            print(f"{prefix}.a_per_step_ms_x1000 {ns_to_ms_x1000(a_total / args.steps)}")
            print(f"{prefix}.b_per_step_ms_x1000 {ns_to_ms_x1000(b_total / args.steps)}")
            print(f"{prefix}.busy_per_step_ms_x1000 {ns_to_ms_x1000(busy_total / args.steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
