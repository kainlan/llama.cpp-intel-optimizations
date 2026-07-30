#!/usr/bin/env python3
"""Break a SYCL timeline's device-gap classification down by *cause*, and rank
event transitions *within* a class rather than across all classes pooled.

`parse-sycl-timeline.py` answers "how much gap time is `host_overlap` /
`queue_serialization` / `runtime_idle`".  It deliberately does not answer the two
follow-up questions that decide whether a dominant `runtime_idle` is actionable:

1. **Why** did each gap land in `runtime_idle`?  That class is the `else` branch
   of the classifier, so it absorbs both genuine runtime latency and every gap
   the first two tests merely failed to *prove* something about.  A verdict of
   "`runtime_idle` dominates" is only actionable once the residual is separated
   from the signal.  See `gap_cause_names` for the split.

   One of those causes is not an instrument artifact but a second mechanism.
   `device_gap_has_dependency` recognises a dependency only through an explicit
   `depends_on` event id, and ggml-sycl's GPU queues are all in-order (see the
   queue construction in `ggml/src/ggml-sycl/common.hpp`), so they serialize
   *implicitly*, declaring no edge at all.  `queue_serialization` is therefore
   close to unreachable on this backend whatever the truth is, and an unknown
   share of the residual may be serialization wearing the residual's name.
   `implicit_queue_serialization` separates it out.

2. **Which transitions** carry a given class?  `parse-sycl-timeline.py`'s
   `--top-gaps` ranks transitions with all three classes pooled together.  That
   ranking is actively misleading for fix selection, because `host_overlap` and
   `runtime_idle` call for opposite responses -- the first says the host is busy
   and overlap is working, the second says nobody is doing anything.  A
   transition can top the pooled table purely on `host_overlap` and be nearly
   absent from the class you actually intend to fix.

Both outputs are diagnostics.  This script does not reclassify anything and does
not change `parse-sycl-timeline.py`'s verdict; it explains it.

Usage:
    parse-sycl-gap-causes.py <trace.json> [--queue compute] [--top-transitions 12]

Note the trace is Chrome Trace format: events live under `traceEvents`, not
`events`.  A parse keyed on `events` returns zero and is indistinguishable from
an empty capture.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

# The sibling parser is the single source of truth for the classification
# itself.  Load it by path rather than reimplementing: any drift between the two
# would silently invalidate every number this script prints.  Its filename
# contains dashes, so a plain import will not reach it.
_TIMELINE_PARSER_PATH = pathlib.Path(__file__).with_name("parse-sycl-timeline.py")


def load_timeline_parser(path: pathlib.Path = _TIMELINE_PARSER_PATH) -> Any:
    spec = importlib.util.spec_from_file_location("parse_sycl_timeline", path)
    if spec is None or spec.loader is None:
        raise OSError(f"cannot load timeline parser from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Why a gap landed in `runtime_idle`.  Only `truly_idle` is a finding; the rest
# are instrument artifacts and must be subtracted before any fix is scoped from
# the class total.
#
# No `sycl.submit` record for one or both events, so `host_overlap` could not be
# tested at all.  A large share here means the class total is an artifact of
# missing instrumentation, not a measurement -- treat the whole verdict as void
# until it is explained.
CAUSE_NO_SUBMIT_SPAN = "no_submit_span"

# The next op's submit began before the previous op's submit ended, i.e. the
# host ran ahead and batched submissions.  `device_gap_has_host_overlap` returns
# early here, so these never get an overlap test.
CAUSE_SUBMIT_PIPELINED_AHEAD = "submit_pipelined_ahead"

# Host work *does* cover at least half the gap once the covering spans are
# summed, but no single span clears the bar.  Only meaningful under the "max"
# coverage policy, and only as a diagnostic *of that policy's defect*: these are
# `host_overlap` misfiled as `runtime_idle`.  Retired under "union", which
# already counts merged coverage -- there is no broader metric left to appeal
# to, and a plain sum would only differ by double-counting overlapping spans,
# which is an error rather than a reclassification.
CAUSE_SUM_COVERS_MAX_DOES_NOT = "sum_covers_max_does_not"

# The successor was already handed to the queue when the predecessor finished,
# so the in-order queue -- not the host, and not any declared dependency -- is
# what held it back.  This is the class `device_gap_has_dependency` cannot see:
# an in-order queue serializes without declaring a `depends_on` edge, so these
# gaps fall through to `runtime_idle` by construction.  Not an instrument
# artifact; a second mechanism, and the one that has to be excluded before a
# residual gap can be read as launch overhead.
CAUSE_IMPLICIT_QUEUE_SERIALIZATION = "implicit_queue_serialization"

# Submits are present and ordered, host `compute_forward_node` work does not
# cover the gap, and the successor was not yet queued when the predecessor
# finished.  The real residual: time explained by neither our host work, nor a
# declared dependency, nor the queue's own ordering.
CAUSE_TRULY_IDLE = "truly_idle"

# `sycl-kernel-profiler.cpp` emits `device_submit_ns=0` when the runtime gave it
# no submit timestamp, so 0 means "unavailable", not "submitted at time zero".
# Taken at face value it precedes every predecessor's completion and would make
# every such gap read as implicit serialization.
DEVICE_SUBMIT_UNAVAILABLE_NS = 0.0


def gap_cause_names(coverage: str) -> tuple[str, ...]:
    """Causes reachable under a given `HOST_OVERLAP_COVERAGE` policy.

    Kept policy-dependent rather than fixed so a retired cause reports as
    *absent from the schema* instead of as a hard zero -- a zero would be
    indistinguishable from a real instrument defect that never fired.
    """
    if coverage == "max":
        return (
            CAUSE_NO_SUBMIT_SPAN,
            CAUSE_SUBMIT_PIPELINED_AHEAD,
            CAUSE_SUM_COVERS_MAX_DOES_NOT,
            CAUSE_IMPLICIT_QUEUE_SERIALIZATION,
            CAUSE_TRULY_IDLE,
        )
    return (
        CAUSE_NO_SUBMIT_SPAN,
        CAUSE_SUBMIT_PIPELINED_AHEAD,
        CAUSE_IMPLICIT_QUEUE_SERIALIZATION,
        CAUSE_TRULY_IDLE,
    )


def sum_host_node_overlap_us(nodes: list[tuple[float, float, str]], start_us: float, end_us: float) -> float:
    """Total host-node coverage of [start_us, end_us).

    The classifier uses `max_host_node_overlap_us` -- the single longest
    covering span -- which under-reports coverage whenever a gap is spanned by
    many short ops.  A decode step submitting ~461 launches is exactly that
    shape, so the two figures are reported side by side rather than one being
    assumed correct.
    """
    total = 0.0
    if end_us <= start_us:
        return total
    for node_start, node_end, _ in nodes:
        if node_end <= start_us:
            continue
        if node_start >= end_us:
            break
        total += min(node_end, end_us) - max(node_start, start_us)
    return total


def device_submit_ns(timeline: Any, event: dict[str, Any]) -> float | None:
    """The successor's submit timestamp on the *device* profiling clock.

    `sycl-kernel-profiler.cpp` writes `device_submit_ns` into the `sycl.event`
    metadata alongside `device_start_ns` / `device_end_ns`, all three on the same
    device clock -- which is what makes them comparable.  `host_submit_*_us` is
    on the host steady clock and is *not* comparable with a device timestamp, so
    it cannot answer this question.

    Returns None when the field is absent (an older capture) or carries the
    profiler's 0 sentinel.  A present-but-non-numeric value is left to raise, as
    every other numeric arg in this parser does: a broken instrument should be
    loud rather than silently reported as "no serialization observed".
    """
    raw = timeline.arg_or_metadata_field(event, "device_submit_ns")
    if raw in (None, ""):
        return None
    value = timeline.numeric_arg_field(event, "device_submit_ns")
    if value <= DEVICE_SUBMIT_UNAVAILABLE_NS:
        return None
    return value


def device_gap_has_implicit_serialization(
    timeline: Any,
    next_event: dict[str, Any],
    gap_start_ns: float,
) -> bool:
    """Was the successor already queued when the predecessor finished?

    Callers form gaps per (device, queue_kind), so `next_event` is on the same
    queue as the predecessor whose completion is `gap_start_ns`, and every
    ggml-sycl GPU queue is in-order.  A successor submitted before that
    completion could therefore not have started any earlier however idle the
    device looked -- the queue's own ordering held it, and the gap is
    serialization rather than idleness.

    Strict `<`: a submit landing exactly at the predecessor's completion is not
    proof the queue held anything.
    """
    submit_ns = device_submit_ns(timeline, next_event)
    if submit_ns is None:
        return False
    return submit_ns < gap_start_ns


def classify_runtime_idle_cause(
    timeline: Any,
    previous_event: dict[str, Any],
    next_event: dict[str, Any],
    gap_start_ns: float,
    gap_ns: float,
    submits_by_event_id: dict[str, tuple[float, float, str, dict[str, Any], str | None]],
    nodes: list[tuple[float, float, str]],
) -> str:
    """Explain why one already-`runtime_idle` gap failed the `host_overlap` test.

    The first three causes mirror `device_gap_has_host_overlap`'s early returns
    in order, so each corresponds to exactly one branch of that function.
    `CAUSE_IMPLICIT_QUEUE_SERIALIZATION` is then tested last, immediately before
    the residual, so it carves out of `CAUSE_TRULY_IDLE` and nothing else --
    every other cause keeps the meaning and the value it had before it existed.

    Consequence worth stating: the reported implicit-serialization figure is a
    **lower bound**.  Gaps already attributed to `no_submit_span` or
    `submit_pipelined_ahead` are not re-examined, and some of them may be
    serialization too.  Re-testing them here would double-count, which is why
    the partition is ordered rather than overlapping.
    """
    previous_submit = timeline.submit_span_for_event(previous_event, submits_by_event_id)
    next_submit = timeline.submit_span_for_event(next_event, submits_by_event_id)
    if previous_submit is None or next_submit is None:
        return CAUSE_NO_SUBMIT_SPAN
    if next_submit[0] <= previous_submit[1]:
        return CAUSE_SUBMIT_PIPELINED_AHEAD
    required_overlap_us = (gap_ns / 1000.0) * timeline.HOST_OVERLAP_MIN_FRACTION
    if timeline.HOST_OVERLAP_COVERAGE == "max" and (
        sum_host_node_overlap_us(nodes, previous_submit[1], next_submit[0]) >= required_overlap_us
    ):
        return CAUSE_SUM_COVERS_MAX_DOES_NOT
    if device_gap_has_implicit_serialization(timeline, next_event, gap_start_ns):
        return CAUSE_IMPLICIT_QUEUE_SERIALIZATION
    return CAUSE_TRULY_IDLE


def summarize_gap_causes(
    timeline: Any,
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Re-run the sibling parser's classification, retaining per-gap detail.

    Returns, per (device, queue_kind): class totals/counts, `runtime_idle` cause
    totals/counts, and per-class transition totals -- all in the parser's
    `_ms_x1000` integer scale.
    """
    _, submits_by_event_id = timeline.collect_submit_spans(events)
    nodes = timeline.collect_host_nodes(events)

    ranges_by_queue: dict[tuple[str, str], list[tuple[float, float, str, dict[str, Any]]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or not timeline.is_sycl_event_category(str(event.get("cat", "unknown"))):
            continue
        event_range = timeline.device_range(event)
        if event_range is None:
            continue
        device, queue_kind, start_ns, end_ns = event_range
        ranges_by_queue[(device, queue_kind)].append((start_ns, end_ns, str(event.get("name", "unknown")), event))

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for queue, ranges in ranges_by_queue.items():
        class_total: Counter[str] = Counter()
        class_count: Counter[str] = Counter()
        cause_total: Counter[str] = Counter()
        cause_count: Counter[str] = Counter()
        transition_total: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        transition_count: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

        sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1], item[2]))
        if not sorted_ranges:
            continue
        previous_end = sorted_ranges[0][1]
        previous_event = sorted_ranges[0][3]
        previous_name = sorted_ranges[0][2]
        for start_ns, end_ns, name, event in sorted_ranges[1:]:
            if start_ns > previous_end:
                gap_ns = start_ns - previous_end
                gap_total = timeline.ns_to_ms_x1000(gap_ns)
                if timeline.device_gap_has_host_overlap(previous_event, event, gap_ns, submits_by_event_id, nodes):
                    gap_class = "host_overlap"
                elif gap_ns <= timeline.QUEUE_SERIALIZATION_EPSILON_NS or timeline.device_gap_has_dependency(
                    previous_event, event, submits_by_event_id
                ):
                    gap_class = "queue_serialization"
                else:
                    gap_class = "runtime_idle"
                    cause = classify_runtime_idle_cause(
                        timeline, previous_event, event, previous_end, gap_ns, submits_by_event_id, nodes
                    )
                    cause_total[cause] += gap_total
                    cause_count[cause] += 1
                class_total[gap_class] += gap_total
                class_count[gap_class] += 1
                transition_total[gap_class][(previous_name, name)] += gap_total
                transition_count[gap_class][(previous_name, name)] += 1
            if end_ns >= previous_end:
                previous_end = end_ns
                previous_event = event
                previous_name = name

        result[queue] = {
            "class_total": class_total,
            "class_count": class_count,
            "cause_total": cause_total,
            "cause_count": cause_count,
            "transition_total": transition_total,
            "transition_count": transition_count,
        }
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Explain a SYCL timeline's runtime_idle gaps by cause, and rank transitions within each class",
        allow_abbrev=False,
    )
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument(
        "--queue",
        default="compute",
        help="queue kind to report transitions for; use 'all' for every queue (default: compute)",
    )
    parser.add_argument(
        "--top-transitions",
        type=int,
        default=12,
        help="per-class transition rows to print; 0 disables (default: 12)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="decode steps in the capture window; when >0 also print per-step figures",
    )
    args = parser.parse_args(argv)

    if args.top_transitions < 0:
        parser.error("--top-transitions must be non-negative")
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    try:
        timeline = load_timeline_parser()
        events = timeline.load_trace_events(args.trace)
        summaries = summarize_gap_causes(timeline, events)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse timeline: {exc}")
        return 2

    for (device, queue_kind), summary in sorted(summaries.items()):
        if args.queue != "all" and queue_kind != args.queue:
            continue
        prefix = f"gap_cause.device{device}.{queue_kind}"
        for gap_class in timeline.GAP_CLASS_NAMES:
            print(f"{prefix}.class.{gap_class}.total_ms_x1000 {summary['class_total'][gap_class]}")
            print(f"{prefix}.class.{gap_class}.count {summary['class_count'][gap_class]}")
        print(f"{prefix}.host_overlap_coverage {timeline.HOST_OVERLAP_COVERAGE}")
        for cause in gap_cause_names(timeline.HOST_OVERLAP_COVERAGE):
            print(f"{prefix}.runtime_idle.{cause}.total_ms_x1000 {summary['cause_total'][cause]}")
            print(f"{prefix}.runtime_idle.{cause}.count {summary['cause_count'][cause]}")
            if args.steps > 0:
                per_step = summary["cause_total"][cause] / args.steps
                print(f"{prefix}.runtime_idle.{cause}.per_step_ms_x1000 {int(round(per_step))}")

        if args.top_transitions <= 0:
            continue
        for gap_class in timeline.GAP_CLASS_NAMES:
            rows = sorted(
                summary["transition_total"][gap_class].items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[: args.top_transitions]
            for (previous_name, next_name), total in rows:
                count = summary["transition_count"][gap_class][(previous_name, next_name)]
                token = (
                    f"{timeline.sanitize_metric_token(previous_name)}"
                    f"--to--{timeline.sanitize_metric_token(next_name)}"
                )
                row_prefix = f"gap_transition_by_class.device{device}.{queue_kind}.{gap_class}.{token}"
                print(f"{row_prefix}.total_ms_x1000 {total}")
                print(f"{row_prefix}.count {count}")
                if args.steps > 0:
                    print(f"{row_prefix}.per_step_ms_x1000 {int(round(total / args.steps))}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
