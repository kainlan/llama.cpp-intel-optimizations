#!/usr/bin/env python3
"""The runtime_idle cause split must exhaustively partition the runtime_idle class.

parse-sycl-timeline.py reports runtime_idle as a single number, and Task 9 of
docs/plans/2026-07-25-decode-host-overhead-findings.md reads a dominant
runtime_idle as evidence about the L0/UR submit path.  That reading is only
valid for the part of the class that is genuinely unexplained.  runtime_idle is
the classifier's `else` branch, so it also absorbs gaps that merely *failed* the
host_overlap test for instrument reasons -- a missing submit span, a submit the
host batched ahead, or host work that no single ggml node covers even though the
spans together do -- and gaps that an in-order queue explains without declaring
any dependency edge.

parse-sycl-gap-causes.py separates those.  This test gates four properties of
that split:

  * **Conservation.**  The causes must sum to the runtime_idle class total, in
    both time and count.  Unlike parse-sycl-timeline.py's class split, which is
    force-balanced by a rounding correction, nothing makes this sum come out
    right -- a gap that reaches the `else` branch and matches no cause would be
    silently dropped, and the class would read as more explained than it is.

  * **Non-vacuity.**  Every cause must be observed at least once across the
    fixtures.  A cause that can never fire is indistinguishable from a cause
    that is simply absent from the capture, and would let a real instrument
    defect report as zero.

  * **The coverage policy.**  `HOST_OVERLAP_COVERAGE` decides whether
    `host_overlap` needs one contiguous host op to cover the gap ("max") or
    merged coverage ("union").  The checked-in fixture is built so the two
    answers differ, so flipping the policy fails here rather than silently
    restating a published split.

  * **`truly_idle` does not move.**  `truly_idle` is the budget every task in
    docs/plans/2026-07-25-decode-host-overhead-findings.md is scoped from
    (960153, 9.602 ms/step, n=18840).  Any *new* cause carves out of it, so a
    new cause that fires on a gap it should not have would silently restate that
    published number.  The checked-in fixture contains no
    `implicit_queue_serialization` gap -- it carries no `device_submit_ns` field
    at all -- so its `truly_idle` figure is pinned to an exact value here.

Driven by two synthetic traces, never live hardware:

  * `test-sycl-gap-causes.trace.json`, checked in beside this file, which builds
    one gap per instrument-artifact cause; see that file's `_comment`.
  * `IMPLICIT_SERIALIZATION_TRACE` below, materialized to a temp file at run
    time.  It is kept out of the checked-in fixture on purpose: adding
    `device_submit_ns` fields there would perturb both the coverage-policy pin
    and the `truly_idle` baseline that the same fixture is holding.

cause_partition_failures() is additionally run against a deliberately corrupted
copy of the metrics and must reject it, so a test that stopped checking anything
cannot pass.

Plain script, not pytest: run with `python3 tests/<this file>`.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARSER = ROOT / "scripts" / "parse-sycl-gap-causes.py"
TRACE = HERE / "test-sycl-gap-causes.trace.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FAIL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Read the cause schema and the active coverage policy from the parser itself
# rather than restating them, so the two cannot drift.
CAUSES_MODULE = load_module(PARSER, "parse_sycl_gap_causes")
TIMELINE = CAUSES_MODULE.load_timeline_parser()
COVERAGE = TIMELINE.HOST_OVERLAP_COVERAGE
GAP_CAUSES = CAUSES_MODULE.gap_cause_names(COVERAGE)

TRULY_IDLE = CAUSES_MODULE.CAUSE_TRULY_IDLE

# The cause under test.  Resolved from the parser when it exists so the two
# cannot drift, with the literal as a fallback purely so a parser that does not
# implement it yet fails on the *classification* assertions below rather than on
# an AttributeError at import time.
#
# That fallback looks like dead code once the parser defines the constant, and
# has already been flagged as such once.  It is not: it is what lets this file
# run against a parser PREDATING the cause it tests.  That is how the RED step
# was demonstrated, how the spec review independently reproduced it, and what
# keeps a `git bisect` across the commit that added the cause reporting a
# readable classification failure instead of dying at import.  Do not "simplify"
# it to a plain attribute read.
IMPLICIT_CAUSE = getattr(CAUSES_MODULE, "CAUSE_IMPLICIT_QUEUE_SERIALIZATION", "implicit_queue_serialization")

# The queue the checked-in fixture builds its gaps on.  Keep in sync with the
# fixture's device/queue_kind args.
FIXTURE_QUEUE = "device0.compute"

# The fixture's k2->k3 gap is spanned by three 400 us nodes: sum/union 1200 us,
# max 400 us, against a 1000 us bar.  So it classifies as host_overlap under
# "union" and as runtime_idle/sum_covers_max_does_not under "max" -- which makes
# the fixture a *policy gate*, not just a conservation check.  Flipping
# HOST_OVERLAP_COVERAGE without updating this table fails the test loudly
# instead of silently restating a published number.
EXPECTED_BY_COVERAGE = {
    "union": {"host_overlap_count": 1, "runtime_idle_count": 3},
    "max": {"host_overlap_count": 0, "runtime_idle_count": 4},
}

# The checked-in fixture's k1->k2 gap is its only `truly_idle` one, 2,000,000 ns
# wide, under either coverage policy.  Pinned so a new cause that reaches gaps
# it should not have -- the one way to silently restate the published
# `truly_idle` budget -- fails here.
CHECKED_IN_TRULY_IDLE = {"total_ms_x1000": 2000, "count": 1}


# Synthetic trace for the implicit in-order serialization cause.  Device queues
# in ggml-sycl are all in-order, so they serialize without declaring a
# `depends_on` edge; the signal is the successor's *device* submit timestamp
# (`device_submit_ns`, same clock as `device_start_ns`/`device_end_ns`) sitting
# before the predecessor's completion.
#
# Every gap here is 2,000,000 ns on the compute queue, and every host submit
# window between consecutive events is 900 us wide against the 1000 us that
# HOST_OVERLAP_MIN_FRACTION demands of a 2 ms gap -- so no gap can reach
# host_overlap, and there are no ggml.op nodes at all.  Each therefore reaches
# the runtime_idle cause split, with submits present and ordered, and differs
# from its neighbours only in `device_submit_ns`:
#
#   s1->s2  implicit   s2 submitted at 1,500,000, before s1 ended at 2,000,000.
#   s2->s3  truly_idle s3 submitted at 6,000,000, after s2 ended at 5,000,000 --
#                      the device really was waiting for the host.
#   s3->s4  truly_idle s4 submitted at exactly s3's end (8,000,000).  Equality
#                      is not proof the queue held it, so the test is strict.
#   s4->s5  truly_idle s5 carries device_submit_ns=0, the profiler's
#                      "unavailable" sentinel (sycl-kernel-profiler.cpp emits 0
#                      when the runtime supplies no submit timestamp).  A parser
#                      that took 0 at face value would call every such gap
#                      implicit, since 0 precedes every predecessor's end.
#
# The device0/copy queue is the different-queue control: c2's submit
# (4,000,000) is *after* its own predecessor c1's end (3,000,000) so it must not
# be implicit, but *before* the compute queue's s2 end (5,000,000), so a parser
# that paired events across queues would misfile it.
IMPLICIT_SERIALIZATION_TRACE = {
    "traceEvents": [
        {
            "ph": "X", "cat": "sycl.event", "name": "s1", "ts": 50, "dur": 50,
            "args": {
                "device": 0, "queue_kind": "compute", "event_id": 11,
                "device_submit_ns": 900000, "device_start_ns": 1000000, "device_end_ns": 2000000,
                "host_submit_begin_us": 50, "host_submit_end_us": 100,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "s2", "ts": 1000, "dur": 100,
            "args": {
                "device": 0, "queue_kind": "compute", "event_id": 12,
                "device_submit_ns": 1500000, "device_start_ns": 4000000, "device_end_ns": 5000000,
                "host_submit_begin_us": 1000, "host_submit_end_us": 1100,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "s3", "ts": 2000, "dur": 100,
            "args": {
                "device": 0, "queue_kind": "compute", "event_id": 13,
                "device_submit_ns": 6000000, "device_start_ns": 7000000, "device_end_ns": 8000000,
                "host_submit_begin_us": 2000, "host_submit_end_us": 2100,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "s4", "ts": 3000, "dur": 100,
            "args": {
                "device": 0, "queue_kind": "compute", "event_id": 14,
                "device_submit_ns": 8000000, "device_start_ns": 10000000, "device_end_ns": 11000000,
                "host_submit_begin_us": 3000, "host_submit_end_us": 3100,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "s5", "ts": 4000, "dur": 100,
            "args": {
                "device": 0, "queue_kind": "compute", "event_id": 15,
                "device_submit_ns": 0, "device_start_ns": 13000000, "device_end_ns": 14000000,
                "host_submit_begin_us": 4000, "host_submit_end_us": 4100,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "c1", "ts": 200, "dur": 50,
            "args": {
                "device": 0, "queue_kind": "copy", "event_id": 21,
                "device_submit_ns": 1400000, "device_start_ns": 1500000, "device_end_ns": 3000000,
                "host_submit_begin_us": 200, "host_submit_end_us": 250,
            },
        },
        {
            "ph": "X", "cat": "sycl.event", "name": "c2", "ts": 1200, "dur": 50,
            "args": {
                "device": 0, "queue_kind": "copy", "event_id": 22,
                "device_submit_ns": 4000000, "device_start_ns": 6000000, "device_end_ns": 6500000,
                "host_submit_begin_us": 1200, "host_submit_end_us": 1250,
            },
        },
    ]
}

# Expected cause counts and times for the trace above, per queue.  A cause not
# listed for a queue must be absent from it entirely.
IMPLICIT_FIXTURE_EXPECTED = {
    "device0.compute": {
        "runtime_idle_count": 4,
        "causes": {IMPLICIT_CAUSE: (1, 2000), TRULY_IDLE: (3, 6000)},
    },
    "device0.copy": {
        "runtime_idle_count": 1,
        "causes": {TRULY_IDLE: (1, 3000)},
    },
}


def class_prefix(queue: str) -> str:
    return f"gap_cause.{queue}.class.runtime_idle"


def cause_prefix(queue: str) -> str:
    return f"gap_cause.{queue}.runtime_idle"


CLASS_PREFIX = class_prefix(FIXTURE_QUEUE)
CAUSE_PREFIX = cause_prefix(FIXTURE_QUEUE)


def parse_metrics(trace_path: Path, queue: str = "compute") -> dict[str, int]:
    result = subprocess.run(
        [sys.executable, str(PARSER), str(trace_path), "--queue", queue, "--top-transitions", "0"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"FAIL: parser exited {result.returncode} on {trace_path}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    metrics: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            metrics[parts[0]] = int(parts[1])
        except ValueError:
            continue
    if not metrics:
        raise SystemExit(f"FAIL: parser emitted no metrics for {trace_path}\nstdout:\n{result.stdout}")
    return metrics


def cause_partition_failures(
    metrics: dict[str, int],
    queue: str = FIXTURE_QUEUE,
    required_causes: tuple[str, ...] = GAP_CAUSES,
) -> list[str]:
    """Conservation and non-vacuity checks; empty list means the split is sound.

    Conservation is checked over the whole cause schema -- a gap filed under a
    cause this fixture does not intend still has to be accounted for.
    `required_causes` narrows only the non-vacuity check, so one fixture can own
    the causes it was built to reach without asserting the others are reachable
    from it.
    """
    failures: list[str] = []
    class_prefix_for_queue = class_prefix(queue)
    cause_prefix_for_queue = cause_prefix(queue)

    for suffix in ("total_ms_x1000", "count"):
        class_key = f"{class_prefix_for_queue}.{suffix}"
        if class_key not in metrics:
            failures.append(f"missing metric {class_key}")
            continue
        cause_keys = [f"{cause_prefix_for_queue}.{cause}.{suffix}" for cause in GAP_CAUSES]
        missing = [key for key in cause_keys if key not in metrics]
        if missing:
            failures.append(f"missing metrics {', '.join(missing)}")
            continue
        cause_sum = sum(metrics[key] for key in cause_keys)
        if cause_sum != metrics[class_key]:
            failures.append(
                f"{queue} {suffix}: causes sum to {cause_sum} but runtime_idle class is {metrics[class_key]}"
            )

    for cause in required_causes:
        count_key = f"{cause_prefix_for_queue}.{cause}.count"
        total_key = f"{cause_prefix_for_queue}.{cause}.total_ms_x1000"
        if metrics.get(count_key, 0) <= 0:
            failures.append(f"cause {cause} never observed ({count_key} is {metrics.get(count_key)})")
        if metrics.get(total_key, 0) <= 0:
            failures.append(f"cause {cause} contributed no time ({total_key} is {metrics.get(total_key)})")

    return failures


def observed_causes(metrics: dict[str, int], queues: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    for queue in queues:
        for cause in GAP_CAUSES:
            if metrics.get(f"{cause_prefix(queue)}.{cause}.count", 0) > 0:
                seen.add(cause)
    return seen


def schema_failures() -> list[str]:
    """The new cause has to be part of the reported schema, not an ad-hoc extra.

    A cause the parser classifies but never prints would be invisible to the
    conservation check, which reads the schema from `gap_cause_names()`.
    """
    if IMPLICIT_CAUSE in GAP_CAUSES:
        return []
    return [
        f"cause {IMPLICIT_CAUSE!r} is absent from gap_cause_names({COVERAGE!r}) = {GAP_CAUSES}; "
        "the implicit in-order serialization cause is not reported, so it cannot be told apart "
        "from truly_idle"
    ]


def checked_in_fixture_failures(metrics: dict[str, int]) -> list[str]:
    """The checked-in fixture's split, including the pinned `truly_idle` budget."""
    failures: list[str] = []

    # Pin the coverage policy.  The fixture is built so the two policies give
    # different class counts, so this catches a flip of HOST_OVERLAP_COVERAGE
    # that was not accompanied by a deliberate re-statement of the published
    # split in the findings doc.
    expected = EXPECTED_BY_COVERAGE.get(COVERAGE)
    if expected is None:
        return [f"unknown HOST_OVERLAP_COVERAGE {COVERAGE!r}; expected one of {sorted(EXPECTED_BY_COVERAGE)}"]
    observed_overlap = metrics.get(f"gap_cause.{FIXTURE_QUEUE}.class.host_overlap.count")
    observed_idle = metrics.get(f"{CLASS_PREFIX}.count")
    if observed_overlap != expected["host_overlap_count"] or observed_idle != expected["runtime_idle_count"]:
        failures.append(
            f"coverage policy {COVERAGE!r} gave host_overlap={observed_overlap} "
            f"runtime_idle={observed_idle}, expected {expected['host_overlap_count']} "
            f"and {expected['runtime_idle_count']}"
        )

    # This fixture declares no `device_submit_ns` anywhere, so nothing in it can
    # be proven to be implicit serialization and `truly_idle` must be exactly
    # what it was before the cause existed.
    implicit_count = metrics.get(f"{CAUSE_PREFIX}.{IMPLICIT_CAUSE}.count", 0)
    implicit_total = metrics.get(f"{CAUSE_PREFIX}.{IMPLICIT_CAUSE}.total_ms_x1000", 0)
    if implicit_count != 0 or implicit_total != 0:
        failures.append(
            f"fixture carries no device_submit_ns field, yet {IMPLICIT_CAUSE} claimed "
            f"count={implicit_count} total={implicit_total}"
        )
    for suffix, expected_value in CHECKED_IN_TRULY_IDLE.items():
        key = f"{CAUSE_PREFIX}.{TRULY_IDLE}.{suffix}"
        if metrics.get(key) != expected_value:
            failures.append(
                f"{key} is {metrics.get(key)}, expected {expected_value}; a new cause moved the "
                "published truly_idle budget on an input that contains no implicit serialization"
            )
    return failures


def implicit_fixture_failures(metrics: dict[str, int]) -> list[str]:
    """Positive case and its three same-queue plus one cross-queue controls."""
    failures: list[str] = []
    for queue, expected in sorted(IMPLICIT_FIXTURE_EXPECTED.items()):
        idle_count_key = f"{class_prefix(queue)}.count"
        if metrics.get(idle_count_key) != expected["runtime_idle_count"]:
            failures.append(
                f"{idle_count_key} is {metrics.get(idle_count_key)}, expected "
                f"{expected['runtime_idle_count']}; the fixture no longer reaches the cause split"
            )
        for cause in GAP_CAUSES:
            count, total = expected["causes"].get(cause, (0, 0))
            count_key = f"{cause_prefix(queue)}.{cause}.count"
            total_key = f"{cause_prefix(queue)}.{cause}.total_ms_x1000"
            if metrics.get(count_key) != count or metrics.get(total_key) != total:
                failures.append(
                    f"{queue} cause {cause}: count={metrics.get(count_key)} "
                    f"total={metrics.get(total_key)}, expected count={count} total={total}"
                )
    return failures


def negative_control_failures(metrics: dict[str, int]) -> list[str]:
    """cause_partition_failures() must reject metrics that violate the split."""
    failures: list[str] = []

    # A checker that has stopped checking would accept anything.  Move one
    # cause's time without moving the class total and require a complaint, so
    # the assertions are proven live rather than merely satisfied.
    corrupted = dict(metrics)
    corrupted[f"{CAUSE_PREFIX}.{TRULY_IDLE}.total_ms_x1000"] += 1
    if not cause_partition_failures(corrupted):
        failures.append("cause_partition_failures accepted metrics whose causes overshoot the class total")

    # Same for non-vacuity: zeroing a cause must be rejected even though the
    # sums then still balance if the class total is lowered to match.
    zeroed = dict(metrics)
    cause_total = zeroed[f"{CAUSE_PREFIX}.{TRULY_IDLE}.total_ms_x1000"]
    cause_count = zeroed[f"{CAUSE_PREFIX}.{TRULY_IDLE}.count"]
    zeroed[f"{CAUSE_PREFIX}.{TRULY_IDLE}.total_ms_x1000"] = 0
    zeroed[f"{CAUSE_PREFIX}.{TRULY_IDLE}.count"] = 0
    zeroed[f"{CLASS_PREFIX}.total_ms_x1000"] -= cause_total
    zeroed[f"{CLASS_PREFIX}.count"] -= cause_count
    if not cause_partition_failures(zeroed):
        failures.append("cause_partition_failures accepted a cause that was never observed")

    return failures


def main() -> int:
    if not PARSER.is_file():
        print(f"FAIL: parser not found at {PARSER}")
        return 1
    if not TRACE.is_file():
        print(f"FAIL: fixture trace not found at {TRACE}")
        return 1

    metrics = parse_metrics(TRACE)
    checked_in_required = tuple(cause for cause in GAP_CAUSES if cause != IMPLICIT_CAUSE)

    failures = schema_failures()
    failures += cause_partition_failures(metrics, FIXTURE_QUEUE, checked_in_required)
    failures += checked_in_fixture_failures(metrics)

    with tempfile.TemporaryDirectory() as tmp:
        implicit_trace = Path(tmp) / "implicit-serialization.trace.json"
        implicit_trace.write_text(json.dumps(IMPLICIT_SERIALIZATION_TRACE), encoding="utf-8")
        implicit_metrics = parse_metrics(implicit_trace, queue="all")

    implicit_queues = tuple(sorted(IMPLICIT_FIXTURE_EXPECTED))
    for queue in implicit_queues:
        required = (IMPLICIT_CAUSE, TRULY_IDLE) if queue == "device0.compute" else (TRULY_IDLE,)
        failures += cause_partition_failures(implicit_metrics, queue, required)
    failures += implicit_fixture_failures(implicit_metrics)

    # Non-vacuity over the whole schema: no cause may be unreachable from every
    # fixture, or a real instrument defect could report as a hard zero.
    seen = observed_causes(metrics, (FIXTURE_QUEUE,)) | observed_causes(implicit_metrics, implicit_queues)
    unreachable = [cause for cause in GAP_CAUSES if cause not in seen]
    if unreachable:
        failures.append(f"causes never observed by any fixture: {', '.join(unreachable)}")

    if not failures:
        failures += negative_control_failures(metrics)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    observed = ", ".join(f"{cause}={metrics[f'{CAUSE_PREFIX}.{cause}.count']}" for cause in GAP_CAUSES)
    implicit_prefix = cause_prefix("device0.compute")
    implicit_observed = ", ".join(
        f"{cause}={implicit_metrics[f'{implicit_prefix}.{cause}.count']}" for cause in GAP_CAUSES
    )
    print(
        f"PASS: coverage={COVERAGE}; runtime_idle cause split conserves and is non-vacuous "
        f"(checked-in fixture: {observed}; implicit-serialization fixture compute queue: {implicit_observed}); "
        f"truly_idle pinned at {CHECKED_IN_TRULY_IDLE['total_ms_x1000']} on an input with no implicit serialization"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
