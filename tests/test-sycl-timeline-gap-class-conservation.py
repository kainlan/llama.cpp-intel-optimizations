#!/usr/bin/env python3
"""Gap classes must exhaustively partition each queue's total gap time.

Task 6 of docs/plans/2026-07-25-sycl-decode-host-overhead-attribution.md reads
runtime_idle as evidence about the L0 submit path.  That is only valid if the
three classes sum to the queue total, and if the rounding correction that
parse-sycl-timeline.py applies toward runtime_idle is small enough to ignore.
The parser *forces* the sum to balance, so the sum alone proves nothing about
how much unexplained time was folded in -- that is what the
gap_class.*.rounding_delta_ms_x1000 metric is for, and this test requires it.

Driven by a checked-in synthetic trace (never live hardware).  The fixture is
built so the test cannot pass vacuously:

  * device0/compute exercises all three classification branches, non-zero;
  * its per-gap rounding disagrees with the queue-total rounding by 2, so the
    rounding delta under test is provably non-zero rather than a constant 0;
  * conservation_failures() is additionally run against a deliberately
    corrupted copy of the metrics and must reject it.

Plain script, not pytest: run with `python3 tests/<this file>`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARSER = ROOT / "scripts" / "parse-sycl-timeline.py"
TRACE = HERE / "test-sycl-timeline-gap-class-conservation.trace.json"

GAP_CLASSES = ("host_overlap", "queue_serialization", "runtime_idle")

GAP_PREFIX = "gap."
GAP_SUFFIX = ".total_ms_x1000"

# The queue the fixture builds all three classes and a non-zero rounding delta
# on.  Keep in sync with the fixture's device/queue_kind args.
FIXTURE_QUEUE = "device0.compute"
FIXTURE_ROUNDING_DELTA = 2


def parse_metrics(trace_path: Path) -> dict[str, int]:
    result = subprocess.run(
        [sys.executable, str(PARSER), str(trace_path)],
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
        if len(parts) == 2:
            try:
                metrics[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return metrics


def gap_totals_by_queue(metrics: dict[str, int]) -> dict[str, int]:
    return {
        key[len(GAP_PREFIX):-len(GAP_SUFFIX)]: value
        for key, value in metrics.items()
        if key.startswith(GAP_PREFIX) and key.endswith(GAP_SUFFIX)
    }


def conservation_failures(metrics: dict[str, int]) -> list[str]:
    """Every queue's three gap classes must sum to its gap total.

    Also requires the rounding-delta metric, so the amount of unexplained time
    the parser folds into a class is observable rather than assumed harmless.
    """
    failures: list[str] = []
    for queue, gap_total in sorted(gap_totals_by_queue(metrics).items()):
        class_sum = 0
        missing = False
        for gap_class in GAP_CLASSES:
            class_key = f"gap_class.{queue}.{gap_class}{GAP_SUFFIX}"
            if class_key not in metrics:
                failures.append(f"missing {class_key}")
                missing = True
                continue
            class_sum += metrics[class_key]
        if not missing and class_sum != gap_total:
            failures.append(f"{queue} classes sum to {class_sum}, gap total is {gap_total}")

        delta_key = f"gap_class.{queue}.rounding_delta_ms_x1000"
        if delta_key not in metrics:
            failures.append(f"missing {delta_key} -- rounding correction is not observable")
    return failures


def fixture_drives_the_parser(metrics: dict[str, int]) -> list[str]:
    """Guard against a green run that measured nothing.

    A conservation check over an empty or all-zero metric set passes for any
    state of the parser.  These assertions fail instead.
    """
    failures: list[str] = []

    gap_totals = gap_totals_by_queue(metrics)
    if not gap_totals:
        failures.append("parser emitted no gap totals for the synthetic trace")
        return failures
    for queue, gap_total in sorted(gap_totals.items()):
        if gap_total <= 0:
            failures.append(f"gap total for {queue} is {gap_total}; fixture measured no gap time")

    if FIXTURE_QUEUE not in gap_totals:
        failures.append(f"fixture queue {FIXTURE_QUEUE} absent; the trace no longer reaches the parser")
        return failures

    for gap_class in GAP_CLASSES:
        class_key = f"gap_class.{FIXTURE_QUEUE}.{gap_class}{GAP_SUFFIX}"
        if metrics.get(class_key, 0) <= 0:
            failures.append(f"{class_key} is {metrics.get(class_key)}; fixture no longer exercises {gap_class}")

    delta_key = f"gap_class.{FIXTURE_QUEUE}.rounding_delta_ms_x1000"
    if delta_key in metrics and metrics[delta_key] != FIXTURE_ROUNDING_DELTA:
        failures.append(
            f"{delta_key} is {metrics[delta_key]}, expected {FIXTURE_ROUNDING_DELTA}; "
            "the fixture no longer exercises the rounding correction, so a present-but-always-zero "
            "delta metric would go undetected"
        )
    return failures


def negative_control_failures(metrics: dict[str, int]) -> list[str]:
    """conservation_failures() must reject metrics that violate the invariant."""
    failures: list[str] = []

    unbalanced = dict(metrics)
    unbalanced[f"gap_class.{FIXTURE_QUEUE}.runtime_idle{GAP_SUFFIX}"] += 1
    if not conservation_failures(unbalanced):
        failures.append("conservation_failures() accepted a class sum that does not match the gap total")

    without_delta = dict(metrics)
    del without_delta[f"gap_class.{FIXTURE_QUEUE}.rounding_delta_ms_x1000"]
    if not conservation_failures(without_delta):
        failures.append("conservation_failures() accepted metrics with no rounding_delta metric")

    return failures


def main() -> int:
    if not TRACE.is_file():
        print(f"FAIL: synthetic trace fixture missing: {TRACE}")
        return 1

    metrics = parse_metrics(TRACE)

    failures = fixture_drives_the_parser(metrics)
    failures += conservation_failures(metrics)
    if not failures:
        failures += negative_control_failures(metrics)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("PASS: gap classes partition every queue total, and the rounding delta is observable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
