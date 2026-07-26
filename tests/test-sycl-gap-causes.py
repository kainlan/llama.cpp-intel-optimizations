#!/usr/bin/env python3
"""The runtime_idle cause split must exhaustively partition the runtime_idle class.

parse-sycl-timeline.py reports runtime_idle as a single number, and Task 9 of
docs/plans/2026-07-25-decode-host-overhead-findings.md reads a dominant
runtime_idle as evidence about the L0/UR submit path.  That reading is only
valid for the part of the class that is genuinely unexplained.  runtime_idle is
the classifier's `else` branch, so it also absorbs gaps that merely *failed* the
host_overlap test for instrument reasons -- a missing submit span, a submit the
host batched ahead, or host work that no single ggml node covers even though the
spans together do.

parse-sycl-gap-causes.py separates those.  This test gates two properties of
that split:

  * **Conservation.**  The four causes must sum to the runtime_idle class total,
    in both time and count.  Unlike parse-sycl-timeline.py's class split, which
    is force-balanced by a rounding correction, nothing makes this sum come out
    right -- a gap that reaches the `else` branch and matches no cause would be
    silently dropped, and the class would read as more explained than it is.

  * **Non-vacuity.**  Every cause must be observed at least once.  A cause that
    can never fire is indistinguishable from a cause that is simply absent from
    the capture, and would let a real instrument defect report as zero.

  * **The coverage policy.**  `HOST_OVERLAP_COVERAGE` decides whether
    `host_overlap` needs one contiguous host op to cover the gap ("max") or
    merged coverage ("union").  The fixture is built so the two answers differ,
    so flipping the policy fails here rather than silently restating a published
    split.

Driven by a checked-in synthetic trace (never live hardware) built so each
reachable cause occurs exactly once; see that file's `_comment` for how each gap
is constructed.  cause_partition_failures() is additionally run against a
deliberately corrupted copy of the metrics and must reject it, so a test that
stopped checking anything cannot pass.

Plain script, not pytest: run with `python3 tests/<this file>`.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
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

# The queue the fixture builds its gaps on.  Keep in sync with the fixture's
# device/queue_kind args.
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

CLASS_PREFIX = f"gap_cause.{FIXTURE_QUEUE}.class.runtime_idle"
CAUSE_PREFIX = f"gap_cause.{FIXTURE_QUEUE}.runtime_idle"


def parse_metrics(trace_path: Path) -> dict[str, int]:
    result = subprocess.run(
        [sys.executable, str(PARSER), str(trace_path), "--queue", "compute", "--top-transitions", "0"],
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


def cause_partition_failures(metrics: dict[str, int]) -> list[str]:
    """Conservation and non-vacuity checks; empty list means the split is sound."""
    failures: list[str] = []

    for suffix in ("total_ms_x1000", "count"):
        class_key = f"{CLASS_PREFIX}.{suffix}"
        if class_key not in metrics:
            failures.append(f"missing metric {class_key}")
            continue
        cause_keys = [f"{CAUSE_PREFIX}.{cause}.{suffix}" for cause in GAP_CAUSES]
        missing = [key for key in cause_keys if key not in metrics]
        if missing:
            failures.append(f"missing metrics {', '.join(missing)}")
            continue
        cause_sum = sum(metrics[key] for key in cause_keys)
        if cause_sum != metrics[class_key]:
            failures.append(
                f"{suffix}: causes sum to {cause_sum} but runtime_idle class is {metrics[class_key]}"
            )

    for cause in GAP_CAUSES:
        count_key = f"{CAUSE_PREFIX}.{cause}.count"
        total_key = f"{CAUSE_PREFIX}.{cause}.total_ms_x1000"
        if metrics.get(count_key, 0) <= 0:
            failures.append(f"cause {cause} never observed ({count_key} is {metrics.get(count_key)})")
        if metrics.get(total_key, 0) <= 0:
            failures.append(f"cause {cause} contributed no time ({total_key} is {metrics.get(total_key)})")

    return failures


def main() -> int:
    if not PARSER.is_file():
        print(f"FAIL: parser not found at {PARSER}")
        return 1
    if not TRACE.is_file():
        print(f"FAIL: fixture trace not found at {TRACE}")
        return 1

    metrics = parse_metrics(TRACE)

    failures = cause_partition_failures(metrics)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    # Pin the coverage policy.  The fixture is built so the two policies give
    # different class counts, so this catches a flip of HOST_OVERLAP_COVERAGE
    # that was not accompanied by a deliberate re-statement of the published
    # split in the findings doc.
    expected = EXPECTED_BY_COVERAGE.get(COVERAGE)
    if expected is None:
        print(f"FAIL: unknown HOST_OVERLAP_COVERAGE {COVERAGE!r}; expected one of {sorted(EXPECTED_BY_COVERAGE)}")
        return 1
    observed_overlap = metrics.get(f"gap_cause.{FIXTURE_QUEUE}.class.host_overlap.count")
    observed_idle = metrics.get(f"{CLASS_PREFIX}.count")
    if observed_overlap != expected["host_overlap_count"] or observed_idle != expected["runtime_idle_count"]:
        print(
            f"FAIL: coverage policy {COVERAGE!r} gave host_overlap={observed_overlap} "
            f"runtime_idle={observed_idle}, expected {expected['host_overlap_count']} "
            f"and {expected['runtime_idle_count']}"
        )
        return 1

    # A checker that has stopped checking would pass the block above on any
    # input.  Move one cause's time without moving the class total and require a
    # complaint, so the assertions are proven live rather than merely satisfied.
    corrupted = dict(metrics)
    corrupted[f"{CAUSE_PREFIX}.truly_idle.total_ms_x1000"] += 1
    if not cause_partition_failures(corrupted):
        print("FAIL: cause_partition_failures accepted metrics whose causes overshoot the class total")
        return 1

    # Same for non-vacuity: zeroing a cause must be rejected even though the
    # sums then still balance if the class total is lowered to match.
    zeroed = dict(metrics)
    cause_total = zeroed[f"{CAUSE_PREFIX}.truly_idle.total_ms_x1000"]
    cause_count = zeroed[f"{CAUSE_PREFIX}.truly_idle.count"]
    zeroed[f"{CAUSE_PREFIX}.truly_idle.total_ms_x1000"] = 0
    zeroed[f"{CAUSE_PREFIX}.truly_idle.count"] = 0
    zeroed[f"{CLASS_PREFIX}.total_ms_x1000"] -= cause_total
    zeroed[f"{CLASS_PREFIX}.count"] -= cause_count
    if not cause_partition_failures(zeroed):
        print("FAIL: cause_partition_failures accepted a cause that was never observed")
        return 1

    observed = ", ".join(f"{cause}={metrics[f'{CAUSE_PREFIX}.{cause}.count']}" for cause in GAP_CAUSES)
    print(f"PASS: coverage={COVERAGE}; runtime_idle cause split conserves and is non-vacuous ({observed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
