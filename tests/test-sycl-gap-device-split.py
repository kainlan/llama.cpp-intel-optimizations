#!/usr/bin/env python3
"""Regression gate for parse-sycl-gap-device-split.py.

Filed as llama.cpp-u9fr: the script crashed with a ZeroDivisionError at line
231 on ANY trace with zero device-event gaps -- no malformed data required, just
a queue with one event, or two events where the second is fully contained in
the first (`current[0] > previous[1]` never fires, so `rows` stays empty and
`gap_total` is 0.0). The existing "no device events at all" guard a few lines
above (:178-179) does not cover this: it only catches the case with FEWER than
one event, not the case with events but no *gap* between them.

This file checks four things:

  * The zero-gap crash is fixed: a single-event trace and a fully-contained-pair
    trace must both exit cleanly (return 2, matching the shape of the existing
    empty-input guard) instead of raising.
  * The normal A/B split is arithmetically correct on a small, hand-computed
    fixture (`test-sycl-gap-device-split.trace.json`, checked in beside this
    file) -- asserting the actual numbers, not just a zero exit code.
  * The dispatch-latency floor calibration case -- a transition whose two
    events share one `node_idx` -- reports `dnode_median == 0`, which is the
    precondition the findings doc's 0.35-0.72 us floor claim rests on.
  * A malformed event (missing `dur` on a `sycl.submit`, missing `ts` on a
    `ggml.op` node) is skipped rather than raising, exercising the
    `collect_device_events`-style field-presence guard added to
    `collect_submit_spans` / `collect_host_nodes`.

Plain script, not pytest, following the sibling gates' precedent (plain script
+ `llama_test_cmd`, not a pytest probe) -- run with `python3 tests/<this file>`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARSER = ROOT / "scripts" / "parse-sycl-gap-device-split.py"
TRACE = HERE / "test-sycl-gap-device-split.trace.json"


def metadata(**fields: object) -> str:
    return ";".join(f"{key}={value}" for key, value in fields.items())


def device_event(name: str, ts: float, dur: float, **fields: object) -> dict:
    return {
        "ph": "X",
        "cat": "sycl.event",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {"metadata": metadata(**fields)},
    }


# A single device event on the queue: no successor, so the main loop never
# runs and `rows` stays empty -- the first of the two well-formed inputs the
# lead reproduced the crash with.
SINGLE_EVENT_TRACE = {
    "traceEvents": [
        device_event(
            "k1", 50, 50,
            device_start_ns=1000000, device_end_ns=2000000, device_submit_ns=900000,
            queue_kind="compute", node_idx=1, submit_id="s1", node_tensor="t1",
        ),
    ]
}

# Two device events where the second is fully CONTAINED in the first
# (k2 ends before k1 does), so `current[0] > previous[1]` never fires and no
# gap is ever recorded -- the second well-formed reproduction from the task.
CONTAINED_EVENTS_TRACE = {
    "traceEvents": [
        device_event(
            "k1", 50, 50,
            device_start_ns=1000000, device_end_ns=5000000, device_submit_ns=900000,
            queue_kind="compute", node_idx=1, submit_id="s1", node_tensor="t1",
        ),
        device_event(
            "k2", 100, 50,
            device_start_ns=2000000, device_end_ns=3000000, device_submit_ns=1900000,
            queue_kind="compute", node_idx=2, submit_id="s2", node_tensor="t2",
        ),
    ]
}

# One well-formed gap (k1->k2, same shape as the checked-in fixture's k1->k2)
# plus a `sycl.submit` missing `dur` and a `ggml.op` node missing `ts`. Neither
# malformed helper field is on the critical path for the gap itself, so if
# collect_submit_spans / collect_host_nodes raise instead of skipping, the
# whole parse dies before printing anything.
MALFORMED_TRACE = {
    "traceEvents": [
        device_event(
            "k1", 50, 50,
            device_start_ns=1000000, device_end_ns=2000000, device_submit_ns=900000,
            queue_kind="compute", node_idx=1, submit_id="s1", node_tensor="t1",
        ),
        device_event(
            "k2", 1000, 100,
            device_start_ns=5000000, device_end_ns=6000000, device_submit_ns=4500000,
            queue_kind="compute", node_idx=1, submit_id="s2", node_tensor="t2",
        ),
        {"ph": "X", "cat": "sycl.submit", "name": "submit-bad", "ts": 10,
         "args": {"metadata": "event_id=bad"}},  # no "dur"
        {"ph": "X", "cat": "ggml.op", "name": "compute_forward_node", "dur": 10,
         "args": {"metadata": "op=MUL_MAT"}},  # no "ts"
    ]
}


def write_trace(directory: Path, name: str, document: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def run_parser(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PARSER), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def parse_metrics(trace_path: Path, queue: str = "compute") -> dict[str, int]:
    result = run_parser(str(trace_path), "--queue", queue, "--top-transitions", "10", "--format", "metrics")
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


EXPECTED = {
    "gap_device_split.compute.transition.k1--to--k2.n": 1,
    "gap_device_split.compute.transition.k1--to--k2.gap_ms_x1000": 3000,
    "gap_device_split.compute.transition.k1--to--k2.a_ms_x1000": 2500,
    "gap_device_split.compute.transition.k1--to--k2.b_ms_x1000": 500,
    "gap_device_split.compute.transition.k1--to--k2.b_pct_x10": 167,
    "gap_device_split.compute.transition.k1--to--k2.dnode_median": 0,
    "gap_device_split.compute.transition.k2--to--k3.n": 1,
    "gap_device_split.compute.transition.k2--to--k3.gap_ms_x1000": 3000,
    "gap_device_split.compute.transition.k2--to--k3.a_ms_x1000": 2700,
    "gap_device_split.compute.transition.k2--to--k3.b_ms_x1000": 300,
    "gap_device_split.compute.transition.k2--to--k3.b_pct_x10": 100,
    "gap_device_split.compute.transition.k2--to--k3.dnode_median": 4,
    "gap_device_split.compute.total.n": 2,
    "gap_device_split.compute.total.gap_ms_x1000": 6000,
    "gap_device_split.compute.total.a_ms_x1000": 5200,
    "gap_device_split.compute.total.b_ms_x1000": 800,
    "gap_device_split.compute.total.b_pct_x10": 133,
    "gap_device_split.compute.total.busy_ms_x1000": 3000,
    "gap_device_split.compute.total.events": 3,
}


def checked_in_fixture_failures() -> list[str]:
    """Normal A/B split, asserted numerically, plus the same-node_idx floor
    calibration case (dnode_median == 0 on k1->k2)."""
    metrics = parse_metrics(TRACE)
    failures = []
    for key, expected_value in EXPECTED.items():
        actual = metrics.get(key)
        if actual != expected_value:
            failures.append(f"{key} is {actual}, expected {expected_value}")
    return failures


def zero_gap_failures() -> list[str]:
    """The regression test for the crash: a trace with events but no gap
    between them must exit cleanly, not raise ZeroDivisionError.

    Deliberately invoked with the DEFAULT (table) output -- no `--format` --
    so this reproduces the crash exactly as filed, independent of whichever
    output-mode flag the fix happens to add alongside it."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cases = {
            "single event": write_trace(tmp_path, "single.json", SINGLE_EVENT_TRACE),
            "fully-contained pair": write_trace(tmp_path, "contained.json", CONTAINED_EVENTS_TRACE),
        }
        for label, trace_path in cases.items():
            result = run_parser(str(trace_path), "--queue", "compute")
            if "Traceback" in result.stderr or "ZeroDivisionError" in result.stderr:
                failures.append(f"{label}: parser raised instead of exiting cleanly:\n{result.stderr}")
                continue
            if result.returncode != 2:
                failures.append(
                    f"{label}: expected return code 2 (matching the empty-input guard), "
                    f"got {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
    return failures


def malformed_input_failures() -> list[str]:
    """A malformed submit/host-node event must be skipped, not crash the parse
    -- the well-formed k1->k2 gap elsewhere in the same trace must still be
    reported."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = write_trace(Path(tmp), "malformed.json", MALFORMED_TRACE)
        result = run_parser(str(trace_path), "--queue", "compute", "--format", "metrics", "--top-transitions", "10")
        if result.returncode != 0:
            failures.append(
                f"malformed trace: expected return code 0 (bad helper events skipped), "
                f"got {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            return failures
        metrics: dict[str, int] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = int(parts[1])
                except ValueError:
                    continue
        key = "gap_device_split.compute.transition.k1--to--k2.gap_ms_x1000"
        if metrics.get(key) != 3000:
            failures.append(f"malformed trace: {key} is {metrics.get(key)}, expected 3000")
    return failures


def main() -> int:
    if not PARSER.is_file():
        print(f"FAIL: parser not found at {PARSER}")
        return 1
    if not TRACE.is_file():
        print(f"FAIL: fixture trace not found at {TRACE}")
        return 1

    failures: list[str] = []
    failures += zero_gap_failures()
    failures += checked_in_fixture_failures()
    failures += malformed_input_failures()

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(
        "PASS: zero-gap traces (single event; fully-contained pair) exit cleanly; "
        "k1->k2/k2->k3 A/B split matches hand-computed totals; dnode_median floor "
        "calibration (k1->k2, equal node_idx) is 0; a malformed submit/host-node "
        "event is skipped rather than crashing the parse"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
