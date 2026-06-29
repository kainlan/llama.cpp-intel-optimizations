#!/usr/bin/env python3
"""Offline quality/speed ceiling gate for MXFP4 algorithmic route captures."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REQUIRED_KEYS = (
    "baseline_output",
    "candidate_output",
    "baseline_logits_top10",
    "candidate_logits_top10",
    "baseline_ms_per_token",
    "candidate_ms_per_token",
)

RELATIVE_L2_LIMIT = 1e-3
TOP10_LOGIT_MAE_LIMIT = 1e-2
MAX_SAFE_ABS = math.sqrt(sys.float_info.max)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def load_capture(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("empty algorithmic capture")

    try:
        parsed = json.loads(text, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("algorithmic capture is not a JSON object")
    return parsed


def require_keys(capture: dict[str, Any]) -> None:
    for key in REQUIRED_KEYS:
        if key not in capture:
            raise ValueError(f"missing key: {key}")


def require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"invalid {name}: numeric value is too large") from exc
    if not math.isfinite(number):
        raise ValueError(f"invalid {name}")
    if abs(number) > MAX_SAFE_ABS:
        raise ValueError(f"invalid {name}: numeric value is too large")
    return number


def require_number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} is not a list")
    numbers = [require_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if not numbers:
        raise ValueError(f"{name} is empty")
    return numbers


def relative_l2(baseline: list[float], candidate: list[float]) -> float:
    numerator = sum((a - b) * (a - b) for a, b in zip(baseline, candidate))
    denominator = max(sum(a * a for a in baseline), 1e-30)
    return math.sqrt(numerator / denominator)


def mean_abs_error(baseline: list[float], candidate: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(baseline, candidate)) / len(baseline)


def analyze(capture: dict[str, Any]) -> dict[str, float | str]:
    require_keys(capture)

    baseline_output = require_number_list(capture["baseline_output"], "baseline_output")
    candidate_output = require_number_list(capture["candidate_output"], "candidate_output")
    if len(baseline_output) != len(candidate_output):
        raise ValueError("baseline_output and candidate_output length mismatch")

    baseline_logits = require_number_list(capture["baseline_logits_top10"], "baseline_logits_top10")
    candidate_logits = require_number_list(capture["candidate_logits_top10"], "candidate_logits_top10")
    if len(baseline_logits) != 10:
        raise ValueError("baseline_logits_top10 must contain exactly 10 values")
    if len(candidate_logits) != 10:
        raise ValueError("candidate_logits_top10 must contain exactly 10 values")

    baseline_ms = require_finite_number(capture["baseline_ms_per_token"], "baseline_ms_per_token")
    candidate_ms = require_finite_number(capture["candidate_ms_per_token"], "candidate_ms_per_token")
    if baseline_ms <= 0.0:
        raise ValueError("baseline_ms_per_token must be positive")
    if candidate_ms <= 0.0:
        raise ValueError("candidate_ms_per_token must be positive")

    rel_l2 = relative_l2(baseline_output, candidate_output)
    logit_mae = mean_abs_error(baseline_logits, candidate_logits)
    speed_ceiling = 1000.0 / candidate_ms
    speedup = baseline_ms / candidate_ms
    recommendation = "pass"
    if rel_l2 > RELATIVE_L2_LIMIT or logit_mae > TOP10_LOGIT_MAE_LIMIT:
        recommendation = "kill"

    return {
        "relative_l2": rel_l2,
        "top10_logit_mae": logit_mae,
        "speed_ceiling_tok_s": speed_ceiling,
        "baseline_ms_per_token": baseline_ms,
        "candidate_ms_per_token": candidate_ms,
        "speedup_vs_baseline": speedup,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate offline MXFP4 algorithmic-route quality and speed ceiling captures"
    )
    parser.add_argument("capture", type=Path, help="JSON file containing exactly one algorithmic capture object")
    args = parser.parse_args()

    if not args.capture.exists():
        return fail(f"missing algorithmic capture: {args.capture}")
    if not args.capture.is_file():
        return fail(f"algorithmic capture is not a file: {args.capture}")

    try:
        results = analyze(load_capture(args.capture))
    except ValueError as exc:
        return fail(str(exc))

    print(f"relative_l2 {results['relative_l2']:.12g}")
    print(f"top10_logit_mae {results['top10_logit_mae']:.12g}")
    print(f"speed_ceiling_tok_s {results['speed_ceiling_tok_s']:.6f}")
    print(f"baseline_ms_per_token {results['baseline_ms_per_token']:.6f}")
    print(f"candidate_ms_per_token {results['candidate_ms_per_token']:.6f}")
    print(f"speedup_vs_baseline {results['speedup_vs_baseline']:.6f}")
    print(f"recommendation {results['recommendation']}")

    if results["recommendation"] != "pass":
        return fail(
            f"algorithmic route exceeds quality thresholds: relative_l2>{RELATIVE_L2_LIMIT:g} "
            f"or top10_logit_mae>{TOP10_LOGIT_MAE_LIMIT:g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
