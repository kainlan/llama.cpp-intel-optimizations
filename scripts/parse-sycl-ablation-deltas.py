#!/usr/bin/env python3
"""Convert SYCL MXFP4 TG microbench route evidence into ablation deltas."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, Callable


class AblationDeltaError(ValueError):
    """Expected parse failure that should be reported without a traceback."""


def load_mxfp4_parser() -> Any:
    path = pathlib.Path(__file__).resolve().with_name("parse-sycl-mxfp4-tg-bench.py")
    spec = importlib.util.spec_from_file_location("parse_sycl_mxfp4_tg_bench", path)
    if spec is None or spec.loader is None:
        raise AblationDeltaError(f"failed to load microbench parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def average_saving_ms(records: list[dict[str, Any]], route: str) -> float:
    savings: list[float] = []
    for record in records:
        if record.get("route") != route:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise AblationDeltaError(f"route record has invalid metrics: {route}")
        saving = metrics.get("saving_vs_baseline_ms")
        if not isinstance(saving, (int, float)) or isinstance(saving, bool):
            raise AblationDeltaError(f"route record has invalid metrics.saving_vs_baseline_ms: {route}")
        savings.append(float(saving))
    if not savings:
        raise AblationDeltaError(f"route not found: {route}")
    return sum(savings) / len(savings)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse SYCL microbench route evidence into ablation delta JSON")
    parser.add_argument("--microbench-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--route", required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if not args.kernel:
            raise AblationDeltaError("missing kernel")
        if not args.route:
            raise AblationDeltaError("missing route")

        module = load_mxfp4_parser()
        load_records: Callable[[pathlib.Path], list[dict[str, Any]]] = module.load_records
        records = load_records(args.microbench_jsonl)
        avg_saving_ms = average_saving_ms(records, args.route)
        delta_ms_x1000 = int(round(avg_saving_ms * 1000.0))
        output = {
            "deltas": [
                {
                    "kernel": args.kernel,
                    "route": args.route,
                    "delta_ms_x1000": delta_ms_x1000,
                }
            ]
        }
        print(json.dumps(output, separators=(",", ":")))
    except (OSError, ValueError, AblationDeltaError) as exc:
        print(f"failed to parse ablation deltas: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
