#!/usr/bin/env python3
"""Emit a layered SYCL profiling ledger from parser summary artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
E2E_STAGE_RE = re.compile(r"\[SYCL-E2E-TG-STAGE\].*?\bhost=([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+ms\b")
INTEGER_RE = re.compile(r"[+-]?\d+")


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timeline_parser = load_module("parse_sycl_timeline", ROOT / "scripts" / "parse-sycl-timeline.py")
kernel_parser = load_module("parse_sycl_kernel_profile", ROOT / "scripts" / "parse-sycl-kernel-profile.py")


def read_parse_file(path: pathlib.Path) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected key value row")
        key, value = fields
        result[key] = int(value) if INTEGER_RE.fullmatch(value) else value
    return result


def parse_e2e_stderr(path: pathlib.Path) -> int:
    total = 0.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if "[SYCL-E2E-TG-STAGE]" not in line:
            continue
        match = E2E_STAGE_RE.search(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed SYCL-E2E-TG-STAGE row")
        total += float(match.group(1))
    return int(round(total))


def int_metric(summary: dict[str, int | str], key: str, path: pathlib.Path) -> int:
    value = summary.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"{path}: missing integer metric {key}")


def parse_optional_summary(path: pathlib.Path | None, key: str) -> tuple[int, bool]:
    if path is None:
        return 0, True
    summary = read_parse_file(path)
    return int_metric(summary, key, path), False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Emit a layered SYCL profiling ledger", allow_abbrev=False)
    parser.add_argument("--timeline", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-profile", type=pathlib.Path, required=True)
    parser.add_argument("--l0-summary", type=pathlib.Path)
    parser.add_argument("--ur-summary", type=pathlib.Path)
    parser.add_argument("--vtune-summary", type=pathlib.Path)
    parser.add_argument("--bench-stderr", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        events = timeline_parser.load_trace_events(args.timeline)
        category_totals, _, wall_us, _, _, _ = timeline_parser.summarize_events(events)
        rows = kernel_parser.load_rows(args.kernel_profile)
        kernel_totals, _ = kernel_parser.aggregate_rows(rows)

        l0_api, missing_l0 = parse_optional_summary(args.l0_summary, "l0.total_ms_x1000")
        ur_api, missing_ur = parse_optional_summary(args.ur_summary, "ur.total_ms_x1000")
        vtune_gpu, missing_vtune = parse_optional_summary(args.vtune_summary, "vtune.kernel_total_ms_x1000")

        e2e_host_total = parse_e2e_stderr(args.bench_stderr) if args.bench_stderr is not None else None
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError, AttributeError) as exc:
        print(f"failed to parse layer ledger: {exc}")
        return 2

    wall_total = timeline_parser.us_to_ms_x1000(wall_us)
    sycl_submit_host = int(category_totals["sycl.submit"])
    if e2e_host_total is None:
        app_host = sum(int(total) for category, total in category_totals.items() if category != "sycl.submit")
    else:
        app_host = min(wall_total, e2e_host_total)
    gpu_kernel = kernel_parser.ns_to_ms_x1000(sum(totals["total_ns"] for totals in kernel_totals.values()))
    unknown = max(0, wall_total - app_host - sycl_submit_host - ur_api - l0_api - gpu_kernel)

    missing_layers = []
    if missing_l0:
        missing_layers.append("l0")
    if missing_ur:
        missing_layers.append("ur")
    if missing_vtune:
        missing_layers.append("vtune")
    status = "missing_layers" if missing_layers else "ok"

    print(f"layer.wall_ms_x1000 {wall_total}")
    print(f"layer.app_host_ms_x1000 {app_host}")
    print(f"layer.sycl_submit_host_ms_x1000 {sycl_submit_host}")
    print(f"layer.ur_api_ms_x1000 {ur_api}")
    print(f"layer.level_zero_api_ms_x1000 {l0_api}")
    print(f"layer.gpu_kernel_ms_x1000 {gpu_kernel}")
    print(f"layer.vtune_gpu_ms_x1000 {vtune_gpu}")
    print(f"layer.unknown_wall_ms_x1000 {unknown}")
    for layer in missing_layers:
        print(f"coverage.missing_layer {layer}")
    print(f"coverage.layer_status {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
