#!/usr/bin/env python3
"""Summarize SYCL named kernel profiler CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any


MXFP4_BLOCK_ELEMS = 32
MXFP4_BLOCK_BYTES = 17  # one shared E8M0 scale byte + 32 packed 4-bit elements

# Model geometry needed to turn a kernel's wall time into achieved bandwidth.
# Keys are the `--geometry` choices.
GEOMETRY_PRESETS: dict[str, dict[str, Any]] = {
    "gpt-oss-20b": {
        "layers": 24,
        "experts": 32,
        "experts_active": 4,  # top-4 routing
        "expert_ncols": 2880,
        "expert_nrows": 2880,
        # Expert matrices streamed per kernel call, by role.
        #
        # CRITICAL: gate and up are TWO separate 2880x2880 expert matrices, so
        # the fused gate/up kernel streams 2 x top_k = 8 matrices = 33.6 MiB
        # per call. The down projection is ONE matrix per expert, so it streams
        # top_k = 4 matrices = 16.8 MiB -- half the traffic. Swapping these two
        # inverts the roofline verdict (the gate/up kernel is the fast one).
        "role_matrices": {
            "gateup": 2,
            "down": 1,
        },
        # Kernel-name substring -> role, checked in order; first match wins.
        # In the GPT-OSS TG capture the fused gate/up takes the packed-q8 XMX
        # path, which leaves the SOA batched matvec (`role=matvec` in its
        # profiler metadata) carrying the down projection.
        "role_rules": [
            ("gateup", "gateup"),
            ("down", "down"),
            ("soa.batched", "down"),
        ],
    },
}


def metric_name(raw: str) -> str:
    return raw.replace(" ", "_")


def mxfp4_bytes(ncols: int, nrows: int) -> int:
    """Stored size of an `ncols` x `nrows` MXFP4 tensor.

    MXFP4 packs 32 elements plus one shared scale into 17 bytes. A trailing
    partial block still occupies a whole block.
    """

    elems = ncols * nrows
    blocks = (elems + MXFP4_BLOCK_ELEMS - 1) // MXFP4_BLOCK_ELEMS
    return blocks * MXFP4_BLOCK_BYTES


def achieved_gbs(bytes_moved: float, seconds: float) -> float:
    """Achieved bandwidth in GB/s (decimal 10^9, matching vendor peak figures)."""

    if seconds <= 0:
        return 0.0
    return bytes_moved / seconds / 1e9


def percent_of_peak(achieved: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return 100.0 * achieved / peak


def validate_geometry_preset(name: str, preset: dict[str, Any]) -> None:
    """Reject a preset whose `role_rules` name a role `role_matrices` lacks.

    Adding a preset is meant to be a pure data edit, so the one mistake such an
    edit makes silently -- a typo'd role key -- is raised as the same ValueError
    the rest of this script uses, instead of surfacing later as a bare KeyError
    traceback out of `geometry_kernel_bytes`.
    """

    missing = {role for _, role in preset["role_rules"]} - set(preset["role_matrices"])
    if missing:
        raise ValueError(f"geometry preset {name}: role_rules roles with no role_matrices entry: {sorted(missing)}")


def geometry_preset(name: str) -> dict[str, Any]:
    preset = GEOMETRY_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown geometry preset: {name}")
    validate_geometry_preset(name, preset)
    return preset


# A malformed preset is a data-edit mistake, so surface it at import time
# rather than only when a capture happens to name the affected kernel.
for _preset_name, _preset in GEOMETRY_PRESETS.items():
    validate_geometry_preset(_preset_name, _preset)


def geometry_expert_matrix_bytes(name: str) -> int:
    preset = geometry_preset(name)
    return mxfp4_bytes(preset["expert_ncols"], preset["expert_nrows"])


def geometry_role(name: str, kernel: str) -> str | None:
    preset = geometry_preset(name)
    for needle, role in preset["role_rules"]:
        if needle in kernel:
            return role
    return None


def geometry_kernel_bytes(name: str, kernel: str) -> int | None:
    """Weight bytes one call of `kernel` streams, or None if the preset has no rule."""

    role = geometry_role(name, kernel)
    if role is None:
        return None
    preset = geometry_preset(name)
    matrices = preset["role_matrices"][role]
    return matrices * preset["experts_active"] * geometry_expert_matrix_bytes(name)


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" or text.lstrip().startswith("{"):
        obj = json.loads(text)
        return [dict(row) for row in obj.get("kernels", [])]
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines()):
        rows.append(dict(row))
    return rows


def int_field(row: dict[str, Any], name: str) -> int:
    raw = row.get(name, 0)
    if raw in (None, ""):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        kernel = row.get("name", "unknown")
        raise ValueError(f"invalid integer field {name} for kernel {kernel}: {raw!r}") from exc


def ns_to_ms_x1000(ns: int) -> int:
    return int(round(ns / 1000.0))


def parse_min_total(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=MS")
    name, value = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("expected NAME=MS")
    try:
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid millisecond value: {raw}") from exc


def parse_wall_ms(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid wall millisecond value: {raw}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("--wall-ms must be finite and greater than zero")
    return value


def parse_kernel_bytes(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=BYTES")
    name, value = raw.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("expected NAME=BYTES")
    try:
        bytes_per_event = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte count: {raw}") from exc
    if bytes_per_event < 0:
        raise argparse.ArgumentTypeError("kernel bytes must be non-negative")
    return name, bytes_per_event


def parse_peak_gbs(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid peak bandwidth: {raw}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("--peak-gbs must be finite and greater than zero")
    return value


def parse_top_n(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid top-kernels value: {raw}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("--top-kernels must be non-negative")
    return value


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Counter[str]], Counter[str]]:
    """Aggregate rows by kernel name and category.

    Profiler artifacts are emitted at `(name, category, metadata)` granularity,
    but this parser's `--require-kernel NAME` and counter output are intentionally
    name-based so one kernel can be checked even when several metadata variants
    appear in a single artifact.
    """

    kernel_totals: dict[str, Counter[str]] = {}
    category_totals: Counter[str] = Counter()
    for row in rows:
        name = str(row.get("name", "unknown"))
        category = metric_name(str(row.get("category", "unknown")))
        count = int_field(row, "count")
        total_ns = int_field(row, "total_ns")
        failed = int_field(row, "failed_timestamps")
        totals = kernel_totals.setdefault(name, Counter())
        totals["count"] += count
        totals["total_ns"] += total_ns
        totals["failed_timestamps"] += failed
        category_totals[category] += total_ns
    return kernel_totals, category_totals


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize SYCL named kernel profile artifacts")
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument("--require-kernel", action="append", default=[])
    parser.add_argument("--min-total-ms", action="append", type=parse_min_total, default=[])
    parser.add_argument("--wall-ms", type=parse_wall_ms)
    parser.add_argument("--kernel-bytes", action="append", type=parse_kernel_bytes, default=[])
    parser.add_argument("--top-kernels", type=parse_top_n)
    parser.add_argument(
        "--geometry",
        choices=sorted(GEOMETRY_PRESETS),
        help="derive per-kernel bytes from a named model geometry preset",
    )
    parser.add_argument(
        "--peak-gbs",
        type=parse_peak_gbs,
        help="device peak bandwidth in GB/s; reports each kernel's percentage of it",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.profile)
        kernel_totals, category_totals = aggregate_rows(rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to parse profile: {exc}")
        return 2

    ok = True

    for name in args.require_kernel:
        if name not in kernel_totals:
            print(f"missing required kernel: {name}")
            ok = False

    for name, threshold_ms in args.min_total_ms:
        totals = kernel_totals.get(name)
        if totals is None:
            print(f"missing threshold kernel: {name}")
            ok = False
            continue
        total_ms = totals["total_ns"] / 1_000_000.0
        if total_ms < threshold_ms:
            print(f"kernel below total_ms threshold: {name} observed={total_ms:.3f} required={threshold_ms:.3f}")
            ok = False

    kernel_bytes = dict(args.kernel_bytes)
    for name in kernel_bytes:
        if name not in kernel_totals:
            print(f"missing bytes kernel: {name}")
            ok = False

    if args.geometry is not None:
        preset = geometry_preset(args.geometry)
        matrix_bytes = geometry_expert_matrix_bytes(args.geometry)
        print(f"geometry.{args.geometry}.layers {preset['layers']}")
        print(f"geometry.{args.geometry}.experts {preset['experts']}")
        print(f"geometry.{args.geometry}.experts_active {preset['experts_active']}")
        print(f"geometry.{args.geometry}.expert_matrix_bytes {matrix_bytes}")
        moe_bytes_per_token = 0
        for role, matrices in sorted(preset["role_matrices"].items()):
            role_bytes = matrices * preset["experts_active"] * matrix_bytes
            moe_bytes_per_token += role_bytes
            print(f"geometry.{args.geometry}.role.{role}.bytes_per_call {role_bytes}")
        print(f"geometry.{args.geometry}.moe_bytes_per_token {preset['layers'] * moe_bytes_per_token}")
    if args.peak_gbs is not None:
        print(f"profile.peak_gbs_x1000 {int(round(args.peak_gbs * 1000.0))}")

    kernel_sum_total_ns = sum(totals["total_ns"] for totals in kernel_totals.values())
    print(f"profile.kernel_sum_total_ms_x1000 {ns_to_ms_x1000(kernel_sum_total_ns)}")
    ranked_kernels = sorted(kernel_totals.items(), key=lambda item: (-item[1]["total_ns"], item[0]))
    if args.top_kernels is not None and ranked_kernels:
        top_name, top_totals = ranked_kernels[0]
        print(f"cost.top1_kernel {top_name} {ns_to_ms_x1000(top_totals['total_ns'])}")
    if args.top_kernels is not None:
        for rank, (name, totals) in enumerate(ranked_kernels[: args.top_kernels], start=1):
            print(f"cost.kernel.rank.{rank}.name {name}")
            print(f"cost.kernel.rank.{rank}.total_ms_x1000 {ns_to_ms_x1000(totals['total_ns'])}")
            print(f"cost.kernel.rank.{rank}.count {totals['count']}")
            print(f"cost.kernel.rank.{rank}.failed_timestamps {totals['failed_timestamps']}")
    if args.wall_ms is not None:
        kernel_sum_ms = kernel_sum_total_ns / 1_000_000.0
        print(f"profile.decode_wall_ms_x1000 {int(round(args.wall_ms * 1000.0))}")
        print(f"profile.kernel_coverage_pct_x1000 {int(round(100000.0 * kernel_sum_ms / args.wall_ms))}")

    for name, totals in sorted(kernel_totals.items(), key=lambda item: item[1]["total_ns"], reverse=True):
        metric = metric_name(name)
        print(f"kernel.{metric}.count {totals['count']}")
        print(f"kernel.{metric}.total_ms_x1000 {ns_to_ms_x1000(totals['total_ns'])}")
        print(f"kernel.{metric}.failed_timestamps {totals['failed_timestamps']}")
        # Bytes per profiler event: an explicit `--kernel-bytes` always wins
        # over the `--geometry` preset so a capture can be corrected by hand.
        bytes_per_event = kernel_bytes.get(name)
        if bytes_per_event is None and args.geometry is not None:
            bytes_per_event = geometry_kernel_bytes(args.geometry, name)
            if bytes_per_event is not None:
                print(f"kernel.{metric}.geometry_bytes_per_call {bytes_per_event}")
        if bytes_per_event is not None:
            # Achieved GB/s is (bytes_per_event * event_count) / total_seconds;
            # the printed metric is x1000 scaled.
            achieved = achieved_gbs(bytes_per_event * totals["count"], totals["total_ns"] / 1e9)
            print(f"kernel.{metric}.achieved_gbps_x1000 {int(round(1000.0 * achieved))}")
            if args.peak_gbs is not None:
                pct = percent_of_peak(achieved, args.peak_gbs)
                print(f"kernel.{metric}.pct_of_peak_x1000 {int(round(1000.0 * pct))}")

    for category, total_ns in sorted(category_totals.items()):
        print(f"category.{category}.total_ms_x1000 {ns_to_ms_x1000(total_ns)}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
