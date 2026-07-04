#!/usr/bin/env python3
"""Normalize VTune CSV exports for SYCL profiling ledgers."""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys
from typing import Any

UNKNOWN_VALUES = {"", "[Unknown]", "[Unknown source file]"}


def validate_row_shape(row: dict[str | None, str | list[str] | None]) -> dict[str, str]:
    if None in row:
        raise ValueError("malformed CSV row contains surplus fields")
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None or isinstance(value, list):
            raise ValueError("malformed CSV row shape")
        normalized[key] = "" if value is None else value
    return normalized


def csv_dialect(path: pathlib.Path) -> type[csv.Dialect]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first_line = handle.readline()
    return csv.excel_tab if "\t" in first_line else csv.excel


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=csv_dialect(path))
        return [validate_row_shape(row) for row in reader]


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            return value
    return ""


def parse_time_ms(raw: str) -> int:
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        raise ValueError("missing time value")
    return int(round(float(cleaned) * 1000.0))


def row_matches_kernel(row: dict[str, str], required: str | None) -> bool:
    if required is None:
        return True
    joined = " ".join(value for value in row.values() if value)
    return required in joined


def row_has_known_source(row: dict[str, str]) -> bool:
    candidates = (row.get("Source Line", ""), row.get("Source File", ""), row.get("Source File Path", ""))
    return any(candidate not in UNKNOWN_VALUES for candidate in candidates)


def parse_kernel_rows(path: pathlib.Path) -> tuple[int, dict[str, dict[str, int]]]:
    totals: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"total": 0, "count": 0})
    for row in read_csv_rows(path):
        task = first_present(row, ("Computing Task", "Task"))
        if not task:
            continue
        time_value = first_present(row, ("Computing Task:Total Time", "GPU Time", "Total Time"))
        total = parse_time_ms(time_value)
        totals[task]["total"] += total
        totals[task]["count"] += 1
    return sum(item["total"] for item in totals.values()), totals


def parse_api_rows(path: pathlib.Path) -> int:
    total = 0
    for row in read_csv_rows(path):
        name = first_present(row, ("Function", "API", "Task"))
        if not name:
            continue
        time_value = first_present(row, ("CPU Time", "Total Time", "Duration"))
        total += parse_time_ms(time_value)
    return total


def parse_source_rows(path: pathlib.Path, required_kernel: str | None) -> tuple[int, int]:
    known = 0
    unknown = 0
    for row in read_csv_rows(path):
        if not row_matches_kernel(row, required_kernel):
            continue
        if row_has_known_source(row):
            known += 1
        else:
            unknown += 1
    return known, unknown


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Normalize VTune CSV exports for SYCL profiling")
    parser.add_argument("--kernel-csv", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--api-csv", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--source-csv", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--require-kernel")
    args = parser.parse_args(argv)

    try:
        kernel_total = 0
        kernel_totals: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"total": 0, "count": 0})
        for path in args.kernel_csv:
            path_total, path_totals = parse_kernel_rows(path)
            kernel_total += path_total
            for name, totals in path_totals.items():
                kernel_totals[name]["total"] += totals["total"]
                kernel_totals[name]["count"] += totals["count"]

        api_total = 0
        for path in args.api_csv:
            api_total += parse_api_rows(path)

        known_source = 0
        unknown_source = 0
        for path in args.source_csv:
            known, unknown = parse_source_rows(path, args.require_kernel)
            known_source += known
            unknown_source += unknown
    except (OSError, csv.Error, ValueError, TypeError, IndexError) as exc:
        print(f"failed to parse VTune exports: {exc}")
        return 2

    print(f"vtune.kernel_total_ms_x1000 {kernel_total}")
    ranked_kernels = sorted(kernel_totals.items(), key=lambda item: (-item[1]["total"], item[0]))
    for rank, (name, totals) in enumerate(ranked_kernels, start=1):
        print(f"vtune.kernel.rank.{rank}.name {name}")
        print(f"vtune.kernel.rank.{rank}.total_ms_x1000 {totals['total']}")
        print(f"vtune.kernel.rank.{rank}.count {totals['count']}")
    print(f"vtune.api_total_ms_x1000 {api_total}")
    print(f"vtune.source.known_rows {known_source}")
    print(f"vtune.source.unknown_rows {unknown_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
