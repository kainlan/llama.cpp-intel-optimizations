#!/usr/bin/env python3
"""Extract selected VTune SYCL computing task timing from CSV/TSV exports."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

TASK_COLUMNS = ("Computing Task", "Task", "Source Computing Task")
TIME_COLUMNS = ("Computing Task:Total Time", "GPU Time", "Total Time")


def csv_dialect(path: pathlib.Path) -> type[csv.Dialect]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first_line = handle.readline()
    return csv.excel_tab if "\t" in first_line else csv.excel


def validate_row_shape(row: dict[str | None, str | list[str] | None]) -> dict[str, str]:
    if None in row:
        raise ValueError("malformed VTune task row contains surplus fields")
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None or isinstance(value, list):
            raise ValueError("malformed VTune task row shape")
        normalized[key] = "" if value is None else value
    return normalized


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def parse_time_ms_x1000(raw: str) -> int:
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        return 0
    return int(round(float(cleaned) * 1000.0))


def read_tasks(path: pathlib.Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=csv_dialect(path))
        if reader.fieldnames is None:
            raise ValueError("missing VTune task header")
        for row in reader:
            normalized = validate_row_shape(row)
            task = first_present(normalized, TASK_COLUMNS)
            if not task:
                raise ValueError("missing task name value")
            time_value = first_present(normalized, TIME_COLUMNS)
            tasks.append((task, parse_time_ms_x1000(time_value)))
    if not tasks:
        raise ValueError("no VTune task rows found")
    return tasks


def select_task(tasks: list[tuple[str, int]], match: str) -> tuple[str, int]:
    ranked = sorted(tasks, key=lambda item: (-item[1], item[0]))
    for task, time_ms_x1000 in ranked:
        if match in task:
            return task, time_ms_x1000
    raise ValueError(f"no task matched {match}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract selected VTune SYCL computing task timing")
    parser.add_argument("tasks", type=pathlib.Path, help="VTune computing task CSV/TSV export")
    parser.add_argument("--match", required=True, help="Substring to match in the computing task name")
    args = parser.parse_args(argv)

    try:
        tasks = read_tasks(args.tasks)
        selected, selected_time_ms_x1000 = select_task(tasks, args.match)
    except (OSError, csv.Error, ValueError, TypeError) as exc:
        print(f"failed to parse VTune tasks: {exc}")
        return 2

    print("vtune_task.status ok")
    print(f"vtune_task.match {args.match}")
    print(f"vtune_task.selected {selected}")
    print(f"vtune_task.selected_time_ms_x1000 {selected_time_ms_x1000}")
    print(f"vtune_task.count {len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
