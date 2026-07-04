#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

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


def row_matches_kernel(row: dict[str, str], required: str | None) -> bool:
    if required is None:
        return True
    joined = " ".join(value for value in row.values() if value)
    return required in joined


def row_has_known_source(row: dict[str, str]) -> bool:
    candidates = [row.get("Source Line", ""), row.get("Source File", ""), row.get("Source File Path", "")]
    return any(candidate not in UNKNOWN_VALUES for candidate in candidates)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check VTune GPU source-line feasibility for SYCL kernels")
    parser.add_argument("--readelf-sections", required=True, type=pathlib.Path)
    parser.add_argument("--vtune-csv", required=True, type=pathlib.Path)
    parser.add_argument("--require-kernel")
    args = parser.parse_args(argv)

    try:
        sections = args.readelf_sections.read_text(encoding="utf-8", errors="replace")
        debug_line_present = ".debug_line" in sections
        with args.vtune_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.excel_tab if "\t" in sample.splitlines()[0] else csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            non_unknown_rows = 0
            for raw_row in reader:
                row = validate_row_shape(raw_row)
                if row_matches_kernel(row, args.require_kernel) and row_has_known_source(row):
                    non_unknown_rows += 1
    except (OSError, csv.Error, IndexError, TypeError, ValueError) as exc:
        print(f"failed to check source lines: {exc}")
        return 2

    passed = debug_line_present and non_unknown_rows > 0
    if passed:
        blocker = "none"
    elif not debug_line_present:
        blocker = "missing_debug_line"
    else:
        blocker = "vtune_unknown_source"

    print(f"source_line.debug_line_present {1 if debug_line_present else 0}")
    print(f"source_line.non_unknown_rows {non_unknown_rows}")
    if args.require_kernel is not None:
        print(f"source_line.required_kernel {args.require_kernel}")
    print(f"source_line.blocker {blocker}")
    print(f"source_line.status {'pass' if passed else 'fail'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
