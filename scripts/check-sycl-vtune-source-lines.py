#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import re
import sys
from typing import Any

UNKNOWN_VALUES = {"", "[Unknown]", "[Unknown source file]"}
DEBUG_LINE_SECTION_RE = re.compile(r"(?m)^\s*\[\s*\d+\]\s+\.debug_line(?:\s|$)")


def load_line_table_parser() -> Any:
    path = pathlib.Path(__file__).resolve().with_name("parse-sycl-zebin-line-table.py")
    spec = importlib.util.spec_from_file_location("parse_sycl_zebin_line_table", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load ZEBin line-table parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    parser.add_argument("--dwarf-line-dump", type=pathlib.Path)
    parser.add_argument("--require-source-path")
    args = parser.parse_args(argv)

    try:
        sections = args.readelf_sections.read_text(encoding="utf-8", errors="replace")
        debug_line_present = DEBUG_LINE_SECTION_RE.search(sections) is not None
        with args.vtune_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            sample_lines = sample.splitlines()
            dialect = csv.excel_tab if sample_lines and "\t" in sample_lines[0] else csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            non_unknown_rows = 0
            for raw_row in reader:
                row = validate_row_shape(raw_row)
                if row_matches_kernel(row, args.require_kernel) and row_has_known_source(row):
                    non_unknown_rows += 1
        dwarf_status = "not_checked"
        dwarf_source_rows = 0
        dwarf_required_path_present = True
        if args.dwarf_line_dump is not None:
            module = load_line_table_parser()
            parsed = module.parse_line_table(
                args.dwarf_line_dump.read_text(encoding="utf-8", errors="replace"),
                args.require_source_path,
            )
            dwarf_status = str(parsed["status"])
            dwarf_source_rows = int(parsed["source_rows"])
            dwarf_required_path_present = bool(parsed["required_path_present"])
    except (OSError, csv.Error, IndexError, TypeError, ValueError) as exc:
        print(f"failed to check source lines: {exc}")
        return 2

    if not debug_line_present:
        passed = False
        blocker = "missing_debug_line"
    elif args.dwarf_line_dump is not None and not dwarf_required_path_present:
        passed = False
        blocker = "missing_dwarf_source_path"
    elif non_unknown_rows > 0:
        passed = True
        blocker = "none"
    else:
        passed = False
        blocker = "vtune_unknown_source"

    print(f"source_line.debug_line_present {1 if debug_line_present else 0}")
    print(f"source_line.non_unknown_rows {non_unknown_rows}")
    if args.require_kernel is not None:
        print(f"source_line.required_kernel {args.require_kernel}")
    if args.dwarf_line_dump is not None:
        print(f"source_line.dwarf_status {dwarf_status}")
        print(f"source_line.dwarf_source_rows {dwarf_source_rows}")
        print(f"source_line.dwarf_required_path_present {1 if dwarf_required_path_present else 0}")
    print(f"source_line.blocker {blocker}")
    print(f"source_line.status {'pass' if passed else 'fail'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
