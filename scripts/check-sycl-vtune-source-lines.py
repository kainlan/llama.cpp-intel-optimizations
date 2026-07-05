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
DWARF_ATTRIBUTION_MODE = "dwarf-line-table"
VTUNE_ATTRIBUTION_MODE = "vtune-sampled-exact"
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


def row_attribution_mode(row: dict[str, str]) -> str:
    return row.get("Source Attribution Mode", "").strip()


def row_is_dwarf_line_table(row: dict[str, str]) -> bool:
    return row_attribution_mode(row) == DWARF_ATTRIBUTION_MODE


def read_source_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        sample_lines = sample.splitlines()
        dialect = csv.excel_tab if sample_lines and "\t" in sample_lines[0] else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [validate_row_shape(raw_row) for raw_row in reader]


def count_vtune_sampled_known_rows(rows: list[dict[str, str]], required_kernel: str | None) -> int:
    return sum(
        1
        for row in rows
        if row_matches_kernel(row, required_kernel) and row_has_known_source(row) and not row_is_dwarf_line_table(row)
    )


def count_dwarf_line_table_known_rows(rows: list[dict[str, str]], required_kernel: str | None) -> int:
    return sum(
        1
        for row in rows
        if row_matches_kernel(row, required_kernel) and row_has_known_source(row) and row_is_dwarf_line_table(row)
    )


def parse_dwarf_line_table(module: Any, text: str, require_path: str | None) -> dict[str, object]:
    try:
        return module.parse_line_table(text, require_path)
    except Exception as exc:
        line_table_error = getattr(module, "LineTableError", None)
        if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
            raise ValueError(str(exc)) from None
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check VTune GPU source-line feasibility for SYCL kernels")
    parser.add_argument("--readelf-sections", required=True, type=pathlib.Path)
    parser.add_argument("--vtune-csv", type=pathlib.Path)
    parser.add_argument("--require-kernel")
    parser.add_argument("--dwarf-line-dump", type=pathlib.Path)
    parser.add_argument("--require-source-path")
    parser.add_argument(
        "--dwarf-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from decoded ZEBin DWARF line tables",
    )
    parser.add_argument(
        "--allow-dwarf-line-table-only",
        action="store_true",
        help="allow DWARF line-table CSV rows to pass with source_line.status dwarf-line-table-only when VTune rows are unavailable",
    )
    args = parser.parse_args(argv)

    if args.vtune_csv is None and not (args.allow_dwarf_line_table_only and args.dwarf_source_lines_csv is not None):
        print("failed to check source lines: --vtune-csv is required unless --allow-dwarf-line-table-only and --dwarf-source-lines-csv are both provided")
        return 2

    try:
        sections = args.readelf_sections.read_text(encoding="utf-8", errors="replace")
        debug_line_present = DEBUG_LINE_SECTION_RE.search(sections) is not None
        non_unknown_rows = 0
        if args.vtune_csv is not None:
            vtune_rows = read_source_csv(args.vtune_csv)
            non_unknown_rows = count_vtune_sampled_known_rows(vtune_rows, args.require_kernel)

        dwarf_status = "not_checked"
        dwarf_source_rows = 0
        dwarf_required_path_present = True
        if args.dwarf_line_dump is not None:
            module = load_line_table_parser()
            parsed = parse_dwarf_line_table(
                module,
                args.dwarf_line_dump.read_text(encoding="utf-8", errors="replace"),
                args.require_source_path,
            )
            dwarf_status = str(parsed["status"])
            dwarf_source_rows = int(parsed["source_rows"])
            dwarf_required_path_present = bool(parsed["required_path_present"])

        dwarf_source_line_rows = 0
        if args.dwarf_source_lines_csv is not None:
            dwarf_source_line_rows = count_dwarf_line_table_known_rows(
                read_source_csv(args.dwarf_source_lines_csv),
                args.require_kernel,
            )
    except (OSError, csv.Error, IndexError, TypeError, ValueError) as exc:
        print(f"failed to check source lines: {exc}")
        return 2

    source_attribution_mode = "none"
    status = "fail"
    if not debug_line_present:
        blocker = "missing_debug_line"
    elif args.dwarf_line_dump is not None and not dwarf_required_path_present:
        blocker = "missing_dwarf_source_path"
    elif non_unknown_rows > 0:
        blocker = "none"
        status = "pass"
        source_attribution_mode = VTUNE_ATTRIBUTION_MODE
    elif args.allow_dwarf_line_table_only and dwarf_source_line_rows > 0:
        blocker = "none"
        status = "dwarf-line-table-only"
        source_attribution_mode = DWARF_ATTRIBUTION_MODE
    else:
        blocker = "vtune_unknown_source"

    print(f"source_line.debug_line_present {1 if debug_line_present else 0}")
    print(f"source_line.non_unknown_rows {non_unknown_rows}")
    print(f"source_line.vtune_sampled_non_unknown_rows {non_unknown_rows}")
    if args.require_kernel is not None:
        print(f"source_line.required_kernel {args.require_kernel}")
    if args.dwarf_line_dump is not None:
        print(f"source_line.dwarf_status {dwarf_status}")
        print(f"source_line.dwarf_source_rows {dwarf_source_rows}")
        print(f"source_line.dwarf_required_path_present {1 if dwarf_required_path_present else 0}")
    if args.dwarf_source_lines_csv is not None:
        print(f"source_line.dwarf_source_line_rows {dwarf_source_line_rows}")
        print(f"source_line.allow_dwarf_line_table_only {1 if args.allow_dwarf_line_table_only else 0}")
    print(f"source_line.source_attribution_mode {source_attribution_mode}")
    print(f"source_line.blocker {blocker}")
    print(f"source_line.status {status}")
    return 0 if status in {"pass", "dwarf-line-table-only"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
