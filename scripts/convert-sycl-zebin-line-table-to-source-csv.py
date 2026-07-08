#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import pathlib
import posixpath
import sys
from collections.abc import Iterator
from typing import Any, TextIO

CSV_FIELDS = [
    "Source Line",
    "Source File",
    "Source File Path",
    "Source Computing Task",
    "Address",
    "Line",
    "Column",
    "Source Attribution Mode",
    "Source Attribution Status",
]

ATTRIBUTION_MODE = "dwarf-line-table"
ATTRIBUTION_STATUS = "dwarf_line_table_only"


def load_line_table_parser() -> Any:
    path = pathlib.Path(__file__).resolve().with_name("parse-sycl-zebin-line-table.py")
    spec = importlib.util.spec_from_file_location("parse_sycl_zebin_line_table", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load ZEBin line-table parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_text(input_path: pathlib.Path | None) -> str:
    if input_path is None:
        return sys.stdin.read()
    return input_path.read_text(encoding="utf-8", errors="replace")


@contextlib.contextmanager
def open_output(output_path: pathlib.Path | None) -> Iterator[TextIO]:
    if output_path is None:
        yield sys.stdout
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        yield handle


def source_file_name(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert decoded llvm-dwarfdump --debug-line text from a SYCL ZEBin into checker-compatible source-line CSV rows."
    )
    parser.add_argument("--input", type=pathlib.Path, help="decoded debug-line text file; defaults to stdin")
    parser.add_argument("--output", type=pathlib.Path, help="CSV output path; defaults to stdout")
    parser.add_argument("--source-computing-task", default="", help="optional kernel/task name to write into Source Computing Task")
    args = parser.parse_args(argv)

    try:
        module = load_line_table_parser()
        line_table_error = getattr(module, "LineTableError", None)
        try:
            rows = module.parse_line_table_rows(read_text(args.input))
        except Exception as exc:
            if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
                raise ValueError(str(exc)) from None
            raise
        with open_output(args.output) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Source Line": f"{row.file_path}:{row.line}",
                        "Source File": source_file_name(row.file_path),
                        "Source File Path": row.file_path,
                        "Source Computing Task": args.source_computing_task,
                        "Address": row.address,
                        "Line": str(row.line),
                        "Column": str(row.column),
                        "Source Attribution Mode": ATTRIBUTION_MODE,
                        "Source Attribution Status": ATTRIBUTION_STATUS,
                    }
                )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"failed to convert ZEBin line table: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
