#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import contextlib
import csv
import importlib.util
import pathlib
import posixpath
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TextIO

CSV_FIELDS = [
    "Source Line",
    "Source File",
    "Source File Path",
    "Source Computing Task",
    "Sample Count",
    "Sample Kind",
    "Source Attribution Mode",
    "Source Attribution Status",
    "source_file",
    "source_line",
    "sample_count",
    "kernel",
]

ATTRIBUTION_MODE = "sampled-pc-line"
ATTRIBUTION_STATUS = "sampled_line_cost"
NO_MATCH_BLOCKER = "no_pc_sample_source_matches"


class ResolveError(ValueError):
    pass


@dataclass(frozen=True)
class PcSample:
    kernel: str
    pc: int
    sample_count: int
    sample_kind: str


@dataclass
class LineAggregate:
    file_path: str
    line: int
    column: int
    address: int
    sample_kind: str
    sample_count: int = 0


def load_module(module_name: str, file_name: str) -> Any:
    path = pathlib.Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ResolveError(f"failed to load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_existing_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise ResolveError(f"{label} file does not exist: {path}")


def source_file_name(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def path_matches(path: str, required_path: str) -> bool:
    if not required_path:
        return True
    candidate = posixpath.normpath(path.strip())
    required = posixpath.normpath(required_path.strip())
    if required.startswith("/"):
        return candidate == required
    return candidate == required or candidate.endswith("/" + required)


def parse_int(raw: str, field: str) -> int:
    text = raw.strip().replace(",", "")
    if not text:
        raise ResolveError(f"missing {field}")
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    except ValueError as exc:
        raise ResolveError(f"invalid {field}: {raw}") from exc


def validate_row_shape(row: dict[str | None, str | list[str] | None]) -> dict[str, str]:
    if None in row:
        raise ResolveError("malformed PC sample CSV row contains surplus fields")
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None or isinstance(value, list):
            raise ResolveError("malformed PC sample CSV row shape")
        normalized[key] = "" if value is None else value
    return normalized


def load_samples(path: pathlib.Path, source_computing_task: str) -> list[PcSample]:
    require_existing_file(path, "PC samples")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"kernel", "pc", "sample_count", "sample_kind"}
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ResolveError(f"PC sample CSV missing required columns: {', '.join(missing)}")
        samples: list[PcSample] = []
        for raw_row in reader:
            row = validate_row_shape(raw_row)
            kernel = row["kernel"].strip()
            if kernel != source_computing_task:
                continue
            sample_kind = row["sample_kind"].strip()
            if not sample_kind:
                raise ResolveError("missing sample_kind")
            sample_count = parse_int(row["sample_count"], "sample_count")
            if sample_count <= 0:
                raise ResolveError("sample_count must be positive")
            samples.append(
                PcSample(
                    kernel=kernel,
                    pc=parse_int(row["pc"], "pc"),
                    sample_count=sample_count,
                    sample_kind=sample_kind,
                )
            )
    return samples


def load_source_rows(dwarf_line_dump: pathlib.Path) -> list[Any]:
    require_existing_file(dwarf_line_dump, "DWARF line dump")
    module = load_module("parse_sycl_zebin_line_table", "parse-sycl-zebin-line-table.py")
    text = dwarf_line_dump.read_text(encoding="utf-8", errors="replace")
    try:
        rows = module.parse_line_table_rows(text)
    except Exception as exc:
        line_table_error = getattr(module, "LineTableError", None)
        if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
            raise ResolveError(str(exc)) from None
        raise
    return sorted(rows, key=lambda row: parse_int(row.address, "DWARF address"))


def source_row_for_pc(source_rows: list[Any], addresses: list[int], pc: int) -> Any | None:
    index = bisect.bisect_right(addresses, pc) - 1
    if index < 0:
        return None
    if index + 1 < len(addresses):
        if pc < addresses[index + 1]:
            return source_rows[index]
        return None
    return source_rows[index]


def aggregate_rows(source_rows: list[Any], samples: list[PcSample], require_source_path: str) -> tuple[list[LineAggregate], int, int]:
    addresses = [parse_int(row.address, "DWARF address") for row in source_rows]
    aggregates: dict[tuple[str, int, str], LineAggregate] = {}
    mapped = 0
    unmapped = 0

    for sample in samples:
        source_row = source_row_for_pc(source_rows, addresses, sample.pc)
        if source_row is None:
            unmapped += sample.sample_count
            continue
        if require_source_path and not path_matches(source_row.file_path, require_source_path):
            unmapped += sample.sample_count
            continue

        source_address = parse_int(source_row.address, "DWARF address")
        source_column = int(source_row.column)
        key = (source_row.file_path, int(source_row.line), sample.sample_kind)
        aggregate = aggregates.get(key)
        if aggregate is None:
            aggregate = LineAggregate(
                file_path=source_row.file_path,
                line=int(source_row.line),
                column=source_column,
                address=source_address,
                sample_kind=sample.sample_kind,
            )
            aggregates[key] = aggregate
        else:
            aggregate.column = min(aggregate.column, source_column)
            aggregate.address = min(aggregate.address, source_address)
        aggregate.sample_count += sample.sample_count
        mapped += sample.sample_count

    rows = sorted(aggregates.values(), key=lambda row: (-row.sample_count, row.file_path, row.line, row.sample_kind))
    return rows, mapped, unmapped


@contextlib.contextmanager
def open_output(output_path: pathlib.Path | None) -> Iterator[TextIO]:
    if output_path is None:
        yield sys.stdout
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        yield handle


def row_to_csv(
    row: LineAggregate,
    source_computing_task: str,
    attribution_mode: str,
    attribution_status: str,
) -> dict[str, str]:
    source_line = f"{row.file_path}:{row.line}"
    sample_count = str(row.sample_count)
    return {
        "Source Line": source_line,
        "Source File": source_file_name(row.file_path),
        "Source File Path": row.file_path,
        "Source Computing Task": source_computing_task,
        "Sample Count": sample_count,
        "Sample Kind": row.sample_kind,
        "Source Attribution Mode": attribution_mode,
        "Source Attribution Status": attribution_status,
        "source_file": row.file_path,
        "source_line": str(row.line),
        "sample_count": sample_count,
        "kernel": source_computing_task,
    }


def write_csv(
    rows: list[LineAggregate],
    output_path: pathlib.Path | None,
    source_computing_task: str,
    attribution_mode: str,
    attribution_status: str,
) -> None:
    with open_output(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row, source_computing_task, attribution_mode, attribution_status))
        handle.flush()


def write_summary(
    rows: list[LineAggregate],
    summary_output: pathlib.Path | None,
    mapped: int,
    unmapped: int,
    summary_prefix: str,
) -> None:
    if summary_output is None:
        return
    lines = [
        f"{summary_prefix}.status {'ok' if rows else NO_MATCH_BLOCKER}",
        f"{summary_prefix}.mapped_sample_count {mapped}",
        f"{summary_prefix}.unmapped_sample_count {unmapped}",
        f"{summary_prefix}.source_line_rows {len(rows)}",
    ]
    if rows:
        top = rows[0]
        lines.extend(
            [
                f"{summary_prefix}.top_source_line {top.file_path}:{top.line}",
                f"{summary_prefix}.top_sample_count {top.sample_count}",
                f"{summary_prefix}.top_sample_kind {top.sample_kind}",
            ]
        )
    else:
        lines.append(f"{summary_prefix}.blocker {NO_MATCH_BLOCKER}")
    summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve dynamic SYCL EU PC sample counts to DWARF debug-line source rows. "
            "Samples are matched to nearest-preceding line-table ranges; output is sampled runtime line cost, "
            "not static instruction cost and not VTune exact source rows."
        )
    )
    parser.add_argument("--dwarf-line-dump", required=True, type=pathlib.Path)
    parser.add_argument("--pc-samples", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="CSV output path; defaults to stdout")
    parser.add_argument("--summary-output", type=pathlib.Path, help="optional parse-style summary output")
    parser.add_argument("--source-computing-task", required=True, help="required kernel/task name to write into output rows")
    parser.add_argument("--require-source-path", default="", help="optional source path suffix that mapped rows must match")
    parser.add_argument(
        "--attribution-mode",
        default=ATTRIBUTION_MODE,
        help="Source Attribution Mode label for output rows; default is sampled-pc-line",
    )
    parser.add_argument(
        "--attribution-status",
        default=ATTRIBUTION_STATUS,
        help="Source Attribution Status label for output rows; default is sampled_line_cost",
    )
    parser.add_argument(
        "--summary-prefix",
        default="pc_sample_source",
        help="prefix for parse-style summary keys; default is pc_sample_source",
    )
    args = parser.parse_args(argv)

    try:
        source_rows = load_source_rows(args.dwarf_line_dump)
        samples = load_samples(args.pc_samples, args.source_computing_task)
        rows, mapped, unmapped = aggregate_rows(source_rows, samples, args.require_source_path)
        write_csv(rows, args.output, args.source_computing_task, args.attribution_mode, args.attribution_status)
        write_summary(rows, args.summary_output, mapped, unmapped, args.summary_prefix)
        if mapped == 0:
            print(
                f"failed to resolve SYCL PC sample source lines: no mapped PC sample source rows ({NO_MATCH_BLOCKER})",
                file=sys.stderr,
            )
            return 2
    except (OSError, UnicodeDecodeError, ResolveError) as exc:
        print(f"failed to resolve SYCL PC sample source lines: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
