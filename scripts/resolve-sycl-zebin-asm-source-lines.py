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
    "Address",
    "Line",
    "Column",
    "Static Instruction Count",
    "Static Dpas Count",
    "Static Send Ugm Count",
    "Static Send Count",
    "Static Math Count",
    "Static Score",
    "Source Attribution Mode",
    "Source Attribution Status",
    "source_file",
    "source_line",
    "instruction_count",
    "static_score",
    "kernel",
]

ATTRIBUTION_MODE = "asm-line-static"
ATTRIBUTION_STATUS = "asm_line_static_cost"
NO_MATCH_BLOCKER = "no_asm_source_matches"


class ResolveError(ValueError):
    pass


@dataclass
class LineAggregate:
    file_path: str
    line: int
    column: int
    address: int
    instruction_count: int = 0
    dpas_count: int = 0
    send_ugm_count: int = 0
    send_count: int = 0
    math_count: int = 0

    def score(self) -> int:
        return self.instruction_count + self.dpas_count * 8 + self.send_ugm_count * 4 + self.math_count * 2


def load_module(module_name: str, file_name: str) -> Any:
    path = pathlib.Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ResolveError(f"failed to load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_file_name(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def parse_hex_address(raw: str) -> int:
    return int(raw, 16)


def require_existing_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise ResolveError(f"{label} file does not exist: {path}")


def load_source_rows(dwarf_line_dump: pathlib.Path) -> list[Any]:
    require_existing_file(dwarf_line_dump, "DWARF line dump")
    module = load_module("parse_sycl_zebin_line_table", "parse-sycl-zebin-line-table.py")
    try:
        rows = module.parse_line_table_rows(dwarf_line_dump.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        line_table_error = getattr(module, "LineTableError", None)
        if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
            raise ResolveError(str(exc)) from None
        raise
    return sorted(rows, key=lambda row: parse_hex_address(row.address))


def load_instructions(asm_path: pathlib.Path) -> list[Any]:
    require_existing_file(asm_path, "ASM")
    module = load_module("parse_sycl_vtune_kernel_asm", "parse-sycl-vtune-kernel-asm.py")
    return sorted(module.parse_asm_instructions(asm_path), key=lambda row: row.address)


def path_matches(path: str, required_path: str) -> bool:
    if not required_path:
        return True
    candidate = posixpath.normpath(path.strip())
    required = posixpath.normpath(required_path.strip())
    if required.startswith("/"):
        return candidate == required
    return candidate == required or candidate.endswith("/" + required)


def source_row_for_instruction(source_rows: list[Any], addresses: list[int], instruction_address: int) -> Any | None:
    index = bisect.bisect_right(addresses, instruction_address) - 1
    if index < 0:
        return None
    if index + 1 < len(addresses):
        if instruction_address < addresses[index + 1]:
            return source_rows[index]
        return None
    if instruction_address == addresses[index]:
        return source_rows[index]
    return None


def aggregate_rows(source_rows: list[Any], instructions: list[Any], require_source_path: str) -> tuple[list[LineAggregate], int, int]:
    addresses = [parse_hex_address(row.address) for row in source_rows]
    aggregates: dict[tuple[str, int], LineAggregate] = {}
    mapped = 0
    unmapped = 0

    for instruction in instructions:
        source_row = source_row_for_instruction(source_rows, addresses, instruction.address)
        if source_row is None:
            unmapped += 1
            continue
        if require_source_path and not path_matches(source_row.file_path, require_source_path):
            unmapped += 1
            continue

        source_address = parse_hex_address(source_row.address)
        source_column = int(source_row.column)
        key = (source_row.file_path, int(source_row.line))
        aggregate = aggregates.get(key)
        if aggregate is None:
            aggregate = LineAggregate(
                file_path=source_row.file_path,
                line=int(source_row.line),
                column=source_column,
                address=source_address,
            )
            aggregates[key] = aggregate
        else:
            aggregate.column = min(aggregate.column, source_column)
            aggregate.address = min(aggregate.address, source_address)

        aggregate.instruction_count += 1
        if instruction.opcode.startswith("dpas"):
            aggregate.dpas_count += 1
        if instruction.opcode.startswith("send"):
            aggregate.send_count += 1
        if instruction.opcode == "send.ugm":
            aggregate.send_ugm_count += 1
        if instruction.opcode.startswith("math."):
            aggregate.math_count += 1
        mapped += 1

    rows = sorted(aggregates.values(), key=lambda row: (-row.score(), row.file_path, row.line, row.column))
    return rows, mapped, unmapped


@contextlib.contextmanager
def open_output(output_path: pathlib.Path | None) -> Iterator[TextIO]:
    if output_path is None:
        yield sys.stdout
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        yield handle


def row_to_csv(row: LineAggregate, source_computing_task: str) -> dict[str, str]:
    source_line = f"{row.file_path}:{row.line}"
    instruction_count = str(row.instruction_count)
    static_score = str(row.score())
    return {
        "Source Line": source_line,
        "Source File": source_file_name(row.file_path),
        "Source File Path": row.file_path,
        "Source Computing Task": source_computing_task,
        "Address": hex(row.address),
        "Line": str(row.line),
        "Column": str(row.column),
        "Static Instruction Count": instruction_count,
        "Static Dpas Count": str(row.dpas_count),
        "Static Send Ugm Count": str(row.send_ugm_count),
        "Static Send Count": str(row.send_count),
        "Static Math Count": str(row.math_count),
        "Static Score": static_score,
        "Source Attribution Mode": ATTRIBUTION_MODE,
        "Source Attribution Status": ATTRIBUTION_STATUS,
        "source_file": row.file_path,
        "source_line": str(row.line),
        "instruction_count": instruction_count,
        "static_score": static_score,
        "kernel": source_computing_task,
    }


def write_csv(rows: list[LineAggregate], output_path: pathlib.Path | None, source_computing_task: str) -> None:
    with open_output(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row, source_computing_task))


def write_summary(rows: list[LineAggregate], summary_output: pathlib.Path | None, mapped: int, unmapped: int) -> None:
    if summary_output is None:
        return
    lines = [
        f"asm_source.status {'ok' if rows else NO_MATCH_BLOCKER}",
        f"asm_source.mapped_instruction_count {mapped}",
        f"asm_source.unmapped_instruction_count {unmapped}",
        f"asm_source.source_line_rows {len(rows)}",
    ]
    if rows:
        top = rows[0]
        lines.extend(
            [
                f"asm_source.top_source_line {top.file_path}:{top.line}",
                f"asm_source.top_static_score {top.score()}",
                f"asm_source.top_instruction_count {top.instruction_count}",
            ]
        )
    else:
        lines.append(f"asm_source.blocker {NO_MATCH_BLOCKER}")
    summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve addressed ZEBin EU assembly instructions to DWARF debug-line source rows. "
            "Instruction addresses are matched to closed known line-table ranges: each row covers "
            "addresses from its address up to the next row address, and the final row only covers "
            "an instruction at that exact address."
        )
    )
    parser.add_argument("--dwarf-line-dump", required=True, type=pathlib.Path)
    parser.add_argument("--asm", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="CSV output path; defaults to stdout")
    parser.add_argument("--summary-output", type=pathlib.Path, help="optional parse-style summary output")
    parser.add_argument("--source-computing-task", required=True, help="required kernel/task name to write into output rows")
    parser.add_argument("--require-source-path", default="", help="optional source path suffix that mapped rows must match")
    args = parser.parse_args(argv)

    try:
        source_rows = load_source_rows(args.dwarf_line_dump)
        instructions = load_instructions(args.asm)
        rows, mapped, unmapped = aggregate_rows(source_rows, instructions, args.require_source_path)
        write_csv(rows, args.output, args.source_computing_task)
        write_summary(rows, args.summary_output, mapped, unmapped)
    except (OSError, UnicodeDecodeError, ResolveError) as exc:
        print(f"failed to resolve ZEBin ASM source lines: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
