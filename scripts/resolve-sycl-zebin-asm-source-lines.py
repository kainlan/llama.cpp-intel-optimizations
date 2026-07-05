#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import contextlib
import csv
import importlib.util
import pathlib
import posixpath
import re
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

ASM_KERNEL_MARKER_PATTERNS = (
    re.compile(r"^\s*\.kernel\s+\"?([^\"\s]+)\"?", re.IGNORECASE),
    re.compile(r"^\s*(?://\s*)?(?:kernel(?:\s+name)?|function|computing\s+task)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?://\s*)?disassembly\s+of\s+(?:kernel|function)\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*disassembly\s+of\s+section\s+\.text(?:[.$])?([^:]+):\s*$", re.IGNORECASE),
)
DWARF_KERNEL_MARKER_PATTERNS = (
    re.compile(r"^\s*(?://\s*)?(?:kernel(?:\s+name)?|function|computing\s+task)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?://\s*)?debug[_ -]?line\s+(?:for|kernel|function)\s+(.+?)\s*$", re.IGNORECASE),
)


class ResolveError(ValueError):
    pass


@dataclass(frozen=True)
class NamedTextSection:
    name: str | None
    text: str


@dataclass(frozen=True)
class PcInstructionRow:
    address: int
    opcode: str
    text: str
    raw: str
    send_comment: str


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


def normalize_task_text(raw: str) -> str:
    return raw.strip().strip('"\'').rstrip(":")


def task_matches(name: str | None, required_task: str) -> bool:
    if not required_task:
        return True
    if name is None:
        return False
    name_text = normalize_task_text(name)
    required_text = normalize_task_text(required_task)
    return name_text == required_text


def kernel_marker_name(raw: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.match(raw)
        if match:
            return normalize_task_text(match.group(1))
    return None


def split_named_sections(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[list[NamedTextSection], bool]:
    sections: list[NamedTextSection] = []
    current_name: str | None = None
    current_lines: list[str] = []
    found_named_section = False

    for raw in text.splitlines():
        marker_name = kernel_marker_name(raw, patterns)
        if marker_name is not None:
            if current_name is not None or current_lines:
                sections.append(NamedTextSection(current_name, "\n".join(current_lines) + "\n"))
            current_name = marker_name
            current_lines = [raw]
            found_named_section = True
            continue
        current_lines.append(raw)

    if current_name is not None or current_lines:
        sections.append(NamedTextSection(current_name, "\n".join(current_lines) + "\n"))
    return sections, found_named_section


def select_text_for_task(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    source_computing_task: str,
    *,
    require_named_section: bool = False,
    section_label: str = "input",
) -> str:
    sections, found_named_section = split_named_sections(text, patterns)
    if not found_named_section:
        if require_named_section and source_computing_task:
            raise ResolveError(
                f"{section_label} does not identify required source-computing-task {source_computing_task}"
            )
        return text
    return "".join(section.text for section in sections if task_matches(section.name, source_computing_task))


def parse_source_rows(module: Any, text: str, allow_empty: bool = False) -> list[Any]:
    try:
        return module.parse_line_table_rows(text)
    except Exception as exc:
        line_table_error = getattr(module, "LineTableError", None)
        if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
            if allow_empty and str(exc) == "no source rows found":
                return []
            raise ResolveError(str(exc)) from None
        raise


def load_source_rows(dwarf_line_dump: pathlib.Path, source_computing_task: str) -> list[Any]:
    require_existing_file(dwarf_line_dump, "DWARF line dump")
    module = load_module("parse_sycl_zebin_line_table", "parse-sycl-zebin-line-table.py")
    text = dwarf_line_dump.read_text(encoding="utf-8", errors="replace")
    selected_text = select_text_for_task(text, DWARF_KERNEL_MARKER_PATTERNS, source_computing_task)
    if not selected_text.strip():
        return []
    rows = parse_source_rows(module, selected_text, allow_empty=selected_text != text)
    return sorted(rows, key=lambda row: parse_hex_address(row.address))


def load_instructions(asm_path: pathlib.Path, source_computing_task: str) -> list[Any]:
    require_existing_file(asm_path, "ASM")
    module = load_module("parse_sycl_vtune_kernel_asm", "parse-sycl-vtune-kernel-asm.py")
    text = asm_path.read_text(encoding="utf-8", errors="replace")
    selected_text = select_text_for_task(
        text,
        ASM_KERNEL_MARKER_PATTERNS,
        source_computing_task,
        require_named_section=True,
        section_label="ASM",
    )
    if not selected_text.strip():
        return []
    return sorted(module.parse_asm_instructions_text(selected_text), key=lambda row: row.address)


def load_iga_instructions(csv_path: pathlib.Path, source_computing_task: str, pc_base: int) -> list[PcInstructionRow]:
    require_existing_file(csv_path, "IGA PC instruction CSV")
    rows: list[PcInstructionRow] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_kernel = row.get("kernel") or ""
            if row_kernel != source_computing_task:
                raise ResolveError(
                    f"IGA PC instruction CSV contains kernel {row_kernel} but expected {source_computing_task}"
                )
            pc_raw = row.get("pc") or ""
            if not pc_raw.isdigit():
                continue
            rows.append(
                PcInstructionRow(
                    address=int(pc_raw) + pc_base,
                    opcode=(row.get("opcode") or "").lower(),
                    text=row.get("text") or "",
                    raw=row.get("raw") or "",
                    send_comment=row.get("send_comment") or "",
                )
            )
    return sorted(rows, key=lambda row: row.address)


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
    return source_rows[index]


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
        handle.flush()


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
            "Instruction addresses are matched to nearest-preceding line-table ranges: each row covers "
            "addresses from its address up to the next row address, and the final row covers later "
            "instructions until another row is present."
        )
    )
    parser.add_argument("--dwarf-line-dump", required=True, type=pathlib.Path)
    parser.add_argument("--asm", type=pathlib.Path)
    parser.add_argument("--iga-instructions-csv", type=pathlib.Path)
    parser.add_argument("--pc-base", default="0", help="base address added to IGA section-relative PCs")
    parser.add_argument("--output", type=pathlib.Path, help="CSV output path; defaults to stdout")
    parser.add_argument("--summary-output", type=pathlib.Path, help="optional parse-style summary output")
    parser.add_argument("--source-computing-task", required=True, help="required kernel/task name to write into output rows")
    parser.add_argument("--require-source-path", default="", help="optional source path suffix that mapped rows must match")
    args = parser.parse_args(argv)

    try:
        if bool(args.asm) == bool(args.iga_instructions_csv):
            raise ResolveError("pass exactly one of --asm or --iga-instructions-csv")
        source_rows = load_source_rows(args.dwarf_line_dump, args.source_computing_task)
        instructions = (
            load_instructions(args.asm, args.source_computing_task)
            if args.asm
            else load_iga_instructions(
                args.iga_instructions_csv,
                args.source_computing_task,
                parse_hex_address(args.pc_base),
            )
        )
        rows, mapped, unmapped = aggregate_rows(source_rows, instructions, args.require_source_path)
        write_csv(rows, args.output, args.source_computing_task)
        write_summary(rows, args.summary_output, mapped, unmapped)
        if mapped == 0:
            print(
                f"failed to resolve ZEBin ASM source lines: no mapped ASM source rows ({NO_MATCH_BLOCKER})",
                file=sys.stderr,
            )
            return 2
    except (OSError, UnicodeDecodeError, ResolveError) as exc:
        print(f"failed to resolve ZEBin ASM source lines: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
