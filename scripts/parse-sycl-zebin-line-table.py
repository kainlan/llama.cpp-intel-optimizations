#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path


INCLUDE_RE = re.compile(r"^\s*include_directories\[\s*(\d+)\s*\]\s*=\s*(.*?)\s*$")
FILE_RE = re.compile(r"^\s*file_names\[\s*(\d+)\s*\]:\s*$")
NAME_RE = re.compile(r"^\s*name:\s*(.*?)\s*$")
DIR_INDEX_RE = re.compile(r"^\s*dir_index:\s*(\d+)\s*$")
ROW_RE = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+")
TABLE_START_RE = re.compile(r"^\s*(?:\.debug_line contents:|debug_line\[[^\]]+\])\s*$")


class LineTableError(Exception):
    pass


@dataclass
class FileEntry:
    name: str = ""
    dir_index: int = 0


@dataclass(frozen=True)
class SourceRow:
    address: str
    line: int
    column: int
    file_id: int
    file_path: str


def unquote_llvm_string(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in {'"', "'"} and stripped[-1] == stripped[0]:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped[1:-1]
        if isinstance(parsed, str):
            return parsed
    return stripped


def normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path.strip())
    return "" if normalized == "." else normalized


def resolve_file_path(entry: FileEntry, include_directories: dict[int, str]) -> str:
    name = entry.name.strip()
    if not name:
        raise LineTableError("file entry is missing name")
    if name.startswith("/"):
        return normalize_path(name)
    directory = include_directories.get(entry.dir_index, "")
    if directory:
        return normalize_path(posixpath.join(directory, name))
    return normalize_path(name)


def path_matches(path: str, required_path: str) -> bool:
    required = normalize_path(required_path)
    candidate = normalize_path(path)
    if not required:
        return False
    if required.startswith("/"):
        return candidate == required
    return candidate == required or candidate.endswith("/" + required)


def parse_line_table_rows(text: str) -> list[SourceRow]:
    include_directories: dict[int, str] = {}
    file_entries: dict[int, FileEntry] = {}
    current_file_index: int | None = None
    rows: list[SourceRow] = []

    for line in text.splitlines():
        if TABLE_START_RE.match(line):
            include_directories = {}
            file_entries = {}
            current_file_index = None
            continue

        include_match = INCLUDE_RE.match(line)
        if include_match:
            include_directories[int(include_match.group(1))] = normalize_path(unquote_llvm_string(include_match.group(2)))
            current_file_index = None
            continue

        file_match = FILE_RE.match(line)
        if file_match:
            current_file_index = int(file_match.group(1))
            file_entries[current_file_index] = FileEntry()
            continue

        if current_file_index is not None:
            name_match = NAME_RE.match(line)
            if name_match:
                file_entries[current_file_index].name = unquote_llvm_string(name_match.group(1))
                continue
            dir_index_match = DIR_INDEX_RE.match(line)
            if dir_index_match:
                file_entries[current_file_index].dir_index = int(dir_index_match.group(1))
                continue

        row_match = ROW_RE.match(line)
        if row_match:
            address = row_match.group(1)
            line_number = int(row_match.group(2))
            column_number = int(row_match.group(3))
            file_id = int(row_match.group(4))
            if line_number > 0:
                entry = file_entries.get(file_id)
                if entry is None:
                    raise LineTableError(f"source row references unknown file id {file_id}")
                rows.append(
                    SourceRow(
                        address=address,
                        line=line_number,
                        column=column_number,
                        file_id=file_id,
                        file_path=resolve_file_path(entry, include_directories),
                    )
                )
            current_file_index = None

    if not rows:
        raise LineTableError("no source rows found")

    return rows


def parse_line_table(text: str, require_path: str | None = None) -> dict[str, object]:
    rows = parse_line_table_rows(text)
    files = sorted({row.file_path for row in rows})
    if not files:
        raise LineTableError("no files found")

    required_path = require_path or ""
    required_path_present = any(path_matches(path, required_path) for path in files) if required_path else False

    return {
        "file_count": len(files),
        "files": files,
        "required_path": required_path,
        "required_path_present": required_path_present,
        "source_rows": len(rows),
        "status": "ok",
    }


def read_text(input_path: str) -> str:
    if input_path == "-":
        return sys.stdin.read()
    return Path(input_path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse decoded llvm-dwarfdump --debug-line text from a SYCL ZEBin.")
    parser.add_argument("--input", default="-", help="decoded debug-line text file, or '-' for stdin")
    parser.add_argument("--require-path", default=None, help="source path that must appear in parsed line-table rows")
    args = parser.parse_args(argv)

    try:
        result = parse_line_table(read_text(args.input), args.require_path)
    except (LineTableError, OSError, UnicodeDecodeError) as exc:
        print(f"failed to parse ZEBin line table: {exc}")
        return 2

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
