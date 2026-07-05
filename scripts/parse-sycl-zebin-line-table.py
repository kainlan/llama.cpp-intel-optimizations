#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
ROW_RE = re.compile(r"^\s*0x[0-9a-fA-F]+\s+(\d+)\s+(\d+)\s+(\d+)\s+")


class LineTableError(Exception):
    pass


@dataclass
class FileEntry:
    name: str = ""
    dir_index: int = 0


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


def parse_line_table(text: str, require_path: str | None = None) -> dict[str, object]:
    include_directories: dict[int, str] = {}
    file_entries: dict[int, FileEntry] = {}
    current_file_index: int | None = None
    source_row_file_ids: list[int] = []

    for line in text.splitlines():
        include_match = INCLUDE_RE.match(line)
        if include_match:
            include_directories[int(include_match.group(1))] = normalize_path(include_match.group(2))
            current_file_index = None
            continue

        file_match = FILE_RE.match(line)
        if file_match:
            current_file_index = int(file_match.group(1))
            file_entries.setdefault(current_file_index, FileEntry())
            continue

        if current_file_index is not None:
            name_match = NAME_RE.match(line)
            if name_match:
                file_entries[current_file_index].name = name_match.group(1).strip()
                continue
            dir_index_match = DIR_INDEX_RE.match(line)
            if dir_index_match:
                file_entries[current_file_index].dir_index = int(dir_index_match.group(1))
                continue

        row_match = ROW_RE.match(line)
        if row_match:
            line_number = int(row_match.group(1))
            file_id = int(row_match.group(3))
            if line_number > 0:
                source_row_file_ids.append(file_id)
            current_file_index = None

    if not source_row_file_ids:
        raise LineTableError("no source rows found")

    row_files: set[str] = set()
    for file_id in source_row_file_ids:
        entry = file_entries.get(file_id)
        if entry is None:
            raise LineTableError(f"source row references unknown file id {file_id}")
        row_files.add(resolve_file_path(entry, include_directories))

    files = sorted(row_files)
    if not files:
        raise LineTableError("no files found")

    required_path = require_path or ""
    required_path_present = any(path_matches(path, required_path) for path in files) if required_path else False

    return {
        "file_count": len(files),
        "files": files,
        "required_path": required_path,
        "required_path_present": required_path_present,
        "source_rows": len(source_row_file_ids),
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
