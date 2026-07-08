#!/usr/bin/env python3
"""Convert SYCL_UR_TRACE=2 stderr/stdout rows into normalized UR_TRACE rows."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ENTER_RE = re.compile(r"^\s*--->\s+(ur[A-Za-z0-9_]*)\b")
EXIT_RE = re.compile(r"^\s*<---\s+(ur[A-Za-z0-9_]*)\s*\(.*\)\s*->\s*(UR_RESULT_[A-Z_]+);")


class ConvertError(ValueError):
    pass


def convert(path: pathlib.Path) -> list[str]:
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ENTER_RE.match(raw):
            continue
        match = EXIT_RE.match(raw)
        if not match:
            continue
        name, result = match.groups()
        rows.append(f"UR_TRACE name={name} dur_us=0 evidence=counts_only result={result}")
    if not rows:
        raise ConvertError("no UR API rows found")
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert SYCL_UR_TRACE=2 stderr/stdout logs to normalized UR_TRACE rows")
    parser.add_argument("trace", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        rows = convert(args.trace)
    except (OSError, ConvertError) as exc:
        print(f"failed to convert UR stderr: {exc}")
        return 2

    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
