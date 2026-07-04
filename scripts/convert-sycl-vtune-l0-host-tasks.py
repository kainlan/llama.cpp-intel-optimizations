#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROW_RE = re.compile(r"^\s*(ze[A-Za-z0-9_]+)\s+([0-9]+(?:\.[0-9]+)?)s\s+")


class ConvertError(ValueError):
    pass


def convert(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW_RE.match(raw)
        if match is None:
            continue
        name = match.group(1)
        seconds = float(match.group(2))
        rows.append(
            {
                "name": name,
                "ts_us": 0,
                "dur_us": int(round(seconds * 1_000_000)),
                "source": "vtune_host_task_summary",
            }
        )
    if not rows:
        raise ConvertError("no Level Zero host task rows found")
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert VTune Level Zero host task summary rows to PTI/L0 JSONL")
    parser.add_argument("summary", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        rows = convert(args.summary)
    except (OSError, ValueError) as exc:
        print(f"failed to convert VTune L0 host tasks: {exc}")
        return 2
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
