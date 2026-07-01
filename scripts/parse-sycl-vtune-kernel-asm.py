#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
from typing import Any


def parse_asm(path: pathlib.Path) -> dict[str, Any]:
    opcodes: collections.Counter[str] = collections.Counter()
    send_comments: collections.Counter[str] = collections.Counter()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.endswith(":"):
            continue
        line = re.sub(r"^\([^)]*\)\s*", "", line)
        match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)", line)
        if match:
            opcodes[match.group(1).lower()] += 1
        if "//" in raw:
            comment = raw.split("//", 1)[1].strip()
            if comment.startswith("wr:"):
                send_comments[comment] += 1
    return {
        "path": str(path),
        "opcodes": dict(sorted(opcodes.items())),
        "send_comments": dict(sorted(send_comments.items())),
    }


def parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_instr_tsv(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            task = row.get("Computing Task") or row.get("Task") or ""
            if not task:
                continue
            rows.append(
                {
                    "task": task,
                    "gpu_instructions": parse_int(row.get("GPU Instructions Executed")),
                    "simd_width": parse_int(row.get("Computing Task:SIMD Width")),
                    "spill_memory_size": parse_int(row.get("Computing Task:Spill Memory Size")),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize VTune/ocloc SYCL kernel evidence")
    parser.add_argument("--asm", action="append", default=[], help="ocloc/IGA assembly file to summarize")
    parser.add_argument("--instr-tsv", action="append", default=[], help="VTune instruction-count TSV report")
    args = parser.parse_args()

    output: dict[str, Any] = {}
    if args.asm:
        summaries = [parse_asm(pathlib.Path(path)) for path in args.asm]
        output["asm"] = summaries[0] if len(summaries) == 1 else summaries
    if args.instr_tsv:
        rows: list[dict[str, Any]] = []
        for path in args.instr_tsv:
            rows.extend(parse_instr_tsv(pathlib.Path(path)))
        output["instr_tsv"] = rows
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
