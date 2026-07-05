#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from dataclasses import dataclass

FIELDS = ["kernel", "pc", "pc_hex", "opcode", "text", "raw", "send_comment", "source"]
TEXT_PC_PATTERNS = (
    re.compile(r"^\s*(?:0x)?([0-9A-Fa-f]+):\s*(.+)$"),
    re.compile(r"^\s*PC\s*=\s*(?:0x)?([0-9A-Fa-f]+)\s+(.+)$", re.IGNORECASE),
)


@dataclass(frozen=True)
class PcInstruction:
    kernel: str
    pc: int
    opcode: str
    text: str
    raw: str
    send_comment: str
    source: str

    def to_row(self) -> dict[str, str]:
        return {
            "kernel": self.kernel,
            "pc": str(self.pc),
            "pc_hex": hex(self.pc),
            "opcode": self.opcode,
            "text": self.text,
            "raw": self.raw,
            "send_comment": self.send_comment,
            "source": self.source,
        }


def opcode_from_text(text: str) -> str:
    cleaned = re.sub(r"^\([^)]*\)\s*", "", text.strip())
    match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)", cleaned)
    return match.group(1).lower() if match else "unknown"


def send_comment(raw: str) -> str:
    if "//" not in raw:
        return ""
    comment = raw.split("//", 1)[1].strip()
    return comment if comment.startswith("wr:") else ""


def parse_pc(raw: object) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("0x"):
            return int(text, 16)
        if text.isdigit():
            return int(text, 10)
    return None


def parse_json_rows(text: str, kernel: str) -> list[PcInstruction]:
    data = json.loads(text)
    kernels = data.get("kernels") if isinstance(data, dict) else None
    if kernels is None and isinstance(data, dict):
        kernels = [data]
    rows: list[PcInstruction] = []
    for item in kernels or []:
        name = str(item.get("name") or item.get("kernel") or kernel)
        if name != kernel:
            continue
        for elem in item.get("elems", []):
            if elem.get("kind") not in {"inst", "instruction"}:
                continue
            pc = parse_pc(elem.get("pc"))
            if pc is None:
                continue
            text_row = str(elem.get("syntax") or elem.get("text") or elem.get("op") or "")
            rows.append(
                PcInstruction(
                    kernel=kernel,
                    pc=pc,
                    opcode=str(elem.get("op") or opcode_from_text(text_row)).lower(),
                    text=text_row,
                    raw=json.dumps(elem, sort_keys=True),
                    send_comment=send_comment(text_row),
                    source="iga-json",
                )
            )
    return sorted(rows, key=lambda row: row.pc)


def parse_text_rows(text: str, kernel: str) -> list[PcInstruction]:
    rows: list[PcInstruction] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.endswith(":") or stripped.startswith("//"):
            continue
        for pattern in TEXT_PC_PATTERNS:
            match = pattern.match(raw)
            if not match:
                continue
            pc = int(match.group(1), 16)
            inst = match.group(2).strip()
            rows.append(
                PcInstruction(
                    kernel=kernel,
                    pc=pc,
                    opcode=opcode_from_text(inst),
                    text=inst,
                    raw=raw,
                    send_comment=send_comment(raw),
                    source="iga-text",
                )
            )
            break
    return sorted(rows, key=lambda row: row.pc)


def write_csv(rows: list[PcInstruction]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_row())


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse IGA PC disassembly rows")
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--format", choices=("json", "text"), required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8", errors="replace")
    rows = parse_json_rows(text, args.kernel) if args.format == "json" else parse_text_rows(text, args.kernel)
    write_csv(rows)
    if not rows:
        print("iga_pc.status no_pc_rows")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
