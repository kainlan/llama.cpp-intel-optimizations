#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any


ADDRESS_PATTERNS = (
    re.compile(r"^\s*(?:0x)?([0-9A-Fa-f]+):\s*(.*)$"),
    re.compile(r"^\s*/\*\s*(?:0x)?([0-9A-Fa-f]+)\s*\*/\s*(.*)$"),
    re.compile(r"^\s*\[\s*(?:0x)?([0-9A-Fa-f]+)\s*\]\s*(.*)$"),
)


@dataclass(frozen=True)
class AsmInstruction:
    address: int
    opcode: str
    text: str
    raw: str
    send_comment: str

    def to_json(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "address_hex": hex(self.address),
            "opcode": self.opcode,
            "text": self.text,
            "raw": self.raw,
            "send_comment": self.send_comment,
        }


def split_address(raw: str) -> tuple[int | None, str]:
    for pattern in ADDRESS_PATTERNS:
        match = pattern.match(raw)
        if match:
            return int(match.group(1), 16), match.group(2).strip()
    return None, raw.strip()


def strip_predicate(text: str) -> str:
    return re.sub(r"^\([^)]*\)\s*", "", text.strip())


def parse_opcode(text: str) -> str | None:
    cleaned = strip_predicate(text)
    match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)", cleaned)
    return match.group(1).lower() if match else None


def extract_send_comment(raw: str) -> str:
    if "//" not in raw:
        return ""
    comment = raw.split("//", 1)[1].strip()
    return comment if comment.startswith("wr:") else ""


def parse_asm_instructions_text(text: str) -> list[AsmInstruction]:
    rows: list[AsmInstruction] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("//") or stripped.endswith(":"):
            continue
        address, instruction_text = split_address(raw)
        if address is None:
            continue
        opcode = parse_opcode(instruction_text)
        if opcode is None:
            continue
        rows.append(
            AsmInstruction(
                address=address,
                opcode=opcode,
                text=instruction_text,
                raw=raw.rstrip("\n"),
                send_comment=extract_send_comment(raw),
            )
        )
    return rows


def parse_asm(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    opcodes: collections.Counter[str] = collections.Counter()
    send_comments: collections.Counter[str] = collections.Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.endswith(":"):
            continue
        _, instruction_text = split_address(raw)
        opcode = parse_opcode(instruction_text)
        if opcode:
            opcodes[opcode] += 1
        send_comment = extract_send_comment(raw)
        if send_comment:
            send_comments[send_comment] += 1
    return {
        "path": str(path),
        "opcodes": dict(sorted(opcodes.items())),
        "send_comments": dict(sorted(send_comments.items())),
    }


def parse_asm_instructions(path: pathlib.Path) -> list[AsmInstruction]:
    return parse_asm_instructions_text(path.read_text(encoding="utf-8", errors="replace"))


def classify_bound(opcodes: dict[str, int]) -> dict[str, Any]:
    dpas = int(opcodes.get("dpas.8x8", 0))
    send_ugm = int(opcodes.get("send.ugm", 0))
    math_exp = int(opcodes.get("math.exp", 0))
    math_inv = int(opcodes.get("math.inv", 0))
    if dpas > 0 and send_ugm >= dpas * 16:
        verdict = "memory_or_address_bound"
    elif dpas > 0:
        verdict = "compute_xmx_bound"
    elif send_ugm > 0:
        verdict = "memory_bound"
    elif math_exp + math_inv > 0:
        verdict = "math_bound"
    else:
        verdict = "unknown"
    return {
        "verdict": verdict,
        "dpas": dpas,
        "send_ugm": send_ugm,
        "math_exp": math_exp,
        "math_inv": math_inv,
        "send_per_dpas_x1000": 0 if dpas == 0 else int(round(1000.0 * send_ugm / dpas)),
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
    parser.add_argument("--classify-bound", action="store_true")
    parser.add_argument("--emit-instructions", action="store_true", help="emit addressed assembly instruction rows")
    args = parser.parse_args()

    output: dict[str, Any] = {}
    if args.asm:
        summaries = [parse_asm(pathlib.Path(path)) for path in args.asm]
        if args.classify_bound:
            for summary in summaries:
                summary["bound"] = classify_bound(summary.get("opcodes", {}))
        output["asm"] = summaries[0] if len(summaries) == 1 else summaries
        if args.emit_instructions:
            instruction_rows: list[dict[str, Any]] = []
            for path in args.asm:
                instruction_rows.extend(row.to_json() for row in parse_asm_instructions(pathlib.Path(path)))
            output["instructions"] = instruction_rows
    if args.instr_tsv:
        rows: list[dict[str, Any]] = []
        for path in args.instr_tsv:
            rows.extend(parse_instr_tsv(pathlib.Path(path)))
        output["instr_tsv"] = rows
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
