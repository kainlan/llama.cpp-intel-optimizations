#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import struct
import sys
from dataclasses import dataclass
from typing import Any, TextIO

PC_FIELDS = [
    "kernel",
    "pc",
    "pc_hex",
    "sample_count",
    "sample_kind",
    "bbl_id",
    "opcode",
    "text",
]

BBL_FIELDS = [
    "kernel",
    "bbl_id",
    "dynamic_count",
    "instruction_count",
    "send_count",
]

DEFAULT_SAMPLE_KIND = "gtpin-bbl-instruction-exec-count"
STATUS_OK = "ok"
STATUS_NO_PC_COUNTS = "no_pc_counts"


class ExtractError(ValueError):
    pass


@dataclass(frozen=True)
class SendDescriptor:
    offset: int
    payload_len: int
    addr_width: int


@dataclass
class TraceData:
    descriptor_u32_count: int
    register_size_bits: int
    timestamp_included: int
    num_tiles: int
    bbl_sends: dict[int, list[SendDescriptor]]
    bbl_counts: collections.Counter[int]
    send_counts: collections.Counter[int]
    dynamic_bbl_records: int
    profiled_thread_count: int


@dataclass(frozen=True)
class Instruction:
    pc: int
    opcode: str
    text: str


def require_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise ExtractError(f"{label} file does not exist: {path}")


def send_descriptor_from_words(words: list[int]) -> SendDescriptor:
    if len(words) == 20:
        # VTune/GTPin 2025.10 memorytrace.so writes one extra u32 between
        # isFence and addrWidth compared with the older MAAT.py decoder.
        return SendDescriptor(offset=words[0], payload_len=words[12], addr_width=words[9])
    if len(words) == 19:
        # Older MAAT.py ABI.
        return SendDescriptor(offset=words[0], payload_len=words[11], addr_width=words[8])
    raise ExtractError(f"unsupported send descriptor length: {len(words)}")


def parse_trace_with_descriptor_words(data: bytes, descriptor_u32_count: int) -> TraceData:
    pos = 0
    size = len(data)

    def u32() -> int:
        nonlocal pos
        if pos + 4 > size:
            raise ExtractError(f"unexpected EOF reading u32 at byte {pos} of {size}")
        value = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return value

    def skip_bytes(count: int, label: str) -> None:
        nonlocal pos
        if count < 0:
            raise ExtractError(f"negative skip while reading {label}: {count}")
        if pos + count > size:
            raise ExtractError(f"unexpected EOF reading {label} at byte {pos} of {size}")
        pos += count

    bbl_header_count = u32()
    if bbl_header_count <= 0 or bbl_header_count > 1_000_000:
        raise ExtractError(f"implausible BBL header count: {bbl_header_count}")

    bbl_sends: dict[int, list[SendDescriptor]] = {}
    for _ in range(bbl_header_count):
        bbl_id = u32()
        send_count = u32()
        if send_count > 1_000_000:
            raise ExtractError(f"implausible send count for BBL {bbl_id}: {send_count}")
        sends: list[SendDescriptor] = []
        for _send_index in range(send_count):
            words = [u32() for _ in range(descriptor_u32_count)]
            send = send_descriptor_from_words(words)
            if send.addr_width not in (32, 64):
                raise ExtractError(f"implausible address width for BBL {bbl_id}: {send.addr_width}")
            sends.append(send)
        bbl_sends[bbl_id] = sends

    register_size_u32 = u32()
    register_size_bits = register_size_u32 * 8
    if register_size_bits not in (256, 512, 1024):
        raise ExtractError(f"implausible register size bits: {register_size_bits}")

    timestamp_included = u32()
    if timestamp_included not in (0, 1):
        raise ExtractError(f"implausible timestamp flag: {timestamp_included}")

    num_tiles = u32()
    if num_tiles <= 0 or num_tiles > 4096:
        raise ExtractError(f"implausible tile count: {num_tiles}")

    bbl_counts: collections.Counter[int] = collections.Counter()
    send_counts: collections.Counter[int] = collections.Counter()
    dynamic_bbl_records = 0
    profiled_thread_count = 0

    for _tile_index in range(num_tiles):
        _tile_id = u32()
        num_threads = u32()
        if num_threads > 10_000_000:
            raise ExtractError(f"implausible profiled thread count: {num_threads}")
        for _thread_index in range(num_threads):
            # SliceID, DualSubSliceID, SubSliceID, EuId, hardware-thread Id.
            for _ in range(5):
                u32()
            profiled_thread_count += 1
            num_records = u32()
            if num_records > 1_000_000_000:
                raise ExtractError(f"implausible record count: {num_records}")
            for _record_index in range(num_records):
                bbl_id = u32()
                _exec_mask = u32()
                if timestamp_included:
                    skip_bytes(8, "record timestamp")
                bbl_counts[bbl_id] += 1
                dynamic_bbl_records += 1
                for send_index, send in enumerate(bbl_sends.get(bbl_id, [])):
                    if timestamp_included and send_index != 0:
                        # MAAT.py's current timestamp path skips the saved TID
                        # tuple, timestamp, and remaining GRF spill before each
                        # additional send in the same BBL.
                        skip_bytes(4 * 5, "per-send tid")
                        skip_bytes(8, "per-send timestamp")
                        skip_bytes(max(0, register_size_bits // 8 - 4 * 7), "per-send register spill")
                    if send.payload_len > 0:
                        address_count = send.payload_len * register_size_bits // send.addr_width
                        address_bytes = address_count * (4 if send.addr_width == 32 else 8)
                        skip_bytes(address_bytes, "send address payload")
                    send_counts[send.offset] += 1

    if pos != size:
        raise ExtractError(f"trace parser stopped at byte {pos} but file has {size} bytes")

    return TraceData(
        descriptor_u32_count=descriptor_u32_count,
        register_size_bits=register_size_bits,
        timestamp_included=timestamp_included,
        num_tiles=num_tiles,
        bbl_sends=bbl_sends,
        bbl_counts=bbl_counts,
        send_counts=send_counts,
        dynamic_bbl_records=dynamic_bbl_records,
        profiled_thread_count=profiled_thread_count,
    )


def parse_trace(path: pathlib.Path, descriptor_u32_count: str) -> TraceData:
    require_file(path, "GTPin memorytrace")
    data = path.read_bytes()
    candidates = [20, 19] if descriptor_u32_count == "auto" else [int(descriptor_u32_count)]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return parse_trace_with_descriptor_words(data, candidate)
        except ExtractError as exc:
            errors.append(f"{candidate} u32 descriptor: {exc}")
    raise ExtractError("failed to parse GTPin memorytrace (" + "; ".join(errors) + ")")


def parse_session_bbl_asm(path: pathlib.Path) -> dict[int, list[int]]:
    require_file(path, "GTPin Session_Final ASM")
    bbl_to_ids: dict[int, list[int]] = {}
    current_bbl: int | None = None
    bbl_re = re.compile(r"^//\s*BBL(\d+)\b")
    id_re = re.compile(r"^\[\s*(\d+)\]")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        bbl_match = bbl_re.match(line.strip())
        if bbl_match:
            current_bbl = int(bbl_match.group(1))
            bbl_to_ids.setdefault(current_bbl, [])
            continue
        id_match = id_re.match(line)
        if id_match and current_bbl is not None:
            bbl_to_ids[current_bbl].append(int(id_match.group(1)))
    if not bbl_to_ids:
        raise ExtractError(f"no BBL markers found in Session_Final ASM: {path}")
    return bbl_to_ids


def opcode_from_text(text: str) -> str:
    parts = text.strip().split()
    if not parts:
        return ""
    if parts[0].startswith("(") and len(parts) > 1:
        return parts[1]
    return parts[0]


def select_kernel_report(obj: dict[str, Any], kernel_match: str) -> dict[str, Any]:
    kernels = obj.get("kernels")
    if not isinstance(kernels, dict):
        raise ExtractError("app report JSON does not contain a kernels object")
    matches: list[dict[str, Any]] = []
    for key, value in kernels.items():
        if not isinstance(value, dict):
            continue
        name = str(value.get("name", key))
        if kernel_match in key or kernel_match in name:
            matches.append(value)
    if not matches:
        raise ExtractError(f"no app report kernel matched: {kernel_match}")
    if len(matches) > 1:
        # Prefer the non-synthetic per-kernel entry. MAAT also emits All_kernels.
        non_all = [item for item in matches if str(item.get("name", "")) != "All_kernels"]
        matches = non_all or matches
    return matches[0]


def load_app_report_instructions(path: pathlib.Path, kernel_match: str) -> dict[int, Instruction]:
    require_file(path, "MAAT app report JSON")
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ExtractError(f"failed to parse app report JSON: {exc}") from exc
    kernel = select_kernel_report(obj, kernel_match)
    asm = kernel.get("asm")
    if not isinstance(asm, list):
        raise ExtractError(f"matched app report kernel has no asm list: {kernel_match}")
    instructions: dict[int, Instruction] = {}
    for entry in asm:
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        inst_id, pc, text = entry[:3]
        if not isinstance(inst_id, int) or not isinstance(pc, int) or not isinstance(text, str):
            continue
        if inst_id < 0 or pc < 0:
            continue
        instructions[inst_id] = Instruction(pc=pc, opcode=opcode_from_text(text), text=text.strip())
    if not instructions:
        raise ExtractError(f"no instructions found in app report kernel: {kernel_match}")
    return instructions


def choose_id_shift(
    trace: TraceData,
    bbl_to_ids: dict[int, list[int]],
    instructions: dict[int, Instruction],
) -> tuple[int, int]:
    best_shift = 0
    best_matches = -1
    send_offsets_by_bbl = {bbl: {send.offset for send in sends} for bbl, sends in trace.bbl_sends.items()}
    for shift in range(-4, 5):
        matches = 0
        for bbl, send_offsets in send_offsets_by_bbl.items():
            if not send_offsets:
                continue
            for session_id in bbl_to_ids.get(bbl, []):
                inst = instructions.get(session_id + shift)
                if inst is not None and inst.pc in send_offsets:
                    matches += 1
        if matches > best_matches:
            best_shift = shift
            best_matches = matches
    return best_shift, max(best_matches, 0)


def pc_counts_from_bbl_counts(
    trace: TraceData,
    bbl_to_ids: dict[int, list[int]],
    instructions: dict[int, Instruction],
    id_shift: int,
) -> tuple[collections.Counter[int], dict[int, tuple[int, Instruction]], int, int]:
    pc_counts: collections.Counter[int] = collections.Counter()
    pc_meta: dict[int, tuple[int, Instruction]] = {}
    missing_bbl_record_count = 0
    missing_instruction_count = 0
    for bbl_id, dynamic_count in trace.bbl_counts.items():
        session_ids = bbl_to_ids.get(bbl_id)
        if not session_ids:
            missing_bbl_record_count += dynamic_count
            continue
        for session_id in session_ids:
            inst = instructions.get(session_id + id_shift)
            if inst is None:
                missing_instruction_count += dynamic_count
                continue
            pc_counts[inst.pc] += dynamic_count
            pc_meta.setdefault(inst.pc, (bbl_id, inst))
    return pc_counts, pc_meta, missing_bbl_record_count, missing_instruction_count


def open_csv_output(path: pathlib.Path | None) -> TextIO:
    if path is None:
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def write_pc_counts(
    rows: collections.Counter[int],
    pc_meta: dict[int, tuple[int, Instruction]],
    output_path: pathlib.Path | None,
    source_computing_task: str,
    sample_kind: str,
) -> None:
    handle = open_csv_output(output_path)
    should_close = handle is not sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=PC_FIELDS)
        writer.writeheader()
        for pc, count in sorted(rows.items(), key=lambda item: (-item[1], item[0])):
            bbl_id, inst = pc_meta[pc]
            writer.writerow(
                {
                    "kernel": source_computing_task,
                    "pc": str(pc),
                    "pc_hex": hex(pc),
                    "sample_count": str(count),
                    "sample_kind": sample_kind,
                    "bbl_id": str(bbl_id),
                    "opcode": inst.opcode,
                    "text": inst.text,
                }
            )
        handle.flush()
    finally:
        if should_close:
            handle.close()


def write_bbl_counts(
    trace: TraceData,
    bbl_to_ids: dict[int, list[int]],
    output_path: pathlib.Path | None,
    source_computing_task: str,
) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BBL_FIELDS)
        writer.writeheader()
        for bbl_id, count in sorted(trace.bbl_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow(
                {
                    "kernel": source_computing_task,
                    "bbl_id": str(bbl_id),
                    "dynamic_count": str(count),
                    "instruction_count": str(len(bbl_to_ids.get(bbl_id, []))),
                    "send_count": str(len(trace.bbl_sends.get(bbl_id, []))),
                }
            )


def write_summary(
    output_path: pathlib.Path | None,
    trace: TraceData,
    pc_counts: collections.Counter[int],
    id_shift: int,
    id_shift_send_matches: int,
    missing_bbl_record_count: int,
    missing_instruction_count: int,
    sample_kind: str,
) -> None:
    if output_path is None:
        return
    dynamic_instruction_count = sum(pc_counts.values())
    lines = [
        f"gtpin_bbl_pc.status {STATUS_OK if dynamic_instruction_count else STATUS_NO_PC_COUNTS}",
        f"gtpin_bbl_pc.trace_send_descriptor_u32_count {trace.descriptor_u32_count}",
        f"gtpin_bbl_pc.register_size_bits {trace.register_size_bits}",
        f"gtpin_bbl_pc.timestamp_included {trace.timestamp_included}",
        f"gtpin_bbl_pc.tile_count {trace.num_tiles}",
        f"gtpin_bbl_pc.profiled_thread_count {trace.profiled_thread_count}",
        f"gtpin_bbl_pc.bbl_header_count {len(trace.bbl_sends)}",
        f"gtpin_bbl_pc.dynamic_bbl_records {trace.dynamic_bbl_records}",
        f"gtpin_bbl_pc.dynamic_instruction_count {dynamic_instruction_count}",
        f"gtpin_bbl_pc.pc_rows {len(pc_counts)}",
        f"gtpin_bbl_pc.send_pc_rows {len(trace.send_counts)}",
        f"gtpin_bbl_pc.id_shift {id_shift}",
        f"gtpin_bbl_pc.id_shift_send_matches {id_shift_send_matches}",
        f"gtpin_bbl_pc.missing_bbl_record_count {missing_bbl_record_count}",
        f"gtpin_bbl_pc.missing_instruction_count {missing_instruction_count}",
        f"gtpin_bbl_pc.sample_kind {sample_kind}",
    ]
    if pc_counts:
        top_pc, top_count = max(pc_counts.items(), key=lambda item: (item[1], -item[0]))
        lines.extend(
            [
                f"gtpin_bbl_pc.top_pc {hex(top_pc)}",
                f"gtpin_bbl_pc.top_count {top_count}",
            ]
        )
    else:
        lines.append(f"gtpin_bbl_pc.blocker {STATUS_NO_PC_COUNTS}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract positive runtime PC count rows from a GTPin memorytrace.so BBL trace. "
            "The output uses the pc-samples.csv schema for downstream source-line resolvers, "
            "but sample_kind identifies the data as GTPin BBL instruction execution counts, not VTune samples."
        )
    )
    parser.add_argument("--trace", required=True, type=pathlib.Path, help="memorytrace_compressed.bin from GTPin memorytrace.so")
    parser.add_argument("--session-asm", required=True, type=pathlib.Path, help="GTPin Session_Final ISA .asm with // BBL markers")
    parser.add_argument("--app-report-json", required=True, type=pathlib.Path, help="MAAT app.report.json with instruction IDs and PC offsets")
    parser.add_argument("--kernel-match", required=True, help="substring identifying the mangled kernel in app.report.json")
    parser.add_argument("--source-computing-task", required=True, help="friendly kernel/task name to write into output rows")
    parser.add_argument("--output", type=pathlib.Path, help="pc-count CSV output path; defaults to stdout")
    parser.add_argument("--bbl-output", type=pathlib.Path, help="optional BBL dynamic-count CSV output")
    parser.add_argument("--summary-output", type=pathlib.Path, help="optional parse-style summary output")
    parser.add_argument("--sample-kind", default=DEFAULT_SAMPLE_KIND)
    parser.add_argument("--send-descriptor-u32-count", default="auto", choices=("auto", "19", "20"))
    args = parser.parse_args(argv)

    try:
        trace = parse_trace(args.trace, args.send_descriptor_u32_count)
        bbl_to_ids = parse_session_bbl_asm(args.session_asm)
        instructions = load_app_report_instructions(args.app_report_json, args.kernel_match)
        id_shift, shift_matches = choose_id_shift(trace, bbl_to_ids, instructions)
        pc_counts, pc_meta, missing_bbl_records, missing_instructions = pc_counts_from_bbl_counts(
            trace, bbl_to_ids, instructions, id_shift
        )
        write_pc_counts(pc_counts, pc_meta, args.output, args.source_computing_task, args.sample_kind)
        write_bbl_counts(trace, bbl_to_ids, args.bbl_output, args.source_computing_task)
        write_summary(
            args.summary_output,
            trace,
            pc_counts,
            id_shift,
            shift_matches,
            missing_bbl_records,
            missing_instructions,
            args.sample_kind,
        )
        if not pc_counts:
            print("failed to extract GTPin BBL PC counts: no PC counts", file=sys.stderr)
            return 2
    except (OSError, UnicodeDecodeError, ExtractError) as exc:
        print(f"failed to extract GTPin BBL PC counts: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
