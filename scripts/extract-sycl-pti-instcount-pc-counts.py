#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from typing import Any, TextIO

CSV_FIELDS = ["kernel", "pc", "sample_count", "sample_kind"]
SAMPLE_KIND = "pti-instcount-instruction-exec-count"


class InstcountError(ValueError):
    pass


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise InstcountError(f"invalid PTI instcount JSON: {exc}") from exc


def kernel_matches(name: str, required: str) -> bool:
    return not required or required in name


def require_list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise InstcountError(f"{description} must be a list")
    return value


def require_dict(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InstcountError(f"{description} must be an object")
    return value


def parse_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InstcountError(f"{description} must be an integer")
    return value


def aggregate_kernel(kernel_obj: dict[str, Any]) -> Counter[int]:
    totals: Counter[int] = Counter()
    invocations = require_list(kernel_obj.get("invocations", []), "kernel invocations")
    for inv_index, inv_raw in enumerate(invocations):
        inv = require_dict(inv_raw, f"invocations[{inv_index}]")
        tiles = require_list(inv.get("tiles", []), f"invocations[{inv_index}].tiles")
        for tile_index, tile_raw in enumerate(tiles):
            tile = require_dict(tile_raw, f"invocations[{inv_index}].tiles[{tile_index}]")
            results = require_list(tile.get("results", []), f"invocations[{inv_index}].tiles[{tile_index}].results")
            for result_index, result_raw in enumerate(results):
                result = require_dict(
                    result_raw,
                    f"invocations[{inv_index}].tiles[{tile_index}].results[{result_index}]",
                )
                offset = parse_int(result.get("offset"), "offset")
                count = parse_int(result.get("instruction_counter"), "instruction_counter")
                if count > 0:
                    totals[offset] += count
    return totals


def write_csv(rows: list[dict[str, str]], output: pathlib.Path | None) -> None:
    handle: TextIO
    if output is None:
        handle = sys.stdout
        close = False
    else:
        handle = output.open("w", encoding="utf-8", newline="")
        close = True
    try:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close:
            handle.close()


def write_summary(
    summary_output: pathlib.Path | None,
    matched_kernel_count: int,
    output_kernel: str,
    rows: list[dict[str, str]],
) -> None:
    if summary_output is None:
        return
    total = sum(int(row["sample_count"]) for row in rows)
    lines = [
        f"pti_instcount.status {'ok' if rows else 'no_matching_kernel'}",
        f"pti_instcount.matched_kernel_count {matched_kernel_count}",
        f"pti_instcount.output_kernel {output_kernel}",
        f"pti_instcount.pc_rows {len(rows)}",
        f"pti_instcount.total_instruction_count {total}",
        f"pti_instcount.sample_kind {SAMPLE_KIND}",
    ]
    if rows:
        top = max(rows, key=lambda row: int(row["sample_count"]))
        lines.extend(
            [
                f"pti_instcount.top_pc {top['pc']}",
                f"pti_instcount.top_sample_count {top['sample_count']}",
            ]
        )
    else:
        lines.append("pti_instcount.blocker no_matching_kernel")
    summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PTI instcount dynamic instruction execution counts into the "
            "kernel,pc,sample_count,sample_kind CSV schema used by the SYCL source-line resolver. "
            "This is runtime instruction counting, not sampled PC evidence."
        )
    )
    parser.add_argument("--instcount-json", required=True, type=pathlib.Path)
    parser.add_argument("--kernel-match", required=True, help="substring that identifies the PTI kernel_name to extract")
    parser.add_argument(
        "--source-computing-task",
        required=True,
        help="kernel/task name to write in the output CSV for downstream resolver matching",
    )
    parser.add_argument("--output", type=pathlib.Path, help="CSV output path; defaults to stdout")
    parser.add_argument("--summary-output", type=pathlib.Path, help="optional parse-style summary output")
    args = parser.parse_args(argv)

    try:
        data = load_json(args.instcount_json)
        root = require_dict(data, "PTI instcount root")
        kernels = require_list(root.get("kernels"), "kernels")
        totals: Counter[int] = Counter()
        matched = 0
        for kernel_index, kernel_raw in enumerate(kernels):
            kernel_obj = require_dict(kernel_raw, f"kernels[{kernel_index}]")
            kernel_name = str(kernel_obj.get("kernel_name", ""))
            if not kernel_matches(kernel_name, args.kernel_match):
                continue
            matched += 1
            totals.update(aggregate_kernel(kernel_obj))
        rows = [
            {
                "kernel": args.source_computing_task,
                "pc": hex(offset),
                "sample_count": str(count),
                "sample_kind": SAMPLE_KIND,
            }
            for offset, count in sorted(totals.items())
            if count > 0
        ]
        write_csv(rows, args.output)
        write_summary(args.summary_output, matched, args.source_computing_task, rows)
        if not rows:
            print("failed to extract PTI instcount rows: no matching positive instruction counts", file=sys.stderr)
            return 2
    except (OSError, UnicodeDecodeError, InstcountError) as exc:
        print(f"failed to extract PTI instcount rows: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
