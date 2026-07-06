#!/usr/bin/env python3
"""Map hottest SYCL kernels to source regions with optional ablation evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


class SourceAttributionError(ValueError):
    """Expected parse failure that should be reported without a traceback."""


def load_parse_rows(path: pathlib.Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, sep, value = stripped.partition(" ")
        if not sep:
            rows[key] = ""
        else:
            rows[key] = value.strip()
    return rows


def parse_top_kernel(rows: dict[str, str]) -> str:
    value = rows.get("cost.top1_kernel", "")
    kernel = value.split(maxsplit=1)[0] if value else ""
    if not kernel:
        raise SourceAttributionError("missing cost.top1_kernel")
    return kernel


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SourceAttributionError(f"invalid JSON in {path}: {exc}") from exc


def load_region_map(path: pathlib.Path) -> dict[str, Any]:
    obj = load_json(path)
    kernels = obj.get("kernels") if isinstance(obj, dict) else None
    if not isinstance(kernels, dict):
        raise SourceAttributionError("region map missing kernels object")
    return kernels


def require_json_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceAttributionError(f"{description} must be an integer")
    return value


def validate_region(kernel: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SourceAttributionError(f"invalid region entry for {kernel}")
    region = dict(raw)
    for name in ("file", "line_start", "line_end", "label_line"):
        if name not in region:
            raise SourceAttributionError(f"region entry for {kernel} missing {name}")
    if not isinstance(region["file"], str) or not region["file"]:
        raise SourceAttributionError(f"region entry for {kernel} has invalid file")
    for name in ("line_start", "line_end", "label_line"):
        region[name] = require_json_int(region[name], f"region entry for {kernel} {name}")
    return region


def source_line_kernel_matches(source_line_kernel: str, top_kernel: str) -> bool:
    if not source_line_kernel:
        return True
    return source_line_kernel == top_kernel or source_line_kernel in top_kernel or top_kernel in source_line_kernel


def load_ablation_delta(path: pathlib.Path | None, kernel: str) -> int | None:
    if path is None:
        return None
    obj = load_json(path)
    deltas = obj.get("deltas") if isinstance(obj, dict) else None
    if not isinstance(deltas, list):
        raise SourceAttributionError("ablation JSON missing deltas list")
    for item in deltas:
        if not isinstance(item, dict) or item.get("kernel") != kernel:
            continue
        if "delta_ms_x1000" not in item:
            raise SourceAttributionError(f"ablation delta for {kernel} missing delta_ms_x1000")
        return require_json_int(item["delta_ms_x1000"], f"ablation delta for {kernel} delta_ms_x1000")
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Map SYCL hot-kernel cost rows to source attribution rows")
    parser.add_argument("--cost-ranking", required=True, type=pathlib.Path)
    parser.add_argument("--source-line", required=True, type=pathlib.Path)
    parser.add_argument("--region-map", required=True, type=pathlib.Path)
    parser.add_argument("--ablation-json", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        cost_rows = load_parse_rows(args.cost_ranking)
        source_rows = load_parse_rows(args.source_line)
        top_kernel = parse_top_kernel(cost_rows)

        source_status = source_rows.get("source_line.status", "")
        if not source_status:
            raise SourceAttributionError("missing source_line.status")
        if source_status not in {"pass", "fail", "sampled-line-cost", "gtpin-bbl-runtime-cost", "asm-line-static-cost", "dwarf-line-table-only"}:
            raise SourceAttributionError(f"invalid source_line.status {source_status}")
        source_line_kernel = source_rows.get("source_line.required_kernel", "")
        source_line_matches_top_kernel = source_line_kernel_matches(source_line_kernel, top_kernel)
        sampled_line_matches_top_kernel = bool(source_line_kernel) and source_line_matches_top_kernel
        exact_pass = source_status == "pass" and source_line_matches_top_kernel
        sampled_line_cost = source_status == "sampled-line-cost" and sampled_line_matches_top_kernel
        gtpin_bbl_runtime_cost = source_status == "gtpin-bbl-runtime-cost" and sampled_line_matches_top_kernel
        asm_line_static_cost = source_status == "asm-line-static-cost" and source_line_matches_top_kernel
        dwarf_line_table_only = source_status == "dwarf-line-table-only" and source_line_matches_top_kernel
        exact_blocker = source_rows.get("source_line.blocker", "unknown")
        if (source_status in {"pass", "asm-line-static-cost", "dwarf-line-table-only"} and not source_line_matches_top_kernel) or (
            source_status in {"sampled-line-cost", "gtpin-bbl-runtime-cost"} and not sampled_line_matches_top_kernel
        ):
            exact_blocker = f"source_line_kernel_mismatch:{source_line_kernel}"

        regions = load_region_map(args.region_map)
        raw_region = regions.get(top_kernel)
        region = validate_region(top_kernel, raw_region) if raw_region is not None else None

        delta = load_ablation_delta(args.ablation_json, top_kernel)
        if exact_pass:
            status = "exact_source_line"
        elif sampled_line_cost:
            status = "sampled_line_cost"
        elif gtpin_bbl_runtime_cost:
            status = "gtpin_bbl_runtime_cost"
        elif asm_line_static_cost:
            status = "asm_line_static_cost"
        elif dwarf_line_table_only:
            status = "dwarf_line_table_only"
        else:
            if region is None:
                raise SourceAttributionError(f"missing source-region map entry for top kernel {top_kernel}")
            status = "source_region_plus_ablation" if delta is not None else "source_region"

        print(f"source_attribution.status {status}")
        print(f"source_attribution.kernel {top_kernel}")
        if source_line_kernel:
            print(f"source_attribution.source_line_kernel {source_line_kernel}")
        if sampled_line_cost:
            print("source_attribution.source_line_status sampled-line-cost")
            sampled_top_source_line = source_rows.get("source_line.sampled_top_source_line", "")
            sampled_top_sample_count = source_rows.get("source_line.sampled_top_sample_count", "")
            if sampled_top_source_line:
                print(f"source_attribution.sampled_top_source_line {sampled_top_source_line}")
            if sampled_top_sample_count:
                print(f"source_attribution.sampled_top_sample_count {sampled_top_sample_count}")
        if gtpin_bbl_runtime_cost:
            print("source_attribution.source_line_status gtpin-bbl-runtime-cost")
            gtpin_bbl_top_source_line = source_rows.get("source_line.gtpin_bbl_top_source_line", "")
            gtpin_bbl_top_sample_count = source_rows.get("source_line.gtpin_bbl_top_sample_count", "")
            if gtpin_bbl_top_source_line:
                print(f"source_attribution.gtpin_bbl_top_source_line {gtpin_bbl_top_source_line}")
            if gtpin_bbl_top_sample_count:
                print(f"source_attribution.gtpin_bbl_top_sample_count {gtpin_bbl_top_sample_count}")
        if asm_line_static_cost:
            print("source_attribution.source_line_status asm-line-static-cost")
            asm_top_source_line = source_rows.get("source_line.asm_top_source_line", "")
            asm_top_static_score = source_rows.get("source_line.asm_top_static_score", "")
            if asm_top_source_line:
                print(f"source_attribution.asm_top_source_line {asm_top_source_line}")
            if asm_top_static_score:
                print(f"source_attribution.asm_top_static_score {asm_top_static_score}")
        if dwarf_line_table_only:
            print("source_attribution.source_line_status dwarf-line-table-only")
        if region is not None:
            print(f"source_attribution.file {region['file']}")
            print(f"source_attribution.line_start {region['line_start']}")
            print(f"source_attribution.line_end {region['line_end']}")
            print(f"source_attribution.label_line {region['label_line']}")
        if not exact_pass and not sampled_line_cost and not gtpin_bbl_runtime_cost and not asm_line_static_cost and not dwarf_line_table_only:
            print(f"source_attribution.exact_line_blocker {exact_blocker}")
        if delta is not None:
            print(f"source_attribution.ablation_delta_ms_x1000 {delta}")
    except (OSError, SourceAttributionError) as exc:
        print(f"failed to parse source attribution: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
