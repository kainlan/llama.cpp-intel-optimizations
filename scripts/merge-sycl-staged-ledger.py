#!/usr/bin/env python3
"""Merge staged SYCL profiling artifacts into a strict closure ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layer_ledger = load_module("parse_sycl_layer_ledger", ROOT / "scripts" / "parse-sycl-layer-ledger.py")
stage_manifest = load_module("parse_sycl_stage_manifest", ROOT / "scripts" / "parse-sycl-stage-manifest.py")


def validate_manifests(paths: list[pathlib.Path]) -> None:
    manifests = [stage_manifest.load_manifest(path) for path in paths]
    identity = stage_manifest.merge_identity(manifests[0])
    for obj in manifests[1:]:
        if stage_manifest.merge_identity(obj) != identity:
            raise stage_manifest.ManifestError("metadata mismatch across stage manifests")


def summary_lines(path: pathlib.Path) -> list[str]:
    # Reuse the layer-ledger parse-file validator so malformed summary rows fail
    # consistently, then preserve the source summary rows in ledger output.
    layer_ledger.read_parse_file(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Merge staged SYCL profiling artifacts into a closure ledger", allow_abbrev=False)
    parser.add_argument("--manifest", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--timeline", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-profile", type=pathlib.Path, required=True)
    parser.add_argument("--bench-stderr", type=pathlib.Path, required=True)
    parser.add_argument("--l0-summary", type=pathlib.Path, required=True)
    parser.add_argument("--ur-summary", type=pathlib.Path, required=True)
    parser.add_argument("--vtune-summary", type=pathlib.Path, required=True)
    parser.add_argument("--source-line", type=pathlib.Path, required=True)
    parser.add_argument("--source-attribution", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    try:
        validate_manifests(args.manifest)
    except (OSError, json.JSONDecodeError, stage_manifest.ManifestError) as exc:
        if "metadata mismatch" in str(exc):
            print("coverage.layer_status metadata_mismatch")
            print("failed to merge staged ledger: metadata mismatch across stage manifests")
        else:
            print(f"failed to merge staged ledger: {exc}")
        return 2

    try:
        events = layer_ledger.timeline_parser.load_trace_events(args.timeline)
        category_totals, _, wall_us, _, _, _ = layer_ledger.timeline_parser.summarize_events(events)
        rows = layer_ledger.kernel_parser.load_rows(args.kernel_profile)
        kernel_totals, _ = layer_ledger.kernel_parser.aggregate_rows(rows)

        l0_summary = layer_ledger.read_parse_file(args.l0_summary)
        ur_summary = layer_ledger.read_parse_file(args.ur_summary)
        vtune_summary = layer_ledger.read_parse_file(args.vtune_summary)
        source_line_rows = summary_lines(args.source_line)
        source_line_summary = layer_ledger.read_parse_file(args.source_line)
        source_attribution_rows = summary_lines(args.source_attribution)
        source_attribution_summary = layer_ledger.read_parse_file(args.source_attribution)

        l0_api = layer_ledger.int_metric(l0_summary, "l0.total_ms_x1000", args.l0_summary)
        ur_api = layer_ledger.int_metric(ur_summary, "ur.total_ms_x1000", args.ur_summary)
        vtune_gpu = layer_ledger.int_metric(vtune_summary, "vtune.kernel_total_ms_x1000", args.vtune_summary)

        status = source_attribution_summary.get("source_attribution.status")
        source_line_status = source_line_summary.get("source_line.status")
        if status not in {"exact_source_line", "source_region_plus_ablation", "sampled_line_cost", "gtpin_bbl_runtime_cost", "asm_line_static_cost", "dwarf_line_table_only"}:
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: source attribution incomplete")
            return 2
        if status == "exact_source_line" and source_line_status != "pass":
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_line_rows:
                print(line)
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: exact source attribution requires source_line.status pass")
            return 2
        if status == "sampled_line_cost" and source_line_status != "sampled-line-cost":
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_line_rows:
                print(line)
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: sampled PC source attribution requires source_line.status sampled-line-cost")
            return 2
        if status == "gtpin_bbl_runtime_cost" and source_line_status != "gtpin-bbl-runtime-cost":
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_line_rows:
                print(line)
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: GTPin BBL source attribution requires source_line.status gtpin-bbl-runtime-cost")
            return 2
        if status == "asm_line_static_cost" and source_line_status != "asm-line-static-cost":
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_line_rows:
                print(line)
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: ASM static source attribution requires source_line.status asm-line-static-cost")
            return 2
        if status == "dwarf_line_table_only" and source_line_status != "dwarf-line-table-only":
            print("coverage.layer_status source_attribution_incomplete")
            for line in source_line_rows:
                print(line)
            for line in source_attribution_rows:
                print(line)
            print("failed to merge staged ledger: DWARF line-table attribution requires source_line.status dwarf-line-table-only")
            return 2
        if status == "source_region_plus_ablation":
            try:
                layer_ledger.int_metric(
                    source_attribution_summary,
                    "source_attribution.ablation_delta_ms_x1000",
                    args.source_attribution,
                )
            except ValueError as exc:
                print("coverage.layer_status source_attribution_incomplete")
                for line in source_attribution_rows:
                    print(line)
                print(f"failed to merge staged ledger: {exc}")
                return 2

        e2e_host_total = layer_ledger.parse_e2e_stderr(args.bench_stderr)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError, AttributeError) as exc:
        print(f"failed to merge staged ledger: {exc}")
        return 2

    wall_total = layer_ledger.timeline_parser.us_to_ms_x1000(wall_us)
    sycl_submit_host = int(category_totals["sycl.submit"])
    app_host = min(wall_total, e2e_host_total)
    gpu_kernel = layer_ledger.kernel_parser.ns_to_ms_x1000(sum(totals["total_ns"] for totals in kernel_totals.values()))
    unknown = max(0, wall_total - sycl_submit_host - ur_api - l0_api - gpu_kernel)

    print(f"layer.wall_ms_x1000 {wall_total}")
    print(f"layer.app_host_ms_x1000 {app_host}")
    print(f"layer.sycl_submit_host_ms_x1000 {sycl_submit_host}")
    print(f"layer.ur_api_ms_x1000 {ur_api}")
    print(f"layer.level_zero_api_ms_x1000 {l0_api}")
    print(f"layer.gpu_kernel_ms_x1000 {gpu_kernel}")
    print(f"layer.vtune_gpu_ms_x1000 {vtune_gpu}")
    print(f"layer.unknown_wall_ms_x1000 {unknown}")
    print("coverage.layer_status ok")
    for line in source_line_rows:
        print(line)
    for line in source_attribution_rows:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
