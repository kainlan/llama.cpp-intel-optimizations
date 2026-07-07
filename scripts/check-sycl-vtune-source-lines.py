#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import re
import sys
from typing import Any

UNKNOWN_VALUES = {"", "[Unknown]", "[Unknown source file]"}
DWARF_ATTRIBUTION_MODE = "dwarf-line-table"
ASM_ATTRIBUTION_MODE = "asm-line-static"
ASM_ATTRIBUTION_STATUS = "asm_line_static_cost"
SAMPLED_PC_ATTRIBUTION_MODE = "sampled-pc-line"
SAMPLED_PC_ATTRIBUTION_STATUS = "sampled_line_cost"
GTPIN_BBL_ATTRIBUTION_MODE = "gtpin-bbl-line"
GTPIN_BBL_ATTRIBUTION_STATUS = "gtpin_bbl_runtime_cost"
PTI_INSTCOUNT_ATTRIBUTION_MODE = "pti-instcount-line"
PTI_INSTCOUNT_ATTRIBUTION_STATUS = "pti_instcount_runtime_cost"
VTUNE_ATTRIBUTION_MODE = "vtune-sampled-exact"
DEBUG_LINE_SECTION_RE = re.compile(r"(?m)^\s*\[\s*\d+\]\s+\.debug_line(?:\s|$)")
NO_GPU_SIDE_TRACE_RE = re.compile(r"no GPU-side trace.{0,120}data was collected", re.IGNORECASE | re.DOTALL)
GTPIN_NO_KERNELS_RE = re.compile(r"GTPin didn't find any kernels", re.IGNORECASE)
GTPIN_REGISTER_PRESSURE_RE = re.compile(r"Not enough free registers", re.IGNORECASE)


def load_line_table_parser() -> Any:
    path = pathlib.Path(__file__).resolve().with_name("parse-sycl-zebin-line-table.py")
    spec = importlib.util.spec_from_file_location("parse_sycl_zebin_line_table", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load ZEBin line-table parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_row_shape(row: dict[str | None, str | list[str] | None]) -> dict[str, str]:
    if None in row:
        raise ValueError("malformed CSV row contains surplus fields")
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None or isinstance(value, list):
            raise ValueError("malformed CSV row shape")
        normalized[key] = "" if value is None else value
    return normalized


def row_matches_kernel(row: dict[str, str], required: str | None) -> bool:
    if required is None:
        return True
    joined = " ".join(value for value in row.values() if value)
    return required in joined


def row_has_known_source_line(row: dict[str, str]) -> bool:
    source_line = row.get("Source Line", "").strip()
    return source_line not in UNKNOWN_VALUES


def row_attribution_mode(row: dict[str, str]) -> str:
    return row.get("Source Attribution Mode", "").strip()


def row_attribution_status(row: dict[str, str]) -> str:
    return row.get("Source Attribution Status", "").strip()


def row_is_dwarf_line_table(row: dict[str, str]) -> bool:
    return row_attribution_mode(row) == DWARF_ATTRIBUTION_MODE and row_attribution_status(row) not in {
        ASM_ATTRIBUTION_STATUS,
        SAMPLED_PC_ATTRIBUTION_STATUS,
        GTPIN_BBL_ATTRIBUTION_STATUS,
        PTI_INSTCOUNT_ATTRIBUTION_STATUS,
    }


def row_is_asm_line_static(row: dict[str, str]) -> bool:
    mode = row_attribution_mode(row)
    status = row_attribution_status(row)
    return (mode == ASM_ATTRIBUTION_MODE and status in {"", ASM_ATTRIBUTION_STATUS}) or (
        status == ASM_ATTRIBUTION_STATUS and mode in {"", ASM_ATTRIBUTION_MODE}
    )


def row_is_sampled_pc_line(row: dict[str, str]) -> bool:
    return row_attribution_mode(row) == SAMPLED_PC_ATTRIBUTION_MODE and row_attribution_status(row) == SAMPLED_PC_ATTRIBUTION_STATUS


def row_is_gtpin_bbl_line(row: dict[str, str]) -> bool:
    return row_attribution_mode(row) == GTPIN_BBL_ATTRIBUTION_MODE and row_attribution_status(row) == GTPIN_BBL_ATTRIBUTION_STATUS


def row_is_pti_instcount_line(row: dict[str, str]) -> bool:
    return (
        row_attribution_mode(row) == PTI_INSTCOUNT_ATTRIBUTION_MODE
        and row_attribution_status(row) == PTI_INSTCOUNT_ATTRIBUTION_STATUS
    )


def row_has_non_vtune_attribution_marker(row: dict[str, str]) -> bool:
    mode = row_attribution_mode(row)
    status = row_attribution_status(row)
    return mode in {
        DWARF_ATTRIBUTION_MODE,
        ASM_ATTRIBUTION_MODE,
        SAMPLED_PC_ATTRIBUTION_MODE,
        GTPIN_BBL_ATTRIBUTION_MODE,
        PTI_INSTCOUNT_ATTRIBUTION_MODE,
    } or status in {
        ASM_ATTRIBUTION_STATUS,
        SAMPLED_PC_ATTRIBUTION_STATUS,
        GTPIN_BBL_ATTRIBUTION_STATUS,
        PTI_INSTCOUNT_ATTRIBUTION_STATUS,
    }


def parse_int_field(row: dict[str, str], field: str) -> int:
    raw = row.get(field, "").replace(",", "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def read_source_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        sample_lines = sample.splitlines()
        dialect = csv.excel_tab if sample_lines and "\t" in sample_lines[0] else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [validate_row_shape(raw_row) for raw_row in reader]


def count_vtune_sampled_known_rows(rows: list[dict[str, str]], required_kernel: str | None) -> int:
    return sum(
        1
        for row in rows
        if row_matches_kernel(row, required_kernel)
        and row_has_known_source_line(row)
        and not row_has_non_vtune_attribution_marker(row)
    )


def count_dwarf_line_table_known_rows(rows: list[dict[str, str]], required_kernel: str | None) -> int:
    return sum(
        1
        for row in rows
        if row_matches_kernel(row, required_kernel) and row_has_known_source_line(row) and row_is_dwarf_line_table(row)
    )


def asm_line_static_rows(rows: list[dict[str, str]], required_kernel: str | None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_matches_kernel(row, required_kernel) and row_has_known_source_line(row) and row_is_asm_line_static(row)
    ]


def sampled_pc_line_rows(rows: list[dict[str, str]], required_kernel: str | None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_matches_kernel(row, required_kernel)
        and row_has_known_source_line(row)
        and row_is_sampled_pc_line(row)
        and (parse_int_field(row, "Sample Count") > 0 or parse_int_field(row, "sample_count") > 0)
    ]


def gtpin_bbl_line_rows(rows: list[dict[str, str]], required_kernel: str | None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_matches_kernel(row, required_kernel)
        and row_has_known_source_line(row)
        and row_is_gtpin_bbl_line(row)
        and (parse_int_field(row, "Sample Count") > 0 or parse_int_field(row, "sample_count") > 0)
    ]


def pti_instcount_line_rows(rows: list[dict[str, str]], required_kernel: str | None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_matches_kernel(row, required_kernel)
        and row_has_known_source_line(row)
        and row_is_pti_instcount_line(row)
        and (parse_int_field(row, "Sample Count") > 0 or parse_int_field(row, "sample_count") > 0)
    ]


def top_sampled_pc_line_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (parse_int_field(row, "Sample Count"), parse_int_field(row, "sample_count")))


def top_asm_line_static_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (parse_int_field(row, "Static Score"), parse_int_field(row, "Static Instruction Count")))


def parse_dwarf_line_table(module: Any, text: str, require_path: str | None) -> dict[str, object]:
    try:
        return module.parse_line_table(text, require_path)
    except Exception as exc:
        line_table_error = getattr(module, "LineTableError", None)
        if isinstance(line_table_error, type) and isinstance(exc, line_table_error):
            raise ValueError(str(exc)) from None
        raise


def read_optional_text(paths: list[pathlib.Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check VTune GPU source-line feasibility for SYCL kernels")
    parser.add_argument("--readelf-sections", required=True, type=pathlib.Path)
    parser.add_argument("--vtune-csv", type=pathlib.Path)
    parser.add_argument("--require-kernel")
    parser.add_argument("--dwarf-line-dump", type=pathlib.Path)
    parser.add_argument("--require-source-path")
    parser.add_argument(
        "--dwarf-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from decoded ZEBin DWARF line tables",
    )
    parser.add_argument(
        "--allow-dwarf-line-table-only",
        action="store_true",
        help="allow DWARF line-table CSV rows to pass with source_line.status dwarf-line-table-only when VTune rows are unavailable",
    )
    parser.add_argument(
        "--sampled-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from dynamic PC samples plus ZEBin DWARF line tables",
    )
    parser.add_argument(
        "--gtpin-bbl-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from GTPin memorytrace BBL counts plus ZEBin DWARF line tables",
    )
    parser.add_argument(
        "--allow-gtpin-bbl-runtime-cost",
        action="store_true",
        help="allow GTPin BBL runtime-count rows to pass with source_line.status gtpin-bbl-runtime-cost",
    )
    parser.add_argument(
        "--pti-instcount-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from PTI instcount instruction counts plus ZEBin DWARF line tables",
    )
    parser.add_argument(
        "--allow-pti-instcount-runtime-cost",
        action="store_true",
        help="allow PTI instcount runtime-count rows to pass with source_line.status pti-instcount-runtime-cost",
    )
    parser.add_argument(
        "--asm-source-lines-csv",
        type=pathlib.Path,
        help="checker-compatible CSV generated from ZEBin DWARF plus assembly instruction addresses",
    )
    parser.add_argument(
        "--allow-asm-line-static-cost",
        action="store_true",
        help="allow ASM-resolved static source-line rows to pass with source_line.status asm-line-static-cost",
    )
    parser.add_argument("--vtune-stdout", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--vtune-stderr", action="append", default=[], type=pathlib.Path)
    args = parser.parse_args(argv)

    has_allowed_dwarf_only = args.allow_dwarf_line_table_only and args.dwarf_source_lines_csv is not None
    has_allowed_asm_static = args.allow_asm_line_static_cost and args.asm_source_lines_csv is not None
    has_allowed_gtpin_bbl = args.allow_gtpin_bbl_runtime_cost and args.gtpin_bbl_source_lines_csv is not None
    has_allowed_pti_instcount = args.allow_pti_instcount_runtime_cost and args.pti_instcount_source_lines_csv is not None
    has_sampled_pc_lines = args.sampled_source_lines_csv is not None
    if args.vtune_csv is None and not (
        has_sampled_pc_lines or has_allowed_gtpin_bbl or has_allowed_pti_instcount or has_allowed_asm_static or has_allowed_dwarf_only
    ):
        print(
            "failed to check source lines: --vtune-csv is required unless --sampled-source-lines-csv or an explicitly "
            "allowed --gtpin-bbl-source-lines-csv, --pti-instcount-source-lines-csv, --asm-source-lines-csv, "
            "or --dwarf-source-lines-csv fallback is provided"
        )
        return 2

    try:
        vtune_log_text = read_optional_text([*args.vtune_stdout, *args.vtune_stderr])
        vtune_no_gpu_side_trace = NO_GPU_SIDE_TRACE_RE.search(vtune_log_text) is not None
        gtpin_no_kernels = GTPIN_NO_KERNELS_RE.search(vtune_log_text) is not None
        gtpin_register_pressure = GTPIN_REGISTER_PRESSURE_RE.search(vtune_log_text) is not None
        sections = args.readelf_sections.read_text(encoding="utf-8", errors="replace")
        debug_line_present = DEBUG_LINE_SECTION_RE.search(sections) is not None
        non_unknown_rows = 0
        if args.vtune_csv is not None:
            vtune_rows = read_source_csv(args.vtune_csv)
            non_unknown_rows = count_vtune_sampled_known_rows(vtune_rows, args.require_kernel)

        dwarf_status = "not_checked"
        dwarf_error = ""
        dwarf_source_rows = 0
        dwarf_required_path_present = True
        if args.dwarf_line_dump is not None:
            module = load_line_table_parser()
            try:
                parsed = parse_dwarf_line_table(
                    module,
                    args.dwarf_line_dump.read_text(encoding="utf-8", errors="replace"),
                    args.require_source_path,
                )
            except ValueError as exc:
                dwarf_status = "error"
                dwarf_error = str(exc)
                dwarf_required_path_present = args.require_source_path is None
            else:
                dwarf_status = str(parsed["status"])
                dwarf_source_rows = int(parsed["source_rows"])
                dwarf_required_path_present = True if args.require_source_path is None else bool(parsed["required_path_present"])

        dwarf_source_line_rows = 0
        if args.dwarf_source_lines_csv is not None:
            dwarf_source_line_rows = count_dwarf_line_table_known_rows(
                read_source_csv(args.dwarf_source_lines_csv),
                args.require_kernel,
            )

        sampled_source_line_rows = 0
        sampled_top_row: dict[str, str] | None = None
        if (
            args.sampled_source_lines_csv is not None
            and args.sampled_source_lines_csv.is_file()
            and args.sampled_source_lines_csv.stat().st_size > 0
        ):
            sampled_rows = sampled_pc_line_rows(read_source_csv(args.sampled_source_lines_csv), args.require_kernel)
            sampled_source_line_rows = len(sampled_rows)
            sampled_top_row = top_sampled_pc_line_row(sampled_rows)

        gtpin_bbl_source_line_rows = 0
        gtpin_bbl_top_row: dict[str, str] | None = None
        if (
            args.gtpin_bbl_source_lines_csv is not None
            and args.gtpin_bbl_source_lines_csv.is_file()
            and args.gtpin_bbl_source_lines_csv.stat().st_size > 0
        ):
            gtpin_rows = gtpin_bbl_line_rows(read_source_csv(args.gtpin_bbl_source_lines_csv), args.require_kernel)
            gtpin_bbl_source_line_rows = len(gtpin_rows)
            gtpin_bbl_top_row = top_sampled_pc_line_row(gtpin_rows)

        pti_instcount_source_line_rows = 0
        pti_instcount_top_row: dict[str, str] | None = None
        if (
            args.pti_instcount_source_lines_csv is not None
            and args.pti_instcount_source_lines_csv.is_file()
            and args.pti_instcount_source_lines_csv.stat().st_size > 0
        ):
            pti_rows = pti_instcount_line_rows(read_source_csv(args.pti_instcount_source_lines_csv), args.require_kernel)
            pti_instcount_source_line_rows = len(pti_rows)
            pti_instcount_top_row = top_sampled_pc_line_row(pti_rows)

        asm_source_line_rows = 0
        asm_top_row: dict[str, str] | None = None
        if args.asm_source_lines_csv is not None and args.asm_source_lines_csv.is_file() and args.asm_source_lines_csv.stat().st_size > 0:
            asm_rows = asm_line_static_rows(read_source_csv(args.asm_source_lines_csv), args.require_kernel)
            asm_source_line_rows = len(asm_rows)
            asm_top_row = top_asm_line_static_row(asm_rows)
    except (OSError, csv.Error, IndexError, TypeError, ValueError) as exc:
        print(f"failed to check source lines: {exc}")
        return 2

    source_attribution_mode = "none"
    status = "fail"
    if not debug_line_present:
        blocker = "missing_debug_line"
    elif args.dwarf_line_dump is not None and dwarf_status == "error":
        blocker = "dwarf_" + re.sub(r"[^a-z0-9]+", "_", dwarf_error.lower()).strip("_")
        if blocker == "dwarf_":
            blocker = "dwarf_parse_error"
    elif args.dwarf_line_dump is not None and not dwarf_required_path_present:
        blocker = "missing_dwarf_source_path"
    elif non_unknown_rows > 0:
        blocker = "none"
        status = "pass"
        source_attribution_mode = VTUNE_ATTRIBUTION_MODE
    elif sampled_source_line_rows > 0:
        blocker = "none"
        status = "sampled-line-cost"
        source_attribution_mode = SAMPLED_PC_ATTRIBUTION_MODE
    elif args.allow_gtpin_bbl_runtime_cost and gtpin_bbl_source_line_rows > 0:
        blocker = "none"
        status = "gtpin-bbl-runtime-cost"
        source_attribution_mode = GTPIN_BBL_ATTRIBUTION_MODE
    elif args.allow_pti_instcount_runtime_cost and pti_instcount_source_line_rows > 0:
        blocker = "none"
        status = "pti-instcount-runtime-cost"
        source_attribution_mode = PTI_INSTCOUNT_ATTRIBUTION_MODE
    elif args.allow_asm_line_static_cost and asm_source_line_rows > 0:
        blocker = "none"
        status = "asm-line-static-cost"
        source_attribution_mode = ASM_ATTRIBUTION_MODE
    elif args.allow_dwarf_line_table_only and dwarf_source_line_rows > 0:
        blocker = "none"
        status = "dwarf-line-table-only"
        source_attribution_mode = DWARF_ATTRIBUTION_MODE
    elif vtune_no_gpu_side_trace:
        blocker = "vtune_no_gpu_side_trace"
    else:
        blocker = "vtune_unknown_source"

    print(f"source_line.debug_line_present {1 if debug_line_present else 0}")
    print(f"source_line.non_unknown_rows {non_unknown_rows}")
    print(f"source_line.vtune_sampled_non_unknown_rows {non_unknown_rows}")
    print(f"source_line.vtune_no_gpu_side_trace {1 if vtune_no_gpu_side_trace else 0}")
    print(f"source_line.gtpin_no_kernels {1 if gtpin_no_kernels else 0}")
    print(f"source_line.gtpin_register_pressure {1 if gtpin_register_pressure else 0}")
    if args.require_kernel is not None:
        print(f"source_line.required_kernel {args.require_kernel}")
    if args.dwarf_line_dump is not None:
        print(f"source_line.dwarf_status {dwarf_status}")
        if dwarf_error:
            print(f"source_line.dwarf_error {dwarf_error}")
        print(f"source_line.dwarf_source_rows {dwarf_source_rows}")
        print(f"source_line.dwarf_required_path_present {1 if dwarf_required_path_present else 0}")
    if args.dwarf_source_lines_csv is not None:
        print(f"source_line.dwarf_source_line_rows {dwarf_source_line_rows}")
        print(f"source_line.allow_dwarf_line_table_only {1 if args.allow_dwarf_line_table_only else 0}")
    if args.sampled_source_lines_csv is not None:
        print(f"source_line.sampled_source_line_rows {sampled_source_line_rows}")
        if sampled_top_row is not None:
            print(f"source_line.sampled_top_source_line {sampled_top_row.get('Source Line', '')}")
            print(
                "source_line.sampled_top_sample_count "
                f"{max(parse_int_field(sampled_top_row, 'Sample Count'), parse_int_field(sampled_top_row, 'sample_count'))}"
            )
    if args.gtpin_bbl_source_lines_csv is not None:
        print(f"source_line.gtpin_bbl_source_line_rows {gtpin_bbl_source_line_rows}")
        print(f"source_line.allow_gtpin_bbl_runtime_cost {1 if args.allow_gtpin_bbl_runtime_cost else 0}")
        if gtpin_bbl_top_row is not None:
            print(f"source_line.gtpin_bbl_top_source_line {gtpin_bbl_top_row.get('Source Line', '')}")
            print(
                "source_line.gtpin_bbl_top_sample_count "
                f"{max(parse_int_field(gtpin_bbl_top_row, 'Sample Count'), parse_int_field(gtpin_bbl_top_row, 'sample_count'))}"
            )
    if args.pti_instcount_source_lines_csv is not None:
        print(f"source_line.pti_instcount_source_line_rows {pti_instcount_source_line_rows}")
        print(f"source_line.allow_pti_instcount_runtime_cost {1 if args.allow_pti_instcount_runtime_cost else 0}")
        if pti_instcount_top_row is not None:
            print(f"source_line.pti_instcount_top_source_line {pti_instcount_top_row.get('Source Line', '')}")
            print(
                "source_line.pti_instcount_top_sample_count "
                f"{max(parse_int_field(pti_instcount_top_row, 'Sample Count'), parse_int_field(pti_instcount_top_row, 'sample_count'))}"
            )
    if args.asm_source_lines_csv is not None:
        print(f"source_line.asm_source_line_rows {asm_source_line_rows}")
        print(f"source_line.allow_asm_line_static_cost {1 if args.allow_asm_line_static_cost else 0}")
        if asm_top_row is not None:
            print(f"source_line.asm_top_source_line {asm_top_row.get('Source Line', '')}")
            print(f"source_line.asm_top_static_score {parse_int_field(asm_top_row, 'Static Score')}")
            print(f"source_line.asm_top_instruction_count {parse_int_field(asm_top_row, 'Static Instruction Count')}")
    print(f"source_line.source_attribution_mode {source_attribution_mode}")
    print(f"source_line.blocker {blocker}")
    print(f"source_line.status {status}")
    return 0 if status in {
        "pass",
        "sampled-line-cost",
        "gtpin-bbl-runtime-cost",
        "pti-instcount-runtime-cost",
        "asm-line-static-cost",
        "dwarf-line-table-only",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
