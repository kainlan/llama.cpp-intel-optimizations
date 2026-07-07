#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts" / "extract-sycl-pti-instcount-pc-counts.py"


def run_extractor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXTRACTOR), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_instcount_json(path: pathlib.Path) -> None:
    path.write_text(
        json.dumps(
            {
                "app_name": "",
                "kernels": [
                    {
                        "kernel_name": "ignored_kernel",
                        "invocations": [
                            {"tiles": [{"results": [{"offset": 16, "instruction_counter": 99}]}]},
                        ],
                    },
                    {
                        "kernel_name": "_ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE",
                        "invocations": [
                            {
                                "tiles": [
                                    {
                                        "results": [
                                            {"offset": 16, "instruction_counter": 7, "simd_active_lane_counter": 112},
                                            {"offset": 32, "instruction_counter": 0, "simd_active_lane_counter": 0},
                                        ]
                                    }
                                ]
                            },
                            {
                                "tiles": [
                                    {
                                        "results": [
                                            {"offset": 16, "instruction_counter": 5, "simd_active_lane_counter": 80},
                                            {"offset": 48, "instruction_counter": 3, "simd_active_lane_counter": 48},
                                        ]
                                    }
                                ]
                            },
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_extracts_positive_instruction_counts_as_pc_sample_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        instcount_json = tmp / "instcount.json"
        output = tmp / "pc-counts.csv"
        summary = tmp / "summary.parse"
        write_instcount_json(instcount_json)

        result = run_extractor(
            "--instcount-json",
            str(instcount_json),
            "--kernel-match",
            "mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel",
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        )

        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader(output.open(newline="")))
        assert rows == [
            {
                "kernel": "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",
                "pc": "0x10",
                "sample_count": "12",
                "sample_kind": "pti-instcount-instruction-exec-count",
            },
            {
                "kernel": "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",
                "pc": "0x30",
                "sample_count": "3",
                "sample_kind": "pti-instcount-instruction-exec-count",
            },
        ]
        text = summary.read_text(encoding="utf-8")
        assert "pti_instcount.status ok" in text
        assert "pti_instcount.matched_kernel_count 1" in text
        assert "pti_instcount.pc_rows 2" in text
        assert "pti_instcount.total_instruction_count 15" in text
        assert "pti_instcount.top_pc 0x10" in text


def test_reports_no_matching_kernel_without_synthesizing_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        instcount_json = tmp / "instcount.json"
        output = tmp / "pc-counts.csv"
        summary = tmp / "summary.parse"
        write_instcount_json(instcount_json)

        result = run_extractor(
            "--instcount-json",
            str(instcount_json),
            "--kernel-match",
            "not_present",
            "--source-computing-task",
            "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        )

        assert result.returncode == 2
        rows = list(csv.DictReader(output.open(newline="")))
        assert rows == []
        text = summary.read_text(encoding="utf-8")
        assert "pti_instcount.status no_matching_kernel" in text
        assert "pti_instcount.blocker no_matching_kernel" in text
