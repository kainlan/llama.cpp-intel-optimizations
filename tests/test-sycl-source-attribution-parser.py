#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-source-attribution.py"


def run_parser(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_common(tmp: pathlib.Path) -> dict[str, pathlib.Path]:
    cost = tmp / "cost.parse"
    cost.write_text("cost.top1_kernel mxfp4.gateup.xmx_tiled_dpas_m2 706354\n", encoding="utf-8")
    source = tmp / "source.parse"
    source.write_text(
        "source_line.debug_line_present 0\n"
        "source_line.non_unknown_rows 0\n"
        "source_line.blocker missing_debug_line\n"
        "source_line.status fail\n",
        encoding="utf-8",
    )
    region = tmp / "regions.json"
    region.write_text(
        json.dumps(
            {
                "kernels": {
                    "mxfp4.gateup.xmx_tiled_dpas_m2": {
                        "file": "ggml/src/ggml-sycl/mmvq.cpp",
                        "line_start": 9730,
                        "line_end": 9955,
                        "label_line": 9767,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ablation = tmp / "ablation.json"
    ablation.write_text(
        json.dumps({"deltas": [{"kernel": "mxfp4.gateup.xmx_tiled_dpas_m2", "delta_ms_x1000": 120000}]}),
        encoding="utf-8",
    )
    return {"cost": cost, "source": source, "region": region, "ablation": ablation}


def test_source_attribution_uses_region_and_ablation_when_exact_lines_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
            "--ablation-json",
            str(p["ablation"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status source_region_plus_ablation" in result.stdout
        assert "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2" in result.stdout
        assert "source_attribution.file ggml/src/ggml-sycl/mmvq.cpp" in result.stdout
        assert "source_attribution.line_start 9730" in result.stdout
        assert "source_attribution.line_end 9955" in result.stdout
        assert "source_attribution.label_line 9767" in result.stdout
        assert "source_attribution.ablation_delta_ms_x1000 120000" in result.stdout
        assert "source_attribution.exact_line_blocker missing_debug_line" in result.stdout


def test_source_attribution_uses_region_when_exact_lines_fail_without_ablation() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status source_region" in result.stdout
        assert "source_attribution.ablation_delta_ms_x1000" not in result.stdout


def test_source_attribution_keeps_exact_source_line_status_when_exact_lines_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text("source_line.blocker none\nsource_line.status pass\n", encoding="utf-8")
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
            "--ablation-json",
            str(p["ablation"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status exact_source_line" in result.stdout
        assert "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2" in result.stdout
        assert "source_attribution.status source_region" not in result.stdout
        assert "source_attribution.exact_line_blocker" not in result.stdout


def test_source_attribution_does_not_claim_exact_when_source_line_kernel_mismatches_top_kernel() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text(
            "source_line.required_kernel sycl_source_line_probe\nsource_line.blocker none\nsource_line.status pass\n",
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
            "--ablation-json",
            str(p["ablation"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status source_region_plus_ablation" in result.stdout
        assert "source_attribution.source_line_kernel sycl_source_line_probe" in result.stdout
        assert "source_attribution.exact_line_blocker source_line_kernel_mismatch:sycl_source_line_probe" in result.stdout
        assert "source_attribution.status exact_source_line" not in result.stdout


def test_source_attribution_accepts_dwarf_line_table_only_as_distinct_status() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text("source_line.blocker none\nsource_line.status dwarf-line-table-only\n", encoding="utf-8")
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status dwarf_line_table_only" in result.stdout
        assert "source_attribution.source_line_status dwarf-line-table-only" in result.stdout
        assert "source_attribution.file ggml/src/ggml-sycl/mmvq.cpp" in result.stdout
        assert "source_attribution.status exact_source_line" not in result.stdout
        assert "source_attribution.exact_line_blocker" not in result.stdout


def test_source_attribution_accepts_asm_line_static_cost_as_distinct_status() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text(
            "source_line.required_kernel mxfp4.gateup.xmx_tiled_dpas_m2\n"
            "source_line.blocker none\n"
            "source_line.status asm-line-static-cost\n"
            "source_line.asm_top_source_line ggml/src/ggml-sycl/mmvq.cpp:6800\n"
            "source_line.asm_top_static_score 14\n",
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status asm_line_static_cost" in result.stdout
        assert "source_attribution.source_line_status asm-line-static-cost" in result.stdout
        assert "source_attribution.asm_top_source_line ggml/src/ggml-sycl/mmvq.cpp:6800" in result.stdout
        assert "source_attribution.asm_top_static_score 14" in result.stdout
        assert "source_attribution.status exact_source_line" not in result.stdout


def test_source_attribution_does_not_claim_dwarf_line_table_when_source_line_kernel_mismatches_top_kernel() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text(
            "source_line.required_kernel sycl_source_line_probe\nsource_line.blocker none\nsource_line.status dwarf-line-table-only\n",
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status source_region" in result.stdout
        assert "source_attribution.source_line_kernel sycl_source_line_probe" in result.stdout
        assert "source_attribution.exact_line_blocker source_line_kernel_mismatch:sycl_source_line_probe" in result.stdout
        assert "source_attribution.status dwarf_line_table_only" not in result.stdout


def test_source_attribution_does_not_claim_asm_static_when_source_line_kernel_mismatches_top_kernel() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text(
            "source_line.required_kernel sycl_source_line_probe\n"
            "source_line.blocker none\n"
            "source_line.status asm-line-static-cost\n"
            "source_line.asm_top_source_line main.cpp:148\n"
            "source_line.asm_top_static_score 3\n",
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 0, result.stdout
        assert "source_attribution.status source_region" in result.stdout
        assert "source_attribution.exact_line_blocker source_line_kernel_mismatch:sycl_source_line_probe" in result.stdout
        assert "source_attribution.status asm_line_static_cost" not in result.stdout


def test_source_attribution_reports_missing_region_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["region"].write_text(json.dumps({"kernels": {}}), encoding="utf-8")
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 2
        assert "failed to parse source attribution" in result.stdout
        assert "Traceback" not in result.stdout


def test_source_attribution_reports_missing_top_kernel_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["cost"].write_text("cost.kernel.rank.1.name mxfp4.gateup.xmx_tiled_dpas_m2\n", encoding="utf-8")
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 2
        assert "failed to parse source attribution" in result.stdout
        assert "Traceback" not in result.stdout


def test_source_attribution_rejects_invalid_source_line_status_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["source"].write_text("source_line.blocker missing_debug_line\nsource_line.status maybe\n", encoding="utf-8")
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 2
        assert "failed to parse source attribution" in result.stdout
        assert "invalid source_line.status maybe" in result.stdout
        assert "Traceback" not in result.stdout


def test_source_attribution_rejects_non_integer_region_and_ablation_numbers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["region"].write_text(
            json.dumps(
                {
                    "kernels": {
                        "mxfp4.gateup.xmx_tiled_dpas_m2": {
                            "file": "ggml/src/ggml-sycl/mmvq.cpp",
                            "line_start": 9730.5,
                            "line_end": 9955,
                            "label_line": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
        )
        assert result.returncode == 2
        assert "must be an integer" in result.stdout
        assert "Traceback" not in result.stdout

    with tempfile.TemporaryDirectory() as tmp_raw:
        p = write_common(pathlib.Path(tmp_raw))
        p["ablation"].write_text(
            json.dumps({"deltas": [{"kernel": "mxfp4.gateup.xmx_tiled_dpas_m2", "delta_ms_x1000": 120000.5}]}),
            encoding="utf-8",
        )
        result = run_parser(
            "--cost-ranking",
            str(p["cost"]),
            "--source-line",
            str(p["source"]),
            "--region-map",
            str(p["region"]),
            "--ablation-json",
            str(p["ablation"]),
        )
        assert result.returncode == 2
        assert "must be an integer" in result.stdout
        assert "Traceback" not in result.stdout
