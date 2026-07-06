#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MERGER = ROOT / "scripts" / "merge-sycl-staged-ledger.py"


def manifest(stage: str, root: str, build_sha: str = "abc123") -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": stage,
        "artifact_root": root,
        "build_sha": build_sha,
        "model": {"path": "/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf", "size": 12101000000},
        "device_selector": "level_zero:1",
        "fa": 1,
        "moe_knobs": {
            "GGML_SYCL_MOE_PHASE_MATERIALIZE": "1",
            "GGML_SYCL_MOE_PHASE_BULK_XMX": "1",
            "GGML_SYCL_MOE_DOWN_SUM_DIRECT": "1",
        },
        "prompt_tokens": 512,
        "gen_tokens": 128,
        "repeat": 1,
        "artifacts": {"summary": f"{root}/summary.parse"},
    }


def write_fixture(tmp: pathlib.Path, mismatch: bool = False) -> dict[str, pathlib.Path]:
    timeline = tmp / "sycl-timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "decode", "cat": "app.compute", "ph": "X", "ts": 0, "dur": 1000},
                    {"name": "submit", "cat": "sycl.submit", "ph": "X", "ts": 100, "dur": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    kernels = tmp / "sycl-kernels.csv"
    kernels.write_text(
        "name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded\n"
        "mxfp4.gateup.xmx_tiled_dpas_m2,compute,,0,in_order,1,300000,300000,300000,300000,300000,300000,0,0,0\n",
        encoding="utf-8",
    )
    bench_stderr = tmp / "bench.stderr"
    bench_stderr.write_text(
        "[SYCL-E2E-TG-STAGE] stage=moe calls=1 host=0.200 ms device=0.000 ms bytes=0 last_path=MUL_MAT_ID\n",
        encoding="utf-8",
    )
    l0 = tmp / "l0.parse"
    l0.write_text("l0.total_ms_x1000 80\n", encoding="utf-8")
    ur = tmp / "ur.parse"
    ur.write_text("ur.total_ms_x1000 50\n", encoding="utf-8")
    vtune = tmp / "vtune.parse"
    vtune.write_text("vtune.kernel_total_ms_x1000 310\n", encoding="utf-8")
    source_line = tmp / "source-line.parse"
    source_line.write_text("source_line.status fail\nsource_line.blocker vtune_unknown_source\n", encoding="utf-8")
    source_attr = tmp / "source-attribution.parse"
    source_attr.write_text(
        "source_attribution.status source_region_plus_ablation\n"
        "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n"
        "source_attribution.ablation_delta_ms_x1000 120000\n",
        encoding="utf-8",
    )
    base_manifest = tmp / "base.manifest.json"
    l0_manifest = tmp / "l0.manifest.json"
    base_manifest.write_text(json.dumps(manifest("base", str(tmp / "base"), "abc123")), encoding="utf-8")
    l0_manifest.write_text(
        json.dumps(manifest("l0", str(tmp / "l0"), "def456" if mismatch else "abc123")),
        encoding="utf-8",
    )
    return {
        "timeline": timeline,
        "kernels": kernels,
        "bench_stderr": bench_stderr,
        "l0": l0,
        "ur": ur,
        "vtune": vtune,
        "source_line": source_line,
        "source_attr": source_attr,
        "base_manifest": base_manifest,
        "l0_manifest": l0_manifest,
    }


def run_merger(paths: dict[str, pathlib.Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(MERGER),
            "--manifest",
            str(paths["base_manifest"]),
            "--manifest",
            str(paths["l0_manifest"]),
            "--timeline",
            str(paths["timeline"]),
            "--kernel-profile",
            str(paths["kernels"]),
            "--bench-stderr",
            str(paths["bench_stderr"]),
            "--l0-summary",
            str(paths["l0"]),
            "--ur-summary",
            str(paths["ur"]),
            "--vtune-summary",
            str(paths["vtune"]),
            "--source-line",
            str(paths["source_line"]),
            "--source-attribution",
            str(paths["source_attr"]),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_staged_merger_emits_ok_layer_status_and_unknown_wall() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "layer.wall_ms_x1000 1000" in result.stdout
        assert "layer.sycl_submit_host_ms_x1000 20" in result.stdout
        assert "layer.ur_api_ms_x1000 50" in result.stdout
        assert "layer.level_zero_api_ms_x1000 80" in result.stdout
        assert "layer.gpu_kernel_ms_x1000 300" in result.stdout
        assert "layer.unknown_wall_ms_x1000 550" in result.stdout
        assert "source_line.status fail" in result.stdout
        assert "source_attribution.status source_region_plus_ablation" in result.stdout


def test_staged_merger_accepts_exact_source_line_when_source_line_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status pass\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status exact_source_line\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "source_line.status pass" in result.stdout
        assert "source_attribution.status exact_source_line" in result.stdout


def test_staged_merger_rejects_exact_source_line_when_source_line_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_attr"].write_text(
            "source_attribution.status exact_source_line\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "source_line.status fail" in result.stdout
        assert "source_attribution.status exact_source_line" in result.stdout
        assert "exact source attribution requires source_line.status pass" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_accepts_dwarf_line_table_only_when_source_line_matches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status dwarf-line-table-only\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status dwarf_line_table_only\n"
            "source_attribution.source_line_status dwarf-line-table-only\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "source_line.status dwarf-line-table-only" in result.stdout
        assert "source_attribution.status dwarf_line_table_only" in result.stdout


def test_staged_merger_accepts_asm_line_static_cost_when_source_line_matches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status asm-line-static-cost\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status asm_line_static_cost\n"
            "source_attribution.source_line_status asm-line-static-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "source_line.status asm-line-static-cost" in result.stdout
        assert "source_attribution.status asm_line_static_cost" in result.stdout


def test_staged_merger_accepts_sampled_line_cost_when_source_line_matches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status sampled-line-cost\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status sampled_line_cost\n"
            "source_attribution.source_line_status sampled-line-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "source_line.status sampled-line-cost" in result.stdout
        assert "source_attribution.status sampled_line_cost" in result.stdout


def test_staged_merger_accepts_gtpin_bbl_runtime_cost_when_source_line_matches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status gtpin-bbl-runtime-cost\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status gtpin_bbl_runtime_cost\n"
            "source_attribution.source_line_status gtpin-bbl-runtime-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 0, result.stdout
        assert "coverage.layer_status ok" in result.stdout
        assert "source_line.status gtpin-bbl-runtime-cost" in result.stdout
        assert "source_attribution.status gtpin_bbl_runtime_cost" in result.stdout


def test_staged_merger_rejects_sampled_line_cost_when_source_line_mismatches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status pass\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status sampled_line_cost\n"
            "source_attribution.source_line_status sampled-line-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "sampled PC source attribution requires source_line.status sampled-line-cost" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_gtpin_bbl_runtime_cost_when_source_line_mismatches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status sampled-line-cost\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status gtpin_bbl_runtime_cost\n"
            "source_attribution.source_line_status gtpin-bbl-runtime-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "GTPin BBL source attribution requires source_line.status gtpin-bbl-runtime-cost" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_asm_line_static_cost_when_source_line_mismatches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status dwarf-line-table-only\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status asm_line_static_cost\n"
            "source_attribution.source_line_status asm-line-static-cost\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "ASM static source attribution requires source_line.status asm-line-static-cost" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_dwarf_line_table_only_when_source_line_mismatches() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_line"].write_text("source_line.status pass\nsource_line.blocker none\n", encoding="utf-8")
        paths["source_attr"].write_text(
            "source_attribution.status dwarf_line_table_only\n"
            "source_attribution.source_line_status dwarf-line-table-only\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "source_line.status pass" in result.stdout
        assert "source_attribution.status dwarf_line_table_only" in result.stdout
        assert "DWARF line-table attribution requires source_line.status dwarf-line-table-only" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_metadata_mismatch_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw), mismatch=True)
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status metadata_mismatch" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_plain_source_region_for_strict_closure() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_attr"].write_text(
            "source_attribution.status source_region\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "source_attribution.status source_region" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_requires_ablation_delta_for_plus_ablation_status() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_attr"].write_text(
            "source_attribution.status source_region_plus_ablation\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "missing integer metric source_attribution.ablation_delta_ms_x1000" in result.stdout
        assert "Traceback" not in result.stdout


def test_staged_merger_rejects_non_integer_ablation_delta_for_plus_ablation_status() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        paths = write_fixture(pathlib.Path(tmp_raw))
        paths["source_attr"].write_text(
            "source_attribution.status source_region_plus_ablation\n"
            "source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2\n"
            "source_attribution.ablation_delta_ms_x1000 bogus\n",
            encoding="utf-8",
        )
        result = run_merger(paths)
        assert result.returncode == 2
        assert "coverage.layer_status source_attribution_incomplete" in result.stdout
        assert "source_attribution.ablation_delta_ms_x1000 bogus" in result.stdout
        assert "missing integer metric source_attribution.ablation_delta_ms_x1000" in result.stdout
        assert "Traceback" not in result.stdout
