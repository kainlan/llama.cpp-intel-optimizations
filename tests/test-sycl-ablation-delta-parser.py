#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-ablation-deltas.py"
SOURCE_ATTR = ROOT / "scripts" / "parse-sycl-source-attribution.py"


def record(route: str, saving: float) -> dict[str, object]:
    return {
        "route": route,
        "mode": "dry-run",
        "shape": {"ncols": 2880, "hidden": 2880, "topk": 4, "layers": 24, "tokens": 128},
        "metrics": {
            "prepack_us": 0.0,
            "compute_us": 1.0,
            "launch_us": 1.0,
            "host_bounce_us": 0.0,
            "total_gateup_equiv_ms": 4.0,
            "saving_vs_baseline_ms": saving,
            "p50_us": 1.0,
            "p90_us": 1.0,
            "p99_us": 1.0,
        },
        "correct": {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0},
        "fatal": {"total": 0},
        "evidence": {"path": "synthetic", "dry_run": True, "device": "level_zero:1"},
    }


def test_ablation_delta_parser_emits_json_consumed_by_source_attribution() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        micro = tmp / "micro.jsonl"
        out = tmp / "ablation.json"
        micro.write_text("\n".join(json.dumps(row) for row in [record("prepack", 1.25), record("prepack", 1.75)]) + "\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(PARSER), "--microbench-jsonl", str(micro), "--kernel", "mxfp4.gateup.xmx_tiled_dpas_m2", "--route", "prepack"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 0, result.stdout
        out.write_text(result.stdout, encoding="utf-8")
        parsed = json.loads(result.stdout)
        assert parsed == {"deltas": [{"kernel": "mxfp4.gateup.xmx_tiled_dpas_m2", "route": "prepack", "delta_ms_x1000": 1500}]}

        cost = tmp / "cost.parse"
        source = tmp / "source.parse"
        region = tmp / "region.json"
        cost.write_text("cost.top1_kernel mxfp4.gateup.xmx_tiled_dpas_m2 706354\n", encoding="utf-8")
        source.write_text("source_line.status fail\nsource_line.blocker vtune_unknown_source\n", encoding="utf-8")
        region.write_text(json.dumps({"kernels": {"mxfp4.gateup.xmx_tiled_dpas_m2": {"file": "ggml/src/ggml-sycl/mmvq.cpp", "line_start": 9730, "line_end": 9955, "label_line": 9767}}}), encoding="utf-8")
        attr = subprocess.run([sys.executable, str(SOURCE_ATTR), "--cost-ranking", str(cost), "--source-line", str(source), "--region-map", str(region), "--ablation-json", str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert attr.returncode == 0, attr.stdout
        assert "source_attribution.status source_region_plus_ablation" in attr.stdout
        assert "source_attribution.ablation_delta_ms_x1000 1500" in attr.stdout


def test_ablation_delta_parser_rejects_missing_route_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        micro = pathlib.Path(tmp_raw) / "micro.jsonl"
        micro.write_text(json.dumps(record("baseline", 0.0)) + "\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(PARSER), "--microbench-jsonl", str(micro), "--kernel", "mxfp4.gateup.xmx_tiled_dpas_m2", "--route", "prepack"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 2
        assert "failed to parse ablation deltas" in result.stdout
        assert "route not found: prepack" in result.stdout
        assert "Traceback" not in result.stdout


def test_ablation_delta_parser_rejects_non_finite_saving_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        micro = pathlib.Path(tmp_raw) / "micro.jsonl"
        micro.write_text(json.dumps(record("prepack", float("nan"))) + "\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(PARSER), "--microbench-jsonl", str(micro), "--kernel", "mxfp4.gateup.xmx_tiled_dpas_m2", "--route", "prepack"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 2
        assert "failed to parse ablation deltas" in result.stdout
        assert "non-standard JSON constant" in result.stdout
        assert "Traceback" not in result.stdout


def test_ablation_delta_parser_rejects_nonzero_fatal_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        micro = pathlib.Path(tmp_raw) / "micro.jsonl"
        bad = record("prepack", 1.0)
        fatal = bad["fatal"]
        assert isinstance(fatal, dict)
        fatal["total"] = 1
        micro.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(PARSER), "--microbench-jsonl", str(micro), "--kernel", "mxfp4.gateup.xmx_tiled_dpas_m2", "--route", "prepack"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 2
        assert "failed to parse ablation deltas" in result.stdout
        assert "fatal.total is non-zero" in result.stdout
        assert "Traceback" not in result.stdout
