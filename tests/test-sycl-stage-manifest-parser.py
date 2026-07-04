#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-stage-manifest.py"


def run_parser(*paths: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *(str(path) for path in paths)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


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


def test_manifest_parser_accepts_matching_stage_manifests() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        base = tmp / "base.json"
        ur = tmp / "ur.json"
        base.write_text(json.dumps(manifest("base", "/tmp/base")), encoding="utf-8")
        ur.write_text(json.dumps(manifest("ur", "/tmp/ur")), encoding="utf-8")
        result = run_parser(base, ur)
        assert result.returncode == 0, result.stdout
        assert "manifest.status ok" in result.stdout
        assert "manifest.count 2" in result.stdout
        assert "manifest.stage.base.root /tmp/base" in result.stdout
        assert "manifest.stage.ur.root /tmp/ur" in result.stdout
        assert "manifest.schema_version 1" in result.stdout
        assert "manifest.merge_key" in result.stdout


def test_manifest_parser_rejects_metadata_mismatch_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        base = tmp / "base.json"
        l0 = tmp / "l0.json"
        base.write_text(json.dumps(manifest("base", "/tmp/base", "abc123")), encoding="utf-8")
        l0.write_text(json.dumps(manifest("l0", "/tmp/l0", "def456")), encoding="utf-8")
        result = run_parser(base, l0)
        assert result.returncode == 2
        assert "failed to parse stage manifests" in result.stdout
        assert "metadata mismatch" in result.stdout
        assert "Traceback" not in result.stdout
