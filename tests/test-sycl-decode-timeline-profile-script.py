#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-decode-timeline-profile.sh"
DOC = ROOT / "docs" / "backend" / "SYCL.md"


REQUIRED_DOC_STRINGS = [
    "GGML_SYCL_TIMELINE",
    "GGML_SYCL_TIMELINE_OUTPUT",
    "scripts/parse-sycl-timeline.py",
    "Perfetto",
    "host-side file:line",
    "VTune GPU source-line",
    "workers must not run",
    "timeline.gaps.parse",
    "gap_transition.",
    "host_gap_overlap.",
    "not by itself proof of GPU idle",
]

REQUIRED_DRY_RUN_STRINGS = [
    "DRY RUN",
    "ONEAPI_DEVICE_SELECTOR=level_zero:1",
    "GGML_SYCL_TIMELINE=timeline+events",
    "GGML_SYCL_TIMELINE_OUTPUT=",
    "GGML_SYCL_TIMELINE_TOKEN_START=1",
    "GGML_SYCL_KERNEL_PROFILE=1",
    "GGML_SYCL_KERNEL_PROFILE_FORMAT=both",
    "GGML_SYCL_KERNEL_PROFILE_RAW=1",
    "GGML_SYCL_KERNEL_PROFILE_TOP_N=80",
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
    "/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf",
    "./build/bin/llama-bench",
    "-ngl 99",
    "-fa 1",
    "-p 512",
    "-n 128",
    "-r 1",
    "sycl-timeline.json",
    "sycl-kernels",
    "bench.stdout",
    "bench.stderr",
    "timeline.parse",
    "timeline.gaps.parse",
    "kernels.parse",
    "cost-ranking.parse",
    "wall-ledger.parse",
    "scripts/parse-sycl-timeline.py",
    "scripts/parse-sycl-kernel-profile.py",
    "scripts/parse-sycl-profile-ledger.py",
    "--top-gaps 20",
    "--top-host-gap-overlaps 40",
    "--top-kernels 30",
]


EXECUTE_BRANCH_STRINGS = [
    "mkdir -p \"${OUT_ROOT}\"",
    "print_cmd >\"${OUT_ROOT}/command.txt\"",
    "env \"${env_args[@]}\" \"${bench_args[@]}\" >\"${OUT_ROOT}/bench.stdout\" 2>\"${OUT_ROOT}/bench.stderr\"",
    "python3 scripts/parse-sycl-timeline.py \"${OUT_ROOT}/sycl-timeline.json\" >\"${OUT_ROOT}/timeline.parse\"",
    "python3 scripts/parse-sycl-timeline.py \\",
    "--top-gaps 20 \\",
    "--top-host-gap-overlaps 40 \\",
    "\"${OUT_ROOT}/sycl-timeline.json\" >\"${timeline_gaps_parse}\"",
    "python3 scripts/parse-sycl-kernel-profile.py \"${OUT_ROOT}/sycl-kernels.csv\" >\"${OUT_ROOT}/kernels.parse\"",
    "python3 scripts/parse-sycl-kernel-profile.py --top-kernels 30 \"${OUT_ROOT}/sycl-kernels.csv\" >\"${OUT_ROOT}/cost-ranking.parse\"",
    "python3 scripts/parse-sycl-profile-ledger.py \"${OUT_ROOT}/sycl-timeline.json\" \"${OUT_ROOT}/sycl-kernels.csv\" >\"${OUT_ROOT}/wall-ledger.parse\"",
    "printf 'Artifacts: %s\\n' \"${OUT_ROOT}\"",
]


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_timeline_profile_script_is_dry_run_by_default() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for required in REQUIRED_DRY_RUN_STRINGS:
        assert required in result.stdout


def test_timeline_profile_script_execute_branch_wires_artifact_layout() -> None:
    text = _script_text()
    execute_start = 'mkdir -p "${OUT_ROOT}"'
    assert execute_start in text
    assert 'timeline_gaps_parse="${OUT_ROOT}/timeline.gaps.parse"' in text
    execute_branch = text[text.index(execute_start) :]
    for required in EXECUTE_BRANCH_STRINGS:
        assert required in execute_branch


def test_timeline_profile_script_refuses_execute_without_ack() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--execute"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2
    assert "requires --i-understand-this-runs-gpu-models" in result.stdout


def test_decode_timeline_profiler_docs_contain_required_contract_terms() -> None:
    text = _doc_text()
    for required in REQUIRED_DOC_STRINGS:
        assert required in text


def test_decode_timeline_profiler_section_follows_e2e_tg_ledger() -> None:
    text = _doc_text()
    ledger = "### GPT-OSS MXFP4 end-to-end TG profile ledger"
    timeline = "### SYCL decode timeline profiler"
    assert ledger in text
    assert timeline in text
    assert text.index(ledger) < text.index(timeline)


def test_decode_timeline_profiler_documents_supported_parser_callsite_option() -> None:
    text = _doc_text()
    assert "--top-callsites" in text
    assert "--top " not in text
