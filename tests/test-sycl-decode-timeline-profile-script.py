#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "backend" / "SYCL.md"


REQUIRED_DOC_STRINGS = [
    "GGML_SYCL_TIMELINE",
    "GGML_SYCL_TIMELINE_OUTPUT",
    "scripts/parse-sycl-timeline.py",
    "Perfetto",
    "host-side file:line",
    "VTune GPU source-line",
    "workers must not run",
]


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


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
